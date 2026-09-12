"""
tasks.py — ARQ background workers.

Two tasks:
  run_pipeline        — full job: routing → [search] → story gen → verify → voiceover → stitch → upload
  run_edit_pipeline   — surgical edit: segment text change or image swap → re-voiceover/re-stitch → upload

Run the worker:
    python -m worker.tasks
"""
from __future__ import annotations
import asyncio
import os
import shutil

from arq import cron
from arq.connections import RedisSettings

from config import get_settings
from core.graph import run_graph
from services.supabase_service import (
    get_supabase,
    update_job_status,
    get_job_with_story_and_segments,
    upload_file,
    get_public_url,
)
from services.redis_service import publish_job_progress

settings = get_settings()


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_pipeline(
    ctx,
    job_id: str,
    prompt: str,
    image_url: str,
    wpm: int = 150,
    video_length_min: float = 10.0,
    voice_model: str = "af_heart",
) -> None:
    sb = get_supabase()
    update_job_status(sb, job_id, "running")
    publish_job_progress(job_id, 0, 0, "pipeline started")

    initial_state = {
        "job_id": job_id,
        "prompt": prompt,
        "image_url": image_url,
        "wpm": wpm,
        "video_length_min": video_length_min,
        "voice_model": voice_model,
    }

    try:
        # Run the full LangGraph pipeline in a thread (all nodes are sync)
        result = await asyncio.to_thread(run_graph, job_id, initial_state)

        if result.get("error"):
            update_job_status(sb, job_id, "failed", error_message=result["error"])
            publish_job_progress(job_id, 0, 0, f"failed: {result['error']}")
        # supabase_node already updates the record to 'completed' on success

    except Exception as e:
        update_job_status(sb, job_id, "failed", error_message=str(e))
        publish_job_progress(job_id, 0, 0, f"failed: {e}")


# ── Edit pipeline ──────────────────────────────────────────────────────────────

async def run_edit_pipeline(
    ctx,
    job_id: str,
    edit_type: str,       # "segment" | "image"
    segment_index: int | None = None,
    new_text: str | None = None,
    new_image_url: str | None = None,
) -> None:
    """
    edit_type="segment" → update text in DB, re-assemble, re-voiceover, re-stitch, re-upload.
    edit_type="image"   → update image_url in DB, re-stitch only with existing audio, re-upload.
    """
    sb = get_supabase()
    update_job_status(sb, job_id, "running")
    publish_job_progress(job_id, 0, 0, f"edit started ({edit_type})")

    try:
        data = get_job_with_story_and_segments(sb, job_id)
        if not data:
            raise ValueError(f"job {job_id} not found")

        job = data["job"]
        segments = data["segments"]

        tmp_dir = os.path.join(settings.tmp_dir, job_id)
        os.makedirs(tmp_dir, exist_ok=True)

        if edit_type == "segment":
            await _edit_segment(sb, job_id, job, segments, segment_index, new_text, tmp_dir)

        elif edit_type == "image":
            await _swap_image(sb, job_id, job, new_image_url, tmp_dir)

        else:
            raise ValueError(f"unknown edit_type: {edit_type}")

    except Exception as e:
        update_job_status(sb, job_id, "failed", error_message=str(e))
        publish_job_progress(job_id, 0, 0, f"edit failed: {e}")


async def _edit_segment(sb, job_id, job, segments, segment_index, new_text, tmp_dir):
    from story_gen import assemble_story

    # Update segment in DB
    sb.table("segments").update(
        {"text": new_text, "status": "edited"}
    ).eq("job_id", job_id).eq("segment_index", segment_index).execute()

    # Re-assemble story text with updated segment
    updated_segments = [
        {**s, "text": new_text if s["segment_index"] == segment_index else s["text"]}
        for s in segments
    ]
    story_dicts = [
        {"id": s["segment_index"], "text": s["text"], "status": s.get("status", "kept")}
        for s in updated_segments
    ]
    assembled = assemble_story(story_dicts, job.get("title", ""))
    story_text = assembled["story_text"]

    # Re-generate voiceover
    publish_job_progress(job_id, 0, 0, "re-generating voiceover…")
    audio_path = os.path.join(tmp_dir, "audio.mp3")
    await _run_voiceover(story_text, audio_path, job.get("wpm", 150), job_id)

    # Re-stitch
    image_url = job.get("image_url", "")
    video_path = os.path.join(tmp_dir, "video.mp4")
    await _run_stitch(image_url, audio_path, video_path, job_id, tmp_dir)

    # Re-upload
    await _upload_results(sb, job_id, audio_path, video_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)


async def _swap_image(sb, job_id, job, new_image_url, tmp_dir):
    # Update image_url in job record
    sb.table("jobs").update({"image_url": new_image_url}).eq("id", job_id).execute()

    # Download existing audio from Supabase
    publish_job_progress(job_id, 0, 0, "downloading existing audio…")
    audio_url = job.get("audio_url", "")
    audio_path = os.path.join(tmp_dir, "audio.mp3")
    if audio_url:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(audio_url)
            r.raise_for_status()
            with open(audio_path, "wb") as f:
                f.write(r.content)
    else:
        raise ValueError("no existing audio to re-stitch against")

    # Re-stitch with new image
    video_path = os.path.join(tmp_dir, "video.mp4")
    await _run_stitch(new_image_url, audio_path, video_path, job_id, tmp_dir)

    # Upload new video only
    storage_path = f"{job_id}/video.mp4"
    upload_file(sb, settings.video_bucket, storage_path, video_path, "video/mp4")
    video_url = get_public_url(sb, settings.video_bucket, storage_path)
    sb.table("jobs").update(
        {"status": "completed", "video_url": video_url}
    ).eq("id", job_id).execute()
    publish_job_progress(job_id, 0, 0, "complete")

    shutil.rmtree(tmp_dir, ignore_errors=True)


async def _run_voiceover(story_text, audio_path, wpm, job_id):
    from voiceover import generate_voiceover
    publish_job_progress(job_id, 0, 0, "generating voiceover…")
    await asyncio.to_thread(
        generate_voiceover,
        story_text,
        audio_path,
        target_wpm=float(wpm),
        cache_dir=os.path.join(settings.tmp_dir, ".kokoro_cache"),
        use_cuda=settings.use_cuda,
    )


async def _run_stitch(image_url, audio_path, video_path, job_id, tmp_dir):
    import subprocess
    import httpx
    publish_job_progress(job_id, 0, 0, "stitching video…")
    image_path = os.path.join(tmp_dir, "image.jpg")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(image_url)
        r.raise_for_status()
        with open(image_path, "wb") as f:
            f.write(r.content)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-c:v", "libx264", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest", "-movflags", "+faststart",
        video_path,
    ]
    await asyncio.to_thread(subprocess.run, cmd, check=True, capture_output=True)


async def _upload_results(sb, job_id, audio_path, video_path):
    publish_job_progress(job_id, 0, 0, "uploading…")
    audio_storage = f"{job_id}/audio.mp3"
    video_storage = f"{job_id}/video.mp4"
    upload_file(sb, settings.audio_bucket, audio_storage, audio_path, "audio/mpeg")
    upload_file(sb, settings.video_bucket, video_storage, video_path, "video/mp4")
    audio_url = get_public_url(sb, settings.audio_bucket, audio_storage)
    video_url = get_public_url(sb, settings.video_bucket, video_storage)
    sb.table("jobs").update(
        {"status": "completed", "audio_url": audio_url, "video_url": video_url}
    ).eq("id", job_id).execute()
    publish_job_progress(job_id, 0, 0, "complete")


# ── ARQ worker settings ────────────────────────────────────────────────────────

class WorkerSettings:
    functions = [run_pipeline, run_edit_pipeline]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = settings.max_jobs
    job_timeout = 60 * 60  # 1 hour max per job


if __name__ == "__main__":
    import arq
    arq.run_worker(WorkerSettings)
