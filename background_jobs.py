"""
background_jobs.py — the actual job functions, run via FastAPI's BackgroundTasks
instead of a separate ARQ worker process. One process (uvicorn) does everything:
serves the API and runs pipeline/edit jobs in the background after the response
is sent.

Trade-off you're accepting by doing it this way: if the API process restarts
mid-job, that job is just gone (ARQ + Redis would have retried it from its
queue). For a single-instance deploy that's usually fine. If you ever need
multi-instance horizontal scaling or job retries, this is the piece you'd
swap back to a real queue.

A semaphore caps how many jobs run at once (settings.max_jobs) so multiple
concurrent TTS + ffmpeg jobs don't fight over CPU/RAM on the same box.
"""
from __future__ import annotations
import asyncio
import logging
import os
import shutil

from logging_config import configure_logging
from config import get_settings
from core.graph import run_graph, resume_graph, is_awaiting_approval, get_pending_state, patch_pending_state
from services.supabase_service import (
    get_supabase,
    update_job_status,
    get_job_with_story_and_segments,
    upload_file,
    get_public_url,
    upsert_story_record,
    upsert_segment_records,
)
from services.redis_service import publish_job_progress
from errors import SupabaseError, HTTPFetchError, MediaProcessingError

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

# Caps how many pipeline/edit jobs run concurrently in this process.
_JOB_SEMAPHORE = asyncio.Semaphore(settings.max_jobs)

# Hard ceiling so a stuck job can't hang forever and quietly eat a worker slot.
_JOB_TIMEOUT_SEC = 60 * 60  # 1 hour


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_pipeline(
    job_id: str,
    prompt: str,
    image_url: str,
    wpm: int = 150,
    video_length_min: float = 10.0,
    voice_model: str = "af_heart",
) -> None:
    async with _JOB_SEMAPHORE:
        logger.info("[job=%s] run_pipeline starting (wpm=%s, video_length_min=%s, voice_model=%s)",
                    job_id, wpm, video_length_min, voice_model)
        sb = get_supabase()

        try:
            update_job_status(sb, job_id, "running", error_message=None)
        except SupabaseError:
            logger.exception("[job=%s] could not mark job as running, continuing anyway", job_id)

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
            # Run the full LangGraph pipeline in a thread (all nodes are sync) with
            # a hard timeout so one stuck job can't hold its semaphore slot forever.
            # This stops either at the approval gate (before voiceover) or at the
            # real end of the pipeline — see is_awaiting_approval below.
            result = await asyncio.wait_for(
                asyncio.to_thread(run_graph, job_id, initial_state),
                timeout=_JOB_TIMEOUT_SEC,
            )

            paused = await asyncio.to_thread(is_awaiting_approval, job_id)
            if paused:
                await _persist_pending_script(job_id, result)
                update_job_status(sb, job_id, "awaiting_approval", error_message=None)
                logger.info("[job=%s] script ready, awaiting approval before voiceover", job_id)
                publish_job_progress(job_id, 0, 0, "awaiting_approval: script ready for review")
                return

            if result.get("error"):
                logger.error("[job=%s] pipeline finished with an error: %s", job_id, result["error"])
                update_job_status(sb, job_id, "failed", error_message=result["error"])
                publish_job_progress(job_id, 0, 0, f"failed: {result['error']}")
            else:
                logger.info("[job=%s] pipeline finished successfully", job_id)
            # supabase_node already updates the record to 'completed' on success

        except asyncio.TimeoutError:
            logger.error("[job=%s] pipeline timed out after %ds", job_id, _JOB_TIMEOUT_SEC)
            try:
                update_job_status(sb, job_id, "failed", error_message="pipeline timed out")
            except SupabaseError:
                logger.exception("[job=%s] could not mark timed-out job as failed", job_id)
            publish_job_progress(job_id, 0, 0, "failed: pipeline timed out")

        except Exception as e:
            logger.exception("[job=%s] run_pipeline crashed", job_id)
            try:
                update_job_status(sb, job_id, "failed", error_message=str(e))
            except SupabaseError:
                logger.exception("[job=%s] could not even mark job as failed", job_id)
            publish_job_progress(job_id, 0, 0, f"failed: {e}")


# ── Approval gate (script review before voiceover) ─────────────────────────────

async def _persist_pending_script(job_id: str, state: dict) -> None:
    """
    Called when the graph pauses at the approval gate. Writes the generated
    story/segments to the DB right away so GET /jobs/{job_id} can show the
    script for review — otherwise it only exists inside the LangGraph
    checkpoint, which the API never reads directly.
    """
    sb = get_supabase()
    try:
        upsert_story_record(
            sb, job_id,
            title=state.get("title", ""),
            story_arc=state.get("story_arc", ""),
            characters=state.get("characters", []),
            locations=state.get("locations", []),
            segment_plans=state.get("segment_plans", {}),
            word_count=len((state.get("story_text", "")).split()),
        )
    except SupabaseError:
        logger.exception("[job=%s] failed to persist pending story record", job_id)

    segments = state.get("segments", [])
    if segments:
        try:
            upsert_segment_records(sb, job_id, segments)
        except SupabaseError:
            logger.exception("[job=%s] failed to persist pending segment records", job_id)
    else:
        logger.warning("[job=%s] approval gate reached with no segments to persist", job_id)


async def edit_pending_segment(job_id: str, segment_index: int, new_text: str) -> None:
    """
    Edits a segment's text while the job is sitting at the approval gate.
    No voiceover/stitch has happened yet, so this just patches the checkpointed
    graph state (so the edit is picked up when resumed) and keeps the DB copy
    of the script in sync for anyone reading it via GET /jobs/{job_id}.
    """
    from core.story_gen import assemble_story

    logger.info("[job=%s] editing pending segment %s (pre-approval)", job_id, segment_index)

    state = await asyncio.to_thread(get_pending_state, job_id)
    if not state:
        raise ValueError(f"job {job_id} has no pending script to edit")

    segments = list(state.get("segments", []))
    found = False
    for s in segments:
        if s["id"] == segment_index:
            s["text"] = new_text
            s["status"] = "edited"
            found = True
            break
    if not found:
        raise ValueError(f"segment {segment_index} not found in pending script")

    assembled = assemble_story(segments, state.get("title", ""))

    await asyncio.to_thread(
        patch_pending_state, job_id, {"segments": segments, "story_text": assembled["story_text"]}
    )

    sb = get_supabase()
    try:
        await asyncio.to_thread(upsert_segment_records, sb, job_id, segments)
    except SupabaseError:
        logger.exception("[job=%s] failed to sync edited segment to DB (graph state was still patched)", job_id)

    publish_job_progress(job_id, 0, 0, f"segment {segment_index} updated, still awaiting approval")
    logger.info("[job=%s] pending segment %s edited successfully", job_id, segment_index)


async def resume_pipeline_after_approval(job_id: str) -> None:
    """
    Resumes a job sitting at the approval gate — continues on to voiceover,
    stitch, and upload using whatever script is currently checkpointed
    (including any edits made via PATCH /jobs/{job_id}/segments/{idx} while
    it was awaiting approval).
    """
    async with _JOB_SEMAPHORE:
        logger.info("[job=%s] resuming pipeline after approval", job_id)
        sb = get_supabase()

        try:
            update_job_status(sb, job_id, "running", error_message=None)
        except SupabaseError:
            logger.exception("[job=%s] could not mark job as running, continuing anyway", job_id)

        publish_job_progress(job_id, 0, 0, "approved — generating voiceover…")

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(resume_graph, job_id),
                timeout=_JOB_TIMEOUT_SEC,
            )

            if result.get("error"):
                logger.error("[job=%s] pipeline failed after approval: %s", job_id, result["error"])
                update_job_status(sb, job_id, "failed", error_message=result["error"])
                publish_job_progress(job_id, 0, 0, f"failed: {result['error']}")
            else:
                logger.info("[job=%s] pipeline finished successfully after approval", job_id)

        except asyncio.TimeoutError:
            logger.error("[job=%s] pipeline timed out after approval (%ds)", job_id, _JOB_TIMEOUT_SEC)
            try:
                update_job_status(sb, job_id, "failed", error_message="pipeline timed out")
            except SupabaseError:
                logger.exception("[job=%s] could not mark timed-out job as failed", job_id)
            publish_job_progress(job_id, 0, 0, "failed: pipeline timed out")

        except Exception as e:
            logger.exception("[job=%s] resume_pipeline_after_approval crashed", job_id)
            try:
                update_job_status(sb, job_id, "failed", error_message=str(e))
            except SupabaseError:
                logger.exception("[job=%s] could not even mark job as failed", job_id)
            publish_job_progress(job_id, 0, 0, f"failed: {e}")


# ── Edit pipeline ──────────────────────────────────────────────────────────────

async def run_edit_pipeline(
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
    async with _JOB_SEMAPHORE:
        logger.info("[job=%s] run_edit_pipeline starting (edit_type=%s, segment_index=%s)",
                    job_id, edit_type, segment_index)
        sb = get_supabase()

        try:
            update_job_status(sb, job_id, "running", error_message=None)
        except SupabaseError:
            logger.exception("[job=%s] could not mark job as running, continuing anyway", job_id)

        publish_job_progress(job_id, 0, 0, f"edit started ({edit_type})")

        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(get_job_with_story_and_segments, sb, job_id),
                timeout=30,
            )
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

            logger.info("[job=%s] run_edit_pipeline finished successfully", job_id)

        except Exception as e:
            logger.exception("[job=%s] run_edit_pipeline failed", job_id)
            try:
                update_job_status(sb, job_id, "failed", error_message=str(e))
            except SupabaseError:
                logger.exception("[job=%s] could not even mark job as failed", job_id)
            publish_job_progress(job_id, 0, 0, f"edit failed: {e}")


async def _edit_segment(sb, job_id, job, segments, segment_index, new_text, tmp_dir):
    from core.story_gen import assemble_story

    logger.info("[job=%s] editing segment %s", job_id, segment_index)

    try:
        sb.table("segments").update(
            {"text": new_text, "status": "edited"}
        ).eq("job_id", job_id).eq("segment_index", segment_index).execute()
        logger.info("[job=%s] segment %s text updated in DB", job_id, segment_index)
    except Exception as e:
        logger.exception("[job=%s] failed to update segment %s in DB", job_id, segment_index)
        raise SupabaseError(f"failed to update segment {segment_index}: {e}", e) from e

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
    logger.info("[job=%s] story re-assembled after edit (%d words)", job_id, assembled["word_count"])

    # Preserve the job's original voice — don't silently fall back to the global default.
    voice_model = job.get("voice_model") or settings.default_voice
    publish_job_progress(job_id, 0, 0, "re-generating voiceover…")
    audio_path = os.path.join(tmp_dir, "audio.mp3")
    await _run_voiceover(story_text, audio_path, job.get("wpm", 150), voice_model, job_id)

    image_url = job.get("image_url", "")
    video_path = os.path.join(tmp_dir, "video.mp4")
    await _run_stitch(image_url, audio_path, video_path, job_id, tmp_dir)

    await _upload_results(sb, job_id, audio_path, video_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("[job=%s] segment edit complete, tmp dir cleaned up", job_id)


async def _swap_image(sb, job_id, job, new_image_url, tmp_dir):
    logger.info("[job=%s] swapping image -> %s", job_id, new_image_url)

    try:
        sb.table("jobs").update({"image_url": new_image_url}).eq("id", job_id).execute()
    except Exception as e:
        logger.exception("[job=%s] failed to update image_url in DB", job_id)
        raise SupabaseError(f"failed to update image_url: {e}", e) from e

    publish_job_progress(job_id, 0, 0, "downloading existing audio…")
    audio_url = job.get("audio_url", "")
    audio_path = os.path.join(tmp_dir, "audio.mp3")
    if not audio_url:
        raise ValueError("no existing audio to re-stitch against")

    try:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(audio_url)
            r.raise_for_status()
            with open(audio_path, "wb") as f:
                f.write(r.content)
        logger.info("[job=%s] existing audio downloaded (%d bytes)", job_id, len(r.content))
    except Exception as e:
        logger.exception("[job=%s] failed to download existing audio from %s", job_id, audio_url)
        raise HTTPFetchError(f"failed to download existing audio: {e}", e) from e

    video_path = os.path.join(tmp_dir, "video.mp4")
    await _run_stitch(new_image_url, audio_path, video_path, job_id, tmp_dir)

    storage_path = f"{job_id}/video.mp4"
    upload_file(sb, settings.video_bucket, storage_path, video_path, "video/mp4")
    video_url = get_public_url(sb, settings.video_bucket, storage_path)

    try:
        sb.table("jobs").update(
            {"status": "completed", "video_url": video_url}
        ).eq("id", job_id).execute()
    except Exception as e:
        logger.exception("[job=%s] failed to update job record after image swap", job_id)
        raise SupabaseError(f"failed to update job record: {e}", e) from e

    publish_job_progress(job_id, 0, 0, "complete")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("[job=%s] image swap complete, tmp dir cleaned up", job_id)


async def _run_voiceover(story_text, audio_path, wpm, voice_model, job_id):
    from core.voiceover import generate_voiceover
    logger.info("[job=%s] regenerating voiceover (voice=%s, wpm=%s)", job_id, voice_model, wpm)
    publish_job_progress(job_id, 0, 0, "generating voiceover…")
    try:
        await asyncio.to_thread(
            generate_voiceover,
            story_text,
            audio_path,
            voice_model=voice_model,
            target_wpm=float(wpm),
            cache_dir=os.path.join(settings.tmp_dir, ".kokoro_cache"),
            use_cuda=settings.use_cuda,
        )
        logger.info("[job=%s] voiceover regeneration complete", job_id)
    except MediaProcessingError:
        logger.exception("[job=%s] voiceover regeneration failed", job_id)
        raise
    except Exception as e:
        logger.exception("[job=%s] voiceover regeneration failed (unexpected)", job_id)
        raise MediaProcessingError(f"voiceover regeneration failed: {e}", e) from e


async def _run_stitch(image_url, audio_path, video_path, job_id, tmp_dir):
    import subprocess
    import httpx
    logger.info("[job=%s] re-stitching video (image=%s)", job_id, image_url)
    publish_job_progress(job_id, 0, 0, "stitching video…")
    image_path = os.path.join(tmp_dir, "image.jpg")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(image_url)
            r.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(r.content)
        logger.info("[job=%s] image downloaded for stitching (%d bytes)", job_id, len(r.content))
    except Exception as e:
        logger.exception("[job=%s] failed to download image for stitching", job_id)
        raise HTTPFetchError(f"failed to download image {image_url}: {e}", e) from e

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
    try:
        await asyncio.to_thread(subprocess.run, cmd, check=True, capture_output=True)
        logger.info("[job=%s] ffmpeg stitch complete -> %s", job_id, video_path)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace")[:500] if e.stderr else ""
        logger.exception("[job=%s] ffmpeg stitch failed: %s", job_id, stderr)
        raise MediaProcessingError(f"ffmpeg stitch failed: {stderr}", e) from e
    except FileNotFoundError as e:
        logger.exception("[job=%s] ffmpeg binary not found", job_id)
        raise MediaProcessingError("ffmpeg is not installed or not on PATH", e) from e


async def _upload_results(sb, job_id, audio_path, video_path):
    logger.info("[job=%s] uploading regenerated audio+video", job_id)
    publish_job_progress(job_id, 0, 0, "uploading…")
    audio_storage = f"{job_id}/audio.mp3"
    video_storage = f"{job_id}/video.mp4"

    upload_file(sb, settings.audio_bucket, audio_storage, audio_path, "audio/mpeg")
    upload_file(sb, settings.video_bucket, video_storage, video_path, "video/mp4")
    audio_url = get_public_url(sb, settings.audio_bucket, audio_storage)
    video_url = get_public_url(sb, settings.video_bucket, video_storage)

    try:
        sb.table("jobs").update(
            {"status": "completed", "audio_url": audio_url, "video_url": video_url}
        ).eq("id", job_id).execute()
    except Exception as e:
        logger.exception("[job=%s] failed to update job record after upload", job_id)
        raise SupabaseError(f"failed to update job record: {e}", e) from e

    logger.info("[job=%s] upload complete: audio_url=%s video_url=%s", job_id, audio_url, video_url)
    publish_job_progress(job_id, 0, 0, "complete")
