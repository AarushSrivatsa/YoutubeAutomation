"""
background_jobs.py — the actual job functions, run via FastAPI's BackgroundTasks
instead of a separate ARQ worker process. One process (uvicorn) does everything:
serves the API and runs pipeline/edit jobs in the background after the response
is sent.
"""
from __future__ import annotations
import asyncio
import logging
import os
import shutil

from logging_config import configure_logging
from config import get_settings
from core.graph import (
    run_graph,
    resume_graph,
    reset_thread,
    is_awaiting_approval,
    get_pending_state,
    patch_pending_state,
)
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

_JOB_SEMAPHORE = asyncio.Semaphore(settings.max_jobs)
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

        # run_pipeline always means "run the graph from START with this input" —
        # true both for a brand-new job and for a /retry of a previously-failed
        # one. But thread_id == job_id, and a retry reuses the same job_id, so
        # without clearing the old checkpoint first, LangGraph would silently
        # resume from wherever that thread's last run left off instead of
        # actually starting over (skipping the approval-gate interrupt,
        # carrying stale revision counters, etc.). No-op for a genuinely new
        # job that has no prior checkpoint yet.
        await asyncio.to_thread(reset_thread, job_id)

        initial_state = {
            "job_id": job_id,
            "prompt": prompt,
            "image_url": image_url,
            "wpm": wpm,
            "video_length_min": video_length_min,
            "voice_model": voice_model,
        }

        try:
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


# ── Approval gate ──────────────────────────────────────────────────────────────

async def _persist_pending_script(job_id: str, state: dict) -> None:
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


async def edit_pending_segment(job_id: str, segment_index: int, new_text: str) -> None:
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
        logger.exception("[job=%s] failed to sync edited segment to DB", job_id)

    publish_job_progress(job_id, 0, 0, f"segment {segment_index} updated, still awaiting approval")


async def _resume_and_finalize(job_id: str, log_label: str, progress_msg: str) -> None:
    """
    Shared tail end of every "call resume_graph() and see what happened" path.
    resume_graph() just does graph.invoke(None, config) — LangGraph figures out
    where a thread left off from its checkpoint, regardless of *why* it stopped
    there (deliberate interrupt_before gate vs. process getting killed mid-node).
    So the outcome-handling here is identical for both callers below:
      - approved script -> voiceover/stitch/supabase run, may finish or error
      - crash-recovered job -> whatever node it died in re-runs, then continues
        onward, and CAN legitimately land back on the approval gate again if
        the crash happened before the script was ever approved.
    """
    async with _JOB_SEMAPHORE:
        logger.info("[job=%s] %s", job_id, log_label)
        sb = get_supabase()

        try:
            update_job_status(sb, job_id, "running", error_message=None)
        except SupabaseError:
            logger.exception("[job=%s] could not mark job as running, continuing anyway", job_id)

        publish_job_progress(job_id, 0, 0, progress_msg)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(resume_graph, job_id),
                timeout=_JOB_TIMEOUT_SEC,
            )

            paused = await asyncio.to_thread(is_awaiting_approval, job_id)
            if paused:
                await _persist_pending_script(job_id, result)
                update_job_status(sb, job_id, "awaiting_approval", error_message=None)
                logger.info("[job=%s] landed back on the approval gate after resume", job_id)
                publish_job_progress(job_id, 0, 0, "awaiting_approval: script ready for review")
                return

            if result.get("error"):
                logger.error("[job=%s] pipeline failed after resume: %s", job_id, result["error"])
                update_job_status(sb, job_id, "failed", error_message=result["error"])
                publish_job_progress(job_id, 0, 0, f"failed: {result['error']}")

        except asyncio.TimeoutError:
            logger.error("[job=%s] pipeline timed out after resume (%ds)", job_id, _JOB_TIMEOUT_SEC)
            try:
                update_job_status(sb, job_id, "failed", error_message="pipeline timed out")
            except SupabaseError:
                logger.exception("[job=%s] could not mark timed-out job as failed", job_id)
            publish_job_progress(job_id, 0, 0, "failed: pipeline timed out")

        except Exception as e:
            logger.exception("[job=%s] %s crashed", job_id, log_label)
            try:
                update_job_status(sb, job_id, "failed", error_message=str(e))
            except SupabaseError:
                logger.exception("[job=%s] could not even mark job as failed", job_id)
            publish_job_progress(job_id, 0, 0, f"failed: {e}")


async def resume_pipeline_after_approval(job_id: str) -> None:
    await _resume_and_finalize(
        job_id,
        log_label="resuming pipeline after approval",
        progress_msg="approved — generating voiceover…",
    )


async def recover_interrupted_job(job_id: str) -> None:
    """
    Called from main.py's startup sweep for a job that was still 'running' when
    the previous process died (crash, OOM, deploy, Ctrl+C) but whose LangGraph
    checkpoint thread shows pending work (core.graph.is_resumable() == True).

    Unlike the old behavior (mark failed, force a full /retry from the original
    prompt), this continues the SAME thread_id from exactly the node it never
    finished — Postgres checkpointer already has every node's output up to
    that point, so story generation / voiceover / etc. already done doesn't
    get redone.
    """
    await _resume_and_finalize(
        job_id,
        log_label="recovering interrupted job from last checkpoint",
        progress_msg="recovered after restart — resuming from last checkpoint…",
    )


# ── Edit pipeline ──────────────────────────────────────────────────────────────

async def run_edit_pipeline(
    job_id: str,
    edit_type: str,
    segment_index: int | None = None,
    new_text: str | None = None,
    new_image_url: str | None = None,
) -> None:
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
    except Exception as e:
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

    voice_model = job.get("voice_model") or settings.default_voice
    publish_job_progress(job_id, 0, 0, "re-generating voiceover…")
    audio_path = os.path.join(tmp_dir, "audio.mp3")
    await _run_voiceover(story_text, audio_path, job.get("wpm", 150), voice_model, job_id)

    image_url = job.get("image_url", "")
    video_path = os.path.join(tmp_dir, "video.mp4")
    await _run_stitch(image_url, audio_path, video_path, job_id, tmp_dir)

    await _upload_results(sb, job_id, audio_path, video_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("[job=%s] segment edit complete", job_id)


async def _swap_image(sb, job_id, job, new_image_url, tmp_dir):
    logger.info("[job=%s] swapping image -> %s", job_id, new_image_url)

    try:
        sb.table("jobs").update({"image_url": new_image_url}).eq("id", job_id).execute()
    except Exception as e:
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
        raise SupabaseError(f"failed to update job record: {e}", e) from e

    publish_job_progress(job_id, 0, 0, "complete")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("[job=%s] image swap complete", job_id)


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
    except Exception as e:
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
        raise HTTPFetchError(f"failed to download image {image_url}: {e}", e) from e

    # pad filter rounds width/height up to even numbers — libx264 requires this.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
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
        raise MediaProcessingError(f"ffmpeg stitch failed: {stderr}", e) from e
    except FileNotFoundError as e:
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
        raise SupabaseError(f"failed to update job record: {e}", e) from e

    logger.info("[job=%s] upload complete", job_id)
    publish_job_progress(job_id, 0, 0, "complete")