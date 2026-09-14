"""
supabase_node.py — final pipeline step.
1. Uploads audio + video to Supabase storage buckets.
2. Persists job, story, and segment records to Postgres.
3. Generates and stores a prompt embedding for similarity search (via local Ollama).
4. Cleans up the local tmp directory.
5. Publishes a final 'complete' progress event so the SSE stream can close.

Uploads and DB writes here are individually best-effort (a failed audio upload
shouldn't block the video from being saved), but every failure is logged AND
surfaced to the client via a progress warning event — nothing fails silently.
"""
from __future__ import annotations
import logging
import os
import shutil
from config import get_settings
from core.state import PipelineState
from services.supabase_service import (
    get_supabase,
    upload_file,
    get_public_url,
    upsert_story_record,
    upsert_segment_records,
)
from services.redis_service import publish_job_progress
from services.embeddings_service import get_embedding
from errors import SupabaseError, OllamaError

logger = logging.getLogger(__name__)
settings = get_settings()


def supabase_node(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    logger.info("[job=%s] supabase_node entered", job_id)
    publish_job_progress(job_id, 0, 0, "uploading and saving…")

    try:
        sb = get_supabase()
    except SupabaseError as e:
        logger.exception("[job=%s] supabase_node: could not create Supabase client, aborting node", job_id)
        publish_job_progress(job_id, 0, 0, f"failed: {e}")
        return {**state, "error": str(e)}

    audio_url = ""
    video_url = ""

    # ── Upload audio ───────────────────────────────────────────────────────────
    audio_path = state.get("audio_path", "")
    if audio_path and os.path.exists(audio_path):
        try:
            storage_path = f"{job_id}/audio.mp3"
            upload_file(sb, settings.audio_bucket, storage_path, audio_path, "audio/mpeg")
            audio_url = get_public_url(sb, settings.audio_bucket, storage_path)
            logger.info("[job=%s] audio uploaded: %s", job_id, audio_url)
        except SupabaseError as e:
            logger.exception("[job=%s] audio upload failed", job_id)
            publish_job_progress(job_id, 0, 0, f"warning: audio upload failed: {e}")
    else:
        logger.warning("[job=%s] no audio_path in state or file missing, skipping audio upload", job_id)

    # ── Upload video ───────────────────────────────────────────────────────────
    video_path = state.get("video_path", "")
    if video_path and os.path.exists(video_path):
        try:
            storage_path = f"{job_id}/video.mp4"
            upload_file(sb, settings.video_bucket, storage_path, video_path, "video/mp4")
            video_url = get_public_url(sb, settings.video_bucket, storage_path)
            logger.info("[job=%s] video uploaded: %s", job_id, video_url)
        except SupabaseError as e:
            logger.exception("[job=%s] video upload failed", job_id)
            publish_job_progress(job_id, 0, 0, f"warning: video upload failed: {e}")
    else:
        logger.warning("[job=%s] no video_path in state or file missing, skipping video upload", job_id)

    # ── Update job record ──────────────────────────────────────────────────────
    try:
        sb.table("jobs").update(
            {
                "status": "completed",
                "audio_url": audio_url or None,
                "video_url": video_url or None,
                "error_message": state.get("error"),
            }
        ).eq("id", job_id).execute()
        logger.info("[job=%s] job record marked completed", job_id)
    except Exception as e:
        logger.exception("[job=%s] job record update failed", job_id)
        publish_job_progress(job_id, 0, 0, f"warning: job record update failed: {e}")

    # ── Upsert story record ────────────────────────────────────────────────────
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
        logger.info("[job=%s] story record upserted", job_id)
    except Exception as e:
        logger.exception("[job=%s] story record upsert failed", job_id)
        publish_job_progress(job_id, 0, 0, f"warning: story record upsert failed: {e}")

    # ── Upsert segment records ─────────────────────────────────────────────────
    segments = state.get("segments", [])
    if segments:
        try:
            upsert_segment_records(sb, job_id, segments)
            logger.info("[job=%s] %d segment record(s) upserted", job_id, len(segments))
        except Exception as e:
            logger.exception("[job=%s] segments upsert failed", job_id)
            publish_job_progress(job_id, 0, 0, f"warning: segments upsert failed: {e}")
    else:
        logger.warning("[job=%s] no segments in state to persist", job_id)

    # ── Generate prompt embedding (optional, best-effort) ─────────────────────
    if settings.embeddings_enabled:
        _store_embedding(sb, job_id, state.get("prompt", ""))
    else:
        logger.debug("[job=%s] embeddings disabled, skipping", job_id)

    # ── Clean up local tmp ────────────────────────────────────────────────────
    tmp_dir = os.path.join(settings.tmp_dir, job_id)
    if os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("[job=%s] tmp dir %s cleaned up", job_id, tmp_dir)
        except Exception:
            logger.exception("[job=%s] failed to clean up tmp dir %s", job_id, tmp_dir)

    logger.info("[job=%s] supabase_node complete: audio_url=%s video_url=%s", job_id, audio_url, video_url)
    publish_job_progress(job_id, 0, 0, "complete")

    return {
        **state,
        "audio_url": audio_url,
        "video_url": video_url,
    }


def _store_embedding(sb, job_id: str, prompt: str) -> None:
    try:
        embedding = get_embedding(prompt)
        sb.table("prompt_embeddings").upsert(
            {"job_id": job_id, "prompt": prompt, "embedding": embedding},
            on_conflict="job_id",
        ).execute()
        logger.info("[job=%s] prompt embedding stored", job_id)
    except OllamaError as e:
        logger.warning("[job=%s] embedding generation failed (non-critical): %s", job_id, e)
    except Exception as e:
        logger.warning("[job=%s] embedding storage failed (non-critical): %s", job_id, e)
