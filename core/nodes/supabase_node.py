"""
supabase_node.py — final pipeline step.
1. Uploads audio + video to Supabase storage buckets.
2. Persists job, story, and segment records to Postgres.
3. Generates and stores a prompt embedding for similarity search (via local Ollama).
4. Cleans up the local tmp directory.
5. Publishes a final 'complete' progress event so the SSE stream can close.
"""
from __future__ import annotations
import json
import os
import shutil
from config import get_settings
from core.state import PipelineState
from services.supabase_service import (
    get_supabase,
    upload_file,
    get_public_url,
)
from services.redis_service import publish_job_progress
from services.embeddings_service import get_embedding

settings = get_settings()


def supabase_node(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    publish_job_progress(job_id, 0, 0, "uploading and saving…")

    sb = get_supabase()
    audio_url = ""
    video_url = ""

    # ── Upload audio ───────────────────────────────────────────────────────────
    audio_path = state.get("audio_path", "")
    if audio_path and os.path.exists(audio_path):
        try:
            storage_path = f"{job_id}/audio.mp3"
            upload_file(sb, settings.audio_bucket, storage_path, audio_path, "audio/mpeg")
            audio_url = get_public_url(sb, settings.audio_bucket, storage_path)
        except Exception as e:
            publish_job_progress(job_id, 0, 0, f"warning: audio upload failed: {e}")

    # ── Upload video ───────────────────────────────────────────────────────────
    video_path = state.get("video_path", "")
    if video_path and os.path.exists(video_path):
        try:
            storage_path = f"{job_id}/video.mp4"
            upload_file(sb, settings.video_bucket, storage_path, video_path, "video/mp4")
            video_url = get_public_url(sb, settings.video_bucket, storage_path)
        except Exception as e:
            publish_job_progress(job_id, 0, 0, f"warning: video upload failed: {e}")

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
    except Exception as e:
        publish_job_progress(job_id, 0, 0, f"warning: job record update failed: {e}")

    # ── Upsert story record ────────────────────────────────────────────────────
    try:
        sb.table("stories").upsert(
            {
                "job_id": job_id,
                "title": state.get("title", ""),
                "story_arc": state.get("story_arc", ""),
                "characters": state.get("characters", []),
                "locations": state.get("locations", []),
                "segment_plans": state.get("segment_plans", {}),
                "word_count": len((state.get("story_text", "")).split()),
            },
            on_conflict="job_id",
        ).execute()
    except Exception as e:
        publish_job_progress(job_id, 0, 0, f"warning: story record upsert failed: {e}")

    # ── Upsert segment records ─────────────────────────────────────────────────
    segments = state.get("segments", [])
    if segments:
        try:
            rows = [
                {
                    "job_id": job_id,
                    "segment_index": s["id"],
                    "text": s["text"],
                    "original_text": s.get("original_text", s["text"]),
                    "status": s.get("status", "kept"),
                }
                for s in segments
            ]
            sb.table("segments").upsert(rows, on_conflict="job_id,segment_index").execute()
        except Exception as e:
            publish_job_progress(job_id, 0, 0, f"warning: segments upsert failed: {e}")

    # ── Generate prompt embedding (optional) ──────────────────────────────────
    if settings.embeddings_enabled:
        _store_embedding(sb, job_id, state.get("prompt", ""))

    # ── Clean up local tmp ────────────────────────────────────────────────────
    tmp_dir = os.path.join(settings.tmp_dir, job_id)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

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
    except Exception:
        pass  # embedding is non-critical
