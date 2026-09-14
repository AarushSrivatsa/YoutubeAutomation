"""
supabase_service.py — thin wrappers around supabase-py v2 for storage and DB operations.
The client is created per-call (supabase-py v2 is not thread-safe for the same instance).
For high-concurrency workloads, consider using the Postgres connection pool directly.

Every function that talks to Supabase (storage or DB) is wrapped in a try/except.
Failures are logged with full context and re-raised as SupabaseError so callers can
decide whether to fail the whole job, or (for best-effort steps like embeddings)
swallow it.
"""
from __future__ import annotations
import logging
from supabase import create_client, Client
from config import get_settings
from errors import SupabaseError

logger = logging.getLogger(__name__)
settings = get_settings()


def get_supabase() -> Client:
    logger.debug("Creating Supabase client for %s", settings.supabase_url)
    try:
        return create_client(settings.supabase_url, settings.supabase_service_key)
    except Exception as e:
        logger.exception("Failed to create Supabase client")
        raise SupabaseError("failed to create client", e) from e


# ── Storage helpers ────────────────────────────────────────────────────────────

def upload_file(
    sb: Client,
    bucket: str,
    storage_path: str,
    local_path: str,
    content_type: str,
) -> None:
    """Upload a local file to a Supabase storage bucket."""
    logger.info("Uploading local file %s -> bucket=%s path=%s", local_path, bucket, storage_path)
    try:
        with open(local_path, "rb") as f:
            data = f.read()
    except OSError as e:
        logger.exception("Could not read local file %s before upload", local_path)
        raise SupabaseError(f"could not read local file {local_path}: {e}", e) from e

    try:
        sb.storage.from_(bucket).upload(
            storage_path,
            data,
            {"content-type": content_type, "upsert": "true"},
        )
        logger.info("Upload complete: bucket=%s path=%s (%d bytes)", bucket, storage_path, len(data))
    except Exception as e:
        logger.exception("Supabase storage upload failed: bucket=%s path=%s", bucket, storage_path)
        raise SupabaseError(f"upload_file failed for {bucket}/{storage_path}: {e}", e) from e


def upload_bytes(
    sb: Client,
    bucket: str,
    storage_path: str,
    data: bytes,
    content_type: str,
) -> None:
    """Upload raw bytes to a Supabase storage bucket."""
    logger.info("Uploading %d bytes -> bucket=%s path=%s", len(data), bucket, storage_path)
    try:
        sb.storage.from_(bucket).upload(
            storage_path,
            data,
            {"content-type": content_type, "upsert": "true"},
        )
        logger.info("Upload complete: bucket=%s path=%s", bucket, storage_path)
    except Exception as e:
        logger.exception("Supabase storage upload_bytes failed: bucket=%s path=%s", bucket, storage_path)
        raise SupabaseError(f"upload_bytes failed for {bucket}/{storage_path}: {e}", e) from e


def get_public_url(sb: Client, bucket: str, storage_path: str) -> str:
    try:
        result = sb.storage.from_(bucket).get_public_url(storage_path)
        logger.debug("Resolved public URL for %s/%s: %s", bucket, storage_path, result)
        return result
    except Exception as e:
        logger.exception("Failed to resolve public URL for %s/%s", bucket, storage_path)
        raise SupabaseError(f"get_public_url failed for {bucket}/{storage_path}: {e}", e) from e


def download_file_bytes(sb: Client, bucket: str, storage_path: str) -> bytes:
    logger.info("Downloading bucket=%s path=%s", bucket, storage_path)
    try:
        data = sb.storage.from_(bucket).download(storage_path)
        logger.info("Downloaded %d bytes from %s/%s", len(data), bucket, storage_path)
        return data
    except Exception as e:
        logger.exception("Supabase storage download failed: bucket=%s path=%s", bucket, storage_path)
        raise SupabaseError(f"download_file_bytes failed for {bucket}/{storage_path}: {e}", e) from e


# ── Job DB helpers ─────────────────────────────────────────────────────────────

def create_job_record(
    sb: Client,
    job_id: str,
    prompt: str,
    image_url: str,
    wpm: int,
    video_length_min: float,
    voice_model: str = "af_heart",
) -> dict:
    logger.info("Creating job record %s", job_id)
    try:
        result = sb.table("jobs").insert(
            {
                "id": job_id,
                "status": "pending",
                "prompt": prompt,
                "image_url": image_url,
                "wpm": wpm,
                "video_length_min": video_length_min,
                "voice_model": voice_model,
                "thread_id": job_id,
            }
        ).execute()
        logger.info("Job record %s created", job_id)
        return result.data[0] if result.data else {}
    except Exception as e:
        logger.exception("Failed to create job record %s", job_id)
        raise SupabaseError(f"create_job_record failed for job {job_id}: {e}", e) from e


def get_job_record(sb: Client, job_id: str) -> dict | None:
    logger.debug("Fetching job record %s", job_id)
    try:
        result = sb.table("jobs").select("*").eq("id", job_id).single().execute()
        return result.data
    except Exception as e:
        # PGRST116 = "no rows / multiple rows" from PostgREST's .single() — treat as not-found.
        if "PGRST116" in str(e):
            logger.warning("Job record %s not found", job_id)
            return None
        logger.exception("Failed to fetch job record %s", job_id)
        raise SupabaseError(f"get_job_record failed for job {job_id}: {e}", e) from e


def try_claim_job(sb: Client, job_id: str, expected_status: str, new_status: str) -> bool:
    """
    Atomically transitions a job's status ONLY if it is currently `expected_status`,
    via a single conditional UPDATE (WHERE id=... AND status=...).

    Why this exists: endpoints like /approve and /retry used to do a plain
    read (get_job_with_story_and_segments) followed by a status check, then
    scheduled a background pipeline run. That's a classic read-then-write
    race — two near-simultaneous requests (a double-click, a retried HTTP
    call, etc.) can both read the same starting status before either write
    lands, both pass the check, and both get scheduled. Both then invoke the
    LangGraph pipeline for the *same* job_id/thread_id concurrently, and
    since every node writes to the same tmp_dir (keyed only by job_id), the
    two runs can stomp on each other's image/audio/video files mid-write —
    this is what produced the "height not divisible by 2" ffmpeg crash
    right after a job had already completed successfully once.

    Returns True if THIS call won the claim (i.e. it performed the update),
    False if another request already transitioned the job first — in which
    case the caller should reject the request instead of scheduling
    anything.
    """
    logger.info("Attempting to claim job %s: %s -> %s", job_id, expected_status, new_status)
    try:
        result = (
            sb.table("jobs")
            .update({"status": new_status})
            .eq("id", job_id)
            .eq("status", expected_status)
            .execute()
        )
        claimed = bool(result.data)
        logger.info(
            "Claim %s for job %s (%s -> %s)",
            "succeeded" if claimed else "lost (already transitioned by another request)",
            job_id, expected_status, new_status,
        )
        return claimed
    except Exception as e:
        logger.exception("try_claim_job failed for job %s", job_id)
        raise SupabaseError(f"try_claim_job failed for job {job_id}: {e}", e) from e


def update_job_status(sb: Client, job_id: str, status: str, **kwargs) -> None:
    logger.info("Updating job %s -> status=%s extra=%s", job_id, status, list(kwargs.keys()))
    update_data = {"status": status, **kwargs}
    try:
        sb.table("jobs").update(update_data).eq("id", job_id).execute()
        logger.info("Job %s status updated to %s", job_id, status)
    except Exception as e:
        logger.exception("Failed to update job %s status to %s", job_id, status)
        raise SupabaseError(f"update_job_status failed for job {job_id}: {e}", e) from e


def list_jobs_by_status(sb: Client, status: str) -> list[dict]:
    """Used on startup to find jobs orphaned by a server restart mid-job."""
    logger.debug("Listing jobs with status=%s", status)
    try:
        result = sb.table("jobs").select("id, prompt, status, updated_at").eq("status", status).execute()
        return result.data or []
    except Exception as e:
        logger.exception("Failed to list jobs with status=%s", status)
        raise SupabaseError(f"list_jobs_by_status failed for status {status}: {e}", e) from e


def get_job_with_story_and_segments(sb: Client, job_id: str) -> dict | None:
    logger.debug("Fetching job+story+segments for %s", job_id)
    job = get_job_record(sb, job_id)
    if not job:
        return None

    try:
        story_res = sb.table("stories").select("*").eq("job_id", job_id).execute()
        story = story_res.data[0] if story_res.data else None
    except Exception as e:
        logger.exception("Failed to fetch story for job %s", job_id)
        raise SupabaseError(f"failed to fetch story for job {job_id}: {e}", e) from e

    try:
        segs_res = (
            sb.table("segments")
            .select("*")
            .eq("job_id", job_id)
            .order("segment_index")
            .execute()
        )
        segments = segs_res.data or []
    except Exception as e:
        logger.exception("Failed to fetch segments for job %s", job_id)
        raise SupabaseError(f"failed to fetch segments for job {job_id}: {e}", e) from e

    return {"job": job, "story": story, "segments": segments}


def list_jobs(sb: Client, limit: int = 20, offset: int = 0) -> list[dict]:
    logger.debug("Listing jobs limit=%d offset=%d", limit, offset)
    try:
        result = (
            sb.table("jobs")
            .select("id, status, prompt, video_url, created_at")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.exception("Failed to list jobs")
        raise SupabaseError(f"list_jobs failed: {e}", e) from e


def similarity_search(sb: Client, embedding: list[float], threshold: float = 0.7, limit: int = 10) -> list[dict]:
    logger.debug("Running similarity_search threshold=%s limit=%d", threshold, limit)
    try:
        result = sb.rpc(
            "match_jobs",
            {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": limit,
            },
        ).execute()
        return result.data or []
    except Exception as e:
        logger.exception("similarity_search RPC failed")
        raise SupabaseError(f"similarity_search failed: {e}", e) from e


# ── Story/segment persistence (shared by supabase_node and the approval-gate path) ──

def upsert_story_record(
    sb: Client,
    job_id: str,
    title: str,
    story_arc: str,
    characters: list,
    locations: list,
    segment_plans: dict,
    word_count: int,
) -> None:
    logger.info("Upserting story record for job %s", job_id)
    try:
        sb.table("stories").upsert(
            {
                "job_id": job_id,
                "title": title,
                "story_arc": story_arc,
                "characters": characters,
                "locations": locations,
                "segment_plans": segment_plans,
                "word_count": word_count,
            },
            on_conflict="job_id",
        ).execute()
    except Exception as e:
        logger.exception("Failed to upsert story record for job %s", job_id)
        raise SupabaseError(f"upsert_story_record failed for job {job_id}: {e}", e) from e


def upsert_segment_records(sb: Client, job_id: str, segments: list[dict]) -> None:
    if not segments:
        return
    logger.info("Upserting %d segment record(s) for job %s", len(segments), job_id)
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
        logger.exception("Failed to upsert segments for job %s", job_id)
        raise SupabaseError(f"upsert_segment_records failed for job {job_id}: {e}", e) from e