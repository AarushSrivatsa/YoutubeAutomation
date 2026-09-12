"""
supabase_service.py — thin wrappers around supabase-py v2 for storage and DB operations.
The client is created per-call (supabase-py v2 is not thread-safe for the same instance).
For high-concurrency workloads, consider using the Postgres connection pool directly.
"""
from __future__ import annotations
import os
from supabase import create_client, Client
from config import get_settings

settings = get_settings()


def get_supabase() -> Client:
    return create_client(settings.supabase_url, settings.supabase_service_key)


# ── Storage helpers ────────────────────────────────────────────────────────────

def upload_file(
    sb: Client,
    bucket: str,
    storage_path: str,
    local_path: str,
    content_type: str,
) -> None:
    """Upload a local file to a Supabase storage bucket."""
    with open(local_path, "rb") as f:
        data = f.read()
    sb.storage.from_(bucket).upload(
        storage_path,
        data,
        {"content-type": content_type, "upsert": "true"},
    )


def upload_bytes(
    sb: Client,
    bucket: str,
    storage_path: str,
    data: bytes,
    content_type: str,
) -> None:
    """Upload raw bytes to a Supabase storage bucket."""
    sb.storage.from_(bucket).upload(
        storage_path,
        data,
        {"content-type": content_type, "upsert": "true"},
    )


def get_public_url(sb: Client, bucket: str, storage_path: str) -> str:
    result = sb.storage.from_(bucket).get_public_url(storage_path)
    return result


def download_file_bytes(sb: Client, bucket: str, storage_path: str) -> bytes:
    return sb.storage.from_(bucket).download(storage_path)


# ── Job DB helpers ─────────────────────────────────────────────────────────────

def create_job_record(
    sb: Client,
    job_id: str,
    prompt: str,
    image_url: str,
    wpm: int,
    video_length_min: float,
) -> dict:
    result = sb.table("jobs").insert(
        {
            "id": job_id,
            "status": "pending",
            "prompt": prompt,
            "image_url": image_url,
            "wpm": wpm,
            "video_length_min": video_length_min,
            "thread_id": job_id,
        }
    ).execute()
    return result.data[0] if result.data else {}


def get_job_record(sb: Client, job_id: str) -> dict | None:
    result = sb.table("jobs").select("*").eq("id", job_id).single().execute()
    return result.data


def update_job_status(sb: Client, job_id: str, status: str, **kwargs) -> None:
    update_data = {"status": status, **kwargs}
    sb.table("jobs").update(update_data).eq("id", job_id).execute()


def get_job_with_story_and_segments(sb: Client, job_id: str) -> dict | None:
    job = get_job_record(sb, job_id)
    if not job:
        return None

    story_res = sb.table("stories").select("*").eq("job_id", job_id).execute()
    story = story_res.data[0] if story_res.data else None

    segs_res = (
        sb.table("segments")
        .select("*")
        .eq("job_id", job_id)
        .order("segment_index")
        .execute()
    )
    segments = segs_res.data or []

    return {"job": job, "story": story, "segments": segments}


def list_jobs(sb: Client, limit: int = 20, offset: int = 0) -> list[dict]:
    result = (
        sb.table("jobs")
        .select("id, status, prompt, video_url, created_at")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return result.data or []


def similarity_search(sb: Client, embedding: list[float], threshold: float = 0.7, limit: int = 10) -> list[dict]:
    result = sb.rpc(
        "match_jobs",
        {
            "query_embedding": embedding,
            "match_threshold": threshold,
            "match_count": limit,
        },
    ).execute()
    return result.data or []
