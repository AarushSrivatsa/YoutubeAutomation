"""
redis_service.py — Redis clients and progress publishing helpers.

Two clients:
  • Sync  (redis.Redis)       — used inside LangGraph nodes and on_progress callbacks
                               (nodes are sync; can't await inside them)
  • Async (redis.asyncio)     — used in FastAPI SSE endpoint and ARQ task management
"""
from __future__ import annotations
import json
from functools import lru_cache

import redis
import redis.asyncio as aioredis

from config import get_settings

settings = get_settings()

# ── Sync client (module-level singleton) ──────────────────────────────────────

_sync_redis: redis.Redis | None = None


def get_sync_redis() -> redis.Redis:
    global _sync_redis
    if _sync_redis is None:
        _sync_redis = redis.from_url(settings.redis_url, decode_responses=False)
    return _sync_redis


# ── Async client factory (one per event loop — don't cache across loops) ──────

def get_async_redis() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_url, decode_responses=False)


# ── Progress publishing ───────────────────────────────────────────────────────

def publish_job_progress(job_id: str, current: int, total: int, stage: str) -> None:
    """Sync publish — safe to call from inside LangGraph nodes."""
    if not job_id:
        return
    try:
        payload = json.dumps({"current": current, "total": total, "stage": stage})
        get_sync_redis().publish(f"job:{job_id}:progress", payload)
    except Exception:
        pass  # never crash the pipeline over a progress event


def make_progress_callback(job_id: str):
    """
    Returns a sync callable that matches the on_progress(dict) signature
    expected by generate_story() in story_gen.py.
    """
    def callback(progress: dict) -> None:
        publish_job_progress(
            job_id,
            current=progress.get("current", 0),
            total=progress.get("total", 0),
            stage=progress.get("stage", ""),
        )
    return callback


async def publish_job_progress_async(job_id: str, current: int, total: int, stage: str) -> None:
    """Async publish — for use in FastAPI route handlers."""
    if not job_id:
        return
    try:
        r = get_async_redis()
        payload = json.dumps({"current": current, "total": total, "stage": stage})
        await r.publish(f"job:{job_id}:progress", payload)
        await r.aclose()
    except Exception:
        pass
