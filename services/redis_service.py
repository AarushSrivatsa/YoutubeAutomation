"""
redis_service.py — Redis clients and progress publishing helpers.

Two clients:
  • Sync  (redis.Redis)       — used inside LangGraph nodes and on_progress callbacks
                               (nodes are sync; can't await inside them)
  • Async (redis.asyncio)     — used in FastAPI SSE endpoint (progress streaming)

Progress publishing is intentionally fail-open (a dropped progress event should
never crash the pipeline), but every failure is still logged so it's visible
in the terminal instead of silently vanishing.
"""
from __future__ import annotations
import json
import logging

import redis
import redis.asyncio as aioredis

from config import get_settings
from errors import RedisError

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Sync client (module-level singleton) ──────────────────────────────────────

_sync_redis: redis.Redis | None = None


def get_sync_redis() -> redis.Redis:
    global _sync_redis
    if _sync_redis is None:
        logger.info("Creating sync Redis client for %s", settings.redis_url)
        try:
            _sync_redis = redis.from_url(settings.redis_url, decode_responses=False)
        except Exception as e:
            logger.exception("Failed to create sync Redis client")
            raise RedisError(f"failed to create sync client: {e}", e) from e
    return _sync_redis


# ── Async client factory (one per event loop — don't cache across loops) ──────

def get_async_redis() -> aioredis.Redis:
    logger.debug("Creating async Redis client for %s", settings.redis_url)
    try:
        return aioredis.from_url(settings.redis_url, decode_responses=False)
    except Exception as e:
        logger.exception("Failed to create async Redis client")
        raise RedisError(f"failed to create async client: {e}", e) from e


# ── Progress publishing ───────────────────────────────────────────────────────

def publish_job_progress(job_id: str, current: int, total: int, stage: str) -> None:
    """Sync publish — safe to call from inside LangGraph nodes. Never raises."""
    if not job_id:
        return
    logger.info("[job=%s] progress %s/%s — %s", job_id, current, total, stage)
    try:
        payload = json.dumps({"current": current, "total": total, "stage": stage})
        get_sync_redis().publish(f"job:{job_id}:progress", payload)
    except Exception as e:
        # Never crash the pipeline over a progress event, but don't hide it either.
        logger.warning("[job=%s] failed to publish progress event (%s): %s", job_id, stage, e)


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
    """Async publish — for use in FastAPI route handlers. Never raises."""
    if not job_id:
        return
    logger.info("[job=%s] progress (async) %s/%s — %s", job_id, current, total, stage)
    try:
        r = get_async_redis()
        payload = json.dumps({"current": current, "total": total, "stage": stage})
        await r.publish(f"job:{job_id}:progress", payload)
        await r.aclose()
    except Exception as e:
        logger.warning("[job=%s] failed to publish async progress event (%s): %s", job_id, stage, e)
