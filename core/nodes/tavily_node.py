"""
tavily_node.py — performs web search using Tavily and appends results to state.
Results are cached in Redis keyed by the search query to avoid burning API calls
on repeat prompts.

A Tavily failure is non-fatal: the pipeline proceeds without search results
rather than failing the whole job, but the failure is always logged.
"""
from __future__ import annotations
import logging
from tavily import TavilyClient
from config import get_settings
from core.state import PipelineState
from services.redis_service import get_sync_redis, publish_job_progress
from errors import TavilyAPIError, RedisError

logger = logging.getLogger(__name__)
settings = get_settings()

_CACHE_TTL_SEC = 60 * 60 * 24  # 24 hours
_MAX_RESULTS = 5

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        if not settings.tavily_api_key:
            raise TavilyAPIError("TAVILY_API_KEY is not set")
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


def _cache_key(query: str) -> str:
    return f"tavily:search:{query.strip().lower()[:200]}"


def tavily_node(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    query = state.get("search_query", "") or state.get("prompt", "")
    logger.info("[job=%s] tavily_node entered, query=%r", job_id, query[:120])
    publish_job_progress(job_id, 0, 0, f"searching: {query[:60]}…")

    # Check Redis cache first (best-effort — a cache miss due to Redis being down
    # should just fall through to a live search, not fail the job).
    cache_key = _cache_key(query)
    try:
        r = get_sync_redis()
        cached = r.get(cache_key)
    except (RedisError, Exception) as e:
        logger.warning("[job=%s] tavily cache lookup failed, proceeding without cache: %s", job_id, e)
        r = None
        cached = None

    if cached:
        logger.info("[job=%s] tavily search results loaded from cache", job_id)
        publish_job_progress(job_id, 0, 0, "search results loaded from cache")
        return {**state, "search_results": cached.decode()}

    try:
        response = _get_client().search(
            query=query,
            search_depth="advanced",
            max_results=_MAX_RESULTS,
            include_answer=True,
        )
        # Flatten into a single readable block for the story gen prompt
        parts: list[str] = []
        if response.get("answer"):
            parts.append(f"Summary: {response['answer']}")
        for r_item in response.get("results", []):
            title = r_item.get("title", "")
            content = r_item.get("content", "")
            if title or content:
                parts.append(f"— {title}\n{content}")

        search_text = "\n\n".join(parts)
        logger.info("[job=%s] tavily search returned %d result(s)", job_id, len(response.get("results", [])))

        # Cache it (best-effort)
        if r is not None:
            try:
                r.set(cache_key, search_text, ex=_CACHE_TTL_SEC)
            except Exception as e:
                logger.warning("[job=%s] failed to cache tavily results: %s", job_id, e)

        publish_job_progress(job_id, 0, 0, "search complete")
        return {**state, "search_results": search_text}

    except Exception as e:
        # Non-fatal: proceed without search results, but log it loudly.
        logger.exception("[job=%s] Tavily search failed", job_id)
        publish_job_progress(job_id, 0, 0, f"search failed, continuing without it: {e}")
        return {
            **state,
            "search_results": "",
            "error": str(TavilyAPIError(f"search failed for query {query!r}", e)),
        }
