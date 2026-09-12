"""
tavily_node.py — performs web search using Tavily and appends results to state.
Results are cached in Redis keyed by the search query to avoid burning API calls
on repeat prompts.
"""
from __future__ import annotations
import json
from tavily import TavilyClient
from config import get_settings
from core.state import PipelineState
from services.redis_service import get_sync_redis, publish_job_progress

settings = get_settings()

_CACHE_TTL_SEC = 60 * 60 * 24  # 24 hours
_MAX_RESULTS = 5

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings.tavily_api_key)
    return _client


def _cache_key(query: str) -> str:
    return f"tavily:search:{query.strip().lower()[:200]}"


def tavily_node(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    query = state.get("search_query", "") or state.get("prompt", "")
    publish_job_progress(job_id, 0, 0, f"searching: {query[:60]}…")

    # Check Redis cache first
    r = get_sync_redis()
    cache_key = _cache_key(query)
    cached = r.get(cache_key)
    if cached:
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
        # Cache it
        r.set(cache_key, search_text, ex=_CACHE_TTL_SEC)
        publish_job_progress(job_id, 0, 0, "search complete")
        return {**state, "search_results": search_text}

    except Exception as e:
        # Non-fatal: proceed without search results
        return {
            **state,
            "search_results": "",
            "error": f"Tavily search failed: {e}",
        }
