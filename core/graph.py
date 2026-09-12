"""
graph.py — assembles the LangGraph pipeline graph.
Compiled once at startup; each job uses its own thread_id (= job_id) for checkpointing.

Graph flow:
  START
    → core_llm  (routing mode)
        → [tavily]  →  story
        → story
    story → core_llm  (verification mode)
        → story  (revision, max 2 rounds)
        → voiceover
    voiceover → stitch → supabase → END
"""
from __future__ import annotations
from functools import lru_cache

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from config import get_settings
from core.state import PipelineState
from core.nodes.core_llm import core_llm_node, route_after_core_llm
from core.nodes.tavily_node import tavily_node
from core.nodes.story_node import story_node
from core.nodes.voiceover_node import voiceover_node
from core.nodes.stitch_node import stitch_node
from core.nodes.supabase_node import supabase_node

settings = get_settings()

_pool: ConnectionPool | None = None
_graph = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.supabase_db_url,
            max_size=5,
            kwargs={"autocommit": True},
        )
    return _pool


def get_graph():
    """Returns the compiled LangGraph graph (singleton)."""
    global _graph
    if _graph is not None:
        return _graph

    pool = _get_pool()
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # idempotent — creates langgraph checkpoint tables if missing

    g = StateGraph(PipelineState)

    g.add_node("core_llm", core_llm_node)
    g.add_node("tavily", tavily_node)
    g.add_node("story", story_node)
    g.add_node("voiceover", voiceover_node)
    g.add_node("stitch", stitch_node)
    g.add_node("supabase", supabase_node)

    g.add_edge(START, "core_llm")
    g.add_conditional_edges(
        "core_llm",
        route_after_core_llm,
        {
            "tavily": "tavily",
            "story": "story",
            "voiceover": "voiceover",
        },
    )
    g.add_edge("tavily", "story")
    g.add_edge("story", "core_llm")
    g.add_edge("voiceover", "stitch")
    g.add_edge("stitch", "supabase")
    g.add_edge("supabase", END)

    _graph = g.compile(checkpointer=checkpointer)
    return _graph


def run_graph(job_id: str, initial_state: dict) -> dict:
    """
    Synchronous graph invocation. Called inside asyncio.to_thread() from the ARQ worker
    so it doesn't block the event loop.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": job_id}}
    result = graph.invoke(initial_state, config)
    return result


def resume_graph_from(job_id: str, new_state_values: dict, from_node: str | None = None) -> dict:
    """
    Resume a checkpointed graph after an edit.
    Updates state then re-invokes from the current checkpoint position.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": job_id}}
    # Update checkpoint state with new values
    graph.update_state(config, new_state_values)
    result = graph.invoke(None, config)
    return result
