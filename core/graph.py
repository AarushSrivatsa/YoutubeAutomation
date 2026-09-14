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
    -- PAUSE HERE for human approval (interrupt_before=["voiceover"]) --
    voiceover → stitch → supabase → END

The graph pauses right before the voiceover node so the generated script can
be reviewed/edited by a human before any TTS or ffmpeg time gets spent on it.
The Postgres checkpointer is what makes this possible — the graph's exact
state (segments, story_text, etc.) is persisted and sits there untouched
until something calls resume_graph(job_id).
"""
from __future__ import annotations
import logging

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
from errors import ExternalServiceError

logger = logging.getLogger(__name__)
settings = get_settings()

_pool: ConnectionPool | None = None
_graph = None

# Node the graph pauses before, waiting for human approval of the script.
APPROVAL_GATE_NODE = "voiceover"


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        if not settings.supabase_db_url:
            raise ExternalServiceError("postgres", "SUPABASE_DB_URL is not set")
        logger.info("Creating Postgres connection pool for LangGraph checkpointing")
        try:
            # psycopg_pool 3.2+ requires the pool to be explicitly opened.
            _pool = ConnectionPool(
                conninfo=settings.supabase_db_url,
                max_size=5,
                kwargs={"autocommit": True},
                open=False,
            )
            _pool.open()
        except Exception as e:
            logger.exception("Failed to create/open Postgres connection pool")
            raise ExternalServiceError("postgres", f"failed to create connection pool: {e}", e) from e
        logger.info("Postgres connection pool ready")
    return _pool


def get_graph():
    """Returns the compiled LangGraph graph (singleton)."""
    global _graph
    if _graph is not None:
        return _graph

    logger.info("Compiling LangGraph pipeline graph")
    pool = _get_pool()

    try:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()  # idempotent — creates langgraph checkpoint tables if missing
    except Exception as e:
        logger.exception("Failed to set up Postgres checkpointer")
        raise ExternalServiceError("postgres", f"checkpointer setup failed: {e}", e) from e

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

    _graph = g.compile(checkpointer=checkpointer, interrupt_before=[APPROVAL_GATE_NODE])
    logger.info("LangGraph pipeline graph compiled (pauses before '%s' for approval)", APPROVAL_GATE_NODE)
    return _graph


def _config_for(job_id: str) -> dict:
    return {"configurable": {"thread_id": job_id}}


def reset_thread(job_id: str) -> None:
    """
    Deletes any existing checkpoint history for this job's LangGraph thread.

    thread_id == job_id in this app. run_pipeline() is meant to always start
    the graph fresh at START with a brand-new initial_state — that's true for
    a brand-new job, but ALSO for a /retry of a previously-failed job, which
    reuses the same job_id and therefore the same thread_id.

    LangGraph does not treat graph.invoke(initial_state, config) as "restart
    at START" when a checkpoint already exists for that thread — it resumes
    from wherever that thread's last checkpoint left off. Without this reset,
    a retried job silently continues from old accumulated state instead of
    actually starting over: the approval-gate interrupt can get skipped
    entirely (because that thread already consumed it in an earlier run),
    and counters like revision_round keep climbing across "restarts" instead
    of resetting (e.g. hitting round 5 despite a coded cap of 2).

    Safe to call even when no checkpoint exists yet (a brand-new job) — it's
    a no-op in that case, so this can unconditionally run before every fresh
    pipeline invocation.
    """
    logger.info("[job=%s] resetting LangGraph checkpoint thread before fresh run", job_id)
    graph = get_graph()
    try:
        graph.checkpointer.delete_thread(job_id)
        logger.info("[job=%s] checkpoint thread reset", job_id)
    except Exception:
        # Non-fatal: worst case a retry behaves like the old (buggy) resume
        # behavior for this one run, but we don't want a checkpoint-cleanup
        # hiccup to block the retry outright.
        logger.exception("[job=%s] failed to reset checkpoint thread before run (continuing anyway)", job_id)


def run_graph(job_id: str, initial_state: dict) -> dict:
    """
    Synchronous graph invocation. Called inside asyncio.to_thread() from a FastAPI
    background task so it doesn't block the event loop.

    Returns the state as of wherever the graph stopped — either the approval gate
    (see is_awaiting_approval) or the real end of the graph.
    """
    logger.info("[job=%s] run_graph starting", job_id)
    try:
        graph = get_graph()
        result = graph.invoke(initial_state, _config_for(job_id))
        logger.info("[job=%s] run_graph finished (error=%s)", job_id, result.get("error"))
        return result
    except Exception:
        logger.exception("[job=%s] run_graph raised an unhandled exception", job_id)
        raise


def is_awaiting_approval(job_id: str) -> bool:
    """
    True if the graph stopped because it hit the approval gate (as opposed to
    reaching END normally, or never having run at all).
    """
    graph = get_graph()
    snapshot = graph.get_state(_config_for(job_id))
    paused = bool(snapshot and snapshot.next and APPROVAL_GATE_NODE in snapshot.next)
    logger.debug("[job=%s] is_awaiting_approval=%s (next=%s)", job_id, paused, getattr(snapshot, "next", None))
    return paused


def get_pending_state(job_id: str) -> dict | None:
    """Returns the current checkpointed state values for a job, or None if there is none."""
    graph = get_graph()
    snapshot = graph.get_state(_config_for(job_id))
    return snapshot.values if snapshot and snapshot.values else None


def patch_pending_state(job_id: str, patch: dict) -> None:
    """
    Updates the checkpointed state for a job that's paused at the approval gate —
    used to apply a script edit before resuming, so the voiceover node picks up
    the edited text instead of the original.
    """
    logger.info("[job=%s] patching pending graph state (keys=%s)", job_id, list(patch.keys()))
    graph = get_graph()
    try:
        graph.update_state(_config_for(job_id), patch)
    except Exception:
        logger.exception("[job=%s] failed to patch pending graph state", job_id)
        raise


def resume_graph(job_id: str) -> dict:
    """
    Resumes a graph paused at the approval gate (or anywhere else) from exactly
    where it left off, using its checkpointed state — no new input needed.
    """
    logger.info("[job=%s] resume_graph starting", job_id)
    try:
        graph = get_graph()
        result = graph.invoke(None, _config_for(job_id))
        logger.info("[job=%s] resume_graph finished (error=%s)", job_id, result.get("error"))
        return result
    except Exception:
        logger.exception("[job=%s] resume_graph raised an unhandled exception", job_id)
        raise