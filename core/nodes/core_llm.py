"""
core_llm.py — single node, two modes.
  • No segments in state  → routing mode  (does this prompt need web search?)
  • Segments in state     → verification mode  (is the story good? what needs fixing?)

Uses a fast Groq model with strict JSON output for both.

Both modes are intentionally fail-open: a Groq failure here should not kill the
whole pipeline (routing falls back to "no search needed", verification falls
back to "approved"). Every failure is still logged loudly so it's visible.
"""
from __future__ import annotations
import json
import logging
from groq import Groq
from config import get_settings
from core.state import PipelineState
from services.redis_service import publish_job_progress
from errors import GroqAPIError

logger = logging.getLogger(__name__)
settings = get_settings()

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise GroqAPIError("GROQ_API_KEY is not set")
        _client = Groq(api_key=settings.groq_api_key)
    return _client


# ── Routing ───────────────────────────────────────────────────────────────────

_ROUTING_SYSTEM = (
    "You are a routing agent. Given a user's story/content prompt, decide whether "
    "web search is needed to generate the content accurately. Search is needed when "
    "the prompt references real events, real people, factual accuracy, current affairs, "
    "specific historical details, or anything that benefits from up-to-date information. "
    "Search is NOT needed for purely fictional or open-ended creative prompts. "
    "Respond only with the JSON schema provided."
)

_ROUTING_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_search": {"type": "boolean"},
        "search_query": {
            "type": "string",
            "description": "Optimised search query if needs_search is true, else empty string."
        },
        "reasoning": {"type": "string"},
    },
    "required": ["needs_search", "search_query", "reasoning"],
    "additionalProperties": False,
}

# ── Verification ──────────────────────────────────────────────────────────────

_VERIFICATION_SYSTEM = (
    "You are a story quality reviewer. Read the story segments and evaluate: "
    "consistency, pacing, narrative continuity, tonal coherence, and faithfulness to "
    "the original prompt. Flag only segments that have a genuine problem — do not flag "
    "segments that are simply average. If the story is acceptable, approve it. "
    "Each instruction in segment_feedback must be specific and actionable. "
    "Respond only with the JSON schema provided."
)

_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["approved", "needs_revision"],
        },
        "overall_notes": {"type": "string"},
        "segment_feedback": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "integer"},
                    "instruction": {"type": "string"},
                },
                "required": ["segment_id", "instruction"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "overall_notes", "segment_feedback"],
    "additionalProperties": False,
}


def core_llm_node(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    segments = state.get("segments")

    logger.info("[job=%s] core_llm_node entered (mode=%s)", job_id, "verification" if segments else "routing")
    if not segments:
        return _routing_pass(state, job_id)
    else:
        return _verification_pass(state, job_id)


def _routing_pass(state: PipelineState, job_id: str) -> PipelineState:
    publish_job_progress(job_id, 0, 0, "routing: analysing prompt")
    prompt = state.get("prompt", "")
    try:
        resp = _get_client().chat.completions.create(
            model=settings.core_llm_model,
            messages=[
                {"role": "system", "content": _ROUTING_SYSTEM},
                {"role": "user", "content": f"Prompt:\n{prompt}"},
            ],
            temperature=0.1,
            max_completion_tokens=512,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "routing_decision",
                    "strict": True,
                    "schema": _ROUTING_SCHEMA,
                },
            },
        )
        data = json.loads(resp.choices[0].message.content)
        logger.info(
            "[job=%s] routing decision: needs_search=%s query=%r",
            job_id, data["needs_search"], data.get("search_query", ""),
        )
        publish_job_progress(
            job_id, 0, 0,
            f"routing done — {'search needed' if data['needs_search'] else 'no search needed'}"
        )
        return {
            **state,
            "needs_search": data["needs_search"],
            "search_query": data.get("search_query", ""),
        }
    except Exception as e:
        # Fail open: skip search on routing error, but log it loudly.
        logger.exception("[job=%s] routing pass failed — falling back to needs_search=False", job_id)
        publish_job_progress(job_id, 0, 0, f"routing failed, continuing without search: {e}")
        return {**state, "needs_search": False, "search_query": "", "error": str(GroqAPIError("routing failed", e))}


def _verification_pass(state: PipelineState, job_id: str) -> PipelineState:
    revision_round = state.get("revision_round", 0)
    publish_job_progress(job_id, 0, 0, f"verifying story (round {revision_round + 1})")

    prompt = state.get("prompt", "")
    segments = state.get("segments", [])
    segments_text = "\n\n".join(
        f"[Segment {s['id']}]\n{s['text']}" for s in segments
    )

    try:
        resp = _get_client().chat.completions.create(
            model=settings.core_llm_model,
            messages=[
                {"role": "system", "content": _VERIFICATION_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Original prompt:\n{prompt}\n\n"
                        f"Story segments:\n{segments_text}"
                    ),
                },
            ],
            temperature=0.2,
            max_completion_tokens=1024,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "verification_result",
                    "strict": True,
                    "schema": _VERIFICATION_SCHEMA,
                },
            },
        )
        data = json.loads(resp.choices[0].message.content)
        logger.info(
            "[job=%s] verification result: status=%s flagged_segments=%d",
            job_id, data["status"], len(data.get("segment_feedback", [])),
        )
        publish_job_progress(job_id, 0, 0, f"verification: {data['status']}")
        return {
            **state,
            "verification_status": data["status"],
            "segment_feedback": data.get("segment_feedback", []),
            "revision_round": revision_round + 1,
        }
    except Exception as e:
        # Fail open: approve on verification error so pipeline doesn't stall.
        logger.exception("[job=%s] verification pass failed — approving story as-is", job_id)
        publish_job_progress(job_id, 0, 0, f"verification failed, approving as-is: {e}")
        return {
            **state,
            "verification_status": "approved",
            "segment_feedback": [],
            "revision_round": revision_round + 1,
            "error": str(GroqAPIError("verification failed", e)),
        }


# ── Conditional routing function (used by LangGraph graph edges) ──────────────

def route_after_core_llm(state: PipelineState) -> str:
    """
    Called by LangGraph after core_llm_node to decide the next node.
    No segments yet  → routing just happened → go to tavily or story.
    Segments exist   → verification just happened → go to story (revision) or voiceover.
    """
    job_id = state.get("job_id", "")
    if not state.get("segments"):
        next_node = "tavily" if state.get("needs_search") else "story"
        logger.info("[job=%s] route_after_core_llm (routing mode) -> %s", job_id, next_node)
        return next_node

    status = state.get("verification_status", "approved")
    revision_round = state.get("revision_round", 0)
    if status == "needs_revision" and revision_round < 2:
        logger.info("[job=%s] route_after_core_llm -> story (revision round %d)", job_id, revision_round)
        return "story"
    logger.info("[job=%s] route_after_core_llm -> voiceover (status=%s, round=%d)", job_id, status, revision_round)
    return "voiceover"
