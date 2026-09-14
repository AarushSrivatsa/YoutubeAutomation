"""
story_node.py — two modes:
  • No segments in state  → full story generation via generate_story()
  • Segments + feedback   → surgical revision of flagged segments only via regenerate_segment()
"""
from __future__ import annotations
import logging

from core.story_gen import generate_story, regenerate_segment, assemble_story
from config import get_settings
from core.state import PipelineState
from services.redis_service import publish_job_progress, make_progress_callback

logger = logging.getLogger(__name__)
settings = get_settings()


def story_node(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    segments = state.get("segments")
    logger.info("[job=%s] story_node entered (mode=%s)", job_id, "revision" if segments else "full_generation")

    if not segments:
        return _full_generation(state)
    else:
        return _revision_pass(state)


# ── Full generation ────────────────────────────────────────────────────────────

def _full_generation(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    publish_job_progress(job_id, 0, 0, "generating story plan…")

    prompt = state.get("prompt", "")
    search_results = state.get("search_results", "")

    # Enrich the prompt with search results if available
    if search_results:
        enriched = (
            f"{prompt}\n\n"
            f"--- Background research (use for factual accuracy) ---\n{search_results}"
        )
    else:
        enriched = prompt

    wpm = state.get("wpm", 150)
    video_length_min = state.get("video_length_min", 10.0)

    # Progress callback publishes to Redis so SSE can stream it
    on_progress = make_progress_callback(job_id)

    logger.info("[job=%s] calling generate_story (wpm=%s, video_length_min=%s)", job_id, wpm, video_length_min)
    result = generate_story(
        idea=enriched,
        wpm=wpm,
        video_length_min=video_length_min,
        model=settings.groq_model,
        on_progress=on_progress,
    )

    if not result["success"]:
        logger.error("[job=%s] story generation failed: %s", job_id, result.get("error"))
        return {**state, "error": result.get("error", "story generation failed")}

    assembled = assemble_story(result["segments"], result["title"])

    # segment_plans keys must be strings for JSON serialization
    segment_plans = {str(k): v for k, v in (result.get("segment_plans") or {}).items()}

    logger.info("[job=%s] story generation complete: %d segments, %d words",
                job_id, len(result["segments"]), assembled["word_count"])
    publish_job_progress(job_id, 0, 0, "story generation complete")
    return {
        **state,
        "enriched_prompt": enriched,
        "title": result["title"],
        "story_arc": result.get("story_arc", ""),
        "characters": result.get("characters", []),
        "locations": result.get("locations", []),
        "segments": result["segments"],
        "segment_plans": segment_plans,
        "target_words": result.get("target_words", 0),
        "story_text": assembled["story_text"],
    }


# ── Targeted revision pass ────────────────────────────────────────────────────

def _revision_pass(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    feedback = state.get("segment_feedback", [])
    segments = list(state.get("segments", []))
    segment_plans = state.get("segment_plans", {})
    enriched_prompt = state.get("enriched_prompt", state.get("prompt", ""))
    revision_round = state.get("revision_round", 1)

    if not feedback:
        # Nothing to revise — skip straight through
        logger.info("[job=%s] revision pass called with no feedback, skipping", job_id)
        return state

    logger.info("[job=%s] revising %d segment(s), round %d", job_id, len(feedback), revision_round)
    publish_job_progress(
        job_id, 0, 0,
        f"revising {len(feedback)} segment(s) (round {revision_round})…"
    )

    # Build lookup by segment id for O(1) access
    seg_by_id = {s["id"]: s for s in segments}

    for item in feedback:
        sid = item.get("segment_id")
        instruction = item.get("instruction", "")
        seg = seg_by_id.get(sid)
        if not seg:
            logger.warning("[job=%s] revision feedback referenced unknown segment_id=%s, skipping", job_id, sid)
            continue

        plan = segment_plans.get(str(sid)) or segment_plans.get(sid)
        logger.info("[job=%s] regenerating segment %s: %r", job_id, sid, instruction[:120])
        result = regenerate_segment(
            original_text=seg["original_text"],
            instruction=instruction,
            idea=enriched_prompt,
            segment_plan=plan,
            model=settings.groq_model,
        )
        if result["success"]:
            seg["text"] = result["text"]
            seg["status"] = "regenerated"
            logger.info("[job=%s] segment %s regenerated successfully", job_id, sid)
        else:
            logger.warning("[job=%s] segment %s regeneration failed, keeping previous text: %s",
                           job_id, sid, result.get("error"))

    # Re-assemble full story text after revision
    assembled = assemble_story(segments, state.get("title", ""))
    logger.info("[job=%s] revision round %d complete", job_id, revision_round)
    publish_job_progress(job_id, 0, 0, f"revision round {revision_round} complete")

    return {
        **state,
        "segments": segments,
        "story_text": assembled["story_text"],
    }
