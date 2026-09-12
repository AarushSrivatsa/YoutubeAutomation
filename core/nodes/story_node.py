"""
story_node.py — two modes:
  • No segments in state  → full story generation via generate_story()
  • Segments + feedback   → surgical revision of flagged segments only via regenerate_segment()
"""
from __future__ import annotations

from core.story_gen import generate_story, regenerate_segment, assemble_story
from config import get_settings
from core.state import PipelineState
from services.redis_service import publish_job_progress, make_progress_callback

settings = get_settings()


def story_node(state: PipelineState) -> PipelineState:
    segments = state.get("segments")

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

    result = generate_story(
        idea=enriched,
        wpm=wpm,
        video_length_min=video_length_min,
        model=settings.groq_model,
        on_progress=on_progress,
    )

    if not result["success"]:
        return {**state, "error": result.get("error", "story generation failed")}

    assembled = assemble_story(result["segments"], result["title"])

    # segment_plans keys must be strings for JSON serialization
    segment_plans = {str(k): v for k, v in (result.get("segment_plans") or {}).items()}

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
        return state

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
            continue

        plan = segment_plans.get(str(sid)) or segment_plans.get(sid)
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

    # Re-assemble full story text after revision
    assembled = assemble_story(segments, state.get("title", ""))
    publish_job_progress(job_id, 0, 0, f"revision round {revision_round} complete")

    return {
        **state,
        "segments": segments,
        "story_text": assembled["story_text"],
    }
