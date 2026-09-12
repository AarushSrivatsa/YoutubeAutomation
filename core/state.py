"""
LangGraph state for the full pipeline.
All values must be JSON-serializable — the Postgres checkpointer serializes state to JSON.
No callables, no objects, no numpy arrays here.
"""
from __future__ import annotations
from typing import Optional, TypedDict


class PipelineState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    job_id: str
    prompt: str
    image_url: str
    wpm: int
    video_length_min: float
    voice_model: str

    # ── Routing (core_llm, routing mode) ─────────────────────────────────────
    needs_search: bool
    search_query: str

    # ── Search result (tavily node) ───────────────────────────────────────────
    search_results: str          # plain text, injected into story gen prompt

    # ── Story generation (story node) ─────────────────────────────────────────
    enriched_prompt: str         # prompt + search_results combined
    title: str
    story_arc: str
    characters: list[str]
    locations: list[str]
    segments: list[dict]         # [{id, text, original_text, status}]
    segment_plans: dict          # {str(id): plan_dict}
    target_words: int

    # ── Verification (core_llm, verification mode) ────────────────────────────
    verification_status: str     # "approved" | "needs_revision"
    segment_feedback: list[dict] # [{segment_id, instruction}]
    revision_round: int          # incremented each revision pass; capped at 2

    # ── Audio (voiceover node) ────────────────────────────────────────────────
    story_text: str
    audio_path: str              # local tmp path

    # ── Video (stitch node) ───────────────────────────────────────────────────
    video_path: str              # local tmp path

    # ── Final URLs (supabase node) ────────────────────────────────────────────
    audio_url: str
    video_url: str

    # ── Error ─────────────────────────────────────────────────────────────────
    error: Optional[str]
