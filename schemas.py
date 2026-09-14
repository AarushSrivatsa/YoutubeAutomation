from __future__ import annotations
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


# ── Job creation ──────────────────────────────────────────────────────────────

class JobCreateForm(BaseModel):
    """Parsed from multipart form (image is a separate UploadFile)."""
    prompt: str = Field(..., min_length=1, max_length=4000)
    wpm: int = Field(default=150, ge=80, le=250)
    video_length_min: float = Field(default=10.0, ge=1.0, le=120.0)
    voice_model: str = Field(default="af_heart")


# ── Job response ──────────────────────────────────────────────────────────────

class SegmentOut(BaseModel):
    segment_index: int
    text: str
    original_text: Optional[str] = None
    status: str


class StoryOut(BaseModel):
    title: Optional[str] = None
    story_arc: Optional[str] = None
    characters: list[str] = []
    locations: list[str] = []
    word_count: Optional[int] = None


class JobOut(BaseModel):
    id: UUID
    status: str
    prompt: str
    image_url: Optional[str] = None
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    error_message: Optional[str] = None
    wpm: int
    video_length_min: float
    voice_model: str = "af_heart"
    story: Optional[StoryOut] = None
    segments: list[SegmentOut] = []
    created_at: datetime
    updated_at: datetime


class JobListOut(BaseModel):
    id: UUID
    status: str
    prompt: str
    video_url: Optional[str] = None
    created_at: datetime


# ── Edit endpoints ────────────────────────────────────────────────────────────

class SegmentEditRequest(BaseModel):
    """Provide text for a direct edit, or instruction for AI-assisted rewrite."""
    text: Optional[str] = Field(default=None, description="Direct replacement text")
    instruction: Optional[str] = Field(
        default=None,
        description="AI rewrite instruction, used if text is not provided"
    )


# ── Progress (SSE payload shape) ──────────────────────────────────────────────

class ProgressEvent(BaseModel):
    current: int
    total: int
    stage: str


# ── Generic responses ─────────────────────────────────────────────────────────

class JobEnqueued(BaseModel):
    job_id: UUID
    status: str = "pending"
    message: str = "Job started in the background. Poll /jobs/{job_id}/progress for updates."


class EditEnqueued(BaseModel):
    job_id: UUID
    message: str
