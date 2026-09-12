"""
routes.py — all API endpoints.

POST   /jobs                          Create + enqueue a new job
GET    /jobs                          List jobs (optional ?q= similarity search)
GET    /jobs/{job_id}                 Full job details with story + segments
GET    /jobs/{job_id}/progress        SSE stream of pipeline progress events
PATCH  /jobs/{job_id}/segments/{idx}  Edit a segment (direct text or AI rewrite)
PATCH  /jobs/{job_id}/image           Swap the image → re-stitch only
DELETE /jobs/{job_id}                 Delete job + storage objects
"""
from __future__ import annotations
import json
import uuid
from typing import Annotated, AsyncGenerator, Optional

import httpx
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from config import get_settings
from schemas import (
    EditEnqueued,
    JobEnqueued,
    JobListOut,
    JobOut,
    SegmentEditRequest,
    SegmentOut,
    StoryOut,
)
from services.redis_service import get_async_redis
from services.embeddings_service import get_embedding
from services.supabase_service import (
    create_job_record,
    get_job_with_story_and_segments,
    get_supabase,
    list_jobs,
    update_job_status,
    upload_bytes,
    get_public_url,
    similarity_search,
)
from core.story_gen import regenerate_segment

settings = get_settings()
router = APIRouter()


# ── ARQ pool dependency ────────────────────────────────────────────────────────

async def get_arq():
    pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    try:
        yield pool
    finally:
        await pool.aclose()


# ── POST /jobs ─────────────────────────────────────────────────────────────────

@router.post("/jobs", response_model=JobEnqueued, status_code=202)
async def create_job(
    prompt: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
    wpm: Annotated[int, Form()] = 150,
    video_length_min: Annotated[float, Form()] = 10.0,
    voice_model: Annotated[str, Form()] = "af_heart",
    arq=Depends(get_arq),
):
    job_id = str(uuid.uuid4())
    sb = get_supabase()

    # Upload image immediately — pipeline only stores/passes the URL
    image_bytes = await image.read()
    content_type = image.content_type or "image/jpeg"
    ext = image.filename.rsplit(".", 1)[-1] if image.filename else "jpg"
    storage_path = f"{job_id}/image.{ext}"

    try:
        upload_bytes(sb, settings.image_bucket, storage_path, image_bytes, content_type)
        image_url = get_public_url(sb, settings.image_bucket, storage_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {e}")

    # Create job record in DB
    try:
        create_job_record(sb, job_id, prompt, image_url, wpm, video_length_min)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB record creation failed: {e}")

    # Enqueue background job
    await arq.enqueue_job(
        "run_pipeline",
        job_id=job_id,
        prompt=prompt,
        image_url=image_url,
        wpm=wpm,
        video_length_min=video_length_min,
        voice_model=voice_model,
    )

    return JobEnqueued(job_id=job_id)


# ── GET /jobs ──────────────────────────────────────────────────────────────────

@router.get("/jobs", response_model=list[JobListOut])
async def list_all_jobs(
    limit: int = 20,
    offset: int = 0,
    q: Optional[str] = None,  # optional similarity search query
):
    sb = get_supabase()

    if q and settings.embeddings_enabled:
        try:
            embedding = get_embedding(q)
            matches = similarity_search(sb, embedding, limit=limit)
            job_ids = [m["job_id"] for m in matches]
            # Fetch full job records for matched ids
            if not job_ids:
                return []
            result = sb.table("jobs").select(
                "id, status, prompt, video_url, created_at"
            ).in_("id", job_ids).execute()
            return result.data or []
        except Exception:
            pass  # fall through to regular list

    return list_jobs(sb, limit=limit, offset=offset)


# ── GET /jobs/{job_id} ────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str):
    sb = get_supabase()
    data = get_job_with_story_and_segments(sb, job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = data["job"]
    story = data.get("story")
    segments = data.get("segments", [])

    return JobOut(
        id=job["id"],
        status=job["status"],
        prompt=job["prompt"],
        image_url=job.get("image_url"),
        audio_url=job.get("audio_url"),
        video_url=job.get("video_url"),
        error_message=job.get("error_message"),
        wpm=job["wpm"],
        video_length_min=job["video_length_min"],
        story=StoryOut(
            title=story.get("title") if story else None,
            story_arc=story.get("story_arc") if story else None,
            characters=story.get("characters", []) if story else [],
            locations=story.get("locations", []) if story else [],
            word_count=story.get("word_count") if story else None,
        ) if story else None,
        segments=[
            SegmentOut(
                segment_index=s["segment_index"],
                text=s["text"],
                original_text=s.get("original_text"),
                status=s.get("status", "kept"),
            )
            for s in segments
        ],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


# ── GET /jobs/{job_id}/progress  (SSE) ───────────────────────────────────────

@router.get("/jobs/{job_id}/progress")
async def job_progress(job_id: str, request: Request):
    """
    Server-Sent Events stream. Publishes JSON progress payloads:
      { "current": int, "total": int, "stage": str }
    Stream closes automatically when stage == "complete" or "failed".
    """
    async def event_stream() -> AsyncGenerator[str, None]:
        r = get_async_redis()
        pubsub = r.pubsub()
        await pubsub.subscribe(f"job:{job_id}:progress")
        try:
            async for message in pubsub.listen():
                if await request.is_disconnected():
                    break
                if message["type"] != "message":
                    continue
                raw = message["data"]
                data_str = raw.decode() if isinstance(raw, bytes) else raw
                yield f"data: {data_str}\n\n"
                try:
                    parsed = json.loads(data_str)
                    stage = parsed.get("stage", "")
                    if stage in ("complete", "failed") or stage.startswith("failed"):
                        break
                except Exception:
                    pass
        finally:
            await pubsub.unsubscribe(f"job:{job_id}:progress")
            await pubsub.aclose()
            await r.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


# ── PATCH /jobs/{job_id}/segments/{segment_index} ─────────────────────────────

@router.patch("/jobs/{job_id}/segments/{segment_index}", response_model=EditEnqueued)
async def edit_segment(
    job_id: str,
    segment_index: int,
    body: SegmentEditRequest,
    arq=Depends(get_arq),
):
    sb = get_supabase()
    data = get_job_with_story_and_segments(sb, job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = data["job"]
    if job["status"] not in ("completed", "failed"):
        raise HTTPException(status_code=409, detail="Job must be completed before editing")

    # Find the segment
    seg = next(
        (s for s in data["segments"] if s["segment_index"] == segment_index), None
    )
    if not seg:
        raise HTTPException(status_code=404, detail=f"Segment {segment_index} not found")

    # Determine final text
    if body.text:
        new_text = body.text
    elif body.instruction:
        # AI-assisted rewrite — do it synchronously here (it's fast)
        story_result = sb.table("stories").select("segment_plans").eq("job_id", job_id).execute()
        segment_plans = {}
        if story_result.data:
            segment_plans = story_result.data[0].get("segment_plans", {})
        plan = segment_plans.get(str(segment_index))
        result = regenerate_segment(
            original_text=seg["original_text"] or seg["text"],
            instruction=body.instruction,
            idea=job["prompt"],
            segment_plan=plan,
            model=settings.groq_model,
        )
        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"Rewrite failed: {result['error']}")
        new_text = result["text"]
    else:
        raise HTTPException(status_code=422, detail="Provide either text or instruction")

    # Enqueue edit pipeline (re-voiceover + re-stitch)
    await arq.enqueue_job(
        "run_edit_pipeline",
        job_id=job_id,
        edit_type="segment",
        segment_index=segment_index,
        new_text=new_text,
    )

    return EditEnqueued(
        job_id=job_id,
        message=f"Segment {segment_index} edit enqueued. Poll /jobs/{job_id}/progress for updates.",
    )


# ── PATCH /jobs/{job_id}/image ────────────────────────────────────────────────

@router.patch("/jobs/{job_id}/image", response_model=EditEnqueued)
async def swap_image(
    job_id: str,
    image: Annotated[UploadFile, File()],
    arq=Depends(get_arq),
):
    sb = get_supabase()
    data = get_job_with_story_and_segments(sb, job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    if data["job"]["status"] not in ("completed", "failed"):
        raise HTTPException(status_code=409, detail="Job must be completed before swapping image")

    # Upload new image
    image_bytes = await image.read()
    content_type = image.content_type or "image/jpeg"
    ext = image.filename.rsplit(".", 1)[-1] if image.filename else "jpg"
    storage_path = f"{job_id}/image_v2.{ext}"

    try:
        upload_bytes(sb, settings.image_bucket, storage_path, image_bytes, content_type)
        new_image_url = get_public_url(sb, settings.image_bucket, storage_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {e}")

    await arq.enqueue_job(
        "run_edit_pipeline",
        job_id=job_id,
        edit_type="image",
        new_image_url=new_image_url,
    )

    return EditEnqueued(
        job_id=job_id,
        message="Image swap enqueued. Poll /jobs/{job_id}/progress for updates.",
    )


# ── DELETE /jobs/{job_id} ─────────────────────────────────────────────────────

@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    sb = get_supabase()
    data = get_job_with_story_and_segments(sb, job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete storage objects (best-effort)
    for bucket, path in [
        (settings.image_bucket, f"{job_id}/image.jpg"),
        (settings.audio_bucket, f"{job_id}/audio.mp3"),
        (settings.video_bucket, f"{job_id}/video.mp4"),
    ]:
        try:
            sb.storage.from_(bucket).remove([path])
        except Exception:
            pass

    # Cascade delete in DB (FK constraints handle stories + segments)
    sb.table("jobs").delete().eq("id", job_id).execute()
