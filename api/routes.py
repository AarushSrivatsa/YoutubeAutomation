"""
routes.py — all API endpoints.

POST   /jobs                          Create + enqueue a new job
GET    /jobs                          List jobs (optional ?q= similarity search)
GET    /jobs/{job_id}                 Full job details with story + segments
GET    /jobs/{job_id}/progress        SSE stream of pipeline progress events
POST   /jobs/{job_id}/retry           Re-run a failed job from scratch with the same inputs
POST   /jobs/{job_id}/approve         Approve a reviewed script and continue to voiceover
PATCH  /jobs/{job_id}/segments/{idx}  Edit a segment (direct text or AI rewrite)
PATCH  /jobs/{job_id}/image           Swap the image → re-stitch only
DELETE /jobs/{job_id}                 Delete job + storage objects

Human-in-the-loop script review: a job pauses at status='awaiting_approval'
right after the script is written and verified, before any TTS/ffmpeg time is
spent on it. GET /jobs/{job_id} shows the script (story + segments) at that
point; PATCH the segments to edit it; POST /approve when it's ready to become
a video.

NOTE on job status transitions (approve/retry): these use try_claim_job(),
an atomic conditional UPDATE, instead of "read status, check it, then write
status elsewhere" — the latter is a read-then-write race where two
near-simultaneous requests for the same job can both pass the check and
both get scheduled as background pipeline runs, which then collide with
each other over the same tmp_dir/thread_id (this is what caused a stray
second `resume_graph` invocation to crash ffmpeg with a dimension error
right after a job had already completed successfully once).
"""
from __future__ import annotations
import json
import logging
import uuid
from typing import Annotated, AsyncGenerator, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
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
    try_claim_job,
    upload_bytes,
    get_public_url,
    similarity_search,
)
from core.story_gen import regenerate_segment
from background_jobs import run_pipeline, run_edit_pipeline, resume_pipeline_after_approval, edit_pending_segment
from errors import SupabaseError, OllamaError, GroqAPIError

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

# NOTE: jobs run as FastAPI BackgroundTasks in this same process — there is no
# separate worker to start. background_tasks.add_task() schedules the coroutine
# to run right after the response is sent; it does not block the request.


# ── POST /jobs ─────────────────────────────────────────────────────────────────

@router.post("/jobs", response_model=JobEnqueued, status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    prompt: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
    wpm: Annotated[int, Form()] = 150,
    video_length_min: Annotated[float, Form()] = 10.0,
    voice_model: Annotated[str, Form()] = "af_heart",
):
    job_id = str(uuid.uuid4())
    logger.info("[job=%s] create_job: prompt=%r wpm=%s video_length_min=%s voice_model=%s",
                job_id, prompt[:80], wpm, video_length_min, voice_model)
    sb = get_supabase()

    # Upload image immediately — pipeline only stores/passes the URL
    image_bytes = await image.read()
    content_type = image.content_type or "image/jpeg"
    ext = image.filename.rsplit(".", 1)[-1] if image.filename else "jpg"
    storage_path = f"{job_id}/image.{ext}"

    try:
        upload_bytes(sb, settings.image_bucket, storage_path, image_bytes, content_type)
        image_url = get_public_url(sb, settings.image_bucket, storage_path)
    except SupabaseError as e:
        logger.exception("[job=%s] image upload failed", job_id)
        raise HTTPException(status_code=502, detail=f"Image upload failed: {e}")

    # Create job record in DB
    try:
        create_job_record(sb, job_id, prompt, image_url, wpm, video_length_min, voice_model)
    except SupabaseError as e:
        logger.exception("[job=%s] DB record creation failed", job_id)
        raise HTTPException(status_code=502, detail=f"DB record creation failed: {e}")

    # Schedule the pipeline to run right after this response is sent.
    background_tasks.add_task(
        run_pipeline,
        job_id=job_id,
        prompt=prompt,
        image_url=image_url,
        wpm=wpm,
        video_length_min=video_length_min,
        voice_model=voice_model,
    )

    logger.info("[job=%s] background pipeline task scheduled", job_id)
    return JobEnqueued(job_id=job_id)


# ── GET /jobs ──────────────────────────────────────────────────────────────────

@router.get("/jobs", response_model=list[JobListOut])
async def list_all_jobs(
    limit: int = 20,
    offset: int = 0,
    q: Optional[str] = None,  # optional similarity search query
):
    logger.info("list_all_jobs: limit=%d offset=%d q=%r", limit, offset, q)
    sb = get_supabase()

    if q and settings.embeddings_enabled:
        try:
            embedding = get_embedding(q)
            matches = similarity_search(sb, embedding, limit=limit)
            job_ids = [m["job_id"] for m in matches]
            if not job_ids:
                logger.info("similarity_search for %r returned no matches", q)
                return []
            result = sb.table("jobs").select(
                "id, status, prompt, video_url, created_at"
            ).in_("id", job_ids).execute()
            return result.data or []
        except (OllamaError, SupabaseError) as e:
            logger.warning("similarity search failed for %r, falling back to plain list: %s", q, e)
        except Exception:
            logger.exception("unexpected error during similarity search for %r, falling back", q)

    try:
        return list_jobs(sb, limit=limit, offset=offset)
    except SupabaseError as e:
        logger.exception("list_jobs failed")
        raise HTTPException(status_code=502, detail=f"Failed to list jobs: {e}")


# ── GET /jobs/{job_id} ────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str):
    logger.info("[job=%s] get_job", job_id)
    sb = get_supabase()
    try:
        data = get_job_with_story_and_segments(sb, job_id)
    except SupabaseError as e:
        logger.exception("[job=%s] get_job failed", job_id)
        raise HTTPException(status_code=502, detail=f"Failed to fetch job: {e}")

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
        voice_model=job.get("voice_model", "af_heart"),
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


# ── POST /jobs/{job_id}/retry ─────────────────────────────────────────────────

@router.post("/jobs/{job_id}/retry", response_model=JobEnqueued, status_code=202)
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Re-runs a failed job from scratch using its original prompt/image/settings.
    This is a full restart, not a resume-from-where-it-broke — the pipeline
    doesn't currently halt early on error, so by the time a job is marked
    'failed' its checkpoint has already run to completion anyway.
    """
    logger.info("[job=%s] retry requested", job_id)
    sb = get_supabase()
    try:
        data = get_job_with_story_and_segments(sb, job_id)
    except SupabaseError as e:
        logger.exception("[job=%s] failed to fetch job for retry", job_id)
        raise HTTPException(status_code=502, detail=f"Failed to fetch job: {e}")

    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = data["job"]

    if not job.get("image_url"):
        raise HTTPException(status_code=422, detail="Job has no stored image_url to retry with")

    # Atomic claim: only one concurrent /retry call for this job can win this,
    # so run_pipeline can never be scheduled twice for the same job_id.
    if not try_claim_job(sb, job_id, expected_status="failed", new_status="pending"):
        raise HTTPException(
            status_code=409,
            detail=f"Only failed jobs can be retried (current status: {job['status']})",
        )

    background_tasks.add_task(
        run_pipeline,
        job_id=job_id,
        prompt=job["prompt"],
        image_url=job["image_url"],
        wpm=job["wpm"],
        video_length_min=job["video_length_min"],
        voice_model=job.get("voice_model") or "af_heart",
    )

    logger.info("[job=%s] retry scheduled", job_id)
    return JobEnqueued(
        job_id=job_id,
        status="pending",
        message=f"Retry started. Poll /jobs/{job_id}/progress for updates.",
    )


# ── POST /jobs/{job_id}/approve ───────────────────────────────────────────────

@router.post("/jobs/{job_id}/approve", response_model=JobEnqueued, status_code=202)
async def approve_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Approves the script generated for this job and lets the pipeline continue
    to voiceover → stitch → upload. Only valid while status='awaiting_approval'.
    Review the script first via GET /jobs/{job_id}, and edit it via
    PATCH /jobs/{job_id}/segments/{idx} if it needs changes before approving.
    """
    logger.info("[job=%s] approve requested", job_id)
    sb = get_supabase()
    try:
        data = get_job_with_story_and_segments(sb, job_id)
    except SupabaseError as e:
        logger.exception("[job=%s] failed to fetch job for approval", job_id)
        raise HTTPException(status_code=502, detail=f"Failed to fetch job: {e}")

    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    # Atomic claim instead of read-then-write: only one concurrent /approve
    # call for this job can ever win this, so resume_pipeline_after_approval
    # can never be scheduled twice for the same job_id/thread_id.
    if not try_claim_job(sb, job_id, expected_status="awaiting_approval", new_status="running"):
        raise HTTPException(
            status_code=409,
            detail="Job is not awaiting approval (already approved, running, or in another state)",
        )

    background_tasks.add_task(resume_pipeline_after_approval, job_id=job_id)

    logger.info("[job=%s] approval accepted, resuming pipeline", job_id)
    return JobEnqueued(
        job_id=job_id,
        status="running",
        message=f"Approved. Poll /jobs/{job_id}/progress for updates.",
    )


# ── GET /jobs/{job_id}/progress  (SSE) ───────────────────────────────────────

@router.get("/jobs/{job_id}/progress")
async def job_progress(job_id: str, request: Request):
    """
    Server-Sent Events stream. Publishes JSON progress payloads:
      { "current": int, "total": int, "stage": str }
    Stream closes automatically when stage == "complete" or "failed".
    """
    logger.info("[job=%s] SSE progress stream opened", job_id)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            r = get_async_redis()
            pubsub = r.pubsub()
            await pubsub.subscribe(f"job:{job_id}:progress")
        except Exception as e:
            logger.exception("[job=%s] failed to open Redis pubsub for progress stream", job_id)
            yield f"data: {json.dumps({'current': 0, 'total': 0, 'stage': f'failed: {e}'})}\n\n"
            return

        try:
            async for message in pubsub.listen():
                if await request.is_disconnected():
                    logger.info("[job=%s] client disconnected from progress stream", job_id)
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
                        logger.info("[job=%s] progress stream closing (stage=%s)", job_id, stage)
                        break
                except Exception:
                    logger.warning("[job=%s] could not parse progress payload: %s", job_id, data_str)
        except Exception:
            logger.exception("[job=%s] progress stream crashed", job_id)
        finally:
            try:
                await pubsub.unsubscribe(f"job:{job_id}:progress")
                await pubsub.aclose()
                await r.aclose()
            except Exception:
                logger.exception("[job=%s] failed to clean up Redis pubsub connection", job_id)

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
    background_tasks: BackgroundTasks,
):
    logger.info("[job=%s] edit_segment %d requested", job_id, segment_index)
    sb = get_supabase()
    try:
        data = get_job_with_story_and_segments(sb, job_id)
    except SupabaseError as e:
        logger.exception("[job=%s] failed to fetch job for segment edit", job_id)
        raise HTTPException(status_code=502, detail=f"Failed to fetch job: {e}")

    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    job = data["job"]
    if job["status"] not in ("awaiting_approval", "completed", "failed"):
        raise HTTPException(
            status_code=409,
            detail="Job must be awaiting approval, completed, or failed before editing",
        )

    # Find the segment
    seg = next(
        (s for s in data["segments"] if s["segment_index"] == segment_index), None
    )
    if not seg:
        raise HTTPException(status_code=404, detail=f"Segment {segment_index} not found")

    # Determine final text
    if body.text:
        new_text = body.text
        logger.info("[job=%s] segment %d: direct text replacement", job_id, segment_index)
    elif body.instruction:
        # AI-assisted rewrite — do it synchronously here (it's fast)
        logger.info("[job=%s] segment %d: AI rewrite requested: %r",
                    job_id, segment_index, body.instruction[:120])
        try:
            story_result = sb.table("stories").select("segment_plans").eq("job_id", job_id).execute()
        except Exception as e:
            logger.exception("[job=%s] failed to fetch segment_plans", job_id)
            raise HTTPException(status_code=502, detail=f"Failed to fetch story plan: {e}")

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
            logger.error("[job=%s] segment %d AI rewrite failed: %s", job_id, segment_index, result["error"])
            raise HTTPException(status_code=502, detail=f"Rewrite failed: {result['error']}")
        new_text = result["text"]
    else:
        raise HTTPException(status_code=422, detail="Provide either text or instruction")

    if job["status"] == "awaiting_approval":
        # Pre-audio edit — no voiceover/stitch has happened yet, so just patch
        # the script (checkpointed graph state + DB) and leave it pending.
        try:
            await edit_pending_segment(job_id, segment_index, new_text)
        except ValueError as e:
            logger.error("[job=%s] pending segment edit failed: %s", job_id, e)
            raise HTTPException(status_code=404, detail=str(e))
        except SupabaseError as e:
            logger.exception("[job=%s] pending segment edit failed (Supabase)", job_id)
            raise HTTPException(status_code=502, detail=str(e))

        logger.info("[job=%s] segment %d updated while awaiting approval", job_id, segment_index)
        return EditEnqueued(
            job_id=job_id,
            message=f"Segment {segment_index} updated. Still awaiting approval — "
                    f"POST /jobs/{job_id}/approve when ready.",
        )

    # Otherwise (completed/failed): schedule the full edit pipeline
    # (re-voiceover + re-stitch) to run in the background.
    background_tasks.add_task(
        run_edit_pipeline,
        job_id=job_id,
        edit_type="segment",
        segment_index=segment_index,
        new_text=new_text,
    )

    logger.info("[job=%s] segment %d edit task scheduled", job_id, segment_index)
    return EditEnqueued(
        job_id=job_id,
        message=f"Segment {segment_index} edit enqueued. Poll /jobs/{job_id}/progress for updates.",
    )


# ── PATCH /jobs/{job_id}/image ────────────────────────────────────────────────

@router.patch("/jobs/{job_id}/image", response_model=EditEnqueued)
async def swap_image(
    job_id: str,
    background_tasks: BackgroundTasks,
    image: Annotated[UploadFile, File()],
):
    logger.info("[job=%s] swap_image requested", job_id)
    sb = get_supabase()
    try:
        data = get_job_with_story_and_segments(sb, job_id)
    except SupabaseError as e:
        logger.exception("[job=%s] failed to fetch job for image swap", job_id)
        raise HTTPException(status_code=502, detail=f"Failed to fetch job: {e}")

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
    except SupabaseError as e:
        logger.exception("[job=%s] new image upload failed", job_id)
        raise HTTPException(status_code=502, detail=f"Image upload failed: {e}")

    background_tasks.add_task(
        run_edit_pipeline,
        job_id=job_id,
        edit_type="image",
        new_image_url=new_image_url,
    )

    logger.info("[job=%s] image swap task scheduled", job_id)
    return EditEnqueued(
        job_id=job_id,
        message="Image swap enqueued. Poll /jobs/{job_id}/progress for updates.",
    )


# ── DELETE /jobs/{job_id} ─────────────────────────────────────────────────────

@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    logger.info("[job=%s] delete_job requested", job_id)
    sb = get_supabase()
    try:
        data = get_job_with_story_and_segments(sb, job_id)
    except SupabaseError as e:
        logger.exception("[job=%s] failed to fetch job before delete", job_id)
        raise HTTPException(status_code=502, detail=f"Failed to fetch job: {e}")

    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete storage objects (best-effort — a missing object shouldn't block the DB delete)
    for bucket, path in [
        (settings.image_bucket, f"{job_id}/image.jpg"),
        (settings.audio_bucket, f"{job_id}/audio.mp3"),
        (settings.video_bucket, f"{job_id}/video.mp4"),
    ]:
        try:
            sb.storage.from_(bucket).remove([path])
            logger.info("[job=%s] removed storage object %s/%s", job_id, bucket, path)
        except Exception as e:
            logger.warning("[job=%s] failed to remove storage object %s/%s (continuing): %s",
                           job_id, bucket, path, e)

    # Cascade delete in DB (FK constraints handle stories + segments)
    try:
        sb.table("jobs").delete().eq("id", job_id).execute()
        logger.info("[job=%s] job record deleted", job_id)
    except Exception as e:
        logger.exception("[job=%s] failed to delete job record", job_id)
        raise HTTPException(status_code=502, detail=f"Failed to delete job record: {e}")