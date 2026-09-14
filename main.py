"""
main.py — FastAPI application entry point. This is the only process you run.

Start it:
    uvicorn main:app --host 0.0.0.0 --port 8000

There is no separate worker process to start — pipeline and edit jobs run
in-process as FastAPI BackgroundTasks (see background_jobs.py). Redis is
still used, but only for publishing/streaming progress events over SSE.

The frontend lives in static/ and is served directly at "/" — open
http://localhost:8000/ in a browser once this is running.
"""
from __future__ import annotations
from fastapi.responses import FileResponse
import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

from api.routes import router
from config import get_settings
from services.supabase_service import get_supabase, list_jobs_by_status, update_job_status
from core.graph import is_resumable
from background_jobs import recover_interrupted_job
from errors import SupabaseError

# Holds references to fire-and-forget recovery tasks scheduled at startup so
# asyncio doesn't garbage-collect them mid-flight (a real risk for tasks with
# no other referrer — see asyncio docs on create_task). Each task removes
# itself once done.
_startup_recovery_tasks: set[asyncio.Task] = set()

settings = get_settings()

# Set LangSmith env vars before any LangGraph imports
if settings.langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    logger.info("LangSmith tracing enabled (project=%s)", settings.langchain_project)

if settings.groq_api_key:
    os.environ["GROQ_API_KEY"] = settings.groq_api_key
else:
    logger.warning("GROQ_API_KEY is not set — story generation and routing will fail")


def _sweep_orphaned_jobs() -> None:
    """
    Since jobs run as in-process BackgroundTasks, any job still marked 'running'
    when this process starts up means the previous process died mid-job (crash,
    deploy, Ctrl+C, OOM-kill) — nothing out there is still working on it.

    Old behavior: mark every one of these 'failed' unconditionally. That made
    /retry — a full restart from the original prompt — the only way forward,
    throwing away story generation / voiceover / whatever had already run.

    New behavior: LangGraph's Postgres checkpointer already has that work
    saved per node, keyed by thread_id == job_id. So for each orphaned job,
    check the checkpoint via core.graph.is_resumable() before giving up on it:
      - checkpoint exists and still has steps queued -> resumable. Schedule
        recover_interrupted_job() to continue that thread from exactly the
        node it never finished, instead of failing it.
      - no checkpoint, or the thread already reached END -> genuinely dead
        (died before the graph took a single step, or died after finishing
        but before the status write landed). Mark failed and clean up its
        tmp dir, same as before.
    """
    try:
        sb = get_supabase()
        orphaned = list_jobs_by_status(sb, "running")
    except SupabaseError:
        logger.exception("Startup sweep: could not check for orphaned jobs, skipping")
        return

    if not orphaned:
        logger.info("Startup sweep: no orphaned 'running' jobs found")
        return

    logger.warning("Startup sweep: found %d orphaned job(s) from a previous run", len(orphaned))

    for job in orphaned:
        job_id = job["id"]

        try:
            resumable = is_resumable(job_id)
        except Exception:
            # Checkpoint check itself blew up (e.g. Postgres briefly unreachable
            # during startup) — don't let that decide the job's fate silently;
            # fall through to the same "mark failed" path as a real orphan.
            logger.exception("[job=%s] could not read checkpoint state, treating as unresumable", job_id)
            resumable = False

        if resumable:
            logger.warning("[job=%s] resumable checkpoint found — recovering instead of failing it", job_id)
            task = asyncio.create_task(recover_interrupted_job(job_id))
            _startup_recovery_tasks.add(task)
            task.add_done_callback(_startup_recovery_tasks.discard)
            continue

        logger.warning("[job=%s] no resumable checkpoint, marking failed", job_id)
        try:
            update_job_status(
                sb, job_id, "failed",
                error_message="orphaned: server restarted while this job was running",
            )
            logger.info("[job=%s] marked failed (orphaned on startup)", job_id)
        except SupabaseError:
            logger.exception("[job=%s] failed to mark orphaned job as failed", job_id)

        tmp_dir = os.path.join(settings.tmp_dir, job_id)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("[job=%s] cleaned up orphaned tmp dir %s", job_id, tmp_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up")
    try:
        os.makedirs(settings.tmp_dir, exist_ok=True)
        logger.info("tmp dir ready: %s", settings.tmp_dir)
    except OSError:
        logger.exception("Failed to create tmp dir %s — background job file operations may fail",
                          settings.tmp_dir)

    _sweep_orphaned_jobs()

    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="Narrative Video Generator",
    description="Prompt + image → story → voiceover → stitched video. Powered by LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Last-resort safety net: log every unhandled exception with full context instead
    of letting it disappear into a bare 500 with no trace in the logs.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(router, prefix="/api/v1")


# Static assets (CSS/JS) live at /static/*, matching the hrefs in index.html.
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join("static", "index.html"))