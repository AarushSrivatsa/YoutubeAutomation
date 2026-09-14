# Debug + hardening pass — summary

## Real bugs fixed (not just style)

1. **Broken imports that would crash every segment edit.**
   The old `worker/tasks.py` had `from story_gen import assemble_story` and
   `from voiceover import generate_voiceover` — those top-level modules don't
   exist; the real code lives at `core/story_gen.py` and `core/voiceover.py`.
   The first time anyone edited a segment, this would raise
   `ModuleNotFoundError` and fail the job. Fixed in `background_jobs.py`.

2. **`voice_model` was never persisted, so edits silently lost the user's voice.**
   `jobs` table had no `voice_model` column, `create_job_record()` didn't
   accept or insert it, and `_edit_segment()` always regenerated voiceover
   with the hardcoded default instead of the job's actual voice. Fixed:
   - `schema.sql` — added `voice_model` column (`ALTER TABLE` snippet included
     for existing databases).
   - `services/supabase_service.py` — `create_job_record()` now takes and
     stores `voice_model`.
   - `api/routes.py` — passes it through on job creation, returns it in
     `GET /jobs/{id}`.
   - `background_jobs.py` — `_edit_segment()` now reads `job.get("voice_model")`
     and passes it into voiceover regeneration.

3. **`ffmpeg` encode step in `core/voiceover.py` had no error handling.**
   `_encode_final_output()` called `subprocess.run(..., check=True)` with
   nothing catching `CalledProcessError` — a bad install or missing `ffmpeg`
   binary would produce a raw, low-context traceback. Now raises
   `MediaProcessingError` with the actual ffmpeg stderr attached.

4. **`psycopg_pool.ConnectionPool` never explicitly opened.**
   psycopg_pool 3.2+ expects an explicit `.open()` call; the old code relied
   on implicit-open behavior that's deprecated. `core/graph.py` now creates
   the pool with `open=False` and opens it explicitly, with errors caught and
   reported clearly instead of surfacing as an opaque psycopg exception deep
   in graph compilation.

## Removed ARQ, switched to FastAPI `BackgroundTasks`

There's no more separate worker process / `python -m worker.tasks` command.
- `worker/tasks.py` → replaced by `background_jobs.py` (same logic, no `ctx`
  param, no ARQ decorators).
- `api/routes.py` — `arq.enqueue_job(...)` calls replaced with
  `background_tasks.add_task(...)`.
- An in-process `asyncio.Semaphore(settings.max_jobs)` caps concurrent jobs
  so multiple heavy TTS/ffmpeg jobs don't fight over CPU/RAM.
- A 1-hour `asyncio.wait_for` timeout wraps the graph invocation so a stuck
  job can't hold a semaphore slot forever.
- `requirements.txt` — dropped `arq`. Redis is still used, just only for
  progress pub/sub (SSE), not job queuing.
- `docker-compose.yml` — merged into a single `api` service; the old
  `worker` service is gone.

**Trade-off worth knowing:** if the API process restarts mid-job, that job is
gone — there's no queue to redeliver it (ARQ + Redis would have retried it).
Fine for a single-instance deploy; if you need multi-instance scaling or
automatic retries later, that's the piece to bring back.

## Logging — every step now goes to the terminal

New `logging_config.py`, called once from `main.py`. Every module does
`logger = logging.getLogger(__name__)` and logs:
- Entry/exit of every node and every background job, with `job_id` prefixed
  on every line so you can `grep` a single job's full trace.
- Every external call (Groq, Tavily, Supabase, Redis, Ollama, ffmpeg, image/
  audio HTTP downloads) — both the attempt and the outcome.
- Every fallback path (routing fails open, verification fails open, cache
  miss vs. hit, embedding skipped) so "silent" behavior is now visible.
- `print()`-based progress bar in `core/story_gen.py` replaced with proper
  logging (the `on_progress` callback for Redis/SSE is unchanged).

Set `LOG_LEVEL=DEBUG` in `.env` for more detail (e.g. calibration cache
hits, per-paragraph synthesis).

## try/except around every external call

New `errors.py` defines typed exceptions (`GroqAPIError`, `TavilyAPIError`,
`SupabaseError`, `RedisError`, `OllamaError`, `MediaProcessingError`,
`HTTPFetchError`) so callers can tell "an external dependency failed" apart
from a bug in our own code, and every failure is logged in exactly one
place before it's re-raised or handled.

Wrapped everywhere that wasn't already:
- `services/supabase_service.py` had **zero** try/except before — every DB
  and storage call goes through Supabase (an external service) and can now
  fail cleanly instead of throwing a raw supabase-py exception up the stack.
- `services/embeddings_service.py` — Ollama HTTP call.
- `core/voiceover.py` — Kokoro pipeline load/synthesis and the ffmpeg encode.
- `core/graph.py` — Postgres pool creation, checkpointer setup, graph
  invocation.
- `api/routes.py` — image uploads, DB reads/writes, and the "unhandled
  exception" middleware in `main.py` is a last-resort net so a bug anywhere
  in the app returns a clean 500 with a full stack trace in the logs instead
  of an ugly default error page.
- Fail-open behaviors (routing, verification, Tavily search, embeddings) are
  preserved exactly as designed — they just log loudly now instead of
  silently swallowing the error.

## Config validation

`config.py` now has `Settings.validate()`, called once at `get_settings()`
time, which logs a warning for every missing critical env var
(`GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SUPABASE_DB_URL`,
`TAVILY_API_KEY`) instead of letting a missing key surface as a confusing
failure three layers deep the first time it's used.

## Human-in-the-loop script approval (new)

The pipeline now pauses right after the script is written and verified, and
right before any TTS or ffmpeg time gets spent on it:

- `core/graph.py` compiles the graph with `interrupt_before=["voiceover"]` —
  LangGraph's built-in mechanism for this. The Postgres checkpointer is what
  makes the pause durable: the paused state survives a server restart with
  no extra code (unlike a `running` job, which the startup sweep treats as
  orphaned, an `awaiting_approval` job is left alone on purpose).
- When the graph hits that pause, `background_jobs.py` immediately writes
  the story + segments to the DB (`upsert_story_record` /
  `upsert_segment_records` — new shared helpers in `services/supabase_service.py`,
  also now used by `supabase_node.py` so there's one code path for this, not
  two) and sets the job's status to `awaiting_approval`.
- `GET /jobs/{job_id}` now shows that script for review, same as it would for
  a completed job.
- `PATCH /jobs/{job_id}/segments/{idx}` works during `awaiting_approval` too,
  but takes a much lighter path than it does after voiceover exists: it
  patches the segment text directly in the checkpointed graph state (via the
  new `patch_pending_state`) and syncs it to the DB — no TTS, no ffmpeg, no
  background job at all, just an instant edit.
- `POST /jobs/{job_id}/approve` resumes the graph exactly where it paused
  (`resume_graph`), picking up any edits made while it was waiting, and
  carries on through voiceover → stitch → upload.

Job status now has five values: `pending → running → awaiting_approval →
running → completed` (or `failed` at any point).

## Running it now

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

That's it — one process, no worker to start separately.
