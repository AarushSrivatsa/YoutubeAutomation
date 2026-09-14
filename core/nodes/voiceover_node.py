"""
voiceover_node.py — wraps generate_voiceover() from voiceover.py.
Writes audio to a job-specific tmp dir. Uploads happen later in supabase_node.
"""
from __future__ import annotations
import logging
import os

from core.voiceover import generate_voiceover
from config import get_settings
from core.state import PipelineState
from services.redis_service import publish_job_progress
from errors import MediaProcessingError

logger = logging.getLogger(__name__)
settings = get_settings()


def voiceover_node(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    story_text = state.get("story_text", "")
    logger.info("[job=%s] voiceover_node entered (%d chars of story text)", job_id, len(story_text))

    if not story_text:
        logger.error("[job=%s] voiceover_node: story_text is empty, aborting", job_id)
        return {**state, "error": "voiceover_node: story_text is empty"}

    publish_job_progress(job_id, 0, 0, "generating voiceover…")

    tmp_dir = os.path.join(settings.tmp_dir, job_id)
    try:
        os.makedirs(tmp_dir, exist_ok=True)
    except OSError as e:
        logger.exception("[job=%s] voiceover_node: could not create tmp dir %s", job_id, tmp_dir)
        return {**state, "error": f"voiceover_node: could not create tmp dir {tmp_dir}: {e}"}

    audio_path = os.path.join(tmp_dir, "audio.mp3")

    try:
        result = generate_voiceover(
            story_text=story_text,
            output_path=audio_path,
            voice_model=state.get("voice_model", settings.default_voice),
            target_wpm=float(state.get("wpm", 150)),
            cache_dir=os.path.join(settings.tmp_dir, ".kokoro_cache"),
            use_cuda=settings.use_cuda,
        )
        logger.info(
            "[job=%s] voiceover_node: done — %.0fs at %s wpm (path=%s)",
            job_id, result["duration_sec"], result["actual_wpm"], result["audio_path"],
        )
        publish_job_progress(
            job_id, 0, 0,
            f"voiceover done — {result['duration_sec']:.0f}s at {result['actual_wpm']} wpm"
        )
        return {**state, "audio_path": result["audio_path"]}

    except MediaProcessingError as e:
        logger.exception("[job=%s] voiceover_node failed (media processing)", job_id)
        publish_job_progress(job_id, 0, 0, f"voiceover failed: {e}")
        return {**state, "error": f"voiceover_node failed: {e}"}
    except Exception as e:
        logger.exception("[job=%s] voiceover_node failed (unexpected)", job_id)
        publish_job_progress(job_id, 0, 0, f"voiceover failed: {e}")
        return {**state, "error": f"voiceover_node failed: {e}"}
