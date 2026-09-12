"""
voiceover_node.py — wraps generate_voiceover() from voiceover.py.
Writes audio to a job-specific tmp dir. Uploads happen later in supabase_node.
"""
from __future__ import annotations
import os

from core.voiceover import generate_voiceover
from config import get_settings
from core.state import PipelineState
from services.redis_service import publish_job_progress

settings = get_settings()


def voiceover_node(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    story_text = state.get("story_text", "")

    if not story_text:
        return {**state, "error": "voiceover_node: story_text is empty"}

    publish_job_progress(job_id, 0, 0, "generating voiceover…")

    tmp_dir = os.path.join(settings.tmp_dir, job_id)
    os.makedirs(tmp_dir, exist_ok=True)
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
        publish_job_progress(
            job_id, 0, 0,
            f"voiceover done — {result['duration_sec']:.0f}s at {result['actual_wpm']} wpm"
        )
        return {**state, "audio_path": result["audio_path"]}

    except Exception as e:
        return {**state, "error": f"voiceover_node failed: {e}"}
