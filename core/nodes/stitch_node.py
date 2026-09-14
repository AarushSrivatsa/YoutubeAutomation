"""
stitch_node.py — downloads the image from Supabase URL, stitches with audio via ffmpeg.
Output: a local .mp4 in the job tmp dir.
"""
from __future__ import annotations
import logging
import os
import subprocess
import httpx
from config import get_settings
from core.state import PipelineState
from services.redis_service import publish_job_progress
from errors import HTTPFetchError, MediaProcessingError

logger = logging.getLogger(__name__)
settings = get_settings()


def stitch_node(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    image_url = state.get("image_url", "")
    audio_path = state.get("audio_path", "")

    logger.info("[job=%s] stitch_node entered (image_url=%s, audio_path=%s)", job_id, image_url, audio_path)

    if not image_url:
        logger.error("[job=%s] stitch_node: image_url is missing", job_id)
        return {**state, "error": "stitch_node: image_url is missing"}
    if not audio_path or not os.path.exists(audio_path):
        logger.error("[job=%s] stitch_node: audio file not found at %s", job_id, audio_path)
        return {**state, "error": f"stitch_node: audio file not found at {audio_path}"}

    publish_job_progress(job_id, 0, 0, "stitching video…")

    tmp_dir = os.path.join(settings.tmp_dir, job_id)
    try:
        os.makedirs(tmp_dir, exist_ok=True)
    except OSError as e:
        logger.exception("[job=%s] stitch_node: could not create tmp dir %s", job_id, tmp_dir)
        return {**state, "error": f"stitch_node: could not create tmp dir {tmp_dir}: {e}"}

    image_path = os.path.join(tmp_dir, "image.jpg")
    video_path = os.path.join(tmp_dir, "video.mp4")

    # Download image from Supabase public URL
    logger.info("[job=%s] stitch_node: downloading image from %s", job_id, image_url)
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(image_url)
            r.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(r.content)
        logger.info("[job=%s] stitch_node: image downloaded (%d bytes)", job_id, len(r.content))
    except httpx.HTTPError as e:
        logger.exception("[job=%s] stitch_node: image download failed", job_id)
        return {**state, "error": str(HTTPFetchError(f"image download failed: {e}", e))}
    except OSError as e:
        logger.exception("[job=%s] stitch_node: could not write image to %s", job_id, image_path)
        return {**state, "error": str(HTTPFetchError(f"could not write image locally: {e}", e))}

    # ffmpeg: loop image over audio duration
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1",
        "-i", image_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        "-movflags", "+faststart",
        video_path,
    ]

    logger.info("[job=%s] stitch_node: running ffmpeg: %s", job_id, " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace")[:500] if e.stderr else ""
        logger.exception("[job=%s] stitch_node: ffmpeg failed: %s", job_id, stderr)
        return {
            **state,
            "error": str(MediaProcessingError(f"ffmpeg failed: {stderr}", e)),
        }
    except FileNotFoundError as e:
        logger.exception("[job=%s] stitch_node: ffmpeg binary not found", job_id)
        return {**state, "error": str(MediaProcessingError("ffmpeg is not installed or not on PATH", e))}

    if not os.path.exists(video_path):
        logger.error("[job=%s] stitch_node: video file not created at %s", job_id, video_path)
        return {**state, "error": "stitch_node: video file not created"}

    logger.info("[job=%s] stitch_node: video stitched successfully at %s", job_id, video_path)
    publish_job_progress(job_id, 0, 0, "video stitched")
    return {**state, "video_path": video_path}
