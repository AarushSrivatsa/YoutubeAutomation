"""
stitch_node.py — downloads the image from Supabase URL, stitches with audio via ffmpeg.
Output: a local .mp4 in the job tmp dir.
"""
from __future__ import annotations
import os
import subprocess
import httpx
from config import get_settings
from core.state import PipelineState
from services.redis_service import publish_job_progress

settings = get_settings()


def stitch_node(state: PipelineState) -> PipelineState:
    job_id = state.get("job_id", "")
    image_url = state.get("image_url", "")
    audio_path = state.get("audio_path", "")

    if not image_url:
        return {**state, "error": "stitch_node: image_url is missing"}
    if not audio_path or not os.path.exists(audio_path):
        return {**state, "error": f"stitch_node: audio file not found at {audio_path}"}

    publish_job_progress(job_id, 0, 0, "stitching video…")

    tmp_dir = os.path.join(settings.tmp_dir, job_id)
    os.makedirs(tmp_dir, exist_ok=True)
    image_path = os.path.join(tmp_dir, "image.jpg")
    video_path = os.path.join(tmp_dir, "video.mp4")

    # Download image from Supabase public URL
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(image_url)
            r.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(r.content)
    except Exception as e:
        return {**state, "error": f"stitch_node: image download failed: {e}"}

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

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return {
            **state,
            "error": f"stitch_node: ffmpeg failed: {e.stderr.decode()[:500]}",
        }

    if not os.path.exists(video_path):
        return {**state, "error": "stitch_node: video file not created"}

    publish_job_progress(job_id, 0, 0, "video stitched")
    return {**state, "video_path": video_path}
