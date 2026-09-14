"""
voiceover.py — Kokoro-82M TTS narration generator.
Apache-2.0 licensed, commercial-safe. Targets a specific WPM pace with a calibration+correction pass.
"""
from __future__ import annotations
import json, logging, re, shutil, subprocess, wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np
from kokoro import KPipeline

from errors import MediaProcessingError

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000
DEFAULT_CACHE_DIR = "./kokoro_cache"
DEFAULT_SENTENCE_SILENCE_SEC = 0.35
DEFAULT_PARAGRAPH_SILENCE_SEC = 0.9
WPM_TOLERANCE_PCT = 2.0
MAX_CORRECTION_PASSES = 1
MIN_SPEED = 0.5
MAX_SPEED = 2.0

_CALIBRATION_TEXT = (
    "The old lighthouse stood quietly at the edge of the cliff, its light sweeping "
    "slowly across the dark water below. Waves rolled in without hurry, one after "
    "another, folding softly onto the sand. Somewhere in the distance a bell buoy "
    "rang, a low and steady sound that seemed to belong to the night itself. Nothing "
    "moved quickly here. Even the wind took its time, drifting through the tall "
    "grass on the dunes and settling again. A single boat light blinked far out on "
    "the horizon, patient and unhurried, as if it too understood there was nowhere "
    "else it needed to be."
)

_PIPELINE_CACHE: dict[str, KPipeline] = {}


@dataclass
class VoiceoverResult:
    audio_path: str
    duration_sec: float
    word_count: int
    target_wpm: float
    actual_wpm: float
    speed_used: float
    voice_model: str
    correction_passes_used: int

    def as_dict(self) -> dict:
        return asdict(self)


def _lang_code_for_voice(voice_model: str) -> str:
    return voice_model[0]


def _load_pipeline(lang_code: str, use_cuda: bool = False) -> KPipeline:
    cache_key = f"{lang_code}:{use_cuda}"
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    logger.info("Loading Kokoro pipeline (lang_code=%s, use_cuda=%s)", lang_code, use_cuda)
    if use_cuda:
        try:
            import torch
        except ImportError as e:
            raise MediaProcessingError("use_cuda=True but torch is not installed", e) from e
        if not torch.cuda.is_available():
            raise MediaProcessingError("use_cuda=True but no CUDA device available")
        device = "cuda"
    else:
        device = "cpu"

    try:
        pipeline = KPipeline(lang_code=lang_code, device=device, repo_id="hexgrad/Kokoro-82M")
    except Exception as e:
        logger.exception("Failed to load Kokoro pipeline")
        raise MediaProcessingError(f"failed to load Kokoro pipeline: {e}", e) from e

    _PIPELINE_CACHE[cache_key] = pipeline
    logger.info("Kokoro pipeline loaded (lang_code=%s, device=%s)", lang_code, device)
    return pipeline


def _extract_audio(result) -> np.ndarray:
    if hasattr(result, "audio"):
        audio = result.audio
    elif hasattr(result, "output") and hasattr(result.output, "audio"):
        audio = result.output.audio
    else:
        audio = result[2]
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio)


def _calibration_cache_path(cache_dir: str) -> Path:
    return Path(cache_dir) / ".wpm_calibration_cache.json"


def _get_words_per_sec_at_speed1(pipeline: KPipeline, voice_model: str, cache_dir: str) -> float:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = _calibration_cache_path(cache_dir)
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Calibration cache at %s unreadable, rebuilding: %s", cache_path, e)
            cache = {}
    if voice_model in cache:
        logger.debug("Calibration cache hit for voice=%s", voice_model)
        return cache[voice_model]

    logger.info("Calibrating words/sec for voice=%s (no cache hit)", voice_model)
    try:
        total_samples = 0
        for result in pipeline(_CALIBRATION_TEXT, voice=voice_model, speed=1.0):
            total_samples += len(_extract_audio(result))
    except Exception as e:
        logger.exception("Kokoro calibration synthesis failed for voice=%s", voice_model)
        raise MediaProcessingError(f"calibration synthesis failed for voice {voice_model}: {e}", e) from e

    duration_sec = total_samples / SAMPLE_RATE
    if duration_sec <= 0:
        raise MediaProcessingError(f"calibration produced zero-length audio for voice {voice_model}")
    word_count = len(_CALIBRATION_TEXT.split())
    words_per_sec = word_count / duration_sec
    cache[voice_model] = words_per_sec
    try:
        cache_path.write_text(json.dumps(cache, indent=2))
    except OSError as e:
        logger.warning("Failed to persist calibration cache at %s: %s", cache_path, e)
    logger.info("Calibration complete for voice=%s: %.2f words/sec", voice_model, words_per_sec)
    return words_per_sec


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [" ".join(p.split()) for p in paragraphs if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [p for p in parts if p.strip()]


def _silence_bytes(sample_rate: int, sample_width: int, n_channels: int, seconds: float) -> bytes:
    n_frames = max(0, int(sample_rate * seconds))
    return bytes(n_frames * sample_width * n_channels)


def _synthesize_paragraphs_to_wav(
    pipeline, voice_model, paragraphs, wav_path, speed, sentence_silence, paragraph_silence
) -> float:
    sample_width = 2
    n_channels = 1
    total_frames = 0
    logger.info("Synthesizing %d paragraph(s) at speed=%.3f", len(paragraphs), speed)
    try:
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.setsampwidth(sample_width)
            wav_file.setnchannels(n_channels)
            for p_idx, paragraph in enumerate(paragraphs):
                sentences = _split_sentences(paragraph)
                chunk_text = "\n".join(sentences)
                results = list(pipeline(chunk_text, voice=voice_model, speed=speed, split_pattern=r"\n+"))
                for c_idx, result in enumerate(results):
                    audio = _extract_audio(result)
                    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                    wav_file.writeframes(pcm)
                    total_frames += len(pcm) // (sample_width * n_channels)
                    if c_idx != len(results) - 1:
                        sil = _silence_bytes(SAMPLE_RATE, sample_width, n_channels, sentence_silence)
                        wav_file.writeframes(sil)
                        total_frames += len(sil) // (sample_width * n_channels)
                if p_idx != len(paragraphs) - 1:
                    sil = _silence_bytes(SAMPLE_RATE, sample_width, n_channels, paragraph_silence)
                    wav_file.writeframes(sil)
                    total_frames += len(sil) // (sample_width * n_channels)
                logger.debug("Synthesized paragraph %d/%d", p_idx + 1, len(paragraphs))
    except MediaProcessingError:
        raise
    except Exception as e:
        logger.exception("Kokoro synthesis failed while writing %s", wav_path)
        raise MediaProcessingError(f"synthesis failed: {e}", e) from e

    duration_sec = total_frames / SAMPLE_RATE
    logger.info("Synthesis complete: %.1fs of audio written to %s", duration_sec, wav_path)
    return duration_sec


def _encode_final_output(wav_path: Path, output_path: Path) -> None:
    if output_path.suffix.lower() == ".wav":
        shutil.move(str(wav_path), str(output_path))
        logger.info("Output is .wav, moved directly to %s", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-ar", "44100"]
    if output_path.suffix.lower() == ".mp3":
        cmd += ["-codec:a", "libmp3lame", "-b:a", "128k"]
    cmd += [str(output_path)]
    logger.info("Encoding final audio via ffmpeg: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("ffmpeg encode complete: %s", output_path)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace")[:500] if e.stderr else ""
        logger.exception("ffmpeg encode failed: %s", stderr)
        raise MediaProcessingError(f"ffmpeg encode failed: {stderr}", e) from e
    except FileNotFoundError as e:
        logger.exception("ffmpeg binary not found on PATH")
        raise MediaProcessingError("ffmpeg is not installed or not on PATH", e) from e
    finally:
        wav_path.unlink(missing_ok=True)


def generate_voiceover(
    story_text: str,
    output_path: str,
    voice_model: str = DEFAULT_VOICE,
    target_wpm: float = 140.0,
    cache_dir: str = DEFAULT_CACHE_DIR,
    sentence_silence: float = DEFAULT_SENTENCE_SILENCE_SEC,
    paragraph_silence: float = DEFAULT_PARAGRAPH_SILENCE_SEC,
    use_cuda: bool = False,
) -> dict:
    logger.info("generate_voiceover: voice=%s target_wpm=%s output=%s", voice_model, target_wpm, output_path)
    if not (100 <= target_wpm <= 200):
        logger.warning("generate_voiceover: %s wpm outside typical 100-200 range", target_wpm)

    lang_code = _lang_code_for_voice(voice_model)
    pipeline = _load_pipeline(lang_code, use_cuda)
    words_per_sec_at_1 = _get_words_per_sec_at_speed1(pipeline, voice_model, cache_dir)

    paragraphs = _split_paragraphs(story_text)
    if not paragraphs:
        raise MediaProcessingError("story_text is empty after cleanup")

    word_count = len(story_text.split())
    total_sentences = sum(max(1, len(_split_sentences(p))) for p in paragraphs)
    n_paragraphs = len(paragraphs)
    estimated_pause_sec = (
        (total_sentences - n_paragraphs) * sentence_silence
        + max(0, n_paragraphs - 1) * paragraph_silence
    )
    speech_duration_at_1 = word_count / words_per_sec_at_1
    target_total_duration = word_count / target_wpm * 60.0
    target_speech_duration = max(1.0, target_total_duration - estimated_pause_sec)
    speed = speech_duration_at_1 / target_speech_duration
    speed = min(max(speed, MIN_SPEED), MAX_SPEED)

    out_path = Path(output_path)
    tmp_wav = out_path.with_suffix(".tmp.wav")
    passes_used = 0
    duration_sec = 0.0
    actual_wpm = 0.0

    for attempt in range(MAX_CORRECTION_PASSES + 1):
        logger.info("generate_voiceover: synthesis attempt %d/%d, speed=%.3f",
                    attempt + 1, MAX_CORRECTION_PASSES + 1, speed)
        duration_sec = _synthesize_paragraphs_to_wav(
            pipeline, voice_model, paragraphs, tmp_wav, speed, sentence_silence, paragraph_silence
        )
        actual_wpm = word_count / (duration_sec / 60.0)
        drift_pct = abs(actual_wpm - target_wpm) / target_wpm * 100.0
        logger.info("generate_voiceover: actual_wpm=%.1f target_wpm=%.1f drift=%.2f%%",
                    actual_wpm, target_wpm, drift_pct)
        if drift_pct <= WPM_TOLERANCE_PCT or attempt == MAX_CORRECTION_PASSES:
            break
        correction_ratio = duration_sec / target_total_duration
        speed = min(max(speed * correction_ratio, MIN_SPEED), MAX_SPEED)
        passes_used += 1
        logger.info("generate_voiceover: drift too high, correcting speed -> %.3f", speed)

    _encode_final_output(tmp_wav, out_path)

    logger.info("generate_voiceover: done, %.1fs audio at %.1f wpm -> %s",
                duration_sec, actual_wpm, out_path)
    return VoiceoverResult(
        audio_path=str(out_path),
        duration_sec=round(duration_sec, 2),
        word_count=word_count,
        target_wpm=target_wpm,
        actual_wpm=round(actual_wpm, 1),
        speed_used=round(speed, 4),
        voice_model=voice_model,
        correction_passes_used=passes_used,
    ).as_dict()
