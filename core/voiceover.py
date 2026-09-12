"""
voiceover.py — Kokoro-82M TTS narration generator.
Apache-2.0 licensed, commercial-safe. Targets a specific WPM pace with a calibration+correction pass.
"""
from __future__ import annotations
import json, re, shutil, subprocess, wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np
from kokoro import KPipeline

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
    if use_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("use_cuda=True but no CUDA device available.")
        device = "cuda"
    else:
        device = "cpu"
    pipeline = KPipeline(lang_code=lang_code, device=device, repo_id="hexgrad/Kokoro-82M")
    _PIPELINE_CACHE[cache_key] = pipeline
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
        except (json.JSONDecodeError, OSError):
            cache = {}
    if voice_model in cache:
        return cache[voice_model]
    total_samples = 0
    for result in pipeline(_CALIBRATION_TEXT, voice=voice_model, speed=1.0):
        total_samples += len(_extract_audio(result))
    duration_sec = total_samples / SAMPLE_RATE
    word_count = len(_CALIBRATION_TEXT.split())
    words_per_sec = word_count / duration_sec
    cache[voice_model] = words_per_sec
    cache_path.write_text(json.dumps(cache, indent=2))
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
    return total_frames / SAMPLE_RATE


def _encode_final_output(wav_path: Path, output_path: Path) -> None:
    if output_path.suffix.lower() == ".wav":
        shutil.move(str(wav_path), str(output_path))
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-ar", "44100"]
    if output_path.suffix.lower() == ".mp3":
        cmd += ["-codec:a", "libmp3lame", "-b:a", "128k"]
    cmd += [str(output_path)]
    try:
        subprocess.run(cmd, check=True)
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
    if not (100 <= target_wpm <= 200):
        print(f"[voiceover] warning: {target_wpm} wpm outside typical 100-200 range.")
    lang_code = _lang_code_for_voice(voice_model)
    pipeline = _load_pipeline(lang_code, use_cuda)
    words_per_sec_at_1 = _get_words_per_sec_at_speed1(pipeline, voice_model, cache_dir)
    paragraphs = _split_paragraphs(story_text)
    if not paragraphs:
        raise ValueError("story_text is empty after cleanup")
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
    for attempt in range(MAX_CORRECTION_PASSES + 1):
        duration_sec = _synthesize_paragraphs_to_wav(
            pipeline, voice_model, paragraphs, tmp_wav, speed, sentence_silence, paragraph_silence
        )
        actual_wpm = word_count / (duration_sec / 60.0)
        drift_pct = abs(actual_wpm - target_wpm) / target_wpm * 100.0
        if drift_pct <= WPM_TOLERANCE_PCT or attempt == MAX_CORRECTION_PASSES:
            break
        correction_ratio = duration_sec / target_total_duration
        speed = min(max(speed * correction_ratio, MIN_SPEED), MAX_SPEED)
        passes_used += 1
    _encode_final_output(tmp_wav, out_path)
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
