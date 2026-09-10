"""
voiceover.py
============
Generate natural-sounding narration voiceovers from story text, using
Kokoro-82M TTS, targeting a specific words-per-minute (WPM) pace.

Why Kokoro, not Piper or XTTS:
    XTTS-v2's weights are Coqui Public Model License (CPML) -- NON-COMMERCIAL
    use only. A YPP-monetized channel is commercial use, so XTTS narration
    on a monetized channel is a licensing violation regardless of what
    YouTube itself checks.
    Piper (MIT) is commercial-safe but its small VITS models have a
    naturalness ceiling -- flat, "typewriter" prosody that noise_scale/
    length_scale tuning only marginally improves.
    Kokoro-82M is Apache-2.0 (commercial-safe, confirmed on its model card:
    hf.co/hexgrad/Kokoro-82M) and, despite being only 82M params, is a
    StyleTTS2-based model -- a real step up in naturalness over Piper's VITS
    architecture, not just a parameter tweak.

Usage as a plain function (LangGraph-node-ready -- pure in, dict out, no
globals mutated except an on-disk WPM-calibration cache):

    from voiceover import generate_voiceover

    result = generate_voiceover(
        story_text=my_story,
        output_path="output/ep01.mp3",
        target_wpm=140,
    )
    # result["audio_path"], result["actual_wpm"], result["duration_sec"], ...

Setup (first run downloads ~300MB of Kokoro weights from Hugging Face via
the `kokoro` package's own huggingface_hub caching -- no manual model
management needed, unlike Piper):

    pip install kokoro soundfile
    sudo apt-get install espeak-ng   # English out-of-dictionary word fallback

Kokoro voice names encode language + gender in their prefix:
    af_/am_ = American English ('a'), bf_/bm_ = British English ('b').
Mixing a voice with the wrong lang_code silently degrades pronunciation, so
this module derives lang_code from the voice name automatically -- don't
pass en_US/en_GB-style names here, those are Piper's naming scheme, not
Kokoro's.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from kokoro import KPipeline

# --------------------------------------------------------------------------- #
# Config / constants
# --------------------------------------------------------------------------- #

DEFAULT_VOICE = "af_heart"   # warm American female -- good default for sleep narration
# Other calm picks:
#   "am_michael"  (calm American male)
#   "bm_george"   (calm British male)

SAMPLE_RATE = 24000   # fixed by Kokoro, not configurable
DEFAULT_CACHE_DIR = "./kokoro_cache"   # holds only OUR wpm-calibration cache --
                                        # Kokoro's own weights live in the
                                        # standard huggingface_hub cache
DEFAULT_SENTENCE_SILENCE_SEC = 0.35   # pause between sentences
DEFAULT_PARAGRAPH_SILENCE_SEC = 0.9   # longer pause between paragraphs (blank line)
WPM_TOLERANCE_PCT = 2.0               # acceptable drift before a correction pass
MAX_CORRECTION_PASSES = 1
MIN_SPEED = 0.5
MAX_SPEED = 2.0

# A ~120-word passage with normal sentence variety, used once per voice to
# measure that voice's intrinsic speaking rate at speed=1.0.
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

# module-level cache so repeated calls (e.g. many LangGraph node invocations in
# one process) don't reload the pipeline/model every time
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


# --------------------------------------------------------------------------- #
# Pipeline loading + calibration
# --------------------------------------------------------------------------- #

def _lang_code_for_voice(voice_model: str) -> str:
    """Kokoro voice names encode lang as their first letter (af_/am_ -> 'a',
    bf_/bm_ -> 'b', jf_/jm_ -> 'j', etc). Holds for every documented voicepack."""
    return voice_model[0]


def _load_pipeline(lang_code: str, use_cuda: bool = False) -> KPipeline:
    cache_key = f"{lang_code}:{use_cuda}"
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    if use_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("use_cuda=True but no CUDA device is available.")
        device = "cuda"
    else:
        device = "cpu"

    pipeline = KPipeline(lang_code=lang_code, device=device, repo_id="hexgrad/Kokoro-82M")
    _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def _extract_audio(result) -> np.ndarray:
    """kokoro's pipeline() yields a Result whose .audio is a torch.FloatTensor
    (older versions/forks may yield a plain (graphemes, phonemes, audio)
    tuple instead) -- handle both, and convert from torch to numpy either way
    so a package-version bump doesn't silently break this."""
    if hasattr(result, "audio"):
        audio = result.audio
    elif hasattr(result, "output") and hasattr(result.output, "audio"):
        audio = result.output.audio
    else:
        audio = result[2]  # tuple form: (graphemes, phonemes, audio)
    if hasattr(audio, "detach"):  # torch.Tensor -> numpy
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio)


def _calibration_cache_path(cache_dir: str) -> Path:
    return Path(cache_dir) / ".wpm_calibration_cache.json"


def _get_words_per_sec_at_speed1(pipeline: KPipeline, voice_model: str,
                                  cache_dir: str) -> float:
    """Words/sec at speed=1.0 for this voice. Cached to disk so it's only
    measured once per voice, ever (not once per process)."""
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
    word_count = _count_words(_CALIBRATION_TEXT)
    words_per_sec = word_count / duration_sec

    cache[voice_model] = words_per_sec
    cache_path.write_text(json.dumps(cache, indent=2))
    return words_per_sec


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

def _count_words(text: str) -> int:
    return len(text.split())


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [" ".join(p.split()) for p in paragraphs if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [p for p in parts if p.strip()]


def _estimate_sentence_count(paragraph: str) -> int:
    return max(1, len(_split_sentences(paragraph)))


def _silence_bytes(sample_rate: int, sample_width: int, n_channels: int,
                    seconds: float) -> bytes:
    n_frames = max(0, int(sample_rate * seconds))
    return bytes(n_frames * sample_width * n_channels)


# --------------------------------------------------------------------------- #
# Core synthesis
# --------------------------------------------------------------------------- #

def _synthesize_paragraphs_to_wav(
    pipeline: KPipeline,
    voice_model: str,
    paragraphs: list[str],
    wav_path: Path,
    speed: float,
    sentence_silence: float,
    paragraph_silence: float,
) -> float:
    """Writes narration to wav_path, returns duration in seconds."""
    sample_width = 2  # 16-bit PCM
    n_channels = 1
    total_frames = 0

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.setsampwidth(sample_width)
        wav_file.setnchannels(n_channels)

        for p_idx, paragraph in enumerate(paragraphs):
            sentences = _split_sentences(paragraph)
            # Newline-join so Kokoro's default split_pattern (r'\n+') yields
            # one result per sentence -- gives us the same sentence-level
            # pause control the Piper version had.
            chunk_text = "\n".join(sentences)
            results = list(pipeline(chunk_text, voice=voice_model, speed=speed,
                                     split_pattern=r"\n+"))

            for c_idx, result in enumerate(results):
                audio = _extract_audio(result)
                pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                wav_file.writeframes(pcm)
                total_frames += len(pcm) // (sample_width * n_channels)

                is_last_sentence_in_paragraph = c_idx == len(results) - 1
                if not is_last_sentence_in_paragraph:
                    sil = _silence_bytes(SAMPLE_RATE, sample_width, n_channels,
                                         sentence_silence)
                    wav_file.writeframes(sil)
                    total_frames += len(sil) // (sample_width * n_channels)

            is_last_paragraph = p_idx == len(paragraphs) - 1
            if not is_last_paragraph:
                sil = _silence_bytes(SAMPLE_RATE, sample_width, n_channels,
                                     paragraph_silence)
                wav_file.writeframes(sil)
                total_frames += len(sil) // (sample_width * n_channels)

    return total_frames / SAMPLE_RATE


def _encode_final_output(wav_path: Path, output_path: Path) -> None:
    """Converts the intermediate wav to the requested container/codec via
    ffmpeg, if output_path isn't already .wav."""
    if output_path.suffix.lower() == ".wav":
        shutil.move(str(wav_path), str(output_path))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(wav_path),
        "-ar", "44100",
    ]
    if output_path.suffix.lower() == ".mp3":
        cmd += ["-codec:a", "libmp3lame", "-b:a", "128k"]
    cmd += [str(output_path)]

    try:
        subprocess.run(cmd, check=True)
    finally:
        wav_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Public function
# --------------------------------------------------------------------------- #

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
    """
    Generate a narration voiceover for `story_text` at `target_wpm`, written to
    `output_path` (.wav or .mp3 -- anything else is passed to ffmpeg as-is).

    Returns a plain dict (LangGraph-friendly):
        {
          "audio_path": str,
          "duration_sec": float,
          "word_count": int,
          "target_wpm": float,
          "actual_wpm": float,
          "speed_used": float,
          "voice_model": str,
          "correction_passes_used": int,
        }
    """
    if not (100 <= target_wpm <= 200):
        # not a hard block -- just flags that you've left "natural narration" territory
        print(f"[voiceover] warning: {target_wpm} wpm is outside the typical "
              f"100-200 natural-narration range; audio may sound off.")

    lang_code = _lang_code_for_voice(voice_model)
    pipeline = _load_pipeline(lang_code, use_cuda)
    words_per_sec_at_1 = _get_words_per_sec_at_speed1(pipeline, voice_model, cache_dir)

    paragraphs = _split_paragraphs(story_text)
    if not paragraphs:
        raise ValueError("story_text is empty after cleanup")

    word_count = _count_words(story_text)
    sentence_counts = [_estimate_sentence_count(p) for p in paragraphs]
    total_sentences = sum(sentence_counts)
    n_paragraphs = len(paragraphs)

    # estimated non-speech (pause) time baked into the final render
    estimated_pause_sec = (
        (total_sentences - n_paragraphs) * sentence_silence
        + max(0, n_paragraphs - 1) * paragraph_silence
    )

    speech_duration_at_1 = word_count / words_per_sec_at_1
    target_total_duration = word_count / target_wpm * 60.0
    target_speech_duration = max(1.0, target_total_duration - estimated_pause_sec)

    # Kokoro's `speed` is a direct multiplier: higher = faster = shorter audio.
    # This is the INVERSE relationship of Piper's length_scale (higher =
    # slower = longer), so the calibration math is flipped accordingly.
    speed = speech_duration_at_1 / target_speech_duration
    speed = min(max(speed, MIN_SPEED), MAX_SPEED)

    out_path = Path(output_path)
    tmp_wav = out_path.with_suffix(".tmp.wav")

    passes_used = 0
    for attempt in range(MAX_CORRECTION_PASSES + 1):
        duration_sec = _synthesize_paragraphs_to_wav(
            pipeline, voice_model, paragraphs, tmp_wav, speed,
            sentence_silence, paragraph_silence,
        )
        actual_wpm = word_count / (duration_sec / 60.0)
        drift_pct = abs(actual_wpm - target_wpm) / target_wpm * 100.0

        if drift_pct <= WPM_TOLERANCE_PCT or attempt == MAX_CORRECTION_PASSES:
            break

        # correct proportionally and re-render (inverse relationship: if the
        # render came out too long, INCREASE speed, don't decrease it)
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


# --------------------------------------------------------------------------- #
# CLI (manual testing)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a narration voiceover")
    parser.add_argument("text_file", help="Path to a .txt file with the story")
    parser.add_argument("output_path", help="Output audio path (.wav or .mp3)")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--wpm", type=float, default=140.0)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    text = Path(args.text_file).read_text()
    result = generate_voiceover(
        story_text=text,
        output_path=args.output_path,
        voice_model=args.voice,
        target_wpm=args.wpm,
        cache_dir=args.cache_dir,
        use_cuda=args.cuda,
    )
    print(json.dumps(result, indent=2))