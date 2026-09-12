"""
story_gen.py — general-purpose long-form story generation.
prompt -> plan (blueprint) -> story bible -> batched prose (<=2 segs/call) -> assembly.
Tone, genre, and style are fully driven by the user prompt. No hardcoded framing.
"""

import os
import json
import time
import math
from groq import Groq
from config import get_settings
settings = get_settings()


MODEL = "openai/gpt-oss-120b"
MAX_OUTPUT_TOKENS = 6000
SEGMENTS_PER_BATCH = 2
MAX_RETRIES = 2
RETRY_DELAY_SEC = 5
WORDS_PER_SEGMENT_TARGET = 375

PLANNER_TEMP = 0.3
PLANNER_REASONING = "medium"
PROSE_TEMP = 0.7
PROSE_REASONING = "low"

_PLAN_SEGMENT_ITEM = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "purpose": {"type": "string"},
        "location": {"type": "string"},
        "time": {"type": "string"},
        "main_event": {"type": "string"},
        "character_development": {"type": "string"},
        "key_details": {"type": "array", "items": {"type": "string"}},
        "avoid_repeating": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "id", "purpose", "location", "time", "main_event",
        "character_development", "key_details", "avoid_repeating",
    ],
    "additionalProperties": False,
}

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "story_arc": {"type": "string"},
        "characters": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
        "segments": {"type": "array", "items": _PLAN_SEGMENT_ITEM},
    },
    "required": ["title", "story_arc", "characters", "locations", "segments"],
    "additionalProperties": False,
}

_PROSE_SEGMENT_ITEM = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "text": {"type": "string"},
    },
    "required": ["id", "text"],
    "additionalProperties": False,
}

_PROSE_BATCH_SCHEMA = {
    "type": "object",
    "properties": {"segments": {"type": "array", "items": _PROSE_SEGMENT_ITEM}},
    "required": ["segments"],
    "additionalProperties": False,
}

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def _sys_prompt_planner() -> str:
    return (
        "You are a story planner. Given a topic or prompt, produce a structured blueprint: "
        "an overall arc, a cast of characters, key locations, and a segment-by-segment plan. "
        "Match the tone, genre, and register the prompt implies — do not impose any fixed style. "
        "Every segment must have a distinct narrative purpose: advancing the plot, developing "
        "characters, building the world, escalating tension, delivering payoff, or resolving "
        "conflict. No two segments should serve the same purpose. For each segment, list concrete "
        "key details introduced there and any details that should not be repeated later. "
        "Follow the JSON schema exactly."
    )


def _sys_prompt_prose() -> str:
    return (
        "You are a prose writer. Given a story plan and bible, write engaging narrative prose "
        "that follows the plan exactly. Match the tone, genre, and register implied by the prompt "
        "and plan — do not impose any fixed style. Maintain continuity with previous context; "
        "do not restate or recap what has already been written. Give each segment its assigned "
        "narrative purpose. Avoid unnecessary repetition of wording, imagery, or details already "
        "used. Follow the JSON schema exactly."
    )


def _print_progress_bar(current: int, total: int, stage: str, width: int = 30) -> None:
    frac = current / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(frac * 100)
    print(f"\r[{bar}] {pct:3d}% ({current}/{total}) {stage}", end="", flush=True)
    if current >= total:
        print()


def _emit_progress(progress: dict, stage: str, increment: bool = True) -> None:
    if increment:
        progress["current"] += 1
    progress["stage"] = stage
    cb = progress.get("on_progress")
    if cb:
        cb({"current": progress["current"], "total": progress["total"], "stage": stage})
    else:
        _print_progress_bar(progress["current"], progress["total"], stage)


def _init_progress(total: int, on_progress=None) -> dict:
    return {"current": 0, "total": total, "stage": "", "on_progress": on_progress}


def calculate_segment_count(
    target_words: int,
    words_per_segment_target: int = WORDS_PER_SEGMENT_TARGET,
) -> dict:
    num_segments = max(1, round(target_words / words_per_segment_target))
    base = target_words // num_segments
    remainder = target_words - base * num_segments
    words_per_segment = [base + 1 if i < remainder else base for i in range(num_segments)]
    return {"num_segments": num_segments, "words_per_segment": words_per_segment}


def create_segment_batches(segments: list, batch_size: int = SEGMENTS_PER_BATCH) -> list:
    return [segments[i : i + batch_size] for i in range(0, len(segments), batch_size)]


def plan_story(
    idea: str,
    num_segments: int,
    words_per_segment: list,
    model: str = MODEL,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict:
    client = _get_client()
    user_payload = (
        f"TOPIC:\n{idea}\n\n"
        f"NUMBER OF SEGMENTS: {num_segments}\n"
        f"APPROX WORDS PER SEGMENT: {words_per_segment}\n\n"
        f"Produce the full blueprint now, one plan entry per segment id 1..{num_segments}, in order."
    )

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _sys_prompt_planner()},
                    {"role": "user", "content": user_payload},
                ],
                temperature=PLANNER_TEMP,
                reasoning_effort=PLANNER_REASONING,
                include_reasoning=False,
                max_completion_tokens=max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "story_plan",
                        "strict": True,
                        "schema": _PLAN_SCHEMA,
                    },
                },
            )
            data = json.loads(resp.choices[0].message.content)
            return {
                "success": True,
                "title": data.get("title", ""),
                "story_arc": data.get("story_arc", ""),
                "characters": data.get("characters", []),
                "locations": data.get("locations", []),
                "segments": data.get("segments", []),
                "error": None,
            }
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    return {
        "success": False,
        "title": "",
        "story_arc": "",
        "characters": [],
        "locations": [],
        "segments": [],
        "error": f"planning failed after {MAX_RETRIES + 1} attempts: {last_err}",
    }


def create_story_bible(idea: str, plan: dict) -> dict:
    return {
        "original_topic": idea,
        "title": plan.get("title", ""),
        "story_arc": plan.get("story_arc", ""),
        "characters": plan.get("characters", []),
        "locations": plan.get("locations", []),
        "segment_plans": {s["id"]: s for s in plan.get("segments", [])},
        "completed_segment_ids": [],
        "current_segment_id": 1,
        "used_details": [],
        "used_locations": [],
        "covered_events": [],
        "covered_character_development": [],
        "continuity_facts": [],
        "avoid_repeating": [],
    }


def update_story_bible(
    bible: dict,
    segment_plan: dict,
    generated_text: str,
    max_tracked: int = 30,
) -> dict:
    sid = segment_plan.get("id")
    bible["completed_segment_ids"].append(sid)
    bible["current_segment_id"] = sid + 1

    bible["used_details"].extend(segment_plan.get("key_details", []))
    bible["used_details"] = bible["used_details"][-max_tracked:]

    loc = segment_plan.get("location")
    if loc and loc not in bible["used_locations"]:
        bible["used_locations"].append(loc)

    event = segment_plan.get("main_event")
    if event:
        bible["covered_events"].append(event)
        bible["covered_events"] = bible["covered_events"][-max_tracked:]

    cdev = segment_plan.get("character_development")
    if cdev:
        bible["covered_character_development"].append(cdev)
        bible["covered_character_development"] = bible["covered_character_development"][-max_tracked:]

    bible["avoid_repeating"].extend(segment_plan.get("avoid_repeating", []))
    bible["avoid_repeating"] = bible["avoid_repeating"][-max_tracked:]

    bible["continuity_facts"].append(generated_text[-400:])
    bible["continuity_facts"] = bible["continuity_facts"][-3:]

    return bible


def generate_segment_batch(
    idea: str,
    batch_segments: list,
    bible: dict,
    context_tail: str,
    model: str = MODEL,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> dict:
    client = _get_client()
    words_per_segment = {s["id"]: s.get("target_words") for s in batch_segments}
    bible_compact = {
        "story_arc": bible.get("story_arc", ""),
        "characters": bible.get("characters", []),
        "locations": bible.get("locations", []),
        "used_locations": bible.get("used_locations", []),
        "covered_events": bible.get("covered_events", []),
        "covered_character_development": bible.get("covered_character_development", []),
    }

    user_payload = (
        f"TOPIC:\n{idea}\n\n"
        f"STORY PLAN FOR CURRENT SEGMENTS:\n{json.dumps(batch_segments, ensure_ascii=False)}\n\n"
        f"STORY BIBLE:\n{json.dumps(bible_compact, ensure_ascii=False)}\n\n"
        f"PREVIOUS CONTEXT (do not restate, continue from here):\n\"...{context_tail}\"\n\n"
        f"USED DETAILS (avoid unnecessary reuse):\n{json.dumps(bible.get('used_details', []), ensure_ascii=False)}\n\n"
        f"AVOID REPEATING:\n{json.dumps(bible.get('avoid_repeating', []), ensure_ascii=False)}\n\n"
        f"TARGET WORD COUNT PER SEGMENT:\n{json.dumps(words_per_segment, ensure_ascii=False)}\n\n"
        "Write the CURRENT SEGMENTS listed above, in order, each following its own plan's purpose exactly."
    )

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _sys_prompt_prose()},
                    {"role": "user", "content": user_payload},
                ],
                temperature=PROSE_TEMP,
                reasoning_effort=PROSE_REASONING,
                include_reasoning=False,
                max_completion_tokens=max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "story_batch",
                        "strict": True,
                        "schema": _PROSE_BATCH_SCHEMA,
                    },
                },
            )
            data = json.loads(resp.choices[0].message.content)
            return {"success": True, "segments": data["segments"], "error": None}
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    return {
        "success": False,
        "segments": [],
        "error": f"batch failed after {MAX_RETRIES + 1} attempts: {last_err}",
    }


def generate_story(
    idea: str,
    wpm: int = 150,
    video_length_min: float = 10.0,
    target_words: int = None,
    num_segments: int = None,
    model: str = MODEL,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    words_per_segment_target: int = WORDS_PER_SEGMENT_TARGET,
    batch_size: int = SEGMENTS_PER_BATCH,
    on_progress=None,
) -> dict:
    """
    idea -> plan -> story bible -> batched prose generation.
    on_progress(dict{current, total, stage}) called on each step if provided.
    """
    idea = (idea or "").strip()
    if not idea:
        return {"success": False, "error": "idea is empty", "segments": []}
    if wpm <= 0 or video_length_min <= 0:
        return {"success": False, "error": "wpm and video_length_min must be > 0", "segments": []}

    target_words = target_words or round(wpm * video_length_min)

    if num_segments:
        base = target_words // num_segments
        remainder = target_words - base * num_segments
        seg_word_targets = [base + 1 if i < remainder else base for i in range(num_segments)]
    else:
        sizing = calculate_segment_count(target_words, words_per_segment_target)
        num_segments = sizing["num_segments"]
        seg_word_targets = sizing["words_per_segment"]

    total_batches = math.ceil(num_segments / batch_size)
    progress = _init_progress(1 + total_batches, on_progress)

    _emit_progress(progress, "planning story", increment=False)
    plan = plan_story(idea, num_segments, seg_word_targets, model, max_output_tokens)
    if not plan["success"]:
        return {
            "success": False,
            "error": plan["error"],
            "idea": idea,
            "title": "",
            "segments": [],
            "wpm": wpm,
            "video_length_min": video_length_min,
            "target_words": target_words,
        }
    _emit_progress(progress, "story planned")

    plan_segments = plan["segments"]
    for i, s in enumerate(plan_segments):
        s["target_words"] = (
            seg_word_targets[i] if i < len(seg_word_targets) else words_per_segment_target
        )

    bible = create_story_bible(idea, plan)
    batches = create_segment_batches(plan_segments, batch_size)

    all_segments = []
    context_tail = ""

    for batch in batches:
        ids = [s["id"] for s in batch]
        _emit_progress(progress, f"generating segments {ids}", increment=False)
        result = generate_segment_batch(idea, batch, bible, context_tail, model, max_output_tokens)
        if not result["success"]:
            return {
                "success": False,
                "error": result["error"],
                "idea": idea,
                "title": plan["title"],
                "segments": all_segments,
                "wpm": wpm,
                "video_length_min": video_length_min,
                "target_words": target_words,
                "segment_plans": bible["segment_plans"],
            }

        for s_plan, s_out in zip(batch, result["segments"]):
            text = s_out["text"]
            all_segments.append(
                {
                    "id": s_plan["id"],
                    "text": text,
                    "original_text": text,
                    "status": "pending",
                }
            )
            bible = update_story_bible(bible, s_plan, text)
            context_tail = text[-400:]
        _emit_progress(progress, f"segments {ids} done")

    return {
        "success": True,
        "idea": idea,
        "title": plan["title"],
        "segments": all_segments,
        "wpm": wpm,
        "video_length_min": video_length_min,
        "target_words": target_words,
        "story_arc": plan["story_arc"],
        "characters": plan["characters"],
        "locations": plan["locations"],
        "segment_plans": bible["segment_plans"],
        "model_used": model,
        "error": None,
    }


def regenerate_segment(
    original_text: str,
    instruction: str,
    idea: str,
    segment_plan: dict = None,
    model: str = MODEL,
    max_output_tokens: int = 2000,
) -> dict:
    """Rewrite one segment per instruction, preserving its planned narrative purpose."""
    plan_note = (
        f"\nSegment purpose: {segment_plan.get('purpose')}" if segment_plan else ""
    )
    prompt = (
        f"Story: {idea}{plan_note}\n\nOriginal segment:\n{original_text}\n\n"
        f"Rewrite per instruction: {instruction or 'improve clarity and pacing'}. "
        "Keep the same rough length and narrative purpose. "
        "Output only the rewritten segment text, no preamble."
    )
    try:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=PROSE_TEMP,
            reasoning_effort=PROSE_REASONING,
            include_reasoning=False,
            max_completion_tokens=max_output_tokens,
        )
        return {
            "success": True,
            "text": resp.choices[0].message.content.strip(),
            "error": None,
        }
    except Exception as e:
        return {"success": False, "text": original_text, "error": str(e)}


def assemble_story(segments: list, title: str = "") -> dict:
    parts = [s["text"] for s in segments if s.get("status") != "skipped"]
    story_text = "\n\n".join(parts)
    return {
        "story_text": story_text,
        "word_count": len(story_text.split()),
        "title": title,
    }
