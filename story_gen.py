"""
story_gen.py — planned, bible-driven story generation + human-in-loop segment review.
Idea -> plan (blueprint) -> story bible -> batched prose (<=2 segs/call) -> human keep/edit/skip/regenerate -> final text.

target_words = wpm * video_length_min (unless target_words passed explicitly). wpm should match
voiceover.py's target_wpm so script length lines up with intended audio duration.
"""

import os
import json
import time
from groq import Groq
from langgraph.types import interrupt

MODEL = "openai/gpt-oss-120b"       # Apache-2.0, Groq production tier, strict JSON support
MAX_OUTPUT_TOKENS = 6000            # per-call completion cap, stays well under 65536 limit + TPM-friendly
SEGMENTS_PER_BATCH = 2               # max prose segments requested per call (adaptive, never exceeded)
MAX_RETRIES = 2                      # per-call retry on API/JSON failure (rate limits etc.)
RETRY_DELAY_SEC = 5

DEFAULT_WPM = 150                    # match voiceover.py target_wpm default
DEFAULT_VIDEO_LENGTH_MIN = 60
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
        "avoid_repeating": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["id", "purpose", "location", "time", "main_event",
                 "character_development", "key_details", "avoid_repeating"],
    "additionalProperties": False
}

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "story_arc": {"type": "string"},
        "characters": {"type": "array", "items": {"type": "string"}},
        "locations": {"type": "array", "items": {"type": "string"}},
        "segments": {"type": "array", "items": _PLAN_SEGMENT_ITEM}
    },
    "required": ["title", "story_arc", "characters", "locations", "segments"],
    "additionalProperties": False
}

_PROSE_SEGMENT_ITEM = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "text": {"type": "string"}
    },
    "required": ["id", "text"],
    "additionalProperties": False
}

_PROSE_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {"type": "array", "items": _PROSE_SEGMENT_ITEM}
    },
    "required": ["segments"],
    "additionalProperties": False
}

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _sys_prompt_planner() -> str:
    """Generic planner behavior only. No topic/character/location specifics live here."""
    return (
        "You are a story planner for a relaxation/sleep YouTube channel. Given a topic, produce a "
        "structured blueprint for a calm, slow-paced sleep story: an overall arc, character and "
        "location lists, and a segment-by-segment plan. Every segment must have a distinct "
        "narrative purpose — it should advance event progression, character development, "
        "relationship development, location progression, discovery, realization, decision, "
        "life-stage progression, or the consequence of a prior event. Atmosphere alone is not a "
        "valid segment purpose. Only the final segment should naturally transition toward sleep or "
        "rest unless the topic itself calls for a different structure; earlier segments should "
        "continue the story. Strip out violence, danger, action, conflict, jump-scares, sudden "
        "events, and cliffhangers, while keeping the topic's world, characters, and identity "
        "recognizable throughout every segment. For each segment, list concrete key details "
        "introduced there and any details or imagery that should not be repeated later. Follow the "
        "JSON schema exactly."
    )


def _sys_prompt_prose() -> str:
    """Generic prose-writer behavior only. No topic/character/location specifics live here."""
    return (
        "You write sleep stories for a relaxation YouTube channel. Priority: comfortable, not "
        "beautiful. Plain, warm, unpolished-sounding sentences — not literature, not purple prose, "
        "no metaphor stacking, no thesaurus words. Write like a real person talking softly to "
        "someone falling asleep next to them at 1am. Short simple sentences. Everyday words. Some "
        "repetition of sentence rhythm is fine and sounds human, but avoid unnecessary repetition of "
        "wording, imagery, actions, and sensory details already used in the story. Intimate, close, "
        "low-key, slow pacing. Follow the supplied story plan and story bible exactly: stay in the "
        "provided topic's world, keep the supplied characters and locations consistent, and give "
        "each segment the narrative purpose it was assigned rather than inventing a new one. Do not "
        "introduce conflict, danger, or abrupt events unless the supplied plan requires it. Do not "
        "end every segment on falling asleep, drifting off, breathing slowing, returning to bed, or "
        "sitting quietly — only wrap toward sleep when the current segment's plan calls for it. "
        "Maintain continuity with the previous context and do not restate or recap it. Follow the "
        "JSON schema exactly."
    )


def calculate_segment_count(target_words: int, words_per_segment_target: int = WORDS_PER_SEGMENT_TARGET) -> dict:
    """target_words -> num_segments + evenly distributed per-segment word targets (no large gaps).
    OUT: num_segments, words_per_segment (list, len == num_segments)"""
    num_segments = max(1, round(target_words / words_per_segment_target))
    base = target_words // num_segments
    remainder = target_words - base * num_segments
    words_per_segment = [base + 1 if i < remainder else base for i in range(num_segments)]
    return {"num_segments": num_segments, "words_per_segment": words_per_segment}


def create_segment_batches(segments: list, batch_size: int = SEGMENTS_PER_BATCH) -> list:
    """Split the planner's segment list into sequential batches, never exceeding batch_size."""
    return [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]


def plan_story(idea: str, num_segments: int, words_per_segment: list,
               model: str = MODEL, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    """idea + segment/word targets -> structured blueprint (title, arc, cast, locations, per-segment
    purpose/location/time/event/character_development/key_details/avoid_repeating).
    OUT: success, title, story_arc, characters, locations, segments, error"""
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
                    {"role": "user", "content": user_payload}
                ],
                temperature=PLANNER_TEMP,
                reasoning_effort=PLANNER_REASONING,
                include_reasoning=False,
                max_completion_tokens=max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "story_plan", "strict": True, "schema": _PLAN_SCHEMA}
                }
            )
            data = json.loads(resp.choices[0].message.content)
            return {
                "success": True,
                "title": data.get("title", ""),
                "story_arc": data.get("story_arc", ""),
                "characters": data.get("characters", []),
                "locations": data.get("locations", []),
                "segments": data.get("segments", []),
                "error": None
            }
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    return {
        "success": False, "title": "", "story_arc": "", "characters": [], "locations": [],
        "segments": [], "error": f"planning failed after {MAX_RETRIES + 1} attempts: {last_err}"
    }


def create_story_bible(idea: str, plan: dict) -> dict:
    """Persistent continuity state generated from the planner. Primary continuity mechanism —
    context_tail is kept only as a short-range supplement."""
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
        "avoid_repeating": []
    }


def update_story_bible(bible: dict, segment_plan: dict, generated_text: str, max_tracked: int = 30) -> dict:
    """Fold a just-generated segment's plan + text back into the bible. Lists are capped so later
    prompts stay compact rather than growing unbounded."""
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


def generate_segment_batch(idea: str, batch_segments: list, bible: dict, context_tail: str,
                            model: str = MODEL, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    """Generate prose for one batch (1-2 planned segments), driven by story bible + dynamic
    repetition control rather than hardcoded phrase lists.
    OUT: success, segments:[{id,text}], error"""
    client = _get_client()

    words_per_segment = {s["id"]: s.get("target_words") for s in batch_segments}
    bible_compact = {
        "story_arc": bible.get("story_arc", ""),
        "characters": bible.get("characters", []),
        "locations": bible.get("locations", []),
        "used_locations": bible.get("used_locations", []),
        "covered_events": bible.get("covered_events", []),
        "covered_character_development": bible.get("covered_character_development", [])
    }

    user_payload = (
        f"TOPIC:\n{idea}\n\n"
        f"STORY PLAN FOR CURRENT SEGMENTS:\n{json.dumps(batch_segments, ensure_ascii=False)}\n\n"
        f"STORY BIBLE:\n{json.dumps(bible_compact, ensure_ascii=False)}\n\n"
        f"PREVIOUS CONTEXT (do not restate, continue from here):\n\"...{context_tail}\"\n\n"
        f"USED DETAILS (avoid unnecessary reuse):\n{json.dumps(bible.get('used_details', []), ensure_ascii=False)}\n\n"
        f"AVOID REPEATING:\n{json.dumps(bible.get('avoid_repeating', []), ensure_ascii=False)}\n\n"
        f"TARGET WORD COUNT PER SEGMENT:\n{json.dumps(words_per_segment, ensure_ascii=False)}\n\n"
        "Write the CURRENT SEGMENTS listed above, in order, each following its own plan's purpose "
        "exactly. Output only the requested segment ids."
    )

    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _sys_prompt_prose()},
                    {"role": "user", "content": user_payload}
                ],
                temperature=PROSE_TEMP,
                reasoning_effort=PROSE_REASONING,
                include_reasoning=False,
                max_completion_tokens=max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "sleep_story_batch", "strict": True, "schema": _PROSE_BATCH_SCHEMA}
                }
            )
            data = json.loads(resp.choices[0].message.content)
            return {"success": True, "segments": data["segments"], "error": None}
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)

    return {"success": False, "segments": [], "error": f"batch failed after {MAX_RETRIES + 1} attempts: {last_err}"}


def generate_story(idea: str, wpm: int = DEFAULT_WPM, video_length_min: float = DEFAULT_VIDEO_LENGTH_MIN,
                    target_words: int = None, num_segments: int = None,
                    model: str = MODEL, max_output_tokens: int = MAX_OUTPUT_TOKENS,
                    words_per_segment_target: int = WORDS_PER_SEGMENT_TARGET,
                    batch_size: int = SEGMENTS_PER_BATCH) -> dict:
    """
    idea -> plan -> story bible -> batched (<=2/call) prose generation.
    target_words derives from wpm * video_length_min unless passed directly.
    OUT: success, idea, title, segments:[{id,text,original_text,status}], wpm, video_length_min,
         target_words, story_arc, characters, locations, segment_plans, model_used, error
    """
    idea = (idea or "").strip()
    if not idea:
        return {"success": False, "error": "idea empty", "segments": []}

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

    plan = plan_story(idea, num_segments, seg_word_targets, model, max_output_tokens)
    if not plan["success"]:
        return {
            "success": False, "error": plan["error"], "idea": idea, "title": "", "segments": [],
            "wpm": wpm, "video_length_min": video_length_min, "target_words": target_words
        }

    plan_segments = plan["segments"]
    for i, s in enumerate(plan_segments):
        s["target_words"] = seg_word_targets[i] if i < len(seg_word_targets) else words_per_segment_target

    bible = create_story_bible(idea, plan)
    batches = create_segment_batches(plan_segments, batch_size)

    all_segments = []
    context_tail = ""

    for batch in batches:
        result = generate_segment_batch(idea, batch, bible, context_tail, model, max_output_tokens)
        if not result["success"]:
            return {
                "success": False, "error": result["error"], "idea": idea, "title": plan["title"],
                "segments": all_segments, "wpm": wpm, "video_length_min": video_length_min,
                "target_words": target_words, "segment_plans": bible["segment_plans"]
            }

        for s_plan, s_out in zip(batch, result["segments"]):
            text = s_out["text"]
            all_segments.append({
                "id": s_plan["id"], "text": text, "original_text": text, "status": "pending"
            })
            bible = update_story_bible(bible, s_plan, text)
            context_tail = text[-400:]

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
        "error": None
    }


def regenerate_segment(original_text: str, instruction: str, idea: str, segment_plan: dict = None,
                        model: str = MODEL, max_output_tokens: int = 2000) -> dict:
    """Rewrite one segment per human instruction, keeping its planned narrative purpose.
    OUT: success, text, error"""
    plan_note = f"\nSegment purpose: {segment_plan.get('purpose')}" if segment_plan else ""
    prompt = (
        f"Story idea: {idea}{plan_note}\nOriginal segment:\n{original_text}\n\n"
        f"Rewrite per instruction: {instruction or 'improve pacing and imagery, keep calm sleep-story tone'}. "
        "Keep same rough length and the same narrative purpose. Output only the rewritten segment text, no preamble."
    )
    try:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=PROSE_TEMP,
            reasoning_effort=PROSE_REASONING,
            include_reasoning=False,
            max_completion_tokens=max_output_tokens
        )
        return {"success": True, "text": resp.choices[0].message.content.strip(), "error": None}
    except Exception as e:
        return {"success": False, "text": original_text, "error": str(e)}


def assemble_story(segments: list, title: str = "") -> dict:
    """Join non-skipped segments. OUT: story_text, word_count, title"""
    parts = [s["text"] for s in segments if s["status"] != "skipped"]
    story_text = "\n\n".join(parts)
    return {"story_text": story_text, "word_count": len(story_text.split()), "title": title}


# ---------- LangGraph node wrappers (state dict in -> dict out) ----------

def story_gen_node(state: dict) -> dict:
    return generate_story(
        idea=state.get("idea", ""),
        wpm=state.get("wpm", DEFAULT_WPM),
        video_length_min=state.get("video_length_min", DEFAULT_VIDEO_LENGTH_MIN),
        target_words=state.get("target_words"),
        num_segments=state.get("num_segments"),
        model=state.get("model", MODEL),
        max_output_tokens=state.get("max_output_tokens", MAX_OUTPUT_TOKENS),
        words_per_segment_target=state.get("words_per_segment_target", WORDS_PER_SEGMENT_TARGET),
        batch_size=state.get("batch_size", SEGMENTS_PER_BATCH)
    )


def review_segments_node(state: dict) -> dict:
    """
    Pauses graph, surfaces segments to human.
    Resume payload: {segment_id(str): {"action": "keep"|"edit"|"skip"|"regenerate",
                                        "text": str, "instruction": str}}
    """
    segments = state["segments"]
    idea = state.get("idea", "")
    model = state.get("model", MODEL)
    segment_plans = state.get("segment_plans", {})

    decisions = interrupt({
        "task": "review_story_segments",
        "segments": [{"id": s["id"], "text": s["text"]} for s in segments]
    })

    updated = []
    for s in segments:
        d = decisions.get(str(s["id"])) or decisions.get(s["id"]) or {"action": "keep"}
        action = d.get("action", "keep")

        if action == "skip":
            s["status"] = "skipped"
        elif action == "edit":
            s["text"] = d.get("text", s["text"])
            s["status"] = "edited"
        elif action == "regenerate":
            plan = segment_plans.get(s["id"]) or segment_plans.get(str(s["id"]))
            r = regenerate_segment(s["original_text"], d.get("instruction", ""), idea, plan, model)
            s["text"] = r["text"]
            s["status"] = "regenerated"
        else:
            s["status"] = "kept"
        updated.append(s)

    return {"segments": updated, "review_done": True}


def assemble_story_node(state: dict) -> dict:
    return assemble_story(state["segments"], state.get("title", ""))


# ---------- demo graph wiring ----------

if __name__ == "__main__":
    from typing import TypedDict, List, Dict
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    class StoryState(TypedDict, total=False):
        idea: str
        wpm: int
        video_length_min: float
        target_words: int
        num_segments: int
        model: str
        max_output_tokens: int
        words_per_segment_target: int
        batch_size: int
        title: str
        story_arc: str
        characters: List[str]
        locations: List[str]
        segments: List[dict]
        segment_plans: Dict[int, dict]
        review_done: bool
        story_text: str
        word_count: int

    g = StateGraph(StoryState)
    g.add_node("generate", story_gen_node)
    g.add_node("review", review_segments_node)
    g.add_node("assemble", assemble_story_node)
    g.add_edge(START, "generate")
    g.add_edge("generate", "review")
    g.add_edge("review", "assemble")
    g.add_edge("assemble", END)

    graph = g.compile(checkpointer=InMemorySaver())

    # user input — topic + target params, no hardcoded idea
    idea = input("Story topic/idea: ").strip()
    wpm_raw = input(f"WPM [{DEFAULT_WPM}]: ").strip()
    len_raw = input(f"Video length, minutes [{DEFAULT_VIDEO_LENGTH_MIN}]: ").strip()
    wpm = int(wpm_raw) if wpm_raw else DEFAULT_WPM
    video_length_min = float(len_raw) if len_raw else DEFAULT_VIDEO_LENGTH_MIN

    cfg = {"configurable": {"thread_id": "story-1"}}
    result = graph.invoke({"idea": idea, "wpm": wpm, "video_length_min": video_length_min}, cfg)

    if result.get("success") is False:
        print(f"generation failed: {result.get('error')}")
        print(f"got {len(result.get('segments', []))} segments before failing")
        raise SystemExit(1)

    print(result["__interrupt__"][0].value)

    decisions = {"1": {"action": "keep"}, "2": {"action": "skip"}}
    final = graph.invoke(Command(resume=decisions), cfg)
    print(final["story_text"])