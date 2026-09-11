"""
story_gen.py — batched story generation + human-in-loop segment review.
Idea -> Groq LLM (batched calls) -> segmented story -> human keep/edit/skip/regenerate -> final text.
"""

import os
import json
from groq import Groq
from langgraph.types import interrupt

MODEL = "openai/gpt-oss-120b"       # Apache-2.0, Groq production tier, strict JSON support
MAX_OUTPUT_TOKENS = 6000            # per-call completion cap, stays well under 65536 limit + TPM-friendly
SEGMENTS_PER_CALL = 4                # segments requested per API call (batching = handles multi-hour stories)

_SEGMENT_ITEM = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "text": {"type": "string"}
    },
    "required": ["id", "text"],
    "additionalProperties": False
}

_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "segments": {"type": "array", "items": _SEGMENT_ITEM}
    },
    "required": ["title", "segments"],
    "additionalProperties": False
}

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _sys_prompt(batch_n: int, words_per_segment: int, is_first: bool, context_tail: str) -> str:
    base = (
        "Write sleep stories for a relaxation YouTube channel. Priority: comfortable, not "
        "beautiful. Plain, warm, unpolished-sounding sentences — not literature, not purple "
        "prose, no metaphor stacking, no thesaurus words. Write like a real person talking "
        "softly to someone falling asleep next to them at 1am. Short simple sentences. "
        "Everyday words. Some repetition is fine and sounds more human, not less. Intimate, "
        "close, low-key — like the narrator is right there in the room, half-whispering. "
        "No violence, no jump-scares, no sudden events, no cliffhangers. Slow pacing."
    )
    if is_first:
        task = (
            f" Write the OPENING {batch_n} segments of the story, ~{words_per_segment} words each. "
            "Set the scene, unhurried."
        )
    else:
        task = (
            f" CONTINUE the same story from where it left off. Story so far ends with:\n"
            f"\"...{context_tail}\"\n"
            f"Write the NEXT {batch_n} segments, ~{words_per_segment} words each. Keep same tone, "
            "same pacing, do not restart or recap."
        )
    return base + task + " Follow JSON schema exactly."


def generate_story(idea: str, target_words: int = 3000, num_segments: int = None,
                    model: str = MODEL, max_output_tokens: int = MAX_OUTPUT_TOKENS,
                    segments_per_call: int = SEGMENTS_PER_CALL) -> dict:
    """
    idea -> segmented sleep story, generated in batches (handles multi-hour targets, avoids
    per-call token limit errors).
    OUT: success, idea, title, segments:[{id,text,original_text,status}], model_used, error
    """
    idea = (idea or "").strip()
    if not idea:
        return {"success": False, "error": "idea empty", "segments": []}

    num_segments = num_segments or max(4, target_words // 400)
    words_per_segment = max(100, target_words // num_segments)

    client = _get_client()
    all_segments = []
    title = ""
    next_id = 1
    context_tail = ""

    while len(all_segments) < num_segments:
        batch_n = min(segments_per_call, num_segments - len(all_segments))
        sys_prompt = _sys_prompt(batch_n, words_per_segment, next_id == 1, context_tail)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Story idea: {idea}"}
                ],
                temperature=0.9,
                reasoning_effort="low",
                include_reasoning=False,
                max_completion_tokens=max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "sleep_story_batch", "strict": True, "schema": _BATCH_SCHEMA}
                }
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception as e:
            return {"success": False, "error": str(e), "segments": all_segments}

        if next_id == 1:
            title = data.get("title", "")

        for s in data["segments"]:
            all_segments.append({
                "id": next_id, "text": s["text"], "original_text": s["text"], "status": "pending"
            })
            next_id += 1

        context_tail = all_segments[-1]["text"][-400:]  # tail for next batch's continuity

    return {
        "success": True,
        "idea": idea,
        "title": title,
        "segments": all_segments[:num_segments],
        "model_used": model,
        "error": None
    }


def regenerate_segment(original_text: str, instruction: str, idea: str,
                        model: str = MODEL, max_output_tokens: int = 2000) -> dict:
    """Rewrite one segment per human instruction. OUT: success, text, error"""
    prompt = (
        f"Story idea: {idea}\nOriginal segment:\n{original_text}\n\n"
        f"Rewrite per instruction: {instruction or 'improve pacing and imagery, keep calm sleep-story tone'}. "
        "Keep same rough length. Output only the rewritten segment text, no preamble."
    )
    try:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            reasoning_effort="low",
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
        target_words=state.get("target_words", 3000),
        num_segments=state.get("num_segments"),
        model=state.get("model", MODEL),
        max_output_tokens=state.get("max_output_tokens", MAX_OUTPUT_TOKENS),
        segments_per_call=state.get("segments_per_call", SEGMENTS_PER_CALL)
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
            r = regenerate_segment(s["original_text"], d.get("instruction", ""), idea, model)
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
    from typing import TypedDict, List
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    class StoryState(TypedDict, total=False):
        idea: str
        target_words: int
        num_segments: int
        model: str
        max_output_tokens: int
        segments_per_call: int
        title: str
        segments: List[dict]
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

    cfg = {"configurable": {"thread_id": "story-1"}}
    result = graph.invoke({"idea": "a lighthouse keeper counting stars", "target_words": 2000}, cfg)
    print(result["__interrupt__"][0].value)

    decisions = {"1": {"action": "keep"}, "2": {"action": "skip"}}
    final = graph.invoke(Command(resume=decisions), cfg)
    print(final["story_text"])