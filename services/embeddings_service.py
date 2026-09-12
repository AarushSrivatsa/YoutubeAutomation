"""
embeddings_service.py — local embeddings via Ollama.

Replaces OpenAI's text-embedding-3-small with a locally-hosted model
(default: nomic-embed-text, 768-dim). No API key, no per-call cost —
just needs `ollama pull nomic-embed-text` and the Ollama daemon running
at settings.ollama_base_url.
"""
from __future__ import annotations
import httpx

from config import get_settings

settings = get_settings()


def get_embedding(text: str) -> list[float]:
    """
    Calls Ollama's /api/embeddings endpoint and returns the embedding vector.
    Raises on failure — callers should catch and treat embeddings as best-effort.
    """
    resp = httpx.post(
        f"{settings.ollama_base_url}/api/embeddings",
        json={"model": settings.ollama_embed_model, "prompt": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]
