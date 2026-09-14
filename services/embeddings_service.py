"""
embeddings_service.py — local embeddings via Ollama.

Replaces OpenAI's text-embedding-3-small with a locally-hosted model
(default: nomic-embed-text, 768-dim). No API key, no per-call cost —
just needs `ollama pull nomic-embed-text` and the Ollama daemon running
at settings.ollama_base_url.
"""
from __future__ import annotations
import logging
import httpx

from config import get_settings
from errors import OllamaError

logger = logging.getLogger(__name__)
settings = get_settings()


def get_embedding(text: str) -> list[float]:
    """
    Calls Ollama's /api/embeddings endpoint and returns the embedding vector.
    Raises OllamaError on failure — callers should catch it and treat embeddings
    as best-effort (this is exactly what supabase_node._store_embedding does).
    """
    preview = (text or "")[:60].replace("\n", " ")
    logger.debug("Requesting embedding from Ollama for text: %r...", preview)
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/embeddings",
            json={"model": settings.ollama_embed_model, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        embedding = resp.json()["embedding"]
        logger.debug("Received embedding of length %d", len(embedding))
        return embedding
    except httpx.HTTPError as e:
        logger.exception("Ollama embeddings request failed (HTTP)")
        raise OllamaError(f"embeddings request failed: {e}", e) from e
    except (KeyError, ValueError) as e:
        logger.exception("Ollama embeddings response was malformed")
        raise OllamaError(f"malformed embeddings response: {e}", e) from e
    except Exception as e:
        logger.exception("Unexpected error getting embedding from Ollama")
        raise OllamaError(f"unexpected error: {e}", e) from e
