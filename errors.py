"""
errors.py — custom exception types for external service failures.

Every call out to a third-party service (Groq, Tavily, Supabase, Redis,
Ollama, ffmpeg/Kokoro, generic HTTP downloads) is wrapped in a try/except
that logs the failure and raises one of these, so callers can tell "an
external dependency failed" apart from "the code itself is broken", and so
every failure has exactly one place it was logged.
"""
from __future__ import annotations


class ExternalServiceError(Exception):
    """Base class for all external-service failures."""

    def __init__(self, service: str, message: str, original: Exception | None = None):
        self.service = service
        self.original = original
        super().__init__(f"[{service}] {message}")


class GroqAPIError(ExternalServiceError):
    def __init__(self, message: str, original: Exception | None = None):
        super().__init__("groq", message, original)


class TavilyAPIError(ExternalServiceError):
    def __init__(self, message: str, original: Exception | None = None):
        super().__init__("tavily", message, original)


class SupabaseError(ExternalServiceError):
    def __init__(self, message: str, original: Exception | None = None):
        super().__init__("supabase", message, original)


class RedisError(ExternalServiceError):
    def __init__(self, message: str, original: Exception | None = None):
        super().__init__("redis", message, original)


class OllamaError(ExternalServiceError):
    def __init__(self, message: str, original: Exception | None = None):
        super().__init__("ollama", message, original)


class MediaProcessingError(ExternalServiceError):
    """ffmpeg / Kokoro TTS / any local media-processing subprocess failure."""

    def __init__(self, message: str, original: Exception | None = None):
        super().__init__("media", message, original)


class HTTPFetchError(ExternalServiceError):
    """Generic httpx fetch failure (image/audio download etc.)."""

    def __init__(self, message: str, original: Exception | None = None):
        super().__init__("http", message, original)
