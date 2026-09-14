import logging
import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Settings:
    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    core_llm_model: str = os.getenv("CORE_LLM_MODEL", "llama-3.3-70b-versatile")

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")
    supabase_db_url: str = os.getenv("SUPABASE_DB_URL", "")  # direct postgres connection string

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # ── Tavily ────────────────────────────────────────────────────────────────
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # ── Ollama embeddings (local, free — used for similarity search) ──────────
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # ── LangSmith (optional) ──────────────────────────────────────────────────
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_tracing_v2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "narrative-gen")

    # ── Supabase storage buckets ──────────────────────────────────────────────
    image_bucket: str = os.getenv("IMAGE_BUCKET", "images")
    audio_bucket: str = os.getenv("AUDIO_BUCKET", "audio")
    video_bucket: str = os.getenv("VIDEO_BUCKET", "videos")

    # ── Background jobs ───────────────────────────────────────────────────────
    # Jobs run as FastAPI BackgroundTasks in this same process (see
    # background_jobs.py) — max_jobs caps how many run concurrently via an
    # in-process semaphore, since TTS + ffmpeg are CPU/RAM heavy.
    tmp_dir: str = os.getenv("TMP_DIR", "/tmp/narrative_gen")
    max_jobs: int = int(os.getenv("MAX_JOBS", "2"))

    # ── Voiceover defaults ────────────────────────────────────────────────────
    default_voice: str = os.getenv("DEFAULT_VOICE", "af_heart")
    use_cuda: bool = os.getenv("USE_CUDA", "false").lower() == "true"

    @property
    def embeddings_enabled(self) -> bool:
        return bool(self.ollama_embed_model)

    def validate(self) -> list[str]:
        """
        Returns human-readable warnings for missing/likely-misconfigured settings.
        Does not raise — the API process and the worker process need different
        subsets of these, so it's up to the entrypoint whether a warning here is
        actually fatal for it.
        """
        warnings: list[str] = []
        required = {
            "GROQ_API_KEY": self.groq_api_key,
            "SUPABASE_URL": self.supabase_url,
            "SUPABASE_SERVICE_KEY": self.supabase_service_key,
            "SUPABASE_DB_URL": self.supabase_db_url,
        }
        for name, value in required.items():
            if not value:
                warnings.append(f"{name} is not set")
        if not self.tavily_api_key:
            warnings.append(
                "TAVILY_API_KEY is not set — the routing node will still run, "
                "but any prompt that needs search will silently skip it"
            )
        return warnings


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for warning in settings.validate():
        logger.warning("Config warning: %s", warning)
    return settings
