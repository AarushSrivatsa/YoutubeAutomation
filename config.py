import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings:
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

    # ── Worker ────────────────────────────────────────────────────────────────
    tmp_dir: str = os.getenv("TMP_DIR", "/tmp/narrative_gen")
    max_jobs: int = int(os.getenv("MAX_JOBS", "2"))

    # ── Voiceover defaults ────────────────────────────────────────────────────
    default_voice: str = os.getenv("DEFAULT_VOICE", "af_heart")
    use_cuda: bool = os.getenv("USE_CUDA", "false").lower() == "true"

    @property
    def embeddings_enabled(self) -> bool:
        return bool(self.ollama_embed_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()