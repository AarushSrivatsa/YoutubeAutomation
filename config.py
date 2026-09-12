from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"
    core_llm_model: str = "llama-3.3-70b-versatile"

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str
    supabase_service_key: str
    supabase_db_url: str  # direct postgres connection string

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"

    # ── Tavily ────────────────────────────────────────────────────────────────
    tavily_api_key: str

    # ── Ollama embeddings (local, free — used for similarity search) ──────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # ── LangSmith (optional) ──────────────────────────────────────────────────
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "narrative-gen"

    # ── Supabase storage buckets ──────────────────────────────────────────────
    image_bucket: str = "images"
    audio_bucket: str = "audio"
    video_bucket: str = "videos"

    # ── Worker ────────────────────────────────────────────────────────────────
    tmp_dir: str = "/tmp/narrative_gen"
    max_jobs: int = 2

    # ── Voiceover defaults ────────────────────────────────────────────────────
    default_voice: str = "af_heart"
    use_cuda: bool = False

    @property
    def embeddings_enabled(self) -> bool:
        # Ollama is local — embeddings are on by default. Set to "" to disable.
        return bool(self.ollama_embed_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
