-- Run this once against your Supabase project.
-- Supabase has pgvector pre-installed; the extension just needs enabling.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Jobs ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jobs (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status        TEXT NOT NULL DEFAULT 'pending',   -- pending | running | completed | failed
    prompt        TEXT NOT NULL,
    image_url     TEXT,
    audio_url     TEXT,
    video_url     TEXT,
    error_message TEXT,
    wpm           INTEGER NOT NULL DEFAULT 150,
    video_length_min FLOAT NOT NULL DEFAULT 10.0,
    -- LangGraph uses this as thread_id for checkpointing
    thread_id     TEXT UNIQUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Stories ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stories (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id        UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    title         TEXT,
    story_arc     TEXT,
    characters    JSONB NOT NULL DEFAULT '[]',
    locations     JSONB NOT NULL DEFAULT '[]',
    segment_plans JSONB NOT NULL DEFAULT '{}',
    word_count    INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (job_id)
);

-- ── Segments ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS segments (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id         UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    segment_index  INTEGER NOT NULL,   -- 1-based, matches story_gen segment id
    text           TEXT NOT NULL,
    original_text  TEXT,
    status         TEXT NOT NULL DEFAULT 'kept',  -- kept | edited | regenerated | skipped
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (job_id, segment_index)
);

-- ── Prompt embeddings (pgvector similarity search) ────────────────────────────
-- Populated via local Ollama embeddings (nomic-embed-text, 768-dim). Falls
-- back to plain text search if Ollama isn't reachable — see embeddings_enabled
-- in config.py.
CREATE TABLE IF NOT EXISTS prompt_embeddings (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id     UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    prompt     TEXT NOT NULL,
    embedding  vector(768),   -- ollama nomic-embed-text dimension
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (job_id)
);

CREATE INDEX IF NOT EXISTS prompt_embeddings_ivfflat_idx
    ON prompt_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ── updated_at auto-trigger ───────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS jobs_updated_at ON jobs;
CREATE TRIGGER jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS segments_updated_at ON segments;
CREATE TRIGGER segments_updated_at
    BEFORE UPDATE ON segments
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ── Similarity search RPC (called from supabase-py .rpc()) ───────────────────
CREATE OR REPLACE FUNCTION match_jobs(
    query_embedding vector(768),
    match_threshold float DEFAULT 0.7,
    match_count     int   DEFAULT 10
)
RETURNS TABLE (
    job_id     UUID,
    prompt     TEXT,
    similarity float
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        pe.job_id,
        pe.prompt,
        1 - (pe.embedding <=> query_embedding) AS similarity
    FROM prompt_embeddings pe
    WHERE 1 - (pe.embedding <=> query_embedding) > match_threshold
    ORDER BY pe.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
