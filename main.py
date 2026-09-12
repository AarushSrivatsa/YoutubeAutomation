"""
main.py — FastAPI application entry point.

Start the API:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Start the worker (separate terminal/container):
    python -m worker.tasks
"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config import get_settings

settings = get_settings()

# Set LangSmith env vars before any LangGraph imports
if settings.langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

os.environ["GROQ_API_KEY"] = settings.groq_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure tmp dir exists
    os.makedirs(settings.tmp_dir, exist_ok=True)
    yield
    # Cleanup on shutdown (optional)


app = FastAPI(
    title="Narrative Video Generator",
    description="Prompt + image → story → voiceover → stitched video. Powered by LangGraph.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}
