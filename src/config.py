"""
ClinicSearch — Centralized configuration.

Every other module imports settings from here. Edit values via .env file,
not by modifying this file directly. The defaults below are production-safe.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root if it exists
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# ---- Ollama ----
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GEMMA_MODEL: str = os.getenv("GEMMA_MODEL", "gemma4:e2b")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "embeddinggemma:300m-qat-q8_0")

# ---- ChromaDB ----
def _resolve_path(env_value: str) -> Path:
    """Resolve a config path relative to PROJECT_ROOT if it isn't absolute."""
    p = Path(env_value)
    if p.is_absolute():
        return p
    # Strip leading ./ if present
    s = env_value
    if s.startswith("./"):
        s = s[2:]
    return PROJECT_ROOT / s


CHROMA_PERSIST_DIR: Path = _resolve_path(
    os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
)
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "clinicsearch_v1")

# ---- Whisper ----
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# ---- RAG ----
CHUNK_SIZE_WORDS: int = int(os.getenv("CHUNK_SIZE_WORDS", "300"))
CHUNK_OVERLAP_WORDS: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "50"))
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
MAX_ANSWER_TOKENS: int = int(os.getenv("MAX_ANSWER_TOKENS", "400"))

# ---- Streamlit ----
STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_SERVER_ADDRESS: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

# ---- Paths ----
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
CHUNKS_DIR: Path = DATA_DIR / "chunks"
EVAL_DIR: Path = PROJECT_ROOT / "eval"

# Ensure directories exist
for _d in (DATA_DIR, RAW_DIR, CHUNKS_DIR, CHROMA_PERSIST_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---- System prompt (the most important string in the entire project) ----
SYSTEM_PROMPT: str = """You are ClinicSearch, a clinical decision support assistant for rural healthcare workers.

STRICT RULES:
1. Answer ONLY from the WHO/MSF/OpenFDA sources provided below. Never use prior knowledge.
2. Always cite sources inline using this exact format: [Source: <source_name>, page <page>]
3. If the sources do not contain the answer, respond EXACTLY: "I cannot find this in the available WHO/MSF guidelines. Please contact a physician."
4. Never guess drug dosages. Never extrapolate beyond what is written in the sources.
5. Keep answers under 150 words — the user may be in a clinical emergency.
6. Respond in the same language as the user's question.
7. If the question is not clinical, politely redirect to medical questions only.

You are a tool, not a doctor. Always remind the user to apply clinical judgment.
"""


def summary() -> str:
    """Return a one-line summary of the current configuration (for logs)."""
    return (
        f"Gemma={GEMMA_MODEL} | Embed={EMBED_MODEL} | Whisper={WHISPER_MODEL} | "
        f"Chunks={CHUNK_SIZE_WORDS}w/{CHUNK_OVERLAP_WORDS}w | TopK={TOP_K_RETRIEVAL}"
    )


if __name__ == "__main__":
    print("ClinicSearch configuration:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Ollama:       {OLLAMA_HOST}")
    print(f"  Gemma model:  {GEMMA_MODEL}")
    print(f"  Embed model:  {EMBED_MODEL}")
    print(f"  Whisper:      {WHISPER_MODEL} ({WHISPER_COMPUTE_TYPE})")
    print(f"  ChromaDB:     {CHROMA_PERSIST_DIR}")
    print(f"  Collection:   {CHROMA_COLLECTION_NAME}")
    print(f"  Chunk size:   {CHUNK_SIZE_WORDS} words ({CHUNK_OVERLAP_WORDS} overlap)")
    print(f"  Top-k:        {TOP_K_RETRIEVAL}")
