"""
ClinicSearch — Embedding & indexing pipeline.

Reads chunks from data/chunks/chunks.jsonl, embeds each via EmbeddingGemma
(through Ollama), and stores the results in a persistent ChromaDB collection.

Key properties:
  - Idempotent: chunks with ids already in the DB are skipped.
  - Resumable: kill it and restart, it picks up where it left off.
  - Batched: BATCH_SIZE controls memory and request size.

Usage:
    python -m src.embed
    # or
    python scripts/build_index.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterator

import chromadb
import ollama
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Embedding batch size. EmbeddingGemma is fast on M2; 32 is a comfortable balance.
BATCH_SIZE = 32

# Retry behavior for Ollama hiccups
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


def load_chunks(path: Path) -> list[dict]:
    """Load all chunks from the JSONL file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {path}. Run `python -m src.ingest` first."
        )
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def batched(items: list, n: int) -> Iterator[list]:
    """Yield successive n-sized batches from items."""
    for i in range(0, len(items), n):
        yield items[i : i + n]


def embed_batch(client: ollama.Client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with retry logic. Returns one vector per input text."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            result = client.embed(model=config.EMBED_MODEL, input=texts)
            vectors = result.get("embeddings", [])
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Expected {len(texts)} vectors, got {len(vectors)}"
                )
            return vectors
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_SECONDS * (2**attempt)
                print(
                    f"\n  {YELLOW}Embed batch failed (attempt {attempt + 1}): {exc}; "
                    f"retrying in {wait:.1f}s{RESET}"
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Embed batch failed after retries: {last_exc}")


def run() -> int:
    print(f"\n{BLUE}ClinicSearch — Embedding & indexing{RESET}")
    print(f"  Config: {config.summary()}")

    # 1. Load all chunks
    chunks_path = config.CHUNKS_DIR / "chunks.jsonl"
    print(f"\n  Loading chunks from {chunks_path}...")
    try:
        all_chunks = load_chunks(chunks_path)
    except FileNotFoundError as exc:
        print(f"  {RED}✗ {exc}{RESET}")
        return 1
    print(f"  {GREEN}✓{RESET} Loaded {len(all_chunks):,} chunks")

    if not all_chunks:
        print(f"  {RED}No chunks to embed. Run `python -m src.ingest` first.{RESET}")
        return 1

    # 2. Connect to ChromaDB
    print(f"\n  Opening ChromaDB at {config.CHROMA_PERSIST_DIR}...")
    chroma = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    collection = chroma.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    existing_count = collection.count()
    print(
        f"  {GREEN}✓{RESET} Collection `{config.CHROMA_COLLECTION_NAME}` "
        f"({existing_count:,} existing chunks)"
    )

    # 3. Filter to only chunks not yet indexed (idempotent resume)
    if existing_count > 0:
        print(f"\n  Checking which chunks are already indexed...")
        # Fetch existing ids in batches (ChromaDB has a limit per get())
        existing_ids: set[str] = set()
        # We'll just request all ids — ChromaDB handles paging internally
        try:
            got = collection.get(include=[])  # only ids
            existing_ids = set(got.get("ids", []))
        except Exception as exc:
            print(f"  {YELLOW}Could not enumerate existing ids: {exc}{RESET}")
            print(f"  {YELLOW}Will re-add all chunks (duplicates will overwrite).{RESET}")

        chunks_to_embed = [c for c in all_chunks if c["id"] not in existing_ids]
        skipped = len(all_chunks) - len(chunks_to_embed)
        if skipped > 0:
            print(
                f"  {GREEN}✓{RESET} Skipping {skipped:,} already-indexed chunks"
            )
    else:
        chunks_to_embed = all_chunks

    if not chunks_to_embed:
        print(f"\n  {GREEN}✓ All chunks already indexed. Nothing to do.{RESET}")
        return 0

    # 4. Embed and add in batches
    ollama_client = ollama.Client(host=config.OLLAMA_HOST)

    print(
        f"\n  Embedding {len(chunks_to_embed):,} chunks in batches of {BATCH_SIZE}..."
    )

    pbar = tqdm(
        total=len(chunks_to_embed), unit="chunk", desc="Embedding", smoothing=0.1
    )
    errors = 0

    try:
        for batch in batched(chunks_to_embed, BATCH_SIZE):
            try:
                texts = [c["text"] for c in batch]
                vectors = embed_batch(ollama_client, texts)
                collection.add(
                    ids=[c["id"] for c in batch],
                    embeddings=vectors,
                    documents=texts,
                    metadatas=[
                        {"source": c["source"], "page": c["page"]} for c in batch
                    ],
                )
                pbar.update(len(batch))
            except Exception as exc:
                errors += len(batch)
                pbar.write(f"  {RED}Batch error: {exc}{RESET}")
                pbar.update(len(batch))
    finally:
        pbar.close()

    # 5. Report
    final_count = collection.count()
    print(f"\n{BLUE}{'─' * 60}{RESET}")
    print(f"{BLUE}Indexing summary{RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")
    print(f"  Final collection size: {final_count:,} chunks")
    print(f"  Newly added:           {final_count - existing_count:,}")
    if errors:
        print(f"  {YELLOW}Errors: {errors:,} chunks could not be embedded{RESET}")
    print(f"  Index location:        {config.CHROMA_PERSIST_DIR}")

    if final_count == 0:
        print(f"\n  {RED}Index is empty. Something went wrong with embedding.{RESET}")
        return 1

    print(
        f"\n  {GREEN}Next:{RESET} test retrieval with `python scripts/ask_question.py`"
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())
