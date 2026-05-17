"""
ClinicSearch — Ingestion pipeline.

Reads raw documents from data/raw/ and produces chunks ready for embedding.
Each chunk has:
  - id:        stable hash, for idempotent indexing
  - text:      the chunk text
  - source:    human-readable source name (shown in citations)
  - page:      page number (for PDFs) or record id (for OpenFDA)
  - char_len:  for debugging chunk-size distribution

Chunks are written to data/chunks/chunks.jsonl (one JSON object per line).

Usage:
    python -m src.ingest
    # or
    python scripts/build_chunks.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from tqdm import tqdm

# Allow running this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


@dataclass
class Chunk:
    """A single retrievable unit of text with provenance."""

    id: str
    text: str
    source: str
    page: int  # 1-based for PDFs; -1 for non-paginated sources
    char_len: int

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def stable_id(source: str, page: int, text: str) -> str:
    """Deterministic chunk id so re-ingesting doesn't duplicate work downstream."""
    h = hashlib.sha1(f"{source}|{page}|{text[:200]}".encode("utf-8")).hexdigest()
    return f"{source.replace(' ', '_').lower()}_{page}_{h[:10]}"


def clean_text(s: str) -> str:
    """Normalize whitespace and strip obvious junk."""
    # Collapse all whitespace runs to single spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_garbage(text: str) -> bool:
    """Heuristic filter: skip chunks that are mostly numbers, symbols, or too short.

    Tuned for clinical documents which legitimately contain dense numeric
    content (dosages, mg amounts, page references) mixed with drug names.
    """
    if len(text) < 50:
        return True
    # Strip everything except letters; if very little remains, it's a pure table or figure
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) < len(text) * 0.25:
        return True
    return False


def chunk_text_by_words(
    text: str, words_per_chunk: int, overlap_words: int
) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text] if words else []

    step = max(1, words_per_chunk - overlap_words)
    chunks: list[str] = []
    for i in range(0, len(words), step):
        window = words[i : i + words_per_chunk]
        if not window:
            break
        chunks.append(" ".join(window))
        if i + words_per_chunk >= len(words):
            break
    return chunks


def ingest_pdf(path: Path, source_name: str) -> Iterator[Chunk]:
    """Yield chunks from a single PDF, page by page."""
    doc = fitz.open(path)
    try:
        for page_idx, page in enumerate(
            tqdm(doc, desc=f"PDF: {source_name}", leave=False), start=1
        ):
            raw = page.get_text()
            cleaned = clean_text(raw)
            if not cleaned:
                continue
            for chunk_text in chunk_text_by_words(
                cleaned,
                words_per_chunk=config.CHUNK_SIZE_WORDS,
                overlap_words=config.CHUNK_OVERLAP_WORDS,
            ):
                if is_garbage(chunk_text):
                    continue
                yield Chunk(
                    id=stable_id(source_name, page_idx, chunk_text),
                    text=chunk_text,
                    source=source_name,
                    page=page_idx,
                    char_len=len(chunk_text),
                )
    finally:
        doc.close()


def ingest_openfda(path: Path, source_name: str = "OpenFDA") -> Iterator[Chunk]:
    """Yield chunks from the OpenFDA JSON dump, preserving structured fields."""
    data = json.loads(path.read_text())
    results = data.get("results", [])

    for record in tqdm(results, desc="OpenFDA", leave=False):
        # Concatenate the most useful clinical fields
        parts: list[str] = []
        for field in (
            "drug_interactions",
            "warnings",
            "contraindications",
            "dosage_and_administration",
            "indications_and_usage",
        ):
            v = record.get(field)
            if isinstance(v, list):
                parts.extend(str(x) for x in v if x)
            elif isinstance(v, str):
                parts.append(v)

        if not parts:
            continue

        combined = clean_text(" ".join(parts))
        if is_garbage(combined):
            continue

        # Use the openFDA record id as the "page" (sha1-based for determinism)
        record_id_str = str(record.get("id", record.get("set_id", "unknown")))
        record_page = int(
            hashlib.sha1(record_id_str.encode()).hexdigest()[:6], 16
        )

        # Extract brand/generic name for the source citation if available
        brand = ""
        openfda_block = record.get("openfda", {})
        if isinstance(openfda_block, dict):
            brand_list = openfda_block.get("brand_name") or openfda_block.get(
                "generic_name"
            ) or []
            if isinstance(brand_list, list) and brand_list:
                brand = brand_list[0]

        display_source = f"OpenFDA — {brand}" if brand else "OpenFDA"

        # Long OpenFDA records get chunked too
        for chunk_text in chunk_text_by_words(
            combined,
            words_per_chunk=config.CHUNK_SIZE_WORDS,
            overlap_words=config.CHUNK_OVERLAP_WORDS,
        ):
            if is_garbage(chunk_text):
                continue
            yield Chunk(
                id=stable_id(display_source, record_page, chunk_text),
                text=chunk_text,
                source=display_source,
                page=-1,
                char_len=len(chunk_text),
            )


# Map of recognized filenames → (ingester function, human-readable source name)
DEFAULT_SOURCES: list[tuple[str, str, str]] = [
    # (filename in data/raw/, kind: pdf|fda, source name shown to user)
    ("who_eml.pdf", "pdf", "WHO Essential Medicines List"),
    ("msf_guidelines.pdf", "pdf", "MSF Clinical Guidelines"),
    ("openfda_interactions.json", "fda", "OpenFDA"),
    ("who_malaria.pdf", "pdf", "WHO Malaria Guidelines"),
    ("who_tb.pdf", "pdf", "WHO Tuberculosis Guidelines"),
    ("who_antenatal.pdf", "pdf", "WHO Antenatal Care Guidelines"),
]


def run() -> int:
    print(f"\n{BLUE}ClinicSearch — Ingestion{RESET}")
    print(f"  Config: {config.summary()}")
    print(f"  Reading from: {config.RAW_DIR}")

    out_path = config.CHUNKS_DIR / "chunks.jsonl"
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    per_source_counts: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as out_f:
        for filename, kind, source_name in DEFAULT_SOURCES:
            input_path = config.RAW_DIR / filename
            if not input_path.exists():
                print(
                    f"  {YELLOW}⚠{RESET}  {filename} not found in {config.RAW_DIR}; skipping"
                )
                continue

            print(f"\n  {BLUE}▶{RESET} Ingesting {source_name} ({filename})...")
            count = 0
            try:
                if kind == "pdf":
                    iterator = ingest_pdf(input_path, source_name)
                elif kind == "fda":
                    iterator = ingest_openfda(input_path, source_name)
                else:
                    print(f"    {RED}✗ Unknown kind: {kind}{RESET}")
                    continue

                for chunk in iterator:
                    out_f.write(chunk.to_jsonl() + "\n")
                    count += 1
                    total_chunks += 1

            except Exception as exc:
                print(f"    {RED}✗ Error ingesting {filename}: {exc}{RESET}")
                continue

            per_source_counts[source_name] = count
            print(f"    {GREEN}✓{RESET} {count:,} chunks from {source_name}")

    # Summary
    print(f"\n{BLUE}{'─' * 60}{RESET}")
    print(f"{BLUE}Ingestion summary{RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")
    for source, count in per_source_counts.items():
        print(f"  {source:40s}  {count:>8,} chunks")
    print(f"  {'TOTAL':40s}  {total_chunks:>8,} chunks")
    print(f"\n  Output: {out_path}")

    if total_chunks == 0:
        print(
            f"\n  {RED}No chunks produced. Did you run scripts/download_data.py "
            f"and place the PDFs in {config.RAW_DIR}?{RESET}"
        )
        return 1

    print(
        f"\n  {GREEN}Next:{RESET} build the vector index with `python -m src.embed`"
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())
