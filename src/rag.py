"""
ClinicSearch — Core RAG pipeline.

The heart of the system. Given a clinical question (in any language),
this module:
  1. Embeds the query using EmbeddingGemma.
  2. Retrieves top-k chunks from ChromaDB by cosine similarity.
  3. Constructs a prompt with strict citation requirements.
  4. Calls Gemma 4 E2B via Ollama to generate a cited answer.
  5. Returns a structured response with sources and timing.

The system prompt is carefully engineered to:
  - Force citations on every clinical claim
  - Refuse to answer outside the corpus
  - Respond in the user's language
  - Stay under 150 words (clinical emergencies need fast answers)

Usage:
    from src.rag import ClinicRAG
    rag = ClinicRAG()
    response = rag.ask("First-line malaria treatment for children")
    print(response.answer)
    for s in response.sources:
        print(f"  - {s['source']} (p.{s['page']}, score={s['score']:.2f})")
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import ollama

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

# Exact string returned for out-of-scope queries.
# We check for this string in the UI to render the refusal differently.
REFUSAL_STRING = (
    "I cannot find this in the available WHO/MSF guidelines. "
    "Please contact a physician."
)


@dataclass
class RetrievedChunk:
    """A single chunk returned from the vector store."""

    text: str
    source: str
    page: int
    score: float  # 1 - cosine_distance, so higher = better

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "score": self.score,
        }


@dataclass
class RAGResponse:
    """The full structured result of a RAG query."""

    answer: str
    sources: list[dict]
    retrieval_ms: float
    generation_ms: float
    is_refusal: bool = False
    error: str | None = None

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "retrieval_ms": self.retrieval_ms,
            "generation_ms": self.generation_ms,
            "total_ms": self.total_ms,
            "is_refusal": self.is_refusal,
            "error": self.error,
        }


class ClinicRAG:
    """Stateful RAG pipeline. Reuses Ollama and ChromaDB clients across queries."""

    def __init__(
        self,
        ollama_host: str | None = None,
        embed_model: str | None = None,
        gen_model: str | None = None,
        chroma_dir: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.ollama_host = ollama_host or config.OLLAMA_HOST
        self.embed_model = embed_model or config.EMBED_MODEL
        self.gen_model = gen_model or config.GEMMA_MODEL
        chroma_dir = chroma_dir or config.CHROMA_PERSIST_DIR
        collection_name = collection_name or config.CHROMA_COLLECTION_NAME

        self.ollama = ollama.Client(host=self.ollama_host)

        self.chroma = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            self.collection = self.chroma.get_collection(name=collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{collection_name}' not found at "
                f"{chroma_dir}. Run `python -m src.embed` first."
            ) from exc

    # ---- Retrieval ----

    def _generate_hypothetical_answer(self, query: str) -> str:
        """Generate a brief hypothetical answer to use as the retrieval query.

        This is the HyDE (Hypothetical Document Embeddings) technique. We embed
        a model-generated answer rather than the user's question because the
        hypothetical answer is written in the same style as source documents,
        sitting closer in embedding space to the real answer than the question is.
        """
        try:
            response = self.ollama.chat(
                model=self.gen_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a medical assistant. Given a clinical question "
                            "in any language, write a brief one-sentence hypothetical "
                            "answer in ENGLISH in the style of a clinical guideline. "
                            "Be specific about drug names, conditions, and patient "
                            "populations. Do not hedge or refuse. Always respond in "
                            "English regardless of the input language. Maximum 40 words."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                think=False,
                options={
                    "temperature": 0.3,
                    "num_predict": 80,
                },
            )
            hypothetical = response["message"]["content"].strip()
            # Fall back to the original query if Gemma refused or returned empty
            if not hypothetical or len(hypothetical) < 20:
                return query
            # Combine: hypothetical answer + original query for best of both worlds
            return f"{hypothetical} {query}"
        except Exception:
            # If HyDE generation fails for any reason, gracefully fall back
            return query

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks, stratified across sources.

        Because OpenFDA has ~74K chunks vs WHO/MSF at a few thousand each,
        unbalanced search would return only OpenFDA results. This method
        queries each source group separately, then merges by score.
        """
        top_k = top_k or config.TOP_K_RETRIEVAL

        # HyDE: expand the query into a hypothetical answer for better retrieval
        expanded_query = self._generate_hypothetical_answer(query)

        # Embed the expanded query
        emb_result = self.ollama.embed(model=self.embed_model, input=expanded_query)
        query_vector = emb_result["embeddings"][0]

        # Per-source quotas: ensure WHO and MSF always get a seat at the table
        # Allocate roughly 40% WHO, 30% MSF, 30% OpenFDA of the top_k slots
        quotas = {
            "who": max(2, int(top_k * 0.4)),
            "msf": max(1, int(top_k * 0.3)),
            "fda": max(1, int(top_k * 0.3)),
        }

        all_results: list[RetrievedChunk] = []
        seen_ids: set[str] = set()

        # Query each source group with its quota
        for source_key, n in quotas.items():
            if source_key == "who":
                where_filter = {"source": {"$in": ["WHO Essential Medicines List", "WHO Malaria Guidelines", "WHO Tuberculosis Guidelines", "WHO Antenatal Care Guidelines"]}}
            elif source_key == "msf":
                where_filter = {"source": {"$eq": "MSF Clinical Guidelines"}}
            else:  # OpenFDA — query without filter (OpenFDA dominates corpus)
                where_filter = None

            try:
                query_kwargs = {
                    "query_embeddings": [query_vector],
                    "n_results": n,
                    "include": ["documents", "metadatas", "distances"],
                }
                if where_filter is not None:
                    query_kwargs["where"] = where_filter
                result = self.collection.query(**query_kwargs)
                docs = result["documents"][0]
                metas = result["metadatas"][0]
                dists = result["distances"][0]
                for doc, meta, dist in zip(docs, metas, dists):
                    chunk_id = f"{meta.get('source','')}_{meta.get('page','')}_{doc[:30]}"
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)
                    all_results.append(
                        RetrievedChunk(
                            text=doc,
                            source=meta.get("source", "Unknown"),
                            page=int(meta.get("page", -1)),
                            score=max(0.0, 1.0 - float(dist)),
                        )
                    )
            except Exception:
                # If a source has no matches or filter syntax fails, skip it
                continue

        # Sort by score descending and return top_k
        all_results.sort(key=lambda c: c.score, reverse=True)
        return all_results[:top_k]

    # ---- Generation ----

    def _build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Compose the user-side prompt with source blocks."""
        if not chunks:
            return (
                f"No sources found.\n\nQuestion: {question}\n\n"
                f"Per the rules, you must reply exactly with the refusal phrase."
            )

        source_blocks: list[str] = []
        for i, c in enumerate(chunks, start=1):
            page_str = f"page {c.page}" if c.page > 0 else "record"
            source_blocks.append(
                f"[Source {i}: {c.source}, {page_str}]\n{c.text}"
            )

        sources_section = "\n\n".join(source_blocks)
        return (
            f"SOURCES (cite these by their bracketed labels):\n\n"
            f"{sources_section}\n\n"
            f"QUESTION: {question}"
        )

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Call Gemma 4 with the retrieved context."""
        prompt = self._build_prompt(question, chunks)
        response = self.ollama.chat(
            model=self.gen_model,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.2,  # low for clinical accuracy
                "num_predict": config.MAX_ANSWER_TOKENS,
                "top_p": 0.9,
            },
            think=False,  # Gemma 4 E2B defaults to thinking mode; disable it
        )
        return response["message"]["content"].strip()

    # ---- End-to-end ----

    def ask(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Full pipeline: retrieve, generate, return structured response."""
        try:
            t0 = time.perf_counter()
            chunks = self.retrieve(question, top_k=top_k)
            t1 = time.perf_counter()
            answer = self.generate(question, chunks)
            t2 = time.perf_counter()

            return RAGResponse(
                answer=answer,
                sources=[c.to_dict() for c in chunks],
                retrieval_ms=(t1 - t0) * 1000,
                generation_ms=(t2 - t1) * 1000,
                is_refusal=REFUSAL_STRING.lower() in answer.lower(),
            )
        except Exception as exc:
            return RAGResponse(
                answer="An internal error occurred. Please try again or contact a physician.",
                sources=[],
                retrieval_ms=0.0,
                generation_ms=0.0,
                error=str(exc),
            )


# ---- Quick CLI for smoke testing ----

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick RAG smoke test")
    parser.add_argument(
        "question",
        nargs="?",
        default="First-line treatment for uncomplicated malaria in children",
        help="The clinical question to ask",
    )
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    print(f"\nQuestion: {args.question}\n")
    rag = ClinicRAG()
    resp = rag.ask(args.question, top_k=args.top_k)

    print("─" * 60)
    print("Answer:")
    print(resp.answer)
    print("─" * 60)
    print(f"Sources ({len(resp.sources)}):")
    for i, s in enumerate(resp.sources, 1):
        page_label = f"p.{s['page']}" if s["page"] > 0 else "record"
        print(f"  {i}. {s['source']} ({page_label})  score={s['score']:.2f}")
    print("─" * 60)
    print(
        f"Timing: retrieval={resp.retrieval_ms:.0f}ms  "
        f"generation={resp.generation_ms:.0f}ms  "
        f"total={resp.total_ms:.0f}ms"
    )
    if resp.is_refusal:
        print("[REFUSAL] Question was out of scope.")
    if resp.error:
        print(f"[ERROR] {resp.error}")
