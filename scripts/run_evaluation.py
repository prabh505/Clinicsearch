"""
ClinicSearch — Evaluation harness.

Runs all queries in eval/test_queries.json against the live system and
reports:
  - precision@1, precision@3, precision@5 (broken down by language)
  - refusal accuracy (did out-of-scope queries get refused?)
  - latency distribution
  - failure cases (for manual review)

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --top-k 5 --out eval/eval_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import ClinicRAG, REFUSAL_STRING


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
GRAY = "\033[90m"
RESET = "\033[0m"


def source_matches(source_name: str, expected_substring: str) -> bool:
    """Lenient match: case-insensitive substring."""
    return expected_substring.lower() in source_name.lower()


def evaluate(rag: ClinicRAG, queries: list[dict], top_k: int) -> dict:
    """Run all queries and return aggregated metrics + per-query results."""
    results: list[dict] = []
    latencies: list[float] = []

    for i, q in enumerate(queries, 1):
        print(
            f"{GRAY}[{i:2d}/{len(queries)}]{RESET} [{q['lang']}] {q['query'][:60]}..."
        )

        response = rag.ask(q["query"], top_k=top_k)
        latencies.append(response.total_ms)

        result = {
            "id": q["id"],
            "lang": q["lang"],
            "category": q["category"],
            "query": q["query"],
            "answer": response.answer,
            "is_refusal": response.is_refusal,
            "expect_refusal": q.get("expect_refusal", False),
            "retrieval_ms": response.retrieval_ms,
            "generation_ms": response.generation_ms,
            "sources": response.sources,
            "error": response.error,
        }

        # Compute precision@k for in-scope queries with expected_source_contains
        expected_source = q.get("expected_source_contains")
        if expected_source and not q.get("expect_refusal"):
            sources_returned = [s["source"] for s in response.sources]
            result["correct_in_top_1"] = bool(
                sources_returned and source_matches(sources_returned[0], expected_source)
            )
            result["correct_in_top_3"] = any(
                source_matches(s, expected_source) for s in sources_returned[:3]
            )
            result["correct_in_top_5"] = any(
                source_matches(s, expected_source) for s in sources_returned[:5]
            )

        # Refusal accuracy
        if q.get("expect_refusal"):
            result["refusal_correct"] = response.is_refusal

        results.append(result)

    # ---- Aggregate metrics ----
    metrics: dict = {
        "total_queries": len(queries),
        "by_language": {},
        "by_category": {},
        "refusal_metrics": {},
        "latency": {
            "mean_ms": statistics.mean(latencies) if latencies else 0,
            "median_ms": statistics.median(latencies) if latencies else 0,
            "p90_ms": (
                statistics.quantiles(latencies, n=10)[8] if len(latencies) > 10 else 0
            ),
            "max_ms": max(latencies) if latencies else 0,
        },
    }

    # Per-language precision
    lang_buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if "correct_in_top_3" in r:
            lang_buckets[r["lang"]].append(r)

    for lang, items in lang_buckets.items():
        n = len(items)
        p1 = sum(1 for r in items if r.get("correct_in_top_1")) / n if n else 0
        p3 = sum(1 for r in items if r.get("correct_in_top_3")) / n if n else 0
        p5 = sum(1 for r in items if r.get("correct_in_top_5")) / n if n else 0
        metrics["by_language"][lang] = {
            "count": n,
            "precision_at_1": p1,
            "precision_at_3": p3,
            "precision_at_5": p5,
        }

    # Per-category precision
    cat_buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if "correct_in_top_3" in r:
            cat_buckets[r["category"]].append(r)

    for cat, items in cat_buckets.items():
        n = len(items)
        p3 = sum(1 for r in items if r.get("correct_in_top_3")) / n if n else 0
        metrics["by_category"][cat] = {
            "count": n,
            "precision_at_3": p3,
        }

    # Refusal metrics
    refusal_items = [r for r in results if r.get("expect_refusal")]
    if refusal_items:
        correct_refusals = sum(1 for r in refusal_items if r.get("refusal_correct"))
        metrics["refusal_metrics"] = {
            "count": len(refusal_items),
            "correct": correct_refusals,
            "accuracy": correct_refusals / len(refusal_items),
        }

    # Overall precision@3 (only over queries that have an expected source)
    in_scope_items = [r for r in results if "correct_in_top_3" in r]
    if in_scope_items:
        overall_p3 = sum(
            1 for r in in_scope_items if r.get("correct_in_top_3")
        ) / len(in_scope_items)
        metrics["overall_precision_at_3"] = overall_p3

    return {"metrics": metrics, "results": results}


def print_summary(metrics: dict) -> None:
    print(f"\n{BLUE}{'═' * 60}{RESET}")
    print(f"{BLUE}Evaluation summary{RESET}")
    print(f"{BLUE}{'═' * 60}{RESET}")

    if "overall_precision_at_3" in metrics:
        p3 = metrics["overall_precision_at_3"]
        color = GREEN if p3 >= 0.7 else (YELLOW if p3 >= 0.5 else RED)
        print(f"\n  Overall precision@3: {color}{p3:.1%}{RESET}")

    if metrics.get("by_language"):
        print(f"\n  {BLUE}By language:{RESET}")
        print(f"  {'lang':<6} {'count':>5}  {'P@1':>6}  {'P@3':>6}  {'P@5':>6}")
        for lang, m in sorted(metrics["by_language"].items()):
            print(
                f"  {lang:<6} {m['count']:>5}  "
                f"{m['precision_at_1']:>6.1%}  "
                f"{m['precision_at_3']:>6.1%}  "
                f"{m['precision_at_5']:>6.1%}"
            )

    if metrics.get("by_category"):
        print(f"\n  {BLUE}By category:{RESET}")
        print(f"  {'category':<22} {'count':>5}  {'P@3':>6}")
        for cat, m in sorted(metrics["by_category"].items()):
            print(f"  {cat:<22} {m['count']:>5}  {m['precision_at_3']:>6.1%}")

    if metrics.get("refusal_metrics"):
        rm = metrics["refusal_metrics"]
        color = GREEN if rm["accuracy"] >= 0.75 else YELLOW
        print(f"\n  {BLUE}Refusal accuracy:{RESET}")
        print(
            f"  Out-of-scope queries: {rm['correct']}/{rm['count']} "
            f"correctly refused ({color}{rm['accuracy']:.1%}{RESET})"
        )

    lat = metrics.get("latency", {})
    if lat:
        print(f"\n  {BLUE}Latency:{RESET}")
        print(f"  Mean:   {lat['mean_ms'] / 1000:6.2f}s")
        print(f"  Median: {lat['median_ms'] / 1000:6.2f}s")
        if lat.get("p90_ms"):
            print(f"  p90:    {lat['p90_ms'] / 1000:6.2f}s")
        print(f"  Max:    {lat['max_ms'] / 1000:6.2f}s")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="ClinicSearch evaluation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--queries",
        default=str(PROJECT_ROOT / "eval" / "test_queries.json"),
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval" / "eval_results.json"),
    )
    args = parser.parse_args()

    # Load queries
    with open(args.queries) as f:
        data = json.load(f)
    queries = data["queries"]
    print(f"{BLUE}Loaded {len(queries)} evaluation queries{RESET}\n")

    # Initialize RAG
    print(f"{BLUE}Initializing RAG pipeline...{RESET}")
    rag = ClinicRAG()

    # Run evaluation
    start_time = time.time()
    output = evaluate(rag, queries, top_k=args.top_k)
    elapsed = time.time() - start_time

    print(f"\n{GREEN}Evaluation complete in {elapsed / 60:.1f} minutes{RESET}")

    # Write results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results written to {args.out}")

    # Print summary
    print_summary(output["metrics"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
