"""
ClinicSearch — Command-line question interface.

Usage:
    # Single question
    python scripts/ask_question.py "First-line treatment for malaria in children"

    # Interactive REPL
    python scripts/ask_question.py

    # With custom top-k
    python scripts/ask_question.py "..." --top-k 8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag import ClinicRAG


GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
GRAY = "\033[90m"
RESET = "\033[0m"


def print_response(response, top_k_actual: int) -> None:
    print(f"\n{BLUE}{'─' * 60}{RESET}")
    if response.is_refusal:
        print(f"{YELLOW}REFUSAL (out of scope):{RESET}")
    else:
        print(f"{GREEN}Answer:{RESET}")
    print(response.answer)
    print(f"{BLUE}{'─' * 60}{RESET}")

    if response.sources:
        print(f"Sources ({len(response.sources)}):")
        for i, s in enumerate(response.sources, 1):
            page_label = f"p.{s['page']}" if s["page"] > 0 else "record"
            print(
                f"  {i}. {s['source']} ({page_label})  "
                f"score={s['score']:.2f}"
            )

    print(
        f"{GRAY}retrieval={response.retrieval_ms:.0f}ms  "
        f"generation={response.generation_ms:.0f}ms  "
        f"total={response.total_ms:.0f}ms{RESET}"
    )
    if response.error:
        print(f"\nError: {response.error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ClinicSearch CLI")
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="The question to ask (omit for interactive mode)",
    )
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    print(f"{BLUE}Initializing ClinicSearch...{RESET}")
    try:
        rag = ClinicRAG()
    except Exception as exc:
        print(f"Failed to initialize: {exc}")
        return 1

    if args.question:
        response = rag.ask(args.question, top_k=args.top_k)
        print_response(response, args.top_k or 5)
        return 0

    # Interactive REPL
    print(
        f"\n{GREEN}ClinicSearch CLI{RESET} — type a question (or 'quit' to exit)\n"
    )
    while True:
        try:
            question = input(f"{BLUE}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        response = rag.ask(question, top_k=args.top_k)
        print_response(response, args.top_k or 5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
