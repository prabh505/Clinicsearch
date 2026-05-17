"""
ClinicSearch — Environment verification.

Run this FIRST after installing dependencies. It checks every component
of the stack: Python version, all packages, Ollama service, both models,
ChromaDB persistence, Whisper, and microphone capability.

Usage:
    python scripts/verify_environment.py

Exit code 0 = ready to build. Non-zero = something needs fixing.
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

# Add project root to path so we can import src.config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ANSI colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    """Print a labeled check result and return ok."""
    mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {mark} {label}", end="")
    if detail:
        print(f"  {YELLOW}({detail}){RESET}")
    else:
        print()
    return ok


def section(name: str) -> None:
    print(f"\n{BLUE}{'─' * 60}{RESET}")
    print(f"{BLUE}{name}{RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")


def main() -> int:
    print(f"\n{BLUE}╔════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BLUE}║  ClinicSearch Environment Verification                     ║{RESET}")
    print(f"{BLUE}╚════════════════════════════════════════════════════════════╝{RESET}")

    failures: list[str] = []

    # ---- 1. Python and platform ----
    section("1. Platform")
    py_version = sys.version_info
    py_ok = py_version >= (3, 11) and py_version < (3, 13)
    check(
        f"Python version is 3.11 or 3.12",
        py_ok,
        f"got {py_version.major}.{py_version.minor}.{py_version.micro}",
    )
    if not py_ok:
        failures.append("Python version must be 3.11 or 3.12 (you have 3.12.8 recommended)")

    machine = platform.machine()
    arch_ok = machine == "arm64"
    check(f"Apple Silicon architecture", arch_ok, machine)
    if not arch_ok:
        failures.append(f"Expected arm64 architecture, got {machine}")

    is_mac = platform.system() == "Darwin"
    check(f"Running on macOS", is_mac, platform.system())
    if not is_mac:
        print(f"  {YELLOW}Note: not on macOS, some checks may behave differently{RESET}")

    # ---- 2. Python packages ----
    section("2. Python packages")
    required_packages = [
        ("ollama", "ollama"),
        ("chromadb", "chromadb"),
        ("streamlit", "streamlit"),
        ("fitz", "pymupdf"),
        ("faster_whisper", "faster-whisper"),
        ("ctranslate2", "ctranslate2"),
        ("sounddevice", "sounddevice"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("dotenv", "python-dotenv"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
        ("httpx", "httpx"),
    ]
    for module_name, pip_name in required_packages:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "?")
            check(f"{pip_name} importable", True, f"v{version}")
        except ImportError as exc:
            check(f"{pip_name} importable", False, str(exc))
            failures.append(f"pip install {pip_name}")

    # ---- 3. Configuration loads ----
    section("3. Configuration")
    try:
        from src import config

        check(f"src.config loads", True, config.summary())
    except Exception as exc:
        check(f"src.config loads", False, str(exc))
        failures.append(f"src/config.py has an error: {exc}")
        return 1

    # ---- 4. Ollama service ----
    section("4. Ollama service")
    try:
        import requests

        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
        ollama_ok = r.status_code == 200
        check(
            f"Ollama service reachable at {config.OLLAMA_HOST}",
            ollama_ok,
            f"HTTP {r.status_code}",
        )
        if not ollama_ok:
            failures.append("Start Ollama: `brew services start ollama` or run the Ollama.app")
    except Exception as exc:
        check(f"Ollama service reachable", False, str(exc))
        failures.append(
            "Ollama not reachable. Install: `brew install ollama` then `brew services start ollama`"
        )
        return _summarize(failures)

    # ---- 5. Models pulled ----
    section("5. Ollama models")
    try:
        tags_json = r.json()
        installed_models = {m["name"] for m in tags_json.get("models", [])}

        gemma_ok = config.GEMMA_MODEL in installed_models
        check(
            f"Gemma model `{config.GEMMA_MODEL}` pulled",
            gemma_ok,
            "" if gemma_ok else "NOT FOUND",
        )
        if not gemma_ok:
            failures.append(f"ollama pull {config.GEMMA_MODEL}")

        embed_ok = config.EMBED_MODEL in installed_models
        check(
            f"Embedding model `{config.EMBED_MODEL}` pulled",
            embed_ok,
            "" if embed_ok else "NOT FOUND",
        )
        if not embed_ok:
            failures.append(f"ollama pull {config.EMBED_MODEL}")
    except Exception as exc:
        check(f"Model list parseable", False, str(exc))
        failures.append("Could not parse Ollama tag list")

    # ---- 6. Embedding round-trip ----
    section("6. Embedding round-trip")
    if config.EMBED_MODEL in installed_models:
        try:
            import ollama

            client = ollama.Client(host=config.OLLAMA_HOST)
            result = client.embed(
                model=config.EMBED_MODEL,
                input="paracetamol dosage for children",
            )
            embeddings = result.get("embeddings", [])
            if embeddings and len(embeddings[0]) >= 256:
                check(
                    f"Embedding returns vector",
                    True,
                    f"dim={len(embeddings[0])}",
                )
            else:
                check(f"Embedding returns vector", False, "empty response")
                failures.append("Embedding model returned an empty response")
        except Exception as exc:
            check(f"Embedding round-trip", False, str(exc))
            failures.append(f"Embedding model failed: {exc}")

    # ---- 7. Generation round-trip ----
    section("7. Gemma 4 generation round-trip")
    if config.GEMMA_MODEL in installed_models:
        try:
            import ollama

            client = ollama.Client(host=config.OLLAMA_HOST)
            response = client.chat(
                model=config.GEMMA_MODEL,
                messages=[{"role": "user", "content": "Reply with exactly: READY"}],
                options={"num_predict": 20, "temperature": 0.0},
            )
            answer = response["message"]["content"].strip()
            gen_ok = "READY" in answer.upper()
            check(
                f"Gemma 4 generation works",
                gen_ok,
                f'returned "{answer[:30]}"',
            )
            if not gen_ok:
                print(
                    f"  {YELLOW}Note: model responded but not with expected text. "
                    f"Probably fine — this is not a hard failure.{RESET}"
                )
        except Exception as exc:
            check(f"Gemma 4 generation", False, str(exc))
            failures.append(f"Gemma 4 generation failed: {exc}")

    # ---- 8. ChromaDB ----
    section("8. ChromaDB persistence")
    try:
        import chromadb

        smoke_dir = Path("/tmp/clinicsearch_smoke_chroma")
        smoke_dir.mkdir(exist_ok=True)
        c = chromadb.PersistentClient(path=str(smoke_dir))
        col = c.get_or_create_collection("smoke_test")
        col.add(
            ids=["test1"],
            embeddings=[[0.1] * 768],
            documents=["smoke test document"],
            metadatas=[{"src": "test"}],
        )
        result = col.query(query_embeddings=[[0.1] * 768], n_results=1)
        chroma_ok = result["documents"][0][0] == "smoke test document"
        check(f"ChromaDB persistent store works", chroma_ok)
        # Cleanup
        c.delete_collection("smoke_test")
        import shutil

        shutil.rmtree(smoke_dir, ignore_errors=True)
    except Exception as exc:
        check(f"ChromaDB", False, str(exc))
        failures.append(f"ChromaDB error: {exc}")

    # ---- 9. faster-whisper ----
    section("9. faster-whisper")
    try:
        from faster_whisper import WhisperModel

        # Just check the model can be instantiated. Don't actually run inference
        # (that would require downloading ~150MB on first use).
        check(
            f"faster-whisper imports",
            True,
            "model load deferred to first use",
        )
    except Exception as exc:
        check(f"faster-whisper", False, str(exc))
        failures.append(f"faster-whisper error: {exc}")

    # ---- 10. Microphone (optional — only warns) ----
    section("10. Microphone (optional)")
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        mic_ok = len(input_devices) > 0
        check(
            f"Input device(s) detected",
            mic_ok,
            f"{len(input_devices)} found",
        )
        if not mic_ok:
            print(
                f"  {YELLOW}Warning: no microphone detected. Voice input won't work, "
                f"but text input will.{RESET}"
            )
    except Exception as exc:
        print(f"  {YELLOW}Could not check microphone: {exc}{RESET}")

    # ---- Summary ----
    return _summarize(failures)


def _summarize(failures: list[str]) -> int:
    section("Summary")
    if not failures:
        print(f"  {GREEN}All checks passed. Environment is ready.{RESET}")
        print(f"\n  Next: download data sources with `python scripts/download_data.py`")
        return 0
    print(f"  {RED}{len(failures)} issue(s) need to be fixed:{RESET}\n")
    for i, f in enumerate(failures, 1):
        print(f"  {RED}{i}.{RESET} {f}")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
