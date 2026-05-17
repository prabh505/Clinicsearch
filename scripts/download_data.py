"""
ClinicSearch — Data source downloader.

Fetches all three corpus sources:
  1. WHO Essential Medicines List (PDF) - 23rd list, 2023
  2. MSF Clinical Guidelines (PDF) - latest June 2025 edition
  3. OpenFDA drug-interaction sample (JSON, ~10,000 records)

All download URLs are direct PDF links from the official publishers,
verified working as of May 2026. The script tries multiple mirrors per
source. Each download is idempotent (skips if file exists).

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force   # re-download even if exists
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Realistic browser user-agent — some WHO/MSF endpoints reject empty UAs
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ---- Download targets ----
# Each target has a list of candidate URLs tried in order.
# A target is considered downloaded once any candidate succeeds.
WHO_EML_URLS = [
    # WHO IRIS direct bitstream (canonical, working as of May 2026)
    "https://iris.who.int/server/api/core/bitstreams/289a875c-cc89-4914-90ad-eb3c578ebaf6/content",
    # WHO IRIS legacy path
    "https://iris.who.int/bitstream/handle/10665/371090/WHO-MHP-HPS-EML-2023.02-eng.pdf?sequence=1",
    # Apps WHO mirror
    "https://apps.who.int/iris/bitstream/handle/10665/371090/WHO-MHP-HPS-EML-2023.02-eng.pdf",
    # DRA Pakistan mirror (works when WHO is slow)
    "https://www.dra.gov.pk/wp-content/uploads/2023/10/WHO-Essential-Medicines-List-02.10.2023.pdf",
]

MSF_GUIDELINES_URLS = [
    # MSF Clinical Guidelines - Diagnosis and treatment manual (June 2025)
    "https://medicalguidelines.msf.org/sites/default/files/2025-07/guideline-170-en.pdf",
    # Older 2022 edition (fallback)
    "https://medicalguidelines.msf.org/sites/default/files/pdf/guideline-170-en-2022-10-26.pdf",
    # Tuberculosis 2025 (different document, same source — last-resort fallback)
    "https://medicalguidelines.msf.org/sites/default/files/2025-06/Tuberculosis%202025_EN.pdf",
]


def _download_with_progress(url: str, dest: Path, timeout: float = 120.0) -> bool:
    """Download a single URL to dest with a progress bar. Returns True on success."""
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    try:
        with httpx.stream(
            "GET",
            url,
            headers=headers,
            follow_redirects=True,
            timeout=timeout,
        ) as r:
            if r.status_code != 200:
                print(f"    {YELLOW}HTTP {r.status_code}{RESET}")
                return False

            total = int(r.headers.get("content-length", 0))
            content_type = r.headers.get("content-type", "")

            # Sanity: PDF endpoints should not return HTML
            if "html" in content_type.lower() and dest.suffix == ".pdf":
                print(f"    {YELLOW}Got HTML instead of PDF{RESET}")
                return False

            tmp_path = dest.with_suffix(dest.suffix + ".part")
            with tmp_path.open("wb") as f:
                with tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    leave=False,
                ) as pbar:
                    for chunk in r.iter_bytes(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            actual_size = tmp_path.stat().st_size
            if actual_size < 100_000:  # PDFs should be >100 KB
                print(
                    f"    {YELLOW}Suspicious size ({actual_size} bytes); discarding{RESET}"
                )
                tmp_path.unlink()
                return False

            # Verify PDF magic bytes
            if dest.suffix == ".pdf":
                with tmp_path.open("rb") as f:
                    magic = f.read(5)
                if not magic.startswith(b"%PDF-"):
                    print(
                        f"    {YELLOW}Not a valid PDF (magic={magic!r}); discarding{RESET}"
                    )
                    tmp_path.unlink()
                    return False

            tmp_path.rename(dest)
            return True
    except httpx.TimeoutException:
        print(f"    {YELLOW}Timeout{RESET}")
        return False
    except Exception as exc:
        print(f"    {YELLOW}{type(exc).__name__}: {exc}{RESET}")
        return False


def download_pdf(
    name: str, urls: list[str], dest: Path, force: bool = False
) -> bool:
    """Try each URL in order until one succeeds. Returns True if dest exists."""
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  {GREEN}✓{RESET} {name} already exists at {dest} ({size_mb:.1f} MB)")
        return True

    print(f"  {BLUE}Downloading {name}...{RESET}")
    for i, url in enumerate(urls, 1):
        # Truncate long URLs for display
        display_url = url if len(url) <= 75 else url[:72] + "..."
        print(f"    Attempt {i}/{len(urls)}: {display_url}")
        if _download_with_progress(url, dest):
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(
                f"  {GREEN}✓{RESET} {name} downloaded successfully ({size_mb:.1f} MB)"
            )
            return True

    print(f"  {RED}✗{RESET} All {len(urls)} attempts failed for {name}")
    return False


def download_openfda(limit: int = 10000, force: bool = False) -> bool:
    """Download a sample of OpenFDA drug-label data with drug_interactions field."""
    out = config.RAW_DIR / "openfda_interactions.json"
    if out.exists() and not force:
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"  {GREEN}✓{RESET} OpenFDA data already exists at {out} ({size_mb:.1f} MB)")
        return True

    print(f"  {BLUE}Downloading OpenFDA drug-interaction records (limit={limit})...{RESET}")

    base_url = "https://api.fda.gov/drug/label.json"
    all_results: list[dict] = []
    batch_size = 100
    skip = 0

    with tqdm(total=limit, desc="OpenFDA", unit="rec") as pbar:
        while len(all_results) < limit:
            try:
                params = {
                    "search": "_exists_:drug_interactions",
                    "limit": batch_size,
                    "skip": skip,
                }
                r = httpx.get(
                    base_url,
                    params=params,
                    timeout=30.0,
                    headers={"User-Agent": USER_AGENT},
                )
                if r.status_code != 200:
                    print(
                        f"\n  {YELLOW}OpenFDA returned HTTP {r.status_code}; stopping.{RESET}"
                    )
                    break
                data = r.json()
                results = data.get("results", [])
                if not results:
                    break
                all_results.extend(results)
                pbar.update(len(results))
                skip += batch_size
                if skip >= 25000:  # openFDA caps skip at 25000
                    break
            except Exception as exc:
                print(f"\n  {YELLOW}OpenFDA request failed: {exc}{RESET}")
                break

    all_results = all_results[:limit]
    if not all_results:
        print(f"  {RED}✗ OpenFDA returned 0 records.{RESET}")
        return False

    out.write_text(json.dumps({"results": all_results}, indent=2))
    print(f"  {GREEN}✓{RESET} Saved {len(all_results):,} OpenFDA records to {out}")
    return True


def print_manual_instructions(missing: list[str]) -> None:
    """If automatic downloads failed, tell the user what to do."""
    if not missing:
        return

    raw = config.RAW_DIR
    print(f"\n{YELLOW}╔════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{YELLOW}║  Manual download needed                                    ║{RESET}")
    print(f"{YELLOW}╚════════════════════════════════════════════════════════════╝{RESET}")

    if "WHO EML" in missing:
        print(
            f"\n{YELLOW}WHO Essential Medicines List (could not auto-download){RESET}\n"
            f"  Step 1. Open in a browser:\n"
            f"          https://www.who.int/publications/i/item/WHO-MHP-HPS-EML-2023.02\n"
            f"  Step 2. Click 'Download' (English PDF, ~5 MB)\n"
            f"  Step 3. Save the file as: {raw}/who_eml.pdf"
        )

    if "MSF Guidelines" in missing:
        print(
            f"\n{YELLOW}MSF Clinical Guidelines (could not auto-download){RESET}\n"
            f"  Step 1. Open in a browser:\n"
            f"          https://medicalguidelines.msf.org/en/viewport/CG/english/clinical-guidelines-16686604.html\n"
            f"  Step 2. Click 'Download PDF' on the page\n"
            f"  Step 3. Save the file as: {raw}/msf_guidelines.pdf"
        )

    if "OpenFDA" in missing:
        print(
            f"\n{YELLOW}OpenFDA could not be reached.{RESET}\n"
            f"  This is unusual. Check your internet connection and re-run this script."
        )

    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ClinicSearch data sources")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist"
    )
    args = parser.parse_args()

    print(f"\n{BLUE}ClinicSearch — Data source downloader{RESET}\n")
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []

    # 1. WHO EML
    who_path = config.RAW_DIR / "who_eml.pdf"
    if not download_pdf("WHO EML", WHO_EML_URLS, who_path, force=args.force):
        missing.append("WHO EML")

    # 2. MSF Clinical Guidelines
    msf_path = config.RAW_DIR / "msf_guidelines.pdf"
    if not download_pdf(
        "MSF Clinical Guidelines", MSF_GUIDELINES_URLS, msf_path, force=args.force
    ):
        missing.append("MSF Guidelines")

    # 3. OpenFDA
    if not download_openfda(limit=10000, force=args.force):
        missing.append("OpenFDA")

    # Show manual fallback if needed
    print_manual_instructions(missing)

    # Summary
    print(f"\n{BLUE}Final state of {config.RAW_DIR}:{RESET}")
    files = sorted(
        f for f in config.RAW_DIR.glob("*") if not f.name.startswith(".")
    )
    if not files:
        print(f"  {RED}(empty){RESET}")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  • {f.name:35s}  {size_mb:6.2f} MB")

    expected = {"who_eml.pdf", "msf_guidelines.pdf", "openfda_interactions.json"}
    present = {f.name for f in files}
    have_all = expected.issubset(present)

    if have_all:
        print(f"\n  {GREEN}✓ All three data sources present. Ready to ingest.{RESET}")
        print(f"  Next: python -m src.ingest")
        return 0
    else:
        missing_files = expected - present
        print(
            f"\n  {YELLOW}⚠ Missing {len(missing_files)} of 3 sources: "
            f"{', '.join(sorted(missing_files))}{RESET}"
        )
        print(
            f"  Follow the manual download instructions above, then re-run this script."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
