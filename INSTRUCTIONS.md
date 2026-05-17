# ClinicSearch — Executor Instructions

**Audience:** Local Claude Opus 4.6 (or a human developer) executing the build on a MacBook M2.

This document walks through every phase of the build. Each phase has clear inputs, the commands to run, expected outputs, and verification criteria. **Do not skip phases. Verify each phase before moving to the next.**

---

## Pre-flight check

Before starting, confirm the host environment:

```bash
# 1. Check macOS
uname -s        # → Darwin
uname -m        # → arm64

# 2. Check disk space (need 25+ GB free)
df -g .

# 3. Check the project files unzipped correctly
ls clinicsearch/
# Should show: app.py, setup.sh, requirements.txt, src/, scripts/, eval/, apk_wrapper/, data/
```

---

## Phase 1 — Environment setup

**Goal:** Install all system and Python dependencies. Pull both Ollama models.

**Time:** 30–60 minutes (model downloads dominate).

### Steps

```bash
cd clinicsearch
bash setup.sh
```

The script is idempotent — safe to re-run if something fails partway through.

### Verification

After setup completes, the script automatically runs `python scripts/verify_environment.py`. **All 10 sections must pass.** If anything fails, fix it before continuing.

Common issues and fixes:

| Symptom | Fix |
|---|---|
| `Ollama service not reachable` | `brew services start ollama` |
| `Model gemma4:e2b NOT FOUND` | `ollama pull gemma4:e2b` |
| `python version must be 3.11 or 3.12` | `pyenv install 3.12.8 && pyenv local 3.12.8` |
| `faster-whisper not importable` | `source .venv/bin/activate && pip install -r requirements.txt` |
| Build wheel error for `hnswlib` | Confirm Python is 3.12 (not 3.13) |

### Sign-off criterion

`python scripts/verify_environment.py` exits with code 0 and prints "All checks passed."

---

## Phase 2 — Data acquisition

**Goal:** Get the raw corpus into `data/raw/`.

**Time:** 5–10 minutes (OpenFDA auto-download) plus ~5 minutes for manual PDF downloads.

### Steps

```bash
source .venv/bin/activate
python scripts/download_data.py
```

The script downloads OpenFDA automatically (~10,000 records) and prints instructions for the two PDFs to download manually:

1. **WHO Essential Medicines List** from <https://list.essentialmeds.org/>
   - Save as `data/raw/who_eml.pdf`

2. **MSF Clinical Guidelines** from <https://medicalguidelines.msf.org/en>
   - Save as `data/raw/msf_guidelines.pdf`

### Verification

```bash
ls -lh data/raw/
```

Expected output:

```
who_eml.pdf                3-8 MB
msf_guidelines.pdf         5-20 MB
openfda_interactions.json  5-15 MB
```

### Sign-off criterion

All three files are present in `data/raw/` with non-zero file sizes.

---

## Phase 3 — Chunking

**Goal:** Convert raw documents into discrete searchable chunks.

**Time:** 1–3 minutes.

### Steps

```bash
python -m src.ingest
```

### Expected output

The script prints per-source chunk counts. Expected ranges:

| Source | Chunk count |
|---|---|
| WHO Essential Medicines List | 1,000–4,000 |
| MSF Clinical Guidelines | 2,000–10,000 |
| OpenFDA | 30,000–50,000 |
| **TOTAL** | **~35,000–60,000** |

### Verification

```bash
# Count chunks
wc -l data/chunks/chunks.jsonl

# Inspect a few
head -1 data/chunks/chunks.jsonl | python -m json.tool

# Confirm all three sources are represented
python -c "
import json
sources = {}
with open('data/chunks/chunks.jsonl') as f:
    for line in f:
        s = json.loads(line)['source']
        sources[s] = sources.get(s, 0) + 1
for src, n in sorted(sources.items(), key=lambda x: -x[1]):
    print(f'{n:>7,}  {src}')
"
```

### Sign-off criterion

- `data/chunks/chunks.jsonl` exists and has >5,000 lines.
- At least 2 distinct sources are represented (WHO + OpenFDA at minimum).
- A sample chunk has `text`, `source`, `page`, and `id` fields.

---

## Phase 4 — Building the vector index

**Goal:** Embed every chunk and store in ChromaDB.

**Time:** 10–20 minutes for 50,000 chunks on M2.

### Steps

```bash
python -m src.embed
```

### Notes

- The script is idempotent — interrupting and restarting picks up where it left off.
- A progress bar shows real-time throughput (~50–100 chunks/sec).
- If you see warnings about specific batch failures, that's recoverable — the script retries with backoff.

### Verification

```bash
# Check the persistent DB exists
du -sh data/chroma_db/

# Count vectors in the collection
python -c "
import chromadb
from src import config
c = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
col = c.get_collection(config.CHROMA_COLLECTION_NAME)
print(f'{col.count():,} vectors in collection')
"
```

### Sign-off criterion

- `data/chroma_db/` directory exists and is >100 MB.
- Collection count matches (within a few percent) the chunk count from Phase 3.

---

## Phase 5 — End-to-end RAG smoke test

**Goal:** Confirm the full pipeline works.

**Time:** 1–2 minutes (~15 seconds per query).

### Steps

```bash
python scripts/ask_question.py "First-line treatment for uncomplicated malaria in children"
```

### Expected output

The CLI prints:
1. A retrieved answer of 100–200 words, with inline source citations.
2. A list of 5 sources, each with source name, page, and match score.
3. Timing breakdown: retrieval (~50ms), generation (~9000ms), total (~9–14s).

### Verification

Run two more test queries to confirm consistency:

```bash
# Multilingual test (Portuguese)
python scripts/ask_question.py "Qual é o tratamento de primeira linha para malária em crianças?"

# Out-of-scope test — should refuse
python scripts/ask_question.py "What's the best smartphone to buy?"
```

The Portuguese query should retrieve a relevant clinical answer. The smartphone query should return the refusal string.

### Sign-off criterion

- English malaria query returns an answer citing WHO sources.
- Portuguese malaria query returns a clinical answer.
- The smartphone query returns the refusal string.

---

## Phase 6 — Voice transcription test

**Goal:** Verify Whisper works on macOS.

**Time:** 2–5 minutes (first run downloads ~150MB model).

### Steps

```bash
python -m src.voice --record
```

- The script records 5 seconds from the default microphone.
- Speak a clinical question in any language.
- On first run, macOS prompts for microphone permission — approve it.

### Expected output

```
Detected language: en
Transcription: First-line treatment for malaria in children
```

### Sign-off criterion

The transcribed text reasonably matches what was spoken.

---

## Phase 7 — UI smoke test

**Goal:** Launch the Streamlit app and verify it loads in a browser.

**Time:** 1 minute.

### Steps

```bash
streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false
```

Open `http://localhost:8501` in a browser on the same Mac.

### Verification

- The page loads with the title "🩺 ClinicSearch".
- No Streamlit menu or footer visible (hidden by CSS).
- A microphone input button and a text input box are visible.
- Type a test question and click Ask — an answer appears within 15 seconds.
- The Source passages expander shows 5 retrieved chunks.

### Sign-off criterion

A test query through the UI returns a sensible cited answer within 15 seconds.

---

## Phase 8 — Mobile WiFi test

**Goal:** Verify the phone can reach the laptop's app.

**Time:** 5 minutes.

### Steps

1. Find the laptop's WiFi IP:
   ```bash
   ipconfig getifaddr en0
   ```
   Example output: `192.168.1.42`.

2. Ensure macOS firewall is not blocking incoming connections to Python:
   - System Settings → Network → Firewall → Options
   - If firewall is on, set Python to "Allow incoming connections"

3. On the Android phone (same WiFi), open Chrome and navigate to:
   ```
   http://192.168.1.42:8501
   ```
   (Replace with your actual laptop IP.)

### Verification

The Streamlit UI loads on the phone. Touch input works. Ask a test query — it returns within 15 seconds.

### Sign-off criterion

Phone successfully connects to the laptop's app over local WiFi and a query returns successfully.

---

## Phase 9 — Evaluation run

**Goal:** Generate quantitative metrics for the write-up.

**Time:** 8–12 minutes (48 queries × ~10 sec each).

### Steps

```bash
python scripts/run_evaluation.py
```

### Expected output

The summary prints:
- Overall precision@3 (expect 70–85%)
- Per-language breakdown
- Per-category breakdown
- Refusal accuracy (expect 75–100%)
- Latency distribution

### Verification

```bash
# Inspect the JSON
python -c "
import json
data = json.load(open('eval/eval_results.json'))
m = data['metrics']
print(f'Overall P@3: {m[\"overall_precision_at_3\"]:.1%}')
print(f'Refusal accuracy: {m[\"refusal_metrics\"][\"accuracy\"]:.1%}')
print(f'Mean latency: {m[\"latency\"][\"mean_ms\"]/1000:.1f}s')
"
```

### Sign-off criterion

`eval/eval_results.json` exists and shows overall precision@3 ≥ 60%. (Below 60% would suggest a problem with chunking or embedding quality.)

---

## Phase 10 — APK build

**Goal:** Build a sideloadable Android APK.

**Time:** 5–15 minutes (most of which is downloading the Android SDK on first run).

### Steps

1. Confirm Streamlit is running (Phase 7).

2. Edit the APK config:
   ```bash
   # Replace YOUR_LAPTOP_IP with your actual WiFi IP
   sed -i '' "s/YOUR_LAPTOP_IP/$(ipconfig getifaddr en0)/" apk_wrapper/webapk.conf

   # Verify the substitution
   cat apk_wrapper/webapk.conf | grep mainURL
   ```

3. Build:
   ```bash
   bash apk_wrapper/build_apk.sh
   ```

### Verification

```bash
ls -lh apk_wrapper/clinicsearch.apk
file apk_wrapper/clinicsearch.apk
```

Expected: APK file ~3–6 MB.

### Transfer to phone

Choose one method:
- **AirDrop / Files app** (if on iOS Mac with Files): send the APK to the phone.
- **Google Drive**: upload, then download on phone.
- **USB**: connect phone, drag-and-drop into a folder, then open the file manager on the phone.

On the phone:
1. Open Settings → Apps → Special access → Install unknown apps
2. Grant permission to the file source you used (Files, Drive, etc.)
3. Tap the APK to install
4. Open the ClinicSearch app

### Sign-off criterion

The APK installs successfully and opens to the ClinicSearch UI in fullscreen. A test query returns an answer.

---

## Phase 11 — Demo recording

**Goal:** Record the 2-minute submission video.

### Storyboard

```
0:00–0:20  PROBLEM
"Every year, preventable deaths happen because a rural health worker
couldn't access WHO protocols in time. No internet. No physician.
No reference in their language."

0:20–0:35  THE PERSON
"Meet Amina, a community health worker in rural Tanzania. A child
arrives with fever. She has a $100 phone and no connectivity."

0:35–1:35  LIVE DEMO (no cuts)
[Show: phone, ClinicSearch app, voice input in Swahili]
"Dawa ya kwanza ya malaria kwa watoto" → answer appears with WHO citation
[Tap source → show WHO passage]
[Repeat in Portuguese] "Qual é o tratamento de primeira linha
para malária em crianças?"

1:35–1:50  HOW IT WORKS
"Gemma 4 and EmbeddingGemma run entirely offline on a laptop.
The phone is a thin client over local WiFi. No data leaves the room."

1:50–2:00  SCALE
"1 billion people live more than an hour from a hospital.
ClinicSearch fits in their pocket."
```

### Recording

- Use QuickTime (Mac) or OBS for screen recording the laptop.
- Use Android phone's built-in screen recorder for the phone footage.
- Edit together with iMovie or DaVinci Resolve.
- Upload to YouTube as **Unlisted**, enable auto-captions, verify they are readable.

### Sign-off criterion

Video is 1:50–2:10 in length, has captions, and the link is shareable.

---

## Phase 12 — Final submission

**Goal:** Submit on Kaggle by May 18, 2026, 18:00 UTC (NOT midnight — leave buffer).

### Steps

1. Push code to a public GitHub repo.
2. Create a Kaggle submission with:
   - Project title and description (use the one-sentence pitch from the README)
   - GitHub repo URL (public, includes README, requirements.txt)
   - YouTube video link
   - Write-up describing problem, architecture, evaluation, limitations

### Sign-off criterion

Submission confirmation received from Kaggle.

---

## Troubleshooting

### Ollama out-of-memory

Symptom: `model requires more memory than available`.

Fix: ensure no other large applications are running. Close Chrome, Slack, etc. Set in shell:

```bash
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_MAX_LOADED_MODELS=1
```

Then restart Ollama: `brew services restart ollama`.

### Slow generation (>30 seconds per query)

- Check that no other Ollama models are loaded: `ollama ps`
- Stop and restart Ollama
- Confirm Metal acceleration is active in the Ollama logs

### APK installs but shows blank screen

- Streamlit isn't running on the laptop. Start it.
- The IP in `webapk.conf` doesn't match the current laptop IP. Rebuild with corrected IP.
- The phone and laptop are on different WiFi networks. Reconnect.

### Whisper transcription is poor for Hindi/Swahili

- The `tiny` model has known accuracy problems on these languages.
- In `.env`, set `WHISPER_MODEL=small` (488 MB, much better).

---

## Daily checklist (for the 8-day build calendar)

- **Day 1:** Phase 1 (setup) + Phase 2 (data acquisition).
- **Day 2:** Phase 3 (chunking) + start Phase 4 (embedding, run overnight if needed).
- **Day 3:** Phase 4 finish + Phase 5 (end-to-end smoke test).
- **Day 4:** Phase 6 (voice) + Phase 7 (UI).
- **Day 5:** Phase 8 (mobile WiFi test) + UI polish.
- **Day 6:** Phase 9 (evaluation) + write-up draft.
- **Day 7:** Phase 10 (APK build) + Phase 11 (video recording).
- **Day 8:** Phase 12 (submit by 18:00 UTC).

Don't wait until the last day to record the video. Phones, microphones, and screen recorders fail in surprising ways.
