"""
ClinicSearch — Streamlit web UI.

Mobile-first clinical assistant interface. Hides Streamlit's default chrome
so the WebView APK wrapper looks like a native app.

Run with:
    streamlit run app.py \\
        --server.address=0.0.0.0 \\
        --server.port=8501 \\
        --server.headless=true \\
        --browser.gatherUsageStats=false

Then on the Android device, navigate to:
    http://<your-laptop-ip>:8501

Or build the APK wrapper (see apk_wrapper/build_apk.sh) which embeds
this URL as a fullscreen WebView app.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.rag import REFUSAL_STRING, ClinicRAG


# ---- Page config (must be first Streamlit call) ----
st.set_page_config(
    page_title="ClinicSearch",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ---- Hide Streamlit chrome for clean APK look ----
HIDE_CHROME_CSS = """
<style>
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden; height: 0;}
  header {visibility: hidden; height: 0;}
  .stDeployButton {display: none;}
  .stAppViewBlockContainer {padding-top: 1rem; padding-bottom: 2rem;}
  /* Larger touch targets for mobile */
  .stTextInput input, .stTextArea textarea {
      font-size: 16px;
  }
  .stButton > button {
      min-height: 44px;
      font-size: 16px;
  }
  /* Cleaner spacing */
  div[data-testid="stVerticalBlock"] > div {gap: 0.5rem;}
  /* Answer card */
  .answer-card {
      background: rgba(14, 165, 233, 0.15);
      border-left: 4px solid #0EA5E9;
      padding: 1rem;
      border-radius: 0.5rem;
      margin: 1rem 0;
      color: inherit;
  }
  .refusal-card {
      background: rgba(217, 119, 6, 0.18);
      border-left: 4px solid #D97706;
      padding: 1rem;
      border-radius: 0.5rem;
      margin: 1rem 0;
      color: inherit;
  }
  .refusal-card strong {
      color: #FCD34D;
  }
  .metric-row {
      font-size: 0.85rem;
      color: #6b7280;
      margin-top: 0.5rem;
  }
  .source-chip {
      display: inline-block;
      background: #e5e7eb;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 0.8rem;
      margin-right: 4px;
  }
</style>
"""
st.markdown(HIDE_CHROME_CSS, unsafe_allow_html=True)


# ---- Cached RAG client ----
@st.cache_resource
def get_rag() -> ClinicRAG:
    """Initialize the RAG pipeline once, reuse for every query."""
    return ClinicRAG()


def safe_get_rag():
    """Wrap RAG initialization with friendly errors."""
    try:
        return get_rag(), None
    except Exception as exc:
        return None, str(exc)


# ---- Header ----
st.title("🩺 ClinicSearch")
st.caption("Offline clinical knowledge for rural healthcare workers — WHO, MSF, OpenFDA")

# ---- Initialize RAG (cached) ----
rag, init_error = safe_get_rag()
if init_error:
    st.error(
        f"**Could not initialize the system.** Make sure Ollama is running "
        f"and the vector index has been built.\n\n"
        f"Error details: `{init_error}`\n\n"
        f"To fix this, run:\n"
        f"```bash\n"
        f"ollama serve  # in a separate terminal\n"
        f"python -m src.ingest\n"
        f"python -m src.embed\n"
        f"```"
    )
    st.stop()


# ---- Input section ----
# Language selector (purely a hint to the user — Whisper auto-detects)
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("##### Ask a clinical question")
with col2:
    language_hint = st.selectbox(
        "Language",
        ["Auto-detect", "English", "Português", "Swahili", "हिन्दी", "Français"],
        index=0,
        label_visibility="collapsed",
    )

# Voice input
voice_query: str | None = None
audio_bytes = st.audio_input(
    "🎙️ Tap to speak (any language)",
    label_visibility="visible",
)
if audio_bytes is not None:
    with st.spinner("Transcribing..."):
        try:
            from src.voice import transcribe_wav_bytes

            wav_bytes = audio_bytes.getvalue() if hasattr(audio_bytes, "getvalue") else audio_bytes
            voice_query, detected_lang = transcribe_wav_bytes(wav_bytes)
            if voice_query:
                st.success(
                    f"Heard ({detected_lang}): _{voice_query}_"
                )
        except Exception as exc:
            st.warning(f"Voice transcription failed: {exc}")

# Text input (pre-filled with voice query if available)
text_query = st.text_input(
    "Or type your question",
    value=voice_query or "",
    placeholder="e.g. First-line treatment for malaria in children",
    label_visibility="collapsed",
)

# Submit
ask_clicked = st.button("Ask", type="primary", use_container_width=True)


# ---- Handle query ----
query = text_query.strip() if text_query else ""

if ask_clicked and query:
    with st.spinner("Searching WHO/MSF/FDA guidelines..."):
        response = rag.ask(query)

    # ---- Render answer ----
    if response.error:
        st.error(f"Something went wrong: {response.error}")
    elif response.is_refusal:
        st.markdown(
            f'<div class="refusal-card">'
            f'<strong>⚠️ Out of scope</strong><br>{response.answer}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="answer-card">{response.answer}</div>',
            unsafe_allow_html=True,
        )

    # Metrics row
    st.markdown(
        f'<div class="metric-row">'
        f'⏱ {response.total_ms / 1000:.1f}s '
        f'(retrieval {response.retrieval_ms:.0f}ms + generation '
        f'{response.generation_ms:.0f}ms) · '
        f'{len(response.sources)} sources consulted'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---- Sources ----
    if response.sources and not response.is_refusal:
        with st.expander(f"📚 Source passages ({len(response.sources)})"):
            for i, src in enumerate(response.sources, 1):
                page_label = (
                    f"page {src['page']}" if src["page"] > 0 else "record"
                )
                st.markdown(
                    f"**Source {i}: {src['source']}** "
                    f"_{page_label}_ · match score {src['score']:.0%}"
                )
                st.caption(src["text"][:500] + ("..." if len(src["text"]) > 500 else ""))
                st.divider()

elif ask_clicked and not query:
    st.info("Please type a question or use voice input above.")


# ---- Footer ----
st.divider()
st.caption(
    "⚠️ ClinicSearch is a decision-support tool, not a substitute for "
    "clinical judgment. Always verify with a qualified physician. "
    "Sources: WHO Essential Medicines List, MSF Clinical Guidelines, "
    "OpenFDA drug data."
)
