"""
ClinicSearch — Voice transcription via faster-whisper.

Multilingual speech-to-text using faster-whisper with the `base` model
(quantized to INT8). Runs entirely on CPU at ~4x real-time on M2.

Why faster-whisper instead of openai-whisper:
  - 4x faster on CPU thanks to CTranslate2
  - No PyTorch dependency (avoids MPS/Metal headaches)
  - Identical model weights and quality
  - Drop-in CPU-only deployment

Why `base` not `tiny`:
  - Whisper tiny shows >40% WER on Hindi and Swahili in published research
  - base model: 142 MB, ~250 MB RAM, dramatically better non-English accuracy
  - For clinical multilingual context, the size tradeoff is non-negotiable

Usage:
    from src.voice import transcribe
    text, language = transcribe(audio_array_or_path)
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

AudioInput = Union[np.ndarray, str, Path, bytes]


@lru_cache(maxsize=1)
def _get_model():
    """Lazy-load and cache the WhisperModel. First call downloads ~142MB."""
    from faster_whisper import WhisperModel

    print(f"Loading faster-whisper model `{config.WHISPER_MODEL}` "
          f"(compute_type={config.WHISPER_COMPUTE_TYPE})...")
    model = WhisperModel(
        config.WHISPER_MODEL,
        device="cpu",
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )
    return model


def transcribe_audio_array(
    audio: np.ndarray,
    sample_rate: int = 16000,
    language: str | None = None,
) -> tuple[str, str]:
    """Transcribe a numpy audio array. Returns (text, detected_language).

    Args:
        audio: 1-D float32 numpy array of audio samples
        sample_rate: sample rate of the audio (default 16000)
        language: ISO language code to force; None = auto-detect

    Returns:
        (transcribed_text, language_code)
    """
    # faster-whisper expects float32, mono, 16kHz
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    # Squeeze any extra dims
    audio = np.squeeze(audio)
    if audio.ndim != 1:
        # Take first channel if stereo
        audio = audio[:, 0] if audio.shape[1] > 0 else audio.flatten()

    # Resample to 16kHz if needed (using simple decimation; for production use librosa)
    if sample_rate != 16000:
        # Simple linear resampling - good enough for speech
        target_len = int(len(audio) * 16000 / sample_rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    model = _get_model()
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=True,  # voice activity detection trims silence
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=(
            "Clinical question about patient treatment, drug dosage, or medical "
            "condition. क्लिनिकल प्रश्न। Pergunta clínica. Swali la kiafya. "
            "Question clinique."
        ),
        condition_on_previous_text=False,
    )

    text = " ".join(s.text for s in segments).strip()
    return text, info.language


def transcribe_wav_bytes(wav_bytes: bytes) -> tuple[str, str]:
    """Convenience wrapper: take raw WAV bytes (from Streamlit's audio_input)
    and return (text, language).
    """
    import io
    import soundfile as sf

    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return transcribe_audio_array(audio, sample_rate=sample_rate)


def transcribe(audio: AudioInput) -> tuple[str, str]:
    """Polymorphic entry point: accepts numpy array, file path, or WAV bytes."""
    if isinstance(audio, np.ndarray):
        return transcribe_audio_array(audio)
    if isinstance(audio, (str, Path)):
        model = _get_model()
        segments, info = model.transcribe(
            str(audio),
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(s.text for s in segments).strip()
        return text, info.language
    if isinstance(audio, (bytes, bytearray)):
        return transcribe_wav_bytes(bytes(audio))
    raise TypeError(f"Unsupported audio input type: {type(audio)}")


# ---- CLI smoke test ----

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test voice transcription")
    parser.add_argument(
        "audio_path",
        nargs="?",
        help="Path to an audio file (WAV, MP3, etc.) to transcribe",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record 5 seconds from the default microphone instead",
    )
    args = parser.parse_args()

    if args.record:
        import sounddevice as sd

        print("Recording for 5 seconds... speak now.")
        sample_rate = 16000
        audio = sd.rec(
            int(5 * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("Recording complete. Transcribing...")
        text, lang = transcribe_audio_array(audio.flatten(), sample_rate=sample_rate)
    elif args.audio_path:
        text, lang = transcribe(args.audio_path)
    else:
        print("Provide --record or an audio file path.")
        sys.exit(1)

    print(f"\nDetected language: {lang}")
    print(f"Transcription: {text}")
