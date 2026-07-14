"""VoiceBrain — local speech-to-text.

Wraps faster-whisper (CTranslate2 port of OpenAI Whisper). The model runs
fully on CPU; the first call downloads the weights (~460 MB for "small")
into the local Hugging Face cache and every call after that is offline.

"small" is the sweet spot for multilingual accuracy on CPU. Set
WHISPER_MODEL=tiny or base in .env for faster/lighter transcription.
"""

import logging
import os
from pathlib import Path

from faster_whisper import WhisperModel

log = logging.getLogger("voicebrain.transcribe")

_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """Lazy-load the model once per process (loading takes seconds)."""
    global _model
    if _model is None:
        log.info("Loading whisper model '%s' (first run downloads it)...", _MODEL_NAME)
        _model = WhisperModel(_MODEL_NAME, device="cpu", compute_type="int8")
        log.info("Whisper model ready.")
    return _model


def transcribe(audio_path: Path) -> tuple[str, str]:
    """Transcribe an audio file. Returns (text, detected_language)."""
    segments, info = _get_model().transcribe(str(audio_path), vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    log.info(
        "Transcribed %s: lang=%s (p=%.2f), %d chars",
        audio_path.name, info.language, info.language_probability, len(text),
    )
    return text, info.language
