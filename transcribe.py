"""Local speech-to-text with faster-whisper (CPU).

Names from the user's past notes are passed as initial_prompt so whisper
stops mishearing them.
"""

import logging
import os
from pathlib import Path

from faster_whisper import WhisperModel

from storage import known_people

log = logging.getLogger("voicebrain.transcribe")

_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")
_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        log.info("Loading whisper model '%s'...", _MODEL_NAME)
        _model = WhisperModel(_MODEL_NAME, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: Path, user_id: int | None = None) -> tuple[str, str]:
    """Returns (text, detected_language)."""
    # bare names only: a full sentence here biases whisper's output language
    prompt = None
    if user_id is not None:
        names = known_people(user_id)
        if names:
            prompt = ", ".join(names[:30])

    segments, info = _get_model().transcribe(
        str(audio_path), vad_filter=True, initial_prompt=prompt
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    log.info(
        "Transcribed %s: lang=%s (p=%.2f), %d chars, glossary=%d names",
        audio_path.name,
        info.language,
        info.language_probability,
        len(text),
        len(prompt.split(",")) if prompt else 0,
    )
    return text, info.language
