"""VoiceBrain — local speech-to-text.

Wraps faster-whisper (CTranslate2 port of OpenAI Whisper), fully on CPU.
Default model is large-v3-turbo: near large-v3 accuracy at a fraction of
the compute. Set WHISPER_MODEL=small (or tiny/base) in .env for lighter
hosts — the 8GB-VPS "lite" profile uses small.

Name glossary: Whisper often mangles names it has rarely seen (Tunisian
names especially: "Walid" -> "we did"). Whisper accepts an initial_prompt
that biases decoding, so we feed it the names already present in THIS
user's notes. Once a name is written once, it is recognized afterwards —
personalized ASR adaptation for free.
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
    """Lazy-load the model once per process (loading takes seconds)."""
    global _model
    if _model is None:
        log.info("Loading whisper model '%s' (first run downloads it)...", _MODEL_NAME)
        _model = WhisperModel(_MODEL_NAME, device="cpu", compute_type="int8")
        log.info("Whisper model ready.")
    return _model


def transcribe(audio_path: Path, user_id: int | None = None) -> tuple[str, str]:
    """Transcribe an audio file. Returns (text, detected_language)."""
    prompt = None
    if user_id is not None:
        names = known_people(user_id)
        if names:
            prompt = "Names that may appear: " + ", ".join(names[:30]) + "."

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
