"""Local text-to-speech via the Piper CLI. One voice per language, English
fallback. Piper outputs WAV; Telegram voice messages need OGG/Opus, so we
transcode with PyAV."""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import av
from langdetect import LangDetectException, detect

log = logging.getLogger("voicebrain.tts")

_BIN_NAME = "piper.exe" if sys.platform == "win32" else "piper"
PIPER_BIN = Path("piper") / "piper" / _BIN_NAME
VOICES_DIR = Path("models") / "voices"

VOICES = {
    "en": "en_US-lessac-medium.onnx",
    "fr": "fr_FR-siwis-medium.onnx",
}
DEFAULT_LANG = "en"


def available() -> bool:
    return PIPER_BIN.exists()


def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return DEFAULT_LANG


def synthesize(text: str, lang: str) -> Path | None:
    """Text -> OGG/Opus file, or None if piper isn't installed."""
    if not available():
        return None
    voice = VOICES.get(lang, VOICES[DEFAULT_LANG])
    model = VOICES_DIR / voice
    if not model.exists():
        model = VOICES_DIR / VOICES[DEFAULT_LANG]

    wav_path = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.run(
        [str(PIPER_BIN), "--model", str(model), "--output_file", str(wav_path)],
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
    )

    ogg_path = wav_path.with_suffix(".ogg")
    _wav_to_opus(wav_path, ogg_path)
    wav_path.unlink(missing_ok=True)
    log.info("Synthesized %d chars (%s) -> %s", len(text), lang, ogg_path.name)
    return ogg_path


def _wav_to_opus(wav: Path, ogg: Path) -> None:
    with av.open(str(wav)) as inp, av.open(str(ogg), "w", format="ogg") as out:
        stream = out.add_stream("libopus", rate=48000)
        for frame in inp.decode(audio=0):
            frame.pts = None
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
