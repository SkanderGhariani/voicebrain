"""Measure pipeline latencies on this machine: python scripts/bench.py

Uses a real voice note if one is dropped in bench_audio/ (any .ogg),
otherwise generates spoken input with piper. Times cold (first call,
includes model load) and warm runs.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SENTENCE = (
    "Tomorrow I need to pick up the dry cleaning, then meet Thomas for coffee "
    "at four, and I should also call the garage about the car before Friday."
)


def timed(label: str, fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    print(f"  {label}: {time.perf_counter() - t0:.1f}s")
    return result


def main():
    real = sorted((ROOT / "bench_audio").glob("*.ogg")) if (ROOT / "bench_audio").exists() else []
    if real:
        audio = real[0]
        print(f"input: real voice note {audio.name}")
    else:
        from tts import synthesize

        print("input: synthetic speech (piper); real speech may be a bit slower to transcribe")
        audio = timed("tts synthesis", synthesize, SENTENCE, "en")

    from transcribe import transcribe

    print("transcription:")
    timed("cold (model load included)", transcribe, audio)
    text, lang = timed("warm", transcribe, audio)

    from extract import extract

    print("extraction:")
    timed("cold (model load included)", extract, text, lang)
    timed("warm", extract, text, lang)

    import sqlite3

    from storage import _connect

    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT user_id, COUNT(*) n FROM notes WHERE user_id IS NOT NULL "
            "GROUP BY user_id ORDER BY n DESC LIMIT 1"
        ).fetchone()
    if not row:
        print("search/ask: skipped, no notes in local DB")
        return

    from memory import ask, search

    print(f"search (over {row['n']} notes):")
    timed("cold (model loads included)", search, row["user_id"], "coffee with Thomas")
    timed("warm", search, row["user_id"], "car repair")

    print(f"ask (over {row['n']} notes):")
    timed("warm models", ask, row["user_id"], "what do I need to do this week?")


if __name__ == "__main__":
    main()
