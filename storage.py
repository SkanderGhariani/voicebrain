"""VoiceBrain — note storage.

SQLite, one file, zero setup. Each note keeps the raw transcript, the
detected language, and the structured extraction (as JSON columns).
Vector embeddings for semantic search are added in Phase 4.
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data") / "voicebrain.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            language TEXT,
            transcript TEXT NOT NULL,
            summary TEXT,
            tasks TEXT,    -- JSON array
            dates TEXT,    -- JSON array
            people TEXT,   -- JSON array
            topics TEXT    -- JSON array
        )"""
    )
    return conn


def save_note(transcript: str, language: str, extracted: dict) -> int:
    """Persist a note; returns its id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO notes (created_at, language, transcript, summary, tasks, dates, people, topics) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                language,
                transcript,
                extracted["summary"],
                json.dumps(extracted["tasks"], ensure_ascii=False),
                json.dumps(extracted["dates"], ensure_ascii=False),
                json.dumps(extracted["people"], ensure_ascii=False),
                json.dumps(extracted["topics"], ensure_ascii=False),
            ),
        )
        return cur.lastrowid


def recent_notes(limit: int = 5) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM notes ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
