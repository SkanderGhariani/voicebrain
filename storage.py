"""SQLite note storage. Every read path filters on user_id."""

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
            user_id INTEGER,
            created_at REAL NOT NULL,
            language TEXT,
            transcript TEXT NOT NULL,
            summary TEXT,
            tasks TEXT,    -- JSON array
            dates TEXT,    -- JSON array
            people TEXT,   -- JSON array
            topics TEXT,   -- JSON array
            embedding BLOB
        )"""
    )
    # migrations for older DBs
    cols = [r[1] for r in conn.execute("PRAGMA table_info(notes)")]
    if "user_id" not in cols:
        conn.execute("ALTER TABLE notes ADD COLUMN user_id INTEGER")
    if "embedding" not in cols:
        conn.execute("ALTER TABLE notes ADD COLUMN embedding BLOB")
    return conn


def save_note(user_id: int, transcript: str, language: str, extracted: dict) -> int:
    """Persist a note for a user; returns its id."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO notes (user_id, created_at, language, transcript, "
            "summary, tasks, dates, people, topics) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
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


def recent_notes(user_id: int, limit: int = 5) -> list[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM notes WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_note(user_id: int, note_id: int) -> bool:
    """Delete a note if it belongs to this user. Returns True if deleted."""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM notes WHERE id = ? AND user_id = ?", (note_id, user_id)
        )
        return cur.rowcount > 0


def known_people(user_id: int) -> list[str]:
    """Distinct people names across this user's notes, for the whisper glossary."""
    names: set[str] = set()
    with _connect() as conn:
        for (people_json,) in conn.execute(
            "SELECT people FROM notes WHERE user_id = ? AND people IS NOT NULL",
            (user_id,),
        ):
            try:
                names.update(n.strip() for n in json.loads(people_json) if n.strip())
            except (json.JSONDecodeError, TypeError):
                continue
    return sorted(names)


def delete_all_notes(user_id: int) -> int:
    """Delete ALL of this user's notes. Returns how many were removed."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM notes WHERE user_id = ?", (user_id,))
        return cur.rowcount
