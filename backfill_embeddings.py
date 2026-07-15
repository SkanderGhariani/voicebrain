"""Re-embed all notes (used after changing the embedding model).
Run: python backfill_embeddings.py
"""

import json
import logging
import sqlite3

logging.basicConfig(level=logging.WARNING)

from memory import store_embedding
from storage import _connect

with _connect() as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, transcript, summary, topics FROM notes").fetchall()

print(f"Re-embedding {len(rows)} notes...")
for r in rows:
    try:
        topics = json.loads(r["topics"] or "[]")
    except json.JSONDecodeError:
        topics = []
    store_embedding(r["id"], r["transcript"], r["summary"] or "", topics)
    print(f"  #{r['id']} embedded")
print("Done.")
