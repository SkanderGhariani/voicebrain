"""One-off: embed notes saved before the memory layer existed.
Run: python backfill_embeddings.py
"""

import logging
import sqlite3

logging.basicConfig(level=logging.INFO)

from memory import _ensure_column, store_embedding
from storage import _connect

_ensure_column()
with _connect() as conn:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, transcript, summary FROM notes WHERE embedding IS NULL"
    ).fetchall()

print(f"Backfilling {len(rows)} notes...")
for r in rows:
    store_embedding(r["id"], r["transcript"], r["summary"] or "")
    print(f"  #{r['id']} embedded")
print("Done.")
