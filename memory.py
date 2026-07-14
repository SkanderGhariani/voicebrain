"""VoiceBrain — semantic memory.

Embeds every note with a multilingual sentence-transformer and searches by
cosine similarity. Embeddings are stored as float32 blobs in SQLite and
compared brute-force with numpy: at personal-notes scale (thousands, not
millions) a vector index would add complexity for zero measurable gain.

/ask is retrieval-augmented generation (RAG): retrieve the top-k relevant
notes, hand them to the local LLM as context, and answer ONLY from them,
citing note ids.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime

import numpy as np

from storage import DB_PATH, _connect

log = logging.getLogger("voicebrain.memory")

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model '%s'...", _MODEL_NAME)
        _encoder = SentenceTransformer(_MODEL_NAME, device="cpu")
        log.info("Embedding model ready.")
    return _encoder


def embed_text(text: str) -> np.ndarray:
    """Normalized embedding vector for a text."""
    vec = _get_encoder().encode([text], normalize_embeddings=True)[0]
    return vec.astype(np.float32)


def store_embedding(note_id: int, transcript: str, summary: str) -> None:
    """Embed transcript+summary (covers original language AND English) and save."""
    vec = embed_text(f"{summary}\n{transcript}")
    with _connect() as conn:
        conn.execute(
            "UPDATE notes SET embedding = ? WHERE id = ?", (vec.tobytes(), note_id)
        )


def search(user_id: int, query: str, k: int = 5) -> list[dict]:
    """Top-k of THIS USER's notes by cosine similarity to the query."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM notes WHERE embedding IS NOT NULL AND user_id = ?",
                (user_id,),
            )
        ]
    if not rows:
        return []

    q = embed_text(query)
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = mat @ q  # vectors are normalized, so dot product == cosine similarity
    order = np.argsort(-sims)[:k]
    results = []
    for i in order:
        r = rows[int(i)]
        r["score"] = float(sims[int(i)])
        results.append(r)
    return results


def ask(user_id: int, question: str, k: int = 4) -> str:
    """Answer a question from THIS USER's notes (RAG), citing note ids."""
    from extract import _get_llm  # reuse the already-loaded LLM

    hits = search(user_id, question, k)
    if not hits:
        return "I have no notes to answer from yet."

    context_blocks = []
    for h in hits:
        day = datetime.fromtimestamp(h["created_at"]).strftime("%Y-%m-%d")
        context_blocks.append(f"[Note #{h['id']} — {day}] {h['transcript']}")
    context = "\n".join(context_blocks)

    resp = _get_llm().create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about the user's personal voice notes. "
                    "Use ONLY the notes provided. Cite the note ids you used, "
                    "like (#3). If the notes don't contain the answer, say so "
                    "plainly. Answer in the language of the question. Be brief."
                ),
            },
            {
                "role": "user",
                "content": f"My notes:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.1,
        max_tokens=300,
    )
    return resp["choices"][0]["message"]["content"].strip()
