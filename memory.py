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
import os
import sqlite3
import time
from datetime import datetime

import numpy as np

from storage import DB_PATH, _connect

log = logging.getLogger("voicebrain.memory")

_MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
# e5 models are trained with these prefixes; retrieval quality drops without them.
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "

MIN_SCORE = 0.15  # below this, a hit is noise — don't feed it to the LLM

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model '%s'...", _MODEL_NAME)
        _encoder = SentenceTransformer(_MODEL_NAME, device="cpu")
        log.info("Embedding model ready.")
    return _encoder


def _embed(text: str) -> np.ndarray:
    """Normalized embedding vector for a text."""
    vec = _get_encoder().encode([text], normalize_embeddings=True)[0]
    return vec.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    return _embed(_QUERY_PREFIX + text)


def embed_passage(text: str) -> np.ndarray:
    return _embed(_PASSAGE_PREFIX + text)


def store_embedding(note_id: int, transcript: str, summary: str, topics: list[str] | None = None) -> None:
    """Embed summary+transcript+topics (original language AND English) and save."""
    parts = [summary, transcript]
    if topics:
        parts.append(", ".join(topics))
    vec = embed_passage("\n".join(p for p in parts if p))
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

    q = embed_query(query)
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = mat @ q  # vectors are normalized, so dot product == cosine similarity
    order = np.argsort(-sims)[:k]
    results = []
    for i in order:
        r = rows[int(i)]
        r["score"] = float(sims[int(i)])
        results.append(r)
    return results


def ask(user_id: int, question: str, k: int = 6) -> str:
    """Answer a question from THIS USER's notes (RAG), citing note ids."""
    from extract import _get_llm  # reuse the already-loaded LLM

    hits = [h for h in search(user_id, question, k) if h["score"] >= MIN_SCORE]
    if not hits:
        return "I have no notes about that yet."

    context_blocks = []
    for h in hits:
        day = datetime.fromtimestamp(h["created_at"]).strftime("%Y-%m-%d")
        summary = h["summary"] or ""
        context_blocks.append(
            f"[Note #{h['id']} — recorded {day}]\n"
            f"Transcript: {h['transcript']}\n"
            f"Summary: {summary}"
        )
    context = "\n\n".join(context_blocks)

    resp = _get_llm().create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about the user's personal voice notes. "
                    "Use ONLY the notes provided. Answer the question directly "
                    "first, then cite the note ids used, like (#3). "
                    "The transcripts come from speech recognition, so names may "
                    "be spelled inconsistently: treat similar-sounding names as "
                    "the same person (e.g. 'Sava' and 'Sarra') and answer using "
                    "the name from the question. If the notes truly don't "
                    "contain the answer, say so plainly. Answer in the language "
                    "of the question. Be brief."
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
