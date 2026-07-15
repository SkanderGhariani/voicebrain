"""Semantic memory: embeddings, two-stage search (recall + rerank), and RAG /ask.

Embeddings are stored as float32 blobs in SQLite and compared brute force with
numpy, which is plenty at this scale.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime

import numpy as np

from storage import _connect

log = logging.getLogger("voicebrain.memory")

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
# e5 was trained with these prefixes; retrieval degrades without them
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANK_ENABLED = os.getenv("RERANK", "1") != "0"  # lite profile can disable
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.0"))

RECALL_K = 10  # stage-1 shortlist size

_encoder = None
_reranker = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model '%s'...", EMBED_MODEL)
        _encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
        log.info("Embedding model ready.")
    return _encoder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        log.info("Loading reranker '%s'...", RERANK_MODEL)
        _reranker = CrossEncoder(RERANK_MODEL, device="cpu")
        log.info("Reranker ready.")
    return _reranker


def _embed(text: str) -> np.ndarray:
    vec = _get_encoder().encode([text], normalize_embeddings=True)[0]
    return vec.astype(np.float32)


def _note_text(summary: str, transcript: str, topics: list[str] | None = None) -> str:
    parts = [summary or "", transcript or ""]
    if topics:
        parts.append(", ".join(topics))
    return "\n".join(p for p in parts if p)


def store_embedding(
    note_id: int, transcript: str, summary: str, topics: list[str] | None = None
) -> None:
    """Embed a note (original language + English summary + topics) and save."""
    vec = _embed(_PASSAGE_PREFIX + _note_text(summary, transcript, topics))
    with _connect() as conn:
        conn.execute(
            "UPDATE notes SET embedding = ? WHERE id = ?", (vec.tobytes(), note_id)
        )


def _recall(user_id: int, query: str, k: int) -> list[dict]:
    """Stage 1: cosine shortlist of this user's notes."""
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

    q = _embed(_QUERY_PREFIX + query)
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = mat @ q  # vectors are normalized, dot == cosine
    order = np.argsort(-sims)
    results = []
    for i in order[:k]:
        r = rows[int(i)]
        r["score"] = float(sims[int(i)])
        results.append(r)
    return results


def search(user_id: int, query: str, k: int = 5) -> list[dict]:
    """Cosine shortlist, then cross-encoder rerank. May return fewer than k
    (or none) once candidates under RERANK_THRESHOLD are dropped."""
    candidates = _recall(user_id, query, RECALL_K)
    if not candidates or not RERANK_ENABLED:
        return candidates[:k]

    pairs = [(query, _note_text(c["summary"], c["transcript"])) for c in candidates]
    scores = _get_reranker().predict(pairs)
    for c, s in zip(candidates, scores):
        c["score"] = float(s)
    candidates.sort(key=lambda c: -c["score"])
    return [c for c in candidates if c["score"] >= RERANK_THRESHOLD][:k]


def ask(user_id: int, question: str, k: int = 8) -> str:
    """Answer a question from the user's notes, citing note ids."""
    from extract import _get_llm

    hits = search(user_id, question, k)
    if not hits:
        return "I have no notes about that yet."

    context_blocks = []
    for h in hits:
        day = datetime.fromtimestamp(h["created_at"]).strftime("%A %Y-%m-%d")
        block = [f"[Note #{h['id']} — recorded {day}]", f"Transcript: {h['transcript']}"]
        if h.get("summary"):
            block.append(f"Summary: {h['summary']}")
        for field in ("tasks", "dates", "people"):
            try:
                values = json.loads(h.get(field) or "[]")
            except json.JSONDecodeError:
                values = []
            if values:
                block.append(f"{field.capitalize()}: {', '.join(values)}")
        context_blocks.append("\n".join(block))
    context = "\n\n".join(context_blocks)

    today = datetime.now().strftime("%A, %Y-%m-%d")
    resp = _get_llm().create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    f"Today is {today}. "
                    "You answer questions about the user's personal voice notes. "
                    "Use ONLY the notes provided. Answer the question directly "
                    "first, then cite the note ids used, like (#3). Use today's "
                    "date to resolve relative dates (tomorrow, next Thursday...). "
                    "The transcripts come from speech recognition, so names may "
                    "be spelled inconsistently: treat similar-sounding names as "
                    "the same person and answer using the name from the question. "
                    "If the notes truly don't contain the answer, say so plainly. "
                    "Answer in the language of the question. Be brief."
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
