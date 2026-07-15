"""Memory layer tests with fake models: ranking, filtering, isolation.

No real models are loaded: a deterministic fake encoder maps keyword hits to
axes of a small vector space, and a fake reranker scores by word overlap.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import memory
import storage

ALICE, BOB = 111, 222

EXTRACTED = {
    "summary": "",
    "tasks": [],
    "dates": [],
    "people": [],
    "topics": [],
}

_KEYWORDS = ["pizza", "plumber", "wedding"]


class FakeEncoder:
    """Text -> unit vector; axis i lights up if keyword i appears."""

    def encode(self, texts, normalize_embeddings=True):
        out = []
        for text in texts:
            v = np.zeros(len(_KEYWORDS) + 1, dtype=np.float32)
            lower = text.lower()
            for i, kw in enumerate(_KEYWORDS):
                if kw in lower:
                    v[i] = 1.0
            if not v.any():
                v[-1] = 1.0  # "other" axis
            out.append(v / np.linalg.norm(v))
        return out


class FakeReranker:
    """Positive score iff a content word from the query appears in the passage."""

    def predict(self, pairs):
        scores = []
        for query, passage in pairs:
            words = {w for w in query.lower().split() if len(w) > 3}
            hit = any(w in passage.lower() for w in words)
            scores.append(5.0 if hit else -5.0)
        return scores


@pytest.fixture(autouse=True)
def fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(memory, "_encoder", FakeEncoder())
    monkeypatch.setattr(memory, "_reranker", FakeReranker())
    monkeypatch.setattr(memory, "RERANK_ENABLED", True)


def _add_note(user_id, transcript, summary):
    note_id = storage.save_note(user_id, transcript, "en", {**EXTRACTED, "summary": summary})
    memory.store_embedding(note_id, transcript, summary)
    return note_id


def test_relevant_note_ranks_first_and_junk_is_filtered():
    pizza_id = _add_note(ALICE, "try the new pizza place", "pizza plan")
    _add_note(ALICE, "call the plumber about the sink", "plumber call")

    hits = memory.search(ALICE, "where was that pizza restaurant?")
    assert [h["id"] for h in hits] == [pizza_id], "only the pizza note is relevant"


def test_no_relevant_notes_returns_empty():
    _add_note(ALICE, "call the plumber about the sink", "plumber call")
    assert memory.search(ALICE, "couscous recipe ideas") == []


def test_search_is_user_isolated():
    _add_note(ALICE, "try the new pizza place", "pizza plan")
    assert memory.search(BOB, "pizza restaurant") == []


def test_rerank_disabled_falls_back_to_cosine_order():
    pizza_id = _add_note(ALICE, "try the new pizza place", "pizza plan")
    plumber_id = _add_note(ALICE, "call the plumber", "plumber call")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(memory, "RERANK_ENABLED", False)
        hits = memory.search(ALICE, "pizza night", 5)

    ids = [h["id"] for h in hits]
    assert ids[0] == pizza_id
    assert plumber_id in ids, "without reranking nothing is filtered out"


def test_threshold_is_tunable():
    _add_note(ALICE, "try the new pizza place", "pizza plan")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(memory, "RERANK_THRESHOLD", 10.0)  # above the fake's max score
        assert memory.search(ALICE, "pizza restaurant") == []
