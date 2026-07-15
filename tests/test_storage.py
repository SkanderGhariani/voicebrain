"""Storage tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import storage

EXTRACTED = {
    "summary": "Buy a gift for Yasmine",
    "tasks": ["buy gift"],
    "dates": ["Saturday"],
    "people": ["Yasmine"],
    "topics": ["birthday"],
}

ALICE, BOB = 111, 222


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")


def test_save_and_recent_roundtrip():
    note_id = storage.save_note(ALICE, "transcript text", "en", EXTRACTED)
    notes = storage.recent_notes(ALICE)
    assert len(notes) == 1
    assert notes[0]["id"] == note_id
    assert notes[0]["transcript"] == "transcript text"
    assert notes[0]["summary"] == EXTRACTED["summary"]


def test_users_cannot_see_each_others_notes():
    storage.save_note(ALICE, "alice secret", "en", EXTRACTED)
    storage.save_note(BOB, "bob secret", "en", EXTRACTED)

    alice_notes = storage.recent_notes(ALICE)
    bob_notes = storage.recent_notes(BOB)

    assert [n["transcript"] for n in alice_notes] == ["alice secret"]
    assert [n["transcript"] for n in bob_notes] == ["bob secret"]


def test_delete_only_own_notes():
    note_id = storage.save_note(ALICE, "mine", "en", EXTRACTED)
    assert storage.delete_note(BOB, note_id) is False  # not Bob's note
    assert storage.recent_notes(ALICE), "note must survive foreign delete"
    assert storage.delete_note(ALICE, note_id) is True
    assert storage.recent_notes(ALICE) == []


def test_delete_all_scoped_to_user():
    storage.save_note(ALICE, "a1", "en", EXTRACTED)
    storage.save_note(ALICE, "a2", "en", EXTRACTED)
    storage.save_note(BOB, "b1", "en", EXTRACTED)

    assert storage.delete_all_notes(ALICE) == 2
    assert storage.recent_notes(ALICE) == []
    assert len(storage.recent_notes(BOB)) == 1


def test_known_people_distinct_and_scoped():
    storage.save_note(ALICE, "t", "en", {**EXTRACTED, "people": ["Walid", "Sarra"]})
    storage.save_note(ALICE, "t", "en", {**EXTRACTED, "people": ["Walid"]})
    storage.save_note(BOB, "t", "en", {**EXTRACTED, "people": ["Rami"]})

    assert storage.known_people(ALICE) == ["Sarra", "Walid"]
    assert storage.known_people(BOB) == ["Rami"]
