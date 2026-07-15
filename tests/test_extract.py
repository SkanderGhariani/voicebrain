"""Extraction contract tests: schema shape and prompt regression guards.

The LLM itself is never loaded here — these tests pin down the contract that
the rest of the system (storage, bot formatting) depends on, and guard the
prompt rules that past bugs taught us to keep (no invention, date fidelity).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import extract


def test_schema_has_all_required_fields():
    assert set(extract.SCHEMA["required"]) == {"summary", "tasks", "dates", "people", "topics"}


def test_schema_types():
    props = extract.SCHEMA["properties"]
    assert props["summary"]["type"] == "string"
    for field in ("tasks", "dates", "people", "topics"):
        assert props[field]["type"] == "array"
        assert props[field]["items"]["type"] == "string"


def test_prompt_keeps_do_not_invent_rule():
    assert "Do not invent" in extract.SYSTEM_PROMPT


def test_prompt_keeps_date_fidelity_rule():
    # Regression guard: the model once swapped Thursday for Friday.
    assert "EXACTLY" in extract.SYSTEM_PROMPT
    assert "EVERY date" in extract.SYSTEM_PROMPT
