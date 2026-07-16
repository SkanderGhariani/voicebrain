"""Schema and prompt-rule tests (no LLM load)."""

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
    assert "EXACTLY" in extract.SYSTEM_PROMPT
    assert "EVERY date" in extract.SYSTEM_PROMPT


def test_prompt_keeps_every_task_rule():
    assert "EVERY concrete action item" in extract.SYSTEM_PROMPT
