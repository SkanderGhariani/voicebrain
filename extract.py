"""VoiceBrain — structured extraction with a local LLM.

Takes a raw transcript (any language) and produces structured English data:
summary, tasks, dates, people, topics. Runs Qwen2.5-3B-Instruct (GGUF, Q4)
fully locally via llama.cpp.

The key technique: JSON-schema-constrained generation. llama.cpp compiles the
schema into a GBNF grammar that masks invalid tokens at every decoding step,
so the model is physically unable to produce anything but valid JSON matching
the schema. No retries, no "please output JSON" prayers.
"""

import json
import logging
import os

log = logging.getLogger("voicebrain.extract")

MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "One-sentence summary in English"},
        "tasks": {"type": "array", "items": {"type": "string"}},
        "dates": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "tasks", "dates", "people", "topics"],
}

SYSTEM_PROMPT = (
    "You extract structured information from voice-note transcripts. "
    "The transcript may be in any language; your output is always in English "
    "(keep proper names as they are). Extract:\n"
    "- summary: one short sentence, what the note is about\n"
    "- tasks: concrete action items mentioned (empty list if none)\n"
    "- dates: EVERY date, day or time mentioned, none skipped — copy each EXACTLY "
    "as spoken, translated to English but never changed (if the transcript says "
    "Thursday, write Thursday, not any other day)\n"
    "- people: names of people mentioned\n"
    "- topics: 1-3 short topic tags\n"
    "Only extract what is actually in the transcript. Do not invent."
)

_llm = None


def _get_llm():
    """Lazy-load the model once per process (several GB into RAM, takes seconds).

    llama_cpp is imported here rather than at module level so the module can
    be imported (e.g. by tests and CI) without the native dependency installed.
    """
    global _llm
    if _llm is None:
        from llama_cpp import Llama

        log.info("Loading LLM from %s ...", MODEL_PATH)
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=max(1, (os.cpu_count() or 4) - 1),
            verbose=False,
        )
        log.info("LLM ready.")
    return _llm


def extract(transcript: str, language: str) -> dict:
    """Extract structured data from a transcript. Always returns schema-valid dict."""
    resp = _get_llm().create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript (language: {language}):\n{transcript}"},
        ],
        response_format={"type": "json_object", "schema": SCHEMA},
        temperature=0.1,
        max_tokens=512,
    )
    data = json.loads(resp["choices"][0]["message"]["content"])
    log.info("Extracted: %d tasks, %d people, topics=%s",
             len(data["tasks"]), len(data["people"]), data["topics"])
    return data
