# VoiceBrain

A self-hosted, multilingual voice-notes brain on Telegram.

Send it a voice note in French, Arabic, English, Italian or German. It transcribes it locally,
extracts the structure (tasks, dates, people, topics) with a local quantized LLM, remembers
everything, and lets you search your own voice history semantically or ask questions about it.

**No cloud AI APIs. Everything runs on your own machine.**

- 🎙️ Transcription: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (multilingual, CPU)
- 🧠 Extraction: [llama.cpp](https://github.com/ggerganov/llama.cpp) + Qwen2.5-3B-Instruct (GGUF, Q4), with JSON-schema-constrained output
- 🔎 Memory: sentence-transformers embeddings + SQLite
- 🤖 Interface: Telegram bot (polling — no domain, no SSL, runs anywhere)

> 🚧 Work in progress — building in public.

## Status

- [x] Phase 0 — project setup
- [x] Phase 1 — echo bot (Telegram loop)
- [x] Phase 2 — ears (local transcription)
- [x] Phase 3 — brain (structured extraction, local LLM)
- [x] Phase 4 — memory (semantic search + /ask RAG)
- [ ] Phase 5 — polish (Docker, docs)
- [ ] Phase 6 — deploy guide

## License

MIT
