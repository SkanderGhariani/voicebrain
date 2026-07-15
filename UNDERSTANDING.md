# UNDERSTANDING.md — your guided tour (read before any interview)

*This file is for YOU, not for visitors. Work through it once, slowly, with the code open.
Budget: ~1 hour. After it, you can defend every line of this project.*

---

## 1. The story in one paragraph (memorize this shape)

> "VoiceBrain is a Telegram bot that turns voice notes in any language into structured,
> searchable memory — fully self-hosted, no cloud AI APIs. Whisper transcribes locally,
> a quantized Qwen model extracts tasks/dates/people as grammar-constrained JSON, notes are
> embedded for semantic search, and /ask does RAG over your own notes with citations —
> it even answers by voice. It runs on a €5 CPU server."

## 2. The pipeline, top-down

1. **Telegram → `bot.py`.** The bot POLLS Telegram ("any new messages?" every few seconds).
   No webhook = no domain/SSL/open ports = runs anywhere. The voice file is downloaded
   and a job is queued.
2. **The queue (`bot.py`).** ONE worker processes notes serially. Why: each note runs
   Whisper + a 7B model; two at once doubles peak RAM and thrashes the CPU. Users see
   "queued, N ahead of you."
3. **Ears (`transcribe.py`).** faster-whisper (CTranslate2 port of Whisper), int8, CPU.
   Auto language detect. The *name glossary*: names from your past notes are passed as
   `initial_prompt` so Whisper recognizes YOUR people.
4. **Brain (`extract.py`).** llama.cpp runs Qwen2.5 (GGUF, Q4 quantized). The JSON schema
   is compiled to a GBNF grammar; during generation every token that would break the JSON
   is masked out. Output is ALWAYS valid JSON. Prompt rules ("do not invent", "copy dates
   exactly") guard the CONTENT.
5. **Memory (`storage.py` + `memory.py`).** SQLite, one file. Every note stores its
   embedding (a ~384-dim float vector from multilingual-e5) as a BLOB. `user_id` on every
   row; every query filters on it — users cannot see each other's notes.
6. **Search (`memory.py`).** Two stages: (1) cosine similarity over embeddings shortlists
   10 candidates — fast, great recall, but scores bunch together; (2) a cross-encoder
   reranker reads query+note together and gives real relevance scores (+4 relevant,
   −8 junk) — below-threshold results are dropped, so /search never shows garbage.
7. **/ask (`memory.py`).** RAG = the search above (Retrieve) + Qwen answering from the
   retrieved notes only (Generate), citing note ids. Today's date is injected so
   "next Thursday" and "this week" resolve correctly.
8. **Mouth (`tts.py`).** Piper (local neural TTS) speaks the answer; PyAV transcodes
   WAV → OGG/Opus because that's Telegram's voice-message format.

## 3. Key concepts you must be able to explain

- **Polling vs webhook:** poll = you call the API in a loop (works behind NAT); webhook =
  they call your public URL (needs server+SSL, better at scale). It's a message queue —
  the laptop is a consumer; offline messages wait in the bot's mailbox (~24h).
- **Quantization (Q4/GGUF):** storing model weights in 4 bits instead of 16 → ~4x smaller,
  runs on CPU RAM, small accuracy cost. GGUF is llama.cpp's file format.
- **Grammar-constrained decoding:** the LLM outputs probabilities over ALL tokens each
  step; llama.cpp zeroes the probability of every token that would violate the compiled
  JSON grammar. The model *cannot* produce invalid JSON. (Guarantees form, NOT truth.)
- **Embeddings:** text → vector where meaning determines position; "pizza place" and
  "restaurant italien" land close. Cosine similarity = the angle between vectors.
- **Bi-encoder vs cross-encoder:** bi-encoder embeds query and note SEPARATELY (fast,
  cacheable, coarse); cross-encoder reads them TOGETHER (slow per pair, precise). Standard
  production pattern: bi-encoder recall → cross-encoder rerank.
- **RAG:** Retrieve relevant docs, feed them to the LLM as context, answer only from them.
  Kills hallucination, adds citations, keeps knowledge fresh without retraining.

## 4. The three bugs we hit (tell these stories — engineers love scars)

1. **The English prompt that broke French.** The name glossary was "Names that may appear:
   Walid..." — an ENGLISH sentence. Whisper treats `initial_prompt` as conversation
   context, so it biased output to English: a French note came out as broken English with
   the days translated WRONG. Fix: bare names only. Lesson: prompts influence more than
   content — they leak language.
2. **The shared-memory leak.** V1 had one notes table with no owner column: any user of
   the bot could read MY notes via /recent. Fix: `user_id` on every row + every query
   filtered. Lesson: multi-tenancy is a day-one column, not an afterthought.
3. **The concurrent RAM bomb.** Two simultaneous voice notes each loaded Whisper+LLM work
   → double peak RAM on a small host. Fix: serial queue with position feedback. Lesson:
   on CPU hosts you budget memory like money.

Plus one meta-story: **the measured revert.** We upgraded MiniLM→e5 expecting better
search; the benchmark showed identical top-1 and *worse* score readability. Instead of
keeping the "upgrade" on vibes, we solved relevance with a reranker. Benchmarks over hype.

## 5. Likely interview questions (with the strong answer's skeleton)

1. *Why local models instead of OpenAI?* — Privacy (personal voice data), zero marginal
   cost, no vendor dependency, and it proves I understand the layer beneath the API.
   Tradeoff: quality/latency vs GPT-class — acceptable here, measured openly in the README.
2. *How do you guarantee the LLM returns valid JSON?* — Grammar-constrained decoding
   (explain token masking). Contrast with beg-and-retry (which I did at CURE in 2025 —
   honest growth story).
3. *Why no vector database?* — Scale math: thousands of vectors = one numpy matmul in
   microseconds. Qdrant/pgvector earn their complexity at millions. Right tool, right scale.
4. *How does search work?* — Two-stage: e5 bi-encoder recall (top-10) → cross-encoder
   rerank with a relevance threshold. Explain WHY two stages (score compression vs precision).
5. *What breaks at 10,000 users?* — Serial queue becomes the bottleneck → worker pool with
   per-user fairness; SQLite → Postgres; brute-force search → ANN index (then a vector DB
   earns its place); polling → webhook behind a load balancer; model servers split from
   the bot process.
6. *Why polling?* — (see concepts). Deliberate tradeoff, documented.
7. *Biggest bug?* — Pick one of the three above, tell it with the lesson.
8. *How do you test AI code?* — Separate the deterministic shell from the stochastic core:
   unit tests with fake encoder/reranker for ranking/filtering/isolation logic (0.2s, CI),
   real-model behavior tested via a repeatable manual benchmark (6 notes incl. traps like
   negation and no-task notes).
9. *Whisper limitations?* — One language per note, name cold-start, hallucination on
   noise. All documented; glossary mitigates names.
10. *Why Telegram?* — Free bot API, no approval process, voice-message UX is native, and
    the interface costs zero frontend work — the project's value is the pipeline, not a UI.
11. *What would you do next?* — Reminders from extracted dates (scheduling + date
    normalization), hybrid keyword+vector retrieval, streaming.
12. *Is the reranker overkill for 20 notes?* — Product-wise borderline, engineering-wise
    it fixed a real UX failure (junk results at deceptive scores) with the standard
    production pattern. I timeboxed it and measured before/after.
13. *Why Q4 and not Q8/F16?* — RAM budget: 7B@Q4 ≈ 4.7GB fits the profile; Q4_K_M is the
    accepted sweet spot of size vs quality loss.
14. *Security?* — Token in .env (never committed), per-user data isolation, no shell-out
    on user input except Piper with fixed args, length caps on audio.
15. *What did you NOT build, and why?* — Web UI, user accounts, vector DB, GPU serving,
    package restructure — each cut for scale-appropriateness. Knowing what not to build
    is the actual skill.

## 6. Numbers to have in your pocket

~7.5GB RAM total (quality profile) · extraction ~10-15s, /ask ~40-60s on CPU ·
reranker separation: +4 vs −8 logits where cosine gave 0.80 vs 0.76 ·
14 unit tests, 0.2s, zero model loads · models: whisper-turbo 1.6GB, Qwen7B 4.4GB,
e5-small 0.5GB, reranker 0.5GB, Piper voices 60MB each.
