"""VoiceBrain — Telegram bot entry point.

Voice note -> local transcription (faster-whisper) -> structured extraction
with a local LLM (llama.cpp, schema-constrained JSON) -> per-user SQLite
memory with embeddings (/search, /ask, /recent, /delete).

Heavy work (whisper + LLM) runs through a single-worker queue: on CPU hosts,
processing two notes concurrently would double RAM and thrash both jobs.
Serial processing keeps resource usage bounded; users see their queue position.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from telegram import Message, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from extract import extract
from memory import ask, search, store_embedding
from storage import delete_all_notes, delete_note, recent_notes, save_note
from transcribe import transcribe
from tts import detect_lang, synthesize

load_dotenv()
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

AUDIO_DIR = Path("audio_tmp")
AUDIO_DIR.mkdir(exist_ok=True)

MAX_VOICE_SECONDS = 180

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("voicebrain")


# ---------- processing queue ----------

@dataclass
class VoiceJob:
    user_id: int
    audio_path: Path
    status: Message  # the "queued/processing..." message we keep editing


JOBS: asyncio.Queue[VoiceJob] = asyncio.Queue()


async def _worker() -> None:
    """Single consumer: processes voice jobs one at a time, forever."""
    while True:
        job = await JOBS.get()
        try:
            await _process(job)
        except Exception:
            log.exception("Voice job failed for user %s", job.user_id)
            try:
                await job.status.edit_text(
                    "\U0001F4A5 Something went wrong processing that note. Try again?"
                )
            except Exception:
                pass
        finally:
            job.audio_path.unlink(missing_ok=True)
            JOBS.task_done()


async def _process(job: VoiceJob) -> None:
    await job.status.edit_text("\U0001F442 Transcribing locally...")
    text, lang = await asyncio.to_thread(transcribe, job.audio_path, job.user_id)

    if not text:
        await job.status.edit_text("\U0001F92B I couldn't hear anything in that note.")
        return

    await job.status.edit_text("\U0001F9E0 Extracting structure...")
    data = await asyncio.to_thread(extract, text, lang)
    note_id = save_note(job.user_id, text, lang, data)
    await asyncio.to_thread(store_embedding, note_id, text, data["summary"], data["topics"])

    await job.status.edit_text(_format_note(note_id, lang, text, data))


# ---------- handlers ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi, I'm VoiceBrain \U0001F9E0\n\n"
        "Send me a voice note in any language. I transcribe it locally, "
        "extract tasks, dates and people, and remember everything.\n\n"
        "Commands:\n"
        "/recent — your latest notes\n"
        "/search <query> — semantic search over your notes\n"
        "/ask <question> — ask your own memory\n"
        "/delete <id> — remove a note\n"
        "/reset — wipe your whole memory"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "I work with voice notes — send me one!\n"
        "Or use /search, /ask, /recent, /delete."
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    voice = update.message.voice
    if voice.duration > MAX_VOICE_SECONDS:
        await update.message.reply_text(
            f"\U0001F62C That's {voice.duration}s — I only take notes up to "
            f"{MAX_VOICE_SECONDS}s. Shorter thoughts, please!"
        )
        return

    tg_file = await voice.get_file()
    dest = AUDIO_DIR / f"{tg_file.file_unique_id}.ogg"
    await tg_file.download_to_drive(custom_path=dest)
    log.info("Voice note queued: %s (%ss)", dest.name, voice.duration)

    waiting = JOBS.qsize()
    label = (
        "\U0001F442 Got it, processing..."
        if waiting == 0
        else f"\U0001F4E5 Queued — {waiting} note(s) ahead of you."
    )
    status = await update.message.reply_text(label)
    await JOBS.put(VoiceJob(update.effective_user.id, dest, status))


def _format_note(note_id: int, lang: str, transcript: str, data: dict) -> str:
    lines = [
        f"\U0001F4DD Note #{note_id} ({lang})",
        f"“{transcript}”",
        "",
        f"Summary: {data['summary']}",
    ]
    if data["tasks"]:
        lines.append("Tasks:\n" + "\n".join(f"  • {t}" for t in data["tasks"]))
    if data["dates"]:
        lines.append("Dates: " + ", ".join(data["dates"]))
    if data["people"]:
        lines.append("People: " + ", ".join(data["people"]))
    if data["topics"]:
        lines.append("Topics: " + ", ".join(data["topics"]))
    return "\n".join(lines)


async def search_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Usage: /search <what you're looking for>")
        return
    hits = await asyncio.to_thread(search, update.effective_user.id, query, 5)
    if not hits:
        await update.message.reply_text("No notes yet. Send me a voice note!")
        return
    lines = ["\U0001F50E Closest notes:"]
    for h in hits:
        lines.append(f"#{h['id']} ({h['score']:.2f}) — {h['summary']}")
    await update.message.reply_text("\n".join(lines))


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = " ".join(context.args) if context.args else ""
    if not question:
        await update.message.reply_text("Usage: /ask <question about your notes>")
        return
    status = await update.message.reply_text("\U0001F4DA Reading my memory...")
    answer = await asyncio.to_thread(ask, update.effective_user.id, question)
    await status.edit_text(f"\U0001F4A1 {answer}")

    # Speak the answer back (local TTS). Text stays even if TTS fails.
    try:
        lang = detect_lang(answer)
        ogg = await asyncio.to_thread(synthesize, answer, lang)
        if ogg:
            with ogg.open("rb") as f:
                await update.message.reply_voice(voice=f)
            ogg.unlink(missing_ok=True)
    except Exception:
        log.exception("TTS failed; text answer already delivered")


async def recent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    notes = recent_notes(update.effective_user.id, 5)
    if not notes:
        await update.message.reply_text("No notes yet. Send me a voice note!")
        return
    blocks = []
    for n in notes:
        topics = ", ".join(json.loads(n["topics"] or "[]"))
        blocks.append(f"#{n['id']} — {n['summary']}" + (f"  [{topics}]" if topics else ""))
    await update.message.reply_text("\U0001F5C3 Recent notes:\n" + "\n".join(blocks))


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args and context.args[0].lower() == "confirm":
        n = delete_all_notes(update.effective_user.id)
        await update.message.reply_text(f"\U0001F9F9 Wiped {n} note(s). Fresh brain.")
    else:
        await update.message.reply_text(
            "⚠️ This deletes ALL your notes, permanently.\n"
            "If you're sure: /reset confirm"
        )


async def delete_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        note_id = int(context.args[0])
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /delete <note id>")
        return
    if delete_note(update.effective_user.id, note_id):
        await update.message.reply_text(f"\U0001F5D1 Note #{note_id} deleted.")
    else:
        await update.message.reply_text(f"No note #{note_id} in your memory.")


# ---------- app ----------

async def _post_init(app: Application) -> None:
    # Keep a reference in bot_data so the worker task isn't garbage-collected.
    app.bot_data["worker"] = asyncio.create_task(_worker())


def main() -> None:
    app = Application.builder().token(TOKEN).post_init(_post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("recent", recent))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("delete", delete_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    log.info("VoiceBrain is polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
