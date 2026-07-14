"""VoiceBrain — Telegram bot entry point.

Phase 3: voice note -> local transcription (faster-whisper) -> structured
extraction with a local LLM (llama.cpp, schema-constrained JSON) -> SQLite.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from extract import extract
from memory import ask, search, store_embedding
from storage import recent_notes, save_note
from transcribe import transcribe

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

AUDIO_DIR = Path("audio_tmp")
AUDIO_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
# The HTTP library is chatty at INFO; keep the log readable.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("voicebrain")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi, I'm VoiceBrain \U0001F9E0\n\n"
        "Send me a voice note in any language and I'll turn it into "
        "structured, searchable memory.\n\n"
        "(Phase 1: I can only receive audio so far — brain installation in progress.)"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Echo: {update.message.text}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    voice = update.message.voice
    tg_file = await voice.get_file()
    dest = AUDIO_DIR / f"{tg_file.file_unique_id}.ogg"
    await tg_file.download_to_drive(custom_path=dest)
    log.info("Voice note saved: %s (%ss)", dest.name, voice.duration)

    status = await update.message.reply_text("\U0001F442 Transcribing locally...")
    # Whisper and the LLM are CPU-blocking; run them in a worker thread
    # so the bot stays responsive to other messages.
    text, lang = await asyncio.to_thread(transcribe, dest)
    dest.unlink(missing_ok=True)  # audio no longer needed once we have the text

    if not text:
        await status.edit_text("\U0001F92B I couldn't hear anything in that note.")
        return

    await status.edit_text("\U0001F9E0 Extracting structure...")
    data = await asyncio.to_thread(extract, text, lang)
    note_id = save_note(text, lang, data)
    await asyncio.to_thread(store_embedding, note_id, text, data["summary"])

    await status.edit_text(_format_note(note_id, lang, text, data))


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
    hits = await asyncio.to_thread(search, query, 5)
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
    answer = await asyncio.to_thread(ask, question)
    await status.edit_text(f"\U0001F4A1 {answer}")


async def recent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    notes = recent_notes(5)
    if not notes:
        await update.message.reply_text("No notes yet. Send me a voice note!")
        return
    blocks = []
    for n in notes:
        topics = ", ".join(json.loads(n["topics"] or "[]"))
        blocks.append(f"#{n['id']} — {n['summary']}" + (f"  [{topics}]" if topics else ""))
    await update.message.reply_text("\U0001F5C3 Recent notes:\n" + "\n".join(blocks))


def main() -> None:
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("recent", recent))
    app.add_handler(CommandHandler("search", search_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    log.info("VoiceBrain is polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
