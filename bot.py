"""VoiceBrain — Telegram bot entry point.

Phase 2: the bot transcribes voice notes locally (faster-whisper) and
replies with the text + detected language. Extraction comes in Phase 3.
"""

import asyncio
import logging
import os
from pathlib import Path

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
    # Whisper is CPU-blocking; run it in a worker thread so the bot stays responsive.
    text, lang = await asyncio.to_thread(transcribe, dest)
    dest.unlink(missing_ok=True)  # audio no longer needed once we have the text

    if not text:
        await status.edit_text("\U0001F92B I couldn't hear anything in that note.")
        return

    await status.edit_text(
        f"\U0001F5E3 Detected language: {lang}\n\n"
        f"“{text}”\n\n"
        f"(Structured extraction coming in Phase 3.)"
    )


def main() -> None:
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    log.info("VoiceBrain is polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
