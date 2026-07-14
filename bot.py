"""VoiceBrain — Telegram bot entry point.

Phase 1: prove the Telegram loop. The bot answers /start, echoes text,
and downloads voice notes to a local scratch folder, replying with the
file's metadata. Transcription and extraction come in later phases.
"""

import logging
import os
from pathlib import Path

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

    size_kb = dest.stat().st_size / 1024
    log.info("Voice note saved: %s (%.1f KB, %ss)", dest.name, size_kb, voice.duration)
    await update.message.reply_text(
        f"\U0001F3A4 Got your voice note!\n"
        f"Duration: {voice.duration}s\n"
        f"Size: {size_kb:.1f} KB\n"
        f"Saved as: {dest.name}\n\n"
        f"Transcription coming in Phase 2."
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
