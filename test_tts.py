"""Quick local test for tts.py — run: python test_tts.py"""

import logging

logging.basicConfig(level=logging.INFO)

from tts import available, detect_lang, synthesize

print("piper available:", available())

for text in [
    "Hello Skander, your voice brain is now able to speak.",
    "Salut Skander, ton deuxième cerveau sait parler maintenant.",
]:
    lang = detect_lang(text)
    ogg = synthesize(text, lang)
    print(f"[{lang}] -> {ogg} ({ogg.stat().st_size // 1024} KB)" if ogg else "FAILED")
