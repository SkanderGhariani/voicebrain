"""Quick local test for extract.py — run: python test_extract.py"""

import json
import logging

logging.basicConfig(level=logging.WARNING)

from extract import extract

SAMPLES = [
    ("en", "Remind me to call the dentist on Tuesday and ask Sami about the invoice before Friday."),
    ("fr", "Il faut que je pense à acheter un cadeau pour l'anniversaire de Yasmine samedi, et appeler maman ce soir."),
]

for lang, transcript in SAMPLES:
    print(f"\n=== {lang} ===\n{transcript}")
    data = extract(transcript, lang)
    print(json.dumps(data, indent=2, ensure_ascii=False))
