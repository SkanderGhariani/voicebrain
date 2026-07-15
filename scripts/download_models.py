"""Download models for a profile: python scripts/download_models.py [lite|quality]"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICES = ROOT / "models" / "voices"

PROFILES = {
    "lite": {
        "whisper": "small",
        "gguf": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "gguf_repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
    },
    "quality": {
        "whisper": "large-v3-turbo",
        "gguf": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "gguf_repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
    },
}

PIPER_VOICES = [
    ("en/en_US/lessac/medium", "en_US-lessac-medium"),
    ("fr/fr_FR/siwis/medium", "fr_FR-siwis-medium"),
]


def main() -> None:
    profile = PROFILES[sys.argv[1] if len(sys.argv) > 1 else "lite"]

    from huggingface_hub import hf_hub_download

    print(f"1/4 LLM: {profile['gguf']} ...")
    hf_hub_download(
        repo_id=profile["gguf_repo"],
        filename=profile["gguf"],
        local_dir=ROOT / "models",
    )

    print(f"2/4 Whisper: {profile['whisper']} ...")
    from faster_whisper import WhisperModel

    WhisperModel(profile["whisper"], device="cpu", compute_type="int8")

    print("3/4 Embedding + reranker models ...")
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu")

    print("4/4 Piper voices ...")
    import shutil

    VOICES.mkdir(parents=True, exist_ok=True)
    for hf_path, name in PIPER_VOICES:
        for ext in (".onnx", ".onnx.json"):
            downloaded = hf_hub_download(
                repo_id="rhasspy/piper-voices",
                filename=f"{hf_path}/{name}{ext}",
                local_dir=VOICES,
            )
            flat = VOICES / f"{name}{ext}"  # tts.py expects flat filenames
            if Path(downloaded) != flat:
                shutil.copy(downloaded, flat)

    print("Done. Set WHISPER_MODEL / LLM_MODEL_PATH in .env to match the profile.")


if __name__ == "__main__":
    main()
