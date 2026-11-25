"""Fetch the official source snapshot used for the BuilderHelm Core ML export."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "bodhan-ai/indic-transcribe-flex"
REVISION = "ecba9bac9013c0340fe814b336897ee51ed80be9"
SOURCE_FILES = [
    "*.json",
    "*.py",
    "*.safetensors",
    "*.model",
    "*LICENSE*",
    "Bodhan_AI_Open_Model_License.md",
    "indic-open-license.md",
    "requirements.txt",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Read an authorized Hugging Face token from this file, or use normal HF authentication",
    )
    args = parser.parse_args()
    token = args.token_file.read_text().strip() if args.token_file else None
    snapshot = snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=SOURCE_FILES,
        ignore_patterns=["nemo/*"],
        token=token,
    )
    print(snapshot)


if __name__ == "__main__":
    main()
