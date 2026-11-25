"""Export Swift-readable vocabulary and prompts from the official tokenizer."""

import argparse
import json
import sys
from pathlib import Path

from download import REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.snapshot.resolve()))
    from tokenization_indic_canary import IndicCanaryTokenizer, PROMPT_LANGS

    tokenizer = IndicCanaryTokenizer.from_pretrained(str(args.snapshot))
    prompts = {
        language: {
            "native": tokenizer.encode_prompt(language),
            "mixed": tokenizer.encode_prompt(language, itn=True),
            "romanized": tokenizer.encode_prompt(language, romanized=True),
        }
        for language in PROMPT_LANGS
    }
    config = json.loads((args.snapshot / "generation_config.json").read_text())
    payload = {
        "sourceRevision": REVISION,
        "pieces": tokenizer.ids_to_pieces(range(tokenizer.vocab_size)),
        "prompts": prompts,
        "endTokenIDs": config["eos_token_id"],
        "languageDetectionPrefix": prompts["hi"]["native"][:3],
        "languageTokenIDs": {lang: modes["native"][3] for lang, modes in prompts.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
