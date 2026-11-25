"""Check a local Core ML transcription against Bodhan on the same recording."""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch


def model(directory, name, units):
    return ct.models.MLModel(str(directory / f"{name}.mlpackage"), compute_units=units)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("packages", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--mode", choices=("native", "mixed", "romanized"), required=True)
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.snapshot.resolve()))
    from indic_transcribe import IndicTranscribe, RECOMMENDED_LANGS
    from lid import language_token_map, lid_prefix_ids
    from tokenization_indic_canary import IndicCanaryTokenizer

    torch.set_num_threads(4)
    units = ct.ComputeUnit.CPU_AND_GPU
    reference = IndicTranscribe.from_pretrained(str(args.snapshot), device="cpu")
    tokenizer = IndicCanaryTokenizer.from_pretrained(str(args.snapshot))
    waveform = reference._load(str(args.audio))
    if not 0 < waveform.numel() <= 480000:
        raise ValueError("Supply an audio clip between zero and 30 seconds")
    if waveform.numel() < 16000:
        missing = 16000 - waveform.numel()
        left = round(missing / 2)
        waveform = torch.nn.functional.pad(waveform, (left, missing - left))

    with torch.inference_mode():
        reference_features, mask = reference._features(waveform)
        reference_encoder = reference.model.get_encoder()(
            reference_features, attention_mask=mask,
        ).last_hidden_state.numpy()
    frontend = model(args.packages, "frontend", ct.ComputeUnit.CPU_ONLY)
    features = frontend.predict({"audio": waveform[None, :].numpy()})["features"]
    np.testing.assert_allclose(features, reference_features.numpy(), atol=1e-3, rtol=3e-4)
    encoder = model(args.packages, "encoder", units)
    states = encoder.predict({"features": features})["encoder_states"]
    if states.shape != reference_encoder.shape or not np.isfinite(states).all():
        raise RuntimeError("Unexpected encoder output shape or nonfinite activations")
    print(f"Encoder max absolute delta: {np.abs(states - reference_encoder).max():.6f}", flush=True)

    decoder = model(args.packages, "decoder_cached" if args.cached else "decoder", units)

    def next_logits(ids, state=None, offset=0):
        positions = np.arange(offset, offset + len(ids), dtype=np.int32)[None, :]
        available = np.arange(544, dtype=np.int32) if state is not None else positions
        mask = (positions[:, :, None] < available).astype(np.float32)
        inputs = {
            "ids": np.array([ids], dtype=np.int32),
            "encoder_states": states,
            "positions": positions,
            "mask": mask[:, None, :, :] * -10000.0,
        }
        return decoder.predict(inputs, state=state)["logits"][0, -1].copy()

    def prefill(ids, state):
        if state is None:
            return next_logits(ids)
        for position, token in enumerate(ids):
            logits = next_logits([token], state, position)
        return logits

    language = args.language
    if language == "auto":
        expected, lid = reference.transcribe(
            str(args.audio), mode=args.mode, return_lid=True, do_sample=False,
        )
        logits = prefill(lid_prefix_ids(tokenizer), decoder.make_state() if args.cached else None)
        candidates = language_token_map(tokenizer, RECOMMENDED_LANGS)
        language = candidates[max(candidates, key=lambda token: logits[token])]
        print(f"Detected language: Bodhan={lid['lang']}, Core ML={language}", flush=True)
        if language != lid["lang"]:
            raise SystemExit("Detected languages differ")
    else:
        expected = reference(str(args.audio), lang=language, mode=args.mode, do_sample=False)

    prompt = tokenizer.encode_prompt(
        language, itn=args.mode == "mixed", romanized=args.mode == "romanized",
    )
    ids = list(prompt)
    generation = reference.model.generation_config
    eos = np.atleast_1d(generation.eos_token_id)
    limit = 512
    state = decoder.make_state() if args.cached else None
    for step in range(limit):
        if state is not None and step > 0:
            logits = next_logits(ids[-1:], state, len(ids) - 1)
        else:
            logits = prefill(ids, state)
        for token in generation.suppress_tokens or []:
            logits[token] = -np.inf
        if len(ids) == len(prompt):
            for token in generation.begin_suppress_tokens or []:
                logits[token] = -np.inf
        token = int(logits.argmax())
        if token in eos:
            break
        ids.append(token)
    else:
        raise RuntimeError(f"Decoder reached its {limit}-token limit before ending the transcript")

    actual = tokenizer.decode(tokenizer.strip_prompt_and_trim(ids, prompt))
    print(f"Bodhan: {expected}\nCore ML: {actual}")
    if actual != expected:
        raise SystemExit("Transcripts differ")


if __name__ == "__main__":
    main()
