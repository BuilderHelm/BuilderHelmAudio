# Local Parakeet Unified EN exports (in development)

Parakeet Unified EN 0.6B is NVIDIA's English transducer trained for both offline and streaming recognition with one set of weights. This SDK runs its **offline mode** through the same runtime as V3: staged FP32 frontend, shared-weight 15/30-second encoder functions, quiet-pause long-audio cuts, overlap token consensus and bounded blank recovery. Decoding is greedy RNN-T (at most 10 symbols per frame) instead of TDT. There is no published descriptor yet; locally built folders use `.experimentalParakeetUnifiedEn`:

```swift
let recognizer = try SpeechRecognizer(
    directory: modelDirectory,
    model: .experimentalParakeetUnifiedEn,
    compiledModelsDirectory: cacheDirectory
)
try await recognizer.prepare()   // compiles both windows on the Neural Engine, about 10 s each the first time
let transcript = try await recognizer.transcribe(file: audioURL)
print(transcript.text, transcript.words)
```

```sh
swift run -c release bhaudio transcribe speech.wav --model parakeet-unified-en-0.6b-local --model-dir /path/to/release --json
```

Construct local recognizers off the main thread. Like `.experimentalParakeetV3`, the descriptor has no repository or revision, is excluded from `AsrModel.all` (it is listed in `AsrModel.local`), and `ModelStore` cannot download, verify or remove its folder.

## Capabilities

`[.rnnt, .tokenTimestamps, .wordTimestamps]`, selected with the default `DecodingMode.tdt` (the transducer mode). `decodesTdt` is false. Text includes punctuation and capitalization. Word timestamps follow NeMo's RNN-T timestamps exactly: each token spans its emission frame, a word starts at its first token and ends one frame after its last, and punctuation is not moved (NeMo refines punctuation only for TDT). Confidence is NeMo `max_prob` over the 1024 tokens and blank, with the `min` word aggregate.

CTC, keyword spotting, ranked alternatives, segment timestamps and streaming are unavailable. Ranked alternatives remain a TDT beam search; the RNN-T beam did not improve held-out accuracy.

## Folders

Build with [`Scripts/ParakeetUnified/build_release.py`](../Scripts/ParakeetUnified/README.md). The release root is the FP16 model (`Encoder` and `EncoderTail` programs). `variants/int8/` holds the INT8-weight encoder and a config whose checksums cover the root's shared frontend, predictor, joint and vocabulary; to load it locally, place those shared files beside the variant's `Encoder.mlpackage` and `config.json`. The runtime checks the config's decoder contract (`architecture: parakeet-unified-en-0.6b`, `decoder: rnnt`, blank 1024, 1024 pieces, no durations) and every checksum before inference, and the descriptor and folder must agree: the V3 descriptor refuses a Unified folder and the reverse.

Local exports require independent parity and accuracy qualification before deployment.
