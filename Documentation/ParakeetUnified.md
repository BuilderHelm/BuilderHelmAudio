# Parakeet Unified EN

Parakeet Unified EN 0.6B is NVIDIA's English transducer trained for both offline and streaming recognition with one set of weights. The SDK runs its **offline mode** through the same runtime as V3: staged FP32 frontend, shared-weight encoder windows, quiet-pause long-audio cuts, overlap token consensus and bounded blank recovery. Decoding is greedy RNN-T (at most 10 symbols per frame), with punctuation and capitalization in the text.

| Descriptor | Encoder | Download |
|---|---|---:|
| `.parakeetUnifiedEnInt8` | Weight-only symmetric INT8 per output channel, FP16 compute | 0.71 GB |
| `.parakeetUnifiedEn` | Unquantized FP16, two sequential programs | 1.34 GB |

Both pin [BuilderHelm/Parakeet-Unified-EN-0.6B-CoreML](https://huggingface.co/BuilderHelm/Parakeet-Unified-EN-0.6B-CoreML) 1.0.0 (revision `97e349c3`). The INT8 descriptor is a variant: it downloads the variant's encoder and config plus the shared frontend, predictor, joint and vocabulary into its own folder (`Models/<repository>/Variants/<revision>/int8/`), so removing one download never touches the other.

```swift
let recognizer = try await SpeechRecognizer.load(.parakeetUnifiedEnInt8)
try await recognizer.prepare()   // compiles every window on the Neural Engine, once per macOS build
let transcript = try await recognizer.transcribe(file: audioURL)
print(transcript.text, transcript.words)
```

## Capabilities

`[.rnnt, .tokenTimestamps, .wordTimestamps]`, selected with the default `DecodingMode.tdt` (the transducer mode). `decodesTdt` is false. Word timestamps follow NeMo's RNN-T timestamps exactly: each token spans its emission frame, a word starts at its first token and ends one frame after its last, and punctuation is not moved (NeMo refines punctuation only for TDT). Confidence is NeMo `max_prob` over the 1024 tokens and blank, with the `min` word aggregate. CTC, keyword spotting, ranked alternatives, segment timestamps and streaming are unavailable.

## Speed

Each request uses the smallest window that holds it: 5, 10, 15, 20 or 30 s. The windows share weights, so they add about 3% to the download. INT8 encoder time on the M5 Neural Engine: 9.5, 14.3, 26.4, 34.6 and 82.4 ms. On macOS every window stays loaded after first use (`keepsAllWindows`), because reloading a window function cost about 100 ms whenever consecutive requests needed different windows; iOS keeps only the latest window to bound memory. The folder's `RnntDecoder.bin` holds the original FP32 predictor and joint, which the runtime runs on the CPU with Accelerate.

The first load after installation or a macOS update compiles every window, about 45 s for INT8 on M5. Later loads take about 1.5 s.

## Validation (M5, Neural Engine)

Against the original NeMo model, greedy, WER with Whisper's English normalizer:

| Set | NeMo | FP16 | INT8 |
|---|---:|---:|---:|
| 1,304 clips ≤ 15 s | 7.28% | 7.31% | 7.27% |
| 937 held-out clips | 10.70% | 10.69% | 10.72% |
| 393 clips of 15–30 s | 2.02% | 2.02% | 2.02% |
| 97 recordings of 73–322 s | 1.61% (whole recording) | 1.53% | 1.54% |

ZenVoice Speech Bench v1 (1,382 clips), through ZenVoice's engine path: INT8 8.0% WER, 219× real time, 0.03 s typical delay; FP16 8.0%, 184×, 0.04 s; the GGUF Q8_0 build on transcribe.cpp 8.1%, 104×, 0.07 s. With the GPU busy (13.4 TFLOPS load), typical delay was 0.037 s (INT8) against 0.465 s (GGUF). Swift output is token-identical to a Python reference running the same release files.

## Local folders

`.experimentalParakeetUnifiedEn` loads a folder you build yourself with [`Scripts/ParakeetUnified/build_release.py`](../Scripts/ParakeetUnified/README.md), through `SpeechRecognizer(directory:model:)` or `bhaudio --model-dir <folder> --model parakeet-unified-en-0.6b-local`. Like other local descriptors it has no repository, is listed in `AsrModel.local` rather than `AsrModel.all`, and `ModelStore` cannot download or remove its folder. To load the INT8 variant locally, place the release root's shared files beside `variants/int8/Encoder.mlpackage` and `config.json`. The runtime checks the decoder contract (`architecture: parakeet-unified-en-0.6b`, `decoder: rnnt`, blank 1024, 1024 pieces, no durations) and every checksum before inference; the V3 descriptor refuses a Unified folder and the reverse. Local exports require independent parity and accuracy qualification before deployment.
