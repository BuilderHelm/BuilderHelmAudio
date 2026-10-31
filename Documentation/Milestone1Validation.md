# Milestone 1: runtime architecture and CI

31 October 2026. Implementation is on the local `feature/multiple-runtimes-ci` branch. No release or model revision was changed.

## Result

The recognizer dispatches through an internal runtime interface and factory. Model descriptors now carry architecture, language, capability and optional CTC-layout metadata. The 110M implementation remains the only production runtime adapter; the interface can accept a separate v3 adapter in the next milestone without coupling the recognizer to it.

The existing 110M decoder, encoder loading, audio processing and long-recording algorithms in `Sources/ParakeetRuntime` have no source changes. Download pins/hashes and the privacy manifest are unchanged. Existing load/transcribe/spot examples, String CTC access, default CTC layout and CTC-only mode fallback are preserved. Optional CTC output and optional keyword-spotter access make absent capabilities explicit for future adapters.

## Local environment

- Apple M5 Mac, macOS 27.2.
- Xcode 27.1 (27A9275), Swift 6.4; `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- iOS 27.0 Simulator, using the installed iPhone 17 Pro.
- Baseline SDK commit: `c71582b3fe6d3ea2c9bd3b4f928f05167c662860` (published 0.1.1).
- Model packages remain pinned at TDT+CTC `85a37ceaebc43645bc3ad4fd9594cdab0f38e745` and CTC `cf9ce52bbc83def86524cbf2af1e3756f0b68133`.

## Checks

| Check | Local result |
|---|---|
| macOS release build | Passed |
| Public API examples | All eight README Swift usage examples and both architecture examples typechecked as external clients |
| Default macOS tests | Passed; real-model suite and corpus snapshots are opt-in |
| iOS Simulator package build/tests | Passed using `Scripts/ci-ios.sh` |
| Real-model integration on CPU | Passed: both packages, short/sample paths, long-audio pipeline, adapter parity and repeated requests |
| Real-model integration on Neural Engine | Passed with the same integration suite |
| Before/after frozen-corpus regression | 568 of 568 output records exactly equal |
| 110M decoder source diff | No changes |
| Privacy plist | `plutil -lint` passed |
| Workflow and shell validation | `actionlint` 1.7.12 and `bash -n` passed |

### Frozen-corpus evidence

Before modifying the SDK implementation, `runtimeRegressionSnapshot` captured the existing API on 262 Common Voice clips, 20 Long20 recordings and two bundled fixtures. Both shipped models were run in their default modes: 284 files per model, 568 output records total. The candidate was then compared on the same Mac/Neural Engine path.

- Audio-content hashes and exact hashes of primary text, CTC text and token text/times matched for every record.
- All four bundled-fixture probability snapshots and keyword snapshots matched exactly, including scores and hit start/end times.
- The snapshot file is byte-identical before/after, SHA-256 `d8799999c9ce65ace3240ed6632042118321b595081c5491c3e3c72a7de6b1be`.
- Snapshots are local `/tmp/bhaudio-runtime-before.json` and `/tmp/bhaudio-runtime-after.json`; neither contains audio or corpus transcript text. `/tmp` artifacts are temporary and may be cleared by the system.
- Corpus manifests came from the existing R&D `results/110m/final-comparison-20261030/{cv-cases,long20-cases}.json`. This is regression evidence on existing evaluation audio, not a new accuracy benchmark or independent held-out result.

The reproducible real-model integration suite additionally compares the adapter directly with `ParakeetTdtCtc110M` using the short and long bundled fixtures, then the short fixture again. It covers the full package's TDT and CTC modes and the standalone CTC package. Text, token text/times, full CTC probabilities and keyword scores/times must be exactly equal on the selected compute path.

### Boundary and cancellation coverage

Unit tests cover nonfinal blank IDs, non-110M CTC dimensions/frame durations, invalid probability layouts, unavailable CTC/keyword operations, unsupported architecture rejection before downloads or filesystem access, cancelled queued work, cancellation during synchronous work and subsequent queue reuse, pre-cancelled storage requests, and active URLSession cancellation followed by downloader reuse. These tests do not construct an unimplemented model or claim another architecture has been converted.

## CI configuration and remaining gate

The automatic workflow defines macOS release builds/unit tests with Xcode 16.0 and 16.4, and an iOS Simulator job with Xcode 16.4. The manual integration workflow uses CPU inference, pinned model downloads and a model-only cache. Actions are pinned to commit hashes.

Hosted GitHub Actions have not run: the local branch has not been pushed. The local machine has Xcode 27.1, so the configured hosted Xcode 16.0/16.4 matrix is still an explicit verification gate. Local success is not a claim of hosted CI success. No new iPhone device-performance claim, v3 support or live transcription is included in this milestone.
