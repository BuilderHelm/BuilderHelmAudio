# Runtime architecture (unreleased)

The public `SpeechRecognizer` owns an internal `AsrRuntime` implementation. Model metadata selects the implementation; decoder-specific result conversion belongs to that implementation.

The first adapter wraps the `ParakeetTdtCtc110M` library. Both shipped packages use the `parakeet110M` architecture with different capabilities. This work prepares the SDK for other implementations; it does not add Parakeet v3 or advertise another supported model.

The pinned 110M compatibility exports store the 20-second preprocessor's sample indices as constants. Encoder/decoder weights, vocabulary, normalization and window policies retain the v1.1.0 behavior. [The export patch script](../Scripts/fold-preprocessor-indices.py) reproduces the specification change from the original pinned preprocessor. Runtime tensor copies honor CoreML's actual strides, including padded output rows. Frozen mel-feature reference points cover every channel throughout a 20-second speech window.

## Model metadata and dispatch

`AsrModel` exposes:

| Property | Meaning |
|---|---|
| `architecture` | Stable runtime-family identifier, separate from the download ID |
| `supportedLanguages` | Advertised language codes; not a detected-language result |
| `capabilities` | TDT, CTC, token/word/segment timestamps and keyword support, independently represented |
| `ctcConfiguration` | CTC class count, blank class and frame duration, or nil |
| `decodesTdt` | Existing compatibility accessor, derived from capabilities |

`RuntimeSelection.resolve` validates the descriptor before downloads or filesystem/model loading. `RuntimeFactory` dispatches only to implemented adapters. Unknown architectures throw `unsupportedArchitecture`; incompatible metadata throws `invalidModelConfiguration`.

The 110M adapter requires its actual vocabulary/layout and English capability set. The existing CTC-only package still automatically selects `.ctc`, including when the caller leaves the default `.tdt` argument unchanged. A full package in `.ctc` mode reports only its effective CTC/keyword capabilities.

Bundled directories without an `AsrModel` retain the old 110M loading convention. Future architectures need explicit model metadata.

## Results and unavailable operations

Existing `load`, `transcribe`, `spot`, `ctcText` and `keywordSpotter` call sites still compile.

- `Transcript.ctcTranscript` is optional: nil means no CTC output, while `""` can mean a CTC head recognized silence.
- Existing `Transcript.ctcText` remains a String accessor and returns `""` for absent CTC. New architecture-aware callers should use the optional accessor.
- `SpeechRecognizer.capabilities` describes the selected runtime/mode. Unsupported CTC-probability requests and `spot` calls throw `unsupportedOperation` before inference.
- `keywordSpotterIfAvailable` is nil when the runtime lacks a compatible spotter. The legacy nonoptional `keywordSpotter` remains available; an unavailable instance has `isAvailable == false` and returns no hits.
- The nonthrowing spotter APIs return nil/no hits for incompatible CTC layouts. Runtime-backed `spot` throws if a runtime claims keyword support but omits requested probabilities.

```swift
let recognizer = try await SpeechRecognizer.load(.parakeetTdtCtc110M)
if recognizer.capabilities.contains(.ctc) {
    let transcript = try await recognizer.transcribe(file: audioURL, includeCtcLogProbabilities: true)
    if let probabilities = transcript.ctcLogProbabilities,
       let spotter = recognizer.keywordSpotterIfAvailable {
        print(spotter.detect(["Neural Engine"], in: probabilities))
    }
}
```

## CTC layouts

`CtcLogProbabilities(values:)` and the static `classCount`, `blank` and `frameDuration` constants retain the 110M defaults for source compatibility. Other layouts use the throwing initializer and instance configuration:

```swift
let configuration = try CtcConfiguration(classCount: classCount, blank: blankID, frameDuration: secondsPerFrame)
let probabilities = try CtcLogProbabilities(values: flatValues, configuration: configuration)
let spotter = try KeywordSpotter(vocabulary: pieces, configuration: configuration)
```

The vocabulary contains non-blank pieces in class order. The configurable spotter maps IDs around the blank even when blank is not the last class. It uses the instance's frame duration for hit timing and checks layout compatibility before indexing probabilities. Constructors reject invalid dimensions, blank IDs, frame durations, partial frames and mismatched vocabulary counts. These generic layout tests do not establish support for a new real model.

## Lifecycle and cancellation

One recognizer serializes all inference requests on its own queue. Adapters must not share mutable scratch state across recognizers and must reset per-request decoding state or keep it request-local.

Cancelling an async load or transcription returns `CancellationError` when cancellation is observed. Queued cancelled work does not begin. If synchronous CoreML preparation or prediction is already running, it may finish before the task returns cancellation; the queue is not released for another request until that work finishes. There is no promise of immediate preemption, frame-level interruption or a live-streaming reset API in this milestone.

Model downloads check cancellation before storage work, between files and before writing the verified-ready marker. Cancellation while awaiting a URLSession download cancels the underlying download task, cleans temporary files, and does not retry the cancellation. Verification/hashing may finish its current synchronous operation before observing cancellation. Previously verified files remain available for file-level recovery on the next request.

## Adding a runtime

1. Implement `AsrRuntime` in a separate adapter file, returning SDK-level `RuntimeResult` values. Map model-specific token IDs and times inside the adapter.
2. Add the architecture identifier and descriptor with its actual languages, capabilities, optional CTC layout and pinned file manifest.
3. Extend `RuntimeSelection.resolve` with that family's validation and mode-selection rules and add factory dispatch. Do not pass another model's dimensions to the 110M implementation.
4. Return nil for absent CTC output and no spotter when keyword support is unavailable. Implement the cancellation/state-isolation contract.
5. Add real-model parity, capability, repeated-request and lifecycle tests before advertising support.

## CI and local checks

The default [CI workflow](../.github/workflows/ci.yml) builds and unit-tests on macOS with Xcode 16.0 and 16.4, and runs iOS Simulator tests with Xcode 16.4. It downloads no model weights. `Scripts/ci-ios.sh` selects an installed iPhone runtime compatible with the selected simulator SDK.

The separate [Real-model integration workflow](../.github/workflows/model-integration.yml) runs for pull requests that change runtime sources, tests, the package manifest or the integration workflow. It can also be started manually (`workflow_dispatch`). It caches only pinned downloaded models, rehashes those files, and runs the serialized real-model suite in release mode on CPU. It does not claim Neural Engine or phone performance.

```bash
swift test
bash Scripts/ci-ios.sh
BUILDERHELM_INTEGRATION=1 swift test -c release --filter RealModelTests
BUILDERHELM_INTEGRATION=1 BUILDERHELM_TEST_COMPUTE=cpu swift test -c release --filter RealModelTests
```

Use a full Xcode installation for CoreML compilation and simulator testing; set `DEVELOPER_DIR` if `xcode-select` points to Command Line Tools.

The optional `runtimeRegressionSnapshot` test accepts `BUILDERHELM_REGRESSION_OUT`, optional `BUILDERHELM_REGRESSION_BASELINE`, and colon-separated `BUILDERHELM_REGRESSION_CASES` manifests (`{"files":[{"path":"..."}]}`). It hashes input audio, text/token-time output, and fixture CTC probabilities/keyword results. Snapshots remain external to the repository; no corpus audio or transcripts are committed. Capture the baseline before a code change, then compare the candidate on the same hardware/compute path. The test is skipped unless an output path is provided.
