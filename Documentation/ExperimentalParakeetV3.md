# Experimental local Parakeet v3

The separate v3 runtime loads locally managed, checksum-pinned CoreML packages. It supports TDT text across the checkpoint's 25 languages. The existing 110M descriptors, download manifests and runtime remain available unchanged.

```swift
let recognizer = try SpeechRecognizer(
    directory: modelDirectory,
    model: .experimentalParakeetV3,
    computeUnits: .neuralEngine
)
let transcript = try await recognizer.transcribe(file: audioURL)
```

Load the recognizer off the main thread. Calls on a recognizer are serialized. Cancellation is observed between synchronous model predictions, between decoder steps and at window boundaries; an active CoreML prediction is allowed to finish before cancellation returns.

The experimental descriptor is local-only and is excluded from `AsrModel.all`. It has no published repository/revision, and download APIs reject it before creating storage. `maxRequestDuration` is nil: offline files and PCM can exceed 15 seconds and are processed as bounded windows. It advertises only `.tdt`: CTC, keyword spotting and timestamp capabilities are unavailable. `ctcTranscript` is nil and `tokens` is empty. Decoder emission frames are used internally for experimental long-window ownership and are not presented as aligned word times.

A local export folder contains `Preprocessor.mlpackage`, `Encoder.mlpackage`, `Predictor.mlpackage`, `Joint.mlpackage`, `vocab.json` and `config.json`. The config declares the v3 architecture, precision, fixed 240,000-sample window, 8,192-piece vocabulary, blank ID 8,192, duration bins `[0,1,2,3,4]`, maximum ten symbols per step, punctuation cleanup and SHA-256 checksums for all model/tokenizer files. All checksums and tensor contracts are validated before inference. Model packages compile into the caller's cache, keyed by package content and OS build.

The baseline keeps its frontend and model interfaces in FP32. Separate FP32 and unquantized FP16 encoder/predictor/joint exports are validated against the pinned source. Precision is explicit in the export manifest; an FP16 baseline is not a quantized release.

Each request/window starts with owned zero LSTM state. Model-load temporaries are drained before inference, and prediction temporaries are released at each bounded window rather than after the whole file. Predictor outputs on blanks are cached without committing candidate state. State/projection outputs are copied from their actual strides into owned buffers before another prediction can reuse an output. The joint returns all 8,193 token logits and five duration logits; the host preserves first-maximum argmax behavior and the source maximum-symbol guard.

Legacy single-window exports use up to 240,001 buffered input samples. The release candidate also supports shared-weight `w15` and `w30` functions: requests up to 15 seconds use `w15`; longer recordings use `w30`, with at most 480,001 buffered samples and one resident encoder function. It prefers the middle of a quiet run of at least 160 ms between 8 and 14 seconds. Quiet means a 10 ms RMS below the greater of 0.0005 and 2% of the window peak RMS. These cuts partition the input without dropping audio. If there is no eligible pause, overlapping frame-aligned windows supply context. A join prefers matching token sequences in the overlap and falls back to emission-frame ownership. An entirely blank speech-bearing long window is retried with at most two subdivision levels; nonempty primary results are preserved. Set the low-level initializer's `allowExperimentalLongAudio: false` for an explicit single-window budget.

The earlier 15-second-window validation scored 2.32% WER on 20 known long composites; final release qualification is being recorded separately. All 20 file and PCM paths have identical text, token IDs and internal frames. Fresh multilingual long composites score 13.74%, versus 13.66% for native whole-context NVIDIA FP32 on the same audio. These are limited development/confirmatory results, not an assertion of field-conversation robustness or streaming support. Silence, continuous speech and timestamp quality remain distinct concerns; timestamps are still not exposed.

The Mac compatibility target is all Apple Silicon Macs, from M1 onward, running macOS 15 or later. M5 is the available Mac validation machine. Access to an M1 is not a release prerequisite; performance and memory measurements on other chips remain additional hardware coverage. The Swift package targets generic arm64/macOS 15, and the selected FP16 models are portable `.mlpackage` exports, compiled locally for each machine and OS build. The runtime contains no M5-specific code path. It requests CPU plus Neural Engine compute; the SDK also exposes CPU-only operation.

M5 accuracy/long-file validation and physical iPhone 13 validation have completed. The iPhone passed a 20-minute paced dictation workload and a separate 45-file long check. Five latency outliers remain under investigation. Natural long-form multilingual coverage, controlled cold preparation, energy and total accelerator memory remain open qualification work. No converted model is automatically downloaded or published by this change.
