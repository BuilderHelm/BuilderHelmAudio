# Migrating from 0.1.1 to 0.2.0

Existing `SpeechRecognizer.load`, file/PCM transcription, keyword spotting and 110M model identifiers remain available. The 110M model pins and decoding behavior are retained. Version 0.2.0 adds the V3 model and capability-aware APIs.

## Error switches

Four cases were added to `BuilderHelmAudioError`:

- `unsupportedArchitecture(String)`
- `unsupportedOperation(String)`
- `invalidModelConfiguration(String)`
- `invalidCtcConfiguration(String)`

Callers that switch exhaustively over this enum must handle the new cases or add a fallback. Swift's API digester reports these four additions as source-breaking changes relative to 0.1.1. The comparison found no removed APIs. This is an intentional compatibility change in the 0.2 release.

```swift
catch let error as BuilderHelmAudioError {
    switch error {
    case .unsupportedOperation(let operation):
        print("Selected model does not support \(operation)")
    default:
        print(error.localizedDescription)
    }
}
```

## Model capabilities

The new `.parakeetTdtV3` model returns TDT text only. Check `recognizer.capabilities` before requesting CTC probabilities or keyword spotting. `ctcTranscript` is nil for an absent CTC head; the existing `ctcText` accessor returns an empty string for compatibility. `tokens` is empty when token timestamps are unavailable. Use `keywordSpotterIfAvailable` for a conditional spotter.

```swift
let transcript = try await recognizer.transcribe(
    file: audioURL,
    includeCtcLogProbabilities: recognizer.capabilities.contains(.ctc)
)
```

Async load/download/transcription can now report `CancellationError` when their task is cancelled. Active synchronous CoreML work may finish before cancellation is observed; a recognizer is reused only after that work finishes.

`AsrModel.all` now contains three models, including V3. Code that iterates every model should use capability metadata rather than assuming every model provides CTC, timestamps or keyword spotting.
