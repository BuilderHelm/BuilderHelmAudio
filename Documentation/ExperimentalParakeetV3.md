# Local Parakeet v3 exports

The released model uses `.parakeetTdtV3`. See [Parakeet V3](ParakeetV3.md) for the public API, capabilities, evaluation and release scope.

The existing `.experimentalParakeetV3` descriptor remains available for locally managed exports:

```swift
let recognizer = try SpeechRecognizer(
    directory: modelDirectory,
    model: .experimentalParakeetV3,
    compiledModelsDirectory: cacheDirectory
)
let transcript = try await recognizer.transcribe(file: audioURL)
```

Construct local recognizers off the main thread. This descriptor has no remote repository or revision, is excluded from `AsrModel.all`, and cannot be downloaded, verified or removed by `ModelStore`. The caller manages its folder. It advertises TDT text only, with the same capability limits as the released model.

Folders contain `Preprocessor.mlpackage`, `Encoder.mlpackage`, `Predictor.mlpackage`, `Joint.mlpackage`, `vocab.json` and `config.json`. All package/tokenizer checksums and tensor contracts are validated before inference. Legacy single-function 15-second configs remain supported. The release config additionally declares shared-weight `w15`/`w30` functions. Packages compile into a cache keyed by content and OS build; distribute portable `.mlpackage` files rather than a cache from a development machine.

[Reproduction instructions](../Scripts/ParakeetV3/README.md) describe the pinned source and exporter. Local exports require independent parity and accuracy qualification before deployment. `.experimentalParakeetV3` does not make an arbitrary export equivalent to the released files.
