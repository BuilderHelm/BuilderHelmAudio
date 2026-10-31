# BuilderHelmAudio

On-device speech recognition for Apple platforms. Swift, CoreML, Neural Engine.

BuilderHelmAudio downloads [BuilderHelm's Parakeet models](https://huggingface.co/BuilderHelm) from Hugging Face, checks every file against a pinned SHA-256, keeps them on the device and runs them locally. No audio leaves the device.

- **Fast**: about 1,000× real time on a Mac (M5): 5 minutes of audio in 0.3 seconds
- **Accurate**: 2.46% WER on LibriSpeech test-clean (TDT)
- **Keyword spotting**: find names and terms in audio without a tokenizer
- **Simple**: one call to download and load, one call to transcribe

## Requirements

- iOS 18+ or macOS 15+
- Swift 6 (Xcode 16 or later)

## Installation

In Xcode: **File → Add Package Dependencies…** and enter `https://github.com/BuilderHelm/BuilderHelmAudio`.

Or in `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/BuilderHelm/BuilderHelmAudio", from: "0.1.0"),
],
targets: [
    .target(name: "MyApp", dependencies: ["BuilderHelmAudio"]),
]
```

## Quick Start

```swift
import BuilderHelmAudio

// Downloads the model on first use (144 MB), then loads it from the device
let recognizer = try await SpeechRecognizer.load(.parakeetTdtCtc110M) { progress in
    print("Downloading \(Int(progress.fraction * 100))%")
}

let transcript = try await recognizer.transcribe(file: audioURL)
print(transcript.text)                       // TDT transcript
print(transcript.tokens.first?.time ?? 0)    // token times in seconds
```

Any audio format AVFoundation can read works; it is converted to 16 kHz mono. You can also pass samples:

```swift
let transcript = try await recognizer.transcribe(samples: samples16kHzMono)
```

Load the recognizer once and reuse it. Calls run off the caller's thread, one at a time.

## Models

| Model | Use | Download | Short clips | Long audio |
|-------|-----|---------:|------------:|-----------:|
| `.parakeetTdtCtc110M` | Transcription with token times, plus CTC | 144 MB | 860× | 791× |
| `.parakeetCtc110M` | Fastest transcription and keyword spotting | 124 MB | 1,071× | 998× |

Speed is × real time on an Apple M5 Mac, from the model cards: [TDT + CTC](https://huggingface.co/BuilderHelm/Parakeet-TDT-CTC-110M-CoreML), [CTC](https://huggingface.co/BuilderHelm/Parakeet-CTC-110M-CoreML). English only. iPhone results will be published after on-device testing.

Each model is pinned to a Hugging Face commit, so a given SDK version always runs the same files.

## Keyword Spotting

```swift
let recognizer = try await SpeechRecognizer.load(.parakeetCtc110M)
let hits = try await recognizer.spot(["BuilderHelm", "Neural Engine"], inFile: audioURL)
for hit in hits {
    print(hit.keyword, hit.score, hit.start, hit.end)   // score near 0 = confident
}
```

Phrases are case-sensitive. The default threshold is −1; spoken words usually score above −0.1. To score many phrases against one recording, transcribe once with `includeCtcLogProbabilities: true` and call `recognizer.keywordSpotter.detect(_:in:)` on `transcript.ctcLogProbabilities`.

## Model Storage

Models are kept in `Application Support/BuilderHelmAudio` (excluded from iCloud backup on iOS):

```swift
let store = ModelStore.shared
store.isDownloaded(.parakeetCtc110M)
try await store.download(.parakeetCtc110M) { print($0.fraction) }
try await store.verify(.parakeetCtc110M)     // re-hash every file
try await store.remove(.parakeetCtc110M)
```

Interrupted downloads resume file by file. A file that fails its checksum is discarded. To ship a model inside your app instead, load its folder directly:

```swift
let recognizer = try SpeechRecognizer(directory: bundledModelFolder, model: .parakeetCtc110M)
```

## Command Line (macOS)

```bash
swift run -c release bhaudio models
swift run -c release bhaudio transcribe audio.wav
swift run -c release bhaudio transcribe audio.wav --model parakeet-ctc-110m --json
swift run -c release bhaudio spot audio.wav "Neural Engine" BuilderHelm
swift run -c release bhaudio benchmark cases.json --rounds 3
```

`benchmark` takes a JSON array of `{"path": "...", "text": "optional reference"}` and reports real-time factor, latency percentiles and WER.

## License

Code: [MIT](LICENSE). Models: CC-BY-4.0, from NVIDIA's [Parakeet-TDT-CTC-110M](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m); see each model card.
