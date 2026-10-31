<p align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0E1B26,50:1E66D8,100:13283A&height=190&section=header&text=BuilderHelmAudio&fontSize=52&fontColor=FEFEFF&animation=fadeIn&fontAlignY=36&desc=On-device%20speech%20recognition%20for%20iOS%20and%20macOS&descAlignY=60&descSize=18" alt="BuilderHelmAudio — on-device speech recognition for iOS and macOS" />
</p>

<p align="center">
  <img src="Resources/Brand/BuilderHelmLogo.png" width="120" alt="BuilderHelm logo">
</p>

<p align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=20&duration=3200&pause=900&color=8C8CFF&center=true&vCenter=true&width=780&lines=Speech+to+text+on+the+Neural+Engine.;One+call+to+load.+One+call+to+transcribe.;No+audio+leaves+the+device." alt="Speech to text on the Neural Engine." />
  </a>
</p>

<p align="center">
  <img alt="iOS 18+" src="https://img.shields.io/badge/iOS-18%2B-1E66D8?style=for-the-badge&logo=apple&logoColor=white">
  <img alt="macOS 15+" src="https://img.shields.io/badge/macOS-15%2B-1E66D8?style=for-the-badge&logo=apple&logoColor=white">
  <img alt="Swift 6" src="https://img.shields.io/badge/Swift-6-F05138?style=for-the-badge&logo=swift&logoColor=white">
  <img alt="Swift Package Manager" src="https://img.shields.io/badge/SPM-Compatible-111111?style=for-the-badge&logo=swift&logoColor=white">
</p>

<p align="center">
  <img alt="On-device" src="https://img.shields.io/badge/Privacy-On--device-0D855E?style=for-the-badge">
  <img alt="MIT" src="https://img.shields.io/badge/License-MIT-0D855E?style=for-the-badge">
  <img src="https://img.shields.io/badge/Models-Parakeet%20TDT%20%7C%20Parakeet%20CTC-1E66D8?style=for-the-badge" alt="Supported models">
</p>

<p align="center">
  <a href="https://github.com/BuilderHelm/BuilderHelmAudio/releases/latest">
    <img src="https://img.shields.io/github/v/release/BuilderHelm/BuilderHelmAudio?style=for-the-badge&logo=github&logoColor=white&color=1E66D8&label=Release" alt="Latest release" />
  </a>
  <a href="https://huggingface.co/BuilderHelm">
    <img src="https://img.shields.io/badge/Models-Hugging%20Face-111111?style=for-the-badge&logo=huggingface&logoColor=FFD21E" alt="BuilderHelm models on Hugging Face" />
  </a>
</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#supported-models">Models</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#api-reference">API</a> ·
  <a href="#command-line">Command line</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#privacy">Privacy</a> ·
  <a href="#development">Development</a>
</p>

BuilderHelmAudio is a Swift package for speech recognition on iPhone, iPad and Mac. Load a model, pass it a recording, and get the transcript with token times.

It downloads [BuilderHelm's Parakeet models](https://huggingface.co/BuilderHelm) from Hugging Face, checks every file against a pinned SHA-256, keeps them on the device and runs them locally on the Neural Engine. No audio leaves the device.

```swift
let recognizer = try await SpeechRecognizer.load(.parakeetTdtCtc110M)
let transcript = try await recognizer.transcribe(file: audioURL)
print(transcript.text)
```

---

## Features

- **Fast** — about 1,000× real time on a Mac (M5), 400–600× on an iPhone 13.
- **Accurate** — 2.46% WER on LibriSpeech test-clean (TDT).
- **Simple** — one call to download and load, one call to transcribe.
- **Token times** — TDT transcripts carry the time of every token, plus a CTC transcript from the same pass.
- **Keyword spotting** — find names and terms in audio, with start and end times. No tokenizer needed.
- **Verified downloads** — every file is pinned to a Hugging Face commit and checked by SHA-256. Interrupted downloads resume file by file.
- **Any audio** — any format AVFoundation reads, or raw samples; converted to 16 kHz mono.
- **Any length** — long recordings are split into overlapping windows and joined without repeating words.
- **Command line** — `bhaudio` transcribes, spots keywords and benchmarks on macOS.

---

## Supported models

| Model | Best for | Output | Download | Model card |
|---|---|---|---:|---|
| **`.parakeetTdtCtc110M`** | Transcription with token times | TDT text + token times, CTC text | 144 MB | [Parakeet-TDT-CTC-110M-CoreML](https://huggingface.co/BuilderHelm/Parakeet-TDT-CTC-110M-CoreML) |
| **`.parakeetCtc110M`** | Fastest transcription, keyword spotting | CTC text | 124 MB | [Parakeet-CTC-110M-CoreML](https://huggingface.co/BuilderHelm/Parakeet-CTC-110M-CoreML) |

Both are NVIDIA's [Parakeet-TDT-CTC-110M](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m), converted to CoreML with an int8 encoder. English only. Each model is pinned to a Hugging Face commit, so a given SDK version always runs the same files.

---

## Installation

**Xcode:** choose **File → Add Package Dependencies…** and enter:

```text
https://github.com/BuilderHelm/BuilderHelmAudio
```

**Swift Package Manager:** add it to `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/BuilderHelm/BuilderHelmAudio", from: "0.1.0"),
],
targets: [
    .target(name: "MyApp", dependencies: ["BuilderHelmAudio"]),
]
```

---

## Usage

### Transcribe a file

```swift
import BuilderHelmAudio

// Downloads the model on first use (144 MB), then loads it from the device
let recognizer = try await SpeechRecognizer.load(.parakeetTdtCtc110M) { progress in
    print("Downloading \(Int(progress.fraction * 100))%")
}

let transcript = try await recognizer.transcribe(file: audioURL)
print(transcript.text)                       // TDT transcript
print(transcript.ctcText)                    // CTC transcript from the same pass
print(transcript.tokens.first?.time ?? 0)    // token times in seconds
print(transcript.realTimeFactor)             // audio seconds per processing second
```

Any format AVFoundation reads works (WAV, M4A, MP3, CAF and more). Long files are read in blocks, so they don't need to fit in memory.

Load the recognizer once and reuse it. Calls run off the caller's thread, one at a time, so one instance can be shared.

### Transcribe samples

```swift
let transcript = try await recognizer.transcribe(samples: samples16kHzMono)
```

Samples must be 16 kHz mono `Float`. `SpeechRecognizer.loadAudio(_:)` converts any supported file to that format.

### Keyword spotting

```swift
let recognizer = try await SpeechRecognizer.load(.parakeetCtc110M)
let hits = try await recognizer.spot(["BuilderHelm", "Neural Engine"], inFile: audioURL)
for hit in hits {
    print(hit.keyword, hit.score, hit.start, hit.end)   // score near 0 = confident
}
```

Phrases are case-sensitive. The default threshold is −1; spoken words usually score above −0.1. To score many phrase lists against one recording, transcribe once and reuse the probabilities:

```swift
let transcript = try await recognizer.transcribe(file: audioURL, includeCtcLogProbabilities: true)
let hits = recognizer.keywordSpotter.detect(["Christmas", "Mediterranean"], in: transcript.ctcLogProbabilities!)
```

### Decoding mode and compute units

```swift
// CTC only on the TDT + CTC model: skips the TDT decoder
let fast = try await SpeechRecognizer.load(.parakeetTdtCtc110M, mode: .ctc)

// Run the encoder on the GPU or CPU instead of the Neural Engine
let gpu = try await SpeechRecognizer.load(.parakeetCtc110M, computeUnits: .gpu)
```

The Neural Engine (`.neuralEngine`, the default) is fastest on iPhone and Mac.

### Model storage

Models are kept in `Application Support/BuilderHelmAudio`, excluded from iCloud backup on iOS:

```swift
let store = ModelStore.shared
store.isDownloaded(.parakeetCtc110M)
try await store.download(.parakeetCtc110M) { print($0.fraction) }
try await store.verify(.parakeetCtc110M)     // re-hash every file
try await store.remove(.parakeetCtc110M)
```

A file that fails its checksum is discarded. To download from a mirror with the same `<repo>/resolve/<revision>/<path>` layout, use `ModelStore(baseURL:)`.

### Ship a model inside your app

Bundle a model folder and load it directly. Nothing is downloaded:

```swift
let recognizer = try SpeechRecognizer(directory: bundledModelFolder, model: .parakeetCtc110M)
```

This initializer blocks while CoreML loads the model; call it off the main thread.

### Errors

| `BuilderHelmAudioError` | When |
|---|---|
| `.downloadFailed(file:status:)` | A model file could not be downloaded |
| `.checksumMismatch(file:)` | A downloaded file did not match its pinned SHA-256; it was discarded |
| `.unsupportedAudio(_:)` | AVFoundation could not read the audio |
| `.modelLoadFailed(_:)` | CoreML could not load the model |

---

## API reference

| API | Purpose |
|---|---|
| `SpeechRecognizer.load(_:mode:computeUnits:store:progress:)` | Download if needed, then load a model |
| `SpeechRecognizer(directory:model:mode:computeUnits:compiledModelsDirectory:)` | Load a model folder you manage |
| `transcribe(file:includeCtcLogProbabilities:)` | Transcribe an audio file |
| `transcribe(samples:includeCtcLogProbabilities:)` | Transcribe 16 kHz mono samples |
| `spot(_:inFile:threshold:)` | Find keywords in an audio file |
| `keywordSpotter.detect(_:in:threshold:)` | Find keywords in saved CTC probabilities |
| `SpeechRecognizer.loadAudio(_:)` | Read any supported file as 16 kHz mono samples |
| `ModelStore.shared` | `download`, `isDownloaded`, `verify`, `remove`, `directory(for:)` |
| `AsrModel.all`, `AsrModel.named(_:)` | List models, or look one up by id |

| `Transcript` | |
|---|---|
| `text` | TDT transcript (CTC on CTC-only models) |
| `ctcText` | CTC transcript |
| `tokens` | `[TimedToken]`: each token's text and time in seconds |
| `duration`, `processingTime`, `realTimeFactor` | Audio length, processing time and speed |
| `ctcLogProbabilities` | Per-frame CTC probabilities, when requested |

---

## Command line

`bhaudio` runs on macOS:

```bash
swift run -c release bhaudio models
swift run -c release bhaudio transcribe audio.wav
swift run -c release bhaudio transcribe audio.wav --model parakeet-ctc-110m --json
swift run -c release bhaudio spot audio.wav "Neural Engine" BuilderHelm
swift run -c release bhaudio benchmark cases.json --rounds 3
```

`benchmark` takes a JSON array of `{"path": "...", "text": "optional reference"}` and reports real-time factor, latency percentiles and WER.

---

## Performance

Speed, × real time (higher is faster):

| Model | Mac M5, short clips | Mac M5, long audio | iPhone 13, 6 s clip | iPhone 13, 72 s recording |
|---|---:|---:|---:|---:|
| **`.parakeetTdtCtc110M`** | 860× | 791× | 393× (15 ms) | 308× (233 ms) |
| **`.parakeetCtc110M`** | 1,071× | 998× | 604× (10 ms) | 415× (173 ms) |

- **Mac** numbers and accuracy are from the model cards: [TDT + CTC](https://huggingface.co/BuilderHelm/Parakeet-TDT-CTC-110M-CoreML), [CTC](https://huggingface.co/BuilderHelm/Parakeet-CTC-110M-CoreML).
- **iPhone 13** numbers are the median of 10 runs of the demo app's samples on iOS 27.2; transcripts match the Mac word for word.
- **First load** on a device also compiles the model for CoreML. Later loads reuse the compiled copy.

---

## Requirements

- iOS 18+ or macOS 15+
- Swift 6 (Xcode 16 or later)
- A device with a Neural Engine for full speed; `computeUnits: .gpu` or `.cpu` also work
- Disk: about twice the download per model (124–144 MB), including the copy CoreML compiles on first load
- Network on first load only, to download the model from Hugging Face

---

## How it works

```text
Audio file or samples
  → 16 kHz mono
  → log-mel features (CPU)
  → FastConformer encoder (Neural Engine), in overlapping windows for long audio
  → TDT decoder (CPU, Accelerate) and greedy CTC
  → transcript, token times, keyword hits
```

The CPU decodes one window while the Neural Engine encodes the next.

---

## Privacy

BuilderHelmAudio makes one kind of network request: downloading model files from Hugging Face.

| What | Where it goes |
|---|---|
| Audio | Stays on the device. Never uploaded. |
| Transcripts | Returned to your app. Never stored or uploaded by the SDK. |
| Model files | Downloaded from `huggingface.co` at a pinned commit, or from your mirror via `ModelStore(baseURL:)` |
| Analytics | None |

---

## Development

### Build

```bash
git clone https://github.com/BuilderHelm/BuilderHelmAudio.git
cd BuilderHelmAudio
swift build -c release
```

### Test

```bash
swift test                                   # unit tests
BUILDERHELM_INTEGRATION=1 swift test         # also downloads both models and transcribes
xcodebuild test -scheme BuilderHelmAudio-Package -destination 'platform=iOS Simulator,name=iPhone 17 Pro'
```

The iOS Simulator has no Neural Engine, so its speed is not representative. For device numbers, use the [demo app](#demo-app).

### Architecture

```mermaid
flowchart LR
    App[Your app] --> Recognizer[SpeechRecognizer]
    Recognizer --> Store[ModelStore]
    Store --> HF[(Hugging Face, pinned + SHA-256)]
    Recognizer --> Runtime[ParakeetRuntime]
    Runtime --> ANE[Encoder on the Neural Engine]
    Runtime --> CPU[TDT + CTC decoding on the CPU]
    Recognizer --> Spotter[KeywordSpotter]
```

| Target | Responsibility |
|---|---|
| `BuilderHelmAudio` | Public API: `SpeechRecognizer`, `ModelStore`, `KeywordSpotter`, `AsrModel` |
| `ParakeetRuntime` | CoreML + Accelerate Parakeet runtime, shared with the BuilderHelm model repositories |
| `bhaudio` | Command-line tool for macOS |
| `BuilderHelmAudioTests` | Unit, download and long-audio pipeline tests |

### Project layout

```text
BuilderHelmAudio/
├── Sources/
│   ├── BuilderHelmAudio/     # Public API
│   ├── ParakeetRuntime/      # CoreML + Accelerate runtime
│   └── bhaudio/              # Command-line tool
├── Tests/BuilderHelmAudioTests/
├── Examples/BuilderHelmAudioDemo.swiftpm/
└── Resources/Brand/
```

### Demo app

[`Examples/BuilderHelmAudioDemo.swiftpm`](Examples/BuilderHelmAudioDemo.swiftpm) is a small iPhone and iPad app: download a model, transcribe the bundled samples or the microphone, spot keywords and benchmark speed. Open it in Xcode, pick your team under **Signing & Capabilities**, and run it on a device. See its [README](Examples/BuilderHelmAudioDemo.swiftpm/README.md).

---

## Status

Version 0.1.0, the first release. Tested on an Apple M5 Mac (macOS 27.2) and an iPhone 13 (iOS 27.2). Other devices, battery use and heat are not yet measured. English only.

[File an issue](https://github.com/BuilderHelm/BuilderHelmAudio/issues/new) if something breaks.

---

## License

Code: [MIT](LICENSE). Models: CC-BY-4.0, from NVIDIA's [Parakeet-TDT-CTC-110M](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m); see each model card.

<p align="center">
  <em>Speak. It transcribes. On the device.</em>
</p>

<p align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1E66D8,50:13283A,100:0E1B26&height=120&section=footer" alt="" />
</p>
