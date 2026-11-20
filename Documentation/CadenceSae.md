# Cadence Sae

`.cadenceSaeEn` is Cadence Sae, BuilderHelm's English speech model for ZenVoice. It is NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) with the top 8 encoder layers, the predictor and the joint fine-tuned. The training data was Common Voice 17 English, mostly accent-labelled speakers outside US English, and AMI meeting speech. The encoder weights are 6-bit k-means palettized.

```swift
let recognizer = try await SpeechRecognizer.load(.cadenceSaeEn)
try await recognizer.prepare()
let transcript = try await recognizer.transcribe(file: audioURL)
```

| | |
|---|---|
| Repository | [BuilderHelm/Cadence-Sae-EN](https://huggingface.co/BuilderHelm/Cadence-Sae-EN) at `4702453cd0c6f922b1480857223371db53aa6d40`, folder `coreml/6bit/` (variant `6bit`) |
| Download | 19 files, 542 MB, each pinned by size and SHA-256 |
| Languages | English (`["en"]`) |
| Runtime | The V3 runtime: 15/30-second shared-weight windows, quiet-pause long-audio joins |
| Capabilities | TDT text, token and word timestamps with confidences, ranked alternatives; no CTC or keyword spotting |
| Licence | CC BY 4.0, a modified NVIDIA model; keep the repository's NOTICE.md with redistributions |

The repository also holds FP16, mixed 4/6-bit and 4-bit builds and the NeMo checkpoint. Only the 6-bit build is published as a descriptor. For another build, download its folder and pass `.experimentalParakeetV3` to `SpeechRecognizer(directory:model:)`.

## Evaluation

On ZenVoice Speech Bench v1 the M5 Mac ran the benchmark's own protocol: an accuracy pass, a reverse-order timing pass and three cold starts.

| Build | WER | 95% range | Speed | Typical delay | Load + first clip |
|---|--:|--:|--:|--:|--:|
| `.cadenceSaeEn` (6-bit) | 7.5% | 7.0–8.1% | 147× | 0.05 s | 0.43 s |
| FP16 build (not a descriptor) | 7.5% | 7.0–8.0% | 140× | 0.05 s | 0.43 s |

Paired against the V3 conversion on the same 1,332 speech clips, the 6-bit build differs by −0.02 points (95% interval −0.23 to +0.19). AMI meetings improve from 7.9% to 6.8%. Accented clips are less often transcribed as another language. Recorded Indian English dictation is level with V3, and EdAcc conversational accents are slightly worse.

On an iPhone 13 (iOS 27.2) the 6-bit build ran 25.5 minutes of Speech Bench audio at 43× real time: 97 ms median per clip, a 1.7 s warm load and warm-up, and a 5-minute sustained run at 25–33×. Its transcripts matched the Mac's on 237 of 238 clips. The first launch on a device compiles the model, which takes about 1 minute on iPhone 13 and 15–30 s on M5. Call `prepare()` during setup. The model card has the full results, the compression decision and the limitations.
