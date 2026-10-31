# Parakeet TDT 0.6B v3

BuilderHelmAudio 0.2.0 adds `.parakeetTdtV3`, a pinned conversion of NVIDIA's multilingual Parakeet v3. It ships **unquantized FP16** encoder, predictor and joint weights, with an FP32 frontend and floating-point interfaces. There is no additional training. The download is 1,267,647,148 bytes, plus CoreML's locally compiled cache.

```swift
let recognizer = try await SpeechRecognizer.load(.parakeetTdtV3)
let transcript = try await recognizer.transcribe(file: recordingURL)
print(transcript.text)
```

The descriptor advertises the checkpoint's 25 languages and TDT text only. `ctcTranscript` is nil, `tokens` is empty, and CTC probability/keyword requests throw `unsupportedOperation`. The SDK does not expose aligned V3 word/segment timestamps, live partial results, subtitles or speaker labels.

## Runtime and long recordings

Requests up to 15 seconds use `w15`; longer requests use `w30`. The frontend and encoder share constants across both functions, adding about 3.3% to the 15-second bundle. The runtime keeps one encoder function resident. Reuse the recognizer; a function switch can incur preparation cost.

Files are read in bounded blocks of at most 480,001 input samples, with no fixed SDK recording-duration limit. The caller owns the full array when using the PCM API. Encoder outputs are consumed one window at a time. Model-load/feature temporaries are drained, compiled package URLs are cached per recognizer, and encoder output buffers/prediction options are reused.

The planner searches 10 ms RMS values for a quiet run of at least 160 ms, using the greater of 0.0005 and 2% of peak RMS as the threshold. Eligible cut ranges are 8–12 seconds for `w15` and 20–27 seconds for `w30`. Input coverage is retained and cuts are aligned to the encoder's 80 ms frame grid. A full encoder window retains right context. With no eligible pause, frame-aligned overlap supplies context. Joins prefer at least three matching token IDs near the same audio time and fall back to emission-frame ownership. Entirely blank speech-bearing windows can receive at most two subdivision levels.

A 30-second window can also partially blank active speech on some backends. Token-empty spans of about five seconds or more receive a bounded 15-second retry only when RMS energy is both active and varying. Recovery inserts at least three candidate tokens between matching three-token anchors near the same audio time; it preserves every primary token. One-sided anchors handle leading/trailing gaps. The retry reuses the existing short-window planner, consumes at most four child windows, and does not recurse. Primary prediction temporaries are released before switching encoder functions. This is a recovery guard, not a voice activity detector or an aligned timestamp API.

Each window resets the original LSTM predictor. State carry across windows was tested and rejected after a large WER regression. Internal emission frames are ownership hints, not aligned public word times. Cancellation is cooperative between predictions, decoder steps and windows; active synchronous CoreML calls can finish first. A recognizer serializes requests and can be reused afterwards.

## Quantization decision

The selection set contains 50 FLEURS clips, two per language, separate from the engineering accuracy holdouts. The frozen gates required pooled WER drift no greater than 0.3 percentage points, every language drift no greater than 2 points, and no additional failures or empty speech. Small language denominators make these results coarse.

| Encoder weights | Selection WER | Worst language drift vs FP16 | Decision |
|---|---:|---:|---|
| FP16 | 12.02% | baseline | Ship |
| INT8 block 32 | 11.91% | +3.57 pp, Maltese | Reject |
| INT8 block 16 | 12.02% | +2.70 pp, Croatian | Reject |
| INT4 | 14.68% | +14.29 pp, Swedish | Reject |

Those weight-only trials used the 15-second export. The thresholds were not relaxed after observing results. INT8/INT4 exports are not part of this release, and FP16 must not be described as an integer-quantized model.

## Accuracy evidence

English uses Whisper EnglishTextNormalizer; other languages use BasicTextNormalizer with NFC and diacritics retained. Missing or failed speech remains in the denominator as empty hypotheses.

| Corpus | Files | Native NVIDIA CPU FP32 WER | Release FP16 WER |
|---|---:|---:|---:|
| Reserved multilingual short speech | 200 | 14.52% | 14.46% |
| Separate multilingual short confirmation | 75 | 13.44% | 13.21% |
| Pooled unique short recordings | 275 | 14.24% | 14.14% |
| Natural continuous English call excerpts | 29 | 10.74% | 10.78% |
| Separate call confirmation intervals | 18 | 11.92% | 11.86% |

The short holdouts exclude model-selection IDs, audio hashes and normalized references. They are held out from this engineering process; pretraining overlap and speaker disjointness are unknown. The natural intervals recover exactly adjacent source chunks, with no inserted silence or reordered audio. They come from a corpus previously used for 110M research. The second set uses disjoint intervals from the same calls and is not speaker-independent. The natural sets contain 1,275 and 720 seconds respectively.

Long20 is a known 20-file LibriSpeech composite regression set. The original V3 overlap policy scored 14.03% WER, the intermediate 15-second repair 2.32%, and the shared-window release 2.22%, below the 2.44% engineering target. These are regression results, not new held-out conversation accuracy.

There is a tradeoff on synthetic multilingual long composites. The 50-file development set worsens from 14.84% under the previous 15-second policy to 15.45%. The separate 25-file confirmation improves from 13.74% to 13.44% (native whole-context FP32: 13.66%). Pooled over 75 files, WER changes from 14.55% to 14.92%, within the 0.5-point pooled drift gate. These composites reuse the short recordings and are not independent natural conversations. The release does not improve every corpus or establish natural long-form multilingual robustness.

The final buffer-reuse runtime preserves exact CPU token sequences on 351/351 baseline clips. Final file and PCM paths have identical text, token IDs and internal frames on all 20 Long20 files and 200 reserved short clips. New integration CI downloads and verifies the public export, exercises short → long → short reuse, file/PCM parity, silence, cancellation and capability rejection.

## Devices and compatibility

The target is **all Apple Silicon Macs, M1 onward, running macOS 15+**, plus iOS 18+. Physical qualification uses M5 and iPhone 13. Portable `.mlpackage` files compile locally; generic arm64/macOS 15 builds and CoreML8/specification-9 contracts are checked. No M5-only binary or compiled cache is distributed. A deployment audit and generic compilation do not reproduce execution or performance on every chip, old OS or 8 GB machine.

The default requests CPU plus Neural Engine for model components, and CPU for the frontend. Per-operation placement is determined by CoreML. CPU and GPU modes are available. Hosted CI proves compilation, CPU inference and SDK behavior; it does not establish Neural Engine or physical-device performance.

Final sustained-device results are recorded before publication. A prior no-monitor 20-minute phone run reproduced encoder latency stalls while thermal state stayed nominal. Stage timing isolated the encoder. A controlled 55-call comparison with retained output buffers measured 186 ms median, 274 ms p95 and 283 ms maximum. The final release runtime is separately qualified for 20 minutes and then on 63 long recordings; controlled comparison figures alone are not the sustained result.

On M5, the public SDK's synthetic 6.07-second fixture had a 52 ms warm median over ten runs; the 71.80-second fixture had a 656 ms warm median. Initial load including compilation took 11.5 seconds, and the first long-function request took 12.6 seconds including preparation. Subsequent short reuse after long requests remained stable. These small fixture timings are separate from the accuracy corpus and do not predict every recording's latency.

Process footprint does not include all accelerator allocations and is not total model memory. Charging-device battery changes do not establish inference energy. Natural multilingual conversations, other-generation hardware performance and controlled energy measurements remain additional coverage.

## Source and reproduction

- [Published model and aggregate evidence](https://huggingface.co/BuilderHelm/Parakeet-TDT-0.6B-V3-CoreML)
- [NVIDIA source at pinned revision](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/541d1f99c6b0c3cd0b11a95167540bb8edefd82b)
- Original checkpoint SHA-256: `3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d`
- [Exporter and merger](../Scripts/ParakeetV3/README.md), with pinned dependencies and no additional training
- [Locally managed exports](ExperimentalParakeetV3.md)

The model weights remain NVIDIA CC BY 4.0, with attribution and a conversion notice. SDK/exporter source uses the repository license. Audio, evaluation transcripts, device caches and source checkpoints are excluded from model distribution.
