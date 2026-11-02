# Parakeet TDT 0.6B v3

BuilderHelmAudio 0.2.0 adds `.parakeetTdtV3`, a pinned conversion of NVIDIA's multilingual Parakeet v3. It ships **unquantized FP16** encoder, predictor and joint weights, with an FP32 frontend and floating-point interfaces. There is no additional training. The download is 1,267,682,061 bytes, plus CoreML's locally compiled cache.

```swift
let recognizer = try await SpeechRecognizer.load(.parakeetTdtV3)
let transcript = try await recognizer.transcribe(file: recordingURL)
print(transcript.text)
```

The descriptor advertises the checkpoint's 25 languages and TDT text only. `ctcTranscript` is nil, `tokens` is empty, and CTC probability/keyword requests throw `unsupportedOperation`. The SDK does not expose aligned V3 word/segment timestamps, live partial results, subtitles or speaker labels.

## Runtime and long recordings

The SDK now overlaps encoding window N+1 with CPU decoding of window N for recordings longer than 30 seconds. One encoder model set is retained, with two bounded feature snapshots. Requests on a recognizer remain serialized, output order and join/recovery rules stay unchanged, and cancellation drains the encoder producer before the session can be reused. Recovery waits for the producer before switching model functions. Short requests, diagnostic traces and older bundles without shared windows retain sequential execution.

This changes the runtime only; it uses the same pinned FP16 model download. A five-block M5 Long20 prototype comparison observed about 47% higher throughput than the released sequential runtime with exact text, tokens and frames. That is a known regression workload on one active desktop, not a competitive speed claim. The identical runtime and current model passed the corrected iPhone 13 qualification: 172 paced requests over 20 minutes, 213 ms p95, 280 ms maximum, all 63 long requests and reuse checks passed. Fresh engineering confirmation preserved all 2,510 released FP16 transcripts exactly. On 2,500 multilingual clips, pooled WER was 14.13%, versus 15.11% for Fluid default and 14.68% for INT8-linear v2, with paired recording-cluster intervals favoring this model. On ten full English calls, WER was 12.63%; the difference versus Fluid v2 was inconclusive. Five paired M5 Long20 PCM blocks measured only 0.664× Fluid default throughput and 0.632× v2. The competitive speed target failed, so broader release qualification stopped and no release is established by this work. These public corpora do not establish speaker independence or pretraining novelty; energy and unmeasured hardware remain unmeasured.

Requests up to 15 seconds use `w15`; longer requests use `w30`. Each frontend/encoder package shares its own constants across both functions. The FP16 Conformer is partitioned into sequential `Encoder` and `EncoderTail` programs, each below 1 GB. This preserves the weights while retaining an approximately 1.27 GB download; the second window adds about 3.3%. The runtime retains the two stages for one window size at a time. Reuse the recognizer; a function switch can incur preparation cost.

The FP32 frontend extracts at most 401 raw log-mel frames per DFT prediction, then normalizes the assembled complete 15/30-second window. Pre-emphasis is evaluated in FP32 in the runtime. All weight bytes and the original frontend functions are preserved. This bounded path makes repeated full decoding stable on the hosted macOS 15 CPU runner; frozen features match the validated full-window reference within 0.0000005. The older model 1.0.0 remains immutable for reproduction.

Files are read in bounded blocks of at most 480,001 input samples, with no fixed SDK recording-duration limit. The caller owns the full array when using the PCM API. Encoder outputs are consumed one window at a time. Model-load/feature temporaries are drained and compiled package URLs are cached per recognizer. Owned inputs and feature snapshots use page-aligned storage; model-returned tensors are read inside scoped buffer access. Partitioned encoder stages use framework-managed outputs and default prediction options. The stage boundary retains its producers until the final features are copied into owned storage. Existing single-encoder exports retain their CPU/Neural Engine output-buffer reuse.

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

Long20 is a known 20-file LibriSpeech composite regression set. The original V3 overlap policy scored 14.03% WER, the intermediate 15-second repair 2.32%, and the shared-window release with partial recovery 1.83%, below the 2.44% engineering target. These are regression results, not new held-out conversation accuracy.

On synthetic multilingual long composites, the 50-file development set scores 14.84% WER, matching the previous 15-second policy. The separate 25-file confirmation scores 13.44%, versus 13.74% previously and 13.66% for native whole-context FP32. Pooled over 75 files, WER is 14.47%, versus 14.55% under the previous policy. These composites reuse the short recordings and are not independent natural conversations. The release does not establish natural long-form multilingual conversation robustness.

The final buffer-reuse runtime preserves exact CPU token sequences on 351/351 baseline clips. Final file and PCM paths have identical text, token IDs and internal frames on all 20 Long20 files and all 25 multilingual long confirmation files. The earlier file/PCM short qualification also matched all 200 reserved clips. New integration CI downloads and verifies the public export, exercises short → long → short reuse, file/PCM parity, silence, cancellation and capability rejection.

## Devices and compatibility

The target is **all Apple Silicon Macs, M1 onward, running macOS 15+**, plus iOS 18+. Physical qualification uses M5 and iPhone 13. Portable `.mlpackage` files compile locally; generic arm64/macOS 15 builds and CoreML8/specification-9 contracts are checked. No M5-only binary or compiled cache is distributed. A deployment audit and generic compilation do not reproduce execution or performance on every chip, old OS or 8 GB machine.

The default requests CPU plus Neural Engine for the encoder, and CPU for the frontend, predictor and joint. Per-operation placement is determined by CoreML. CPU is also available on iOS. GPU encoder mode is supported on Mac; V3 GPU requests on iOS fail during loading with an explicit error because repeated device diagnostics stalled on long-window switching. Hosted CI proves compilation, CPU inference and SDK behavior; it does not establish Neural Engine or physical-device performance.

The production-SDK iPhone 13 run completed 172 paced requests over 20.0 minutes: 177 ms median, 215 ms p95 and 327 ms maximum, with 0 requests over one second. Repeated transcripts stayed stable and the app remained active. Thermal state stayed at or below fair. All 63 following long-file requests completed. File/PCM reuse, short reuse after long audio and long silence checks passed. Phone WER was 11.98% on the 18 natural confirmation excerpts, 1.83% on Long20 and 13.44% on 25 multilingual long composites. This is paced dictation followed by a bounded long batch; it does not measure all-day thermals or inference energy.

Warm fixture timings and cold compilation/preparation costs are reported separately in the model card. Reusing a recognizer avoids repeated package loading; function changes can still require preparation. These small synthetic fixtures do not predict every recording's latency.

Process footprint does not include all accelerator allocations and is not total model memory. Charging-device battery changes do not establish inference energy. Natural multilingual conversations, other-generation hardware performance and controlled energy measurements remain additional coverage.

## Source and reproduction

- [Published model and aggregate evidence](https://huggingface.co/BuilderHelm/Parakeet-TDT-0.6B-V3-CoreML)
- [NVIDIA source at pinned revision](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/541d1f99c6b0c3cd0b11a95167540bb8edefd82b)
- Original checkpoint SHA-256: `3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d`
- [Exporter and merger](../Scripts/ParakeetV3/README.md), with pinned dependencies and no additional training
- [Locally managed exports](ExperimentalParakeetV3.md)

The model weights remain NVIDIA CC BY 4.0, with attribution and a conversion notice. SDK/exporter source uses the repository license. Audio, evaluation transcripts, device caches and source checkpoints are excluded from model distribution.
