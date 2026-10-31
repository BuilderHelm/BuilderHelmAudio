# BuilderHelmAudio roadmap

Planned on 31 October 2026, starting from SDK 0.1.1.

Implement these milestones one at a time, in the order below. Milestone 1 is implemented locally with validation documented below; hosted CI verification is pending the first push. Remaining milestones are planned. Complete and review each milestone's evidence before starting the next. Release numbers and dates are assigned when the scope and validation are ready.

## Existing behavior to preserve

- Keep both 110M English models as supported, smaller options.
- Preserve file/sample transcription, TDT token times, CTC text and keyword spotting on compatible models.
- Preserve pinned model revisions, per-file SHA-256 download verification, explicit cache verification, mirror/bundled-model support and the privacy manifest.
- Preserve existing public call sites where practical; document any required migration before release.
- Keep inference local. New optional features must not become mandatory model downloads.

## Working rules

- Each milestone produces an implementation, relevant tests, updated documentation and a short validation report.
- Define benchmark workloads and acceptance limits before tuning. Separate development audio from held-out evaluation audio.
- Report model revision, runtime revision, device, OS, compute precision and compute units with measurements.
- Measure loading separately from transcription. Separate initial compilation from reuse of compiled models, and report memory with consistent units and measurement methods.
- Real iPhone requirements need real-device measurements. Simulator results establish compatibility, not device performance; unavailable hardware leaves that validation pending.
- Do not infer streaming latency, power use or multilingual accuracy from the existing English file benchmarks.

## 1. Model architecture support and CI

Status: implemented locally. See [architecture](Documentation/RuntimeArchitecture.md) and [validation](Documentation/Milestone1Validation.md). Hosted GitHub Actions execution is pending the first push; v3 implementation has not started.

Work:

- Introduce an internal ASR runtime interface and adapt the existing 110M implementation without changing its decoding behavior.
- Extend model metadata with architecture, supported languages and capabilities, including TDT, CTC, token/word/segment timing and keyword support.
- Select the runtime from model metadata rather than directly constructing the 110M runtime in the public recognizer.
- Make CTC class count, blank ID and frame duration model-specific. Retain compatibility for existing 110M probability construction and callers.
- Represent unavailable outputs explicitly and reject unsupported operations with useful errors. Adding TDT support must not imply CTC or keyword support.
- Establish cancellation/lifecycle contracts for runtime work and downloads so later APIs can use them consistently.
- Add CI for builds and unit tests on macOS and iOS Simulator. Keep real-model integration tests as a separate explicit job with reproducible model pins and caching.

Complete when:

- Existing examples build and existing tests pass.
- The frozen 110M corpus preserves text, CTC output and token timing within documented numerical tolerances.
- Existing keyword fixtures preserve detections and scores within documented tolerances.
- Unsupported capability combinations fail predictably, without loading an incompatible runtime.
- CI runs from the repository; the architecture and compatibility decisions are documented.

## 2. Parakeet TDT 0.6B v3: unquantized baseline

Depends on milestone 1.

Work:

- Pin NVIDIA's original checkpoint and tokenizer, and preserve a reference inference path.
- Build a separate v3 runtime and reproducible CoreML conversion. Record all preprocessing, tokenizer, decoder and timestamp conventions.
- Validate against the source model before reducing weight precision. Use the source FP32 reference where available; label FP32 and FP16 CoreML execution explicitly. FP16 is an unquantized deployment baseline, not FP32 full precision.
- Add a v3 model descriptor and download manifest with pinned revisions, sizes and hashes.
- Expose v3 through the same load/transcribe API, with truthful capability metadata. Keep keyword spotting on the compatible 110M model until a v3-compatible path is independently validated.
- Create multilingual evaluation manifests covering the advertised 25 languages. Report per-language results, short/long speech, silence/noise behavior and timestamp alignment; do not hide weak languages behind one average.
- Measure unquantized model storage, compilation, warm loading, transcription latency/throughput and peak memory on Mac and a real iPhone.

Complete when:

- Conversion parity and multilingual accuracy meet acceptance limits fixed before tuning.
- Model loading, long recordings, timestamp ordering and existing 110M behavior pass validation.
- Device results establish the supported hardware/OS envelope and document any constraints.
- A reproducible unquantized baseline report exists before quantization work starts. If the baseline does not fit a target iPhone, record that result rather than claiming device support.

Language scope: v3 supports 25 European languages including English. Hindi, Japanese and Mandarin require additional model coverage; converting v3 does not add them.

## 3. V3 quantization and release readiness

Depends on milestone 2.

Work:

- Evaluate supported CoreML weight-compression options against the frozen unquantized baseline, beginning with INT8 as a candidate rather than a promised winning configuration.
- Compare per-language accuracy, timestamps, package size, compilation, warm load, latency and memory using the same workloads.
- Validate the selected configuration on real iPhone hardware, including sustained sessions and observed thermal behavior. Measure energy only with a stated measurement method.
- Preserve the unquantized reference, conversion scripts and evidence even if a compressed model becomes the recommended download.
- Publish the model card, licenses/attribution, supported-device results, download manifest and SDK usage documentation for the selected configuration.

Complete when:

- A selected configuration meets the declared accuracy and device limits; do not ship compression solely because its archive is smaller.
- A fresh SDK consumer can download, verify, load and transcribe with v3.
- The existing 110M models remain usable and pass regression checks.

## 4. Voice activity detection

Depends on milestone 3.

Work:

- Evaluate Silero VAD as a separate optional component, with pinned source/model revisions and a supported Apple runtime path.
- Add speech probabilities and speech-start/speech-end events with absolute audio timestamps.
- Configure speech/silence thresholds, minimum durations and padding; preserve detector state across input chunks.
- Support reset and cancellation. Bound memory and validate sample-rate/channel conversion at the API boundary.
- Evaluate missed speech, false triggers, onset/offset delay and clipped initial/final words on clean, quiet and noisy speech.

Complete when:

- Chunked and continuous input produce consistent boundary behavior within documented tolerances.
- Reset, cancellation and repeated sessions do not leak state.
- Detection quality and CPU/memory measurements meet predefined limits on Mac and iPhone.

VAD detects speech presence. End-of-speech silence is not automatically a reliable conversational end-of-turn decision.

## 5. Buffered live transcription

Depends on milestones 1 and 4.

Work:

- Add a session API that accepts successive PCM buffers and emits partial/final transcript events, for example through AsyncThrowingStream.
- Keep microphone capture as a separable adapter/example so host applications can supply their own audio.
- Implement bounded buffering, processing cadence, explicit backpressure/overflow behavior, overlap handling and absolute timestamps.
- Define revisable partial text versus committed final text. Handle final flushing so the last words are not lost.
- Implement cancellation, finish, reset and repeated-session behavior. The microphone adapter handles interruptions and audio-route changes.
- Measure first-partial latency, finalization delay, text stability, accuracy, sustained memory and device thermal/energy behavior using stated methods.
- Compare the buffered design against preset acceptance limits. If it falls short, evaluate a dedicated streaming model as a separate decision before expanding that scope.

Complete when:

- Live input stays within the declared buffer limits, with no silent audio loss or duplicate committed phrases.
- Speech boundaries, final flushes, interruptions and repeated sessions pass tests with real audio.
- Supported devices meet predefined quality and latency targets during sustained sessions.
- Documentation clearly distinguishes buffered ASR from a model with native streaming state.

Candidate for a later streaming-model evaluation: NVIDIA Parakeet Realtime EOU 120M. It is English-only and would need its own runtime integration and model validation.

## 6. Transcript output and developer experience

Depends on milestone 5.

Work:

- Expose word/segment timestamps where validated, while retaining token results. Do not present token emission times as validated word end times.
- Add SRT/VTT export with monotonic timestamps and tested handling of long/multilingual recordings.
- Complete consistent cancellation behavior across downloads, preparation and inference; document any operations that cannot stop immediately.
- Distinguish download, verification, compilation and ready states. Handle insufficient disk space, retries and interrupted preparation clearly.
- Add optional locale-aware normalization for numbers, dates and currencies, preserving raw recognized text alongside formatted output.
- Document keyword spotting separately from vocabulary biasing. Keep the current spotter; scope vocabulary biasing independently if demanded by users.
- Expand examples and API documentation and keep them building in CI.

Complete when:

- Fresh-app setup, first download, cached use, cancellation and recovery are covered by integration checks.
- Word/segment alignment and subtitle exports meet documented limits.
- Formatting has locale-specific fixtures and never silently replaces the raw transcript.

## 7. Speaker diarization for meeting applications

Depends on milestone 6.

Work:

- Evaluate a separate optional diarization model/pipeline using meeting recordings with speaker annotations.
- Start with offline speaker turns, then align transcript words/segments to anonymous speaker labels.
- Define behavior for overlap, unknown speaker count and short turns. Evaluate diarization error and speaker-attributed transcription quality.
- Keep streaming diarization as a later extension, with its own latency and label-stability requirements.

Complete when:

- Speaker boundaries, attribution quality, memory and latency meet declared limits on target devices.
- The API represents uncertain or overlapping assignments honestly.

Speaker labels such as Speaker 1 and Speaker 2 do not establish a person's identity.

## 8. Speech synthesis for voice applications

Depends on milestone 7 in this initial sequence. Reorder explicitly if voice assistants become the primary product goal.

Work:

- Select a TTS model using required languages, voice quality, licensing, model size and Apple-device performance.
- Add an optional synthesis API with cancellation and a separable playback adapter.
- Validate pronunciations, supported languages, first-audio latency, sustained synthesis speed and memory on real devices.
- Add incremental audio output if the chosen model supports it and the application requires it.

Complete when:

- Supported languages/voices and device limits are documented with reproducible evidence.
- Synthesis and playback lifecycle behavior passes integration checks.
- ASR-only users do not download TTS models.

Voice cloning is additional scope, not a requirement of the first TTS release.

## 9. Audio enhancement

Depends on milestone 8.

Work:

- Collect representative microphone failures and evaluate noise suppression against actual transcription outcomes.
- Evaluate echo cancellation separately for duplex applications, including availability and timing of the playback reference signal.
- Compare raw versus processed audio for accuracy, added delay, device load and speech damage.
- Expose enhancement as optional processing and preserve an unprocessed path.

Complete when:

- Processing improves predefined real-world workloads without unacceptable accuracy regressions or latency.
- Supported input/reference formats, device costs and limitations are documented.

## Deferred decisions

- Hindi/Hinglish and wider language coverage: evaluate an additional model family against the required languages and writing scripts. Whisper is a candidate, not an implementation commitment.
- Native streaming model: evaluate if the buffered session misses its targets.
- Vocabulary biasing: separate from detecting keywords in recorded audio.
- Online diarization, voice cloning and additional enhancement models: schedule after the initial corresponding capability is validated.

## Primary references

- [Current SDK](https://github.com/BuilderHelm/BuilderHelmAudio)
- [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [NVIDIA Parakeet Realtime EOU 120M](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)
- [Whisper language definitions](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)
