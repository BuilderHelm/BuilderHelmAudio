import AVFoundation
import CoreML
import Foundation
import ParakeetRuntime

/// A token of the TDT transcript and when it was spoken.
public struct TimedToken: Sendable, Equatable {
    /// Token text; a leading space marks the start of a word.
    public let text: String
    /// Emission time in seconds from the start of the audio.
    public let time: Double
}

public struct Transcript: Sendable {
    /// TDT transcript, or the CTC transcript for CTC-only models and `.ctc` mode.
    public let text: String
    /// Greedy CTC transcript, or nil if the selected model has no CTC output.
    public let ctcTranscript: String?
    /// Compatibility accessor. Use `ctcTranscript` to distinguish absent CTC from recognized silence.
    public var ctcText: String { ctcTranscript ?? "" }
    /// TDT tokens with times when tokenTimestamps is supported; otherwise empty.
    public let tokens: [TimedToken]
    /// Audio length in seconds.
    public let duration: Double
    /// Transcription time in seconds, excluding model loading.
    public let processingTime: Double
    /// Present when requested with `includeCtcLogProbabilities: true`.
    public let ctcLogProbabilities: CtcLogProbabilities?
    /// Audio seconds per second of processing.
    public var realTimeFactor: Double { processingTime > 0 ? duration / processingTime : 0 }
}

/// Compute devices CoreML may use; actual placement and speed depend on the model and hardware.
public enum ComputeUnits: Sendable {
    case neuralEngine, gpu, cpu

    var coreML: MLComputeUnits {
        switch self {
        case .neuralEngine: .cpuAndNeuralEngine
        case .gpu: .cpuAndGPU
        case .cpu: .cpuOnly
        }
    }
}

/// How a model decodes. CTC-only models automatically select `.ctc` for compatibility.
public enum DecodingMode: Sendable {
    /// TDT transcript, with timestamps and CTC output when the selected model supports them.
    case tdt
    /// CTC only: skips the TDT decoder.
    case ctc
}

/// Speech-to-text with a BuilderHelm model, selected through its runtime architecture.
///
/// ```swift
/// let recognizer = try await SpeechRecognizer.load(.parakeetTdtCtc110M)
/// let transcript = try await recognizer.transcribe(file: url)
/// print(transcript.text)
/// ```
///
/// Calls are queued and run one at a time off the caller's thread, so one instance can be shared.
/// Cancellation is checked before queued work starts and after synchronous inference returns.
/// An active CoreML prediction is not interrupted; the next request starts only after it finishes.
public final class SpeechRecognizer: @unchecked Sendable {
    public let model: AsrModel?
    public let mode: DecodingMode
    /// Capabilities available in the selected decoding mode.
    public let capabilities: AsrCapabilities
    public var supportedLanguages: [String] { model?.supportedLanguages ?? ["en"] }
    /// Nil for models that do not support the CTC keyword spotter.
    public let keywordSpotterIfAvailable: KeywordSpotter?
    /// Legacy nonoptional access. An unavailable spotter returns no hits; prefer `spot` (throws)
    /// or `keywordSpotterIfAvailable` when using different architectures.
    public var keywordSpotter: KeywordSpotter { keywordSpotterIfAvailable ?? .unavailable }

    private let runtime: any AsrRuntime
    private let queue = DispatchQueue(label: "BuilderHelmAudio.SpeechRecognizer", qos: .userInitiated, autoreleaseFrequency: .workItem)

    /// Downloads the model if needed, then loads it. The first load on a device also compiles it for CoreML.
    public static func load(_ model: AsrModel = .parakeetTdtCtc110M, mode: DecodingMode = .tdt,
                            computeUnits: ComputeUnits = .neuralEngine, store: ModelStore = .shared,
                            progress: (@Sendable (DownloadProgress) -> Void)? = nil) async throws -> SpeechRecognizer {
        try Task.checkCancellation()
        guard !model.repository.isEmpty else {
            throw BuilderHelmAudioError.invalidModelConfiguration("Experimental v3 is local-only; use SpeechRecognizer(directory:model:)")
        }
        _ = try RuntimeSelection.resolve(model: model, mode: mode)
        let directory = try await store.download(model, progress: progress)
        let compiled = store.compiledModelsDirectory
        return try await runOnQueue(.global(qos: .userInitiated)) { _ in
            try SpeechRecognizer(directory: directory, model: model, mode: mode,
                                 computeUnits: computeUnits, compiledModelsDirectory: compiled)
        }
    }

    /// Loads a model folder you manage yourself, for example one bundled with your app.
    /// Blocks while CoreML loads the model; call it off the main thread.
    public convenience init(directory: URL, model: AsrModel? = nil, mode: DecodingMode = .tdt, computeUnits: ComputeUnits = .neuralEngine,
                compiledModelsDirectory: URL? = nil) throws {
        // No descriptor retains the legacy bundled-110M behavior. Other architectures require a descriptor.
        let selection = try RuntimeSelection.resolve(model: model, mode: mode)
        let runtime: any AsrRuntime
        do {
            runtime = try RuntimeFactory.make(selection: selection, directory: directory, computeUnits: computeUnits,
                                              cache: compiledModelsDirectory ?? ModelStore.defaultRoot
                                                .appendingPathComponent("CompiledModels", isDirectory: true))
        } catch {
            throw BuilderHelmAudioError.modelLoadFailed(String(describing: error))
        }
        self.init(runtime: runtime, model: model)
    }

    init(runtime: any AsrRuntime, model: AsrModel? = nil) {
        self.runtime = runtime
        self.model = model
        mode = runtime.mode
        capabilities = runtime.capabilities
        keywordSpotterIfAvailable = runtime.keywordSpotter
    }

    /// Transcribes an audio file in any format AVFoundation reads; it is converted to 16 kHz mono.
    /// The file is read in blocks, so long recordings don't need to fit in memory as audio.
    public func transcribe(file: URL, includeCtcLogProbabilities: Bool = false) async throws -> Transcript {
        try validateRequest(includeCtcLogProbabilities: includeCtcLogProbabilities)
        return try await runOnQueue(queue) { cancellation in
            let audio = try AVAudioFile(forReading: file)
            let duration = Double(audio.length) / audio.fileFormat.sampleRate
            let start = DispatchTime.now().uptimeNanoseconds
            let result = try self.runtime.transcribe(file: file, request: RuntimeRequest(
                includeCtcLogProbabilities: includeCtcLogProbabilities, cancellation: cancellation))
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            return self.transcript(result, duration: duration, elapsed: elapsed)
        }
    }

    /// Transcribes 16 kHz mono samples in the range −1…1.
    public func transcribe(samples: [Float], includeCtcLogProbabilities: Bool = false) async throws -> Transcript {
        try validateRequest(includeCtcLogProbabilities: includeCtcLogProbabilities)
        return try await runOnQueue(queue) { cancellation in
            let start = DispatchTime.now().uptimeNanoseconds
            let result = try self.runtime.transcribe(samples: samples, request: RuntimeRequest(
                includeCtcLogProbabilities: includeCtcLogProbabilities, cancellation: cancellation))
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            return self.transcript(result, duration: Double(samples.count) / 16000, elapsed: elapsed)
        }
    }

    /// Transcribes the file and returns the keywords found in it, best first.
    public func spot(_ keywords: [String], inFile file: URL,
                     threshold: Float = KeywordSpotter.defaultThreshold) async throws -> [KeywordHit] {
        guard capabilities.contains(.keywordSpotting), let spotter = keywordSpotterIfAvailable else {
            throw BuilderHelmAudioError.unsupportedOperation("keyword spotting")
        }
        let transcript = try await transcribe(file: file, includeCtcLogProbabilities: true)
        guard let probabilities = transcript.ctcLogProbabilities else {
            throw BuilderHelmAudioError.invalidModelConfiguration("The keyword runtime did not return requested CTC probabilities")
        }
        return spotter.detect(keywords, in: probabilities, threshold: threshold)
    }

    /// Reads any AVFoundation-supported audio file as 16 kHz mono samples.
    public static func loadAudio(_ file: URL) throws -> [Float] {
        do { return try ParakeetAudio.read(file) } catch { throw BuilderHelmAudioError.unsupportedAudio(String(describing: error)) }
    }

    private func transcript(_ result: RuntimeResult, duration: Double, elapsed: Double) -> Transcript {
        Transcript(text: result.text, ctcTranscript: result.ctcText, tokens: result.tokens, duration: duration,
                          processingTime: elapsed,
                          ctcLogProbabilities: result.ctcLogProbabilities)
    }

    private func validateRequest(includeCtcLogProbabilities: Bool) throws {
        if includeCtcLogProbabilities && !capabilities.contains(.ctc) {
            throw BuilderHelmAudioError.unsupportedOperation("CTC probabilities")
        }
    }
}
