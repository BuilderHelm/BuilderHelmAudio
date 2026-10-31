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
    /// Greedy CTC transcript from the same encoder pass.
    public let ctcText: String
    /// TDT tokens with times; empty in CTC mode.
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

/// Where the encoder runs. The Neural Engine is fastest on iPhone and Mac.
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

/// How a TDT + CTC model decodes. CTC-only models always use `.ctc`.
public enum DecodingMode: Sendable {
    /// TDT transcript and token times, plus the CTC transcript.
    case tdt
    /// CTC only: skips the TDT decoder.
    case ctc
}

/// Speech-to-text with a BuilderHelm Parakeet model.
///
/// ```swift
/// let recognizer = try await SpeechRecognizer.load(.parakeetTdtCtc110M)
/// let transcript = try await recognizer.transcribe(file: url)
/// print(transcript.text)
/// ```
///
/// Calls are queued and run one at a time off the caller's thread, so one instance can be shared.
public final class SpeechRecognizer: @unchecked Sendable {
    public let model: AsrModel?
    public let mode: DecodingMode
    /// Spotter for this model's vocabulary.
    public let keywordSpotter: KeywordSpotter

    private let runtime: ParakeetTdtCtc110M
    private let pieces: [String]
    private let queue = DispatchQueue(label: "BuilderHelmAudio.SpeechRecognizer", qos: .userInitiated)

    /// Downloads the model if needed, then loads it. The first load on a device also compiles it for CoreML.
    public static func load(_ model: AsrModel = .parakeetTdtCtc110M, mode: DecodingMode = .tdt,
                            computeUnits: ComputeUnits = .neuralEngine, store: ModelStore = .shared,
                            progress: (@Sendable (DownloadProgress) -> Void)? = nil) async throws -> SpeechRecognizer {
        let directory = try await store.download(model, progress: progress)
        let compiled = store.compiledModelsDirectory
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Result {
                    try SpeechRecognizer(directory: directory, model: model, mode: model.decodesTdt ? mode : .ctc,
                                         computeUnits: computeUnits, compiledModelsDirectory: compiled)
                })
            }
        }
    }

    /// Loads a model folder you manage yourself, for example one bundled with your app.
    /// Blocks while CoreML loads the model; call it off the main thread.
    public init(directory: URL, model: AsrModel? = nil, mode: DecodingMode = .tdt, computeUnits: ComputeUnits = .neuralEngine,
                compiledModelsDirectory: URL? = nil) throws {
        let ctcOnly = mode == .ctc || model?.decodesTdt == false
        do {
            runtime = try ParakeetTdtCtc110M(directory: directory, computeUnits: computeUnits.coreML, ctcOnly: ctcOnly,
                                             cacheDirectory: compiledModelsDirectory ?? ModelStore.defaultRoot
                                                 .appendingPathComponent("CompiledModels", isDirectory: true))
            pieces = try JSONDecoder().decode([String].self, from: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
        } catch {
            throw BuilderHelmAudioError.modelLoadFailed(String(describing: error))
        }
        self.model = model
        self.mode = ctcOnly ? .ctc : .tdt
        keywordSpotter = KeywordSpotter(vocabulary: pieces)
    }

    /// Transcribes an audio file in any format AVFoundation reads; it is converted to 16 kHz mono.
    /// The file is read in blocks, so long recordings don't need to fit in memory as audio.
    public func transcribe(file: URL, includeCtcLogProbabilities: Bool = false) async throws -> Transcript {
        try await run {
            let audio = try AVAudioFile(forReading: file)
            let duration = Double(audio.length) / audio.fileFormat.sampleRate
            let start = DispatchTime.now().uptimeNanoseconds
            let result = try self.runtime.transcribe(file: file, includeCtcLogProbs: includeCtcLogProbabilities)
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            return self.transcript(result, duration: duration, elapsed: elapsed)
        }
    }

    /// Transcribes 16 kHz mono samples in the range −1…1.
    public func transcribe(samples: [Float], includeCtcLogProbabilities: Bool = false) async throws -> Transcript {
        try await run {
            let start = DispatchTime.now().uptimeNanoseconds
            let result = try self.runtime.transcribe(samples, includeCtcLogProbs: includeCtcLogProbabilities)
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            return self.transcript(result, duration: Double(samples.count) / 16000, elapsed: elapsed)
        }
    }

    /// Transcribes the file and returns the keywords found in it, best first.
    public func spot(_ keywords: [String], inFile file: URL,
                     threshold: Float = KeywordSpotter.defaultThreshold) async throws -> [KeywordHit] {
        let transcript = try await transcribe(file: file, includeCtcLogProbabilities: true)
        guard let probabilities = transcript.ctcLogProbabilities else { return [] }
        return keywordSpotter.detect(keywords, in: probabilities, threshold: threshold)
    }

    /// Reads any AVFoundation-supported audio file as 16 kHz mono samples.
    public static func loadAudio(_ file: URL) throws -> [Float] {
        do { return try ParakeetAudio.read(file) } catch { throw BuilderHelmAudioError.unsupportedAudio(String(describing: error)) }
    }

    private func transcript(_ result: ParakeetResult, duration: Double, elapsed: Double) -> Transcript {
        let tokens = zip(result.tokens, result.tokenTimes).map { id, time in
            TimedToken(text: pieces.indices.contains(id) ? pieces[id].replacingOccurrences(of: "▁", with: " ") : "", time: time)
        }
        return Transcript(text: result.text, ctcText: result.ctcText, tokens: tokens, duration: duration,
                          processingTime: elapsed,
                          ctcLogProbabilities: result.ctcLogProbabilities.map { CtcLogProbabilities(values: $0.values) })
    }

    private func run<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { continuation.resume(with: Result { try work() }) }
        }
    }
}
