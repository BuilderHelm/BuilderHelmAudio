import Foundation

/// Cancellation is cooperative. A synchronous CoreML prediction may finish before cancellation is observed.
final class CancellationSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    var isCancelled: Bool { lock.withLock { cancelled } }
    func cancel() { lock.withLock { cancelled = true } }
    func check() throws { if isCancelled { throw CancellationError() } }
}

/// A queued operation observes cancellation before starting and before returning a result.
/// Completion waits for active synchronous work to finish; the runtime is never reused concurrently.
func runOnQueue<T: Sendable>(_ queue: DispatchQueue,
                            work: @escaping @Sendable (CancellationSignal) throws -> T) async throws -> T {
    let cancellation = CancellationSignal()
    return try await withTaskCancellationHandler {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    try cancellation.check()
                    let value = try work(cancellation)
                    try cancellation.check()
                    continuation.resume(returning: value)
                } catch {
                    continuation.resume(throwing: cancellation.isCancelled ? CancellationError() : error)
                }
            }
        }
    } onCancel: { cancellation.cancel() }
}

struct RuntimeRequest: Sendable {
    let includeCtcLogProbabilities: Bool
    let cancellation: CancellationSignal
}

/// An absent CTC head is nil, distinct from a present head recognizing silence as an empty string.
struct RuntimeResult: Sendable {
    let text: String
    let ctcText: String?
    let tokens: [TimedToken]
    let ctcLogProbabilities: CtcLogProbabilities?
    var words: [TimedWord] = []
}

/// Implementations are owned by a recognizer and used only on its serial queue.
/// Scratch inference state is request-local or reset by the adapter between calls.
protocol AsrRuntime: AnyObject {
    var mode: DecodingMode { get }
    var capabilities: AsrCapabilities { get }
    var keywordSpotter: KeywordSpotter? { get }
    func transcribe(file: URL, request: RuntimeRequest) throws -> RuntimeResult
    func transcribe(samples: [Float], request: RuntimeRequest) throws -> RuntimeResult
    /// Compiles and runs every model function once so the first real request doesn't pay for it.
    func warmUp() throws
    func alternatives(samples: [Float], beamSize: Int, nbest: Int, request: RuntimeRequest) throws -> [TranscriptAlternative]
}

extension AsrRuntime {
    func warmUp() throws { _ = try transcribe(samples: [Float](repeating: 0, count: 16000), request: RuntimeRequest(
        includeCtcLogProbabilities: false, cancellation: CancellationSignal())) }
    func alternatives(samples: [Float], beamSize: Int, nbest: Int, request: RuntimeRequest) throws -> [TranscriptAlternative] {
        throw BuilderHelmAudioError.unsupportedOperation("ranked alternatives")
    }
}

struct RuntimeSelection: Sendable {
    let architecture: AsrArchitecture
    let mode: DecodingMode
    let capabilities: AsrCapabilities
    let ctcConfiguration: CtcConfiguration?

    /// Validate before network access, compilation or runtime construction.
    static func resolve(model: AsrModel?, mode: DecodingMode) throws -> Self {
        let metadata = model ?? .parakeetTdtCtc110M
        let architecture = metadata.architecture
        if architecture == .parakeetV3 {
            guard mode == .tdt, metadata.capabilities == .parakeetV3Tdt, metadata.ctcConfiguration == nil,
                  Set(metadata.supportedLanguages) == Set(AsrModel.experimentalParakeetV3.supportedLanguages) else {
                throw BuilderHelmAudioError.invalidModelConfiguration("V3 supports TDT text with token and word timestamps; CTC is unavailable")
            }
            return Self(architecture: .parakeetV3, mode: .tdt, capabilities: .parakeetV3Tdt, ctcConfiguration: nil)
        }
        if architecture == .parakeetUnifiedEn {
            guard mode == .tdt, metadata.capabilities == .parakeetUnifiedRnnt, metadata.ctcConfiguration == nil,
                  metadata.supportedLanguages == ["en"] else {
                throw BuilderHelmAudioError.invalidModelConfiguration("Parakeet Unified supports English RNN-T text with token and word timestamps; CTC is unavailable")
            }
            return Self(architecture: .parakeetUnifiedEn, mode: .tdt, capabilities: .parakeetUnifiedRnnt, ctcConfiguration: nil)
        }
        guard architecture == .parakeet110M else {
            throw BuilderHelmAudioError.unsupportedArchitecture(architecture.rawValue)
        }
        let capabilities = metadata.capabilities
        let ctc = metadata.ctcConfiguration
        guard capabilities == .parakeet110MTdt || capabilities == .parakeet110MCtc,
              ctc == .parakeet110M, metadata.supportedLanguages == ["en"] else {
            throw BuilderHelmAudioError.invalidModelConfiguration("The 110M runtime requires its English TDT/CTC or CTC-only capabilities and CTC layout")
        }
        // Preserve the existing automatic CTC fallback for CTC-only packages.
        let effectiveMode: DecodingMode = capabilities.contains(.tdt) ? mode : .ctc
        return Self(architecture: architecture, mode: effectiveMode,
                    capabilities: effectiveMode == .ctc ? .parakeet110MCtc : capabilities,
                    ctcConfiguration: .parakeet110M)
    }
}

enum RuntimeFactory {
    static func make(selection: RuntimeSelection, directory: URL, computeUnits: ComputeUnits,
                     cache: URL) throws -> any AsrRuntime {
        switch selection.architecture {
        case .parakeetV3:
            return try ParakeetV3Runtime(directory: directory, computeUnits: computeUnits.coreML, cache: cache)
        case .parakeetUnifiedEn:
            return try ParakeetV3Runtime(directory: directory, computeUnits: computeUnits.coreML, cache: cache,
                                         capabilities: .parakeetUnifiedRnnt)
        case .parakeet110M:
            return try Parakeet110MRuntime(directory: directory, selection: selection,
                                          computeUnits: computeUnits.coreML, cache: cache)
        default:
            throw BuilderHelmAudioError.unsupportedArchitecture(selection.architecture.rawValue)
        }
    }
}
