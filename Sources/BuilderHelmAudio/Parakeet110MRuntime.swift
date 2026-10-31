import CoreML
import Foundation
import ParakeetRuntime

/// Adapter around the shared 110M inference implementation.
final class Parakeet110MRuntime: AsrRuntime {
    let mode: DecodingMode
    let capabilities: AsrCapabilities
    let keywordSpotter: KeywordSpotter?
    private let model: ParakeetTdtCtc110M
    private let pieces: [String]
    private let ctcConfiguration: CtcConfiguration

    init(directory: URL, selection: RuntimeSelection, computeUnits: MLComputeUnits, cache: URL) throws {
        guard let ctcConfiguration = selection.ctcConfiguration else {
            throw BuilderHelmAudioError.invalidModelConfiguration("The 110M adapter requires a CTC layout")
        }
        mode = selection.mode
        capabilities = selection.capabilities
        self.ctcConfiguration = ctcConfiguration
        model = try ParakeetTdtCtc110M(directory: directory, computeUnits: computeUnits,
                                     ctcOnly: mode == .ctc, cacheDirectory: cache)
        pieces = try JSONDecoder().decode([String].self, from: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
        keywordSpotter = try KeywordSpotter(vocabulary: pieces, configuration: ctcConfiguration)
    }

    func transcribe(file: URL, request: RuntimeRequest) throws -> RuntimeResult {
        try request.cancellation.check()
        let result = try model.transcribe(file: file, includeCtcLogProbs: request.includeCtcLogProbabilities)
        try request.cancellation.check()
        return try convert(result)
    }

    func transcribe(samples: [Float], request: RuntimeRequest) throws -> RuntimeResult {
        try request.cancellation.check()
        let result = try model.transcribe(samples, includeCtcLogProbs: request.includeCtcLogProbabilities)
        try request.cancellation.check()
        return try convert(result)
    }

    private func convert(_ result: ParakeetResult) throws -> RuntimeResult {
        let tokens = zip(result.tokens, result.tokenTimes).map { id, time in
            TimedToken(text: pieces.indices.contains(id) ? pieces[id].replacingOccurrences(of: "▁", with: " ") : "", time: time)
        }
        return RuntimeResult(text: result.text, ctcText: result.ctcText, tokens: tokens,
                             ctcLogProbabilities: try result.ctcLogProbabilities.map {
                                 try CtcLogProbabilities(values: $0.values, configuration: ctcConfiguration)
                             })
    }
}
