import CoreML
import Foundation
import ParakeetRuntime

final class ParakeetV3Runtime: AsrRuntime {
    let mode: DecodingMode = .tdt
    let capabilities: AsrCapabilities = .parakeetV3Tdt
    let keywordSpotter: KeywordSpotter? = nil
    let model: ParakeetTdtV3
    init(directory: URL, computeUnits: MLComputeUnits, cache: URL) throws {
        model = try ParakeetTdtV3(directory: directory, computeUnits: computeUnits, cacheDirectory: cache,
                                 decoderComputeUnits: .cpuOnly)
    }
    func transcribe(file: URL, request: RuntimeRequest) throws -> RuntimeResult {
        try convert(model.transcribe(file: file, checkCancellation: request.cancellation.check))
    }
    func transcribe(samples: [Float], request: RuntimeRequest) throws -> RuntimeResult {
        try convert(model.transcribe(samples, checkCancellation: request.cancellation.check))
    }
    func warmUp() throws { try model.warmUp() }
    func alternatives(samples: [Float], beamSize: Int, nbest: Int, request: RuntimeRequest) throws -> [TranscriptAlternative] {
        try model.alternatives(samples, beamSize: beamSize, nbest: nbest, checkCancellation: request.cancellation.check).map {
            TranscriptAlternative(text: $0.text, score: $0.score, normalizedScore: $0.normalizedScore)
        }
    }
    private func convert(_ result: ParakeetTdtV3.Result) -> RuntimeResult {
        let tokens = zip(result.tokenIDs, result.emissionFrames).map { id, frame in
            TimedToken(text: model.piece(id).replacingOccurrences(of: "\u{2581}", with: " "),
                       time: Double(frame) * ParakeetWords.frameSeconds)
        }
        let words = model.words(result).map {
            TimedWord(text: $0.text, start: $0.start, end: $0.end, confidence: $0.confidence)
        }
        return RuntimeResult(text: result.text, ctcText: nil, tokens: tokens, ctcLogProbabilities: nil, words: words)
    }
}
