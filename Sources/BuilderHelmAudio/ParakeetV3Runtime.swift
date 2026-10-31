import CoreML
import Foundation
import ParakeetRuntime

final class ParakeetV3Runtime: AsrRuntime {
    let mode: DecodingMode = .tdt
    let capabilities: AsrCapabilities = [.tdt]
    let keywordSpotter: KeywordSpotter? = nil
    private let model: ParakeetTdtV3
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
    private func convert(_ result: ParakeetTdtV3.Result) -> RuntimeResult {
        RuntimeResult(text: result.text, ctcText: nil, tokens: [], ctcLogProbabilities: nil)
    }
}
