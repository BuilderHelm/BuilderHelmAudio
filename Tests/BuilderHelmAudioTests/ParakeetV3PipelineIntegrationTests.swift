import CoreML
import Foundation
import Testing
@testable import BuilderHelmAudio
@testable import ParakeetRuntime

extension RealModelTests {
    /// Cancel after the pipeline has started, then reuse its mutable model buffers.
    @Test func v3PipelineCancellationDrainsAndPreservesSubsequentResults() async throws {
        let directory = try await ModelStore.shared.download(.parakeetTdtV3)
        let cache = ModelStore.shared.compiledModelsDirectory
        let fixture = try #require(Bundle.module.url(forResource: "long", withExtension: "wav", subdirectory: "Fixtures"))
        let samples = try SpeechRecognizer.loadAudio(fixture)
        try #require(samples.count > 480000)
        let expected = try autoreleasepool {
            let sequential = try ParakeetTdtV3(directory: directory, computeUnits: realModelTestComputeUnits.coreML,
                                              cacheDirectory: cache, decoderComputeUnits: .cpuOnly)
            return try sequential.transcribe(samples)
        }
        let adapter = try ParakeetV3Runtime(directory: directory, computeUnits: realModelTestComputeUnits.coreML, cache: cache)
        var checks = 0
        #expect(throws: CancellationError.self) {
            _ = try adapter.model.transcribe(samples, checkCancellation: {
                checks += 1
                if checks >= 20 { throw CancellationError() }
            })
        }
        #expect(checks >= 20)
        for result in [try adapter.model.transcribe(samples), try adapter.model.transcribe(file: fixture)] {
            #expect(result.text == expected.text)
            #expect(result.tokenIDs == expected.tokenIDs)
            #expect(result.emissionFrames == expected.emissionFrames)
        }
    }
}
