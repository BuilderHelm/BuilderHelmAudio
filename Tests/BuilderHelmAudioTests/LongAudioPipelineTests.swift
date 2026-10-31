import Foundation
import Testing
@testable import BuilderHelmAudio
import ParakeetRuntime

private let long = Bundle.module.url(forResource: "long", withExtension: "wav", subdirectory: "Fixtures")!

extension RealModelTests {
/// The CPU/Neural Engine pipeline and the smallest-fit final window must not change CTC output.
@Test
func ctcPipelineMatchesSequentialDecoding() async throws {
    let directory = try await ModelStore.shared.download(.parakeetCtc110M)
    let cache = ModelStore.shared.compiledModelsDirectory
    let samples = try SpeechRecognizer.loadAudio(long)
    var texts: [String: String] = [:]
    for (name, fit, pipeline) in [("plain", false, false), ("pipeline", false, true), ("fit", true, false), ("shipped", true, true)] {
        let runtime = try ParakeetTdtCtc110M(directory: directory, computeUnits: realModelTestComputeUnits.coreML,
                                             ctcOnly: true, cacheDirectory: cache,
                                             ctcFitLastWindow: fit, ctcPipeline: pipeline)
        let first = try runtime.transcribe(samples)
        let again = try runtime.transcribe(samples)
        #expect(first.ctcText.lowercased().contains("she began her article"), "\(name) dropped the end of the recording")
        #expect(first.ctcText == again.ctcText, "\(name) is not deterministic")
        let sameProbabilities = first.ctcLogProbsFlat == again.ctcLogProbsFlat
        #expect(sameProbabilities, "\(name) probabilities are not deterministic")
        texts[name] = first.ctcText
    }
    #expect(texts["pipeline"] == texts["plain"])
    #expect(texts["shipped"] == texts["fit"])
}
}
