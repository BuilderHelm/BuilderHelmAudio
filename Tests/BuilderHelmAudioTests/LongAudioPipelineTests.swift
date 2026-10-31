import Foundation
import Testing
import BuilderHelmAudio
import ParakeetRuntime

private let enabled = ProcessInfo.processInfo.environment["BUILDERHELM_INTEGRATION"] == "1"
private let long = Bundle.module.url(forResource: "long", withExtension: "wav", subdirectory: "Fixtures")!

/// The CPU/Neural Engine pipeline and the smallest-fit final window must not change CTC output.
@Test(.enabled(if: enabled))
func ctcPipelineMatchesSequentialDecoding() async throws {
    let directory = try await ModelStore.shared.download(.parakeetCtc110M)
    let cache = ModelStore.shared.compiledModelsDirectory
    let samples = try SpeechRecognizer.loadAudio(long)
    var texts: [String: String] = [:]
    for (name, fit, pipeline) in [("plain", false, false), ("pipeline", false, true), ("fit", true, false), ("shipped", true, true)] {
        let runtime = try ParakeetTdtCtc110M(directory: directory, ctcOnly: true, cacheDirectory: cache,
                                             ctcFitLastWindow: fit, ctcPipeline: pipeline)
        let first = try runtime.transcribe(samples)
        let again = try runtime.transcribe(samples)
        #expect(first.ctcText == again.ctcText, "\(name) is not deterministic")
        #expect(first.ctcLogProbsFlat == again.ctcLogProbsFlat, "\(name) probabilities are not deterministic")
        texts[name] = first.ctcText
    }
    #expect(texts["pipeline"] == texts["plain"])
    #expect(texts["shipped"] == texts["fit"])
}
