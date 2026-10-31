import Foundation
import Testing
@testable import BuilderHelmAudio
import ParakeetRuntime

extension RealModelTests {
    /// Compare SDK dispatch with the unchanged model runtime, including both modes and repeated requests.
    @Test(arguments: AsrModel.all)
    func adapterMatchesPublishedRuntime(model: AsrModel) async throws {
        let store = ModelStore.shared
        let directory = try await store.download(model)
        #expect(try await store.verify(model))
        let vocabulary = try JSONDecoder().decode([String].self, from: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
        let spotter = KeywordSpotter(vocabulary: vocabulary)
        for mode: DecodingMode in model.decodesTdt ? [.tdt, .ctc] : [.ctc] {
            let recognizer = try SpeechRecognizer(directory: directory, model: model, mode: mode,
                                                   computeUnits: realModelTestComputeUnits,
                                                   compiledModelsDirectory: store.compiledModelsDirectory)
            let direct = try ParakeetTdtCtc110M(directory: directory, computeUnits: realModelTestComputeUnits.coreML,
                                                ctcOnly: mode == .ctc, cacheDirectory: store.compiledModelsDirectory)
            for name in ["speech", "long", "speech"] {
                let url = Bundle.module.url(forResource: name, withExtension: "wav", subdirectory: "Fixtures")!
                let expected = try direct.transcribe(file: url, includeCtcLogProbs: true)
                let actual = try await recognizer.transcribe(file: url, includeCtcLogProbabilities: true)
                #expect(!expected.text.isEmpty, "Published runtime must recognize the speech fixture")
                #expect(!actual.text.isEmpty, "SDK runtime must recognize the speech fixture")
                #expect(actual.text == expected.text)
                #expect(actual.ctcTranscript == expected.ctcText)
                #expect(actual.tokens.map(\.time) == expected.tokenTimes)
                #expect(actual.tokens.map(\.text) == expected.tokens.map { vocabulary[$0].replacingOccurrences(of: "▁", with: " ") })
                let probabilities = try #require(actual.ctcLogProbabilities)
                #expect(probabilities.values == expected.ctcLogProbsFlat)
                let phrases = ["fox", "Neural Engine", "Maria", "lighthouse", "zebra"]
                let originalProbabilities = CtcLogProbabilities(values: try #require(expected.ctcLogProbsFlat))
                let priorHits = spotter.detect(phrases, in: originalProbabilities)
                let hits = recognizer.keywordSpotter.detect(phrases, in: probabilities)
                #expect(hits.map(\.keyword) == priorHits.map(\.keyword))
                #expect(hits.map(\.score) == priorHits.map(\.score))
                #expect(hits.map(\.start) == priorHits.map(\.start))
                #expect(hits.map(\.end) == priorHits.map(\.end))
            }
        }
    }
}
