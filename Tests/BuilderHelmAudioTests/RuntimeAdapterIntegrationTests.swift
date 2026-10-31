import Foundation
import Testing
@testable import BuilderHelmAudio
import ParakeetRuntime

extension RealModelTests {
    /// Compare SDK dispatch with the model runtime, including both modes and repeated requests.
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
                let sameProbabilities = probabilities.values == expected.ctcLogProbsFlat
                #expect(sameProbabilities)
                #expect(greedyText(probabilities, vocabulary: vocabulary) == actual.ctcText,
                        "CTC text must agree with its returned probability frames")
                let phrases = ["fox", "Neural Engine", "Maria", "lighthouse", "zebra"]
                let originalProbabilities = CtcLogProbabilities(values: try #require(expected.ctcLogProbsFlat))
                #expect(greedyText(originalProbabilities, vocabulary: vocabulary) == expected.ctcText,
                        "Reference CTC text must agree with its returned probability frames")
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

private func greedyText(_ probabilities: CtcLogProbabilities, vocabulary: [String]) -> String {
    let classes = probabilities.configuration.classCount, blank = probabilities.configuration.blank
    var previous = blank, pieces: [String] = []
    probabilities.values.withUnsafeBufferPointer { values in
        for frame in 0..<probabilities.frameCount {
            let start = frame * classes
            var best = 0
            for token in 1..<classes where values[start + token] > values[start + best] { best = token }
            if best != blank && best != previous { pieces.append(vocabulary[best < blank ? best : best - 1]) }
            previous = best
        }
    }
    return pieces.joined().replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
}
