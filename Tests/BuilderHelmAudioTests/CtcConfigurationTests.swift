import Testing
@testable import BuilderHelmAudio

@Test func legacyCtcConstructionRetains110MDefaults() {
    let values = [Float](repeating: -1, count: 2050)
    let probabilities = CtcLogProbabilities(values: values)
    #expect(probabilities.configuration == .parakeet110M)
    #expect(probabilities.frameCount == 2)
    #expect(Array(probabilities[1]) == Array(values[1025...]))
}

@Test(arguments: [0, 1, 2]) func keywordTimesUseModelLayoutIncludingNonfinalBlank(blank: Int) throws {
    let configuration = try CtcConfiguration(classCount: 3, blank: blank, frameDuration: 0.04)
    let ids = (0..<3).filter { $0 != blank }
    let frames = [blank, ids[0], ids[1], blank]
    var values = [Float](repeating: -20, count: 12)
    for (frame, token) in frames.enumerated() { values[frame * 3 + token] = -0.001 }
    let probabilities = try CtcLogProbabilities(values: values, configuration: configuration)
    let spotter = try KeywordSpotter(vocabulary: ["▁a", "a"], configuration: configuration)
    let hit = try #require(spotter.score("aa", in: probabilities))
    #expect(probabilities.frameCount == 4)
    #expect(Array(probabilities[1]) == Array(values[3..<6]))
    #expect(hit.score > -0.01)
    #expect(abs(hit.start - 0.04) < 1e-9)
    #expect(abs(hit.end - 0.12) < 1e-9)
}

@Test func invalidCtcLayoutAndPartialFramesAreRejected() throws {
    for (classes, blank, duration) in [(0, 0, 0.08), (1, 0, 0.08), (3, -1, 0.08), (3, 3, 0.08),
                                       (3, 0, 0), (3, 0, -.infinity), (3, 0, .nan)] {
        #expect(throws: BuilderHelmAudioError.self) { try CtcConfiguration(classCount: classes, blank: blank, frameDuration: duration) }
    }
    let configuration = try CtcConfiguration(classCount: 3, blank: 0, frameDuration: 0.04)
    #expect(throws: BuilderHelmAudioError.self) { try CtcLogProbabilities(values: [-1, -2], configuration: configuration) }
    #expect(throws: BuilderHelmAudioError.self) { try KeywordSpotter(vocabulary: ["▁a"], configuration: configuration) }
    let empty = try CtcLogProbabilities(values: [], configuration: configuration)
    #expect(empty.frameCount == 0)
    let probabilities = try CtcLogProbabilities(values: [-1, -2, -3], configuration: configuration)
    #expect(KeywordSpotter(vocabulary: ["▁a", "a"]).score("aa", in: probabilities) == nil)
}
