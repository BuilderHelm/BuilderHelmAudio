import Foundation
import Testing
@testable import BuilderHelmAudio

/// Log-probabilities where `frames[i]` is the token emitted on frame i (blank elsewhere), confidence ~1.
private func probabilities(_ frames: [Int]) -> CtcLogProbabilities {
    var values = [Float](repeating: -20, count: frames.count * CtcLogProbabilities.classCount)
    for (frame, token) in frames.enumerated() { values[frame * CtcLogProbabilities.classCount + token] = -0.001 }
    return CtcLogProbabilities(values: values)
}

private let vocabulary: [String] = {
    var pieces = (0..<1024).map { "x\($0)" }
    pieces[10] = "▁fir"; pieces[11] = "▁f"; pieces[12] = "ir"; pieces[13] = "▁tree"; pieces[14] = "▁dog"; pieces[15] = "▁t"; pieces[16] = "ree"
    return pieces
}()

@Test func findsAKeywordAndItsTime() throws {
    let blank = CtcLogProbabilities.blank
    // "fir tree" spelled as ▁f ir ▁t ree, starting at frame 3.
    let probs = probabilities([blank, blank, blank, 11, 12, blank, 15, 16, blank, blank])
    let hit = try #require(KeywordSpotter(vocabulary: vocabulary).score("fir tree", in: probs))
    #expect(hit.score > -0.01)
    #expect(abs(hit.start - 0.24) < 1e-9)
    #expect(abs(hit.end - 0.64) < 1e-9)
}

@Test func rejectsAbsentAndUnspellableKeywords() {
    let blank = CtcLogProbabilities.blank
    let probs = probabilities([blank, 11, 12, blank, 13, blank])
    let spotter = KeywordSpotter(vocabulary: vocabulary)
    #expect(spotter.detect(["dog"], in: probs).isEmpty)
    #expect(spotter.score("zebra", in: probs) == nil)             // the vocabulary cannot spell it
    #expect(spotter.detect(["fir tree", "dog"], in: probs).map(\.keyword) == ["fir tree"])
}

@Test func repeatedTokenNeedsABlankBetween() {
    var pieces = vocabulary
    pieces[20] = "▁a"; pieces[21] = "a"
    let spotter = KeywordSpotter(vocabulary: pieces)
    let blank = CtcLogProbabilities.blank
    // "aa" = ▁a a; also check that a collapsed repeat does not count as two tokens.
    #expect(spotter.score("aa", in: probabilities([20, 21, blank]))!.score > -0.01)
    let joined = probabilities([20, 20, blank])          // ▁a ▁a collapses to one ▁a
    #expect(spotter.score("aa", in: joined)!.score < -1)
}
