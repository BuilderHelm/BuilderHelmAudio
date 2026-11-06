import CoreML
import Testing
@testable import ParakeetRuntime

// Pieces: 0 <unk>, 1 ▁Well, 2 ",", 3 ▁I, 4 ▁don, 5 "'", 6 t, 7 ▁know, 8 ".", 9 ▁▁ (special)
private let pieces = ["<unk>", "\u{2581}Well", ",", "\u{2581}I", "\u{2581}don", "'", "t", "\u{2581}know", ".", "\u{2581}\u{2581}"]

@Test func punctuationComesFromNonSpecialPieces() {
    #expect(ParakeetWords.punctuation(in: pieces) == [",", "'", "."])
    #expect(ParakeetWords.punctuation(in: ["\u{2581},", "<,>", "[.]", "##!"]).isEmpty)
}

@Test func wordsFollowNeMoGroupingTimingAndMinConfidence() {
    let result = ParakeetTdtV3.Result(
        text: "Well, I don't know.",
        tokenIDs: [1, 2, 3, 4, 5, 6, 7, 8],
        emissionFrames: [6, 8, 10, 12, 14, 15, 17, 20],
        durations: [2, 1, 1, 2, 1, 1, 3, 1],
        confidences: [0.9, 0.8, 0.95, 0.7, 0.6, 0.99, 0.85, 0.5])
    let words = ParakeetWords.group(result, pieces: pieces, punctuation: ParakeetWords.punctuation(in: pieces))
    #expect(words.map(\.text) == ["Well,", "I", "don't", "know."])
    // "Well," starts at its first frame; the comma is pinned to "Well"'s end (6 + 2 = 8).
    #expect(words[0].start == 6 * 0.08 && words[0].end == 8 * 0.08)
    #expect(words[0].confidence == 0.8)
    #expect(words[1].start == 10 * 0.08 && words[1].end == 11 * 0.08)
    // The apostrophe is punctuation, pinned to "don"'s end (14); "t" ends at 15 + 1.
    #expect(words[2].start == 12 * 0.08 && words[2].end == 16 * 0.08)
    #expect(words[2].confidence == 0.6)
    // The final period is pinned to "know"'s end (17 + 3 = 20).
    #expect(words[3].end == 20 * 0.08 && words[3].confidence == 0.5)
}

@Test func leadingPunctuationAndEmptyResultsAreSafe() {
    let empty = ParakeetTdtV3.Result(text: "", tokenIDs: [], emissionFrames: [])
    #expect(ParakeetWords.group(empty, pieces: pieces, punctuation: [","]).isEmpty)
    let lone = ParakeetTdtV3.Result(text: ".", tokenIDs: [8], emissionFrames: [3], durations: [1], confidences: [0.4])
    let words = ParakeetWords.group(lone, pieces: pieces, punctuation: ["."])
    #expect(words == [ParakeetWord(text: ".", start: 3 * 0.08, end: 4 * 0.08, confidence: 0.4)])
}

@Test func maxProbConfidenceMatchesNeMoFormula() throws {
    let logits = try MLMultiArray(shape: [1, 1, 1, 6], dataType: .float32)
    for (i, v) in [2.0, 0.5, -1.0, 0.0, 9.0, 9.0].enumerated() { logits[i] = NSNumber(value: v) }
    // Over the first 4 logits only (tokens + blank; durations excluded).
    let values = [2.0, 0.5, -1.0, 0.0]
    let p = exp(2.0) / values.map(exp).reduce(0, +)
    let expected = Float((4 * p - 1) / 3)
    let confidence = try ParakeetTdtV3.maxProbConfidence(logits, token: 0, count: 4)
    #expect(abs(confidence - expected) < 1e-6)
    // A uniform distribution has zero confidence.
    for i in 0..<4 { logits[i] = 1 }
    #expect(abs(try ParakeetTdtV3.maxProbConfidence(logits, token: 2, count: 4)) < 1e-6)
}

@Test func recoveryAndSplitsCarryTokenMetadata() throws {
    let primary = ParakeetTdtV3.Result(text: "", tokenIDs: [1, 2, 3, 8, 9, 10], emissionFrames: [0, 1, 2, 100, 101, 102],
                                       durations: [1, 2, 3, 4, 0, 1], confidences: [0.1, 0.2, 0.3, 0.8, 0.9, 1])
    let candidate = ParakeetTdtV3.Result(text: "", tokenIDs: Array(1...10), emissionFrames: [0, 1, 2, 20, 40, 60, 80, 100, 101, 102],
                                         durations: Array(repeating: 2, count: 10), confidences: Array(repeating: 0.5, count: 10))
    let gaps = [ParakeetTdtV3.BlankSpan(index: 3, left: 2, right: 100)]
    let recovered = try #require(ParakeetTdtV3.insertRecoveredTokens(primary, candidate: candidate, gaps: gaps))
    #expect(recovered.durations == [1, 2, 3, 2, 2, 2, 2, 4, 0, 1])
    #expect(recovered.confidences == [0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 0.9, 1])
    var track = ParakeetTdtV3.TokenTrack()
    track.append(primary)
    track.append(primary, frameOffset: 50)
    #expect(track.frames.suffix(3) == [150, 151, 152])
    #expect(track.durations == primary.durations + primary.durations)
}

@Test func beamHelpersMatchParakeetCpp() {
    let logp = ParakeetTdtV3.logSoftmax([1, 2, 3][...])
    #expect(abs(logp.map { exp(Double($0)) }.reduce(0, +) - 1) < 1e-6)
    #expect(ParakeetTdtV3.topK([0.5, 2, 2, -1], count: 4, 3).map(\.index) == [1, 2, 0])
    #expect(ParakeetTdtV3.topK([0.5, 2], count: 2, 5).count == 2)
    #expect(abs(ParakeetTdtV3.logAddExp(log(0.25), log(0.5)) - log(Float(0.75))) < 1e-6)
}
