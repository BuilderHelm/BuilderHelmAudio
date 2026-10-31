import Foundation

/// CTC log-probabilities: one row of 1,025 values (1,024 tokens, then blank) every 80 ms.
public struct CtcLogProbabilities: Sendable {
    public static let classCount = 1025
    public static let blank = 1024
    public static let frameDuration = 0.08
    /// Frame-major values, `frameCount * 1025`.
    public let values: [Float]
    public var frameCount: Int { values.count / Self.classCount }

    public init(values: [Float]) {
        precondition(values.count % Self.classCount == 0, "values must hold whole frames of 1,025 classes")
        self.values = values
    }

    public subscript(frame: Int) -> ArraySlice<Float> {
        values[(frame * Self.classCount)..<((frame + 1) * Self.classCount)]
    }
}

/// A keyword found in the audio.
public struct KeywordHit: Sendable, Equatable {
    public let keyword: String
    /// Average log-probability per character of the best path. Closer to 0 is more confident.
    public let score: Float
    /// Start and end of the best path, in seconds from the start of the audio.
    public let start: Double
    public let end: Double
}

/// Finds words and phrases in CTC log-probabilities, without a tokenizer.
///
/// For each phrase it searches every way the vocabulary can spell it and keeps the best-scoring path
/// anywhere in the audio. Phrases are case-sensitive, as the vocabulary is.
public struct KeywordSpotter: Sendable {
    /// Default detection threshold. Spoken words usually score above −0.1 and absent ones below −2;
    /// tune it on your own audio.
    public static let defaultThreshold: Float = -1

    private let index: [String: Int]

    /// - Parameter vocabulary: the model's `vocab.json` pieces, in token order.
    public init(vocabulary: [String]) {
        index = Dictionary(vocabulary.enumerated().map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
    }

    /// Best hit for `phrase`, or nil when the vocabulary cannot spell it or the audio is empty.
    public func score(_ phrase: String, in probabilities: CtcLogProbabilities) -> KeywordHit? {
        let text = Array(phrase.split(separator: " ").map { "▁" + $0 }.joined())
        guard text.count > 1, probabilities.frameCount > 0 else { return nil }
        // Every vocabulary piece that spells text[from..<to].
        var arcs: [(from: Int, to: Int, id: Int)] = []
        for i in text.indices {
            for j in (i + 1)...text.count {
                if let id = index[String(text[i..<j])] { arcs.append((i, j, id)) }
            }
        }
        guard arcs.contains(where: { $0.to == text.count }) else { return nil }
        let incoming = arcs.indices.map { a in arcs.indices.filter { arcs[$0].to == arcs[a].from && arcs[$0].id != arcs[a].id } }
        let endingAt = (0...text.count).map { k in arcs.indices.filter { arcs[$0].to == k } }

        let none = -Float.infinity
        var onArc = [Float](repeating: none, count: arcs.count), arcStart = [Int](repeating: 0, count: arcs.count)
        var onBlank = [Float](repeating: none, count: text.count), blankStart = [Int](repeating: 0, count: text.count)
        var best = none, bestStart = 0, bestEnd = 0
        let values = probabilities.values, classes = CtcLogProbabilities.classCount
        for frame in 0..<probabilities.frameCount {
            let row = frame * classes
            var nextArc = onArc, nextArcStart = arcStart
            for a in arcs.indices {
                var previous = onArc[a], start = arcStart[a]
                if arcs[a].from == 0 {
                    if 0 > previous { previous = 0; start = frame }          // a keyword can start on any frame
                } else if onBlank[arcs[a].from] > previous {
                    previous = onBlank[arcs[a].from]; start = blankStart[arcs[a].from]
                }
                for b in incoming[a] where onArc[b] > previous { previous = onArc[b]; start = arcStart[b] }
                nextArc[a] = previous + values[row + arcs[a].id]
                nextArcStart[a] = start
            }
            var nextBlank = onBlank, nextBlankStart = blankStart
            for k in 1..<text.count {
                var previous = onBlank[k], start = blankStart[k]
                for b in endingAt[k] where onArc[b] > previous { previous = onArc[b]; start = arcStart[b] }
                nextBlank[k] = previous + values[row + CtcLogProbabilities.blank]
                nextBlankStart[k] = start
            }
            onArc = nextArc; arcStart = nextArcStart; onBlank = nextBlank; blankStart = nextBlankStart
            for a in endingAt[text.count] where onArc[a] > best { best = onArc[a]; bestStart = arcStart[a]; bestEnd = frame }
        }
        guard best > none else { return nil }
        return KeywordHit(keyword: phrase, score: best / Float(text.count),
                          start: Double(bestStart) * CtcLogProbabilities.frameDuration,
                          end: Double(bestEnd + 1) * CtcLogProbabilities.frameDuration)
    }

    /// Hits scoring at or above `threshold`, best first.
    public func detect(_ phrases: [String], in probabilities: CtcLogProbabilities,
                       threshold: Float = KeywordSpotter.defaultThreshold) -> [KeywordHit] {
        phrases.compactMap { score($0, in: probabilities) }.filter { $0.score >= threshold }.sorted { $0.score > $1.score }
    }
}
