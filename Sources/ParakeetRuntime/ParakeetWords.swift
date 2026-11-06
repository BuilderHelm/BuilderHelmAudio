import Foundation

/// A word with NeMo-compatible timing and confidence.
public struct ParakeetWord: Sendable, Equatable {
    public let text: String
    /// Seconds from the start of the audio.
    public let start: Double
    public let end: Double
    /// Lowest token confidence in the word (NeMo's `min` aggregate), (0, 1].
    public let confidence: Float
}

/// Groups decoded SentencePiece tokens into words, as NeMo's `get_words_offsets` plus
/// `_refine_timestamps_tdt` do (`timestamps=True`). A port of parakeet.cpp's `pk::group_words`,
/// which matches NeMo exactly: a word starts at a `▁` piece, supported punctuation joins the
/// word before it, start = first token's frame, end = last token's frame + its duration.
public enum ParakeetWords {
    /// Encoder frame length: hop 160 × subsampling 8 / 16 kHz.
    public static let frameSeconds = 0.08
    private static let metaSpace = "\u{2581}"

    /// NeMo `extract_punctuation_from_vocab`: single ASCII punctuation characters (Unicode
    /// category P) found in pieces that are not special (bracketed, `##`, `▁`-prefixed or blank).
    public static func punctuation(in pieces: [String]) -> Set<String> {
        let marks = Set("!\"#%&'()*,-./:;?@[\\]_{}")
        var found = Set<String>()
        for piece in pieces where !isSpecial(piece) {
            for character in piece where marks.contains(character) { found.insert(String(character)) }
        }
        return found
    }

    public static func group(_ result: ParakeetTdtV3.Result, pieces: [String], punctuation: Set<String>) -> [ParakeetWord] {
        let n = result.tokenIDs.count
        guard n > 0, result.emissionFrames.count == n, result.durations.count == n, result.confidences.count == n else { return [] }
        let delimiter = " "
        let token = result.tokenIDs.map { pieces.indices.contains($0) ? pieces[$0] : "" }
        let text = token.map(pieceText)
        var start = result.emissionFrames
        var end = zip(result.emissionFrames, result.durations).map { $0 + $1 }
        let confidence = result.confidences
        func isPunctuation(_ value: String) -> Bool { !value.isEmpty && value != delimiter && punctuation.contains(value) }

        // TDT refinement: a token that begins with punctuation is pinned to the previous token's end.
        for i in 1..<max(1, n) where !text[i].isEmpty && isPunctuation(String(text[i].first!)) {
            start[i] = end[i - 1]
            end[i] = start[i]
        }

        func word(_ indices: [Int], from first: Int) -> ParakeetWord {
            ParakeetWord(text: detokenize(indices.map { result.tokenIDs[$0] }, pieces: pieces),
                         start: Double(start[first]) * frameSeconds,
                         end: Double(end[indices.last!]) * frameSeconds,
                         confidence: indices.map { confidence[$0] }.min()!)
        }

        var words: [ParakeetWord] = []
        var built: [Int] = []
        var previous = 0
        for i in 0..<n {
            let current = text[i]
            let currentIsPunctuation = isPunctuation(current)
            var nextText = ""
            var j = i
            while nextText.isEmpty && j < n - 1 {
                j += 1
                if text[j] != delimiter { nextText = text[j] }
            }
            let nextIsPunctuation = !nextText.isEmpty && isPunctuation(nextText)
            let startsWord = token[i] != current || (current == delimiter && !nextIsPunctuation)

            if startsWord && !currentIsPunctuation {
                if !built.isEmpty { words.append(word(built, from: previous)) }
                built.removeAll()
                if current != delimiter {
                    built.append(i)
                    previous = i
                }
            } else if currentIsPunctuation && built.isEmpty && !words.isEmpty {
                // Punctuation with no open word attaches to the previous word.
                let last = words.removeLast()
                var joined = last.text
                if joined.hasSuffix(" ") { joined.removeLast() }
                words.append(ParakeetWord(text: joined + current, start: last.start,
                                          end: Double(end[i]) * frameSeconds,
                                          confidence: min(last.confidence, confidence[i])))
            } else if currentIsPunctuation && !built.isEmpty {
                if let last = built.last, [" ", "_", metaSpace].contains(token[last]) { built.removeLast() }
                built.append(i)
            } else {
                if built.isEmpty { previous = i }
                built.append(i)
            }
        }

        if !words.isEmpty {
            let first = words[0]
            words[0] = ParakeetWord(text: first.text, start: Double(start[0]) * frameSeconds, end: first.end, confidence: first.confidence)
            if !built.isEmpty { words.append(word(built, from: previous)) }
        } else if !built.isEmpty {
            words.append(word(built, from: 0))
        }
        return words
    }

    /// One piece's text: `▁` becomes a space, then one leading space is dropped.
    static func pieceText(_ piece: String) -> String {
        var value = piece.replacingOccurrences(of: metaSpace, with: " ")
        if value.hasPrefix(" ") { value.removeFirst() }
        return value
    }

    /// SentencePiece decoding without NeMo's punctuation-spacing pass, as parakeet.cpp's words use.
    static func detokenize(_ ids: [Int], pieces: [String]) -> String {
        pieceText(ids.map { pieces.indices.contains($0) ? pieces[$0] : "" }.joined())
    }

    private static func isSpecial(_ piece: String) -> Bool {
        if piece.isEmpty || piece.allSatisfy(\.isWhitespace) { return true }
        if piece.hasPrefix("[") && piece.hasSuffix("]") { return true }
        if piece.hasPrefix("<") && piece.hasSuffix(">") { return true }
        return piece.hasPrefix("##") || piece.hasPrefix(metaSpace)
    }
}
