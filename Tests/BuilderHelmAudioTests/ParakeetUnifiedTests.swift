import Foundation
import Testing
@testable import BuilderHelmAudio
@testable import ParakeetRuntime

@Test func unifiedSelectionHasTruthfulCapabilitiesAndRejectsCtc() async throws {
    let model = AsrModel.experimentalParakeetUnifiedEn
    let selection = try RuntimeSelection.resolve(model: model, mode: .tdt)
    #expect(selection.architecture == .parakeetUnifiedEn)
    #expect(selection.capabilities == [.rnnt, .tokenTimestamps, .wordTimestamps])
    #expect(!model.decodesTdt && !selection.capabilities.contains(.rankedAlternatives))
    #expect(selection.ctcConfiguration == nil && model.supportedLanguages == ["en"])
    #expect(!AsrModel.all.contains(model) && AsrModel.local.contains(model))
    #expect(throws: BuilderHelmAudioError.self) { try RuntimeSelection.resolve(model: model, mode: .ctc) }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    await #expect(throws: BuilderHelmAudioError.self) { try await SpeechRecognizer.load(model, store: ModelStore(root: root)) }
    #expect(!FileManager.default.fileExists(atPath: root.path))
}

private func config(_ fields: [String: Any]) throws -> ParakeetTdtV3.Config {
    var base: [String: Any] = ["precision": "fp16", "sample_rate": 16000, "window_samples": 240000, "max_symbols": 10,
                               "punctuation": [",", "."], "files": [String: String]()]
    base.merge(fields) { _, new in new }
    return try JSONDecoder().decode(ParakeetTdtV3.Config.self, from: JSONSerialization.data(withJSONObject: base))
}

@Test func transducerContractComesFromConfigAndRejectsMismatches() throws {
    let v3: [String: Any] = ["architecture": "parakeet-tdt-0.6b-v3", "blank_id": 8192, "vocabulary_size": 8192, "durations": [0, 1, 2, 3, 4]]
    let unified: [String: Any] = ["architecture": "parakeet-unified-en-0.6b", "decoder": "rnnt", "blank_id": 1024, "vocabulary_size": 1024]
    #expect(ParakeetTdtV3.Contract.resolve(try config(v3)) == .v3)
    #expect(ParakeetTdtV3.Contract.resolve(try config(v3.merging(["decoder": "tdt"]) { $1 })) == .v3)
    let rnnt = try #require(ParakeetTdtV3.Contract.resolve(try config(unified)))
    #expect(rnnt == .unifiedEn && rnnt.isRNNT && rnnt.logitCount == 1025)
    #expect(ParakeetTdtV3.Contract.v3.logitCount == 8198 && !ParakeetTdtV3.Contract.v3.isRNNT)
    for bad in [unified.merging(["decoder": "tdt"]) { $1 },                      // Unified is never TDT
                unified.filter { $0.key != "decoder" },                            // decoder must be explicit
                unified.merging(["durations": [0, 1, 2, 3, 4]]) { $1 },            // no duration head
                unified.merging(["blank_id": 8192, "vocabulary_size": 8192]) { $1 },
                v3.merging(["decoder": "rnnt"]) { $1 },
                v3.merging(["durations": [0, 1, 2]]) { $1 }] {
        #expect(ParakeetTdtV3.Contract.resolve(try config(bad)) == nil)
    }
}

// Pieces for NeMo's own Unified timestamps of "…her daughter, then…" (LibriSpeech 6930-75918-0004).
private let unifiedPieces = ["<unk>", "\u{2581}her", "\u{2581}d", "a", "ught", "er", ",", "\u{2581}the", "n"]

@Test func rnntWordsKeepPunctuationFramesAsNeMoDoes() {
    let result = ParakeetTdtV3.Result(
        text: "her daughter, then,", tokenIDs: [1, 2, 3, 4, 5, 6, 7, 8, 6],
        emissionFrames: [16, 17, 17, 17, 17, 25, 26, 27, 31], durations: Array(repeating: 1, count: 9),
        confidences: [0.9, 0.8, 0.95, 0.7, 0.6, 0.99, 0.85, 0.5, 0.75])
    let punctuation = ParakeetWords.punctuation(in: unifiedPieces)
    let words = ParakeetWords.group(result, pieces: unifiedPieces, punctuation: punctuation, refinePunctuation: false)
    // NeMo: her 16–17, daughter, 17–26, then, 26–32 (no RNN-T refinement; tokens span one frame).
    #expect(words.map(\.text) == ["her", "daughter,", "then,"])
    #expect(words.map { [$0.start / 0.08, $0.end / 0.08].map { $0.rounded() } } == [[16, 17], [17, 26], [26, 32]])
    #expect(words[1].confidence == 0.6 && words[2].confidence == 0.5)
    // TDT refinement would pin the comma to "er"'s end and end "daughter," at 18.
    let refined = ParakeetWords.group(result, pieces: unifiedPieces, punctuation: punctuation)
    #expect((refined[1].end / 0.08).rounded() == 18)
}

private let unifiedDirectory = ProcessInfo.processInfo.environment["BUILDERHELM_UNIFIED_DIR"]

/// Set BUILDERHELM_UNIFIED_DIR to a built release folder (or an assembled INT8 variant folder).
@Test(.enabled(if: unifiedDirectory != nil)) func unifiedRealModelTranscribesWithWords() async throws {
    let recognizer = try SpeechRecognizer(directory: URL(fileURLWithPath: unifiedDirectory!), model: .experimentalParakeetUnifiedEn,
                                          computeUnits: realModelTestComputeUnits)
    #expect(recognizer.capabilities.contains(.rnnt))
    let speech = try #require(Bundle.module.url(forResource: "speech", withExtension: "wav", subdirectory: "Fixtures"))
    let transcript = try await recognizer.transcribe(file: speech)
    #expect(!transcript.text.isEmpty && !transcript.words.isEmpty)
    #expect(transcript.words.allSatisfy { $0.end >= $0.start && $0.confidence > 0 && $0.confidence <= 1 })
    #expect(transcript.tokens.count >= transcript.words.count)
    await #expect(throws: BuilderHelmAudioError.self) { try await recognizer.alternatives(samples: [Float](repeating: 0, count: 16000)) }
    // The V3 descriptor must not load a Unified folder.
    #expect(throws: BuilderHelmAudioError.self) {
        try SpeechRecognizer(directory: URL(fileURLWithPath: unifiedDirectory!), model: .experimentalParakeetV3)
    }
}

@Test func windowLadderPicksTheSmallestWindowThatHoldsTheRequest() {
    let ladder = [80000, 160000, 240000, 320000, 480000]
    let picks = [1, 80000, 80001, 160000, 200000, 240000, 240001, 320000, 320001, 480000, 480001, 2_000_000]
        .map { ParakeetTdtV3.windowSamples(for: $0, windows: ladder) }
    #expect(picks == [80000, 80000, 160000, 160000, 240000, 240000, 320000, 320000, 480000, 480000, 480000, 480000])
    // v3's two windows and legacy single-window configs keep their previous choices.
    #expect([240000, 240001, 600000].map { ParakeetTdtV3.windowSamples(for: $0, windows: [240000, 480000]) } == [240000, 480000, 480000])
    #expect(ParakeetTdtV3.windowSamples(for: 400000, windows: nil) == 240000)
}

@Test func windowListsMustBeOrderedStandardWindowsWithFifteenAndThirtySeconds() throws {
    func error(for windows: [[String: Any]]) throws -> String {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config: [String: Any] = ["architecture": "parakeet-unified-en-0.6b", "decoder": "rnnt", "precision": "fp16", "sample_rate": 16000,
                                     "window_samples": 240000, "blank_id": 1024, "vocabulary_size": 1024, "max_symbols": 10,
                                     "punctuation": [","], "files": [String: String](), "windows": windows]
        try JSONSerialization.data(withJSONObject: config).write(to: directory.appendingPathComponent("config.json"))
        do { _ = try ParakeetTdtV3(directory: directory, cacheDirectory: directory); return "" }
        catch ParakeetError.badConfig(let reason) { return reason }
    }
    let window = { (seconds: Int) -> [String: Any] in ["samples": seconds * 16000, "function": "w\(seconds)"] }
    // An accepted ladder fails later, at the missing package checksums, not at the window check.
    #expect(try error(for: [5, 10, 15, 20, 30].map(window)).contains("checksum"))
    #expect(try error(for: [15, 30].map(window)).contains("checksum"))
    for bad in [[30, 15], [5, 15], [15, 20], [15, 15, 30], [15, 25, 30]] {
        #expect(try error(for: bad.map(window)).contains("window functions"), "\(bad)")
    }
    #expect(try error(for: [["samples": 240000, "function": "w15"], ["samples": 480000, "function": "long"]]).contains("window functions"))
}
