import CryptoKit
import Foundation
import Testing
import BuilderHelmAudio

private let regressionEnabled = ProcessInfo.processInfo.environment["BUILDERHELM_REGRESSION_OUT"] != nil

/// Optional local corpus regression. Snapshots contain hashes, never audio or corpus transcripts.
/// Capture before a runtime change, then set BUILDERHELM_REGRESSION_BASELINE to compare afterwards.
@Test(.enabled(if: regressionEnabled))
func runtimeRegressionSnapshot() async throws {
    struct Corpus: Decodable { let files: [Clip] }
    struct Clip: Decodable { let path: String }
    struct Snapshot: Codable, Equatable {
        let model: String
        let mode: String
        let clip: String
        let audioSHA256: String
        let transcriptSHA256: String
        let probabilitySHA256: String?
        let keywordSHA256: String?
    }
    struct Token: Encodable { let text: String; let time: Double }
    struct Text: Encodable { let text: String; let ctcText: String; let tokens: [Token] }
    struct Hit: Encodable { let keyword: String; let score: Float; let start: Double; let end: Double }
    let environment = ProcessInfo.processInfo.environment
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    let fixtureURLs = ["speech", "long"].map {
        Bundle.module.url(forResource: $0, withExtension: "wav", subdirectory: "Fixtures")!
    }
    var corpus: [URL] = []
    for path in (environment["BUILDERHELM_REGRESSION_CASES"] ?? "").split(separator: ":") {
        let data = try Data(contentsOf: URL(fileURLWithPath: String(path)))
        corpus += try JSONDecoder().decode(Corpus.self, from: data).files.map { URL(fileURLWithPath: $0.path) }
    }
    let urls = fixtureURLs + corpus
    let audioHashes = try urls.map { hash(try Data(contentsOf: $0)) }
    var snapshots: [Snapshot] = []
    for model in AsrModel.all.filter({ $0.architecture == .parakeet110M }) {
        let recognizer = try await SpeechRecognizer.load(model)
        for (i, url) in urls.enumerated() {
            let transcript = try await recognizer.transcribe(file: url, includeCtcLogProbabilities: i < 2)
            let text = Text(text: transcript.text, ctcText: transcript.ctcText,
                            tokens: transcript.tokens.map { Token(text: $0.text, time: $0.time) })
            var probabilityHash: String?
            var keywordHash: String?
            if let probabilities = transcript.ctcLogProbabilities {
                let bytes = probabilities.values.withUnsafeBytes { Data($0) }
                probabilityHash = hash(bytes)
                let hits = recognizer.keywordSpotter.detect(["fox", "Neural Engine", "Maria", "lighthouse", "zebra"], in: probabilities)
                keywordHash = hash(try encoder.encode(hits.map { Hit(keyword: $0.keyword, score: $0.score, start: $0.start, end: $0.end) }))
            }
            snapshots.append(Snapshot(model: model.id, mode: model.decodesTdt ? "tdt" : "ctc",
                                      clip: "\(i):\(url.lastPathComponent)", audioSHA256: audioHashes[i],
                                      transcriptSHA256: hash(try encoder.encode(text)),
                                      probabilitySHA256: probabilityHash, keywordSHA256: keywordHash))
        }
    }
    let output = try #require(environment["BUILDERHELM_REGRESSION_OUT"])
    try encoder.encode(snapshots).write(to: URL(fileURLWithPath: output), options: .atomic)
    if let baseline = environment["BUILDERHELM_REGRESSION_BASELINE"] {
        let expected = try JSONDecoder().decode([Snapshot].self, from: Data(contentsOf: URL(fileURLWithPath: baseline)))
        #expect(snapshots.count == expected.count)
        for (actual, prior) in zip(snapshots, expected) { #expect(actual == prior, "Regression for \(actual.model) / \(actual.clip)") }
    }
    print("Runtime regression: \(snapshots.count) outputs; \(fixtureURLs.count) fixtures and \(corpus.count) corpus files per model")
}
