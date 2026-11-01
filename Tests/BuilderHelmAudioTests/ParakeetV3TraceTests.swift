import CryptoKit
import Foundation
import Testing
@testable import BuilderHelmAudio
@testable import ParakeetRuntime

private final class V3TraceLog: @unchecked Sendable {
    struct Event: Equatable, Sendable {
        let stage: String
        let digest: String
        let count: Int
        let minimum: Float
        let maximum: Float
    }
    private let lock = NSLock()
    private var events: [Event] = []

    func record(_ stage: String, _ values: [Float]) {
        // Treat signed zero identically, as tensor equality does.
        let canonical = values.map { $0 == 0 ? Float.zero : $0 }
        let digest = canonical.withUnsafeBytes { SHA256.hash(data: Data($0)) }
            .map { String(format: "%02x", $0) }.joined()
        let event = Event(stage: stage, digest: digest, count: values.count,
                          minimum: values.min() ?? 0, maximum: values.max() ?? 0)
        lock.withLock { events.append(event) }
    }
    func take() -> [Event] {
        lock.withLock {
            let result = events
            events.removeAll(keepingCapacity: true)
            return result
        }
    }
}

extension RealModelTests {
    @Test func v3FrontendMatchesFrozenFullWindowFeatures() async throws {
        struct Reference: Decodable {
            struct Window: Decodable {
                let function: String
                let sampleCount: Int
                let frames: [Int]
                let channels: [[Float]]
            }
            let fixtureSHA256: String
            let windows: [Window]
        }
        let fixture = Bundle.module.url(forResource: "long", withExtension: "wav", subdirectory: "Fixtures")!
        let url = Bundle.module.url(forResource: "v3-mel-reference", withExtension: "json", subdirectory: "Fixtures")!
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
        let digest = SHA256.hash(data: try Data(contentsOf: fixture)).map { String(format: "%02x", $0) }.joined()
        try #require(digest == reference.fixtureSHA256)
        let original = try await ModelStore.shared.download(.parakeetTdtV3)
        let directory = ProcessInfo.processInfo.environment["BUILDERHELM_V3_MODEL_DIRECTORY"]
            .map { URL(fileURLWithPath: $0) } ?? original
        let runtime = try ParakeetTdtV3(directory: directory, computeUnits: .cpuOnly,
                                       cacheDirectory: ModelStore.shared.compiledModelsDirectory,
                                       decoderComputeUnits: .cpuOnly)
        let samples = try SpeechRecognizer.loadAudio(fixture)
        for window in reference.windows {
            var observed: [Float]?
            runtime.diagnostics = { stage, values in
                if stage == "mel" {
                    observed = values
                    runtime.diagnostics = nil
                }
            }
            _ = try runtime.transcribe(Array(samples.prefix(window.sampleCount)))
            runtime.diagnostics = nil
            let mel = try #require(observed)
            let frames = window.sampleCount / 160 + 1
            try #require(mel.count == 128 * frames && window.channels.count == 128)
            var maximumError: Float = 0
            for (channel, values) in window.channels.enumerated() {
                for (frame, expected) in zip(window.frames, values) {
                    let actual = mel[channel * frames + frame]
                    maximumError = actual.isFinite ? max(maximumError, abs(actual - expected)) : .infinity
                }
            }
            print("V3 frozen \(window.function): maximum mel error \(maximumError)")
            #expect(maximumError < 0.01, "V3 \(window.function) differs from frozen full-window features")
        }
    }

    /// Trace the actual adapter on SpeechRecognizer's serial queue, including window switches.
    @Test func v3FullDecodeTraceIsRepeatable() async throws {
        let original = try await ModelStore.shared.download(.parakeetTdtV3)
        let directory = ProcessInfo.processInfo.environment["BUILDERHELM_V3_MODEL_DIRECTORY"]
            .map { URL(fileURLWithPath: $0) } ?? original
        let runtime = try ParakeetV3Runtime(directory: directory, computeUnits: .cpuOnly,
                                            cache: ModelStore.shared.compiledModelsDirectory)
        let log = V3TraceLog()
        runtime.model.diagnostics = { stage, values in log.record(stage, values) }
        let recognizer = SpeechRecognizer(runtime: runtime, model: .parakeetTdtV3)
        let speech = Bundle.module.url(forResource: "speech", withExtension: "wav", subdirectory: "Fixtures")!
        let long = Bundle.module.url(forResource: "long", withExtension: "wav", subdirectory: "Fixtures")!

        func compare(_ actual: [V3TraceLog.Event], _ expected: [V3TraceLog.Event], label: String) {
            var firstDifference: String?
            for index in 0..<max(actual.count, expected.count) {
                let lhs = index < actual.count ? actual[index] : nil
                let rhs = index < expected.count ? expected[index] : nil
                if lhs != rhs {
                    firstDifference = "\(label), event \(index): actual \(String(describing: lhs)); reference \(String(describing: rhs))"
                    break
                }
            }
            print("V3 decode trace \(label): \(actual.count) events; \(firstDifference ?? "exact")")
            #expect(firstDifference == nil, "\(firstDifference ?? "exact")")
        }

        let shortReference = try await recognizer.transcribe(file: speech)
        let shortTrace = log.take()
        #expect(shortTrace.count > 10)
        let shortPCM = try await recognizer.transcribe(samples: SpeechRecognizer.loadAudio(speech))
        compare(log.take(), shortTrace, label: "short file/PCM")
        #expect(shortPCM.text == shortReference.text)
        let longFile = try await recognizer.transcribe(file: long)
        let longTrace = log.take()
        let longPCM = try await recognizer.transcribe(samples: SpeechRecognizer.loadAudio(long))
        compare(log.take(), longTrace, label: "long file/PCM")
        #expect(longPCM.text == longFile.text)
        let shortAfter = try await recognizer.transcribe(file: speech)
        compare(log.take(), shortTrace, label: "short after long")
        #expect(shortAfter.text == shortReference.text)
    }
}
