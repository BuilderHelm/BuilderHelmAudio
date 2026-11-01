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
    /// Trace the actual adapter on SpeechRecognizer's serial queue, including window switches.
    @Test func v3FullDecodeTraceIsRepeatable() async throws {
        let directory = try await ModelStore.shared.download(.parakeetTdtV3)
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
