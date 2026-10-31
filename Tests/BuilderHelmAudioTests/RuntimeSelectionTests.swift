import Foundation
import Testing
@testable import BuilderHelmAudio

private func descriptor(architecture: AsrArchitecture = .parakeet110M,
                        capabilities: AsrCapabilities = .parakeet110MTdt,
                        ctc: CtcConfiguration? = .parakeet110M) -> AsrModel {
    AsrModel(id: "configuration-test", repository: "Test/NoDownloads", version: "0",
             revision: String(repeating: "a", count: 40), architecture: architecture,
             supportedLanguages: ["en"], capabilities: capabilities, ctcConfiguration: ctc, files: [])
}

@Test func shippedModelCapabilitiesAndModeFallback() throws {
    let tdt = AsrModel.parakeetTdtCtc110M, ctc = AsrModel.parakeetCtc110M
    #expect(tdt.architecture == .parakeet110M)
    #expect(tdt.supportedLanguages == ["en"])
    #expect(tdt.capabilities == [.tdt, .ctc, .tokenTimestamps, .keywordSpotting])
    #expect(ctc.capabilities == [.ctc, .keywordSpotting])
    #expect(tdt.ctcConfiguration == .parakeet110M)
    #expect(try RuntimeSelection.resolve(model: tdt, mode: .tdt).mode == .tdt)
    #expect(try RuntimeSelection.resolve(model: tdt, mode: .ctc).capabilities == [.ctc, .keywordSpotting])
    #expect(try RuntimeSelection.resolve(model: ctc, mode: .tdt).mode == .ctc)
    #expect(try RuntimeSelection.resolve(model: nil, mode: .ctc).mode == .ctc)
}

@Test func unsupportedArchitectureFailsBeforeDownloadOrFilesystemAccess() async throws {
    let architecture = AsrArchitecture(rawValue: "not-implemented")
    let model = descriptor(architecture: architecture)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let store = ModelStore(root: root)
    await #expect(throws: BuilderHelmAudioError.unsupportedArchitecture(architecture.rawValue)) {
        try await SpeechRecognizer.load(model, store: store)
    }
    #expect(throws: BuilderHelmAudioError.unsupportedArchitecture(architecture.rawValue)) {
        try SpeechRecognizer(directory: root, model: model)
    }
    #expect(!FileManager.default.fileExists(atPath: root.path))
}

@Test func incompatibleMetadataCannotReachThe110MRuntime() throws {
    let models = [
        descriptor(capabilities: [.tdt]),
        descriptor(capabilities: [.keywordSpotting]),
        descriptor(capabilities: [.ctc, .wordTimestamps]),
        descriptor(ctc: nil),
        descriptor(ctc: try CtcConfiguration(classCount: 3, blank: 0, frameDuration: 0.04)),
    ]
    for model in models {
        #expect(throws: BuilderHelmAudioError.self) { try RuntimeSelection.resolve(model: model, mode: .tdt) }
    }
}

/// A non-CTC boundary test, without constructing or claiming support for a new model.
private final class TextRuntime: AsrRuntime {
    let mode: DecodingMode = .tdt
    let capabilities: AsrCapabilities = [.tdt]
    let keywordSpotter: KeywordSpotter? = nil
    func transcribe(file: URL, request: RuntimeRequest) throws -> RuntimeResult {
        Issue.record("An unsupported operation reached inference")
        return try transcribe(samples: [], request: request)
    }
    func transcribe(samples: [Float], request: RuntimeRequest) throws -> RuntimeResult {
        RuntimeResult(text: "recognized", ctcText: nil, tokens: [], ctcLogProbabilities: nil)
    }
}

@Test func missingOutputsAreExplicitAndUnsupportedRequestsThrow() async throws {
    let recognizer = SpeechRecognizer(runtime: TextRuntime())
    #expect(recognizer.keywordSpotterIfAvailable == nil)
    #expect(!recognizer.keywordSpotter.isAvailable)
    let transcript = try await recognizer.transcribe(samples: [])
    #expect(transcript.text == "recognized")
    #expect(transcript.ctcTranscript == nil)
    #expect(transcript.ctcText.isEmpty) // Existing String accessor still compiles.
    await #expect(throws: BuilderHelmAudioError.unsupportedOperation("CTC probabilities")) {
        try await recognizer.transcribe(samples: [], includeCtcLogProbabilities: true)
    }
    await #expect(throws: BuilderHelmAudioError.unsupportedOperation("keyword spotting")) {
        try await recognizer.spot(["word"], inFile: URL(fileURLWithPath: "/does-not-exist.wav"))
    }
}

@Test(.timeLimit(.minutes(1))) func queuedCancellationDoesNotRunWork() async throws {
    let queue = DispatchQueue(label: "test.cancel.queued")
    queue.suspend()
    let task = Task {
        try await runOnQueue(queue) { _ in
            Issue.record("Cancelled queued work ran")
            return 1
        }
    }
    task.cancel()
    queue.resume()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(try await runOnQueue(queue) { _ in 2 } == 2)
}

@Test(.timeLimit(.minutes(1))) func activeCancellationWaitsForSynchronousWorkAndAllowsReuse() async throws {
    let queue = DispatchQueue(label: "test.cancel.active")
    let entered = AsyncStream<Void>.makeStream()
    let release = DispatchSemaphore(value: 0)
    let task = Task {
        try await runOnQueue(queue) { _ in
            entered.continuation.yield(())
            release.wait()
            return 1
        }
    }
    var iterator = entered.stream.makeAsyncIterator()
    _ = await iterator.next()
    task.cancel()
    release.signal()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(try await runOnQueue(queue) { _ in 2 } == 2)
    entered.continuation.finish()
}
