import Foundation
import Testing
import BuilderHelmAudio

extension RealModelTests {
    /// Downloads the pinned public export and exercises function switches through the public API.
    @Test func v3RecognizesShortAndLongRecordingsAndCanBeReused() async throws {
        let descriptor = AsrModel.parakeetTdtV3
        let store = ModelStore.shared
        _ = try await store.download(descriptor)
        #expect(try await store.verify(descriptor))
        let recognizer = try await SpeechRecognizer.load(descriptor, computeUnits: realModelTestComputeUnits)
        #expect(recognizer.capabilities == [.tdt])
        #expect(recognizer.keywordSpotterIfAvailable == nil)
        var shortText: String?
        for name in ["speech", "long", "speech"] {
            let url = Bundle.module.url(forResource: name, withExtension: "wav", subdirectory: "Fixtures")!
            let file = try await recognizer.transcribe(file: url)
            let pcm = try await recognizer.transcribe(samples: SpeechRecognizer.loadAudio(url))
            #expect(!file.text.isEmpty)
            #expect(file.text == pcm.text)
            #expect(file.ctcTranscript == nil)
            #expect(file.ctcLogProbabilities == nil)
            #expect(file.tokens.isEmpty)
            #expect(abs(file.duration - pcm.duration) < 0.001)
            if name == "speech" {
                #expect(file.text.lowercased().contains("quick brown fox jumps over the lazy dog"))
                if let shortText { #expect(file.text == shortText) }
                shortText = file.text
            } else {
                #expect(file.duration > 60)
                #expect(file.text.split(separator: " ").count > 50)
            }
        }
        let silence = try await recognizer.transcribe(samples: [Float](repeating: 0, count: 31 * 16000))
        #expect(silence.text.isEmpty)
        let speech = Bundle.module.url(forResource: "speech", withExtension: "wav", subdirectory: "Fixtures")!
        await #expect(throws: BuilderHelmAudioError.unsupportedOperation("CTC probabilities")) {
            try await recognizer.transcribe(file: speech, includeCtcLogProbabilities: true)
        }
        await #expect(throws: BuilderHelmAudioError.unsupportedOperation("keyword spotting")) {
            try await recognizer.spot(["fox"], inFile: speech)
        }
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await recognizer.transcribe(file: speech)
        }
        await #expect(throws: CancellationError.self) { try await cancelled.value }
        #expect(try await recognizer.transcribe(file: speech).text == shortText)
    }
}
