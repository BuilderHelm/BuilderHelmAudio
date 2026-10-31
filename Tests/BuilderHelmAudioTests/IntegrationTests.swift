import Foundation
import Testing
import BuilderHelmAudio

/// Downloads the real models from Hugging Face. Set BUILDERHELM_INTEGRATION=1
/// (TEST_RUNNER_BUILDERHELM_INTEGRATION=1 with xcodebuild) to run.
private let enabled = ProcessInfo.processInfo.environment["BUILDERHELM_INTEGRATION"] == "1"
private let speech = Bundle.module.url(forResource: "speech", withExtension: "wav", subdirectory: "Fixtures")!

@Test(.enabled(if: enabled), arguments: AsrModel.all)
func transcribesAndSpots(model: AsrModel) async throws {
    let recognizer = try await SpeechRecognizer.load(model)
    let transcript = try await recognizer.transcribe(file: speech)
    print("\(model.id): \(transcript.text) [\(Int(transcript.realTimeFactor))× real time]")
    #expect(transcript.text.lowercased().contains("quick brown fox jumps over the lazy dog"))
    #expect(transcript.ctcText.lowercased().contains("neural engine"))
    #expect(abs(transcript.duration - 6.1) < 0.2)
    #expect(transcript.tokens.isEmpty != model.decodesTdt)
    #expect(transcript.tokens.map(\.time) == transcript.tokens.map(\.time).sorted())

    let hits = try await recognizer.spot(["fox", "Neural Engine", "zebra", "Mediterranean"], inFile: speech)
    #expect(Set(hits.map(\.keyword)) == ["fox", "Neural Engine"])
    let samples = try SpeechRecognizer.loadAudio(speech)
    #expect(try await recognizer.transcribe(samples: samples).text == transcript.text)
}
