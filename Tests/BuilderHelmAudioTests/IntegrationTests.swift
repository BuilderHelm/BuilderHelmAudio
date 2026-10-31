import Foundation
import Testing
import BuilderHelmAudio

/// Downloads the real models from Hugging Face. Set BUILDERHELM_INTEGRATION=1
/// (TEST_RUNNER_BUILDERHELM_INTEGRATION=1 with xcodebuild) to run.
private let enabled = ProcessInfo.processInfo.environment["BUILDERHELM_INTEGRATION"] == "1"
let realModelTestComputeUnits: ComputeUnits = ProcessInfo.processInfo.environment["BUILDERHELM_TEST_COMPUTE"] == "cpu" ? .cpu : .neuralEngine
private let speech = Bundle.module.url(forResource: "speech", withExtension: "wav", subdirectory: "Fixtures")!

// Serial execution also prevents simultaneous first downloads of the same model through a reentrant actor.
@Suite(.serialized, .enabled(if: enabled))
struct RealModelTests {
@Test(arguments: AsrModel.all.filter { $0.architecture == .parakeet110M })
func transcribesAndSpots(model: AsrModel) async throws {
    if ModelStore.shared.isDownloaded(model) { #expect(try await ModelStore.shared.verify(model)) }
    let recognizer = try await SpeechRecognizer.load(model, computeUnits: realModelTestComputeUnits)
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
}
