import Foundation
import Testing
@testable import BuilderHelmAudio

@Test func cadenceSaeDescriptorPinsTheSixBitFolderOfOneRevision() throws {
    let model = AsrModel.cadenceSaeEn
    #expect(AsrModel.all.contains(model) && AsrModel.named("cadence-sae-en") == model)
    #expect(model.repository == "BuilderHelm/Cadence-Sae-EN" && model.revision.count == 40 && model.variant == "6bit")
    #expect(model.architecture == .parakeetV3 && model.supportedLanguages == ["en"] && model.ctcConfiguration == nil)
    // Same runtime and capabilities as the multilingual v3 release.
    let selection = try RuntimeSelection.resolve(model: model, mode: .tdt)
    #expect(selection.capabilities == (try RuntimeSelection.resolve(model: .parakeetTdtV3, mode: .tdt)).capabilities)
    #expect(throws: BuilderHelmAudioError.self) { try RuntimeSelection.resolve(model: model, mode: .ctc) }
    // Every file the v3 runtime opens, plus the export records config.json pins, all from coreml/6bit/.
    #expect(model.files.count == 19 && Set(model.files.map(\.path)).count == 19)
    #expect(Set(AsrModel.parakeetTdtV3.files.map(\.path)).isSubset(of: model.files.map(\.path)))
    #expect(model.files.allSatisfy { $0.repositoryPath == "coreml/6bit/" + $0.path && $0.sha256.count == 64 && $0.size > 0 })
    #expect((540_000_000...545_000_000).contains(model.totalBytes))
    let store = ModelStore(root: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
    #expect(store.directory(for: model) != store.directory(for: .parakeetTdtV3))
}

@Test func v3RuntimeAcceptsLanguageSubsetsButNothingElse() throws {
    func v3(_ languages: [String]) -> AsrModel {
        AsrModel(id: "v3-languages", repository: "Test/NoDownloads", version: "0", revision: String(repeating: "a", count: 40),
                 architecture: .parakeetV3, supportedLanguages: languages, capabilities: .parakeetV3Tdt,
                 ctcConfiguration: nil, files: [])
    }
    #expect(try RuntimeSelection.resolve(model: v3(["en"]), mode: .tdt).architecture == .parakeetV3)
    #expect(try RuntimeSelection.resolve(model: v3(["en", "de", "fr"]), mode: .tdt).architecture == .parakeetV3)
    for languages in [[], ["hi"], ["en", "ja"]] {
        #expect(throws: BuilderHelmAudioError.self, "\(languages)") { try RuntimeSelection.resolve(model: v3(languages), mode: .tdt) }
    }
}

extension RealModelTests {
    /// Downloads the pinned Cadence Sae 6-bit folder and transcribes through the public API.
    @Test func cadenceSaeDownloadsVerifiesAndTranscribes() async throws {
        let descriptor = AsrModel.cadenceSaeEn
        _ = try await ModelStore.shared.download(descriptor)
        #expect(try await ModelStore.shared.verify(descriptor))
        let recognizer = try await SpeechRecognizer.load(descriptor, computeUnits: realModelTestComputeUnits)
        #expect(recognizer.capabilities == [.tdt, .tokenTimestamps, .wordTimestamps, .rankedAlternatives])
        #expect(recognizer.supportedLanguages == ["en"])
        for name in ["speech", "long"] {
            let url = Bundle.module.url(forResource: name, withExtension: "wav", subdirectory: "Fixtures")!
            let transcript = try await recognizer.transcribe(file: url)
            #expect(transcript.words.map(\.text).joined(separator: " ") == transcript.text)
            if name == "speech" {
                #expect(transcript.text.lowercased().contains("quick brown fox jumps over the lazy dog"))
            } else {
                #expect(transcript.duration > 60 && transcript.text.split(separator: " ").count > 50)
            }
        }
    }
}
