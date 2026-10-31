import CoreML
import Foundation
import Testing
@testable import BuilderHelmAudio
@testable import ParakeetRuntime

extension RealModelTests {
    @Test(arguments: AsrModel.all)
    func encoderOutputsSurviveTheNextPrediction(model: AsrModel) async throws {
        let directory = try await ModelStore.shared.download(model)
        let url = Bundle.module.url(forResource: "long", withExtension: "wav", subdirectory: "Fixtures")!
        let samples = try ParakeetAudio.read(url)
        let runtime = try ParakeetTdtCtc110M(directory: directory, computeUnits: realModelTestComputeUnits.coreML,
                                            ctcOnly: !model.decodesTdt, cacheDirectory: ModelStore.shared.compiledModelsDirectory)
        let first = try runtime.runEncoder(runtime.prepare(samples.prefix(320_000), firstFrame: 0))
        let before = ParakeetTdtCtc110M.contiguous(first.ctc, rows: first.frames)
        let second = try runtime.runEncoder(runtime.prepare(samples.dropFirst(224_000).prefix(320_000), firstFrame: 175))
        let next = ParakeetTdtCtc110M.contiguous(second.ctc, rows: second.frames)
        let after = ParakeetTdtCtc110M.contiguous(first.ctc, rows: first.frames)
        let unchanged = before == after
        #expect(unchanged, "A subsequent prediction invalidated retained CTC output")
        let changed = before != next
        #expect(changed, "Different speech windows must produce different CTC probabilities")
    }

    @Test(arguments: AsrModel.all)
    func allWindowShapesPreservePreprocessing(model: AsrModel) async throws {
        let directory = try await ModelStore.shared.download(model)
        let url = Bundle.module.url(forResource: "long", withExtension: "wav", subdirectory: "Fixtures")!
        let samples = try ParakeetAudio.read(url)
        let runtime = try ParakeetTdtCtc110M(directory: directory, computeUnits: realModelTestComputeUnits.coreML,
                                            ctcOnly: true, cacheDirectory: ModelStore.shared.compiledModelsDirectory)
        for window in runtime.config.windows {
            let chunk = samples.prefix(window.samples)
            let prepared = try runtime.prepare(chunk, firstFrame: 0)
            let reference = try predict(prepared.rung.preprocessor, inputs: [
                "audio": prepared.rung.audio, "audio_length": runtime.length,
            ])
            let mel = try #require(reference.featureValue(for: "mel")?.multiArrayValue)
            let converted = try ParakeetTdtCtc110M.encoderMelInput(mel)
            var mismatches = 0
            for i in 0..<mel.count {
                if Float(Float16(mel[i].floatValue)) != converted[i].floatValue { mismatches += 1 }
            }
            print("\(model.id) \(window.function): mel strides=\(mel.strides), length=\(prepared.melLength[0]), conversion mismatches=\(mismatches)")
            #expect(mismatches == 0, "\(window.function) changed logical mel features during conversion")
            #expect(prepared.melLength[0].intValue == chunk.count / 160)
            let encoded = try runtime.runEncoder(prepared)
            #expect(encoded.frames == (prepared.melLength[0].intValue + 7) / 8)
        }
    }

    /// Validate semantic outputs at each boundary; two equally empty transcripts are not parity evidence.
    @Test(arguments: AsrModel.all)
    func inferenceStagesPreserveSpeech(model: AsrModel) async throws {
        let directory = try await ModelStore.shared.download(model)
        let url = Bundle.module.url(forResource: "speech", withExtension: "wav", subdirectory: "Fixtures")!
        let samples = try ParakeetAudio.read(url)
        let peak = samples.map(abs).max() ?? 0
        print("\(model.id) PCM: count=\(samples.count), peak=\(peak)")
        try #require(samples.count > 320 && peak > 0.001, "Speech fixture must decode to non-silent PCM")
        let runtime = try ParakeetTdtCtc110M(directory: directory, computeUnits: realModelTestComputeUnits.coreML,
                                            ctcOnly: true, cacheDirectory: ModelStore.shared.compiledModelsDirectory)
        let prepared = try runtime.prepare(samples[...], firstFrame: 0)
        let features = try predict(prepared.rung.preprocessor, inputs: [
            "audio": prepared.rung.audio, "audio_length": runtime.length,
        ])
        let mel = try #require(features.featureValue(for: "mel")?.multiArrayValue)
        let melLength = try #require(features.featureValue(for: "mel_length")?.multiArrayValue)
        print("\(model.id) raw mel: \(tensorSummary(mel)); length=\(melLength[0]), type=\(melLength.dataType)")
        print("\(model.id) encoder input mel: \(tensorSummary(prepared.mel)); length=\(prepared.melLength[0])")
        try #require(melLength[0].intValue > 0, "Preprocessing must preserve the speech length")
        let output = try predict(prepared.rung.encoder, inputs: [
            "mel": prepared.mel, "mel_length": prepared.melLength,
        ])
        let length = try #require(output.featureValue(for: "encoder_length")?.multiArrayValue)
        let probabilities = try #require(output.featureValue(for: "ctc_logprobs")?.multiArrayValue)
        print("\(model.id) encoder: length=\(length[0]), type=\(length.dataType), CTC \(tensorSummary(probabilities))")
        try #require(length[0].intValue > 0, "Encoder must produce frames for speech")
        let encoded = try runtime.runEncoder(prepared)
        #expect(encoded.frames == length[0].intValue)
        let rows = (0..<encoded.frames).map { ParakeetTdtCtc110M.row(encoded.ctc, $0) }
        #expect(rows.allSatisfy { !$0.contains(where: \.isNaN) }, "CTC rows must not contain NaNs")
        let nonblank = rows.filter { row in
            row.indices.max(by: { row[$0] < row[$1] }) != runtime.config.blank_id
        }.count
        print("\(model.id) speech frames=\(encoded.frames), nonblank CTC frames=\(nonblank)")
        #expect(nonblank > 0, "Speech fixture must not collapse to all blanks")
    }
}

private func predict(_ model: MLModel, inputs: [String: Any]) throws -> any MLFeatureProvider {
    try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
}

private func tensorSummary(_ array: MLMultiArray) -> String {
    var finite = 0, nan = 0
    var minimum = Float.infinity, maximum = -Float.infinity
    for i in 0..<array.count {
        let value = array[i].floatValue
        if value.isNaN { nan += 1 }
        if value.isFinite { finite += 1; minimum = min(minimum, value); maximum = max(maximum, value) }
    }
    return "shape=\(array.shape), strides=\(array.strides), type=\(array.dataType), finite=\(finite)/\(array.count), NaN=\(nan), range=\(minimum)...\(maximum)"
}
