import CoreML
import Foundation
import Testing
@testable import BuilderHelmAudio
@testable import ParakeetRuntime

extension RealModelTests {
    /// Compare independent inputs and scalar reads so layout errors cannot make both paths agree.
    @Test(arguments: AsrModel.all)
    func longWindowsMatchIndependentInputs(model: AsrModel) async throws {
        let directory = try await ModelStore.shared.download(model)
        let url = Bundle.module.url(forResource: "long", withExtension: "wav", subdirectory: "Fixtures")!
        let samples = try ParakeetAudio.read(url)
        let runtime = try ParakeetTdtCtc110M(directory: directory, computeUnits: realModelTestComputeUnits.coreML,
                                            ctcOnly: true, cacheDirectory: ModelStore.shared.compiledModelsDirectory)
        let independent = try ParakeetTdtCtc110M(directory: directory, computeUnits: realModelTestComputeUnits.coreML,
                                                ctcOnly: true, cacheDirectory: ModelStore.shared.compiledModelsDirectory)
        // Exercise a change of function and input length before reusing the long function.
        _ = try runtime.encode(samples.prefix(96_000), firstFrame: 0)
        for start in stride(from: 0, to: samples.count - 96_000, by: 224_000) {
            let chunk = samples[start..<min(start + 320_000, samples.count)]
            let prepared = try runtime.prepare(chunk, firstFrame: start / 1280, forcedWindowSamples: 320_000)
            let audio = try MLMultiArray(shape: [1, 320_000], dataType: .float32)
            audio.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, strides in
                for i in 0..<320_000 { buffer[i * strides[1]] = i < chunk.count ? chunk[chunk.startIndex + i] : 0 }
            }
            let length = try MLMultiArray(shape: [1], dataType: .int32)
            length[0] = NSNumber(value: chunk.count)
            let sameAudio = (0..<320_000).allSatisfy { audio[$0] == prepared.audio[$0] }
            #expect(sameAudio, "\(model.id) window \(start): prepared PCM differs from independent PCM")
            let reference = try predict(prepared.rung.preprocessor, inputs: ["audio": audio, "audio_length": length])
            let rawMel = try #require(reference.featureValue(for: "mel")?.multiArrayValue)
            let melDifferences = (0..<rawMel.count).filter { Float(Float16(rawMel[$0].floatValue)) != prepared.mel[$0].floatValue }.count
            let maximumMelError = (0..<rawMel.count).map { abs(Float(Float16(rawMel[$0].floatValue)) - prepared.mel[$0].floatValue) }.max() ?? 0
            #expect(prepared.audioLength[0] == length[0])
            #expect(melDifferences == 0, "\(model.id) window \(start): \(melDifferences) mel values differ from independent input")
            let output = try predict(prepared.rung.encoder, inputs: ["mel": prepared.mel, "mel_length": prepared.melLength])
            let rawCtc = try #require(output.featureValue(for: "ctc_logprobs")?.multiArrayValue)
            let frames = try #require(output.featureValue(for: "encoder_length")?.multiArrayValue)[0].intValue
            let flat = ParakeetTdtCtc110M.contiguous(rawCtc, rows: frames)
            let readDifferences = flat.indices.filter { flat[$0] != rawCtc[$0].floatValue }.count
            #expect(readDifferences == 0, "\(model.id) window \(start): \(readDifferences) CTC values differ from scalar reads")
            let snapshot = try ParakeetTdtCtc110M.snapshotTensor(rawCtc)
            let copyDifferences = flat.indices.filter { rawCtc[$0] != snapshot[$0] }.count
            #expect(copyDifferences == 0, "\(model.id) window \(start): \(copyDifferences) CTC values changed in snapshot")
            let encoded = try runtime.runEncoder(prepared)
            let repeated = ParakeetTdtCtc110M.contiguous(encoded.ctc, rows: encoded.frames)
            let repeatDifferences = zip(flat, repeated).filter { $0 != $1 }.count
            #expect(repeatDifferences == 0, "\(model.id) window \(start): \(repeatDifferences) CTC values changed on repeated prediction")
            let other = try independent.prepare(chunk, firstFrame: start / 1280, forcedWindowSamples: 320_000)
            let instanceMelDifferences = (0..<prepared.mel.count).filter { prepared.mel[$0] != other.mel[$0] }.count
            #expect(instanceMelDifferences == 0, "\(model.id) window \(start): mel differs across runtime instances")
            let otherEncoded = try independent.runEncoder(other)
            let otherFlat = ParakeetTdtCtc110M.contiguous(otherEncoded.ctc, rows: otherEncoded.frames)
            #expect(otherEncoded.frames == frames)
            let instanceDifferences = zip(flat, otherFlat).filter { $0 != $1 }.count
            #expect(instanceDifferences == 0, "\(model.id) window \(start): \(instanceDifferences) CTC values differ across runtime instances")
            print("\(model.id) window \(start): frames=\(frames), PCM equal=\(sameAudio), max mel error=\(maximumMelError), mel/read/copy/repeat/instance-mel/instance-ctc differences=\(melDifferences)/\(readDifferences)/\(copyDifferences)/\(repeatDifferences)/\(instanceMelDifferences)/\(instanceDifferences)")
        }
    }

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
                "audio": prepared.audio, "audio_length": prepared.audioLength,
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
            "audio": prepared.audio, "audio_length": prepared.audioLength,
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
