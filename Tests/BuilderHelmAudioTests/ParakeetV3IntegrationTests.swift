import CoreML
import Foundation
import Testing
@testable import BuilderHelmAudio
@testable import ParakeetRuntime

extension RealModelTests {
    /// Isolate backend instability before the autoregressive decoder amplifies it.
    @Test func v3StagesRepeatIdenticalInputs() async throws {
        let directory = try await ModelStore.shared.download(.parakeetTdtV3)
        let cache = ModelStore.shared.compiledModelsDirectory
        func load(_ name: String, function: String? = nil) throws -> MLModel {
            let url = try ParakeetTdtCtc110M.modelURL(directory, name, cache: cache, allowBundledCompiled: false)
            return try ParakeetTdtCtc110M.load(url, .cpuOnly, function)
        }
        func provider(_ values: [String: MLMultiArray]) throws -> MLDictionaryFeatureProvider {
            try MLDictionaryFeatureProvider(dictionary: values.mapValues { MLFeatureValue(multiArray: $0) })
        }
        func repeatStage(_ name: String, model: MLModel, input: MLFeatureProvider) throws -> MLFeatureProvider {
            // Snapshot through scalar indexing, independently of the runtime's tensor-copy helper.
            func snapshot(_ output: MLFeatureProvider) throws -> [String: [Float]] {
                var values: [String: [Float]] = [:]
                for key in output.featureNames.sorted() {
                    let array = try #require(output.featureValue(for: key)?.multiArrayValue)
                    var copied: [Float]?
                    if array.dataType == .float32 {
                        // Copy before scalar indexing can materialize framework-owned output storage.
                        let owned = try ParakeetTdtV3.alignedOutput(array.shape.map(\.intValue))
                        try ParakeetTdtV3.copy(array, into: owned)
                        copied = (0..<owned.count).map { owned[$0].floatValue }
                    }
                    let scalarValues = (0..<array.count).map { array[$0].floatValue }
                    if let copied {
                        let mismatches = zip(scalarValues, copied).filter { $0 != $1 }.count
                        #expect(mismatches == 0, "V3 \(name).\(key): tensor copy differs from scalar reads")
                    }
                    values[key] = scalarValues
                }
                return values
            }
            let first = try model.prediction(from: input)
            let reference = try snapshot(first)
            for iteration in 1...3 {
                let output = try model.prediction(from: input)
                let actual = try snapshot(output)
                for key in reference.keys.sorted() {
                    let expected = reference[key]!, values = actual[key] ?? []
                    try #require(values.count == expected.count)
                    let mismatches = zip(expected, values).filter { !$0.isFinite || !$1.isFinite || $0 != $1 }.count
                    print("V3 stage \(name).\(key), repeat \(iteration): \(mismatches)/\(values.count) changed values")
                    #expect(mismatches == 0, "V3 \(name).\(key) changed for identical input on repeat \(iteration)")
                }
            }
            return first
        }
        let speech = Bundle.module.url(forResource: "speech", withExtension: "wav", subdirectory: "Fixtures")!
        let samples = try SpeechRecognizer.loadAudio(speech)
        let audio = try ParakeetTdtV3.alignedOutput([1,240000])
        for i in samples.indices { audio[i] = NSNumber(value: samples[i]) }
        let length = try ParakeetTdtV3.alignedOutput([1], .int32)
        length[0] = NSNumber(value: samples.count)
        let frontend = try load("Preprocessor", function: "w15")
        let mel = try repeatStage("Preprocessor", model: frontend,
                                  input: provider(["audio_signal": audio, "audio_length": length]))
        let encoder = try load("Encoder", function: "w15")
        let encoded = try repeatStage("Encoder", model: encoder, input: mel)
        let tail = try load("EncoderTail", function: "w15")
        var tailValues: [String: MLFeatureValue] = [:]
        for key in tail.modelDescription.inputDescriptionsByName.keys {
            guard let value = encoded.featureValue(for: key) ?? mel.featureValue(for: key) else {
                throw ParakeetError.badConfig("Missing test encoder-stage input: \(key)")
            }
            tailValues[key] = value
        }
        _ = try repeatStage("EncoderTail", model: tail, input: MLDictionaryFeatureProvider(dictionary: tailValues))
        let predictor = try load("Predictor")
        let target = try ParakeetTdtV3.alignedOutput([1,1], .int32)
        target[0] = 8192
        let state = try ParakeetTdtV3.alignedOutput([2,1,640])
        let predicted = try repeatStage("Predictor", model: predictor,
                                       input: provider(["targets": target, "h_in": state, "c_in": state]))
        let joint = try load("Joint")
        let step = try ParakeetTdtV3.alignedOutput([1,1024,1])
        let decoder = try #require(predicted.featureValue(for: "decoder")?.multiArrayValue)
        _ = try repeatStage("Joint", model: joint, input: provider(["encoder_step": step, "decoder_step": decoder]))
    }

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
