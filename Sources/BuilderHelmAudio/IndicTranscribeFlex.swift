import CoreML
import Foundation
import ParakeetRuntime

/// Local, whole-clip inference with the BuilderHelm Indic-Transcribe Flex packages.
public actor IndicTranscribeFlex {
    public enum OutputMode: String, Sendable {
        case native, mixed, romanized
    }

    public struct Transcript: Sendable {
        public let text: String
        public let language: String
    }

    public enum InferenceError: Error {
        case invalidAudioLength
        case invalidAudioSamples
        case unsupportedLanguage(String)
        case invalidTokenizer
        case invalidModelOutput(String)
        case tokenLimitReached
    }

    private struct Tokenizer: Decodable {
        let pieces: [String]
        let prompts: [String: [String: [Int32]]]
        let endTokenIDs: Set<Int>
        let languageDetectionPrefix: [Int32]
        let languageTokenIDs: [String: Int]
    }

    private let frontend: MLModel
    private let encoder: MLModel
    private let decoder: MLModel
    private let usesCachedDecoder: Bool
    private let tokenizer: Tokenizer

    public init(modelDirectory: URL, useCachedDecoder: Bool = false) throws {
        let tokenizerURL = modelDirectory.appendingPathComponent("tokenizer.json")
        let metadata = try JSONDecoder().decode(Tokenizer.self, from: Data(contentsOf: tokenizerURL))
        let tokenIndices = metadata.pieces.indices
        let modes: Set<String> = ["native", "mixed", "romanized"]
        let validPrompts = metadata.prompts.values.allSatisfy { languagePrompts in
            Set(languagePrompts.keys) == modes && languagePrompts.values.allSatisfy { prompt in
                prompt.count == 10 && prompt.allSatisfy { tokenIndices.contains(Int($0)) }
            }
        }
        guard metadata.pieces.count == 7_152,
              !metadata.endTokenIDs.isEmpty, metadata.endTokenIDs.allSatisfy(tokenIndices.contains),
              metadata.languageDetectionPrefix.count == 3,
              metadata.languageDetectionPrefix.allSatisfy({ tokenIndices.contains(Int($0)) }),
              metadata.languageTokenIDs.count == 27,
              Set(metadata.languageTokenIDs.keys) == Set(metadata.prompts.keys),
              metadata.languageTokenIDs.values.allSatisfy(tokenIndices.contains),
              validPrompts else {
            throw InferenceError.invalidTokenizer
        }
        tokenizer = metadata
        frontend = try Self.load("frontend", from: modelDirectory, units: .cpuOnly)
        encoder = try Self.load("encoder", from: modelDirectory, units: .cpuAndGPU)
        usesCachedDecoder = useCachedDecoder
        decoder = try Self.load(
            useCachedDecoder ? "decoder_cached" : "decoder", from: modelDirectory, units: .cpuAndGPU
        )
    }

    public var supportedLanguages: [String] { tokenizer.prompts.keys.sorted() }

    /// Reads a whole clip using the SDK's shared 16 kHz mono audio conversion.
    public func transcribe(
        file: URL,
        language: String? = nil,
        mode: OutputMode = .native
    ) throws -> Transcript {
        try Task.checkCancellation()
        let reader = try ParakeetAudio.read(file, maximumSamples: 480_001)
        return try transcribe(samples: reader, language: language, mode: mode)
    }

    /// Accepts mono 16 kHz samples, up to 30 seconds. Omit language for detection.
    public func transcribe(
        samples: [Float],
        language: String? = nil,
        mode: OutputMode = .native
    ) throws -> Transcript {
        guard (1...480_000).contains(samples.count) else { throw InferenceError.invalidAudioLength }
        guard samples.allSatisfy(\.isFinite) else { throw InferenceError.invalidAudioSamples }
        if let language, tokenizer.prompts[language] == nil {
            throw InferenceError.unsupportedLanguage(language)
        }
        try Task.checkCancellation()
        let padding = max(0, 16_000 - samples.count)
        let left = Int((Double(padding) / 2).rounded(.toNearestOrEven))
        let audio = try MLMultiArray(shape: [1, NSNumber(value: samples.count + padding)], dataType: .float32)
        for index in 0..<audio.count {
            let source = index - left
            audio[index] = NSNumber(value: samples.indices.contains(source) ? samples[source] : 0)
        }

        let features = try Self.predict(frontend, ["audio": audio], output: "features")
        let states = try Self.predict(encoder, ["features": features], output: "encoder_states")
        let selected: String
        if let language {
            selected = language
        } else {
            selected = try detectLanguage(states)
        }
        guard var ids = tokenizer.prompts[selected]?[mode.rawValue] else {
            throw InferenceError.unsupportedLanguage(selected)
        }
        let promptCount = ids.count
        let state = usesCachedDecoder ? decoder.makeState() : nil
        for step in 0..<512 {
            try Task.checkCancellation()
            let logits: MLMultiArray
            if let state, step > 0, let token = ids.last {
                logits = try nextLogits([token], states: states, state: state, offset: ids.count - 1)
            } else {
                logits = try prefill(ids, states: states, state: state)
            }
            guard let token = tokenizer.pieces.indices.max(by: {
                logits[$0].floatValue < logits[$1].floatValue
            }) else { throw InferenceError.invalidModelOutput("logits") }
            if tokenizer.endTokenIDs.contains(token) {
                let text = ids.dropFirst(promptCount).map { tokenizer.pieces[Int($0)] }.joined()
                    .replacingOccurrences(of: "▁", with: " ")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return Transcript(text: text, language: selected)
            }
            ids.append(Int32(token))
        }
        throw InferenceError.tokenLimitReached
    }

    private func detectLanguage(_ states: MLMultiArray) throws -> String {
        let state = usesCachedDecoder ? decoder.makeState() : nil
        let logits = try prefill(tokenizer.languageDetectionPrefix, states: states, state: state)
        // Bodhan's automatic mode excludes these two explicit-only candidates.
        let candidates = tokenizer.languageTokenIDs.filter { $0.key != "bgc" && $0.key != "bhb" }
        guard let detected = candidates.max(by: {
            let lhs = logits[$0.value].floatValue
            let rhs = logits[$1.value].floatValue
            return lhs == rhs ? $0.value > $1.value : lhs < rhs
        }) else { throw InferenceError.invalidModelOutput("language") }
        return detected.key
    }

    private func prefill(_ ids: [Int32], states: MLMultiArray, state: MLState?) throws -> MLMultiArray {
        guard let state else { return try nextLogits(ids, states: states) }
        var logits: MLMultiArray?
        for (offset, token) in ids.enumerated() {
            try Task.checkCancellation()
            logits = try nextLogits([token], states: states, state: state, offset: offset)
        }
        guard let logits else { throw InferenceError.invalidTokenizer }
        return logits
    }

    private func nextLogits(
        _ ids: [Int32], states: MLMultiArray, state: MLState? = nil, offset: Int = 0
    ) throws -> MLMultiArray {
        let length = NSNumber(value: ids.count)
        let maskWidth = state == nil ? ids.count : 544
        let tokens = try MLMultiArray(shape: [1, length], dataType: .int32)
        let positions = try MLMultiArray(shape: [1, length], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, 1, length, NSNumber(value: maskWidth)], dataType: .float32)
        for (row, token) in ids.enumerated() {
            tokens[row] = NSNumber(value: token)
            let position = offset + row
            positions[row] = NSNumber(value: position)
            for column in 0..<maskWidth {
                mask[row * maskWidth + column] = NSNumber(value: column > position ? Float(-10_000) : 0)
            }
        }
        let logits = try Self.predict(
            decoder, ["ids": tokens, "positions": positions, "mask": mask, "encoder_states": states],
            output: "logits", state: state
        )
        guard logits.count == tokenizer.pieces.count else {
            throw InferenceError.invalidModelOutput("logits")
        }
        return logits
    }

    private static func load(_ name: String, from directory: URL, units: MLComputeUnits) throws -> MLModel {
        let compiled = directory.appendingPathComponent("\(name).mlmodelc")
        let url = FileManager.default.fileExists(atPath: compiled.path)
            ? compiled : try MLModel.compileModel(at: directory.appendingPathComponent("\(name).mlpackage"))
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    private static func predict(
        _ model: MLModel,
        _ inputs: [String: MLMultiArray],
        output: String,
        state: MLState? = nil
    ) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let result = try state.map { try model.prediction(from: provider, using: $0) }
            ?? model.prediction(from: provider)
        guard let value = result.featureValue(for: output)?.multiArrayValue else {
            throw InferenceError.invalidModelOutput(output)
        }
        return value
    }
}
