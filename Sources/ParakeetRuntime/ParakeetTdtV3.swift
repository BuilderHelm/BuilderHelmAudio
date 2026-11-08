import CoreML
import CryptoKit
import Foundation
import Darwin

/// Offline runtime for the 0.6B Parakeet transducers with bounded windows: TDT v3 and the RNN-T
/// Parakeet Unified EN (offline mode), selected by `config.json`. Serialize calls on each instance.
/// Each token carries NeMo's timing (emission frame and duration, 80 ms frames; RNN-T tokens span
/// one frame) and `max_prob` confidence, from which `words(_:)` builds word timestamps.
public final class ParakeetTdtV3 {
    public struct Result {
        public let text: String
        public let tokenIDs: [Int]
        /// Encoder frame (80 ms) at which each token was emitted, from the start of the audio.
        public let emissionFrames: [Int]
        /// Frames each token covers: the TDT duration predicted with it (0...4), or 1 for RNN-T.
        public let durations: [Int]
        /// NeMo `max_prob` confidence of each token, (N·p − 1)/(N − 1) over tokens and blank.
        public let confidences: [Float]

        /// Durations and confidences default to 1 for results built without them (window tests).
        public init(text: String, tokenIDs: [Int], emissionFrames: [Int], durations: [Int]? = nil, confidences: [Float]? = nil) {
            self.text = text
            self.tokenIDs = tokenIDs
            self.emissionFrames = emissionFrames
            self.durations = durations ?? Array(repeating: 1, count: tokenIDs.count)
            self.confidences = confidences ?? Array(repeating: 1, count: tokenIDs.count)
        }
    }

    /// Tokens collected from one or more windows, in audio order.
    struct TokenTrack {
        var ids: [Int] = [], frames: [Int] = [], durations: [Int] = [], confidences: [Float] = []

        mutating func append(_ result: Result, at index: Int, frameOffset: Int = 0) {
            ids.append(result.tokenIDs[index])
            frames.append(frameOffset + result.emissionFrames[index])
            durations.append(result.durations[index])
            confidences.append(result.confidences[index])
        }

        mutating func append(_ result: Result, frameOffset: Int = 0) {
            for index in result.tokenIDs.indices { append(result, at: index, frameOffset: frameOffset) }
        }

        func result(text: String) -> Result {
            Result(text: text, tokenIDs: ids, emissionFrames: frames, durations: durations, confidences: confidences)
        }
    }
    struct Window: Decodable {
        let samples: Int
        let function: String?
        var melFrames: Int { samples / 160 + 1 }
        var encoderFrames: Int { (melFrames + 7) / 8 }
    }
    struct Config: Decodable {
        let architecture: String
        /// "rnnt" for Parakeet Unified; absent (TDT) for v3.
        let decoder: String?
        let precision: String
        let sample_rate: Int
        let window_samples: Int
        let blank_id: Int
        let vocabulary_size: Int
        /// TDT duration classes; absent for RNN-T.
        let durations: [Int]?
        let max_symbols: Int
        let punctuation: [String]
        let files: [String: String]
        let windows: [Window]?
        let encoder_stages: [String]?
        let staged_preprocessor: Bool?
        /// FP32 predictor and joint weights for the CPU decoder (RNN-T only).
        let cpu_decoder: CpuDecoder?
    }
    struct CpuDecoder: Decodable {
        struct Tensor: Decodable, Equatable {
            let name: String, shape: [Int]
            init(name: String, shape: [Int]) { self.name = name; self.shape = shape }
            init(from decoder: Decoder) throws {   // ["name", [shape...]]
                var pair = try decoder.unkeyedContainer()
                name = try pair.decode(String.self); shape = try pair.decode([Int].self)
            }
        }
        let file: String
        let dtype: String
        let tensors: [Tensor]
    }

    /// Joint layout and step rule of one supported model.
    struct Contract: Equatable {
        let architecture: String
        let blank: Int
        /// TDT duration classes after the token logits; empty for RNN-T.
        let durations: [Int]
        var isRNNT: Bool { durations.isEmpty }
        /// Token logits including blank, then one logit per duration class.
        var logitCount: Int { blank + 1 + durations.count }

        static let v3 = Contract(architecture: "parakeet-tdt-0.6b-v3", blank: 8192, durations: [0, 1, 2, 3, 4])
        static let unifiedEn = Contract(architecture: "parakeet-unified-en-0.6b", blank: 1024, durations: [])

        static func resolve(_ config: Config) -> Contract? {
            let contract: Contract
            switch (config.architecture, config.decoder) {
            case (Self.v3.architecture, nil), (Self.v3.architecture, "tdt"): contract = .v3
            case (Self.unifiedEn.architecture, "rnnt"): contract = .unifiedEn
            default: return nil
            }
            guard config.blank_id == contract.blank, config.vocabulary_size == contract.blank,
                  (config.durations ?? []) == contract.durations else { return nil }
            return contract
        }
    }
    private let allowExperimentalLongAudio: Bool
    private let config: Config
    let contract: Contract
    /// True for Parakeet Unified's RNN-T contract, false for TDT v3.
    public var decodesRNNT: Bool { contract.isRNNT }
    private let pieces: [String]
    private var wordPunctuation: Set<String> = []
    private let predictor: MLModel, joint: MLModel
    /// RNN-T decoding on the CPU when the folder ships `cpu_decoder` weights; otherwise the Core ML
    /// predictor and joint are used.
    private let cpuDecoder: RnntDecoder?
    private let frontendURL: URL, encoderURL: URL
    private let featureExtractor: MLModel?
    private let featureAudio: MLMultiArray?
    private let encoderTailURL: URL?
    private let encoderComputeUnits: MLComputeUnits
    private var requestWindowSamples = 240000
    private var loadedWindows: [Int: WindowModels] = [:]
    private let keepsAllWindows: Bool
    // Internal integration-test probe; absent during normal inference.
    var diagnostics: ((String, [Float]) -> Void)?
    private final class WindowModels {
        let window: Window
        let frontend: MLModel, encoder: MLModel
        let encoderTail: MLModel?
        let audio: MLMultiArray, audioLength: MLMultiArray
        let mel: MLMultiArray, melLength: MLMultiArray, featuresSnapshot: MLMultiArray
        let encoderOptions: MLPredictionOptions?
        init(window: Window, frontend: MLModel, encoder: MLModel, encoderTail: MLModel? = nil, stagedPreprocessor: Bool = false) throws {
            self.window = window; self.frontend = frontend; self.encoder = encoder; self.encoderTail = encoderTail
            audio = try ParakeetTdtV3.array([1,window.samples])
            audioLength = try ParakeetTdtV3.array([1],.int32)
            mel = try ParakeetTdtV3.array([1,128,window.melFrames])
            melLength = try ParakeetTdtV3.array([1],.int32)
            featuresSnapshot = try ParakeetTdtV3.array([1,1024,window.encoderFrames])
            if encoderTail == nil && encoder.configuration.computeUnits != .cpuAndGPU && encoder.configuration.computeUnits != .all {
                let options = MLPredictionOptions()
                options.outputBackings = [
                    "encoder":try ParakeetTdtV3.alignedOutput([1,1024,window.encoderFrames]),
                    "encoder_length":try ParakeetTdtV3.alignedOutput([1],.int32),
                ]
                encoderOptions = options
            } else { encoderOptions = nil }
            let frontendInputs = stagedPreprocessor
                ? ["raw_mel":[1,128,window.melFrames],"audio_length":[1]]
                : ["audio_signal":[1,window.samples],"audio_length":[1]]
            try ParakeetTdtV3.validate(frontend, inputs: frontendInputs, outputs:["mel":[1,128,window.melFrames],"mel_length":[1]])
            if let tail = encoderTail {
                try ParakeetTdtV3.validate(encoder, inputs: ["mel":[1,128,window.melFrames],"mel_length":[1]], outputs:[:])
                let producers = encoder.modelDescription.inputDescriptionsByName.merging(encoder.modelDescription.outputDescriptionsByName) { _, output in output }
                for (name, input) in tail.modelDescription.inputDescriptionsByName {
                    guard let from = producers[name]?.multiArrayConstraint, let to = input.multiArrayConstraint,
                          from.shape.map(\.intValue) == to.shape.map(\.intValue), from.dataType == to.dataType else {
                        throw ParakeetError.badConfig("Incompatible encoder-stage connection: \(name)")
                    }
                }
                try ParakeetTdtV3.validate(tail, inputs:[:], outputs:["encoder":[1,1024,window.encoderFrames],"encoder_length":[1]])
            } else {
                try ParakeetTdtV3.validate(encoder, inputs: ["mel":[1,128,window.melFrames],"mel_length":[1]], outputs:["encoder":[1,1024,window.encoderFrames],"encoder_length":[1]])
            }
        }
    }
    private let targets: MLMultiArray, hidden: MLMultiArray, cell: MLMultiArray
    private let candidateHidden: MLMultiArray, candidateCell: MLMultiArray
    private let decoderStep: MLMultiArray, encoderStep: MLMultiArray

    /// `keepsAllWindows` keeps every window function loaded after first use (default on macOS);
    /// otherwise only the window of the latest request stays loaded (default on iOS).
    public init(directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine, cacheDirectory: URL,
                allowExperimentalLongAudio: Bool = true, decoderComputeUnits: MLComputeUnits? = nil,
                keepsAllWindows: Bool? = nil) throws {
        #if os(iOS)
        self.keepsAllWindows = keepsAllWindows ?? false
        #else
        self.keepsAllWindows = keepsAllWindows ?? true
        #endif
        #if os(iOS)
        guard computeUnits != .cpuAndGPU && computeUnits != .all,
              decoderComputeUnits != .cpuAndGPU && decoderComputeUnits != .all else {
            throw ParakeetError.badConfig("V3 GPU inference is not qualified on iOS; use CPU or Neural Engine")
        }
        #endif
        self.allowExperimentalLongAudio = allowExperimentalLongAudio
        self.encoderComputeUnits = computeUnits
        let decoder = JSONDecoder()
        config = try decoder.decode(Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard let contract = Contract.resolve(config), ["fp32", "fp16"].contains(config.precision),
              config.sample_rate == 16000, config.window_samples == 240000, config.max_symbols == 10
        else { throw ParakeetError.badConfig("Unsupported v3 tensor/decoder contract") }
        self.contract = contract
        if let windows = config.windows {
            // Ascending standard windows named w<seconds>; long audio relies on the 15 s and 30 s ones.
            let samples = windows.map(\.samples)
            guard samples == samples.sorted(), Set(samples).count == samples.count,
                  Set(samples).isSubset(of: [80000, 160000, 240000, 320000, 480000]), samples.contains(240000), samples.contains(480000),
                  windows.allSatisfy({ $0.function == "w\($0.samples / 16000)" }) else {
                throw ParakeetError.badConfig("Expected ordered w<seconds> window functions including w15 and w30")
            }
        }
        guard config.encoder_stages == nil || config.encoder_stages == ["Encoder", "EncoderTail"] else {
            throw ParakeetError.badConfig("Unsupported encoder-stage contract")
        }
        let required = ["Preprocessor", "Predictor", "Joint"] + (config.encoder_stages ?? ["Encoder"])
        guard config.files["vocab.json"] != nil else { throw ParakeetError.badConfig("Missing vocabulary checksum") }
        for name in required {
            for file in ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin"] {
                guard config.files[name + ".mlpackage/" + file] != nil else {
                    throw ParakeetError.badConfig("Missing v3 model checksum: \(name)/\(file)")
                }
            }
        }
        let manifest = config.files
        if let cpu = config.cpu_decoder {
            guard contract.isRNNT, cpu.dtype == "float32", manifest[cpu.file] != nil,
                  cpu.tensors == RnntDecoder.layout(tokens: contract.blank + 1).map({ CpuDecoder.Tensor(name: $0.0, shape: $0.1) })
            else { throw ParakeetError.badConfig("Unsupported CPU decoder contract") }
        }
        try Self.verify(manifest, in: directory, cache: cacheDirectory)
        pieces = try decoder.decode([String].self, from: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
        guard pieces.count == contract.blank, pieces.first == "<unk>" else {
            throw ParakeetError.badConfig("Expected \(contract.blank) tokenizer pieces")
        }
        wordPunctuation = ParakeetWords.punctuation(in: pieces)
        func load(_ name: String, _ units: MLComputeUnits) throws -> MLModel {
            try autoreleasepool {
                let url = try ParakeetTdtCtc110M.modelURL(directory, name, cache: cacheDirectory, allowBundledCompiled: false,
                                                                manifest: manifest)
                return try ParakeetTdtCtc110M.load(url, units, nil)
            }
        }
        let hasTail = config.encoder_stages != nil
        let compiled = try autoreleasepool { () throws -> (URL,URL,URL?) in
            let front = try ParakeetTdtCtc110M.modelURL(directory,"Preprocessor",cache:cacheDirectory,allowBundledCompiled:false,manifest:manifest)
            let enc = try ParakeetTdtCtc110M.modelURL(directory,"Encoder",cache:cacheDirectory,allowBundledCompiled:false,manifest:manifest)
            let tail = hasTail ? try ParakeetTdtCtc110M.modelURL(directory,"EncoderTail",cache:cacheDirectory,allowBundledCompiled:false,manifest:manifest) : nil
            return (front,enc,tail)
        }
        frontendURL = compiled.0; encoderURL = compiled.1; encoderTailURL = compiled.2
        if config.staged_preprocessor == true {
            guard config.windows != nil else { throw ParakeetError.badConfig("Staged frontend requires w15/w30 functions") }
            featureExtractor = try ParakeetTdtCtc110M.load(compiled.0, .cpuOnly, "features")
            featureAudio = try Self.array([1,1,64512])
            try Self.validate(featureExtractor!, inputs:["waveform":[1,1,64512]], outputs:["raw_mel":[1,128,401]])
        } else { featureExtractor = nil; featureAudio = nil }
        predictor = try load("Predictor", decoderComputeUnits ?? computeUnits)
        joint = try load("Joint", decoderComputeUnits ?? computeUnits)
        cpuDecoder = try config.cpu_decoder.map {
            try RnntDecoder(url: directory.appendingPathComponent($0.file), tokens: contract.blank + 1)
        }
        targets = try Self.array([1, 1], .int32)
        hidden = try Self.array([2, 1, 640]); cell = try Self.array([2, 1, 640])
        candidateHidden = try Self.array([2, 1, 640]); candidateCell = try Self.array([2, 1, 640])
        decoderStep = try Self.array([1, 640, 1]); encoderStep = try Self.array([1, 1024, 1])
        _ = try windowModels()
        try Self.validate(predictor, inputs: ["targets": [1,1], "h_in": [2,1,640], "c_in": [2,1,640]], outputs: ["decoder": [1,640,1], "h_out": [2,1,640], "c_out": [2,1,640]])
        try Self.validate(joint, inputs: ["encoder_step": [1,1024,1], "decoder_step": [1,640,1]], outputs: ["logits": [1,1,1,contract.logitCount]])
    }

    /// Checks every manifest file against its SHA-256. A passing check is recorded in the cache with
    /// each file's device, inode, size and modification and change times, so later loads of the same
    /// unchanged files only stat them (hashing the FP16 export takes about 0.6 s). Writing, replacing or
    /// touching any file, or a different manifest, hashes everything again.
    static func verify(_ files: [String: String], in directory: URL, cache: URL) throws {
        let manifest = files.sorted { $0.key < $1.key }
        for (path, expected) in manifest {
            guard !path.hasPrefix("/"), !path.split(separator: "/").contains(".."), expected.count == 64 else {
                throw ParakeetError.badConfig("Invalid model manifest path or checksum")
            }
        }
        // Taken before hashing, so a file changed while it is read fails the next comparison.
        let identity = Data(manifest.map { path, expected in
            "\(path)\t\(expected)\t\(fileIdentity(directory.appendingPathComponent(path)) ?? "missing")\n"
        }.joined().utf8)
        let key = SHA256.hash(data: Data(directory.resolvingSymlinksInPath().path.utf8))
        let record = cache.appendingPathComponent("verified-" + key.prefix(8).map { String(format: "%02x", $0) }.joined())
        if (try? Data(contentsOf: record)) == identity { return }
        for (path, expected) in manifest {
            let file = try FileHandle(forReadingFrom: directory.appendingPathComponent(path))
            defer { try? file.close() }
            var hash = SHA256()
            while try autoreleasepool(invoking: { () throws -> Bool in
                guard let bytes = try file.read(upToCount: 8 * 1024 * 1024), !bytes.isEmpty else { return false }
                hash.update(data: bytes); return true
            }) {}
            guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == expected else {
                throw ParakeetError.badConfig("Checksum mismatch: \(path)")
            }
        }
        // Best effort: an unwritable cache hashes on every load.
        try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        try? identity.write(to: record, options: .atomic)
    }

    static func fileIdentity(_ url: URL) -> String? {
        var info = stat()
        guard stat(url.path, &info) == 0 else { return nil }
        return "\(info.st_dev):\(info.st_ino):\(info.st_size):\(info.st_mtimespec.tv_sec).\(info.st_mtimespec.tv_nsec)"
            + ":\(info.st_ctimespec.tv_sec).\(info.st_ctimespec.tv_nsec)"
    }

    private func windowSize(for count: Int) -> Int {
        Self.windowSamples(for: count, windows: config.windows?.map(\.samples))
    }
    /// The smallest window that holds the request; longer audio is cut into 30 s windows.
    /// Legacy single-window configs always use 15 s.
    static func windowSamples(for count: Int, windows: [Int]?) -> Int {
        guard let windows else { return 240000 }
        return windows.first(where: { count <= $0 }) ?? 480000
    }
    private func windowModels() throws -> WindowModels {
        if let loaded = loadedWindows[requestWindowSamples] { return loaded }
        // On iOS, retain one encoder function at a time to bound device memory. On the Mac every
        // window stays loaded: reloading a function costs about 100 ms per switch of window size.
        if !keepsAllWindows { autoreleasepool { loadedWindows.removeAll() } }
        let window = config.windows?.first(where: { $0.samples == requestWindowSamples })
            ?? Window(samples: 240000, function: nil)
        let current = try autoreleasepool { () throws -> WindowModels in
            func load(_ name: String, _ units: MLComputeUnits) throws -> MLModel {
                let url: URL
                if name == "Preprocessor" { url = frontendURL }
                else if name == "EncoderTail", let tail = encoderTailURL { url = tail }
                else { url = encoderURL }
                let function = name == "Preprocessor" && config.staged_preprocessor == true
                    ? window.function.map { "normalize_" + $0 } : window.function
                return try ParakeetTdtCtc110M.load(url, units, function)
            }
            return try WindowModels(window:window,frontend:load("Preprocessor",.cpuOnly),encoder:load("Encoder",encoderComputeUnits),encoderTail:encoderTailURL == nil ? nil : load("EncoderTail",encoderComputeUnits),stagedPreprocessor:config.staged_preprocessor == true)
        }
        loadedWindows[requestWindowSamples] = current
        return current
    }

    /// Bounded offline segmentation. Prefer quiet pauses; retain overlap context when no pause is available.
    public func transcribe(file: URL, checkCancellation: () throws -> Void = {}) throws -> Result {
        let reader = try ParakeetAudio.Reader(file)
        var buffer = try reader.read(config.windows == nil ? 240001 : 480001), start = 0
        requestWindowSamples = windowSize(for: buffer.count)
        var track = TokenTrack()
        var pending: PendingWindow?
        try Self.validateSampleCount(buffer.count, allowLongAudio: allowExperimentalLongAudio)
        // Keep short file and PCM requests on exactly the same decode path.
        if buffer.count <= requestWindowSamples {
            return try autoreleasepool {
                try requestWindowSamples > 240000
                    ? decodeLongWindow(buffer, checkCancellation: checkCancellation)
                    : decode(buffer, checkCancellation: checkCancellation)
            }
        }
        while !buffer.isEmpty {
            try checkCancellation()
            let plan = Self.planWindow(buffer, windowSamples: requestWindowSamples)
            let window = try autoreleasepool { try decodeLongWindow(Array(buffer.prefix(plan.count)), checkCancellation: checkCancellation) }
            Self.consume(window, start: start, seam: start + plan.right, pending: &pending, track: &track)
            if buffer.count <= requestWindowSamples { break }
            buffer.removeFirst(plan.advance)
            buffer.append(contentsOf: try reader.read(plan.advance))
            start += plan.advance
        }
        if let last = pending { Self.flush(last, through: last.result.tokenIDs.count, track: &track) }
        return track.result(text: text(track.ids))
    }

    public func transcribe(_ samples: [Float], checkCancellation: () throws -> Void = {}) throws -> Result {
        guard samples.allSatisfy(\.isFinite) else { throw ParakeetError.invalidAudio("PCM samples must be finite") }
        try Self.validateSampleCount(samples.count, allowLongAudio: allowExperimentalLongAudio)
        requestWindowSamples = windowSize(for: samples.count)
        if samples.count <= requestWindowSamples {
            return try autoreleasepool {
                try requestWindowSamples > 240000
                    ? decodeLongWindow(samples, checkCancellation: checkCancellation)
                    : decode(samples, checkCancellation: checkCancellation)
            }
        }
        var start = 0, track = TokenTrack()
        var pending: PendingWindow?
        while start < samples.count {
            try checkCancellation()
            let buffer = Array(samples[start..<min(start + requestWindowSamples + 1, samples.count)])
            let plan = Self.planWindow(buffer, windowSamples: requestWindowSamples)
            let window = try autoreleasepool { try decodeLongWindow(Array(buffer.prefix(plan.count)), checkCancellation: checkCancellation) }
            Self.consume(window, start: start, seam: start + plan.right, pending: &pending, track: &track)
            if buffer.count <= requestWindowSamples { break }
            start += plan.advance
        }
        if let last = pending { Self.flush(last, through: last.result.tokenIDs.count, track: &track) }
        return track.result(text: text(track.ids))
    }

    struct WindowPlan { let count: Int; let advance: Int; let right: Int }
    static func planWindow(_ samples: [Float], windowSamples: Int = 240000) -> WindowPlan {
        if samples.count <= windowSamples {
            return WindowPlan(count: samples.count, advance: samples.count, right: samples.count)
        }
        // 10 ms RMS blocks. Relative threshold adapts to gain; the floor covers digital silence.
        var energy = [Float](repeating: 0, count: windowSamples/160)
        for block in energy.indices {
            var sum: Float = 0
            for index in (block * 160)..<((block + 1) * 160) { sum += samples[index] * samples[index] }
            energy[block] = sqrt(sum / 160)
        }
        let threshold = max(0.0005, (energy.max() ?? 0) * 0.02)
        let cuts = windowSamples == 240000 ? 800...1200 : 2000...2700
        var begin: Int?, best: (length: Int, cut: Int)?
        for index in 0...energy.count {
            if index < energy.count && energy[index] < threshold {
                if begin == nil { begin = index }
            } else if let first = begin {
                let length = index - first, cut = (first + index) / 2
                if length >= 16 && cuts.contains(cut), best == nil || length >= best!.length {
                    best = (length, cut)
                }
                begin = nil
            }
        }
        if let pause = best {
            let cut = ((pause.cut * 160 + 640) / 1280) * 1280
            return WindowPlan(count: windowSamples, advance: cut, right: cut)
        }
        // Continuous speech keeps the original frame-aligned context ownership.
        let stride = ((windowSamples - 63360) / 1280) * 1280
        return WindowPlan(count: windowSamples, advance: stride, right: stride+32000)
    }
    private struct PendingWindow {
        let result: Result
        let start: Int
        let first: Int
        let seam: Int
    }
    private static func flush(_ window: PendingWindow, through end: Int, track: inout TokenTrack) {
        for index in window.first..<max(window.first, end) {
            track.append(window.result, at: index, frameOffset: window.start/1280)
        }
    }
    private static func consume(_ result: Result, start: Int, seam: Int, pending: inout PendingWindow?,
                                track: inout TokenTrack) {
        var first = 0
        if let previous = pending {
            let cut = join(previous.result, previousStart:previous.start, previousFirst:previous.first,
                           next:result, nextStart:start, seam:previous.seam)
            flush(previous, through:cut.previousEnd, track:&track)
            first = cut.nextBegin
        }
        pending = PendingWindow(result:result,start:start,first:first,seam:seam)
    }
    /// Prefer a common token sequence in overlapping audio; emission frames are a fallback.
    static func join(_ previous: Result, previousStart: Int, previousFirst: Int,
                     next: Result, nextStart: Int, seam: Int) -> (previousEnd: Int, nextBegin: Int) {
        if previous.tokenIDs.isEmpty { return (0,0) }
        if next.tokenIDs.isEmpty { return (previous.tokenIDs.count,0) }
        var best: (score: Int, previousEnd: Int, nextBegin: Int)?
        for i in previousFirst..<previous.tokenIDs.count {
            let a = previousStart + previous.emissionFrames[i]*1280
            if abs(a-seam)>48000 { continue }
            for j in next.tokenIDs.indices {
                let b = nextStart + next.emissionFrames[j]*1280
                if abs(b-seam)>48000 || abs(a-b)>16000 { continue }
                var length = 0
                while i+length<previous.tokenIDs.count && j+length<next.tokenIDs.count
                    && length<8 && previous.tokenIDs[i+length]==next.tokenIDs[j+length] {
                    length += 1
                    if length<3 { continue }
                    let endA = previousStart + previous.emissionFrames[i+length-1]*1280
                    let endB = nextStart + next.emissionFrames[j+length-1]*1280
                    if abs(endA-seam)>32000 || abs(endB-seam)>32000 || abs(endA-endB)>16000 { continue }
                    let score = abs(endA-seam)+abs(endB-seam)-length*80
                    if best == nil || score<best!.score { best=(score,i+length,j+length) }
                }
            }
        }
        if let best = best { return (best.previousEnd,best.nextBegin) }
        let end = previous.emissionFrames.firstIndex { previousStart+$0*1280>=seam } ?? previous.tokenIDs.count
        let begin = next.emissionFrames.firstIndex { nextStart+$0*1280>=seam } ?? next.tokenIDs.count
        return (max(previousFirst,end),begin)
    }

    static func validateSampleCount(_ count: Int, allowLongAudio: Bool) throws {
        guard count <= 240000 || allowLongAudio else {
            throw ParakeetError.invalidAudio("This v3 instance is restricted to a single 15-second window")
        }
    }

    private func decodeLongWindow(_ samples: [Float], retryDepth: Int = 0, checkCancellation: () throws -> Void) throws -> Result {
        let first = try autoreleasepool { try decode(samples, checkCancellation: checkCancellation) }
        if retryDepth == 0, requestWindowSamples > 240000, !first.tokenIDs.isEmpty {
            return try recoverPartialBlanks(first, samples: samples, checkCancellation: checkCancellation)
        }
        // The source model can blank an entire speech window at an unlucky cut.
        // Retry with at most two subdivision levels; never erase a nonempty primary result.
        guard first.tokenIDs.isEmpty, samples.count >= 80000 else { return first }
        var squared: Float = 0
        for sample in samples { squared += sample * sample }
        guard sqrt(squared / Float(samples.count)) > 0.0000001 else { return first }
        let cut = samples.count / 2560 * 1280
        let left = try retryDepth < 1
            ? decodeLongWindow(Array(samples[..<cut]), retryDepth: retryDepth + 1, checkCancellation: checkCancellation)
            : decode(Array(samples[..<cut]), checkCancellation: checkCancellation)
        let right = try retryDepth < 1
            ? decodeLongWindow(Array(samples[cut...]), retryDepth: retryDepth + 1, checkCancellation: checkCancellation)
            : decode(Array(samples[cut...]), checkCancellation: checkCancellation)
        var track = TokenTrack()
        track.append(left)
        track.append(right, frameOffset: cut / 1280)
        return track.result(text: text(track.ids))
    }

    struct BlankSpan { let index: Int; let left: Int; let right: Int }

    /// A large token-empty span is retried only when its audio is active and varies in energy.
    static func blankSpans(_ primary: Result, samples: [Float]) -> [BlankSpan] {
        let frames = (samples.count / 160 + 7) / 8
        var gaps: [BlankSpan] = []
        for index in 0...primary.tokenIDs.count {
            let left = index == 0 ? -1 : primary.emissionFrames[index - 1]
            let right = index == primary.tokenIDs.count ? frames : primary.emissionFrames[index]
            if right - left >= 64 { gaps.append(BlankSpan(index: index, left: left, right: right)) }
        }
        guard !gaps.isEmpty else { return [] }
        var energy: [Float] = []
        for begin in stride(from: 0, to: samples.count - 159, by: 160) {
            var squared: Float = 0
            for i in begin..<(begin + 160) { squared += samples[i] * samples[i] }
            energy.append(sqrt(squared / 160))
        }
        let threshold = max(0.0005, (energy.max() ?? 0) * 0.05)
        gaps = gaps.filter { gap in
            let begin = max(0, (gap.left + 1) * 8), end = min(energy.count, gap.right * 8)
            guard end > begin else { return false }
            let values = energy[begin..<end].sorted()
            return values.filter { $0 > threshold }.count * 2 >= values.count
                && values[values.count * 3 / 4] > max(0.000001, values[values.count / 4]) * 1.5
        }
        return gaps
    }

    /// Keep every primary token; fill only gaps bounded by matching tokens at the same audio time.
    static func insertRecoveredTokens(_ primary: Result, candidate: Result, gaps: [BlankSpan]) -> Result? {
        guard !primary.tokenIDs.isEmpty, primary.tokenIDs.count == primary.emissionFrames.count,
              candidate.tokenIDs.count == candidate.emissionFrames.count else { return nil }
        let candidateIDs = candidate.tokenIDs, candidateFrames = candidate.emissionFrames
        func anchor(_ first: Int) -> Int? {
            guard first >= 0, first + 3 <= primary.tokenIDs.count, candidateIDs.count >= 3 else { return nil }
            var best: (score: Int, index: Int)?
            for j in 0...(candidateIDs.count - 3) {
                var score = 0, matches = true
                for k in 0..<3 {
                    let distance = abs(primary.emissionFrames[first + k] - candidateFrames[j + k])
                    if primary.tokenIDs[first + k] != candidateIDs[j + k] || distance > 16 { matches = false; break }
                    score += distance
                }
                if matches, best == nil || score < best!.score { best = (score, j) }
            }
            return best?.index
        }
        var insertions: [Int: Range<Int>] = [:]
        for gap in gaps {
            let begin: Int, end: Int
            if gap.index == 0 { begin = 0 }
            else if let left = anchor(gap.index - 3) { begin = left + 3 }
            else { continue }
            if gap.index == primary.tokenIDs.count { end = candidateIDs.count }
            else if let right = anchor(gap.index) { end = right }
            else { continue }
            guard end - begin >= 3,
                  candidateFrames[begin] >= gap.left,
                  candidateFrames[end - 1] <= gap.right else { continue }
            insertions[gap.index] = begin..<end
        }
        guard !insertions.isEmpty else { return nil }
        var track = TokenTrack()
        for i in 0...primary.tokenIDs.count {
            if let added = insertions[i] { for j in added { track.append(candidate, at: j) } }
            if i < primary.tokenIDs.count { track.append(primary, at: i) }
        }
        return track.result(text: "")
    }

    private func recoverPartialBlanks(_ primary: Result, samples: [Float], checkCancellation: () throws -> Void) throws -> Result {
        let gaps = Self.blankSpans(primary, samples: samples)
        guard !gaps.isEmpty else { return primary }
        let originalSize = requestWindowSamples
        requestWindowSamples = 240000
        defer { requestWindowSamples = originalSize }
        var start = 0, candidates = TokenTrack()
        var pending: PendingWindow?
        while start < samples.count {
            try checkCancellation()
            let buffer = Array(samples[start..<min(start + 240001, samples.count)])
            let plan = Self.planWindow(buffer, windowSamples: 240000)
            let candidate = try autoreleasepool { try decode(Array(buffer.prefix(plan.count)), checkCancellation: checkCancellation) }
            Self.consume(candidate, start: start, seam: start + plan.right, pending: &pending, track: &candidates)
            if buffer.count <= 240000 { break }
            start += plan.advance
        }
        if let last = pending { Self.flush(last, through: last.result.tokenIDs.count, track: &candidates) }
        let candidate = candidates.result(text: "")
        guard let recovered = Self.insertRecoveredTokens(primary, candidate: candidate, gaps: gaps) else { return primary }
        return Result(text: text(recovered.tokenIDs), tokenIDs: recovered.tokenIDs, emissionFrames: recovered.emissionFrames,
                      durations: recovered.durations, confidences: recovered.confidences)
    }

    private func preprocess(_ samples: [Float], models: WindowModels) throws -> MLFeatureProvider {
        guard let featureExtractor, let featureAudio else {
            return try models.frontend.prediction(from: Self.provider(["audio_signal": models.audio, "audio_length": models.audioLength]))
        }
        let frames = models.window.melFrames
        let rawFeatures = try Self.array([1,128,frames])
        for firstFrame in stride(from: 0, to: frames, by: 401) {
            featureAudio.withUnsafeMutableBufferPointer(ofType: Float.self) { destination, strides in
                for i in 0..<64512 {
                    let index = firstFrame * 160 + i - 256
                    let value: Float
                    if index >= 0 && index < samples.count {
                        value = samples[index] - 0.97 * (index > 0 ? samples[index - 1] : 0)
                    } else { value = 0 }
                    destination[i * strides[2]] = value
                }
            }
            let input = try Self.provider(["waveform": featureAudio])
            let output = try featureExtractor.prediction(from: input)
            let raw = try Self.output(output, "raw_mel", [1,128,401])
            guard raw.dataType == .float32 else { throw ParakeetError.badConfig("Expected FP32 raw mel features") }
            withExtendedLifetime((input, output)) {
                raw.withUnsafeBufferPointer(ofType: Float.self) { source in
                    let strides = raw.strides.map(\.intValue)
                    rawFeatures.withUnsafeMutableBufferPointer(ofType: Float.self) { destination, targetStrides in
                        for channel in 0..<128 {
                            for frame in 0..<min(401, frames - firstFrame) {
                                destination[channel * targetStrides[1] + (firstFrame + frame) * targetStrides[2]] = source[channel * strides[1] + frame * strides[2]]
                            }
                        }
                    }
                }
            }
        }
        let input = try Self.provider(["raw_mel": rawFeatures, "audio_length": models.audioLength])
        let output = try models.frontend.prediction(from: input)
        // Snapshot while the full-window normalization inputs and outputs are retained.
        return try withExtendedLifetime((input, output)) {
            let mel = try Self.array([1,128,frames])
            try Self.copy(Self.output(output, "mel", [1,128,frames]), into: mel)
            let length = try Self.array([1], .int32)
            length[0] = NSNumber(value: try Self.scalarLength(Self.output(output, "mel_length", [1])))
            return try Self.provider(["mel": mel, "mel_length": length])
        }
    }

    /// Frontend and encoder for one window: the encoder output snapshot and its valid frame count.
    private func encode(_ samples: [Float], checkCancellation: () throws -> Void) throws -> (features: MLMultiArray, frames: Int) {
        let models = try windowModels()
        let audio = models.audio, audioLength = models.audioLength
        Self.zero(audio)
        samples.withUnsafeBufferPointer { p in audio.dataPointer.assumingMemoryBound(to: Float.self).update(from: p.baseAddress!, count: samples.count) }
        audioLength[0] = NSNumber(value: samples.count)
        let frames = try autoreleasepool { () throws -> Int in
            let melOutput = try preprocess(samples, models: models)
            try checkCancellation()
            let mel = try Self.output(melOutput, "mel", [1,128,models.window.melFrames])
            let length = try Self.scalarLength(Self.output(melOutput, "mel_length", [1]))
            guard length == samples.count / 160 else { throw ParakeetError.badConfig("Incorrect v3 mel valid length") }
            try withExtendedLifetime((melOutput, models.frontend)) { try Self.copy(mel, into: models.mel) }
            diagnose("mel", models.mel)
            models.melLength[0] = NSNumber(value: length)
            let input = try Self.provider(["mel": models.mel, "mel_length": models.melLength])
            let firstEncoded: MLFeatureProvider
            if let options = models.encoderOptions { firstEncoded = try models.encoder.prediction(from: input, options: options) }
            else { firstEncoded = try models.encoder.prediction(from: input) }
            try checkCancellation()
            let encoded: MLFeatureProvider, tailInput: MLFeatureProvider?
            if let tail = models.encoderTail {
                var values: [String: MLFeatureValue] = [:]
                for name in tail.modelDescription.inputDescriptionsByName.keys {
                    guard let value = firstEncoded.featureValue(for: name) ?? input.featureValue(for: name) else {
                        throw ParakeetError.badConfig("Missing encoder-stage input: \(name)")
                    }
                    values[name] = value
                }
                let provider = try MLDictionaryFeatureProvider(dictionary: values)
                tailInput = provider
                encoded = try tail.prediction(from: provider)
                try checkCancellation()
            } else { tailInput = nil; encoded = firstEncoded }
            let features = try Self.output(encoded, "encoder", [1,1024,models.window.encoderFrames])
            let frames = try Self.scalarLength(Self.output(encoded, "encoder_length", [1]))
            guard frames == (samples.count / 160 + 7) / 8 else { throw ParakeetError.badConfig("Incorrect v3 encoder valid length") }
            try withExtendedLifetime((input, firstEncoded, tailInput, encoded, models.encoder, models.encoderTail)) { try Self.copy(features, into: models.featuresSnapshot) }
            return frames
        }
        return (models.featuresSnapshot, frames)
    }

    private func decode(_ samples: [Float], checkCancellation: () throws -> Void) throws -> Result {
        try checkCancellation()
        guard samples.count <= requestWindowSamples, samples.allSatisfy(\.isFinite) else { throw ParakeetError.invalidAudio("Invalid v3 window") }
        diagnostics?("pcm", samples)
        // Fewer than one source mel hop has no valid encoder frames.
        if samples.count < 160 { return Result(text: "", tokenIDs: [], emissionFrames: []) }
        Self.zero(hidden); Self.zero(cell)
        let (features, frames) = try encode(samples, checkCancellation: checkCancellation)
        diagnose("encoder", features)
        if let cpuDecoder { return try decodeOnCPU(cpuDecoder, features: features, frames: frames, checkCancellation: checkCancellation) }
        let blank = contract.blank, logitCount = contract.logitCount
        var time = 0, symbols = 0, lastToken = blank, needsPrediction = true
        var ids: [Int] = [], times: [Int] = [], spans: [Int] = [], confidences: [Float] = []
        while time < frames {
            try checkCancellation()
            if needsPrediction {
                targets[0] = NSNumber(value: lastToken)
                diagnostics?("target", [Float(lastToken)])
                diagnose("h_in", hidden); diagnose("c_in", cell)
                try autoreleasepool {
                    // Publish the current mutable state for each prediction. Keep the provider
                    // and its output alive until all framework-owned tensors have been copied.
                    let input = try Self.provider(["targets": targets, "h_in": hidden, "c_in": cell])
                    let predicted = try predictor.prediction(from: input)
                    try withExtendedLifetime((input, predicted)) {
                        try Self.copy(Self.output(predicted, "decoder", [1,640,1]), into: decoderStep)
                        try Self.copy(Self.output(predicted, "h_out", [2,1,640]), into: candidateHidden)
                        try Self.copy(Self.output(predicted, "c_out", [2,1,640]), into: candidateCell)
                        diagnose("decoder", decoderStep)
                        diagnose("h_out", candidateHidden); diagnose("c_out", candidateCell)
                    }
                }
                needsPrediction = false
            }
            let src = features.dataPointer.assumingMemoryBound(to: Float.self), dst = encoderStep.dataPointer.assumingMemoryBound(to: Float.self)
            guard features.dataType == .float32 else { throw ParakeetError.badConfig("Expected float32 encoder output") }
            let strides = features.strides.map(\.intValue)
            for channel in 0..<1024 { dst[channel] = src[channel * strides[1] + time * strides[2]] }
            diagnose("encoder_step", encoderStep)
            let (token, duration, confidence) = try autoreleasepool {
                let input = try Self.provider(["encoder_step": encoderStep, "decoder_step": decoderStep])
                let output = try joint.prediction(from: input)
                return try withExtendedLifetime((input, output)) {
                    let logits = try Self.output(output, "logits", [1,1,1,logitCount])
                    diagnose("logits", logits)
                    let token = try Self.argmax(logits, 0..<(blank + 1))
                    // RNN-T stays on the frame after a token and advances one frame on blank.
                    let duration = contract.isRNNT ? (token == blank ? 1 : 0)
                        : contract.durations[try Self.argmax(logits, (blank + 1)..<logitCount) - blank - 1]
                    // Only emitted tokens need a confidence; blanks skip the softmax.
                    var confidence: Float = 0
                    if token != blank { confidence = try Self.maxProbConfidence(logits, token: token, count: blank + 1) }
                    return (token, duration, confidence)
                }
            }
            diagnostics?("decision", [Float(time), Float(token), Float(duration)])
            if token != blank {
                // NeMo gives an RNN-T token a one-frame span.
                let span = contract.isRNNT ? 1 : duration
                ids.append(token); times.append(time); spans.append(span); confidences.append(confidence); lastToken = token
                try Self.copy(candidateHidden, into: hidden); try Self.copy(candidateCell, into: cell)
                needsPrediction = true
            }
            time += duration; symbols += 1
            if duration > 0 || symbols == config.max_symbols {
                if symbols == config.max_symbols { time += 1 }
                symbols = 0
            }
        }
        return Result(text: text(ids), tokenIDs: ids, emissionFrames: times, durations: spans, confidences: confidences)
    }

    /// Greedy RNN-T with the CPU decoder, with the same stepping rule as `decode`: stay on the frame
    /// after a token, advance on blank or after `max_symbols` tokens.
    private func decodeOnCPU(_ decoder: RnntDecoder, features: MLMultiArray, frames: Int,
                             checkCancellation: () throws -> Void) throws -> Result {
        let shape = features.shape.map(\.intValue), strides = features.strides.map(\.intValue)
        guard features.dataType == .float32, shape.count == 3, shape[1] == 1024, strides[2] == 1, strides[1] == shape[2] else {
            throw ParakeetError.badConfig("Expected a contiguous float32 encoder snapshot")
        }
        decoder.project(features.dataPointer.assumingMemoryBound(to: Float.self), frames: frames, capacity: shape[2])
        decoder.reset()
        let blank = contract.blank
        var time = 0, symbols = 0
        var ids: [Int] = [], times: [Int] = [], confidences: [Float] = []
        while time < frames {
            if symbols == 0 { try checkCancellation() }
            let (token, confidence) = decoder.decide(frame: time)
            if token == blank {
                time += 1; symbols = 0
                continue
            }
            ids.append(token); times.append(time); confidences.append(confidence)
            decoder.step(token)
            symbols += 1
            if symbols == config.max_symbols { time += 1; symbols = 0 }
        }
        return Result(text: text(ids), tokenIDs: ids, emissionFrames: times,
                      durations: Array(repeating: 1, count: ids.count), confidences: confidences)
    }

    // MARK: - Warm-up

    /// Loads and runs every encoder window function once (15 s, then 30 s), so the Neural Engine
    /// compiles them now rather than during a user's first short or first long recording.
    /// Every window then stays loaded on macOS (only the 15 s window on iOS). Safe to call again;
    /// later calls hit the compile cache.
    public func warmUp() throws {
        let sizes = config.windows?.map(\.samples) ?? [240000]
        for size in sizes.reversed() {
            requestWindowSamples = size
            _ = try autoreleasepool { try decode([Float](repeating: 0, count: min(size, 320000)), checkCancellation: {}) }
        }
        requestWindowSamples = 240000
    }

    // MARK: - Ranked alternatives

    /// One beam-search hypothesis, best first in `alternatives`.
    public struct Hypothesis: Sendable, Equatable {
        public let text: String
        public let tokenIDs: [Int]
        /// Total log probability (tokens and durations).
        public let score: Float
        /// `score / (tokens + 1)`, the ranking score when normalization is on.
        public let normalizedScore: Float
    }

    private struct PredictorState {
        let hidden: MLMultiArray, cell: MLMultiArray
    }

    private struct BeamState {
        var tokens: [Int] = []
        var score: Float = 0
        var state: PredictorState
        var next: PredictorState?
        var output: MLMultiArray?
        var lastFrame = 0
        var lastToken = 8192
        var predictionValid = false
    }

    /// Ranked TDT beam search over one window of at most 30 s, as parakeet.cpp's `tdt_beam_search`
    /// (NeMo's default TDT beam): the same expansions, duplicate merging, pruning and length
    /// normalization. Used for vocabulary-aware choices among close candidates, not plain transcription.
    public func alternatives(_ samples: [Float], beamSize: Int = 4, nbest: Int = 4, normalizeScores: Bool = true,
                             checkCancellation: () throws -> Void = {}) throws -> [Hypothesis] {
        guard !contract.isRNNT else { throw ParakeetError.badConfig("Ranked alternatives require a TDT model") }
        guard (1...16).contains(beamSize), (1...beamSize).contains(nbest) else {
            throw ParakeetError.badConfig("Require 16 >= beam size >= n-best >= 1")
        }
        guard samples.allSatisfy(\.isFinite) else { throw ParakeetError.invalidAudio("PCM samples must be finite") }
        requestWindowSamples = windowSize(for: samples.count)
        guard samples.count <= requestWindowSamples else {
            throw ParakeetError.invalidAudio("Alternatives need audio that fits one window (30 s or less)")
        }
        if samples.count < 160 { return [Hypothesis(text: "", tokenIDs: [], score: 0, normalizedScore: 0)] }
        let (features, frames) = try encode(samples, checkCancellation: checkCancellation)
        return try beamSearch(features: features, frames: frames, beam: min(beamSize, 8192), nbest: nbest,
                              normalize: normalizeScores, checkCancellation: checkCancellation)
    }

    private func predict(_ label: Int, from state: PredictorState) throws -> (output: MLMultiArray, next: PredictorState) {
        try autoreleasepool {
            targets[0] = NSNumber(value: label)
            let input = try Self.provider(["targets": targets, "h_in": state.hidden, "c_in": state.cell])
            let predicted = try predictor.prediction(from: input)
            return try withExtendedLifetime((input, predicted)) {
                let output = try Self.array([1, 640, 1]), hidden = try Self.array([2, 1, 640]), cell = try Self.array([2, 1, 640])
                try Self.copy(Self.output(predicted, "decoder", [1,640,1]), into: output)
                try Self.copy(Self.output(predicted, "h_out", [2,1,640]), into: hidden)
                try Self.copy(Self.output(predicted, "c_out", [2,1,640]), into: cell)
                return (output, PredictorState(hidden: hidden, cell: cell))
            }
        }
    }

    private func jointLogits(_ decoderOutput: MLMultiArray) throws -> [Float] {
        try autoreleasepool {
            let input = try Self.provider(["encoder_step": encoderStep, "decoder_step": decoderOutput])
            let output = try joint.prediction(from: input)
            return try withExtendedLifetime((input, output)) {
                let logits = try Self.output(output, "logits", [1,1,1,8198])
                guard logits.dataType == .float32 else { throw ParakeetError.badConfig("Expected float32 joint logits") }
                return logits.withUnsafeBufferPointer(ofType: Float.self) { buffer in
                    let stride = logits.strides[3].intValue
                    return (0..<8198).map { buffer[$0 * stride] }
                }
            }
        }
    }

    static func logSoftmax(_ values: ArraySlice<Float>) -> [Float] {
        let maximum = values.max() ?? 0
        var sum = 0.0
        for value in values { sum += exp(Double(value) - Double(maximum)) }
        let denominator = log(sum)
        return values.map { Float(Double($0) - Double(maximum) - denominator) }
    }

    /// Indices of the `k` best scores, ties to the lower index.
    static func topK(_ scores: [Float], count n: Int, _ k: Int) -> [(score: Float, index: Int)] {
        // Typed and split up: Swift 6.0 and 6.1 time out inferring the chain.
        var ranked: [(score: Float, index: Int)] = (0..<n).map { (score: scores[$0], index: $0) }
        ranked.sort { (a: (score: Float, index: Int), b: (score: Float, index: Int)) -> Bool in
            if a.score != b.score { return a.score > b.score }
            return a.index < b.index
        }
        return Array(ranked.prefix(min(n, k)))
    }

    static func logAddExp(_ a: Float, _ b: Float) -> Float {
        let high = max(a, b), low = min(a, b)
        return high + log1p(exp(low - high))
    }

    private static func mergeDuplicates(_ hypotheses: inout [BeamState]) {
        hypotheses.sort { $0.score > $1.score }
        var merged: [BeamState] = []
        for candidate in hypotheses {
            if let index = merged.firstIndex(where: { $0.lastFrame == candidate.lastFrame && $0.tokens == candidate.tokens }) {
                merged[index].score = logAddExp(merged[index].score, candidate.score)
            } else {
                merged.append(candidate)
            }
        }
        hypotheses = merged
    }

    private func beamSearch(features: MLMultiArray, frames: Int, beam: Int, nbest: Int, normalize: Bool,
                            checkCancellation: () throws -> Void) throws -> [Hypothesis] {
        let blank = 8192, durations = contract.durations
        let tokenBeam = min(beam, blank - 1), durationBeam = min(beam, durations.count)
        let zeroDuration = durations.firstIndex(of: 0)
        guard features.dataType == .float32 else { throw ParakeetError.badConfig("Expected float32 encoder output") }
        let zeroState = PredictorState(hidden: try Self.array([2, 1, 640]), cell: try Self.array([2, 1, 640]))
        Self.zero(zeroState.hidden); Self.zero(zeroState.cell)
        var kept = [BeamState(state: zeroState)]
        let source = features.dataPointer.assumingMemoryBound(to: Float.self), strides = features.strides.map(\.intValue)
        let step = encoderStep.dataPointer.assumingMemoryBound(to: Float.self)

        for time in 0..<frames {
            try checkCancellation()
            for channel in 0..<1024 { step[channel] = source[channel * strides[1] + time * strides[2]] }
            var current = kept.filter { $0.lastFrame == time }
            kept = kept.filter { $0.lastFrame > time }

            while !current.isEmpty {
                let bestIndex = current.indices.max { current[$0].score < current[$1].score }!
                var best = current.remove(at: bestIndex)
                if !best.predictionValid {
                    let predicted = try predict(best.lastToken, from: best.state)
                    best.output = predicted.output; best.next = predicted.next; best.predictionValid = true
                }
                let logits = try jointLogits(best.output!)
                let tokenLogp = Self.logSoftmax(logits[0..<8193])
                let durationLogp = Self.logSoftmax(logits[8193..<8198])
                guard tokenLogp.allSatisfy(\.isFinite), durationLogp.allSatisfy(\.isFinite) else {
                    throw ParakeetError.badConfig("Nonfinite beam log probability")
                }
                let bestDurations = Self.topK(durationLogp, count: durations.count, durationBeam)
                let bestTokens = Self.topK(tokenLogp, count: blank, tokenBeam)
                var pairs: [(score: Float, token: Int, duration: Int)] = []
                for d in bestDurations { for t in bestTokens { pairs.append((d.score + t.score, t.index, d.index)) } }
                pairs.sort {
                    if $0.score != $1.score { return $0.score > $1.score }
                    if $0.duration != $1.duration { return $0.duration < $1.duration }
                    return $0.token < $1.token
                }
                for pair in pairs.prefix(tokenBeam) {
                    let duration = durations[pair.duration]
                    var child = best
                    child.score += pair.score
                    // A zero-duration step that costs nothing (log p rounds to 0 in float32) could
                    // expand forever. parakeet.cpp aborts the whole search there; skip just that step.
                    if duration == 0 && !(child.score < best.score) { continue }
                    child.tokens.append(pair.token)
                    child.state = best.next!
                    child.next = nil; child.output = nil
                    child.lastToken = pair.token
                    child.predictionValid = false
                    child.lastFrame += duration
                    if duration == 0 { current.append(child) } else { kept.append(child) }
                }
                for ranked in bestDurations {
                    var index = ranked.index
                    if index == zeroDuration {
                        guard bestDurations.count == 1 else { continue }
                        index = durations.indices.filter { durations[$0] > 0 }.max { durationLogp[$0] < durationLogp[$1] }!
                    }
                    var child = best
                    child.score += tokenLogp[blank] + durationLogp[index]
                    child.lastFrame += durations[index]
                    kept.append(child)
                }

                Self.mergeDuplicates(&kept)
                if let nextScore = current.map(\.score).max() {
                    let moreProbable = kept.filter { $0.score > nextScore }
                    if moreProbable.count >= beam {
                        kept = moreProbable
                        break
                    }
                } else {
                    kept.sort { $0.score > $1.score }
                    if kept.count > beam { kept.removeLast(kept.count - beam) }
                }
            }
        }

        func rank(_ h: BeamState) -> Float { normalize ? h.score / Float(h.tokens.count + 1) : h.score }
        kept.sort { rank($0) > rank($1) }
        return kept.prefix(nbest).map {
            Hypothesis(text: text($0.tokens), tokenIDs: $0.tokens, score: $0.score,
                       normalizedScore: $0.score / Float($0.tokens.count + 1))
        }
    }

    private func diagnose(_ name: String, _ array: MLMultiArray) {
        guard let diagnostics else { return }
        let shape = array.shape.map(\.intValue)
        let values = array.withUnsafeBufferPointer(ofType: Float.self) { buffer in
            let strides = array.strides.map(\.intValue)
            return (0..<array.count).map { index -> Float in
                var remainder = index, offset = 0
                for axis in shape.indices.reversed() {
                    offset += (remainder % shape[axis]) * strides[axis]
                    remainder /= shape[axis]
                }
                return buffer[offset]
            }
        }
        diagnostics(name, values)
    }

    /// Word timestamps and confidences for a result from this model.
    public func words(_ result: Result) -> [ParakeetWord] {
        ParakeetWords.group(result, pieces: pieces, punctuation: wordPunctuation, refinePunctuation: !contract.isRNNT)
    }

    /// The tokenizer piece for an id, or "" when out of range.
    public func piece(_ id: Int) -> String { pieces.indices.contains(id) ? pieces[id] : "" }

    private func text(_ ids: [Int]) -> String { Self.detokenize(ids, pieces: pieces, punctuation: config.punctuation) }
    static func detokenize(_ ids: [Int], pieces: [String], punctuation supported: [String]) -> String {
        // This pinned SentencePiece model has no control/byte tokens. Unknown ID 0
        // has the literal surface " ⁇ "; remove only the initial dummy metaspace.
        var value = ids.map { $0 == 0 ? " ⁇ " : pieces[$0] }.joined().replacingOccurrences(of: "▁", with: " ")
        if let first = ids.first, first != 0, pieces[first].hasPrefix("▁"), value.hasPrefix(" ") { value.removeFirst() }
        // Mirror NeMo's single-whitespace removal before supported punctuation.
        let punctuation = supported.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|")
        if !punctuation.isEmpty { value = value.replacingOccurrences(of: "(\\s)(" + punctuation + ")", with: "$2", options: .regularExpression) }
        return value
    }
    private final class TensorStorage {
        let address: UnsafeMutableRawPointer
        init(bytes: Int, alignment: Int) throws {
            var pointer: UnsafeMutableRawPointer?
            let status = posix_memalign(&pointer, alignment, bytes)
            guard status == 0, let pointer else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(status)) }
            address = pointer
            memset(address, 0, bytes)
        }
        deinit { free(address) }
    }

    /// Stable CPU-owned storage avoids pixel-buffer locks and satisfies CoreML's page-alignment guidance.
    static func alignedOutput(_ shape: [Int], _ type: MLMultiArrayDataType = .float32) throws -> MLMultiArray {
        guard !shape.isEmpty, type == .float32 || type == .int32 else {
            throw ParakeetError.badConfig("Expected positive dimensions and a 32-bit tensor dtype")
        }
        var bytes = 4
        for dimension in shape {
            guard dimension > 0, dimension <= Int.max / bytes else { throw ParakeetError.badConfig("Tensor shape exceeds addressable memory") }
            bytes *= dimension
        }
        let page = Int(getpagesize())
        guard bytes <= Int.max - (page - 1) else { throw ParakeetError.badConfig("Tensor allocation exceeds addressable memory") }
        let storage = try TensorStorage(bytes: (bytes + page - 1) / page * page, alignment: page)
        var strides = [Int](repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) { strides[i] = strides[i + 1] * shape[i + 1] }
        return try MLMultiArray(dataPointer: storage.address, shape: shape.map(NSNumber.init), dataType: type,
                                strides: strides.map(NSNumber.init), deallocator: { _ in withExtendedLifetime(storage) {} })
    }

    private static func array(_ shape: [Int], _ type: MLMultiArrayDataType = .float32) throws -> MLMultiArray {
        try alignedOutput(shape, type)
    }
    private static func zero(_ a: MLMultiArray) { a.dataPointer.assumingMemoryBound(to: Float.self).update(repeating: 0, count: a.count) }
    private static func provider(_ arrays: [String: MLMultiArray]) throws -> MLDictionaryFeatureProvider {
        try MLDictionaryFeatureProvider(dictionary: arrays.mapValues { MLFeatureValue(multiArray: $0) })
    }
    private static func output(_ provider: MLFeatureProvider, _ key: String, _ shape: [Int]) throws -> MLMultiArray {
        guard let a = provider.featureValue(for: key)?.multiArrayValue, a.shape.map(\.intValue) == shape else { throw ParakeetError.badConfig("Incorrect v3 output: \(key)") }
        return a
    }
    private static func scalarLength(_ value: MLMultiArray) throws -> Int {
        guard value.dataType == .int32, value.count == 1 else { throw ParakeetError.badConfig("Expected an int32 length") }
        return value.withUnsafeBufferPointer(ofType: Int32.self) { Int($0[0]) }
    }
    static func copy(_ source: MLMultiArray, into target: MLMultiArray) throws {
        guard source.dataType == .float32, target.dataType == .float32, source.shape == target.shape else { throw ParakeetError.badConfig("Incorrect predictor state layout") }
        let shape = source.shape.map(\.intValue)
        try source.withUnsafeBufferPointer(ofType: Float.self) { inputBuffer in
            let sourceStrides = source.strides.map(\.intValue)
            try target.withUnsafeMutableBufferPointer(ofType: Float.self) { outputBuffer, targetStrides in
                let input = inputBuffer.baseAddress!, output = outputBuffer.baseAddress!
                if shape.count == 3 {
                    for a in 0..<shape[0] { for b in 0..<shape[1] {
                        let from = a * sourceStrides[0] + b * sourceStrides[1]
                        let to = a * targetStrides[0] + b * targetStrides[1]
                        for c in 0..<shape[2] {
                            let value = input[from + c * sourceStrides[2]]
                            guard value.isFinite else { throw ParakeetError.badConfig("Nonfinite tensor output") }
                            output[to + c * targetStrides[2]] = value
                        }
                    } }
                } else {
                    for index in 0..<source.count {
                        var remainder = index, from = 0, to = 0
                        for axis in shape.indices.reversed() {
                            let coordinate = remainder % shape[axis]
                            from += coordinate * sourceStrides[axis]; to += coordinate * targetStrides[axis]
                            remainder /= shape[axis]
                        }
                        let value = input[from]
                        guard value.isFinite else { throw ParakeetError.badConfig("Nonfinite tensor output") }
                        output[to] = value
                    }
                }
            }
        }
    }
    static func argmax(_ a: MLMultiArray, _ range: Range<Int>) throws -> Int {
        guard a.dataType == .float32, a.shape.count == 4, a.shape.prefix(3).allSatisfy({ $0.intValue == 1 }), !range.isEmpty, range.lowerBound >= 0, range.upperBound <= a.count else { throw ParakeetError.badConfig("Expected float32 joint logits") }
        return try a.withUnsafeBufferPointer(ofType: Float.self) { buffer in
            let pointer = buffer.baseAddress!, stride = a.strides[3].intValue
            var best = range.lowerBound, maximum = -Float.infinity
            for index in range {
                let value = pointer[index * stride]
                guard value.isFinite else { throw ParakeetError.badConfig("Nonfinite joint logits") }
                if value > maximum { maximum = value; best = index }
            }
            return best
        }
    }

    /// NeMo `max_prob` confidence (α = 1) of `token` over the first `count` logits (tokens and blank,
    /// not durations): (N·p − 1)/(N − 1). Same arithmetic as parakeet.cpp's `decode_max_prob_conf`.
    static func maxProbConfidence(_ a: MLMultiArray, token: Int, count: Int) throws -> Float {
        guard a.dataType == .float32, count > 1, token >= 0, token < count, count <= a.count else {
            throw ParakeetError.badConfig("Expected float32 joint logits")
        }
        return a.withUnsafeBufferPointer(ofType: Float.self) { buffer in
            let pointer = buffer.baseAddress!, stride = a.strides[3].intValue
            var maximum = -Float.infinity
            for index in 0..<count { maximum = max(maximum, pointer[index * stride]) }
            var denominator = 0.0
            for index in 0..<count { denominator += exp(Double(pointer[index * stride] - maximum)) }
            let p = exp(Double(pointer[token * stride] - maximum)) / denominator
            let n = Double(count)
            return Float((n * p - 1) / (n - 1))
        }
    }

    private static func validate(_ model: MLModel, inputs: [String: [Int]], outputs: [String: [Int]]) throws {
        for (features, shapes) in [(model.modelDescription.inputDescriptionsByName,inputs),(model.modelDescription.outputDescriptionsByName,outputs)] {
            for (name,shape) in shapes {
                guard let constraint = features[name]?.multiArrayConstraint, constraint.shape.map(\.intValue) == shape,
                      constraint.dataType == (name.hasSuffix("length") || name == "targets" ? .int32 : .float32) else {
                    throw ParakeetError.badConfig("Unexpected v3 tensor contract: \(name)")
                }
            }
        }
    }
}
