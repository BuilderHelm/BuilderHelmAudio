import CoreML
import CryptoKit
import Foundation
import Darwin

/// Offline v3 runtime with bounded windows. Serialize calls on each instance.
/// Transcript text is supported; emission frames are internal window-ownership markers, not aligned word times.
public final class ParakeetTdtV3 {
    public struct Result {
        public let text: String
        public let tokenIDs: [Int]
        public let emissionFrames: [Int]
    }
    struct Window: Decodable {
        let samples: Int
        let function: String?
        var melFrames: Int { samples / 160 + 1 }
        var encoderFrames: Int { (melFrames + 7) / 8 }
    }
    struct Config: Decodable {
        let architecture: String
        let precision: String
        let sample_rate: Int
        let window_samples: Int
        let blank_id: Int
        let vocabulary_size: Int
        let durations: [Int]
        let max_symbols: Int
        let punctuation: [String]
        let files: [String: String]
        let windows: [Window]?
        let encoder_stages: [String]?
    }
    private let allowExperimentalLongAudio: Bool
    private let config: Config
    private let pieces: [String]
    private let predictor: MLModel, joint: MLModel
    private let frontendURL: URL, encoderURL: URL
    private let encoderTailURL: URL?
    private let encoderComputeUnits: MLComputeUnits
    private var requestWindowSamples = 240000
    private var activeWindow: WindowModels?
    // Internal integration-test probe; absent during normal inference.
    var diagnostics: ((String, [Float]) -> Void)?
    private final class WindowModels {
        let window: Window
        let frontend: MLModel, encoder: MLModel
        let encoderTail: MLModel?
        let audio: MLMultiArray, audioLength: MLMultiArray
        let mel: MLMultiArray, melLength: MLMultiArray, featuresSnapshot: MLMultiArray
        let encoderOptions: MLPredictionOptions?
        init(window: Window, frontend: MLModel, encoder: MLModel, encoderTail: MLModel? = nil) throws {
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
            try ParakeetTdtV3.validate(frontend, inputs: ["audio_signal":[1,window.samples],"audio_length":[1]], outputs:["mel":[1,128,window.melFrames],"mel_length":[1]])
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

    public init(directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine, cacheDirectory: URL,
                allowExperimentalLongAudio: Bool = true, decoderComputeUnits: MLComputeUnits? = nil) throws {
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
        guard config.architecture == "parakeet-tdt-0.6b-v3", ["fp32", "fp16"].contains(config.precision),
              config.sample_rate == 16000, config.window_samples == 240000, config.blank_id == 8192,
              config.vocabulary_size == 8192, config.durations == [0, 1, 2, 3, 4], config.max_symbols == 10
        else { throw ParakeetError.badConfig("Unsupported v3 tensor/decoder contract") }
        if let windows = config.windows {
            guard windows.map(\.samples) == [240000,480000], windows.map(\.function) == ["w15","w30"] else {
                throw ParakeetError.badConfig("Expected ordered w15/w30 window functions")
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
        for (path, expected) in config.files {
            guard !path.hasPrefix("/"), !path.split(separator: "/").contains(".."), expected.count == 64 else {
                throw ParakeetError.badConfig("Invalid model manifest path or checksum")
            }
            let url = directory.appendingPathComponent(path)
            let file = try FileHandle(forReadingFrom: url)
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
        pieces = try decoder.decode([String].self, from: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
        guard pieces.count == 8192, pieces.first == "<unk>" else { throw ParakeetError.badConfig("Expected 8192 tokenizer pieces") }
        func load(_ name: String, _ units: MLComputeUnits) throws -> MLModel {
            try autoreleasepool {
                let url = try ParakeetTdtCtc110M.modelURL(directory, name, cache: cacheDirectory, allowBundledCompiled: false)
                return try ParakeetTdtCtc110M.load(url, units, nil)
            }
        }
        let hasTail = config.encoder_stages != nil
        let compiled = try autoreleasepool { () throws -> (URL,URL,URL?) in
            let front = try ParakeetTdtCtc110M.modelURL(directory,"Preprocessor",cache:cacheDirectory,allowBundledCompiled:false)
            let enc = try ParakeetTdtCtc110M.modelURL(directory,"Encoder",cache:cacheDirectory,allowBundledCompiled:false)
            let tail = hasTail ? try ParakeetTdtCtc110M.modelURL(directory,"EncoderTail",cache:cacheDirectory,allowBundledCompiled:false) : nil
            return (front,enc,tail)
        }
        frontendURL = compiled.0; encoderURL = compiled.1; encoderTailURL = compiled.2
        predictor = try load("Predictor", decoderComputeUnits ?? computeUnits)
        joint = try load("Joint", decoderComputeUnits ?? computeUnits)
        targets = try Self.array([1, 1], .int32)
        hidden = try Self.array([2, 1, 640]); cell = try Self.array([2, 1, 640])
        candidateHidden = try Self.array([2, 1, 640]); candidateCell = try Self.array([2, 1, 640])
        decoderStep = try Self.array([1, 640, 1]); encoderStep = try Self.array([1, 1024, 1])
        _ = try windowModels()
        try Self.validate(predictor, inputs: ["targets": [1,1], "h_in": [2,1,640], "c_in": [2,1,640]], outputs: ["decoder": [1,640,1], "h_out": [2,1,640], "c_out": [2,1,640]])
        try Self.validate(joint, inputs: ["encoder_step": [1,1024,1], "decoder_step": [1,640,1]], outputs: ["logits": [1,1,1,8198]])
    }

    private func windowSize(for count: Int) -> Int {
        count > 240000 && config.windows != nil ? 480000 : 240000
    }
    private func windowModels() throws -> WindowModels {
        if activeWindow?.window.samples == requestWindowSamples { return activeWindow! }
        // Retain one encoder function at a time to bound device memory.
        autoreleasepool { activeWindow = nil }
        let window = config.windows?.first(where: { $0.samples == requestWindowSamples })
            ?? Window(samples: 240000, function: nil)
        let current = try autoreleasepool { () throws -> WindowModels in
            func load(_ name: String, _ units: MLComputeUnits) throws -> MLModel {
                let url: URL
                if name == "Preprocessor" { url = frontendURL }
                else if name == "EncoderTail", let tail = encoderTailURL { url = tail }
                else { url = encoderURL }
                return try ParakeetTdtCtc110M.load(url, units, window.function)
            }
            return try WindowModels(window:window,frontend:load("Preprocessor",.cpuOnly),encoder:load("Encoder",encoderComputeUnits),encoderTail:encoderTailURL == nil ? nil : load("EncoderTail",encoderComputeUnits))
        }
        activeWindow = current
        return current
    }

    /// Bounded offline segmentation. Prefer quiet pauses; retain overlap context when no pause is available.
    public func transcribe(file: URL, checkCancellation: () throws -> Void = {}) throws -> Result {
        let reader = try ParakeetAudio.Reader(file)
        var buffer = try reader.read(config.windows == nil ? 240001 : 480001), start = 0
        requestWindowSamples = windowSize(for: buffer.count)
        var allIDs: [Int] = [], allFrames: [Int] = []
        var pending: PendingWindow?
        try Self.validateSampleCount(buffer.count, allowLongAudio: allowExperimentalLongAudio)
        // Keep short file and PCM requests on exactly the same decode path.
        if buffer.count <= requestWindowSamples {
            return try autoreleasepool {
                try requestWindowSamples == 480000
                    ? decodeLongWindow(buffer, checkCancellation: checkCancellation)
                    : decode(buffer, checkCancellation: checkCancellation)
            }
        }
        while !buffer.isEmpty {
            try checkCancellation()
            let plan = Self.planWindow(buffer, windowSamples: requestWindowSamples)
            let window = try autoreleasepool { try decodeLongWindow(Array(buffer.prefix(plan.count)), checkCancellation: checkCancellation) }
            Self.consume(window, start: start, seam: start + plan.right, pending: &pending, ids: &allIDs, frames: &allFrames)
            if buffer.count <= requestWindowSamples { break }
            buffer.removeFirst(plan.advance)
            buffer.append(contentsOf: try reader.read(plan.advance))
            start += plan.advance
        }
        if let last = pending { Self.flush(last, through: last.result.tokenIDs.count, ids: &allIDs, frames: &allFrames) }
        return Result(text: text(allIDs), tokenIDs: allIDs, emissionFrames: allFrames)
    }

    public func transcribe(_ samples: [Float], checkCancellation: () throws -> Void = {}) throws -> Result {
        guard samples.allSatisfy(\.isFinite) else { throw ParakeetError.invalidAudio("PCM samples must be finite") }
        try Self.validateSampleCount(samples.count, allowLongAudio: allowExperimentalLongAudio)
        requestWindowSamples = windowSize(for: samples.count)
        if samples.count <= requestWindowSamples {
            return try autoreleasepool {
                try requestWindowSamples == 480000
                    ? decodeLongWindow(samples, checkCancellation: checkCancellation)
                    : decode(samples, checkCancellation: checkCancellation)
            }
        }
        var start = 0, allIDs: [Int] = [], allFrames: [Int] = []
        var pending: PendingWindow?
        while start < samples.count {
            try checkCancellation()
            let buffer = Array(samples[start..<min(start + requestWindowSamples + 1, samples.count)])
            let plan = Self.planWindow(buffer, windowSamples: requestWindowSamples)
            let window = try autoreleasepool { try decodeLongWindow(Array(buffer.prefix(plan.count)), checkCancellation: checkCancellation) }
            Self.consume(window, start: start, seam: start + plan.right, pending: &pending, ids: &allIDs, frames: &allFrames)
            if buffer.count <= requestWindowSamples { break }
            start += plan.advance
        }
        if let last = pending { Self.flush(last, through: last.result.tokenIDs.count, ids: &allIDs, frames: &allFrames) }
        return Result(text: text(allIDs), tokenIDs: allIDs, emissionFrames: allFrames)
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
    private static func flush(_ window: PendingWindow, through end: Int, ids: inout [Int], frames: inout [Int]) {
        for index in window.first..<max(window.first, end) {
            ids.append(window.result.tokenIDs[index])
            frames.append(window.start/1280 + window.result.emissionFrames[index])
        }
    }
    private static func consume(_ result: Result, start: Int, seam: Int, pending: inout PendingWindow?,
                                ids: inout [Int], frames: inout [Int]) {
        var first = 0
        if let previous = pending {
            let cut = join(previous.result, previousStart:previous.start, previousFirst:previous.first,
                           next:result, nextStart:start, seam:previous.seam)
            flush(previous, through:cut.previousEnd, ids:&ids, frames:&frames)
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
        if retryDepth == 0, requestWindowSamples == 480000, !first.tokenIDs.isEmpty {
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
        let ids = left.tokenIDs + right.tokenIDs
        return Result(text: text(ids), tokenIDs: ids, emissionFrames: left.emissionFrames + right.emissionFrames.map { $0 + cut / 1280 })
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
    static func insertRecoveredTokens(_ primary: Result, candidate: Result, gaps: [BlankSpan]) -> (ids: [Int], frames: [Int])? {
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
        var ids: [Int] = [], times: [Int] = []
        for i in 0...primary.tokenIDs.count {
            if let added = insertions[i] { ids += candidateIDs[added]; times += candidateFrames[added] }
            if i < primary.tokenIDs.count { ids.append(primary.tokenIDs[i]); times.append(primary.emissionFrames[i]) }
        }
        return (ids, times)
    }

    private func recoverPartialBlanks(_ primary: Result, samples: [Float], checkCancellation: () throws -> Void) throws -> Result {
        let gaps = Self.blankSpans(primary, samples: samples)
        guard !gaps.isEmpty else { return primary }
        let originalSize = requestWindowSamples
        requestWindowSamples = 240000
        defer { requestWindowSamples = originalSize }
        var start = 0, candidateIDs: [Int] = [], candidateFrames: [Int] = []
        var pending: PendingWindow?
        while start < samples.count {
            try checkCancellation()
            let buffer = Array(samples[start..<min(start + 240001, samples.count)])
            let plan = Self.planWindow(buffer, windowSamples: 240000)
            let candidate = try autoreleasepool { try decode(Array(buffer.prefix(plan.count)), checkCancellation: checkCancellation) }
            Self.consume(candidate, start: start, seam: start + plan.right, pending: &pending, ids: &candidateIDs, frames: &candidateFrames)
            if buffer.count <= 240000 { break }
            start += plan.advance
        }
        if let last = pending { Self.flush(last, through: last.result.tokenIDs.count, ids: &candidateIDs, frames: &candidateFrames) }
        let candidate = Result(text: "", tokenIDs: candidateIDs, emissionFrames: candidateFrames)
        guard let recovered = Self.insertRecoveredTokens(primary, candidate: candidate, gaps: gaps) else { return primary }
        return Result(text: text(recovered.ids), tokenIDs: recovered.ids, emissionFrames: recovered.frames)
    }

    private func decode(_ samples: [Float], checkCancellation: () throws -> Void) throws -> Result {
        try checkCancellation()
        guard samples.count <= requestWindowSamples, samples.allSatisfy(\.isFinite) else { throw ParakeetError.invalidAudio("Invalid v3 window") }
        diagnostics?("pcm", samples)
        // Fewer than one source mel hop has no valid encoder frames.
        if samples.count < 160 { return Result(text: "", tokenIDs: [], emissionFrames: []) }
        let models = try windowModels()
        let audio = models.audio, audioLength = models.audioLength
        Self.zero(audio); Self.zero(hidden); Self.zero(cell)
        samples.withUnsafeBufferPointer { p in audio.dataPointer.assumingMemoryBound(to: Float.self).update(from: p.baseAddress!, count: samples.count) }
        audioLength[0] = NSNumber(value: samples.count)
        let frames = try autoreleasepool { () throws -> Int in
            let melOutput = try models.frontend.prediction(from: Self.provider(["audio_signal": audio, "audio_length": audioLength]))
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
        let features = models.featuresSnapshot
        diagnose("encoder", features)
        var time = 0, symbols = 0, lastToken = 8192, needsPrediction = true
        var ids: [Int] = [], times: [Int] = []
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
            let (token, duration) = try autoreleasepool {
                let input = try Self.provider(["encoder_step": encoderStep, "decoder_step": decoderStep])
                let output = try joint.prediction(from: input)
                return try withExtendedLifetime((input, output)) {
                    let logits = try Self.output(output, "logits", [1,1,1,8198])
                    diagnose("logits", logits)
                    return (try Self.argmax(logits, 0..<8193), try Self.argmax(logits, 8193..<8198) - 8193)
                }
            }
            diagnostics?("decision", [Float(time), Float(token), Float(duration)])
            if token != 8192 {
                ids.append(token); times.append(time); lastToken = token
                try Self.copy(candidateHidden, into: hidden); try Self.copy(candidateCell, into: cell)
                needsPrediction = true
            }
            time += duration; symbols += 1
            if duration > 0 || symbols == config.max_symbols {
                if symbols == config.max_symbols { time += 1 }
                symbols = 0
            }
        }
        return Result(text: text(ids), tokenIDs: ids, emissionFrames: times)
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
