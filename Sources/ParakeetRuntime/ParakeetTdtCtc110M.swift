// Reference Swift runtime for the Parakeet TDT-CTC 110M Core ML models.
//
//   let model = try ParakeetTdtCtc110M(directory: modelFolder)
//   let result = try model.transcribe(samples)   // 16 kHz mono Float PCM
//   print(result.text)                           // TDT transcript
//   print(result.ctcText)                        // greedy CTC transcript
//   result.ctcLogProbabilities?.values           // optional flat frames x 1025, for keyword spotting
//
// One encoder call (Neural Engine) serves both decoders. Each window uses the
// smallest encoder function for short recordings. Adaptive long audio reuses
// the 20 s function, including its partial tail, and overlaps by 6 s. Windows
// are stitched at a frame both CTC heads call blank; CTC-only joins also require the
// stitched tokens near the cut to match one window's own decoding.
import Accelerate
import CoreML
import CryptoKit
import Foundation

public struct ParakeetCtcLogProbabilities: Sendable {
    public static let classCount = 1025
    /// Contiguous, frame-major storage. Index = frame * 1025 + token.
    public let values: [Float]
    public var frameCount: Int { values.count / Self.classCount }
    public subscript(frame: Int) -> ArraySlice<Float> {
        values[frame * Self.classCount..<(frame + 1) * Self.classCount]
    }
}

public struct ParakeetResult: Sendable {
    /// TDT transcript, or the CTC transcript when initialized with ctcOnly: true.
    public let text: String
    /// TDT token ids and the time (seconds) each was emitted.
    public let tokens: [Int]
    public let tokenTimes: [Double]
    /// Greedy CTC transcript.
    public let ctcText: String
    /// Number of valid CTC frames, also available when probabilities are omitted.
    public let ctcFrameCount: Int
    /// Optional flat CTC storage. Blank token = 1024; each frame spans 80 ms.
    public let ctcLogProbabilities: ParakeetCtcLogProbabilities?
    public var ctcLogProbsFlat: [Float]? { ctcLogProbabilities?.values }
    /// Compatibility row view. Materializes arrays when accessed; prefer flat storage.
    public var ctcLogProbs: [[Float]] {
        guard let probabilities = ctcLogProbabilities else { return [] }
        return (0..<probabilities.frameCount).map { Array(probabilities[$0]) }
    }
    init(text: String, tokens: [Int], tokenTimes: [Double], ctcText: String,
         ctcFrameCount: Int, ctcValues: [Float]?) {
        self.text = text; self.tokens = tokens; self.tokenTimes = tokenTimes; self.ctcText = ctcText
        self.ctcFrameCount = ctcFrameCount
        ctcLogProbabilities = ctcValues.map { ParakeetCtcLogProbabilities(values: $0) }
    }
    // Compatibility initializer for the historical standalone evaluation harnesses.
    init(text: String, tokens: [Int], tokenTimes: [Double], ctcText: String, ctcLogProbs: [[Float]]) {
        self.init(text: text, tokens: tokens, tokenTimes: tokenTimes, ctcText: ctcText,
                  ctcFrameCount: ctcLogProbs.count, ctcValues: ctcLogProbs.flatMap { $0 })
    }
}

public enum ParakeetError: Error {
    case missingModel(String)
    case badConfig(String)
    case invalidAudio(String)
}

/// Instances reuse inference buffers. Serialize calls on each instance.
public final class ParakeetTdtCtc110M {
    struct Config: Decodable {
        struct Window: Decodable { let function: String; let seconds: Double; let samples: Int }
        struct LongAudio: Decodable { let window_function: String; let overlap_seconds: Double; let adaptive_long_audio: Bool? }
        struct Tensor: Decodable, Equatable { let name: String; let shape: [Int] }
        struct DecoderFile: Decodable { let file: String; let sha256: String?; let tensors: [Tensor] }
        let blank_id: Int
        let sample_rate: Int
        let tdt_durations: [Int]
        let tdt_max_symbols_per_frame: Int
        let encoder_frame_samples: Int
        let encoder_frame_seconds: Double
        let windows: [Window]
        let long_audio: LongAudio
        /// Swift decoder weights; absent in v1.0.0 folders, which decode with DecoderJoint/JointBlock.
        let tdt_decoder: DecoderFile?
    }

    final class Rung {
        let samples: Int
        private let function: String, preprocessorURL: URL, encoderURL: URL, units: MLComputeUnits
        private struct Models { let preprocessor: MLModel; let encoder: MLModel; let audio: MLMultiArray }
        private var models: Models?
        var lastUse: UInt64 = 0
        var isLoaded: Bool { models != nil }
        var preprocessor: MLModel { models!.preprocessor }
        var encoder: MLModel { models!.encoder }
        var audio: MLMultiArray { models!.audio }
        init(_ window: Config.Window, preprocessorURL: URL, encoderURL: URL, units: MLComputeUnits) {
            samples = window.samples
            function = window.function; self.preprocessorURL = preprocessorURL; self.encoderURL = encoderURL; self.units = units
        }
        func load() throws {
            guard models == nil else { return }
            models = try Models(preprocessor: ParakeetTdtCtc110M.load(preprocessorURL, .cpuOnly, function),
                                encoder: ParakeetTdtCtc110M.load(encoderURL, units, function),
                                audio: MLMultiArray(shape: [1, NSNumber(value: samples)], dataType: .float32))
        }
        func unload() { models = nil }
    }

    struct Window {
        let firstFrame: Int
        let frames: Int
        let values: [Float]?
        let blankLogProbs: [Float]
        let ctcTokens: [Int]
        let projection: MLMultiArray?     // [1, T, 640]; absent in dedicated CTC exports
        var logProbs: [[Float]] {
            guard let values else { return [] }
            return (0..<frames).map { Array(values[$0 * 1025..<($0 + 1) * 1025]) }
        }
        init(firstFrame: Int, frames: Int, values: [Float], retainProbabilities: Bool = true, projection: MLMultiArray?) {
            self.firstFrame = firstFrame; self.frames = frames; self.projection = projection
            self.values = retainProbabilities ? values : nil
            var blank: [Float] = [], tokens: [Int] = []
            blank.reserveCapacity(frames); tokens.reserveCapacity(frames)
            values.withUnsafeBufferPointer { data in
                for frame in 0..<frames {
                    let offset = frame * 1025
                    var best: vDSP_Length = 0
                    var maximum: Float = 0
                    vDSP_maxvi(data.baseAddress! + offset, 1, &maximum, &best, 1025)
                    tokens.append(Int(best)); blank.append(data[offset + 1024])
                }
            }
            blankLogProbs = blank; ctcTokens = tokens
        }
        init(firstFrame: Int, frames: Int, logProbs: [[Float]], projection: MLMultiArray?) {
            self.init(firstFrame: firstFrame, frames: frames, values: logProbs.flatMap { $0 }, projection: projection)
        }
    }

    let config: Config
    let rungs: [Rung]
    /// The config's long_audio window; exports may add larger functions for CTC-only policies.
    private var longRungIndex = 0
    var longRung: Rung { rungs[longRungIndex] }
    private let loadCacheDirectory: URL
    public let maximumLoadedWindows: Int
    private var rungUseClock: UInt64 = 0
    /// Loaded encoder/preprocessor window sizes; useful for tuning a dictation preload.
    public var loadedWindowSeconds: [Double] { rungs.filter(\.isLoaded).map { Double($0.samples) / 16000 } }
    let decoder: TdtDecoder?
    let decoderJoint: MLModel?
    let jointBlock: MLModel?
    /// Explicit state-carry request. Adaptive mode can also enable carry on long recordings.
    public let carryTdtStateAcrossWindows: Bool
    /// At 60 seconds and above, use at least six seconds of overlap and carry TDT state.
    public let adaptiveLongAudio: Bool
    private var usesCarriedTdtState: Bool
    private var nextTdtFrame: Int?
    /// Experimental CTC-only long-recording policy. Nil retains the shipped policy.
    private let ctcLongWindowSamples: Int?
    private let ctcLongOverlapSeconds: Double?
    /// CTC-only: encode a partial final long-audio window with the smallest function that fits.
    private let ctcFitLastWindow: Bool
    /// CTC-only: prepare window i+1 and finish window i-1 on the CPU while the Neural Engine encodes window i.
    private let ctcPipeline: Bool
    private let ctcConsistentJoin: Bool
    var decodesTdt: Bool { decoder != nil || decoderJoint != nil }
    let pieces: [String]
    let block: Int
    let hidden = 640
    let blocks: MLMultiArray              // [1, block, 640]
    let token: MLMultiArray
    let length: MLMultiArray

    /// - Parameters:
    ///   - directory: folder with Preprocessor/Encoder bundles, config.json, vocab.json,
    ///     and TdtDecoder.bin (or the legacy DecoderJoint/JointBlock bundles).
    ///   - computeUnits: where the encoder runs (default: Neural Engine).
    ///   - ctcOnly: load only Preprocessor and Encoder; skip TDT decoding.
    ///     Otherwise TDT runs on the CPU from TdtDecoder.bin when config.json lists it,
    ///     else with the DecoderJoint and JointBlock models.
    ///   - cacheDirectory: where .mlpackage bundles are compiled once and kept
    ///     (default: `defaultCacheDirectory`). Unused for .mlmodelc bundles.
    ///   - carryTdtStateAcrossWindows: preserve the Swift decoder state and duration
    ///     across owned window boundaries. Changes long-audio transcripts; requires
    ///     TdtDecoder.bin. Measured on Long20 only, so opt in explicitly.
    ///   - adaptiveLongAudio: at 60 seconds and above, carry TDT state and use at least
    ///     six seconds of overlap. Below 60 seconds, retain configured overlap and state
    ///     policy. Defaults to config.json long_audio.adaptive_long_audio, or false.
    ///   - preferredShortWindowSeconds: short window to preload alongside the largest (default 4 s); nil loads only the largest.
    ///   - maximumLoadedWindows: bound on resident model windows (default 4). Only two preload;
    ///     other sizes stay cached after first use. Smaller limits trade mixed-duration latency for retention.
    ///   - ctcLongWindowSeconds, ctcLongOverlapSeconds: CTC-only long-audio window (an exported
    ///     function) and overlap. Default: the configured long window and policy.
    ///   - ctcFitLastWindow: CTC-only; encode the final long-audio window with the smallest
    ///     function that fits instead of the largest. Transcripts are unchanged.
    ///   - ctcPipeline: CTC-only; overlap mel and decoding on the CPU with the encoder on the
    ///     Neural Engine for long audio. Transcripts are unchanged.
    ///   - ctcConsistentJoin: CTC-only; join long-audio windows where the stitched tokens match what
    ///     one window decoded alone, so words at a join are not repeated.
    public init(directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine, ctcOnly: Bool = false,
                cacheDirectory: URL? = nil, carryTdtStateAcrossWindows: Bool = false, adaptiveLongAudio: Bool? = nil,
                preferredShortWindowSeconds: Double? = 4, maximumLoadedWindows: Int = 4,
                ctcLongWindowSeconds: Double? = nil, ctcLongOverlapSeconds: Double? = nil,
                ctcFitLastWindow: Bool = true, ctcPipeline: Bool = true, ctcConsistentJoin: Bool = true) throws {
        self.ctcConsistentJoin = ctcConsistentJoin
        self.ctcFitLastWindow = ctcFitLastWindow
        self.ctcPipeline = ctcPipeline
        guard maximumLoadedWindows > 0, preferredShortWindowSeconds.map({ $0.isFinite && $0 > 0 }) ?? true
        else { throw ParakeetError.badConfig("Invalid window loading policy") }
        self.maximumLoadedWindows = maximumLoadedWindows
        loadCacheDirectory = cacheDirectory ?? Self.defaultCacheDirectory
        self.carryTdtStateAcrossWindows = carryTdtStateAcrossWindows
        usesCarriedTdtState = carryTdtStateAcrossWindows
        config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        self.adaptiveLongAudio = adaptiveLongAudio ?? config.long_audio.adaptive_long_audio ?? false
        pieces = try JSONDecoder().decode([String].self, from: Data(contentsOf: directory.appendingPathComponent("vocab.json")))
        let sorted = config.windows.sorted { $0.samples < $1.samples }
        if let seconds = ctcLongWindowSeconds {
            guard ctcOnly, let window = sorted.first(where: { $0.seconds == seconds })
            else { throw ParakeetError.badConfig("CTC long window must name an exported function in CTC-only mode") }
            ctcLongWindowSamples = window.samples
        } else { ctcLongWindowSamples = nil }
        if let overlap = ctcLongOverlapSeconds {
            guard ctcOnly, overlap.isFinite, overlap >= 0.08,
                  overlap * 16000 < Double((ctcLongWindowSamples ?? sorted.last?.samples ?? 0) - 1280)
            else { throw ParakeetError.badConfig("Invalid CTC long overlap") }
        }
        self.ctcLongOverlapSeconds = ctcLongOverlapSeconds
        guard config.sample_rate == 16000, config.blank_id == 1024, pieces.count == 1024,
              config.encoder_frame_samples == 1280, config.encoder_frame_seconds == 0.08,
              config.tdt_durations == [0, 1, 2, 3, 4], config.tdt_max_symbols_per_frame > 0,
              !sorted.isEmpty, sorted.allSatisfy({ $0.samples >= 320 && !$0.function.isEmpty }),
              Set(sorted.map(\.samples)).count == sorted.count,
              Set(sorted.map(\.function)).count == sorted.count,
              let largest = sorted.first(where: { [longFunction = config.long_audio.window_function] in $0.function == longFunction }),
              largest.samples % config.encoder_frame_samples == 0,
              config.long_audio.overlap_seconds.isFinite, config.long_audio.overlap_seconds >= 0.08,
              config.long_audio.overlap_seconds * 16000 < Double(largest.samples - 1280)
        else { throw ParakeetError.badConfig("Incompatible architecture, vocabulary, windows or overlap") }
        guard !self.adaptiveLongAudio || largest.samples > 6 * 16000 + 1280
        else { throw ParakeetError.badConfig("Adaptive long audio needs a window longer than six seconds") }
        if let samples = ctcLongWindowSamples {
            guard (ctcLongOverlapSeconds ?? max(6, config.long_audio.overlap_seconds)) * 16000 < Double(samples - 1280)
            else { throw ParakeetError.badConfig("CTC long window must be larger than its overlap") }
        }
        // Compile each portable package once, then load its named window functions.
        // Window functions load concurrently: warm loads take about half as long.
        let cache = loadCacheDirectory
        if !ctcOnly, let file = config.tdt_decoder {
            let outputs = pieces.count + 1 + config.tdt_durations.count
            guard file.tensors == TdtDecoder.layout(tokens: pieces.count + 1, outputs: outputs).map({ .init(name: $0.0, shape: $0.1) }),
                  !file.file.contains("/") else { throw ParakeetError.badConfig("Unexpected tdt_decoder layout") }
            let url = directory.appendingPathComponent(file.file)
            guard FileManager.default.fileExists(atPath: url.path) else { throw ParakeetError.missingModel(file.file) }
            let tokenCount = pieces.count + 1
            decoder = try autoreleasepool {
                try TdtDecoder(url: url, tokens: tokenCount, outputs: outputs, expectedSHA256: file.sha256)
            }
        } else { decoder = nil }
        if (carryTdtStateAcrossWindows || self.adaptiveLongAudio) && !ctcOnly && decoder == nil {
            throw ParakeetError.badConfig("Carrying TDT state requires the Swift decoder weights")
        }
        let coreMLDecoder = !ctcOnly && decoder == nil
        let preloadSamples = ctcLongWindowSamples ?? largest.samples
        (rungs, decoderJoint, jointBlock) = try Self.withLoadLock(cache) {
            try autoreleasepool {
            let preURL = try Self.modelURL(directory, "Preprocessor", cache: cache)
            let encURL = try Self.modelURL(directory, "Encoder", cache: cache)
            let windows = sorted.map { Rung($0, preprocessorURL: preURL, encoderURL: encURL, units: computeUnits) }
            let longest = windows.first { $0.samples == preloadSamples }!
            var preload = [longest]
            if maximumLoadedWindows > 1, let preferredShortWindowSeconds,
               let short = windows.first(where: { Double($0.samples) / 16000 == preferredShortWindowSeconds }), short !== longest {
                preload.append(short)
            }
            for (i, window) in preload.enumerated() { try window.load(); window.lastUse = UInt64(i + 1) }
            return (windows,
                    coreMLDecoder ? try Self.load(Self.modelURL(directory, "DecoderJoint", cache: cache), .cpuOnly, nil) : nil,
                    coreMLDecoder ? try Self.load(Self.modelURL(directory, "JointBlock", cache: cache), .cpuOnly, nil) : nil)
            }
        }
        longRungIndex = rungs.firstIndex { $0.samples == largest.samples }!
        let shape: [NSNumber]
        if let jointBlock {
            guard let input = jointBlock.modelDescription.inputDescriptionsByName["encoder_frames"]?.multiArrayConstraint?.shape,
                  input.count == 3, input[0].intValue == 1, input[1].intValue > 0, input[2].intValue == hidden
            else { throw ParakeetError.badConfig("Invalid JointBlock encoder_frames shape") }
            shape = input
        } else { shape = [1, 16, 640] }
        block = shape[1].intValue
        blocks = try MLMultiArray(shape: shape, dataType: .float32)
        token = try MLMultiArray(shape: [1, 1], dataType: .int32)
        length = try MLMultiArray(shape: [1], dataType: .int32)
        rungUseClock = UInt64(rungs.count)
    }

    private func ensureLoaded(_ rung: Rung, protecting busy: Rung? = nil) throws {
        rungUseClock += 1; rung.lastUse = rungUseClock
        guard !rung.isLoaded else { return }
        while rungs.filter(\.isLoaded).count >= maximumLoadedWindows {
            let candidates = rungs.filter { $0.isLoaded && $0 !== rung && $0 !== busy }
            let short = candidates.filter { $0 !== longRung }
            guard let victim = (short.isEmpty ? candidates : short).min(by: { $0.lastUse < $1.lastUse }) else { break }
            victim.unload()
        }
        try Self.withLoadLock(loadCacheDirectory) { try rung.load() }
    }

    /// Where .mlpackage bundles are compiled: Application Support/ParakeetTdtCtc110M/CompiledModels.
    public static var defaultCacheDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ParakeetTdtCtc110M/CompiledModels", isDirectory: true)
    }

    /// Compiles portable bundles and prepares the default preloaded windows. Other
    /// functions load on first use and can pay a one-time Neural Engine preparation cost.
    public static func prepare(directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                               ctcOnly: Bool = false, cacheDirectory: URL? = nil) throws {
        _ = try ParakeetTdtCtc110M(directory: directory, computeUnits: computeUnits, ctcOnly: ctcOnly,
                                   cacheDirectory: cacheDirectory)
    }

    /// Core ML can fail ("functionName must be nil unless the model type is ML Program")
    /// when two processes first load the same model path at once, so loading holds a
    /// lock file in the cache. Best effort: an unwritable cache loads without it.
    static func withLoadLock<T>(_ cache: URL, _ body: () throws -> T) rethrows -> T {
        try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let fd = open(cache.appendingPathComponent(".load.lock").path, O_CREAT | O_RDWR | O_CLOEXEC, 0o644)
        if fd >= 0 { flock(fd, LOCK_EX) }
        defer { if fd >= 0 { flock(fd, LOCK_UN); close(fd) } }
        return try body()
    }

    /// A .mlmodelc is used in place. A .mlpackage is compiled once into
    /// `cache/<name>-<content hash>-<macOS build>/` and reused from that fixed path:
    /// Core ML caches each Neural Engine function by model path, so a fresh
    /// temporary path re-pays about 5 s per encoder function on every launch.
    static func modelURL(_ directory: URL, _ name: String, cache: URL) throws -> URL {
        let fm = FileManager.default
        let bundled = directory.appendingPathComponent(name + ".mlmodelc")
        if fm.fileExists(atPath: bundled.path) { return bundled }
        let package = directory.appendingPathComponent(name + ".mlpackage")
        guard fm.fileExists(atPath: package.path) else { throw ParakeetError.missingModel(name) }
        let prefix = "\(name)-\(try contentHash(package))-"
        let entry = cache.appendingPathComponent(prefix + osBuild(), isDirectory: true)
        let url = entry.appendingPathComponent(name + ".mlmodelc")
        if fm.fileExists(atPath: url.path) { return url }

        // Compile into a staging folder, then rename it into place in one step, so an
        // entry is either complete or absent. Another process may win the rename.
        try fm.createDirectory(at: cache, withIntermediateDirectories: true)
        let staging = cache.appendingPathComponent(".staging-" + UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? fm.removeItem(at: staging) }
        let compiled = try MLModel.compileModel(at: package)
        defer { try? fm.removeItem(at: compiled) }
        try fm.moveItem(at: compiled, to: staging.appendingPathComponent(name + ".mlmodelc"))
        do { try fm.moveItem(at: staging, to: entry) } catch where fm.fileExists(atPath: url.path) {}
        // The same package compiled under an earlier macOS build is stale.
        for old in (try? fm.contentsOfDirectory(atPath: cache.path)) ?? []
        where old.hasPrefix(prefix) && old != entry.lastPathComponent {
            try? fm.removeItem(at: cache.appendingPathComponent(old))
        }
        return url
    }

    /// SHA-256 over each file's relative path and bytes (about 0.1 s for all four packages on M5).
    static func contentHash(_ package: URL) throws -> String {
        let root = package.resolvingSymlinksInPath()
        guard let walk = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey])
        else { throw ParakeetError.missingModel(package.lastPathComponent) }
        var files: [(String, URL)] = []
        for case let url as URL in walk where try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
            let path = url.resolvingSymlinksInPath().pathComponents.dropFirst(root.pathComponents.count)
            files.append((path.joined(separator: "/"), url))
        }
        var hash = SHA256()
        for (path, url) in files.sorted(by: { $0.0 < $1.0 }) {
            let data = try Data(contentsOf: url, options: .alwaysMapped)
            hash.update(data: Data("\(path)\u{0}\(data.count)\u{0}".utf8))
            hash.update(data: data)
        }
        return hash.finalize().prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    /// macOS build, e.g. "27C5048a"; a new build gets a fresh compile.
    static func osBuild() -> String {
        var size = 0
        guard sysctlbyname("kern.osversion", nil, &size, nil, 0) == 0, size > 1 else { return "unknown" }
        var chars = [CChar](repeating: 0, count: size)
        guard sysctlbyname("kern.osversion", &chars, &size, nil, 0) == 0 else { return "unknown" }
        return String(cString: chars)
    }

    static func load(_ url: URL, _ units: MLComputeUnits, _ function: String?) throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        if let function { configuration.functionName = function }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    // MARK: - Public

    /// Set includeCtcLogProbs to false for transcription without retaining CTC probabilities.
    /// Both transcripts, token times and ctcFrameCount remain available.
    public func transcribe(_ samples: [Float], includeCtcLogProbs: Bool = true) throws -> ParakeetResult {
        guard samples.allSatisfy(\.isFinite) else { throw ParakeetError.invalidAudio("PCM samples must be finite") }
        return try transcribeWindows(policySampleCount: samples.count, estimatedSampleCount: samples.count,
                                     includeCtcLogProbs: includeCtcLogProbs) { start, largest, stride in
            guard start == 0 || start - stride + largest < samples.count else { return nil }
            return samples[start..<min(start + largest, samples.count)]
        }
    }

    /// File transcription with bounded PCM storage; uses the same fixed-block converter as ParakeetAudio.read.
    /// Probes at most 60 seconds to select the duration policy exactly, then keeps only overlap/window PCM.
    public func transcribe(file: URL, includeCtcLogProbs: Bool = false) throws -> ParakeetResult {
        let reader = try ParakeetAudio.Reader(file)
        var prefetched = try reader.read(60 * config.sample_rate), offset = 0
        let policyCount = prefetched.count
        var window: [Float] = []
        return try transcribeWindows(policySampleCount: policyCount, estimatedSampleCount: reader.estimatedSampleCount,
                                     includeCtcLogProbs: includeCtcLogProbs) { start, largest, stride in
            if start > 0 {
                guard window.count == largest else { return nil }
                window.removeFirst(stride)
            } else { window.reserveCapacity(largest) }
            let retained = window.count
            let fromPrefix = min(largest - window.count, prefetched.count - offset)
            if fromPrefix > 0 {
                window.append(contentsOf: prefetched[offset..<offset + fromPrefix]); offset += fromPrefix
            }
            if offset == prefetched.count { prefetched = []; offset = 0 }
            if window.count < largest { window.append(contentsOf: try reader.read(largest - window.count)) }
            if start > 0 && window.count == retained { return nil }
            return window[...]
        }
    }

    private func transcribeWindows(policySampleCount: Int, estimatedSampleCount: Int, includeCtcLogProbs: Bool,
                                   readWindow: (Int, Int, Int) throws -> ArraySlice<Float>?) throws -> ParakeetResult {
        nextTdtFrame = nil
        let adaptive = adaptiveLongAudio && policySampleCount >= 60 * config.sample_rate
        // An explicit CTC-only policy applies at every length: one window up to its size, else
        // windows of exactly that size with the given overlap.
        let ctcPolicy = !decodesTdt && ctcLongWindowSamples != nil
        usesCarriedTdtState = carryTdtStateAcrossWindows || adaptive
        // Feature normalization requires two valid mel frames (20 ms).
        guard policySampleCount >= 320 else {
            return ParakeetResult(text: "", tokens: [], tokenTimes: [], ctcText: "", ctcFrameCount: 0,
                                  ctcValues: includeCtcLogProbs ? [] : nil)
        }
        let largest = ctcPolicy ? ctcLongWindowSamples! : longRung.samples
        let frameSamples = config.encoder_frame_samples
        let overlapSeconds = ctcPolicy ? (ctcLongOverlapSeconds ?? max(6, config.long_audio.overlap_seconds))
            : adaptive ? max(6, config.long_audio.overlap_seconds) : config.long_audio.overlap_seconds
        let windowed = policySampleCount > largest   // exact below the 60 s probe; longer is always windowed
        let overlap = Int((overlapSeconds * 16000 / Double(frameSamples)).rounded()) * frameSamples
        let stride = (largest - overlap) / frameSamples * frameSamples

        // Window i's TDT starts on the CPU once window i+1 is encoded (which fixes their
        // boundary), so the CPU decodes while the Neural Engine encodes the next window.
        var pending: Window?, keepFrom = 0
        var ids: [Int] = [], times: [Double] = []
        var logProbs: [Float] = [], frameCount = 0, ctcIds: [Int] = [], previous = config.blank_id
        if includeCtcLogProbs { logProbs.reserveCapacity((estimatedSampleCount / 160 + 7) / 8 * 1025) }
        var decodeError: Error?
        let decoding = DispatchQueue(label: "ParakeetTdtCtc110M.tdt", qos: .userInitiated)
        let decodeSlot = DispatchSemaphore(value: 1)
        defer { decoding.sync {} }                   // never leave work running on shared decoder state
        func consume(_ w: Window, from a: Int, to b: Int) {
            frameCount += b - a
            if includeCtcLogProbs { logProbs.append(contentsOf: w.values![a * 1025..<b * 1025]) }
            for best in w.ctcTokens[a..<b] {
                if best != previous && best != config.blank_id { ctcIds.append(best) }
                previous = best
            }
            guard decodesTdt else { return }
            decodeSlot.wait()   // at most one decode job; do not queue retained windows without a bound
            decoding.async {
                autoreleasepool {
                    defer { decodeSlot.signal() }
                    guard decodeError == nil else { return }
                    do {
                        let (tokens, frames) = try self.tdt(w, keepFrom: a, keepTo: b)
                        ids += tokens
                        times += frames.map { Double(w.firstFrame + $0) * self.config.encoder_frame_seconds }
                    } catch { decodeError = error }
                }
            }
        }
        func accept(_ next: Window) {
            if let previousWindow = pending {
                let cut = decodesTdt || !ctcConsistentJoin ? boundary(previousWindow, next) : ctcBoundary(previousWindow, next)
                consume(previousWindow, from: keepFrom, to: cut - previousWindow.firstFrame)
                keepFrom = cut - next.firstFrame
            }
            pending = next
        }
        func forcedWindow(_ count: Int) -> Int? {
            let fit = ctcFitLastWindow && !decodesTdt && count < largest
            return ctcPolicy && windowed ? (fit ? nil : largest) : adaptive ? (fit ? nil : largest) : nil
        }
        var start = 0
        if ctcPipeline && !decodesTdt {
            try withoutActuallyEscaping(readWindow) { readWindow in
                // Worker: finish window i-1 (copy, argmax), then read and preprocess window i+1.
                // Caller: encode window i. Only the worker touches readWindow and rung loading after window 0.
                func prepareWindow(at offset: Int, protecting busy: Rung?) throws -> Prepared? {
                    guard let pcm = try readWindow(offset, largest, stride) else { return nil }
                    return try prepare(pcm, firstFrame: offset / frameSamples, reuseLargestWindow: adaptive && !ctcPolicy,
                                       forcedWindowSamples: forcedWindow(pcm.count), protecting: busy)
                }
                let worker = DispatchQueue(label: "ParakeetTdtCtc110M.ctc-pipeline", qos: .userInitiated)
                var current = try autoreleasepool { try prepareWindow(at: 0, protecting: nil) }
                var encodedPrevious: Encoded?
                while let job = current {
                    var finished: Window?, next: Prepared?, workerError: Error?
                    let toFinish = encodedPrevious, nextStart = start + stride
                    let group = DispatchGroup()
                    worker.async(group: group) {
                        autoreleasepool {
                            do {
                                if let toFinish { finished = self.finish(toFinish, retainCtcProbabilities: includeCtcLogProbs) }
                                next = try prepareWindow(at: nextStart, protecting: job.rung)
                            } catch { workerError = error }
                        }
                    }
                    let encoded: Encoded
                    do { encoded = try autoreleasepool { try runEncoder(job) } } catch { group.wait(); throw error }
                    group.wait()
                    if let workerError { throw workerError }
                    if let finished { accept(finished) }
                    encodedPrevious = encoded; current = next; start = nextStart
                }
                if let encodedPrevious { accept(finish(encodedPrevious, retainCtcProbabilities: includeCtcLogProbs)) }
            }
        } else {
            while let pcm = try readWindow(start, largest, stride) {
                let next = try autoreleasepool {
                    try encode(pcm, firstFrame: start / frameSamples, retainCtcProbabilities: includeCtcLogProbs,
                               reuseLargestWindow: adaptive && !ctcPolicy, forcedWindowSamples: forcedWindow(pcm.count))
                }
                accept(next)
                start += stride
            }
        }
        if let last = pending { consume(last, from: keepFrom, to: last.frames) }
        pending = nil
        decoding.sync {}
        if let decodeError { throw decodeError }
        return ParakeetResult(text: decodesTdt ? detokenize(ids) : detokenize(ctcIds), tokens: ids, tokenTimes: times,
                              ctcText: detokenize(ctcIds), ctcFrameCount: frameCount,
                              ctcValues: includeCtcLogProbs ? logProbs : nil)
    }

    public func detokenize(_ ids: [Int]) -> String {
        ids.compactMap { pieces.indices.contains($0) ? pieces[$0] : nil }.joined().replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Encoder

    struct Prepared { let rung: Rung; let firstFrame: Int; let mel: MLMultiArray; let melLength: MLMultiArray }
    struct Encoded {
        let firstFrame: Int
        let frames: Int
        let ctc: MLMultiArray
        let projection: MLMultiArray?
    }

    func encode(_ samples: ArraySlice<Float>, firstFrame: Int, retainCtcProbabilities: Bool = true,
                reuseLargestWindow: Bool = false, forcedWindowSamples: Int? = nil) throws -> Window {
        let prepared = try prepare(samples, firstFrame: firstFrame, reuseLargestWindow: reuseLargestWindow,
                                   forcedWindowSamples: forcedWindowSamples)
        return finish(try runEncoder(prepared), retainCtcProbabilities: retainCtcProbabilities)
    }

    /// Window choice, PCM copy and the CPU preprocessor. Runs on one thread at a time.
    func prepare(_ samples: ArraySlice<Float>, firstFrame: Int, reuseLargestWindow: Bool = false,
                 forcedWindowSamples: Int? = nil, protecting busy: Rung? = nil) throws -> Prepared {
        let n = samples.count
        let rung = forcedWindowSamples.flatMap { size in rungs.first { $0.samples == size } } ?? (reuseLargestWindow ? longRung : (rungs.first { $0.samples >= n } ?? longRung))
        try ensureLoaded(rung, protecting: busy)
        rung.audio.withUnsafeMutableBufferPointer(ofType: Float.self) { audio, strides in
            let stride = strides[1]
            if stride == 1 {
                samples.withUnsafeBufferPointer { audio.baseAddress!.update(from: $0.baseAddress!, count: n) }
                if n < rung.samples { (audio.baseAddress! + n).update(repeating: 0, count: rung.samples - n) }
            } else {
                for (i, sample) in samples.enumerated() { audio[i * stride] = sample }
                for i in n..<rung.samples { audio[i * stride] = 0 }
            }
        }
        length[0] = NSNumber(value: n)
        let pre = try rung.preprocessor.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "audio": rung.audio, "audio_length": length]))
        let mel32 = pre.featureValue(for: "mel")!.multiArrayValue!
        let mel = try Self.encoderMelInput(mel32)
        let melLength = try MLMultiArray(shape: [1], dataType: .int32)
        melLength[0] = pre.featureValue(for: "mel_length")!.multiArrayValue![0]
        return Prepared(rung: rung, firstFrame: firstFrame, mel: mel, melLength: melLength)
    }

    /// CoreML output rows may be padded (for example, width 801 with row stride 816).
    /// Copy logical values rather than interpreting padding as the next mel channel.
    static func encoderMelInput(_ features: MLMultiArray) throws -> MLMultiArray {
        guard features.shape.count == 3, features.shape[0].intValue == 1,
              features.dataType == .float32 || features.dataType == .float16 else {
            throw ParakeetError.badConfig("Expected float32/float16 mel features with shape [1, channels, frames]")
        }
        let mel = try MLMultiArray(shape: features.shape, dataType: .float16)
        let channels = features.shape[1].intValue, frames = features.shape[2].intValue
        features.withUnsafeBytes { bytes in
            // Buffer access can materialize a new backing layout: read strides inside the scope.
            let sourceRow = features.strides[1].intValue, sourceColumn = features.strides[2].intValue
            mel.withUnsafeMutableBufferPointer(ofType: Float16.self) { destination, strides in
                let targetRow = strides[1], targetColumn = strides[2]
                if features.dataType == .float32 {
                    let source = bytes.baseAddress!.assumingMemoryBound(to: Float.self)
                    for channel in 0..<channels {
                        for frame in 0..<frames {
                            destination[channel * targetRow + frame * targetColumn] = Float16(source[channel * sourceRow + frame * sourceColumn])
                        }
                    }
                } else {
                    let source = bytes.baseAddress!.assumingMemoryBound(to: Float16.self)
                    for channel in 0..<channels {
                        for frame in 0..<frames {
                            destination[channel * targetRow + frame * targetColumn] = source[channel * sourceRow + frame * sourceColumn]
                        }
                    }
                }
            }
        }
        return mel
    }

    /// The Neural Engine stage; touches only the rung's encoder and the job's own arrays.
    func runEncoder(_ job: Prepared) throws -> Encoded {
        let model = job.rung.encoder
        let input = try MLDictionaryFeatureProvider(dictionary: ["mel": job.mel, "mel_length": job.melLength])
        let out = try model.prediction(from: input)
        let frames = out.featureValue(for: "encoder_length")!.multiArrayValue![0].intValue
        let projection = out.featureValue(for: "encoder_proj")?.multiArrayValue
        guard !decodesTdt || projection != nil else { throw ParakeetError.badConfig("TDT decoding requires encoder_proj") }
        // The pipeline consumes these tensors after another prediction may have started.
        // Own their storage before releasing the prediction context or reusing the encoder.
        return try withExtendedLifetime((job, model, input, out)) {
            Encoded(firstFrame: job.firstFrame, frames: frames,
                    ctc: try Self.snapshotTensor(out.featureValue(for: "ctc_logprobs")!.multiArrayValue!),
                    projection: try decodesTdt ? projection.map(Self.snapshotTensor) : nil)
        }
    }

    static func snapshotTensor(_ source: MLMultiArray) throws -> MLMultiArray {
        let owned = try MLMultiArray(shape: source.shape, dataType: source.dataType)
        source.transfer(to: owned)
        return owned
    }

    /// CTC probability copy and per-frame argmax.
    func finish(_ e: Encoded, retainCtcProbabilities: Bool) -> Window {
        Window(firstFrame: e.firstFrame, frames: e.frames, values: Self.contiguous(e.ctc, rows: e.frames),
               retainProbabilities: retainCtcProbabilities, projection: e.projection)
    }

    /// Row r of a [1, T, C] float16/float32 array.
    static func row(_ a: MLMultiArray, _ r: Int) -> [Float] {
        a.withUnsafeBytes { bytes in
            let c = a.shape[2].intValue, rs = a.strides[1].intValue, cs = a.strides[2].intValue
            if a.dataType == .float16 {
                let p = bytes.baseAddress!.assumingMemoryBound(to: Float16.self)
                return (0..<c).map { Float(p[r * rs + $0 * cs]) }
            }
            let p = bytes.baseAddress!.assumingMemoryBound(to: Float.self)
            return (0..<c).map { Float(p[r * rs + $0 * cs]) }
        }
    }

    /// Frames each window keeps: windows meet at the overlap frame both CTC heads call most blank.
    func ownership(_ windows: [Window]) -> [(Int, Int)] {
        let bounds = zip(windows, windows.dropFirst()).map { boundary($0, $1) }
        return windows.indices.map { owned(windows[$0], index: $0, bounds: bounds) }
    }

    /// Window-relative frames window `index` keeps, given the boundaries known so far.
    func owned(_ w: Window, index i: Int, bounds: [Int]) -> (Int, Int) {
        ((i > 0 ? bounds[i - 1] : w.firstFrame) - w.firstFrame,
         (i < bounds.count ? bounds[i] : w.firstFrame + w.frames) - w.firstFrame)
    }

    /// Absolute frame where consecutive windows a and b meet.
    func boundary(_ a: Window, _ b: Window) -> Int {
        let lo = b.firstFrame, hi = a.firstFrame + a.frames
        let margin = max(1, (hi - lo) / 4)
        var candidates = Array((lo + margin)..<max(lo + margin, hi - margin))
        if candidates.isEmpty { candidates = Array(lo..<max(lo + 1, hi)) }
        let score = { (f: Int) in a.blankLogProbs[f - a.firstFrame] + b.blankLogProbs[f - b.firstFrame] }
        return candidates.max { score($0) < score($1) }!
    }

    /// CTC-only join: among the usual candidate frames, prefer a cut where the tokens stitched from about
    /// one second either side equal what one window decoded there alone, so a word both windows heard a
    /// few frames apart is kept once. Falls back to `boundary` when no candidate qualifies.
    func ctcBoundary(_ a: Window, _ b: Window) -> Int {
        let lo = b.firstFrame, hi = a.firstFrame + a.frames
        let margin = max(1, (hi - lo) / 4), reach = 12
        guard hi - lo > 2 * margin else { return boundary(a, b) }
        let blank = config.blank_id
        func collapse(_ frames: ArraySlice<Int>) -> [Int] {
            var out: [Int] = [], previous = blank
            for token in frames {
                if token != previous && token != blank { out.append(token) }
                previous = token
            }
            return out
        }
        let left = { (f: Int) in a.ctcTokens[f - a.firstFrame] }, right = { (f: Int) in b.ctcTokens[f - b.firstFrame] }
        let score = { (f: Int) in a.blankLogProbs[f - a.firstFrame] + b.blankLogProbs[f - b.firstFrame] }
        let consistent = ((lo + margin)..<(hi - margin)).filter { cut in
            let from = max(lo, cut - reach), to = min(hi, cut + reach)
            let stitched = collapse(ArraySlice((from..<cut).map(left) + (cut..<to).map(right)))
            return stitched == collapse(ArraySlice((from..<to).map(left))) || stitched == collapse(ArraySlice((from..<to).map(right)))
        }
        return consistent.max { score($0) < score($1) } ?? boundary(a, b)
    }

    // MARK: - TDT greedy decoding

    func fill(_ w: Window, at t: Int) {
        let p = w.projection!
        p.withUnsafeBytes { source in
            let rs = p.strides[1].intValue, cs = p.strides[2].intValue
            blocks.withUnsafeMutableBufferPointer(ofType: Float.self) { destination, strides in
                for j in 0..<block {
                    for k in 0..<hidden {
                        let index = j * strides[1] + k * strides[2]
                        if t + j >= w.frames { destination[index] = 0 }
                        else if p.dataType == .float16 {
                            destination[index] = Float(source.baseAddress!.assumingMemoryBound(to: Float16.self)[(t + j) * rs + k * cs])
                        } else {
                            destination[index] = source.baseAddress!.assumingMemoryBound(to: Float.self)[(t + j) * rs + k * cs]
                        }
                    }
                }
            }
        }
    }

    static func decisions(_ out: MLFeatureProvider, count: Int) -> ([Int], [Int]) {
        let tk = out.featureValue(for: "tokens")!.multiArrayValue!, du = out.featureValue(for: "durations")!.multiArrayValue!
        let tokens = tk.withUnsafeBufferPointer(ofType: Int32.self) { buffer in
            let stride = tk.strides.last!.intValue
            return (0..<count).map { Int(buffer[$0 * stride]) }
        }
        let durations = du.withUnsafeBufferPointer(ofType: Int32.self) { buffer in
            let stride = du.strides.last!.intValue
            return (0..<count).map { Int(buffer[$0 * stride]) }
        }
        return (tokens, durations)
    }

    /// NeMo-equivalent greedy TDT over one window, keeping tokens from frames keepFrom..<keepTo.
    func tdt(_ w: Window, keepFrom: Int, keepTo: Int) throws -> ([Int], [Int]) {
        if let decoder { return tdtAccelerate(w, decoder, keepFrom: keepFrom, keepTo: keepTo) }
        return try tdtCoreML(w, keepFrom: keepFrom, keepTo: keepTo)
    }

    /// The greedy walk shared by both decoders. `emit(k, t)` feeds token k (the blank id is
    /// the start symbol) to the prediction network; `decide(t)` returns the joint's best
    /// token and duration index at frame t for the latest prediction.
    func walk(frames count: Int, keepFrom: Int, keepTo: Int,
              startAt: Int = 0, beginSequence: Bool = true, didFinish: ((Int) -> Void)? = nil,
              emit: (Int, Int) throws -> Void, decide: (Int) throws -> (Int, Int)) rethrows -> ([Int], [Int]) {
        let blank = config.blank_id, durationTable = config.tdt_durations, maxSymbols = config.tdt_max_symbols_per_frame
        if beginSequence { try emit(blank, 0) }      // start-of-sequence step
        var ids: [Int] = [], frames: [Int] = []
        var t = startAt
        while t < count && t < keepTo {
            var symbols = 0
            while symbols < maxSymbols {
                let (k, duration) = try decide(t)
                var skip = durationTable[duration]
                if k != blank {
                    if t >= keepFrom { ids.append(k); frames.append(t) }
                    try emit(k, t)
                }
                symbols += 1
                if k == blank && skip == 0 { skip = 1 }
                t += skip
                if skip != 0 || t >= count { break }
            }
            if symbols == maxSymbols { t += 1 }
        }
        didFinish?(t)
        return (ids, frames)
    }

    /// One LSTM step per emitted token and one joint per visited frame, on the CPU.
    func tdtAccelerate(_ w: Window, _ decoder: TdtDecoder, keepFrom: Int, keepTo: Int) -> ([Int], [Int]) {
        let frames = Self.contiguous(w.projection!, rows: w.frames)
        let carry = usesCarriedTdtState
        let begin = !carry || nextTdtFrame == nil
        if begin { decoder.reset() }
        let start = carry ? max(keepFrom, (nextTdtFrame ?? w.firstFrame) - w.firstFrame) : 0
        return frames.withUnsafeBufferPointer { p in
            walk(frames: w.frames, keepFrom: keepFrom, keepTo: keepTo,
                 startAt: start, beginSequence: begin,
                 didFinish: { t in if carry { self.nextTdtFrame = w.firstFrame + t } },
                 emit: { k, _ in decoder.step(k) },
                 decide: { t in decoder.decide(p.baseAddress! + t * hidden) })
        }
    }

    /// The first `rows` frames of a [1, T, 640] float16/float32 array as contiguous float32.
    static func contiguous(_ a: MLMultiArray, rows: Int) -> [Float] {
        a.withUnsafeBytes { bytes in
            let c = a.shape[2].intValue, rs = a.strides[1].intValue, cs = a.strides[2].intValue
            return [Float](unsafeUninitializedCapacity: rows * c) { out, initialized in
                if a.dataType == .float16 {
                    var converted = false
                    if cs == 1 && rows > 0 {
                        var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: bytes.baseAddress!), height: vImagePixelCount(rows), width: vImagePixelCount(c), rowBytes: rs * 2)
                        var destination = vImage_Buffer(data: out.baseAddress!, height: vImagePixelCount(rows), width: vImagePixelCount(c), rowBytes: c * 4)
                        converted = vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags)) == kvImageNoError
                    }
                    if !converted {
                        let src = bytes.baseAddress!.assumingMemoryBound(to: Float16.self)
                        for r in 0..<rows { for k in 0..<c { out[r * c + k] = Float(src[r * rs + k * cs]) } }
                    }
                } else {
                    let src = bytes.baseAddress!.assumingMemoryBound(to: Float.self)
                    if cs == 1 {
                        for r in 0..<rows { out.baseAddress!.advanced(by: r * c).update(from: src + r * rs, count: c) }
                    } else {
                        for r in 0..<rows { for k in 0..<c { out[r * c + k] = src[r * rs + k * cs] } }
                    }
                }
                initialized = rows * c
            }
        }
    }

    /// Core ML fallback: a non-blank token advances the LSTM (one DecoderJoint call, which
    /// also scores the next `block` frames); walking past a block costs one JointBlock call.
    func tdtCoreML(_ w: Window, keepFrom: Int, keepTo: Int) throws -> ([Int], [Int]) {
        var h = try MLMultiArray(shape: [1, 1, NSNumber(value: hidden)], dataType: .float32)
        var c = try MLMultiArray(shape: [1, 1, NSNumber(value: hidden)], dataType: .float32)
        h.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, _ in buffer.initialize(repeating: 0) }
        c.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, _ in buffer.initialize(repeating: 0) }
        var pred: MLMultiArray? = nil
        var tokens: [Int] = [], durations: [Int] = []
        var blockStart = 0
        return try walk(frames: w.frames, keepFrom: keepFrom, keepTo: keepTo, emit: { k, t in
            fill(w, at: t)
            token[0] = NSNumber(value: k)
            let out = try decoderJoint!.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "token": token, "h": h, "c": c, "encoder_frames": blocks]))
            pred = out.featureValue(for: "pred_proj")!.multiArrayValue!
            h = out.featureValue(for: "h_out")!.multiArrayValue!
            c = out.featureValue(for: "c_out")!.multiArrayValue!
            (tokens, durations) = Self.decisions(out, count: block)
            blockStart = t
        }, decide: { t in
            if t >= blockStart + block {
                fill(w, at: t)
                let out = try jointBlock!.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                    "encoder_frames": blocks, "pred_proj": pred!]))
                (tokens, durations) = Self.decisions(out, count: block)
                blockStart = t
            }
            return (tokens[t - blockStart], durations[t - blockStart])
        })
    }
}
