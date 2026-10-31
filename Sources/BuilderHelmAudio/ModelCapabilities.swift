import Foundation

/// Runtime family, independent of the model's download repository or decoding mode.
/// Only registered SDK runtime families can be loaded.
public struct AsrArchitecture: RawRepresentable, Hashable, Sendable {
    public let rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }

    public static let parakeetV3 = Self(rawValue: "parakeet-tdt-0.6b-v3")

    public static let parakeet110M = Self(rawValue: "parakeet-tdt-ctc-110m")
}

/// Outputs and operations a model/runtime supports. TDT does not imply CTC or keyword support.
public struct AsrCapabilities: OptionSet, Hashable, Sendable {
    public let rawValue: UInt16
    public init(rawValue: UInt16) { self.rawValue = rawValue }

    public static let tdt = Self(rawValue: 1 << 0)
    public static let ctc = Self(rawValue: 1 << 1)
    public static let tokenTimestamps = Self(rawValue: 1 << 2)
    public static let wordTimestamps = Self(rawValue: 1 << 3)
    public static let segmentTimestamps = Self(rawValue: 1 << 4)
    public static let keywordSpotting = Self(rawValue: 1 << 5)

    static let parakeet110MTdt: Self = [.tdt, .ctc, .tokenTimestamps, .keywordSpotting]
    static let parakeet110MCtc: Self = [.ctc, .keywordSpotting]
}

/// Layout of one model's CTC frames. Vocabulary pieces omit the blank class.
public struct CtcConfiguration: Hashable, Sendable {
    public let classCount: Int
    public let blank: Int
    public let frameDuration: Double

    public init(classCount: Int, blank: Int, frameDuration: Double) throws {
        guard classCount > 1, (0..<classCount).contains(blank), frameDuration.isFinite, frameDuration > 0 else {
            throw BuilderHelmAudioError.invalidCtcConfiguration("Expected at least two classes, a valid blank ID and a finite positive frame duration")
        }
        self.classCount = classCount
        self.blank = blank
        self.frameDuration = frameDuration
    }

    private init() { classCount = 1025; blank = 1024; frameDuration = 0.08 }
    public static let parakeet110M = Self()
}
