import Foundation

/// A speech recognition model published by BuilderHelm on Hugging Face, pinned to an exact revision.
public struct AsrModel: Sendable, Hashable, CustomStringConvertible {
    /// A file in the model repository, with the size and SHA-256 it must have after download.
    public struct File: Sendable, Hashable {
        public let path: String
        public let size: Int64
        public let sha256: String
    }

    /// Stable identifier, used by the CLI and the cache.
    public let id: String
    /// Hugging Face repository, for example `BuilderHelm/Parakeet-CTC-110M-CoreML`.
    public let repository: String
    /// Base model release. Compatibility revisions can fix the export graph without changing trained weights.
    public let version: String
    /// Full commit hash. Downloads use this, so a moved tag or branch cannot change the files.
    public let revision: String
    public let architecture: AsrArchitecture
    /// BCP-47 language codes advertised by this model.
    public let supportedLanguages: [String]
    public let capabilities: AsrCapabilities
    public let ctcConfiguration: CtcConfiguration?
    /// Compatibility accessor. TDT support alone does not imply CTC or token timestamps.
    public var decodesTdt: Bool { capabilities.contains(.tdt) }
    public let files: [File]

    public var totalBytes: Int64 { files.reduce(0) { $0 + $1.size } }
    public var description: String { "\(repository) \(version)" }

    init(id: String, repository: String, version: String, revision: String, architecture: AsrArchitecture,
         supportedLanguages: [String], capabilities: AsrCapabilities, ctcConfiguration: CtcConfiguration?, files: [File]) {
        self.id = id
        self.repository = repository
        self.version = version
        self.revision = revision
        self.architecture = architecture
        self.supportedLanguages = supportedLanguages
        self.capabilities = capabilities
        self.ctcConfiguration = ctcConfiguration
        self.files = files
    }

    // Preserve construction of the existing 110M descriptors and internal manifest fixtures.
    init(id: String, repository: String, version: String, revision: String, decodesTdt: Bool, files: [File]) {
        self.init(id: id, repository: repository, version: version, revision: revision, architecture: .parakeet110M,
                  supportedLanguages: ["en"], capabilities: decodesTdt ? .parakeet110MTdt : .parakeet110MCtc,
                  ctcConfiguration: .parakeet110M, files: files)
    }

    // The pinned compatibility exports use bounded feature predictions followed by full-window normalization before CoreML compilation.
    /// Parakeet TDT + CTC 110M: TDT transcript with token times, plus CTC from the same encoder pass.
    public static let parakeetTdtCtc110M = AsrModel(
        id: "parakeet-tdt-ctc-110m",
        repository: "BuilderHelm/Parakeet-TDT-CTC-110M-CoreML",
        version: "1.1.0",
        revision: "261cc46b2b8a89448c19b1d7daccd5416259c9a8",
        architecture: .parakeet110M,
        supportedLanguages: ["en"],
        capabilities: [.tdt, .ctc, .tokenTimestamps, .keywordSpotting],
        ctcConfiguration: .parakeet110M,
        files: [
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 1_911_845,
                 sha256: "7ff5abdf3c09357b3054f40d910e00e979a48a0eafca7c6e0adbd1047194f10e"),
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 116_090_922,
                 sha256: "b2b8fdc4c8adcfd4f1ea851c862c463ced05cc35a1aebb20fa2eaec8b9b0da27"),
            File(path: "Encoder.mlpackage/Manifest.json", size: 617,
                 sha256: "51dd5c87244a688739c8e2f08d7e60064ad24ce33f74cd77fb9fa03ada548578"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 59_787,
                 sha256: "15edb0776d2dc4195cee57bf8d049787061a50eb7d5941e00d3eed9b5f249428"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 5_890_820,
                 sha256: "41b2384ebe5ae8b3760b4a31c13b4d6145db3196b55c2b4f2a11b0e85b5b7e30"),
            File(path: "Preprocessor.mlpackage/Manifest.json", size: 617,
                 sha256: "422610cf421d9331f6f5117d2543d808532a9a46c0e8f2387d2a3be51cba56aa"),
            File(path: "TdtDecoder.bin", size: 20_033_560,
                 sha256: "1e4c6c9173d76b25cfa158181bad301c8beef667069ef92d5bd1dbeb1cabc784"),
            File(path: "config.json", size: 5_365,
                 sha256: "fa6b8041f6f72953ba1fabe6cf3ae0042ea4e7a640bccaf610795851a9aeb9c0"),
            File(path: "vocab.json", size: 9_004,
                 sha256: "ecbc3fe1c391a3feea1d1c39f8c15af674bd03cbaf0d0f5de4a564771c16bd16"),
        ])

    /// Parakeet CTC 110M: CTC-only encoder for the fastest transcription and keyword spotting.
    public static let parakeetCtc110M = AsrModel(
        id: "parakeet-ctc-110m",
        repository: "BuilderHelm/Parakeet-CTC-110M-CoreML",
        version: "1.1.0",
        revision: "436962caa4bb6d546cf3980bc9313b03361736f7",
        architecture: .parakeet110M,
        supportedLanguages: ["en"],
        capabilities: [.ctc, .keywordSpotting],
        ctcConfiguration: .parakeet110M,
        files: [
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 1_911_663,
                 sha256: "db0ab044580aae0b4b44f0c2ec790f6f94cd12b2accb714cc5d649b8659c26cf"),
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 116_090_922,
                 sha256: "b2b8fdc4c8adcfd4f1ea851c862c463ced05cc35a1aebb20fa2eaec8b9b0da27"),
            File(path: "Encoder.mlpackage/Manifest.json", size: 617,
                 sha256: "19b440d4fea6d14bc88d41fd8d64252a246c459d9de06a75199f869e4170deed"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 59_787,
                 sha256: "15edb0776d2dc4195cee57bf8d049787061a50eb7d5941e00d3eed9b5f249428"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 5_890_820,
                 sha256: "41b2384ebe5ae8b3760b4a31c13b4d6145db3196b55c2b4f2a11b0e85b5b7e30"),
            File(path: "Preprocessor.mlpackage/Manifest.json", size: 617,
                 sha256: "422610cf421d9331f6f5117d2543d808532a9a46c0e8f2387d2a3be51cba56aa"),
            File(path: "config.json", size: 2_928,
                 sha256: "44633b7fb12cb99ac7c5a4837dd3ba2d426bec7e72ab46b939a80da3a9b64e85"),
            File(path: "vocab.json", size: 9_004,
                 sha256: "ecbc3fe1c391a3feea1d1c39f8c15af674bd03cbaf0d0f5de4a564771c16bd16"),
        ])

    public static let all: [AsrModel] = [.parakeetTdtCtc110M, .parakeetCtc110M]

    public static func named(_ id: String) -> AsrModel? { all.first { $0.id == id } }
}
