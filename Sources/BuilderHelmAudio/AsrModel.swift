import Foundation

/// A pinned published recognition model or an explicitly local experimental descriptor.
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
    public var description: String { "\(repository.isEmpty ? id : repository) \(version)" }
    /// Fixed SDK input limit in seconds; nil means no fixed duration limit is imposed by the runtime.
    public var maxRequestDuration: TimeInterval? { nil }

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

    /// Multilingual Parakeet TDT 0.6B v3: FP16 text recognition in 25 languages.
    /// Offline recordings use shared-weight 15/30-second windows. No CTC or aligned timestamps.
    public static let parakeetTdtV3 = AsrModel(
        id: "parakeet-tdt-0.6b-v3",
        repository: "BuilderHelm/Parakeet-TDT-0.6B-V3-CoreML",
        version: "1.0.0",
        revision: "7f5e3fd434ac9858703b800dc1bcb2ccd8122f93",
        architecture: .parakeetV3,
        supportedLanguages: ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"],
        capabilities: [.tdt], ctcConfiguration: nil,
        files: [
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 581_731,
                 sha256: "fe8c3ea5143cfbc6b807eb906ad1df7839da24bb00d3ec1c7f626d28fb7904f6"),
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 615_729_280,
                 sha256: "654f958bb131f6e99bcd27ba2ce02a78e549285d7cc907edb736812dc4c6f14b"),
            File(path: "Encoder.mlpackage/Manifest.json", size: 617,
                 sha256: "717a4c37b8f4413e9404585eb8cfb37825117a313751bee2bf3a6e649eb2efa8"),
            File(path: "EncoderTail.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 557_772,
                 sha256: "cf4da74798755b40eaa387faf2d6df0dc6f60ae559e27d51553679539f1d260e"),
            File(path: "EncoderTail.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 607_061_824,
                 sha256: "9e171b6a97dbe3bda30921bf7710cbaec69c250202271662f60ac63cdab78aaa"),
            File(path: "EncoderTail.mlpackage/Manifest.json", size: 617,
                 sha256: "7768afd24206d54799753fbfc0ee113c310a1a6856464b97843eeacf57ae7ea8"),
            File(path: "Joint.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 4_613,
                 sha256: "33f5e1769a6c2fc5f2d4b3d831883cb270ed3eb76e1074a47015425e6658b606"),
            File(path: "Joint.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 12_642_764,
                 sha256: "4e0e63d840032f7f07ddb1d64446051166281e5491bf22da8a945c41f6eedb3e"),
            File(path: "Joint.mlpackage/Manifest.json", size: 617,
                 sha256: "dab679d1d83a2e038f3de5e9aec52783e82a8ca36295f2821ea2903058c5da14"),
            File(path: "Predictor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 12_011,
                 sha256: "589dbee0b6985ca6ce4cbff6baebcac15130ca5b6432b11c3044f252c48d1d67"),
            File(path: "Predictor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 23_604_992,
                 sha256: "48adf0f0d47c406c8253d4f7fef967436a39da14f5a65e66d5a4b407be355d41"),
            File(path: "Predictor.mlpackage/Manifest.json", size: 617,
                 sha256: "43e2d60191e3cdd2eaf1ee1ebc28e3b06e301bd540755260c900b3ba2b857218"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 23_546,
                 sha256: "67eb57c1f6768d3b041c0136be380c6df36907df9fef72daceb73933d1584e9f"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 7_348_032,
                 sha256: "c7b165bbc95ac8538b0e3dc28e4f7a770dfa31e1cc9f754cb9055c211d224b21"),
            File(path: "Preprocessor.mlpackage/Manifest.json", size: 617,
                 sha256: "f49015764d1241e685821db2253738b10fb6b0f98c6ece34b3ba92b4790bd52d"),
            File(path: "config.json", size: 3_125,
                 sha256: "71c965a403144b3584237081309cf419e4fc73e26ad10833e511fa5448379e88"),
            File(path: "vocab.json", size: 95_041,
                 sha256: "6d23d30bfea120aaf0b911a3e2f66a72b011d0722554550b0bce821e1bd6409b"),
        ])

    /// Locally managed v3 exports. Use `SpeechRecognizer(directory:model:)`.
    /// This descriptor has no remote repository and supports TDT text only.
    public static let experimentalParakeetV3 = AsrModel(
        id: "parakeet-tdt-0.6b-v3-local", repository: "", version: "experimental", revision: "",
        architecture: .parakeetV3,
        supportedLanguages: ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"],
        capabilities: [.tdt], ctcConfiguration: nil, files: [])

    public static let all: [AsrModel] = [.parakeetTdtCtc110M, .parakeetCtc110M, .parakeetTdtV3]

    public static func named(_ id: String) -> AsrModel? { all.first { $0.id == id } }
}
