import Foundation

/// A pinned published recognition model or an explicitly local experimental descriptor.
public struct AsrModel: Sendable, Hashable, CustomStringConvertible {
    /// A file in the model repository, with the size and SHA-256 it must have after download.
    public struct File: Sendable, Hashable {
        /// Path inside the local model folder, where the runtime expects this file.
        public let path: String
        /// Path in the pinned repository; may point into a variant subfolder or a shared package.
        public let repositoryPath: String
        public let size: Int64
        public let sha256: String

        init(path: String, size: Int64, sha256: String, repositoryPath: String? = nil) {
            self.path = path
            self.repositoryPath = repositoryPath ?? path
            self.size = size
            self.sha256 = sha256
        }
    }

    /// Stable identifier, used by the CLI and the cache.
    public let id: String
    /// Hugging Face repository, for example `BuilderHelm/Parakeet-CTC-110M-CoreML`.
    public let repository: String
    /// Base model release. Compatibility revisions can fix the export graph without changing trained weights.
    public let version: String
    /// Full commit hash. Downloads use this, so a moved tag or branch cannot change the files.
    public let revision: String
    /// Optional artifact variant within a repository revision. Nil preserves the original cache path.
    /// Variants have independent download, verification and removal lifecycles.
    public let variant: String?
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
         supportedLanguages: [String], capabilities: AsrCapabilities, ctcConfiguration: CtcConfiguration?, files: [File],
         variant: String? = nil) {
        self.id = id
        self.repository = repository
        self.version = version
        self.revision = revision
        self.variant = variant
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

    /// Multilingual Parakeet TDT 0.6B v3: FP16 text recognition in 25 languages, with NeMo-compatible
    /// token and word timestamps and confidences. Offline recordings use shared-weight 15/30-second windows. No CTC.
    public static let parakeetTdtV3 = AsrModel(
        id: "parakeet-tdt-0.6b-v3",
        repository: "BuilderHelm/Parakeet-TDT-0.6B-V3-CoreML",
        version: "1.0.1",
        revision: "3578e62cd8d94e56337bcc23e8b21e7f25617fbf",
        architecture: .parakeetV3,
        supportedLanguages: ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"],
        capabilities: .parakeetV3Tdt, ctcConfiguration: nil,
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
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 37_760,
                 sha256: "7a67f874be8af28a9a868415387d08df9df11762bedf270aa7c82d212312e964"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 7_348_032,
                 sha256: "c7b165bbc95ac8538b0e3dc28e4f7a770dfa31e1cc9f754cb9055c211d224b21"),
            File(path: "Preprocessor.mlpackage/Manifest.json", size: 617,
                 sha256: "f49015764d1241e685821db2253738b10fb6b0f98c6ece34b3ba92b4790bd52d"),
            File(path: "config.json", size: 3_156,
                 sha256: "c8bfaab9c017738885809f320f564219ee297f8ea32a2456b1d3abe7cba5ef7e"),
            File(path: "vocab.json", size: 95_041,
                 sha256: "6d23d30bfea120aaf0b911a3e2f66a72b011d0722554550b0bce821e1bd6409b"),
        ])

    /// Cadence Sae (English): Parakeet TDT 0.6B v3 with its top 8 encoder layers, predictor and joint fine-tuned
    /// on accented English and meeting speech, encoder weights 6-bit palettized. English TDT text with token and
    /// word timestamps; same runtime, windows and capabilities as `.parakeetTdtV3`. No CTC.
    /// The repository also holds FP16, 4-bit and mixed builds; this descriptor pins the 6-bit folder.
    public static let cadenceSaeEn = AsrModel(
        id: "cadence-sae-en",
        repository: "BuilderHelm/Cadence-Sae-EN",
        version: "1.0.0",
        revision: "4702453cd0c6f922b1480857223371db53aa6d40",
        architecture: .parakeetV3,
        supportedLanguages: ["en"],
        capabilities: .parakeetV3Tdt, ctcConfiguration: nil,
        files: [
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 621_181,
                 sha256: "9f28e452a093756272e0ae9177f5fa9213372cb57aee2ed2d0b3dd25168087a9",
                 repositoryPath: "coreml/6bit/Encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 250_213_696,
                 sha256: "4349fdacba055f9666939541ae05831df5ddfd8bfdcb9c8dfa88b4d01d2d6404",
                 repositoryPath: "coreml/6bit/Encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
            File(path: "Encoder.mlpackage/Manifest.json", size: 617,
                 sha256: "1edef987af8faf263b886966a1cbe845790361a31c2f4af23c50d9e8dcb6281f",
                 repositoryPath: "coreml/6bit/Encoder.mlpackage/Manifest.json"),
            File(path: "EncoderTail.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 595_432,
                 sha256: "81afb64f9b8e106be5313d3f3625805e3a67c1a83b8959e90dcac214121e73cd",
                 repositoryPath: "coreml/6bit/EncoderTail.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
            File(path: "EncoderTail.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 246_942_784,
                 sha256: "3689d1b56a663a100cd7a3f03e5a73531702cba0c7239c27a5e83af5a3f67c76",
                 repositoryPath: "coreml/6bit/EncoderTail.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
            File(path: "EncoderTail.mlpackage/Manifest.json", size: 617,
                 sha256: "79d85be086f84c933f98377380eb0c4513259c30d4f796d63c8d43fa176c8d1a",
                 repositoryPath: "coreml/6bit/EncoderTail.mlpackage/Manifest.json"),
            File(path: "Joint.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 4_647,
                 sha256: "ef71e774a9ca37347188276fccb0aa474481b77e5dcc3fe0faf8af9ba7283d73",
                 repositoryPath: "coreml/6bit/Joint.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
            File(path: "Joint.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 12_642_764,
                 sha256: "a7795226a2c21f50680cbee042ff9452ed30c33027ec547ffc557abf9f144319",
                 repositoryPath: "coreml/6bit/Joint.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
            File(path: "Joint.mlpackage/Manifest.json", size: 617,
                 sha256: "7148f2bcb95efee49f5710c4884c9ba49c76ee723f1d3292dbf96faf681141a2",
                 repositoryPath: "coreml/6bit/Joint.mlpackage/Manifest.json"),
            File(path: "Predictor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 12_045,
                 sha256: "a762164ff413495f702b6d770434319860cfab89d9f51158d6a0b65e1fa208b7",
                 repositoryPath: "coreml/6bit/Predictor.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
            File(path: "Predictor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 23_604_992,
                 sha256: "9f74ceb8e28a6839bb1a8b77e75f2b1e4adce9a95769619a6c5a94bb33fa07e7",
                 repositoryPath: "coreml/6bit/Predictor.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
            File(path: "Predictor.mlpackage/Manifest.json", size: 617,
                 sha256: "90e18803615868935c317c334202cb4390ee288f41c91578fd52550aeee40c60",
                 repositoryPath: "coreml/6bit/Predictor.mlpackage/Manifest.json"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 37_760,
                 sha256: "7a67f874be8af28a9a868415387d08df9df11762bedf270aa7c82d212312e964",
                 repositoryPath: "coreml/6bit/Preprocessor.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 7_348_032,
                 sha256: "c7b165bbc95ac8538b0e3dc28e4f7a770dfa31e1cc9f754cb9055c211d224b21",
                 repositoryPath: "coreml/6bit/Preprocessor.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
            File(path: "Preprocessor.mlpackage/Manifest.json", size: 617,
                 sha256: "ed2bd2f705935ab4db0cc5cf51f85068fa675108426225891f74a1d6d602229c",
                 repositoryPath: "coreml/6bit/Preprocessor.mlpackage/Manifest.json"),
            File(path: "cadence-export.json", size: 822,
                 sha256: "af82a167eb187306f0605580a6cc148061e86c734a178c20c098d014f5e6a66d",
                 repositoryPath: "coreml/6bit/cadence-export.json"),
            File(path: "config.json", size: 3_394,
                 sha256: "f14da96cabcdd23e1f19543e895cd6b554eeff567bafe2d5edc0ddad6d2bce9f",
                 repositoryPath: "coreml/6bit/config.json"),
            File(path: "merge-record.json", size: 554,
                 sha256: "79e69e9b695aa9972bdf43a9ca774c56ec5b299ce72f6e4784a85c02eccb8872",
                 repositoryPath: "coreml/6bit/merge-record.json"),
            File(path: "vocab.json", size: 95_041,
                 sha256: "6d23d30bfea120aaf0b911a3e2f66a72b011d0722554550b0bce821e1bd6409b",
                 repositoryPath: "coreml/6bit/vocab.json"),
        ],
        variant: "6bit")

    /// Cadence Masubi: a multilingual FP16 fine-tune of Parakeet TDT V3.
    /// Uses the same 25 languages and TDT runtime, with token and word timestamps.
    public static let cadenceMasubiMultilingual = AsrModel(
        id: "cadence-masubi-multilingual",
        repository: "BuilderHelm/Cadence-Masubi-Multilingual-CoreML",
        version: "1.0.0",
        revision: "29640e2a8f075b250c352df8b6add73a8ba7ae8e",
        architecture: .parakeetV3,
        supportedLanguages: parakeetTdtV3.supportedLanguages,
        capabilities: .parakeetV3Tdt, ctcConfiguration: nil,
        files: [
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 581_731,
                 sha256: "2df1e07c8498c045bf3e01cf8f4eb9181628037ffec414d80e39078ffbeb5d32"),
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 615_729_280,
                 sha256: "654f958bb131f6e99bcd27ba2ce02a78e549285d7cc907edb736812dc4c6f14b"),
            File(path: "Encoder.mlpackage/Manifest.json", size: 617,
                 sha256: "229cf4fb1d80c98377fc655cf2d1ca6af74fd160b3e4cd192df4c8fd5396a52a"),
            File(path: "EncoderTail.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 557_772,
                 sha256: "5f0f0de748a62918370035c907636cf554b238a8ce02b013ac379d7ac56c67e2"),
            File(path: "EncoderTail.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 607_061_824,
                 sha256: "79deb1670d1a5bdbc0bb7cf2fc237efd9022e6b84324fb508790e4d75419ea81"),
            File(path: "EncoderTail.mlpackage/Manifest.json", size: 617,
                 sha256: "4a264e1c70836d313d39cb912bc5b4d5d6e847ea766c166d663a3d042da7d5df"),
            File(path: "Joint.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 4_647,
                 sha256: "95b669c4eaa7500ae78a8f060dee3c0f0412370d3d61610acd7550ef84c88397"),
            File(path: "Joint.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 12_642_764,
                 sha256: "f42cc3e8b05299bd2093cc55f61ad6106d00bc7d864fe8da8d02f222b680b283"),
            File(path: "Joint.mlpackage/Manifest.json", size: 617,
                 sha256: "ce0429c78636233a41e3e0e122b402fda21f370a3a9f87f110faca071f84833c"),
            File(path: "Predictor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 12_045,
                 sha256: "38c70bb852f8338b9113eef8c6338ee987561ddc7b2bea5723959929166c3e2a"),
            File(path: "Predictor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 23_604_992,
                 sha256: "91d96ac3f1f10068fdfe868d0362d5b28e79b1a2bf2a0fee89077f8640f6023a"),
            File(path: "Predictor.mlpackage/Manifest.json", size: 617,
                 sha256: "96f2ec9e5e5fa5f97361ab9d489887742173748b4c167c1222c2eefb5c2abd15"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 37_760,
                 sha256: "7a67f874be8af28a9a868415387d08df9df11762bedf270aa7c82d212312e964"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 7_348_032,
                 sha256: "c7b165bbc95ac8538b0e3dc28e4f7a770dfa31e1cc9f754cb9055c211d224b21"),
            File(path: "Preprocessor.mlpackage/Manifest.json", size: 617,
                 sha256: "ba1316c90f5445c311355003b6d755abcd514c166ee22c73a6e05e53d123aa82"),
            File(path: "config.json", size: 2_917,
                 sha256: "1260f437d2356dde8d0e1311a80bf2e74774e2c89f267ed176b4f812fbe7ad96"),
            File(path: "vocab.json", size: 95_041,
                 sha256: "6d23d30bfea120aaf0b911a3e2f66a72b011d0722554550b0bce821e1bd6409b"),
        ])

    /// Locally managed v3 exports. Use `SpeechRecognizer(directory:model:)`.
    /// This descriptor has no remote repository and supports TDT text, token and word timestamps.
    public static let experimentalParakeetV3 = AsrModel(
        id: "parakeet-tdt-0.6b-v3-local", repository: "", version: "experimental", revision: "",
        architecture: .parakeetV3,
        supportedLanguages: ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"],
        capabilities: .parakeetV3Tdt, ctcConfiguration: nil, files: [])

    /// Parakeet Unified EN 0.6B (offline mode) with an unquantized FP16 encoder: English RNN-T text with
    /// punctuation and capitalization, NeMo-compatible token and word timestamps and confidences.
    /// Requests use the smallest of five shared-weight windows (5–30 s); longer recordings use 30 s windows.
    public static let parakeetUnifiedEn = AsrModel(
        id: "parakeet-unified-en-0.6b",
        repository: "BuilderHelm/Parakeet-Unified-EN-0.6B-CoreML",
        version: "1.0.0",
        revision: "97e349c3dee2a9f683a435aac1157374a465d613",
        architecture: .parakeetUnifiedEn,
        supportedLanguages: ["en"],
        capabilities: .parakeetUnifiedRnnt, ctcConfiguration: nil,
        files: [
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 1_602_559,
                 sha256: "4748cdb9f17ee96ad48dd64f08004e0c093787d0670011c1a055c61ffea99e5a"),
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 637_704_128,
                 sha256: "dc0dea41e39cd0de8fa4496103ec9ca6d7598958c14ee07ba6a3632687afcbb2"),
            File(path: "Encoder.mlpackage/Manifest.json", size: 617,
                 sha256: "7eed98ace5c004204e33250a94c6afeff722bf7b6d61139a7c7896df51470291"),
            File(path: "EncoderTail.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 1_542_902,
                 sha256: "14db247a37dba8e86c17dde238f910676db5039c283929e1759527909b368c99"),
            File(path: "EncoderTail.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 629_032_384,
                 sha256: "b1ecda2e02d4f26d4c0e699e9d2d2d5c45bc677c709700a28ad2e93f55d3fd53"),
            File(path: "EncoderTail.mlpackage/Manifest.json", size: 617,
                 sha256: "c7ceae9a4ea0f5f31eccaf4e533b2428bf1b4600d5905d07e9c9496c217b360a"),
            File(path: "Joint.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 4_722,
                 sha256: "cc5d61412ce04db6d6266598689049c1c45d3401509ec2a5f2b56c546ace6191"),
            File(path: "Joint.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 3_446_978,
                 sha256: "06831afa6d1beb0c0b10350ebf7886bc37638e951d14e738d7e06fbd2a05012f"),
            File(path: "Joint.mlpackage/Manifest.json", size: 617,
                 sha256: "8c4ef577c0bc99e629664e37bf449284c08b6497a2d2a18868b4a4dd482292e5"),
            File(path: "Predictor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 12_119,
                 sha256: "00e6c57defa6d55421c935973755958c3d479fb26d45b4de285dfea25373ea2b"),
            File(path: "Predictor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 14_429_952,
                 sha256: "96f990461a5986d5e7309ad1a0f36084fbf0f4b28aec35948f8b8d0dcbf8599e"),
            File(path: "Predictor.mlpackage/Manifest.json", size: 617,
                 sha256: "9712c081cb64c445aa13a8dfeb0b0e988b2a1264c34ca0d74697c43f4ef7e397"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 88_752,
                 sha256: "1f6eee5b7b1db05a54a766e7a2f950848d491260b1868a3ff882057ea842e014"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 12_356_416,
                 sha256: "e3a36a729352a6f36f19fe2eec354f456e66fcd42d6aaefadfc83b739b3f2e45"),
            File(path: "Preprocessor.mlpackage/Manifest.json", size: 617,
                 sha256: "ae80310bd602296dbe1b2e108f5970128566f8d200a3c7f3c29e4a6e6f5be724"),
            File(path: "RnntDecoder.bin", size: 35_772_420,
                 sha256: "95b0591f0643ff94f3ed1a43ba4f8917abbfa4f10a9272b983502a60ece64851"),
            File(path: "config.json", size: 4_756,
                 sha256: "dd54d7d5586e1595f9e1e672c8f5cc097cbf621f7517bc776635b06286d78e97"),
            File(path: "vocab.json", size: 10_055,
                 sha256: "8e71323640de5f3be9a7fd7a306231ed3fd3dc654c59afb95cc2d0fabc2003e4"),
        ])

    /// Parakeet Unified EN 0.6B with a weight-only INT8 encoder (FP16 compute): the same accuracy as
    /// `.parakeetUnifiedEn` in every published evaluation, about half the download and faster on the Neural Engine.
    public static let parakeetUnifiedEnInt8 = AsrModel(
        id: "parakeet-unified-en-0.6b-int8",
        repository: "BuilderHelm/Parakeet-Unified-EN-0.6B-CoreML",
        version: "1.0.0",
        revision: "97e349c3dee2a9f683a435aac1157374a465d613",
        architecture: .parakeetUnifiedEn,
        supportedLanguages: ["en"],
        capabilities: .parakeetUnifiedRnnt, ctcConfiguration: nil,
        files: [
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 3_333_384,
                 sha256: "a225e90ab390883b382665a5e9304055d3a262573d2f60b7513601a232c2b46d",
                 repositoryPath: "variants/int8/Encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
            File(path: "Encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 635_155_882,
                 sha256: "1bb65108e67898c13737b9485df2c2afc35516d0904b9a293a1cc43e2a7520fd",
                 repositoryPath: "variants/int8/Encoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
            File(path: "Encoder.mlpackage/Manifest.json", size: 617,
                 sha256: "f4a733e57cdb515cb34fdb0c557848b7fdae0383e119e099c5dafc15b194bd5d",
                 repositoryPath: "variants/int8/Encoder.mlpackage/Manifest.json"),
            File(path: "Joint.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 4_722,
                 sha256: "cc5d61412ce04db6d6266598689049c1c45d3401509ec2a5f2b56c546ace6191"),
            File(path: "Joint.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 3_446_978,
                 sha256: "06831afa6d1beb0c0b10350ebf7886bc37638e951d14e738d7e06fbd2a05012f"),
            File(path: "Joint.mlpackage/Manifest.json", size: 617,
                 sha256: "8c4ef577c0bc99e629664e37bf449284c08b6497a2d2a18868b4a4dd482292e5"),
            File(path: "Predictor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 12_119,
                 sha256: "00e6c57defa6d55421c935973755958c3d479fb26d45b4de285dfea25373ea2b"),
            File(path: "Predictor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 14_429_952,
                 sha256: "96f990461a5986d5e7309ad1a0f36084fbf0f4b28aec35948f8b8d0dcbf8599e"),
            File(path: "Predictor.mlpackage/Manifest.json", size: 617,
                 sha256: "9712c081cb64c445aa13a8dfeb0b0e988b2a1264c34ca0d74697c43f4ef7e397"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 88_752,
                 sha256: "1f6eee5b7b1db05a54a766e7a2f950848d491260b1868a3ff882057ea842e014"),
            File(path: "Preprocessor.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 12_356_416,
                 sha256: "e3a36a729352a6f36f19fe2eec354f456e66fcd42d6aaefadfc83b739b3f2e45"),
            File(path: "Preprocessor.mlpackage/Manifest.json", size: 617,
                 sha256: "ae80310bd602296dbe1b2e108f5970128566f8d200a3c7f3c29e4a6e6f5be724"),
            File(path: "RnntDecoder.bin", size: 35_772_420,
                 sha256: "95b0591f0643ff94f3ed1a43ba4f8917abbfa4f10a9272b983502a60ece64851"),
            File(path: "config.json", size: 4_367,
                 sha256: "37865273f6a7261d69c7475d13370407d6cecb9ce2d3c3a82a9923f8828cf970",
                 repositoryPath: "variants/int8/config.json"),
            File(path: "vocab.json", size: 10_055,
                 sha256: "8e71323640de5f3be9a7fd7a306231ed3fd3dc654c59afb95cc2d0fabc2003e4"),
        ],
        variant: "int8")

    /// Locally managed Parakeet Unified EN 0.6B exports (offline mode, FP16 or the INT8-encoder variant
    /// folder). Use `SpeechRecognizer(directory:model:)`. English RNN-T text with token and word timestamps.
    public static let experimentalParakeetUnifiedEn = AsrModel(
        id: "parakeet-unified-en-0.6b-local", repository: "", version: "experimental", revision: "",
        architecture: .parakeetUnifiedEn, supportedLanguages: ["en"],
        capabilities: .parakeetUnifiedRnnt, ctcConfiguration: nil, files: [])

    public static let all: [AsrModel] = [
        .parakeetTdtCtc110M, .parakeetCtc110M, .parakeetTdtV3,
        .cadenceSaeEn, .cadenceMasubiMultilingual, .parakeetUnifiedEn, .parakeetUnifiedEnInt8,
    ]
    /// Descriptors for model folders you manage yourself, by id.
    public static let local: [AsrModel] = [.experimentalParakeetV3, .experimentalParakeetUnifiedEn]

    public static func named(_ id: String) -> AsrModel? { all.first { $0.id == id } }
}
