import CryptoKit
import Foundation
import Testing
@testable import BuilderHelmAudio

private func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

/// A fake model served from a local folder laid out like Hugging Face (`<repo>/resolve/<revision>/<path>`).
private func fixture(corrupt: Bool = false) throws -> (store: ModelStore, model: AsrModel, root: URL) {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("bh-store-\(UUID().uuidString)")
    let contents = ["a/one.bin": Data(repeating: 7, count: 300_000), "two.json": Data("[1,2,3]".utf8)]
    let revision = String(repeating: "ab", count: 20)
    let served = root.appendingPathComponent("hub/Test/Model/resolve/\(revision)")
    for (path, data) in contents {
        let url = served.appendingPathComponent(path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }
    let files = contents.keys.sorted().map { path in
        AsrModel.File(path: path, size: Int64(contents[path]!.count), sha256: corrupt && path == "two.json" ? String(repeating: "0", count: 64) : sha256(contents[path]!))
    }
    let model = AsrModel(id: "test", repository: "Test/Model", version: "0", revision: revision, decodesTdt: false, files: files)
    let store = ModelStore(root: root.appendingPathComponent("store"), baseURL: root.appendingPathComponent("hub"))
    return (store, model, root)
}

@Test func downloadsVerifiesAndReportsProgress() async throws {
    let (store, model, root) = try fixture()
    defer { try? FileManager.default.removeItem(at: root) }
    #expect(!store.isDownloaded(model))
    let seen = Counter()
    let directory = try await store.download(model) { seen.add($0.completedBytes) }
    #expect(store.isDownloaded(model))
    #expect(try await store.verify(model))
    #expect(FileManager.default.fileExists(atPath: directory.appendingPathComponent("a/one.bin").path))
    #expect(seen.last == model.totalBytes)
    // A damaged file is detected by size and fetched again.
    try Data("x".utf8).write(to: directory.appendingPathComponent("two.json"))
    #expect(!store.isDownloaded(model))
    try await store.download(model)
    #expect(try await store.verify(model))
}

@Test func rejectsChecksumMismatch() async throws {
    let (store, model, root) = try fixture(corrupt: true)
    defer { try? FileManager.default.removeItem(at: root) }
    await #expect(throws: BuilderHelmAudioError.checksumMismatch(file: "two.json")) { try await store.download(model) }
    #expect(!store.isDownloaded(model))
}

@Test func cancelledDownloadDoesNotCreateModelStorage() async throws {
    let (store, model, root) = try fixture()
    defer { try? FileManager.default.removeItem(at: root) }
    let task = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        return try await store.download(model)
    }
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("store").path))
    #expect(!store.isDownloaded(model))
}

@Test func publishedManifestsArePinned() {
    for model in AsrModel.all {
        #expect(model.revision.count == 40 && model.revision.allSatisfy(\.isHexDigit))
        #expect(model.files.allSatisfy { $0.sha256.count == 64 && $0.size > 0 })
        #expect(Set(model.files.map(\.path)).isSuperset(of: ["config.json", "vocab.json", "Encoder.mlpackage/Manifest.json"]))
        #expect(model.files.contains { $0.path == "TdtDecoder.bin" } == (model.architecture == .parakeet110M && model.decodesTdt))
        if model.architecture == .parakeetV3 {
            #expect(model.files.contains { $0.path == "Predictor.mlpackage/Manifest.json" })
            #expect(model.files.contains { $0.path == "Joint.mlpackage/Manifest.json" })
        }
    }
    #expect(Set(AsrModel.all.map(\.id)).count == AsrModel.all.count)
}

@Test func sameRepositoryRevisionVariantsHaveIndependentLifecycles() async throws {
    let (store, original, root) = try fixture()
    defer { try? FileManager.default.removeItem(at: root) }
    let bytes = Data("quantized fixture".utf8)
    let remotePath = "variants/palette6/config.json"
    let served = root.appendingPathComponent("hub/\(original.repository)/resolve/\(original.revision)/\(remotePath)")
    try FileManager.default.createDirectory(at: served.deletingLastPathComponent(), withIntermediateDirectories: true)
    try bytes.write(to: served)
    let quantized = AsrModel(
        id: "test-palette6", repository: original.repository, version: original.version,
        revision: original.revision, architecture: original.architecture,
        supportedLanguages: original.supportedLanguages, capabilities: original.capabilities,
        ctcConfiguration: original.ctcConfiguration,
        files: [.init(path: "config.json", size: Int64(bytes.count), sha256: sha256(bytes), repositoryPath: remotePath),
                original.files.first!], variant: "palette6")
    let originalDirectory = store.directory(for: original)
    #expect(originalDirectory == root.appendingPathComponent("store/Models/Test/Model/\(original.revision)", isDirectory: true))
    #expect(store.directory(for: quantized) != originalDirectory)

    try await store.download(quantized)
    let downloadedConfig = try Data(contentsOf: store.directory(for: quantized).appendingPathComponent("config.json"))
    #expect(downloadedConfig == bytes)
    #expect(FileManager.default.fileExists(atPath: store.directory(for: quantized).appendingPathComponent(original.files.first!.path).path))
    #expect(store.isDownloaded(quantized))
    #expect(!store.isDownloaded(original))
    #expect(!FileManager.default.fileExists(atPath: originalDirectory.path))
    try await store.download(original)
    #expect(try await store.verify(original))
    #expect(try await store.verify(quantized))

    // A same-size corruption invalidates only this variant, and can be repaired independently.
    let quantizedConfig = store.directory(for: quantized).appendingPathComponent("config.json")
    try Data(repeating: 0, count: bytes.count).write(to: quantizedConfig)
    let corruptedVerifies = try await store.verify(quantized)
    #expect(!corruptedVerifies)
    #expect(!store.isDownloaded(quantized))
    #expect(store.isDownloaded(original))
    #expect(try await store.verify(original))
    try await store.download(quantized)
    #expect(try await store.verify(quantized))

    // Legacy removal must not recursively remove the explicit variant at the same revision.
    try await store.remove(original)
    #expect(!store.isDownloaded(original))
    #expect(try await store.verify(quantized))
    #expect(store.isDownloaded(quantized))
    try await store.download(original)
    try await store.remove(quantized)
    #expect(!store.isDownloaded(quantized))
    #expect(store.isDownloaded(original))
    #expect(try await store.verify(original))
}

@Test func explicitVariantsDoNotShareReadinessOrFiles() async throws {
    let (store, original, root) = try fixture()
    defer { try? FileManager.default.removeItem(at: root) }
    func variant(_ name: String, contents: Data) -> AsrModel {
        AsrModel(id: "test-\(name)", repository: original.repository, version: original.version,
                 revision: original.revision, architecture: original.architecture,
                 supportedLanguages: original.supportedLanguages, capabilities: original.capabilities,
                 ctcConfiguration: original.ctcConfiguration,
                 files: [.init(path: "config.json", size: Int64(contents.count), sha256: sha256(contents),
                               repositoryPath: "variants/\(name)/config.json")], variant: name)
    }
    let firstBytes = Data("first".utf8), secondBytes = Data("other".utf8)
    let first = variant("palette6", contents: firstBytes)
    let second = variant("linear8", contents: secondBytes)
    func serve(_ model: AsrModel, _ data: Data) throws {
        let url = root.appendingPathComponent("hub/\(model.repository)/resolve/\(model.revision)/\(model.files[0].repositoryPath)")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }
    try serve(first, firstBytes)
    // The unselected variant does not exist on the server yet. Downloading the first must still succeed.
    try await store.download(first)
    #expect(store.isDownloaded(first))
    #expect(!store.isDownloaded(second))
    #expect(!FileManager.default.fileExists(atPath: store.directory(for: second).path))
    try serve(second, secondBytes)
    try await store.download(second)
    let firstConfig = try Data(contentsOf: store.directory(for: first).appendingPathComponent("config.json"))
    let secondConfig = try Data(contentsOf: store.directory(for: second).appendingPathComponent("config.json"))
    #expect(firstConfig == firstBytes)
    #expect(secondConfig == secondBytes)
    try await store.remove(first)
    #expect(!store.isDownloaded(first))
    #expect(store.isDownloaded(second))
    #expect(try await store.verify(second))
}

final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Int64 = 0
    func add(_ bytes: Int64) { lock.withLock { value = bytes } }
    var last: Int64 { lock.withLock { value } }
}
