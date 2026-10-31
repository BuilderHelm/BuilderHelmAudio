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

final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Int64 = 0
    func add(_ bytes: Int64) { lock.withLock { value = bytes } }
    var last: Int64 { lock.withLock { value } }
}
