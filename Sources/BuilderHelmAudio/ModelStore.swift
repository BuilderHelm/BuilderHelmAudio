import CryptoKit
import Foundation

/// Download progress for one model, across all of its files.
public struct DownloadProgress: Sendable {
    public let completedBytes: Int64
    public let totalBytes: Int64
    /// Repository path of the file being downloaded or verified.
    public let currentFile: String
    public var fraction: Double { totalBytes > 0 ? Double(completedBytes) / Double(totalBytes) : 1 }
}

public enum BuilderHelmAudioError: Error, LocalizedError, Equatable {
    case downloadFailed(file: String, status: Int)
    case checksumMismatch(file: String)
    case unsupportedAudio(String)
    case modelLoadFailed(String)
    case unsupportedArchitecture(String)
    case unsupportedOperation(String)
    case invalidModelConfiguration(String)
    case invalidCtcConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case let .downloadFailed(file, status): "Download of \(file) failed (HTTP \(status))"
        case let .checksumMismatch(file): "\(file) does not match its pinned SHA-256; the download was discarded"
        case let .unsupportedAudio(reason): "Unsupported audio: \(reason)"
        case let .modelLoadFailed(reason): "Model failed to load: \(reason)"
        case let .unsupportedArchitecture(architecture): "No runtime is available for architecture \(architecture)"
        case let .unsupportedOperation(operation): "This model does not support \(operation)"
        case let .invalidModelConfiguration(reason): "Invalid model configuration: \(reason)"
        case let .invalidCtcConfiguration(reason): "Invalid CTC configuration: \(reason)"
        }
    }
}

/// Downloads pinned models from Hugging Face, checks every file's SHA-256 and keeps them on device.
///
/// Files live under `root/Models/<repository>/<revision>/`. A model is only marked ready after every
/// file has been verified, so an interrupted download resumes file by file on the next call.
public actor ModelStore {
    public static let shared = ModelStore()

    /// `Application Support/BuilderHelmAudio`. On iOS the folder is excluded from iCloud backup.
    public static var defaultRoot: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("BuilderHelmAudio", isDirectory: true)
    }

    public let root: URL
    private let baseURL: URL
    private let downloader = FileDownloader()
    private static let marker = ".builderhelm-verified"

    /// - Parameters:
    ///   - root: where models and compiled CoreML caches are kept.
    ///   - baseURL: Hugging Face host, or a mirror with the same `<repo>/resolve/<revision>/<path>` layout.
    public init(root: URL = ModelStore.defaultRoot, baseURL: URL = URL(string: "https://huggingface.co")!) {
        self.root = root
        self.baseURL = baseURL
    }

    /// Folder holding the model's files, whether or not they have been downloaded yet.
    public nonisolated func directory(for model: AsrModel) -> URL {
        root.appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(model.repository, isDirectory: true)
            .appendingPathComponent(model.revision, isDirectory: true)
    }

    /// Where CoreML keeps compiled models for this store.
    public nonisolated var compiledModelsDirectory: URL { root.appendingPathComponent("CompiledModels", isDirectory: true) }

    /// True when every file was verified and still has its pinned size.
    public nonisolated func isDownloaded(_ model: AsrModel) -> Bool {
        let directory = directory(for: model)
        guard let data = try? Data(contentsOf: directory.appendingPathComponent(Self.marker)),
              String(decoding: data, as: UTF8.self) == model.revision else { return false }
        return model.files.allSatisfy { size(of: directory.appendingPathComponent($0.path)) == $0.size }
    }

    /// Downloads and verifies the model if needed, and returns its folder.
    @discardableResult
    public func download(_ model: AsrModel, progress: (@Sendable (DownloadProgress) -> Void)? = nil) async throws -> URL {
        try Task.checkCancellation()
        let directory = directory(for: model)
        if isDownloaded(model) { return directory }
        try prepareRoot()
        try? FileManager.default.removeItem(at: directory.appendingPathComponent(Self.marker))
        var completed: Int64 = 0
        for file in model.files {
            try Task.checkCancellation()
            let target = directory.appendingPathComponent(file.path)
            let done = completed
            progress?(DownloadProgress(completedBytes: done, totalBytes: model.totalBytes, currentFile: file.path))
            if try size(of: target) != file.size || Self.sha256(of: target) != file.sha256 {
                try await fetch(file, of: model, to: target) { written in
                    progress?(DownloadProgress(completedBytes: done + written, totalBytes: model.totalBytes, currentFile: file.path))
                }
            }
            completed += file.size
        }
        try Task.checkCancellation()
        try Data(model.revision.utf8).write(to: directory.appendingPathComponent(Self.marker), options: .atomic)
        progress?(DownloadProgress(completedBytes: completed, totalBytes: model.totalBytes, currentFile: ""))
        return directory
    }

    /// Re-hashes every downloaded file. Returns false and clears the ready mark if any file changed.
    public func verify(_ model: AsrModel) throws -> Bool {
        try Task.checkCancellation()
        let directory = directory(for: model)
        for file in model.files {
            try Task.checkCancellation()
            let url = directory.appendingPathComponent(file.path)
            guard size(of: url) == file.size, try Self.sha256(of: url) == file.sha256 else {
                try? FileManager.default.removeItem(at: directory.appendingPathComponent(Self.marker))
                return false
            }
        }
        return true
    }

    /// Deletes the model's files. Compiled caches are keyed by content and are left in place.
    public func remove(_ model: AsrModel) throws {
        let directory = directory(for: model)
        if FileManager.default.fileExists(atPath: directory.path) { try FileManager.default.removeItem(at: directory) }
    }

    private func fetch(_ file: AsrModel.File, of model: AsrModel, to target: URL,
                       written: @escaping @Sendable (Int64) -> Void) async throws {
        let url = baseURL.appendingPathComponent(model.repository).appendingPathComponent("resolve")
            .appendingPathComponent(model.revision).appendingPathComponent(file.path)
        var lastError: Error = BuilderHelmAudioError.downloadFailed(file: file.path, status: 0)
        for attempt in 0..<3 {
            try Task.checkCancellation()
            if attempt > 0 { try await Task.sleep(for: .seconds(attempt * 2)) }
            do {
                let (temporary, response) = try await downloader.download(url, progress: written)
                defer { try? FileManager.default.removeItem(at: temporary) }
                try Task.checkCancellation()
                if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                    lastError = BuilderHelmAudioError.downloadFailed(file: file.path, status: http.statusCode)
                    if (400..<500).contains(http.statusCode) { break }
                    continue
                }
                guard size(of: temporary) == file.size, try Self.sha256(of: temporary) == file.sha256 else {
                    throw BuilderHelmAudioError.checksumMismatch(file: file.path)
                }
                try Task.checkCancellation()
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                if FileManager.default.fileExists(atPath: target.path) { try FileManager.default.removeItem(at: target) }
                try FileManager.default.moveItem(at: temporary, to: target)
                return
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as BuilderHelmAudioError {
                throw error
            } catch {
                lastError = error
            }
        }
        throw lastError
    }

    private func prepareRoot() throws {
        var models = root.appendingPathComponent("Models", isDirectory: true)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? models.setResourceValues(values)
    }

    private nonisolated func size(of url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
    }

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// URLSession download tasks with byte progress. Task-specific delegates on the async API
/// do not receive progress, so one session-level delegate routes callbacks by task.
final class FileDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    /// Covers cancellation racing with task creation; tasks are registered before being attached here.
    private final class Cancellation: @unchecked Sendable {
        private let lock = NSLock()
        private var task: URLSessionDownloadTask?
        private var cancelled = false
        func attach(_ task: URLSessionDownloadTask) {
            let shouldCancel = lock.withLock { self.task = task; return cancelled }
            if shouldCancel { task.cancel() }
        }
        func cancel() {
            let task = lock.withLock { cancelled = true; return self.task }
            task?.cancel()
        }
    }
    private struct Pending {
        let continuation: CheckedContinuation<(URL, URLResponse), Error>
        let progress: @Sendable (Int64) -> Void
        var file: URL?
    }

    private let lock = NSLock()
    private var pending: [Int: Pending] = [:]
    private let configuration: URLSessionConfiguration
    private lazy var session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)

    init(configuration: URLSessionConfiguration = .default) {
        self.configuration = configuration
        super.init()
    }

    /// Downloads `url` to a temporary file the caller must move or delete.
    func download(_ url: URL, progress: @escaping @Sendable (Int64) -> Void) async throws -> (URL, URLResponse) {
        try Task.checkCancellation()
        let cancellation = Cancellation()
        do {
            let result: (URL, URLResponse) = try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    let task = lock.withLock { session.downloadTask(with: url) }
                    lock.withLock { pending[task.taskIdentifier] = Pending(continuation: continuation, progress: progress) }
                    cancellation.attach(task)
                    task.resume()
                }
            } onCancel: { cancellation.cancel() }
            if Task.isCancelled {
                try? FileManager.default.removeItem(at: result.0)
                throw CancellationError()
            }
            return result
        } catch {
            if Task.isCancelled { throw CancellationError() }
            throw error
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        lock.withLock { pending[downloadTask.taskIdentifier] }?.progress(totalBytesWritten)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // The system deletes `location` when this returns, so keep it under our own name.
        let kept = FileManager.default.temporaryDirectory.appendingPathComponent("BuilderHelmAudio-\(UUID().uuidString)")
        let moved = (try? FileManager.default.moveItem(at: location, to: kept)) != nil
        lock.withLock { pending[downloadTask.taskIdentifier]?.file = moved ? kept : nil }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let entry = lock.withLock({ pending.removeValue(forKey: task.taskIdentifier) }) else { return }
        if let error {
            if let file = entry.file { try? FileManager.default.removeItem(at: file) }
            entry.continuation.resume(throwing: error)
        } else if let file = entry.file, let response = task.response {
            entry.continuation.resume(returning: (file, response))
        } else {
            entry.continuation.resume(throwing: URLError(.cannotCreateFile))
        }
    }
}
