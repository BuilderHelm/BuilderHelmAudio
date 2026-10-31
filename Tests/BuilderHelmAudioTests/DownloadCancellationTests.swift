import Foundation
import Testing
@testable import BuilderHelmAudio

/// A request that stays in flight until URLSession cancels it; no network or timing sleeps.
private final class ControlledDownloadProtocol: URLProtocol, @unchecked Sendable {
    static let started = AsyncStream<Void>.makeStream()
    override class func canInit(with request: URLRequest) -> Bool { request.url?.scheme == "bhaudio-test" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        if request.url?.path == "/stall" {
            Self.started.continuation.yield(())
        } else {
            let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
                                           headerFields: ["Content-Length": "4"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data("done".utf8))
            client?.urlProtocolDidFinishLoading(self)
        }
    }
    override func stopLoading() {}
}

@Test(.timeLimit(.minutes(1))) func activeDownloadCancellationResumesAndDownloaderCanBeReused() async throws {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [ControlledDownloadProtocol.self]
    let downloader = FileDownloader(configuration: configuration)
    let task = Task { try await downloader.download(URL(string: "bhaudio-test://download/stall")!) { _ in } }
    var iterator = ControlledDownloadProtocol.started.stream.makeAsyncIterator()
    _ = await iterator.next()
    task.cancel()
    await #expect(throws: CancellationError.self) { try await task.value }
    let (file, _) = try await downloader.download(URL(string: "bhaudio-test://download/success")!) { _ in }
    defer { try? FileManager.default.removeItem(at: file) }
    #expect(try Data(contentsOf: file) == Data("done".utf8))
}
