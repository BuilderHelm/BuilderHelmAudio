import AVFoundation
import BuilderHelmAudio
import Foundation
import Observation

@MainActor @Observable
final class DemoModel {
    enum Sample: String, CaseIterable, Identifiable {
        case short, long
        var id: String { rawValue }
        var title: String { self == .short ? "Short (6 s)" : "Long (72 s)" }
        var url: URL { Bundle.main.url(forResource: rawValue, withExtension: "wav", subdirectory: "Samples")! }
    }

    var model: AsrModel = .parakeetCtc110M { didSet { if oldValue != model { recognizer = nil; status = "" } } }
    var downloadFraction: Double?
    var status = ""
    var transcript = ""
    var stats = ""
    var keywords = "fox, Neural Engine, lighthouse, zebra"
    var hits: [KeywordHit] = []
    var busy = false
    var recording = false

    private(set) var recognizer: SpeechRecognizer?
    private var recorder: AVAudioRecorder?
    private let recordingURL = FileManager.default.temporaryDirectory.appendingPathComponent("recording.wav")
    private let references: [String: String] = {
        let url = Bundle.main.url(forResource: "mac-reference", withExtension: "json", subdirectory: "Samples")!
        return (try? JSONDecoder().decode([String: String].self, from: Data(contentsOf: url))) ?? [:]
    }()

    var isDownloaded: Bool { ModelStore.shared.isDownloaded(model) }
    var device: String {
        var info = utsname()
        uname(&info)
        let machine = withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        return "\(machine), iOS \(ProcessInfo.processInfo.operatingSystemVersionString)"
    }

    func load() async -> SpeechRecognizer? {
        if let recognizer { return recognizer }
        busy = true
        defer { busy = false; downloadFraction = nil }
        do {
            status = isDownloaded ? "Loading…" : "Downloading…"
            let started = Date()
            let recognizer = try await SpeechRecognizer.load(model) { progress in
                Task { @MainActor in self.downloadFraction = progress.fraction }
            }
            status = String(format: "Loaded in %.1f s", Date().timeIntervalSince(started))
            self.recognizer = recognizer
            return recognizer
        } catch {
            status = "Error: \(error.localizedDescription)"
            return nil
        }
    }

    func transcribe(_ sample: Sample) async {
        await transcribe(url: sample.url, reference: references["\(model.id)/\(sample.rawValue)"])
    }

    func transcribe(url: URL, reference: String? = nil) async {
        guard let recognizer = await load() else { return }
        busy = true
        defer { busy = false }
        do {
            let result = try await recognizer.transcribe(file: url, includeCtcLogProbabilities: recognizer.capabilities.contains(.ctc))
            transcript = result.text
            var line = String(format: "%.1f s audio in %.0f ms · %.0f× real time", result.duration,
                              result.processingTime * 1000, result.realTimeFactor)
            if let reference { line += reference == result.text ? " · matches Mac" : " · differs from Mac" }
            stats = line
            let phrases = keywords.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            hits = result.ctcLogProbabilities.map { recognizer.keywordSpotter.detect(phrases, in: $0) } ?? []
        } catch {
            stats = "Error: \(error.localizedDescription)"
        }
    }

    /// Transcribes the sample `rounds` times after one warm-up and reports median speed and latency.
    func benchmark(_ sample: Sample, rounds: Int = 10) async {
        guard let recognizer = await load() else { return }
        busy = true
        defer { busy = false }
        do {
            _ = try await recognizer.transcribe(file: sample.url)
            var times: [Double] = [], duration = 0.0, text = ""
            for round in 1...rounds {
                stats = "Benchmark \(round)/\(rounds)…"
                let result = try await recognizer.transcribe(file: sample.url)
                times.append(result.processingTime)
                duration = result.duration
                text = result.text
            }
            times.sort()
            let median = times[times.count / 2]
            let match = references["\(model.id)/\(sample.rawValue)"].map { $0 == text ? "matches Mac" : "differs from Mac" } ?? "no frozen reference"
            transcript = text
            stats = String(format: "%@ · %@\n%.1f s audio · median %.0f ms (best %.0f, worst %.0f) · %.0f× real time · %@",
                           model.id, device, duration, median * 1000, times.first! * 1000, times.last! * 1000, duration / median, match)
        } catch {
            stats = "Error: \(error.localizedDescription)"
        }
    }

    func toggleRecording() async {
        if recording {
            recorder?.stop()
            recording = false
            try? AVAudioSession.sharedInstance().setActive(false)
            await transcribe(url: recordingURL)
            return
        }
        guard await AVAudioApplication.requestRecordPermission() else { stats = "Microphone access denied"; return }
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement)
            try session.setActive(true)
            recorder = try AVAudioRecorder(url: recordingURL, settings: [
                AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 16000, AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false,
            ])
            recorder?.record()
            recording = true
            stats = "Recording… tap Stop to transcribe"
        } catch {
            stats = "Error: \(error.localizedDescription)"
        }
    }

    func deleteModel() async {
        recognizer = nil
        try? await ModelStore.shared.remove(model)
        status = "Deleted"
    }
}
