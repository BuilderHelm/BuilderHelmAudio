import BuilderHelmAudio
import Foundation

let usage = """
usage: bhaudio <command> [options]

  models                                   List models and whether they are downloaded
  download <model>                         Download and verify a model
  transcribe <audio> [--model M] [--ctc] [--compute ane|gpu|cpu] [--json]
  spot <audio> <keyword>... [--model M] [--threshold T]
  benchmark <cases.json> [--model M] [--ctc] [--compute ane|gpu|cpu] [--rounds N] [--out results.json]

Models: \(AsrModel.all.map(\.id).joined(separator: ", ")) (default \(AsrModel.parakeetTdtCtc110M.id))
benchmark cases: a JSON array of {"path": "...", "text": "reference (optional)"}
"""

struct UsageError: Error {}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

var arguments = Array(CommandLine.arguments.dropFirst())
@MainActor func option(_ name: String) -> String? {
    guard let i = arguments.firstIndex(of: name) else { return nil }
    guard i + 1 < arguments.count else { fail("\(name) needs a value") }
    let value = arguments[i + 1]
    arguments.removeSubrange(i...(i + 1))
    return value
}
@MainActor func flag(_ name: String) -> Bool {
    guard let i = arguments.firstIndex(of: name) else { return false }
    arguments.remove(at: i)
    return true
}

@MainActor func model() -> AsrModel {
    let id = option("--model") ?? AsrModel.parakeetTdtCtc110M.id
    guard let model = AsrModel.named(id) else { fail("Unknown model \(id). \(usage)") }
    return model
}

func progressPrinter(_ model: AsrModel) -> @Sendable (DownloadProgress) -> Void {
    let start = Date()
    return { progress in
        let mb = Double(progress.completedBytes) / 1e6, total = Double(progress.totalBytes) / 1e6
        let line = String(format: "\r%@ %5.1f / %.1f MB (%3.0f%%)", model.id, mb, total, 100 * progress.fraction)
        FileHandle.standardError.write(Data(line.utf8))
        if progress.completedBytes == progress.totalBytes && progress.currentFile.isEmpty {
            FileHandle.standardError.write(Data(String(format: " in %.1f s\n", Date().timeIntervalSince(start)).utf8))
        }
    }
}

@MainActor func computeUnits() -> ComputeUnits {
    switch option("--compute") ?? "ane" {
    case "ane": .neuralEngine
    case "gpu": .gpu
    case "cpu": .cpu
    case let other: fail("Unknown compute units \(other); use ane, gpu or cpu")
    }
}

func load(_ model: AsrModel, ctc: Bool, compute: ComputeUnits = .neuralEngine) async throws -> SpeechRecognizer {
    let started = Date()
    let recognizer = try await SpeechRecognizer.load(model, mode: ctc ? .ctc : .tdt, computeUnits: compute,
                                                     progress: progressPrinter(model))
    FileHandle.standardError.write(Data(String(format: "Loaded %@ (%@ mode) in %.2f s\n", model.description,
                                               recognizer.mode == .ctc ? "CTC" : "TDT", Date().timeIntervalSince(started)).utf8))
    return recognizer
}

func json(_ value: Any) -> String {
    String(decoding: try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .prettyPrinted]), as: UTF8.self)
}

/// Word error rate after lowercasing and removing punctuation. Simpler than Whisper's English normalizer.
func wordErrorRate(references: [String], hypotheses: [String]) -> (errors: Int, words: Int) {
    func words(_ text: String) -> [String] {
        String(text.lowercased().map { $0.isLetter || $0.isNumber || $0 == "'" ? $0 : " " }).split(separator: " ").map(String.init)
    }
    var errors = 0, total = 0
    for (reference, hypothesis) in zip(references, hypotheses) {
        let r = words(reference), h = words(hypothesis)
        var row = Array(0...h.count)                       // edit distance against the empty reference prefix
        for word in r {
            var next = [row[0] + 1]
            for j in h.indices { next.append(min(row[j + 1] + 1, next[j] + 1, row[j] + (word == h[j] ? 0 : 1))) }
            row = next
        }
        errors += row[h.count]
        total += r.count
    }
    return (errors, total)
}

guard let command = arguments.first else { fail(usage) }
arguments.removeFirst()

do {
    switch command {
    case "models":
        for model in AsrModel.all {
            let state = ModelStore.shared.isDownloaded(model) ? "downloaded" : "not downloaded"
            print(String(format: "%-24@ %@  %.0f MB  %@", model.id as NSString, model.description, Double(model.totalBytes) / 1e6, state))
        }
    case "download":
        guard let id = arguments.first, let model = AsrModel.named(id) else { fail(usage) }
        let directory = try await ModelStore.shared.download(model, progress: progressPrinter(model))
        print(directory.path)
    case "transcribe":
        let model = model(), ctc = flag("--ctc"), asJSON = flag("--json"), compute = computeUnits()
        guard let path = arguments.first else { fail(usage) }
        let recognizer = try await load(model, ctc: ctc, compute: compute)
        let transcript = try await recognizer.transcribe(file: URL(fileURLWithPath: path))
        if asJSON {
            print(json(["text": transcript.text, "ctc_text": transcript.ctcText, "duration": transcript.duration,
                        "processing_time": transcript.processingTime,
                        "tokens": transcript.tokens.map { ["text": $0.text, "time": $0.time] }]))
        } else {
            print(transcript.text)
            FileHandle.standardError.write(Data(String(format: "%.1f s audio in %.3f s (%.0f× real time)\n",
                transcript.duration, transcript.processingTime, transcript.realTimeFactor).utf8))
        }
    case "spot":
        let model = model()
        let threshold = option("--threshold").map { Float($0) ?? KeywordSpotter.defaultThreshold } ?? KeywordSpotter.defaultThreshold
        guard arguments.count >= 2 else { fail(usage) }
        let recognizer = try await load(model, ctc: true)
        let started = Date()
        let hits = try await recognizer.spot(Array(arguments.dropFirst()), inFile: URL(fileURLWithPath: arguments[0]), threshold: threshold)
        for hit in hits { print(String(format: "%@\t%.3f\t%.2f–%.2f s", hit.keyword, hit.score, hit.start, hit.end)) }
        FileHandle.standardError.write(Data(String(format: "%d of %d found in %.3f s\n", hits.count, arguments.count - 1,
                                                   Date().timeIntervalSince(started)).utf8))
    case "benchmark":
        let model = model(), ctc = flag("--ctc"), compute = computeUnits()
        let rounds = Int(option("--rounds") ?? "3") ?? 3, out = option("--out")
        guard let path = arguments.first else { fail(usage) }
        struct Case: Decodable { let path: String; let text: String? }
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        let recognizer = try await load(model, ctc: ctc, compute: compute)
        for item in cases { _ = try await recognizer.transcribe(file: URL(fileURLWithPath: item.path)) }   // warm every window size
        var totals: [Double] = [], latencies: [Double] = [], texts: [String] = []
        var audio = 0.0
        for round in 1...rounds {
            var total = 0.0
            texts = []
            audio = 0
            for item in cases {
                let start = DispatchTime.now().uptimeNanoseconds
                let transcript = try await recognizer.transcribe(file: URL(fileURLWithPath: item.path))
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                total += elapsed
                if round == rounds { latencies.append(elapsed * 1000) }
                texts.append(transcript.text)
                audio += transcript.duration
            }
            totals.append(total)
            FileHandle.standardError.write(Data(String(format: "round %d: %.0f× real time\n", round, audio / total).utf8))
        }
        let median = totals.sorted()[totals.count / 2]
        latencies.sort()
        var summary: [String: Any] = [
            "model": model.description, "mode": recognizer.mode == .ctc ? "ctc" : "tdt", "files": cases.count,
            "audio_seconds": audio, "rounds": rounds, "rtfx": audio / median,
            "p50_ms": latencies[latencies.count / 2], "p95_ms": latencies[Int(0.95 * Double(latencies.count - 1))],
        ]
        let referenced = cases.indices.filter { cases[$0].text != nil }
        if !referenced.isEmpty {
            let score = wordErrorRate(references: referenced.map { cases[$0].text! }, hypotheses: referenced.map { texts[$0] })
            summary["wer_percent"] = 100 * Double(score.errors) / Double(max(score.words, 1))
            summary["wer_normalization"] = "lowercase, punctuation removed"
        }
        print(json(summary))
        if let out {
            summary["texts"] = zip(cases, texts).map { ["path": $0.path, "text": $1] }
            try json(summary).write(toFile: out, atomically: true, encoding: .utf8)
        }
    default:
        fail(usage)
    }
} catch {
    FileHandle.standardError.write(Data("bhaudio: \(error.localizedDescription)\n".utf8))
    exit(1)
}
