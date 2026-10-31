import BuilderHelmAudio
import SwiftUI

struct ContentView: View {
    @State private var demo = DemoModel()

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    Picker("Model", selection: $demo.model) {
                        Text("Parakeet CTC 110M").tag(AsrModel.parakeetCtc110M)
                        Text("Parakeet TDT + CTC 110M").tag(AsrModel.parakeetTdtCtc110M)
                    }
                    if let fraction = demo.downloadFraction {
                        ProgressView(value: fraction) { Text("Downloading \(Int(fraction * 100))%") }
                    }
                    HStack {
                        Text(demo.isDownloaded ? "Downloaded" : "\(demo.model.totalBytes / 1_000_000) MB download")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(demo.status).foregroundStyle(.secondary)
                    }
                    .font(.footnote)
                    Button("Download and Load") { Task { _ = await demo.load() } }
                    if demo.isDownloaded { Button("Delete Model", role: .destructive) { Task { await demo.deleteModel() } } }
                }

                Section("Transcribe") {
                    ForEach(DemoModel.Sample.allCases) { sample in
                        Button("Sample: \(sample.title)") { Task { await demo.transcribe(sample) } }
                    }
                    Button(demo.recording ? "Stop and Transcribe" : "Record from Microphone") {
                        Task { await demo.toggleRecording() }
                    }
                    .tint(demo.recording ? .red : .accentColor)
                    TextField("Keywords, comma separated", text: $demo.keywords)
                        .textInputAutocapitalization(.never)
                }

                Section("Benchmark") {
                    ForEach(DemoModel.Sample.allCases) { sample in
                        Button("Benchmark \(sample.title), 10 runs") { Task { await demo.benchmark(sample) } }
                    }
                }

                if !demo.stats.isEmpty || !demo.transcript.isEmpty {
                    Section("Result") {
                        Text(demo.stats).font(.footnote.monospaced()).textSelection(.enabled)
                        Text(demo.transcript).textSelection(.enabled)
                        ForEach(demo.hits, id: \.keyword) { hit in
                            Text(String(format: "%@  %.2f–%.2f s  score %.3f", hit.keyword, hit.start, hit.end, hit.score))
                                .font(.footnote.monospaced())
                        }
                    }
                }
            }
            .disabled(demo.busy && !demo.recording)
            .navigationTitle("BuilderHelm Audio")
        }
        .task {
            // xcrun simctl launch --console <device> ai.builderhelm.audio.demo --autobenchmark
            guard ProcessInfo.processInfo.arguments.contains("--autobenchmark") else { return }
            for model in AsrModel.all {
                demo.model = model
                for sample in DemoModel.Sample.allCases {
                    await demo.benchmark(sample)
                    print("AUTOBENCHMARK \(demo.stats.replacingOccurrences(of: "\n", with: " | "))")
                    print("AUTOTEXT \(model.id)/\(sample.rawValue) \(demo.transcript)")
                }
            }
            print("AUTOBENCHMARK done")
        }
    }
}
