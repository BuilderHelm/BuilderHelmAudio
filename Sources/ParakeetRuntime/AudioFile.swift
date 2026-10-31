import AVFoundation
import Foundation

public enum ParakeetAudio {
    /// Decode an AVFoundation-supported file to 16 kHz mono Float PCM.
    /// Reads the source in blocks; the returned waveform remains in memory.
    public static func read(_ url: URL) throws -> [Float] {
        try autoreleasepool {
            let reader = try Reader(url)
            var samples: [Float] = []
            samples.reserveCapacity(reader.estimatedSampleCount + 4096)
            while true {
                let block = try reader.read(16384)
                if block.isEmpty { return samples }
                samples.append(contentsOf: block)
            }
        }
    }

    /// Fixed conversion blocks ensure file streaming and `read` use the same resampler sequence.
    final class Reader {
        private let file: AVAudioFile
        private let converter: AVAudioConverter
        private let input: AVAudioPCMBuffer, output: AVAudioPCMBuffer
        private var pending: [Float] = [], offset = 0, ended = false
        let estimatedSampleCount: Int

        init(_ url: URL) throws {
            let file = try AVAudioFile(forReading: url)
            guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                         channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: file.processingFormat, to: format),
              let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 4096),
              let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4096)
            else { throw ParakeetError.invalidAudio("Cannot create audio converter") }
            let expected = ceil(Double(file.length) * 16000 / file.processingFormat.sampleRate)
            guard expected.isFinite, expected >= 0, expected < Double(Int.max - 4096)
            else { throw ParakeetError.invalidAudio("Audio length is out of range") }
            estimatedSampleCount = Int(expected)
            self.file = file; self.converter = converter; self.input = input; self.output = output
        }

        func read(_ maximumSamples: Int) throws -> [Float] {
            var samples: [Float] = []
            samples.reserveCapacity(maximumSamples)
            while samples.count < maximumSamples {
                if offset == pending.count {
                    if ended { break }
                    pending = try autoreleasepool { try nextBlock() }; offset = 0
                    if pending.isEmpty { continue }
                }
                let count = min(maximumSamples - samples.count, pending.count - offset)
                samples.append(contentsOf: pending[offset..<offset + count]); offset += count
            }
            return samples
        }

        private func nextBlock() throws -> [Float] {
            output.frameLength = 0
            var readError: Error?
            var error: NSError?
            let status = converter.convert(to: output, error: &error) { [self] requested, state in
                guard self.file.framePosition < self.file.length else {
                    state.pointee = .endOfStream
                    return nil
                }
                do {
                    try self.file.read(into: self.input, frameCount: min(requested, self.input.frameCapacity))
                    state.pointee = self.input.frameLength == 0 ? .endOfStream : .haveData
                    return self.input.frameLength == 0 ? nil : self.input
                } catch {
                    readError = error
                    state.pointee = .endOfStream
                    return nil
                }
            }
            if let readError { throw readError }
            if let error { throw error }
            ended = status == .endOfStream
            if status == .error { throw ParakeetError.invalidAudio("Audio conversion failed") }
            if let channel = output.floatChannelData?[0], output.frameLength > 0 {
                let block = Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)))
                guard block.allSatisfy(\.isFinite) else { throw ParakeetError.invalidAudio("PCM samples must be finite") }
                return block
            }
            return []
        }
    }
}
