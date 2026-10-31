// TDT prediction network and joint on the CPU with Accelerate, from TdtDecoder.bin.
//
// Same maths as the DecoderJoint and JointBlock Core ML models:
//   (h, c) = LSTM(embed(token), (h, c));  pred = joint.pred(h)
//   logits(t) = joint.out(relu(encoder_proj[t] + pred))
// A Core ML call costs about 0.4 ms; one token here costs about 20 µs.
import Accelerate
import CryptoKit
import Foundation

final class TdtDecoder {
    static let hidden = 640, gates = 4 * 640
    /// Tensor names and shapes in file order; must match config.json `tdt_decoder.tensors`.
    static func layout(tokens: Int, outputs: Int) -> [(String, [Int])] {
        [("embed", [tokens, hidden]), ("lstm_w_ih", [gates, hidden]), ("lstm_w_hh", [gates, hidden]),
         ("lstm_b_ih", [gates]), ("lstm_b_hh", [gates]), ("joint_pred_w", [hidden, hidden]),
         ("joint_pred_b", [hidden]), ("joint_out_w", [outputs, hidden]), ("joint_out_b", [outputs])]
    }

    let tokens: Int, outputs: Int
    /// Input half of the LSTM for every token: W_ih · embed[token] + b_ih + b_hh.
    private let inputGates: [Float]       // [tokens, gates]
    private let recurrent: [Float]        // [gates, hidden]
    private let predWeight: [Float], predBias: [Float]
    private let outWeight: [Float], outBias: [Float]
    /// State and scratch in one allocation, reused for every prediction step.
    private let buffer: UnsafeMutablePointer<Float>
    private let h, c, pred, scratch, joint, gateBuffer, logits: UnsafeMutablePointer<Float>

    init(url: URL, tokens: Int, outputs: Int, expectedSHA256: String? = nil) throws {
        let layout = Self.layout(tokens: tokens, outputs: outputs)
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let counts = layout.map { $0.1.reduce(1, *) }
        guard data.count == counts.reduce(0, +) * MemoryLayout<Float>.size else {
            throw ParakeetError.badConfig("TdtDecoder.bin has \(data.count) bytes; expected \(counts.reduce(0, +) * 4)")
        }
        if let expectedSHA256 {
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard digest == expectedSHA256 else { throw ParakeetError.badConfig("TdtDecoder.bin SHA-256 mismatch") }
        }
        var tensors: [String: [Float]] = [:]
        data.withUnsafeBytes { raw in   // little-endian float32, as on every Apple Silicon Mac
            var offset = 0
            for ((name, _), count) in zip(layout, counts) {
                tensors[name] = [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
                    UnsafeMutableRawBufferPointer(buffer).copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[offset..<offset + count * 4]))
                    initialized = count
                }
                offset += count * 4
            }
        }
        let n = Self.hidden, g = Self.gates
        // inputGates = embed · W_ihᵀ, computed as (W_ih · embedᵀ)ᵀ since vDSP_mmul has no transpose flag.
        var embedT = [Float](repeating: 0, count: tokens * n)
        vDSP_mtrans(tensors["embed"]!, 1, &embedT, 1, vDSP_Length(n), vDSP_Length(tokens))
        var gatesByToken = [Float](repeating: 0, count: g * tokens)
        vDSP_mmul(tensors["lstm_w_ih"]!, 1, embedT, 1, &gatesByToken, 1, vDSP_Length(g), vDSP_Length(tokens), vDSP_Length(n))
        var table = [Float](repeating: 0, count: tokens * g)
        vDSP_mtrans(gatesByToken, 1, &table, 1, vDSP_Length(tokens), vDSP_Length(g))
        var bias = [Float](repeating: 0, count: g)
        vDSP_vadd(tensors["lstm_b_ih"]!, 1, tensors["lstm_b_hh"]!, 1, &bias, 1, vDSP_Length(g))
        table.withUnsafeMutableBufferPointer { p in
            for k in 0..<tokens { vDSP_vadd(p.baseAddress! + k * g, 1, bias, 1, p.baseAddress! + k * g, 1, vDSP_Length(g)) }
        }
        self.tokens = tokens
        self.outputs = outputs
        inputGates = table
        recurrent = tensors["lstm_w_hh"]!
        predWeight = tensors["joint_pred_w"]!
        predBias = tensors["joint_pred_b"]!
        outWeight = tensors["joint_out_w"]!
        outBias = tensors["joint_out_b"]!
        let size = 5 * n + g + outputs
        buffer = .allocate(capacity: size)
        buffer.initialize(repeating: 0, count: size)
        h = buffer; c = h + n; pred = c + n; scratch = pred + n; joint = scratch + n
        gateBuffer = joint + n; logits = gateBuffer + g
    }

    deinit { buffer.deallocate() }

    /// Zero LSTM state, as at the start of each window.
    func reset() {
        vDSP_vclr(h, 1, vDSP_Length(Self.hidden))
        vDSP_vclr(c, 1, vDSP_Length(Self.hidden))
    }

    /// Feeds one token (the blank id is the start symbol) and updates `pred`.
    func step(_ token: Int) {
        let n = Self.hidden, g = Self.gates
        let gates = gateBuffer, h = h, c = c
        vDSP_mmul(recurrent, 1, h, 1, gates, 1, vDSP_Length(g), 1, vDSP_Length(n))
        inputGates.withUnsafeBufferPointer { vDSP_vadd(gates, 1, $0.baseAddress! + token * g, 1, gates, 1, vDSP_Length(g)) }
        // PyTorch gate order: input, forget, cell, output.
        var count = Int32(n), one: Float = 1
        let i = gates, f = gates + n, cell = gates + 2 * n, o = gates + 3 * n
        vvtanhf(cell, cell, &count)
        vDSP_vneg(i, 1, i, 1, vDSP_Length(2 * n))             // sigmoid(x) = 1 / (1 + exp(-x))
        vDSP_vneg(o, 1, o, 1, vDSP_Length(n))
        vvexpf(i, i, &count); vvexpf(f, f, &count); vvexpf(o, o, &count)
        for gate in [i, f, o] {
            vDSP_vsadd(gate, 1, &one, gate, 1, vDSP_Length(n))
            vDSP_svdiv(&one, gate, 1, gate, 1, vDSP_Length(n))
        }
        vDSP_vmma(f, 1, c, 1, i, 1, cell, 1, c, 1, vDSP_Length(n))   // c = f * c + i * cell
        vvtanhf(h, c, &count)
        vDSP_vmul(o, 1, h, 1, h, 1, vDSP_Length(n))                  // h = o * tanh(c)
        vDSP_mmul(predWeight, 1, h, 1, pred, 1, vDSP_Length(n), 1, vDSP_Length(n))
        vDSP_vadd(pred, 1, predBias, 1, pred, 1, vDSP_Length(n))
    }

    /// Best token and best duration index at one encoder frame (640 floats).
    func decide(_ frame: UnsafePointer<Float>) -> (token: Int, duration: Int) {
        let n = Self.hidden
        var zero: Float = 0
        let logits = logits
        vDSP_vadd(frame, 1, pred, 1, scratch, 1, vDSP_Length(n))
        vDSP_vthr(scratch, 1, &zero, joint, 1, vDSP_Length(n))   // relu
        vDSP_mmul(outWeight, 1, joint, 1, logits, 1, vDSP_Length(outputs), 1, vDSP_Length(n))
        vDSP_vadd(logits, 1, outBias, 1, logits, 1, vDSP_Length(outputs))
        // First maximum wins, as torch.argmax.
        var token = 0, duration = tokens
        for k in 1..<tokens where logits[k] > logits[token] { token = k }
        for k in tokens + 1..<outputs where logits[k] > logits[duration] { duration = k }
        return (token, duration - tokens)
    }
}
