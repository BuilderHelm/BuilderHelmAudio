// Parakeet Unified's RNN-T prediction network and joint on the CPU with Accelerate, from RnntDecoder.bin.
//
// Same maths as the Predictor and Joint Core ML models, in the checkpoint's FP32:
//   (h, c) = LSTM₂(embed(token), (h, c));  pred = joint.pred(h₁)
//   logits(t) = joint.out(relu(joint.enc(encoder[t]) + pred))
// joint.enc is applied to every frame of a window in one matrix product, so each greedy step costs
// one 640 × 1025 matrix-vector product, and each emitted token one two-layer LSTM step.
import Accelerate
import Foundation

final class RnntDecoder {
    static let hidden = 640, gates = 4 * 640, encoderWidth = 1024
    /// Tensor names and shapes in file order; must match config.json `cpu_decoder.tensors`.
    static func layout(tokens: Int) -> [(String, [Int])] {
        [("embed", [tokens, hidden])]
            + [0, 1].flatMap { layer in [("lstm\(layer)_w_ih", [gates, hidden]), ("lstm\(layer)_w_hh", [gates, hidden]),
                                         ("lstm\(layer)_b_ih", [gates]), ("lstm\(layer)_b_hh", [gates])] }
            + [("joint_enc_w", [hidden, encoderWidth]), ("joint_enc_b", [hidden]),
               ("joint_pred_w", [hidden, hidden]), ("joint_pred_b", [hidden]),
               ("joint_out_w", [tokens, hidden]), ("joint_out_b", [tokens])]
    }

    /// Tokens including blank (the last id), which is also the joint's output width.
    let tokens: Int
    /// Layer 0's input half for every token: W_ih0 · embed[token] + b_ih0 + b_hh0.
    private let inputGates: [Float]       // [tokens, gates]
    private let recurrent0: [Float], input1: [Float], recurrent1: [Float], bias1: [Float]
    private let encWeight: [Float], encBias: [Float]
    private let predWeight: [Float], predBias: [Float]
    private let outWeight: [Float], outBias: [Float]
    /// State and scratch in one allocation, reused for every step.
    private let buffer: UnsafeMutablePointer<Float>
    private let h0, c0, h1, c1, pred, joint, gateBuffer, recurrentBuffer, logits: UnsafeMutablePointer<Float>
    /// joint.enc of every frame of the current window, [hidden, frameCapacity] row-major.
    private var projected: [Float] = []
    private var frameStride = 0

    init(url: URL, tokens: Int) throws {
        let layout = Self.layout(tokens: tokens)
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let counts = layout.map { $0.1.reduce(1, *) }
        guard data.count == counts.reduce(0, +) * MemoryLayout<Float>.size else {
            throw ParakeetError.badConfig("RnntDecoder.bin has \(data.count) bytes; expected \(counts.reduce(0, +) * 4)")
        }
        var tensors: [String: [Float]] = [:]
        data.withUnsafeBytes { raw in   // little-endian float32, as on every Apple Silicon device
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
        // inputGates = embed · W_ih0ᵀ, computed as (W_ih0 · embedᵀ)ᵀ since vDSP_mmul has no transpose flag.
        var embedT = [Float](repeating: 0, count: tokens * n)
        vDSP_mtrans(tensors["embed"]!, 1, &embedT, 1, vDSP_Length(n), vDSP_Length(tokens))
        var gatesByToken = [Float](repeating: 0, count: g * tokens)
        vDSP_mmul(tensors["lstm0_w_ih"]!, 1, embedT, 1, &gatesByToken, 1, vDSP_Length(g), vDSP_Length(tokens), vDSP_Length(n))
        var table = [Float](repeating: 0, count: tokens * g)
        vDSP_mtrans(gatesByToken, 1, &table, 1, vDSP_Length(tokens), vDSP_Length(g))
        var bias0 = [Float](repeating: 0, count: g), bias1 = [Float](repeating: 0, count: g)
        vDSP_vadd(tensors["lstm0_b_ih"]!, 1, tensors["lstm0_b_hh"]!, 1, &bias0, 1, vDSP_Length(g))
        vDSP_vadd(tensors["lstm1_b_ih"]!, 1, tensors["lstm1_b_hh"]!, 1, &bias1, 1, vDSP_Length(g))
        table.withUnsafeMutableBufferPointer { p in
            for k in 0..<tokens { vDSP_vadd(p.baseAddress! + k * g, 1, bias0, 1, p.baseAddress! + k * g, 1, vDSP_Length(g)) }
        }
        self.tokens = tokens
        inputGates = table
        recurrent0 = tensors["lstm0_w_hh"]!
        input1 = tensors["lstm1_w_ih"]!
        recurrent1 = tensors["lstm1_w_hh"]!
        self.bias1 = bias1
        encWeight = tensors["joint_enc_w"]!
        encBias = tensors["joint_enc_b"]!
        predWeight = tensors["joint_pred_w"]!
        predBias = tensors["joint_pred_b"]!
        outWeight = tensors["joint_out_w"]!
        outBias = tensors["joint_out_b"]!
        let size = 6 * n + 2 * g + tokens
        buffer = .allocate(capacity: size)
        buffer.initialize(repeating: 0, count: size)
        h0 = buffer; c0 = h0 + n; h1 = c0 + n; c1 = h1 + n; pred = c1 + n; joint = pred + n
        gateBuffer = joint + n; recurrentBuffer = gateBuffer + g; logits = recurrentBuffer + g
    }

    deinit { buffer.deallocate() }

    /// Zero LSTM state and feed the blank start symbol, as at the start of each window.
    func reset() {
        vDSP_vclr(buffer, 1, vDSP_Length(4 * Self.hidden))
        step(tokens - 1)
    }

    /// Applies joint.enc to the first `frames` frames of an encoder output [1, 1024, capacity]
    /// with contiguous frames (stride 1) and channel stride `capacity`.
    func project(_ features: UnsafePointer<Float>, frames: Int, capacity: Int) {
        let n = Self.hidden
        if projected.count != n * capacity { projected = [Float](repeating: 0, count: n * capacity) }
        frameStride = capacity
        projected.withUnsafeMutableBufferPointer { out in
            // [640 × 1024] · [1024 × capacity]; columns past `frames` are never read.
            vDSP_mmul(encWeight, 1, features, 1, out.baseAddress!, 1, vDSP_Length(n), vDSP_Length(capacity), vDSP_Length(Self.encoderWidth))
            for j in 0..<n {
                var b = encBias[j]
                vDSP_vsadd(out.baseAddress! + j * capacity, 1, &b, out.baseAddress! + j * capacity, 1, vDSP_Length(frames))
            }
        }
    }

    /// Feeds one token and updates `pred`.
    func step(_ token: Int) {
        let n = Self.hidden, g = Self.gates
        // Layer 0: input half from the token table.
        vDSP_mmul(recurrent0, 1, h0, 1, gateBuffer, 1, vDSP_Length(g), 1, vDSP_Length(n))
        inputGates.withUnsafeBufferPointer { vDSP_vadd(gateBuffer, 1, $0.baseAddress! + token * g, 1, gateBuffer, 1, vDSP_Length(g)) }
        Self.cell(gateBuffer, h: h0, c: c0)
        // Layer 1 reads layer 0's new output.
        vDSP_mmul(input1, 1, h0, 1, gateBuffer, 1, vDSP_Length(g), 1, vDSP_Length(n))
        vDSP_mmul(recurrent1, 1, h1, 1, recurrentBuffer, 1, vDSP_Length(g), 1, vDSP_Length(n))
        vDSP_vadd(gateBuffer, 1, recurrentBuffer, 1, gateBuffer, 1, vDSP_Length(g))
        vDSP_vadd(gateBuffer, 1, bias1, 1, gateBuffer, 1, vDSP_Length(g))
        Self.cell(gateBuffer, h: h1, c: c1)
        vDSP_mmul(predWeight, 1, h1, 1, pred, 1, vDSP_Length(n), 1, vDSP_Length(n))
        vDSP_vadd(pred, 1, predBias, 1, pred, 1, vDSP_Length(n))
    }

    /// Best token at one frame, and NeMo `max_prob` confidence for a non-blank token.
    func decide(frame t: Int) -> (token: Int, confidence: Float) {
        let n = Self.hidden
        var zero: Float = 0
        projected.withUnsafeBufferPointer { p in vDSP_vadd(p.baseAddress! + t, vDSP_Stride(frameStride), pred, 1, joint, 1, vDSP_Length(n)) }
        vDSP_vthr(joint, 1, &zero, joint, 1, vDSP_Length(n))   // relu
        vDSP_mmul(outWeight, 1, joint, 1, logits, 1, vDSP_Length(tokens), 1, vDSP_Length(n))
        vDSP_vadd(logits, 1, outBias, 1, logits, 1, vDSP_Length(tokens))
        // First maximum wins, as torch.argmax.
        var token = 0
        for k in 1..<tokens where logits[k] > logits[token] { token = k }
        guard token != tokens - 1 else { return (token, 0) }
        // (N·p − 1)/(N − 1) with p = softmax(logits)[token], as ParakeetTdtV3.maxProbConfidence.
        let maximum = logits[token]
        var denominator = 0.0
        for k in 0..<tokens { denominator += exp(Double(logits[k] - maximum)) }
        let p = 1 / denominator, count = Double(tokens)
        return (token, Float((count * p - 1) / (count - 1)))
    }

    /// One LSTM cell update from its gate pre-activations, in PyTorch order: input, forget, cell, output.
    private static func cell(_ values: UnsafeMutablePointer<Float>, h: UnsafeMutablePointer<Float>, c: UnsafeMutablePointer<Float>) {
        let n = hidden
        var count = Int32(n), one: Float = 1
        let i = values, f = values + n, cell = values + 2 * n, o = values + 3 * n
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
    }
}
