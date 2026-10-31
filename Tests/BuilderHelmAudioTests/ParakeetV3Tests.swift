import CoreML
import Darwin
import Foundation
import Testing
@testable import BuilderHelmAudio
@testable import ParakeetRuntime

#if os(iOS)
@Test func v3RejectsUnqualifiedGpuBeforeOpeningModelFiles() throws {
    let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    for units in [MLComputeUnits.cpuAndGPU, .all] {
        do {
            _ = try ParakeetTdtV3(directory: missing, computeUnits: units, cacheDirectory: missing)
            Issue.record("Unqualified V3 GPU mode was accepted")
        } catch ParakeetError.badConfig(let reason) {
            #expect(reason.contains("GPU inference is not qualified on iOS"))
        }
    }
    #expect(!FileManager.default.fileExists(atPath: missing.path))
}
#endif

@Test func v3SelectionHasTruthfulCapabilitiesAndRejectsCtc() async throws {
    let model = AsrModel.experimentalParakeetV3
    let selection = try RuntimeSelection.resolve(model: model, mode: .tdt)
    #expect(selection.architecture == .parakeetV3)
    #expect(selection.capabilities == [.tdt])
    #expect(model.maxRequestDuration == nil)
    #expect(AsrModel.parakeetTdtCtc110M.maxRequestDuration == nil)
    #expect(selection.ctcConfiguration == nil)
    #expect(Set(model.supportedLanguages).count == 25)
    #expect(!AsrModel.all.contains(model))
    #expect(throws: BuilderHelmAudioError.self) { try RuntimeSelection.resolve(model: model, mode: .ctc) }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    await #expect(throws: BuilderHelmAudioError.self) { try await SpeechRecognizer.load(model, store: ModelStore(root: root)) }
    let store = ModelStore(root: root)
    await #expect(throws: BuilderHelmAudioError.self) { try await store.download(model) }
    #expect(!store.isDownloaded(model))
    #expect(!FileManager.default.fileExists(atPath: root.path))
}

@Test func v3StateCopyOwnsValuesAndRespectsPaddedStrides() throws {
    let pointer = UnsafeMutablePointer<Float>.allocate(capacity: 18)
    pointer.initialize(repeating: .nan, count: 18)
    for (index, offset) in [0,2,4,9,11,13].enumerated() { pointer[offset] = Float(index + 1) }
    let source = try MLMultiArray(dataPointer: pointer, shape: [2,1,3], dataType: .float32,
                                  strides: [9,9,2], deallocator: { $0.assumingMemoryBound(to: Float.self).deallocate() })
    let target = try MLMultiArray(shape: [2,1,3], dataType: .float32)
    try ParakeetTdtV3.copy(source, into: target)
    #expect((0..<6).map { target[$0].intValue } == [1,2,3,4,5,6])
    pointer[0] = 100
    #expect(target[0].intValue == 1)
    pointer[2] = .nan
    #expect(throws: ParakeetError.self) { try ParakeetTdtV3.copy(source, into: target) }
}

@Test func v3ArgmaxUsesFirstTieAndChecksNonfiniteValues() throws {
    let a = try MLMultiArray(shape: [1,1,1,5], dataType: .float32)
    for (i,value) in [1,4,4,2,3].enumerated() { a[i] = NSNumber(value: value) }
    #expect(try ParakeetTdtV3.argmax(a, 0..<5) == 1)
    a[4] = NSNumber(value: Float.nan)
    #expect(throws: ParakeetError.self) { try ParakeetTdtV3.argmax(a, 0..<5) }
    #expect(throws: ParakeetError.self) { try ParakeetTdtV3.argmax(a, 0..<6) }
}

@Test func v3AllowsAnExplicitSingleWindowRequestBudget() throws {
    try ParakeetTdtV3.validateSampleCount(240000, allowLongAudio: false)
    #expect(throws: ParakeetError.self) { try ParakeetTdtV3.validateSampleCount(240001, allowLongAudio: false) }
    try ParakeetTdtV3.validateSampleCount(240001, allowLongAudio: true)
}

@Test func v3LocalDescriptorCannotVerifyOrRemoveSharedModelStorage() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let preserved = root.appendingPathComponent("Models/preserved.txt")
    try FileManager.default.createDirectory(at: preserved.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("existing model".utf8).write(to: preserved)
    let store = ModelStore(root: root)
    #expect(try await !store.verify(.experimentalParakeetV3))
    await #expect(throws: BuilderHelmAudioError.self) { try await store.remove(.experimentalParakeetV3) }
    #expect(try Data(contentsOf: preserved) == Data("existing model".utf8))
}

@Test func v3TokenizerPreservesSourceUnknownAndWhitespaceSemantics() {
    let pieces = ["<unk>", "▁the", "▁", ","]
    #expect(ParakeetTdtV3.detokenize([0], pieces: pieces, punctuation: [","]) == " ⁇ ")
    #expect(ParakeetTdtV3.detokenize([1,0], pieces: pieces, punctuation: [","]) == "the ⁇ ")
    #expect(ParakeetTdtV3.detokenize([0,1], pieces: pieces, punctuation: [","]) == " ⁇  the")
    #expect(ParakeetTdtV3.detokenize([2,2], pieces: pieces, punctuation: [","]) == " ")
    #expect(ParakeetTdtV3.detokenize([1,2,3], pieces: pieces, punctuation: [","]) == "the,")
}

@Test func v3PausePlannerCutsInAQuietRunWithoutDiscardingSamples() {
    var audio = [Float](repeating: 0.2, count: 240001)
    for i in 160000..<164800 { audio[i] = 0 }
    let plan = ParakeetTdtV3.planWindow(audio)
    #expect(plan.count == 240000)
    #expect((160000..<164800).contains(plan.advance))
    #expect(plan.advance % 1280 == 0)
    #expect(plan.right == plan.advance)
    let next = ParakeetTdtV3.planWindow(Array(audio.dropFirst(plan.advance)))
    #expect(plan.advance + next.count == audio.count)
}

@Test func v3PausePlannerKeepsOverlapForContinuousSpeechAndIgnoresTinyGaps() {
    var audio = [Float](repeating: 0.2, count: 240001)
    for i in 160000..<160800 { audio[i] = 0 }
    let plan = ParakeetTdtV3.planWindow(audio)
    #expect(plan.count == 240000)
    #expect(plan.advance == 176640)
    #expect(plan.right - plan.advance == 32000)
    let tail = ParakeetTdtV3.planWindow([Float](repeating: 0, count: 100))
    #expect(tail.count == 100)
}

@Test func v3PausePlannerIsGainRelativeAndHasBoundedForwardProgress() {
    for gain: Float in [0.01, 0.1, 1] {
        var audio = [Float](repeating: gain, count: 240001)
        for i in 172800..<177920 { audio[i] = gain * 0.001 }
        let plan = ParakeetTdtV3.planWindow(audio)
        #expect((128000...224000).contains(plan.advance))
        #expect((172800..<177920).contains(plan.advance))
        #expect(plan.advance % 1280 == 0)
    }
}

@Test func v3LongWindowPlannerUsesContextAndKeepsFrameAlignedOwnership() {
    var samples = [Float](repeating: 0.2, count: 480001)
    let continuous = ParakeetTdtV3.planWindow(samples, windowSamples: 480000)
    #expect(continuous.count == 480000)
    #expect(continuous.advance % 1280 == 0)
    #expect(continuous.right - continuous.advance == 32000)
    #expect(continuous.right < continuous.count)
    for i in 400000..<405120 { samples[i] = 0 }
    let paused = ParakeetTdtV3.planWindow(samples, windowSamples: 480000)
    #expect(paused.count == 480000)
    #expect((400000..<405120).contains(paused.advance))
    #expect(paused.advance % 1280 == 0)
    #expect(paused.right == paused.advance)
}

@Test func v3ConsensusJoinPreservesWordsWithShiftedEmissionFrames() {
    let a = ParakeetTdtV3.Result(text:"",tokenIDs:[1,2,3,4,5,6],emissionFrames:[90,97,101,103,105,108])
    let b = ParakeetTdtV3.Result(text:"",tokenIDs:[3,4,5,6,7],emissionFrames:[0,2,4,7,10])
    let cut = ParakeetTdtV3.join(a,previousStart:0,previousFirst:0,next:b,nextStart:100*1280,seam:100*1280)
    let merged = Array(a.tokenIDs[..<cut.previousEnd])+Array(b.tokenIDs[cut.nextBegin...])
    #expect(merged == [1,2,3,4,5,6,7])
}

@Test func v3ConsensusJoinRejectsMatchingTextAtDifferentAudioTimes() {
    let a = ParakeetTdtV3.Result(text:"",tokenIDs:[1,2,3],emissionFrames:[10,11,12])
    let b = ParakeetTdtV3.Result(text:"",tokenIDs:[1,2,3],emissionFrames:[0,1,2])
    let cut = ParakeetTdtV3.join(a,previousStart:0,previousFirst:0,next:b,nextStart:100*1280,seam:100*1280)
    #expect(cut.previousEnd == 3)
    #expect(cut.nextBegin == 0)
    let empty = ParakeetTdtV3.Result(text:"",tokenIDs:[],emissionFrames:[])
    let silence = ParakeetTdtV3.join(a,previousStart:0,previousFirst:0,next:empty,nextStart:100*1280,seam:100*1280)
    #expect(silence.previousEnd == 3)
}

@Test func v3RecoveryKeepsPrimaryTokensAndRequiresAnchorsAtBothSides() throws {
    let primary = ParakeetTdtV3.Result(text: "", tokenIDs: [1,2,3,8,9,10], emissionFrames: [0,1,2,100,101,102])
    let candidate = ParakeetTdtV3.Result(text: "", tokenIDs: Array(1...10), emissionFrames: [0,1,2,20,40,60,80,100,101,102])
    let gaps = [ParakeetTdtV3.BlankSpan(index: 3, left: 2, right: 100)]
    let recovered = try #require(ParakeetTdtV3.insertRecoveredTokens(primary, candidate: candidate, gaps: gaps))
    #expect(recovered.ids == Array(1...10))
    #expect(recovered.frames == candidate.emissionFrames)
    let missingAnchor = ParakeetTdtV3.Result(text: "", tokenIDs: [1,2,3,4,5,6,7,8,9,99], emissionFrames: candidate.emissionFrames)
    #expect(ParakeetTdtV3.insertRecoveredTokens(primary, candidate: missingAnchor, gaps: gaps) == nil)
}

@Test func v3RecoveryRejectsRepeatedTextAtAnotherTimeAndTinyAdditions() {
    let primary = ParakeetTdtV3.Result(text: "", tokenIDs: [1,2,3,8,9,10], emissionFrames: [0,1,2,100,101,102])
    let gaps = [ParakeetTdtV3.BlankSpan(index: 3, left: 2, right: 100)]
    let shifted = ParakeetTdtV3.Result(text: "", tokenIDs: Array(1...10), emissionFrames: [200,201,202,220,240,260,280,300,301,302])
    #expect(ParakeetTdtV3.insertRecoveredTokens(primary, candidate: shifted, gaps: gaps) == nil)
    let tiny = ParakeetTdtV3.Result(text: "", tokenIDs: [1,2,3,4,5,8,9,10], emissionFrames: [0,1,2,30,60,100,101,102])
    #expect(ParakeetTdtV3.insertRecoveredTokens(primary, candidate: tiny, gaps: gaps) == nil)
}

@Test func v3RecoveryCanFillLeadingAndTrailingGapsWithoutReplacingPrimaryTokens() throws {
    let primary = ParakeetTdtV3.Result(text: "", tokenIDs: [4,5,6], emissionFrames: [100,101,102])
    let candidate = ParakeetTdtV3.Result(text: "", tokenIDs: Array(1...9), emissionFrames: [0,20,40,100,101,102,180,200,220])
    let gaps = [ParakeetTdtV3.BlankSpan(index: 0, left: -1, right: 100),
                ParakeetTdtV3.BlankSpan(index: 3, left: 102, right: 300)]
    let recovered = try #require(ParakeetTdtV3.insertRecoveredTokens(primary, candidate: candidate, gaps: gaps))
    #expect(recovered.ids == Array(1...9))
    #expect(Array(recovered.frames[3..<6]) == primary.emissionFrames)
}

@Test func v3BlankRecoveryRequiresChangingActiveAudio() {
    let primary = ParakeetTdtV3.Result(text: "", tokenIDs: [1,2,3,4,5,6], emissionFrames: [0,1,2,120,121,122])
    let silence = [Float](repeating: 0, count: 12 * 16000)
    #expect(ParakeetTdtV3.blankSpans(primary, samples: silence).isEmpty)
    var steady = silence, changing = silence
    for i in steady.indices {
        let wave = Float(sin(Double(i) * 2 * .pi * 100 / 16000))
        steady[i] = wave * 0.1
        changing[i] = wave * ((i / 3200) % 2 == 0 ? 0.05 : 0.2)
    }
    #expect(ParakeetTdtV3.blankSpans(primary, samples: steady).isEmpty)
    let gaps = ParakeetTdtV3.blankSpans(primary, samples: changing)
    #expect(gaps.count == 1)
    #expect(gaps.first?.index == 3)
}

@Test func v3OwnedTensorsAreAlignedAndValidateAllocationSizes() throws {
    let tensor = try ParakeetTdtV3.alignedOutput([1,1024,376])
    let length = try ParakeetTdtV3.alignedOutput([1], .int32)
    #expect(Int(bitPattern: tensor.dataPointer) % Int(getpagesize()) == 0)
    #expect(Int(bitPattern: length.dataPointer) % Int(getpagesize()) == 0)
    #expect(tensor.pixelBuffer == nil && length.pixelBuffer == nil)
    #expect(tensor.strides.map(\.intValue) == [385024,376,1])
    tensor[[0,1023,375]] = 4.25
    #expect(tensor[tensor.count - 1].floatValue == 4.25)
    #expect(throws: ParakeetError.self) { try ParakeetTdtV3.alignedOutput([1], .double) }
    #expect(throws: ParakeetError.self) { try ParakeetTdtV3.alignedOutput([Int.max,2]) }
    #expect(throws: ParakeetError.self) { try ParakeetTdtV3.alignedOutput([0]) }
}

@Test func v3ScopedArgmaxHonorsPaddedLogitStorage() throws {
    let pointer = UnsafeMutablePointer<Float>.allocate(capacity: 10)
    pointer.initialize(repeating: 999, count: 10)
    let values: [Float] = [1,5,5,-1,4]
    for (i, value) in values.enumerated() { pointer[i * 2] = value }
    let logits = try MLMultiArray(dataPointer: pointer, shape: [1,1,1,5], dataType: .float32,
        strides: [10,10,10,2], deallocator: { $0.assumingMemoryBound(to: Float.self).deallocate() })
    #expect(try ParakeetTdtV3.argmax(logits, 0..<5) == 1)
    pointer[8] = .infinity
    #expect(throws: ParakeetError.self) { try ParakeetTdtV3.argmax(logits, 0..<5) }
}
