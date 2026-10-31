import CoreML
import CoreVideo
import Testing
@testable import ParakeetRuntime

private func paddedFeatures(halfPrecision: Bool) throws -> (MLMultiArray, [Float]) {
    let count = 16
    let bytes = count * (halfPrecision ? MemoryLayout<Float16>.stride : MemoryLayout<Float>.stride)
    let storage = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: MemoryLayout<Float>.alignment)
    if halfPrecision {
        storage.bindMemory(to: Float16.self, capacity: count).initialize(repeating: .nan, count: count)
    } else {
        storage.bindMemory(to: Float.self, capacity: count).initialize(repeating: .nan, count: count)
    }
    let source: MLMultiArray
    do {
        // Padding between rows and columns, containing NaNs to reveal accidental flat copies.
        source = try MLMultiArray(dataPointer: storage, shape: [1, 2, 3],
                                  dataType: halfPrecision ? .float16 : .float32,
                                  strides: [16, 8, 2], deallocator: { $0.deallocate() })
    } catch { storage.deallocate(); throw error }
    let values: [Float] = [-2.19, 0.12345, 3.26, 1, -1, 0]
    for channel in 0..<2 {
        for frame in 0..<3 {
            source[[0, NSNumber(value: channel), NSNumber(value: frame)]] = NSNumber(value: values[channel * 3 + frame])
        }
    }
    return (source, values)
}

@Test(arguments: [false, true])
func melConversionSkipsPaddingAndPreservesLogicalValues(halfPrecision: Bool) throws {
    let (source, values) = try paddedFeatures(halfPrecision: halfPrecision)
    let result = try ParakeetTdtCtc110M.encoderMelInput(source)
    #expect(result.shape == source.shape)
    #expect(result.dataType == .float16)
    for channel in 0..<2 {
        for frame in 0..<3 {
            let actual = result[[0, NSNumber(value: channel), NSNumber(value: frame)]].floatValue
            #expect(actual == Float(Float16(values[channel * 3 + frame])))
        }
    }
}

@Test(arguments: [false, true])
func tensorReadsPreservePaddedRowsAndColumns(halfPrecision: Bool) throws {
    let (source, original) = try paddedFeatures(halfPrecision: halfPrecision)
    let expected = halfPrecision ? original.map { Float(Float16($0)) } : original
    #expect(ParakeetTdtCtc110M.contiguous(source, rows: 2) == expected)
    #expect(ParakeetTdtCtc110M.row(source, 0) == Array(expected[0..<3]))
    #expect(ParakeetTdtCtc110M.row(source, 1) == Array(expected[3..<6]))
}

@Test(arguments: [false, true])
func tensorSnapshotsOwnTheirStorage(halfPrecision: Bool) throws {
    let (source, original) = try paddedFeatures(halfPrecision: halfPrecision)
    let snapshot = try ParakeetTdtCtc110M.snapshotTensor(source)
    source[[0, 0, 0]] = 999
    let expected = halfPrecision ? original.map { Float(Float16($0)) } : original
    #expect(ParakeetTdtCtc110M.contiguous(snapshot, rows: 2) == expected)
    snapshot[[0, 1, 2]] = 123
    #expect(source[[0, 1, 2]].floatValue == 0)
}

@Test func pixelBufferTensorReadsUseTheCurrentBackingLayout() throws {
    let width = 1025, rows = 3
    let attributes = [kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                      kCVPixelBufferBytesPerRowAlignmentKey as String: 64] as [String: Any]
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, rows, kCVPixelFormatType_OneComponent16Half,
                                    attributes as CFDictionary, &pixelBuffer)
    try #require(status == kCVReturnSuccess)
    let buffer = try #require(pixelBuffer)
    let values = (0..<(width * rows)).map { Float(Float16(Float($0 % 127) / 16 - 4)) }
    try #require(CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess)
    let pointer = try #require(CVPixelBufferGetBaseAddress(buffer)).assumingMemoryBound(to: Float16.self)
    let stride = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float16>.stride
    for r in 0..<rows { for c in 0..<width { pointer[r * stride + c] = Float16(values[r * width + c]) } }
    CVPixelBufferUnlockBaseAddress(buffer, [])
    let array = MLMultiArray(pixelBuffer: buffer, shape: [1, NSNumber(value: rows), NSNumber(value: width)])
    for _ in 0..<3 {
        let equal = ParakeetTdtCtc110M.contiguous(array, rows: rows) == values
        #expect(equal)
        #expect(ParakeetTdtCtc110M.row(array, 1) == Array(values[width..<(width * 2)]))
    }
}

@Test func melConversionRejectsUnsupportedShapeAndType() throws {
    let wrongShape = try MLMultiArray(shape: [2, 3], dataType: .float32)
    let wrongType = try MLMultiArray(shape: [1, 2, 3], dataType: .int32)
    #expect(throws: ParakeetError.self) { try ParakeetTdtCtc110M.encoderMelInput(wrongShape) }
    #expect(throws: ParakeetError.self) { try ParakeetTdtCtc110M.encoderMelInput(wrongType) }
}
