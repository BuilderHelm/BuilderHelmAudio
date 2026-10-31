import CoreML
import Testing
@testable import ParakeetRuntime

@Test(arguments: [false, true])
func melConversionSkipsPaddingAndPreservesLogicalValues(halfPrecision: Bool) throws {
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

@Test func melConversionRejectsUnsupportedShapeAndType() throws {
    let wrongShape = try MLMultiArray(shape: [2, 3], dataType: .float32)
    let wrongType = try MLMultiArray(shape: [1, 2, 3], dataType: .int32)
    #expect(throws: ParakeetError.self) { try ParakeetTdtCtc110M.encoderMelInput(wrongShape) }
    #expect(throws: ParakeetError.self) { try ParakeetTdtCtc110M.encoderMelInput(wrongType) }
}
