import Numerics
import Testing

@testable import HNSW

@Test func `index init should correctly initalize properties`() throws {
    let index1 = try Index(8)
    #expect(index1.nodeCount == 0)
    #expect(index1.vectorDim == 8)
    #expect(index1.quantization == nil)

    let index2 = try Index(16, quantization: .binary)
    #expect(index2.nodeCount == 0)
    #expect(index2.vectorDim == 16)
    #expect(index2.quantization == .binary)

    let index3 = try Index(4, quantization: .q8)
    #expect(index3.nodeCount == 0)
    #expect(index3.vectorDim == 4)
    #expect(index3.quantization == .q8)
}

@Test func `insert node should store node in index`() throws {
    let index = try Index(4)
    let node1 = try index.insert([1, 2, 3, 4], nodeId: 100)
    let node2 = try index.insert([5, 6, 7, 8], nodeId: 200)
    let node3 = try index.insert([8, 3, 4, 9])

    #expect(index.nodeCount == 3)
    #expect([node1.id, node2.id, node3.id] == [100, 200, 1])
}

@Test func `delete node should delete node from index`() throws {
    let index = try Index(4)
    let node1 = try index.insert([1, 2, 3, 4], nodeId: 100)
    let node2 = try index.insert([5, 6, 7, 8], nodeId: 200)
    let node3 = try index.insert([8, 3, 4, 9])

    try index.delete(node1)
    #expect(index.nodeCount == 2)

    try index.delete(node3)
    #expect(index.nodeCount == 1)

    let nodeIndexIds = index.nodes().map { $0.id }
    #expect(nodeIndexIds == [node2.id])
}

@Test func `iterating index should visit all nodes`() throws {
    let index = try Index(4)
    let node1 = try index.insert([1, 2, 3, 4])
    let node2 = try index.insert([5, 6, 7, 8])
    let node3 = try index.insert([8, 3, 4, 9])

    let nodeIds = [node1.id, node2.id, node3.id]
    let nodeIndexIds = index.nodes().map { $0.id }

    #expect(nodeIds.count == nodeIndexIds.count)
    #expect(Set(nodeIds) == Set(nodeIndexIds))
}

@Test func `index should allow to insert and retrieve array of values`() throws {
    let index = try Index(4)
    let node1 = try index.insert([1, 2, 3, 4])
    let node2 = try index.insert([5, 6, 7, 8])
    let node3 = try index.insert([8, 3, 4, 9])

    #expect(allClose(index.vector(of: node1), [1, 2, 3, 4]))
    #expect(allClose(index.vector(of: node2), [5, 6, 7, 8]))
    #expect(allClose(index.vector(of: node3), [8, 3, 4, 9]))
}

@Test func `q8 quantized index should allow to insert and retrieve array of values`() throws {
    let index = try Index(4, quantization: .q8)
    let node1 = try index.insert([1, 2, 3, 4])
    let node2 = try index.insert([5, 6, 7, 8])
    let node3 = try index.insert([8, 3, 4, 9])

    #expect(allClose(index.vector(of: node1), [1, 2, 3, 4], absoluteTolerance: 0.1))
    #expect(allClose(index.vector(of: node2), [5, 6, 7, 8], absoluteTolerance: 0.1))
    #expect(allClose(index.vector(of: node3), [8, 3, 4, 9], absoluteTolerance: 0.1))
}

@Test func `binary quantized index should allow to insert and retrieve array of values`() throws {
    let index = try Index(4, quantization: .binary)

    let node1 = try index.insert([1, -2, 3, -4])
    let node2 = try index.insert([-5, 6, -7, 8])
    let node3 = try index.insert([8, -3, 4, -9])

    #expect(allClose(index.vector(of: node1), [1, -1, 1, -1], absoluteTolerance: 0.1))
    #expect(allClose(index.vector(of: node2), [-1, 1, -1, 1], absoluteTolerance: 0.1))
    #expect(allClose(index.vector(of: node3), [1, -1, 1, -1], absoluteTolerance: 0.1))
}

@Test func `serialized node should preserve serialized data`() throws {
    let index = try Index(3)
    let node = try index.insert([1.5, 2.5, 3.5])

    let serializedNode = try index.serialize(node)
    let serializedNodeVector = serializedNode.vectorData()

    #expect(serializedNode.vectorSizeInBytes == 12)
    #expect(serializedNodeVector == [16, 153, 168, 62, 141, 127, 12, 63, 147, 178, 68, 63])
}

@Test func `search should find nearest neighbors`() throws {
    let index = try Index(4)

    try index.insert([1.0, 0.0, 0.0, 0.0], nodeId: 1)
    try index.insert([0.0, 1.0, 0.0, 0.0], nodeId: 2)
    try index.insert([0.8, 0.0, 1.0, 0.0], nodeId: 3)
    try index.insert([0.9, 0.1, 0.0, 0.0], nodeId: 4)

    let results = index.search([1.0, 0.0, 0.0, 0.0], k: 3)

    #expect(results.count == 3)
    #expect(results[0].distance.isApproximatelyEqual(to: 0.0))
    #expect(results[0].node.id == 1)
    #expect(results[1].node.id == 4)
    #expect(results[2].node.id == 3)
}

@Test func `optimistic insert should successfully insert node`() throws {
    let index = try Index(4)

    try index.insert([1.0, 0.0, 0.0, 0.0], nodeId: 1)
    try index.insert([0.0, 1.0, 0.0, 0.0], nodeId: 2)

    guard let context = index.prepareInsert([0.5, 0.5, 0.0, 0.0], nodeId: 3) else {
        Issue.record("Couldn't create insert context")
        return
    }
    guard let node = index.commitInsert(context) else {
        Issue.record("Couldn't commit insert")
        return
    }

    #expect(node.id == 3)
    #expect(index.nodeCount == 3)
}

@Test func `optimistic insert context can be safely discarded`() throws {
    let index = try Index(4)

    try index.insert([1.0, 0.0, 0.0, 0.0], nodeId: 1)
    try index.insert([0.0, 1.0, 0.0, 0.0], nodeId: 2)

    guard let context = index.prepareInsert([0.5, 0.5, 0.0, 0.0], nodeId: 3) else {
        Issue.record("Couldn't create insert context")
        return
    }
    context.discard()

    #expect(index.nodeCount == 2)
}

@Test func `searchConcurrent should retrieve results`() throws {
    let index = try Index(4)

    try index.insert([1.0, 0.0, 0.0, 0.0], nodeId: 1)
    try index.insert([0.0, 1.0, 0.0, 0.0], nodeId: 2)
    try index.insert([0.8, 0.0, 1.0, 0.0], nodeId: 3)
    try index.insert([0.9, 0.1, 0.0, 0.0], nodeId: 4)

    let results = try index.searchConcurrent([1.0, 0.0, 0.0, 0.0], k: 3)

    #expect(results.count == 3)
    #expect(results[0].distance.isApproximatelyEqual(to: 0.0))
    #expect(results[0].node.id == 1)
    #expect(results[1].node.id == 4)
    #expect(results[2].node.id == 3)
}

@Test func `search should retrieve results`() throws {
    let index = try Index(4)

    try index.insert([1.0, 0.0, 0.0, 0.0], nodeId: 1)
    try index.insert([0.0, 1.0, 0.0, 0.0], nodeId: 2)
    try index.insert([0.8, 0.0, 1.0, 0.0], nodeId: 3)
    try index.insert([0.9, 0.1, 0.0, 0.0], nodeId: 4)

    let results = index.search([1.0, 0.0, 0.0, 0.0], k: 3)

    #expect(results.count == 3)
    #expect(results[0].distance.isApproximatelyEqual(to: 0.0))
    #expect(results[0].node.id == 1)
    #expect(results[1].node.id == 4)
    #expect(results[2].node.id == 3)
}

@Test func `insertSerialized should insert node with serialized data`() throws {
    let index1 = try Index(4)
    let node1 = try index1.insert([1.0, 2.0, 3.0, 4.0], nodeId: 100)
    let serialized = try index1.serialize(node1)

    let index2 = try Index(4)
    let node2 = try index2.insertSerialized(serialized.vectorData(), params: serialized.params())
    index2.deserialize()

    #expect(index2.nodeCount == 1)
    #expect(node2.id == 100)
    #expect(allClose(index2.vector(of: node2), [1.0, 2.0, 3.0, 4.0]))
}

@Test func `serialize empty index should return metadata only`() throws {
    let index = try Index(8)
    let data = try index.serialize()

    // 4 UInt64 values: version, vectorDim, nodeCount, quantization
    #expect(data.count == 32)

    // Parse metadata
    let version = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt64.self) }
    let vectorDim = data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt64.self) }
    let nodeCount = data.withUnsafeBytes { $0.load(fromByteOffset: 16, as: UInt64.self) }
    let quantization = data.withUnsafeBytes { $0.load(fromByteOffset: 24, as: UInt64.self) }

    #expect(version == 1)
    #expect(vectorDim == 8)
    #expect(nodeCount == 0)
    #expect(quantization == 0)
}

@Test func `serialize index should include metadata and node data`() throws {
    let index = try Index(4)
    try index.insert([1.0, 2.0, 3.0, 4.0], nodeId: 100)
    try index.insert([5.0, 6.0, 7.0, 8.0], nodeId: 200)

    let data = try index.serialize()

    // Parse metadata
    let version = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt64.self) }
    let vectorDim = data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt64.self) }
    let nodeCount = data.withUnsafeBytes { $0.load(fromByteOffset: 16, as: UInt64.self) }
    let quantization = data.withUnsafeBytes { $0.load(fromByteOffset: 24, as: UInt64.self) }

    #expect(version == 1)
    #expect(vectorDim == 4)
    #expect(nodeCount == 2)
    #expect(quantization == 0)

    // Data should be larger than just metadata
    #expect(data.count > 32)
}

@Test func `serialize index with quantization should preserve quantization type`() throws {
    let indexQ8 = try Index(4, quantization: .q8)
    let dataQ8 = try indexQ8.serialize()
    let quantizationQ8 = dataQ8.withUnsafeBytes { $0.load(fromByteOffset: 24, as: UInt64.self) }
    #expect(quantizationQ8 == 1)

    let indexBinary = try Index(4, quantization: .binary)
    let dataBinary = try indexBinary.serialize()
    let quantizationBinary = dataBinary.withUnsafeBytes {
        $0.load(fromByteOffset: 24, as: UInt64.self)
    }
    #expect(quantizationBinary == 2)
}

@Test func `serialize and deserialize index should preserve all nodes`() throws {
    let index1 = try Index(4)
    try index1.insert([1.0, 2.0, 3.0, 4.0], nodeId: 100)
    try index1.insert([5.0, 6.0, 7.0, 8.0], nodeId: 200)
    try index1.insert([9.0, 10.0, 11.0, 12.0], nodeId: 300)

    let data = try index1.serialize()

    // Parse and reconstruct
    var offset = 0
    let _ = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }  // version
    offset += 8
    let vectorDim = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
    offset += 8
    let nodeCount = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
    offset += 8
    let _ = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }  // quantization
    offset += 8

    let index2 = try Index(Int(vectorDim))

    for _ in 0..<nodeCount {
        let vectorSize = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
        offset += 8
        let paramsCount = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
        offset += 8

        let vectorData = Array(data[offset..<(offset + Int(vectorSize))])
        offset += Int(vectorSize)

        var params = [UInt64]()
        for _ in 0..<paramsCount {
            let param = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
            params.append(param)
            offset += 8
        }

        try index2.insertSerialized(vectorData, params: params)
    }
    index2.deserialize()

    #expect(index2.nodeCount == 3)

    let nodeIds = Set(index2.nodes().map { $0.id })
    #expect(nodeIds == Set([100, 200, 300]))

    // Verify vectors are preserved
    for node in index2.nodes() {
        let vector = index2.vector(of: node)
        switch node.id {
        case 100:
            #expect(allClose(vector, [1.0, 2.0, 3.0, 4.0]))
        case 200:
            #expect(allClose(vector, [5.0, 6.0, 7.0, 8.0]))
        case 300:
            #expect(allClose(vector, [9.0, 10.0, 11.0, 12.0]))
        default:
            Issue.record("Unexpected node id: \(node.id)")
        }
    }
}

@Test func `serialize and deserialize should preserve search functionality`() throws {
    let index1 = try Index(4)
    try index1.insert([1.0, 0.0, 0.0, 0.0], nodeId: 1)
    try index1.insert([0.0, 1.0, 0.0, 0.0], nodeId: 2)
    try index1.insert([0.9, 0.1, 0.0, 0.0], nodeId: 3)

    let data = try index1.serialize()

    // Reconstruct index
    var offset = 8  // skip version
    let vectorDim = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
    offset += 8
    let nodeCount = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
    offset += 16  // skip nodeCount and quantization

    let index2 = try Index(Int(vectorDim))

    for _ in 0..<nodeCount {
        let vectorSize = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
        offset += 8
        let paramsCount = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
        offset += 8

        let vectorData = Array(data[offset..<(offset + Int(vectorSize))])
        offset += Int(vectorSize)

        var params = [UInt64]()
        for _ in 0..<paramsCount {
            let param = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
            params.append(param)
            offset += 8
        }

        try index2.insertSerialized(vectorData, params: params)
    }
    index2.deserialize()

    // Search should work correctly
    let results = index2.search([1.0, 0.0, 0.0, 0.0], k: 2)
    #expect(results.count == 2)
    #expect(results[0].node.id == 1)
    #expect(results[1].node.id == 3)
}

@Test func `deserialize from data should restore all nodes`() throws {
    let index1 = try Index(4)
    try index1.insert([1.0, 2.0, 3.0, 4.0], nodeId: 100)
    try index1.insert([5.0, 6.0, 7.0, 8.0], nodeId: 200)
    try index1.insert([9.0, 10.0, 11.0, 12.0], nodeId: 300)

    let data = try index1.serialize()

    let index2 = try Index.deserialize(from: data)

    #expect(index2.nodeCount == 3)

    let nodeIds = Set(index2.nodes().map { $0.id })
    #expect(nodeIds == Set([100, 200, 300]))

    for node in index2.nodes() {
        let vector = index2.vector(of: node)
        switch node.id {
        case 100:
            #expect(allClose(vector, [1.0, 2.0, 3.0, 4.0]))
        case 200:
            #expect(allClose(vector, [5.0, 6.0, 7.0, 8.0]))
        case 300:
            #expect(allClose(vector, [9.0, 10.0, 11.0, 12.0]))
        default:
            Issue.record("Unexpected node id: \(node.id)")
        }
    }
}

@Test func `deserialize from data should preserve search functionality`() throws {
    let index1 = try Index(4)
    try index1.insert([1.0, 0.0, 0.0, 0.0], nodeId: 1)
    try index1.insert([0.0, 1.0, 0.0, 0.0], nodeId: 2)
    try index1.insert([0.9, 0.1, 0.0, 0.0], nodeId: 3)

    let data = try index1.serialize()

    let index2 = try Index.deserialize(from: data)

    let results = index2.search([1.0, 0.0, 0.0, 0.0], k: 2)
    #expect(results.count == 2)
    #expect(results[0].node.id == 1)
    #expect(results[1].node.id == 3)
}

@Test func `deserialize from empty data should throw error`() throws {
    #expect(throws: IndexError.deserializationFailed) {
        _ = try Index.deserialize(from: [])
    }
}

@Test func `deserialize from truncated data should throw error`() throws {
    let index1 = try Index(4)
    try index1.insert([1.0, 2.0, 3.0, 4.0], nodeId: 100)
    let data = try index1.serialize()

    // Truncate to less than minimum header size (32 bytes)
    let truncatedData = Array(data[0..<30])

    #expect(throws: IndexError.deserializationFailed) {
        _ = try Index.deserialize(from: truncatedData)
    }
}

@Test func `deserialize with unsupported version should throw error`() throws {
    let index = try Index(4)
    var data = try index.serialize()

    // Corrupt the version field (first 8 bytes) to an unsupported version
    var unsupportedVersion: UInt64 = 999
    withUnsafeBytes(of: &unsupportedVersion) { bytes in
        for i in 0..<8 {
            data[i] = bytes[i]
        }
    }

    #expect(throws: IndexError.unsupportedSerializationVersion(999)) {
        _ = try Index.deserialize(from: data)
    }
}
