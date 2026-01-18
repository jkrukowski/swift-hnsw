import CHNSW

public struct SerializedNode: ~Copyable {
    private let ptr: UnsafeMutablePointer<CHNSW.hnswSerNode>

    deinit {
        CHNSW.hnsw_free_serialized_node(ptr)
    }

    init(_ ptr: UnsafeMutablePointer<CHNSW.hnswSerNode>) {
        self.ptr = ptr
    }

    public var paramsCount: Int {
        Int(ptr.pointee.params_count)
    }

    public var vectorSizeInBytes: Int {
        Int(ptr.pointee.vector_size)
    }

    public func params() -> [UInt64] {
        let buffer = UnsafeBufferPointer(start: ptr.pointee.params, count: paramsCount)
        return Array(buffer)
    }

    public func vectorData() -> [UInt8] {
        let size = vectorSizeInBytes
        return [UInt8](unsafeUninitializedCapacity: size) { buffer, writtenCount in
            let source = ptr.pointee.vector.assumingMemoryBound(to: UInt8.self)
            buffer.baseAddress?.initialize(from: source, count: size)
            writtenCount = size
        }
    }
}
