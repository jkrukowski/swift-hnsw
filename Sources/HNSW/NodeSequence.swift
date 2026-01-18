import CHNSW

public struct NodeSequence: Sequence {
    private let ptr: UnsafeMutablePointer<CHNSW.HNSW>

    init(_ ptr: UnsafeMutablePointer<CHNSW.HNSW>) {
        self.ptr = ptr
    }

    public func makeIterator() -> NodeIterator {
        NodeIterator(ptr)
    }
}
