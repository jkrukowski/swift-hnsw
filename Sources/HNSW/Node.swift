import CHNSW

public struct Node {
    public let id: Int
    public let level: Int

    let ptr: UnsafeMutablePointer<CHNSW.hnswNode>

    init(_ ptr: UnsafeMutablePointer<CHNSW.hnswNode>) {
        self.ptr = ptr
        self.id = Int(ptr.pointee.id)
        self.level = Int(ptr.pointee.level)
    }
}

extension Node: CustomDebugStringConvertible {
    public var debugDescription: String {
        "<Node id: \(id), level: \(level)>"
    }
}
