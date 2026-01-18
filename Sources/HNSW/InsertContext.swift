import CHNSW

public struct InsertContext: ~Copyable {
    let ptr: OpaquePointer

    deinit {
        CHNSW.hnsw_free_insert_context(ptr)
    }

    init(_ ptr: OpaquePointer) {
        self.ptr = ptr
    }

    consuming func discard() {
        discard self
    }
}
