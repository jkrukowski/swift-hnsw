import CHNSW

public struct NodeIterator: IteratorProtocol {
    private var cursor: UnsafeMutablePointer<CHNSW.hnswCursor>?

    init(_ ptr: UnsafeMutablePointer<CHNSW.HNSW>) {
        self.cursor = CHNSW.hnsw_cursor_init(ptr)
    }

    public mutating func next() -> Node? {
        guard let cursor else {
            return nil
        }
        guard CHNSW.hnsw_cursor_acquire_lock(cursor) != 0 else {
            CHNSW.hnsw_cursor_free(cursor)
            self.cursor = nil
            return nil
        }
        guard let ptr = CHNSW.hnsw_cursor_next(cursor) else {
            CHNSW.hnsw_cursor_release_lock(cursor)
            CHNSW.hnsw_cursor_free(cursor)
            self.cursor = nil
            return nil
        }
        CHNSW.hnsw_cursor_release_lock(cursor)
        return Node(ptr)
    }
}
