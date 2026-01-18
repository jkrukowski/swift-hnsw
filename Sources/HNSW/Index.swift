import CHNSW

/// A Hierarchical Navigable Small World (HNSW) index for efficient approximate nearest neighbor search.
///
/// HNSW is a graph-based algorithm that organizes vectors in a hierarchical structure, enabling fast similarity searches
/// with high recall. This implementation is based on the paper by Yu. A. Malkov and D. A. Yashunin.
///
/// ## Features
/// - Thread-safe concurrent search operations
/// - Optional vector quantization (Q8 or binary) for memory efficiency
/// - Bi-directional graph links for better connectivity
/// - True node deletion (not just marking as deleted)
/// - Normalized vectors using cosine similarity/dot product
///
/// ## Example
/// ```swift
/// // Create an index for 128-dimensional vectors with Q8 quantization
/// let index = try Index(128, quantization: .q8, m: 16)
///
/// // Insert vectors
/// let vector = [Float](repeating: 0.5, count: 128)
/// try index.insert(vector)
///
/// // Search for nearest neighbors
/// let results = index.search(vector, k: 10)
/// for result in results {
///     print("Distance: \(result.distance)")
/// }
/// ```
public struct Index: ~Copyable, Sendable {
    nonisolated(unsafe) private let ptr: UnsafeMutablePointer<CHNSW.HNSW>

    deinit {
        CHNSW.hnsw_free(ptr, nil)
    }

    init(_ ptr: UnsafeMutablePointer<CHNSW.HNSW>) {
        self.ptr = ptr
    }

    /// Creates a new HNSW index.
    ///
    /// - Parameters:
    ///   - vectorDim: The dimensionality of the vectors to be stored. Must be greater than 0.
    ///   - quantization: Optional quantization type to compress vectors. Use `.q8` for int8 quantization
    ///                   or `.binary` for binary quantization. Defaults to no quantization.
    ///   - m: The M parameter from the HNSW paper. Controls the number of bi-directional links per node.
    ///        Layer 0 has M*2 neighbors, other layers have M neighbors. Use 0 for the default value (16).
    ///        Valid range: 4 to 4096, or 0 for default.
    ///
    /// - Throws: `IndexError.initializationFailed` if the index cannot be created.
    ///
    /// - Precondition: `vectorDim` must be greater than 0.
    ///
    /// - Note: Vectors are automatically normalized on insertion, making cosine similarity and dot product equivalent.
    ///         This means Euclidean distance cannot be used with this implementation.
    public init(_ vectorDim: Int, quantization: Quantization? = nil, m: Int = 0) throws {
        precondition(vectorDim > 0, "vectorDim must be greater than 0, got \(vectorDim)")

        let quantizationRaw: UInt32 =
            switch quantization {
            case .some(let value):
                UInt32(value.rawValue)
            case .none:
                0
            }
        guard let ptr = CHNSW.hnsw_new(UInt32(vectorDim), quantizationRaw, UInt32(m)) else {
            throw IndexError.initializationFailed
        }
        self.init(ptr)
    }

    /// The dimensionality of vectors stored in this index.
    public var vectorDim: Int {
        Int(ptr.pointee.vector_dim)
    }

    /// The total number of nodes currently in the index.
    public var nodeCount: Int {
        Int(ptr.pointee.node_count)
    }

    /// The quantization type used by this index, if any.
    public var quantization: Quantization? {
        Quantization(rawValue: Int(ptr.pointee.quant_type))
    }

    /// Returns a sequence of all nodes in the index.
    ///
    /// - Returns: A `NodeSequence` that can be iterated to access all nodes.
    ///
    /// - Note: The sequence provides a snapshot of nodes at the time of iteration.
    ///         Nodes added or removed during iteration may or may not be included.
    public func nodes() -> NodeSequence {
        NodeSequence(ptr)
    }
}

// MARK: - Serialization

extension Index {
    /// Serializes a single node to a portable format.
    ///
    /// This method extracts all the necessary information from a node including its vector data,
    /// layer information, links to other nodes (represented as node IDs), and metadata.
    ///
    /// - Parameter node: The node to serialize.
    ///
    /// - Returns: A `SerializedNode` containing the node's vector and parameters.
    ///
    /// - Throws: `IndexError.nodeSerializationFailed` if the node cannot be serialized.
    ///
    /// - Note: The serialized format includes:
    ///   - Vector data (quantized or raw floats)
    ///   - Node ID and level information
    ///   - Layer-specific link counts and neighbor IDs
    ///   - Worst link distance information for each layer
    ///   - L2 norm and quantization range metadata
    public func serialize(_ node: borrowing Node) throws -> SerializedNode {
        guard let nodePtr = CHNSW.hnsw_serialize_node(ptr, node.ptr) else {
            throw IndexError.nodeSerializationFailed
        }
        return SerializedNode(nodePtr)
    }

    /// Serializes the entire index to a byte array.
    ///
    /// This method creates a complete snapshot of the index that can be saved to disk or transmitted
    /// over a network. The serialization format is platform-independent and includes:
    /// - Serialization version (for future compatibility)
    /// - Index metadata (vector dimension, node count, quantization type)
    /// - All nodes with their vectors, links, and metadata
    ///
    /// - Returns: A byte array containing the serialized index.
    ///
    /// - Throws: `IndexError.nodeSerializationFailed` if any node fails to serialize.
    ///
    /// - Note: The serialized data can be restored using `Index.deserialize(from:)`.
    ///         The format includes version information to ensure compatibility.
    public func serialize() throws -> [UInt8] {
        var data = [UInt8]()

        // Serialize version and index metadata
        var versionValue = Constants.serializationVersion
        var vectorDimValue = UInt64(vectorDim)
        var nodeCountValue = UInt64(nodeCount)
        var quantizationValue = UInt64(quantization?.rawValue ?? 0)

        withUnsafeBytes(of: &versionValue) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &vectorDimValue) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &nodeCountValue) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &quantizationValue) { data.append(contentsOf: $0) }

        // Serialize each node
        for node in nodes() {
            let serializedNode = try serialize(node)

            var vectorSize = UInt64(serializedNode.vectorSizeInBytes)
            let vectorData = serializedNode.vectorData()
            let params = serializedNode.params()
            var paramsCount = UInt64(params.count)

            withUnsafeBytes(of: &vectorSize) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &paramsCount) { data.append(contentsOf: $0) }
            data.append(contentsOf: vectorData)
            for var param in params {
                withUnsafeBytes(of: &param) { data.append(contentsOf: $0) }
            }
        }
        return data
    }

    /// Fixes pointer relationships after deserialization.
    ///
    /// After nodes are inserted via `insertSerialized(_:params:)`, this method must be called
    /// to restore the internal pointer relationships between nodes based on their IDs.
    /// It uses random salts for hashing to prevent hash collision attacks.
    ///
    /// - Important: This method must be called after all serialized nodes have been inserted
    ///              and before the index can be used for search operations.
    public func deserialize() {
        CHNSW.hnsw_deserialize_index(
            ptr,
            UInt64.random(in: UInt64.min..<UInt64.max),
            UInt64.random(in: UInt64.min..<UInt64.max)
        )
    }

    /// Deserializes an index from a byte array.
    ///
    /// This static method reconstructs a complete HNSW index from serialized data created by
    /// `serialize()`. It validates the format version, creates a new index with the correct
    /// parameters, inserts all serialized nodes, and fixes internal pointer relationships.
    ///
    /// - Parameter data: The byte array containing the serialized index data.
    ///
    /// - Returns: A fully reconstructed `Index` ready for use.
    ///
    /// - Throws:
    ///   - `IndexError.deserializationFailed` if the data is corrupted or incomplete
    ///   - `IndexError.unsupportedSerializationVersion(_)` if the data format version is not supported
    ///   - `IndexError.initializationFailed` if the index cannot be created
    ///   - `IndexError.nodeInsertionFailed` if any node cannot be inserted
    ///
    /// - Note: The deserialized index will have the same vector dimension, quantization type,
    ///         and M parameter as the original index.
    public static func deserialize(from data: [UInt8]) throws -> Index {
        // Minimum size: 4 UInt64 values (version, vectorDim, nodeCount, quantization)
        guard data.count >= 32 else {
            throw IndexError.deserializationFailed
        }

        var offset = 0

        // Parse and validate version
        let version = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
        offset += 8
        guard version == Constants.serializationVersion else {
            throw IndexError.unsupportedSerializationVersion(version)
        }

        // Parse metadata
        let vectorDim = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
        offset += 8
        let nodeCount = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt64.self) }
        offset += 8
        let quantizationRaw = data.withUnsafeBytes {
            $0.load(fromByteOffset: offset, as: UInt64.self)
        }
        offset += 8

        let quantization = Quantization(rawValue: Int(quantizationRaw))
        let index = try Index(Int(vectorDim), quantization: quantization)

        // Parse and insert each node
        for _ in 0..<nodeCount {
            guard offset + 16 <= data.count else {
                throw IndexError.deserializationFailed
            }

            let vectorSize = data.withUnsafeBytes {
                $0.load(fromByteOffset: offset, as: UInt64.self)
            }
            offset += 8
            let paramsCount = data.withUnsafeBytes {
                $0.load(fromByteOffset: offset, as: UInt64.self)
            }
            offset += 8

            let vectorEnd = offset + Int(vectorSize)
            guard vectorEnd <= data.count else {
                throw IndexError.deserializationFailed
            }

            let vectorData = Array(data[offset..<vectorEnd])
            offset = vectorEnd

            let paramsEnd = offset + Int(paramsCount) * 8
            guard paramsEnd <= data.count else {
                throw IndexError.deserializationFailed
            }

            var params = [UInt64]()
            params.reserveCapacity(Int(paramsCount))
            for _ in 0..<paramsCount {
                let param = data.withUnsafeBytes {
                    $0.load(fromByteOffset: offset, as: UInt64.self)
                }
                params.append(param)
                offset += 8
            }

            try index.insertSerialized(vectorData, params: params)
        }

        // Fix pointer relationships
        index.deserialize()
        return index
    }
}

// MARK: - Insertion / Deletion

extension Index {
    /// Inserts a pre-serialized node into the index.
    ///
    /// This method is used during deserialization to restore nodes from saved data.
    /// The vector data and parameters must match the format produced by `serialize(_:)`.
    ///
    /// - Parameters:
    ///   - vectorData: The serialized vector data (quantized or raw floats).
    ///   - params: Array of parameters containing node metadata, including:
    ///     - Node ID
    ///     - Level and serialization version
    ///     - Per-layer link counts and neighbor IDs
    ///     - Worst link information
    ///     - L2 norm and quantization range
    ///
    /// - Returns: The newly inserted `Node`.
    ///
    /// - Throws: `IndexError.nodeInsertionFailed` if the node cannot be inserted.
    ///
    /// - Important: After inserting all serialized nodes, you must call `deserialize()`
    ///              to fix internal pointer relationships.
    @discardableResult
    public func insertSerialized(
        _ vectorData: [UInt8],
        params: [UInt64]
    ) throws -> Node {
        let nodePtr = vectorData.span.withUnsafeBufferPointer { buffer in
            params.withUnsafeBufferPointer { paramsBuffer in
                CHNSW.hnsw_insert_serialized(
                    ptr,
                    UnsafeMutableRawPointer(mutating: buffer.baseAddress),
                    UnsafeMutablePointer(mutating: paramsBuffer.baseAddress),
                    UInt32(params.count),
                    nil
                )
            }
        }
        guard let nodePtr else {
            throw IndexError.nodeInsertionFailed
        }
        return Node(nodePtr)
    }

    /// Inserts a new vector into the index.
    ///
    /// This method normalizes the vector, optionally quantizes it based on the index's quantization type,
    /// randomly assigns it a level in the hierarchical structure, searches for appropriate neighbors
    /// at each level, and establishes bi-directional links with them. The operation acquires a write lock
    /// to ensure thread safety.
    ///
    /// - Parameters:
    ///   - array: The vector to insert. Must have exactly `vectorDim` dimensions.
    ///   - nodeId: Optional unique identifier for the node. Use 0 for auto-generated IDs. Default is 0.
    ///   - ef: The size of the dynamic candidate list during insertion. Higher values improve accuracy
    ///         but increase insertion time. Default is 200. Use 0 for the library default (200).
    ///
    /// - Returns: The newly inserted `Node`.
    ///
    /// - Throws: `IndexError.nodeInsertionFailed` if the node cannot be inserted.
    ///
    /// - Precondition: `array.count` must equal `vectorDim`.
    ///
    /// - Note: This operation blocks all concurrent searches. For high-throughput scenarios,
    ///         consider using the optimistic insertion API (`prepareInsert`/`commitInsert`).
    @discardableResult
    public func insert(_ array: [Float], nodeId: Int = 0, ef: Int = 200) throws -> Node {
        precondition(array.count == vectorDim)
        let nodePtr = array.span.withUnsafeBufferPointer { buffer in
            CHNSW.hnsw_insert(ptr, buffer.baseAddress, nil, 0, UInt64(nodeId), nil, Int32(ef))
        }
        guard let nodePtr else {
            throw IndexError.nodeInsertionFailed
        }
        return Node(nodePtr)
    }

    /// Prepares an insertion without modifying the index (optimistic insertion API - phase 1).
    ///
    /// This method performs the read-only search phase of insertion, finding good candidate neighbors
    /// for the new node. It only acquires read locks, allowing concurrent searches to continue.
    /// The actual insertion is completed later by calling `commitInsert(_:)`.
    ///
    /// This two-phase API is useful for high-throughput scenarios where you want to minimize
    /// write lock duration. The prepare phase can be done in parallel with searches, and only
    /// the commit phase requires exclusive access.
    ///
    /// - Parameters:
    ///   - array: The vector to insert. Must have exactly `vectorDim` dimensions.
    ///   - nodeId: Optional unique identifier for the node. Use 0 for auto-generated IDs. Default is 0.
    ///   - ef: The size of the dynamic candidate list during the search. Higher values improve accuracy
    ///         but increase preparation time. Default is 200.
    ///
    /// - Returns: An `InsertContext` containing the pre-computed candidates, or `nil` if preparation fails.
    ///
    /// - Precondition: `array.count` must equal `vectorDim`.
    ///
    /// - Note: The returned context must be consumed by calling `commitInsert(_:)`. If nodes are
    ///         deleted between prepare and commit, the commit may fail and return `nil`.
    public func prepareInsert(_ array: [Float], nodeId: Int = 0, ef: Int = 200) -> InsertContext? {
        precondition(array.count == vectorDim)
        let contextPtr = array.span.withUnsafeBufferPointer { buffer in
            CHNSW.hnsw_prepare_insert(ptr, buffer.baseAddress, nil, 0, UInt64(nodeId), Int32(ef))
        }
        guard let contextPtr else {
            return nil
        }
        return InsertContext(contextPtr)
    }

    /// Commits a prepared insertion (optimistic insertion API - phase 2).
    ///
    /// This method attempts to insert the node prepared by `prepareInsert(_:nodeId:ef:)` into the index.
    /// It uses optimistic concurrency control: if no nodes were deleted between prepare and commit,
    /// the insertion succeeds. Otherwise, it fails and returns `nil`.
    ///
    /// The commit phase acquires a write lock, but typically for much less time than a full
    /// `insert(_:nodeId:ef:)` call, since the neighbor search was already performed during prepare.
    ///
    /// - Parameter context: The insertion context returned by `prepareInsert(_:nodeId:ef:)`.
    ///                      This parameter is consuming, so the context cannot be reused.
    ///
    /// - Returns: The newly inserted `Node`, or `nil` if the commit failed due to concurrent modifications.
    ///
    /// - Note: If this method returns `nil`, the insertion must be retried, typically by falling back
    ///         to the simpler `insert(_:nodeId:ef:)` API or preparing a new context.
    @discardableResult
    public func commitInsert(_ context: consuming InsertContext) -> Node? {
        let nodePtr = CHNSW.hnsw_try_commit_insert(ptr, context.ptr, nil)
        context.discard()
        guard let nodePtr else {
            return nil
        }
        return Node(nodePtr)
    }

    /// Deletes a node from the index.
    ///
    /// This method performs true deletion (not just marking as deleted) by:
    /// 1. Removing all bi-directional links to the node
    /// 2. Reconnecting orphaned neighbors to maintain graph connectivity
    /// 3. Freeing the node's memory
    ///
    /// The operation acquires a write lock to ensure thread safety.
    ///
    /// - Parameter node: The node to delete.
    ///
    /// - Throws: `IndexError.nodeDeletionFailed` if the node cannot be deleted.
    ///
    /// - Note: After deletion, the node reference becomes invalid and should not be used.
    ///         Any existing search results containing this node should be discarded.
    public func delete(_ node: Node) throws {
        if CHNSW.hnsw_delete_node(ptr, node.ptr, nil) == 0 {
            throw IndexError.nodeDeletionFailed
        }
    }
}

// MARK: - Search

extension Index {
    /// Searches for the k-nearest neighbors of a query vector (single-threaded).
    ///
    /// This method performs an approximate nearest neighbor search using the HNSW algorithm.
    /// It starts from the entry point at the highest level and progressively narrows down
    /// to the best candidates at level 0.
    ///
    /// - Parameters:
    ///   - query: The query vector. Must have exactly `vectorDim` dimensions.
    ///   - k: The number of nearest neighbors to return. Default is 3.
    ///   - isQueryVectorNormalized: Whether the query vector is already normalized.
    ///                              If `false`, the vector will be normalized internally. Default is `false`.
    ///
    /// - Returns: An array of `SearchResult` objects sorted by distance (closest first), containing at most `k` items.
    ///
    /// - Precondition: `query.count` must equal `vectorDim`.
    ///
    /// - Note: This method uses slot 0 internally and is not thread-safe for concurrent searches.
    ///         For concurrent access, use `searchConcurrent(_:k:isQueryVectorNormalized:)`.
    ///
    /// - Important: Results are only valid while the index is not being modified. Do not access
    ///              result nodes after calling `insert`, `delete`, or any other mutating method.
    public func search(
        _ query: [Float],
        k: Int = 3,
        isQueryVectorNormalized: Bool = false
    ) -> [SearchResult] {
        search(query, k: k, slot: 0, isQueryVectorNormalized: isQueryVectorNormalized)
    }

    /// Searches for the k-nearest neighbors of a query vector (thread-safe).
    ///
    /// This method is the thread-safe version of `search(_:k:isQueryVectorNormalized:)`.
    /// It acquires a read slot before searching, allowing multiple concurrent searches
    /// while preventing conflicts with write operations.
    ///
    /// The implementation supports up to 32 concurrent search threads. Each thread uses
    /// a dedicated slot with its own epoch tracking for efficient visited node management.
    ///
    /// - Parameters:
    ///   - query: The query vector. Must have exactly `vectorDim` dimensions.
    ///   - k: The number of nearest neighbors to return. Default is 3.
    ///   - isQueryVectorNormalized: Whether the query vector is already normalized.
    ///                              If `false`, the vector will be normalized internally. Default is `false`.
    ///
    /// - Returns: An array of `SearchResult` objects sorted by distance (closest first), containing at most `k` items.
    ///
    /// - Throws: `IndexError.acquireReadSlotFailed` if no read slot could be acquired (all 32 slots are in use).
    ///
    /// - Precondition: `query.count` must equal `vectorDim`.
    ///
    /// - Important: The read lock is held for the duration of the search. Results are only valid
    ///              until the method returns. Do not store node references for later use.
    public func searchConcurrent(
        _ query: [Float],
        k: Int = 3,
        isQueryVectorNormalized: Bool = false
    ) throws -> [SearchResult] {
        let slot = try acquireReadSlot()
        defer { releaseReadSlot(slot) }
        return search(query, k: k, slot: slot, isQueryVectorNormalized: isQueryVectorNormalized)
    }

    /// Internal search implementation with explicit slot parameter.
    ///
    /// - Parameters:
    ///   - query: The query vector.
    ///   - k: The number of nearest neighbors to return.
    ///   - slot: The thread slot to use (0-31). Slot 0 is for single-threaded access.
    ///   - isQueryVectorNormalized: Whether the query vector is already normalized.
    ///
    /// - Returns: An array of search results sorted by distance.
    private func search(
        _ query: [Float],
        k: Int,
        slot: Int,
        isQueryVectorNormalized: Bool
    ) -> [SearchResult] {
        precondition(slot >= 0, "slot must be >= 0")
        precondition(slot < Int(HNSW_MAX_THREADS), "slot must be <\(HNSW_MAX_THREADS)")
        let neighbors = UnsafeMutablePointer<UnsafeMutablePointer<CHNSW.hnswNode>?>
            .allocate(capacity: k)
        let distances = UnsafeMutablePointer<Float>
            .allocate(capacity: k)
        defer {
            neighbors.deallocate()
            distances.deallocate()
        }
        let found = query.span.withUnsafeBufferPointer { buffer in
            CHNSW.hnsw_search(
                ptr,
                buffer.baseAddress,
                UInt32(k),
                neighbors,
                distances,
                UInt32(slot),
                isQueryVectorNormalized ? 1 : 0
            )
        }
        let foundCount = Int(found)
        var results = [SearchResult]()
        results.reserveCapacity(foundCount)
        for index in 0..<foundCount {
            if let nodePtr = neighbors[index] {
                results.append(
                    SearchResult(
                        node: Node(nodePtr),
                        distance: distances[index]
                    )
                )
            }
        }
        return results
    }

    /// Acquires a read slot for thread-safe search operations.
    ///
    /// This method tries to acquire one of the 32 available read slots using a non-blocking
    /// approach first. If all slots are busy, it uses atomic increment to select a slot
    /// and blocks until that slot becomes available.
    ///
    /// - Returns: The acquired slot number (0-31).
    ///
    /// - Throws: `IndexError.acquireReadSlotFailed` if the slot cannot be acquired.
    private func acquireReadSlot() throws -> Int {
        let result = Int(CHNSW.hnsw_acquire_read_slot(ptr))
        if result == -1 {
            throw IndexError.acquireReadSlotFailed
        }
        return result
    }

    /// Releases a previously acquired read slot.
    ///
    /// - Parameter slot: The slot number to release (0-31).
    ///
    /// - Important: This must be called after `acquireReadSlot()` to allow other threads
    ///              to use the slot. Always use `defer` to ensure the slot is released.
    private func releaseReadSlot(_ slot: Int) {
        CHNSW.hnsw_release_read_slot(ptr, Int32(slot))
    }
}

// MARK: - Nodes

extension Index {
    /// Computes the distance between two nodes.
    ///
    /// This method calculates the distance using the index's configured distance metric.
    /// Since vectors are normalized on insertion, the distance represents the cosine distance
    /// (1 - dot product) between the original vectors.
    ///
    /// The computation automatically handles the index's quantization type:
    /// - For unquantized vectors: direct dot product of float vectors
    /// - For Q8 quantization: optimized int8 dot product with range scaling
    /// - For binary quantization: efficient XOR-based comparison
    ///
    /// - Parameters:
    ///   - node1: The first node.
    ///   - node2: The second node.
    ///
    /// - Returns: The distance between the two nodes. Lower values indicate greater similarity.
    ///            The distance is in the range [0, 2], where 0 means identical vectors
    ///            and 2 means opposite directions.
    ///
    /// - Note: This method uses SIMD optimizations (AVX2/AVX512 on x86_64) when available
    ///         and the vector dimension is large enough.
    public func distance(from node1: borrowing Node, to node2: borrowing Node) -> Float {
        CHNSW.hnsw_distance(ptr, node1.ptr, node2.ptr)
    }

    /// Retrieves the vector stored in a node.
    ///
    /// This method returns an approximation of the original vector that was inserted.
    /// The vector is de-quantized (if quantization is enabled) and de-normalized to
    /// restore values close to the original scale.
    ///
    /// - Parameter node: The node whose vector to retrieve.
    ///
    /// - Returns: An array of `Float` values representing the reconstructed vector.
    ///            The array will have exactly `vectorDim` elements.
    ///
    /// - Note: For quantized indexes, the returned vector is an approximation:
    ///   - Q8 quantization: restored from int8 values with some loss of precision
    ///   - Binary quantization: restored to -1.0 or 1.0 per dimension (significant information loss)
    ///   - No quantization: exact original vector (normalized during insertion)
    ///
    /// - Important: For Q8 and binary quantization, the L2 norm is restored but the
    ///              per-component values are approximations. The quantization range and L2 norm
    ///              are stored separately to enable more accurate reconstruction.
    public func vector(of node: borrowing Node) -> [Float] {
        let nodePtr = node.ptr
        let vectorDim = Int(ptr.pointee.vector_dim)
        return [Float](unsafeUninitializedCapacity: vectorDim) { buffer, writtenCount in
            CHNSW.hnsw_get_node_vector(ptr, nodePtr, buffer.baseAddress)
            writtenCount = vectorDim
        }
    }
}
