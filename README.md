# `swift-hnsw`

[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjkrukowski%2Fswift-hnsw%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/jkrukowski/swift-hnsw)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fjkrukowski%2Fswift-hnsw%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/jkrukowski/swift-hnsw)

A Swift wrapper around high-performance [Redis HNSW implementation](https://github.com/redis/redis/blob/unstable/modules/vector-sets/hnsw.h) for approximate nearest neighbor search in vector spaces.

## Installation

Add the dependency to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/jkrukowski/swift-hnsw.git", from: "0.0.1")
]
```

## Usage

### Basic Example

```swift
import HNSW

// Create an index for 4-dimensional vectors
let index = try Index(4)

// Insert vectors with optional custom IDs
try index.insert([1.0, 0.0, 0.0, 0.0], nodeId: 1)
try index.insert([0.0, 1.0, 0.0, 0.0], nodeId: 2)
try index.insert([0.9, 0.1, 0.0, 0.0], nodeId: 3)

// Search for nearest neighbors
let results = index.search([1.0, 0.0, 0.0, 0.0], k: 3)
for result in results {
    print("Node ID: \(result.node.id), Distance: \(result.distance)")
}
```

### Quantization

Reduce memory usage with vector quantization:

```swift
// 8-bit quantization
let indexQ8 = try Index(4, quantization: .q8)

// Binary quantization
let indexBinary = try Index(4, quantization: .binary)
```

### Serialization

Save and load indexes:

```swift
// Serialize index to bytes
let data = try index.serialize()

// Deserialize from bytes
let loadedIndex = try Index.deserialize(from: data)
```

### Node Management

```swift
// Delete a node
try index.delete(node)

// Get node count
print("Total nodes: \(index.nodeCount)")

// Iterate over all nodes
for node in index.nodes() {
    let vector = index.vector(of: node)
    print("Node \(node.id): \(vector)")
}

// Calculate distance between nodes
let distance = index.distance(from: node1, to: node2)
```

## Acknowledgements

This project uses HNSW implementation taken from [Redis](https://github.com/redis/redis/blob/unstable/modules/vector-sets/hnsw.h) originally authored by Salvatore Sanfilippo.

## Code Formatting

This project uses [swift-format](https://github.com/swiftlang/swift-format). To format the code run:

```bash
swift format . -i -r --configuration .swift-format
```
