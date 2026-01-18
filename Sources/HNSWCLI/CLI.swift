import Foundation
import HNSW

@main
enum CLI {
    static func main() async throws {
        let index = try Index(8)
        try index.insert([Float](repeating: 1.0, count: 8))
        try index.insert([Float](repeating: 2.0, count: 8))
        try index.insert([Float](repeating: 3.0, count: 8))

        for node in index.nodes() {
            print(node)
        }

        let result = index.search([Float](repeating: 4.0, count: 8))
        print(result)

        let node1 = try index.insert([Float](repeating: 4.0, count: 8))
        let node2 = try index.insert([Float](repeating: 5.0, count: 8))
        print(index.distance(from: node1, to: node2))
    }
}
