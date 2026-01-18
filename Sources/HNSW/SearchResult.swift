public struct SearchResult {
    public let node: Node
    public let distance: Float

    public init(node: Node, distance: Float) {
        self.node = node
        self.distance = distance
    }
}
