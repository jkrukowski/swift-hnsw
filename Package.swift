// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-hnsw",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
        .watchOS(.v26),
        .tvOS(.v26),
        .visionOS(.v26),
    ],
    products: [
        .executable(
            name: "HNSWCLI",
            targets: ["HNSWCLI"]
        ),
        .library(
            name: "HNSW",
            targets: ["HNSW"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apple/swift-numerics.git",
            from: "1.1.1"
        )
    ],
    targets: [
        .executableTarget(
            name: "HNSWCLI",
            dependencies: [
                "HNSW"
            ]
        ),
        .target(
            name: "HNSW",
            dependencies: [
                "CHNSW"
            ]
        ),
        .target(
            name: "CHNSW"
        ),
        .testTarget(
            name: "HNSWTests",
            dependencies: [
                "HNSW",
                .product(name: "Numerics", package: "swift-numerics"),
            ]
        ),
    ]
)
