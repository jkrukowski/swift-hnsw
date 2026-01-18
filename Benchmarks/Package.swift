// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "benchmarks",
    platforms: [
        .macOS(.v26)
    ],
    dependencies: [
        .package(name: "swift-hnsw", path: "../"),
        .package(url: "https://github.com/ordo-one/package-benchmark", from: "1.29.7"),
    ],
    targets: [
        .executableTarget(
            name: "HNSWBenchmarks",
            dependencies: [
                .product(name: "Benchmark", package: "package-benchmark"),
                .product(name: "HNSW", package: "swift-hnsw"),
            ],
            path: "Benchmarks",
            plugins: [
                .plugin(name: "BenchmarkPlugin", package: "package-benchmark")
            ]
        )
    ]
)
