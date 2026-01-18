import Benchmark
import HNSW

func index(
    of count: Int,
    dimension: Int,
    quantization: HNSW.Quantization? = nil
) throws -> HNSW.Index {
    let data = matrix(of: count, dimension: dimension)
    let index = try HNSW.Index(dimension, quantization: quantization)
    for row in data {
        try index.insert(row)
    }
    return index
}

func matrix(of count: Int, dimension: Int) -> [[Float]] {
    var data = [[Float]]()
    data.reserveCapacity(count)
    for _ in 0..<count {
        data.append(vector(of: dimension))
    }
    return data
}

func vector(of dimension: Int) -> [Float] {
    var vector = [Float]()
    vector.reserveCapacity(dimension)
    for _ in 0..<dimension {
        vector.append(Float.random(in: -1..<1))
    }
    return vector
}

let benchmarks: @Sendable () -> Void = {
    let parameterization = [100, 10_000, 100_000]
    let dimension = 1024

    for count in parameterization {
        Benchmark(
            "Insert",
            configuration: .init(tags: ["count": count.description])
        ) { benchmark in
            let index = try index(of: count, dimension: dimension)

            for _ in benchmark.scaledIterations {
                benchmark.startMeasurement()
                blackHole(try index.insert(vector(of: dimension)))
                benchmark.stopMeasurement()
            }
        }

        Benchmark(
            "Insert Q8",
            configuration: .init(tags: ["count": count.description])
        ) { benchmark in
            let index = try index(of: count, dimension: dimension, quantization: .q8)

            for _ in benchmark.scaledIterations {
                benchmark.startMeasurement()
                blackHole(try index.insert(vector(of: dimension)))
                benchmark.stopMeasurement()
            }
        }

        Benchmark(
            "Insert Binary",
            configuration: .init(tags: ["count": count.description])
        ) { benchmark in
            let index = try index(of: count, dimension: dimension, quantization: .binary)

            for _ in benchmark.scaledIterations {
                benchmark.startMeasurement()
                blackHole(try index.insert(vector(of: dimension)))
                benchmark.stopMeasurement()
            }
        }

        Benchmark(
            "Search",
            configuration: .init(tags: ["count": count.description])
        ) { benchmark in
            let index = try index(of: count, dimension: dimension)

            for _ in benchmark.scaledIterations {
                benchmark.startMeasurement()
                blackHole(index.search(vector(of: dimension), k: 10))
                benchmark.stopMeasurement()
            }
        }

        Benchmark(
            "Search Q8",
            configuration: .init(tags: ["count": count.description])
        ) { benchmark in
            let index = try index(of: count, dimension: dimension, quantization: .q8)

            for _ in benchmark.scaledIterations {
                benchmark.startMeasurement()
                blackHole(index.search(vector(of: dimension), k: 10))
                benchmark.stopMeasurement()
            }
        }

        Benchmark(
            "Search Binary",
            configuration: .init(tags: ["count": count.description])
        ) { benchmark in
            let index = try index(of: count, dimension: dimension, quantization: .binary)

            for _ in benchmark.scaledIterations {
                benchmark.startMeasurement()
                blackHole(index.search(vector(of: dimension), k: 10))
                benchmark.stopMeasurement()
            }
        }

        Benchmark(
            "Serialize",
            configuration: .init(tags: ["count": count.description])
        ) {
            benchmark in
            let index = try index(of: count, dimension: dimension)

            for _ in benchmark.scaledIterations {
                benchmark.startMeasurement()
                blackHole(try index.serialize())
                benchmark.stopMeasurement()
            }
        }

        Benchmark(
            "Serialize Q8",
            configuration: .init(tags: ["count": count.description])
        ) {
            benchmark in
            let index = try index(of: count, dimension: dimension, quantization: .q8)

            for _ in benchmark.scaledIterations {
                benchmark.startMeasurement()
                blackHole(try index.serialize())
                benchmark.stopMeasurement()
            }
        }

        Benchmark(
            "Serialize Binary",
            configuration: .init(tags: ["count": count.description])
        ) {
            benchmark in
            let index = try index(of: count, dimension: dimension, quantization: .binary)

            for _ in benchmark.scaledIterations {
                benchmark.startMeasurement()
                blackHole(try index.serialize())
                benchmark.stopMeasurement()
            }
        }
    }
}
