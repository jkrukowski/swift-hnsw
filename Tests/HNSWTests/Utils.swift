import Numerics
import Testing

func allClose<T: Numeric>(
    _ lhs: [T],
    _ rhs: [T],
    absoluteTolerance: T.Magnitude = T.Magnitude.ulpOfOne.squareRoot()
        * T.Magnitude.leastNormalMagnitude,
    relativeTolerance: T.Magnitude = T.Magnitude.ulpOfOne.squareRoot()
) -> Bool where T.Magnitude: FloatingPoint {
    guard lhs.count == rhs.count else {
        Issue.record("Sizes differ: \(lhs.count) vs. \(rhs.count)")
        return false
    }
    for (l, r) in zip(lhs, rhs) {
        guard
            l.isApproximatelyEqual(
                to: r,
                absoluteTolerance: absoluteTolerance,
                relativeTolerance: relativeTolerance
            )
        else {
            Issue.record("Expected \(lhs) to be approximately equal to \(rhs)")
            return false
        }
    }
    return true
}
