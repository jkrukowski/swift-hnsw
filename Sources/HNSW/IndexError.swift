public enum IndexError: Error, Equatable {
    case acquireReadSlotFailed
    case initializationFailed
    case nodeInsertionFailed
    case nodeDeletionFailed
    case nodeSerializationFailed
    case deserializationFailed
    case unsupportedSerializationVersion(UInt64)
}
