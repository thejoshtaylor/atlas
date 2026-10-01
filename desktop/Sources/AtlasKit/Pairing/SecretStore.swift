import Synchronization

public enum SecretStoreError: Error, Equatable {
    case keychain(Int32)
    case malformedItem
}

/// Storage for the single pairing of this Mac.
public protocol SecretStore: Sendable {
    func load() throws -> PairingCredentials?
    func save(_ credentials: PairingCredentials) throws
    func delete() throws
    /// OSStatus of a read that may never show UI (D-25). 0 means readable.
    func readStatus() -> Int32
}

/// In-memory store for tests. It keeps one pairing, like the real store.
public final class InMemorySecretStore: SecretStore {
    private struct State {
        var credentials: PairingCredentials?
        var failNextSave: Int32?
    }

    private let state = Mutex(State())

    public init() {}

    /// When set, the next `save` throws `.keychain(status)` and clears this hook.
    public var failNextSave: Int32? {
        get { state.withLock { $0.failNextSave } }
        set { state.withLock { $0.failNextSave = newValue } }
    }

    public func load() throws -> PairingCredentials? {
        state.withLock { $0.credentials }
    }

    public func save(_ credentials: PairingCredentials) throws {
        try state.withLock { s in
            if let status = s.failNextSave {
                s.failNextSave = nil
                throw SecretStoreError.keychain(status)
            }
            s.credentials = credentials
        }
    }

    public func delete() throws {
        state.withLock { $0.credentials = nil }
    }

    public func readStatus() -> Int32 {
        state.withLock { $0.credentials == nil ? -25300 : 0 }
    }
}
