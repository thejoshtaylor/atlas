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
    /// OSStatus of a read that may never show UI (D-25). 0 means readable with
    /// no prompt. Map it with `KeychainReadOutcome`.
    func readStatus() -> Int32
}

/// In-memory store for tests. It keeps one pairing, like the real store.
public final class InMemorySecretStore: SecretStore {
    private struct State {
        var credentials: PairingCredentials?
        var failNextSave: Int32?
        var failNextLoad: Int32?
        var readStatusOverride: Int32?
    }

    private let state = Mutex(State())

    public init() {}

    /// When set, the next `save` throws `.keychain(status)` and clears this hook.
    public var failNextSave: Int32? {
        get { state.withLock { $0.failNextSave } }
        set { state.withLock { $0.failNextSave = newValue } }
    }

    /// When set, the next `load` throws `.keychain(status)` and clears this hook.
    public var failNextLoad: Int32? {
        get { state.withLock { $0.failNextLoad } }
        set { state.withLock { $0.failNextLoad = newValue } }
    }

    /// When set, `readStatus` returns it, so tests can play a read that needs a prompt.
    public var readStatusOverride: Int32? {
        get { state.withLock { $0.readStatusOverride } }
        set { state.withLock { $0.readStatusOverride = newValue } }
    }

    public func load() throws -> PairingCredentials? {
        try state.withLock { s in
            if let status = s.failNextLoad {
                s.failNextLoad = nil
                throw SecretStoreError.keychain(status)
            }
            return s.credentials
        }
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
        state.withLock { s in
            if let forced = s.readStatusOverride { return forced }
            return s.credentials == nil ? -25300 : 0
        }
    }
}
