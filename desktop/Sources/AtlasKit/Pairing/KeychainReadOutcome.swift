import Security

/// What a Keychain read that may never show UI tells us (D-25, D-29).
///
/// A status of 0 means the read worked with no prompt. A read that needed a
/// prompt must never look like a silent one, so the launch record reports the
/// outcome beside the raw status.
public enum KeychainReadOutcome: Equatable, Sendable {
    case silent
    case notStored
    /// The access list or the partition list asked for a prompt, or a prompt
    /// was refused. This is what a rebuild with a self-signed identity causes.
    case needsInteraction(Int32)
    case failed(Int32)

    public init(status: Int32) {
        switch status {
        case errSecSuccess:
            self = .silent
        case errSecItemNotFound:
            self = .notStored
        case errSecInteractionNotAllowed, errSecInteractionRequired, errSecUserCanceled,
            errSecAuthFailed, errSecNoAccessForItem:
            self = .needsInteraction(status)
        default:
            self = .failed(status)
        }
    }

    /// The text written to `launch.jsonl`.
    public var label: String {
        switch self {
        case .silent: "silent"
        case .notStored: "not_stored"
        case .needsInteraction: "needs_interaction"
        case .failed: "failed"
        }
    }
}

extension SecretStore {
    public func readOutcome() -> KeychainReadOutcome {
        KeychainReadOutcome(status: readStatus())
    }
}
