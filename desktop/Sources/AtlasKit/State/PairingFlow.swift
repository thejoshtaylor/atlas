import Foundation

/// Why a pairing did not work. `message(host:)` gives the UI-SPEC copy, which
/// names the problem and the next step. None of it is an alert.
public enum PairingFailure: Equatable, Sendable {
    case notAtlasLink
    case insecure
    /// The server refused the token on a fresh pairing (D-27 generic refusal).
    case refused
    case unreachable(host: String)
    case versionMismatch
    case keychainWrite

    /// `host` is the name to show. The caller truncates a long host.
    public func message(host: String) -> String {
        switch self {
        case .notAtlasLink:
            "That is not an ATLAS pair link. Open the Macs page, add a Mac, and copy its pair link."
        case .insecure:
            "This link is not secure. ATLAS pairs over wss:// only. Open the Macs page over https and copy a new link."
        case .refused:
            "The server did not accept this token. Add a new Mac on the Macs page and pair again."
        case .unreachable:
            "Could not reach \(host). Check that this Mac is on the right network and try again."
        case .versionMismatch:
            "This app and the server use different versions. Run install.sh again to update the app."
        case .keychainWrite:
            "Could not save the token to Keychain. Unlock your login keychain and try again."
        }
    }
}

public enum PairingFlowState: Equatable, Sendable {
    case idle
    /// A link or typed token waits for the operator to confirm the host (D-03).
    case confirm(PairingTarget)
    /// A link arrived while this Mac is paired. Keep Current is the default.
    case replace(current: String, target: PairingTarget)
    case saving(PairingTarget)
    case connecting(host: String, error: PairingFailure?)
    case paired(host: String)
    case failed(PairingFailure)
}

public enum PairingFlowEvent: Equatable, Sendable {
    case linkOpened(Result<PairingTarget, PairingLinkError>, currentHost: String?)
    case confirmed
    case cancelled
    case keepCurrent
    case replaceChosen
    case saveSucceeded
    case saveFailed
    /// A snapshot of the connection while a pairing is in flight.
    case connection(ConnectionSnapshot)
}

public enum PairingFlowEffect: Equatable, Sendable {
    case saveToken(PairingTarget)
    case startConnection(PairingTarget)
    case removeSavedToken
    case stopCurrentConnection
}

/// The steps of pairing as a pure reducer. The shell runs the effects.
///
/// A fresh pairing whose first dial the server refuses removes the token that
/// was just saved and shows the refused copy. It never runs the revoke flow,
/// because the server never accepted this token (T-14-42).
public struct PairingFlow: Sendable {
    public private(set) var state: PairingFlowState = .idle
    /// Snapshots from before this pairing started can still be in the stream.
    /// The flow ignores them until the first `connecting` snapshot arrives.
    private var sawAttempt = false

    public init() {}

    public static func canSubmit(link: String, server: String, token: String) -> Bool {
        func filled(_ text: String) -> Bool {
            !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return filled(link) || (filled(server) && filled(token))
    }

    public mutating func handle(_ event: PairingFlowEvent) -> [PairingFlowEffect] {
        switch event {
        case .linkOpened(let result, let currentHost):
            return linkOpened(result, currentHost: currentHost)
        case .confirmed:
            guard case .confirm(let target) = state else { return [] }
            state = .saving(target)
            return [.saveToken(target)]
        case .replaceChosen:
            guard case .replace(_, let target) = state else { return [] }
            state = .saving(target)
            return [.stopCurrentConnection, .saveToken(target)]
        case .keepCurrent:
            guard case .replace = state else { return [] }
            state = .idle
            return []
        case .cancelled:
            switch state {
            case .confirm, .replace, .failed: state = .idle
            default: break
            }
            return []
        case .saveSucceeded:
            guard case .saving(let target) = state else { return [] }
            state = .connecting(host: target.host, error: nil)
            sawAttempt = false
            return [.startConnection(target)]
        case .saveFailed:
            guard case .saving = state else { return [] }
            state = .failed(.keychainWrite)
            return []
        case .connection(let snapshot):
            return connection(snapshot)
        }
    }

    private mutating func linkOpened(
        _ result: Result<PairingTarget, PairingLinkError>, currentHost: String?
    ) -> [PairingFlowEffect] {
        switch result {
        case .success(let target):
            if let currentHost {
                state = .replace(current: currentHost, target: target)
            } else {
                state = .confirm(target)
            }
        case .failure(let error):
            state = .failed(error == .insecure ? .insecure : .notAtlasLink)
        }
        return []
    }

    private mutating func connection(_ snapshot: ConnectionSnapshot) -> [PairingFlowEffect] {
        guard case .connecting(let host, _) = state else { return [] }
        if case .connecting = snapshot.status { sawAttempt = true }
        guard sawAttempt else { return [] }

        switch snapshot.status {
        case .connected:
            state = .paired(host: host)
        case .versionMismatch:
            state = .failed(.versionMismatch)
        case .revoked:
            state = .failed(.refused)
            return [.removeSavedToken, .stopCurrentConnection]
        case .offline, .connecting:
            switch snapshot.lastFailure {
            case .refused?:
                state = .failed(.refused)
                return [.removeSavedToken, .stopCurrentConnection]
            case .unreachable?:
                state = .connecting(host: host, error: .unreachable(host: host))
            default:
                break
            }
        case .unpaired:
            break
        }
        return []
    }
}
