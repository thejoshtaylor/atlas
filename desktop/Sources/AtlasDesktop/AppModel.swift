import AppKit
import AtlasKit
import Foundation
import OSLog
import Observation

/// The state the menu bar shows, and the owner of the connection.
///
/// It stays thin: every rule that can be pure lives in AtlasKit (`MenuState`,
/// `PairingFlow`, `ConnectionStateMachine`). It logs the host and the state
/// name only. It never logs a token, a pair link or a header.
@MainActor
@Observable
final class AppModel {
    static let shared = AppModel()

    private(set) var menuState: MenuState
    private(set) var snapshot = ConnectionSnapshot(status: .unpaired, everConnected: false, lastFailure: nil)
    /// The host of the pairing in Keychain. Never the token.
    private(set) var pairedHost: String?
    /// Set when the server revoked this Mac. Cleared at the first Connected.
    private(set) var revokedHost: String?

    /// Plan 14-11 sets this from Location. Away is a menu label only (D-10).
    var away = false {
        didSet { resolveMenu() }
    }

    @ObservationIgnored private let store = KeychainStore()
    @ObservationIgnored let connection: DesktopConnection
    @ObservationIgnored private let systemEvents = SystemEvents()
    @ObservationIgnored private let log = Logger(subsystem: AppIdentity.bundleIdentifier, category: "app")
    @ObservationIgnored private var started = false
    @ObservationIgnored private var activity: NSObjectProtocol?

    private static let revokedHostKey = "atlas.revokedHost"

    private init() {
        let info = Bundle.main.infoDictionary
        let os = ProcessInfo.processInfo.operatingSystemVersion
        let hello = HelloInfo(
            appVersion: info?["CFBundleShortVersionString"] as? String ?? "unbundled",
            osVersion: "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)")
        connection = DesktopConnection(
            transport: URLSessionWebSocketTransport(),
            clock: SystemConnectionClock(),
            store: store,
            helloInfo: hello)
        revokedHost = UserDefaults.standard.string(forKey: Self.revokedHostKey)
        menuState = MenuState.resolve(
            status: nil, revokedHost: UserDefaults.standard.string(forKey: Self.revokedHostKey),
            away: false, now: Date())
    }

    /// Loads the pairing, starts the connection and listens for it. Runs once.
    func start() {
        guard !started else { return }
        started = true

        let stream = connection.snapshots
        Task {
            for await next in stream { handle(next) }
        }
        systemEvents.start(forwardingTo: connection)

        let store = store
        Task {
            // A Keychain read can wait on a prompt after a rebuild (D-29), so it
            // never runs on the main thread.
            let credentials = await Task.detached(priority: .userInitiated) { () -> PairingCredentials? in
                do {
                    return try store.load()
                } catch {
                    return nil
                }
            }.value
            guard let credentials else {
                log.info("No pairing was loaded from Keychain.")
                return
            }
            pairedHost = credentials.host
            log.info("Starting the connection to \(credentials.host, privacy: .public).")
            await connection.start(credentials: credentials)
        }
    }

    // MARK: - Connection snapshots

    private func handle(_ next: ConnectionSnapshot) {
        snapshot = next
        if case .connected(let host, _) = next.status {
            pairedHost = host
            clearRevokedHost()
            holdActivity()
        } else {
            releaseActivity()
        }
        resolveMenu()
        log.info("The connection is now \(self.menuState.accessibilityLabel, privacy: .public).")
    }

    private func resolveMenu() {
        menuState = MenuState.resolve(
            status: snapshot.status, revokedHost: revokedHost, away: away, now: Date())
    }

    private func clearRevokedHost() {
        guard revokedHost != nil else { return }
        revokedHost = nil
        UserDefaults.standard.removeObject(forKey: Self.revokedHostKey)
    }

    // MARK: - App Nap

    /// While connected, App Nap must not starve the heartbeat (RESEARCH Pattern 9).
    private func holdActivity() {
        guard activity == nil else { return }
        activity = ProcessInfo.processInfo.beginActivity(
            options: .userInitiatedAllowingIdleSystemSleep, reason: "ATLAS connection")
    }

    private func releaseActivity() {
        guard let activity else { return }
        ProcessInfo.processInfo.endActivity(activity)
        self.activity = nil
    }
}
