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
    private(set) var pairedHost: String? {
        didSet { if pairedHost != oldValue { refreshHostLocality() } }
    }
    /// Set when the server revoked this Mac. Cleared at the first Connected.
    private(set) var revokedHost: String?

    /// The steps of pairing (D-03). The setup window renders it.
    private(set) var flow = PairingFlow()
    /// True while the Pair form replaces the step list in the setup window.
    private(set) var showingPairForm = false

    /// Set from Location. Away is a menu label only and never changes the
    /// connection (D-10).
    var away = false {
        didSet { resolveMenu() }
    }

    /// Accessibility, read live. Local Network comes from the connection.
    let permissions = PermissionMonitor()
    /// A rough location for Home or Away. It stays on this Mac (D-11).
    let location = LocationController()
    /// The Launch at login toggle. On by default, and it applies on Continue (D-19).
    var launchAtLogin: Bool {
        didSet { UserDefaults.standard.set(launchAtLogin, forKey: Self.launchAtLoginKey) }
    }
    private(set) var loginState: LoginItemState
    /// True when the paired host is on the home network (a name or an address).
    private var hostIsLocal = false

    @ObservationIgnored private let loginItem = LoginItem()
    @ObservationIgnored private let store = KeychainStore()
    @ObservationIgnored let connection: DesktopConnection
    @ObservationIgnored private let systemEvents = SystemEvents()
    @ObservationIgnored private let log = Logger(subsystem: AppIdentity.bundleIdentifier, category: "app")
    @ObservationIgnored private var started = false
    @ObservationIgnored private var activity: NSObjectProtocol?
    /// Connection commands run one after another, in the order they were given.
    @ObservationIgnored private var connectionTail: Task<Void, Never>?
    /// A pair link can arrive before the saved pairing is read. It waits, so a
    /// Mac that is already paired never skips the Replace sheet (T-14-40).
    @ObservationIgnored private var credentialsLoaded = false
    @ObservationIgnored private var pendingLink: URL?

    private static let revokedHostKey = "atlas.revokedHost"
    private static let launchAtLoginKey = "setup.launchAtLogin"
    private static let setupSeenKey = "setup.seen"
    private static let localNetworkGrantedKey = "setup.localNetworkGranted"

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
        launchAtLogin = UserDefaults.standard.object(forKey: Self.launchAtLoginKey) as? Bool ?? true
        loginState = LoginItem().status()
        menuState = MenuState.resolve(
            status: nil, revokedHost: UserDefaults.standard.string(forKey: Self.revokedHostKey),
            away: false, now: Date())
        permissions.onActivate = { [weak self] in self?.refreshSetupState() }
        location.onChange = { [weak self] in self?.updateAway() }
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
            credentialsLoaded = true
            if let credentials {
                pairedHost = credentials.host
                log.info("Starting the connection to \(credentials.host, privacy: .public).")
                enqueue { await $0.start(credentials: credentials) }
            } else {
                log.info("No pairing was loaded from Keychain.")
                openSetup(focus: .pair)
            }
            if let link = pendingLink {
                pendingLink = nil
                receivePairLink(link)
            }
        }
    }

    // MARK: - Pairing (D-03)

    /// An `atlas://pair` link. Nothing is saved until the operator confirms the
    /// host. The URL is parsed and dropped, never logged.
    func receivePairLink(_ url: URL) {
        guard credentialsLoaded else {
            pendingLink = url
            return
        }
        let result: Result<PairingTarget, PairingLinkError>
        do throws(PairingLinkError) {
            result = .success(try PairingLink.parse(url))
        } catch {
            result = .failure(error)
        }
        dispatch(.linkOpened(result, currentHost: pairedHost))
        openSetup(focus: .pairForm)
    }

    /// The Pair form's "Pair" button: a link on its own, or a server and token.
    func submit(link: String, server: String, token: String) {
        let trimmed = link.trimmingCharacters(in: .whitespacesAndNewlines)
        let result: Result<PairingTarget, PairingLinkError>
        do throws(PairingLinkError) {
            if trimmed.isEmpty {
                result = .success(try PairingTarget.manual(server: server, token: token))
            } else if let url = URL(string: trimmed) {
                result = .success(try PairingLink.parse(url))
            } else {
                result = .failure(.notAtlasLink)
            }
        } catch {
            result = .failure(error)
        }
        dispatch(.linkOpened(result, currentHost: pairedHost))
    }

    func confirm() {
        dispatch(.confirmed)
        showingPairForm = false
    }

    func replace() {
        dispatch(.replaceChosen)
        showingPairForm = false
    }

    func keepCurrent() {
        dispatch(.keepCurrent)
        showingPairForm = false
    }

    func cancel() {
        dispatch(.cancelled)
    }

    func openPairForm() {
        dispatch(.cancelled)
        showingPairForm = true
    }

    func closePairForm() {
        dispatch(.cancelled)
        showingPairForm = false
    }

    private func dispatch(_ event: PairingFlowEvent) {
        for effect in flow.handle(event) { run(effect) }
    }

    private func run(_ effect: PairingFlowEffect) {
        switch effect {
        case .saveToken(let target):
            do {
                try store.save(target.credentials)
                dispatch(.saveSucceeded)
            } catch {
                log.error("The pairing could not be saved to Keychain.")
                dispatch(.saveFailed)
            }
        case .startConnection(let target):
            pairedHost = target.host
            let credentials = target.credentials
            enqueue { await $0.start(credentials: credentials) }
        case .stopCurrentConnection:
            enqueue { await $0.stop() }
        case .removeSavedToken:
            try? store.delete()
            pairedHost = nil
            log.info("The token of a refused pairing was removed.")
        }
    }

    private func enqueue(_ command: @escaping @Sendable (DesktopConnection) async -> Void) {
        let previous = connectionTail
        let connection = connection
        connectionTail = Task {
            await previous?.value
            await command(connection)
        }
    }

    // MARK: - Setup window

    func openSetup(focus: SetupFocus) {
        switch focus {
        case .pair:
            if !flowAwaitsChoice { showingPairForm = false }
        case .pairForm:
            showingPairForm = true
        }
        SetupWindowController.shared.show(focus: focus)
    }

    /// The window is about to be visible: refresh every step and start polling.
    func setupWindowWillShow() {
        refreshSetupState()
        permissions.startPolling()
    }

    func setupWindowDidClose() {
        permissions.stopPolling()
        if flowAwaitsChoice { dispatch(.cancelled) }
        showingPairForm = false
    }

    /// Continue: register the login item from the toggle, close the window and
    /// remember that setup was seen (D-19). It is safe to press twice.
    func continueSetup() {
        loginState = loginItem.apply(toggleOn: launchAtLogin)
        UserDefaults.standard.set(true, forKey: Self.setupSeenKey)
        SetupWindowController.shared.close()
    }

    func openLoginItems() {
        loginItem.openApproval()
    }

    private func refreshSetupState() {
        permissions.refresh()
        location.refresh()
        loginState = loginItem.status()
    }

    private func refreshHostLocality() {
        guard let host = pairedHost else {
            hostIsLocal = false
            return
        }
        Task {
            let local = await Task.detached { PermissionMonitor.isLocalNetworkHost(host) }.value
            if pairedHost == host { hostIsLocal = local }
        }
    }

    // MARK: - Setup steps

    private var localNetwork: LocalNetworkStatus {
        LocalNetworkStatus.resolve(
            everConnected: snapshot.everConnected,
            probe: permissions.localNetworkProbe,
            lastFailure: snapshot.lastFailure,
            hostIsLocal: hostIsLocal,
            previouslyGranted: UserDefaults.standard.bool(forKey: Self.localNetworkGrantedKey))
    }

    private var pairStepState: PairStepState {
        if case .connecting(let host, _) = flow.state { return .connecting(host: host) }
        if case .saving(let target) = flow.state { return .connecting(host: target.host) }
        if let host = pairedHost { return .paired(host: host) }
        if revokedHost != nil { return .revoked }
        return .notPaired
    }

    var setupInputs: SetupInputs {
        SetupInputs(
            pair: pairStepState,
            axTrusted: permissions.axTrusted,
            localNetwork: localNetwork,
            location: location.auth,
            homeSavedAt: location.homeSavedAt,
            loginItem: loginState,
            launchAtLogin: launchAtLogin)
    }

    var setupProgress: SetupProgress {
        SetupProgress.evaluate(setupInputs)
    }

    private var flowAwaitsChoice: Bool {
        switch flow.state {
        case .confirm, .replace: true
        default: false
        }
    }

    // MARK: - Connection snapshots

    private func handle(_ next: ConnectionSnapshot) {
        snapshot = next
        if next.everConnected { UserDefaults.standard.set(true, forKey: Self.localNetworkGrantedKey) }
        let wasPairing: Bool
        switch flow.state {
        case .saving, .connecting: wasPairing = true
        default: wasPairing = false
        }
        dispatch(.connection(next))
        if case .paired = flow.state { showingPairForm = false }
        if case .revoked(let host) = next.status, !wasPairing { runRevokeFlow(host: host) }
        if case .connected(let host, _) = next.status {
            pairedHost = host
            clearRevokedHost()
            holdActivity()
        } else {
            releaseActivity()
        }
        if case .offline = next.status { location.requestFixIfOffline() }
        updateAway()
        resolveMenu()
        log.info("The connection is now \(self.menuState.accessibilityLabel, privacy: .public).")
    }

    /// D-21, D-27. The actor has already stopped retrying and deleted the token.
    /// A Mac refused during a fresh pairing never comes here.
    private func runRevokeFlow(host: String) {
        pairedHost = nil
        revokedHost = host
        UserDefaults.standard.set(host, forKey: Self.revokedHostKey)
        resolveMenu()
        log.info("The server revoked this Mac: \(host, privacy: .public).")
        openSetup(focus: .pair)
        RevokeNotifier.postOnce()
    }

    /// Away needs the Mac offline, Location allowed, a home point and a fix more
    /// than 1 km away (D-10). It changes the menu label only.
    private func updateAway() {
        var connected = false
        if case .connected = snapshot.status { connected = true }
        away = HomeAway.isAway(
            current: location.current, home: location.home,
            locationAllowed: location.auth == .allowed, connected: connected)
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
