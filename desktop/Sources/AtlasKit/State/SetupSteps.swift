import Foundation

/// The five setup steps, in the fixed order the window shows them (D-18).
public enum SetupStepID: CaseIterable, Sendable {
    case pair, accessibility, localNetwork, location, launchAtLogin

    public var title: String {
        switch self {
        case .pair: "Pair"
        case .accessibility: "Accessibility"
        case .localNetwork: "Local Network"
        case .location: "Location and Set home"
        case .launchAtLogin: "Launch at login"
        }
    }

    /// Pair, Accessibility and Local Network gate Continue. Location and Launch
    /// at login never do (D-12, D-18, D-19).
    public var isRequired: Bool {
        switch self {
        case .pair, .accessibility, .localNetwork: true
        case .location, .launchAtLogin: false
        }
    }

    public var description: String {
        switch self {
        case .pair:
            "Connect this Mac to your ATLAS server."
        case .accessibility:
            "ATLAS needs this to move and size windows later. It does nothing with it yet."
        case .localNetwork:
            "ATLAS needs this to reach a server on your home network."
        case .location:
            "Optional. ATLAS uses a rough location to tell Home from Away in the menu bar. "
                + "Your location stays on this Mac and is never sent to the server."
        case .launchAtLogin:
            "Open ATLAS when you log in, so it stays connected."
        }
    }
}

// MARK: - Local Network

/// A measured read of the Local Network permission, when one exists. The spike
/// (14-SPIKE.md) found `NWPath.unsatisfiedReason` unreliable on macOS 26, so the
/// shell never supplies one. The case stays so a later OS can.
public enum LocalNetworkProbe: Equatable, Sendable {
    case ready, denied
}

/// The Local Network step. macOS has no API that returns the permission, and
/// the spike never saw `localNetworkDenied`, so the state comes from what a
/// connection attempt did (UI-SPEC fallback):
///
/// - a connection that succeeded, or a server that answered even to refuse,
///   proves the permission: Granted;
/// - a past success stays Granted through a later outage, so a down server never
///   reads as a missing permission;
/// - a first attempt to a LAN host that fails is Not granted;
/// - until one of those exists, Checking…
public enum LocalNetworkStatus: Equatable, Sendable {
    case checking, granted, notGranted

    public static func resolve(
        everConnected: Bool,
        probe: LocalNetworkProbe?,
        lastFailure: DialFailure? = nil,
        hostIsLocal: Bool = false,
        previouslyGranted: Bool = false
    ) -> LocalNetworkStatus {
        if probe == .denied { return .notGranted }
        if probe == .ready || everConnected { return .granted }
        switch lastFailure {
        case .refused?, .closed?:
            // The server answered, so the network path works.
            return .granted
        default:
            break
        }
        if previouslyGranted { return .granted }
        if lastFailure == .unreachable, hostIsLocal { return .notGranted }
        return .checking
    }

    public var stateText: String {
        switch self {
        case .checking: "Checking\u{2026}"
        case .granted: "Granted"
        case .notGranted: "Not granted"
        }
    }
}

/// Whether a host name or address is on the home network. A failed connection
/// to such a host is the only failure that points at the Local Network
/// permission. This is a name and address check only. A public name that
/// resolves to a private address is handled by the shell, which resolves it and
/// calls `isLocalAddress`.
public enum LocalNetworkHost {
    public static func looksLocal(_ host: String) -> Bool {
        var name = host.lowercased()
        if name.hasPrefix("["), name.hasSuffix("]") { name = String(name.dropFirst().dropLast()) }
        if name.hasSuffix(".") { name.removeLast() }
        guard !name.isEmpty else { return false }
        if isLocalAddress(name) { return true }
        // A bare name such as "atlas" only resolves through local DNS or mDNS.
        if !name.contains(".") && !name.contains(":") { return true }
        for suffix in [".local", ".lan", ".home", ".home.arpa", ".internal", ".localdomain"]
        where name.hasSuffix(suffix) { return true }
        return false
    }

    /// Private, link-local and loopback IPv4 and IPv6 literals.
    public static func isLocalAddress(_ literal: String) -> Bool {
        let value = literal.lowercased()
        let parts = value.split(separator: ".", omittingEmptySubsequences: false)
        let octets = parts.compactMap { UInt8($0) }
        if parts.count == 4, octets.count == 4 {
            let (a, b) = (octets[0], octets[1])
            return a == 10 || a == 127 || (a == 172 && (16...31).contains(b))
                || (a == 192 && b == 168) || (a == 169 && b == 254)
        }
        guard value.contains(":") else { return false }
        if value == "::1" { return true }
        if value.hasPrefix("fe8") || value.hasPrefix("fe9") || value.hasPrefix("fea") || value.hasPrefix("feb") {
            return true
        }
        return value.hasPrefix("fc") || value.hasPrefix("fd")
    }
}

// MARK: - Location and login item

public enum LocationAuth: Equatable, Sendable {
    case notAsked, allowed, denied

    public var stateText: String {
        switch self {
        case .notAsked: "Not asked"
        case .allowed: "Allowed"
        case .denied: "Denied"
        }
    }
}

/// `SMAppService.mainApp.status`, reduced to what the setup window needs.
public enum LoginItemState: Equatable, Sendable {
    case on, off, needsApproval

    /// `SMAppService.Status.rawValue`: 0 notRegistered, 1 enabled,
    /// 2 requiresApproval, 3 notFound.
    public static func from(rawStatus: Int) -> LoginItemState {
        switch rawStatus {
        case 1: .on
        case 2: .needsApproval
        default: .off
        }
    }

    /// Registration is idempotent: it runs only when the item is off. A pending
    /// approval is already registered, and registering again would add nothing.
    public static func shouldRegister(toggleOn: Bool, current: LoginItemState) -> Bool {
        toggleOn && current == .off
    }

    public static func shouldUnregister(toggleOn: Bool, current: LoginItemState) -> Bool {
        !toggleOn && (current == .on || current == .needsApproval)
    }
}

// MARK: - Progress

/// The Pair row, which the pairing flow and the revoke flow drive.
public enum PairStepState: Equatable, Sendable {
    case notPaired
    case connecting(host: String)
    case paired(host: String)
    case revoked
}

public struct SetupInputs: Equatable, Sendable {
    public var pair: PairStepState
    public var axTrusted: Bool
    public var localNetwork: LocalNetworkStatus
    public var location: LocationAuth
    public var homeSavedAt: Date?
    public var loginItem: LoginItemState
    /// The Launch at login toggle. On by default (D-19).
    public var launchAtLogin: Bool

    public init(
        pair: PairStepState = .notPaired,
        axTrusted: Bool = false,
        localNetwork: LocalNetworkStatus = .checking,
        location: LocationAuth = .notAsked,
        homeSavedAt: Date? = nil,
        loginItem: LoginItemState = .off,
        launchAtLogin: Bool = true
    ) {
        self.pair = pair
        self.axTrusted = axTrusted
        self.localNetwork = localNetwork
        self.location = location
        self.homeSavedAt = homeSavedAt
        self.loginItem = loginItem
        self.launchAtLogin = launchAtLogin
    }
}

public enum SetupStepStatus: Equatable, Sendable {
    /// `checkmark.circle.fill`, green.
    case done
    /// `circle`, secondary.
    case notDone
    /// `exclamationmark.circle`, secondary: the operator must act in System Settings.
    case needsAction
}

public struct SetupStep: Equatable, Sendable, Identifiable {
    public let id: SetupStepID
    public let status: SetupStepStatus
    public let stateText: String
    /// Lines under the state text, such as "Saved 9:41 AM".
    public let notes: [String]
    /// The row button. `nil` for the Launch at login toggle row with no approval pending.
    public let buttonLabel: String?
    public let accessibilityLabel: String

    public var title: String { id.title }
    public var description: String { id.description }
    public var isRequired: Bool { id.isRequired }
}

public struct SetupProgress: Equatable, Sendable {
    public let steps: [SetupStep]
    /// True only when Pair, Accessibility and Local Network are done (D-12, D-18).
    public let continueEnabled: Bool

    /// The menu shows "Finish Setup…" while this is false and "Setup…" once true.
    public var requiredDone: Bool { continueEnabled }

    public static let locationDeniedNote = "ATLAS works without it. The menu bar shows Offline instead of Away."

    public static func evaluate(
        _ inputs: SetupInputs,
        now: Date = Date(),
        calendar: Calendar = .current,
        locale: Locale = .current
    ) -> SetupProgress {
        let steps = SetupStepID.allCases.map { step($0, inputs, now, calendar, locale) }
        let enabled = steps.filter(\.isRequired).allSatisfy { $0.status == .done }
        return SetupProgress(steps: steps, continueEnabled: enabled)
    }

    private static func step(
        _ id: SetupStepID, _ i: SetupInputs, _ now: Date, _ calendar: Calendar, _ locale: Locale
    ) -> SetupStep {
        let status: SetupStepStatus
        let text: String
        var notes: [String] = []
        var accessibilityState: String?
        let button: String?

        switch id {
        case .pair:
            switch i.pair {
            case .notPaired: (status, text, button) = (.notDone, "Not paired", "Pair\u{2026}")
            case .connecting(let host):
                (status, text, button) = (
                    .notDone, "Connecting to " + MenuState.truncateMiddle(host) + "\u{2026}", "Pair\u{2026}"
                )
            case .paired(let host):
                (status, text, button) = (.done, "Paired with " + MenuState.truncateMiddle(host), "Re-pair\u{2026}")
            case .revoked: (status, text, button) = (.needsAction, "Revoked by the server", "Pair\u{2026}")
            }
        case .accessibility:
            (status, text, button) = i.axTrusted
                ? (.done, "Granted", "Open Settings") : (.needsAction, "Not granted", "Open Settings")
        case .localNetwork:
            text = i.localNetwork.stateText
            button = "Open Settings"
            switch i.localNetwork {
            case .granted: status = .done
            case .checking: status = .notDone
            case .notGranted: status = .needsAction
            }
        case .location:
            text = i.location.stateText
            switch i.location {
            case .notAsked: (status, button) = (.notDone, "Allow Location")
            case .allowed: (status, button) = (.done, "Set Home Here")
            case .denied:
                (status, button) = (.needsAction, "Open Settings")
                notes.append(locationDeniedNote)
            }
            let homeText = i.homeSavedAt == nil ? "Home not set" : "Home set"
            notes.insert(homeText, at: 0)
            if let saved = i.homeSavedAt {
                notes.append("Saved " + MenuState.formatTime(saved, now: now, calendar: calendar, locale: locale))
            }
            accessibilityState = text + ", " + lowercasedFirst(homeText)
        case .launchAtLogin:
            if !i.launchAtLogin {
                (status, text, button) = (.notDone, "Off", nil)
            } else if i.loginItem == .needsApproval {
                (status, text, button) = (.needsAction, "Needs approval", "Open Login Items")
            } else {
                (status, text, button) = (i.loginItem == .on ? .done : .notDone, "On", nil)
            }
        }

        var label = id.title + ", " + lowercasedFirst(accessibilityState ?? text) + "."
        if let button { label += " " + button.replacingOccurrences(of: "\u{2026}", with: "") + "." }
        return SetupStep(
            id: id, status: status, stateText: text, notes: notes, buttonLabel: button,
            accessibilityLabel: label)
    }

    private static func lowercasedFirst(_ text: String) -> String {
        guard let first = text.first else { return text }
        return first.lowercased() + text.dropFirst()
    }
}
