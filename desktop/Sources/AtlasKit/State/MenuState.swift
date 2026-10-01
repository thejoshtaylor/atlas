import Foundation

/// What the menu bar shows (D-20): one of six states, the icon and the first
/// menu line. A pure value, so every rule is a unit test.
///
/// `away` is an input and never derived from the connection (D-10). It only
/// applies when the link is offline.
public struct MenuState: Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        case unpaired, connecting, connected, offline, away, revoked
    }

    public let kind: Kind
    public let firstLine: String
    /// The filled globe shows only while Connected.
    public let iconFilled: Bool
    public let accessibilityLabel: String

    /// The most characters of a host the first menu line shows (280 pt).
    public static let hostLimit = 36

    private static let dot = " \u{00B7} "
    private static let ellipsis = "\u{2026}"

    public static func resolve(
        status: LinkStatus?,
        revokedHost: String?,
        away: Bool,
        now: Date,
        calendar: Calendar = .current,
        locale: Locale = .current
    ) -> MenuState {
        func time(_ date: Date) -> String {
            formatTime(date, now: now, calendar: calendar, locale: locale)
        }
        func offlineLine(_ label: String, host: String, lastConnected: Date?) -> String {
            var line = label + dot + truncateMiddle(host)
            if let lastConnected { line += dot + "last connected " + time(lastConnected) }
            return line
        }

        switch status ?? .unpaired {
        case .unpaired:
            if revokedHost != nil { return make(.revoked, "Revoked \u{2014} pair again") }
            return make(.unpaired, "Not paired")
        case .revoked:
            return make(.revoked, "Revoked \u{2014} pair again")
        case .connecting(let host):
            return make(.connecting, "Connecting to " + truncateMiddle(host) + ellipsis)
        case .connected(let host, let since):
            return make(.connected, "Connected to " + truncateMiddle(host) + dot + "since " + time(since))
        case .offline(let host, let lastConnected):
            if away {
                return make(.away, offlineLine("Away", host: host, lastConnected: lastConnected))
            }
            return make(.offline, offlineLine("Offline", host: host, lastConnected: lastConnected))
        case .versionMismatch(let host):
            // D-20 has six states, so a version mismatch reads as Offline.
            return make(.offline, offlineLine("Offline", host: host, lastConnected: nil))
        }
    }

    /// Keeps the start and the end of a long host and puts "…" between them.
    public static func truncateMiddle(_ host: String, limit: Int = hostLimit) -> String {
        guard limit > 1, host.count > limit else { return host }
        let kept = limit - 1
        let head = (kept + 1) / 2
        let tail = kept / 2
        return String(host.prefix(head)) + ellipsis + String(host.suffix(tail))
    }

    /// A time on the same calendar day as `now` shows the short time only.
    /// Another day shows the short date and the short time.
    public static func formatTime(_ date: Date, now: Date, calendar: Calendar, locale: Locale) -> String {
        let formatter = DateFormatter()
        formatter.locale = locale
        formatter.calendar = calendar
        formatter.timeZone = calendar.timeZone
        formatter.dateStyle = calendar.isDate(date, inSameDayAs: now) ? .none : .short
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    private static func make(_ kind: Kind, _ firstLine: String) -> MenuState {
        let name: String
        switch kind {
        case .unpaired: name = "Not paired"
        case .connecting: name = "Connecting"
        case .connected: name = "Connected"
        case .offline: name = "Offline"
        case .away: name = "Away"
        case .revoked: name = "Revoked"
        }
        return MenuState(
            kind: kind, firstLine: firstLine, iconFilled: kind == .connected,
            accessibilityLabel: "ATLAS, " + name)
    }
}
