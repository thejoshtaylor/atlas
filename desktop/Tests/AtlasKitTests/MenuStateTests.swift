import Foundation
import Testing

@testable import AtlasKit

@Suite struct MenuStateTests {
    private let calendar: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        return c
    }()
    private let locale = Locale(identifier: "en_US")
    private let now = Date(timeIntervalSince1970: 1_790_000_000)

    private func resolve(
        _ status: LinkStatus?, revokedHost: String? = nil, away: Bool = false
    ) -> MenuState {
        MenuState.resolve(
            status: status, revokedHost: revokedHost, away: away, now: now,
            calendar: calendar, locale: locale)
    }

    private func clock(_ date: Date, sameDay: Bool) -> String {
        let f = DateFormatter()
        f.locale = locale
        f.calendar = calendar
        f.timeZone = calendar.timeZone
        f.dateStyle = sameDay ? .none : .short
        f.timeStyle = .short
        return f.string(from: date)
    }

    private var earlierToday: Date { now.addingTimeInterval(-600) }
    private var twoDaysAgo: Date { now.addingTimeInterval(-2 * 86_400) }

    // MARK: the six states

    @Test func nothingPairedIsNotPairedWithTheOutlineIcon() {
        let state = resolve(nil)
        #expect(state.kind == .unpaired)
        #expect(state.firstLine == "Not paired")
        #expect(!state.iconFilled)
        #expect(state.accessibilityLabel == "ATLAS, Not paired")
    }

    @Test func theUnpairedStatusMatchesNoStatus() {
        #expect(resolve(.unpaired) == resolve(nil))
    }

    @Test func connectingNamesTheHostAndKeepsTheOutlineIcon() {
        let state = resolve(.connecting(host: "svr.test"))
        #expect(state.kind == .connecting)
        #expect(state.firstLine == "Connecting to svr.test\u{2026}")
        #expect(!state.iconFilled)
    }

    @Test func connectedIsTheOnlyStateWithTheFilledIcon() {
        let state = resolve(.connected(host: "svr.test", since: earlierToday))
        #expect(state.kind == .connected)
        #expect(state.firstLine == "Connected to svr.test \u{00B7} since " + clock(earlierToday, sameDay: true))
        #expect(state.iconFilled)
        #expect(state.accessibilityLabel == "ATLAS, Connected")

        let others: [LinkStatus?] = [
            nil, .connecting(host: "h"), .offline(host: "h", lastConnected: nil),
            .revoked(host: "h"), .versionMismatch(host: "h"),
        ]
        for status in others { #expect(!resolve(status).iconFilled) }
        #expect(!resolve(.offline(host: "h", lastConnected: nil), away: true).iconFilled)
    }

    @Test func offlineWithNoPriorConnectionShowsNoTime() {
        let state = resolve(.offline(host: "svr.test", lastConnected: nil))
        #expect(state.kind == .offline)
        #expect(state.firstLine == "Offline \u{00B7} svr.test")
    }

    @Test func offlineWithADateShowsLastConnected() {
        let state = resolve(.offline(host: "svr.test", lastConnected: earlierToday))
        #expect(
            state.firstLine
                == "Offline \u{00B7} svr.test \u{00B7} last connected " + clock(earlierToday, sameDay: true))
    }

    @Test func awayReplacesOfflineOnly() {
        let offline = LinkStatus.offline(host: "svr.test", lastConnected: earlierToday)
        let state = resolve(offline, away: true)
        #expect(state.kind == .away)
        #expect(
            state.firstLine
                == "Away \u{00B7} svr.test \u{00B7} last connected " + clock(earlierToday, sameDay: true))
        #expect(state.accessibilityLabel == "ATLAS, Away")

        #expect(resolve(.connected(host: "h", since: earlierToday), away: true).kind == .connected)
        #expect(resolve(.connecting(host: "h"), away: true).kind == .connecting)
        #expect(resolve(nil, away: true).kind == .unpaired)
        #expect(resolve(.revoked(host: "h"), away: true).kind == .revoked)
    }

    @Test func awayWithNoPriorConnectionShowsNoTime() {
        let state = resolve(.offline(host: "svr.test", lastConnected: nil), away: true)
        #expect(state.firstLine == "Away \u{00B7} svr.test")
    }

    @Test func revokedReadsPairAgain() {
        let fromStatus = resolve(.revoked(host: "svr.test"))
        #expect(fromStatus.kind == .revoked)
        #expect(fromStatus.firstLine == "Revoked \u{2014} pair again")
        #expect(!fromStatus.iconFilled)
        #expect(resolve(nil, revokedHost: "svr.test") == fromStatus)
        #expect(resolve(.unpaired, revokedHost: "svr.test") == fromStatus)
    }

    @Test func aLiveStatusWinsOverARememberedRevoke() {
        #expect(resolve(.connecting(host: "new.test"), revokedHost: "svr.test").kind == .connecting)
        #expect(resolve(.connected(host: "new.test", since: earlierToday), revokedHost: "svr.test").kind == .connected)
    }

    @Test func aVersionMismatchReadsAsOffline() {
        let state = resolve(.versionMismatch(host: "svr.test"))
        #expect(state.kind == .offline)
        #expect(state.firstLine == "Offline \u{00B7} svr.test")
    }

    // MARK: time

    @Test func aTimeOnTheSameDayShowsTheShortTimeOnly() {
        let text = MenuState.formatTime(earlierToday, now: now, calendar: calendar, locale: locale)
        #expect(text == clock(earlierToday, sameDay: true))
    }

    @Test func aTimeOnAnotherDayShowsDateAndTime() {
        let text = MenuState.formatTime(twoDaysAgo, now: now, calendar: calendar, locale: locale)
        #expect(text == clock(twoDaysAgo, sameDay: false))
        #expect(text != clock(twoDaysAgo, sameDay: true))
    }

    // MARK: long hosts

    @Test func aSixtyCharacterHostIsMiddleTruncatedToTheLimit() {
        let host = String(repeating: "a", count: 30) + String(repeating: "b", count: 30)
        let shown = MenuState.truncateMiddle(host)
        #expect(host.count == 60)
        #expect(shown.count == MenuState.hostLimit)
        #expect(shown.contains("\u{2026}"))
        #expect(shown.hasPrefix("aaaa"))
        #expect(shown.hasSuffix("bbbb"))
    }

    @Test func aShortHostIsNotChanged() {
        #expect(MenuState.truncateMiddle("svr.test") == "svr.test")
        let exactly = String(repeating: "x", count: MenuState.hostLimit)
        #expect(MenuState.truncateMiddle(exactly) == exactly)
    }

    @Test func offlineAndAwayLinesKeepTheTimeWhenTheHostIsLong() {
        let host = String(repeating: "h", count: 60)
        let when = earlierToday
        let suffix = " \u{00B7} last connected " + clock(when, sameDay: true)
        let offline = resolve(.offline(host: host, lastConnected: when))
        let away = resolve(.offline(host: host, lastConnected: when), away: true)
        #expect(offline.firstLine.hasSuffix(suffix))
        #expect(away.firstLine.hasSuffix(suffix))
        #expect(!offline.firstLine.contains(host))
    }
}
