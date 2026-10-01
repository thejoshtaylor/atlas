import Foundation
import Testing

@testable import AtlasKit

// The ring rules of the panel reducer, with an injected clock (plan 15-07, task 3).
// Times are seconds after `t0`.

private let t0 = Date(timeIntervalSince1970: 1_800_000_000)
private func at(_ seconds: Double) -> Date { t0.addingTimeInterval(seconds) }

private struct PanelRig {
    var state = PanelState()

    @discardableResult
    mutating func send(_ event: PanelEvent, at seconds: Double = 0) -> [PanelEffect] {
        PanelReducer.reduce(&state, event, now: at(seconds))
    }
}

private func ring(_ id: Int, _ kind: String = "timer", _ label: String = "pasta") -> PanelEvent {
    .timerRinging(TimerRinging(timerId: id, kind: kind, label: label))
}
private func stopped(_ id: Int) -> PanelEvent { .timerStopped(TimerStopped(timerId: id)) }
private func wake(_ id: String) -> PanelEvent { .wakeConfirmed(WakeConfirmed(turnId: id)) }
private func partial(_ id: String, _ text: String) -> PanelEvent {
    .partial(TranscriptPartial(turnId: id, text: text))
}
private func ended(_ id: String) -> PanelEvent {
    .turnEnded(TurnEnded(turnId: id, outcome: "completed", followUpWindowMs: 0, playbackMsLeft: 0))
}
private func ringingRig(_ id: Int = 12) -> PanelRig {
    var rig = PanelRig()
    rig.send(ring(id))
    return rig
}

@Suite struct PanelReducerRingTests {
    // MARK: Ring opens (D-14, CARD-02)

    @Test func aRingOpensThePanelWithNoWakeWord() {
        var rig = PanelRig()
        let effects = rig.send(ring(12))
        #expect(rig.state.visible)
        #expect(rig.state.turn == nil)
        #expect(rig.state.ring == PanelRing(timerId: 12, kind: .timer, label: "pasta", stop: .idle))
        #expect(rig.state.nextDeadline == at(130))
        #expect(effects.contains(.show))
        #expect(effects.contains(.announce("Timer ringing, pasta", .high)))
    }

    @Test func theAnnouncementUsesTheKindAndTheEmptyLabelText() {
        var empty = PanelRig()
        #expect(empty.send(ring(1, "timer", "")).contains(.announce("Timer ringing, Time is up", .high)))
        var alarm = PanelRig()
        #expect(alarm.send(ring(2, "alarm", "wake up")).contains(.announce("Alarm ringing, wake up", .high)))
        #expect(alarm.state.ring?.kind == .alarm)
        var emptyAlarm = PanelRig()
        #expect(
            emptyAlarm.send(ring(3, "alarm", "")).contains(.announce("Alarm ringing, Alarm is ringing", .high)))
        var unknown = PanelRig()
        unknown.send(ring(4, "metronome", "x"))
        #expect(unknown.state.ring?.kind == .timer)
    }

    @Test func aRingLabelIsSanitizedAndKeepsMarkdownAsLiteralText() {
        var rig = PanelRig()
        rig.send(ring(1, "timer", "  [pay](https://example.com)\u{202E}  now  "))
        #expect(rig.state.ring?.label == "[pay](https://example.com) now")
        var long = PanelRig()
        long.send(ring(2, "timer", String(repeating: "z", count: 120)))
        #expect(long.state.ring?.label.unicodeScalars.count == 120)
    }

    @Test func aRingDuringATurnTakesOverWhileTheTurnKeepsItsState() {
        var rig = PanelRig()
        rig.send(wake("A"))
        rig.send(partial("A", "hello"), at: 1)
        let effects = rig.send(ring(12), at: 2)
        #expect(rig.state.ring?.timerId == 12)
        #expect(rig.state.turn?.transcript == "hello")
        #expect(!effects.contains(.show))
        rig.send(partial("A", "hello again"), at: 3)
        #expect(rig.state.turn?.transcript == "hello again")
    }

    @Test func aSecondRingReplacesTheIdAndTheLabel() {
        var rig = ringingRig(12)
        rig.send(ring(13, "alarm", "wake up"), at: 5)
        #expect(rig.state.ring == PanelRing(timerId: 13, kind: .alarm, label: "wake up", stop: .idle))
        #expect(rig.state.nextDeadline == at(135))
    }

    @Test func aRepeatOfTheSameRingKeepsTheStopState() {
        var rig = ringingRig(12)
        rig.send(.stopClicked, at: 1)
        let effects = rig.send(ring(12), at: 2)
        #expect(rig.state.ring?.stop == .pending)
        #expect(!effects.contains(where: { if case .announce = $0 { true } else { false } }))
    }

    // MARK: Stop (CARD-03, D-15)

    @Test func aStopClickSendsOneRequestAndShowsPending() {
        var rig = ringingRig(12)
        let effects = rig.send(.stopClicked, at: 10)
        #expect(effects.contains(.sendTimerStop(timerId: 12)))
        #expect(rig.state.ring?.stop == .pending)
        #expect(rig.state.nextDeadline == at(13))
        let again = rig.send(.stopClicked, at: 11)
        #expect(!again.contains(.sendTimerStop(timerId: 12)))
        #expect(rig.state.nextDeadline == at(13))
    }

    @Test func aStopClickWithNoRingDoesNothing() {
        var rig = PanelRig()
        #expect(rig.send(.stopClicked) == [.wakeAt(nil)])
    }

    @Test func noAnswerForThreeSecondsEnablesStopAgain() {
        var rig = ringingRig(12)
        rig.send(.stopClicked, at: 10)
        rig.send(.deadline, at: 12.9)
        #expect(rig.state.ring?.stop == .pending)
        rig.send(.deadline, at: 13)
        #expect(rig.state.ring?.stop == .idle)
        #expect(rig.send(.stopClicked, at: 14).contains(.sendTimerStop(timerId: 12)))
    }

    // MARK: Stopped (D-16)

    @Test func stoppedDuringAnActiveTurnReturnsToTheTurn() {
        var rig = PanelRig()
        rig.send(wake("A"))
        rig.send(partial("A", "hello"), at: 1)
        rig.send(ring(12), at: 2)
        rig.send(stopped(12), at: 3)
        #expect(rig.state.ring == nil)
        #expect(rig.state.turn?.transcript == "hello")
        #expect(rig.state.visible)
    }

    @Test func stoppedWithNoTurnShowsStoppedAndHidesFourSecondsLater() {
        var rig = ringingRig(12)
        rig.send(.stopClicked, at: 10)
        let effects = rig.send(stopped(12), at: 11)
        #expect(rig.state.ring?.stop == .stopped)
        #expect(effects.contains(.relayout))
        #expect(rig.state.nextDeadline == at(15))
        let hide = rig.send(.deadline, at: 15)
        #expect(hide.contains(.hide(animated: true)))
        #expect(!rig.state.visible)
        #expect(rig.state.ring == nil)
    }

    @Test func aStoppedHideWaitsForTheHover() {
        var rig = ringingRig(12)
        rig.send(.hoverChanged(true), at: 1)
        rig.send(stopped(12), at: 2)
        #expect(rig.state.nextDeadline == nil)
        rig.send(.hoverChanged(false), at: 10)
        #expect(rig.state.nextDeadline == at(14))
    }

    @Test func stoppedForAnotherTimerChangesNothing() {
        var rig = ringingRig(12)
        let before = rig.state
        rig.send(stopped(99), at: 1)
        #expect(rig.state == before)
        var none = PanelRig()
        none.send(stopped(12))
        #expect(none.state == PanelState())
    }

    @Test func aNewRingAfterStoppedCancelsTheStoppedHide() {
        var rig = ringingRig(12)
        rig.send(stopped(12), at: 1)  // hide due at 5
        rig.send(ring(13, "timer", "tea"), at: 2)
        #expect(rig.state.ring == PanelRing(timerId: 13, kind: .timer, label: "tea", stop: .idle))
        #expect(rig.state.hideAt == nil)
        rig.send(.deadline, at: 5)
        #expect(rig.state.visible)
    }

    @Test func aTurnThatStartsOverAStoppedRingViewTakesTheViewBack() {
        var rig = ringingRig(12)
        rig.send(stopped(12), at: 1)
        rig.send(wake("B"), at: 2)
        #expect(rig.state.ring == nil)
        #expect(rig.state.turn?.turnId == "B")
        #expect(rig.state.hideAt == nil)
    }

    // MARK: Cap and timers (D-08)

    @Test func aLostStoppedEndsAtTheHundredThirtySecondCap() {
        var rig = ringingRig(12)
        rig.send(.deadline, at: 129)
        #expect(rig.state.visible)
        let effects = rig.send(.deadline, at: 130)
        #expect(effects.contains(.hide(animated: true)))
        #expect(!rig.state.visible)
        #expect(rig.state.ring == nil)
        #expect(rig.state.nextDeadline == nil)
        // The capped ring does not open again for the same id.
        rig.send(ring(12), at: 131)
        #expect(!rig.state.visible)
    }

    @Test func theCapKeepsThePanelWhenATurnRemains() {
        var rig = PanelRig()
        rig.send(ring(12))
        rig.send(wake("A"), at: 120)
        rig.send(partial("A", "hello"), at: 125)
        rig.send(.deadline, at: 130)
        #expect(rig.state.ring == nil)
        #expect(rig.state.visible)
        #expect(rig.state.turn?.transcript == "hello")
    }

    @Test func theWatchdogNeverHidesAnOpenRingView() {
        var rig = PanelRig()
        rig.send(wake("A"))
        rig.send(ring(12), at: 1)
        rig.send(.deadline, at: 30)
        #expect(rig.state.visible)
        #expect(rig.state.ring?.timerId == 12)
        #expect(rig.state.turn == nil)
        #expect(rig.state.watchdogAt == nil)
    }

    @Test func aDueHideTimerOfAnEndedTurnDropsTheTurnAndKeepsTheRing() {
        var rig = PanelRig()
        rig.send(wake("A"))
        rig.send(ended("A"), at: 1)  // hide due at 5
        rig.send(ring(12), at: 2)
        rig.send(.deadline, at: 5)
        #expect(rig.state.visible)
        #expect(rig.state.ring?.timerId == 12)
        #expect(rig.state.turn == nil)
        #expect(rig.state.nextDeadline == at(132))
    }

    // MARK: Close, connection loss, display removal

    @Test func closeHidesTheRingAndAClosedRingNeverOpensAgainButANewIdDoes() {
        var rig = ringingRig(12)
        let effects = rig.send(.closePressed, at: 1)
        #expect(effects.contains(.hide(animated: true)))
        #expect(!rig.state.visible)
        rig.send(ring(12), at: 2)
        #expect(!rig.state.visible)
        #expect(rig.state.ring == nil)
        let reopen = rig.send(ring(13), at: 3)
        #expect(reopen.contains(.show))
        #expect(rig.state.ring?.timerId == 13)
    }

    // A repeating alarm keeps one timer id for every ring (CR-01).
    @Test func aClosedRepeatingAlarmShowsAgainWhenItsNextRingComes() {
        var rig = ringingRig(5)
        rig.send(.closePressed, at: 1)
        rig.send(stopped(5), at: 2)
        let effects = rig.send(ring(5, "alarm", "wake up"), at: 86_400)
        #expect(effects.contains(.show))
        #expect(rig.state.visible)
        #expect(rig.state.ring?.timerId == 5)
    }

    @Test func aStoppedFrameForAClosedRingClearsTheMemoryEvenWithNoRingInView() {
        var rig = ringingRig(5)
        rig.send(.closePressed, at: 1)
        #expect(rig.state.ring == nil)
        rig.send(stopped(5), at: 2)
        #expect(rig.state.closedTimerId == nil)
    }

    @Test func aStoppedFrameForAnotherTimerKeepsTheClosedMemory() {
        var rig = ringingRig(5)
        rig.send(.closePressed, at: 1)
        rig.send(stopped(6), at: 2)
        rig.send(ring(5), at: 3)
        #expect(!rig.state.visible)
    }

    @Test func aLostStoppedAfterCloseForgetsTheClosedRingAfterTheCap() {
        var rig = ringingRig(5)
        rig.send(.closePressed, at: 10)
        rig.send(ring(5), at: 139)  // a replay inside the cap still stays hidden
        #expect(!rig.state.visible)
        rig.send(ring(5), at: 141)  // the cap since the close has passed
        #expect(rig.state.visible)
        #expect(rig.state.ring?.timerId == 5)
    }

    @Test func aCappedRingIsForgottenOneCapLater() {
        var rig = ringingRig(12)
        rig.send(.deadline, at: 130)
        rig.send(ring(12), at: 259)
        #expect(!rig.state.visible)
        rig.send(ring(12), at: 261)
        #expect(rig.state.visible)
    }

    @Test func aDroppedConnectionHidesTheRingView() {
        var rig = ringingRig(12)
        let effects = rig.send(.connectionLost, at: 1)
        #expect(effects.contains(.hide(animated: true)))
        #expect(!rig.state.visible)
        #expect(rig.state.ring == nil)
        #expect(rig.state.nextDeadline == nil)
    }

    @Test func aRingShowsAgainAfterADisplayWasRemoved() {
        var rig = ringingRig(12)
        rig.send(.displayRemoved, at: 1)
        #expect(!rig.state.visible)
        let effects = rig.send(ring(13), at: 2)
        #expect(effects.first == .show)
        #expect(rig.state.visible)
        #expect(rig.state.ring?.timerId == 13)
    }
}
