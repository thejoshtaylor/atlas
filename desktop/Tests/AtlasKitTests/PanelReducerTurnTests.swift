import Foundation
import Testing

@testable import AtlasKit

// The turn rules of the panel reducer, with an injected clock (plan 15-07, task 2).
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

private func wake(_ id: String) -> PanelEvent { .wakeConfirmed(WakeConfirmed(turnId: id)) }
private func word(_ id: String, _ word: String) -> PanelEvent {
    .turnState(TurnStateMessage(turnId: id, state: word))
}
private func partial(_ id: String, _ text: String) -> PanelEvent {
    .partial(TranscriptPartial(turnId: id, text: text))
}
private func final(_ id: String, _ text: String) -> PanelEvent {
    .final(TranscriptFinal(turnId: id, text: text))
}
private func textCard(_ id: String, _ text: String, cardId: String = "reply") -> PanelEvent {
    .card(
        CardMessage(
            turnId: id, cardId: cardId, kind: "text", fallbackText: text, data: .text(TextCardData(text: text))))
}
private func ended(_ id: String, _ outcome: String = "completed", window: Int = 0, playback: Int = 0) -> PanelEvent {
    .turnEnded(TurnEnded(turnId: id, outcome: outcome, followUpWindowMs: window, playbackMsLeft: playback))
}
private func openRig(_ id: String = "A") -> PanelRig {
    var rig = PanelRig()
    rig.send(wake(id))
    return rig
}

@Suite struct PanelReducerTurnTests {
    // MARK: Open and replace

    @Test func aWakeFromHiddenOpensTheEmptyListeningPanel() {
        var rig = PanelRig()
        let effects = rig.send(wake("A"))
        #expect(rig.state.visible)
        #expect(rig.state.turn == PanelTurn(turnId: "A"))
        #expect(rig.state.turn?.word == .listening)
        #expect(rig.state.turn?.transcript == "")
        #expect(effects.contains(.show))
        #expect(effects.contains(.announce("ATLAS is listening", .medium)))
        #expect(rig.state.nextDeadline == at(30))
    }

    @Test func aNewWakeWhileOpenReplacesTheContentKeepsTheScreenAndResetsTheTimers() {
        var rig = openRig("A")
        rig.send(partial("A", "hey atlas what"), at: 5)
        let effects = rig.send(wake("B"), at: 10)
        #expect(rig.state.turn == PanelTurn(turnId: "B"))
        #expect(!effects.contains(.show))
        #expect(effects.contains(.relayout))
        #expect(rig.state.nextDeadline == at(40))
        // Later frames of the replaced turn change nothing.
        rig.send(partial("A", "late"), at: 11)
        rig.send(word("A", "thinking"), at: 11)
        rig.send(final("A", "late"), at: 11)
        #expect(rig.state.turn == PanelTurn(turnId: "B"))
    }

    @Test func theSameWakeAgainChangesNothing() {
        var rig = openRig("A")
        rig.send(partial("A", "hello"), at: 1)
        rig.send(wake("A"), at: 2)
        #expect(rig.state.turn?.transcript == "hello")
    }

    // MARK: Follow-up turns (D-05)

    @Test func aFollowUpTurnStartsInPlaceOnceTheCurrentTurnEnded() {
        var rig = openRig("A")
        rig.send(ended("A"), at: 1)
        let effects = rig.send(word("C", "listening"), at: 2)
        #expect(rig.state.turn == PanelTurn(turnId: "C"))
        #expect(rig.state.visible)
        #expect(!effects.contains(.show))
        #expect(effects.contains(.relayout))
        #expect(rig.state.hideAt == nil)
        #expect(rig.state.retiredTurnIds.contains("A"))
    }

    @Test func aFollowUpTurnCanStartFromAPartial() {
        var rig = openRig("A")
        rig.send(ended("A", window: 6800), at: 1)
        rig.send(partial("C", "and tomorrow"), at: 3)
        #expect(rig.state.turn?.turnId == "C")
        #expect(rig.state.turn?.transcript == "and tomorrow")
    }

    @Test func anUnseenTurnIsIgnoredWhileTheCurrentTurnIsActive() {
        var rig = openRig("A")
        rig.send(word("C", "listening"), at: 1)
        rig.send(partial("C", "parallel"), at: 1)
        #expect(rig.state.turn == PanelTurn(turnId: "A"))
    }

    @Test func anUnseenTurnIsIgnoredWhileHidden() {
        var rig = PanelRig()
        let effects = rig.send(word("C", "listening"))
        rig.send(partial("C", "parallel"))
        #expect(rig.state.turn == nil)
        #expect(!rig.state.visible)
        #expect(effects == [.wakeAt(nil)])
    }

    @Test func aReplacedTurnIsNeverAdoptedAgain() {
        var rig = openRig("A")
        rig.send(wake("B"), at: 1)
        rig.send(ended("B"), at: 2)
        rig.send(word("A", "listening"), at: 3)
        #expect(rig.state.turn?.turnId == "B")
        rig.send(word("D", "listening"), at: 3)
        #expect(rig.state.turn?.turnId == "D")
    }

    @Test func theMemoryOfRetiredTurnsIsBounded() {
        var rig = openRig("T0")
        for index in 1...20 { rig.send(wake("T\(index)"), at: Double(index)) }
        #expect(rig.state.retiredTurnIds.count == PanelTiming.retiredTurnMemory)
        #expect(!rig.state.retiredTurnIds.contains("T0"))
        #expect(rig.state.retiredTurnIds.contains("T19"))
    }

    @Test func framesAfterTurnEndedChangeNothing() {
        var rig = openRig("A")
        rig.send(ended("A", "no_speech"), at: 1)
        let before = rig.state
        rig.send(partial("A", "late"), at: 2)
        rig.send(word("A", "speaking"), at: 2)
        rig.send(textCard("A", "late card"), at: 2)
        #expect(rig.state.turn == before.turn)
    }

    // MARK: Transcript (PANEL-03)

    @Test func aPartialSetsTheTranscriptAndIsNotFinal() {
        var rig = openRig()
        rig.send(partial("A", "hey atlas what"), at: 1)
        #expect(rig.state.turn?.transcript == "hey atlas what")
        #expect(rig.state.turn?.transcriptIsFinal == false)
    }

    @Test func partialsFasterThanTwentyASecondApplyTheNewestOnTheDeadline() {
        var rig = openRig()
        rig.send(partial("A", "a"), at: 0)
        rig.send(partial("A", "b"), at: 0.01)
        rig.send(partial("A", "c"), at: 0.02)
        #expect(rig.state.turn?.transcript == "a")
        #expect(rig.state.nextDeadline == at(0.05))
        let effects = rig.send(.deadline, at: 0.05)
        #expect(rig.state.turn?.transcript == "c")
        #expect(effects.contains(.relayout))
        #expect(rig.state.nextDeadline == at(30.02))
    }

    @Test func aLongPartialKeepsTheNewestFiveHundredCharactersWithALeadingEllipsis() {
        var rig = openRig()
        rig.send(partial("A", String(repeating: "x", count: 500) + String(repeating: "y", count: 500)), at: 1)
        let text = rig.state.turn?.transcript ?? ""
        #expect(text.unicodeScalars.count == 500)
        #expect(text.unicodeScalars.first == "\u{2026}")
        #expect(text.dropFirst().allSatisfy { $0 == "y" })
    }

    @Test func aFinalReplacesTheTranscriptMarksItFinalAndDropsAHeldPartial() {
        var rig = openRig()
        rig.send(partial("A", "hey atlas what"), at: 1)
        rig.send(partial("A", "hey atlas what time"), at: 1.01)
        rig.send(final("A", "what time is it"), at: 1.02)
        #expect(rig.state.turn?.transcript == "what time is it")
        #expect(rig.state.turn?.transcriptIsFinal == true)
        rig.send(.deadline, at: 1.2)
        #expect(rig.state.turn?.transcript == "what time is it")
        rig.send(partial("A", "late partial"), at: 2)
        #expect(rig.state.turn?.transcript == "what time is it")
    }

    // MARK: State word (PANEL-04, D-09)

    @Test func theStateWordFollowsTheServerAndAnUnknownWordKeepsTheLastState() {
        var rig = openRig()
        rig.send(word("A", "thinking"), at: 1)
        #expect(rig.state.turn?.word == .thinking)
        rig.send(word("A", "speaking"), at: 2)
        #expect(rig.state.turn?.word == .speaking)
        rig.send(word("A", "dancing"), at: 3)
        #expect(rig.state.turn?.word == .speaking)
        rig.send(word("A", "done"), at: 4)
        #expect(rig.state.turn?.word == .speaking)
    }

    // MARK: Cards (D-11)

    @Test func aTextCardSetsTheCardAndAnnouncesIt() {
        var rig = openRig()
        let effects = rig.send(textCard("A", "It is sunny."), at: 1)
        #expect(rig.state.turn?.cardId == "reply")
        #expect(rig.state.turn?.cardText == "It is sunny.")
        #expect(effects.contains(.announce("It is sunny.", .medium)))
        #expect(effects.contains(.relayout))
    }

    @Test func aCardWithTheSameIdReplacesTheEarlierOne() {
        var rig = openRig()
        rig.send(textCard("A", "First."), at: 1)
        rig.send(textCard("A", "Second."), at: 2)
        #expect(rig.state.turn?.cardText == "Second.")
        #expect(rig.state.turn?.cardId == "reply")
    }

    @Test func aCardOfAnUnknownKindShowsItsFallbackText() {
        var rig = openRig()
        let card = CardMessage(
            turnId: "A", cardId: "w", kind: "weather", fallbackText: "Sunny, 20 degrees", data: .unknown)
        rig.send(.card(card), at: 1)
        #expect(rig.state.turn?.cardText == "Sunny, 20 degrees")
        #expect(rig.state.turn?.cardId == "w")
    }

    @Test func aCardWithNoUsableTextChangesNothing() {
        var rig = openRig()
        rig.send(textCard("A", "Kept."), at: 1)
        let before = rig.state
        let blank = CardMessage(
            turnId: "A", cardId: "reply", kind: "text", fallbackText: "   ", data: .text(TextCardData(text: "   ")))
        let effects = rig.send(.card(blank), at: 2)
        #expect(rig.state == before)
        #expect(effects == [.wakeAt(before.nextDeadline)])
        let empty = CardMessage(turnId: "A", cardId: "reply", kind: "text", fallbackText: "", data: .text(TextCardData(text: "")))
        rig.send(.card(empty), at: 3)
        #expect(rig.state.turn?.cardText == "Kept.")
    }

    @Test func aCardIsKeptAsLiteralTextWhenItHoldsMarkdownLinkSyntax() {
        var rig = openRig()
        let link = "see [the site](https://example.com) now"
        rig.send(textCard("A", link), at: 1)
        rig.send(partial("A", link), at: 2)
        #expect(rig.state.turn?.cardText == link)
        #expect(rig.state.turn?.transcript == link)
    }

    // MARK: Outcome and hide timing (D-06, PANEL-05)

    @Test func aCompletedTurnEndsAtDoneAndHidesAfterPlaybackPlusFourSeconds() {
        var rig = openRig()
        rig.send(ended("A", "completed", playback: 2000), at: 10)
        #expect(rig.state.turn?.word == .done)
        #expect(rig.state.turn?.outcome == nil)
        #expect(rig.state.turn?.ended == true)
        #expect(rig.state.nextDeadline == at(16))
        let effects = rig.send(.deadline, at: 16)
        #expect(effects.contains(.hide(animated: true)))
        #expect(!rig.state.visible)
        #expect(rig.state.turn == nil)
    }

    @Test func eachOutcomeGivesItsFixedLineAndTheWordStaysDone() {
        let table: [(String, PanelOutcomeLine?)] = [
            ("no_speech", .noSpeech), ("failed", .failed), ("stopped", .stopped),
            ("completed", nil), ("exploded", nil),
        ]
        for (wire, line) in table {
            var rig = openRig()
            rig.send(ended("A", wire), at: 1)
            #expect(rig.state.turn?.outcome == line, "outcome \(wire)")
            #expect(rig.state.turn?.word == .done, "word for \(wire)")
        }
    }

    @Test func aFollowUpWindowShowsListeningKeepsTheCardAndHidesAfterPlaybackPlusTheWindow() {
        var rig = openRig()
        rig.send(textCard("A", "Do you want the forecast?"), at: 1)
        rig.send(ended("A", window: 6800, playback: 1000), at: 10)
        #expect(rig.state.turn?.word == .listening)
        #expect(rig.state.turn?.cardText == "Do you want the forecast?")
        #expect(rig.state.nextDeadline == at(17.8))
    }

    @Test func aServerWindowOfFifteenMinutesIsClampedToOneHundredTwentySeconds() {
        var rig = openRig()
        rig.send(ended("A", window: 900_000), at: 10)
        #expect(rig.state.nextDeadline == at(130))
    }

    // MARK: Hover (D-04)

    @Test func aHoverResumesTheHideTimerWithTheTimeLeftAndAtLeastTwoSeconds() {
        var rig = openRig()
        rig.send(ended("A"), at: 0)  // hide due at 4
        rig.send(.hoverChanged(true), at: 3.5)  // 0.5 s left
        #expect(rig.state.nextDeadline == nil)
        rig.send(.hoverChanged(false), at: 10)
        #expect(rig.state.nextDeadline == at(12))
    }

    @Test func aHoverThatEndsWithMoreThanTwoSecondsLeftKeepsTheTimeLeft() {
        var rig = openRig()
        rig.send(ended("A"), at: 0)  // hide due at 4
        rig.send(.hoverChanged(true), at: 1)  // 3 s left
        rig.send(.hoverChanged(false), at: 2)
        #expect(rig.state.nextDeadline == at(5))
    }

    @Test func aHoverPausesTheWatchdogToo() {
        var rig = openRig()  // watchdog at 30
        rig.send(.hoverChanged(true), at: 10)  // 20 s left
        #expect(rig.state.nextDeadline == nil)
        rig.send(.deadline, at: 100)
        #expect(rig.state.visible)
        rig.send(.hoverChanged(false), at: 100)
        #expect(rig.state.nextDeadline == at(120))
    }

    @Test func aTurnThatEndsDuringAHoverWaitsForTheHoverToEnd() {
        var rig = openRig()
        rig.send(.hoverChanged(true), at: 1)
        rig.send(ended("A"), at: 2)  // hide in 4 s, held
        #expect(rig.state.nextDeadline == nil)
        rig.send(.hoverChanged(false), at: 20)
        #expect(rig.state.nextDeadline == at(24))
    }

    @Test func aHoverOnAHiddenPanelDoesNothing() {
        var rig = PanelRig()
        let effects = rig.send(.hoverChanged(true))
        #expect(!rig.state.hovering)
        #expect(effects == [.wakeAt(nil)])
    }

    // MARK: Watchdog (D-08)

    @Test func theWatchdogHidesAnOpenTurnAfterThirtySecondsWithNoTurnEvent() {
        var rig = openRig()
        rig.send(partial("A", "hello"), at: 20)
        #expect(rig.state.nextDeadline == at(50))
        rig.send(.deadline, at: 49)
        #expect(rig.state.visible)
        let effects = rig.send(.deadline, at: 50)
        #expect(effects.contains(.hide(animated: true)))
        #expect(!rig.state.visible)
        #expect(rig.state.turn == nil)
        #expect(rig.state.nextDeadline == nil)
    }

    @Test func turnEndedClearsTheWatchdog() {
        var rig = openRig()
        rig.send(ended("A"), at: 1)
        #expect(rig.state.watchdogAt == nil)
    }

    // MARK: Close, connection loss, display removal

    @Test func closeHidesAtOnceAndIgnoresLaterEventsOfThatTurn() {
        var rig = openRig()
        rig.send(partial("A", "hello"), at: 1)
        let effects = rig.send(.closePressed, at: 2)
        #expect(effects.contains(.hide(animated: true)))
        #expect(!rig.state.visible)
        let late = rig.send(partial("A", "more"), at: 3)
        rig.send(word("A", "speaking"), at: 3)
        rig.send(wake("A"), at: 3)
        #expect(rig.state.turn == nil && !rig.state.visible)
        #expect(!late.contains(.show))
        let reopen = rig.send(wake("B"), at: 4)
        #expect(rig.state.visible)
        #expect(reopen.contains(.show))
        #expect(rig.state.turn?.turnId == "B")
    }

    @Test func aDroppedConnectionHidesAVisiblePanelAndEmptiesTheState() {
        var rig = openRig()
        rig.send(partial("A", "hello"), at: 1)
        let effects = rig.send(.connectionLost, at: 2)
        #expect(effects.contains(.hide(animated: true)))
        #expect(!rig.state.visible)
        #expect(rig.state.turn == nil)
        #expect(rig.state.nextDeadline == nil)
        let again = rig.send(.connectionLost, at: 3)
        #expect(again == [.wakeAt(nil)])
    }

    @Test func aRemovedDisplayHidesWithoutFadeKeepsTheContentAndTheNextEventShowsIt() {
        var rig = openRig()
        rig.send(partial("A", "hello"), at: 1)
        let removed = rig.send(.displayRemoved, at: 2)
        #expect(removed.contains(.hide(animated: false)))
        #expect(!rig.state.visible)
        #expect(rig.state.turn?.transcript == "hello")
        let next = rig.send(partial("A", "hello again"), at: 3)
        #expect(next.first == .show)
        #expect(rig.state.visible)
        #expect(rig.state.turn?.transcript == "hello again")
        #expect(rig.send(.displayRemoved, at: 4).contains(.hide(animated: false)))
        #expect(rig.send(.displayRemoved, at: 5) == [.wakeAt(rig.state.nextDeadline)])
    }

    // MARK: Contract of every call

    @Test func everyCallReturnsExactlyOneWakeAtLastEqualToTheNextDeadline() {
        var rig = PanelRig()
        let events: [(PanelEvent, Double)] = [
            (wake("A"), 0), (partial("A", "a"), 0.01), (partial("A", "b"), 0.02), (.deadline, 0.05),
            (word("A", "thinking"), 1), (final("A", "x"), 1.5), (textCard("A", "ok"), 2),
            (.hoverChanged(true), 3), (.hoverChanged(false), 4), (ended("A", window: 1000), 5),
            (.deadline, 100), (.closePressed, 101), (.connectionLost, 102), (.displayRemoved, 103),
            (.stopClicked, 104), (.timerStopped(TimerStopped(timerId: 1)), 105),
        ]
        for (event, seconds) in events {
            let effects = rig.send(event, at: seconds)
            let wakes = effects.filter { if case .wakeAt = $0 { true } else { false } }
            #expect(wakes.count == 1)
            #expect(effects.last == .wakeAt(rig.state.nextDeadline))
        }
    }
}
