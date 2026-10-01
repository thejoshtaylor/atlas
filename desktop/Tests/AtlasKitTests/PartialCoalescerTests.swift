import Foundation
import Testing

@testable import AtlasKit

private let t0 = Date(timeIntervalSince1970: 1_800_000_000)
private func partial(_ text: String, turn: String = "A") -> TranscriptPartial {
    TranscriptPartial(turnId: turn, text: text)
}

@Suite struct PartialCoalescerTests {
    @Test func theFirstPartialIsDeliveredAtOnce() {
        var coalescer = PartialCoalescer()
        #expect(coalescer.offer(partial("one"), now: t0) == partial("one"))
        #expect(coalescer.flushAt == nil)
    }

    @Test func aSecondPartialInsideFiftyMillisecondsIsHeld() {
        var coalescer = PartialCoalescer()
        _ = coalescer.offer(partial("one"), now: t0)
        #expect(coalescer.offer(partial("two"), now: t0.addingTimeInterval(0.049)) == nil)
        #expect(coalescer.flushAt == t0.addingTimeInterval(0.05))
    }

    @Test func aThirdPartialInTheSameWindowReplacesTheHeldOne() {
        var coalescer = PartialCoalescer()
        _ = coalescer.offer(partial("one"), now: t0)
        _ = coalescer.offer(partial("two"), now: t0.addingTimeInterval(0.01))
        _ = coalescer.offer(partial("three"), now: t0.addingTimeInterval(0.02))
        #expect(coalescer.flush(now: t0.addingTimeInterval(0.05)) == partial("three"))
        #expect(coalescer.flush(now: t0.addingTimeInterval(0.2)) == nil)
        #expect(coalescer.flushAt == nil)
    }

    @Test func flushBeforeTheWindowEndsKeepsTheHeldPartial() {
        var coalescer = PartialCoalescer()
        _ = coalescer.offer(partial("one"), now: t0)
        _ = coalescer.offer(partial("two"), now: t0.addingTimeInterval(0.01))
        #expect(coalescer.flush(now: t0.addingTimeInterval(0.03)) == nil)
        #expect(coalescer.flush(now: t0.addingTimeInterval(0.06)) == partial("two"))
    }

    @Test func aPartialAfterTheWindowIsDeliveredAndDropsTheHeldOne() {
        var coalescer = PartialCoalescer()
        _ = coalescer.offer(partial("one"), now: t0)
        _ = coalescer.offer(partial("two"), now: t0.addingTimeInterval(0.01))
        #expect(coalescer.offer(partial("three"), now: t0.addingTimeInterval(0.1)) == partial("three"))
        #expect(coalescer.flushAt == nil)
    }

    @Test func aHundredPartialsOverOneSecondDeliverAtMostTwentyOne() {
        var coalescer = PartialCoalescer()
        var delivered = 0
        for index in 0..<100 {
            let now = t0.addingTimeInterval(Double(index) * 0.01)
            if coalescer.flush(now: now) != nil { delivered += 1 }
            if coalescer.offer(partial("p\(index)"), now: now) != nil { delivered += 1 }
        }
        #expect(delivered <= 21)
        #expect(delivered >= 18)
    }

    @Test func flushAtIsNilWithNothingHeld() {
        #expect(PartialCoalescer().flushAt == nil)
    }

    @Test func resetClearsTheHeldPartialAndTheLastDelivery() {
        var coalescer = PartialCoalescer()
        _ = coalescer.offer(partial("one"), now: t0)
        _ = coalescer.offer(partial("two"), now: t0.addingTimeInterval(0.01))
        coalescer.reset()
        #expect(coalescer.flushAt == nil)
        #expect(coalescer.offer(partial("fresh"), now: t0.addingTimeInterval(0.02)) == partial("fresh"))
    }
}

@Suite struct PanelCopyTests {
    @Test func theFourStateWordsAndSymbols() {
        let expected: [(PanelStateWord, String, String)] = [
            (.listening, "Listening", "waveform"),
            (.thinking, "Thinking", "ellipsis"),
            (.speaking, "Speaking", "speaker.wave.2.fill"),
            (.done, "Done", "checkmark.circle.fill"),
        ]
        for (word, text, symbol) in expected {
            #expect(PanelCopy.stateWord(word) == text)
            #expect(PanelCopy.stateSymbol(word) == symbol)
        }
    }

    @Test func theThreeOutcomeLines() {
        #expect(PanelCopy.outcomeLine(.noSpeech) == "Did not catch that")
        #expect(PanelCopy.outcomeLine(.stopped) == "Stopped")
        #expect(PanelCopy.outcomeLine(.failed) == "Could not answer. Try again.")
    }

    @Test func theFixedStringsOfTheCopyTable() {
        #expect(PanelCopy.emptyHint == "Go ahead")
        #expect(PanelCopy.ringKindWord(.timer) == "Timer")
        #expect(PanelCopy.ringKindWord(.alarm) == "Alarm")
        #expect(PanelCopy.emptyRingLabel(.timer) == "Time is up")
        #expect(PanelCopy.emptyRingLabel(.alarm) == "Alarm is ringing")
        #expect(PanelCopy.stopTitle == "Stop")
        #expect(PanelCopy.stoppedLine == "Stopped")
        #expect(PanelCopy.closeLabel == "Close panel")
        #expect(PanelCopy.panelLabel == "ATLAS")
        #expect(PanelCopy.listeningAnnouncement == "ATLAS is listening")
        #expect(PanelCopy.stopRingingMenuItem == "Stop Ringing")
        #expect(PanelCopy.bellSymbol == "bell.fill")
        #expect(PanelCopy.closeSymbol == "xmark")
    }

    @Test func thePendingTitleEndsWithTheEllipsisCharacter() {
        #expect(PanelCopy.stopPendingTitle == "Stopping\u{2026}")
        #expect(PanelCopy.stopPendingTitle.unicodeScalars.last?.value == 0x2026)
    }

    @Test func theStopAccessibilityLabelNamesTheRingOrItsKind() {
        #expect(PanelCopy.stopAccessibilityLabel(label: "pasta", kind: .timer) == "Stop pasta")
        #expect(PanelCopy.stopAccessibilityLabel(label: "", kind: .timer) == "Stop timer")
        #expect(PanelCopy.stopAccessibilityLabel(label: "", kind: .alarm) == "Stop alarm")
    }

    @Test func theRingAnnouncementFallsBackToTheEmptyLabelText() {
        #expect(PanelCopy.ringAnnouncement(kind: .timer, label: "pasta") == "Timer ringing, pasta")
        #expect(PanelCopy.ringAnnouncement(kind: .alarm, label: "wake up") == "Alarm ringing, wake up")
        #expect(PanelCopy.ringAnnouncement(kind: .timer, label: "") == "Timer ringing, Time is up")
        #expect(PanelCopy.ringAnnouncement(kind: .alarm, label: "") == "Alarm ringing, Alarm is ringing")
    }

    @Test func noFixedStringHoldsAContraction() {
        let strings = [
            PanelCopy.emptyHint, PanelCopy.emptyTimerLabel, PanelCopy.emptyAlarmLabel,
            PanelCopy.stoppedLine, PanelCopy.listeningAnnouncement,
            PanelCopy.outcomeLine(.noSpeech), PanelCopy.outcomeLine(.failed),
        ]
        for text in strings {
            #expect(!text.contains("'") && !text.contains("\u{2019}"))
        }
    }
}

@Suite struct PanelEventMappingTests {
    @Test func eachPanelFrameMapsToItsEvent() {
        let wake = WakeConfirmed(turnId: "A")
        let state = TurnStateMessage(turnId: "A", state: "thinking")
        let part = TranscriptPartial(turnId: "A", text: "hey")
        let final = TranscriptFinal(turnId: "A", text: "hey atlas")
        let card = CardMessage(
            turnId: "A", cardId: "reply", kind: "text", fallbackText: "x",
            data: .text(TextCardData(text: "x")))
        let ended = TurnEnded(turnId: "A", outcome: "completed", followUpWindowMs: 0, playbackMsLeft: 0)
        let ringing = TimerRinging(timerId: 1, kind: "timer", label: "pasta")
        let stopped = TimerStopped(timerId: 1)

        #expect(PanelEvent(.wakeConfirmed(wake)) == .wakeConfirmed(wake))
        #expect(PanelEvent(.turnState(state)) == .turnState(state))
        #expect(PanelEvent(.transcriptPartial(part)) == .partial(part))
        #expect(PanelEvent(.transcriptFinal(final)) == .final(final))
        #expect(PanelEvent(.card(card)) == .card(card))
        #expect(PanelEvent(.turnEnded(ended)) == .turnEnded(ended))
        #expect(PanelEvent(.timerRinging(ringing)) == .timerRinging(ringing))
        #expect(PanelEvent(.timerStopped(stopped)) == .timerStopped(stopped))
    }

    @Test func framesThePanelDoesNotReadMapToNil() {
        #expect(PanelEvent(.helloAck(HelloAck(protocolVersion: 1, deviceId: 1, pingIntervalS: 15))) == nil)
        #expect(PanelEvent(.ping(Ping(id: 1))) == nil)
        #expect(PanelEvent(.pong(Pong(id: 1))) == nil)
        #expect(PanelEvent(.error(ErrorMessage(code: "bad", detail: "x"))) == nil)
        #expect(PanelEvent(.unknown(type: "future.thing")) == nil)
    }

    @Test func aFreshStateHasNoDeadline() {
        let state = PanelState()
        #expect(state.nextDeadline == nil)
        #expect(!state.visible)
        #expect(state.turn == nil && state.ring == nil && !state.hovering)
    }
}
