import Foundation

/// The panel's behavior as one pure function (Phase 15). It has no AppKit and no
/// clock of its own: the shell passes `now` and runs the effects it returns. The
/// last effect of every call is `wakeAt(state.nextDeadline)`, so the shell keeps
/// one timer and sets it again after each call.
///
/// Remote text is sanitized here, before it enters `PanelState` (T-15-27). The
/// Mac does not trust the server: it clamps durations, and it ignores a turn it
/// did not see open (T-15-28, T-15-29). Ring rules are in `PanelReducerRing.swift`.
public enum PanelReducer {
    public static func reduce(_ state: inout PanelState, _ event: PanelEvent, now: Date) -> [PanelEffect] {
        var effects: [PanelEffect] = []
        switch event {
        case .wakeConfirmed(let message): onWake(&state, message, now, &effects)
        case .turnState(let message): onTurnState(&state, message, now, &effects)
        case .partial(let message): onPartial(&state, message, now, &effects)
        case .final(let message): onFinal(&state, message, now, &effects)
        case .card(let message): onCard(&state, message, now, &effects)
        case .turnEnded(let message): onTurnEnded(&state, message, now, &effects)
        case .timerRinging(let message): timerRinging(&state, message, now, &effects)
        case .timerStopped(let message): timerStopped(&state, message, now, &effects)
        case .stopClicked: stopClicked(&state, now, &effects)
        case .hoverChanged(let inside):
            if state.visible { state.setHovering(inside, now: now) }
        case .closePressed: onClose(&state, now, &effects)
        case .deadline: onDeadline(&state, now, &effects)
        case .connectionLost: hide(&state, animated: true, &effects)
        case .displayRemoved: onDisplayRemoved(&state, now, &effects)
        }
        effects.append(.wakeAt(state.nextDeadline))
        return effects
    }

    // MARK: Turn events

    /// D-05, D-07: a confirmed wake opens the panel, or replaces its content.
    private static func onWake(
        _ state: inout PanelState, _ message: WakeConfirmed, _ now: Date, _ effects: inout [PanelEffect]
    ) {
        // The same turn again changes nothing. A closed or retired turn never comes back.
        if state.isRetired(message.turnId) || state.turn?.turnId == message.turnId { return }
        let opening = !state.visible
        startTurn(&state, id: message.turnId, word: .listening)
        state.visible = true
        state.awaitingScreen = false
        state.setWatchdog(after: PanelTiming.watchdogS, now: now)
        if opening {
            effects.append(.show)
            effects.append(.announce(PanelCopy.listeningAnnouncement, .medium))
        } else {
            effects.append(.relayout)
        }
    }

    private static func onTurnState(
        _ state: inout PanelState, _ message: TurnStateMessage, _ now: Date, _ effects: inout [PanelEffect]
    ) {
        let word = stateWord(message.state)
        let admission = admit(&state, turnId: message.turnId, mayStart: true, startWord: word ?? .listening)
        guard admission != .rejected else { return }
        accepted(&state, started: admission == .started, now, &effects)
        if let word { state.turn?.word = word }
    }

    private static func onPartial(
        _ state: inout PanelState, _ message: TranscriptPartial, _ now: Date, _ effects: inout [PanelEffect]
    ) {
        let admission = admit(&state, turnId: message.turnId, mayStart: true, startWord: .listening)
        guard admission != .rejected else { return }
        accepted(&state, started: admission == .started, now, &effects)
        if let delivered = state.coalescer.offer(message, now: now) {
            applyTranscript(&state, delivered, &effects)
        }
    }

    private static func onFinal(
        _ state: inout PanelState, _ message: TranscriptFinal, _ now: Date, _ effects: inout [PanelEffect]
    ) {
        guard admit(&state, turnId: message.turnId, mayStart: false, startWord: .listening) == .current else { return }
        accepted(&state, started: false, now, &effects)
        state.turn?.transcript = sanitizedTranscript(message.text)
        state.turn?.transcriptIsFinal = true
        state.coalescer.reset()
        effects.append(.relayout)
    }

    /// D-11: one card slot. A card with no usable text changes nothing.
    private static func onCard(
        _ state: inout PanelState, _ message: CardMessage, _ now: Date, _ effects: inout [PanelEffect]
    ) {
        guard admit(&state, turnId: message.turnId, mayStart: false, startWord: .listening) == .current else { return }
        var text = TextSanitizer.sanitize(message.displayText ?? "", maxScalars: PanelTiming.cardMaxScalars)
        if text.isEmpty {
            text = TextSanitizer.sanitize(message.fallbackText, maxScalars: PanelTiming.cardMaxScalars)
        }
        guard !text.isEmpty else { return }
        accepted(&state, started: false, now, &effects)
        state.turn?.cardId = message.cardId
        state.turn?.cardText = text
        effects.append(.relayout)
        effects.append(.announce(text, .medium))
    }

    /// D-06, D-09, PANEL-05: the Mac sets `done` itself and starts the hide timer.
    private static func onTurnEnded(
        _ state: inout PanelState, _ message: TurnEnded, _ now: Date, _ effects: inout [PanelEffect]
    ) {
        guard admit(&state, turnId: message.turnId, mayStart: false, startWord: .listening) == .current else { return }
        accepted(&state, started: false, now, &effects)
        let playback = clampedSeconds(message.playbackMsLeft)
        let followUp = clampedSeconds(message.followUpWindowMs)
        state.turn?.ended = true
        state.turn?.outcome = outcomeLine(message.outcome)
        state.turn?.word = followUp > 0 ? .listening : .done
        state.watchdogAt = nil
        state.pausedWatchdog = nil
        state.coalescer.reset()
        state.setHide(after: playback + (followUp > 0 ? followUp : PanelTiming.hideAfterTurnS), now: now)
        effects.append(.relayout)
    }

    // MARK: Shell events

    /// D-04: Close hides now, and the turn and the ring it showed never come back.
    private static func onClose(_ state: inout PanelState, _ now: Date, _ effects: inout [PanelEffect]) {
        guard state.visible else { return }
        if let id = state.turn?.turnId { state.closedTurnId = id }
        if let id = state.ring?.timerId { state.rememberClosedRing(id, now: now) }
        hide(&state, animated: true, &effects)
    }

    /// A display went away: the panel is off screen, its content stays, and the
    /// next accepted event shows it again on a screen the shell picks then.
    private static func onDisplayRemoved(_ state: inout PanelState, _ now: Date, _ effects: inout [PanelEffect]) {
        guard state.visible else { return }
        state.setHovering(false, now: now)
        state.visible = false
        state.awaitingScreen = true
        effects.append(.hide(animated: false))
    }

    /// The order is: the ring rules, a held partial, then the turn timers.
    private static func onDeadline(_ state: inout PanelState, _ now: Date, _ effects: inout [PanelEffect]) {
        deadlineRing(&state, now, &effects)
        if let partial = state.coalescer.flush(now: now) { applyTranscript(&state, partial, &effects) }
        let watchdogDue = state.watchdogAt.map { $0 <= now } ?? false
        let hideDue = state.hideAt.map { $0 <= now } ?? false
        guard watchdogDue || hideDue else { return }
        if let ring = state.ring, ring.stop != .stopped {
            // An open ring view outlives its turn: drop the turn and keep the panel.
            state.dropTurn()
            state.hideAt = nil
            state.pausedHide = nil
            effects.append(.relayout)
        } else {
            hide(&state, animated: true, &effects)
        }
    }

    // MARK: Shared helpers (the ring file uses these too)

    /// Hide, and forget everything except the memory of closed and retired ids.
    static func hide(_ state: inout PanelState, animated: Bool, _ effects: inout [PanelEffect]) {
        let wasVisible = state.visible
        state.resetForHide()
        if wasVisible { effects.append(.hide(animated: animated)) }
    }

    /// The panel waited for a screen: show it again, before any other effect.
    static func resumeIfAwaitingScreen(_ state: inout PanelState, _ effects: inout [PanelEffect]) {
        guard state.awaitingScreen else { return }
        state.awaitingScreen = false
        state.visible = true
        effects.append(.show)
    }

    // MARK: Private helpers

    private enum Admission { case rejected, current, started }

    /// Whether a turn-scoped frame applies (D-05, RESEARCH Pitfall 1). The current
    /// turn takes its frames until it ends. An unseen id starts the next turn only
    /// from a state or partial frame, in an open panel, after the current turn
    /// ended. A replaced, hidden or closed turn is never adopted again.
    private static func admit(
        _ state: inout PanelState, turnId: String, mayStart: Bool, startWord: PanelStateWord
    ) -> Admission {
        guard state.isOpen else { return .rejected }
        if let turn = state.turn, turn.turnId == turnId { return turn.ended ? .rejected : .current }
        guard mayStart, !state.isRetired(turnId), !state.hasActiveTurn else { return .rejected }
        startTurn(&state, id: turnId, word: startWord)
        return .started
    }

    /// An accepted frame: show the panel again if it waited for a screen, restart
    /// the watchdog, and measure again when a new turn replaced the content.
    private static func accepted(_ state: inout PanelState, started: Bool, _ now: Date, _ effects: inout [PanelEffect]) {
        resumeIfAwaitingScreen(&state, &effects)
        state.setWatchdog(after: PanelTiming.watchdogS, now: now)
        if started { effects.append(.relayout) }
    }

    private static func startTurn(_ state: inout PanelState, id: String, word: PanelStateWord) {
        if let old = state.turn { state.retire(old.turnId) }
        state.turn = PanelTurn(turnId: id, word: word)
        state.coalescer.reset()
        state.hideAt = nil
        state.pausedHide = nil
        // A "Stopped" ring view must not cover a live turn.
        if state.ring?.stop == .stopped { state.ring = nil }
    }

    private static func applyTranscript(_ state: inout PanelState, _ partial: TranscriptPartial, _ effects: inout [PanelEffect]) {
        guard let turn = state.turn, turn.turnId == partial.turnId, !turn.transcriptIsFinal, !turn.ended else { return }
        state.turn?.transcript = sanitizedTranscript(partial.text)
        effects.append(.relayout)
    }

    private static func sanitizedTranscript(_ text: String) -> String {
        TextSanitizer.sanitize(text, maxScalars: PanelTiming.transcriptKeepScalars, keep: .end)
    }

    private static func stateWord(_ word: String) -> PanelStateWord? {
        switch word {
        case "listening": .listening
        case "thinking": .thinking
        case "speaking": .speaking
        default: nil
        }
    }

    private static func outcomeLine(_ outcome: String) -> PanelOutcomeLine? {
        switch outcome {
        case "no_speech": .noSpeech
        case "stopped": .stopped
        case "failed": .failed
        default: nil
        }
    }

    /// Server milliseconds as seconds, inside 0 and `maxServerWindowS` (T-15-28).
    private static func clampedSeconds(_ milliseconds: Int) -> TimeInterval {
        min(max(TimeInterval(milliseconds) / 1000, 0), PanelTiming.maxServerWindowS)
    }
}
