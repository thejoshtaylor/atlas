import Foundation

// The ring rules (CARD-02, CARD-03, D-14 to D-17). The ring view has priority
// over turn content while a timer rings. Turn frames keep updating the turn
// underneath, so its content returns when the ring ends. The watchdog never
// applies to the ring view, and the 130 s cap covers a lost `timer.stopped`.
extension PanelReducer {
    /// A ring opens the panel with no wake word and takes over a turn (D-14, D-16).
    /// A second ring replaces the label and the id. The same id again (the sticky
    /// replay after a reconnect) changes nothing. A ring the operator closed
    /// never opens again for its id (T-15-28), until the server says that ring ended
    /// or one ring cap has passed. A repeating alarm rings under one id every time.
    static func timerRinging(
        _ state: inout PanelState, _ message: TimerRinging, _ now: Date, _ effects: inout [PanelEffect]
    ) {
        guard !state.isClosedRing(message.timerId, now: now), state.ring?.timerId != message.timerId else { return }
        let kind: RingKind = message.kind == "alarm" ? .alarm : .timer
        let label = TextSanitizer.sanitize(message.label, maxScalars: PanelTiming.labelMaxScalars)
        let opening = !state.visible
        // The hide timer of a "Stopped" view belongs to the ring that ended.
        if state.ring?.stop == .stopped {
            state.hideAt = nil
            state.pausedHide = nil
        }
        state.visible = true
        state.awaitingScreen = false
        state.ring = PanelRing(timerId: message.timerId, kind: kind, label: label)
        state.ringCapAt = now.addingTimeInterval(PanelTiming.ringCapS)
        state.stopRetryAt = nil
        effects.append(opening ? .show : .relayout)
        effects.append(.announce(PanelCopy.ringAnnouncement(kind: kind, label: label), .high))
    }

    /// D-15: one Stop request. A second click while the answer is pending sends nothing.
    static func stopClicked(_ state: inout PanelState, _ now: Date, _ effects: inout [PanelEffect]) {
        guard state.visible, let ring = state.ring, ring.stop == .idle else { return }
        state.ring?.stop = .pending
        state.stopRetryAt = now.addingTimeInterval(PanelTiming.stopRetryS)
        effects.append(.sendTimerStop(timerId: ring.timerId))
    }

    /// D-16: with a turn still active the turn content returns. Otherwise the ring
    /// view shows "Stopped" and the normal hide timer starts. A stop for another
    /// timer changes nothing.
    static func timerStopped(
        _ state: inout PanelState, _ message: TimerStopped, _ now: Date, _ effects: inout [PanelEffect]
    ) {
        // The ring ended on the server, so the memory of a closed ring ends too,
        // even when the panel shows no ring (a repeating alarm rings again).
        if state.closedTimerId == message.timerId { state.forgetClosedRing() }
        guard let ring = state.ring, ring.timerId == message.timerId, ring.stop != .stopped else { return }
        resumeIfAwaitingScreen(&state, &effects)
        state.stopRetryAt = nil
        state.ringCapAt = nil
        if state.hasActiveTurn {
            state.ring = nil
        } else {
            state.dropTurn()
            state.ring?.stop = .stopped
            state.setHide(after: PanelTiming.hideAfterStopS, now: now)
        }
        effects.append(.relayout)
    }

    /// The ring deadlines run before the turn deadlines. No answer to Stop for
    /// 3 s enables the button again. At the cap the ring ends, and the panel
    /// hides when no turn remains.
    static func deadlineRing(_ state: inout PanelState, _ now: Date, _ effects: inout [PanelEffect]) {
        if let at = state.stopRetryAt, at <= now {
            state.stopRetryAt = nil
            if state.ring?.stop == .pending { state.ring?.stop = .idle }
        }
        if let at = state.ringCapAt, at <= now {
            state.ringCapAt = nil
            state.stopRetryAt = nil
            if let id = state.ring?.timerId { state.rememberClosedRing(id, now: now) }
            state.ring = nil
            if state.turn == nil {
                hide(&state, animated: true, &effects)
            } else {
                effects.append(.relayout)
            }
        }
    }
}
