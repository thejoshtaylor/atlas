import Foundation

/// The four words of the state line (D-09). The server sends listening,
/// thinking and speaking. The Mac sets `done` itself, on `turn.ended`.
public enum PanelStateWord: Sendable, Equatable {
    case listening, thinking, speaking, done
}

/// The fixed outcome lines. `completed` and an unknown outcome have none.
public enum PanelOutcomeLine: Sendable, Equatable {
    case noSpeech, stopped, failed
}

/// What the turn view shows.
public struct PanelTurn: Sendable, Equatable {
    public internal(set) var turnId: String
    public internal(set) var word: PanelStateWord
    public internal(set) var transcript: String
    public internal(set) var transcriptIsFinal: Bool
    public internal(set) var cardId: String?
    public internal(set) var cardText: String?
    public internal(set) var outcome: PanelOutcomeLine?
    public internal(set) var ended: Bool

    public init(
        turnId: String, word: PanelStateWord = .listening, transcript: String = "",
        transcriptIsFinal: Bool = false, cardId: String? = nil, cardText: String? = nil,
        outcome: PanelOutcomeLine? = nil, ended: Bool = false
    ) {
        self.turnId = turnId
        self.word = word
        self.transcript = transcript
        self.transcriptIsFinal = transcriptIsFinal
        self.cardId = cardId
        self.cardText = cardText
        self.outcome = outcome
        self.ended = ended
    }
}

public enum RingKind: Sendable, Equatable {
    case timer, alarm
}

/// The Stop button of the ring view (UI-SPEC E6). `stopped` replaces the button
/// with the line "Stopped".
public enum StopButtonState: Sendable, Equatable {
    case idle, pending, stopped
}

/// What the ring view shows. One ring at a time.
public struct PanelRing: Sendable, Equatable {
    public internal(set) var timerId: Int
    public internal(set) var kind: RingKind
    public internal(set) var label: String
    public internal(set) var stop: StopButtonState

    public init(timerId: Int, kind: RingKind, label: String, stop: StopButtonState = .idle) {
        self.timerId = timerId
        self.kind = kind
        self.label = label
        self.stop = stop
    }
}

public enum AnnouncePriority: Sendable, Equatable {
    case medium, high
}

/// Everything the reducer reacts to: a server frame, a pointer or button event
/// from the shell, a clock deadline, and the two system events.
public enum PanelEvent: Sendable, Equatable {
    case wakeConfirmed(WakeConfirmed)
    case turnState(TurnStateMessage)
    case partial(TranscriptPartial)
    case final(TranscriptFinal)
    case card(CardMessage)
    case turnEnded(TurnEnded)
    case timerRinging(TimerRinging)
    case timerStopped(TimerStopped)
    case hoverChanged(Bool)
    case closePressed
    case stopClicked
    /// The shell calls this when `nextDeadline` has passed.
    case deadline
    case connectionLost
    case displayRemoved

    /// The panel event for a server frame, or `nil` for a frame the panel does
    /// not read (hello.ack, ping, pong, error and an unknown type).
    public init?(_ message: ServerMessage) {
        switch message {
        case .wakeConfirmed(let value): self = .wakeConfirmed(value)
        case .turnState(let value): self = .turnState(value)
        case .transcriptPartial(let value): self = .partial(value)
        case .transcriptFinal(let value): self = .final(value)
        case .card(let value): self = .card(value)
        case .turnEnded(let value): self = .turnEnded(value)
        case .timerRinging(let value): self = .timerRinging(value)
        case .timerStopped(let value): self = .timerStopped(value)
        case .helloAck, .ping, .pong, .error, .unknown: return nil
        }
    }
}

/// What the shell must do after an event. `wakeAt` is always last.
public enum PanelEffect: Sendable, Equatable {
    case show
    case hide(animated: Bool)
    /// The content changed. The shell measures it and sets the frame again.
    case relayout
    case announce(String, AnnouncePriority)
    case sendTimerStop(timerId: Int)
    /// Call the reducer with `.deadline` at this time. `nil` means no timer.
    case wakeAt(Date?)
}

/// The panel's whole state. The reducer is the only writer.
public struct PanelState: Sendable, Equatable {
    public internal(set) var visible = false
    public internal(set) var turn: PanelTurn?
    public internal(set) var ring: PanelRing?
    public internal(set) var hovering = false

    // Deadlines. A hover moves `hideAt` and `watchdogAt` into the paused times.
    var hideAt: Date?
    var watchdogAt: Date?
    var ringCapAt: Date?
    var stopRetryAt: Date?
    var pausedHide: TimeInterval?
    var pausedWatchdog: TimeInterval?

    // Memory that survives a hide, so a replayed or late frame cannot pull the
    // panel back (D-04, D-05, RESEARCH Pitfall 1).
    var closedTurnId: String?
    var closedTimerId: Int?
    /// The memory of `closedTimerId` ends here. A repeating alarm keeps one id
    /// for every ring, and a lost `timer.stopped` must not hide it for good.
    var closedTimerUntil: Date?
    var retiredTurnIds: [String] = []

    /// A display went away while the panel showed. The next accepted event
    /// shows it again, and the shell picks the screen then.
    var awaitingScreen = false
    var coalescer = PartialCoalescer()

    public init() {}

    /// The earliest time the shell must call the reducer with `.deadline`.
    public var nextDeadline: Date? {
        [hideAt, watchdogAt, ringCapAt, stopRetryAt, coalescer.flushAt].compactMap { $0 }.min()
    }

    /// The panel shows, or it waits for a screen and keeps its content.
    var isOpen: Bool { visible || awaitingScreen }

    /// A turn that has not ended.
    var hasActiveTurn: Bool { turn.map { !$0.ended } ?? false }

    func isRetired(_ turnId: String) -> Bool {
        turnId == closedTurnId || retiredTurnIds.contains(turnId)
    }

    mutating func retire(_ turnId: String) {
        guard !retiredTurnIds.contains(turnId) else { return }
        retiredTurnIds.append(turnId)
        if retiredTurnIds.count > PanelTiming.retiredTurnMemory { retiredTurnIds.removeFirst() }
    }

    /// Drop the turn content, remembering its id.
    /// Remember that the operator closed (or the cap ended) this ring. The
    /// memory lasts one ring cap, which is as long as any ring plays.
    mutating func rememberClosedRing(_ timerId: Int, now: Date) {
        closedTimerId = timerId
        closedTimerUntil = now.addingTimeInterval(PanelTiming.ringCapS)
    }

    mutating func forgetClosedRing() {
        closedTimerId = nil
        closedTimerUntil = nil
    }

    /// True while a replay of this ring must stay hidden.
    func isClosedRing(_ timerId: Int, now: Date) -> Bool {
        guard timerId == closedTimerId, let until = closedTimerUntil else { return false }
        return until > now
    }

    mutating func dropTurn() {
        if let id = turn?.turnId { retire(id) }
        turn = nil
        watchdogAt = nil
        pausedWatchdog = nil
        coalescer.reset()
    }

    /// Everything except the memory (`closedTurnId`, `closedTimerId`, `closedTimerUntil`, `retiredTurnIds`).
    mutating func resetForHide() {
        dropTurn()
        visible = false
        awaitingScreen = false
        ring = nil
        hovering = false
        hideAt = nil
        ringCapAt = nil
        stopRetryAt = nil
        pausedHide = nil
    }

    /// Start (or restart) the hide timer. While the pointer is on the panel the
    /// time waits in `pausedHide`.
    mutating func setHide(after seconds: TimeInterval, now: Date) {
        if hovering {
            pausedHide = seconds
            hideAt = nil
        } else {
            hideAt = now.addingTimeInterval(seconds)
            pausedHide = nil
        }
    }

    mutating func setWatchdog(after seconds: TimeInterval, now: Date) {
        if hovering {
            pausedWatchdog = seconds
            watchdogAt = nil
        } else {
            watchdogAt = now.addingTimeInterval(seconds)
            pausedWatchdog = nil
        }
    }

    /// Hover in pauses the hide timer and the watchdog. Hover out resumes each
    /// with the time it had left, and never less than `hoverResumeMinS`.
    mutating func setHovering(_ value: Bool, now: Date) {
        guard value != hovering else { return }
        hovering = value
        if value {
            if let at = hideAt { pausedHide = max(0, at.timeIntervalSince(now)); hideAt = nil }
            if let at = watchdogAt { pausedWatchdog = max(0, at.timeIntervalSince(now)); watchdogAt = nil }
        } else {
            if let left = pausedHide {
                hideAt = now.addingTimeInterval(max(left, PanelTiming.hoverResumeMinS))
                pausedHide = nil
            }
            if let left = pausedWatchdog {
                watchdogAt = now.addingTimeInterval(max(left, PanelTiming.hoverResumeMinS))
                pausedWatchdog = nil
            }
        }
    }
}
