import Foundation

/// What the menu shows about the link to the server (D-20).
///
/// `connecting` means a socket open or a hello is in flight. The wait between
/// retries is `offline`, so the menu does not flicker between the two.
public enum LinkStatus: Equatable, Sendable {
    case unpaired
    case connecting(host: String)
    case connected(host: String, since: Date)
    case offline(host: String, lastConnected: Date?)
    case revoked(host: String)
    case versionMismatch(host: String)
}

/// Everything that can happen to the link. The connection actor turns each
/// transport result, timer and system notification into one of these.
public enum ConnectionEvent: Equatable, Sendable {
    case start(host: String)
    case stop
    /// The socket opened and the hello went out.
    case dialSucceeded
    /// `hello.ack` arrived.
    case ackReceived
    case dialFailed(DialFailure)
    /// A live socket ended. `nil` means no close code.
    case socketClosed(code: Int?)
    case pongMissed
    case retryFired
    case pathSatisfied
    case pathUnsatisfied
    case didWake
    case willSleep
}

/// What the actor must do after an event.
public enum ConnectionEffect: Equatable, Sendable {
    case dial
    case scheduleRetry(after: TimeInterval)
    case cancelRetry
    case startHeartbeat
    case stopHeartbeat
    case closeSocket(code: Int)
    /// Stop retrying, then delete the Keychain token (D-21, D-27).
    case unpair
}

/// The reconnect rules of the Mac, with no IO and no clock of its own.
///
/// The type takes no location input of any kind (D-10). A wrong or denied
/// location can change the menu label elsewhere, but it can never change a
/// dial, a wait or an unpair.
public struct ConnectionStateMachine: Sendable {
    private enum Phase { case stopped, dialing, awaitingAck, connected, waiting, revoked, versionMismatch }

    private let policy: ReconnectPolicy
    private var phase: Phase = .stopped
    private var host = ""
    private var pathOK = true
    private var asleep = false
    private var attempt = 0
    /// Handshake 403s since the last accepted hello (D-27).
    private var refusals = 0
    private var connectedAt: Date?
    private var lastConnected: Date?

    public init(policy: ReconnectPolicy = ReconnectPolicy()) {
        self.policy = policy
    }

    public var status: LinkStatus {
        switch phase {
        case .stopped: .unpaired
        case .dialing, .awaitingAck: .connecting(host: host)
        case .connected: .connected(host: host, since: connectedAt ?? Date(timeIntervalSince1970: 0))
        case .waiting: .offline(host: host, lastConnected: lastConnected)
        case .revoked: .revoked(host: host)
        case .versionMismatch: .versionMismatch(host: host)
        }
    }

    private var isLive: Bool { phase == .dialing || phase == .awaitingAck || phase == .connected }
    private var canDial: Bool { pathOK && !asleep }

    public mutating func handle(_ event: ConnectionEvent, now: Date, random: Double) -> [ConnectionEffect] {
        switch event {
        case .start(let newHost): return start(newHost)
        case .stop: return stop()
        case .dialSucceeded:
            if phase == .dialing { phase = .awaitingAck }
            return []
        case .ackReceived: return ack(now: now)
        case .dialFailed(let failure): return failed(failure, now: now, random: random)
        case .socketClosed(let code):
            guard isLive else { return [] }
            if let code { return closed(code: code, now: now, random: random) }
            return backoff(now: now, random: random)
        case .pongMissed:
            guard phase == .connected else { return [] }
            return [.closeSocket(code: 1001)] + backoff(now: now, random: random)
        case .retryFired:
            guard phase == .waiting, canDial else { return [] }
            phase = .dialing
            return [.dial]
        case .pathSatisfied:
            pathOK = true
            guard phase == .waiting, !asleep else { return [] }
            attempt = 0
            return [.cancelRetry, .scheduleRetry(after: policy.wakeDelay)]
        case .pathUnsatisfied:
            pathOK = false
            return suspend(now: now)
        case .willSleep:
            asleep = true
            return suspend(now: now)
        case .didWake:
            asleep = false
            return wake(now: now)
        }
    }

    // MARK: - Start and stop

    private mutating func start(_ newHost: String) -> [ConnectionEffect] {
        var effects: [ConnectionEffect] = []
        if isLive || phase == .waiting {
            effects = [.cancelRetry, .stopHeartbeat, .closeSocket(code: 1000)]
        }
        host = newHost
        attempt = 0
        refusals = 0
        connectedAt = nil
        lastConnected = nil
        guard canDial else {
            phase = .waiting
            return effects
        }
        phase = .dialing
        return effects + [.dial]
    }

    private mutating func stop() -> [ConnectionEffect] {
        guard phase != .stopped, phase != .revoked else { return [] }
        phase = .stopped
        connectedAt = nil
        return [.cancelRetry, .stopHeartbeat, .closeSocket(code: 1000)]
    }

    // MARK: - Ack and failure

    private mutating func ack(now: Date) -> [ConnectionEffect] {
        guard phase == .dialing || phase == .awaitingAck else { return [] }
        phase = .connected
        connectedAt = now
        refusals = 0
        return [.cancelRetry, .startHeartbeat]
    }

    private mutating func failed(_ failure: DialFailure, now: Date, random: Double) -> [ConnectionEffect] {
        guard isLive else { return [] }
        switch failure {
        case .refused:
            refusals += 1
            if refusals >= 2 { return revoke() }
            return backoff(now: now, random: random)
        case .unreachable:
            return backoff(now: now, random: random)
        case .closed(let code):
            return closed(code: code, now: now, random: random)
        }
    }

    private mutating func closed(code: Int, now: Date, random: Double) -> [ConnectionEffect] {
        switch CloseCode(rawValue: code)?.clientAction {
        case .unpair:
            return revoke()
        case .stopUpdateApp:
            let prefix = leaveLive(now: now)
            phase = .versionMismatch
            return prefix + [.cancelRetry]
        case .retryAfterCap:
            let prefix = leaveLive(now: now)
            return prefix + waitThenRetry(after: policy.cap)
        case .reconnect, nil:
            return backoff(now: now, random: random)
        }
    }

    private mutating func revoke() -> [ConnectionEffect] {
        phase = .revoked
        connectedAt = nil
        return [.stopHeartbeat, .cancelRetry, .unpair]
    }

    // MARK: - Waiting

    /// One failed attempt or one lost connection: wait by the policy delay.
    private mutating func backoff(now: Date, random: Double) -> [ConnectionEffect] {
        let prefix = leaveLive(now: now)
        let delay = policy.delay(attempt: attempt, random: random)
        attempt += 1
        return prefix + waitThenRetry(after: delay)
    }

    /// Records the end of a connection and, if it was stable, resets the attempt count.
    private mutating func leaveLive(now: Date) -> [ConnectionEffect] {
        var effects: [ConnectionEffect] = []
        if phase == .connected {
            effects = [.stopHeartbeat]
            lastConnected = now
            if let at = connectedAt, now.timeIntervalSince(at) >= policy.stableAfter { attempt = 0 }
        }
        connectedAt = nil
        return effects
    }

    private mutating func waitThenRetry(after delay: TimeInterval) -> [ConnectionEffect] {
        phase = .waiting
        return canDial ? [.scheduleRetry(after: delay)] : []
    }

    // MARK: - Sleep and network path

    /// Sleep or a lost path: close the socket and stop dialing until it returns.
    private mutating func suspend(now: Date) -> [ConnectionEffect] {
        if isLive {
            _ = leaveLive(now: now)
            phase = .waiting
            return [.stopHeartbeat, .closeSocket(code: 1001), .cancelRetry]
        }
        return phase == .waiting ? [.cancelRetry] : []
    }

    private mutating func wake(now: Date) -> [ConnectionEffect] {
        var effects: [ConnectionEffect] = []
        if isLive {
            // A socket that survived sleep without a willSleep is probably half open.
            effects = suspend(now: now)
        } else if phase != .waiting {
            return []
        }
        guard pathOK else { return effects }
        attempt = 0
        return effects + [.cancelRetry, .scheduleRetry(after: policy.wakeDelay)]
    }
}
