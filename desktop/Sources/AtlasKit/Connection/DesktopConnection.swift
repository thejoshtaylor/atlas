import Foundation
import os

/// What the connection actor reports after every state change.
public struct ConnectionSnapshot: Sendable, Equatable {
    public var status: LinkStatus
    /// True after the first `hello.ack`, and it stays true.
    public var everConnected: Bool
    /// Why the last attempt or connection ended. It clears on the next ack.
    public var lastFailure: DialFailure?

    public init(status: LinkStatus, everConnected: Bool, lastFailure: DialFailure?) {
        self.status = status
        self.everConnected = everConnected
        self.lastFailure = lastFailure
    }
}

/// What the app tells the actor about the system.
public enum SystemEvent: Sendable, Equatable {
    case didWake
    case willSleep
    case pathSatisfied
    case pathUnsatisfied
}

/// The two strings the hello carries besides the protocol number.
public struct HelloInfo: Sendable, Equatable {
    public var appVersion: String
    public var osVersion: String

    public init(appVersion: String, osVersion: String) {
        self.appVersion = appVersion
        self.osVersion = osVersion
    }
}

/// Keeps the Mac connected to the server: dial, hello, heartbeat, retries.
///
/// Every transport result, timer and system event becomes a `ConnectionEvent`
/// for the pure `ConnectionStateMachine`. The actor only runs the effects it
/// returns. The actor reads no location (D-10). It logs close codes and frame
/// kinds, never the token and never a frame body.
public actor DesktopConnection {
    public nonisolated let snapshots: AsyncStream<ConnectionSnapshot>

    private let transport: any WebSocketTransport
    private let clock: any ConnectionClock
    private let store: any SecretStore
    private let helloInfo: HelloInfo
    private let random: @Sendable () -> Double
    private let log = Logger(subsystem: AppIdentity.bundleIdentifier, category: "connection")
    private let continuation: AsyncStream<ConnectionSnapshot>.Continuation

    private var machine: ConnectionStateMachine
    private var credentials: PairingCredentials?
    private var channel: (any WebSocketChannel)?
    /// Each dial gets a number. Results from an older number are ignored.
    private var generation = 0
    private var retryEpoch = 0
    private var dialTask: Task<Void, Never>?
    private var receiveTask: Task<Void, Never>?
    private var ackTimeoutTask: Task<Void, Never>?
    private var retryTask: Task<Void, Never>?
    private var heartbeatTask: Task<Void, Never>?
    private var pongWatchTask: Task<Void, Never>?
    private var heartbeatInterval = TimeInterval(ProtocolConstants.pingIntervalS)
    private var nextPingId = 1
    private var pendingPingId: Int?
    private var everConnected = false
    private var lastFailure: DialFailure?
    private var lastPublished: ConnectionSnapshot

    public init(
        transport: any WebSocketTransport,
        clock: any ConnectionClock = SystemConnectionClock(),
        store: any SecretStore,
        helloInfo: HelloInfo,
        policy: ReconnectPolicy = ReconnectPolicy(),
        random: @escaping @Sendable () -> Double = { Double.random(in: 0..<1) }
    ) {
        self.transport = transport
        self.clock = clock
        self.store = store
        self.helloInfo = helloInfo
        self.random = random
        self.machine = ConnectionStateMachine(policy: policy)
        let initial = ConnectionSnapshot(status: .unpaired, everConnected: false, lastFailure: nil)
        self.lastPublished = initial
        let (stream, continuation) = AsyncStream.makeStream(of: ConnectionSnapshot.self)
        self.snapshots = stream
        self.continuation = continuation
        continuation.yield(initial)
    }

    public var currentSnapshot: ConnectionSnapshot { lastPublished }

    // MARK: - Inputs

    public func start(credentials: PairingCredentials) {
        self.credentials = credentials
        everConnected = false
        lastFailure = nil
        apply(.start(host: credentials.host))
    }

    public func stop() {
        apply(.stop)
        credentials = nil
    }

    public func handle(_ event: SystemEvent) {
        switch event {
        case .didWake: apply(.didWake)
        case .willSleep: apply(.willSleep)
        case .pathSatisfied: apply(.pathSatisfied)
        case .pathUnsatisfied: apply(.pathUnsatisfied)
        }
    }

    // MARK: - Machine and effects

    private func apply(_ event: ConnectionEvent) {
        let effects = machine.handle(event, now: clock.now(), random: random())
        for effect in effects { run(effect) }
        publish()
    }

    private func run(_ effect: ConnectionEffect) {
        switch effect {
        case .dial: startDial()
        case .scheduleRetry(let delay): scheduleRetry(after: delay)
        case .cancelRetry: cancelRetry()
        case .startHeartbeat: startHeartbeat()
        case .stopHeartbeat: stopHeartbeat()
        case .closeSocket(let code): dropSocket(code: code)
        case .unpair:
            dropSocket(code: 1000)
            do {
                try store.delete()
            } catch {
                log.error("The pairing could not be deleted from the store.")
            }
        }
    }

    private func publish() {
        let snapshot = ConnectionSnapshot(
            status: machine.status, everConnected: everConnected, lastFailure: lastFailure)
        guard snapshot != lastPublished else { return }
        lastPublished = snapshot
        continuation.yield(snapshot)
    }

    // MARK: - Dial

    private func startDial() {
        guard let credentials else { return }
        dropSocket(code: 1001)
        generation += 1
        let gen = generation
        let url = ConnectionEndpoint.url(for: credentials.host)
        let bearer = credentials.token
        dialTask = Task { [transport] in
            do {
                let opened = try await transport.open(url: url, bearerToken: bearer)
                await self.dialOpened(opened, generation: gen)
            } catch {
                self.dialFailed(error as? DialFailure ?? .unreachable, generation: gen)
            }
        }
        // One bound for open, hello and ack together.
        ackTimeoutTask = Task { [clock] in
            do { try await clock.sleep(for: TimeInterval(ProtocolConstants.helloTimeoutS)) } catch { return }
            self.ackTimedOut(generation: gen)
        }
    }

    private func dialOpened(_ opened: any WebSocketChannel, generation gen: Int) async {
        guard gen == generation else {
            await opened.close(code: 1001)
            return
        }
        channel = opened
        let hello = Hello(
            protocolVersion: ProtocolConstants.protocolVersion,
            appVersion: helloInfo.appVersion,
            osVersion: helloInfo.osVersion,
            capabilities: [])
        do {
            let text = try WireCodec.encode(.hello(hello))
            try await opened.send(text: text)
        } catch {
            dialFailed(error as? DialFailure ?? .unreachable, generation: gen)
            return
        }
        guard gen == generation else { return }
        apply(.dialSucceeded)
        receiveTask = Task {
            await self.receiveLoop(opened, generation: gen)
        }
    }

    private func dialFailed(_ failure: DialFailure, generation gen: Int) {
        guard gen == generation else { return }
        lastFailure = failure
        dropSocket(code: 1001)
        apply(.dialFailed(failure))
    }

    private func ackTimedOut(generation gen: Int) {
        guard gen == generation, case .connecting = machine.status else { return }
        log.info("No hello.ack within the time limit.")
        dialFailed(.unreachable, generation: gen)
    }

    // MARK: - Receive

    private func receiveLoop(_ opened: any WebSocketChannel, generation gen: Int) async {
        while !Task.isCancelled {
            do {
                let text = try await opened.receive()
                guard gen == generation else { return }
                await received(text, on: opened, generation: gen)
            } catch {
                receiveEnded(error as? DialFailure ?? .unreachable, generation: gen)
                return
            }
        }
    }

    private func received(_ text: String, on opened: any WebSocketChannel, generation gen: Int) async {
        guard let message = try? WireCodec.decodeServer(text) else { return }
        switch message {
        case .helloAck(let ack):
            ackTimeoutTask?.cancel()
            ackTimeoutTask = nil
            heartbeatInterval = (5...60).contains(ack.pingIntervalS)
                ? TimeInterval(ack.pingIntervalS) : TimeInterval(ProtocolConstants.pingIntervalS)
            everConnected = true
            lastFailure = nil
            apply(.ackReceived)
        case .ping(let ping):
            // The Test round trip (D-15): answer at once, change nothing visible.
            if let reply = try? WireCodec.encode(.pong(Pong(id: ping.id))) {
                try? await opened.send(text: reply)
            }
        case .pong(let pong):
            if pong.id == pendingPingId { pendingPingId = nil }
        case .error(let error):
            log.info("The server sent an error frame with code \(error.code, privacy: .public).")
        case .unknown:
            break
        }
    }

    private func receiveEnded(_ failure: DialFailure, generation gen: Int) {
        guard gen == generation else { return }
        lastFailure = failure
        if case .connected = machine.status {
            var code: Int?
            if case .closed(let value) = failure { code = value }
            if let code { log.info("The server closed the socket with code \(code, privacy: .public).") }
            dropSocket(code: 1001)
            apply(.socketClosed(code: code))
        } else {
            dropSocket(code: 1001)
            apply(.dialFailed(failure))
        }
    }

    // MARK: - Socket

    /// Ends the current dial and socket. Every task tied to it is cancelled and
    /// its late results are ignored.
    private func dropSocket(code: Int) {
        generation += 1
        dialTask?.cancel()
        dialTask = nil
        receiveTask?.cancel()
        receiveTask = nil
        ackTimeoutTask?.cancel()
        ackTimeoutTask = nil
        if let old = channel {
            channel = nil
            Task { await old.close(code: code) }
        }
    }

    // MARK: - Retry timer

    private func scheduleRetry(after delay: TimeInterval) {
        retryTask?.cancel()
        retryEpoch += 1
        let epoch = retryEpoch
        retryTask = Task { [clock] in
            do { try await clock.sleep(for: delay) } catch { return }
            self.retryDue(epoch: epoch)
        }
    }

    private func cancelRetry() {
        retryTask?.cancel()
        retryTask = nil
        retryEpoch += 1
    }

    private func retryDue(epoch: Int) {
        guard epoch == retryEpoch else { return }
        retryTask = nil
        apply(.retryFired)
    }

    // MARK: - Heartbeat

    private func startHeartbeat() {
        stopHeartbeat()
        let gen = generation
        let interval = heartbeatInterval
        heartbeatTask = Task {
            await self.heartbeatLoop(generation: gen, interval: interval)
        }
    }

    private func stopHeartbeat() {
        heartbeatTask?.cancel()
        heartbeatTask = nil
        pongWatchTask?.cancel()
        pongWatchTask = nil
        pendingPingId = nil
    }

    private func heartbeatLoop(generation gen: Int, interval: TimeInterval) async {
        while !Task.isCancelled {
            do { try await clock.sleep(for: interval) } catch { return }
            guard gen == generation, let live = channel else { return }
            let id = nextPingId
            nextPingId += 1
            pendingPingId = id
            pongWatchTask?.cancel()
            pongWatchTask = Task { [clock] in
                do { try await clock.sleep(for: TimeInterval(ProtocolConstants.pongTimeoutS)) } catch { return }
                self.pongWatch(id: id, generation: gen)
            }
            if let text = try? WireCodec.encode(.ping(Ping(id: id))) {
                try? await live.send(text: text)
            }
        }
    }

    private func pongWatch(id: Int, generation gen: Int) {
        guard gen == generation, pendingPingId == id else { return }
        log.info("No pong within the time limit. The socket is treated as dead.")
        lastFailure = .unreachable
        apply(.pongMissed)
    }
}
