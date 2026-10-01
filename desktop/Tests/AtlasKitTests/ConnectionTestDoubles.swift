import Foundation
import Synchronization

@testable import AtlasKit

/// A clock that moves only when a test says so. A sleeper wakes when `advance`
/// passes its deadline, in deadline order.
final class TestClock: ConnectionClock {
    private struct Sleeper {
        let deadline: Date
        let continuation: CheckedContinuation<Void, any Error>
    }

    private struct State {
        var now = Date(timeIntervalSince1970: 1_000_000)
        var sleepers: [Int: Sleeper] = [:]
        var cancelledEarly: Set<Int> = []
        var nextId = 0
    }

    private let state = Mutex(State())

    func now() -> Date { state.withLock { $0.now } }

    var sleeperCount: Int { state.withLock { $0.sleepers.count } }

    func sleep(for seconds: TimeInterval) async throws {
        let id = state.withLock { s -> Int in
            s.nextId += 1
            return s.nextId
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                let cancelled = state.withLock { s -> Bool in
                    if s.cancelledEarly.remove(id) != nil { return true }
                    s.sleepers[id] = Sleeper(deadline: s.now.addingTimeInterval(seconds), continuation: continuation)
                    return false
                }
                if cancelled { continuation.resume(throwing: CancellationError()) }
            }
        } onCancel: {
            let waiting = state.withLock { s -> CheckedContinuation<Void, any Error>? in
                if let sleeper = s.sleepers.removeValue(forKey: id) { return sleeper.continuation }
                s.cancelledEarly.insert(id)
                return nil
            }
            waiting?.resume(throwing: CancellationError())
        }
    }

    /// Wakes every sleeper whose deadline falls inside the step, earliest first,
    /// and gives the woken work a moment to run before the next one.
    func advance(by seconds: TimeInterval) async {
        let target = now().addingTimeInterval(seconds)
        while true {
            let next = state.withLock { s -> CheckedContinuation<Void, any Error>? in
                let due = s.sleepers.filter { $0.value.deadline <= target }
                guard let earliest = due.min(by: { $0.value.deadline < $1.value.deadline }) else {
                    s.now = target
                    return nil
                }
                s.sleepers.removeValue(forKey: earliest.key)
                s.now = max(s.now, earliest.value.deadline)
                return earliest.value.continuation
            }
            guard let next else { return }
            next.resume()
            await settle()
        }
    }

    /// Waits until the actor under test has parked at least `count` sleepers.
    func waitForSleepers(atLeast count: Int = 1) async -> Bool {
        await eventually { self.sleeperCount >= count }
    }
}

/// Lets scheduled work run. Used after a step when a test expects nothing to happen.
func settle() async {
    try? await Task.sleep(for: .milliseconds(25))
}

/// Polls a condition for up to two seconds of wall time, so a broken rule fails
/// fast and never hangs the suite.
func eventually(_ condition: @Sendable () async -> Bool) async -> Bool {
    let deadline = ContinuousClock.now + .seconds(2)
    while ContinuousClock.now < deadline {
        if await condition() { return true }
        try? await Task.sleep(for: .milliseconds(2))
    }
    return await condition()
}

/// One scripted socket. Frames the test pushes come out of `receive` in order.
final class FakeChannel: WebSocketChannel {
    private enum Inbound {
        case text(String)
        case failure(DialFailure)
    }

    private struct State {
        var queue: [Inbound] = []
        var waiter: CheckedContinuation<String, any Error>?
        var sent: [String] = []
        var closeCode: Int?
        var failSend: DialFailure?
    }

    private let state = Mutex(State())

    var sent: [String] { state.withLock { $0.sent } }
    var closeCode: Int? { state.withLock { $0.closeCode } }

    func failSends(with failure: DialFailure) { state.withLock { $0.failSend = failure } }

    func push(_ text: String) { deliver(.text(text)) }
    func push(failure: DialFailure) { deliver(.failure(failure)) }

    func pushAck(interval: Int = 15) {
        push(#"{"type":"hello.ack","protocol":1,"device_id":7,"ping_interval_s":\#(interval)}"#)
    }

    func pushPing(id: Int) { push(#"{"type":"ping","id":\#(id)}"#) }
    func pushPong(id: Int) { push(#"{"type":"pong","id":\#(id)}"#) }

    func send(text: String) async throws {
        let failure = state.withLock { s -> DialFailure? in
            if let failure = s.failSend { return failure }
            s.sent.append(text)
            return nil
        }
        if let failure { throw failure }
    }

    func receive() async throws -> String {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, any Error>) in
                let ready = state.withLock { s -> Result<String, any Error>? in
                    if !s.queue.isEmpty {
                        switch s.queue.removeFirst() {
                        case .text(let text): return .success(text)
                        case .failure(let failure): return .failure(failure)
                        }
                    }
                    if s.closeCode != nil { return .failure(DialFailure.unreachable) }
                    s.waiter = continuation
                    return nil
                }
                if let ready { continuation.resume(with: ready) }
            }
        } onCancel: {
            let waiter = state.withLock { s -> CheckedContinuation<String, any Error>? in
                defer { s.waiter = nil }
                return s.waiter
            }
            waiter?.resume(throwing: CancellationError())
        }
    }

    func close(code: Int) async {
        let waiter = state.withLock { s -> CheckedContinuation<String, any Error>? in
            if s.closeCode == nil { s.closeCode = code }
            defer { s.waiter = nil }
            return s.waiter
        }
        waiter?.resume(throwing: DialFailure.unreachable)
    }

    private func deliver(_ item: Inbound) {
        let waiter = state.withLock { s -> CheckedContinuation<String, any Error>? in
            if let waiter = s.waiter {
                s.waiter = nil
                return waiter
            }
            s.queue.append(item)
            return nil
        }
        guard let waiter else { return }
        switch item {
        case .text(let text): waiter.resume(returning: text)
        case .failure(let failure): waiter.resume(throwing: failure)
        }
    }
}

/// Records every open and returns the next scripted result. With nothing
/// scripted it returns a silent channel.
final class FakeTransport: WebSocketTransport {
    enum Step {
        case channel(FakeChannel)
        case failure(DialFailure)
    }

    struct Open: Equatable {
        let url: URL
        let bearerToken: String
    }

    private struct State {
        var script: [Step] = []
        var opens: [Open] = []
        var channels: [FakeChannel] = []
    }

    private let state = Mutex(State())

    var opens: [Open] { state.withLock { $0.opens } }
    var channels: [FakeChannel] { state.withLock { $0.channels } }

    func script(_ step: Step) { state.withLock { $0.script.append(step) } }

    /// Queues a channel and returns it so a test can script its frames.
    @discardableResult
    func scriptChannel() -> FakeChannel {
        let channel = FakeChannel()
        script(.channel(channel))
        return channel
    }

    func open(url: URL, bearerToken: String) async throws -> any WebSocketChannel {
        let step = state.withLock { s -> Step in
            s.opens.append(Open(url: url, bearerToken: bearerToken))
            let next = s.script.isEmpty ? Step.channel(FakeChannel()) : s.script.removeFirst()
            if case .channel(let channel) = next { s.channels.append(channel) }
            return next
        }
        switch step {
        case .channel(let channel): return channel
        case .failure(let failure): throw failure
        }
    }
}
