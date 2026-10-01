import Foundation
import Testing

@testable import AtlasKit

@Suite struct ConnectionStateMachineTests {
    private let t0 = Date(timeIntervalSince1970: 1_000_000)
    private let host = "svr.test"

    private func at(_ seconds: TimeInterval) -> Date { t0.addingTimeInterval(seconds) }

    @discardableResult
    private func send(
        _ machine: inout ConnectionStateMachine, _ event: ConnectionEvent,
        at seconds: TimeInterval = 0, random: Double = 0.5
    ) -> [ConnectionEffect] {
        machine.handle(event, now: at(seconds), random: random)
    }

    private func started() -> ConnectionStateMachine {
        var machine = ConnectionStateMachine()
        send(&machine, .start(host: host))
        return machine
    }

    private func connected(at seconds: TimeInterval = 0) -> ConnectionStateMachine {
        var machine = started()
        send(&machine, .dialSucceeded, at: seconds)
        send(&machine, .ackReceived, at: seconds)
        return machine
    }

    @Test func startDialsAndReportsConnecting() {
        var machine = ConnectionStateMachine()
        #expect(machine.status == .unpaired)
        #expect(send(&machine, .start(host: host)) == [.dial])
        #expect(machine.status == .connecting(host: host))
    }

    @Test func aSocketOpenKeepsConnectingAndTheAckConnects() {
        var machine = started()
        #expect(send(&machine, .dialSucceeded).isEmpty)
        #expect(machine.status == .connecting(host: host))
        #expect(send(&machine, .ackReceived, at: 3) == [.cancelRetry, .startHeartbeat])
        #expect(machine.status == .connected(host: host, since: at(3)))
    }

    @Test func eachFailedDialWaitsLongerAndTheMenuStaysOffline() {
        var machine = started()
        var delays: [TimeInterval] = []
        for _ in 0..<3 {
            let effects = send(&machine, .dialFailed(.unreachable))
            #expect(machine.status == .offline(host: host, lastConnected: nil))
            if case .scheduleRetry(let delay)? = effects.first { delays.append(delay) }
            #expect(send(&machine, .retryFired) == [.dial])
            #expect(machine.status == .connecting(host: host))
        }
        #expect(delays == [1.0, 2.0, 4.0])
    }

    @Test func aLiveCloseWith4001RevokesAtOnceAndStaysRevoked() {
        var machine = connected()
        let effects = send(&machine, .socketClosed(code: 4001), at: 10)
        #expect(effects == [.stopHeartbeat, .cancelRetry, .unpair])
        #expect(machine.status == .revoked(host: host))
        for event: ConnectionEvent in [
            .retryFired, .pathUnsatisfied, .willSleep, .didWake, .pathSatisfied, .pongMissed,
            .dialFailed(.unreachable), .socketClosed(code: nil), .stop,
        ] {
            #expect(send(&machine, event, at: 20).isEmpty)
            #expect(machine.status == .revoked(host: host))
        }
        #expect(send(&machine, .start(host: host), at: 30) == [.dial])
        #expect(machine.status == .connecting(host: host))
    }

    @Test func aClosedFailureOnTheHandshakeWith4001RevokesToo() {
        var machine = started()
        #expect(send(&machine, .dialFailed(.closed(code: 4001))) == [.stopHeartbeat, .cancelRetry, .unpair])
        #expect(machine.status == .revoked(host: host))
    }

    @Test func twoHandshake403sRevoke() {
        var machine = started()
        let first = send(&machine, .dialFailed(.refused))
        #expect(first == [.scheduleRetry(after: 1.0)])
        #expect(machine.status == .offline(host: host, lastConnected: nil))
        send(&machine, .retryFired)
        let second = send(&machine, .dialFailed(.refused))
        #expect(second == [.stopHeartbeat, .cancelRetry, .unpair])
        #expect(machine.status == .revoked(host: host))
    }

    @Test func anUnreachableDialBetweenTwoRefusalsDoesNotBreakThePair() {
        var machine = started()
        send(&machine, .dialFailed(.refused))
        send(&machine, .retryFired)
        send(&machine, .dialFailed(.unreachable))
        send(&machine, .retryFired)
        #expect(send(&machine, .dialFailed(.refused)).contains(.unpair))
        #expect(machine.status == .revoked(host: host))
    }

    @Test func anAcceptedHelloBetweenTwoRefusalsResetsTheCount() {
        var machine = started()
        send(&machine, .dialFailed(.refused))
        send(&machine, .retryFired)
        send(&machine, .dialSucceeded)
        send(&machine, .ackReceived, at: 2)
        send(&machine, .socketClosed(code: nil), at: 5)
        send(&machine, .retryFired, at: 6)
        let effects = send(&machine, .dialFailed(.refused), at: 6)
        #expect(!effects.contains(.unpair))
        #expect(machine.status == .offline(host: host, lastConnected: at(5)))
    }

    @Test func aProtocolMismatchStopsInItsOwnState() {
        var machine = connected()
        let effects = send(&machine, .socketClosed(code: 4002), at: 1)
        #expect(machine.status == .versionMismatch(host: host))
        #expect(effects.contains(.stopHeartbeat))
        #expect(!effects.contains { if case .scheduleRetry = $0 { true } else { false } })
        #expect(send(&machine, .retryFired).isEmpty)
        #expect(send(&machine, .pathSatisfied).isEmpty)
    }

    @Test func aSupersedeWaitsTheFullCapBeforeTheNextDial() {
        var machine = connected()
        let effects = send(&machine, .socketClosed(code: 4000), at: 1)
        #expect(effects.contains(.scheduleRetry(after: 60)))
        #expect(machine.status == .offline(host: host, lastConnected: at(1)))
    }

    @Test func aConnectionThatStayedUpThirtySecondsResetsTheAttempt() {
        var machine = started()
        send(&machine, .dialFailed(.unreachable))
        send(&machine, .retryFired)
        send(&machine, .dialFailed(.unreachable))
        send(&machine, .retryFired)
        send(&machine, .dialSucceeded, at: 100)
        send(&machine, .ackReceived, at: 100)
        let effects = send(&machine, .socketClosed(code: nil), at: 130)
        #expect(effects.contains(.scheduleRetry(after: 1.0)))
    }

    @Test func aConnectionThatClosedSoonerKeepsTheAttemptCount() {
        var machine = started()
        send(&machine, .dialFailed(.unreachable))
        send(&machine, .retryFired)
        send(&machine, .dialFailed(.unreachable))
        send(&machine, .retryFired)
        send(&machine, .dialSucceeded, at: 100)
        send(&machine, .ackReceived, at: 100)
        let effects = send(&machine, .socketClosed(code: nil), at: 129)
        #expect(effects.contains(.scheduleRetry(after: 4.0)))
    }

    @Test func aMissingPongClosesTheSocketAndRetries() {
        var machine = connected()
        let effects = send(&machine, .pongMissed, at: 25)
        #expect(effects == [.closeSocket(code: 1001), .stopHeartbeat, .scheduleRetry(after: 1.0)])
        #expect(machine.status == .offline(host: host, lastConnected: at(25)))
    }

    @Test func aLostPathStopsAllDialingUntilItReturns() {
        var machine = started()
        send(&machine, .dialFailed(.unreachable))
        #expect(send(&machine, .pathUnsatisfied) == [.cancelRetry])
        #expect(send(&machine, .retryFired).isEmpty)
        #expect(machine.status == .offline(host: host, lastConnected: nil))
        let back = send(&machine, .pathSatisfied)
        #expect(back == [.cancelRetry, .scheduleRetry(after: 1.0)])
        #expect(send(&machine, .retryFired) == [.dial])
    }

    @Test func aReturningPathResetsTheAttemptCount() {
        var machine = started()
        for _ in 0..<3 {
            send(&machine, .dialFailed(.unreachable))
            send(&machine, .retryFired)
        }
        send(&machine, .dialFailed(.unreachable))
        send(&machine, .pathSatisfied)
        send(&machine, .retryFired)
        let effects = send(&machine, .dialFailed(.unreachable))
        #expect(effects == [.scheduleRetry(after: 1.0)])
    }

    @Test func aPathLostWhileConnectedClosesTheSocket() {
        var machine = connected()
        let effects = send(&machine, .pathUnsatisfied, at: 40)
        #expect(effects.contains(.closeSocket(code: 1001)))
        #expect(effects.contains(.cancelRetry))
        #expect(!effects.contains(.dial))
        #expect(machine.status == .offline(host: host, lastConnected: at(40)))
    }

    @Test func sleepClosesCleanlyAndTheWakeDialsAboutASecondLater() {
        var machine = connected()
        let sleep = send(&machine, .willSleep, at: 50)
        #expect(sleep == [.stopHeartbeat, .closeSocket(code: 1001), .cancelRetry])
        #expect(send(&machine, .retryFired).isEmpty)
        #expect(send(&machine, .pathSatisfied).isEmpty)
        #expect(machine.status == .offline(host: host, lastConnected: at(50)))
        let wake = send(&machine, .didWake, at: 900)
        #expect(wake.contains(.scheduleRetry(after: 1.0)))
        #expect(send(&machine, .retryFired, at: 901) == [.dial])
    }

    @Test func aWakeWithNoPathWaitsForThePath() {
        var machine = connected()
        send(&machine, .willSleep)
        send(&machine, .pathUnsatisfied)
        #expect(send(&machine, .didWake).isEmpty)
        #expect(send(&machine, .pathSatisfied) == [.cancelRetry, .scheduleRetry(after: 1.0)])
    }

    @Test func aWakeWithALiveSocketDropsItBecauseItIsProbablyHalfOpen() {
        var machine = connected()
        let effects = send(&machine, .didWake, at: 10)
        #expect(effects.contains(.closeSocket(code: 1001)))
        #expect(effects.contains(.scheduleRetry(after: 1.0)))
    }

    @Test func stopUnpairsAndTearsEverythingDown() {
        var machine = connected()
        let effects = send(&machine, .stop)
        #expect(effects == [.cancelRetry, .stopHeartbeat, .closeSocket(code: 1000)])
        #expect(machine.status == .unpaired)
        #expect(send(&machine, .retryFired).isEmpty)
    }

    @Test func aSupersedeWithNoPathSchedulesNothing() {
        var machine = connected()
        send(&machine, .pathUnsatisfied)
        #expect(send(&machine, .socketClosed(code: 4000)).isEmpty)
        #expect(machine.status == .offline(host: host, lastConnected: at(0)))
    }

    @Test func anUnknownCloseCodeBacksOffLikeAnyOtherFailure() {
        var machine = connected()
        let effects = send(&machine, .socketClosed(code: 1011))
        #expect(effects.contains(.scheduleRetry(after: 1.0)))
    }

    @Test func aRestartWhileLiveClosesTheOldSocketBeforeTheNewDial() {
        var machine = connected()
        let effects = send(&machine, .start(host: "other.test"))
        #expect(effects == [.cancelRetry, .stopHeartbeat, .closeSocket(code: 1000), .dial])
        #expect(machine.status == .connecting(host: "other.test"))
    }
}
