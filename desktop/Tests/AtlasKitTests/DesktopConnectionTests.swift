import Foundation
import Synchronization
import Testing

@testable import AtlasKit

/// A connection wired to a fake transport and a clock the test moves by hand.
final class Rig: Sendable {
    let transport = FakeTransport()
    let clock = TestClock()
    let store = InMemorySecretStore()
    let credentials = PairingCredentials(host: "svr.test", token: "t-1")
    let connection: DesktopConnection

    init() {
        try? store.save(credentials)
        connection = DesktopConnection(
            transport: transport, clock: clock, store: store,
            helloInfo: HelloInfo(appVersion: "1.0", osVersion: "15.0"),
            random: { 0.5 })
    }

    func start() async { await connection.start(credentials: credentials) }

    func status() async -> LinkStatus { await connection.currentSnapshot.status }

    func waitConnected() async -> Bool {
        await eventually {
            if case .connected = await self.status() { return true }
            return false
        }
    }

    func waitOffline() async -> Bool {
        await eventually {
            if case .offline = await self.status() { return true }
            return false
        }
    }

    func waitStatus(_ expected: LinkStatus) async -> Bool {
        await eventually { await self.status() == expected }
    }

    func waitForOpens(_ count: Int) async -> Bool {
        await eventually { self.transport.opens.count >= count }
    }

    /// Connects with a channel that already holds a hello.ack.
    @discardableResult
    func connect() async -> FakeChannel {
        let channel = transport.scriptChannel()
        channel.pushAck()
        await start()
        _ = await waitConnected()
        return channel
    }
}

@Suite struct DesktopConnectionTests {
    @Test func dialsOverWssWithTheBearerHeaderAndSendsAHelloFirst() async throws {
        let rig = Rig()
        let channel = rig.transport.scriptChannel()
        await rig.start()
        #expect(await rig.waitForOpens(1))
        let open = try #require(rig.transport.opens.first)
        #expect(open.url.absoluteString == "wss://svr.test/ws/desktop")
        #expect(open.bearerToken == "t-1")
        #expect(!open.url.absoluteString.contains("t-1"))

        #expect(await eventually { !channel.sent.isEmpty })
        let data = Data(try #require(channel.sent.first).utf8)
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(Set(object.keys) == ["type", "protocol", "app_version", "os_version", "capabilities"])
        #expect(object["type"] as? String == "hello")
        #expect(object["protocol"] as? Int == 1)
        #expect((object["capabilities"] as? [Any])?.isEmpty == true)
    }

    @Test func aHelloAckConnectsAndAServerPingGetsASilentPong() async throws {
        let rig = Rig()
        let channel = await rig.connect()
        let before = await rig.connection.currentSnapshot
        guard case .connected(let host, _) = before.status else {
            Issue.record("not connected")
            return
        }
        #expect(host == "svr.test")

        channel.pushPing(id: 3)
        #expect(await eventually { channel.sent.count >= 2 })
        let reply = try WireCodec.decodeClient(try #require(channel.sent.last))
        #expect(reply == .pong(Pong(id: 3)))
        await settle()
        #expect(await rig.connection.currentSnapshot == before)
    }

    @Test func aWakeConfirmedFrameComesOutOfTheInboundStream() async throws {
        let rig = Rig()
        let channel = await rig.connect()

        let url = desktopRoot().appending(path: "protocol/v1/messages/wake_confirmed.json")
        let root = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let message = try #require(root["message"] as? [String: Any])
        let turnId = try #require(message["turn_id"] as? String)
        let text = try #require(
            String(data: try JSONSerialization.data(withJSONObject: message), encoding: .utf8))

        // A bounded wait: the stream must yield, or the task group's timeout wins.
        let first = await withTaskGroup(of: ServerMessage?.self) { group in
            group.addTask {
                var iterator = rig.connection.inbound.makeAsyncIterator()
                return await iterator.next()
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(2))
                return nil
            }
            channel.push(text)
            let result = await group.next() ?? nil
            group.cancelAll()
            return result
        }
        #expect(first == .wakeConfirmed(WakeConfirmed(turnId: turnId)))
    }

    @Test func aMissingPongClosesTheSocketAndTheNextDialFollowsTheBackoff() async throws {
        let rig = Rig()
        let channel = await rig.connect()
        #expect(await rig.clock.waitForSleepers())

        await rig.clock.advance(by: 15)
        #expect(await eventually {
            channel.sent.contains { (try? WireCodec.decodeClient($0)) == .ping(Ping(id: 1)) }
        })
        await rig.clock.advance(by: 10)
        #expect(await rig.waitOffline())
        #expect(channel.closeCode == 1001)
        #expect(await rig.connection.currentSnapshot.lastFailure == .unreachable)

        #expect(await rig.clock.waitForSleepers())
        #expect(rig.transport.opens.count == 1)
        await rig.clock.advance(by: 1)
        #expect(await rig.waitForOpens(2))
    }

    @Test func aPongInTimeKeepsTheSocketOpen() async {
        let rig = Rig()
        let channel = await rig.connect()
        #expect(await rig.clock.waitForSleepers())
        await rig.clock.advance(by: 15)
        #expect(await eventually { channel.sent.count >= 2 })
        channel.pushPong(id: 1)
        await settle()
        await rig.clock.advance(by: 10)
        await settle()
        if case .connected = await rig.status() {} else { Issue.record("the socket was dropped") }
        #expect(channel.closeCode == nil)
    }

    @Test func aLiveCloseWith4001RevokesDeletesTheTokenAndNeverDialsAgain() async throws {
        let rig = Rig()
        let channel = await rig.connect()
        channel.push(failure: .closed(code: 4001))
        #expect(await rig.waitStatus(.revoked(host: "svr.test")))
        #expect(try rig.store.load() == nil)

        await rig.clock.advance(by: 600)
        await settle()
        #expect(rig.transport.opens.count == 1)
    }

    @Test func twoRefusedDialsRevokeAndOneKeepsTheToken() async throws {
        let rig = Rig()
        rig.transport.script(.failure(.refused))
        rig.transport.script(.failure(.refused))
        await rig.start()
        #expect(await rig.waitOffline())
        #expect(try rig.store.load() != nil)

        #expect(await rig.clock.waitForSleepers())
        await rig.clock.advance(by: 1)
        #expect(await rig.waitStatus(.revoked(host: "svr.test")))
        #expect(try rig.store.load() == nil)
        await rig.clock.advance(by: 600)
        await settle()
        #expect(rig.transport.opens.count == 2)
    }

    @Test func aProtocolMismatchStopsDialingAndKeepsTheToken() async throws {
        let rig = Rig()
        let channel = await rig.connect()
        channel.push(failure: .closed(code: 4002))
        #expect(await rig.waitStatus(.versionMismatch(host: "svr.test")))
        await rig.clock.advance(by: 600)
        await settle()
        #expect(rig.transport.opens.count == 1)
        #expect(try rig.store.load() != nil)
    }

    @Test func sleepClosesTheSocketAndTheWakeDialsAboutASecondLater() async {
        let rig = Rig()
        let channel = await rig.connect()
        await rig.connection.handle(.willSleep)
        #expect(await eventually { channel.closeCode == 1001 })
        #expect(await rig.waitOffline())

        await rig.clock.advance(by: 600)
        await settle()
        #expect(rig.transport.opens.count == 1)

        await rig.connection.handle(.didWake)
        #expect(await rig.clock.waitForSleepers())
        #expect(rig.transport.opens.count == 1)
        await rig.clock.advance(by: 1)
        #expect(await rig.waitForOpens(2))
    }

    @Test func aLostPathStopsDialingAndAReturningPathDialsAboutASecondLater() async {
        let rig = Rig()
        let channel = await rig.connect()
        await rig.connection.handle(.pathUnsatisfied)
        #expect(await eventually { channel.closeCode != nil })
        #expect(await rig.waitOffline())

        await rig.clock.advance(by: 600)
        await settle()
        #expect(rig.transport.opens.count == 1)

        await rig.connection.handle(.pathSatisfied)
        #expect(await rig.clock.waitForSleepers())
        await rig.clock.advance(by: 1)
        #expect(await rig.waitForOpens(2))
    }

    @Test func aRetryWaitingWhenThePathDropsIsCancelled() async {
        let rig = Rig()
        rig.transport.script(.failure(.unreachable))
        await rig.start()
        #expect(await rig.waitOffline())
        #expect(await rig.clock.waitForSleepers())
        await rig.connection.handle(.pathUnsatisfied)
        await settle()
        await rig.clock.advance(by: 600)
        await settle()
        #expect(rig.transport.opens.count == 1)
    }

    @Test func noHelloAckInTenSecondsClosesTheChannelAsUnreachable() async {
        let rig = Rig()
        let channel = rig.transport.scriptChannel()
        await rig.start()
        #expect(await rig.waitForOpens(1))
        #expect(await rig.clock.waitForSleepers())
        await rig.clock.advance(by: 10)
        #expect(await rig.waitOffline())
        #expect(channel.closeCode != nil)
        #expect(await rig.connection.currentSnapshot.lastFailure == .unreachable)
    }

    @Test func everConnectedTurnsTrueAtTheFirstAckAndStaysTrue() async {
        let rig = Rig()
        #expect(await rig.connection.currentSnapshot.everConnected == false)
        let channel = await rig.connect()
        #expect(await rig.connection.currentSnapshot.everConnected)
        channel.push(failure: .unreachable)
        #expect(await rig.waitOffline())
        #expect(await rig.connection.currentSnapshot.everConnected)
    }

    @Test func stopClosesTheSocketAndReportsUnpaired() async {
        let rig = Rig()
        let channel = await rig.connect()
        await rig.connection.stop()
        #expect(await rig.status() == .unpaired)
        #expect(await eventually { channel.closeCode == 1000 })
    }

    @Test func theSnapshotStreamReportsEachStateInOrder() async {
        let rig = Rig()
        let seen = Mutex<[LinkStatus]>([])
        let stream = rig.connection.snapshots
        let reader = Task {
            for await snapshot in stream { seen.withLock { $0.append(snapshot.status) } }
        }
        await rig.connect()
        #expect(await eventually { seen.withLock { $0.count >= 3 } })
        reader.cancel()
        let states = seen.withLock { $0 }
        #expect(states.first == .unpaired)
        #expect(states.dropFirst().first == .connecting(host: "svr.test"))
        if case .connected = states[2] {} else { Issue.record("third state was not connected") }
    }
}
