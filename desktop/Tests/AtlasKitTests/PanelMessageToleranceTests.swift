import Foundation
import Testing

@testable import AtlasKit

/// What the Mac accepts that the server never sends, and what it still
/// refuses (D-10, D-11). A newer server must never blank the panel, and an
/// over-cap string must never reach it.
@Suite struct PanelMessageToleranceTests {
    private let turn = "5f1c2a9e7b3d4c60a8e2f7b1c9d04e3a"

    private func card(kind: String, fallback: String = "Fallback.", data: String? = nil) -> String {
        let dataPart = data.map { #","data":\#($0)"# } ?? ""
        return #"{"type":"card","turn_id":"\#(turn)","card_id":"c-1","kind":"\#(kind)","fallback_text":"\#(fallback)"\#(dataPart)}"#
    }

    @Test func anUnknownCardKindKeepsItsFallbackText() throws {
        let decoded = try WireCodec.decodeServer(card(kind: "weather"))
        guard case .card(let value) = decoded else {
            Issue.record("not a card")
            return
        }
        #expect(value.data == .unknown)
        #expect(value.fallbackText == "Fallback.")
        #expect(value.displayText == "Fallback.")
    }

    @Test func anUnknownCardKindWithAnEmptyFallbackStillDecodes() throws {
        let decoded = try WireCodec.decodeServer(card(kind: "weather", fallback: ""))
        guard case .card(let value) = decoded else {
            Issue.record("not a card")
            return
        }
        #expect(value.displayText == nil)
    }

    @Test func anUnknownCardEncodesNoDataKey() throws {
        let decoded = try WireCodec.decodeServer(card(kind: "weather", data: #"{"x":1}"#))
        guard case .card(let value) = decoded else {
            Issue.record("not a card")
            return
        }
        let object = try #require(
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
        #expect(object["data"] == nil)
        #expect(object["kind"] as? String == "weather")
    }

    @Test func aTextCardShowsItsDataText() throws {
        let decoded = try WireCodec.decodeServer(
            card(kind: "text", fallback: "Fallback.", data: #"{"text":"Data text."}"#))
        guard case .card(let value) = decoded else {
            Issue.record("not a card")
            return
        }
        #expect(value.displayText == "Data text.")
    }

    @Test func aStateAndAnOutcomeWordThisBuildDoesNotKnowStillDecode() throws {
        let state = try WireCodec.decodeServer(
            #"{"type":"state","turn_id":"\#(turn)","state":"dancing"}"#)
        #expect(state == .turnState(TurnStateMessage(turnId: turn, state: "dancing")))
        let ended = try WireCodec.decodeServer(
            #"{"type":"turn.ended","turn_id":"\#(turn)","outcome":"exploded","follow_up_window_ms":0,"playback_ms_left":0}"#)
        #expect(
            ended
                == .turnEnded(
                    TurnEnded(turnId: turn, outcome: "exploded", followUpWindowMs: 0, playbackMsLeft: 0)))
    }

    @Test func anExtraKeyIsIgnored() throws {
        let decoded = try WireCodec.decodeServer(
            #"{"type":"timer.stopped","timer_id":5,"future_field":"x"}"#)
        #expect(decoded == .timerStopped(TimerStopped(timerId: 5)))
    }

    @Test func aCardTextOf600ScalarsDecodesAndOf601DoesNot() throws {
        let ok = String(repeating: "x", count: 600)
        let tooLong = String(repeating: "x", count: 601)
        _ = try WireCodec.decodeServer(card(kind: "text", fallback: ok, data: #"{"text":"\#(ok)"}"#))
        #expect(throws: (any Error).self) {
            try WireCodec.decodeServer(
                card(kind: "text", fallback: ok, data: #"{"text":"\#(tooLong)"}"#))
        }
        #expect(throws: (any Error).self) {
            try WireCodec.decodeServer(
                card(kind: "text", fallback: tooLong, data: #"{"text":"\#(ok)"}"#))
        }
    }

    @Test func scalarsNotCharactersAreCounted() throws {
        // A flag is one Character and two scalars. 300 of them are 600 scalars.
        let flags = String(repeating: "\u{1F1E9}\u{1F1EA}", count: 300)
        _ = try WireCodec.decodeServer(
            card(kind: "text", fallback: "ok", data: #"{"text":"\#(flags)"}"#))
        #expect(throws: (any Error).self) {
            try WireCodec.decodeServer(
                card(kind: "text", fallback: "ok", data: #"{"text":"\#(flags)x"}"#))
        }
    }

    @Test func aTimerStopFromTheMacEncodesTheFixtureKeys() throws {
        let text = try WireCodec.encode(.timerStop(TimerStop(timerId: 12)))
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        #expect(Set(object.keys) == ["type", "timer_id"])
        #expect(object["type"] as? String == "timer.stop")
    }

    private func fixtureText(_ name: String) throws -> String {
        let url = desktopRoot().appending(path: "protocol/v1/messages/\(name).json")
        let root = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let message = try #require(root["message"] as? [String: Any])
        return try #require(
            String(data: try JSONSerialization.data(withJSONObject: message), encoding: .utf8))
    }

    @Test func panelFramesComeOutOfTheInboundStreamInOrder() async throws {
        let rig = Rig()
        let channel = await rig.connect()
        let texts = [
            try fixtureText("turn_ended"), try fixtureText("card_text"),
            try fixtureText("timer_ringing"),
        ]

        let received = await withTaskGroup(of: [ServerMessage].self) { group in
            group.addTask {
                var iterator = rig.connection.inbound.makeAsyncIterator()
                var seen: [ServerMessage] = []
                while seen.count < 3, let next = await iterator.next() {
                    if case .frame(let message) = next { seen.append(message) }
                }
                return seen
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(2))
                return []
            }
            for text in texts { channel.push(text) }
            let result = await group.next() ?? []
            group.cancelAll()
            return result
        }

        #expect(received.count == 3)
        guard received.count == 3 else { return }
        guard case .turnEnded = received[0] else { Issue.record("first is not turn.ended"); return }
        guard case .card = received[1] else { Issue.record("second is not a card"); return }
        guard case .timerRinging = received[2] else { Issue.record("third is not timer.ringing"); return }
    }
}
