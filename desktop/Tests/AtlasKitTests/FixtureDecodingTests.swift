import Foundation
import Testing

@testable import AtlasKit

/// The Mac half of the wire contract (D-09, D-21). It reads the same files in
/// `desktop/protocol/v1` that `tests/test_desktop_protocol_contract.py` reads,
/// so a change on either side fails the other side's suite.
@Suite struct FixtureDecodingTests {
    private struct Fixture {
        let name: String
        let direction: String
        let unknown: Bool
        let text: String
        let message: NSDictionary
    }

    private func protocolRoot() -> URL {
        desktopRoot().appending(path: "protocol/v1")
    }

    private func jsonFiles(in directory: String) throws -> [URL] {
        let dir = protocolRoot().appending(path: directory)
        return try FileManager.default
            .contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func readObject(_ url: URL) throws -> NSDictionary {
        let data = try Data(contentsOf: url)
        return try #require(
            try JSONSerialization.jsonObject(with: data) as? NSDictionary,
            "\(url.lastPathComponent) is not a JSON object")
    }

    private func fixtures(in directory: String) throws -> [Fixture] {
        try jsonFiles(in: directory).map { url in
            let root = try readObject(url)
            let message = try #require(root["message"] as? NSDictionary)
            let data = try JSONSerialization.data(withJSONObject: message)
            return Fixture(
                name: url.lastPathComponent,
                direction: try #require(root["direction"] as? String),
                unknown: (root["unknown"] as? Bool) ?? false,
                text: try #require(String(data: data, encoding: .utf8)),
                message: message)
        }
    }

    private func object(fromEncoded text: String) throws -> NSDictionary {
        let data = Data(text.utf8)
        return try #require(try JSONSerialization.jsonObject(with: data) as? NSDictionary)
    }

    private func encodeToObject<T: Encodable>(_ value: T) throws -> NSDictionary {
        let data = try JSONEncoder().encode(value)
        return try #require(try JSONSerialization.jsonObject(with: data) as? NSDictionary)
    }

    @Test func thereAreEnoughFixturesToCheck() throws {
        #expect(try jsonFiles(in: "messages").count >= 19)
        #expect(try jsonFiles(in: "invalid").count >= 18)
    }

    @Test func everyValidClientMessageDecodesAndReEncodesToTheSameObject() throws {
        let client = try fixtures(in: "messages").filter {
            $0.direction == "client_to_server" && !$0.unknown
        }
        #expect(client.count >= 4)
        for fixture in client {
            let decoded = try WireCodec.decodeClient(fixture.text)
            if case .unknown = decoded {
                Issue.record("\(fixture.name) decoded as unknown")
            }
            let encoded = try object(fromEncoded: WireCodec.encode(decoded))
            #expect(encoded.isEqual(fixture.message), "\(fixture.name) did not round-trip")
        }
    }

    @Test func everyValidServerMessageDecodesWithTheFixtureFields() throws {
        let server = try fixtures(in: "messages").filter {
            $0.direction == "server_to_client" && !$0.unknown
        }
        #expect(server.count >= 13)
        for fixture in server {
            let decoded = try WireCodec.decodeServer(fixture.text)
            let reEncoded: NSDictionary
            switch decoded {
            case .helloAck(let value): reEncoded = try encodeToObject(value)
            case .ping(let value): reEncoded = try encodeToObject(value)
            case .pong(let value): reEncoded = try encodeToObject(value)
            case .error(let value): reEncoded = try encodeToObject(value)
            case .wakeConfirmed(let value): reEncoded = try encodeToObject(value)
            case .turnState(let value): reEncoded = try encodeToObject(value)
            case .transcriptPartial(let value): reEncoded = try encodeToObject(value)
            case .transcriptFinal(let value): reEncoded = try encodeToObject(value)
            case .card(let value): reEncoded = try encodeToObject(value)
            case .turnEnded(let value): reEncoded = try encodeToObject(value)
            case .timerRinging(let value): reEncoded = try encodeToObject(value)
            case .timerStopped(let value): reEncoded = try encodeToObject(value)
            case .unknown:
                Issue.record("\(fixture.name) decoded as unknown")
                continue
            }
            #expect(reEncoded.isEqual(fixture.message), "\(fixture.name) fields differ")
        }
    }

    @Test func unknownFixturesDecodeToUnknownInTheirDirection() throws {
        let unknown = try fixtures(in: "messages").filter { $0.unknown }
        #expect(unknown.count >= 2)
        for fixture in unknown {
            if fixture.direction == "client_to_server" {
                let decoded = try WireCodec.decodeClient(fixture.text)
                #expect(decoded == .unknown(type: "example.future"), "\(fixture.name)")
            } else {
                let decoded = try WireCodec.decodeServer(fixture.text)
                #expect(decoded == .unknown(type: "example.future"), "\(fixture.name)")
            }
        }
    }

    @Test func everyInvalidFixtureFailsToDecode() throws {
        let invalid = try fixtures(in: "invalid")
        for fixture in invalid {
            if fixture.direction == "client_to_server" {
                #expect(throws: (any Error).self, "\(fixture.name) decoded") {
                    try WireCodec.decodeClient(fixture.text)
                }
            } else {
                #expect(throws: (any Error).self, "\(fixture.name) decoded") {
                    try WireCodec.decodeServer(fixture.text)
                }
            }
        }
    }

    @Test func invalidFixturesAlsoFailInTheOtherDirectionWhereTheTypeIsShared() throws {
        // A server frame with a bad id is dropped by the app, not trusted.
        #expect(throws: (any Error).self) {
            try WireCodec.decodeServer(#"{"type":"ping","id":-1}"#)
        }
        #expect(throws: (any Error).self) {
            try WireCodec.decodeServer(#"{"type":"pong","id":true}"#)
        }
        #expect(throws: (any Error).self) {
            try WireCodec.decodeServer(#"{"type":"hello.ack","protocol":true,"device_id":1,"ping_interval_s":15}"#)
        }
    }

    @Test func aFloatIsNotAnInteger() throws {
        #expect(throws: (any Error).self) {
            try WireCodec.decodeClient(#"{"type":"ping","id":1.5}"#)
        }
    }

    @Test func idRangeEdges() throws {
        #expect(try WireCodec.decodeClient(#"{"type":"ping","id":0}"#) == .ping(Ping(id: 0)))
        #expect(
            try WireCodec.decodeClient(#"{"type":"ping","id":2147483647}"#)
                == .ping(Ping(id: 2_147_483_647)))
        #expect(throws: (any Error).self) {
            try WireCodec.decodeClient(#"{"type":"ping","id":2147483648}"#)
        }
    }

    @Test func textThatIsNotAnObjectIsMalformed() {
        for text in ["", "not json", "[]", "3", #"{"type":3}"#] {
            #expect(throws: WireError.malformed, "\(text)") {
                try WireCodec.decodeServer(text)
            }
        }
    }

    @Test func capabilityAndStringCapsMatchPython() throws {
        func hello(app: String = "0.1.0", os: String = "26.0", caps: [String] = []) -> String {
            let object: [String: Any] = [
                "type": "hello", "protocol": 1, "app_version": app, "os_version": os,
                "capabilities": caps,
            ]
            let data = try! JSONSerialization.data(withJSONObject: object)
            return String(data: data, encoding: .utf8)!
        }
        _ = try WireCodec.decodeClient(hello(app: String(repeating: "a", count: 32)))
        _ = try WireCodec.decodeClient(hello(caps: Array(repeating: "x", count: 32)))
        _ = try WireCodec.decodeClient(hello(caps: [String(repeating: "c", count: 64)]))
        let bad = [
            hello(app: ""),
            hello(os: ""),
            hello(os: String(repeating: "o", count: 33)),
            hello(caps: Array(repeating: "x", count: 33)),
            hello(caps: [""]),
            hello(caps: [String(repeating: "c", count: 65)]),
        ]
        for text in bad {
            #expect(throws: (any Error).self, "\(text)") { try WireCodec.decodeClient(text) }
        }
    }

    @Test func closeCodesMatchTheFixture() throws {
        let table = try readObject(protocolRoot().appending(path: "close_codes.json"))
        #expect(table.count == CloseCode.allCases.count)
        for case let (key as String, value as NSDictionary) in table {
            let raw = try #require(Int(key))
            let code = try #require(CloseCode(rawValue: raw), "code \(key) is missing in Swift")
            #expect(code.name == value["name"] as? String, "name of \(key)")
            #expect(code.clientAction.rawValue == value["client_action"] as? String, "action of \(key)")
        }
        for code in CloseCode.allCases {
            #expect(table["\(code.rawValue)"] != nil, "\(code) is not in close_codes.json")
        }
    }

    @Test func constantsMatchTheFixture() throws {
        let table = try readObject(protocolRoot().appending(path: "constants.json"))
        let swift: [String: Int] = [
            "protocol": ProtocolConstants.protocolVersion,
            "ping_interval_s": ProtocolConstants.pingIntervalS,
            "pong_timeout_s": ProtocolConstants.pongTimeoutS,
            "hello_timeout_s": ProtocolConstants.helloTimeoutS,
            "idle_timeout_s": ProtocolConstants.idleTimeoutS,
            "max_text_frame_bytes": ProtocolConstants.maxTextFrameBytes,
            "max_invalid_messages": ProtocolConstants.maxInvalidMessages,
            "test_timeout_s": ProtocolConstants.testTimeoutS,
        ]
        #expect(Set(table.allKeys.compactMap { $0 as? String }) == Set(swift.keys))
        for (key, value) in swift {
            #expect((table[key] as? NSNumber)?.intValue == value, "constant \(key)")
        }
    }

    @Test func panelLimitsMatchTheFixture() throws {
        let table = try readObject(protocolRoot().appending(path: "panel_limits.json"))
        let swift: [String: Int] = [
            "id_max": PanelLimits.idMax,
            "word_max": PanelLimits.wordMax,
            "transcript_text_max": PanelLimits.transcriptTextMax,
            "card_text_max": PanelLimits.cardTextMax,
            "label_max": PanelLimits.labelMax,
            "ms_max": PanelLimits.msMax,
        ]
        #expect(Set(table.allKeys.compactMap { $0 as? String }) == Set(swift.keys))
        for (key, value) in swift {
            #expect((table[key] as? NSNumber)?.intValue == value, "limit \(key)")
        }
    }

    @Test func refusalMarkerMatchesTheFixture() throws {
        let table = try readObject(protocolRoot().appending(path: "refusal.json"))
        #expect(Set(table.allKeys.compactMap { $0 as? String }) == ["status", "header", "token_value"])
        #expect((table["status"] as? NSNumber)?.intValue == RefusalMarker.status)
        #expect(table["header"] as? String == RefusalMarker.header)
        #expect(table["token_value"] as? String == RefusalMarker.tokenValue)
    }

    @Test func helloCarriesExactlyFiveKeys() throws {
        let hello = Hello(
            protocolVersion: ProtocolConstants.protocolVersion,
            appVersion: "0.1.0", osVersion: "26.0", capabilities: [])
        let object = try object(fromEncoded: WireCodec.encode(.hello(hello)))
        #expect(
            Set(object.allKeys.compactMap { $0 as? String })
                == ["type", "protocol", "app_version", "os_version", "capabilities"])
        #expect(object["type"] as? String == "hello")
    }
}
