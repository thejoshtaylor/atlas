import Foundation
import Testing

@testable import AtlasKit

@Suite struct PairingLinkTests {
    private func link(
        scheme: String = "atlas", host: String = "pair", server: String? = "svr.test",
        token: String? = "t-1"
    ) throws -> URL {
        var parts = URLComponents()
        parts.scheme = scheme
        parts.host = host
        var items: [URLQueryItem] = []
        if let server { items.append(URLQueryItem(name: "server", value: server)) }
        if let token { items.append(URLQueryItem(name: "token", value: token)) }
        parts.queryItems = items.isEmpty ? nil : items
        return try #require(parts.url)
    }

    private func parseError(_ url: URL) -> PairingLinkError? {
        do {
            _ = try PairingLink.parse(url)
            return nil
        } catch {
            return error
        }
    }

    private func serverError(_ raw: String) -> PairingLinkError? {
        do {
            _ = try ServerAddress.parse(raw)
            return nil
        } catch {
            return error
        }
    }

    // MARK: the link

    @Test func aLinkGivesHostAndToken() throws {
        let target = try PairingLink.parse(try link())
        #expect(target.host == "svr.test")
        #expect(target.token == "t-1")
        #expect(target.credentials == PairingCredentials(host: "svr.test", token: "t-1"))
    }

    @Test func aPortSurvivesPercentEncoding() throws {
        #expect(try PairingLink.parse(try link(server: "svr.test:8443")).host == "svr.test:8443")
        var parts = URLComponents()
        parts.scheme = "atlas"
        parts.host = "pair"
        parts.percentEncodedQuery = "server=svr.test%3A8443&token=t-1"
        #expect(try PairingLink.parse(try #require(parts.url)).host == "svr.test:8443")
    }

    @Test func insecureSchemesInTheServerThrowInsecure() throws {
        for server in ["ws://svr.test", "http://svr.test", "WS://svr.test", "HTTP://svr.test"] {
            #expect(parseError(try link(server: server)) == .insecure, "\(server)")
        }
    }

    @Test func secureSchemesAreStrippedAndOneTrailingSlashIsAccepted() throws {
        #expect(try PairingLink.parse(try link(server: "wss://svr.test")).host == "svr.test")
        #expect(try PairingLink.parse(try link(server: "https://svr.test")).host == "svr.test")
        #expect(try PairingLink.parse(try link(server: "svr.test/")).host == "svr.test")
        #expect(parseError(try link(server: "svr.test//")) == .invalidServer)
    }

    @Test func badServersThrowInvalidServer() throws {
        let bad = [
            "svr.test/x", "u@svr.test", "svr.test?a=b", "svr.test#f", "svr .test", " ",
            "svr.test\u{0007}", "", "svr.test:0", "svr.test:65536", "svr.test:", "svr.test:80a",
            "a:b:c", "[fd00::1", "[nope]:80", "-bad.test", "bad-.test", "bad..test", ".test",
            "svr.test.", "999.1.1.1", "sv\u{00e9}.test", "ftp://svr.test", "svr\\test",
        ]
        for server in bad {
            #expect(serverError(server) == .invalidServer, "\(server.debugDescription)")
        }
    }

    @Test func aLongHostIsRefused() {
        let label = String(repeating: "a", count: 63)
        #expect(serverError(label + ".test") == nil)
        #expect(serverError(label + "a.test") == .invalidServer)
        let long = Array(repeating: label, count: 4).joined(separator: ".")
        #expect(serverError(long) == .invalidServer)
    }

    @Test func goodServersNormalize() throws {
        #expect(try ServerAddress.parse("SVR.Test") == "svr.test")
        #expect(try ServerAddress.parse("10.0.0.5:8443") == "10.0.0.5:8443")
        #expect(try ServerAddress.parse("localhost") == "localhost")
        #expect(try ServerAddress.parse("[fd00::1]:8443") == "[fd00::1]:8443")
        #expect(try ServerAddress.parse("[fd00::1]") == "[fd00::1]")
        #expect(try ServerAddress.parse("svr.test:65535") == "svr.test:65535")
    }

    @Test func wrongSchemeOrHostIsNotAnAtlasLink() throws {
        #expect(parseError(try link(scheme: "https")) == .notAtlasLink)
        #expect(parseError(try link(host: "other")) == .notAtlasLink)
        #expect(parseError(try link(server: nil)) == .notAtlasLink)
        #expect(try PairingLink.parse(try link(scheme: "ATLAS", host: "PAIR")).host == "svr.test")
    }

    @Test func badTokensThrowMissingToken() throws {
        #expect(parseError(try link(token: nil)) == .missingToken)
        #expect(parseError(try link(token: "")) == .missingToken)
        #expect(parseError(try link(token: "a b")) == .missingToken)
        #expect(parseError(try link(token: "a\tb")) == .missingToken)
        #expect(parseError(try link(token: "a" + String(UnicodeScalar(1)) + "b")) == .missingToken)
        #expect(parseError(try link(token: String(repeating: "x", count: 513))) == .missingToken)
        #expect(try PairingLink.parse(try link(token: String(repeating: "x", count: 512))).token.count == 512)
    }

    // MARK: manual entry

    @Test func manualEntryGivesTheSameTargetAsTheLink() throws {
        let manual = try PairingTarget.manual(server: " svr.test ", token: "t-1")
        #expect(manual == (try PairingLink.parse(try link())))
        #expect(manual.host == "svr.test")
    }

    @Test func manualEntryRefusesInsecureAndBadInput() {
        #expect(throws: PairingLinkError.insecure) {
            try PairingTarget.manual(server: "ws://svr.test", token: "t-1")
        }
        #expect(throws: PairingLinkError.invalidServer) {
            try PairingTarget.manual(server: "svr.test/x", token: "t-1")
        }
        #expect(throws: PairingLinkError.missingToken) {
            try PairingTarget.manual(server: "svr.test", token: "  ")
        }
        #expect(throws: PairingLinkError.missingToken) {
            try PairingTarget.manual(server: "svr.test", token: "a b")
        }
    }

    // MARK: the one URL builder

    @Test func theConnectionUrlIsAlwaysWssToTheDesktopPath() {
        #expect(ConnectionEndpoint.url(for: "svr.test").absoluteString == "wss://svr.test/ws/desktop")
        #expect(
            ConnectionEndpoint.url(for: "svr.test:8443").absoluteString
                == "wss://svr.test:8443/ws/desktop")
        let v6 = ConnectionEndpoint.url(for: "[fd00::1]:8443")
        #expect(v6.scheme == "wss")
        #expect(v6.port == 8443)
        #expect(v6.path == "/ws/desktop")
        #expect(v6.absoluteString.contains("fd00::1"))
    }

    @Test func aHostThatWasNeverValidatedStillGivesAWssUrl() {
        for host in ["ws://svr.test", "http://x", "svr.test/x", "", "a b"] {
            let url = ConnectionEndpoint.url(for: host)
            #expect(url.scheme == "wss", "\(host.debugDescription)")
            #expect(url.path == "/ws/desktop", "\(host.debugDescription)")
        }
    }

    // MARK: the token stays out of text

    @Test func textFormsNameTheHostAndNotTheToken() throws {
        let target = try PairingTarget.manual(server: "svr.test", token: "t-1")
        for text in [String(describing: target), String(reflecting: target), "\(target)"] {
            #expect(text.contains("svr.test"))
            #expect(!text.contains("t-1"))
        }
        var dumped = ""
        dump(target, to: &dumped)
        #expect(dumped.contains("svr.test"))
        #expect(!dumped.contains("t-1"))
    }
}
