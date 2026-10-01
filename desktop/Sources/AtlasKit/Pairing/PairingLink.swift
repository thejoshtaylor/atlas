import Foundation

public enum PairingLinkError: Error, Equatable, Sendable {
    /// Not an `atlas://pair` link, or the link has no server.
    case notAtlasLink
    /// The server was given as `ws://` or `http://` (D-08).
    case insecure
    /// The server is not a plain `host[:port]`.
    case invalidServer
    /// The token is missing, too long, or holds whitespace.
    case missingToken
}

/// A server and a token that passed validation but are not saved yet. The
/// shell shows the host and asks before it writes the Keychain (D-03).
///
/// Every text form shows the host only (T-14-31).
public struct PairingTarget: Sendable, Equatable, CustomStringConvertible,
    CustomDebugStringConvertible, CustomReflectable
{
    public let host: String
    public let token: String

    /// Only `PairingLink.parse` and `manual(server:token:)` build a target, so
    /// every target has been through the same checks.
    fileprivate init(host: String, token: String) {
        self.host = host
        self.token = token
    }

    public var credentials: PairingCredentials { PairingCredentials(host: host, token: token) }

    /// Manual entry: the same checks as a link. Whitespace around either value
    /// is trimmed, because a pasted token often carries a newline.
    public static func manual(server: String, token: String) throws(PairingLinkError) -> PairingTarget {
        let host = try ServerAddress.parse(server)
        let cleaned = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard PairingLink.isAcceptableToken(cleaned) else { throw .missingToken }
        return PairingTarget(host: host, token: cleaned)
    }

    public var description: String { "PairingTarget(host: \(host))" }
    public var debugDescription: String { description }
    public var customMirror: Mirror { Mirror(self, children: ["host": host]) }
}

public enum PairingLink {
    static let maxTokenLength = 512

    /// Reads `atlas://pair?server=<host[:port]>&token=<token>` (D-03).
    public static func parse(_ url: URL) throws(PairingLinkError) -> PairingTarget {
        guard let parts = URLComponents(url: url, resolvingAgainstBaseURL: false),
            parts.scheme?.lowercased() == AppIdentity.urlScheme,
            parts.host?.lowercased() == "pair"
        else { throw .notAtlasLink }
        let items = parts.queryItems ?? []
        guard let server = items.first(where: { $0.name == "server" })?.value else {
            throw .notAtlasLink
        }
        let host = try ServerAddress.parse(server)
        guard let token = items.first(where: { $0.name == "token" })?.value,
            isAcceptableToken(token)
        else { throw .missingToken }
        return PairingTarget(host: host, token: token)
    }

    static func isAcceptableToken(_ token: String) -> Bool {
        guard !token.isEmpty, token.count <= maxTokenLength else { return false }
        return token.unicodeScalars.allSatisfy { scalar in
            scalar.value > 0x20 && scalar.value != 0x7F && !scalar.properties.isWhitespace
                && !(0x80...0x9F).contains(scalar.value)
        }
    }
}

/// The only place the app builds a server URL, and it always builds
/// `wss://<host>/ws/desktop` (D-08).
///
/// App Transport Security does not stop a plain `ws://` connection made through
/// `URLSessionWebSocketTask`, so this code does (RESEARCH Pitfall 3). There is
/// no setting, flag or branch that builds a different scheme.
public enum ConnectionEndpoint {
    public static let path = "/ws/desktop"

    /// `host` comes from `ServerAddress.parse`. A host that did not pass it
    /// gives a URL that cannot connect, never a URL with another scheme.
    public static func url(for host: String) -> URL {
        let address = (try? ServerAddress.parse(host)) ?? "invalid.invalid"
        return build(address) ?? build("invalid.invalid")!
    }

    private static func build(_ address: String) -> URL? {
        var name = address
        var port: Int?
        if address.hasPrefix("[") {
            if let close = address.firstIndex(of: "]") {
                name = String(address[...close])
                let rest = address[address.index(after: close)...]
                if rest.hasPrefix(":") { port = Int(rest.dropFirst()) }
            }
        } else if let colon = address.firstIndex(of: ":") {
            name = String(address[..<colon])
            port = Int(address[address.index(after: colon)...])
        }
        var parts = URLComponents()
        parts.scheme = "wss"
        parts.host = name
        parts.port = port
        parts.path = path
        return parts.url
    }
}
