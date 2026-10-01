import Foundation

/// Opens a WebSocket to the server. The real one wraps `URLSessionWebSocketTask`;
/// tests use a fake that returns scripted channels.
public protocol WebSocketTransport: Sendable {
    /// The token travels only in the `Authorization: Bearer` header (D-02).
    /// A refusal, a failed handshake or an unreachable host throws a `DialFailure`.
    func open(url: URL, bearerToken: String) async throws -> any WebSocketChannel
}

/// One open socket.
public protocol WebSocketChannel: Sendable {
    func send(text: String) async throws
    /// The next text frame. Throws a `DialFailure` when the socket ends or the
    /// handshake failed.
    func receive() async throws -> String
    func close(code: Int) async
}
