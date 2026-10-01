/// Why one connection attempt or one live connection ended.
public enum DialFailure: Error, Equatable, Sendable {
    /// HTTP 403 on the handshake: the server does not know this token, or the
    /// Mac was revoked while it was offline (D-27).
    case refused
    /// The server closed a live socket with this code (D-21).
    case closed(code: Int)
    /// Everything else: no network, DNS, TLS, a timeout, a dropped socket.
    case unreachable

    /// `closeCode` 0 means no close frame arrived (`URLSessionWebSocketTask.CloseCode.invalid`).
    public static func classify(httpStatus: Int?, closeCode: Int) -> DialFailure {
        if httpStatus == 403 { return .refused }
        if closeCode != 0 { return .closed(code: closeCode) }
        return .unreachable
    }
}
