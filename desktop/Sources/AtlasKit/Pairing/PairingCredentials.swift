/// One pairing: the server host and the bearer token for this Mac.
///
/// Every text form shows the host only, so the token never reaches a log line,
/// a debugger print or a crash report by way of string interpolation.
public struct PairingCredentials: Sendable, Equatable, CustomStringConvertible,
    CustomDebugStringConvertible, CustomReflectable
{
    public let host: String
    public let token: String

    public init(host: String, token: String) {
        self.host = host
        self.token = token
    }

    public var description: String { "PairingCredentials(host: \(host))" }
    public var debugDescription: String { description }
    public var customMirror: Mirror { Mirror(self, children: ["host": host]) }
}
