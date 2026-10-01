import Foundation

/// Reads the server part of a pair link or of manual entry.
///
/// The result is a normalized `host[:port]`: a DNS name, an IPv4 address or a
/// bracketed IPv6 address, with an optional port 1...65535. Nothing else gets
/// through, so the text can never smuggle in a path, a query or userinfo.
public enum ServerAddress {
    /// Schemes that would send the token in the clear (D-08). They are named
    /// here only to be refused.
    private static let insecurePrefixes = ["ws://", "http://"]
    private static let securePrefixes = ["wss://", "https://"]

    public static func parse(_ raw: String) throws(PairingLinkError) -> String {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = text.lowercased()
        if insecurePrefixes.contains(where: { lower.hasPrefix($0) }) { throw .insecure }
        if let prefix = securePrefixes.first(where: { lower.hasPrefix($0) }) {
            text.removeFirst(prefix.count)
        }
        if text.hasSuffix("/") { text.removeLast() }
        guard !text.isEmpty else { throw .invalidServer }
        // ASCII only: an internationalized name must arrive as punycode.
        for scalar in text.unicodeScalars {
            guard scalar.isASCII, scalar.value > 0x20, scalar.value != 0x7F,
                !"@/?#\\".unicodeScalars.contains(scalar)
            else { throw .invalidServer }
        }

        let hostPart: String
        let portPart: String?
        if text.hasPrefix("[") {
            guard let close = text.firstIndex(of: "]") else { throw .invalidServer }
            let inner = String(text[text.index(after: text.startIndex)..<close])
            guard isIPv6(inner) else { throw .invalidServer }
            hostPart = "[" + inner.lowercased() + "]"
            let rest = text[text.index(after: close)...]
            if rest.isEmpty {
                portPart = nil
            } else if rest.hasPrefix(":") {
                portPart = String(rest.dropFirst())
            } else {
                throw .invalidServer
            }
        } else {
            let pieces = text.split(separator: ":", maxSplits: 2, omittingEmptySubsequences: false)
            guard pieces.count <= 2 else { throw .invalidServer }
            let name = String(pieces[0]).lowercased()
            guard isDNSNameOrIPv4(name) else { throw .invalidServer }
            hostPart = name
            portPart = pieces.count == 2 ? String(pieces[1]) : nil
        }

        guard let portPart else { return hostPart }
        guard !portPart.isEmpty, portPart.utf8.allSatisfy({ $0 >= 0x30 && $0 <= 0x39 }),
            portPart.count <= 5, let port = Int(portPart), (1...65535).contains(port)
        else { throw .invalidServer }
        return hostPart + ":" + String(port)
    }

    private static func isIPv6(_ text: String) -> Bool {
        guard text.contains(":"), text.allSatisfy({ $0.isHexDigit || $0 == ":" || $0 == "." })
        else { return false }
        var address = in6_addr()
        return inet_pton(AF_INET6, text, &address) == 1
    }

    private static func isDNSNameOrIPv4(_ name: String) -> Bool {
        guard !name.isEmpty, name.count <= 253 else { return false }
        let labels = name.split(separator: ".", omittingEmptySubsequences: false)
        for label in labels {
            guard (1...63).contains(label.count), label.first != "-", label.last != "-",
                label.utf8.allSatisfy({ isLabelByte($0) })
            else { return false }
        }
        // A name made only of digits and dots has to be a real IPv4 address.
        if name.utf8.allSatisfy({ ($0 >= 0x30 && $0 <= 0x39) || $0 == 0x2E }) {
            var address = in_addr()
            return labels.count == 4 && inet_pton(AF_INET, name, &address) == 1
        }
        return true
    }

    private static func isLabelByte(_ byte: UInt8) -> Bool {
        (byte >= 0x30 && byte <= 0x39) || (byte >= 0x61 && byte <= 0x7A)
            || (byte >= 0x41 && byte <= 0x5A) || byte == 0x2D
    }
}
