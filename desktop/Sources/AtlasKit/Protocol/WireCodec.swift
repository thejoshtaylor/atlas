import Foundation

public enum WireError: Error, Equatable, Sendable {
    /// The text is not a JSON object, or it has no string `type`.
    case malformed
}

/// Turns frames into text and text into frames. It never logs a frame.
public enum WireCodec {
    /// One frame as compact JSON text. `Hello` writes only its five keys (D-11).
    public static func encode(_ message: ClientMessage) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(message)
        guard let text = String(data: data, encoding: .utf8) else { throw WireError.malformed }
        return text
    }

    public static func decodeServer(_ text: String) throws -> ServerMessage {
        try decode(ServerMessage.self, from: text)
    }

    /// For the contract test: the server reads these frames in Python.
    public static func decodeClient(_ text: String) throws -> ClientMessage {
        try decode(ClientMessage.self, from: text)
    }

    private static func decode<T: Decodable>(_ type: T.Type, from text: String) throws -> T {
        let data = Data(text.utf8)
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            object["type"] is String
        else { throw WireError.malformed }
        return try JSONDecoder().decode(type, from: data)
    }
}
