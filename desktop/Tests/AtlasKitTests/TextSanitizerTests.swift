import Foundation
import Testing

@testable import AtlasKit

/// `TextSanitizer.sanitize` against the vectors that
/// `tests/test_desktop_cards.py` reads for `sanitize_display_text`. Both sides
/// must give the same output (PANEL-03, CARD-08).
@Suite struct TextSanitizerTests {
    private struct Vector {
        let name: String
        let input: String
        let max: Int
        let keep: TextSanitizer.Keep
        let expected: String
    }

    private func vectors() throws -> [Vector] {
        let url = desktopRoot().appending(path: "protocol/v1/hostile_text.json")
        let root = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let list = try #require(root["vectors"] as? [[String: Any]])
        return try list.map { entry in
            Vector(
                name: try #require(entry["name"] as? String),
                input: try #require(entry["input"] as? String),
                max: try #require(entry["max"] as? Int),
                keep: try #require(entry["keep"] as? String) == "end" ? .end : .start,
                expected: try #require(entry["expected"] as? String))
        }
    }

    @Test func thereAreEnoughSharedVectors() throws {
        #expect(try vectors().count >= 12)
    }

    @Test func everySharedVectorGivesTheExpectedOutput() throws {
        for vector in try vectors() {
            let result = TextSanitizer.sanitize(
                vector.input, maxScalars: vector.max, keep: vector.keep)
            #expect(
                Array(result.unicodeScalars) == Array(vector.expected.unicodeScalars),
                "vector \(vector.name)")
            #expect(result.unicodeScalars.count <= vector.max, "vector \(vector.name) is too long")
        }
    }

    @Test func markdownLinkSyntaxStaysLiteral() {
        let text = "[text](https://example.com)"
        #expect(TextSanitizer.sanitize(text, maxScalars: 600) == text)
    }

    @Test func aStringOfOnlySpacesIsEmpty() {
        #expect(TextSanitizer.sanitize("      ", maxScalars: 600) == "")
    }
}
