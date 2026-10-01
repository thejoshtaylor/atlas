import Foundation

/// The one sanitizer for every remote string the panel shows: transcript text,
/// card text and a timer label (Phase 15, CARD-08, PANEL-03). The Mac does not
/// trust the server's sanitizing.
///
/// It is the twin of `sanitize_display_text` in
/// `src/atlas/desktop/display_text.py`, rule for rule, and both read the same
/// vectors from `desktop/protocol/v1/hostile_text.json`. It works over unicode
/// scalars, because Python counts code points.
///
/// - U+0009, U+000A, U+000B, U+000C, U+000D, U+0085, U+2028 and U+2029 become
///   a space.
/// - Every other scalar in U+0000...U+001F and U+007F...U+009F is removed, and
///   so are U+202A...U+202E, U+2066...U+2069, U+200E, U+200F and U+061C.
/// - A run of U+0020 becomes one, and both ends lose U+0020.
/// - Everything else stays, U+00A0 and Markdown characters included.
public enum TextSanitizer {
    public enum Keep: Sendable { case start, end }

    private static let ellipsis: Unicode.Scalar = "\u{2026}"
    private static let space: Unicode.Scalar = " "

    /// `text` as plain text, at most `maxScalars` unicode scalars. Over the
    /// cap, `.start` keeps the first `maxScalars - 1` and ends with U+2026, and
    /// `.end` starts with U+2026 and keeps the last `maxScalars - 1`. Spaces at
    /// the cut are removed.
    public static func sanitize(_ text: String, maxScalars: Int, keep: Keep = .start) -> String {
        precondition(maxScalars >= 2, "maxScalars must leave room for the cut mark")
        var scalars: [Unicode.Scalar] = []
        scalars.reserveCapacity(min(text.unicodeScalars.count, maxScalars + 1))
        var lastWasSpace = true  // a leading space is dropped
        for scalar in text.unicodeScalars {
            let mapped: Unicode.Scalar
            if isSpaceLike(scalar.value) {
                mapped = space
            } else if isRemoved(scalar.value) {
                continue
            } else {
                mapped = scalar
            }
            if mapped == space {
                if lastWasSpace { continue }
                lastWasSpace = true
            } else {
                lastWasSpace = false
            }
            scalars.append(mapped)
        }
        while scalars.last == space { scalars.removeLast() }

        if scalars.count > maxScalars {
            switch keep {
            case .start:
                scalars = Array(scalars.prefix(maxScalars - 1))
                while scalars.last == space { scalars.removeLast() }
                scalars.append(ellipsis)
            case .end:
                var tail = Array(scalars.suffix(maxScalars - 1))
                while tail.first == space { tail.removeFirst() }
                scalars = [ellipsis] + tail
            }
        }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }

    private static func isSpaceLike(_ value: UInt32) -> Bool {
        switch value {
        case 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x85, 0x2028, 0x2029: return true
        default: return false
        }
    }

    private static func isRemoved(_ value: UInt32) -> Bool {
        switch value {
        case 0x00...0x1F, 0x7F...0x9F, 0x202A...0x202E, 0x2066...0x2069, 0x200E, 0x200F, 0x061C:
            return true
        default: return false
        }
    }
}
