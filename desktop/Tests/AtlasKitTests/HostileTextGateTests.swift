import Foundation
import Testing

/// The source gate of the panel (UI-SPEC "Hostile text gate", "Rules for code").
///
/// Every remote string on the panel is plain text. This suite reads the panel
/// sources and fails on any API that could turn text into a link, markup, a web
/// view or a selection, and on any call that could let the panel take focus.
/// It reads files from `desktop/`, so a regression shows in `swift test` and
/// not only on a Mac with a full-screen app.
@Suite struct HostileTextGateTests {
    /// UI-SPEC "Hostile text gate" rule 2, plus the markup and web APIs the
    /// plan names.
    static let hostileTokens = [
        "Text(.init(", "LocalizedStringKey(", "LocalizedStringResource",
        "AttributedString(", "NSAttributedString", "Link(", ".onOpenURL",
        "AsyncImage", "WKWebView", "WebView(", "NSDataDetector",
        ".textSelection(",
    ]

    /// UI-SPEC "Rules for code": the panel never activates the app, never
    /// becomes key, never uses the main screen, and uses a tracking area for
    /// hover.
    static let focusTokens = [
        "NSApp.activate", "makeKeyAndOrderFront", "makeKey(", "NSScreen.main",
        // SwiftUI onHover, in call and trailing-closure form. The shell's own
        // `onHoverChanged` callback is not a match.
        ".onHover(", ".onHover {",
    ]

    private static let textWithoutVerbatim = #"\bText\((?!verbatim:)"#
    private static let buttonWithLiteral = #"\bButton\(\s*""#
    private static let labelWithLiteral = #"\.accessibilityLabel\(\s*""#

    struct SourceLine {
        let file: String
        let number: Int
        let code: String
    }

    private static func isPanelFile(_ name: String) -> Bool {
        ["Panel", "AtlasPanel", "TimerRing", "FirstClickHost"].contains { name.hasPrefix($0) }
    }

    private static func swiftFiles(in relative: String, only: (String) -> Bool = { _ in true }) throws -> [URL] {
        let dir = desktopRoot().appending(path: relative)
        return try FileManager.default
            .contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "swift" && only($0.lastPathComponent) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    static func panelFiles() throws -> [URL] {
        try swiftFiles(in: "Sources/AtlasDesktop", only: isPanelFile)
            + swiftFiles(in: "Sources/AtlasKit/Panel")
    }

    /// The code of one line, with the `//` comment cut off.
    static func stripComment(_ line: String) -> String {
        guard let range = line.range(of: "//") else { return line }
        return String(line[..<range.lowerBound])
    }

    static func codeLines(_ files: [URL]) throws -> [SourceLine] {
        var lines: [SourceLine] = []
        for url in files {
            let text = try String(contentsOf: url, encoding: .utf8)
            for (index, raw) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                lines.append(
                    SourceLine(file: url.lastPathComponent, number: index + 1, code: stripComment(String(raw))))
            }
        }
        return lines
    }

    static func matches(_ pattern: String, in code: String) -> Bool {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return false }
        return regex.firstMatch(in: code, range: NSRange(code.startIndex..., in: code)) != nil
    }

    private func violations(tokens: [String], in lines: [SourceLine]) -> [String] {
        lines.flatMap { line in
            tokens.filter { line.code.contains($0) }.map { "\(line.file):\(line.number) uses \($0)" }
        }
    }

    private func violations(pattern: String, name: String, in lines: [SourceLine]) -> [String] {
        lines.filter { Self.matches(pattern, in: $0.code) }.map { "\($0.file):\($0.number) has \(name)" }
    }

    @Test func gateReadsEnoughPanelFiles() throws {
        let names = try Self.panelFiles().map(\.lastPathComponent)
        #expect(names.count >= 8, "found only \(names)")
        #expect(names.contains("PanelView.swift"))
        #expect(names.contains("PanelController.swift"))
        #expect(names.contains("TimerRingView.swift"), "the ring view must be under the gate")
    }

    @Test func noMarkupLinkWebOrSelectionApi() throws {
        let lines = try Self.codeLines(Self.panelFiles())
        #expect(violations(tokens: Self.hostileTokens, in: lines).isEmpty, "\(violations(tokens: Self.hostileTokens, in: lines))")
    }

    @Test func noFocusActivationOrMainScreenCall() throws {
        let lines = try Self.codeLines(Self.panelFiles())
        let found = violations(tokens: Self.focusTokens, in: lines)
        #expect(found.isEmpty, "\(found)")
    }

    @Test func everyTextIsVerbatim() throws {
        let lines = try Self.codeLines(Self.panelFiles())
        let found = violations(pattern: Self.textWithoutVerbatim, name: "Text( without verbatim:", in: lines)
        #expect(found.isEmpty, "\(found)")
    }

    @Test func noButtonOrAccessibilityLabelFromALiteral() throws {
        let lines = try Self.codeLines(Self.panelFiles())
        let buttons = violations(pattern: Self.buttonWithLiteral, name: "a Button( string literal", in: lines)
        let labels = violations(pattern: Self.labelWithLiteral, name: "an accessibilityLabel string literal", in: lines)
        #expect(buttons.isEmpty, "\(buttons)")
        #expect(labels.isEmpty, "\(labels)")
    }

    /// The gate must be able to fail, or an empty pattern would pass for ever.
    @Test func theRulesCatchBadSource() {
        #expect(Self.matches(Self.textWithoutVerbatim, in: "Text(message.text)"))
        #expect(Self.matches(Self.textWithoutVerbatim, in: "Text(\"Go ahead\")"))
        #expect(!Self.matches(Self.textWithoutVerbatim, in: "Text(verbatim: text)"))
        #expect(!Self.matches(Self.textWithoutVerbatim, in: "TextField(verbatim: text)"))
        #expect(Self.matches(Self.buttonWithLiteral, in: "Button(\"Stop\") {"))
        #expect(!Self.matches(Self.buttonWithLiteral, in: "Button(action: close) {"))
        #expect(Self.matches(Self.labelWithLiteral, in: ".accessibilityLabel(\"Close\")"))
        #expect(!Self.matches(Self.labelWithLiteral, in: ".accessibilityLabel(PanelCopy.closeLabel)"))
        #expect(Self.stripComment("let a = 1 // NSApp.activate()") == "let a = 1 ")
        let sample = [SourceLine(file: "X.swift", number: 1, code: "window.makeKeyAndOrderFront(nil)")]
        #expect(!violations(tokens: Self.focusTokens, in: sample).isEmpty)
        let hover = [SourceLine(file: "X.swift", number: 1, code: "view.onHover { inside in }")]
        #expect(!violations(tokens: Self.focusTokens, in: hover).isEmpty)
        let hostHover = [SourceLine(file: "X.swift", number: 1, code: "host.onHoverChanged = { _ in }")]
        #expect(violations(tokens: Self.focusTokens, in: hostHover).isEmpty)
    }
}
