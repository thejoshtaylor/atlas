import Foundation
import Testing

@testable import AtlasKit

/// Every row of the UI-SPEC "Panel Window" configuration table (D-03, D-04).
/// A regression here would only show over a full-screen app, so the table is
/// pinned in one value and the AtlasPanel source is checked for the overrides
/// the value cannot express.
@Suite struct PanelConfigTests {
    private let spec = PanelWindowSpec.panel

    private func panelSource() throws -> String {
        let url = desktopRoot().appending(path: "Sources/AtlasDesktop/AtlasPanel.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    @Test func styleIsBorderlessAndNonactivating() {
        #expect(spec.style == [.borderless, .nonactivatingPanel])
    }

    @Test func collectionBehaviorKeepsThePanelOnEverySpaceAndOverFullScreen() {
        #expect(
            spec.collectionBehavior
                == [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle])
    }

    @Test func levelIsStatusBar() {
        #expect(spec.level == .statusBar)
    }

    @Test func thePanelNeverTakesKeyboardFocus() {
        #expect(!spec.canBecomeKey)
        #expect(!spec.canBecomeMain)
        #expect(spec.becomesKeyOnlyIfNeeded)
        #expect(!spec.hidesOnDeactivate)
    }

    @Test func thePanelDoesNotMoveButTakesClicks() {
        #expect(!spec.isMovable)
        #expect(!spec.isMovableByWindowBackground)
        #expect(!spec.ignoresMouseEvents)
    }

    @Test func theSurfaceIsTransparentWithASystemShadow() {
        #expect(spec.isFloatingPanel)
        #expect(!spec.isOpaque)
        #expect(spec.hasShadow)
    }

    @Test func nothingInTheSpecIsTheReadbackSentinel() {
        #expect(spec.level != .unmapped)
        #expect(!spec.collectionBehavior.contains(.unmapped))
        #expect(!spec.style.contains(.unmapped))
    }

    @Test func atlasPanelOverridesCanBecomeKeyAndCanBecomeMainToFalse() throws {
        let source = try panelSource()
        #expect(source.contains("override var canBecomeKey: Bool { false }"))
        #expect(source.contains("override var canBecomeMain: Bool { false }"))
    }

    @Test func atlasPanelSetsTheNonactivatingStyleInItsInitAndReadsTheSpec() throws {
        let source = try panelSource()
        #expect(source.contains(".nonactivatingPanel"))
        #expect(source.contains("PanelWindowSpec"))
    }
}
