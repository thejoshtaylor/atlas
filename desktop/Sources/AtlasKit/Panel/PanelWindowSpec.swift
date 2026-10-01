import Foundation

/// The window level the panel sits at. The names match AppKit's
/// `NSWindow.Level` constants without importing AppKit (this module is pure).
public enum PanelLevel: String, Sendable, CaseIterable {
    case floating
    case statusBar
    /// Readback only: a live level that is neither of the two above. It never
    /// appears in a spec the panel is built from, so it can never compare equal.
    case unmapped
}

/// The collection behavior flags the panel needs (D-03).
public enum PanelCollectionBehavior: String, Sendable, CaseIterable, Hashable {
    case canJoinAllSpaces
    case fullScreenAuxiliary
    case stationary
    case ignoresCycle
    /// Readback only: a live flag outside the four above.
    case unmapped
}

/// The window style, as far as this panel uses it.
public enum PanelStyle: String, Sendable, Hashable {
    case borderless
    case nonactivatingPanel
    /// Readback only: a live style bit outside the two above.
    case unmapped
}

/// Every configurable property of the panel window, in one value (D-03, D-04).
///
/// The values are the rows of the UI-SPEC "Panel Window" configuration table.
/// `AtlasPanel` applies them and `liveSpec()` reads them back, so a person never
/// has to find a regression by looking at a full-screen app. The level is the
/// one value the macOS 26 spike may change: see 15-SPIKE.md.
public struct PanelWindowSpec: Equatable, Sendable {
    public let level: PanelLevel
    public let collectionBehavior: Set<PanelCollectionBehavior>
    public let style: Set<PanelStyle>
    public let isFloatingPanel: Bool
    public let canBecomeKey: Bool
    public let canBecomeMain: Bool
    public let becomesKeyOnlyIfNeeded: Bool
    public let hidesOnDeactivate: Bool
    public let isMovable: Bool
    public let isMovableByWindowBackground: Bool
    public let ignoresMouseEvents: Bool
    public let isOpaque: Bool
    public let hasShadow: Bool

    public init(
        level: PanelLevel,
        collectionBehavior: Set<PanelCollectionBehavior>,
        style: Set<PanelStyle>,
        isFloatingPanel: Bool,
        canBecomeKey: Bool,
        canBecomeMain: Bool,
        becomesKeyOnlyIfNeeded: Bool,
        hidesOnDeactivate: Bool,
        isMovable: Bool,
        isMovableByWindowBackground: Bool,
        ignoresMouseEvents: Bool,
        isOpaque: Bool,
        hasShadow: Bool
    ) {
        self.level = level
        self.collectionBehavior = collectionBehavior
        self.style = style
        self.isFloatingPanel = isFloatingPanel
        self.canBecomeKey = canBecomeKey
        self.canBecomeMain = canBecomeMain
        self.becomesKeyOnlyIfNeeded = becomesKeyOnlyIfNeeded
        self.hidesOnDeactivate = hidesOnDeactivate
        self.isMovable = isMovable
        self.isMovableByWindowBackground = isMovableByWindowBackground
        self.ignoresMouseEvents = ignoresMouseEvents
        self.isOpaque = isOpaque
        self.hasShadow = hasShadow
    }

    /// The same spec at another level. The spike harness uses it for `--level`.
    public func withLevel(_ level: PanelLevel) -> PanelWindowSpec {
        PanelWindowSpec(
            level: level, collectionBehavior: collectionBehavior, style: style,
            isFloatingPanel: isFloatingPanel, canBecomeKey: canBecomeKey,
            canBecomeMain: canBecomeMain, becomesKeyOnlyIfNeeded: becomesKeyOnlyIfNeeded,
            hidesOnDeactivate: hidesOnDeactivate, isMovable: isMovable,
            isMovableByWindowBackground: isMovableByWindowBackground,
            ignoresMouseEvents: ignoresMouseEvents, isOpaque: isOpaque, hasShadow: hasShadow)
    }

    /// The panel: it never takes keyboard focus (D-03), it cannot be dragged and
    /// its buttons take clicks (D-04).
    public static let panel = PanelWindowSpec(
        level: .statusBar,
        collectionBehavior: [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle],
        style: [.borderless, .nonactivatingPanel],
        isFloatingPanel: true,
        canBecomeKey: false,
        canBecomeMain: false,
        becomesKeyOnlyIfNeeded: true,
        hidesOnDeactivate: false,
        isMovable: false,
        isMovableByWindowBackground: false,
        ignoresMouseEvents: false,
        isOpaque: false,
        hasShadow: true)
}
