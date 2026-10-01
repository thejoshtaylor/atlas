import AppKit
import AtlasKit

/// The ATLAS panel window: an `NSPanel` that never takes keyboard focus (D-03)
/// and shows over full-screen apps and on every Space.
///
/// Show rule (UI-SPEC "Rules for code"): show it with `orderFrontRegardless()`
/// only. Never activate the app, never call `makeKey` or `makeKeyAndOrderFront`.
/// Build it once, then order it out and in; do not rebuild it per message.
@MainActor
final class AtlasPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    /// The style mask goes in at init: setting `.nonactivatingPanel` later does
    /// not reliably take effect (RESEARCH A1).
    convenience init(
        spec: PanelWindowSpec = .panel,
        contentRect: NSRect = NSRect(x: 0, y: 0, width: 360, height: 120)
    ) {
        self.init(
            contentRect: contentRect,
            styleMask: Self.styleMask(for: spec.style),
            backing: .buffered,
            defer: false)
        isFloatingPanel = spec.isFloatingPanel
        level = Self.windowLevel(for: spec.level)
        collectionBehavior = Self.behavior(for: spec.collectionBehavior)
        becomesKeyOnlyIfNeeded = spec.becomesKeyOnlyIfNeeded
        hidesOnDeactivate = spec.hidesOnDeactivate
        isMovable = spec.isMovable
        isMovableByWindowBackground = spec.isMovableByWindowBackground
        ignoresMouseEvents = spec.ignoresMouseEvents
        isOpaque = spec.isOpaque
        backgroundColor = .clear
        hasShadow = spec.hasShadow
        animationBehavior = .none
        isReleasedWhenClosed = false
        // sharingType is left alone: the panel makes no claim to hide from capture.
    }

    /// The live AppKit values, read back into a spec. A level, a flag or a style
    /// bit outside the known set reads as `.unmapped`, so it never equals a spec.
    func liveSpec() -> PanelWindowSpec {
        PanelWindowSpec(
            level: Self.panelLevel(for: level),
            collectionBehavior: Self.panelBehavior(for: collectionBehavior),
            style: Self.panelStyle(for: styleMask),
            isFloatingPanel: isFloatingPanel,
            canBecomeKey: canBecomeKey,
            canBecomeMain: canBecomeMain,
            becomesKeyOnlyIfNeeded: becomesKeyOnlyIfNeeded,
            hidesOnDeactivate: hidesOnDeactivate,
            isMovable: isMovable,
            isMovableByWindowBackground: isMovableByWindowBackground,
            ignoresMouseEvents: ignoresMouseEvents,
            isOpaque: isOpaque,
            hasShadow: hasShadow)
    }

    // MARK: - Spec to AppKit

    private static func styleMask(for style: Set<PanelStyle>) -> NSWindow.StyleMask {
        // `.borderless` is the empty mask, so only the other bit adds anything.
        style.contains(.nonactivatingPanel) ? [.borderless, .nonactivatingPanel] : [.borderless]
    }

    private static func windowLevel(for level: PanelLevel) -> NSWindow.Level {
        switch level {
        case .statusBar: .statusBar
        case .floating: .floating
        case .unmapped: .normal
        }
    }

    private static func behavior(for set: Set<PanelCollectionBehavior>) -> NSWindow.CollectionBehavior {
        var result: NSWindow.CollectionBehavior = []
        for item in set {
            switch item {
            case .canJoinAllSpaces: result.insert(.canJoinAllSpaces)
            case .fullScreenAuxiliary: result.insert(.fullScreenAuxiliary)
            case .stationary: result.insert(.stationary)
            case .ignoresCycle: result.insert(.ignoresCycle)
            case .unmapped: break
            }
        }
        return result
    }

    // MARK: - AppKit to spec

    private static func panelLevel(for level: NSWindow.Level) -> PanelLevel {
        switch level {
        case .statusBar: .statusBar
        case .floating: .floating
        default: .unmapped
        }
    }

    private static func panelBehavior(for live: NSWindow.CollectionBehavior) -> Set<PanelCollectionBehavior> {
        let known: [(NSWindow.CollectionBehavior, PanelCollectionBehavior)] = [
            (.canJoinAllSpaces, .canJoinAllSpaces),
            (.fullScreenAuxiliary, .fullScreenAuxiliary),
            (.stationary, .stationary),
            (.ignoresCycle, .ignoresCycle),
        ]
        var result = Set<PanelCollectionBehavior>()
        var rest = live
        for (flag, item) in known where live.contains(flag) {
            result.insert(item)
            rest.remove(flag)
        }
        if !rest.isEmpty { result.insert(.unmapped) }
        return result
    }

    private static func panelStyle(for live: NSWindow.StyleMask) -> Set<PanelStyle> {
        var result = Set<PanelStyle>()
        var rest = live
        if live.contains(.nonactivatingPanel) {
            result.insert(.nonactivatingPanel)
            rest.remove(.nonactivatingPanel)
        }
        // Borderless is the empty mask: it holds when no other bit is left.
        result.insert(rest.isEmpty ? .borderless : .unmapped)
        return result
    }
}
