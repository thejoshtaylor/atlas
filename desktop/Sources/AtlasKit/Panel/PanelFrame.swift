import CoreGraphics

/// The panel's fixed measures, in points (UI-SPEC "Spacing Scale" and "Placement").
public enum PanelLayout {
    public static let width: CGFloat = 360
    /// The height stops growing here; past it the content area scrolls (D-02).
    public static let heightCap: CGFloat = 420
    /// The gap between the panel and the top and right edges of `visibleFrame`.
    public static let inset: CGFloat = 16
    public static let padding: CGFloat = 16
    public static let contentWidth: CGFloat = 328
    public static let headerHeight: CGFloat = 28
    public static let closeHitSize: CGFloat = 28
    public static let ringGraphicSize: CGFloat = 48
    public static let sectionGap: CGFloat = 8
    public static let symbolGap: CGFloat = 4
}

/// One display as plain rectangles. The shell copies these out of `NSScreen`,
/// so the placement rule runs in `swift test` without a window server.
public struct PanelScreen: Sendable, Equatable {
    public let frame: CGRect
    /// The frame without the menu bar, the Dock and the notch.
    public let visibleFrame: CGRect

    public init(frame: CGRect, visibleFrame: CGRect) {
        self.frame = frame
        self.visibleFrame = visibleFrame
    }
}

/// Where the panel sits (D-01, D-02).
///
/// Store a corner and an inset, never absolute coordinates. The shell calls
/// these again on a screen-parameters change (UI-SPEC "Placement"). Nothing here
/// reads a main-screen property: AppKit coordinates, origin at the bottom-left
/// of the primary display, y grows upward.
public enum PanelFrame {
    /// The index of the screen whose `frame` holds the pointer. With none, the
    /// screen that holds the center of the frontmost window's frame. With none
    /// again, the first screen. No screens gives nil.
    ///
    /// Containment is closed on every edge, so a pointer on a shared edge picks
    /// the first matching screen in list order.
    public static func chooseScreen(
        screens: [PanelScreen],
        pointer: CGPoint,
        fallbackWindowFrame: CGRect?
    ) -> Int? {
        guard !screens.isEmpty else { return nil }
        if let hit = screens.firstIndex(where: { contains($0.frame, pointer) }) { return hit }
        if let window = fallbackWindowFrame {
            let center = CGPoint(x: window.midX, y: window.midY)
            if let hit = screens.firstIndex(where: { contains($0.frame, center) }) { return hit }
        }
        return 0
    }

    /// The panel frame on `screen`: 360 pt wide, the content height capped at
    /// 420 pt, the top-right corner 16 pt inside the top and right edges of
    /// `visibleFrame`.
    public static func frame(in screen: PanelScreen, contentHeight: CGFloat) -> CGRect {
        let height = clampedHeight(contentHeight)
        let visible = screen.visibleFrame
        return CGRect(
            x: visible.maxX - PanelLayout.inset - PanelLayout.width,
            y: visible.maxY - PanelLayout.inset - height,
            width: PanelLayout.width,
            height: height)
    }

    /// A new height with the top-right corner fixed: growth and shrinkage go
    /// downward.
    public static func resized(_ current: CGRect, toContentHeight contentHeight: CGFloat) -> CGRect {
        let height = clampedHeight(contentHeight)
        return CGRect(
            x: current.maxX - PanelLayout.width,
            y: current.maxY - height,
            width: PanelLayout.width,
            height: height)
    }

    private static func clampedHeight(_ contentHeight: CGFloat) -> CGFloat {
        min(max(contentHeight, 0), PanelLayout.heightCap)
    }

    /// `CGRect.contains` is half-open; the rule here is closed.
    private static func contains(_ rect: CGRect, _ point: CGPoint) -> Bool {
        point.x >= rect.minX && point.x <= rect.maxX && point.y >= rect.minY && point.y <= rect.maxY
    }
}
