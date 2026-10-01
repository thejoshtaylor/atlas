import CoreGraphics
import Testing

@testable import AtlasKit

@Suite struct PanelFrameTests {
    private let primary = PanelScreen(
        frame: CGRect(x: 0, y: 0, width: 1512, height: 982),
        visibleFrame: CGRect(x: 0, y: 0, width: 1512, height: 949))
    private let above = PanelScreen(
        frame: CGRect(x: 0, y: 982, width: 2560, height: 1440),
        visibleFrame: CGRect(x: 0, y: 982, width: 2560, height: 1415))
    private let left = PanelScreen(
        frame: CGRect(x: -2560, y: 0, width: 2560, height: 1440),
        visibleFrame: CGRect(x: -2560, y: 0, width: 2560, height: 1415))

    @Test func oneScreenPlacesTheShortPanelAtTheTopRight() {
        let frame = PanelFrame.frame(in: primary, contentHeight: 200)
        #expect(frame == CGRect(x: 1136, y: 733, width: 360, height: 200))
    }

    @Test func aTallContentHeightIsCappedAtTheSameTopEdge() {
        let frame = PanelFrame.frame(in: primary, contentHeight: 1000)
        #expect(frame.height == PanelLayout.heightCap)
        #expect(frame.height == 420)
        let top: CGFloat = primary.visibleFrame.maxY - 16
        let right: CGFloat = primary.visibleFrame.maxX - 16
        #expect(frame.maxY == top)
        #expect(frame.maxX == right)
        #expect(frame.maxY == 933)
        #expect(frame.maxX == 1496)
    }

    @Test func aDisplayAboveThePrimaryIsChosenByThePointer() {
        let index = PanelFrame.chooseScreen(
            screens: [primary, above], pointer: CGPoint(x: 100, y: 1500), fallbackWindowFrame: nil)
        #expect(index == 1)
        let frame = PanelFrame.frame(in: above, contentHeight: 100)
        #expect(frame.maxY == above.visibleFrame.maxY - 16)
        #expect(frame.maxX == above.visibleFrame.maxX - 16)
    }

    @Test func aDisplayAtANegativeXIsChosenAndKeepsTheInset() {
        let index = PanelFrame.chooseScreen(
            screens: [primary, left], pointer: CGPoint(x: -100, y: 500), fallbackWindowFrame: nil)
        #expect(index == 1)
        let frame = PanelFrame.frame(in: left, contentHeight: 120)
        #expect(frame.maxX == left.visibleFrame.maxX - 16)
        #expect(frame.minX == left.visibleFrame.maxX - 16 - 360)
        #expect(frame.maxY == left.visibleFrame.maxY - 16)
    }

    @Test func aPointerOnASharedEdgePicksTheFirstScreenInListOrder() {
        // The primary's top edge (y 982) is the display above's bottom edge.
        let point = CGPoint(x: 100, y: 982)
        #expect(
            PanelFrame.chooseScreen(screens: [primary, above], pointer: point, fallbackWindowFrame: nil) == 0)
        #expect(
            PanelFrame.chooseScreen(screens: [above, primary], pointer: point, fallbackWindowFrame: nil) == 0)
    }

    @Test func aPointerOnNoScreenFallsBackToTheScreenHoldingTheWindowCenter() {
        let window = CGRect(x: 500, y: 1200, width: 400, height: 300)
        let index = PanelFrame.chooseScreen(
            screens: [primary, above], pointer: CGPoint(x: 9000, y: 9000), fallbackWindowFrame: window)
        #expect(index == 1)
    }

    @Test func aPointerOnNoScreenAndNoFallbackPicksTheFirstScreen() {
        let index = PanelFrame.chooseScreen(
            screens: [primary, above], pointer: CGPoint(x: 9000, y: 9000), fallbackWindowFrame: nil)
        #expect(index == 0)
    }

    @Test func aFallbackWindowOnNoScreenPicksTheFirstScreen() {
        let window = CGRect(x: 9000, y: 9000, width: 10, height: 10)
        let index = PanelFrame.chooseScreen(
            screens: [primary, above], pointer: CGPoint(x: 9000, y: 9000), fallbackWindowFrame: window)
        #expect(index == 0)
    }

    @Test func noScreensGivesNil() {
        #expect(
            PanelFrame.chooseScreen(screens: [], pointer: .zero, fallbackWindowFrame: nil) == nil)
    }

    @Test func aResizeKeepsTheTopRightCornerAndCapsTheHeight() {
        let current = PanelFrame.frame(in: primary, contentHeight: 200)
        let taller = PanelFrame.resized(current, toContentHeight: 300)
        #expect(taller.maxX == current.maxX)
        #expect(taller.maxY == current.maxY)
        #expect(taller.size == CGSize(width: 360, height: 300))

        let capped = PanelFrame.resized(current, toContentHeight: 5000)
        #expect(capped.maxX == current.maxX)
        #expect(capped.maxY == current.maxY)
        #expect(capped.height == 420)

        let shorter = PanelFrame.resized(taller, toContentHeight: 90)
        #expect(shorter.maxX == current.maxX)
        #expect(shorter.maxY == current.maxY)
        #expect(shorter.height == 90)
    }

    @Test func theSpacingTokensMatchTheUISpec() {
        #expect(PanelLayout.width == 360)
        #expect(PanelLayout.contentWidth == PanelLayout.width - 2 * PanelLayout.padding)
        #expect(PanelLayout.contentWidth == 328)
        #expect(PanelLayout.inset == 16)
    }
}
