import Foundation
import Testing

@testable import AtlasKit

/// The ring id that the menu bar item "Stop Ringing" reads (UI-SPEC "Keyboard
/// and accessibility"). It follows the server frames and does not depend on
/// whether the panel shows.
@Suite struct RingTrackerTests {
    private func ringing(_ id: Int) -> PanelEvent {
        .timerRinging(TimerRinging(timerId: id, kind: "timer", label: "pasta"))
    }

    @Test func startsWithNoRing() {
        #expect(RingTracker().timerId == nil)
    }

    @Test func ringingSetsTheId() {
        var tracker = RingTracker()
        tracker.apply(ringing(7))
        #expect(tracker.timerId == 7)
    }

    @Test func aSecondRingReplacesTheId() {
        var tracker = RingTracker()
        tracker.apply(ringing(7))
        tracker.apply(ringing(8))
        #expect(tracker.timerId == 8)
    }

    @Test func stoppedWithTheSameIdClears() {
        var tracker = RingTracker()
        tracker.apply(ringing(7))
        tracker.apply(.timerStopped(TimerStopped(timerId: 7)))
        #expect(tracker.timerId == nil)
    }

    @Test func stoppedWithAnotherIdChangesNothing() {
        var tracker = RingTracker()
        tracker.apply(ringing(8))
        tracker.apply(.timerStopped(TimerStopped(timerId: 7)))
        #expect(tracker.timerId == 8)
    }

    @Test func aLostConnectionClears() {
        var tracker = RingTracker()
        tracker.apply(ringing(7))
        tracker.apply(.connectionLost)
        #expect(tracker.timerId == nil)
    }

    @Test func otherEventsChangeNothing() {
        var tracker = RingTracker()
        tracker.apply(ringing(7))
        tracker.apply(.closePressed)
        tracker.apply(.hoverChanged(true))
        tracker.apply(.deadline)
        #expect(tracker.timerId == 7)
    }

    @Test func noRingMeansNoStopAction() {
        #expect(RingTracker().stopAction(panelRing: nil) == nil)
    }

    @Test func anIdlePanelRingIsStoppedThroughThePanel() {
        var tracker = RingTracker()
        tracker.apply(ringing(7))
        let ring = PanelRing(timerId: 7, kind: .timer, label: "pasta", stop: .idle)
        #expect(tracker.stopAction(panelRing: ring) == .clickInPanel)
    }

    @Test func aClosedPanelSendsTheStopDirectly() {
        var tracker = RingTracker()
        tracker.apply(ringing(7))
        #expect(tracker.stopAction(panelRing: nil) == .send(timerId: 7))
    }

    @Test func aPendingPanelRingSendsTheStopDirectly() {
        var tracker = RingTracker()
        tracker.apply(ringing(7))
        let ring = PanelRing(timerId: 7, kind: .timer, label: "pasta", stop: .pending)
        #expect(tracker.stopAction(panelRing: ring) == .send(timerId: 7))
    }

    @Test func aPanelRingForAnotherIdSendsTheStopDirectly() {
        var tracker = RingTracker()
        tracker.apply(ringing(8))
        let ring = PanelRing(timerId: 7, kind: .timer, label: "pasta", stop: .idle)
        #expect(tracker.stopAction(panelRing: ring) == .send(timerId: 8))
    }
}
