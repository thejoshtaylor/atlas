import Foundation

// The ring rules (Task 3 of plan 15-07). The turn reducer dispatches to these.
extension PanelReducer {
    static func timerRinging(_ state: inout PanelState, _ message: TimerRinging, _ now: Date, _ effects: inout [PanelEffect]) {}

    static func timerStopped(_ state: inout PanelState, _ message: TimerStopped, _ now: Date, _ effects: inout [PanelEffect]) {}

    static func stopClicked(_ state: inout PanelState, _ now: Date, _ effects: inout [PanelEffect]) {}

    static func deadlineRing(_ state: inout PanelState, _ now: Date, _ effects: inout [PanelEffect]) {}
}
