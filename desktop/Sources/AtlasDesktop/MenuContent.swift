import AppKit
import AtlasKit
import SwiftUI

/// The menu under the menu-bar icon. The first line is the state (D-20). The
/// item set is fixed: at most one of "Finish Setup\u{2026}" and "Setup\u{2026}" shows.
struct MenuContent: View {
    let model: AppModel

    var body: some View {
        Text(model.menuState.firstLine)
        Divider()
        if model.ringingTimerId != nil {
            Button {
                model.stopRinging()
            } label: {
                Text(verbatim: PanelCopy.stopRingingMenuItem)
            }
        }
        if !model.setupProgress.requiredDone {
            Button("Finish Setup\u{2026}") {
                model.openSetup(focus: .pair)
            }
        }
        if model.menuState.kind == .unpaired || model.menuState.kind == .revoked {
            Button("Pair\u{2026}") {
                model.openSetup(focus: .pair)
            }
        }
        if model.location.auth == .allowed {
            Button("Set Home Here") {
                model.location.setHomeHere()
            }
        }
        if model.setupProgress.requiredDone {
            Button("Setup\u{2026}") {
                model.openSetup(focus: .pair)
            }
        }
        Divider()
        Button("Quit ATLAS") {
            NSApplication.shared.terminate(nil)
        }
        .keyboardShortcut("q")
    }
}
