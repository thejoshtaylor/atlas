import AppKit
import AtlasKit
import SwiftUI

/// The menu under the menu-bar icon. The first line is the state (D-20). Plan
/// 14-11 adds "Finish Setup…", "Set Home Here" and "Setup…".
struct MenuContent: View {
    let model: AppModel

    var body: some View {
        Text(model.menuState.firstLine)
        Divider()
        if model.menuState.kind == .unpaired || model.menuState.kind == .revoked {
            Button("Pair\u{2026}") {
                // Task 2 opens the setup window here.
            }
            Divider()
        }
        Button("Quit ATLAS") {
            NSApplication.shared.terminate(nil)
        }
        .keyboardShortcut("q")
    }
}
