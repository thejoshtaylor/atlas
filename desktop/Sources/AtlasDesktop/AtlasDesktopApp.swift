import AppKit
import SwiftUI

/// The menu-bar agent. This is the real Unpaired state (D-20). Plan 14-10
/// replaces the content with the live connection state.
struct AtlasDesktopApp: App {
    var body: some Scene {
        MenuBarExtra {
            Text("Not paired")
            Divider()
            Button("Quit ATLAS") {
                NSApplication.shared.terminate(nil)
            }
            .keyboardShortcut("q")
        } label: {
            Image(systemName: "globe.americas")
                .accessibilityLabel("ATLAS, Not paired")
        }
        .menuBarExtraStyle(.menu)
    }
}
