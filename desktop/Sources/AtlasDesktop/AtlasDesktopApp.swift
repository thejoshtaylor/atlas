import AppKit
import SwiftUI

/// The menu-bar agent. The icon and the first menu line follow the live
/// connection (D-20). There is no animation, so Reduce Motion needs no case.
struct AtlasDesktopApp: App {
    private let model = AppModel.shared

    init() {
        AppModel.shared.start()
    }

    var body: some Scene {
        MenuBarExtra {
            MenuContent(model: model)
        } label: {
            Image(systemName: model.menuState.iconFilled ? "globe.americas.fill" : "globe.americas")
                .accessibilityLabel(model.menuState.accessibilityLabel)
        }
        .menuBarExtraStyle(.menu)
    }
}
