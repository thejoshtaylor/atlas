import AppKit
import SwiftUI

/// Where the setup window opens. Plan 14-11 adds the other steps.
enum SetupFocus: Equatable {
    /// The step list, at the Pair row.
    case pair
    /// The Pair form.
    case pairForm
}

/// The one setup window, owned by AppKit so that a URL handler or a revoke
/// event can open it. `openWindow` is only reachable from a view.
@MainActor
final class SetupWindowController: NSObject, NSWindowDelegate {
    static let shared = SetupWindowController()

    private var window: NSWindow?

    func show(focus: SetupFocus) {
        let window = self.window ?? makeWindow()
        self.window = window
        NSApp.activate()
        window.makeKeyAndOrderFront(nil)
        // The agent app can lose the activation race against the browser that
        // opened the link (RESEARCH A7), so check once and fall back.
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(150))
            guard !window.isKeyWindow else { return }
            NSApp.activate(ignoringOtherApps: true)
            window.makeKeyAndOrderFront(nil)
        }
    }

    private func makeWindow() -> NSWindow {
        let hosting = NSHostingController(rootView: SetupView(model: AppModel.shared))
        hosting.sizingOptions = [.preferredContentSize]
        let window = NSWindow(contentViewController: hosting)
        window.title = "ATLAS Setup"
        window.styleMask = [.titled, .closable]
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()
        return window
    }

    func windowWillClose(_ notification: Notification) {
        AppModel.shared.setupWindowDidClose()
    }
}
