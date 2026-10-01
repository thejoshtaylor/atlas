import AppKit
import AtlasKit

/// Single instance, launch, and `atlas://` links.
///
/// The link handler lives here and not in a SwiftUI URL modifier, because the setup
/// window is owned by AppKit and a link also arrives on a cold launch.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var isDuplicate = false

    func applicationWillFinishLaunching(_ notification: Notification) {
        // Only the installed bundle checks, so a bare `swift run` binary still starts.
        guard Bundle.main.bundleIdentifier == AppIdentity.bundleIdentifier else { return }
        let me = ProcessInfo.processInfo.processIdentifier
        let others = NSRunningApplication
            .runningApplications(withBundleIdentifier: AppIdentity.bundleIdentifier)
            .filter { $0.processIdentifier != me }
        // Two sockets with one token would supersede each other (close 4000).
        if let first = others.first {
            first.activate()
            isDuplicate = true
            NSApp.terminate(nil)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard !isDuplicate else { return }
        AppModel.shared.start()
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        guard let link = urls.first(where: { $0.scheme?.lowercased() == AppIdentity.urlScheme }) else { return }
        AppModel.shared.receivePairLink(link)
    }
}
