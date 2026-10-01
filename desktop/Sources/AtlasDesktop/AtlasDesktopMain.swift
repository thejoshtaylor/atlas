import AppKit
import AtlasKit
import Foundation

@main
enum AtlasDesktopMain {
    @MainActor
    static func main() {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.contains("--panel-spike") {
            // No menu, no single-instance check: a scripted probe.
            exit(PanelSpikeCommand.run(arguments: arguments))
        }
        if arguments.contains("--diagnose") {
            // No UI and no single-instance check: this is a scripted probe.
            exit(DiagnoseCommand.run(arguments: arguments))
        }
        // A Keychain read must never delay the menu, so record off the main thread.
        Task.detached(priority: .utility) {
            LaunchDiagnostics.recordLaunch()
        }
        AtlasDesktopApp.main()
    }
}
