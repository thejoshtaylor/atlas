import AtlasKit
import OSLog
import ServiceManagement

/// The login item (D-19). `SMAppService.mainApp` registers the installed app
/// bundle, so a repeated call never adds a second entry. `apply` also checks the
/// state first, so a second Continue does nothing.
@MainActor
struct LoginItem {
    private let log = Logger(subsystem: AppIdentity.bundleIdentifier, category: "login-item")

    func status() -> LoginItemState {
        .from(rawStatus: SMAppService.mainApp.status.rawValue)
    }

    /// Registers or unregisters only when the read-back state calls for it, then
    /// reads the state back again. A failure is logged by type and never shown:
    /// the row reads the real state, and `requiresApproval` is not a failure.
    @discardableResult
    func apply(toggleOn: Bool) -> LoginItemState {
        let service = SMAppService.mainApp
        let current = status()
        do {
            if LoginItemState.shouldRegister(toggleOn: toggleOn, current: current) {
                try service.register()
            } else if LoginItemState.shouldUnregister(toggleOn: toggleOn, current: current) {
                try service.unregister()
            }
        } catch {
            log.error("The login item change failed: \(String(describing: type(of: error)), privacy: .public).")
        }
        return status()
    }

    func openApproval() {
        SMAppService.openSystemSettingsLoginItems()
    }
}
