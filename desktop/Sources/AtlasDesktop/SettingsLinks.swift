import AppKit

/// Opens the System Settings pane for a permission (RESEARCH A11). When the
/// deep link does not open, it opens System Settings itself.
enum SettingsLinks {
    enum Pane {
        case accessibility, localNetwork, location

        fileprivate var url: URL? {
            let name: String
            switch self {
            case .accessibility: name = "Privacy_Accessibility"
            case .localNetwork: name = "Privacy_LocalNetwork"
            case .location: name = "Privacy_LocationServices"
            }
            return URL(string: "x-apple.systempreferences:com.apple.preference.security?" + name)
        }
    }

    @MainActor
    static func open(_ pane: Pane) {
        if let url = pane.url, NSWorkspace.shared.open(url) { return }
        NSWorkspace.shared.open(URL(fileURLWithPath: "/System/Applications/System Settings.app"))
    }
}
