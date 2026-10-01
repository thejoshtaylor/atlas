/// The one place that names the app's identity.
///
/// Changing `bundleIdentifier` resets the Accessibility grant and orphans the
/// Keychain item on every installed Mac. `Info.plist.template` must hold the
/// same value, and a test pins the two together.
public enum AppIdentity {
    public static let bundleIdentifier = "org.atlas-assistant.desktop"
    public static let pairingKeychainService = bundleIdentifier + ".pairing"
    public static let diagnosticsKeychainService = bundleIdentifier + ".diagnostics"
    public static let urlScheme = "atlas"
    public static let logDirectoryName = "ATLAS"
    public static let launchLogFileName = "launch.jsonl"
}
