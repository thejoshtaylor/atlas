import ApplicationServices
import AtlasKit
import Foundation
import OSLog
import Security
import ServiceManagement

/// One line of `launch.jsonl` (D-25). There is no token field, so no code path
/// can write the token here.
struct LaunchRecord: Encodable, Sendable {
    var at: String
    var mode: String
    var build: String
    var appVersion: String
    var cdhash: String?
    var designatedRequirement: String?
    var axTrusted: Bool
    var keychainPairingStatus: Int32
    var keychainProbeStatus: Int32?
    var loginItemStatus: String
    var pairedHost: String?
    // Diagnose-only fields, omitted from a plain launch record.
    var probeHost: String?
    var lnPath: String?
    var lnUnsatisfiedReason: String?
    var wssProbe: String?

    enum CodingKeys: String, CodingKey {
        case at, mode, build, cdhash
        case appVersion = "app_version"
        case designatedRequirement = "designated_requirement"
        case axTrusted = "ax_trusted"
        case keychainPairingStatus = "keychain_pairing_status"
        case keychainProbeStatus = "keychain_probe_status"
        case loginItemStatus = "login_item_status"
        case pairedHost = "paired_host"
        case probeHost = "probe_host"
        case lnPath = "ln_path"
        case lnUnsatisfiedReason = "ln_unsatisfied_reason"
        case wssProbe = "wss_probe"
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(at, forKey: .at)
        try c.encode(mode, forKey: .mode)
        try c.encode(build, forKey: .build)
        try c.encode(appVersion, forKey: .appVersion)
        // Explicit nulls: a reader can tell "unsigned" from "field missing".
        try c.encode(cdhash, forKey: .cdhash)
        try c.encode(designatedRequirement, forKey: .designatedRequirement)
        try c.encode(axTrusted, forKey: .axTrusted)
        try c.encode(keychainPairingStatus, forKey: .keychainPairingStatus)
        try c.encode(keychainProbeStatus, forKey: .keychainProbeStatus)
        try c.encode(loginItemStatus, forKey: .loginItemStatus)
        try c.encode(pairedHost, forKey: .pairedHost)
        try c.encodeIfPresent(probeHost, forKey: .probeHost)
        try c.encodeIfPresent(lnPath, forKey: .lnPath)
        try c.encodeIfPresent(lnUnsatisfiedReason, forKey: .lnUnsatisfiedReason)
        try c.encodeIfPresent(wssProbe, forKey: .wssProbe)
    }
}

enum LaunchDiagnostics {
    static func collect(mode: String) -> LaunchRecord {
        let info = Bundle.main.infoDictionary
        let signing = signingState()
        let store = KeychainStore()
        let pairingStatus = store.readStatus()
        // Read the host only when the silent read worked, so a launch never
        // raises a Keychain prompt of its own.
        let host: String? = pairingStatus == errSecSuccess ? (try? store.load()?.host) : nil
        return LaunchRecord(
            at: ISO8601DateFormatter().string(from: Date()),
            mode: mode,
            build: info?["CFBundleVersion"] as? String ?? "unbundled",
            appVersion: info?["CFBundleShortVersionString"] as? String ?? "unbundled",
            cdhash: signing.cdhash,
            designatedRequirement: signing.requirement,
            axTrusted: accessibilityTrusted(),
            keychainPairingStatus: pairingStatus,
            keychainProbeStatus: nil,
            loginItemStatus: loginItemStatusName(),
            pairedHost: host
        )
    }

    /// Never prompts. The prompt option is the string literal because the
    /// matching Accessibility constant is not concurrency-safe in Swift 6 mode.
    static func accessibilityTrusted() -> Bool {
        let options = ["AXTrustedCheckOptionPrompt": false] as CFDictionary
        return AXIsProcessTrustedWithOptions(options)
    }

    static func loginItemStatusName() -> String {
        switch SMAppService.mainApp.status {
        case .enabled: return "enabled"
        case .requiresApproval: return "requiresApproval"
        case .notRegistered: return "notRegistered"
        case .notFound: return "notFound"
        @unknown default: return "unknown"
        }
    }

    /// The cdhash and designated requirement of the running code. Each failure
    /// becomes nil and never a crash.
    static func signingState() -> (cdhash: String?, requirement: String?) {
        var code: SecCode?
        guard SecCodeCopySelf([], &code) == errSecSuccess, let code else { return (nil, nil) }
        var staticCode: SecStaticCode?
        guard SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess, let staticCode else {
            return (nil, nil)
        }

        var cdhash: String?
        var infoRef: CFDictionary?
        if SecCodeCopySigningInformation(staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &infoRef)
            == errSecSuccess,
            let info = infoRef as? [String: Any],
            let unique = info[kSecCodeInfoUnique as String] as? Data
        {
            cdhash = unique.map { String(format: "%02x", $0) }.joined()
        }

        var requirement: String?
        var req: SecRequirement?
        if SecCodeCopyDesignatedRequirement(staticCode, [], &req) == errSecSuccess, let req {
            var text: CFString?
            if SecRequirementCopyString(req, [], &text) == errSecSuccess, let text {
                requirement = text as String
            }
        }
        return (cdhash, requirement)
    }

    static func logDirectory() -> URL {
        if let override = ProcessInfo.processInfo.environment["ATLAS_LOG_DIR"], !override.isEmpty {
            return URL(filePath: override, directoryHint: .isDirectory)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Library/Logs/\(AppIdentity.logDirectoryName)", directoryHint: .isDirectory)
    }

    /// Appends one JSON line to `launch.jsonl` and logs a one-line summary.
    static func write(_ record: LaunchRecord) {
        let logger = Logger(subsystem: AppIdentity.bundleIdentifier, category: "launch")
        logger.info(
            "mode=\(record.mode, privacy: .public) ax=\(record.axTrusted, privacy: .public) keychain=\(record.keychainPairingStatus, privacy: .public) login_item=\(record.loginItemStatus, privacy: .public)"
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(record) else { return }
        let directory = logDirectory()
        let file = directory.appending(path: AppIdentity.launchLogFileName)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: file.path(percentEncoded: false)) {
                FileManager.default.createFile(atPath: file.path(percentEncoded: false), contents: nil)
            }
            let handle = try FileHandle(forWritingTo: file)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: data + Data("\n".utf8))
        } catch {
            logger.error("could not write launch.jsonl: \(String(describing: error), privacy: .public)")
        }
    }

    static func recordLaunch() {
        write(collect(mode: "launch"))
    }
}
