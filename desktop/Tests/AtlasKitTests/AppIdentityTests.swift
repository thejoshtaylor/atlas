import Foundation
import Testing

@testable import AtlasKit

@Suite struct AppIdentityTests {
    private func plist() throws -> [String: Any] {
        let url = desktopRoot().appending(path: "Resources/Info.plist.template")
        let data = try Data(contentsOf: url)
        let object = try PropertyListSerialization.propertyList(from: data, format: nil)
        return try #require(object as? [String: Any])
    }

    @Test func templateBundleIdentifierMatchesAppIdentity() throws {
        #expect(try plist()["CFBundleIdentifier"] as? String == AppIdentity.bundleIdentifier)
    }

    @Test func keychainServicesAreDerivedFromTheBundleIdentifier() {
        #expect(AppIdentity.pairingKeychainService == AppIdentity.bundleIdentifier + ".pairing")
        #expect(AppIdentity.diagnosticsKeychainService == AppIdentity.bundleIdentifier + ".diagnostics")
    }

    @Test func templateIsAMenuBarAgentOnMacOS15() throws {
        let p = try plist()
        #expect(p["CFBundleExecutable"] as? String == "AtlasDesktop")
        #expect(p["LSUIElement"] as? Bool == true)
        #expect(p["LSMinimumSystemVersion"] as? String == "15.0")
    }

    @Test func templateRegistersTheAtlasURLScheme() throws {
        let types = try #require(try plist()["CFBundleURLTypes"] as? [[String: Any]])
        let schemes = types.flatMap { $0["CFBundleURLSchemes"] as? [String] ?? [] }
        #expect(schemes.contains(AppIdentity.urlScheme))
    }

    @Test func templateHasUsageStringsAndNoTransportSecurityException() throws {
        let p = try plist()
        for key in ["NSLocalNetworkUsageDescription", "NSLocationUsageDescription",
                    "NSLocationWhenInUseUsageDescription"] {
            let value = try #require(p[key] as? String, "missing \(key)")
            #expect(!value.isEmpty)
        }
        #expect(p["NSAppTransportSecurity"] == nil)
    }
}
