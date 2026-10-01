import Foundation
import LocalAuthentication
import Security
import Synchronization

/// The pairing in the legacy login keychain: one generic-password item per
/// service (D-04). The account holds the server host and the data holds the
/// token, so one atomic item binds the two.
///
/// The base query never selects the data-protection keychain. That keychain
/// returns -34018 without a provisioning profile, and this app has none.
/// The token is never logged or printed.
public struct KeychainStore: SecretStore {
    public let service: String

    /// One lock for every Keychain call in this process. `readStatus` turns the
    /// session-wide "no UI" switch off and on, and that switch would also hit a
    /// real `load` running on another thread, so no two calls may overlap.
    private static let access = Mutex(())

    public init(service: String = AppIdentity.pairingKeychainService) {
        self.service = service
    }

    public static func baseQuery(service: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
        ]
    }

    /// Updates the item in place and adds it only when none exists. The old
    /// pairing is never deleted first, so a failed save (a locked keychain, a
    /// cancelled prompt) leaves the working pairing as it was.
    public func save(_ credentials: PairingCredentials) throws {
        try Self.access.withLock { _ in try saveLocked(credentials) }
    }

    private func saveLocked(_ credentials: PairingCredentials) throws {
        let match = Self.baseQuery(service: service)
        let attributes: [String: Any] = [
            kSecAttrAccount as String: credentials.host,
            kSecValueData as String: Data(credentials.token.utf8),
        ]
        var status = SecItemUpdate(match as CFDictionary, attributes as CFDictionary)
        if status == errSecItemNotFound {
            var query = match
            query.merge(attributes) { _, new in new }
            query[kSecAttrLabel as String] = "ATLAS pairing"
            status = SecItemAdd(query as CFDictionary, nil)
        }
        guard status == errSecSuccess else { throw SecretStoreError.keychain(status) }
    }

    public func load() throws -> PairingCredentials? {
        try Self.access.withLock { _ in try loadLocked() }
    }

    private func loadLocked() throws -> PairingCredentials? {
        var query = Self.baseQuery(service: service)
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        query[kSecReturnAttributes as String] = true
        query[kSecReturnData as String] = true
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess else { throw SecretStoreError.keychain(status) }
        guard
            let item = result as? [String: Any],
            let host = item[kSecAttrAccount as String] as? String,
            let data = item[kSecValueData as String] as? Data,
            let token = String(data: data, encoding: .utf8)
        else { throw SecretStoreError.malformedItem }
        return PairingCredentials(host: host, token: token)
    }

    public func delete() throws {
        try Self.access.withLock { _ in try deleteLocked() }
    }

    private func deleteLocked() throws {
        let status = SecItemDelete(Self.baseQuery(service: service) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw SecretStoreError.keychain(status)
        }
    }

    /// A read that must not show UI (D-25, D-29). `kSecUseAuthenticationUIFail`
    /// alone did not do that on the legacy login keychain: the signing spike saw
    /// a status of 0 while Keychain dialogs appeared. So this read turns UI off
    /// two ways: a LocalAuthentication context that forbids interaction, and the
    /// session-wide Keychain switch, restored on exit. A read that would have
    /// prompted then returns a status such as errSecInteractionNotAllowed.
    /// Unit tests cover the status mapping only. The real behavior of both
    /// switches on the login keychain is checked in the phase UAT.
    public func readStatus() -> Int32 {
        Self.access.withLock { _ in readStatusLocked() }
    }

    private func readStatusLocked() -> Int32 {
        let context = LAContext()
        context.interactionNotAllowed = true
        var query = Self.baseQuery(service: service)
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        query[kSecReturnData as String] = true
        query[kSecUseAuthenticationContext as String] = context

        var wasAllowed = DarwinBoolean(true)
        let hadState = SecKeychainGetUserInteractionAllowed(&wasAllowed) == errSecSuccess
        SecKeychainSetUserInteractionAllowed(false)
        defer {
            if hadState { SecKeychainSetUserInteractionAllowed(wasAllowed.boolValue) }
        }
        var result: CFTypeRef?
        return SecItemCopyMatching(query as CFDictionary, &result)
    }
}
