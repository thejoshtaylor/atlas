import Foundation
import Security

/// The pairing in the legacy login keychain: one generic-password item per
/// service (D-04). The account holds the server host and the data holds the
/// token, so one atomic item binds the two.
///
/// The base query never selects the data-protection keychain. That keychain
/// returns -34018 without a provisioning profile, and this app has none.
/// The token is never logged or printed.
public struct KeychainStore: SecretStore {
    public let service: String

    public init(service: String = AppIdentity.pairingKeychainService) {
        self.service = service
    }

    public static func baseQuery(service: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
        ]
    }

    public func save(_ credentials: PairingCredentials) throws {
        try delete()
        var query = Self.baseQuery(service: service)
        query[kSecAttrAccount as String] = credentials.host
        query[kSecValueData as String] = Data(credentials.token.utf8)
        query[kSecAttrLabel as String] = "ATLAS pairing"
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else { throw SecretStoreError.keychain(status) }
    }

    public func load() throws -> PairingCredentials? {
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
        let status = SecItemDelete(Self.baseQuery(service: service) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw SecretStoreError.keychain(status)
        }
    }

    public func readStatus() -> Int32 {
        var query = Self.baseQuery(service: service)
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        query[kSecReturnData as String] = true
        // An access-list mismatch becomes a status here, never a prompt.
        query[kSecUseAuthenticationUI as String] = kSecUseAuthenticationUIFail
        var result: CFTypeRef?
        return SecItemCopyMatching(query as CFDictionary, &result)
    }
}
