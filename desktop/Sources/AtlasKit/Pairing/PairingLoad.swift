/// The result of reading the saved pairing at launch.
///
/// A failed read is not "not paired". A locked keychain, a cancelled prompt or
/// a transient error leaves the pairing in place, so the app must not open the
/// Pair step or save over the item. It keeps the status code (never the token)
/// for the log and tries again.
public enum PairingLoad: Equatable, Sendable {
    case found(PairingCredentials)
    case notPaired
    case failed(status: Int32)

    /// The status used when a read fails with no Keychain status to report.
    public static let unknownStatus: Int32 = -1

    public static func read(from store: some SecretStore) -> PairingLoad {
        do {
            if let credentials = try store.load() { return .found(credentials) }
            return .notPaired
        } catch SecretStoreError.keychain(let status) {
            return .failed(status: status)
        } catch SecretStoreError.malformedItem {
            // A retry cannot repair a damaged item. Pairing again replaces it.
            return .notPaired
        } catch {
            return .failed(status: unknownStatus)
        }
    }
}
