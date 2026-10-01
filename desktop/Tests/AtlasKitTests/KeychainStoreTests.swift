import Foundation
import Security
import Testing

@testable import AtlasKit

// These tests never touch the real Keychain: a CI runner has no unlocked login
// keychain. The real item is proven by the signing spike and the phase UAT.
@Suite struct KeychainStoreTests {
    @Test func baseQueryIsAGenericPasswordForTheServiceOnly() {
        let query = KeychainStore.baseQuery(service: "svc.test")
        #expect(query[kSecClass as String] as? String == kSecClassGenericPassword as String)
        #expect(query[kSecAttrService as String] as? String == "svc.test")
        #expect(query[kSecUseDataProtectionKeychain as String] == nil)
        #expect(query.count == 2)
    }

    @Test func loadIsNilAtFirst() throws {
        #expect(try InMemorySecretStore().load() == nil)
    }

    @Test func saveThenLoadReturnsTheSamePairing() throws {
        let store = InMemorySecretStore()
        try store.save(PairingCredentials(host: "svr.test", token: "t-1"))
        #expect(try store.load() == PairingCredentials(host: "svr.test", token: "t-1"))
    }

    @Test func aSecondSaveReplacesTheFirst() throws {
        let store = InMemorySecretStore()
        try store.save(PairingCredentials(host: "a.test", token: "t-1"))
        try store.save(PairingCredentials(host: "b.test", token: "t-2"))
        #expect(try store.load() == PairingCredentials(host: "b.test", token: "t-2"))
    }

    @Test func deleteThenLoadIsNil() throws {
        let store = InMemorySecretStore()
        try store.save(PairingCredentials(host: "svr.test", token: "t-1"))
        try store.delete()
        #expect(try store.load() == nil)
    }

    @Test func aStoreToldToFailItsNextSaveThrowsTheStatusOnce() throws {
        let store = InMemorySecretStore()
        store.failNextSave = -25293
        #expect(throws: SecretStoreError.keychain(-25293)) {
            try store.save(PairingCredentials(host: "svr.test", token: "t-1"))
        }
        try store.save(PairingCredentials(host: "svr.test", token: "t-1"))
        #expect(try store.load() != nil)
    }

    @Test func textFormsNameTheHostAndNeverTheToken() {
        let credentials = PairingCredentials(host: "svr.test", token: "t-1")
        for text in [String(describing: credentials), String(reflecting: credentials)] {
            #expect(text.contains("svr.test"))
            #expect(!text.contains("t-1"))
        }
        var dumped = ""
        dump(credentials, to: &dumped)
        #expect(!dumped.contains("t-1"))
    }

    // The mapping of a read status to an outcome. Whether the real login
    // keychain returns these statuses with no UI is checked in the phase UAT.
    @Test func aStoredPairingReadsSilently() throws {
        let store = InMemorySecretStore()
        try store.save(PairingCredentials(host: "svr.test", token: "t-1"))
        #expect(store.readOutcome() == .silent)
    }

    @Test func anEmptyStoreIsNotStoredAndNeverNeedsInteraction() {
        #expect(InMemorySecretStore().readOutcome() == .notStored)
    }

    @Test func aReadThatNeededAPromptIsNeverSilent() throws {
        let store = InMemorySecretStore()
        try store.save(PairingCredentials(host: "svr.test", token: "t-1"))
        for status in [errSecInteractionNotAllowed, errSecUserCanceled, errSecAuthFailed, errSecNoAccessForItem] {
            store.readStatusOverride = status
            #expect(store.readOutcome() == .needsInteraction(status))
            #expect(store.readOutcome().label == "needs_interaction")
        }
    }

    @Test func anUnknownStatusIsFailedAndKeepsItsCode() {
        #expect(KeychainReadOutcome(status: -50) == .failed(-50))
        #expect(KeychainReadOutcome(status: -50).label == "failed")
    }

    // WR-02: a failed read must never look like "not paired".
    @Test func aStoredPairingLoadsAsFound() throws {
        let store = InMemorySecretStore()
        let credentials = PairingCredentials(host: "svr.test", token: "t-1")
        try store.save(credentials)
        #expect(PairingLoad.read(from: store) == .found(credentials))
    }

    @Test func anEmptyStoreLoadsAsNotPaired() {
        #expect(PairingLoad.read(from: InMemorySecretStore()) == .notPaired)
    }

    @Test func aFailedReadKeepsItsStatusAndIsNeverNotPaired() throws {
        let store = InMemorySecretStore()
        let credentials = PairingCredentials(host: "svr.test", token: "t-1")
        try store.save(credentials)
        for status in [errSecInteractionNotAllowed, errSecUserCanceled, errSecAuthFailed] {
            store.failNextLoad = status
            #expect(PairingLoad.read(from: store) == .failed(status: status))
        }
        #expect(PairingLoad.read(from: store) == .found(credentials))
    }
}
