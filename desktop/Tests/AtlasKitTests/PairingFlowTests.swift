import Foundation
import Testing

@testable import AtlasKit

@Suite struct PairingFlowTests {
    private func target(_ server: String = "svr.test", token: String = "t-1") throws -> PairingTarget {
        try PairingTarget.manual(server: server, token: token)
    }

    private func snapshot(
        _ status: LinkStatus, failure: DialFailure? = nil, everConnected: Bool = false
    ) -> ConnectionSnapshot {
        ConnectionSnapshot(status: status, everConnected: everConnected, lastFailure: failure)
    }

    /// A flow that has saved the token and is waiting for the first dial.
    private func connectingFlow(_ server: String = "svr.test") throws -> PairingFlow {
        var flow = PairingFlow()
        let t = try target(server)
        _ = flow.handle(.linkOpened(.success(t), currentHost: nil))
        _ = flow.handle(.confirmed)
        _ = flow.handle(.saveSucceeded)
        return flow
    }

    // MARK: the happy path

    @Test func aLinkOnAnUnpairedMacAsksBeforeSaving() throws {
        var flow = PairingFlow()
        let t = try target()
        let effects = flow.handle(.linkOpened(.success(t), currentHost: nil))
        #expect(flow.state == .confirm(t))
        #expect(effects.isEmpty)
    }

    @Test func confirmSavesTheTokenAndTheSaveStartsTheConnection() throws {
        var flow = PairingFlow()
        let t = try target()
        _ = flow.handle(.linkOpened(.success(t), currentHost: nil))
        #expect(flow.handle(.confirmed) == [.saveToken(t)])
        #expect(flow.state == .saving(t))
        #expect(flow.handle(.saveSucceeded) == [.startConnection(t)])
        #expect(flow.state == .connecting(host: "svr.test", error: nil))
    }

    @Test func aConnectedStatusMeansPaired() throws {
        var flow = try connectingFlow()
        _ = flow.handle(.connection(snapshot(.connecting(host: "svr.test"))))
        _ = flow.handle(.connection(snapshot(.connected(host: "svr.test", since: Date()), everConnected: true)))
        #expect(flow.state == .paired(host: "svr.test"))
    }

    @Test func nothingIsSavedBeforeConfirm() throws {
        var flow = PairingFlow()
        let t = try target()
        var all = flow.handle(.linkOpened(.success(t), currentHost: nil))
        all += flow.handle(.cancelled)
        #expect(flow.state == .idle)
        #expect(!all.contains(.saveToken(t)))
        // A confirm with nothing waiting does nothing.
        #expect(flow.handle(.confirmed).isEmpty)
    }

    // MARK: a link while paired

    @Test func aLinkWhileConnectedAsksToReplace() throws {
        var flow = PairingFlow()
        let t = try target("new.test")
        _ = flow.handle(.linkOpened(.success(t), currentHost: "old.test"))
        #expect(flow.state == .replace(current: "old.test", target: t))
    }

    @Test func keepCurrentSavesNothing() throws {
        var flow = PairingFlow()
        let t = try target("new.test")
        _ = flow.handle(.linkOpened(.success(t), currentHost: "old.test"))
        let effects = flow.handle(.keepCurrent)
        #expect(flow.state == .idle)
        #expect(effects.isEmpty)
    }

    @Test func replaceStopsTheOldConnectionAndSavesTheNewToken() throws {
        var flow = PairingFlow()
        let t = try target("new.test")
        _ = flow.handle(.linkOpened(.success(t), currentHost: "old.test"))
        #expect(flow.handle(.replaceChosen) == [.stopCurrentConnection, .saveToken(t)])
        #expect(flow.state == .saving(t))
    }

    @Test func cancelOnTheReplaceSheetAlsoKeepsTheCurrentPairing() throws {
        var flow = PairingFlow()
        _ = flow.handle(.linkOpened(.success(try target("new.test")), currentHost: "old.test"))
        #expect(flow.handle(.cancelled).isEmpty)
        #expect(flow.state == .idle)
    }

    // MARK: bad links

    @Test func anInsecureLinkFailsAsInsecure() {
        var flow = PairingFlow()
        _ = flow.handle(.linkOpened(.failure(.insecure), currentHost: nil))
        #expect(flow.state == .failed(.insecure))
    }

    @Test func everyOtherLinkErrorIsNotAnAtlasLink() {
        for error in [PairingLinkError.notAtlasLink, .invalidServer, .missingToken] {
            var flow = PairingFlow()
            _ = flow.handle(.linkOpened(.failure(error), currentHost: nil))
            #expect(flow.state == .failed(.notAtlasLink))
        }
    }

    // MARK: failures after the save

    @Test func aKeychainWriteFailureIsReported() throws {
        var flow = PairingFlow()
        _ = flow.handle(.linkOpened(.success(try target()), currentHost: nil))
        _ = flow.handle(.confirmed)
        let effects = flow.handle(.saveFailed)
        #expect(flow.state == .failed(.keychainWrite))
        #expect(effects.isEmpty)
    }

    @Test func aRefusedFreshTokenIsRemovedAndNeverRunsTheRevokeFlow() throws {
        var flow = try connectingFlow()
        _ = flow.handle(.connection(snapshot(.connecting(host: "svr.test"))))
        let effects = flow.handle(
            .connection(snapshot(.offline(host: "svr.test", lastConnected: nil), failure: .refused)))
        #expect(flow.state == .failed(.refused))
        #expect(effects == [.removeSavedToken, .stopCurrentConnection])
    }

    @Test func aRevokedStatusDuringAFreshPairingIsTheSameRefusal() throws {
        var flow = try connectingFlow()
        _ = flow.handle(.connection(snapshot(.connecting(host: "svr.test"))))
        let effects = flow.handle(.connection(snapshot(.revoked(host: "svr.test"), failure: .refused)))
        #expect(flow.state == .failed(.refused))
        #expect(effects.contains(.removeSavedToken))
    }

    @Test func anUnreachableServerKeepsConnectingWithAnError() throws {
        var flow = try connectingFlow()
        _ = flow.handle(.connection(snapshot(.connecting(host: "svr.test"))))
        let effects = flow.handle(
            .connection(snapshot(.offline(host: "svr.test", lastConnected: nil), failure: .unreachable)))
        #expect(flow.state == .connecting(host: "svr.test", error: .unreachable(host: "svr.test")))
        #expect(effects.isEmpty)
    }

    @Test func aLaterConnectedStatusClearsTheError() throws {
        var flow = try connectingFlow()
        _ = flow.handle(.connection(snapshot(.connecting(host: "svr.test"))))
        _ = flow.handle(
            .connection(snapshot(.offline(host: "svr.test", lastConnected: nil), failure: .unreachable)))
        _ = flow.handle(.connection(snapshot(.connected(host: "svr.test", since: Date()), everConnected: true)))
        #expect(flow.state == .paired(host: "svr.test"))
    }

    @Test func aVersionMismatchFails() throws {
        var flow = try connectingFlow()
        _ = flow.handle(.connection(snapshot(.connecting(host: "svr.test"))))
        let effects = flow.handle(.connection(snapshot(.versionMismatch(host: "svr.test"))))
        #expect(flow.state == .failed(.versionMismatch))
        #expect(effects.isEmpty)
    }

    @Test func snapshotsFromBeforeThePairingAreIgnored() throws {
        var flow = try connectingFlow("new.test")
        // The old pairing was refused once and offline; these were queued earlier.
        let stale = flow.handle(
            .connection(snapshot(.offline(host: "old.test", lastConnected: nil), failure: .refused)))
        #expect(stale.isEmpty)
        #expect(flow.state == .connecting(host: "new.test", error: nil))
        _ = flow.handle(.connection(snapshot(.connected(host: "old.test", since: Date()), everConnected: true)))
        #expect(flow.state == .connecting(host: "new.test", error: nil))
    }

    @Test func snapshotsOutsideAPairingChangeNothing() {
        var flow = PairingFlow()
        let effects = flow.handle(
            .connection(snapshot(.offline(host: "svr.test", lastConnected: nil), failure: .refused)))
        #expect(flow.state == .idle)
        #expect(effects.isEmpty)
    }

    // MARK: copy

    @Test func theSixMessagesMatchTheUISpec() {
        let host = "svr.test"
        #expect(
            PairingFailure.notAtlasLink.message(host: host)
                == "That is not an ATLAS pair link. Open the Macs page, add a Mac, and copy its pair link.")
        #expect(
            PairingFailure.insecure.message(host: host)
                == "This link is not secure. ATLAS pairs over wss:// only. Open the Macs page over https and copy a new link."
        )
        #expect(
            PairingFailure.refused.message(host: host)
                == "The server did not accept this token. Add a new Mac on the Macs page and pair again.")
        #expect(
            PairingFailure.unreachable(host: host).message(host: host)
                == "Could not reach svr.test. Check that this Mac is on the right network and try again.")
        #expect(
            PairingFailure.versionMismatch.message(host: host)
                == "This app and the server use different versions. Run install.sh again to update the app.")
        #expect(
            PairingFailure.keychainWrite.message(host: host)
                == "Could not save the token to Keychain. Unlock your login keychain and try again.")
    }

    // MARK: the form

    @Test func submitNeedsALinkOrBothAServerAndAToken() {
        #expect(!PairingFlow.canSubmit(link: "", server: "", token: ""))
        #expect(!PairingFlow.canSubmit(link: "", server: "svr.test", token: ""))
        #expect(!PairingFlow.canSubmit(link: "", server: "", token: "t-1"))
        #expect(!PairingFlow.canSubmit(link: "  ", server: " ", token: " "))
        #expect(PairingFlow.canSubmit(link: "", server: "svr.test", token: "t-1"))
        #expect(PairingFlow.canSubmit(link: "atlas://pair?server=svr.test", server: "", token: ""))
    }
}
