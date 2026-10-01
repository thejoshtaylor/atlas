import Foundation
import Testing

@testable import AtlasKit

@Suite struct SetupStepsTests {
    private let calendar: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        return c
    }()
    private let locale = Locale(identifier: "en_US")
    private let now = Date(timeIntervalSince1970: 1_790_000_000)

    private func evaluate(_ inputs: SetupInputs) -> SetupProgress {
        SetupProgress.evaluate(inputs, now: now, calendar: calendar, locale: locale)
    }

    private func step(_ id: SetupStepID, _ inputs: SetupInputs) -> SetupStep {
        evaluate(inputs).steps.first { $0.id == id }!
    }

    private let allRequired = SetupInputs(
        pair: .paired(host: "atlas.example"), axTrusted: true, localNetwork: .granted)

    // MARK: Order and gate

    @Test func fiveStepsInFixedOrder() {
        let ids = evaluate(SetupInputs()).steps.map(\.id)
        #expect(ids == [.pair, .accessibility, .localNetwork, .location, .launchAtLogin])
        #expect(
            evaluate(SetupInputs()).steps.map(\.title)
                == ["Pair", "Accessibility", "Local Network", "Location and Set home", "Launch at login"])
    }

    @Test func nothingPairedDisablesContinue() {
        let progress = evaluate(SetupInputs())
        #expect(!progress.continueEnabled)
        #expect(!progress.requiredDone)
        #expect(step(.pair, SetupInputs()).stateText == "Not paired")
    }

    @Test func eachRequiredStepGatesContinue() {
        var inputs = allRequired
        #expect(evaluate(inputs).continueEnabled)
        inputs.pair = .notPaired
        #expect(!evaluate(inputs).continueEnabled)
        inputs = allRequired
        inputs.axTrusted = false
        #expect(!evaluate(inputs).continueEnabled)
        inputs = allRequired
        inputs.localNetwork = .checking
        #expect(!evaluate(inputs).continueEnabled)
        inputs.localNetwork = .notGranted
        #expect(!evaluate(inputs).continueEnabled)
    }

    @Test func connectingOrRevokedPairDoesNotCount() {
        var inputs = allRequired
        inputs.pair = .connecting(host: "atlas.example")
        #expect(!evaluate(inputs).continueEnabled)
        inputs.pair = .revoked
        #expect(!evaluate(inputs).continueEnabled)
        #expect(step(.pair, inputs).stateText == "Revoked by the server")
    }

    @Test func optionalStepsNeverChangeContinue() {
        var inputs = allRequired
        for location in [LocationAuth.notAsked, .allowed, .denied] {
            for login in [LoginItemState.on, .off, .needsApproval] {
                for toggle in [true, false] {
                    inputs.location = location
                    inputs.loginItem = login
                    inputs.launchAtLogin = toggle
                    #expect(evaluate(inputs).continueEnabled)
                }
            }
        }
    }

    // MARK: Local Network

    @Test func localNetworkChecksUntilAResultExists() {
        #expect(LocalNetworkStatus.resolve(everConnected: false, probe: nil) == .checking)
        #expect(LocalNetworkStatus.resolve(everConnected: true, probe: nil) == .granted)
    }

    @Test func aMeasuredProbeBeatsEverConnected() {
        #expect(LocalNetworkStatus.resolve(everConnected: false, probe: .denied) == .notGranted)
        #expect(LocalNetworkStatus.resolve(everConnected: true, probe: .denied) == .notGranted)
        #expect(LocalNetworkStatus.resolve(everConnected: false, probe: .ready) == .granted)
    }

    @Test func aFirstFailureToALanHostIsNotGranted() {
        #expect(
            LocalNetworkStatus.resolve(
                everConnected: false, probe: nil, lastFailure: .unreachable, hostIsLocal: true) == .notGranted)
    }

    @Test func aFailureToAnOtherHostStaysChecking() {
        #expect(
            LocalNetworkStatus.resolve(
                everConnected: false, probe: nil, lastFailure: .unreachable, hostIsLocal: false) == .checking)
        #expect(
            LocalNetworkStatus.resolve(everConnected: false, probe: nil, lastFailure: nil, hostIsLocal: true)
                == .checking)
    }

    @Test func aServerThatAnsweredProvesTheNetworkPath() {
        for failure in [DialFailure.refused, .closed(code: 4001)] {
            #expect(
                LocalNetworkStatus.resolve(
                    everConnected: false, probe: nil, lastFailure: failure, hostIsLocal: true) == .granted)
        }
    }

    @Test func aPastGrantSurvivesAnOutage() {
        #expect(
            LocalNetworkStatus.resolve(
                everConnected: false, probe: nil, lastFailure: .unreachable, hostIsLocal: true,
                previouslyGranted: true) == .granted)
    }

    @Test func localNetworkStateTexts() {
        #expect(LocalNetworkStatus.checking.stateText == "Checking\u{2026}")
        #expect(LocalNetworkStatus.granted.stateText == "Granted")
        #expect(LocalNetworkStatus.notGranted.stateText == "Not granted")
    }

    @Test func hostClassification() {
        for local in ["192.168.1.20", "10.0.0.5", "172.16.4.1", "172.31.255.1", "169.254.1.1", "atlas", "atlas.local",
            "box.home.arpa", "fd12::1", "fe80::1", "[fd00::2]", "127.0.0.1"]
        {
            #expect(LocalNetworkHost.looksLocal(local), "\(local)")
        }
        for remote in ["atlas.example.com", "8.8.8.8", "172.32.0.1", "172.15.0.1", "2606:4700::1", ""] {
            #expect(!LocalNetworkHost.looksLocal(remote), "\(remote)")
        }
    }

    // MARK: Login item

    @Test func loginItemMapsRawStatus() {
        #expect(LoginItemState.from(rawStatus: 1) == .on)
        #expect(LoginItemState.from(rawStatus: 2) == .needsApproval)
        #expect(LoginItemState.from(rawStatus: 0) == .off)
        #expect(LoginItemState.from(rawStatus: 3) == .off)
    }

    @Test func registrationIsIdempotent() {
        #expect(!LoginItemState.shouldRegister(toggleOn: true, current: .on))
        #expect(!LoginItemState.shouldRegister(toggleOn: true, current: .needsApproval))
        #expect(LoginItemState.shouldRegister(toggleOn: true, current: .off))
        #expect(!LoginItemState.shouldRegister(toggleOn: false, current: .off))
        #expect(LoginItemState.shouldUnregister(toggleOn: false, current: .on))
        #expect(!LoginItemState.shouldUnregister(toggleOn: false, current: .off))
        #expect(!LoginItemState.shouldUnregister(toggleOn: true, current: .on))
    }

    @Test func loginItemRowShowsApprovalNotFailure() {
        var inputs = allRequired
        inputs.loginItem = .needsApproval
        let row = step(.launchAtLogin, inputs)
        #expect(row.stateText == "Needs approval")
        #expect(row.buttonLabel == "Open Login Items")
        #expect(row.status == .needsAction)
        inputs.loginItem = .on
        #expect(step(.launchAtLogin, inputs).stateText == "On")
        #expect(step(.launchAtLogin, inputs).buttonLabel == nil)
        inputs.launchAtLogin = false
        #expect(step(.launchAtLogin, inputs).stateText == "Off")
    }

    // MARK: Texts

    @Test func accessibilityRow() {
        let off = step(.accessibility, SetupInputs())
        #expect(off.stateText == "Not granted")
        #expect(off.status == .needsAction)
        #expect(off.accessibilityLabel == "Accessibility, not granted. Open Settings.")
        let on = step(.accessibility, SetupInputs(axTrusted: true))
        #expect(on.stateText == "Granted")
        #expect(on.status == .done)
    }

    @Test func pairRowTruncatesALongHost() {
        let host = String(repeating: "a", count: 60)
        let row = step(.pair, SetupInputs(pair: .paired(host: host)))
        #expect(row.stateText == "Paired with " + MenuState.truncateMiddle(host))
        #expect(row.stateText.count < 60)
        #expect(row.buttonLabel == "Re-pair\u{2026}")
    }

    @Test func locationRowTexts() {
        var inputs = SetupInputs()
        #expect(step(.location, inputs).stateText == "Not asked")
        #expect(step(.location, inputs).buttonLabel == "Allow Location")
        #expect(step(.location, inputs).notes == ["Home not set"])
        inputs.location = .denied
        let denied = step(.location, inputs)
        #expect(denied.stateText == "Denied")
        #expect(denied.buttonLabel == "Open Settings")
        #expect(denied.notes.contains(SetupProgress.locationDeniedNote))
        inputs.location = .allowed
        inputs.homeSavedAt = now
        let allowed = step(.location, inputs)
        #expect(allowed.stateText == "Allowed")
        #expect(allowed.buttonLabel == "Set Home Here")
        #expect(allowed.notes.first == "Home set")
        #expect(allowed.notes.last?.hasPrefix("Saved ") == true)
    }

    @Test func accessibilityLabelsFollowTheFormula() {
        let progress = evaluate(allRequired)
        #expect(progress.steps[0].accessibilityLabel == "Pair, paired with atlas.example. Re-pair.")
        #expect(progress.steps[2].accessibilityLabel == "Local Network, granted. Open Settings.")
        #expect(progress.steps[4].accessibilityLabel == "Launch at login, on.")
    }
}
