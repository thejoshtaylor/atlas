import Foundation
import Testing

@testable import AtlasKit

@Suite struct ConnectionPolicyTests {
    private let policy = ReconnectPolicy()

    @Test func delayDoublesUpToTheCapWithNoJitterAtTheMiddle() {
        #expect(policy.delay(attempt: 0, random: 0.5) == 1.0)
        #expect(policy.delay(attempt: 1, random: 0.5) == 2.0)
        #expect(policy.delay(attempt: 5, random: 0.5) == 32.0)
        #expect(policy.delay(attempt: 6, random: 0.5) == 60.0)
        #expect(policy.delay(attempt: 20, random: 0.5) == 60.0)
    }

    @Test func jitterSpreadsTheDelayByTwentyPercent() {
        #expect(abs(policy.delay(attempt: 2, random: 0.0) - 3.2) < 1e-9)
        let high = policy.delay(attempt: 2, random: 1.0.nextDown)
        #expect(high <= 4.8)
        #expect(high > 4.79)
    }

    @Test func noDelayIsNegativeOrAboveSeventyTwoSeconds() {
        for attempt in [-5, 0, 3, 6, 50, 1000] {
            for random in [-1.0, 0.0, 0.25, 0.5, 0.99, 1.0, 7.0] {
                let value = policy.delay(attempt: attempt, random: random)
                #expect(value > 0)
                #expect(value <= 72.0)
            }
        }
    }

    @Test func policyDefaultsAreTheDecidedNumbers() {
        #expect(policy.initial == 1.0)
        #expect(policy.factor == 2.0)
        #expect(policy.cap == 60.0)
        #expect(policy.jitter == 0.2)
        #expect(policy.stableAfter == 30.0)
        #expect(policy.wakeDelay == 1.0)
    }

    @Test func aHandshake403IsRefused() {
        #expect(DialFailure.classify(httpStatus: 403, closeCode: 0) == .refused)
    }

    @Test func aCloseCodeWithNoStatusIsClosed() {
        #expect(DialFailure.classify(httpStatus: nil, closeCode: 4001) == .closed(code: 4001))
    }

    @Test func everythingElseIsUnreachable() {
        #expect(DialFailure.classify(httpStatus: nil, closeCode: 0) == .unreachable)
        #expect(DialFailure.classify(httpStatus: 500, closeCode: 0) == .unreachable)
        #expect(DialFailure.classify(httpStatus: 401, closeCode: 0) == .unreachable)
    }
}
