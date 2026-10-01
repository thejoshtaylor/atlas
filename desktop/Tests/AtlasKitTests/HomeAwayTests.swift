import Foundation
import Testing

@testable import AtlasKit

@Suite struct HomeAwayTests {
    private let origin = Coordinate(latitude: 0, longitude: 0)
    private let far = Coordinate(latitude: 0, longitude: 0.01)  // about 1112 m

    @Test func identicalCoordinatesAreZeroMetersApart() {
        #expect(HomeAway.distanceMeters(origin, origin) == 0)
    }

    @Test func oneHundredthOfADegreeOfLongitudeAtTheEquator() {
        let meters = HomeAway.distanceMeters(origin, far)
        #expect(abs(meters - 1112) < 5)
    }

    @Test func distanceIsSymmetric() {
        let a = Coordinate(latitude: 47.6, longitude: -122.3)
        let b = Coordinate(latitude: 47.7, longitude: -122.2)
        #expect(abs(HomeAway.distanceMeters(a, b) - HomeAway.distanceMeters(b, a)) < 0.001)
    }

    @Test func awayNeedsAllFourConditions() {
        #expect(HomeAway.isAway(current: far, home: origin, locationAllowed: true, connected: false))
    }

    @Test func connectedIsNeverAway() {
        #expect(!HomeAway.isAway(current: far, home: origin, locationAllowed: true, connected: true))
    }

    @Test func aDeniedOrUnknownLocationIsNeverAway() {
        #expect(!HomeAway.isAway(current: far, home: origin, locationAllowed: false, connected: false))
        #expect(!HomeAway.isAway(current: far, home: nil, locationAllowed: true, connected: false))
        #expect(!HomeAway.isAway(current: nil, home: origin, locationAllowed: true, connected: false))
    }

    @Test func exactlyAtTheRadiusIsNotAway() {
        let meters = HomeAway.distanceMeters(origin, far)
        #expect(!HomeAway.isAway(current: far, home: origin, locationAllowed: true, connected: false, radiusMeters: meters))
        #expect(HomeAway.isAway(current: far, home: origin, locationAllowed: true, connected: false, radiusMeters: meters - 1))
    }

    @Test func insideTheRadiusIsNotAway() {
        let near = Coordinate(latitude: 0, longitude: 0.005)
        #expect(!HomeAway.isAway(current: near, home: origin, locationAllowed: true, connected: false))
        #expect(HomeAway.radiusMeters == 1000)
    }

    @Test func fixRequestsWaitFifteenMinutes() {
        let now = Date(timeIntervalSince1970: 1_790_000_000)
        #expect(HomeAway.fixDue(lastRequest: nil, now: now))
        #expect(!HomeAway.fixDue(lastRequest: now.addingTimeInterval(-899), now: now))
        #expect(HomeAway.fixDue(lastRequest: now.addingTimeInterval(-900), now: now))
    }
}
