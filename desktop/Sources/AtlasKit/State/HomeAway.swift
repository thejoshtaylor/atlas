import Foundation

/// A rough position. It lives on this Mac only (D-11): it is not a field of
/// the hello, of any frame or of the launch record.
public struct Coordinate: Equatable, Sendable, Codable {
    public var latitude: Double
    public var longitude: Double

    public init(latitude: Double, longitude: Double) {
        self.latitude = latitude
        self.longitude = longitude
    }
}

/// Home or Away, from a rough location (D-10, D-11).
///
/// The answer is a menu label and nothing else. It never changes whether or
/// when the Mac reconnects, and a denied or unknown location never reads as Away.
public enum HomeAway {
    /// UI-SPEC discretion: one kilometre.
    public static let radiusMeters = 1000.0
    /// While offline, a new fix is requested at most this often.
    public static let fixInterval: TimeInterval = 15 * 60

    private static let earthRadiusMeters = 6_371_000.0

    /// Great-circle distance, haversine.
    public static func distanceMeters(_ a: Coordinate, _ b: Coordinate) -> Double {
        let lat1 = a.latitude * .pi / 180
        let lat2 = b.latitude * .pi / 180
        let dLat = lat2 - lat1
        let dLon = (b.longitude - a.longitude) * .pi / 180
        let h = sin(dLat / 2) * sin(dLat / 2) + cos(lat1) * cos(lat2) * sin(dLon / 2) * sin(dLon / 2)
        return 2 * earthRadiusMeters * asin(min(1, sqrt(h)))
    }

    /// True only when the Mac is not connected, Location is allowed, a home point
    /// and a current position exist, and they are more than `radiusMeters` apart.
    public static func isAway(
        current: Coordinate?,
        home: Coordinate?,
        locationAllowed: Bool,
        connected: Bool,
        radiusMeters: Double = HomeAway.radiusMeters
    ) -> Bool {
        guard !connected, locationAllowed, let current, let home else { return false }
        return distanceMeters(current, home) > radiusMeters
    }

    /// The first request is always due. Later ones wait for the interval.
    public static func fixDue(lastRequest: Date?, now: Date, interval: TimeInterval = fixInterval) -> Bool {
        guard let lastRequest else { return true }
        return now.timeIntervalSince(lastRequest) >= interval
    }
}
