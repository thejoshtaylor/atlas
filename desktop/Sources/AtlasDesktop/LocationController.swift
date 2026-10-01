import AtlasKit
import CoreLocation
import Foundation
import Observation

/// A rough location for Home or Away (D-10, D-11, D-12).
///
/// The home point and the current position live here and in `UserDefaults` on
/// this Mac. They never reach the connection, the hello, the launch record or a
/// log line: this file is the only one that names a coordinate, and it logs none.
@MainActor
@Observable
final class LocationController: NSObject, CLLocationManagerDelegate {
    private(set) var auth: LocationAuth = .notAsked
    private(set) var current: Coordinate?
    private(set) var home: Coordinate?
    private(set) var homeSavedAt: Date?

    /// Runs when the auth, the position or the home point changes.
    @ObservationIgnored var onChange: (() -> Void)?

    @ObservationIgnored private let manager = CLLocationManager()
    @ObservationIgnored private var lastRequest: Date?
    /// True while the next fix is the one the operator asked to keep as home.
    @ObservationIgnored private var savingHome = false
    @ObservationIgnored private var wantsFixAfterAuth = false

    private static let latitudeKey = "home.latitude"
    private static let longitudeKey = "home.longitude"
    private static let savedAtKey = "home.savedAt"

    override init() {
        super.init()
        manager.delegate = self
        // A kilometre is plenty for Home or Away, and a coarse fix is quick.
        manager.desiredAccuracy = kCLLocationAccuracyKilometer
        let defaults = UserDefaults.standard
        if defaults.object(forKey: Self.latitudeKey) != nil, defaults.object(forKey: Self.longitudeKey) != nil {
            home = Coordinate(
                latitude: defaults.double(forKey: Self.latitudeKey),
                longitude: defaults.double(forKey: Self.longitudeKey))
            homeSavedAt = defaults.object(forKey: Self.savedAtKey) as? Date
        }
        auth = Self.map(manager.authorizationStatus)
    }

    /// "Allow Location": asks once. macOS shows its own prompt on the first request.
    func allowLocation() {
        refresh()
        switch auth {
        case .notAsked:
            wantsFixAfterAuth = true
            manager.requestWhenInUseAuthorization()
        case .allowed:
            requestFix()
        case .denied:
            SettingsLinks.open(.location)
        }
    }

    /// "Set Home Here": the next fix becomes the home point on this Mac.
    func setHomeHere() {
        guard auth == .allowed else { return }
        savingHome = true
        requestFix()
    }

    /// Asks for a fix when the connection is offline and none was asked in the
    /// last 15 minutes. Never asks while connected.
    func requestFixIfOffline(now: Date = Date()) {
        guard auth == .allowed, HomeAway.fixDue(lastRequest: lastRequest, now: now) else { return }
        requestFix(now: now)
    }

    func refresh() {
        let next = Self.map(manager.authorizationStatus)
        guard next != auth else { return }
        auth = next
        onChange?()
    }

    private func requestFix(now: Date = Date()) {
        lastRequest = now
        manager.requestLocation()
    }

    private static func map(_ status: CLAuthorizationStatus) -> LocationAuth {
        switch status {
        case .notDetermined: .notAsked
        case .authorizedAlways, .authorizedWhenInUse: .allowed
        case .denied, .restricted: .denied
        @unknown default: .denied
        }
    }

    private func received(_ coordinate: Coordinate) {
        current = coordinate
        if savingHome {
            savingHome = false
            home = coordinate
            let saved = Date()
            homeSavedAt = saved
            let defaults = UserDefaults.standard
            defaults.set(coordinate.latitude, forKey: Self.latitudeKey)
            defaults.set(coordinate.longitude, forKey: Self.longitudeKey)
            defaults.set(saved, forKey: Self.savedAtKey)
        }
        onChange?()
    }

    // MARK: CLLocationManagerDelegate

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        Task { @MainActor in
            let next = Self.map(status)
            let changed = next != auth
            auth = next
            if next == .allowed, wantsFixAfterAuth {
                wantsFixAfterAuth = false
                requestFix()
            }
            if changed { onChange?() }
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let last = locations.last else { return }
        let coordinate = Coordinate(
            latitude: last.coordinate.latitude, longitude: last.coordinate.longitude)
        Task { @MainActor in received(coordinate) }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: any Error) {
        // A failed fix leaves the last position in place. The error can carry
        // nothing useful and is not logged.
        Task { @MainActor in savingHome = false }
    }
}
