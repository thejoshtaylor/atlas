import Foundation
import UserNotifications

/// The one notification of this phase (D-21): "ATLAS was unpaired".
///
/// It runs only from the revoke event and never at launch. Permission is asked
/// the first time it is needed. A denied permission skips the notification, and
/// the setup window is then the only signal (RESEARCH A12).
enum RevokeNotifier {
    static let identifier = "atlas.revoked"

    static func postOnce() {
        // A bare `swift run` binary has no bundle, and the center needs one.
        guard Bundle.main.bundleIdentifier != nil else { return }
        Task {
            let center = UNUserNotificationCenter.current()
            let settings = await center.notificationSettings()
            switch settings.authorizationStatus {
            case .denied:
                return
            case .notDetermined:
                guard let granted = try? await center.requestAuthorization(options: [.alert]), granted
                else { return }
            default:
                break
            }
            let content = UNMutableNotificationContent()
            content.title = "ATLAS was unpaired"
            content.body = "The server revoked this Mac. Open ATLAS to pair again."
            // A fixed identifier means a repeat replaces the first and never stacks.
            let request = UNNotificationRequest(identifier: identifier, content: content, trigger: nil)
            try? await center.add(request)
        }
    }
}
