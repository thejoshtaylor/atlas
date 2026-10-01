import AppKit
import ApplicationServices
import AtlasKit
import Foundation
import Observation

/// Accessibility and Local Network, read from the system (D-18).
///
/// Accessibility has no change notification, so a one-second poll runs while
/// the setup window is visible. Everything refreshes again when the app becomes
/// active, which is when the operator returns from System Settings.
///
/// Local Network has no API that returns the permission. The spike
/// (14-SPIKE.md, "Local Network detection: unreliable") saw no prompt over wss
/// and never saw `NWPath.unsatisfiedReason == .localNetworkDenied`, so this
/// type does not probe. The state comes from the connection result alone
/// (`LocalNetworkStatus.resolve`).
@MainActor
@Observable
final class PermissionMonitor {
    private(set) var axTrusted = false
    /// A measured Local Network read. It stays `nil` on macOS 26 (unreliable).
    private(set) var localNetworkProbe: LocalNetworkProbe?

    /// Runs on every activation, so the model can refresh its own values too.
    @ObservationIgnored var onActivate: (() -> Void)?
    @ObservationIgnored private var pollTask: Task<Void, Never>?
    @ObservationIgnored private var activationObserver: NSObjectProtocol?

    init() {
        refresh()
        activationObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.refresh()
                self?.onActivate?()
            }
        }
    }

    func refresh() {
        axTrusted = Self.isTrusted(prompt: false)
    }

    /// Polls once per second until `stopPolling`. Calling it twice keeps one loop.
    func startPolling() {
        guard pollTask == nil else { return }
        refresh()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                guard !Task.isCancelled else { return }
                self?.refresh()
            }
        }
    }

    func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    /// Adds ATLAS to the Accessibility list and shows the system prompt once,
    /// then opens the pane.
    func requestAccessibility() {
        _ = Self.isTrusted(prompt: true)
        SettingsLinks.open(.accessibility)
        refresh()
    }

    private static func isTrusted(prompt: Bool) -> Bool {
        // The string key, because the `kAXTrustedCheckOptionPrompt` symbol does
        // not compile in Swift 6 mode (RESEARCH Pattern 11).
        AXIsProcessTrustedWithOptions(["AXTrustedCheckOptionPrompt": prompt] as CFDictionary)
    }

    /// True when `host` is a LAN name, or resolves to a private or link-local
    /// address. A failed attempt to such a host is the one failure that may mean
    /// Local Network is off. The lookup blocks, so callers run it off the main
    /// thread. DNS does not need the permission.
    nonisolated static func isLocalNetworkHost(_ host: String) -> Bool {
        if LocalNetworkHost.looksLocal(host) { return true }
        var hints = addrinfo()
        hints.ai_socktype = SOCK_STREAM
        var found: UnsafeMutablePointer<addrinfo>?
        guard getaddrinfo(host, nil, &hints, &found) == 0, let first = found else { return false }
        defer { freeaddrinfo(found) }
        var cursor: UnsafeMutablePointer<addrinfo>? = first
        while let info = cursor {
            var buffer = [CChar](repeating: 0, count: Int(INET6_ADDRSTRLEN))
            let family = info.pointee.ai_family
            let text: String? = info.pointee.ai_addr.withMemoryRebound(to: sockaddr.self, capacity: 1) { addr in
                if family == AF_INET {
                    return addr.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { sin in
                        var address = sin.pointee.sin_addr
                        return inet_ntop(AF_INET, &address, &buffer, socklen_t(buffer.count)).map { String(cString: $0) }
                    }
                } else if family == AF_INET6 {
                    return addr.withMemoryRebound(to: sockaddr_in6.self, capacity: 1) { sin6 in
                        var address = sin6.pointee.sin6_addr
                        return inet_ntop(AF_INET6, &address, &buffer, socklen_t(buffer.count)).map { String(cString: $0) }
                    }
                }
                return nil
            }
            if let text, LocalNetworkHost.isLocalAddress(text) { return true }
            cursor = info.pointee.ai_next
        }
        return false
    }
}
