import AtlasKit
import Dispatch
import Foundation
import Network
import Synchronization

/// `AtlasDesktop --diagnose [--write-probe] [--host <host[:port]>]`
///
/// The scriptable half of the signing spike and a troubleshooting tool. It
/// never prompts for Accessibility, never registers the login item and never
/// shows a window. The full argument parser lands in plan 14-07.
enum DiagnoseCommand {
    struct Options {
        var writeProbe = false
        var host: String?
        var port: UInt16 = 443
    }

    /// Runs the diagnosis and returns the process exit code.
    static func run(arguments: [String]) -> Int32 {
        let options: Options
        switch parse(arguments) {
        case .success(let parsed): options = parsed
        case .failure(let message):
            FileHandle.standardError.write(Data("atlas-desktop: \(message)\n".utf8))
            return 2
        }

        var record = LaunchDiagnostics.collect(mode: "diagnose")

        let probeStore = KeychainStore(service: AppIdentity.diagnosticsKeychainService)
        if options.writeProbe, (try? probeStore.load()) == nil {
            // A harmless item, so a later rebuild can prove the access list survived.
            try? probeStore.save(PairingCredentials(host: "probe", token: "probe"))
        }
        record.keychainProbeStatus = probeStore.readStatus()

        if let host = options.host {
            record.probeHost = host
            let slot = Slot<NetworkResult>()
            let done = DispatchSemaphore(value: 0)
            Task.detached {
                slot.set(await probeNetwork(host: host, port: options.port))
                done.signal()
            }
            // The probes never need the main run loop, so blocking here is safe.
            if done.wait(timeout: .now() + 25) == .success, let result = slot.get() {
                record.lnPath = result.lnPath
                record.lnUnsatisfiedReason = result.unsatisfiedReason
                record.wssProbe = result.wss
            } else {
                record.lnPath = "timeout"
                record.wssProbe = "unreachable:timeout"
            }
        }

        LaunchDiagnostics.write(record)
        return 0
    }

    enum ParseResult {
        case success(Options)
        case failure(String)
    }

    static func parse(_ arguments: [String]) -> ParseResult {
        var options = Options()
        var index = 0
        while index < arguments.count {
            switch arguments[index] {
            case "--diagnose": break
            case "--write-probe": options.writeProbe = true
            case "--host":
                index += 1
                guard index < arguments.count else { return .failure("--host needs a value") }
                switch parseHost(arguments[index]) {
                case .success(let (host, port)):
                    options.host = host
                    if let port { options.port = port }
                case .failure(let message): return .failure(message)
                }
            default: return .failure("unknown argument \(arguments[index])")
            }
            index += 1
        }
        return .success(options)
    }

    static func parseHost(_ value: String) -> Result<(String, UInt16?), HostError> {
        let forbidden = value.unicodeScalars.contains {
            CharacterSet.whitespacesAndNewlines.contains($0) || CharacterSet.controlCharacters.contains($0)
        }
        if value.isEmpty || value.contains("://") || value.contains("/") || forbidden {
            return .failure(HostError("--host takes a bare host or host:port"))
        }
        let parts = value.split(separator: ":", omittingEmptySubsequences: false)
        if parts.count == 1 { return .success((value, nil)) }
        guard parts.count == 2, !parts[0].isEmpty, let port = UInt16(parts[1]), port > 0 else {
            return .failure(HostError("--host takes a bare host or host:port"))
        }
        return .success((String(parts[0]), port))
    }

    struct HostError: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }
}

extension DiagnoseCommand.ParseResult {
    fileprivate static func failure(_ error: DiagnoseCommand.HostError) -> Self {
        .failure(error.description)
    }
}

// MARK: - Probes

private final class Slot<T: Sendable>: Sendable {
    private let value = Mutex<T?>(nil)
    func set(_ new: T) { value.withLock { $0 = new } }
    func get() -> T? { value.withLock { $0 } }
}

private struct NetworkResult: Sendable {
    var lnPath: String
    var unsatisfiedReason: String?
    var wss: String
}

private func probeNetwork(host: String, port: UInt16) async -> NetworkResult {
    let path = await probePath(host: host, port: port)
    let wss = await probeWebSocket(host: host, port: port)
    return NetworkResult(lnPath: path.state, unsatisfiedReason: path.reason, wss: wss)
}

/// A TLS `NWConnection` for at most 5 seconds. A local network denial shows as
/// `unsatisfiedReason == .localNetworkDenied` on the path (RESEARCH Pattern 12).
private func probePath(host: String, port: UInt16) async -> (state: String, reason: String?) {
    guard let nwPort = NWEndpoint.Port(rawValue: port) else { return ("failed", nil) }
    let probe = PathProbe(host: NWEndpoint.Host(host), port: nwPort)
    return await probe.run(timeout: 5)
}

private final class PathProbe: @unchecked Sendable {
    // All mutable state is touched only on `queue`.
    private let queue = DispatchQueue(label: "atlas.diagnose.path")
    private let connection: NWConnection
    private var finished = false
    private var sawWaiting = false

    init(host: NWEndpoint.Host, port: NWEndpoint.Port) {
        connection = NWConnection(host: host, port: port, using: .tls)
    }

    func run(timeout: TimeInterval) async -> (state: String, reason: String?) {
        await withCheckedContinuation { continuation in
            let finish: @Sendable (String) -> Void = { [self] state in
                guard !finished else { return }
                finished = true
                // The reason is meaningful only when the path is not satisfied.
                var reason: String?
                if let path = connection.currentPath, path.status != .satisfied {
                    reason = reasonName(path.unsatisfiedReason)
                }
                connection.cancel()
                continuation.resume(returning: (state, reason))
            }
            connection.stateUpdateHandler = { [self] state in
                switch state {
                case .ready: finish("ready")
                case .failed: finish("failed")
                case .waiting: sawWaiting = true
                default: break
                }
            }
            connection.start(queue: queue)
            queue.asyncAfter(deadline: .now() + timeout) { [self] in
                finish(sawWaiting ? "waiting" : "timeout")
            }
        }
    }
}

private func reasonName(_ reason: NWPath.UnsatisfiedReason) -> String {
    switch reason {
    case .notAvailable: return "notAvailable"
    case .cellularDenied: return "cellularDenied"
    case .wifiDenied: return "wifiDenied"
    case .localNetworkDenied: return "localNetworkDenied"
    case .vpnInactive: return "vpnInactive"
    @unknown default: return "unknown"
    }
}

/// A websocket handshake with no Authorization header. The URL scheme is fixed
/// to wss (D-08). A 403 means the server answered and refused (RESEARCH Pitfall 2).
private func probeWebSocket(host: String, port: UInt16) async -> String {
    var components = URLComponents()
    components.scheme = "wss"
    components.host = host
    if port != 443 { components.port = Int(port) }
    components.path = "/ws/desktop"
    guard let url = components.url else { return "unreachable:bad_url" }

    let task = URLSession.shared.webSocketTask(with: url)
    let timedOut = Mutex(false)
    let timer = Task {
        try await Task.sleep(for: .seconds(10))
        timedOut.withLock { $0 = true }
        task.cancel(with: .goingAway, reason: nil)
    }
    task.resume()
    defer { timer.cancel() }
    do {
        _ = try await task.receive()
        task.cancel(with: .normalClosure, reason: nil)
        return "connected_unexpected"
    } catch {
        if timedOut.withLock({ $0 }) { return "unreachable:timeout" }
        if let response = task.response as? HTTPURLResponse, response.statusCode == 403 {
            return "reached_403"
        }
        let code = (error as NSError).code
        if (error as NSError).domain == NSURLErrorDomain, (-1206 ... -1200).contains(code) {
            return "tls_error"
        }
        return "unreachable:\(code)"
    }
}
