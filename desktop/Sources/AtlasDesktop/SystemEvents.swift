import AppKit
import AtlasKit
import Network
import Synchronization

/// Turns what the system reports into `SystemEvent` values for the connection
/// (D-05): the network path, and sleep and wake.
///
/// The path monitor forwards a change only. A path that flaps with the same
/// status never reaches the actor.
@MainActor
final class SystemEvents {
    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "org.atlas-assistant.desktop.path")
    private var observers: [NSObjectProtocol] = []
    private var running = false

    func start(forwardingTo connection: DesktopConnection) {
        guard !running else { return }
        running = true

        let lastSatisfied = Mutex<Bool?>(nil)
        monitor.pathUpdateHandler = { path in
            let satisfied = path.status == .satisfied
            let changed = lastSatisfied.withLock { last -> Bool in
                defer { last = satisfied }
                return last != satisfied
            }
            guard changed else { return }
            Task { await connection.handle(satisfied ? .pathSatisfied : .pathUnsatisfied) }
        }
        monitor.start(queue: queue)

        let center = NSWorkspace.shared.notificationCenter
        observers.append(
            center.addObserver(forName: NSWorkspace.didWakeNotification, object: nil, queue: .main) { _ in
                Task { await connection.handle(.didWake) }
            })
        observers.append(
            center.addObserver(forName: NSWorkspace.willSleepNotification, object: nil, queue: .main) { _ in
                Task { await connection.handle(.willSleep) }
            })
    }

    func stop() {
        monitor.cancel()
        for observer in observers { NSWorkspace.shared.notificationCenter.removeObserver(observer) }
        observers.removeAll()
        running = false
    }
}
