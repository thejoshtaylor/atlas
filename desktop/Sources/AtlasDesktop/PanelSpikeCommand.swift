import AppKit
import AtlasKit
import Foundation
import SwiftUI

/// `AtlasDesktop --panel-spike [--check-config] [--seconds N] [--level statusBar|floating] [--log PATH]`
///
/// A scripted probe for the panel window (PANEL-02). `--check-config` builds one
/// panel without showing it and compares the live values to the spec. The run
/// mode shows the panel every 10 s over whatever a person is doing and logs one
/// JSON line per event. Neither mode ever activates the app or makes a window key.
enum PanelSpikeCommand {
    struct Options {
        var checkConfig = false
        var seconds = 180
        var level: PanelLevel?
        var logPath: String?
    }

    struct ArgumentError: Error { let message: String }

    @MainActor
    static func run(arguments: [String]) -> Int32 {
        let options: Options
        switch parse(arguments) {
        case .success(let parsed): options = parsed
        case .failure(let error):
            FileHandle.standardError.write(Data("atlas-desktop: \(error.message)\n".utf8))
            return 2
        }
        _ = NSApplication.shared
        NSApp.setActivationPolicy(.accessory)

        let expected = options.level.map { PanelWindowSpec.panel.withLevel($0) } ?? .panel
        let log: SpikeLog
        do {
            log = try SpikeLog(path: options.logPath)
        } catch {
            FileHandle.standardError.write(Data("atlas-desktop: cannot open the log: \(error)\n".utf8))
            return 2
        }
        let runner = SpikeRunner(spec: expected, log: log)
        if options.checkConfig {
            let matches = runner.panel.liveSpec() == expected
            log.write("config", runner.configFields().merging(["matches_spec": matches]) { $1 })
            return matches ? 0 : 1
        }
        runner.run(seconds: options.seconds)
        return 0
    }

    static func parse(_ arguments: [String]) -> Result<Options, ArgumentError> {
        var options = Options()
        var index = 0
        while index < arguments.count {
            switch arguments[index] {
            case "--panel-spike": break
            case "--check-config": options.checkConfig = true
            case "--seconds":
                index += 1
                guard index < arguments.count, let value = Int(arguments[index]), (10...900).contains(value) else {
                    return .failure(ArgumentError(message: "--seconds takes a number from 10 to 900"))
                }
                options.seconds = value
            case "--level":
                index += 1
                guard index < arguments.count, let level = PanelLevel(rawValue: arguments[index]),
                    level != .unmapped
                else {
                    return .failure(ArgumentError(message: "--level takes statusBar or floating"))
                }
                options.level = level
            case "--log":
                index += 1
                guard index < arguments.count, !arguments[index].isEmpty else {
                    return .failure(ArgumentError(message: "--log needs a path"))
                }
                options.logPath = arguments[index]
            default: return .failure(ArgumentError(message: "unknown argument \(arguments[index])"))
            }
            index += 1
        }
        return .success(options)
    }
}

/// JSON lines to a file, or to stdout when no path is given.
@MainActor
final class SpikeLog {
    private let handle: FileHandle

    init(path: String?) throws {
        guard let path else {
            handle = .standardOutput
            return
        }
        let url = URL(filePath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
    }

    func write(_ event: String, _ fields: [String: Any] = [:]) {
        var line = fields
        line["event"] = event
        line["t"] = ISO8601DateFormatter().string(from: Date())
        guard let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) else { return }
        handle.write(data + Data("\n".utf8))
    }
}

/// Builds the panel once, then orders it out and in.
@MainActor
final class SpikeRunner {
    let panel: AtlasPanel
    private let host: FirstClickHost<SpikeView>
    private let spec: PanelWindowSpec
    private let log: SpikeLog
    private var scrollMonitor: Any?
    private var lastScrollLog = Date.distantPast

    init(spec: PanelWindowSpec, log: SpikeLog) {
        self.spec = spec
        self.log = log
        panel = AtlasPanel(spec: spec)
        // The closures capture the log, not self, which does not exist yet.
        host = FirstClickHost(
            rootView: SpikeView(onClick: { name in
                log.write(
                    "click",
                    [
                        "button": name, "app_active": NSApp.isActive,
                        "frontmost": NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? NSNull(),
                    ])
            }))
        host.onHoverChanged = { entered in log.write("hover", ["entered": entered]) }
        panel.contentView = host
    }

    func configFields() -> [String: Any] {
        let live = panel.liveSpec()
        return [
            "level": live.level.rawValue,
            "collection_behavior": live.collectionBehavior.map(\.rawValue).sorted(),
            "style": live.style.map(\.rawValue).sorted(),
            "is_floating_panel": live.isFloatingPanel,
            "can_become_key": live.canBecomeKey,
            "can_become_main": live.canBecomeMain,
            "becomes_key_only_if_needed": live.becomesKeyOnlyIfNeeded,
            "hides_on_deactivate": live.hidesOnDeactivate,
            "is_movable": live.isMovable,
            "is_movable_by_window_background": live.isMovableByWindowBackground,
            "ignores_mouse_events": live.ignoresMouseEvents,
            "is_opaque": live.isOpaque,
            "has_shadow": live.hasShadow,
        ]
    }

    func run(seconds: Int) {
        log.write("config", configFields())
        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] event in
            MainActor.assumeIsolated { self?.noteScroll(event) }
            return event
        }
        show()
        Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.show() }
        }
        Timer.scheduledTimer(withTimeInterval: TimeInterval(seconds), repeats: false) { _ in
            MainActor.assumeIsolated {
                NSApp.stop(nil)
                // stop(_:) takes effect after one more event.
                if let wake = NSEvent.otherEvent(
                    with: .applicationDefined, location: .zero, modifierFlags: [], timestamp: 0,
                    windowNumber: 0, context: nil, subtype: 0, data1: 0, data2: 0)
                {
                    NSApp.postEvent(wake, atStart: true)
                }
            }
        }
        NSApp.run()
        if let scrollMonitor { NSEvent.removeMonitor(scrollMonitor) }
        log.write("done")
    }

    /// Order out and in again, top-right of the screen that holds the pointer.
    private func show() {
        panel.orderOut(nil)
        let mouse = NSEvent.mouseLocation
        if let visible = (NSScreen.screens.first { $0.frame.contains(mouse) } ?? NSScreen.screens.first)?.visibleFrame {
            // The frame goes in after the content: a hosting view resizes the panel.
            let height = host.fittingSize.height
            panel.setFrame(
                NSRect(x: visible.maxX - 16 - 360, y: visible.maxY - 16 - height, width: 360, height: height),
                display: true)
        }
        panel.orderFrontRegardless()
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(300))
            log.write(
                "shown",
                [
                    "level": spec.level.rawValue, "cg_layer": cgLayer() ?? NSNull(),
                    "is_on_active_space": panel.isOnActiveSpace, "app_active": NSApp.isActive,
                    "frontmost": NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? NSNull(),
                    "panel_is_key": panel.isKeyWindow,
                ])
        }
    }

    private func cgLayer() -> Int? {
        let info = CGWindowListCopyWindowInfo([.optionIncludingWindow], CGWindowID(panel.windowNumber))
        return (info as? [[String: Any]])?.first?[kCGWindowLayer as String] as? Int
    }

    /// One scroll line per quarter second, so a long scroll does not flood the log.
    private func noteScroll(_ event: NSEvent) {
        guard Date().timeIntervalSince(lastScrollLog) > 0.25 else { return }
        lastScrollLog = Date()
        log.write("scroll", ["delta_y": event.scrollingDeltaY, "over_panel": event.window === panel])
    }
}

private let sampleText = String(
    repeating: "The panel shows a fixed sentence so a person can scroll it and watch it appear over another app. ",
    count: 25)

/// The spike's own content: a header, a scrollable block, Stop and Close.
struct SpikeView: View {
    let onClick: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(verbatim: "ATLAS panel spike").font(.headline)
            ScrollView {
                Text(verbatim: sampleText).frame(width: 328, alignment: .leading)
            }
            .frame(height: 240)
            HStack {
                Button(action: { onClick("stop") }, label: { Text(verbatim: "Stop") })
                Button(action: { onClick("close") }, label: { Text(verbatim: "Close") })
            }
        }
        .padding(16)
        .frame(width: 360, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(.regularMaterial))
    }
}
