import AppKit
import AtlasKit
import CoreGraphics
import Foundation
import OSLog
import Observation
import SwiftUI

/// The root of the hosting view. It reads the controller, so SwiftUI redraws on
/// each state change. `PanelView` itself takes plain values.
struct PanelRoot: View {
    let controller: PanelController

    var body: some View {
        PanelView(
            state: controller.shown,
            onClose: { controller.dispatch(.closePressed) },
            onStop: { controller.dispatch(.stopClicked) })
    }
}

/// Runs the panel: it builds the window once, feeds events to `PanelReducer`,
/// and does what each `PanelEffect` says.
///
/// The panel never takes focus (D-03). The only show call is
/// `orderFrontRegardless()`. This file never activates the app, never makes the
/// panel key and never reads a main-screen property (UI-SPEC "Rules for code";
/// `HostileTextGateTests` checks it). It logs the words "shown" and "hidden"
/// and never a transcript or a card.
@MainActor
@Observable
final class PanelController {
    /// The reducer's state.
    private(set) var state = PanelState()
    /// What the view draws. It follows `state`, except during a fade-out: the
    /// reducer has already cleared the turn then, and a panel that fades must
    /// not flash an empty "Listening" state on the way out.
    private(set) var shown = PanelState()

    @ObservationIgnored var onSendTimerStop: ((Int) -> Void)?
    @ObservationIgnored private var panel: AtlasPanel?
    @ObservationIgnored private var host: FirstClickHost<PanelRoot>?
    @ObservationIgnored private var wakeTask: Task<Void, Never>?
    @ObservationIgnored private var screenTask: Task<Void, Never>?
    /// The display the panel opened on, as plain rectangles, and its number.
    @ObservationIgnored private var remembered: PanelScreen?
    @ObservationIgnored private var rememberedNumber: NSNumber?
    /// Counts shows, so a fade-out that a later show overtook does not order the
    /// panel out.
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private let log = Logger(subsystem: AppIdentity.bundleIdentifier, category: "panel")

    private static let screenNumberKey = NSDeviceDescriptionKey("NSScreenNumber")

    // MARK: - Build once

    /// Builds the panel and its hosting view, then orders the panel out. A second
    /// call does nothing. Done at launch, so a show never waits for a view tree
    /// (UI-SPEC E1 loading).
    func prepare() {
        guard panel == nil else { return }
        let panel = AtlasPanel()

        let backdrop = NSVisualEffectView()
        backdrop.material = .popover
        backdrop.blendingMode = .behindWindow
        backdrop.state = .active
        backdrop.wantsLayer = true
        backdrop.layer?.cornerRadius = 12
        backdrop.layer?.cornerCurve = .continuous
        backdrop.layer?.masksToBounds = true

        let host = FirstClickHost(rootView: PanelRoot(controller: self))
        // The window sizes the content, not the other way round, so the hosting
        // view reports an intrinsic size only and adds no min or max constraint.
        host.sizingOptions = [.intrinsicContentSize]
        host.frame = backdrop.bounds
        host.autoresizingMask = [.width, .height]
        host.onHoverChanged = { [weak self] inside in self?.dispatch(.hoverChanged(inside)) }
        backdrop.addSubview(host)
        panel.contentView = backdrop
        panel.orderOut(nil)

        self.panel = panel
        self.host = host
        screenTask = Task { [weak self] in
            for await _ in NotificationCenter.default.notifications(
                named: NSApplication.didChangeScreenParametersNotification)
            {
                self?.screensChanged()
            }
        }
    }

    // MARK: - Events and effects

    func dispatch(_ event: PanelEvent) {
        let effects = PanelReducer.reduce(&state, event, now: Date())
        let hiding = effects.contains { if case .hide = $0 { true } else { false } }
        if !hiding { shown = state }
        for effect in effects { run(effect) }
    }

    private func run(_ effect: PanelEffect) {
        switch effect {
        case .show: show()
        case .hide(let animated): hide(animated: animated)
        case .relayout: relayout()
        case .announce(let text, let priority): announce(text, priority)
        case .sendTimerStop(let timerId): onSendTimerStop?(timerId)
        case .wakeAt(let date): wake(at: date)
        }
    }

    // MARK: - Show and hide

    private func show() {
        guard let panel, let host else { return }
        let screens = NSScreen.screens
        let plain = screens.map { PanelScreen(frame: $0.frame, visibleFrame: $0.visibleFrame) }
        guard
            let index = PanelFrame.chooseScreen(
                screens: plain, pointer: NSEvent.mouseLocation, fallbackWindowFrame: frontmostWindowFrame())
        else { return }
        remembered = plain[index]
        rememberedNumber = screens[index].deviceDescription[Self.screenNumberKey] as? NSNumber

        // The content is set first, then the frame follows its height (RESEARCH Pitfall 4).
        host.layoutSubtreeIfNeeded()
        panel.setFrame(PanelFrame.frame(in: plain[index], contentHeight: host.fittingSize.height), display: false)
        generation += 1
        let reduce = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        setAlpha(0, duration: 0, timing: .linear)
        panel.orderFrontRegardless()
        setAlpha(1, duration: reduce ? 0 : PanelTiming.fadeInS, timing: .easeOut)
        log.info("The panel is shown.")
        // SwiftUI applies the state at the end of this turn. Measure again then.
        relayout()
    }

    private func hide(animated: Bool) {
        guard let panel, panel.isVisible else { return }
        let reduce = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        let token = generation
        log.info("The panel is hidden.")
        if reduce || !animated {
            setAlpha(0, duration: 0, timing: .linear)
            panel.orderOut(nil)
            shown = state
            return
        }
        // AppKit calls the completion on the main thread.
        setAlpha(0, duration: PanelTiming.fadeOutS, timing: .easeIn) { [weak self] in
            MainActor.assumeIsolated {
                guard let self, token == self.generation else { return }
                self.panel?.orderOut(nil)
                self.shown = self.state
            }
        }
    }

    /// Sets the alpha through an animation group, so that it also replaces an
    /// animation that is still running.
    private func setAlpha(
        _ value: CGFloat, duration: TimeInterval, timing: CAMediaTimingFunctionName,
        completion: (@Sendable () -> Void)? = nil
    ) {
        guard let panel else { return }
        NSAnimationContext.runAnimationGroup(
            { context in
                context.duration = duration
                context.timingFunction = CAMediaTimingFunction(name: timing)
                panel.animator().alphaValue = value
            }, completionHandler: completion)
    }

    private func relayout() {
        Task { @MainActor [weak self] in self?.applyHeight() }
    }

    /// The height follows the content. The top-right corner stays fixed (D-02).
    private func applyHeight() {
        guard let panel, let host, panel.isVisible, state.visible else { return }
        host.layoutSubtreeIfNeeded()
        let next = PanelFrame.resized(panel.frame, toContentHeight: host.fittingSize.height)
        guard next != panel.frame else { return }
        panel.setFrame(next, display: true)
        panel.invalidateShadow()
    }

    // MARK: - Announce, timers, displays

    private func announce(_ text: String, _ priority: AnnouncePriority) {
        let level: NSAccessibilityPriorityLevel = priority == .high ? .high : .medium
        NSAccessibility.post(
            element: NSApp as Any, notification: .announcementRequested,
            userInfo: [.announcement: text, .priority: level.rawValue])
    }

    private func wake(at date: Date?) {
        wakeTask?.cancel()
        wakeTask = nil
        guard let date else { return }
        wakeTask = Task { [weak self] in
            let delay = date.timeIntervalSinceNow
            if delay > 0 { try? await Task.sleep(for: .seconds(delay)) }
            guard !Task.isCancelled else { return }
            self?.dispatch(.deadline)
        }
    }

    /// A display was added, removed or changed. A display that is gone hides the
    /// panel; the next event shows it on the display with the pointer. A display
    /// that only changed size gets the frame again on its new geometry.
    private func screensChanged() {
        guard state.visible, let panel, let host, let number = rememberedNumber else { return }
        guard
            let live = NSScreen.screens.first(where: {
                ($0.deviceDescription[Self.screenNumberKey] as? NSNumber) == number
            })
        else {
            dispatch(.displayRemoved)
            return
        }
        let plain = PanelScreen(frame: live.frame, visibleFrame: live.visibleFrame)
        remembered = plain
        host.layoutSubtreeIfNeeded()
        panel.setFrame(PanelFrame.frame(in: plain, contentHeight: host.fittingSize.height), display: true)
        panel.invalidateShadow()
    }

    /// The frame of the frontmost app's frontmost window, in AppKit coordinates.
    /// It is the fallback when the pointer is on no display (UI-SPEC "Placement").
    private func frontmostWindowFrame() -> CGRect? {
        guard
            let pid = NSWorkspace.shared.frontmostApplication?.processIdentifier,
            let primary = NSScreen.screens.first,
            let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID)
                as? [[String: Any]]
        else { return nil }
        for window in list {
            guard
                (window[kCGWindowOwnerPID as String] as? pid_t) == pid,
                (window[kCGWindowLayer as String] as? Int) == 0,
                let bounds = window[kCGWindowBounds as String] as? NSDictionary,
                let rect = CGRect(dictionaryRepresentation: bounds)
            else { continue }
            // CG puts the origin at the top-left of the primary display, AppKit at the bottom-left.
            return CGRect(x: rect.minX, y: primary.frame.height - rect.maxY, width: rect.width, height: rect.height)
        }
        return nil
    }
}
