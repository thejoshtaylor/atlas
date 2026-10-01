import AtlasKit
import SwiftUI

/// The ring view (UI-SPEC "Timer ring view", CARD-02, CARD-03): the bell and
/// the kind word, the label beside a 48 pt ring graphic, and a full-width Stop
/// button. It takes the place of the turn content while a timer rings (D-16).
///
/// The label comes from a spoken sentence, maybe from a television. It is
/// sanitized in the reducer and drawn with `Text(verbatim:)`, on 2 lines at
/// most (T-15-37). `HostileTextGateTests` scans this file.
struct TimerRingView: View {
    let ring: PanelRing
    let onStop: () -> Void
    let onClose: () -> Void

    private var label: String {
        ring.label.isEmpty ? PanelCopy.emptyRingLabel(ring.kind) : ring.label
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            PanelHeader(onClose: onClose) {
                Image(systemName: PanelCopy.bellSymbol)
                    .font(.system(size: 13, weight: .regular))
                    .foregroundStyle(Color.accentColor)
                    .frame(width: 16, height: 16)
                    .accessibilityHidden(true)
                Text(verbatim: PanelCopy.ringKindWord(ring.kind))
                    .font(.subheadline.weight(.semibold))
            }
            Color.clear.frame(height: PanelLayout.sectionGap)
            HStack(alignment: .center, spacing: PanelLayout.sectionGap) {
                Text(verbatim: label)
                    .font(.title2.weight(.semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                    .truncationMode(.tail)
                    .frame(maxWidth: .infinity, alignment: .leading)
                RingGraphic()
                    .frame(width: PanelLayout.ringGraphicSize, height: PanelLayout.ringGraphicSize)
                    .accessibilityHidden(true)
            }
            .frame(width: PanelLayout.contentWidth)
            Color.clear.frame(height: 16)
            stopArea
        }
        .frame(width: PanelLayout.contentWidth, alignment: .leading)
    }

    @ViewBuilder
    private var stopArea: some View {
        switch ring.stop {
        case .idle, .pending:
            Button(action: onStop) {
                Text(verbatim: ring.stop == .pending ? PanelCopy.stopPendingTitle : PanelCopy.stopTitle)
                    .frame(maxWidth: .infinity, minHeight: 32)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(ring.stop == .pending)
            .frame(width: PanelLayout.contentWidth)
            .accessibilityLabel(PanelCopy.stopAccessibilityLabel(label: ring.label, kind: ring.kind))
        case .stopped:
            Text(verbatim: PanelCopy.stoppedLine)
                .font(.body)
                .foregroundStyle(.secondary)
                .frame(width: PanelLayout.contentWidth, alignment: .leading)
        }
    }
}

/// A circle with a 2 pt accent stroke, and a second circle that grows to 1.5
/// times its size while it fades from 60% to 0, every 1.2 s. Under Reduce Motion
/// the second circle does not show and the first stays still.
private struct RingGraphic: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var rippling = false

    var body: some View {
        ZStack {
            Circle()
                .strokeBorder(Color.accentColor, lineWidth: 2)
            if !reduceMotion {
                Circle()
                    .strokeBorder(Color.accentColor, lineWidth: 2)
                    .scaleEffect(rippling ? 1.5 : 1)
                    .opacity(rippling ? 0 : 0.6)
            }
        }
        .onAppear { startRipple() }
        .onChange(of: reduceMotion) { startRipple() }
    }

    private func startRipple() {
        rippling = false
        guard !reduceMotion else { return }
        withAnimation(.easeOut(duration: 1.2).repeatForever(autoreverses: false)) {
            rippling = true
        }
    }
}
