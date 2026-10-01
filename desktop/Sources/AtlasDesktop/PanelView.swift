import AppKit
import AtlasKit
import SwiftUI

/// One turn on the panel (UI-SPEC "Panel Anatomy"): a header with the state and
/// Close, then a scrolling content area with the live transcript, the hint, the
/// text card and the outcome line.
///
/// Every string is `Text(verbatim:)`, so a remote string is never read as
/// Markdown or as a link (D-11). `HostileTextGateTests` enforces it. While a
/// timer rings, `TimerRingView` takes the place of the turn content.
struct PanelView: View {
    let state: PanelState
    let onClose: () -> Void
    /// The ring view's Stop button. The controller turns it into `.stopClicked`,
    /// and the reducer sends `timer.stop` once (D-15).
    let onStop: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private static let cardAnchor = "card"
    private static let bottomAnchor = "bottom"

    /// The cap for the whole panel, less the header, the gap and the padding.
    private static let contentCap =
        PanelLayout.heightCap - 2 * PanelLayout.padding - PanelLayout.headerHeight - PanelLayout.sectionGap

    private var turn: PanelTurn? { state.turn }
    private var word: PanelStateWord { turn?.word ?? .listening }
    private var hasCard: Bool { !(turn?.cardText ?? "").isEmpty }
    private var transcript: String { turn?.transcript ?? "" }

    var body: some View {
        VStack(alignment: .leading, spacing: PanelLayout.sectionGap) {
            if let ring = state.ring {
                // A ring has priority over the turn (D-16). The turn keeps
                // updating in the state, so it returns when the ring ends.
                TimerRingView(ring: ring, onStop: onStop, onClose: onClose)
            } else {
                header
                content
            }
        }
        .padding(PanelLayout.padding)
        .frame(width: PanelLayout.width, alignment: .topLeading)
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color(nsColor: .separatorColor), lineWidth: 0.5)
                .allowsHitTesting(false)
        )
        .accessibilityElement(children: .contain)
        .accessibilityLabel(PanelCopy.panelLabel)
    }

    // MARK: Header

    private var header: some View {
        PanelHeader(onClose: onClose) {
            Image(systemName: PanelCopy.stateSymbol(word))
                .font(.system(size: 13, weight: .regular))
                .foregroundStyle(word == .done ? Color.green : Color.accentColor)
                .frame(width: 16, height: 16)
                .symbolEffect(.pulse, isActive: word != .done && !reduceMotion)
                .accessibilityHidden(true)
            Text(verbatim: PanelCopy.stateWord(word))
                .font(.subheadline.weight(.semibold))
        }
    }

    // MARK: Content

    private var content: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    turnContent(turn)
                    Color.clear.frame(height: 0).id(Self.bottomAnchor)
                }
                .frame(width: PanelLayout.contentWidth, alignment: .leading)
            }
            .frame(width: PanelLayout.contentWidth)
            .frame(maxHeight: Self.contentCap)
            .fixedSize(horizontal: false, vertical: true)
            .onChange(of: transcript) {
                if !hasCard && word == .listening { proxy.scrollTo(Self.bottomAnchor, anchor: .bottom) }
            }
            .onChange(of: turn?.cardId) {
                if hasCard { proxy.scrollTo(Self.cardAnchor, anchor: .top) }
            }
            .onChange(of: turn?.cardText) {
                if hasCard { proxy.scrollTo(Self.cardAnchor, anchor: .top) }
            }
        }
    }

    @ViewBuilder
    private func turnContent(_ turn: PanelTurn?) -> some View {
        if !transcript.isEmpty {
            Text(verbatim: transcript)
                .font(.title3)
                .foregroundStyle(hasCard || !(turn?.transcriptIsFinal ?? false) ? Color.secondary : Color.primary)
                .lineLimit(hasCard ? 2 : nil)
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: PanelLayout.contentWidth, alignment: .leading)
        } else if !hasCard && turn?.outcome == nil {
            Text(verbatim: PanelCopy.emptyHint)
                .font(.body)
                .foregroundStyle(.secondary)
                .frame(width: PanelLayout.contentWidth, alignment: .leading)
        }
        if let card = turn?.cardText, !card.isEmpty {
            Color.clear.frame(height: PanelLayout.sectionGap)
            Rectangle()
                .fill(Color(nsColor: .separatorColor))
                .frame(height: 1)
                .accessibilityHidden(true)
            Color.clear.frame(height: PanelLayout.sectionGap)
            Text(verbatim: card)
                .font(.title3)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: PanelLayout.contentWidth, alignment: .leading)
                .id(Self.cardAnchor)
        } else if let outcome = turn?.outcome {
            Text(verbatim: PanelCopy.outcomeLine(outcome))
                .font(.body)
                .foregroundStyle(.secondary)
                .frame(width: PanelLayout.contentWidth, alignment: .leading)
        }
    }
}

/// The header row of the panel: the leading content of the view (symbol and
/// word), a spacer and the Close button. The turn view and the ring view share
/// it, so Close has one implementation.
struct PanelHeader<Leading: View>: View {
    let onClose: () -> Void
    @ViewBuilder let leading: () -> Leading

    var body: some View {
        HStack(spacing: PanelLayout.symbolGap) {
            leading()
            Spacer(minLength: 0)
            Button(action: onClose) {
                Image(systemName: PanelCopy.closeSymbol)
                    .font(.system(size: 11, weight: .regular))
                    .foregroundStyle(.secondary)
                    .frame(width: PanelLayout.closeHitSize, height: PanelLayout.closeHitSize)
                    .contentShape(Rectangle())
            }
            .buttonStyle(PanelCloseButtonStyle())
            .accessibilityLabel(PanelCopy.closeLabel)
            .accessibilityAddTraits(.isButton)
        }
        .frame(height: PanelLayout.headerHeight)
        .accessibilityElement(children: .contain)
    }
}

/// The Close button: a 28 pt square with the `quaternarySystemFill` background
/// while it is pressed. SwiftUI `onHover` is not used in the panel (the panel is
/// never key, so it can miss events; UI-SPEC "Rules for code"), so the
/// background shows on press and not on hover.
private struct PanelCloseButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(configuration.isPressed ? Color(nsColor: .quaternarySystemFill) : Color.clear)
            )
    }
}
