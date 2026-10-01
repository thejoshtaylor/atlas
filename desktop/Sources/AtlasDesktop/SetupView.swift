import AtlasKit
import SwiftUI

/// The setup window (D-18): a header and the step list. This plan builds the
/// Pair step. Plan 14-11 adds Accessibility, Local Network, Location, Launch at
/// login and the footer.
///
/// SwiftUI semantic styles and system colors only. No alert, no warning color.
struct SetupView: View {
    let model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Set up ATLAS")
                    .font(.title2.weight(.semibold))
                Text("Pair this Mac, then grant two permissions. Location is optional.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            if model.showingPairForm {
                PairFormView(model: model)
            } else {
                Form {
                    pairRow
                }
                .formStyle(.grouped)
            }
        }
        .padding(16)
        .frame(width: 520)
    }

    // MARK: Pair row

    private enum PairRowState {
        case notPaired, connecting(String), paired(String), revoked

        var done: Bool {
            if case .paired = self { return true }
            return false
        }
    }

    private var pairState: PairRowState {
        if case .connecting(let host, _) = model.flow.state { return .connecting(host) }
        if case .saving(let target) = model.flow.state { return .connecting(target.host) }
        if let host = model.pairedHost { return .paired(host) }
        if model.revokedHost != nil { return .revoked }
        return .notPaired
    }

    private var pairStateText: String {
        switch pairState {
        case .notPaired: "Not paired"
        case .connecting(let host): "Connecting to " + MenuState.truncateMiddle(host) + "\u{2026}"
        case .paired(let host): "Paired with " + MenuState.truncateMiddle(host)
        case .revoked: "Revoked by the server"
        }
    }

    private var pairErrorText: String? {
        switch model.flow.state {
        case .failed(let failure):
            return failure.message(host: failureHost)
        case .connecting(_, let error?):
            return error.message(host: failureHost)
        default:
            return nil
        }
    }

    private var failureHost: String {
        MenuState.truncateMiddle(model.pairedHost ?? connectingHost ?? "the server")
    }

    private var connectingHost: String? {
        if case .connecting(let host, _) = model.flow.state { return host }
        return nil
    }

    private var pairRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: pairState.done ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(pairState.done ? Color.green : Color.secondary)
                    .font(.title3)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Pair")
                        .font(.title3.weight(.semibold))
                    Text(pairStateText)
                        .font(.body)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text("Connect this Mac to your ATLAS server.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                Button(pairState.done ? "Re-pair\u{2026}" : "Pair\u{2026}") {
                    model.openPairForm()
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)
            }
            if let text = pairErrorText {
                Text(text)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Pair, " + pairStateText)
    }
}
