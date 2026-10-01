import AtlasKit
import SwiftUI

/// The setup window (D-18): a header, the five step rows and the Continue
/// footer. Every state and string comes from `SetupProgress`.
///
/// SwiftUI semantic styles and system colors only. No alert, no warning color.
struct SetupView: View {
    let model: AppModel

    var body: some View {
        let progress = model.setupProgress
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
                    ForEach(progress.steps) { step in
                        row(step)
                    }
                }
                .formStyle(.grouped)
                footer(progress)
            }
        }
        .padding(16)
        .frame(width: 520)
    }

    // MARK: Rows

    private func symbol(_ status: SetupStepStatus) -> (name: String, color: Color) {
        switch status {
        case .done: ("checkmark.circle.fill", .green)
        case .notDone: ("circle", .secondary)
        case .needsAction: ("exclamationmark.circle", .secondary)
        }
    }

    @ViewBuilder
    private func row(_ step: SetupStep) -> some View {
        let mark = symbol(step.status)
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: mark.name)
                    .foregroundStyle(mark.color)
                    .font(.title3)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text(step.title)
                        .font(.title3.weight(.semibold))
                    Text(step.stateText)
                        .font(.body)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    ForEach(step.notes, id: \.self) { note in
                        Text(note)
                            .font(.subheadline)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Text(step.description)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 8)
                control(step)
            }
            if step.id == .pair, let text = pairErrorText {
                Text(text)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(step.accessibilityLabel)
    }

    @ViewBuilder
    private func control(_ step: SetupStep) -> some View {
        switch step.id {
        case .pair:
            stepButton(step) { model.openPairForm() }
        case .accessibility:
            stepButton(step) { model.permissions.requestAccessibility() }
        case .localNetwork:
            stepButton(step) { SettingsLinks.open(.localNetwork) }
        case .location:
            EmptyView()
        case .launchAtLogin:
            VStack(alignment: .trailing, spacing: 8) {
                Toggle("Launch at login", isOn: Binding(get: { model.launchAtLogin }, set: { model.launchAtLogin = $0 }))
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .tint(.accentColor)
                if step.buttonLabel != nil {
                    stepButton(step) { model.openLoginItems() }
                }
            }
        }
    }

    @ViewBuilder
    private func stepButton(_ step: SetupStep, action: @escaping () -> Void) -> some View {
        if let label = step.buttonLabel {
            Button(label, action: action)
                .buttonStyle(.bordered)
                .controlSize(.regular)
        }
    }

    // MARK: Footer

    private func footer(_ progress: SetupProgress) -> some View {
        VStack(alignment: .trailing, spacing: 4) {
            Button("Continue") { model.continueSetup() }
                .buttonStyle(.borderedProminent)
                .controlSize(.regular)
                .keyboardShortcut(.defaultAction)
                .disabled(!progress.continueEnabled)
            if !progress.continueEnabled {
                Text("Finish the three required steps to continue.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    // MARK: Pair errors

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
}
