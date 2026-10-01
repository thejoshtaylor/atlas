import AtlasKit
import SwiftUI

/// The Pair form, in place of the step list. A pair link or a typed server and
/// token gets a confirm sheet that names the host before anything is saved (D-03).
///
/// Errors show inline in `.secondary`. There is no alert and no warning color.
struct PairFormView: View {
    let model: AppModel

    @State private var link = ""
    @State private var server = ""
    @State private var token = ""
    @State private var showManual = false

    private var canSubmit: Bool {
        PairingFlow.canSubmit(link: link, server: server, token: token)
    }

    private var errorText: String? {
        if case .failed(let failure) = model.flow.state {
            return failure.message(host: MenuState.truncateMiddle(model.pairedHost ?? "the server"))
        }
        return nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Button("Back") { model.closePairForm() }
                    .buttonStyle(.borderless)
                Spacer()
            }
            Form {
                TextField("Pair link", text: $link, prompt: Text("atlas://pair?server=\u{2026}"))
                    .lineLimit(1)
                    .accessibilityLabel("Pair link")
                DisclosureGroup("Enter a server and token instead", isExpanded: $showManual) {
                    TextField("Server", text: $server)
                        .accessibilityLabel("Server")
                    SecureField("Token", text: $token)
                        .accessibilityLabel("Token")
                }
                if let text = errorText {
                    Text(text)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .formStyle(.grouped)
            HStack {
                Spacer()
                Button("Pair") {
                    model.submit(link: link, server: server, token: token)
                    token = ""
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!canSubmit)
            }
        }
        .sheet(isPresented: sheetShown) {
            sheetContent
        }
    }

    // MARK: Sheets

    private var sheetShown: Binding<Bool> {
        Binding(
            get: {
                switch model.flow.state {
                case .confirm, .replace: true
                default: false
                }
            },
            set: { shown in
                // Escape closes the sheet. Pair and Replace move the flow on first.
                if !shown { model.cancel() }
            })
    }

    @ViewBuilder private var sheetContent: some View {
        switch model.flow.state {
        case .confirm(let target):
            ConfirmSheet(
                title: "Pair with \(MenuState.truncateMiddle(target.host))?",
                message:
                    "This Mac will connect to \(MenuState.truncateMiddle(target.host)) and show ATLAS answers. Pair only with a server you run.",
                secondaryLabel: "Cancel", secondary: { model.cancel() },
                actionLabel: "Pair", action: { model.confirm() })
        case .replace(let current, let target):
            // Keep Current is the default so a stray atlas:// click cannot swap the token.
            ConfirmSheet(
                title: "Replace the current pairing?",
                message:
                    "This Mac is paired with \(MenuState.truncateMiddle(current)). Pairing with \(MenuState.truncateMiddle(target.host)) replaces it.",
                secondaryLabel: "Replace", secondary: { model.replace() },
                actionLabel: "Keep Current", action: { model.keepCurrent() })
        default:
            EmptyView()
        }
    }
}

/// A sheet with a title, a message and two buttons. `action` is the default
/// button (Return). Escape closes the sheet and the form cancels the flow.
private struct ConfirmSheet: View {
    let title: String
    let message: String
    let secondaryLabel: String
    let secondary: () -> Void
    let actionLabel: String
    let action: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title)
                .font(.title3.weight(.semibold))
                .lineLimit(1)
                .truncationMode(.middle)
            Text(message)
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button(secondaryLabel, action: secondary)
                Button(actionLabel, action: action)
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(16)
        .frame(width: 420)
    }
}
