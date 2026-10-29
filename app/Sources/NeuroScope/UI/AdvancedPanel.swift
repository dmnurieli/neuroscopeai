import SwiftUI

/// Every expert control, behind one address (View > Show Advanced, ⌥⌘A). Technical labels are fine here.
struct AdvancedPanel: View {
    @Environment(AppState.self) private var state
    @State private var open: Set<String> = ["Generation", "View"]

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Advanced").font(.headline)
                Spacer()
                Button { state.showAdvanced = false } label: { Image(systemName: "xmark") }
                    .buttonStyle(.borderless).foregroundStyle(.secondary).help("Hide Advanced (⌥⌘A)")
            }
            .padding(.horizontal, 14).padding(.vertical, 10)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    group("Generation") { PromptPanel() }
                    group("View") { FieldToolbar() }
                    group("Runs") { RunsList().frame(height: 200) }
                    group("Edits") { EditsView().frame(minHeight: 140, maxHeight: 320) }
                    group("Custom directions (contrast / abliteration)") { DirectionsView() }
                    group("Run comparison") { CompareView().frame(height: 300) }
                    group("Logit lens (token at cursor)") { LogitLensTable() }
                }
                .padding(.horizontal, 14).padding(.vertical, 10)
            }
        }
        .background(Theme.panel)
    }

    private func group<C: View>(_ title: String, @ViewBuilder _ content: () -> C) -> some View {
        let binding = Binding<Bool>(
            get: { open.contains(title) },
            set: { if $0 { open.insert(title) } else { open.remove(title) } })
        let body = content()
        return DisclosureGroup(isExpanded: binding) {
            body.padding(.top, 6).padding(.bottom, 10)
        } label: {
            Text(title).font(.callout.weight(.semibold))
        }
        .padding(.vertical, 4)
    }
}
