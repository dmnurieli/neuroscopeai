import SwiftUI
import AppKit

struct EditsView: View {
    @Environment(AppState.self) private var state

    private func badge(_ mode: EditMode, _ v: Float) -> (String, Color) {
        switch mode {
        case .ablate: return ("Delete", Theme.ablated)
        case .scale: return ("x\(String(format: "%g", v))", Theme.scaled)
        case .pin: return ("=\(String(format: "%g", v))", Theme.pinned)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Edits (\(state.edits.count))").font(.headline)
            if state.edits.isEmpty {
                Text("Select a neuron in the field, then delete, scale or pin it, or build a direction under Custom directions. Replay to see how the \(state.hasThinking ? "thinking and answer change" : "answer changes").")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            } else {
                ScrollView {
                    VStack(spacing: 4) {
                        ForEach(state.edits.neurons, id: \.id) { e in
                            row("L\(e.layer) · N\(e.index)", e.mode, e.value, .neuron(e.id))
                        }
                        ForEach(state.edits.heads, id: \.id) { e in
                            row("L\(e.layer) · H\(e.head)", e.mode, e.value, .head(e.id))
                        }
                        ForEach(state.edits.directions) { d in dirRow(d) }
                    }
                }
            }
            HStack {
                Button("Clear all") { state.clearEdits() }.disabled(state.edits.isEmpty)
                Button("Save...") { save() }.disabled(state.edits.isEmpty)
                Button("Load...") { load() }
            }
            .controlSize(.small)
            Button { state.replayWithEdits() } label: {
                Text("Replay with edits").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(state.edits.isEmpty || state.isGenerating)
        }
        .padding(10)
    }

    private func row(_ title: String, _ mode: EditMode, _ value: Float, _ sel: Selection) -> some View {
        let (txt, col) = badge(mode, value)
        return HStack {
            Text(title).font(.system(.callout, design: .monospaced))
            Spacer()
            Text(txt).font(.caption.bold()).padding(.horizontal, 6).padding(.vertical, 2)
                .background(Capsule().fill(col.opacity(0.3)))
                .overlay(Capsule().stroke(col, lineWidth: 1))
            Button { state.removeEdit(sel) } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.white.opacity(0.05)))
        .contentShape(Rectangle())
        .onTapGesture { state.select(sel) }
    }

    private func dirRow(_ d: DirectionEdit) -> some View {
        let col = d.mode == .ablate ? Theme.ablated : Theme.scaled
        let txt = d.mode == .ablate ? "Remove" : "Add \(String(format: "%+.2f", d.value))"
        return HStack {
            Text(d.label).font(.system(.callout, design: .monospaced)).lineLimit(1)
            Spacer()
            Text(txt).font(.caption.bold()).padding(.horizontal, 6).padding(.vertical, 2)
                .background(Capsule().fill(col.opacity(0.3)))
                .overlay(Capsule().stroke(col, lineWidth: 1))
            Button { state.removeDirectionEdit(d.id) } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.white.opacity(0.05)))
    }

    private func alert(_ msg: String) {
        let a = NSAlert(); a.messageText = "Edits error"; a.informativeText = msg; a.runModal()
    }

    private func save() {
        let p = NSSavePanel()
        p.allowedContentTypes = [.json]
        p.nameFieldStringValue = "edits.json"
        guard p.runModal() == .OK, let url = p.url else { return }
        do { try state.saveEdits(to: url) } catch { alert(error.localizedDescription) }
    }

    private func load() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.json]
        p.allowsMultipleSelection = false
        guard p.runModal() == .OK, let url = p.url else { return }
        do { try state.loadEdits(from: url) } catch { alert(error.localizedDescription) }
    }
}
