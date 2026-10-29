import SwiftUI

/// Builds a per-token attributed string for one stream; tokens are links ns://<index>.
func nsStreamAttributed(_ run: Run, stream: StreamKind, highlight: Int?, base: Color, italic: Bool) -> AttributedString {
    var out = AttributedString()
    for tok in run.tokens where tok.stream == stream {
        var piece = AttributedString(tok.text.isEmpty ? " " : tok.text)
        piece.foregroundColor = base
        piece.font = italic ? .system(size: 13).italic() : .system(size: 13)
        piece.link = URL(string: "ns://\(tok.index)")
        if tok.index == highlight {
            piece.backgroundColor = Theme.positive.opacity(0.45)
        }
        out.append(piece)
    }
    return out
}

private final class StreamMemo {
    var keys: [String: String] = [:]
    var strings: [String: AttributedString] = [:]
    func get(_ key: String, _ build: () -> AttributedString) -> AttributedString {
        if let s = strings[key] { return s }
        if strings.count > 16 { strings.removeAll() }
        let v = build(); strings[key] = v; return v
    }
}

struct StreamsView: View {
    @Environment(AppState.self) private var state
    @State private var memo = StreamMemo()

    var body: some View {
        if let run = state.activeRun {
            let hi = state.displayIndex
            HStack(spacing: 1) {
                if state.hasThinking {
                    pane("Thinking", run: run, stream: .thinking, color: Theme.thinking.opacity(0.8), italic: true, hi: hi)
                }
                pane("Answer", run: run, stream: .answer, color: Theme.answer, italic: false, hi: hi)
            }
            .background(Theme.background)
        } else {
            Text("No run yet").foregroundStyle(.secondary).frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func pane(_ title: String, run: Run, stream: StreamKind, color: Color, italic: Bool, hi: Int?) -> some View {
        let text = memo.get("\(run.id)|\(run.count)|\(title)|\(hi ?? -1)") {
            nsStreamAttributed(run, stream: stream, highlight: hi, base: color, italic: italic)
        }
        let live = state.cursor == nil
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).font(.caption.bold()).foregroundStyle(color)
                if run.status == .running {
                    Text("generating...").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }
            ScrollViewReader { proxy in
                ScrollView {
                    Text(text)
                        .textSelection(.disabled)
                        .tint(color)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Color.clear.frame(height: 1).id("end")
                }
                .onChange(of: run.count) { _, _ in
                    if live { proxy.scrollTo("end", anchor: .bottom) }
                }
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.panel)
        .environment(\.openURL, OpenURLAction { url in
            if url.scheme == "ns", let h = url.host, let i = Int(h) { state.setCursor(i); return .handled }
            return .discarded
        })
    }
}
