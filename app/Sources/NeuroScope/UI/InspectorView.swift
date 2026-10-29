import SwiftUI

private final class InspectorMemo {
    var key = ""
    var trace: [Float] = []
    var top: [(t: Int, value: Float)] = []
    var scales: (layer: [Float], head: [Float]) = ([], [])
}

private func tokText(_ s: String) -> String {
    s.replacingOccurrences(of: "\n", with: "⏎").replacingOccurrences(of: " ", with: "·")
}

/// The neuron card: plain-language summary of the selected neuron with Turn off / Turn up.
/// "More controls" (collapsed) holds the precise Scale slider, Pin value and head controls.
struct InspectorView: View {
    @Environment(AppState.self) private var state
    @State private var memo = InspectorMemo()
    @State private var info: NeuronInfo?
    @State private var scaleValue: Double = 0.5
    @State private var pinText: String = "0"
    @State private var moreOpen = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                if let run = state.activeRun, let sel = state.selection {
                    switch sel {
                    case .neuron(let n): neuronCard(n, run)
                    case .head(let h): headCard(h, run)
                    }
                } else {
                    Label("Click any dot in the field to look at that desk.", systemImage: "cursorarrow.click")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(Theme.panel)
    }

    // MARK: memo
    private func refresh(_ run: Run) {
        let sel: String
        switch state.selection {
        case .neuron(let n): sel = "n\(n.layer).\(n.index)"
        case .head(let h): sel = "h\(h.layer).\(h.head)"
        case nil: sel = "-"
        }
        let key = "\(run.id)|\(run.count)|\(sel)|\(state.displayIndex ?? -1)"
        guard key != memo.key else { return }
        memo.key = key
        memo.scales = state.scales(for: run)
        switch state.selection {
        case .neuron(let n):
            memo.trace = run.trace(n)
            memo.top = Analysis.topTokens(for: n, in: run, k: 6)
        case .head(let h):
            memo.trace = (0..<run.count).map { run.headNorm(h, at: $0) }
        case nil: break
        }
    }

    private func caption(_ s: String) -> some View {
        Text(markdown(s)).font(.caption).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
    private func chip(_ s: String, _ color: Color, _ action: (() -> Void)? = nil) -> some View {
        Text(tokText(s)).font(.system(.caption, design: .monospaced))
            .padding(.horizontal, 7).padding(.vertical, 3)
            .background(color.opacity(0.18), in: RoundedRectangle(cornerRadius: 5))
            .onTapGesture { action?() }
    }

    private func activityLine(_ v: Float, scale: Float) -> String {
        let r = abs(v) / max(scale, 1e-6)
        if r >= 0.8 { return "Very busy on this word" }
        if r >= 0.4 { return "Fairly busy on this word" }
        if r >= 0.12 { return "A little busy on this word" }
        return "Idle on this word"
    }

    // MARK: neuron
    @ViewBuilder private func neuronCard(_ n: NeuronID, _ run: Run) -> some View {
        let _ = refresh(run)
        Text(verbatim: "Desk #\(n.index) on floor \(n.layer)").font(.headline)
        caption("A desk is one **neuron**; a floor is one **layer**.")
        Button { Task { await state.followDesk(n) } } label: {
            Label(state.isFollowing ? "Following…" : "Follow this desk", systemImage: "point.topleft.down.to.point.bottomright.curvepath")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent).tint(Theme.positive).controlSize(.large)
        .disabled(state.isFollowing || state.isGenerating || run.count == 0)
        .help("Walk through this desk's whole job on the shown word, step by step: what it reads, what it asks, what it writes and who hears it")
        if let t = state.displayIndex, t < run.count {
            let v = run.activation(n, at: t)
            let sc = memo.scales.layer.indices.contains(n.layer) ? memo.scales.layer[n.layer] : 1
            Text(activityLine(v, scale: sc)).font(.callout)
        }
        Sparkline(values: memo.trace, cursor: state.displayIndex, promptCount: run.promptCount) { state.setCursor($0) }
        caption("How busy this desk was on each word-piece. Click to jump there.")

        if !memo.top.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text("Busiest on these words").font(.subheadline.weight(.medium))
                FlowRow {
                    ForEach(Array(memo.top.enumerated()), id: \.offset) { _, e in
                        chip(run.tokens[e.t].text, Theme.positive) { state.setCursor(e.t) }
                    }
                }
            }
        }
        VStack(alignment: .leading, spacing: 6) {
            if let info {
                Text("When this desk is busy, it nudges the reply toward").font(.subheadline.weight(.medium))
                FlowRow { ForEach(Array(info.promotes.prefix(6).enumerated()), id: \.offset) { _, s in chip(s.token, Theme.positive) } }
                Text("...and away from").font(.subheadline.weight(.medium))
                FlowRow { ForEach(Array(info.suppresses.prefix(6).enumerated()), id: \.offset) { _, s in chip(s.token, Theme.negative) } }
            } else {
                caption("Looking up which words this desk pushes for…")
            }
        }
        .task(id: n) {
            info = nil
            info = await state.neuronInfo(n)
        }

        changeBadge(state.edit(for: n).map { ($0.mode, $0.value) }, remove: { state.removeEdit(.neuron(n)) })
        HStack(spacing: 10) {
            Button { state.setEdit(n, mode: .ablate, value: 0) } label: { Label("Turn this desk off", systemImage: "power") }
                .buttonStyle(.borderedProminent).tint(Theme.ablated)
                .help("Silence this desk on every word, then ask again to see what changes")
            Button { state.setEdit(n, mode: .scale, value: 3.0) } label: { Label("Turn it up", systemImage: "arrow.up.circle") }
                .help("Make this desk work 3 times as hard")
                .buttonStyle(.bordered)
        }
        .controlSize(.regular)

        MathDisclosure("What changes in the math") {
            VStack(alignment: .leading, spacing: 6) {
                caption("Turning a desk off or up changes one number, its busyness **h**. The memo then changes by that difference times the desk's own note (its **column**).")
                DeskMath(h: deskH(n, run), edit: state.edit(for: n), outNorm: info?.outNorm)
            }
        }

        DisclosureGroup("More controls", isExpanded: $moreOpen) {
            VStack(alignment: .leading, spacing: 10) {
                scaleRow { state.setEdit(n, mode: .scale, value: Float(scaleValue)) }
                HStack {
                    Text("Hold at value").font(.caption)
                    TextField("value", text: $pinText).textFieldStyle(.roundedBorder).frame(width: 90)
                    Button("Apply") { state.setEdit(n, mode: .pin, value: Float(pinText) ?? 0) }.controlSize(.small)
                }
                if let info { caption(String(format: "Output strength %.3f (how loudly this desk writes into the memo)", info.outNorm)) }
            }.padding(.top, 8)
        }
        .font(.callout)
    }

    private func deskH(_ n: NeuronID, _ run: Run) -> Float? {
        guard let t = state.displayIndex, t < run.count else { return nil }
        return run.activation(n, at: t)
    }

    // MARK: head
    @ViewBuilder private func headCard(_ h: HeadID, _ run: Run) -> some View {
        let _ = refresh(run)
        Text("Attention head \(h.head) on floor \(h.layer)").font(.headline)
        caption("An **attention head** is a reader on this floor who decides which earlier words to look back at.")
        Sparkline(values: memo.trace, cursor: state.displayIndex, promptCount: run.promptCount) { state.setCursor($0) }
        changeBadge(state.edit(for: h).map { ($0.mode, $0.value) }, remove: { state.removeEdit(.head(h)) })
        Button { state.setEdit(h, mode: .ablate, value: 0) } label: { Label("Turn off", systemImage: "power") }
            .buttonStyle(.borderedProminent).tint(Theme.ablated)
        DisclosureGroup("More controls", isExpanded: $moreOpen) {
            scaleRow { state.setEdit(h, mode: .scale, value: Float(scaleValue)) }.padding(.top, 8)
        }
        .font(.callout)
    }

    @ViewBuilder private func scaleRow(apply: @escaping () -> Void) -> some View {
        HStack {
            Text("Scale").font(.caption)
            Slider(value: $scaleValue, in: -3...5)
            Text(String(format: "%.2f", scaleValue)).font(.system(.caption, design: .monospaced)).frame(width: 40)
            Button("Apply", action: apply).controlSize(.small)
        }
    }

    @ViewBuilder private func changeBadge(_ current: (EditMode, Float)?, remove: @escaping () -> Void) -> some View {
        if let (m, v) = current {
            HStack {
                let c = m == .ablate ? Theme.ablated : (m == .scale ? Theme.scaled : Theme.pinned)
                Text(m == .ablate ? "Turned off" : (m == .scale ? String(format: "Turned to %.1f×", v) : String(format: "Held at %.2f", v)))
                    .font(.caption.weight(.semibold)).padding(.horizontal, 8).padding(.vertical, 3)
                    .background(c.opacity(0.3), in: Capsule())
                Button("Undo", action: remove).buttonStyle(.bordered).controlSize(.small)
            }
        }
    }
}

/// The old logit-lens table for the token at the cursor (what each layer would predict next).
/// Embedded by the Advanced panel.
struct LogitLensTable: View {
    @Environment(AppState.self) private var state

    var body: some View {
        if let run = state.activeRun, let t = state.displayIndex, t < run.count, !run.tokens[t].lens.isEmpty {
            let rec = run.tokens[t]
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(rec.lens.enumerated()), id: \.offset) { l, c in
                    HStack(spacing: 6) {
                        Text("L\(l)").font(.system(.caption2, design: .monospaced)).frame(width: 26, alignment: .leading)
                        Text(tokText(state.engine?.tokenText(c.id) ?? "\(c.id)")).font(.system(.caption, design: .monospaced))
                            .frame(width: 80, alignment: .leading).lineLimit(1)
                        GeometryReader { g in
                            Rectangle().fill(Theme.thinking).frame(width: g.size.width * CGFloat(min(max(c.prob, 0), 1)))
                        }.frame(height: 5)
                        Text(String(format: "%.2f", c.prob)).font(.system(.caption2, design: .monospaced)).frame(width: 34, alignment: .trailing)
                    }
                }
            }
        } else {
            Text("No logit lens for this word-piece yet. The logit lens peeks at each floor's best guess of the next word.").font(.caption).foregroundStyle(.secondary)
        }
    }
}

private struct FlowRow<Content: View>: View {
    @ViewBuilder var content: () -> Content
    init(@ViewBuilder _ content: @escaping () -> Content) { self.content = content }
    var body: some View { FlowLayoutInspector(spacing: 4) { content() } }
}

private struct FlowLayoutInspector: Layout {
    var spacing: CGFloat
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        arrange(proposal.width ?? 300, subviews).size
    }
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let r = arrange(bounds.width, subviews)
        for (i, f) in r.frames.enumerated() {
            subviews[i].place(at: CGPoint(x: bounds.minX + f.minX, y: bounds.minY + f.minY), proposal: .unspecified)
        }
    }
    private func arrange(_ width: CGFloat, _ subviews: Subviews) -> (size: CGSize, frames: [CGRect]) {
        var x: CGFloat = 0, y: CGFloat = 0, rowH: CGFloat = 0, maxW: CGFloat = 0
        var frames: [CGRect] = []
        for sv in subviews {
            let s = sv.sizeThatFits(.unspecified)
            if x > 0, x + s.width > width { x = 0; y += rowH + spacing; rowH = 0 }
            frames.append(CGRect(x: x, y: y, width: s.width, height: s.height))
            x += s.width + spacing; rowH = max(rowH, s.height); maxW = max(maxW, x)
        }
        return (CGSize(width: min(maxW, width), height: y + rowH), frames)
    }
}
