import SwiftUI

private final class BusyMemo {
    var key = ""
    var rows: [(id: NeuronID, value: Float)] = []
}

/// Step 2: the field plays as the model thinks; thinking + answer stream in; the busiest neurons are listed.
struct WatchStep: View {
    @Environment(AppState.self) private var state
    @State private var memo = BusyMemo()
    @State private var showThinking = true
    @State private var showBusiest = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            StepHeader(title: "Watch it think",
                       explainer: "Each dot is one desk in the building. The more colorful a dot, the harder that desk is working on the current word.")
            let g = state.geometry
            LearnCard(([
                "Every floor has {desks} desks ({allDesks} in all). Each desk is a **neuron** - a tiny worker that does one small piece of the job. A desk lights up when its worker is busy with the current word: **amber** means it is pushing for \"more of this\", **blue** means \"less of this\", and **grey** means idle.",
                "A note sheet called the **residual stream** - the memo - rides the elevator up all {floors} floors. Each floor reads it and adds its own notes. The top floor reads the finished memo and picks the next word.",
            ] + (state.hasThinking ? [
                "Before answering, the building may write notes to itself on scratch paper. That is **thinking**, shown as dim italic text. It helps the reply, but it is not part of it.",
            ] : []) + [
                "The timeline shows **tokens** - word-pieces, because the building writes in pieces like \"volcan\" + \"oes\". Click any piece to see the desks at that moment.",
                "**Slow motion** stretches one word's trip up the floors so you can watch an **arrow** (direct path) from one desk to the next. Desks never call each other; the arrows show who pushed whose trigger through the memo.",
                "The **cluster** layout reseats the desks so coworkers who get busy at the same moments sit together, in neighborhoods. The real building has no seating plan; desk numbers mean nothing on their own.",
            ]).map(g.fill), words: [
                ("neuron", g.fill("one desk; {desks} on each floor")),
                ("amber / blue / grey", "pushing for more / pushing for less / idle"),
                ("residual stream", "the memo that rides the elevator up; each floor adds notes"),
            ] + (state.hasThinking ? [("thinking", "scratch paper the building writes to itself before answering")] : []) + [
                ("token", "a word-piece; the timeline shows these"),
                ("arrow (direct path)", "how much one desk's note pushed or held back a desk one floor up"),
                ("attention", "the building looking back at earlier words"),
                ("cluster", "desks reseated into neighborhoods of coworkers who get busy together"),
            ])
            MathDisclosure {
                VStack(alignment: .leading, spacing: 8) {
                    Text(markdown("The memo is really a list of **\(g.memoText) numbers** (a **vector**), one list per floor and word. We call it **x**."))
                        .font(.callout).fixedSize(horizontal: false, vertical: true)
                    FormulaRow("h = silu(gate\u{00B7}x) \u{00D7} (up\u{00B7}x)", [
                        ("gate, up", "two lists of \(g.memoText) numbers each desk owns; **\u{00B7}** = multiply number by number, add up"),
                        ("silu", "a soft on-switch: about 0 when its input is negative"),
                        ("h", "the desk's busyness, one number: the dot's color"),
                    ])
                    FormulaRow("memo += h \u{00D7} c      c = column of W_down, tidied", [
                        ("column", "the desk's own note, \(g.memoText) numbers; it writes h copies of it into the memo"),
                        ("tidied", "before a floor's notes join the memo, the floor rescales them to a standard size and multiplies in its own fixed weights"),
                    ])
                    if case .neuron(let n)? = state.selection, let run = state.activeRun,
                       let t = state.displayIndex, t < run.count {
                        Text("Your desk (\(plainNeuronName(n))) on this word: h = \(mathNum(run.activation(n, at: t), 3)).")
                            .font(.system(.caption, design: .monospaced))
                    }
                }
            }
            if let run = state.activeRun {
                statusLine(run)
                text(run)
                DisclosureGroup(isExpanded: $showBusiest) {
                    busiest(run).padding(.top, 6)
                } label: {
                    Text("Show busiest desks").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                }
            } else {
                Text("Nothing yet. Go back to Ask and type a question, so the building has something to work on.").foregroundStyle(.secondary)
            }

            if case .neuron(let n)? = state.selection {
                HStack {
                    Image(systemName: "smallcircle.filled.circle").foregroundStyle(Theme.positive)
                    Text("You picked \(plainNeuronName(n))").font(.callout)
                    Spacer()
                    Button("See what this desk does") { state.step = .change }.controlSize(.small)
                }
                .padding(8)
                .background(Color.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
            } else {
                Label("Click any dot to see what that desk does.", systemImage: "cursorarrow.click")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Button { Task { await state.startSlowMotion(at: state.displayIndex) } } label: {
                Label("Slow motion: watch one word travel up the floors", systemImage: "tortoise").frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered).controlSize(.regular)
            .disabled(!canSlow)

            Button { state.step = .change } label: {
                Label("Change something", systemImage: "arrow.right").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
            .disabled(state.activeRun == nil || state.isGenerating)
        }
    }

    private var canSlow: Bool {
        guard let run = state.activeRun, run.status == .finished, state.displayIndex != nil else { return false }
        return !state.isGenerating && !state.isTracing && !state.slowMotion
    }

    // MARK: pieces

    private func statusLine(_ run: Run) -> some View {
        let words = generatedWordCount(run)
        let (text, color): (String, Color) = {
            switch run.status {
            case .running:
                let last = run.tokens.last?.stream
                if last == .answer { return ("Writing the reply… \(words) word-pieces", Theme.answer) }
                if last == .thinking { return ("Writing on scratch paper… \(words) word-pieces", Theme.thinking) }
                return ("Reading your question on every floor…", .secondary)
            case .finished:
                if let s = run.summary, let n = s.tokensGenerated, let tps = s.decodeTokPerS, tps > 0 {
                    let secs = (s.prefillMs ?? 0) / 1000 + Double(n) / tps
                    return (String(format: "Done in %.1f s · %d word-pieces", secs, words), .secondary)
                }
                return ("Done · \(words) word-pieces", .secondary)
            case .cancelled: return ("Stopped after \(words) word-pieces", .secondary)
            case .failed(let e): return ("Something went wrong: \(e)", Theme.ablated)
            }
        }()
        return HStack(spacing: 8) {
            if run.status == .running { ProgressView().controlSize(.small) }
            Text(text).font(.callout.weight(.medium)).foregroundStyle(color)
            Spacer()
            if state.isGenerating {
                Button("Stop") { state.cancel() }.controlSize(.small)
            }
        }
    }

    @ViewBuilder private func text(_ run: Run) -> some View {
        let thinking = plainStreamText(run, .thinking)
        let answer = plainStreamText(run, .answer)
        VStack(alignment: .leading, spacing: 10) {
            if !thinking.isEmpty {
                DisclosureGroup(isExpanded: $showThinking) {
                    ScrollView {
                        Text(thinking).font(.system(size: 12).italic())
                            .foregroundStyle(Theme.thinking.opacity(0.75))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(maxHeight: 150)
                } label: {
                    Text("Thinking (scratch paper)").font(.caption.weight(.semibold)).foregroundStyle(Theme.thinking)
                }
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Reply").font(.caption.weight(.semibold)).foregroundStyle(Theme.answer)
                Text(answer.isEmpty ? (run.status == .running ? "…" : "(no reply)") : answer)
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.answer)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
        }
        .padding(10)
        .background(Theme.background, in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder private func busiest(_ run: Run) -> some View {
        let rows = busyRows(run)
        if !rows.isEmpty {
            let mx = max(rows.first?.value ?? 1, 1e-6)
            VStack(alignment: .leading, spacing: 6) {
                StepSection("Busiest desks on this word")
                ForEach(rows, id: \.id) { r in
                    let picked = state.selection == .neuron(r.id)
                    Button { state.select(.neuron(r.id)) } label: {
                        HStack(spacing: 8) {
                            Text(plainNeuronName(r.id)).font(.system(.caption, design: .monospaced))
                                .frame(width: 150, alignment: .leading)
                            GeometryReader { g in
                                Capsule().fill(Theme.positive.opacity(0.75))
                                    .frame(width: max(2, g.size.width * CGFloat(min(abs(r.value) / mx, 1))))
                            }
                            .frame(height: 6)
                        }
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .background(picked ? Color.white.opacity(0.1) : Color.clear, in: RoundedRectangle(cornerRadius: 5))
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("\(plainNeuronName(r.id)), one of the busiest desks. Select it.")
                }
            }
        }
    }

    private func busyRows(_ run: Run) -> [(id: NeuronID, value: Float)] {
        guard let t = state.displayIndex, t < run.count else { return [] }
        let key = "\(run.id)|\(t)|\(run.count / 16)"
        if key != memo.key {
            memo.key = key
            memo.rows = Analysis.topNeurons(in: run, at: t, k: 8, scales: state.scales(for: run).layer)
        }
        return memo.rows
    }
}
