import SwiftUI

/// Controls and teaching for slow motion: one word travelling up the floors, drawn arrow by arrow.
struct SlowMotionBar: View {
    @Environment(AppState.self) private var state

    @State private var showMore = false
    private let minFPS = 0.01, maxFPS = 4.0   // slowest ≈ 100 s per floor

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Button { state.tracePaused.toggle() } label: {
                    Image(systemName: state.tracePaused ? "play.fill" : "pause.fill").frame(width: 18)
                }
                .disabled(state.trace == nil)
                .accessibilityLabel(state.tracePaused ? "Play slow motion" : "Pause slow motion")

                Button("Replay") { replay() }
                    .disabled(state.isTracing || state.trace == nil)

                Text(floorText).font(.callout.weight(.medium)).monospacedDigit()
                if state.isTracing { ProgressView().controlSize(.small) }
                Slider(value: speedBinding, in: 0...1).frame(maxWidth: 160)
                    .accessibilityLabel("Playback speed")

                Spacer(minLength: 8)
                Text("Word: \(wordText)").font(.callout).foregroundStyle(Theme.answer).lineLimit(1)
                Button("Done") { state.stopSlowMotion() }.buttonStyle(.borderedProminent)
            }
            if let n = state.traceFocus {
                Button { Task { await state.followDesk(n, at: state.trace?.position) } } label: {
                    Label("Follow the focused desk", systemImage: "point.topleft.down.to.point.bottomright.curvepath")
                }
                .buttonStyle(.borderedProminent).tint(Theme.positive)
                .disabled(state.isFollowing)
                .help("\(plainNeuronName(n)): walk through what it reads, asks, writes and who hears it.")
            }
            DisclosureGroup(isExpanded: $showMore) {
                moreBody
            } label: {
                Text("More").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(Theme.panel)
    }

    @ViewBuilder private var moreBody: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(speedText).font(.caption).foregroundStyle(.secondary)
            if let t = state.trace, let e = explainedText(t) {
                Text(e).font(.caption).foregroundStyle(.secondary)
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    FoldedLearnCard([
                        "Desks never call each other. Each desk writes a note into the memo, and later desks read the memo. An **arrow** shows how much one desk's note pushed (amber) or held back (blue) a desk one floor up. Thicker means stronger.",
                        "Click any desk to ask \"why did this desk get busy?\" - you will see notes from every floor below it.",
                        "The percentages show how much the arrows explain. The rest comes from notes borrowed from earlier words (**attention**, a ten-dollar word for looking back at earlier words) and from the word itself.",
                        realTimeText,
                    ], words: [
                        ("arrow (direct path)", "how much one desk's note pushed or held back a desk one floor up"),
                        ("attention", "the building looking back at earlier words"),
                    ])
                    MathDisclosure { arrowMath }
                }
            }
            .frame(maxHeight: 260)
        }
    }

    // MARK: pieces

    private var floors: Int { max(state.geometry.numLayers, 1) }
    private var memoN: String { state.geometry.memoText }

    /// Real time per word and per floor, from the run's own measured writing speed (this Mac, this model).
    private var realTimeText: String {
        let perFloor = state.realSecondsPerFloor
        let perWord = perFloor * Double(floors)
        let slowest = Self.nice(1 / minFPS)
        return "Real time: the whole trip up all \(floors) floors takes about \(Self.nice(perWord * 1000)) thousandths of a second (~\(Self.nice(perWord * 1000)) ms) per word on this Mac - about \(Self.nice(perFloor * 1000)) ms per floor. Here it is stretched so you can follow it; at the slowest setting one floor takes about \(slowest) seconds."
    }

    private var floorText: String {
        let n = min(floors, max(0, Int(state.traceProgress.rounded(.down))) + (state.traceProgress >= Double(floors) ? 0 : 1))
        return "Floor \(n) of \(floors)"
    }

    private var wordText: String {
        guard let t = state.trace, let run = state.activeRun, t.position >= 0, t.position < run.tokens.count else { return "-" }
        let s = run.tokens[t.position].displayText
            .replacingOccurrences(of: "\n", with: "↵")
            .trimmingCharacters(in: .whitespaces)
        return s.isEmpty ? "(space)" : "\"\(s)\""
    }

    private var speedBinding: Binding<Double> {
        Binding(
            get: { log(state.traceFloorsPerSecond / minFPS) / log(maxFPS / minFPS) },
            set: { state.traceFloorsPerSecond = minFPS * pow(maxFPS / minFPS, $0) })
    }

    private var speedText: String {
        let f = state.slowdownFactor
        guard f > 0 else { return "" }
        let perFloor = 1 / max(state.traceFloorsPerSecond, 1e-6)
        let floorText = perFloor >= 1 ? "\(Self.nice(perFloor)) s per floor" : "\(Self.nice(perFloor * 1000)) ms per floor"
        return "\(floorText) · about \(Self.nice(f))× slower than real"
    }

    static func nice(_ v: Double) -> String {
        if v < 10 { return String(format: "%.1f", v) }
        let mag = pow(10, floor(log10(v)) - 1)
        let r = (v / mag).rounded() * mag
        let f = NumberFormatter(); f.numberStyle = .decimal; f.maximumFractionDigits = 0
        return f.string(from: NSNumber(value: r)) ?? "\(Int(r))"
    }

    private func explainedText(_ t: TraceResult) -> String? {
        guard !t.explained.isEmpty else { return nil }
        let i = min(t.explained.count - 1, max(0, Int(state.traceProgress.rounded(.down))))
        let pct = Int((max(0, min(1, t.explained[i])) * 100).rounded())
        return "The arrows into this floor explain about \(pct)% of what its busiest desks read. The rest comes from earlier words and the word itself."
    }

    // MARK: the math of one arrow

    /// The floor whose incoming arrows are being drawn right now (0-based target floor).
    private var currentFloor: Int {
        max(0, min(floors - 1, Int(state.traceProgress.rounded(.down))))
    }

    private var strongestArrow: TraceEdge? {
        guard let t = state.trace else { return nil }
        let into = t.edges.filter { $0.toLayer == currentFloor }
        let pool = into.isEmpty ? t.edges.filter { $0.fromLayer == currentFloor } : into
        return pool.max { abs($0.weight) < abs($1.weight) }
    }

    @ViewBuilder private var arrowMath: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(markdown("An arrow's weight is the **direct-path push**: how much the source desk's note, read straight through the memo, moves the target desk's trigger (its gate)."))
                .font(.callout).fixedSize(horizontal: false, vertical: true)
            FormulaRow("w = a_src \u{00D7} (W_gate[target] \u{00B7} c_src)      c_src = (w_out \u{2299} W_down[:, src]) / rms(src floor's notes)", [
                ("a_src", "how busy the source desk is on this word (its h)"),
                ("W_down[:, src]", "the source desk's note: its column, \(memoN) numbers"),
                ("w_out \u{2299}, rms", "the source floor tidies its notes on the way out: it rescales them to a standard size (rms is their typical size on this word) and multiplies in its own weights (\u{2299}: number by number)"),
                ("c_src", "the source desk's column as it lands in the memo"),
                ("W_gate[target]", "the target desk's trigger row, \(memoN) numbers; it reads the memo as it is"),
                ("w", "+ = pushed the target up (amber), \u{2212} = held it back (blue)"),
            ])
            if let e = strongestArrow {
                Text("Strongest arrow here: \(plainNeuronName(e.from)) \u{2192} \(plainNeuronName(e.to)), w = \(mathNum(e.weight, 3))")
                    .font(.system(.caption, design: .monospaced))
            } else {
                Text("Arrows appear as the word climbs.").font(.caption).foregroundStyle(.secondary)
            }
            Text("Measured on this model (2026-10-28, at the word \u{201C}La\u{201D}): arrows from the previous floor's busiest 12 desks explain only about 4\u{2013}10% of a middle floor's busiest desks' input (17\u{2013}21% on floors 12 and 13). The rest comes through attention and earlier floors.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func replay() {
        let pos = state.trace?.position
        state.tracePaused = false
        Task { await state.startSlowMotion(at: pos) }
    }
}

/// LearnCard's layout, but starting collapsed so the slow-motion view stays clear until asked.
private struct FoldedLearnCard: View {
    let paragraphs: [String]
    let words: [(term: String, meaning: String)]
    @State private var open = true

    init(_ paragraphs: [String], words: [(term: String, meaning: String)] = []) {
        self.paragraphs = paragraphs
        self.words = words
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button { withAnimation(.easeOut(duration: 0.15)) { open.toggle() } } label: {
                HStack(spacing: 6) {
                    Image(systemName: "lightbulb").foregroundStyle(Theme.pinned)
                    Text("What is happening").font(.caption.weight(.semibold))
                    Spacer()
                    Image(systemName: open ? "chevron.up" : "chevron.down").font(.caption2).foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(open ? "Hide the explanation" : "Show the explanation")
            if open {
                ForEach(Array(paragraphs.enumerated()), id: \.offset) { _, p in
                    Text(markdown(p)).font(.callout).fixedSize(horizontal: false, vertical: true)
                }
                if !words.isEmpty {
                    Divider().opacity(0.4)
                    ForEach(Array(words.enumerated()), id: \.offset) { _, w in
                        Text(markdown("**\(w.term)**: \(w.meaning)")).font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
        .padding(10)
        .background(Theme.pinned.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.pinned.opacity(0.18)))
    }
}
