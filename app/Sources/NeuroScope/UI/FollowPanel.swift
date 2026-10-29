import SwiftUI

/// "Follow this desk": one desk's whole story at one word, told in seven steps. Each step has three layers:
/// a plain sentence in the office-building picture with this desk's real numbers, the formula with every symbol
/// glossed, and the numbers themselves so the arithmetic can be redone by hand.
struct FollowPanel: View {
    @Environment(AppState.self) private var state
    @FocusState private var focused: Bool
    @State private var showMath = false

    static let stepTitles = ["Reads", "Asks", "Decides", "How loud", "Writes", "Who hears", "The word"]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let f = state.follow {
                header(f)
                stepper
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        stepBody(f)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.trailing, 6)
                }
                .frame(maxHeight: 360)
                footer
            } else if state.isFollowing {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Following this desk through its whole job on this word…").font(.callout).foregroundStyle(.secondary)
                    Spacer()
                    Button("Done") { state.stopFollow() }
                }
            }
        }
        .padding(12)
        .background(Theme.panel)
        .focusable()
        .focusEffectDisabled()
        .focused($focused)
        .onKeyPress(.leftArrow) { go(-1); return .handled }
        .onKeyPress(.rightArrow) { go(1); return .handled }
        .onAppear { focused = true }
    }

    // MARK: frame

    private var step: Int { max(0, min(Self.stepTitles.count - 1, state.followStep)) }

    private func go(_ d: Int) {
        let k = max(0, min(Self.stepTitles.count - 1, step + d))
        if k != step { state.setFollowStep(k) }
    }

    @ViewBuilder private func header(_ f: FollowResult) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "point.topleft.down.to.point.bottomright.curvepath").foregroundStyle(Theme.positive)
            VStack(alignment: .leading, spacing: 1) {
                Text(verbatim: "Following desk #\(f.index) on floor \(f.layer)").font(.headline)
                Text("at the word \(wordText(f.position))").font(.caption).foregroundStyle(Theme.answer)
            }
            if state.isFollowing { ProgressView().controlSize(.small) }
            Spacer(minLength: 8)
            Button("Done") { state.stopFollow() }.buttonStyle(.borderedProminent)
                .help("Stop following this desk")
        }
    }

    private var stepper: some View {
        HStack(spacing: 0) {
            ForEach(Array(Self.stepTitles.enumerated()), id: \.offset) { k, title in
                Button { state.setFollowStep(k) } label: {
                    VStack(spacing: 4) {
                        ZStack {
                            Circle().fill(k == step ? Theme.positive : (k < step ? Theme.positive.opacity(0.35) : Color.white.opacity(0.12)))
                                .frame(width: 20, height: 20)
                            Text("\(k + 1)").font(.caption2.weight(.bold)).monospacedDigit()
                                .foregroundStyle(k == step ? Color.black : Theme.answer)
                        }
                        Text(title).font(.caption2.weight(k == step ? .semibold : .regular))
                            .foregroundStyle(k == step ? Theme.answer : .secondary)
                            .lineLimit(1)
                    }
                    .frame(maxWidth: .infinity)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Step \(k + 1): \(title)")
                if k < Self.stepTitles.count - 1 {
                    Rectangle().fill(Color.white.opacity(k < step ? 0.3 : 0.1)).frame(height: 1).frame(maxWidth: 24)
                        .padding(.bottom, 16)
                }
            }
        }
    }

    private var footer: some View {
        HStack {
            Button { go(-1) } label: { Label("Back", systemImage: "chevron.left") }
                .disabled(step == 0)
            Spacer()
            Text("Step \(step + 1) of \(Self.stepTitles.count) · use ← → to move").font(.caption).foregroundStyle(.secondary)
            Spacer()
            Button { go(1) } label: { Label("Next", systemImage: "chevron.right").labelStyle(TrailingIconLabel()) }
                .buttonStyle(.borderedProminent)
                .disabled(step == Self.stepTitles.count - 1)
        }
    }

    // MARK: the seven steps

    @ViewBuilder private func stepBody(_ f: FollowResult) -> some View {
        switch step {
        case 0: reads(f)
        case 1: asks(f)
        case 2: decides(f)
        case 3: howLoud(f)
        case 4: writes(f)
        case 5: whoHears(f)
        default: theWord(f)
        }
    }

    private func desk(_ f: FollowResult) -> String { "Desk #\(f.index) on floor \(f.layer)" }

    private var memoN: String { state.geometry.memoText }

    // 1 Reads the memo. OLMo 2 desks read the memo as it is (no tidy-up on the way in; the floor tidies what it
    // WRITES, step 5). The engine reports rms = 1 for this model.
    @ViewBuilder private func reads(_ f: FollowResult) -> some View {
        plain("\(desk(f)) starts by reading the **memo** (the running note that rides the elevator up the building, \(memoN) numbers long). On \(wordText(f.position)) the memo arriving at floor \(f.layer) is **\(mathNum(f.memoNorm))** long. In this building every desk reads the memo exactly as it arrives.")
        mathFold {
        why("No tidy-up on the way in? Right: this model tidies what each floor **writes** instead (step 5), so the memo never grows loud enough to drown out a question.")
        FormulaRow("x = the memo arriving at floor \(f.layer)      |x| = \u{221A}(x\u{2081}\u{00B2} + \u{2026} + x\u{2099}\u{00B2})", [
            ("x", "the memo arriving at this floor, \(memoN) numbers"),
            ("|x|", "the memo's length (all its numbers squared, added up, square-rooted)"),
            ("n", "\(memoN): how many numbers the memo has"),
        ])
        chips([("|x|", mathNum(f.memoNorm)), ("numbers", memoN)])
        }
    }

    // 2 Asks its question
    @ViewBuilder private func asks(_ f: FollowResult) -> some View {
        plain("\(desk(f)) asks: how much of **my pattern** is in this memo? The answer is **\(signed(f.gate))** - \(gateWords(f.gate)). Its pattern (its **gate row**, \(memoN) fixed numbers) never changes; only the memo does, word to word.")
        mathFold {
        FormulaRow("gate = W_gate[\(f.index)] \u{00B7} x", [
            ("W_gate[\(f.index)]", "this desk's question: its gate row, \(memoN) numbers set in training and never changed"),
            ("\u{00B7}", "dot product: multiply the two lists number by number and add it all up. Big when they agree"),
            ("x", "the memo from step 1"),
        ])
        chips([("gate", signed(f.gate))])
        }

        if !f.wakesOn.isEmpty {
            sub("What wakes it up")
            plain("This desk wakes up for: **\(wakeList(f))**. These are the words whose meaning best matches its pattern, so this is what its question is about. Nobody labelled it - the app found it by comparing the pattern with every word the building knows.")
            wordChips(f.wakesOn, color: Theme.positive)
        }

        if !f.askedBy.isEmpty {
            sub("Who raised the question")
            plain("These earlier desks' notes raised (amber) or lowered (blue) its answer on this word. Click one to follow it instead.")
            edgeRows(f.askedBy, desk: { $0.from }, position: f.position)
            let listed = f.askedBy.reduce(Float(0)) { $0 + $1.weight }
            if abs(listed) >= abs(f.gate) * 0.999 {
                plain("These notes add up to **\(signed(listed))**, more than its whole answer (\(signed(f.gate))). Other inputs pull it back: earlier words (**attention**, notes borrowed from words already read), the word itself, and desks not listed.")
            } else {
                plain("These notes explain **\(pct(f.askedByExplained))** of its answer; the rest comes from earlier words (**attention**, notes borrowed from words already read) and from the word itself.")
            }
        }
    }

    // 3 Decides whether to speak
    @ViewBuilder private func decides(_ f: FollowResult) -> some View {
        plain("Now it decides whether to speak at all. A soft on/off switch, **\u{03C3}** (sigma), turns its answer into a number between 0 (off) and 1 (on). Its answer was \(signed(f.gate)), so the switch reads **\(mathNum(f.sigma, 4))** - \(sigmaWords(f.sigma)). What passes through is **silu = \(signed(f.silu))**.")
        mathFold {
        why("Why a switch? A yes passes nearly whole; a no is squashed toward zero (silu never goes below \u{2212}0.28). That is how a desk stays quiet on words that are not its business.")
        FormulaRow("silu(gate) = gate \u{00D7} \u{03C3}(gate)      \u{03C3}(z) = 1 / (1 + e^\u{2212}z)", [
            ("\u{03C3}(z)", "the switch: near 0 for a big no, 0.5 at zero, near 1 for a big yes"),
            ("e", "the number 2.718…; e^\u{2212}z shrinks fast as z grows"),
            ("silu", "the answer let through the switch"),
        ])
        chips([("gate", signed(f.gate)), ("\u{03C3}", mathNum(f.sigma, 4)), ("gate \u{00D7} \u{03C3}", signed(f.gate * f.sigma)), ("silu", signed(f.silu))])
        SiluPlot(gate: f.gate).frame(height: 110).frame(maxWidth: 360)
        Text("The curve is silu for every possible answer; the dot is this desk's answer on this word.")
            .font(.caption).foregroundStyle(.secondary)
        }
    }

    // 4 How loudly and which way
    @ViewBuilder private func howLoud(_ f: FollowResult) -> some View {
        plain("It asks a second question with another fixed pattern, **up**: how much, and which way? Its answer is **\(signed(f.up))**. Multiply the two: h = \(signed(f.silu)) \u{00D7} \(signed(f.up)) = **\(signed(f.hRaw))**. That number, **h** (how busy the desk is), says how loudly it will speak (\(mathNum(abs(f.hRaw)))) and which way (\(f.hRaw >= 0 ? "positive: it writes its message as is" : "negative: it writes its message backwards, pulling the memo away from it")).")
        if abs(f.h - f.hRaw) > 1e-4 {
            plain("Your change to this desk overrode that: it actually wrote **h = \(signed(f.h))**.")
        }
        mathFold {
        FormulaRow("up = W_up[\(f.index)] \u{00B7} x      h = silu(gate) \u{00D7} up", [
            ("W_up[\(f.index)]", "this desk's second pattern, \(memoN) fixed numbers"),
            ("up", "how much and which way: its sign can flip the note"),
            ("h", "the desk's busyness on this word: the color saturation of its dot (amber +, blue \u{2212})"),
        ])
        chips([("silu", signed(f.silu)), ("up", signed(f.up)), ("h", signed(f.hRaw))]
              + (abs(f.h - f.hRaw) > 1e-4 ? [("h after your change", signed(f.h))] : []))
        }
    }

    // 5 Writes its note. OLMo 2: memo += w_post ⊙ m / rms(m), m = Σ_j h_j W_down[:, j] (the floor's whole output);
    // this desk's share is h × c with c = (w_post ⊙ W_down[:, j]) / rms(m), rms(m) held at its value on this word.
    // The engine's column_norm is |c| (that scale included).
    @ViewBuilder private func writes(_ f: FollowResult) -> some View {
        plain("It writes its note into the memo. The note is its **column** (its fixed message, \(memoN) numbers) times h. On the way out the floor **tidies** everything its desks wrote: it rescales the floor's notes to a standard size and multiplies in its own fixed weights. Tidied, this desk's column is \(mathNum(f.columnNorm)) long, so its note is **\(mathNum(f.writeNorm, 1))** long; the whole memo is **\(mathNum(f.memoNorm, 1))** long - \(voiceWords(f.writeNorm, f.memoNorm)).")
        mathFold {
        why("Why tidy on the way out? The memo grows as it climbs. Rescaling each floor's notes keeps every floor's voice at a steady volume, so no floor can shout over the memo. What decides where a note goes: the message (the column) is the same every time; only its loudness h changes from word to word. Who it reaches is fixed by the wiring of later desks' questions (step 6).")
        FormulaRow("memo += h \u{00D7} c      c = (w_out \u{2299} W_down[:, \(f.index)]) / rms(floor's notes)", [
            ("W_down[:, \(f.index)]", "this desk's column: the message it always writes"),
            ("w_out", "this floor's fixed tidy-up weights, one per memo number"),
            ("\u{2299}", "multiply number by number, keeping it a list"),
            ("rms(floor's notes)", "the typical number size of everything this floor wrote on this word; dividing by it rescales the notes"),
            ("c", "the column as it lands in the memo"),
            ("+=", "add it to the memo, number by number; nothing is erased"),
        ])
        chips([("|c|", mathNum(f.columnNorm)), ("|h|", mathNum(abs(f.h))), ("|note| = |h| \u{00D7} |c|", mathNum(f.writeNorm, 1)), ("|memo|", mathNum(f.memoNorm, 1)),
               ("note / memo", "\(Int((f.writeNorm / max(f.memoNorm, 1e-6) * 100).rounded()))%")])
        }
    }

    // 6 Who hears it
    @ViewBuilder private func whoHears(_ f: FollowResult) -> some View {
        plain("Desks never call each other. Later desks just read the memo, and some of them find this desk's note in it. These are the desks on later floors whose question its note pushed (amber) or held back (blue) the most.")
        if f.heardBy.isEmpty {
            Text("No later desk was moved much by this note on this word.").font(.callout).foregroundStyle(.secondary)
        } else {
            edgeRows(f.heardBy, desk: { $0.to }, position: f.position)
        }
        mathFold {
        why("**What decides where it goes:** its message (its column) is fixed - the same \(memoN) numbers on every word. Only its loudness h changes word to word. Who reacts is set by the later desks' questions: a desk hears this note exactly as much as its own question lines up with the message.")
        FormulaRow("push = h \u{00D7} (W_gate\u{2032}[k] \u{00B7} c)", [
            ("k", "a later desk; \u{2032} means its floor's numbers"),
            ("W_gate\u{2032}[k]", "that desk's question (its gate row); it reads the memo as it is, so nothing else is in between"),
            ("c", "this desk's tidied column from step 5"),
            ("push", "how much this note moved that desk's answer: + amber, \u{2212} blue"),
        ])
        if let e = f.heardBy.first {
            chips([("h", signed(f.h)), ("strongest push", signed(e.weight)), ("to", "floor \(e.toLayer) #\(e.toIndex)")])
        }
        }
        sub("The path of the note")
        plain("Follow the loudest listener, then its loudest listener, and so on. Click a desk to follow it at this same word.")
        pathChain(f)
    }

    // 7 Effect on the final word
    @ViewBuilder private func theWord(_ f: FollowResult) -> some View {
        let next = clean(state.engine?.tokenText(f.nextToken) ?? "#\(f.nextToken)")
        plain("At the top of the building the memo is turned into a score for every possible next word, and the best one is written. Through its own note, straight up, \(desk(f)) \(f.nextEffect >= 0 ? "raised" : "lowered") the score of the chosen word **\"\(next)\"** by **\(mathNum(abs(f.nextEffect)))**.")
        mathFold {
        why(state.geometry.fill("Why so small? Its note is one of about {allDesks} written on the way up ({desks} desks on {floors} floors), and most of its effect travels through the later desks it reached in step 6, not straight to the top."))
        FormulaRow("effect = h \u{00D7} (unembed[word] \u{00B7} (w_final \u{2299} c)) / rms_final", [
            ("unembed[word]", "the word's score pattern at the top: how the memo is read into that word's score (its **logit**)"),
            ("w_final, rms_final", "the last tidy-up before the scores"),
            ("effect", "+ raised the word's score, \u{2212} lowered it"),
        ])
        chips([("h", signed(f.h)), ("effect on \"\(next)\"", signed(f.nextEffect))])
        }
        if !f.wordEffects.isEmpty {
            let toward = f.wordEffects.filter { $0.score > 0 }.sorted { $0.score > $1.score }
            let away = f.wordEffects.filter { $0.score < 0 }.sorted { $0.score < $1.score }
            if !toward.isEmpty {
                sub("Pushed toward")
                wordChips(toward, color: Theme.positive, showScore: true)
            }
            if !away.isEmpty {
                sub("Pushed away from")
                wordChips(away, color: Theme.negative, showScore: true)
            }
        }
    }

    // MARK: building blocks

    private func mathFold<C: View>(@ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { showMath.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "function").foregroundStyle(Theme.thinking)
                    Text("Show the math").font(.caption.weight(.semibold))
                    Image(systemName: showMath ? "chevron.up" : "chevron.down").font(.caption2).foregroundStyle(.secondary)
                    Spacer()
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if showMath { content() }
        }
    }

    private func plain(_ s: String) -> some View {
        Text(markdown(s)).font(.title2).foregroundStyle(Theme.answer)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func why(_ s: String) -> some View {
        Text(markdown(s)).font(.callout).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private func sub(_ s: String) -> some View {
        Text(s).font(.subheadline.weight(.semibold)).padding(.top, 4)
    }

    private func chips(_ items: [(String, String)]) -> some View {
        HStack(spacing: 6) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, it in
                HStack(spacing: 4) {
                    Text(it.0).font(.caption).foregroundStyle(Theme.thinking)
                    Text(it.1).font(.system(.callout, design: .monospaced).weight(.medium)).foregroundStyle(Theme.answer)
                }
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 6))
                .textSelection(.enabled)
            }
        }
    }

    private func wordChips(_ words: [WordScore], color: Color, showScore: Bool = false) -> some View {
        GuideFlowLayout(spacing: 4) {
            ForEach(Array(words.enumerated()), id: \.offset) { _, w in
                HStack(spacing: 4) {
                    Text(chipText(w.token)).font(.system(.callout, design: .monospaced))
                    if showScore { Text(signed(w.score)).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
                }
                .padding(.horizontal, 7).padding(.vertical, 3)
                .background(color.opacity(0.18), in: RoundedRectangle(cornerRadius: 5))
            }
        }
    }

    /// One row per desk: name, a push bar (amber +, blue −) scaled to the strongest, and the value. Clickable.
    private func edgeRows(_ edges: [TraceEdge], desk: @escaping (TraceEdge) -> NeuronID, position: Int) -> some View {
        let peak = max(edges.map { abs($0.weight) }.max() ?? 1, 1e-6)
        return VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(edges.enumerated()), id: \.offset) { _, e in
                let n = desk(e)
                Button { follow(n, position) } label: {
                    HStack(spacing: 8) {
                        Text(verbatim: "floor \(n.layer) · desk #\(n.index)").font(.system(.callout, design: .monospaced))
                            .frame(width: 170, alignment: .leading)
                        GeometryReader { g in
                            Rectangle().fill(e.weight >= 0 ? Theme.positive : Theme.negative)
                                .frame(width: max(2, g.size.width * CGFloat(abs(e.weight) / peak)))
                                .frame(maxHeight: .infinity, alignment: .center)
                        }
                        .frame(height: 8).frame(maxWidth: 180)
                        Text(signed(e.weight)).font(.system(.callout, design: .monospaced))
                            .foregroundStyle(e.weight >= 0 ? Theme.positive : Theme.negative)
                            .frame(width: 64, alignment: .trailing)
                        Spacer(minLength: 0)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Follow this desk at the same word")
            }
        }
    }

    @ViewBuilder private func pathChain(_ f: FollowResult) -> some View {
        let start = NeuronID(layer: f.layer, index: f.index)
        let hops: [NeuronID] = [start] + f.path.map(\.to)
        GuideFlowLayout(spacing: 4) {
            ForEach(Array(hops.enumerated()), id: \.offset) { i, n in
                HStack(spacing: 4) {
                    if i > 0 {
                        let w = f.path[i - 1].weight
                        Image(systemName: "arrow.right").font(.caption).foregroundStyle(w >= 0 ? Theme.positive : Theme.negative)
                    }
                    Button { if n != start { follow(n, f.position) } } label: {
                        Text(verbatim: "floor \(n.layer) #\(n.index)").font(.system(.callout, design: .monospaced))
                            .padding(.horizontal, 7).padding(.vertical, 3)
                            .background((i == 0 ? Theme.pinned : Theme.thinking).opacity(0.2), in: RoundedRectangle(cornerRadius: 5))
                    }
                    .buttonStyle(.plain)
                    .help(i == 0 ? "The desk you are following" : "Follow this desk at the same word")
                }
            }
        }
    }

    private func follow(_ n: NeuronID, _ position: Int) {
        Task { await state.followDesk(n, at: position) }
    }

    // MARK: words for numbers

    private func signed(_ v: Float, _ d: Int = 2) -> String { v > 0 ? "+" + mathNum(v, d) : mathNum(v, d) }

    private func pct(_ v: Float) -> String {
        let p = v > 1.0001 ? v : v * 100
        return "\(Int(max(0, p).rounded()))%"
    }

    private func gateWords(_ g: Float) -> String {
        switch g {
        case 4...: return "a strong yes"
        case 1..<4: return "a yes"
        case -1..<1: return "a shrug, neither yes nor no"
        case -4 ..< -1: return "a no"
        default: return "a firm no"
        }
    }

    private func sigmaWords(_ s: Float) -> String {
        switch s {
        case 0.95...: return "fully on"
        case 0.6..<0.95: return "mostly on"
        case 0.4..<0.6: return "half on"
        case 0.05..<0.4: return "mostly off"
        default: return "off"
        }
    }

    private func voiceWords(_ note: Float, _ memo: Float) -> String {
        let r = note / max(memo, 1e-6)
        if r >= 0.15 { return "a loud voice" }
        if r >= 0.03 { return "an ordinary voice" }
        if r >= 0.005 { return "a quiet voice" }
        return "a whisper"
    }

    private func wakeList(_ f: FollowResult) -> String {
        var seen = Set<String>(), out: [String] = []
        for w in f.wakesOn {
            let c = clean(w.token)
            if !c.isEmpty, seen.insert(c.lowercased()).inserted { out.append(c) }
        }
        return out.prefix(6).joined(separator: ", ") + (out.count > 6 ? " …" : "")
    }

    private func clean(_ s: String) -> String {
        let t = s.replacingOccurrences(of: "\n", with: "↵").trimmingCharacters(in: .whitespaces)
        return t.isEmpty ? "(space)" : t
    }

    private func chipText(_ s: String) -> String {
        s.replacingOccurrences(of: "\n", with: "⏎").replacingOccurrences(of: " ", with: "·")
    }

    private func wordText(_ t: Int) -> String {
        guard let run = state.activeRun, t >= 0, t < run.tokens.count else { return "this word" }
        return "\"\(clean(run.tokens[t].displayText))\""
    }
}

/// Next button with the chevron after the text.
private struct TrailingIconLabel: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 4) { configuration.title; configuration.icon }
    }
}

/// silu(z) = z·σ(z) over a range that includes the desk's gate, with the gate marked.
private struct SiluPlot: View {
    let gate: Float

    var body: some View {
        Canvas { ctx, size in
            let g = Double(gate.isFinite ? gate : 0)
            let lo = min(-6, g - 1), hi = max(6, g + 1)
            func silu(_ z: Double) -> Double { z / (1 + exp(-z)) }
            let yLo = -0.5, yHi = max(silu(hi), 1)
            func pt(_ z: Double, _ y: Double) -> CGPoint {
                CGPoint(x: (z - lo) / (hi - lo) * size.width, y: size.height - (y - yLo) / (yHi - yLo) * size.height)
            }
            var axes = Path()
            axes.move(to: pt(lo, 0)); axes.addLine(to: pt(hi, 0))
            axes.move(to: pt(0, yLo)); axes.addLine(to: pt(0, yHi))
            ctx.stroke(axes, with: .color(.white.opacity(0.18)), lineWidth: 1)

            var curve = Path()
            let steps = 120
            for i in 0...steps {
                let z = lo + (hi - lo) * Double(i) / Double(steps)
                if i == 0 { curve.move(to: pt(z, silu(z))) } else { curve.addLine(to: pt(z, silu(z))) }
            }
            ctx.stroke(curve, with: .color(Theme.thinking), lineWidth: 1.5)

            let p = pt(g, silu(g))
            var guide = Path(); guide.move(to: CGPoint(x: p.x, y: size.height)); guide.addLine(to: p)
            ctx.stroke(guide, with: .color(Theme.positive.opacity(0.5)), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            ctx.fill(Path(ellipseIn: CGRect(x: p.x - 4, y: p.y - 4, width: 8, height: 8)), with: .color(Theme.positive))
            ctx.draw(Text(String(format: "gate %+.2f", g)).font(.caption2).foregroundStyle(Theme.positive),
                     at: CGPoint(x: min(max(p.x, 36), size.width - 36), y: max(p.y - 10, 8)))
            ctx.draw(Text("z").font(.caption2).foregroundStyle(.secondary), at: CGPoint(x: size.width - 6, y: pt(0, 0).y - 8))
        }
        .background(Theme.background, in: RoundedRectangle(cornerRadius: 6))
        .accessibilityLabel("The silu curve with this desk's gate marked")
    }
}
