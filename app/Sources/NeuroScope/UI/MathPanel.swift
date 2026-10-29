import SwiftUI

// "Show the math": reusable pieces that walk a learner from a plain sentence, to the formula, to THIS run's
// real numbers, so they can redo the arithmetic and then build their own edit.

// MARK: formatting

/// One number, fixed decimals, with a real minus sign so columns read cleanly.
func mathNum(_ v: Float, _ digits: Int = 2) -> String {
    guard v.isFinite else { return "-" }
    let s = String(format: "%.\(digits)f", v)
    return s.hasPrefix("-") ? "\u{2212}" + s.dropFirst() : s
}

/// 1018 -> "1,018".
func groupedInt(_ n: Int) -> String {
    let f = NumberFormatter(); f.numberStyle = .decimal
    return f.string(from: NSNumber(value: n)) ?? "\(n)"
}

/// Residual site in words: site 0 is the front door (entering floor 0), site s is the memo leaving floor s-1.
func siteName(_ site: Int) -> String { site <= 0 ? "the front door (before floor 0)" : "floor \(site - 1)" }

// MARK: the disclosure

@MainActor private enum MathMemory {
    static var restored = false
    static let key = "neuroscope.showMath"
}

/// A collapsible "Show the math" section bound to state.showMath and remembered between launches.
struct MathDisclosure<Content: View>: View {
    @Environment(AppState.self) private var state
    let title: String
    let content: () -> Content

    init(_ title: String = "Show the math", @ViewBuilder _ content: @escaping () -> Content) {
        self.title = title
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { state.showMath.toggle() }
                UserDefaults.standard.set(state.showMath, forKey: MathMemory.key)
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "function").foregroundStyle(Theme.thinking)
                    Text(title).font(.caption.weight(.semibold))
                    Spacer()
                    Image(systemName: state.showMath ? "chevron.up" : "chevron.down")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(state.showMath ? "Hide the math" : "Show the math")

            if state.showMath { content() }
        }
        .padding(10)
        .background(Theme.thinking.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.thinking.opacity(0.2)))
        .onAppear {
            guard !MathMemory.restored else { return }
            MathMemory.restored = true
            if let v = UserDefaults.standard.object(forKey: MathMemory.key) as? Bool { state.showMath = v }
        }
    }
}

/// A numbered step: a plain sentence (markdown) over whatever math rows follow.
struct MathStep<Content: View>: View {
    let number: Int
    let title: String
    let plain: String
    let content: () -> Content

    init(_ number: Int, _ title: String, plain: String, @ViewBuilder _ content: @escaping () -> Content) {
        self.number = number; self.title = title; self.plain = plain; self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("\(number)").font(.caption.weight(.bold)).monospacedDigit()
                    .frame(width: 18, height: 18)
                    .background(Theme.thinking.opacity(0.25), in: Circle())
                Text(title).font(.callout.weight(.semibold))
            }
            Text(markdown(plain)).font(.callout).fixedSize(horizontal: false, vertical: true)
            content()
        }
    }
}

// MARK: formula + numbers

/// A formula in monospaced type with each symbol glossed underneath in plain words.
struct FormulaRow: View {
    let formula: String
    let glosses: [(symbol: String, meaning: String)]

    init(_ formula: String, _ glosses: [(symbol: String, meaning: String)] = []) {
        self.formula = formula; self.glosses = glosses
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(formula)
                .font(.system(.body, design: .monospaced).weight(.medium))
                .foregroundStyle(Theme.answer)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            ForEach(Array(glosses.enumerated()), id: \.offset) { _, g in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(g.symbol).font(.system(.caption, design: .monospaced).weight(.semibold))
                        .foregroundStyle(Theme.thinking)
                        .frame(minWidth: 44, alignment: .leading)
                    Text(markdown(g.meaning)).font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.background, in: RoundedRectangle(cornerRadius: 6))
    }
}

/// A labelled short vector: `x = [−1.67, 5.41, −0.59, … 1,018 more]`.
struct NumbersRow: View {
    let label: String
    let values: [Float]
    let total: Int
    var digits: Int = 2
    var note: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(text).font(.system(.callout, design: .monospaced))
                .foregroundStyle(Theme.answer)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            if let note {
                Text(markdown(note)).font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var text: String {
        let shown = values.map { mathNum($0, digits) }.joined(separator: ", ")
        let rest = total - values.count
        return "\(label) = [\(shown)\(rest > 0 ? ", \u{2026} \(groupedInt(rest)) more" : "")]"
    }
}

// MARK: ink chart

/// Per floor: how much highlighter ink (x·d̂, averaged over the reply's words) the memo carries.
/// Before run (amber), what arrives at each site in the After run (dashed), and what leaves after the edit (blue).
struct InkChart: View {
    let before: InkCurve?
    let after: InkCurve?
    var mode: DirectionMode = .ablate
    @State private var hoverX: CGFloat?

    private let left: CGFloat = 38, right: CGFloat = 8, top: CGFloat = 8, bottom: CGFloat = 30

    private var series: [(values: [Float], color: Color, dashed: Bool, name: String)] {
        var s: [(values: [Float], color: Color, dashed: Bool, name: String)] = []
        if let b = before?.arriving, !b.isEmpty { s.append((b, Theme.positive, false, "Before your change")) }
        if let a = after?.arriving, !a.isEmpty {
            s.append((a, Theme.answer.opacity(0.75), true, mode == .ablate ? "After: arriving, before the erase" : "After: arriving, before the ink"))
        }
        if let a = after?.leaving, !a.isEmpty {
            s.append((a, Theme.negative, false, mode == .ablate ? "After: leaving, after the filter" : "After: leaving, after the ink"))
        }
        return s
    }

    private var count: Int { series.map(\.values.count).max() ?? 0 }

    private var range: (lo: Float, hi: Float) {
        let all = series.flatMap(\.values).filter(\.isFinite)
        var lo = min(0, all.min() ?? 0), hi = max(all.max() ?? 1, 1e-3)
        let pad = (hi - lo) * 0.06
        lo -= lo < 0 ? pad : 0; hi += pad
        return (lo, hi)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if count < 2 {
                Text("No ink measured yet.").font(.caption).foregroundStyle(.secondary)
            } else {
                GeometryReader { geo in
                    ZStack(alignment: .topLeading) {
                        Canvas { ctx, size in draw(&ctx, size) }
                        if let hx = hoverX, let i = index(at: hx, width: geo.size.width) {
                            hoverLabel(i)
                                .position(x: min(max(xPos(i, geo.size.width), 80), geo.size.width - 80), y: 34)
                                .allowsHitTesting(false)
                        }
                    }
                    .contentShape(Rectangle())
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(let p): hoverX = p.x
                        case .ended: hoverX = nil
                        }
                    }
                }
                .frame(height: 180)
                .accessibilityLabel(accessibilitySummary)
                legend
            }
        }
    }

    private func xPos(_ i: Int, _ width: CGFloat) -> CGFloat {
        left + (width - left - right) * CGFloat(i) / CGFloat(max(count - 1, 1))
    }
    private func yPos(_ v: Float, _ height: CGFloat) -> CGFloat {
        let r = range
        let t = CGFloat((v - r.lo) / max(r.hi - r.lo, 1e-6))
        return top + (height - top - bottom) * (1 - t)
    }
    private func index(at x: CGFloat, width: CGFloat) -> Int? {
        guard count > 1 else { return nil }
        let t = (x - left) / max(width - left - right, 1)
        return max(0, min(count - 1, Int((t * CGFloat(count - 1)).rounded())))
    }

    private func draw(_ ctx: inout GraphicsContext, _ size: CGSize) {
        let r = range
        // y grid: low, 0, high
        var ticks: [Float] = [r.lo, 0, r.hi]
        if r.lo >= -1e-6 { ticks = [0, r.hi / 2, r.hi] }
        for v in ticks {
            let y = yPos(v, size.height)
            var p = Path(); p.move(to: CGPoint(x: left, y: y)); p.addLine(to: CGPoint(x: size.width - right, y: y))
            ctx.stroke(p, with: .color(.white.opacity(v == 0 ? 0.28 : 0.08)), lineWidth: v == 0 ? 1 : 0.5)
            ctx.draw(Text(mathNum(v, 1)).font(.system(size: 9, design: .monospaced)).foregroundColor(.secondary),
                     at: CGPoint(x: left - 4, y: y), anchor: .trailing)
        }
        // x labels: every 4th site, plus the top
        for i in 0..<count where i % 4 == 0 || i == count - 1 {
            let label = i == 0 ? "in" : "\(i - 1)"
            ctx.draw(Text(label).font(.system(size: 9, design: .monospaced)).foregroundColor(.secondary),
                     at: CGPoint(x: xPos(i, size.width), y: size.height - bottom + 4), anchor: .top)
        }
        ctx.draw(Text("floor (the memo as it leaves that floor; \"in\" = the front door)").font(.system(size: 9)).foregroundColor(.secondary),
                 at: CGPoint(x: (left + size.width - right) / 2, y: size.height - 2), anchor: .bottom)
        // hover guide
        if let hx = hoverX, let i = index(at: hx, width: size.width) {
            var g = Path(); let x = xPos(i, size.width)
            g.move(to: CGPoint(x: x, y: top)); g.addLine(to: CGPoint(x: x, y: size.height - bottom))
            ctx.stroke(g, with: .color(.white.opacity(0.25)), lineWidth: 1)
        }
        // series
        for s in series {
            var p = Path()
            for (i, v) in s.values.enumerated() where v.isFinite {
                let pt = CGPoint(x: xPos(i, size.width), y: yPos(v, size.height))
                if p.isEmpty { p.move(to: pt) } else { p.addLine(to: pt) }
            }
            let style = s.dashed ? StrokeStyle(lineWidth: 1.5, dash: [4, 3]) : StrokeStyle(lineWidth: 1.5)
            ctx.stroke(p, with: .color(s.color), style: style)
            if !s.dashed {
                for (i, v) in s.values.enumerated() where v.isFinite {
                    let pt = CGPoint(x: xPos(i, size.width), y: yPos(v, size.height))
                    ctx.fill(Path(ellipseIn: CGRect(x: pt.x - 2, y: pt.y - 2, width: 4, height: 4)), with: .color(s.color))
                }
            }
        }
    }

    private func hoverLabel(_ i: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(i == 0 ? "At the front door (site 0)" : "Leaving floor \(i - 1) (site \(i))").font(.caption2.weight(.semibold))
            ForEach(Array(series.enumerated()), id: \.offset) { _, s in
                if i < s.values.count {
                    Text("\(s.name): \(mathNum(s.values[i]))").font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(s.color)
                }
            }
        }
        .padding(6)
        .background(Theme.panel.opacity(0.95), in: RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.12)))
    }

    private var legend: some View {
        VStack(alignment: .leading, spacing: 3) {
            ForEach(Array(series.enumerated()), id: \.offset) { _, s in
                HStack(spacing: 6) {
                    Canvas { ctx, size in
                        var p = Path(); p.move(to: CGPoint(x: 0, y: size.height / 2)); p.addLine(to: CGPoint(x: size.width, y: size.height / 2))
                        ctx.stroke(p, with: .color(s.color), style: s.dashed ? StrokeStyle(lineWidth: 1.5, dash: [4, 3]) : StrokeStyle(lineWidth: 2))
                    }
                    .frame(width: 22, height: 8)
                    Text(s.name).font(.caption).foregroundStyle(.secondary)
                }
            }
            Text("Hover the chart to read each floor's number.").font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var accessibilitySummary: String {
        series.map { s in "\(s.name): top floor \(mathNum(s.values.last ?? 0))" }.joined(separator: ". ")
    }
}

// MARK: tryouts

/// The floors the app tried before showing anything: score bar and a plain verdict per row.
struct TryoutTable: View {
    let tryouts: [Tryout]
    let mode: DirectionMode
    var norms: [Float]? = nil
    var chosenSite: Int? = nil
    var chosenValue: Float? = nil

    private var bestIndex: Int? {
        if let s = chosenSite,
           let i = tryouts.firstIndex(where: { $0.site == s && (mode == .ablate || chosenValue == nil || abs($0.value - (chosenValue ?? 0)) < 1e-3) }) {
            return i
        }
        return tryouts.indices.max { tryouts[$0].score < tryouts[$1].score }
    }

    private var maxScore: Float { max(tryouts.map(\.score).max() ?? 1, 1e-3) }

    var body: some View {
        if tryouts.isEmpty {
            Text("No tryouts recorded for this change (it was picked by hand, or came from an earlier session).")
                .font(.caption).foregroundStyle(.secondary)
        } else {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text("Floor").frame(width: 46, alignment: .leading)
                    if mode == .add { Text("\u{03B1} (ink amount)").frame(width: 110, alignment: .leading) }
                    Text("Score").frame(width: 120, alignment: .leading)
                    Text("Verdict").frame(width: 100, alignment: .leading)
                    Text("Sample answer")
                }
                .font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                ForEach(Array(tryouts.enumerated()), id: \.offset) { i, t in row(t, best: i == bestIndex) }
                Text(markdown("**Score** = how far the answers to fresh questions moved toward the target answers, minus how much they still look like the normal answers (0 = no change, higher = better). A row is thrown out (\u{2212}1) if the reply echoed the question, looped, or broke an unrelated answer."))
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func verdict(_ t: Tryout, best: Bool) -> (String, Color) {
        if t.score <= -0.999 { return ("broke the reply", Theme.ablated) }
        if best { return ("worked best", Theme.positive) }
        return ("too weak", .secondary)
    }

    private func row(_ t: Tryout, best: Bool) -> some View {
        let (v, c) = verdict(t, best: best)
        let broke = t.score <= -0.999
        let frac = broke ? 0 : CGFloat(max(0, t.score) / maxScore)
        return HStack(spacing: 8) {
            Text(t.site == 0 ? "door" : "\(t.site - 1)").font(.system(.caption, design: .monospaced))
                .frame(width: 46, alignment: .leading)
            if mode == .add {
                Text(alphaText(t)).font(.system(.caption, design: .monospaced)).frame(width: 110, alignment: .leading)
            }
            HStack(spacing: 4) {
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.06))
                    Capsule().fill(broke ? Theme.ablated.opacity(0.6) : c.opacity(best ? 0.9 : 0.5))
                        .frame(width: broke ? 6 : max(2, 70 * frac))
                }
                .frame(width: 70, height: 6)
                Text(mathNum(t.score)).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
            }
            .frame(width: 120, alignment: .leading)
            Text(v).font(.caption.weight(best ? .semibold : .regular)).foregroundStyle(c)
                .frame(width: 100, alignment: .leading)
            Text(snippet(t.sampleAnswer)).font(.caption.italic()).foregroundStyle(.secondary).lineLimit(1)
                .help(t.sampleAnswer)
        }
        .padding(.vertical, 2).padding(.horizontal, 4)
        .background(best ? Theme.positive.opacity(0.08) : Color.clear, in: RoundedRectangle(cornerRadius: 4))
    }

    private func alphaText(_ t: Tryout) -> String {
        if let n = norms, t.site < n.count, n[t.site] > 0 {
            return "\(mathNum(t.value)) (\(mathNum(t.value / n[t.site], 1))\u{00D7}|d|)"
        }
        return mathNum(t.value)
    }

    private func snippet(_ s: String) -> String {
        let one = s.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces)
        if one.isEmpty { return "(no reply)" }
        return one.count > 70 ? String(one.prefix(70)) + "\u{2026}" : "\u{201C}\(one)\u{201D}"
    }
}

// MARK: desk math

/// What a desk edit does to the numbers: h, h', the column length (out_norm) and the size of the memo change.
struct DeskMath: View {
    @Environment(AppState.self) private var state
    let h: Float?
    let edit: NeuronEdit?
    let outNorm: Float?

    var body: some View {
        let memoN = state.geometry.memoText
        VStack(alignment: .leading, spacing: 8) {
            FormulaRow("h = silu(gate\u{00B7}x) \u{00D7} (up\u{00B7}x)", [
                ("x", "the memo: \(memoN) numbers arriving at this floor"),
                ("gate, up", "two lists of \(memoN) numbers this desk owns; **\u{00B7}** = multiply number by number and add up"),
                ("silu", "a soft on-switch: about 0 for negative inputs, about the input itself for large positive ones"),
                ("h", "how busy the desk is (one number). The dot's color shows it."),
            ])
            if let h {
                Text(markdown("On this word, **h = \(mathNum(h, 3))**.")).font(.callout)
            }
            FormulaRow("note change = \u{0394}h \u{00D7} column of W_down, then tidied", [
                ("W_down", "the table every desk writes through; this desk's **column** is its own \(memoN)-number note"),
                ("|column|", "that note's length, called **out_norm**" + (outNorm.map { ": here \(mathNum($0, 3))" } ?? "")),
                ("\u{0394}h", "how much you changed h (h\u{2032} \u{2212} h)"),
                ("tidied", "on the way out the floor rescales all its desks' notes to a standard size and multiplies in its own weights, so what reaches the memo is this change rescaled (Follow this desk shows the exact tidied column, step 5)"),
            ])
            if let edit {
                changeLine(label: editName(edit), hp: newH(edit))
            } else {
                changeLine(label: "Turn off (h\u{2032} = 0)", hp: 0)
                changeLine(label: "Turn up (h\u{2032} = 3h)", hp: (h ?? 0) * 3)
            }
        }
    }

    private func newH(_ e: NeuronEdit) -> Float {
        switch e.mode {
        case .ablate: return 0
        case .scale: return (h ?? 0) * e.value
        case .pin: return e.value
        }
    }

    private func editName(_ e: NeuronEdit) -> String {
        switch e.mode {
        case .ablate: return "Your change: turned off (h\u{2032} = 0)"
        case .scale: return "Your change: h\u{2032} = \(mathNum(e.value, 1))h"
        case .pin: return "Your change: held at h\u{2032} = v = \(mathNum(e.value))"
        }
    }

    @ViewBuilder private func changeLine(label: String, hp: Float) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption.weight(.semibold))
            if let h {
                let dh = hp - h
                let size = outNorm.map { abs(dh) * $0 }
                Text("h\u{2032} = \(mathNum(hp, 3)),  \u{0394}h = \(mathNum(hp, 3)) \u{2212} \(mathNum(h, 3)) = \(mathNum(dh, 3))")
                    .font(.system(.caption, design: .monospaced))
                if let size, let on = outNorm {
                    Text("note changes by |\u{0394}h| \u{00D7} out_norm = \(mathNum(abs(dh), 3)) \u{00D7} \(mathNum(on, 3)) = \(mathNum(size, 3)), before the tidy-up")
                        .font(.system(.caption, design: .monospaced))
                    Text(markdown("For scale: the memo grows on the way up, from about 8 long entering floor 0 to about 14 at floor 11 and 36 at floor 15 (measured at the word \u{201C}La\u{201D}). One desk's change is a small part of that."))
                        .font(.caption).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else {
                Text("Pick a word in the timeline to see the real numbers.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

}

// MARK: toy example

/// The 3-number toy memo, worked by hand: ink, erase, check, add.
struct ToyExample: View {
    @Environment(AppState.self) private var state
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Try it by hand").font(.callout.weight(.semibold))
            Text("A real memo has \(state.geometry.memoText) numbers. Here is the same math with only 3, so you can do it on paper.")
                .font(.caption).foregroundStyle(.secondary)
            line("1. The memo and the highlighter", "x = (2, 5, 1)    d\u{0302} = (0, 0.6, 0.8)",
                 "d\u{0302} has length 1: 0\u{00B2} + 0.6\u{00B2} + 0.8\u{00B2} = 0 + 0.36 + 0.64 = 1, so it is a pure direction.")
            line("2. How much ink? (dot product)", "x\u{00B7}d\u{0302} = 2\u{00D7}0 + 5\u{00D7}0.6 + 1\u{00D7}0.8 = 0 + 3 + 0.8 = 3.8",
                 "Multiply number by number, then add up. 3.8 is how much highlighter this memo carries.")
            line("3. Erase it", "x_new = x \u{2212} 3.8 \u{00D7} d\u{0302} = (2, 5, 1) \u{2212} (0, 2.28, 3.04) = (2, 2.72, \u{2212}2.04)",
                 "Take away exactly the ink, pointed along the highlighter.")
            line("4. Check", "x_new\u{00B7}d\u{0302} = 2\u{00D7}0 + 2.72\u{00D7}0.6 + (\u{2212}2.04)\u{00D7}0.8 = 0 + 1.632 \u{2212} 1.632 = 0",
                 "No ink left. The first number (2) never pointed along d\u{0302}, so it did not change: everything that is not highlighter stays.")
            line("5. Add instead (\u{03B1} = 2)", "x_new = x + 2 \u{00D7} d\u{0302} = (2, 6.2, 2.6);  ink = 3.8 + 2 = 5.8",
                 "Steering adds exactly \u{03B1} to the ink: 6.2\u{00D7}0.6 + 2.6\u{00D7}0.8 = 3.72 + 2.08 = 5.8.")
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.03), in: RoundedRectangle(cornerRadius: 8))
    }

    private func line(_ title: String, _ math: String, _ plain: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption.weight(.semibold))
            Text(math).font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.answer)
                .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            Text(plain).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: direction math (shared by Change and the custom sheet)

/// Step 1 + 2 of a direction change: how the highlighter is found, then the erase/add formula with real numbers.
struct DirectionMath: View {
    let contrast: ContrastResult
    let site: Int
    let mode: DirectionMode
    let alpha: Float?

    private var norm: Float { site < contrast.norms.count ? contrast.norms[site] : 0 }
    private var sites: Int { contrast.norms.count }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            MathStep(1, "How the highlighter is found",
                     plain: "We ran \(contrast.countA) questions that ask for the idea (**A**) and \(contrast.countB) matched ones that do not (**B**), and read the memo at \(siteName(site)), at the last word of each question. Average each pile, then subtract: what is left is the highlighter.") {
                FormulaRow("d = mean(A memos) \u{2212} mean(B memos)", [
                    ("d", "the highlighter: a list of \(Geometry.grouped(contrast.hiddenSize)) numbers, the average difference"),
                    ("|d|", "its length (its **norm**): \u{221A}(d\u{2081}\u{00B2} + d\u{2082}\u{00B2} + \u{2026}). Here **|d| = \(mathNum(norm))**"),
                ])
                FormulaRow("d\u{0302} = d / |d|", [
                    ("d\u{0302}", "\u{201C}d-hat\u{201D}: the same direction scaled to length 1, a pure direction"),
                ])
                FormulaRow("ink = x\u{00B7}d\u{0302}", [
                    ("x", "the memo at one floor and word (\(Geometry.grouped(contrast.hiddenSize)) numbers)"),
                    ("\u{00B7}", "dot product: multiply number by number, add up. Big = lots of this highlighter"),
                ])
            }
            if mode == .ablate {
                MathStep(2, "Erase it (abliteration)",
                         plain: "Subtract exactly the ink, pointed along the highlighter. This filter runs at all \(sites) stops of the elevator (entering floor 0 and after every floor), because the model rebuilds some of the idea after every erase.") {
                    FormulaRow("x_new = x \u{2212} (x\u{00B7}d\u{0302}) d\u{0302}", [
                        ("x\u{00B7}d\u{0302}", "the ink this memo carries (one number)"),
                        ("(x\u{00B7}d\u{0302}) d\u{0302}", "that much highlighter, as a list of \(Geometry.grouped(contrast.hiddenSize)) numbers"),
                        ("result", "x_new\u{00B7}d\u{0302} = 0, and every part of x that does not point along d\u{0302} is unchanged"),
                    ])
                    Text(markdown("The highlighter came from \(siteName(site)) (site \(site)), the floor whose tryout worked best. The same d\u{0302} is then erased on every floor."))
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            } else {
                let a = alpha ?? 1.5 * norm
                MathStep(2, "Add it (steering)",
                         plain: "On \(siteName(site)) only, add \u{03B1} units of the highlighter to every memo passing by. The floors above read it as if you had asked for the idea.") {
                    FormulaRow("x_new = x + \u{03B1} d\u{0302}", [
                        ("\u{03B1}", "alpha: how much ink to add. Here **\u{03B1} = \(mathNum(a))**, which is \(mathNum(norm > 0 ? a / norm : 0, 1)) \u{00D7} |d| (|d| = \(mathNum(norm)))"),
                        ("result", "the ink x\u{00B7}d\u{0302} rises by exactly \u{03B1}; nothing else in the memo moves"),
                    ])
                    Text("The app tries \u{03B1} = 1.0, 1.5 and 2.0 \u{00D7} |d|: too little does nothing, too much breaks the reply.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}
