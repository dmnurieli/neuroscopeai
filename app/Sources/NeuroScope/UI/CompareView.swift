import SwiftUI

private final class CompareMemo {
    var key = ""
    var div: Int?
    var sides: [String: AttributedString] = [:]
}

struct CompareView: View {
    @Environment(AppState.self) private var state
    @State private var memo = CompareMemo()

    var body: some View {
        if let edited = state.activeRun, let base = state.baselineRun, edited.id != base.id {
            let key = "\(edited.id)|\(edited.count)|\(base.id)|\(base.count)"
            let div: Int? = {
                if memo.key != key { memo.key = key; memo.div = Analysis.firstDivergence(edited, base); memo.sides = [:] }
                return memo.div
            }()
            VStack(spacing: 6) {
                chips(edited.edits)
                header(edited, base, div)
                HStack(spacing: 1) {
                    side("Baseline", base, div)
                    side("Edited", edited, div)
                }
                shiftStrip(edited, base)
            }
            .padding(8)
            .background(Theme.background)
        } else {
            Text("Run an edited replay to compare").foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func chips(_ e: EditSet) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                ForEach(e.neurons, id: \.self) { n in
                    chip("L\(n.layer).\(n.index) \(n.mode.rawValue)\(n.mode == .ablate ? "" : " \(n.value)")", mode: n.mode)
                }
                ForEach(e.directions) { d in
                    chip("\(d.label) \(d.mode == .ablate ? "remove" : "add \(String(format: "%.2f", d.value))")", mode: d.mode == .ablate ? .ablate : .scale)
                }
                ForEach(e.heads, id: \.self) { h in
                    chip("H\(h.layer).\(h.head) \(h.mode.rawValue)\(h.mode == .ablate ? "" : " \(h.value)")", mode: h.mode)
                }
            }
        }
    }

    private func chip(_ s: String, mode: EditMode) -> some View {
        let c: Color = mode == .ablate ? Theme.ablated : mode == .scale ? Theme.scaled : Theme.pinned
        return Text(s).font(.caption2.monospaced()).padding(.horizontal, 6).padding(.vertical, 2)
            .background(c.opacity(0.25), in: Capsule()).overlay(Capsule().stroke(c.opacity(0.7)))
    }

    private func header(_ e: Run, _ b: Run, _ div: Int?) -> some View {
        func tok(_ r: Run) -> String {
            guard let d = div, d < r.tokens.count else { return "(end)" }
            return r.tokens[d].text.debugDescription
        }
        let s = div.map { "Diverged at token \($0): \(tok(b)) -> \(tok(e))" }
            ?? "Identical output - the edit did not change any token"
        return Text(s).font(.callout.bold())
            .foregroundStyle(div == nil ? Theme.negative : Theme.ablated)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func side(_ title: String, _ run: Run, _ div: Int?) -> some View {
        let ck = "\(title)|\(run.id)|\(run.count)"
        var out = memo.sides[ck] ?? AttributedString()
        if memo.sides[ck] == nil {
        for tok in run.tokens where tok.stream == .thinking || tok.stream == .answer {
            var p = AttributedString(tok.text.isEmpty ? " " : tok.text)
            p.foregroundColor = Theme.color(for: tok.stream)
            p.font = tok.stream == .thinking ? .system(size: 13).italic() : .system(size: 13)
            if let d = div {
                if tok.index == d { p.backgroundColor = Theme.ablated.opacity(0.65); p.underlineStyle = .single }
                else if tok.index > d { p.backgroundColor = Theme.scaled.opacity(0.15) }
            }
            out.append(p)
        }
        memo.sides[ck] = out
        }
        return VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption.bold()).foregroundStyle(.secondary)
            ScrollView { Text(out).frame(maxWidth: .infinity, alignment: .leading) }
        }
        .padding(8).frame(maxWidth: .infinity, maxHeight: .infinity).background(Theme.panel)
    }

    private func shiftStrip(_ e: Run, _ b: Run) -> some View {
        let t = state.displayIndex
        let shift: [Float] = {
            guard let t, t < e.count, t < b.count else { return [] }
            return Analysis.layerShift(e, b, at: t)
        }()
        let mx = max(shift.max() ?? 0, 1e-6)
        return VStack(alignment: .leading, spacing: 2) {
            Text("Layer shift at token \(t.map(String.init) ?? "-")").font(.caption2).foregroundStyle(.secondary)
            HStack(alignment: .bottom, spacing: 2) {
                ForEach(0..<max(shift.count, 1), id: \.self) { i in
                    let v = i < shift.count ? CGFloat(shift[i] / mx) : 0
                    Rectangle().fill(Theme.scaled).frame(height: max(1, 36 * v))
                }
            }
            .frame(height: 36, alignment: .bottom)
        }
    }
}
