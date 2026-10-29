import SwiftUI
import AirlineKit

/// Lesson 1's rulebook editor. Each rule reads as a sentence made of pickers:
/// "If [Wind] is [above] [30 mph] and [Rain] is [above] [1.0 in], then [add 60] minutes."
/// Up to 3 rules, 1-2 conditions each. Under the rules: a bar chart on the CHECK split (the 2023 quiz pile).
///
/// Notes:
/// - Holiday week: one picker, "is a holiday week" / "is not" (encoded as above/below 0.5).
/// - Pilot's shoes: comparison offers "above", "below" and, when the rule has only that one condition, "is".
///   "is <colour>" is stored as TWO conditions (above c-0.5 AND below c+0.5), rendered back as one clause;
///   it fills the rule, so "+ and..." is hidden and its x removes the whole rule. Plain above/below use colour
///   names as thresholds (above "Red" means Blue or White).
/// - Score is recomputed off the main actor, debounced 100 ms, on every edit.
struct RulebookEditorView: View {
    @Binding var rulebook: Rulebook
    let ledger: Ledger
    /// Optional third bar ("A network"): nil shows "?".
    var networkScore: Float? = nil

    @State private var baseline: Float?
    @State private var yours: Float?

    private let maxRules = 3

    var body: some View {
        editor
            .controlSize(.large)   // sized for reading at a lesson's distance
    }

    private var editor: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(spacing: 10) {
                ForEach(Array(rulebook.rules.indices), id: \.self) { r in
                    ruleCard(r)
                }
                if rulebook.rules.count < maxRules {
                    Button { addRule() } label: {
                        Label("Add a rule", systemImage: "plus")
                            .font(.system(size: 15))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(12)
                            .background(RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .strokeBorder(Color.white.opacity(0.14), style: StrokeStyle(lineWidth: 1, dash: [4, 4])))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(Theme.thinking)
                    .accessibilityLabel("Add a rule")
                }
            }
            scoreCard
        }
        .task { baseline = await Self.score(Rulebook(), ledger) }
        .task(id: rulebook) {
            try? await Task.sleep(for: .milliseconds(100))
            if Task.isCancelled { return }
            let v = await Self.score(rulebook, ledger)
            if !Task.isCancelled { yours = v }
        }
    }

    private static func score(_ book: Rulebook, _ ledger: Ledger) async -> Float {
        await Task.detached(priority: .userInitiated) { book.meanMinutesOff(.check, in: ledger) }.value
    }

    // MARK: Rules

    private func addRule() {
        guard rulebook.rules.count < maxRules else { return }
        rulebook.rules.append(Rule(conditions: [Condition(fact: .wind, above: true, threshold: 30)], addMinutes: 30))
    }

    private func ruleCard(_ r: Int) -> some View {
        let rule = rulebook.rules[r]
        let rows = clauses(of: rule)
        return HStack(alignment: .top, spacing: 8) {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(rows.indices, id: \.self) { k in
                    conditionRow(rule: r, clause: rows[k], isFirst: k == 0)
                }
                HStack(spacing: 6) {
                    Text(rows.isEmpty ? "Then" : "then").foregroundStyle(.secondary)
                    minutesPicker(r)
                    Text("minutes").foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                    if canAddAnd(rule) {
                        Button { addCondition(r) } label: { Text("+ and\u{2026}").font(.system(size: 15)) }
                            .buttonStyle(.plain).foregroundStyle(Theme.thinking)
                            .accessibilityLabel("Add a second condition to rule \(r + 1)")
                    }
                }
            }
            Button { rulebook.rules.remove(at: r) } label: {
                Image(systemName: "xmark").font(.system(size: 13, weight: .semibold)).padding(6)
            }
            .buttonStyle(.plain).foregroundStyle(.secondary)
            .accessibilityLabel("Remove rule \(r + 1)")
        }
        .font(.system(size: 17))
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color.white.opacity(0.08)))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Rule \(r + 1): \(sentence(rule))")
    }

    /// A clause is one visible "[Fact] is [above] [value]" row: a single condition, or a shoe "is" pair.
    private struct Clause { var indices: [Int]; var shoeIs: Int? }

    private func clauses(of rule: Rule) -> [Clause] {
        if let c = shoePair(rule) { _ = c; return [Clause(indices: [0, 1], shoeIs: shoePair(rule))] }
        return rule.conditions.indices.map { Clause(indices: [$0], shoeIs: nil) }
    }

    private func shoePair(_ rule: Rule) -> Int? {
        guard rule.conditions.count == 2 else { return nil }
        let a = rule.conditions[0], b = rule.conditions[1]
        guard a.fact == .shoe, b.fact == .shoe, a.above, !b.above else { return nil }
        let c = Int((a.threshold + 0.5).rounded())
        return abs(b.threshold - (Float(c) + 0.5)) < 0.01 ? c : nil
    }

    private func canAddAnd(_ rule: Rule) -> Bool { rule.conditions.count == 1 }

    private func addCondition(_ r: Int) {
        guard rulebook.rules[r].conditions.count == 1 else { return }
        let used = rulebook.rules[r].conditions[0].fact
        let fact = Fact.allCases.first { $0 != used && $0 != .shoe } ?? .rain
        rulebook.rules[r].conditions.append(Condition(fact: fact, above: true, threshold: Self.defaultThreshold(fact)))
    }

    private func conditionRow(rule r: Int, clause: Clause, isFirst: Bool) -> some View {
        let first = clause.indices[0]
        let cond = rulebook.rules[r].conditions[first]
        let canRemove = clause.shoeIs == nil && rulebook.rules[r].conditions.count == 2
        return HStack(spacing: 6) {
            Text(isFirst ? "If" : "and").foregroundStyle(.secondary).frame(minWidth: 18, alignment: .leading)
            Picker("Fact for rule \(r + 1)", selection: Binding(
                get: { cond.fact },
                set: { setFact($0, rule: r, index: first, wasPair: clause.shoeIs != nil) })) {
                ForEach(Fact.allCases, id: \.self) { Text($0.title).tag($0) }
            }
            .labelsHidden().fixedSize()
            .accessibilityLabel("Fact for rule \(r + 1)")
            valuePickers(rule: r, clause: clause, cond: cond)
            if canRemove {
                Button { rulebook.rules[r].conditions.remove(at: first) } label: {
                    Image(systemName: "xmark.circle").font(.system(size: 15))
                }
                .buttonStyle(.plain).foregroundStyle(.secondary)
                .accessibilityLabel("Remove this condition from rule \(r + 1)")
            }
            Spacer(minLength: 0)
        }
    }

    private func setFact(_ f: Fact, rule r: Int, index i: Int, wasPair: Bool) {
        if wasPair { rulebook.rules[r].conditions = [] ; rulebook.rules[r].conditions.append(Condition(fact: f, above: true, threshold: Self.defaultThreshold(f))); return }
        rulebook.rules[r].conditions[i] = Condition(fact: f, above: true, threshold: Self.defaultThreshold(f))
    }

    @ViewBuilder
    private func valuePickers(rule r: Int, clause: Clause, cond: Condition) -> some View {
        let i = clause.indices[0]
        if cond.fact == .holiday {
            Picker("Holiday for rule \(r + 1)", selection: Binding(
                get: { cond.above },
                set: { rulebook.rules[r].conditions[i] = Condition(fact: .holiday, above: $0, threshold: 0.5) })) {
                Text("is a holiday week").tag(true)
                Text("is not a holiday week").tag(false)
            }
            .labelsHidden().fixedSize()
            .accessibilityLabel("Holiday week for rule \(r + 1)")
        } else {
            Text("is").foregroundStyle(.secondary)
            Picker("Comparison for rule \(r + 1)", selection: Binding<Int>(
                get: { clause.shoeIs != nil ? 2 : (cond.above ? 0 : 1) },
                set: { setComparison($0, rule: r, index: i, cond: cond) })) {
                Text("above").tag(0)
                Text("below").tag(1)
                if cond.fact == .shoe && rulebook.rules[r].conditions.count == 1 || clause.shoeIs != nil {
                    Text("exactly").tag(2)
                }
            }
            .labelsHidden().fixedSize()
            .accessibilityLabel("Comparison for rule \(r + 1)")
            thresholdPicker(rule: r, clause: clause, cond: cond)
        }
    }

    private func setComparison(_ v: Int, rule r: Int, index i: Int, cond: Condition) {
        if v == 2 {
            let c = min(max(Int(cond.threshold.rounded()), 0), 4)
            rulebook.rules[r].conditions = Self.shoeIs(c)
        } else if rulebook.rules[r].conditions.count == 2, rulebook.rules[r].conditions[0].fact == .shoe, rulebook.rules[r].conditions[1].fact == .shoe {
            let c = Int(rulebook.rules[r].conditions[0].threshold + 0.5)
            rulebook.rules[r].conditions = [Condition(fact: .shoe, above: v == 0, threshold: Float(c))]
        } else {
            rulebook.rules[r].conditions[i].above = (v == 0)
        }
    }

    private static func shoeIs(_ c: Int) -> [Condition] {
        [Condition(fact: .shoe, above: true, threshold: Float(c) - 0.5),
         Condition(fact: .shoe, above: false, threshold: Float(c) + 0.5)]
    }

    private func thresholdPicker(rule r: Int, clause: Clause, cond: Condition) -> some View {
        let opts = Self.options(for: cond.fact)
        let i = clause.indices[0]
        let current: Float = clause.shoeIs.map(Float.init) ?? cond.threshold
        let sel = nearest(opts, to: current)
        return Picker("Value for rule \(r + 1)", selection: Binding(
            get: { sel },
            set: { n in
                let v = opts[n]
                if clause.shoeIs != nil { rulebook.rules[r].conditions = Self.shoeIs(Int(v)) }
                else { rulebook.rules[r].conditions[i].threshold = v }
            })) {
            ForEach(opts.indices, id: \.self) { Text(cond.fact.format(opts[$0])).tag($0) }
        }
        .labelsHidden().fixedSize()
        .accessibilityLabel("Value for rule \(r + 1)")
    }

    private func nearest(_ opts: [Float], to v: Float) -> Int {
        opts.indices.min { abs(opts[$0] - v) < abs(opts[$1] - v) } ?? 0
    }

    private static func options(for f: Fact) -> [Float] {
        switch f {
        case .rain: return stride(from: 0, through: 3, by: 0.25).map { Float($0) }
        case .wind: return stride(from: 0, through: 60, by: 5).map { Float($0) }
        case .controllers: return (4...14).map { Float($0) }
        case .load: return stride(from: 50, through: 100, by: 5).map { Float($0) / 100 }
        case .holiday: return [0.5]
        case .hour: return (6...22).map { Float($0) }
        case .crews: return (0...6).map { Float($0) }
        case .shoe: return (0...4).map { Float($0) }
        }
    }

    private static func defaultThreshold(_ f: Fact) -> Float {
        switch f {
        case .rain: 1; case .wind: 30; case .controllers: 8; case .load: 0.9
        case .holiday: 0.5; case .hour: 17; case .crews: 2; case .shoe: 2
        }
    }

    private func minutesPicker(_ r: Int) -> some View {
        let opts = Array(stride(from: -30, through: 240, by: 10))
        return Picker("Minutes for rule \(r + 1)", selection: Binding(
            get: { opts.min { abs(Float($0) - rulebook.rules[r].addMinutes) < abs(Float($1) - rulebook.rules[r].addMinutes) } ?? 0 },
            set: { rulebook.rules[r].addMinutes = Float($0) })) {
            ForEach(opts, id: \.self) { Text($0 < 0 ? "take off \(-$0)" : "add \($0)").tag($0) }
        }
        .labelsHidden().fixedSize()
        .accessibilityLabel("Minutes for rule \(r + 1)")
    }

    private func sentence(_ rule: Rule) -> String {
        var parts: [String] = []
        for cl in clauses(of: rule) {
            let c = rule.conditions[cl.indices[0]]
            if let s = cl.shoeIs { parts.append("pilot's shoes are \(Fact.shoe.format(Float(s)))"); continue }
            if c.fact == .holiday { parts.append(c.above ? "it is a holiday week" : "it is not a holiday week"); continue }
            parts.append("\(c.fact.title) is \(c.above ? "above" : "below") \(c.fact.format(c.threshold))")
        }
        let m = Int(rule.addMinutes.rounded())
        let tail = m < 0 ? "take off \(-m) minutes" : "add \(m) minutes"
        return "If \(parts.joined(separator: " and ")), \(tail)"
    }

    // MARK: Scores

    private var scoreCard: some View {
        let rows: [(String, Float?, Color)] = [
            ("Everyone on time", baseline, Theme.special),
            ("Your rules", yours, Theme.prompt),
            ("A network", networkScore, Theme.scaled),
        ]
        let top = max(rows.compactMap { $0.1 }.max() ?? 1, 1)
        return VStack(alignment: .leading, spacing: 10) {
            Text("How far off, on average, on the 2023 quiz flights").font(.system(size: 15)).foregroundStyle(.secondary)
            ForEach(rows.indices, id: \.self) { k in
                let row = rows[k]
                HStack(spacing: 10) {
                    Text(row.0).font(.system(size: 15)).frame(width: 150, alignment: .leading)
                    GeometryReader { g in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color.white.opacity(0.05))
                            if let v = row.1 {
                                Capsule().fill(row.2.opacity(0.85))
                                    .frame(width: max(6, g.size.width * CGFloat(v / top)))
                            }
                        }
                    }
                    .frame(height: 10)
                    Text(row.1.map { "off by \(Int($0.rounded())) min" } ?? "?")
                        .font(.system(size: 15).monospacedDigit()).foregroundStyle(Theme.answer)
                        .frame(width: 100, alignment: .trailing)
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(row.0)
                .accessibilityValue(row.1.map { "off by \(Int($0.rounded())) minutes on average" } ?? "not available yet")
            }
            Text("Shorter bar is better.").font(.system(size: 13)).foregroundStyle(.secondary)
            if let y = yours, let b = baseline, y > b + 0.5 {
                Text("Right now your rules do worse than guessing everyone is on time. Try a different threshold, or fewer minutes.")
                    .font(.system(size: 15)).foregroundStyle(Theme.positive)
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color.white.opacity(0.08)))
    }
}
