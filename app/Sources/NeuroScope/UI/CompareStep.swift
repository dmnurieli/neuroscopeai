import SwiftUI

/// Step 4: Before | After answers side by side, a plain sentence on what changed, and the diff field.
struct CompareStep: View {
    @Environment(AppState.self) private var state

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            StepHeader(title: "Compare",
                       explainer: "The same question, answered before and after your change. The field shows which desks changed.")
            LearnCard([
                "Read the two replies side by side: **Before** is the building as it normally is, **After** is the building with your change. Same question, same settings, so any difference comes from your change.",
                "In the field, **red** means a desk worked harder after your change, and **blue** means it worked less. Dim desks barely changed.",
                "Sometimes nothing changes. The building often has backups: other desks and floors carry the same idea, so erasing or adding a little is not enough. Try a stronger change, another floor, or the suggested question.",
                "Watch for **side effects** too: a change aimed at one idea can also make the reply odd, repeat itself, or drift off topic. That tells you the highlighter also carried other things.",
            ], words: [
                ("red / blue", "a desk worked harder / less after your change"),
                ("side effect", "a change you did not aim for, like a reply that repeats or drifts"),
            ])
            if let before = state.beforeRun {
                let after = state.afterRun
                Text(sentence(before, after)).font(.callout.weight(.medium))
                    .fixedSize(horizontal: false, vertical: true)
                HStack(alignment: .top, spacing: 8) {
                    column("Before", before)
                    if let after { column("After", after) }
                    else { placeholder }
                }
                if !state.edits.isEmpty { mathSection } else { legend }
            } else {
                Text("Nothing to compare yet. Ask a question first, then make a change.").foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            Button { state.step = .change } label: {
                Label("Try another change", systemImage: "arrow.uturn.backward").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
            .disabled(state.isGenerating)
            Button { state.startOver() } label: {
                Text("Start over").frame(maxWidth: .infinity)
            }
            .controlSize(.large)
            .disabled(state.isGenerating)
        }
    }

    private func sentence(_ before: Run, _ after: Run?) -> String {
        guard let after else { return "Asking again with your change…" }
        if before.status == .running || after.status == .running {
            return after.status == .running ? "Asking again with your change…" : "Asking the original building first…"
        }
        guard let d = Analysis.firstDivergence(after, before) else {
            return "The reply did not change. Other desks covered for it; try a stronger change or a different question."
        }
        let word = max(1, d - before.promptCount + 1)
        func tok(_ r: Run) -> String {
            guard d < r.tokens.count else { return "(the end)" }
            let t = r.tokens[d].text.trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? "(a space)" : "\u{201C}\(t)\u{201D}"
        }
        if word == 1 { return "The reply changed from the very first word: \(tok(before)) became \(tok(after))." }
        return "The reply changed starting at word \(word): \(tok(before)) became \(tok(after))."
    }

    private func column(_ title: String, _ run: Run) -> some View {
        let thinking = plainStreamText(run, .thinking)
        let answer = plainStreamText(run, .answer)
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                if run.status == .running { ProgressView().controlSize(.mini) }
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    if !thinking.isEmpty {
                        DisclosureGroup {
                            Text(thinking).font(.system(size: 11).italic())
                                .foregroundStyle(Theme.thinking.opacity(0.75))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                        } label: {
                            Text("Thinking (scratch paper)").font(.caption2).foregroundStyle(Theme.thinking)
                        }
                    }
                    Text(answer.isEmpty ? (run.status == .running ? "…" : "(no reply)") : answer)
                        .font(.system(size: 12)).foregroundStyle(Theme.answer)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
            }
            .frame(minHeight: 160, maxHeight: 320)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(Theme.background, in: RoundedRectangle(cornerRadius: 8))
        .contentShape(Rectangle())
        .onTapGesture { state.showRun(run.id) }
        .overlay(RoundedRectangle(cornerRadius: 8)
            .stroke(state.activeRunID == run.id ? Theme.positive.opacity(0.6) : Color.clear))
        .help("Show this reply's desks in the field")
        .accessibilityLabel("\(title) reply. Show its desks in the field.")
    }

    private var placeholder: some View {
        VStack { ProgressView(); Text("Waiting for the reply…").font(.caption).foregroundStyle(.secondary) }
            .frame(maxWidth: .infinity, minHeight: 160)
            .background(Theme.background, in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: show the math

    private var dirMode: DirectionMode? { state.edits.directions.first?.mode }

    private var needsMeasure: Bool {
        guard dirMode != nil else { return false }
        return state.inkBefore == nil || state.inkAfter == nil || state.mathSample == nil
    }

    private var canMeasure: Bool {
        guard let a = state.afterRun, a.status == .finished, let b = state.beforeRun, b.status == .finished else { return false }
        return !state.isMeasuringMath && !state.isGenerating
    }

    private var legend: some View {
        HStack(spacing: 14) {
            legendDot(Theme.ablated, "desk worked harder after")
            legendDot(Theme.negative, "desk worked less after")
        }
    }

    @ViewBuilder private var mathSection: some View {
        MathDisclosure("More: colors and the math") {
            VStack(alignment: .leading, spacing: 16) {
                legend
                if let mode = dirMode {
                    inkPart(mode)
                    samplePart(mode)
                    measureButton
                } else {
                    Text("Your change edits desks, not a highlighter. Each desk edit changes its h, and the memo moves by \u{0394}h \u{00D7} the desk's column (open the desk card for its real numbers).")
                        .font(.callout).fixedSize(horizontal: false, vertical: true)
                }
                ToyExample()
            }
            .task(id: state.afterRun?.status == .finished) {
                if state.showMath, needsMeasure, canMeasure { await state.measureMath() }
            }
        }
    }

    @ViewBuilder private func inkPart(_ mode: DirectionMode) -> some View {
        MathStep(1, "How much ink, floor by floor",
                 plain: "For every floor we measured the **ink**, x\u{00B7}d\u{0302} (how much highlighter the memo carries), at each word of the reply and averaged it. " + (mode == .ablate
                    ? "The dashed line is what arrives at each floor before the filter; the blue line is what leaves after it."
                    : "The dashed line is what arrives at each floor; the blue line is what leaves, so on the inked floor it jumps by \u{03B1}.")) {
            if state.inkBefore != nil || state.inkAfter != nil {
                InkChart(before: state.inkBefore, after: state.inkAfter, mode: mode)
                if let line = inkSentence(mode) {
                    Text(markdown(line)).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            Text(markdown(mode == .ablate
                ? "Measured on this engine (Remove French): normally the French ink stays near zero on the low floors, 0.40 after floor 11, then jumps to 2.80 after floor 12 and reaches 3.92 at the top. With the filter on, ink **arriving** after floor 12 is still 2.69, and after each filter about 0.002. The model rebuilds some French after every erase, which is **why the filter must run on every floor**."
                : "Adding ink on one floor jumps the line there by exactly \u{03B1}; the floors above then carry it (and change it) on their own."))
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func inkSentence(_ mode: DirectionMode) -> String? {
        guard let b = state.inkBefore?.arriving.last else { return nil }
        var s = "This run: before your change the top floor carried **\(mathNum(b))** ink."
        if let a = state.inkAfter, let arr = a.arriving.last, let lv = a.leaving.last {
            s += mode == .ablate
                ? " After it, **\(mathNum(arr))** arrives at the top and **\(mathNum(lv))** leaves after the filter."
                : " After it, the top floor carries **\(mathNum(lv))**."
        }
        return s
    }

    @ViewBuilder private func samplePart(_ mode: DirectionMode) -> some View {
        if let s = state.mathSample {
            let word = sampleWord(s.position)
            MathStep(2, "One memo, worked through",
                     plain: "Here is one real memo: \(siteName(s.site)) (site \(s.site)), at the word \(word). It has \(state.geometry.memoText) numbers; here are the first 6.") {
                let total = state.geometry.hiddenSize
                NumbersRow(label: "x", values: Array(s.x.prefix(6)), total: total, note: "the memo before the edit")
                NumbersRow(label: "d\u{0302}", values: Array(s.dHat.prefix(6)), total: total, digits: 3,
                           note: "the highlighter, length 1 (from d with |d| = \(mathNum(s.dNorm)))")
                FormulaRow(dotLine(s), [("x\u{00B7}d\u{0302}", "multiply number by number and add up all \(state.geometry.memoText) products: the ink at this word")])
                FormulaRow(mode == .ablate
                           ? "x_new = x \u{2212} (\(mathNum(s.dot))) \u{00D7} d\u{0302}"
                           : "x_new = x + \(mathNum(state.experimentValue ?? (s.dotAfter - s.dot))) \u{00D7} d\u{0302}",
                           [("first number", firstNumberLine(s, mode))])
                NumbersRow(label: "x_new", values: Array(s.xNew.prefix(6)), total: total, note: "the memo after the edit")
                Text("Ink before: x\u{00B7}d\u{0302} = \(mathNum(s.dot))   \u{2192}   after: x_new\u{00B7}d\u{0302} = \(mathNum(s.dotAfter))")
                    .font(.system(.callout, design: .monospaced))
                Text("Length: |x| = \(mathNum(s.xNorm))   \u{2192}   |x_new| = \(mathNum(s.xNewNorm))")
                    .font(.system(.callout, design: .monospaced))
                Text(markdown(mode == .ablate
                    ? "The memo barely shrinks: only the highlighter part (\(mathNum(abs(s.dot))) out of a length of \(mathNum(s.xNorm))) was removed; everything else is untouched. The model stores numbers at reduced precision (**bf16**, about 3 significant digits), so \u{201C}after\u{201D} reads about 0.00\u{2013}0.05, not exactly 0."
                    : "The ink rose by exactly \u{03B1}; every other direction in the memo is unchanged. Numbers are stored at reduced precision (**bf16**, about 3 significant digits), so the last digit can be off a little."))
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func dotLine(_ s: ProjectSample) -> String {
        let terms = zip(s.x.prefix(3), s.dHat.prefix(3)).map { "(\(mathNum($0, 2))\u{00D7}\(mathNum($1, 3)))" }
        return "x\u{00B7}d\u{0302} = " + terms.joined(separator: " + ") + " + \u{2026} = \(mathNum(s.dot))"
    }

    private func firstNumberLine(_ s: ProjectSample, _ mode: DirectionMode) -> String {
        guard let x0 = s.x.first, let d0 = s.dHat.first, let n0 = s.xNew.first else { return "" }
        if mode == .ablate {
            return "\(mathNum(x0)) \u{2212} (\(mathNum(s.dot)) \u{00D7} \(mathNum(d0, 3))) = \(mathNum(x0 - s.dot * d0)) (measured: \(mathNum(n0)))"
        }
        let a = state.experimentValue ?? (s.dotAfter - s.dot)
        return "\(mathNum(x0)) + (\(mathNum(a)) \u{00D7} \(mathNum(d0, 3))) = \(mathNum(x0 + a * d0)) (measured: \(mathNum(n0)))"
    }

    private func sampleWord(_ position: Int) -> String {
        guard let r = state.afterRun, position >= 0, position < r.tokens.count else { return "#\(position)" }
        let t = r.tokens[position].displayText.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? "(a space)" : "\u{201C}\(t)\u{201D}"
    }

    @ViewBuilder private var measureButton: some View {
        if state.isMeasuringMath {
            HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Measuring the ink on every floor\u{2026}").font(.caption).foregroundStyle(.secondary) }
        } else if needsMeasure {
            Button { Task { await state.measureMath() } } label: {
                Label("Measure the math", systemImage: "ruler").frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(!canMeasure)
            .help("Measure the ink x\u{00B7}d\u{0302} at every floor for the Before and After replies")
        }
    }

    private func legendDot(_ c: Color, _ text: String) -> some View {
        HStack(spacing: 5) {
            Circle().fill(c).frame(width: 9, height: 9)
            Text(text).font(.caption).foregroundStyle(.secondary)
        }
    }
}
