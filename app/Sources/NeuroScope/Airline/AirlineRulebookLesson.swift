import SwiftUI
import AirlineKit

/// Lesson 1, "The rulebook": the learner writes if-then rules, scores them on the 2023 quiz pile, meets the
/// near-miss flight where no rule fires, and sees why combinations outrun hand-written rules.
@MainActor @Observable
final class AirlineRulebookLesson: AirlineLessonModel {
    let title = AirlineLessonID.rulebook.title
    let pageCount = 5
    let chor = Choreographer()

    private(set) var ledger: Ledger?
    private(set) var page = 0
    private(set) var everyoneOnTime: Float?
    private(set) var rulesScore: Float?
    var rulebook: Rulebook = AirlineProgress.rulebook {
        didSet {
            AirlineProgress.rulebook = rulebook
            rescore()
        }
    }
    private var scoreTask: Task<Void, Never>?

    var canGoBack: Bool { page > 0 }
    var canWatchAgain: Bool { page == 3 && !chor.reduceMotion && !chor.isPlaying }
    var networkScore: Float? { AirlineProgress.networkScore }

    init() { chor.summary = { [weak self] in self?.summary } }

    func load() async {
        guard ledger == nil else { return }
        let l = await AirlineLedger.load()
        ledger = l
        everyoneOnTime = Rulebook().meanMinutesOff(.check, in: l)
        rescore()
    }

    func stop() { scoreTask?.cancel() }

    private func rescore() {
        guard let l = ledger else { return }
        let book = rulebook
        nearMiss = l.nearMissFlight(for: book)
        scoreTask?.cancel()
        scoreTask = Task { [weak self] in
            let s = await Task.detached(priority: .userInitiated) { book.meanMinutesOff(.check, in: l) }.value
            guard !Task.isCancelled else { return }
            self?.rulesScore = s
        }
    }

    /// The near-miss flight for the learner's own rules (re-picked whenever the rulebook changes).
    private(set) var nearMiss: Int?

    // MARK: the pages

    /// The facts and pairs a rule writer would have to cover.
    static var factCount: Int { Fact.allCases.count }
    static var pairs: Int { factCount * (factCount - 1) / 2 }
    static var triples: Int { factCount * (factCount - 1) * (factCount - 2) / 6 }

    var copy: AirlinePageCopy {
        guard let l = ledger else {
            return AirlinePageCopy(text: "Loading the logbook…", button: "Continue", enabled: false, working: true)
        }
        switch page {
        case 0:
            return AirlinePageCopy(
                text: "Before networks, programmers wrote **rules** by hand: if this, then that.",
                detail: "A rule is an instruction a person writes in advance. The computer just follows it.",
                button: "Continue")
        case 1:
            return AirlinePageCopy(
                text: "Write your own rulebook: up to three rules.",
                detail: "The bars show how far off your rules are, on average, on the \(l.indices(.check).count.formatted()) flights from 2023.",
                button: "Continue")
        case 2:
            guard let i = nearMiss else {
                return AirlinePageCopy(text: "Rules stay quiet just below their thresholds.", detail: nil, button: "Continue")
            }
            return AirlinePageCopy(text: nearMissSentence(i, l), detail: nearMissDetail(i, l), button: "Continue")
        case 3:
            return AirlinePageCopy(
                text: "Rules miss the in-between cases, and the combinations.",
                detail: "\(Self.factCount) facts make \(Self.pairs) pairs and \(Self.triples) triples, each needing its own thresholds. Nobody can write them all by hand.",
                button: "Continue")
        default:
            let net = networkScore.map { "Your two-row network got to \(AirlineState.minutes($0)) min off." }
                ?? "You will fill in the network's bar yourself, in Layover hubs."
            return AirlinePageCopy(
                text: "A network finds the combinations by itself, from the logbook.",
                detail: net, button: "Done")
        }
    }

    /// "Wind was 29 mph, not above 30, so your rule stayed quiet. It was really 95 min late."
    private func nearMissSentence(_ i: Int, _ l: Ledger) -> String {
        let fired = rulebook.firingRules(flight: i, in: l)
        let truth = AirlineState.phrase(l.delay(i))
        if fired.isEmpty, let miss = closestMiss(i, l) {
            let c = miss.condition
            return "\(c.fact.title) was \(c.fact.format(l.fact(c.fact, i))), not \(c.above ? "above" : "below") \(c.fact.format(c.threshold)), so your rule stayed quiet. It was really \(truth)."
        }
        if fired.isEmpty { return "None of your rules fired on this flight. It was really \(truth)." }
        return "\(fired.count == 1 ? "One rule" : "\(fired.count) rules") fired here. Your rules guessed \(AirlineState.phrase(rulebook.predict(flight: i, in: l))); it was really \(truth)."
    }

    private func nearMissDetail(_ i: Int, _ l: Ledger) -> String {
        let f = l.flight(i)
        let facts = [Fact.wind, .rain, .controllers].map { "\($0.title.lowercased()) \($0.format(f.facts[$0.rawValue]))" }.joined(separator: ", ")
        return "A stormy, short-staffed flight from 2023 (\(facts)). Your rules guessed \(AirlineState.phrase(rulebook.predict(flight: i, in: l)))."
    }

    /// The failed condition closest to its threshold (in standard units): the near miss.
    func closestMiss(_ i: Int, _ l: Ledger) -> (rule: Int, condition: Condition)? {
        l.closestMiss(flight: i, of: rulebook).map { ($0.rule, $0.condition) }
    }

    func primary() -> Bool {
        if chor.isPlaying { chor.skip(); return false }
        if page == pageCount - 1 { return true }
        go(to: page + 1)
        return false
    }

    func back() { if page > 0 { go(to: page - 1) } }
    func watchAgain() { if page == 3 { startCount() } }

    private func go(to p: Int) {
        page = p
        if p == 3 { startCount() } else { chor.frame(.wide, move: 0.01) }
    }

    /// 8 → 28 → 56, counted up in turn.
    private func startCount() {
        chor.start(.reveal, keys: [CameraKey(at: 0, focus: .wide, move: 0.01)], duration: 2.0, captionAt: 0)
    }

    var summary: String? {
        page == 3 ? "\(Self.factCount) facts, \(Self.pairs) pairs, \(Self.triples) triples." : nil
    }
}

// MARK: - the view

struct AirlineRulebookLessonView: View {
    var onExit: (_ completed: Bool) -> Void
    @State private var model = AirlineRulebookLesson()

    var body: some View {
        AirlineLessonShell(model: model, onExit: onExit) {
            Group {
                if let l = model.ledger {
                    stage(l)
                } else {
                    Spacer(minLength: 0)
                }
            }
            .frame(maxWidth: 900, maxHeight: .infinity)
            .padding(.horizontal, 28)
        } accessory: { _ in EmptyView() }
        .task { await model.load() }
    }

    @ViewBuilder private func stage(_ l: Ledger) -> some View {
        switch model.page {
        case 0:
            Spacer(minLength: 0)
            if let rule = Rulebook.beginner.rules.first { RuleSentence(rule: rule, state: nil).scaleEffect(1.25) }
            Spacer(minLength: 0)
        case 1:
            // centred at a reading size; scrolls only when three two-condition rules outgrow the window
            let editor = RulebookEditorView(rulebook: Bindable(model).rulebook, ledger: l, networkScore: model.networkScore)
                .frame(maxWidth: 760)
                .padding(.vertical, 18)
            ViewThatFits(in: .vertical) {
                editor.frame(maxHeight: .infinity)
                ScrollView { editor.frame(maxWidth: .infinity) }
            }
        case 2:
            if let i = model.nearMiss {
                VStack(spacing: 18) {
                    FlightCardView(flight: l.flight(i), revealed: true)
                    FactGrid(flight: l.flight(i))
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(Array(model.rulebook.rules.enumerated()), id: \.offset) { n, rule in
                            RuleSentence(rule: rule, state: RuleSentence.Check(flight: i, ledger: l,
                                                                                 fired: model.rulebook.firingRules(flight: i, in: l).contains(n)))
                        }
                        if model.rulebook.rules.isEmpty {
                            Text("Your rulebook is empty: it always guesses on time.").foregroundStyle(.secondary)
                        }
                    }
                }
                .frame(maxHeight: .infinity)
            } else {
                Spacer(minLength: 0)
            }
        case 3:
            CombinationCounter(run: model.chor.run, active: model.chor.timelineActive)
                .frame(maxHeight: .infinity)
        default:
            ScoreBars(bars: [
                .init(label: "Everyone on time", minutes: model.everyoneOnTime),
                .init(label: "Your rules", minutes: model.rulesScore),
                .init(label: "A network", minutes: model.networkScore, highlight: true),
            ])
            .frame(maxHeight: .infinity)
        }
    }
}

/// One rule as a sentence; with a flight, each condition is checked (✓ held, ✗ did not) and the rule shows
/// whether it fired.
struct RuleSentence: View {
    struct Check { let flight: Int; let ledger: Ledger; let fired: Bool }
    let rule: Rule
    let state: Check?

    var body: some View {
        HStack(spacing: 8) {
            if let s = state {
                Image(systemName: s.fired ? "checkmark.circle.fill" : "moon.zzz")
                    .foregroundStyle(s.fired ? Theme.prompt : Color.white.opacity(0.45))
                    .accessibilityLabel(s.fired ? "fired" : "stayed quiet")
            }
            Text("If").foregroundStyle(.secondary)
            ForEach(Array(rule.conditions.enumerated()), id: \.offset) { n, c in
                if n > 0 { Text("and").foregroundStyle(.secondary) }
                condition(c)
            }
            Text("add \(Int(rule.addMinutes.rounded())) minutes").foregroundStyle(.secondary)
        }
        .font(.system(size: 15))
        .padding(.horizontal, 16).padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Color.white.opacity(0.1)))
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder private func condition(_ c: Condition) -> some View {
        let clause = "\(c.fact.title.lowercased()) is \(c.above ? "above" : "below") \(c.fact.format(c.threshold))"
        if let s = state {
            let v = s.ledger.fact(c.fact, s.flight)
            let holds = c.above ? v > c.threshold : v < c.threshold
            HStack(spacing: 4) {
                Text(clause).fontWeight(.semibold)
                Text("(\(c.fact.format(v)) \(holds ? "✓" : "✗"))")
                    .foregroundStyle(holds ? Theme.prompt : MapModel.blame)
                    .monospacedDigit()
            }
        } else {
            Text(clause).fontWeight(.semibold).foregroundStyle(Theme.answer)
        }
    }
}

/// The flight's eight facts, as on the clipboard.
struct FactGrid: View {
    let flight: Flight
    var body: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 10), count: 4), spacing: 8) {
            ForEach(Fact.allCases, id: \.self) { f in
                HStack {
                    Text(f.title).font(.system(size: 12)).foregroundStyle(.secondary)
                    Spacer(minLength: 4)
                    Text(f.format(flight.facts[f.rawValue])).font(.system(size: 13, weight: .medium, design: .monospaced))
                        .foregroundStyle(Theme.answer)
                }
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Theme.panel))
            }
        }
        .frame(maxWidth: 760)
        .accessibilityElement(children: .combine)
    }
}

/// 8 facts → 28 pairs → 56 triples, each number counting up in turn (end state at once under Reduce Motion).
private struct CombinationCounter: View {
    let run: AirlineRun?
    let active: Bool

    var body: some View {
        SwiftUI.TimelineView(.animation(minimumInterval: nil, paused: !active)) { tl in
            let t = run?.elapsed(at: active ? tl.date : Date()) ?? 10
            HStack(spacing: 26) {
                number(AirlineRulebookLesson.factCount, "facts", Ease.out(Ease.ramp(t, 0, 0.4)))
                arrow(Ease.ramp(t, 0.4, 0.6))
                number(AirlineRulebookLesson.pairs, "pairs", Ease.out(Ease.ramp(t, 0.5, 1.1)))
                arrow(Ease.ramp(t, 1.1, 1.3))
                number(AirlineRulebookLesson.triples, "triples", Ease.out(Ease.ramp(t, 1.2, 1.9)))
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(AirlineRulebookLesson.factCount) facts, \(AirlineRulebookLesson.pairs) pairs, \(AirlineRulebookLesson.triples) triples")
    }

    private func number(_ n: Int, _ label: String, _ u: Double) -> some View {
        VStack(spacing: 4) {
            Text("\(Int((Double(n) * u).rounded()))")
                .font(.system(size: 54, weight: .semibold, design: .rounded))
                .foregroundStyle(Theme.answer)
                .monospacedDigit()
            Text(label).font(.system(size: 14)).foregroundStyle(.secondary)
        }
        .opacity(0.25 + 0.75 * u)
    }

    private func arrow(_ u: Double) -> some View {
        Image(systemName: "arrow.right").font(.system(size: 22)).foregroundStyle(.secondary).opacity(u)
    }
}
