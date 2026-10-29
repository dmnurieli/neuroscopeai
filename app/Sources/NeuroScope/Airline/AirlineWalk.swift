import SwiftUI
import AirlineKit

// The backpropagation walk: the one-flight nudge, taught one layer at a time, right to left. Every number comes
// from the engine's single LearnStep on the showcase flight (its `hubBlame`, its route changes) and from the
// network's forward pass on that flight before the step.
//
// One unit scheme throughout, stated on the first routes page:
//   how busy  = a stop's activation (a hub's passengers after its gate agent; for a fact, how far it sits from an
//               ordinary flight, in standard units), one decimal;
//   blame     = blame points: the backward signal scaled so the gate's blame reads 1.0, signed (+ = should have
//               been higher / busier, − = lower / quieter);
//   change    = route change in the same points: how busy × blame. The real nudge is that × a small step
//               (the line-searched stride × the gate's raw blame), so every tile on the rule card multiplies exactly.

enum WalkStage: Equatable {
    case knob            // hotter or colder: turn one route by hand
    case gateBlame       // the gate's blame
    case routes(Int)     // route layer n changes
    case hubs(Int)       // the hubs of layer n receive blame, their resting stress moves
    case together        // everything at its new value; replay

    /// The walk for a network with `routeLayers` route layers: right to left.
    static func stages(routeLayers: Int) -> [WalkStage] {
        var out: [WalkStage] = [.knob, .gateBlame]
        for l in stride(from: routeLayers - 1, through: 0, by: -1) {
            out.append(.routes(l))
            if l > 0 { out.append(.hubs(l)) }
        }
        out.append(.together)
        return out
    }
}

final class WalkPlan {
    let sizes: [Int]
    let before: Network
    let after: Network
    /// The line-searched stride the step used.
    let stride: Float
    /// |the gate's raw backward signal|: blame points × this = the engine's ∂loss/∂(pre-activation).
    let gateScale: Float
    /// Per layer 0…L−1 (clipboard, then hub rows): how busy each stop was on this flight, before the step.
    let busy: [[Float]]
    /// Per hidden row: the tally before the gate agent (≤ 0 = closed).
    let pre: [[Float]]
    /// Per layer 1…L (index k − 1): blame points, signed (+ = should have been higher / busier).
    let blame: [[Float]]
    /// Per route layer, flat [i * out + j]: busy_i × blame_j, in blame points.
    let change: [[Float]]
    /// Per route layer: indices by |change|, biggest first.
    let ranked: [[Int]]
    /// The route into the gate with the largest |gradient| (the hotter-or-colder knob).
    let knob: RouteID
    let guessBefore: Float, guessAfter: Float, truth: Float
    let flight: Int

    init(step: LearnStep, before: Network, after: Network, stride: Float, ledger: Ledger) {
        let sizes = before.layerSizes
        self.sizes = sizes
        self.before = before
        self.after = after
        self.stride = stride
        flight = step.flight
        let acts = before.forward(flight: step.flight, in: ledger)
        busy = [acts.inputs] + acts.hubs
        pre = acts.preActivation
        let g = abs(step.hubBlame.last?.first ?? 1)
        gateScale = max(g, 1e-6)
        blame = step.hubBlame.map { row in row.map { -$0 / max(g, 1e-6) } }
        var change: [[Float]] = []
        for l in 0..<(sizes.count - 1) {
            let out = sizes[l + 1]
            var c = [Float](repeating: 0, count: sizes[l] * out)
            for i in 0..<sizes[l] { for j in 0..<out { c[i * out + j] = busy[l][i] * blame[l][j] } }
            change.append(c)
        }
        self.change = change
        ranked = change.map { c in c.indices.sorted { abs(c[$0]) != abs(c[$1]) ? abs(c[$0]) > abs(c[$1]) : $0 < $1 } }
        let last = sizes.count - 2
        let into = step.changes.filter { $0.route.layer == last }.max { $0.blame < $1.blame }
        knob = into?.route ?? RouteID(layer: last, from: 0, to: 0)
        guessBefore = step.guessBefore
        guessAfter = step.guessAfter
        truth = step.truth
    }

    var routeLayers: Int { sizes.count - 1 }
    /// The real weight change per blame point: the small step.
    var smallStep: Float { stride * gateScale }
    func blame(layer k: Int, _ j: Int) -> Float { blame[k - 1][j] }
    func isClosed(layer k: Int, _ j: Int) -> Bool { k >= 1 && k < sizes.count - 1 && pre[k - 1][j] <= 0 }

    /// Name of a stop for copy: a fact's title, a hub's code, "the gate".
    func name(layer k: Int, _ i: Int) -> String {
        if k == 0 { return before.architecture.inputs[i].title }
        if k == sizes.count - 1 { return "the gate" }
        return MapModel.hubName(row: k - 1, index: i)
    }

    /// The weight the knob page starts from, and how far the knob may turn: enough to reach the true delay and
    /// overshoot it a little, so both warmer and colder are reachable.
    var knobRange: ClosedRange<Float> {
        let w = before.weights[knob.layer][knob.from * sizes[knob.layer + 1] + knob.to]
        let a = max(busy[knob.layer][knob.from], 0.05)
        let needed = abs(truth - guessBefore) / (a * before.delaySD)
        let reach = min(max(1.5, 1.3 * needed), 8)
        return (w - reach)...(w + reach)
    }
    var knobStart: Float { before.weights[knob.layer][knob.from * sizes[knob.layer + 1] + knob.to] }
}

// MARK: - the rule card (in the caption)

struct RuleTile {
    var label: String
    var value: String
    var note: String?
    /// −1…1: a small bar under the value (nil: none).
    var bar: Double?
    var blame = false
}

struct RuleCard {
    var tiles: [RuleTile]       // three: A × B = C
    var footnote: String?
}

enum WalkFormat {
    /// Blame points and changes: one decimal (two below 0.1), always signed, real minus sign.
    static func points(_ v: Float) -> String {
        let a = abs(v)
        let body = a >= 10 ? String(format: "%.0f", Double(a)) : (a >= 0.095 || a == 0 ? String(format: "%.1f", Double(a)) : String(format: "%.2f", Double(a)))
        return (v < 0 ? "−" : "+") + body
    }
    /// How busy: one decimal; signed only for facts (they can sit below an ordinary flight).
    static func busy(_ v: Float, signed: Bool) -> String {
        let s = String(format: signed ? "%+.1f" : "%.1f", Double(v))
        return s.replacingOccurrences(of: "-", with: "−")
    }
    static func arrow(_ v: Float) -> String { v > 0 ? "↑" : (v < 0 ? "↓" : "·") }
    static func chevron(_ v: Float) -> String { v > 0 ? "▲" : (v < 0 ? "▼" : "") }
}
