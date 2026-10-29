import SwiftUI
import AirlineKit

// The lesson's choreography: every page action is a RUN, a timeline of shots. Each shot frames part of the map
// (the camera) and says what traces. The map draws a run by reading its elapsed time each frame; nothing here
// moves on its own. Every moving thing renders a real number from the engine (a contribution, a gradient, a
// tally); the timings only decide WHEN each one is shown.

// MARK: - easing

enum Ease {
    static func clamp(_ x: Double) -> Double { min(max(x, 0), 1) }
    /// 0…1 between `a` and `b`.
    static func ramp(_ t: Double, _ a: Double, _ b: Double) -> Double { b <= a ? (t >= b ? 1 : 0) : clamp((t - a) / (b - a)) }
    static func inOut(_ x: Double) -> Double {
        let u = clamp(x)
        return u < 0.5 ? 4 * u * u * u : 1 - pow(-2 * u + 2, 3) / 2
    }
    static func out(_ x: Double) -> Double { let u = clamp(x); return 1 - pow(1 - u, 3) }
    static func smooth(_ x: Double) -> Double { let u = clamp(x); return u * u * (3 - 2 * u) }
}

// MARK: - the camera

/// What a shot frames, resolved against the map's geometry at draw time (so a resized window reframes itself).
enum Focus: Equatable {
    case wide            // the whole map
    case clipboard       // the fact cards
    case clipboardDials  // the fact cards and the "how unusual" dials beside them
    case between(Int)    // the routes from layer n to layer n+1, with the stops at both ends
    case gate            // the gate and the routes arriving at it
    case gateScale       // the gate plus the minutes scale under it
    case gateRoutes      // the last routes, the gate and the minutes scale (hotter or colder)
}

struct MapCamera: Equatable {
    var center: CGPoint   // map coordinates
    var scale: CGFloat

    static let minScale: CGFloat = 1
    static let maxScale: CGFloat = 2.4

    /// As close as fits (1…2.4×) with the whole rect visible, centred on it.
    static func framing(_ rect: CGRect, in size: CGSize, padding: CGFloat = 12) -> MapCamera {
        let fit = min((size.width - 2 * padding) / max(rect.width, 1), (size.height - 2 * padding) / max(rect.height, 1))
        return MapCamera(center: CGPoint(x: rect.midX, y: rect.midY), scale: min(max(fit, minScale), maxScale))
    }
    static func wide(_ size: CGSize) -> MapCamera { MapCamera(center: CGPoint(x: size.width / 2, y: size.height / 2), scale: 1) }

    /// Centre moves linearly, zoom moves in log space (so zooming in and out feel the same speed).
    func mix(_ to: MapCamera, _ u: Double) -> MapCamera {
        let k = CGFloat(u)
        return MapCamera(center: CGPoint(x: center.x + (to.center.x - center.x) * k, y: center.y + (to.center.y - center.y) * k),
                      scale: exp(log(scale) + (log(to.scale) - log(scale)) * k))
    }

    func apply(_ ctx: inout GraphicsContext, size: CGSize) {
        ctx.translateBy(x: size.width / 2, y: size.height / 2)
        ctx.scaleBy(x: scale, y: scale)
        ctx.translateBy(x: -center.x, y: -center.y)
    }
}

/// From `at` seconds into the run, ease the camera to `focus` over `move` seconds.
struct CameraKey: Equatable {
    var at: Double
    var focus: Focus
    var move: Double
}

// MARK: - a run

/// One choreographed sequence. Immutable once started except `skipped` (the learner pressed the button: jump to
/// the end state).
struct AirlineRun {
    enum Kind: Equatable { case frame, drawIn, guess, reveal, walk, together, unseen }

    /// Where the camera starts: a resting focus, or wherever an interrupted run's camera was.
    indirect enum Origin {
        case focus(Focus)
        case midway(AirlineRun, Double)
    }

    let id: Int
    let kind: Kind
    let start: Date
    let duration: Double
    /// When the caption's numbers may appear (the moment the sequence reaches them).
    let captionAt: Double
    let keys: [CameraKey]
    var origin: Origin
    var skipped: Bool

    // What the run draws (precomputed once, read every frame).
    var forward: ForwardPlan?         // guess, unseen, the walk's replay (the network after the step)
    var forwardTiming: ForwardTiming?
    var before: ForwardPlan?          // together: the hub fills before the step
    var walk: WalkPlan?               // the backpropagation walk's numbers
    var stage: WalkStage?             // which walk page this run belongs to
    var flipFrom: [Float]?            // unseen: the previous flight's facts, flipped over

    func elapsed(at date: Date) -> Double {
        skipped ? duration : min(max(date.timeIntervalSince(start), 0), duration)
    }
    func isPlaying(at date: Date) -> Bool { !skipped && date.timeIntervalSince(start) < duration }
    var endFocus: Focus { keys.last?.focus ?? .wide }

    /// The focus the camera was heading for at `t` (used to flatten an interrupted run's origin).
    func focus(at t: Double) -> Focus {
        var f: Focus = { if case .focus(let f) = origin { return f }; return .wide }()
        for k in keys where k.at <= t { f = k.focus }
        return f
    }

    func camera(at t: Double, _ g: MapGeometry) -> MapCamera {
        var cam: MapCamera
        switch origin {
        case .focus(let f): cam = g.camera(f)
        case .midway(let r, let e): cam = r.camera(at: e, g)
        }
        for (n, k) in keys.enumerated() {
            guard t >= k.at else { break }
            let until = n + 1 < keys.count ? min(t, keys[n + 1].at) : t
            cam = cam.mix(g.camera(k.focus), Ease.inOut((until - k.at) / max(k.move, 0.001)))
        }
        return cam
    }
}

// MARK: - the forward pass, traced

/// When each part of a forward trace happens. `f` compresses the whole pass (1 = the full guess, ~4.7 s).
struct ForwardTiming {
    let start: Double
    let f: Double
    let sizes: [Int]
    private var layerStart: [Double] = [], stagger: [Double] = [], travelTime: [Double] = []
    private var decisionTime: [Double] = []   // per layer index (1…hidden), when its gate agents decide
    private(set) var countStart: Double = 0, countEnd: Double = 0

    static let decisionLength = 0.3
    static let tail = 0.28   // fraction of a route the fading tail covers

    init(start: Double, compress f: Double, sizes: [Int]) {
        self.start = start
        self.f = f
        self.sizes = sizes
        var t = 0.15
        decisionTime = [0]
        for l in 0..<(sizes.count - 1) {
            let st = l == 0 ? 0.08 : 0.03, tr = l == 0 ? 0.9 : (l == sizes.count - 2 ? 0.6 : 0.7)
            layerStart.append(t); stagger.append(st); travelTime.append(tr)
            let arrival = t + st * Double(max(sizes[l] - 1, 0)) + tr
            if l + 1 < sizes.count - 1 {
                decisionTime.append(arrival + 0.05)
                t = arrival + 0.4
            } else {
                countStart = arrival + 0.05
                countEnd = countStart + 0.6
            }
        }
    }

    private func at(_ x: Double) -> Double { start + f * x }
    var end: Double { at(countEnd) }
    var countWindow: (Double, Double) { (at(countStart), at(countEnd)) }
    /// The fact card i lights up as its passengers leave.
    func cardPeak(_ i: Int) -> Double { at(0.08 * Double(i) + 0.05) }
    func headBegin(_ l: Int, _ i: Int) -> Double { at(layerStart[l] + stagger[l] * Double(i)) }
    func travel(_ l: Int) -> Double { f * travelTime[l] }
    /// 0…1 of the head's trip along route layer l from source i (beyond 1 while the tail drains in).
    func progress(_ l: Int, _ i: Int, _ t: Double) -> Double { (t - headBegin(l, i)) / max(travel(l), 0.001) }
    /// 0…1 of the gate agents' decision for hidden layer index k (1…).
    func decided(_ k: Int, _ t: Double) -> Double {
        Ease.ramp(t, at(decisionTime[k]), at(decisionTime[k]) + f * Self.decisionLength)
    }
    var routeLayers: Int { sizes.count - 1 }

    /// MapCamera keys that follow the passengers: clipboard, each gap as the traces cross it, the gate.
    func followKeys(fromClipboard: Bool) -> [CameraKey] {
        let move = max(0.55, 0.8 * f)
        var keys: [CameraKey] = fromClipboard ? [CameraKey(at: start, focus: .clipboard, move: max(0.5, 0.6 * f))] : []
        keys.append(CameraKey(at: at(0.75), focus: .between(0), move: move))
        for l in 1..<max(routeLayers, 1) {
            let focus: Focus = l + 1 == routeLayers ? .gate : .between(l)
            keys.append(CameraKey(at: headBegin(l, 0) - 0.1 * f, focus: focus, move: move))
        }
        return keys
    }
}

/// The numbers a forward trace draws: every route's contribution (passengers actually carried), each hub's
/// tally and whether its gate agent let it through. Built once per run from the engine's own forward pass.
final class ForwardPlan {
    let weights: [[Float]]
    let sizes: [Int]
    /// Per route layer, flat [i * out + j]: activation_i × weight_ij.
    let contribution: [[Float]]
    /// |contribution| / the layer's largest, 0…1.
    let share: [[Float]]
    /// |standardised input| / the largest, 0…1 (how far each fact sits from an ordinary flight).
    let inputShare: [Float]
    let bias: [[Float]]
    let pre: [[Float]]        // per hidden row: the tally incl. resting stress
    let level: [[Double]]     // per hidden row: fill saturation (MapModel.hubLevel)
    let guess: Float          // minutes

    init(network: Network, flight: Int, ledger: Ledger) {
        let acts = network.forward(flight: flight, in: ledger)
        let sizes = network.layerSizes
        self.sizes = sizes
        weights = network.weights
        bias = network.biases
        pre = acts.preActivation
        level = acts.hubs.map { row in row.indices.map { MapModel.hubLevel(row, $0) } }
        guess = acts.gateMinutes
        var contribution: [[Float]] = [], share: [[Float]] = []
        for l in 0..<(sizes.count - 1) {
            let from: [Float] = l == 0 ? acts.inputs : acts.hubs[l - 1]
            let out = sizes[l + 1]
            var c = [Float](repeating: 0, count: sizes[l] * out)
            for i in 0..<sizes[l] { for j in 0..<out { c[i * out + j] = from[i] * network.weights[l][i * out + j] } }
            let top = max(c.map(abs).max() ?? 0, 1e-6)
            contribution.append(c)
            share.append(c.map { abs($0) / top })
        }
        self.contribution = contribution
        self.share = share
        let topIn = max(acts.inputs.map(abs).max() ?? 0, 1e-6)
        inputShare = acts.inputs.map { abs($0) / topIn }
    }
}

// MARK: - the shot lists

/// Durations and camera moves per page action. Times in seconds from the run's start.
enum Shots {
    static let frameMove = 0.8

    /// A plain camera move to a page's resting framing.
    static func frame(_ focus: Focus, move: Double = frameMove) -> (keys: [CameraKey], duration: Double) {
        ([CameraKey(at: 0, focus: focus, move: move)], move)
    }

    // p2: routes draw in layer by layer, then the dials spin and settle on the real random weights.
    static let drawLength = 0.5
    static func drawBegin(_ l: Int, _ i: Int) -> Double { 0.1 + 0.6 * Double(l) + (l == 0 ? 0.05 : 0.03) * Double(i) }
    static let jitter: (Double, Double) = (2.05, 2.6)
    static let drawInDuration = 2.65

    // p4: the hand lifts, the scale shows the gap.
    static let revealGuess: (Double, Double) = (0.3, 0.6)
    static let revealReal: (Double, Double) = (0.9, 1.4)
    static let revealBracket: (Double, Double) = (1.4, 2.1)
    static let revealDuration = 2.2

    // p7: cards flip, a compressed forward pass, then guess and real side by side.
    static func flip(_ i: Int) -> (Double, Double) { let a = 0.25 + 0.06 * Double(i); return (a, a + 0.3) }
    static let unseenForwardStart = 1.0
    static let unseenCompress = 0.55

    // the walk (AirlineWalk): per page, when its pieces arrive
    static let walkCamera = 0.8
    static let gateArrow: (Double, Double) = (0.6, 1.2)
    static let gateBlameDuration = 1.3
    /// routes pages: the rule card's route lights, chips appear, thicknesses ease to their new values
    static let ruleRoute = 0.5
    static func chip(_ n: Int) -> Double { 1.1 + 0.07 * Double(n) }
    static let thicken: (Double, Double) = (1.9, 2.7)
    static let routesDuration = 2.8
    /// hubs pages: blame flows back along the routes, meters fill, resting stress dials tick
    static let backFlow: (Double, Double) = (0.3, 1.0)
    static func meter(_ j: Int) -> (Double, Double) { let a = 0.9 + 0.06 * Double(j); return (a, a + 0.6) }
    static let dial: (Double, Double) = (1.7, 2.5)
    static let hubsDuration = 2.6
    static let chipCount = 10
    /// Clipboard routes crowd near the cards: fewer chips there, spread along their routes.
    static func chips(layer l: Int) -> Int { l == 0 ? 5 : chipCount }
    static func chipAlong(layer l: Int, rank: Int) -> Double { l == 0 ? 0.22 + 0.13 * Double(rank) : 0.3 + 0.1 * Double(rank % 3) }

    // the walk's last page: replay with every route at its new size
    static let replayStart = 0.5
    static let replayCompress = 0.4
}
