import SwiftUI
import AirlineKit

/// The airline map: the clipboard on the left, rows of layover hubs, the destination gate on the right, and a
/// route between every stop of neighbouring columns. Route thickness = |weight| (scaled per layer), amber solid =
/// adds delay, blue dashed = relief flight. A hub's fill SATURATES with its activation (grey at rest → amber);
/// nothing glows.
///
/// The whole map sits under one camera. Page actions play as runs (AirlineChoreography): the map reads the run's
/// elapsed time every frame (TimelineView) and draws the real numbers it carries: passengers per route on the way
/// forward; on the backpropagation walk (AirlineWalk), each layer's blame and route changes. The timeline only
/// runs while something moves.
struct AirlineMapView: View {
    /// The frame's snapshot, given Reduce Motion (taken on the main actor, once per model change).
    let sceneFor: (Bool) -> MapScene?
    let timelineActive: Bool
    var summary: String? = nil
    /// Editing: tapping a clipboard card switches that fact on or off (the camera stays wide on such pages).
    var onToggleFact: ((Fact) -> Void)? = nil
    /// Tapping the map anywhere else: the point in map coordinates (through the camera) and the geometry to hit-test.
    var onTapMap: ((CGPoint, MapGeometry) -> Void)? = nil

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    init(sceneFor: @escaping (Bool) -> MapScene?, timelineActive: Bool, summary: String? = nil,
         onToggleFact: ((Fact) -> Void)? = nil, onTapMap: ((CGPoint, MapGeometry) -> Void)? = nil) {
        self.sceneFor = sceneFor
        self.timelineActive = timelineActive
        self.summary = summary
        self.onToggleFact = onToggleFact
        self.onTapMap = onTapMap
    }

    init(model: AirlineState) {
        self.init(sceneFor: { model.mapScene(reduceMotion: $0) }, timelineActive: model.timelineActive,
                  summary: model.sequenceSummary)
    }

    var body: some View {
        if let scene = sceneFor(reduceMotion) {
            SwiftUI.TimelineView(.animation(minimumInterval: nil, paused: !timelineActive)) { timeline in
                // Paused means every run has reached its end state, so "now" is simply the present.
                let now = timelineActive ? timeline.date : Date()
                ZStack {
                    Canvas { ctx, size in
                        let painter = MapPainter(scene: scene, size: size, now: now)
                        painter.withCamera(&ctx) { painter.drawClipboard(&$0) }
                    }
                    Canvas { ctx, size in
                        let painter = MapPainter(scene: scene, size: size, now: now)
                        painter.withCamera(&ctx) { painter.drawNetwork(&$0) }
                    }
                    .opacity(scene.networkVisible ? 1 : 0)
                    .animation(.easeInOut(duration: 0.6), value: scene.networkVisible)
                    Canvas { ctx, size in
                        MapPainter(scene: scene, size: size, now: now).drawEdgeFade(&ctx)
                    }
                    .allowsHitTesting(false)
                }
            }
            .overlay { if let toggle = onToggleFact { cardButtons(scene, toggle) } }
            .overlay {
                if let tap = onTapMap {
                    GeometryReader { geo in
                        Color.clear
                            .contentShape(Rectangle())
                            .gesture(SpatialTapGesture().onEnded { e in
                                let painter = MapPainter(scene: scene, size: geo.size, now: Date())
                                tap(painter.toMap(e.location), MapGeometry.of(scene, size: geo.size))
                            })
                    }
                }
            }
            .clipped()
            .accessibilityElement(children: .contain)
            .accessibilityChildren { accessibilitySummary(scene) }
        }
    }

    /// Invisible buttons over the clipboard cards (geometry at the wide camera).
    private func cardButtons(_ scene: MapScene, _ toggle: @escaping (Fact) -> Void) -> some View {
        GeometryReader { geo in
            let g = MapGeometry.of(scene, size: geo.size)
            let painter = MapPainter(scene: scene, size: geo.size, now: Date())
            ForEach(Fact.allCases, id: \.self) { f in
                let r = painter.toScreen(g.cardRect(slot: f.rawValue))
                let on = scene.network.architecture.inputs.contains(f)
                Button { toggle(f) } label: { Color.clear.contentShape(Rectangle()) }
                    .buttonStyle(.plain)
                    .frame(width: r.width, height: r.height)
                    .position(x: r.midX, y: r.midY)
                    .help(on ? "Switch \(f.title) off" : "Switch \(f.title) on")
                    .accessibilityLabel("\(f.title), \(on ? "on" : "off")")
                    .accessibilityHint(on ? "Switches this fact off" : "Switches this fact on")
            }
        }
    }

    // MARK: VoiceOver: the clipboard, each hub, a summary of the routes, the gate, what the last sequence showed

    @ViewBuilder private func accessibilitySummary(_ scene: MapScene) -> some View {
        let network = scene.network, flight = scene.flight
        VStack {
            Text("Clipboard: " + network.architecture.inputs.map { "\($0.title) \($0.format(flight.facts[$0.rawValue]))" }.joined(separator: ", "))
            if scene.networkVisible {
                ForEach(Array(network.architecture.hidden.enumerated()), id: \.offset) { row, count in
                    ForEach(0..<count, id: \.self) { k in
                        Text("\(MapModel.hubName(row: row, index: k)) hub, \(busyWord(scene.rest, row: row, index: k))")
                    }
                }
                Text(routeSummary(scene.weights))
                Text("Gate: " + (scene.rest.map { "guesses \(AirlineState.phrase($0.gateMinutes))" } ?? "no guess yet"))
            }
            if let s = summary { Text(s) }
        }
    }

    private func busyWord(_ a: Activations?, row: Int, index k: Int) -> String {
        guard let a, a.hubs.indices.contains(row), a.hubs[row].indices.contains(k) else { return "at rest" }
        let f = MapModel.hubLevel(a.hubs[row], k)
        return f > 0.6 ? "busy" : (f > 0.15 ? "a little busy" : "quiet")
    }

    private func routeSummary(_ weights: [[Float]]) -> String {
        let all = weights.flatMap { $0 }
        let up = all.filter { $0 > 0 }.count
        return "\(all.count) routes: \(up) amber routes add delay, \(all.count - up) dashed blue relief flights take it away."
    }
}

// MARK: - what one frame needs (a snapshot of the model, taken on the main actor)

struct MinutesScale {
    var guess: Float          // the guess being graded
    var truth: Float
    var newGuess: Float?      // the walk's last page: after the step
    var arrow = false         // the gate's blame page: an arrow instead of the bracket
}

struct LossPoint {
    var flights: Int
    var minutes: Float
    var at: Date
}

struct MapScene {
    var network: Network
    /// The weights to draw now (on the walk: each route layer before or after the step, by page).
    var weights: [[Float]]
    /// Route indices per layer, thinnest first (strong routes draw on top). Built once per snapshot.
    var routeOrder: [[Int]]
    /// Training only: the previous chunk's weights, eased from.
    var previousWeights: [[Float]]? = nil
    var snapshotDate: Date = .distantPast
    var flight: Flight
    /// The network on this flight, once it has guessed (nil: hubs grey, gate "?").
    var rest: Activations? = nil
    var networkVisible: Bool = true
    var run: AirlineRun? = nil
    var minutes: MinutesScale? = nil
    var loss: [LossPoint] = []
    var lossTotal: Int = 0
    /// Training: the routes the last chunk moved most (they flicker in the blame colour).
    var hotRoutes: Set<RouteID> = []
    var wasGuess: Float? = nil
    // the backpropagation walk
    var walk: WalkPlan? = nil
    var stage: WalkStage? = nil
    var knobWeight: Float? = nil
    var knobGuess: Float? = nil
    var reduceMotion: Bool = false
    // the course's other lessons
    /// Draw all eight clipboard cards in fixed slots; facts the network does not use are shown switched off.
    var allFacts: Bool = false
    /// A card that wobbles while the current frame run plays ("would the pilot's shoes matter?").
    var wobble: Fact? = nil
    /// Per fact (rawValue order): how far this flight's value is from an ordinary flight, in standard units.
    var zScores: [Float]? = nil
    /// A hub the page is about (row, index): ringed in the answer colour.
    var pointAt: (row: Int, index: Int)? = nil
    /// Clipboard-only pages: the cards gather into a tight column so the camera can come in close; they spread
    /// back out (over 0.8 s from `compactSince`) when the network appears.
    var compactCards: Bool = false
    var compactSince: Date = .distantPast
    /// Close a route: parts closed as an overlay (the network itself is unchanged; `rest` is the closed one's pass).
    var closedRoutes: Set<RouteID> = []
    var closedHubs: Set<HubRef> = []
}

enum MapModel {
    /// Eight per row (rows hold 1…8 hubs). Never one of the ledger's airports.
    static let hubNames = [["ATL", "ORD", "DFW", "DEN", "PHX", "SEA", "MSY", "BNA"],
                           ["CLT", "MSP", "DTW", "IAH", "SLC", "PHL", "STL", "PDX"]]
    static func hubName(row: Int, index k: Int) -> String {
        let names = hubNames[row % hubNames.count]
        return k < names.count ? names[k] : "H\(row + 1).\(k + 1)"
    }

    /// 0…1: how busy hub k is, against the busiest hub in its row (at least 1 standard unit, so a row of
    /// barely-active hubs stays pale instead of the largest of them looking busy).
    static func hubLevel(_ row: [Float], _ k: Int) -> Double {
        let top = max(row.max() ?? 0, 1)
        return Double(min(max(row[k] / top, 0), 1))
    }

    /// Route indices per layer sorted by |weight|, thinnest first.
    static func routeOrder(_ weights: [[Float]]) -> [[Int]] {
        weights.map { w in w.indices.sorted { abs(w[$0]) < abs(w[$1]) } }
    }

    /// One colour for blame everywhere (the walk, the loss bracket, practice's busiest routes).
    static let blame = Theme.ablated
    static let warmer = Theme.positive
    static let colder = Theme.negative
    /// What is not being explained on a walk page steps back to this opacity.
    static let dimmed = 0.25
}

// MARK: - the painter

struct MapPainter {
    let s: MapScene
    let g: MapGeometry
    let now: Date
    let t: Double          // the run's elapsed time (its duration once it ended or was skipped)
    let run: AirlineRun?

    init(scene: MapScene, size: CGSize, now: Date) {
        s = scene
        g = MapGeometry.of(scene, size: size, now: now)
        self.now = now
        run = scene.run
        t = scene.run?.elapsed(at: now) ?? 0
    }

    private var sizes: [Int] { g.sizes }
    private var routeLayers: Int { g.sizes.count - 1 }
    private static let tail = ForwardTiming.tail

    var camera: MapCamera {
        (s.reduceMotion || run == nil) ? .wide(g.size) : run!.camera(at: t, g)
    }

    /// A screen point back in map coordinates (taps).
    func toMap(_ p: CGPoint) -> CGPoint {
        let c = camera
        return CGPoint(x: (p.x - g.size.width / 2) / c.scale + c.center.x, y: (p.y - g.size.height / 2) / c.scale + c.center.y)
    }

    /// A map rect as it lands on screen under the current camera (for controls laid over the map).
    func toScreen(_ r: CGRect) -> CGRect {
        let c = camera
        return CGRect(x: (r.minX - c.center.x) * c.scale + g.size.width / 2, y: (r.minY - c.center.y) * c.scale + g.size.height / 2,
                      width: r.width * c.scale, height: r.height * c.scale)
    }

    func withCamera(_ ctx: inout GraphicsContext, _ draw: (inout GraphicsContext) -> Void) {
        var c = ctx
        camera.apply(&c, size: g.size)
        draw(&c)
    }

    /// When the camera is in close, stops and routes fade into the background at the frame's edges instead of
    /// being cut mid-word (drawn over everything, in screen space; no offscreen mask).
    func drawEdgeFade(_ ctx: inout GraphicsContext) {
        let cam = camera, wide = MapCamera.wide(g.size)
        let off = max(Double(cam.scale - 1) / 0.12, Double(hypot(cam.center.x - wide.center.x, cam.center.y - wide.center.y)) / 30)
        let k = min(max(off, 0), 1)
        guard k > 0.01 else { return }
        let w: CGFloat = 24, size = g.size
        let solid = Theme.background.opacity(k), clear = Theme.background.opacity(0)
        let edges: [(CGRect, CGPoint, CGPoint)] = [
            (CGRect(x: 0, y: 0, width: w, height: size.height), CGPoint(x: 0, y: 0), CGPoint(x: w, y: 0)),
            (CGRect(x: size.width - w, y: 0, width: w, height: size.height), CGPoint(x: size.width, y: 0), CGPoint(x: size.width - w, y: 0)),
            (CGRect(x: 0, y: 0, width: size.width, height: w), CGPoint(x: 0, y: 0), CGPoint(x: 0, y: w)),
            (CGRect(x: 0, y: size.height - w, width: size.width, height: w), CGPoint(x: 0, y: size.height), CGPoint(x: 0, y: size.height - w)),
        ]
        for (rect, a, b) in edges {
            ctx.fill(Path(rect), with: .linearGradient(Gradient(colors: [solid, clear]), startPoint: a, endPoint: b))
        }
    }

    // MARK: what the walk is explaining (everything else steps back)

    private func stopAlpha(_ k: Int, _ i: Int) -> Double {
        guard let stage = s.stage else { return 1 }
        switch stage {
        case .knob:
            guard let w = s.walk else { return 1 }
            return k == routeLayers || (k == w.knob.layer && i == w.knob.from) ? 1 : MapModel.dimmed
        case .gateBlame: return k == routeLayers ? 1 : MapModel.dimmed
        case .routes(let l), .hubs(let l): return k == l || k == l + 1 ? 1 : MapModel.dimmed
        case .together: return 1
        }
    }

    private func routeAlpha(_ l: Int, _ i: Int, _ j: Int) -> Double {
        guard let stage = s.stage else { return 1 }
        switch stage {
        case .knob:
            guard let w = s.walk else { return 1 }
            return RouteID(layer: l, from: i, to: j) == w.knob ? 1 : MapModel.dimmed
        case .gateBlame: return MapModel.dimmed
        case .routes(let n), .hubs(let n): return l == n ? 1 : MapModel.dimmed
        case .together: return 1
        }
    }

    private var scaleAlpha: Double {
        guard let stage = s.stage else { return 1 }
        switch stage {
        case .knob, .gateBlame, .together: return 1
        default: return MapModel.dimmed
        }
    }

    // MARK: weights for this frame

    private func weight(_ l: Int, _ idx: Int) -> Float {
        if let run {
            switch run.kind {
            case .drawIn where t >= Shots.jitter.0 && t < Shots.jitter.1:
                // the toddler spinning the dials: random draws from the same spread as the real initial routes,
                // settling (amplitude → 0) exactly on the real random weights
                let real = s.weights[l][idx]
                let amp = Float(1 - Ease.smooth(Ease.ramp(t, Shots.jitter.0, Shots.jitter.1)))
                let spread = Float((2 / Double(max(sizes[l], 1))).squareRoot())
                return real * (1 - amp) + Self.dial(l, idx, Int(t * 18)) * spread * amp
            case .guess, .unseen:
                if let p = run.forward { return p.weights[l][idx] }
            case .walk:
                // this page's layer eases from its old size to its new one
                if case .routes(let n) = run.stage, n == l, let w = run.walk {
                    let u = Float(Ease.inOut(Ease.ramp(t, Shots.thicken.0, Shots.thicken.1)))
                    let a = w.before.weights[l][idx], b = w.after.weights[l][idx]
                    return a + (b - a) * u
                }
            default: break
            }
        }
        if s.stage == .knob, let kw = s.knobWeight, let w = s.walk, w.knob.layer == l,
           idx == w.knob.from * sizes[l + 1] + w.knob.to {
            return kw
        }
        let now = s.weights[l][idx]
        guard !s.reduceMotion, let prev = s.previousWeights, prev.indices.contains(l), prev[l].indices.contains(idx) else { return now }
        let u = Float(Ease.clamp(self.now.timeIntervalSince(s.snapshotDate) / 0.15))
        return prev[l][idx] + (now - prev[l][idx]) * u
    }

    /// A deterministic standard-normal draw per route per dial tick.
    private static func dial(_ l: Int, _ idx: Int, _ tick: Int) -> Float {
        var rng = SplitMix64(seed: UInt64(truncatingIfNeeded: (l &* 1_000_003) ^ (idx &* 7919) ^ (tick &* 104_729)))
        let u1 = Double(rng.next() >> 11) / Double(1 << 53), u2 = Double(rng.next() >> 11) / Double(1 << 53)
        return Float((-2 * log(max(u1, 1e-12))).squareRoot() * cos(2 * .pi * u2))
    }

    /// Static routes step back while traces travel over them.
    private var routeOpacity: Double {
        guard let run, let ft = run.forwardTiming else { return 0.6 }
        let b0 = ft.headBegin(0, 0)
        let dim = Ease.ramp(t, b0 - 0.1, b0 + 0.3) * (1 - Ease.ramp(t, ft.end, ft.end + 0.5))
        return 0.6 - 0.32 * dim
    }

    // MARK: clipboard

    func drawClipboard(_ ctx: inout GraphicsContext) {
        var head = ctx
        head.opacity = s.stage == nil || s.stage == .together ? 1 : stopAlpha(0, -1)
        header(&head, "Clipboard", at: CGPoint(x: g.cardX + g.cardW / 2, y: g.cardY(slot: 0) - g.cardH / 2 - 18))
        let forward = run.flatMap { r in r.forwardTiming.flatMap { ft in r.forward.map { (ft, $0) } } }
        let inputs = s.network.architecture.inputs
        let facts: [Fact] = s.allFacts ? Fact.allCases : inputs
        for f in facts {
            let input = inputs.firstIndex(of: f)
            let slot = s.allFacts ? f.rawValue : (input ?? 0)
            let r = g.cardRect(slot: slot)
            // the card lights as its passengers leave, more for a fact far from an ordinary flight
            var lit = 0.0
            if let i = input, let (ft, plan) = forward, i < plan.inputShare.count {
                let peak = ft.cardPeak(i)
                lit = (0.35 + 0.65 * Double(plan.inputShare[i])) * Ease.ramp(t, peak - 0.08, peak) * (1 - Ease.ramp(t, peak + 0.1, peak + 0.6))
            }
            // the unseen flight's cards flip over (value swaps at the half-turn)
            var value = s.flight.facts[f.rawValue]
            var squash = 1.0
            if let r0 = run, r0.kind == .unseen, let from = r0.flipFrom, from.indices.contains(f.rawValue) {
                let (a, b) = Shots.flip(input ?? slot)
                let u = Ease.ramp(t, a, b)
                if u < 0.5 { value = from[f.rawValue] }
                squash = max(abs(cos(.pi * u)), 0.04)
            }
            var c = ctx
            c.opacity = input.map { stopAlpha(0, $0) } ?? 0.45
            c.translateBy(x: r.midX, y: r.midY)
            if let w = s.wobble, w == f, let r0 = run, r0.kind == .frame, !s.reduceMotion {
                let fade = 1 - Ease.ramp(t, 0, r0.duration)
                c.rotate(by: .degrees(3.5 * sin(t * 2 * .pi * 3) * fade))
            }
            if squash < 1 { c.scaleBy(x: 1, y: squash) }
            c.translateBy(x: -r.midX, y: -r.midY)
            let shape = Path(roundedRect: r, cornerRadius: 8, style: .continuous)
            c.fill(shape, with: .color(Theme.panel))
            if lit > 0 { c.fill(shape, with: .color(.white.opacity(0.13 * lit))) }
            c.stroke(shape, with: .color(.white.opacity(input == nil ? 0.05 : 0.09 + 0.45 * lit)),
                     style: StrokeStyle(lineWidth: 1, dash: input == nil ? [3, 3] : []))
            c.draw(Text(f.title).font(.system(size: 12)).foregroundColor(.secondary),
                   at: CGPoint(x: r.minX + 12, y: r.midY), anchor: .leading)
            c.draw(Text(input == nil ? "off" : f.format(value))
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                    .foregroundColor(input == nil ? .white.opacity(0.4) : Theme.answer),
                   at: CGPoint(x: r.maxX - 12, y: r.midY), anchor: .trailing)
            if input != nil, let z = s.zScores, z.indices.contains(f.rawValue) { zDial(&ctx, z[f.rawValue], card: r) }
        }
    }

    /// How far a fact sits from an ordinary flight: a scale (−3…+3 standard units) right of its card, the dot at
    /// this flight's value, the words and the number beside it.
    private func zDial(_ ctx: inout GraphicsContext, _ z: Float, card r: CGRect) {
        let x0 = r.maxX + 16, w: CGFloat = 110, y = r.midY - 6
        var line = Path()
        line.move(to: CGPoint(x: x0, y: y)); line.addLine(to: CGPoint(x: x0 + w, y: y))
        ctx.stroke(line, with: .color(.white.opacity(0.3)), style: StrokeStyle(lineWidth: 2, lineCap: .round))
        var mid = Path()
        mid.move(to: CGPoint(x: x0 + w / 2, y: y - 6)); mid.addLine(to: CGPoint(x: x0 + w / 2, y: y + 6))
        ctx.stroke(mid, with: .color(.white.opacity(0.5)), lineWidth: 1.5)
        let p = CGPoint(x: x0 + w / 2 + CGFloat(min(max(z, -3), 3) / 3) * w / 2, y: y)
        let far = abs(z) >= 1.5
        ctx.fill(Path(ellipseIn: CGRect(x: p.x - 6, y: p.y - 6, width: 12, height: 12)), with: .color(far ? Theme.positive : Theme.answer))
        ctx.draw(Text(Self.zWords(z)).font(.system(size: 12, weight: far ? .semibold : .regular)).foregroundColor(.white.opacity(far ? 0.9 : 0.6)),
                 at: CGPoint(x: x0, y: y + 14), anchor: .leading)
        ctx.draw(Text(WalkFormat.busy(z, signed: true)).font(.system(size: 12, weight: .semibold, design: .monospaced)).foregroundColor(.white.opacity(0.75)),
                 at: CGPoint(x: x0 + w + 8, y: y), anchor: .leading)
    }

    static func zWords(_ z: Float) -> String {
        let a = abs(z)
        if a < 0.5 { return "ordinary" }
        let side = z > 0 ? "above" : "below"
        return a < 1.5 ? "\(side) normal" : "far \(side) normal"
    }

    // MARK: the network

    func drawNetwork(_ ctx: inout GraphicsContext) {
        drawRoutes(&ctx)
        if let run, let plan = run.forward, let ft = run.forwardTiming { drawForward(&ctx, plan, ft) }
        if let w = s.walk, let stage = s.stage { drawWalkUnder(&ctx, w, stage) }
        drawHubs(&ctx)
        drawGate(&ctx)
        if let w = s.walk, let stage = s.stage { drawWalkOver(&ctx, w, stage) }
        if let m = s.minutes {
            var c = ctx
            c.opacity = scaleAlpha
            drawMinutes(&c, m)
        }
        if s.loss.count >= 1 { drawLoss(&ctx) }
    }

    private func layerAlpha(_ k: Int) -> Double {
        guard let run, run.kind == .drawIn, k >= 1 else { return 1 }
        let a = Shots.drawBegin(k - 1, 0) + 0.15
        return Ease.ramp(t, a, a + 0.3)
    }

    private func drawRoutes(_ ctx: inout GraphicsContext) {
        let opacity = routeOpacity
        let drawing = run?.kind == .drawIn
        let flickerAge = now.timeIntervalSince(s.snapshotDate)
        let flicker = (s.hotRoutes.isEmpty || s.reduceMotion) ? 0 : max(0, 1 - flickerAge / 0.45)
        for l in 0..<routeLayers {
            let outN = sizes[l + 1], count = sizes[l] * outN
            var biggest: Float = 1e-6
            for idx in 0..<count { biggest = max(biggest, abs(weight(l, idx))) }
            let order = s.routeOrder.indices.contains(l) && s.routeOrder[l].count == count ? s.routeOrder[l] : Array(0..<count)
            for idx in order {
                let i = idx / outN, j = idx % outN
                let w = weight(l, idx)
                let a0 = g.exit(layer: l, i)
                var a1 = g.entry(layer: l + 1, j)
                if drawing {
                    let u = Ease.out(Ease.ramp(t, Shots.drawBegin(l, i), Shots.drawBegin(l, i) + Shots.drawLength))
                    guard u > 0 else { continue }
                    a1 = lerp(a0, a1, u)
                }
                let width = max(0.4, CGFloat(6 * abs(w) / biggest))
                var p = Path()
                p.move(to: a0)
                p.addLine(to: a1)
                let rid = RouteID(layer: l, from: i, to: j)
                let shut = s.closedRoutes.contains(rid) || (l >= 1 && s.closedHubs.contains(HubRef(row: l - 1, index: i)))
                if shut {
                    // closed: nothing flies here; a faint dotted line, and a cross on routes closed one by one
                    ctx.stroke(p, with: .color(.white.opacity(0.2)), style: StrokeStyle(lineWidth: 1, lineCap: .round, dash: [2, 4]))
                    if s.closedRoutes.contains(rid) {
                        let m = lerp(a0, a1, 0.5)
                        var x = Path()
                        x.move(to: CGPoint(x: m.x - 4, y: m.y - 4)); x.addLine(to: CGPoint(x: m.x + 4, y: m.y + 4))
                        x.move(to: CGPoint(x: m.x - 4, y: m.y + 4)); x.addLine(to: CGPoint(x: m.x + 4, y: m.y - 4))
                        ctx.stroke(x, with: .color(MapModel.blame), style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    }
                    continue
                }
                let style = StrokeStyle(lineWidth: width, lineCap: .round, dash: w < 0 ? [5, 4] : [])
                let alpha = routeAlpha(l, i, j)
                ctx.stroke(p, with: .color((w < 0 ? Theme.negative : Theme.positive).opacity(opacity * alpha)), style: style)
                if flicker > 0, s.hotRoutes.contains(RouteID(layer: l, from: i, to: j)) {
                    ctx.stroke(p, with: .color(MapModel.blame.opacity(0.55 * flicker)),
                               style: StrokeStyle(lineWidth: max(1.2, width), lineCap: .round))
                }
            }
        }
    }

    /// A moving head with a fading tail along a0 → a1; `prog` 0…1 (past 1 the tail drains into a1).
    private func trace(_ ctx: inout GraphicsContext, _ a0: CGPoint, _ a1: CGPoint, _ prog: Double, _ share: Float,
                       _ color: Color, _ fade: Double = 1) {
        let head = min(prog, 1), from = max(0, prog - Self.tail)
        guard head > from else { return }
        let sh = Double(share)
        let p0 = lerp(a0, a1, from), p1 = lerp(a0, a1, head)
        var p = Path()
        p.move(to: p0)
        p.addLine(to: p1)
        ctx.stroke(p, with: .linearGradient(Gradient(colors: [color.opacity(0), color.opacity((0.35 + 0.65 * sh) * fade)]),
                                            startPoint: p0, endPoint: p1),
                   style: StrokeStyle(lineWidth: 1 + 3.5 * sh, lineCap: .round))
        if prog <= 1 {
            let r = 1.4 + 2.4 * sh
            ctx.fill(Path(ellipseIn: CGRect(x: p1.x - r, y: p1.y - r, width: 2 * r, height: 2 * r)),
                     with: .color(color.opacity((0.55 + 0.45 * sh) * fade)))
        }
    }

    /// Passengers actually carried: activation × weight per route, amber adds, blue relieves.
    private func drawForward(_ ctx: inout GraphicsContext, _ plan: ForwardPlan, _ ft: ForwardTiming) {
        for l in 0..<min(routeLayers, plan.share.count) {
            let outN = sizes[l + 1]
            for i in 0..<sizes[l] {
                let prog = ft.progress(l, i, t)
                guard prog > 0, prog < 1 + Self.tail else { continue }
                for j in 0..<outN {
                    let idx = i * outN + j, share = plan.share[l][idx]
                    guard share > 0.03 else { continue }
                    trace(&ctx, g.exit(layer: l, i), g.entry(layer: l + 1, j), prog, share,
                          plan.contribution[l][idx] > 0 ? Theme.positive : Theme.negative)
                }
            }
        }
    }

    // MARK: the walk

    private func hubCenter(_ k: Int, _ j: Int) -> CGPoint { CGPoint(x: g.x(layer: k), y: g.y(layer: k, j)) }

    /// Under the hubs: the route the rule card is about, blame flowing back along routes.
    private func drawWalkUnder(_ ctx: inout GraphicsContext, _ w: WalkPlan, _ stage: WalkStage) {
        switch stage {
        case .routes(let l):
            guard let idx = w.ranked[l].first else { return }
            let out = sizes[l + 1], a = Ease.ramp(t, Shots.ruleRoute, Shots.ruleRoute + 0.3)
            var p = Path()
            p.move(to: g.exit(layer: l, idx / out)); p.addLine(to: g.entry(layer: l + 1, idx % out))
            ctx.stroke(p, with: .color(MapModel.blame.opacity(0.45 * a)), style: StrokeStyle(lineWidth: 9, lineCap: .round))
        case .hubs(let k):
            // the blame of layer k+1 passes back along each route: width ∝ |route size × blame at its end|
            let out = sizes[k + 1]
            var top: Float = 1e-6
            for i in 0..<sizes[k] { for j in 0..<out { top = max(top, abs(w.before.weights[k][i * out + j] * w.blame(layer: k + 1, j))) } }
            let prog = Ease.ramp(t, Shots.backFlow.0, Shots.backFlow.1) * (1 + Self.tail)
            guard prog > 0, prog < 1 + Self.tail else { return }
            for i in 0..<sizes[k] where !w.isClosed(layer: k, i) {
                for j in 0..<out {
                    let share = abs(w.before.weights[k][i * out + j] * w.blame(layer: k + 1, j)) / top
                    guard share > 0.03 else { continue }
                    trace(&ctx, g.entry(layer: k + 1, j), g.exit(layer: k, i), prog, share, MapModel.blame)
                }
            }
        default: break
        }
    }

    /// Over everything: chips (▲/▼ + change), blame meters, resting-stress dials, the gate's blame, the knob.
    private func drawWalkOver(_ ctx: inout GraphicsContext, _ w: WalkPlan, _ stage: WalkStage) {
        switch stage {
        case .knob:
            guard let kg = s.knobGuess else { return }
            let gap0 = abs(w.guessBefore - w.truth), gap = abs(kg - w.truth)
            guard abs(gap - gap0) > 0.5 else { return }
            let warmer = gap < gap0
            ctx.draw(Text(warmer ? "warmer" : "colder").font(.system(size: 13, weight: .bold)).foregroundColor(warmer ? MapModel.warmer : MapModel.colder),
                     at: CGPoint(x: g.gateRect.midX, y: g.gateRect.maxY + 13))
        case .gateBlame:
            let a = Ease.ramp(t, Shots.gateArrow.0 + 0.3, Shots.gateArrow.1)
            let b = w.blame.last?.first ?? 1
            ctx.draw(Text("\(WalkFormat.arrow(b)) blame \(WalkFormat.points(b))").font(.system(size: 12, weight: .bold, design: .monospaced))
                        .foregroundColor(MapModel.blame.opacity(a)),
                     at: CGPoint(x: g.gateRect.midX, y: g.gateRect.maxY + 13))
        case .routes(let l):
            let out = sizes[l + 1]
            let n = min(Shots.chips(layer: l), w.ranked[l].count)
            for (rank, idx) in w.ranked[l].prefix(n).enumerated() {
                let a = Ease.ramp(t, Shots.chip(rank), Shots.chip(rank) + 0.25)
                guard a > 0 else { continue }
                let i = idx / out, j = idx % out
                let v = w.change[l][idx]
                let text = v == 0 ? "0" : "\(WalkFormat.chevron(v)) \(WalkFormat.points(v))"
                let along = Shots.chipAlong(layer: l, rank: rank)
                chip(&ctx, text, at: lerp(g.exit(layer: l, i), g.entry(layer: l + 1, j), along), alpha: a, strong: rank == 0)
            }
            if l >= 1 {
                for i in 0..<sizes[l] where w.isClosed(layer: l, i) {
                    closedGlyph(&ctx, hubCenter(l, i), alpha: Ease.ramp(t, Shots.chip(0), Shots.chip(0) + 0.3), held: heldCount(w, l, i))
                }
            }
            // the gate's own resting stress moves with the last routes
            if l == routeLayers - 1 {
                let u = Ease.inOut(Ease.ramp(t, Shots.thicken.0, Shots.thicken.1))
                let b0 = w.before.biases[l][0], b1 = w.after.biases[l][0]
                dial(&ctx, CGPoint(x: g.gateRect.maxX - 12, y: g.gateRect.minY + 12), b0 + (b1 - b0) * Float(u))
            }
        case .hubs(let k):
            let row = w.blame[k - 1]
            let top = max(row.map(abs).max() ?? 0, 1e-6)
            for j in 0..<sizes[k] {
                let c = hubCenter(k, j)
                if w.isClosed(layer: k, j) {
                    closedGlyph(&ctx, c, alpha: Ease.ramp(t, Shots.meter(j).0, Shots.meter(j).1), held: heldCount(w, k, j))
                    continue
                }
                let u = Ease.out(Ease.ramp(t, Shots.meter(j).0, Shots.meter(j).1))
                let v = row[j], frac = Double(abs(v) / top) * u
                if frac > 0 {
                    var arc = Path()
                    arc.addArc(center: c, radius: g.hubR + 4, startAngle: .degrees(-90), endAngle: .degrees(-90 + 360 * frac), clockwise: false)
                    ctx.stroke(arc, with: .color(MapModel.blame), style: StrokeStyle(lineWidth: 3.5, lineCap: .round))
                }
                ctx.draw(Text("\(WalkFormat.arrow(v)) \(WalkFormat.points(v))").font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundColor(MapModel.blame.opacity(u)),
                         at: CGPoint(x: c.x, y: c.y - g.hubR - 13))
                let d = Ease.inOut(Ease.ramp(t, Shots.dial.0, Shots.dial.1))
                let b0 = w.before.biases[k - 1][j], b1 = w.after.biases[k - 1][j]
                dial(&ctx, CGPoint(x: c.x + g.hubR + 10, y: c.y + g.hubR - 2), b0 + (b1 - b0) * Float(d))
            }
        case .together:
            break
        }
    }

    /// Passengers held at a closed hub: how many arrived, against the busiest arrivals in its row (1…4 dots).
    private func heldCount(_ w: WalkPlan, _ k: Int, _ j: Int) -> Int {
        let out = sizes[k]
        func arrived(_ jj: Int) -> Float {
            var s: Float = 0
            for i in 0..<sizes[k - 1] { s += abs(w.busy[k - 1][i] * w.before.weights[k - 1][i * out + jj]) }
            return s
        }
        var top: Float = 1e-6
        for jj in 0..<out { top = max(top, arrived(jj)) }
        return 1 + Int((3 * arrived(j) / top).rounded())
    }

    private func chip(_ ctx: inout GraphicsContext, _ text: String, at p: CGPoint, alpha: Double, strong: Bool) {
        let label = ctx.resolve(Text(text).font(.system(size: 10, weight: .bold, design: .monospaced)).foregroundColor(Theme.answer))
        let size = label.measure(in: CGSize(width: 200, height: 40))
        let rect = CGRect(x: p.x - size.width / 2 - 5, y: p.y - size.height / 2 - 2, width: size.width + 10, height: size.height + 4)
        var c = ctx
        c.opacity = alpha
        let shape = Path(roundedRect: rect, cornerRadius: rect.height / 2, style: .continuous)
        c.fill(shape, with: .color(Color(white: 0.09)))
        c.stroke(shape, with: .color(strong ? MapModel.blame : .white.opacity(0.3)), lineWidth: strong ? 1.6 : 1)
        c.draw(label, at: p)
    }

    /// A resting-stress dial: the needle's angle is the hub's resting stress (bias).
    private func dial(_ ctx: inout GraphicsContext, _ c: CGPoint, _ bias: Float) {
        let r: CGFloat = 6
        ctx.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)), with: .color(Color(white: 0.1)))
        ctx.stroke(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)), with: .color(.white.opacity(0.45)), lineWidth: 1)
        let angle = (-90 + Double(min(max(bias, -1.25), 1.25)) * 120) * .pi / 180
        var needle = Path()
        needle.move(to: c)
        needle.addLine(to: CGPoint(x: c.x + cos(angle) * (r - 1), y: c.y + sin(angle) * (r - 1)))
        ctx.stroke(needle, with: .color(Theme.answer), style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
    }

    /// A closed hub: its gate agent held the passengers. A '0' badge (nothing leaves), a closed boom gate at the
    /// exit, and the held passengers waiting inside.
    private func closedGlyph(_ ctx: inout GraphicsContext, _ c: CGPoint, alpha: Double, held: Int) {
        guard alpha > 0 else { return }
        var x = ctx
        x.opacity = alpha
        let badge = CGRect(x: c.x - 9, y: c.y - g.hubR - 19, width: 18, height: 13)
        x.fill(Path(roundedRect: badge, cornerRadius: 6.5, style: .continuous), with: .color(Color(white: 0.12)))
        x.stroke(Path(roundedRect: badge, cornerRadius: 6.5, style: .continuous), with: .color(.white.opacity(0.5)), lineWidth: 1)
        x.draw(Text("0").font(.system(size: 9, weight: .bold, design: .monospaced)).foregroundColor(.white), at: CGPoint(x: badge.midX, y: badge.midY))
        let px = c.x + g.hubR + 3
        var post = Path()
        post.move(to: CGPoint(x: px, y: c.y - 8)); post.addLine(to: CGPoint(x: px, y: c.y + 8))
        x.stroke(post, with: .color(.white.opacity(0.85)), style: StrokeStyle(lineWidth: 2, lineCap: .round))
        let arm = CGRect(x: px, y: c.y - 1.75, width: 15, height: 3.5)
        x.fill(Path(arm), with: .color(.white.opacity(0.9)))
        for k in 0..<2 { x.fill(Path(CGRect(x: px + 3.5 + CGFloat(k) * 6, y: arm.minY, width: 3, height: arm.height)), with: .color(MapModel.blame)) }
        for n in 0..<min(held, 4) {
            let p = CGPoint(x: c.x + g.hubR - 5, y: c.y - 6 + CGFloat(n) * 4)
            x.fill(Path(ellipseIn: CGRect(x: p.x - 1.5, y: p.y - 1.5, width: 3, height: 3)), with: .color(Theme.answer))
        }
    }

    // MARK: hubs

    /// 0…1 fill for hub k of a row this frame.
    private func hubLevel(_ row: Int, _ k: Int) -> Double {
        if let run, let p = run.forward, let ft = run.forwardTiming, row < p.level.count {
            let d = Ease.inOut(ft.decided(row + 1, t))
            if run.kind == .together, let b = run.before, row < b.level.count {
                return b.level[row][k] + (p.level[row][k] - b.level[row][k]) * d
            }
            return p.level[row][k] * d
        }
        guard let a = s.rest, a.hubs.indices.contains(row), a.hubs[row].indices.contains(k) else { return 0 }
        return MapModel.hubLevel(a.hubs[row], k)
    }

    private func drawHubs(_ ctx: inout GraphicsContext) {
        // the tally: passengers arriving + resting stress, counted as they land (the guess and the unseen flight)
        let tally: (ForwardPlan, ForwardTiming)? = {
            guard let run, run.kind == .guess || run.kind == .unseen, let p = run.forward, let ft = run.forwardTiming else { return nil }
            return (p, ft)
        }()
        for (row, count) in s.network.architecture.hidden.enumerated() {
            let layerIndex = row + 1
            var c = ctx
            c.opacity = layerAlpha(layerIndex) * (s.stage == nil || s.stage == .together ? 1 : stopAlpha(layerIndex, -1))
            header(&c, "Hubs, row \(row + 1)", at: CGPoint(x: g.x(layer: layerIndex), y: g.top - 18))
            for k in 0..<count {
                var h = ctx
                h.opacity = layerAlpha(layerIndex) * stopAlpha(layerIndex, k)
                let center = hubCenter(layerIndex, k)
                let rect = CGRect(x: center.x - g.hubR, y: center.y - g.hubR, width: 2 * g.hubR, height: 2 * g.hubR)
                let level = hubLevel(row, k)
                h.fill(Path(ellipseIn: rect), with: .color(Color(hue: 0.088, saturation: 0.85 * level, brightness: 0.36 + 0.6 * level)))
                h.stroke(Path(ellipseIn: rect), with: .color(.white.opacity(0.22)), lineWidth: 1)
                if s.closedHubs.contains(HubRef(row: row, index: k)) {
                    closedGlyph(&h, center, alpha: 1, held: 0)
                }
                if let p = s.pointAt, p.row == row, p.index == k {
                    let ring = rect.insetBy(dx: -5, dy: -5)
                    h.stroke(Path(ellipseIn: ring), with: .color(Theme.answer), lineWidth: 2)
                }
                h.draw(Text(MapModel.hubName(row: row, index: k)).font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundColor(level > 0.55 ? .black.opacity(0.75) : .white.opacity(0.7)),
                       at: center)
                if let (plan, ft) = tally, row < plan.pre.count { drawTally(&h, plan, ft, row, k, center) }
            }
        }
    }

    private func drawTally(_ ctx: inout GraphicsContext, _ plan: ForwardPlan, _ ft: ForwardTiming, _ row: Int, _ k: Int, _ center: CGPoint) {
        let begin = ft.headBegin(row, 0)
        let alpha = Ease.ramp(t, begin, begin + 0.2) * (1 - Ease.ramp(t, ft.end + 0.4, ft.end + 0.7))
        guard alpha > 0 else { return }
        let outN = sizes[row + 1]
        var value = plan.bias[row][k]
        var arrived: Float = 0
        for i in 0..<sizes[row] {
            let c = plan.contribution[row][i * outN + k]
            value += c * Float(Ease.smooth(ft.progress(row, i, t)))
            arrived += abs(c)
        }
        let decided = ft.decided(row + 1, t)
        if plan.pre[row][k] <= 0 && decided > 0 {
            // the gate agent holds the passengers: a '0' badge, a closed gate, the held passengers stop here
            var top: Float = 1e-6
            for j in 0..<outN {
                var s: Float = 0
                for i in 0..<sizes[row] { s += abs(plan.contribution[row][i * outN + j]) }
                top = max(top, s)
            }
            closedGlyph(&ctx, center, alpha: alpha * decided, held: 1 + Int((3 * arrived / top).rounded()))
        } else {
            ctx.draw(Text(Self.signed(value)).font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundColor(Theme.answer.opacity(0.9 * alpha)),
                     at: CGPoint(x: center.x, y: center.y - g.hubR - 8))
        }
    }

    static func signed(_ v: Float) -> String {
        let s = String(format: "%+.1f", Double(v))
        return s.replacingOccurrences(of: "-", with: "−")
    }

    // MARK: the gate

    private func gateText() -> (main: String, sub: String?) {
        if let run, let plan = run.forward, let ft = run.forwardTiming {
            let (a, b) = ft.countWindow
            switch run.kind {
            case .guess, .unseen:
                if t < a { return ("?", nil) }
                return (AirlineState.phrase(plan.guess * Float(Ease.out(Ease.ramp(t, a, b)))), nil)
            case .together:
                guard let w = run.walk else { break }
                let u = Float(Ease.out(Ease.ramp(t, a, b)))
                let minutes = w.guessBefore + (w.guessAfter - w.guessBefore) * u
                return (AirlineState.phrase(minutes), t >= a ? "was \(AirlineState.phrase(w.guessBefore))" : nil)
            default: break
            }
        }
        if s.stage == .knob, let kg = s.knobGuess { return (AirlineState.phrase(kg), nil) }
        guard let a = s.rest else { return ("?", nil) }
        return (AirlineState.phrase(a.gateMinutes), s.wasGuess.map { "was \(AirlineState.phrase($0))" })
    }

    private func drawGate(_ ctx: inout GraphicsContext) {
        var c = ctx
        c.opacity = layerAlpha(routeLayers) * stopAlpha(routeLayers, 0)
        let gate = g.gateRect
        let shape = Path(roundedRect: gate, cornerRadius: 12, style: .continuous)
        c.fill(shape, with: .color(Theme.panel))
        c.stroke(shape, with: .color(.white.opacity(0.14)), lineWidth: 1)
        header(&c, "Gate \(s.flight.destination)", at: CGPoint(x: gate.midX, y: gate.minY - 14))
        let (main, sub) = gateText()
        c.draw(Text(main).font(.system(size: main == "?" ? 26 : 17, weight: .semibold)).foregroundColor(Theme.answer),
               at: CGPoint(x: gate.midX, y: gate.midY - (sub == nil ? 0 : 8)))
        if let sub {
            c.draw(Text(sub).font(.system(size: 11)).foregroundColor(.white.opacity(0.5)),
                   at: CGPoint(x: gate.midX, y: gate.midY + 15))
        }
    }

    // MARK: the minutes scale (the loss as a visible gap)

    private func drawMinutes(_ ctx: inout GraphicsContext, _ m: MinutesScale) {
        var guessA = 1.0, realA = 1.0, bracket = 1.0, slide = 1.0
        if let run {
            switch run.kind {
            case .reveal:
                guessA = Ease.ramp(t, Shots.revealGuess.0, Shots.revealGuess.1)
                realA = Ease.ramp(t, Shots.revealReal.0, Shots.revealReal.1)
                bracket = Ease.ramp(t, Shots.revealBracket.0, Shots.revealBracket.1)
            case .unseen:
                if let b = run.forwardTiming?.countWindow.1 {
                    guessA = Ease.ramp(t, b, b + 0.3)
                    realA = Ease.ramp(t, b + 0.5, b + 0.9)
                    bracket = Ease.ramp(t, b + 0.9, b + 1.4)
                }
            case .together:
                if let e = run.forwardTiming?.end { slide = Ease.inOut(Ease.ramp(t, e, e + 0.6)) }
            case .walk where m.arrow:
                bracket = Ease.ramp(t, Shots.gateArrow.0, Shots.gateArrow.1)
            default: break
            }
        }
        let r = g.scaleRect
        var values = [m.guess, m.truth]
        if let n = m.newGuess { values.append(n) }
        let lo = min(0, Float((Double(values.min()!) / 30).rounded(.down) * 30))
        let hi = max(60, Float((Double(values.max()!) / 60).rounded(.up) * 60))
        func y(_ v: Float) -> CGFloat { r.maxY - CGFloat((v - lo) / (hi - lo)) * r.height }
        let x0 = r.minX + r.width * 0.5

        var c = ctx
        c.opacity = ctx.opacity * max(guessA, realA)
        header(&c, "Minutes late", at: CGPoint(x: x0, y: r.minY - 12))
        var axis = Path()
        axis.move(to: CGPoint(x: x0, y: y(hi))); axis.addLine(to: CGPoint(x: x0, y: y(lo)))
        c.stroke(axis, with: .color(.white.opacity(0.25)), lineWidth: 1)
        let step: Float = hi - lo > 240 ? 60 : 30
        var v = (lo / step).rounded(.up) * step
        while v <= hi {
            var tick = Path()
            tick.move(to: CGPoint(x: x0 - 3, y: y(v))); tick.addLine(to: CGPoint(x: x0 + 3, y: y(v)))
            c.stroke(tick, with: .color(.white.opacity(0.25)), lineWidth: 1)
            v += step
        }
        c.draw(Text("0").font(.system(size: 9, design: .monospaced)).foregroundColor(.white.opacity(0.35)),
               at: CGPoint(x: x0 + 7, y: y(0)), anchor: .leading)

        // markers (labels on the left, pushed apart when they would overlap)
        let shownGuess = m.newGuess.map { m.guess + ($0 - m.guess) * Float(slide) } ?? m.guess
        let realDrop = Ease.out(realA)
        let realY = y(hi) + (y(m.truth) - y(hi)) * CGFloat(realDrop)
        var marks: [(y: CGFloat, text: String, color: Color, alpha: Double)] = []
        if m.newGuess != nil && slide > 0 {
            marks.append((y(m.guess), "was \(Self.whole(m.guess))", .white, 0.45 * slide))
        }
        marks.append((y(shownGuess), "guess \(Self.whole(shownGuess))", Theme.answer, guessA))
        marks.append((realY, "real \(Self.whole(m.truth))", Theme.prompt, realA))
        let labelYs = Self.spread(marks.map(\.y), gap: 13)
        for (n, mk) in marks.enumerated() where mk.alpha > 0 {
            var line = Path()
            line.move(to: CGPoint(x: x0 - 9, y: mk.y)); line.addLine(to: CGPoint(x: x0 + 9, y: mk.y))
            c.stroke(line, with: .color(mk.color.opacity(mk.alpha)), style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
            c.draw(Text(mk.text).font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundColor(mk.color.opacity(mk.alpha)),
                   at: CGPoint(x: x0 - 14, y: labelYs[n]), anchor: .trailing)
        }

        // the bracket: the loss, the gap between guess and real (or, on the gate's blame page, the way it must move)
        guard bracket > 0 else { return }
        let gy = y(shownGuess), endY = gy + (y(m.truth) - gy) * CGFloat(Ease.out(bracket))
        let bx = x0 + 16
        var br = Path()
        if m.arrow {
            br.move(to: CGPoint(x: bx, y: gy)); br.addLine(to: CGPoint(x: bx, y: endY))
            let dir: CGFloat = endY < gy ? -1 : 1
            br.move(to: CGPoint(x: bx - 5, y: endY - 6 * dir)); br.addLine(to: CGPoint(x: bx, y: endY)); br.addLine(to: CGPoint(x: bx + 5, y: endY - 6 * dir))
            c.stroke(br, with: .color(MapModel.blame), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
        } else {
            br.move(to: CGPoint(x: bx - 5, y: gy)); br.addLine(to: CGPoint(x: bx, y: gy))
            br.addLine(to: CGPoint(x: bx, y: endY)); br.addLine(to: CGPoint(x: bx - 5, y: endY))
            c.stroke(br, with: .color(MapModel.blame.opacity(0.95)), style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
        }
        let off = Int(abs(shownGuess - m.truth).rounded())
        let label = m.arrow ? (m.truth > shownGuess ? "needs to\nbe higher" : "needs to\nbe lower") : "off by\n\(off) min"
        c.opacity = ctx.opacity * Ease.ramp(bracket, 0.5, 1)
        c.draw(Text(label).font(.system(size: 10, weight: .semibold)).foregroundColor(MapModel.blame),
               at: CGPoint(x: bx + 7, y: (gy + endY) / 2), anchor: .leading)
    }

    private static func whole(_ v: Float) -> String {
        let r = Int(v.rounded())
        return r < 0 ? "−\(-r) min" : "\(r) min"
    }

    /// Label positions near `ys`, at least `gap` apart, keeping their order.
    private static func spread(_ ys: [CGFloat], gap: CGFloat) -> [CGFloat] {
        let order = ys.indices.sorted { ys[$0] < ys[$1] }
        var out = ys
        for n in 1..<max(order.count, 1) where out[order[n]] - out[order[n - 1]] < gap {
            out[order[n]] = out[order[n - 1]] + gap
        }
        return out
    }

    // MARK: the practice curve (the quiz score: minutes off on the 2023 flights, point by point)

    private func drawLoss(_ ctx: inout GraphicsContext) {
        let pts = s.loss
        let r = g.scaleRect
        let top = max(pts.map(\.minutes).max() ?? 1, 1) * 1.1
        let plot = CGRect(x: r.minX + 8, y: r.minY + 6, width: r.width - 16, height: r.height - 26)
        func p(_ k: Int) -> CGPoint {
            CGPoint(x: plot.minX + plot.width * CGFloat(pts[k].flights) / CGFloat(max(s.lossTotal, 1)),
                    y: plot.maxY - plot.height * CGFloat(pts[k].minutes / top))
        }
        header(&ctx, "Quiz score: minutes off", at: CGPoint(x: r.midX, y: r.minY - 12))
        var base = Path()
        base.move(to: CGPoint(x: plot.minX, y: plot.maxY)); base.addLine(to: CGPoint(x: plot.maxX, y: plot.maxY))
        ctx.stroke(base, with: .color(.white.opacity(0.22)), lineWidth: 1)
        ctx.draw(Text("0").font(.system(size: 9, design: .monospaced)).foregroundColor(.white.opacity(0.35)),
                 at: CGPoint(x: plot.minX, y: plot.maxY + 8))
        ctx.draw(Text("\(s.lossTotal.formatted()) studied").font(.system(size: 9, design: .monospaced)).foregroundColor(.white.opacity(0.35)),
                 at: CGPoint(x: plot.maxX, y: plot.maxY + 8), anchor: .trailing)
        var line = Path()
        line.move(to: p(0))
        let last = pts.count - 1
        for k in stride(from: 1, to: last, by: 1) { line.addLine(to: p(k)) }
        var end = p(last)
        if last > 0 {
            let u = s.reduceMotion ? 1 : Ease.out(now.timeIntervalSince(pts[last].at) / 0.25)
            end = lerp(p(last - 1), p(last), u)
            line.addLine(to: end)
        }
        ctx.stroke(line, with: .color(Theme.answer.opacity(0.9)), style: StrokeStyle(lineWidth: 1.8, lineCap: .round, lineJoin: .round))
        for k in 0..<last {
            let q = p(k)
            ctx.fill(Path(ellipseIn: CGRect(x: q.x - 1.6, y: q.y - 1.6, width: 3.2, height: 3.2)), with: .color(Theme.answer.opacity(0.7)))
        }
        ctx.fill(Path(ellipseIn: CGRect(x: end.x - 2.6, y: end.y - 2.6, width: 5.2, height: 5.2)), with: .color(Theme.answer))
        ctx.draw(Text("\(Int(pts[0].minutes.rounded()))").font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundColor(.white.opacity(0.6)),
                 at: CGPoint(x: p(0).x + 4, y: p(0).y - 9), anchor: .leading)
        if last > 0 {
            ctx.draw(Text("\(Int(pts[last].minutes.rounded())) min").font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundColor(Theme.answer),
                     at: CGPoint(x: end.x, y: end.y - 10), anchor: .trailing)
        }
    }

    private func header(_ ctx: inout GraphicsContext, _ s: String, at p: CGPoint) {
        ctx.draw(Text(s.uppercased()).font(.system(size: 10, weight: .semibold)).kerning(0.8).foregroundColor(.white.opacity(0.4)), at: p)
    }
}

@inline(__always) private func lerp(_ a: CGPoint, _ b: CGPoint, _ u: Double) -> CGPoint {
    CGPoint(x: a.x + (b.x - a.x) * CGFloat(u), y: a.y + (b.y - a.y) * CGFloat(u))
}

// MARK: - where everything sits (map coordinates; the camera scales and moves the whole map)

struct MapGeometry {
    let size: CGSize
    let sizes: [Int]
    /// Clipboard: the slot each input's card sits in, out of `slotCount` (nil: input i in slot i).
    var slots: [Int]? = nil
    var slotCount: Int? = nil
    /// 0 = cards spread over the full height, 1 = a tight column (clipboard-only pages).
    var compact: CGFloat = 0

    static func of(_ scene: MapScene, size: CGSize, now: Date = Date()) -> MapGeometry {
        let sizes = scene.network.layerSizes
        let u = scene.reduceMotion ? 1 : Ease.inOut(now.timeIntervalSince(scene.compactSince) / 0.8)
        let compact = CGFloat(scene.compactCards ? u : 1 - u)
        guard scene.allFacts else { return MapGeometry(size: size, sizes: sizes, compact: compact) }
        return MapGeometry(size: size, sizes: sizes, slots: scene.network.architecture.inputs.map(\.rawValue),
                           slotCount: Fact.allCases.count, compact: compact)
    }
    private var cardSlots: Int { max(1, slotCount ?? sizes[0]) }

    func cardRect(slot: Int) -> CGRect {
        CGRect(x: cardX, y: cardY(slot: slot) - cardH / 2, width: cardW, height: cardH)
    }
    func cardY(slot: Int) -> CGFloat {
        let n = cardSlots
        if n <= 1 { return (top + bottom) / 2 }
        let full = bottom - top - cardH, tight = min(full, CGFloat(n - 1) * (cardH + 12))
        let span = full + (tight - full) * compact
        return (top + bottom) / 2 - span / 2 + span * CGFloat(slot) / CGFloat(n - 1)
    }
    var top: CGFloat { 28 }
    var bottom: CGFloat { size.height - 8 }
    var cardW: CGFloat { min(210, size.width * 0.22) }
    var cardX: CGFloat { max(8, size.width * 0.03) }
    var cardH: CGFloat { min(34, max(18, (bottom - top) / CGFloat(cardSlots) - 8)) }
    var hubR: CGFloat { min(17, max(9, (bottom - top) / 14)) }
    var gateRect: CGRect {
        let w = min(170, size.width * 0.17), h: CGFloat = 70
        return CGRect(x: size.width - w - max(8, size.width * 0.03), y: (top + bottom) / 2 - h / 2, width: w, height: h)
    }
    /// Under the gate: the minutes scale (reveal, the walk, the unseen flight) or the quiz curve (practice).
    var scaleRect: CGRect {
        let g = gateRect, y0 = g.maxY + 40
        return CGRect(x: g.minX - 20, y: y0, width: g.width + 20, height: max(90, bottom - 6 - y0))
    }

    /// Column centre for a layer: clipboard cards, hub rows spread between them, then the gate.
    func x(layer l: Int) -> CGFloat {
        if l == 0 { return cardX + cardW / 2 }
        if l == sizes.count - 1 { return gateRect.midX }
        let hiddenCount = sizes.count - 2
        let left = cardX + cardW + size.width * 0.12, right = gateRect.minX - size.width * 0.12
        return hiddenCount == 1 ? (left + right) / 2 : left + (right - left) * CGFloat(l - 1) / CGFloat(hiddenCount - 1)
    }

    func y(layer l: Int, _ i: Int) -> CGFloat {
        if l == 0 { return cardY(slot: slots.map { $0.indices.contains(i) ? $0[i] : i } ?? i) }
        let n = sizes[l]
        if l == sizes.count - 1 || n <= 1 { return (top + bottom) / 2 }
        let span = l == 0 ? bottom - top - cardH : min(bottom - top - 2 * hubR, CGFloat(n - 1) * hubR * 4.2)
        let start = (top + bottom) / 2 - span / 2
        return start + span * CGFloat(i) / CGFloat(n - 1)
    }

    /// Where routes leave a stop (right edge) and where they arrive (left edge).
    func exit(layer l: Int, _ i: Int) -> CGPoint {
        CGPoint(x: l == 0 ? cardX + cardW : x(layer: l) + hubR, y: y(layer: l, i))
    }
    func entry(layer l: Int, _ j: Int) -> CGPoint {
        CGPoint(x: l == sizes.count - 1 ? gateRect.minX : x(layer: l) - hubR, y: y(layer: l, j))
    }

    /// The vertical extent of a hub row or the gate, with room for labels above.
    private func extent(_ l: Int) -> ClosedRange<CGFloat> {
        if l == sizes.count - 1 { return (gateRect.minY - 30)...(gateRect.maxY + 22) }
        let n = sizes[l]
        return (y(layer: l, 0) - hubR - 24)...(y(layer: l, max(n - 1, 0)) + hubR + 10)
    }

    /// The rect a shot frames. Every stop the shot is about stays fully in view; routes may run off the edge.
    func rect(_ f: Focus) -> CGRect {
        let last = sizes.count - 1
        switch f {
        case .wide:
            return CGRect(origin: .zero, size: size)
        case .clipboard, .clipboardDials:
            let y0 = cardY(slot: 0) - cardH / 2 - 30, y1 = cardY(slot: cardSlots - 1) + cardH / 2 + 6
            let extra: CGFloat = f == .clipboardDials ? 210 : 0
            return CGRect(x: cardX - 6, y: y0, width: cardW + 12 + extra, height: y1 - y0)
        case .between(let a):
            let b = min(a + 1, last)
            let x0 = a == 0 ? cardX + cardW - 70 : x(layer: a) - hubR - 46
            let x1 = b == last ? gateRect.maxX + 8 : x(layer: b) + hubR + 46
            let ranges = [a, b].filter { $0 > 0 }.map(extent)
            let y0 = ranges.map(\.lowerBound).min() ?? top, y1 = ranges.map(\.upperBound).max() ?? bottom
            return CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
        case .gate:
            let gr = gateRect
            return CGRect(x: gr.minX - 120, y: gr.midY - 120, width: gr.width + 128, height: 240)
        case .gateScale:
            let gr = gateRect, sr = scaleRect
            return CGRect(x: sr.minX - 50, y: gr.minY - 30, width: gr.maxX + 8 - (sr.minX - 50), height: sr.maxY + 6 - (gr.minY - 30))
        case .gateRoutes:
            return rect(.between(max(last - 1, 0))).union(rect(.gateScale))
        }
    }

    func camera(_ f: Focus) -> MapCamera { f == .wide ? .wide(size) : .framing(rect(f), in: size) }

    /// The hub under a map point (a little larger than the circle, for an easy target).
    func hub(at p: CGPoint) -> HubRef? {
        for l in 1..<max(sizes.count - 1, 1) {
            for k in 0..<sizes[l] {
                let c = CGPoint(x: x(layer: l), y: y(layer: l, k))
                if hypot(p.x - c.x, p.y - c.y) <= hubR + 5 { return HubRef(row: l - 1, index: k) }
            }
        }
        return nil
    }

    /// The route nearest a map point, within `tolerance`.
    func route(at p: CGPoint, tolerance: CGFloat = 7) -> RouteID? {
        var best: (RouteID, CGFloat)?
        for l in 0..<(sizes.count - 1) {
            for i in 0..<sizes[l] {
                for j in 0..<sizes[l + 1] {
                    let a = exit(layer: l, i), b = entry(layer: l + 1, j)
                    let dx = b.x - a.x, dy = b.y - a.y, len2 = max(dx * dx + dy * dy, 1e-6)
                    let u = min(max(((p.x - a.x) * dx + (p.y - a.y) * dy) / len2, 0), 1)
                    let d = hypot(p.x - (a.x + u * dx), p.y - (a.y + u * dy))
                    if d <= tolerance, best == nil || d < best!.1 { best = (RouteID(layer: l, from: i, to: j), d) }
                }
            }
        }
        return best?.0
    }
}
