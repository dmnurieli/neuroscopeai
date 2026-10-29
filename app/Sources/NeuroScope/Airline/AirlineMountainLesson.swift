import SwiftUI
import AirlineKit

/// Lesson 5, "The foggy mountain": training as a blindfolded hiker. The mountain is a real slice of the loss:
/// minutes off as two routes into one row-1 hub vary, every other route held still. The hiker's path is real
/// gradient steps on those two routes (MountainHike); the step-size slider is the learning rate, on a log scale.
@MainActor @Observable
final class AirlineMountainLesson: AirlineLessonModel {
    let title = AirlineLessonID.mountain.title
    let pageCount = 3
    let chor = Choreographer()

    static let steps = 60
    static let stepTime = 0.07
    static let defaultStride: Float = 0.5
    /// log10 of the stride: 0.003 … 20,000 (huge strides tumble).
    static let strideRange: ClosedRange<Double> = -2.5...4.3

    private(set) var ledger: Ledger?
    private(set) var network: Network?
    private(set) var slice: MountainSlice?
    private(set) var routes: (RouteID, RouteID)?
    private(set) var hike: MountainHike?
    private(set) var hikeStart = Date()
    private(set) var page = 0
    private(set) var stride: Float = AirlineMountainLesson.defaultStride
    private var hikeTask: Task<Void, Never>?
    private var serial = 0

    /// One-tap strides that show each kind of hike on THIS slice (scanned after load; these until then).
    struct Preset: Identifiable { let label: String; let stride: Float; var id: String { label } }
    private(set) var presets: [Preset] = [
        Preset(label: "Tiny steps", stride: 0.01), Preset(label: "Just right", stride: 0.5),
        Preset(label: "Too big", stride: 3.2), Preset(label: "Giant", stride: 10_000),
    ]

    var canGoBack: Bool { page > 0 }
    var canWatchAgain: Bool { !chor.reduceMotion && hike != nil && !chor.isPlaying }
    var isReady: Bool { slice != nil }

    init() { chor.summary = { [weak self] in self?.summary } }

    /// The lesson network (the learner's clipboard, two rows of six), trained two passes, then the slice.
    func load() async {
        guard ledger == nil else { return }
        let l = await AirlineLedger.load()
        ledger = l
        var inputs = AirlineProgress.clipboard
        if inputs.count < 2 { inputs = Fact.allCases }
        let arch = Architecture(inputs: inputs, hidden: [6, 6])
        let built = await Task.detached(priority: .userInitiated) { () -> (Network, MountainSlice, (RouteID, RouteID))? in
            let t = Trainer(network: Network(architecture: arch, ledger: l, seed: Architecture.lessonSeed), ledger: l)
            _ = t.train(batches: 2 * ((l.indices(.practice).count + 63) / 64))
            let net = t.network
            guard let r = net.mountainRoutes(in: l) else { return nil }
            let s = net.mountain(a: r.0, b: r.1, span: 1.5, resolution: 41, sampleCount: 2_000, seed: 11, in: l)
            return (net, s, r)
        }.value
        guard let built else { return }
        network = built.0
        slice = built.1
        routes = built.2
        rehike()
        scanPresets()
    }

    /// Runs real hikes to find a stride for each preset on this slice: the largest that settles cleanly, one that
    /// only crawls, the largest that swings back and forth across the valley without tumbling, the smallest that
    /// tumbles.
    private func scanPresets() {
        guard let net = network, let s = slice, let l = ledger, let p = start else { return }
        let count = Self.steps
        Task { [weak self] in
            let found = await Task.detached(priority: .utility) { () -> [Preset] in
                func hike(_ st: Float) -> MountainHike { net.hike(on: s, from: p, stepSize: st, steps: count, seed: 11, in: l) }
                let right = [Float(1), 0.5, 0.3, 0.1].first { Self.outcome(hike($0), minLoss: s.minLoss) == .settles } ?? 0.5
                let tiny = [right / 50, right / 150].first { Self.outcome(hike($0), minLoss: s.minLoss) == .crawls } ?? right / 150
                let big = [Float(5.6), 4.2, 3.2, 2.4, 1.8].first { st in
                    let h = hike(st)
                    return !h.tumbled && Self.rises(h) >= 6 && (h.losses.last ?? .infinity) < (h.losses.first ?? 0)
                } ?? 3.2
                let giant = [Float(10), 30, 100, 1_000, 10_000].first { hike($0).tumbled } ?? 10_000
                return [Preset(label: "Tiny steps", stride: tiny), Preset(label: "Just right", stride: right),
                        Preset(label: "Too big", stride: big), Preset(label: "Giant", stride: giant)]
            }.value
            self?.presets = found
        }
    }

    func choose(_ p: Preset) {
        stride = p.stride
        rehike()
    }

    func stop() { hikeTask?.cancel() }

    /// Where every hike starts: up the slope, away from the valley.
    var start: SIMD2<Float>? {
        guard let s = slice else { return nil }
        let a = s.aRange, b = s.bRange
        return SIMD2(a.lowerBound + 0.12 * (a.upperBound - a.lowerBound), b.lowerBound + 0.88 * (b.upperBound - b.lowerBound))
    }

    func setStride(log10 v: Double) {
        stride = Float(pow(10, v))
        rehike()
    }

    private func rehike() {
        guard let net = network, let s = slice, let l = ledger, let p = start else { return }
        serial += 1
        let id = serial, step = stride, count = Self.steps
        hikeTask?.cancel()
        hikeTask = Task { [weak self] in
            let h = await Task.detached(priority: .userInitiated) {
                net.hike(on: s, from: p, stepSize: step, steps: count, seed: 11, in: l)
            }.value
            guard let self, !Task.isCancelled, self.serial == id else { return }
            self.hike = h
            self.walk()
        }
    }

    /// The hiker walks its path, one step at a time; the button or Return jumps to the end.
    private func walk() {
        guard let h = hike else { return }
        hikeStart = Date()
        chor.start(.reveal, keys: [CameraKey(at: 0, focus: .wide, move: 0.01)],
                   duration: Double(max(h.points.count - 1, 1)) * Self.stepTime, captionAt: 0)
    }

    /// How the hike went, in words.
    enum Outcome { case settles, crawls, leaps, tumbles }
    var outcome: Outcome? {
        guard let h = hike, let s = slice else { return nil }
        return Self.outcome(h, minLoss: s.minLoss)
    }

    nonisolated static func rises(_ h: MountainHike) -> Int {
        var n = 0
        for k in 1..<max(h.losses.count, 1) where h.losses[k] > h.losses[k - 1] * 1.02 { n += 1 }
        return n
    }

    nonisolated static func outcome(_ h: MountainHike, minLoss: Float) -> Outcome? {
        guard let first = h.losses.first, let last = h.losses.last else { return nil }
        if h.tumbled { return .tumbles }
        if rises(h) >= 3 || last > first { return .leaps }
        let closed = (first - last) / max(first - minLoss, 1e-6)
        return closed > 0.8 ? .settles : .crawls
    }

    private var outcomeWords: String {
        guard let h = hike, let o = outcome, let first = h.losses.first, let last = h.losses.last else { return "" }
        let a = AirlineState.minutes(first), b = AirlineState.minutes(last)
        switch o {
        case .settles: return "It settles in the valley: off by \(a) → \(b) min in \(h.points.count - 1) steps."
        case .crawls: return "It crawls: off by \(a) → \(b) min after \(h.points.count - 1) steps, still walking."
        case .leaps: return "It leaps past the valley and back: off by \(a) min at the start, \(b) at the end."
        case .tumbles: return "It tumbles off the mountain: the step was so big the numbers blew up."
        }
    }

    var routeNames: (String, String) {
        guard let r = routes, let net = network else { return ("route A", "route B") }
        func name(_ id: RouteID) -> String {
            let from = id.layer == 0 ? net.architecture.inputs[id.from].title : MapModel.hubName(row: id.layer - 1, index: id.from)
            return "\(from) → \(MapModel.hubName(row: id.layer, index: id.to))"
        }
        return (name(r.0), name(r.1))
    }

    // MARK: the pages

    var copy: AirlinePageCopy {
        guard slice != nil else {
            return AirlinePageCopy(text: "Getting the mountain ready…", detail: "Training a two-row network on the logbook first.",
                                   button: "Continue", enabled: false, working: true)
        }
        switch page {
        case 0:
            let (a, b) = routeNames
            return AirlinePageCopy(
                text: "Training is a blindfolded hiker on a foggy mountain, feeling for downhill.",
                detail: "Height = how far off the guesses are. Left–right is the route \(a), up–down is \(b); every other route is held still. Dark is low.",
                button: "Continue")
        case 1:
            return AirlinePageCopy(
                text: "Try the stride: too small and it crawls, too big and it leaps past the valley.",
                detail: "Stride \(String(format: "%.3g", Double(stride))). " + outcomeWords,
                button: "Continue", reserveDetail: ["Stride 0.000. It leaps past the valley and back: off by 000 min at the start, 000 at the end."])
        default:
            return AirlinePageCopy(
                text: "Each step: feel the slope, step downhill, take a new flight from the logbook.",
                detail: "Training does this for every route at once, thousands of times. If a step is so big the numbers blow up, the trainer stops and says it tumbled.",
                button: "Done")
        }
    }

    func primary() -> Bool {
        guard slice != nil else { return false }
        if chor.isPlaying { chor.skip(); return false }
        if page == pageCount - 1 { return true }
        page += 1
        if page == 2, stride != Self.defaultStride { stride = Self.defaultStride; rehike() }
        return false
    }

    func back() {
        guard page > 0 else { return }
        page -= 1
    }

    func watchAgain() { walk() }

    var summary: String? {
        guard hike != nil else { return nil }
        return "Stride \(String(format: "%.3g", Double(stride))): " + outcomeWords
    }
}

// MARK: - the view

struct AirlineMountainLessonView: View {
    var onExit: (_ completed: Bool) -> Void
    @State private var model = AirlineMountainLesson()

    var body: some View {
        AirlineLessonShell(model: model, onExit: onExit) {
            Group {
                if let s = model.slice {
                    MountainView(slice: s, hike: model.hike, run: model.chor.run, active: model.chor.timelineActive,
                                 names: model.routeNames)
                        .padding(.horizontal, 40).padding(.vertical, 18)
                } else {
                    Spacer(minLength: 0)
                }
            }
            .frame(maxWidth: 900, maxHeight: .infinity)
        } accessory: { _ in
            if model.page == 1 {
                VStack(spacing: 10) {
                    HStack(spacing: 8) {
                        ForEach(model.presets) { p in
                            Button(p.label) { model.choose(p) }
                                .buttonStyle(PillButtonStyle(quiet: abs(model.stride - p.stride) > p.stride * 0.001))
                                .help("Stride \(String(format: "%.3g", Double(p.stride)))")
                        }
                    }
                    HStack(spacing: 10) {
                        Text("tiny").font(.system(size: 12)).foregroundStyle(.secondary)
                        Slider(value: Binding(get: { log10(Double(model.stride)) }, set: { model.setStride(log10: $0) }),
                               in: AirlineMountainLesson.strideRange)
                            .frame(width: 300)
                            .accessibilityLabel("Stride")
                            .accessibilityValue(String(format: "%.3g", Double(model.stride)))
                        Text("giant").font(.system(size: 12)).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .task { await model.load() }
    }
}

/// The slice as a contour map: dark = few minutes off (the valley), pale = many. The hiker is a dot with its
/// trail; every point is a real step of gradient descent on the two routes.
struct MountainView: View {
    let slice: MountainSlice
    let hike: MountainHike?
    let run: AirlineRun?
    let active: Bool
    let names: (String, String)

    var body: some View {
        VStack(spacing: 8) {
            Text("A SLICE OF THE MOUNTAIN: TWO ROUTES, THE REST HELD STILL")
                .font(.system(size: 12, weight: .semibold)).kerning(0.8).foregroundStyle(.white.opacity(0.55))
            ZStack {
                // the map is drawn once per slice; only the hiker redraws per frame
                Canvas { ctx, size in drawMap(&ctx, size) }
                SwiftUI.TimelineView(.animation(minimumInterval: nil, paused: !active)) { tl in
                    let t = run?.elapsed(at: active ? tl.date : Date()) ?? .infinity
                    Canvas { ctx, size in drawHiker(&ctx, size, t) }
                }
            }
            .aspectRatio(1, contentMode: .fit)
            HStack {
                Text("← \(names.0) →").font(.system(size: 14, weight: .medium)).foregroundStyle(.secondary)
                Spacer()
                Text("↕ \(names.1)").font(.system(size: 14, weight: .medium)).foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("A slice of the mountain: minutes off as \(names.0) and \(names.1) change. Lowest \(AirlineState.minutes(slice.minLoss)) minutes, highest \(AirlineState.minutes(slice.maxLoss)).")
    }

    private func drawMap(_ ctx: inout GraphicsContext, _ size: CGSize) {
        let n = slice.resolution
        guard n > 1, slice.loss.count == n * n else { return }
        let side = min(size.width, size.height)
        let cell = side / CGFloat(n)
        let lo = slice.minLoss, span = max(slice.maxLoss - slice.minLoss, 1e-6)
        func v(_ a: Int, _ b: Int) -> Float { sqrt(max(slice.loss[b * n + a] - lo, 0) / span) }   // sqrt: detail near the valley
        // height as lightness and saturation (no glow): deep teal valley → pale sand peaks
        for b in 0..<n {
            for a in 0..<n {
                let x = Double(v(a, b))
                let color = Color(hue: 0.52 - 0.42 * x, saturation: 0.55 - 0.3 * x, brightness: 0.18 + 0.7 * x)
                ctx.fill(Path(CGRect(x: CGFloat(a) * cell, y: side - CGFloat(b + 1) * cell, width: cell + 0.5, height: cell + 0.5)),
                         with: .color(color))
            }
        }
        // contour lines (marching squares) at even steps of the scaled height
        var contours = Path()
        for level in stride(from: Float(0.1), to: 1, by: 0.1) {
            for b in 0..<(n - 1) {
                for a in 0..<(n - 1) {
                    let c = [v(a, b), v(a + 1, b), v(a + 1, b + 1), v(a, b + 1)]
                    let p = [CGPoint(x: CGFloat(a) + 0.5, y: CGFloat(b) + 0.5), CGPoint(x: CGFloat(a) + 1.5, y: CGFloat(b) + 0.5),
                             CGPoint(x: CGFloat(a) + 1.5, y: CGFloat(b) + 1.5), CGPoint(x: CGFloat(a) + 0.5, y: CGFloat(b) + 1.5)]
                    var hits: [CGPoint] = []
                    for e in 0..<4 {
                        let (i, j) = (e, (e + 1) % 4)
                        if (c[i] < level) != (c[j] < level) {
                            let u = CGFloat((level - c[i]) / (c[j] - c[i]))
                            hits.append(CGPoint(x: p[i].x + (p[j].x - p[i].x) * u, y: p[i].y + (p[j].y - p[i].y) * u))
                        }
                    }
                    var k = 0
                    while k + 1 < hits.count {
                        contours.move(to: CGPoint(x: hits[k].x * cell, y: side - hits[k].y * cell))
                        contours.addLine(to: CGPoint(x: hits[k + 1].x * cell, y: side - hits[k + 1].y * cell))
                        k += 2
                    }
                }
            }
        }
        ctx.stroke(contours, with: .color(.white.opacity(0.18)), lineWidth: 0.8)

        // the valley floor of this slice
        if let m = slice.loss.indices.min(by: { slice.loss[$0] < slice.loss[$1] }) {
            let q = CGPoint(x: (CGFloat(m % n) + 0.5) * cell, y: side - (CGFloat(m / n) + 0.5) * cell)
            ctx.stroke(Path(ellipseIn: CGRect(x: q.x - 6, y: q.y - 6, width: 12, height: 12)), with: .color(Theme.prompt), lineWidth: 1.5)
        }

    }

    private func drawHiker(_ ctx: inout GraphicsContext, _ size: CGSize, _ t: Double) {
        let side = min(size.width, size.height)
        guard let h = hike, !h.points.isEmpty else { return }
        let a0 = slice.aRange.lowerBound, aw = slice.aRange.upperBound - a0
        let b0 = slice.bRange.lowerBound, bw = slice.bRange.upperBound - b0
        func pt(_ p: SIMD2<Float>) -> CGPoint {
            let x = CGFloat((p.x - a0) / aw), y = CGFloat((p.y - b0) / bw)
            return CGPoint(x: min(max(x, -0.05), 1.05) * side, y: side - min(max(y, -0.05), 1.05) * side)
        }
        let shown = min(h.points.count - 1, Int(t / AirlineMountainLesson.stepTime))
        var trail = Path()
        trail.move(to: pt(h.points[0]))
        for k in 0...max(shown, 0) where k > 0 { trail.addLine(to: pt(h.points[k])) }
        ctx.stroke(trail, with: .color(Theme.answer.opacity(0.85)), style: StrokeStyle(lineWidth: 1.6, lineCap: .round, lineJoin: .round))
        for k in 0...max(shown, 0) {
            let q = pt(h.points[k])
            ctx.fill(Path(ellipseIn: CGRect(x: q.x - 1.8, y: q.y - 1.8, width: 3.6, height: 3.6)), with: .color(Theme.answer.opacity(0.6)))
        }
        let here = pt(h.points[max(shown, 0)])
        ctx.fill(Path(ellipseIn: CGRect(x: here.x - 6, y: here.y - 6, width: 12, height: 12)), with: .color(Theme.answer))
        ctx.stroke(Path(ellipseIn: CGRect(x: here.x - 6, y: here.y - 6, width: 12, height: 12)), with: .color(.black.opacity(0.7)), lineWidth: 1.5)
        if h.tumbled && shown >= h.points.count - 1 {
            ctx.draw(Text("tumbled").font(.system(size: 12, weight: .bold)).foregroundColor(MapModel.blame),
                     at: CGPoint(x: min(max(here.x, 40), side - 40), y: min(max(here.y - 16, 12), side - 12)))
        }
        if let loss = h.losses.indices.contains(max(shown, 0)) ? h.losses[max(shown, 0)] : nil, loss.isFinite {
            ctx.draw(Text("off by \(AirlineState.minutes(loss)) min").font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundColor(Theme.answer),
                     at: CGPoint(x: min(max(here.x + 12, 0), side - 90), y: min(max(here.y + 14, 10), side - 10)), anchor: .leading)
        }
    }
}
