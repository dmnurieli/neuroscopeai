import SwiftUI
import AirlineKit

// The airline course's shared pieces: which lessons exist and their order, what carries between lessons
// (persisted), one ledger for every lesson, the background trainer, the train panel, the score bars and the
// edit strip for building a network.

// MARK: - the lessons

enum AirlineLessonID: String, CaseIterable {
    case loop = "airline-loop"
    case rulebook = "airline-rulebook"
    case clipboard = "airline-clipboard"
    case direct = "airline-direct"
    case hubs = "airline-hubs"
    case mountain = "airline-mountain"
    case hubNames = "airline-hubnames"
    case exam = "airline-exam"
    case close = "airline-close"
    case graduation = "airline-graduation"
    case builder = "airline-builder"

    /// The "Start here" rows, in unlock order (the Builder is separate: it opens after `hubs`).
    static let course: [AirlineLessonID] = [.loop, .rulebook, .clipboard, .direct, .hubs, .mountain, .hubNames, .exam, .close, .graduation]

    var title: String {
        switch self {
        case .loop: "Your first network"
        case .rulebook: "The rulebook"
        case .clipboard: "The clipboard"
        case .direct: "Direct routes"
        case .hubs: "Layover hubs"
        case .mountain: "The foggy mountain"
        case .hubNames: "Name your hubs"
        case .exam: "Exam day"
        case .close: "Close a route"
        case .graduation: "Graduation"
        case .builder: "Builder"
        }
    }
    var subtitle: String {
        switch self {
        case .loop: "Watch a tiny network guess a flight delay, get it wrong, and learn"
        case .rulebook: "Write the rules yourself, and see where rules run out"
        case .clipboard: "Choose the facts the network gets to see"
        case .direct: "The simplest network: every fact straight to the gate"
        case .hubs: "Add layover hubs where facts meet, then a second row"
        case .mountain: "Training as a blindfolded hiker feeling for downhill"
        case .hubNames: "Find out what each hub learned, and name it"
        case .exam: "The sealed years open: memorising versus understanding"
        case .close: "Close a route or a hub and see what breaks"
        case .graduation: "Meet a real, fully open AI and watch it learn, from noise to answers"
        case .builder: "Build any network you like and train it"
        }
    }
    var minutes: Int {
        switch self {
        case .loop: 4
        case .rulebook, .clipboard, .mountain, .exam: 3
        case .direct, .hubs, .hubNames, .close: 4
        case .graduation: 6
        case .builder: 5
        }
    }
    var symbol: String {
        switch self {
        case .loop: "airplane"
        case .rulebook: "list.bullet.rectangle"
        case .clipboard: "list.clipboard"
        case .direct: "arrow.right"
        case .hubs: "point.3.connected.trianglepath.dotted"
        case .mountain: "mountain.2"
        case .hubNames: "tag"
        case .exam: "lock.open"
        case .close: "xmark.octagon"
        case .graduation: "graduationcap"
        case .builder: "hammer"
        }
    }
}

// MARK: - what carries between lessons

@MainActor enum AirlineProgress {
    private static let d = UserDefaults.standard
    private static let clipboardKey = "neuroscope.airline.clipboard"
    private static let rulebookKey = "neuroscope.airline.rulebook"
    private static let directKey = "neuroscope.airline.directScore"
    private static let networkKey = "neuroscope.airline.networkScore"

    /// The clipboard cards chosen in lesson 2; lessons 3–5 and the Builder start from them. Default: all eight.
    static var clipboard: [Fact] {
        get {
            guard let raw = d.array(forKey: clipboardKey) as? [Int] else { return Fact.allCases }
            return raw.compactMap(Fact.init(rawValue:)).sorted { $0.rawValue < $1.rawValue }
        }
        set { d.set(newValue.map(\.rawValue).sorted(), forKey: clipboardKey) }
    }
    /// The learner's rulebook from lesson 1 (default: the beginner's two rules).
    static var rulebook: Rulebook {
        get {
            guard let data = d.data(forKey: rulebookKey), let r = try? JSONDecoder().decode(Rulebook.self, from: data) else { return .beginner }
            return r
        }
        set { if let data = try? JSONEncoder().encode(newValue) { d.set(data, forKey: rulebookKey) } }
    }
    /// Quiz scores (minutes off on the 2023 flights) of the learner's own trained networks.
    static var directScore: Float? {
        get { d.object(forKey: directKey) as? Float ?? (d.object(forKey: directKey) as? Double).map(Float.init) }
        set { if let v = newValue { d.set(Double(v), forKey: directKey) } else { d.removeObject(forKey: directKey) } }
    }
    static var networkScore: Float? {
        get { (d.object(forKey: networkKey) as? Double).map(Float.init) }
        set { if let v = newValue { d.set(Double(v), forKey: networkKey) } else { d.removeObject(forKey: networkKey) } }
    }
    /// The names the learner gave hubs in lesson 6, by hub code ("ATL" → "Storm spotter").
    static var hubNames: [String: String] {
        get { d.dictionary(forKey: "neuroscope.airline.hubNames") as? [String: String] ?? [:] }
        set { d.set(newValue, forKey: "neuroscope.airline.hubNames") }
    }

    /// The learner's trained two-row network from lesson 4, kept as JSON in Application Support so lessons 6–8
    /// study the network they built. A missing or unreadable file means "train one again".
    static var trainedURL: URL? {
        guard let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else { return nil }
        return dir.appendingPathComponent("NeuroScope", isDirectory: true).appendingPathComponent("airline-network.json")
    }
    static func saveTrained(_ net: Network) {
        guard let url = trainedURL, let data = try? JSONEncoder().encode(net) else { return }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }
    static func savedTrained() -> Network? {
        guard let url = trainedURL, let data = try? Data(contentsOf: url) else { return nil }
        guard let net = try? JSONDecoder().decode(Network.self, from: data), net.architecture.hidden.count == 2 else {
            NSLog("NeuroScope: ignoring an unreadable airline-network.json")
            return nil
        }
        return net
    }

    /// The learner's trained network, or (if lesson 4's was never saved) the same recipe trained now: their
    /// clipboard, two rows of six, six passes from the lesson seed.
    static func trainedNetwork(_ ledger: Ledger) async -> Network {
        if let net = savedTrained() { return net }
        var inputs = clipboard
        if inputs.isEmpty { inputs = Fact.allCases }
        let arch = Architecture(inputs: inputs, hidden: [6, 6])
        let net = await Task.detached(priority: .userInitiated) { () -> Network in
            let t = Trainer(network: Network(architecture: arch, ledger: ledger, seed: Architecture.lessonSeed), ledger: ledger)
            _ = t.train(batches: 6 * ((ledger.indices(.practice).count + 63) / 64))
            return t.network
        }.value
        saveTrained(net)
        return net
    }
}

extension HubLabel {
    /// "mostly holiday rush", "linked to storm", "a mix of storm and perfect storm", "a quiet hub".
    var words: String {
        switch self {
        case .mostly(let c): "mostly \(c.title.lowercased())"
        case .linked(let c): "linked to \(c.title.lowercased())"
        case .mix(let a, let b): "a mix of \(a.title.lowercased()) and \(b.title.lowercased())"
        case .quiet: "a quiet hub"
        }
    }
}

// MARK: - one ledger for every lesson

@MainActor enum AirlineLedger {
    private static var cached: Ledger?
    private static var loading: Task<Ledger, Never>?
    static func load() async -> Ledger {
        if let l = cached { return l }
        if let t = loading { return await t.value }
        let t = Task.detached(priority: .userInitiated) { Ledger.generate() }
        loading = t
        let l = await t.value
        cached = l
        return l
    }
}

// MARK: - the trainer, on a background task in chunks

/// Trains a network on the study pile (practice years) in chunks; after each chunk the map reshapes and the
/// quiz score (minutes off on the 2023 flights) gains a point. Any architecture edit rebuilds from the lesson seed.
@MainActor @Observable
final class TrainerHost {
    enum State: Equatable { case idle, running, paused, done, tumbled }

    let ledger: Ledger
    private(set) var network: Network
    private(set) var previousWeights: [[Float]]?
    private(set) var snapshotDate = Date.distantPast
    private(set) var routeOrder: [[Int]]
    private(set) var hotRoutes: Set<RouteID> = []
    private(set) var state: State = .idle
    private(set) var flightsStudied = 0
    private(set) var quiz: [LossPoint] = []
    private(set) var quizStart: Float
    private(set) var quizNow: Float
    /// The learning rate (the hiker's stride); applied at the next chunk.
    var stepSize: Float = TrainConfig().stepSize
    /// How many study flights one Train press covers (six passes through the pile by default).
    var budget: Int
    /// Called after every snapshot (the lesson keeps its map timeline ticking).
    @ObservationIgnored var onSnapshot: () -> Void = {}

    @ObservationIgnored private var trainer: Trainer
    @ObservationIgnored private var task: Task<Void, Never>?
    /// The chunk running on the background right now. A Trainer is touched by one chunk at a time: a resume waits
    /// for it, and a rebuild makes a new Trainer instead of resetting the busy one.
    @ObservationIgnored private var inFlight: Task<(TrainStatus, Network, Float, Int), Never>?
    @ObservationIgnored private var generation = 0

    static func sixPasses(_ ledger: Ledger) -> Int { 6 * ledger.indices(.practice).count }

    var studyPile: Int { ledger.indices(.practice).count }

    /// "six passes through the 84,000-flight study pile", so a count like 504,000 is not mysterious.
    var passesWords: String? {
        guard flightsStudied > 0, studyPile > 0 else { return nil }
        let p = Double(flightsStudied) / Double(studyPile)
        let pile = "the \(studyPile.formatted())-flight study pile"
        if p < 0.98 { return "\(Int((p * 100).rounded()))% of one pass through \(pile)" }
        let whole = p.rounded()
        if abs(p - whole) < 0.02 {
            let words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
            let n = Int(whole)
            let w = n <= words.count ? words[n - 1] : "\(n)"
            return "\(w) pass\(n == 1 ? "" : "es") through \(pile)"
        }
        return String(format: "%.1f passes through ", p) + pile
    }

    init(ledger: Ledger, architecture: Architecture) {
        self.ledger = ledger
        let net = Network(architecture: architecture, ledger: ledger, seed: Architecture.lessonSeed)
        network = net
        routeOrder = MapModel.routeOrder(net.weights)
        trainer = Trainer(network: net, ledger: ledger)
        let q = net.meanMinutesOff(.check, in: ledger)
        quizStart = q
        quizNow = q
        budget = Self.sixPasses(ledger)
    }

    var architecture: Architecture { network.architecture }
    var isRunning: Bool { state == .running }
    var hasTrained: Bool { flightsStudied > 0 }

    /// A new network for `architecture`, from the lesson seed; any training stops and starts over.
    func rebuild(_ architecture: Architecture) {
        stop()
        generation += 1
        inFlight = nil
        let net = Network(architecture: architecture, ledger: ledger, seed: Architecture.lessonSeed)
        trainer = Trainer(network: net, ledger: ledger)
        previousWeights = nil
        network = net
        routeOrder = MapModel.routeOrder(net.weights)
        hotRoutes = []
        snapshotDate = Date()
        flightsStudied = 0
        quiz = []
        quizStart = net.meanMinutesOff(.check, in: ledger)
        quizNow = quizStart
        state = .idle
        onSnapshot()
    }

    func stop() {
        task?.cancel()
        task = nil
        if state == .running { state = .paused }
    }

    func pause() { stop() }

    /// Trains until `budget` flights have been studied (resumes from a pause; a finished run starts another).
    func train() {
        guard state != .running, state != .tumbled else { return }
        if state == .done { budget += Self.sixPasses(ledger) }
        if quiz.isEmpty { quiz = [LossPoint(flights: 0, minutes: quizNow, at: Date())] }
        state = .running
        onSnapshot()
        let t = trainer, l = ledger, gen = generation
        let batchSize = max(1, t.config.batchSize)
        let chunkBatches = max(20, budget / batchSize / 60)
        let previous = inFlight
        task = Task { [weak self] in
            _ = await previous?.value   // a paused chunk may still be finishing on this trainer
            while let self, !Task.isCancelled, self.generation == gen, self.flightsStudied < self.budget {
                let step = self.stepSize
                let left = (self.budget - self.flightsStudied + batchSize - 1) / batchSize
                let n = min(chunkBatches, max(left, 1))
                let chunk = Task.detached(priority: .userInitiated) { () -> (TrainStatus, Network, Float, Int) in
                    t.config.stepSize = step
                    let s = t.train(batches: n)
                    let net = t.network
                    return (s, net, net.meanMinutesOff(.check, in: l), t.flightsSeen)
                }
                self.inFlight = chunk
                let (status, net, off, seen) = await chunk.value
                guard !Task.isCancelled, self.generation == gen else { return }
                self.publish(net)
                self.flightsStudied = seen
                if status == .tumbled { self.state = .tumbled; self.hotRoutes = []; self.onSnapshot(); return }
                self.quizNow = off
                self.quiz.append(LossPoint(flights: seen, minutes: off, at: Date()))
                try? await Task.sleep(for: .milliseconds(30))
            }
            guard let self, !Task.isCancelled, self.generation == gen else { return }
            self.state = .done
            self.hotRoutes = []
            self.onSnapshot()
        }
    }

    /// After a tumble: the same architecture, fresh from the lesson seed.
    func reset() { rebuild(network.architecture) }

    private func publish(_ net: Network) {
        let old = network.weights
        previousWeights = old
        network = net
        snapshotDate = Date()
        routeOrder = MapModel.routeOrder(net.weights)
        let sizes = net.layerSizes
        var moved: [(Float, RouteID)] = []
        for l in net.weights.indices where old.indices.contains(l) && old[l].count == net.weights[l].count {
            for idx in net.weights[l].indices {
                moved.append((abs(net.weights[l][idx] - old[l][idx]), RouteID(layer: l, from: idx / sizes[l + 1], to: idx % sizes[l + 1])))
            }
        }
        hotRoutes = Set(moved.sorted { $0.0 > $1.0 }.prefix(12).filter { $0.0 > 0 }.map(\.1))
        onSnapshot()
    }
}

// MARK: - architecture edits (the map's +/− controls)

enum ArchitectureEdit {
    static let maxHubs = 8, maxRows = 2

    static func toggling(_ f: Fact, in a: Architecture) -> Architecture {
        var inputs = a.inputs
        if let i = inputs.firstIndex(of: f) { inputs.remove(at: i) } else { inputs.append(f) }
        return Architecture(inputs: inputs.sorted { $0.rawValue < $1.rawValue }, hidden: a.hidden)
    }
    /// +1 / −1 hub on a row (the last hub removed removes the row; a hub added to a missing first row creates it).
    static func hubs(_ delta: Int, row: Int, in a: Architecture) -> Architecture {
        var hidden = a.hidden
        if row >= hidden.count {
            guard delta > 0, row == hidden.count, hidden.count < maxRows else { return a }
            hidden.append(1)
        } else {
            hidden[row] = min(max(hidden[row] + delta, 0), maxHubs)
            if hidden[row] == 0 { hidden.remove(at: row) }
        }
        return Architecture(inputs: a.inputs, hidden: hidden)
    }
    static func addRow(_ a: Architecture, hubs: Int = 6) -> Architecture {
        guard a.hidden.count < maxRows else { return a }
        return Architecture(inputs: a.inputs, hidden: a.hidden + [min(max(hubs, 1), maxHubs)])
    }
    static func removeRow(_ a: Architecture) -> Architecture {
        guard !a.hidden.isEmpty else { return a }
        return Architecture(inputs: a.inputs, hidden: Array(a.hidden.dropLast()))
    }
}

/// The edit controls a page allows, in a strip under the map: each control sits under the column it changes
/// (lay it out at the map's width). Pill buttons with the hub count between them; "+ row" where the next row
/// would go.
struct EditStrip: View {
    struct Allowed: OptionSet {
        let rawValue: Int
        static let hubsRow1 = Allowed(rawValue: 1), hubsRow2 = Allowed(rawValue: 2), rows = Allowed(rawValue: 4), cards = Allowed(rawValue: 8)
        static let all: Allowed = [.hubsRow1, .hubsRow2, .rows, .cards]
    }
    let architecture: Architecture
    let allowed: Allowed
    let apply: (Architecture) -> Void

    static let height: CGFloat = 52

    var body: some View {
        GeometryReader { geo in
            let g = MapGeometry(size: CGSize(width: geo.size.width, height: 400), sizes: [architecture.inputs.count] + architecture.hidden + [1])
            let mid = geo.size.height / 2
            ZStack {
                if allowed.contains(.cards) {
                    Text("Tap a card to switch it on or off")
                        .font(.system(size: 13)).foregroundStyle(.secondary)
                        .position(x: g.x(layer: 0), y: mid)
                }
                ForEach(Array(architecture.hidden.enumerated()), id: \.offset) { row, n in
                    if rowAllowed(row) {
                        rowControls(row, n).position(x: g.x(layer: row + 1), y: mid)
                    }
                }
                if architecture.hidden.isEmpty && allowed.contains(.hubsRow1) {
                    Button("+ hub") { apply(ArchitectureEdit.hubs(1, row: 0, in: architecture)) }
                        .buttonStyle(PillButtonStyle())
                        .position(x: (g.cardX + g.cardW + g.gateRect.minX) / 2, y: mid)
                }
                if allowed.contains(.rows) && architecture.hidden.count < ArchitectureEdit.maxRows {
                    let lastX = architecture.hidden.isEmpty ? g.cardX + g.cardW : g.x(layer: architecture.hidden.count)
                    Button("+ row") { apply(ArchitectureEdit.addRow(architecture, hubs: architecture.hidden.last ?? 6)) }
                        .buttonStyle(PillButtonStyle())
                        // clear of the last row's controls (about 250 pt wide) or of the "+ hub" pill
                        .position(x: architecture.hidden.isEmpty ? (lastX + g.gateRect.minX) / 2 + 95
                                    : max((lastX + g.gateRect.minX) / 2, lastX + 170), y: mid)
                        .help("Add a row of hubs")
                }
            }
        }
        .frame(height: Self.height)
    }

    private func rowAllowed(_ row: Int) -> Bool {
        (row == 0 && allowed.contains(.hubsRow1)) || (row == 1 && allowed.contains(.hubsRow2))
    }

    private func rowControls(_ row: Int, _ n: Int) -> some View {
        HStack(spacing: 8) {
            Button { apply(ArchitectureEdit.hubs(-1, row: row, in: architecture)) } label: { Image(systemName: "minus") }
                .buttonStyle(PillButtonStyle())
                .help(n == 1 ? "Remove the last hub (and the row)" : "One hub fewer")
                .accessibilityLabel("One hub fewer in row \(row + 1)")
            Text("\(n) hub\(n == 1 ? "" : "s")")
                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                .frame(minWidth: 62)
                .accessibilityLabel("\(n) hubs in row \(row + 1)")
            Button { apply(ArchitectureEdit.hubs(1, row: row, in: architecture)) } label: { Image(systemName: "plus") }
                .buttonStyle(PillButtonStyle())
                .disabled(n >= ArchitectureEdit.maxHubs)
                .help("One hub more")
                .accessibilityLabel("One hub more in row \(row + 1)")
            if allowed.contains(.rows) && row == architecture.hidden.count - 1 {
                Button("− row") { apply(ArchitectureEdit.removeRow(architecture)) }
                    .buttonStyle(PillButtonStyle(quiet: true))
                    .help("Remove this row")
            }
        }
    }
}

/// A clear, roomy pill for map edits (14 pt, a generous hit target).
struct PillButtonStyle: ButtonStyle {
    var quiet = false
    func makeBody(configuration: Configuration) -> some View { Pill(configuration: configuration, quiet: quiet) }

    private struct Pill: View {
        let configuration: ButtonStyleConfiguration
        let quiet: Bool
        @Environment(\.isEnabled) private var enabled
        var body: some View {
            configuration.label
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(enabled ? (quiet ? Color.secondary : Theme.answer) : Color.white.opacity(0.3))
                .padding(.horizontal, 14).padding(.vertical, 8)
                .frame(minWidth: 40, minHeight: 32)
                .background(Capsule().fill(Color.white.opacity(configuration.isPressed ? 0.2 : (quiet ? 0.05 : 0.11))))
                .overlay(Capsule().stroke(Color.white.opacity(enabled ? 0.2 : 0.08)))
                .contentShape(Capsule())
        }
    }
}

// MARK: - the train panel

struct TrainPanel: View {
    let host: TrainerHost
    /// The Builder shows its own Train / Pause; lessons drive training from the caption's one button.
    var showsControls = false
    var showsStepSize = false

    var body: some View {
        HStack(alignment: .center, spacing: 20) {
            if showsControls {
                Button(host.isRunning ? "Pause" : (host.state == .tumbled ? "Reset" : "Train")) {
                    switch host.state {
                    case .running: host.pause()
                    case .tumbled: host.reset()
                    default: host.train()
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.answer.opacity(0.9))
                .foregroundStyle(Color.black)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text("FLIGHTS STUDIED").font(.system(size: 9, weight: .semibold)).kerning(0.6).foregroundStyle(.white.opacity(0.45))
                Text(host.flightsStudied.formatted()).font(.system(size: 15, weight: .semibold, design: .monospaced))
                Text(host.passesWords ?? "study pile: \(host.studyPile.formatted()) flights")
                    .font(.system(size: 10)).foregroundStyle(.white.opacity(0.45))
            }
            VStack(alignment: .leading, spacing: 3) {
                Text("QUIZ SCORE (2023)").font(.system(size: 9, weight: .semibold)).kerning(0.6).foregroundStyle(.white.opacity(0.45))
                Text("off by \(AirlineState.minutes(host.quizNow)) min").font(.system(size: 15, weight: .semibold, design: .monospaced))
                    .foregroundStyle(Theme.answer)
            }
            QuizCurve(points: host.quiz, total: host.budget, start: host.quizStart)
                .frame(width: 220, height: 54)
            if showsStepSize {
                VStack(alignment: .leading, spacing: 2) {
                    Text("STEP SIZE \(String(format: "%.3g", Double(host.stepSize)))")
                        .font(.system(size: 9, weight: .semibold)).kerning(0.6).foregroundStyle(.white.opacity(0.45))
                    Slider(value: Binding(get: { log10(Double(host.stepSize)) }, set: { host.stepSize = Float(pow(10, $0)) }),
                           in: -3...1)
                        .frame(width: 130)
                        .accessibilityLabel("Step size")
                        .accessibilityValue(String(format: "%.3g", Double(host.stepSize)))
                }
            }
            if host.state == .tumbled {
                Text("Tumbled: a step was too big.").font(.system(size: 12, weight: .semibold)).foregroundStyle(MapModel.blame)
            }
        }
        .padding(.horizontal, 18).padding(.vertical, 10)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color.white.opacity(0.08)))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Studied \(host.flightsStudied) flights\(host.passesWords.map { ", \($0)" } ?? ""); quiz score off by \(AirlineState.minutes(host.quizNow)) minutes")
    }
}

/// Minutes off on the quiz pile against flights studied, point by point.
struct QuizCurve: View {
    let points: [LossPoint]
    let total: Int
    let start: Float

    var body: some View {
        Canvas { ctx, size in
            let top = max(points.map(\.minutes).max() ?? start, start, 1) * 1.1
            let span = max(total, points.last?.flights ?? 1, 1)
            func p(_ q: LossPoint) -> CGPoint {
                CGPoint(x: size.width * CGFloat(q.flights) / CGFloat(span), y: size.height - size.height * CGFloat(q.minutes / top))
            }
            var base = Path()
            base.move(to: CGPoint(x: 0, y: size.height - 0.5)); base.addLine(to: CGPoint(x: size.width, y: size.height - 0.5))
            ctx.stroke(base, with: .color(.white.opacity(0.2)), lineWidth: 1)
            guard let first = points.first else {
                ctx.draw(Text("quiz score: minutes off").font(.system(size: 10)).foregroundColor(.white.opacity(0.35)),
                         at: CGPoint(x: size.width / 2, y: size.height / 2))
                return
            }
            var line = Path()
            line.move(to: p(first))
            for q in points.dropFirst() { line.addLine(to: p(q)) }
            ctx.stroke(line, with: .color(Theme.answer.opacity(0.9)), style: StrokeStyle(lineWidth: 1.6, lineCap: .round, lineJoin: .round))
            if let last = points.last {
                let e = p(last)
                ctx.fill(Path(ellipseIn: CGRect(x: e.x - 2.5, y: e.y - 2.5, width: 5, height: 5)), with: .color(Theme.answer))
            }
        }
        .accessibilityHidden(true)
    }
}

// MARK: - score bars (shorter = closer)

struct ScoreBars: View {
    struct Bar: Identifiable {
        var label: String
        var minutes: Float?    // nil: "?"
        var highlight = false
        var id: String { label }
    }
    let bars: [Bar]

    var body: some View {
        let top = max(bars.compactMap(\.minutes).max() ?? 1, 1)
        VStack(alignment: .leading, spacing: 8) {
            ForEach(bars) { bar in
                HStack(spacing: 10) {
                    Text(bar.label).font(.system(size: 12)).foregroundStyle(.secondary).frame(width: 120, alignment: .trailing)
                    GeometryReader { geo in
                        Capsule().fill(bar.highlight ? Theme.answer : Color.white.opacity(0.3))
                            .frame(width: max(4, geo.size.width * CGFloat((bar.minutes ?? 0) / top)), height: 8)
                            .frame(maxHeight: .infinity)
                            .opacity(bar.minutes == nil ? 0 : 1)
                    }
                    .frame(width: 220, height: 14)
                    Text(bar.minutes.map { "\(AirlineState.minutes($0)) min off" } ?? "?")
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(bar.highlight ? Theme.answer : .secondary)
                        .frame(width: 90, alignment: .leading)
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("\(bar.label): " + (bar.minutes.map { "off by \(AirlineState.minutes($0)) minutes" } ?? "not known yet"))
            }
        }
    }
}
