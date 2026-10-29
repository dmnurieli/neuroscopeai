// AirlineKit: the tiny teaching network and its synthetic flight ledger. Pure Swift, no UI, no I/O.
//
// The public API of AirlineKit; the app codes against it. Implementations live in World.swift, NetCore.swift,
// Lessons.swift and HubCards.swift.
// Course spec: docs/airline-course.md
//
// Vocabulary (the course's): facts on the dispatcher's CLIPBOARD (inputs) → rows of layover HUBS (hidden
// ReLU units) → the destination GATE (one output: predicted delay in minutes). Route = weight, resting
// stress = bias, the gate agent's filter = ReLU.

import Foundation

// MARK: - Randomness we own (identical on every Mac, every run)

public struct SplitMix64: RandomNumberGenerator, Sendable {
    public var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

// MARK: - The clipboard facts

/// One fact on the dispatcher's clipboard. Order = column order in `Ledger.facts`.
public enum Fact: Int, CaseIterable, Codable, Sendable, Hashable {
    case rain        // inches of rain at the origin
    case wind        // wind speed, mph
    case controllers // air traffic controllers on duty at the destination
    case load        // passengers ÷ seats, 0…1
    case holiday     // 1 = holiday week, 0 = not
    case hour        // scheduled departure hour, 6…22
    case crews       // spare maintenance crews on shift
    case shoe        // the pilot's shoe colour, 0…4 (a decoy: means nothing)

    /// Plain-English label for the clipboard card, e.g. "Rain".
    public var title: String {
        switch self {
        case .rain: "Rain"
        case .wind: "Wind"
        case .controllers: "Controllers on duty"
        case .load: "Plane"
        case .holiday: "Holiday week"
        case .hour: "Departure"
        case .crews: "Spare crews"
        case .shoe: "Pilot's shoes"
        }
    }
    /// Unit or scale for display, e.g. "inches", "mph", "% full".
    public var unit: String {
        switch self {
        case .rain: "inches"
        case .wind: "mph"
        case .controllers: "on duty"
        case .load: "% full"
        case .holiday: "yes or no"
        case .hour: "time of day"
        case .crews: "on shift"
        case .shoe: "colour"
        }
    }
    /// Formats a raw value for a card, e.g. 1.8 → "1.8 in", 0.96 → "96% full", 1 → "Holiday week".
    public func format(_ value: Float) -> String {
        let whole = Int(value.rounded())
        switch self {
        case .rain: return String(format: "%.1f in", Double(value))
        case .wind: return "\(whole) mph"
        case .controllers, .crews: return "\(whole)"
        case .load: return "\(Int((value * 100).rounded()))% full"
        case .holiday: return value >= 0.5 ? "Yes" : "No"
        case .hour:
            let h = ((whole % 24) + 24) % 24
            return h == 0 ? "12 am" : h < 12 ? "\(h) am" : h == 12 ? "12 pm" : "\(h - 12) pm"
        case .shoe: return Fact.shoeColours[min(max(whole, 0), Fact.shoeColours.count - 1)]
        }
    }
    static let shoeColours = ["Black", "Brown", "Red", "Blue", "White"]
}

/// The hidden rulebook's concepts, stored per flight so the app can CHECK what a hub learned.
/// Each is 0…1. Never shown as an input; the network never sees them.
public enum Concept: Int, CaseIterable, Codable, Sendable {
    case storm        // rain AND wind together
    case shortStaff   // too few controllers
    case holidayRush  // holiday week AND a full plane
    case perfectStorm // storm while short-staffed, worse in a rush
    public var title: String {   // "Storm", "Short-staffed", "Holiday rush", "Perfect storm"
        switch self {
        case .storm: "Storm"
        case .shortStaff: "Short-staffed"
        case .holidayRush: "Holiday rush"
        case .perfectStorm: "Perfect storm"
        }
    }
}

// MARK: - The ledger

public enum Split: Sendable {
    case practice   // years 0…6 (labelled 2016–2022): training only
    case check      // year 7 (2023): comparisons, hub naming, rulebook score while editing
    case exam       // years 8…9 (2024–2025): sealed; read ONLY on exam day
}

public struct Flight: Sendable {
    public var index: Int
    public var year: Int              // 0…9 → 2016 + year
    public var dayOfYear: Int         // 1…365
    public var origin: String         // one of MIA JFK BOS LAX SFO LAS MCO IAD SAN AUS (never a hub name)
    public var destination: String    // ditto, != origin
    public var scheduledDeparture: Int  // minutes after midnight
    public var scheduledArrival: Int
    public var actualDeparture: Int
    public var actualArrival: Int     // scheduledArrival + round(delay)
    public var passengers: Int
    public var seats: Int
    public var facts: [Float]         // raw values, Fact.allCases order
    public var delay: Float           // ground truth, minutes late (negative = early), clamped −15…300
    public var concepts: [Float]      // Concept.allCases order
}

public final class Ledger: @unchecked Sendable {
    public static let defaultSeed: UInt64 = 2026
    public static let defaultCount = 120_000

    /// Deterministic: the same seed and count always give byte-identical flights (the world:
    /// weather SYSTEMS make stormy days wet AND windy; drizzly-calm and dry-breezy days happen alone;
    /// delay comes from the conjunctions, not the raw facts). < 0.3 s for the default size.
    public static func generate(seed: UInt64 = defaultSeed, count: Int = defaultCount) -> Ledger {
        Ledger(columns: World.generate(seed: seed, count: max(count, 0)))
    }

    public var count: Int { columns.count }
    public func flight(_ i: Int) -> Flight {
        let c = columns
        let late = Int(c.delays[i].rounded())
        let departure = Int(c.scheduledDeparture[i]), arrival = Int(c.scheduledArrival[i])
        return Flight(
            index: i, year: Int(c.years[i]), dayOfYear: Int(c.days[i]),
            origin: World.airports[Int(c.origins[i])], destination: World.airports[Int(c.destinations[i])],
            scheduledDeparture: departure, scheduledArrival: arrival,
            actualDeparture: departure + max(0, late),   // never pushes back early; early arrivals make it up in the air
            actualArrival: arrival + late,
            passengers: Int(c.passengers[i]), seats: Int(c.seats[i]),
            facts: Array(c.facts[(i * World.factCount)..<((i + 1) * World.factCount)]),
            delay: c.delays[i],
            concepts: Array(c.concepts[(i * World.conceptCount)..<((i + 1) * World.conceptCount)]))
    }
    /// Flight indices in a split, ascending.
    public func indices(_ split: Split) -> [Int] {
        switch split {
        case .practice: practiceIndices
        case .check: checkIndices
        case .exam: examIndices
        }
    }
    /// Raw fact value for flight i.
    public func fact(_ f: Fact, _ i: Int) -> Float { columns.facts[i * World.factCount + f.rawValue] }
    public func delay(_ i: Int) -> Float { columns.delays[i] }
    public func concept(_ c: Concept, _ i: Int) -> Float { columns.concept(c, i) }

    /// Mean and standard deviation of each fact and of delay, measured on PRACTICE years only.
    public var factMean: [Float] { practiceStats.factMean }
    public var factSD: [Float] { practiceStats.factSD }
    public var delayMean: Float { practiceStats.delayMean }
    public var delaySD: Float { practiceStats.delaySD }

    /// The memorable flight for the minimal loop: the practice flight with the largest perfect-storm value
    /// (ties → lowest index). "The Thanksgiving flight."
    public var showcaseFlight: Int { showcaseIndex }
    /// A CHECK-year flight the network never trained on, with storm ≥ 0.8 (lowest index).
    public var unseenStormFlight: Int { unseenStormIndex }   // (no such flight → the stormiest check flight)

    // Storage (World.swift): flat columns built once by `generate`, plus what is derived from them.
    let columns: WorldColumns
    let practiceIndices: [Int], checkIndices: [Int], examIndices: [Int]
    let practiceStats: WorldStats
    let showcaseIndex: Int, unseenStormIndex: Int

    init(columns: WorldColumns) {
        self.columns = columns
        let s = columns.splitIndices()
        practiceIndices = s.practice; checkIndices = s.check; examIndices = s.exam
        practiceStats = columns.stats(over: s.practice)
        showcaseIndex = columns.showcase(in: s.practice)
        unseenStormIndex = columns.firstStorm(in: s.check)
    }
}

// MARK: - The network

public struct Architecture: Codable, Equatable, Hashable, Sendable {
    public var inputs: [Fact]     // clipboard cards in use (may be empty: the network can only guess the average)
    public var hidden: [Int]      // 0…2 rows, each 1…8 hubs
    public init(inputs: [Fact], hidden: [Int]) { self.inputs = inputs; self.hidden = hidden }
    public static let lesson = Architecture(inputs: Fact.allCases, hidden: [6, 6])
    /// The network seed every lesson uses. Tested (AirlineKitTests): on this seed a row-1 hub learns "storm"
    /// (signed r ≈ 0.81). Seed 2026 learns storm only in row 2, so the lesson must not use it.
    public static let lessonSeed: UInt64 = 42
}

/// One forward pass, kept for drawing: what every hub tallied and whether its gate agent let it through.
public struct Activations: Sendable {
    public var inputs: [Float]        // standardised inputs actually fed in
    public var hubs: [[Float]]        // per hidden row, after ReLU
    public var preActivation: [[Float]] // per hidden row, before ReLU (the tally incl. resting stress)
    public var gateMinutes: Float     // the prediction, in minutes
}

public struct Network: Codable, Equatable, Sendable {
    public var architecture: Architecture
    /// weights[l][i * outCount + j] = route from unit i of layer l to unit j of layer l+1 (row-major).
    /// Layer 0 = clipboard; last = gate (1 unit).
    public var weights: [[Float]]
    public var biases: [[Float]]      // resting stress per hub row, then the gate
    /// Normalisation copied from the ledger at creation (practice years), so a saved network stands alone.
    public var inputMean: [Float]
    public var inputSD: [Float]
    public var delayMean: Float
    public var delaySD: Float

    /// He-initialised from the seed (the toddler spinning the dials). Same seed + arch + ledger → same network.
    public init(architecture: Architecture, ledger: Ledger, seed: UInt64) {
        self.architecture = architecture
        inputMean = architecture.inputs.map { ledger.factMean[$0.rawValue] }
        inputSD = architecture.inputs.map { ledger.factSD[$0.rawValue] }
        delayMean = ledger.delayMean
        delaySD = ledger.delaySD
        let sizes = [architecture.inputs.count] + architecture.hidden + [1]
        var rng = SplitMix64(seed: seed)
        var w: [[Float]] = [], b: [[Float]] = []
        for l in 0..<(sizes.count - 1) {
            let fanIn = sizes[l], fanOut = sizes[l + 1]
            let sd = fanIn > 0 ? (2 / Double(fanIn)).squareRoot() : 0
            w.append((0..<(fanIn * fanOut)).map { _ in Float(rng.normal(0, sd)) })
            b.append([Float](repeating: 0, count: fanOut))
        }
        weights = w
        biases = b
    }

    public var layerSizes: [Int] { [architecture.inputs.count] + architecture.hidden + [1] }   // [inputs.count, hidden..., 1]
    public func forward(flight i: Int, in ledger: Ledger) -> Activations {
        let x = architecture.inputs.enumerated().map { k, f in (ledger.fact(f, i) - inputMean[k]) / inputSD[k] }
        var hubs: [[Float]] = [], pre: [[Float]] = []
        var current = x, out: Float = 0
        let sizes = layerSizes
        for l in 0..<(sizes.count - 1) {
            let nin = sizes[l], nout = sizes[l + 1]
            var z = [Float](repeating: 0, count: nout)
            for j in 0..<nout {
                var s = biases[l][j]
                for i in 0..<nin { s += current[i] * weights[l][i * nout + j] }
                z[j] = s
            }
            if l == sizes.count - 2 {
                out = z[0]
            } else {
                pre.append(z)
                current = z.map { max(0, $0) }
                hubs.append(current)
            }
        }
        return Activations(inputs: x, hubs: hubs, preActivation: pre, gateMinutes: out * delaySD + delayMean)
    }
    public func predict(flight i: Int, in ledger: Ledger) -> Float { forward(flight: i, in: ledger).gateMinutes }   // minutes
    /// Average minutes off over a split (or over the given indices).
    public func meanMinutesOff(_ split: Split, in ledger: Ledger) -> Float {
        meanMinutesOff(indices: ledger.indices(split), in: ledger)
    }
    public func meanMinutesOff(indices: [Int], in ledger: Ledger) -> Float {
        guard !indices.isEmpty else { return 0 }
        let core = NetCore(self), cols = inputColumns
        let x = UnsafeMutablePointer<Float>.allocate(capacity: max(cols.count, 1))
        defer { x.deallocate() }
        var total = 0.0
        for i in indices {
            standardise(flight: i, in: ledger, columns: cols, into: x)
            total += Double(abs(core.forward(x) * delaySD + delayMean - ledger.delay(i)))
        }
        return Float(total / Double(indices.count))
    }
}

/// A route identified by (layer, from, to): layer 0 = clipboard → first row (or → gate when no hubs).
public struct RouteID: Hashable, Codable, Sendable {
    public var layer: Int, from: Int, to: Int
    public init(layer: Int, from: Int, to: Int) { self.layer = layer; self.from = from; self.to = to }
}

public struct RouteChange: Sendable {
    public var route: RouteID
    public var before: Float
    public var after: Float
    /// How much of the blame this route carried (|gradient|), for colouring the backward walk.
    public var blame: Float
}

/// One hide-and-seek round on a single flight: hide the answer, guess, grade, step downhill once.
public struct LearnStep: Sendable {
    public var flight: Int
    public var guessBefore: Float     // minutes
    public var truth: Float
    public var guessAfter: Float      // same flight, after the step
    public var changes: [RouteChange] // every route, sorted by |after − before| descending
    /// The backward signal on this flight, BEFORE the step: ∂loss/∂(pre-activation)
    /// of every stop after its gate agent's filter, one array per hidden row, then the gate's (count 1). Signed, in
    /// the standardised-loss units the step uses: positive = this stop's tally pushed the guess too high (it gets
    /// turned down), negative = too low, exactly 0 where the gate agent held the passengers (ReLU shut), so no
    /// blame passes back through that hub. A route's blame is |activation_from × hubBlame_to|, where
    /// activation_from is the standardised input (clipboard) or the hub's activation. Also filled when a step
    /// tumbles and is undone (it is the signal that was computed, not the change made).
    public var hubBlame: [[Float]]
}

public struct TrainConfig: Codable, Equatable, Sendable {
    public var stepSize: Float = 0.05    // the hiker's stride (learning rate)
    public var batchSize: Int = 64
    public var routeUpkeep: Float = 3e-4 // L1 on clipboard routes: useless routes slowly close
    public init() {}
}

public enum TrainStatus: Equatable, Sendable {
    case ok
    case tumbled   // a weight became NaN/∞ (step too big): training stopped, network left as before the bad step
}

/// Synchronous training core. The app runs it on a background task and publishes snapshots.
/// Uses PRACTICE flights only. Mini-batches are drawn from a shuffle seeded by `seed`.
public final class Trainer: @unchecked Sendable {
    public private(set) var network: Network
    public var config: TrainConfig
    public private(set) var flightsSeen: Int = 0
    /// `studyPile`: train on just these flights (exam day's "crammer"); anything
    /// outside the practice years is dropped, so the sealed years stay sealed. nil = every practice flight.
    public init(network: Network, ledger: Ledger, config: TrainConfig = TrainConfig(), seed: UInt64 = 7,
                studyPile: [Int]? = nil) {
        self.network = network
        self.config = config
        self.ledger = ledger
        self.seed = seed
        if let pile = studyPile {
            let allowed = Set(ledger.indices(.practice))
            practice = pile.filter { allowed.contains($0) }
        } else {
            practice = ledger.indices(.practice)
        }
        rng = SplitMix64(seed: seed)
        core = NetCore(network)
        rebuild()
    }

    // Working state: the network's parameters live in `core` while training and are copied back to `network`
    // after every call; practice inputs are standardised once per network into `xs` / `ys`.
    private let ledger: Ledger
    private let seed: UInt64
    private let practice: [Int]
    private var rng: SplitMix64
    private var core: NetCore
    private var order: [Int32] = []          // shuffled practice positions
    private var position = 0                 // next unread entry in `order`
    private var xs: UnsafeMutablePointer<Float> = .allocate(capacity: 1)
    private var ys: UnsafeMutablePointer<Float> = .allocate(capacity: 1)
    private var scratch: UnsafeMutablePointer<Float> = .allocate(capacity: 1)
    private var maxInput: Float = 1          // largest |standardised input| among practice flights

    deinit { xs.deallocate(); ys.deallocate(); scratch.deallocate() }

    /// Fresh core, standardised practice columns and shuffle for the current network.
    private func rebuild() {
        core = NetCore(network)
        let cols = network.inputColumns, nIn = cols.count, n = practice.count
        xs.deallocate(); ys.deallocate(); scratch.deallocate()
        xs = .allocate(capacity: max(n * nIn, 1))
        ys = .allocate(capacity: max(n, 1))
        scratch = .allocate(capacity: max(nIn, 1))
        for (k, i) in practice.enumerated() {
            network.standardise(flight: i, in: ledger, columns: cols, into: xs + k * nIn)
            ys[k] = network.standardisedTarget(flight: i, in: ledger)
        }
        maxInput = 1
        for k in 0..<(n * nIn) { maxInput = max(maxInput, abs(xs[k])) }
        rng = SplitMix64(seed: seed)
        order = (0..<Int32(n)).map { $0 }
        rng.shuffle(&order)
        position = 0
    }

    /// Runs `batches` mini-batch steps (Huber(1) on the standardised target + routeUpkeep·|w| on clipboard routes).
    /// `.tumbled` when a step leaves a route or stress NaN/∞, or so large that the guess itself would overflow to
    /// ∞; the network is then put back as it was before that step.
    public func train(batches: Int) -> TrainStatus {
        let n = practice.count, nIn = network.architecture.inputs.count
        guard n > 0, batches > 0 else { return .ok }
        let size = max(config.batchSize, 1)
        for _ in 0..<batches {
            if position >= n { rng.shuffle(&order); position = 0 }
            let count = min(size, n - position)   // the last batch of a pass may be short, as in any epoch
            let scale = 1 / Float(count)
            core.save()
            core.zeroGradient()
            for k in position..<(position + count) {
                let s = Int(order[k])
                _ = core.forward(xs + s * nIn)
                _ = core.accumulate(target: ys[s], scale: scale)
            }
            core.addUpkeep(config.routeUpkeep)
            core.step(config.stepSize)
            guard !core.hasTumbled(maxInput: maxInput) else {
                core.restore()
                core.store(into: &network)
                return .tumbled
            }
            position += count
            flightsSeen += count
        }
        core.store(into: &network)
        return .ok
    }
    /// One round on one flight with the given stride (the minimal loop uses a bigger stride so the change is
    /// visible; same algorithm). Returns the before/after guess and every route's change.
    ///
    /// The step is the plain Huber gradient on this one flight; route upkeep is left out so every change on the
    /// map is the flight's own blame. Any flight index works (the app uses practice flights). A step that would
    /// tumble is undone, so `changes` then shows no movement.
    public func learn(fromFlight i: Int, stepSize: Float) -> LearnStep {
        let cols = network.inputColumns
        network.standardise(flight: i, in: ledger, columns: cols, into: scratch)
        let target = network.standardisedTarget(flight: i, in: ledger)
        let (sd, mean) = (network.delaySD, network.delayMean)
        let before = core.forward(scratch) * sd + mean
        core.zeroGradient()
        _ = core.accumulate(target: target, scale: 1)
        let hubBlame = core.deltas()   // read before the step: the signal this flight sent back
        core.save()
        core.step(stepSize)
        var reach = maxInput
        for k in 0..<cols.count { reach = max(reach, abs(scratch[k])) }
        if core.hasTumbled(maxInput: reach) { core.restore() }
        let after = core.forward(scratch) * sd + mean
        var changes: [(order: Int, change: RouteChange)] = []
        for l in 0..<core.layers {
            let nin = core.sizes[l], nout = core.sizes[l + 1], o = core.wOff[l]
            for from in 0..<nin { for to in 0..<nout {
                let k = o + from * nout + to
                changes.append((changes.count, RouteChange(route: RouteID(layer: l, from: from, to: to),
                                                           before: core.saved[k], after: core.p[k], blame: abs(core.g[k]))))
            } }
        }
        changes.sort {
            let (a, b) = (abs($0.change.after - $0.change.before), abs($1.change.after - $1.change.before))
            return a != b ? a > b : $0.order < $1.order
        }
        core.store(into: &network)
        flightsSeen += 1
        return LearnStep(flight: i, guessBefore: before, truth: ledger.delay(i), guessAfter: after,
                         changes: changes.map(\.change), hubBlame: hubBlame)
    }
    /// Replace the network (e.g. after an architecture edit); resets flightsSeen.
    /// The shuffle restarts from the trainer's seed, so a rebuilt map trains the same way every time.
    public func reset(to network: Network) {
        self.network = network
        flightsSeen = 0
        rebuild()
    }
}

// MARK: - The learner's rulebook

public struct Condition: Codable, Equatable, Hashable, Sendable {
    public var fact: Fact
    public var above: Bool        // true: "is above", false: "is below"
    public var threshold: Float   // raw units
    public init(fact: Fact, above: Bool, threshold: Float) { self.fact = fact; self.above = above; self.threshold = threshold }
}

public struct Rule: Codable, Equatable, Hashable, Sendable {
    public var conditions: [Condition]   // 1…2, joined by AND
    public var addMinutes: Float
    public init(conditions: [Condition], addMinutes: Float) { self.conditions = conditions; self.addMinutes = addMinutes }
}

public struct Rulebook: Codable, Equatable, Sendable {
    public var rules: [Rule]             // 0…3
    public init(rules: [Rule] = []) { self.rules = rules }
    /// "Everyone arrives on time" plus every rule that fires.
    public func predict(flight i: Int, in ledger: Ledger) -> Float {
        var minutes: Float = 0
        for rule in rules where rule.conditions.allSatisfy({ c in
            let v = ledger.fact(c.fact, i)
            return c.above ? v > c.threshold : v < c.threshold
        }) {
            minutes += rule.addMinutes
        }
        return minutes
    }
    public func meanMinutesOff(_ split: Split, in ledger: Ledger) -> Float {
        let idx = ledger.indices(split)
        guard !idx.isEmpty else { return 0 }
        var total = 0.0
        for i in idx { total += Double(abs(predict(flight: i, in: ledger) - ledger.delay(i))) }
        return Float(total / Double(idx.count))
    }
    /// The rulebook a beginner writes first: wind above 30 → +60; rain above 2 → +30.
    public static let beginner = Rulebook(rules: [
        Rule(conditions: [Condition(fact: .wind, above: true, threshold: 30)], addMinutes: 60),
        Rule(conditions: [Condition(fact: .rain, above: true, threshold: 2)], addMinutes: 30),
    ])
}

// MARK: - Lesson helpers and the foggy mountain

extension Rulebook {
    /// Indices of the rules that fire for flight i (all conditions hold).
    public func firingRules(flight i: Int, in ledger: Ledger) -> [Int] { _firingRules(flight: i, in: ledger) }
}

extension Ledger {
    /// Lowest-index flight in `split` satisfying `predicate` (raw facts, delay, concepts readable via the ledger).
    public func firstFlight(in split: Split, where predicate: (Int) -> Bool) -> Int? { _firstFlight(in: split, where: predicate) }
    /// The rulebook lesson's "29-mph flight": a CHECK-year flight with wind 25–30 mph (just under the beginner's
    /// 30-mph rule), rain ≥ 1 in, controllers ≤ 8 and delay ≥ 90 min; wind closest to (below) 29.5 wins, so the card reads "29 mph" (then larger delay). nil if none.
    public var nearMissFlight: Int? { _nearMissFlight }
    /// The 29-mph flight for a learner's OWN rulebook: `nearMissFlight` while none of `book`'s rules fire on it;
    /// otherwise the stormy (rain ≥ 1 in), short-staffed (controllers ≤ 8), ≥ 90-min-late CHECK flight where no
    /// rule fires and one condition misses its threshold by the least, in standard units (then larger delay).
    /// Falls back to `nearMissFlight` when the rules catch every such flight.
    public func nearMissFlight(for book: Rulebook) -> Int? { _nearMissFlight(for: book) }
    /// The failed condition of `book` closest to its threshold on flight i, with its gap in standard units.
    public func closestMiss(flight i: Int, of book: Rulebook) -> (rule: Int, condition: Condition, gap: Float)? {
        _closestMiss(flight: i, of: book)
    }
    /// A CHECK-year compound flight for the direct-routes lesson: perfect storm ≥ 0.7 (largest delay wins).
    public var compoundFlight: Int? { _compoundFlight }
}

/// The foggy mountain: the training loss as a function of TWO routes, every other weight held still. It is a
/// slice of the real landscape (say so in the UI), measured on a fixed sample of practice flights.
public struct MountainSlice: Sendable {
    public var routeA: RouteID
    public var routeB: RouteID
    public var aRange: ClosedRange<Float>     // weight values on the x axis
    public var bRange: ClosedRange<Float>     // weight values on the y axis
    public var resolution: Int                // grid is resolution × resolution
    public var loss: [Float]                  // row-major [b][a], mean minutes off (MAE in minutes) on the sample
    public var minLoss: Float
    public var maxLoss: Float
    /// The practice flights the loss was measured on (internal: `hike` descends on the same sample).
    var sample: [Int] = []
}

/// One hike: the positions visited by plain gradient steps on the slice (the two routes only), starting from `start`.
public struct MountainHike: Sendable {
    public var points: [SIMD2<Float>]         // (a, b) after each step, starting point first
    public var losses: [Float]                // minutes off at each point
    public var tumbled: Bool                  // left the grid's range or went non-finite (stride too big)
}

extension Network {
    /// Builds the slice around the current values of `a` and `b` (±`span` each), on `sampleCount` practice flights
    /// chosen by `seed`. The lesson uses two routes INTO one row-1 hub (upstream, so the hill can have bumps).
    /// resolution 41, sampleCount 2,000 must run < 0.3 s in release.
    public func mountain(a: RouteID, b: RouteID, span: Float, resolution: Int, sampleCount: Int, seed: UInt64,
                         in ledger: Ledger) -> MountainSlice { _mountain(a: a, b: b, span: span, resolution: resolution,
                                                                         sampleCount: sampleCount, seed: seed, in: ledger) }
    /// `steps` gradient steps on just those two routes (same loss as training, on the slice's sample), from `start`.
    public func hike(on slice: MountainSlice, from start: SIMD2<Float>, stepSize: Float, steps: Int, seed: UInt64,
                     in ledger: Ledger) -> MountainHike { _hike(on: slice, from: start, stepSize: stepSize, steps: steps,
                                                                 seed: seed, in: ledger) }
    /// The two routes the mountain lesson uses: the two largest-|weight| clipboard routes into the row-1 hub most
    /// linked to Concept.storm on the check year (signed). nil for networks with no hidden rows.
    public func mountainRoutes(in ledger: Ledger) -> (RouteID, RouteID)? { _mountainRoutes(in: ledger) }
}

// Implementations: Lessons.swift (the underscored helpers above).

// MARK: - Hub cards, closing parts, the crammer's study pile

/// A hub by row (0 = first hidden row) and index.
public struct HubRef: Hashable, Codable, Sendable {
    public var row: Int, index: Int
    public init(row: Int, index: Int) { self.row = row; self.index = index }
}

extension Concept {
    /// Concepts that are narrower cases of this one (2026-10-28): a perfect storm IS a storm (storm while
    /// short-staffed), and it is also short-staffing. A hub that tracks the broad concept naturally tracks its narrower
    /// cases too, so they never count against it in the label's 1.5× ratio.
    ///
    ///   storm       → perfectStorm
    ///   shortStaff  → perfectStorm
    ///   holidayRush → (none)
    ///   perfectStorm → (none)
    public var narrowerCases: [Concept] {
        switch self {
        case .storm, .shortStaff: [.perfectStorm]
        case .holidayRush, .perfectStorm: []
        }
    }
}

/// What a hub card says about a hub, from the CHECK year only (lesson 6's label rule).
public enum HubLabel: Equatable, Sendable {
    case mostly(Concept)              // signed match ≥ 0.7, ≥ 1.5× the best concept NOT nested under it
                                      // (Concept.narrowerCases), and closing it hurts those flights
    case linked(Concept)              // leans one way, but not all three tests
    case mix(Concept, Concept)        // two ideas about equally
    case quiet                        // fires on < 1% of flights
}

public struct HubCard: Sendable {
    public var hub: HubRef
    /// Share of check flights it lets passengers through on (activation > 0).
    public var firingShare: Float
    /// Signed Pearson match of its activation with each Concept (Concept.allCases order) and each raw fact
    /// (Fact.allCases order), on the check year.
    public var conceptMatch: [Float]
    public var factMatch: [Float]
    /// Up to 3 check flights it fires on most, and up to 3 it ignores (shut) that rank highest on its strongest
    /// raw fact: the contrast ("rainy, but calm: quiet").
    public var firesOn: [Int]
    public var ignores: [Int]
    public var topConcept: Concept
    /// The close-the-hub test on check flights where `topConcept` ≥ 0.5: minutes off with the hub open vs closed.
    public var conceptFlights: Int
    public var errorOpen: Float
    public var errorClosed: Float
    public var label: HubLabel
}

extension Network {
    /// A temporary overlay, no rebuild: the same network with these routes closed (weight 0) and these hubs closed
    /// (nothing leaves them: every route out of the hub is 0, so its output never reaches the next stop).
    public func closing(routes: Set<RouteID> = [], hubs: Set<HubRef> = []) -> Network { _closing(routes: routes, hubs: hubs) }
    /// The hub's card on the check year (one forward pass per check flight, plus the close test).
    public func hubCard(_ hub: HubRef, in ledger: Ledger) -> HubCard { _hubCard(hub, in: ledger) }
}

extension Ledger {
    /// `count` practice flights chosen by `seed` (a small study pile for exam day's crammer).
    public func studyPile(count: Int, seed: UInt64) -> [Int] { _studyPile(count: count, seed: seed) }
}
