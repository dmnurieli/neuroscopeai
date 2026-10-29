import SwiftUI
import AppKit
import AirlineKit

/// The minimal loop's model: one ledger, one lesson network, one trainer, and the page the learner is on.
/// Every number a page shows is read from the engine at that moment; nothing is typed in.
///
/// Pages: the flight, the brand-new network, its guess, the reveal, then the backpropagation walk (AirlineWalk:
/// hotter or colder by hand, the gate's blame, then each layer right to left, then all together), practice, an
/// unseen flight, the end. Page actions play as choreographed runs (AirlineChoreography). The button or Return
/// skips a playing run to its end state; the caption's numbers appear when the run reaches them.
@MainActor @Observable
final class AirlineState: AirlineLessonModel {
    let title = "Your first network"
    var pageCount: Int { Self.pageCount }
    var canGoBack: Bool { page > 0 && practice != .running }

    static let walk = WalkStage.stages(routeLayers: Architecture.lesson.hidden.count + 1)
    static let walkStart = 4
    static let practicePage = walkStart + walk.count
    static let unseenPage = practicePage + 1
    static let pageCount = unseenPage + 2
    static func page(of stage: WalkStage) -> Int { walkStart + (walk.firstIndex(of: stage) ?? 0) }

    /// Strides tried for the one-flight nudge, biggest first. Same backward walk as training; only the step
    /// size is chosen, so the guess visibly moves toward the truth without leaping past it.
    static let nudgeStrides: [Float] = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    /// The nudge may close at most this share of the gap (a little step, not a teleport).
    static let nudgeMaxClose: Float = 0.6
    static let practiceFlights = 50_000
    static let practiceChunks = 40
    static let hotRouteCount = 12

    enum Practice: Equatable { case idle, running, done, tumbled }

    // the engine
    private(set) var ledger: Ledger?
    private var trainer: Trainer?
    /// The network as last published. Never read from the trainer while it trains.
    private(set) var network: Network?
    /// Practice: the weights drawn just before `network`, so the map can ease from one chunk to the next.
    private(set) var previousWeights: [[Float]]?
    private(set) var snapshotDate = Date.distantPast
    /// Route draw order for `network` (thinnest first), built once per snapshot.
    private(set) var routeOrder: [[Int]] = []
    /// The network on the flight on screen, once it has guessed (cached per snapshot and page).
    private(set) var restActivations: Activations?

    // the two flights the lesson is about
    private(set) var showcase = 0
    private(set) var unseen = 0
    private var practiceIndices: [Int] = []

    // progress through the loop
    private(set) var page = 0
    private(set) var guess: Float?               // the brand-new network's guess on the showcase flight
    private(set) var guessNetwork: Network?      // the network that made it (Watch again replays it)
    private(set) var learn: LearnStep?           // the one nudge, taught by the walk
    private(set) var walkPlan: WalkPlan?
    private(set) var practice: Practice = .idle
    private(set) var checkStart: Float?          // quiz score (minutes off on the 2023 flights) before practice
    private(set) var checkNow: Float?
    private(set) var checkCount = 0
    private(set) var flightsPractised = 0

    // hotter or colder: the knob turns one route on a copy of the network; the real weight is never touched
    private(set) var knobWeight: Float?
    private(set) var knobGuess: Float?
    private(set) var knobMoved = false

    private var practiceTask: Task<Void, Never>?

    // the choreography: the run on screen, and whether the map's timeline must tick (AirlineShell)
    let chor = Choreographer()
    var run: AirlineRun? { chor.run }
    var captionReady: Bool { chor.captionReady }
    var timelineActive: Bool { chor.timelineActive }
    private var played: Set<Int> = []
    /// When the clipboard last gathered (page 1) or spread out (leaving it): the map eases between the layouts.
    private(set) var compactSince = Date.distantPast            // pages whose arrival sequence has played once

    // practice extras: the study pile flipping on the flight card, the quiz curve under the gate
    private(set) var practiceStart = Date()
    private(set) var lossHistory: [LossPoint] = []
    private(set) var hotRoutes: Set<RouteID> = []

    var isReady: Bool { network != nil }
    private var reduceMotion: Bool { chor.reduceMotion }

    init() {
        chor.hold = { [weak self] in self?.practice == .running }
        chor.summary = { [weak self] in self?.sequenceSummary }
    }
    var stage: WalkStage? {
        let n = page - Self.walkStart
        return Self.walk.indices.contains(n) ? Self.walk[n] : nil
    }
    var studyCount: Int { practiceIndices.count }

    // MARK: loading

    /// Generates the ledger and the starting network off the main actor (well under a second), once.
    func load() async {
        guard ledger == nil else { return }
        let built = await Task.detached(priority: .userInitiated) { () -> (Ledger, Trainer, Float, Int) in
            let ledger = Ledger.generate()
            let net = Network(architecture: .lesson, ledger: ledger, seed: Architecture.lessonSeed)
            let trainer = Trainer(network: net, ledger: ledger)
            return (ledger, trainer, net.meanMinutesOff(.check, in: ledger), ledger.indices(.check).count)
        }.value
        ledger = built.0
        trainer = built.1
        showcase = built.0.showcaseFlight
        unseen = built.0.unseenStormFlight
        practiceIndices = built.0.indices(.practice)
        checkStart = built.2
        checkNow = built.2
        checkCount = built.3
        publish(built.1.network)
        startFrame(.clipboard, move: 0)
    }

    func stop() {
        practiceTask?.cancel()
        practiceTask = nil
        if practice == .running { practice = .idle }
        keepAlive(0)
    }

    private func publish(_ net: Network) {
        let previous = network?.weights
        previousWeights = previous
        network = net
        snapshotDate = Date()
        routeOrder = MapModel.routeOrder(net.weights)
        hotRoutes = practice == .running ? Self.mostChanged(from: previous, to: net, count: Self.hotRouteCount) : []
        refreshRest()
        keepAlive(0.5)
    }

    /// The routes a snapshot moved most (practice's faint blame flicker).
    private static func mostChanged(from old: [[Float]]?, to net: Network, count: Int) -> Set<RouteID> {
        guard let old, old.count == net.weights.count else { return [] }
        let sizes = net.layerSizes
        var moved: [(Float, RouteID)] = []
        for l in net.weights.indices where old[l].count == net.weights[l].count {
            let outN = sizes[l + 1]
            for idx in net.weights[l].indices {
                moved.append((abs(net.weights[l][idx] - old[l][idx]), RouteID(layer: l, from: idx / outN, to: idx % outN)))
            }
        }
        return Set(moved.sorted { $0.0 > $1.0 }.prefix(count).filter { $0.0 > 0 }.map(\.1))
    }

    /// The network whose state a page shows: before the step through the walk (the hubs as they were when blamed),
    /// after it on the walk's last page, the live network from practice on.
    private var pageNetwork: Network? {
        if let w = walkPlan, page <= Self.page(of: .together) {
            return page == Self.page(of: .together) ? w.after : w.before
        }
        return network
    }

    private func refreshRest() {
        guard let net = pageNetwork, let l = ledger, guessVisible else { restActivations = nil; return }
        restActivations = net.forward(flight: displayFlight, in: l)
    }

    /// Route sizes to draw: through the walk each layer flips to its new size on its own page.
    private var displayWeights: [[Float]] {
        guard let net = network else { return [] }
        guard let w = walkPlan, page <= Self.page(of: .together) else { return net.weights }
        return (0..<w.routeLayers).map { l in page >= Self.page(of: .routes(l)) ? w.after.weights[l] : w.before.weights[l] }
    }

    // MARK: what is on screen

    /// The flight the card and the map are about: the showcase flight, then the unseen one on the last pages.
    var displayFlight: Int { page >= Self.unseenPage ? unseen : showcase }
    /// Whether the real arrival is shown (lifted from under the hand).
    var truthRevealed: Bool { page >= 3 }
    var networkVisible: Bool { page >= 1 }
    /// The network has guessed at least once, so the map shows hub activity and the guess.
    var guessVisible: Bool { guess != nil && page >= 2 }

    func flight(_ i: Int) -> Flight? { ledger?.flight(i) }

    /// While practising: a real study-pile flight from the ledger, sampled ~12 times a second.
    func logbookFlight(at date: Date) -> Flight? {
        guard let l = ledger, !practiceIndices.isEmpty else { return nil }
        var rng = SplitMix64(seed: UInt64(max(0, date.timeIntervalSince(practiceStart)) * 12) &+ 0x10_6B00)
        return l.flight(practiceIndices[Int(rng.next() % UInt64(practiceIndices.count))])
    }

    /// Everything the map draws this frame, snapshotted on the main actor.
    func mapScene(reduceMotion: Bool) -> MapScene? {
        guard let net = network, let l = ledger else { return nil }
        let f = l.flight(displayFlight)
        var minutes: MinutesScale?
        if let st = stage, let w = walkPlan {
            switch st {
            case .knob: minutes = MinutesScale(guess: knobGuess ?? w.guessBefore, truth: w.truth)
            case .gateBlame: minutes = MinutesScale(guess: w.guessBefore, truth: w.truth, arrow: true)
            case .together: minutes = MinutesScale(guess: w.guessBefore, truth: w.truth, newGuess: w.guessAfter)
            default: minutes = MinutesScale(guess: w.guessBefore, truth: w.truth)
            }
        } else if page == 3, let g = guess {
            minutes = MinutesScale(guess: g, truth: f.delay)
        } else if page == Self.unseenPage {
            let g = run?.kind == .unseen ? run?.forward?.guess : restActivations?.gateMinutes
            if let g { minutes = MinutesScale(guess: g, truth: f.delay) }
        }
        let onPractice = page == Self.practicePage
        return MapScene(
            network: net, weights: displayWeights, routeOrder: routeOrder,
            previousWeights: onPractice ? previousWeights : nil, snapshotDate: snapshotDate,
            flight: f, rest: restActivations, networkVisible: networkVisible, run: run, minutes: minutes,
            loss: onPractice ? lossHistory : [], lossTotal: Self.practiceFlights,
            hotRoutes: practice == .running ? hotRoutes : [],
            wasGuess: stage == .together ? walkPlan?.guessBefore : nil,
            walk: stage == nil ? nil : walkPlan, stage: stage,
            knobWeight: knobWeight, knobGuess: knobGuess,
            reduceMotion: reduceMotion, compactCards: page == 0, compactSince: compactSince)
    }

    // MARK: the pages

    typealias PageCopy = AirlinePageCopy

    /// The current run has reached its numbers (true when the page's run is of another kind).
    private func reached(_ kind: AirlineRun.Kind) -> Bool { chor.reached(kind) }

    var copy: PageCopy {
        guard let l = ledger, let net = network else {
            return PageCopy(text: "Loading the logbook…", button: "Continue", enabled: false, working: true)
        }
        if let st = stage, let w = walkPlan { return walkCopy(st, w) }
        switch page {
        case 0:
            return PageCopy(
                text: "This is one flight from the airline's **ledger**, its logbook of past flights.",
                detail: "On the left is the dispatcher's **clipboard**: the facts known before take-off. The real arrival is hidden under a hand.",
                button: "Continue")
        case 1:
            return PageCopy(
                text: "Here is a brand-new **network**. Its routes were sized at random, like a toddler spinning dials.",
                detail: "Each line is a **route** (a weight: how much one stop passes on). Solid amber routes add delay; dashed blue relief flights take it away.",
                button: "Continue")
        case 2:
            let flow = "Passengers flow through the hubs to the gate, more of them on the busier routes."
            if let g = guess {
                let said = "It guesses **\(Self.phrase(g))**."
                return PageCopy(text: "Let it guess how late this flight was.",
                                detail: reached(.guess) ? said : flow, button: "Continue", reserveDetail: [flow, said])
            }
            return PageCopy(text: "Let it guess how late this flight was.", detail: flow, button: "Guess")
        case 3:
            let truth = l.delay(showcase)
            let g = guess ?? net.predict(flight: showcase, in: l)
            let full = "Now lift the hand: it was really \(Self.phrase(truth)). The guess was off by \(Self.minutes(g - truth)) minutes."
            return PageCopy(
                text: reached(.reveal) ? full : "Now lift the hand.",
                detail: "That gap is the **loss**, how wrong the guess was. Next: how it learns from it.",
                button: "Continue", reserveText: [full])
        case Self.practicePage:
            let text = "One flight is not much. Let it practise on the ledger."
            let start = Self.minutes(checkStart ?? 0), now = Self.minutes(checkNow ?? 0)
            let piles = "It studies \(Self.practiceFlights.formatted()) flights from 2016–2022 (\(Self.shareWords(Double(Self.practiceFlights) / Double(max(studyCount, 1)))) the \(studyCount.formatted()) in the study pile), then quizzes itself on \(checkCount.formatted()) flights from 2023 it has never studied."
            switch practice {
            case .idle:
                return PageCopy(text: text, detail: piles, button: "Study \(Self.practiceFlights.formatted()) flights")
            case .running:
                return PageCopy(
                    text: text,
                    detail: "Studied \(flightsPractised.formatted()) flights. Quiz score: off by \(start) min → \(now) min on average.",
                    button: "Studying…", enabled: false, working: true, reserveDetail: [piles])
            case .done:
                return PageCopy(text: text,
                                detail: "Quiz score on the \(checkCount.formatted()) flights from 2023 it never studied: off by \(start) min → \(now) min on average.",
                                button: "Continue", reserveDetail: [piles])
            case .tumbled:
                return PageCopy(
                    text: text,
                    detail: "The hiker tumbled: a step was too big and the numbers blew up. Reset starts from a fresh network.",
                    button: "Reset")
            }
        case Self.unseenPage:
            let f = l.flight(unseen)
            let said = "It guesses **\(Self.phrase(net.predict(flight: unseen, in: l)))**. Really: **\(Self.phrase(f.delay))**."
            let flow = "Its facts flow through the same routes."
            return PageCopy(text: "Here is a flight it has never seen.",
                            detail: reached(.unseen) ? said : flow, button: "Continue", reserveDetail: [said, flow])
        default:
            return PageCopy(
                text: "That is all a network does: guess, check, nudge, repeat.",
                detail: "The big language models learn the same way, with far more routes.",
                button: "Done")
        }
    }

    /// "about half of", "a little over half of": how much of the study pile one practice run reads.
    nonisolated static func shareWords(_ r: Double) -> String {
        switch r {
        case ..<0.3: return "under a third of"
        case ..<0.45: return "under half of"
        case ..<0.55: return "about half of"
        case ..<0.75: return "a little over half of"
        case ..<0.95: return "most of"
        case ..<1.05: return "all of"
        default: return "more than one pass through"
        }
    }

    // MARK: the walk's words (every number from the step)

    private func walkCopy(_ st: WalkStage, _ w: WalkPlan) -> PageCopy {
        let L = w.routeLayers
        let gateBlame = w.blame.last?.first ?? 1
        let gap0 = Self.minutes(w.guessBefore - w.truth)
        let higher = w.truth > w.guessBefore
        let routeCount = w.before.weights.reduce(0) { $0 + $1.count }
        switch st {
        case .knob:
            let hub = w.name(layer: w.knob.layer, w.knob.from)
            return PageCopy(
                text: "**Hotter or colder?** Turn this one route up or down and watch the gap.",
                detail: "It is the route from \(hub) into the gate, the one that matters most for this flight. Testing every route this way would take forever: this tiny network has \(routeCount) routes; a real AI has billions.",
                button: "Continue", enabled: knobMoved, knob: true)
        case .gateBlame:
            return PageCopy(
                text: "**Backpropagation** (the backward walk) answers hotter-or-colder for every route at once, starting at the gate.",
                detail: "The gate's guess needed to be \(higher ? "higher" : "lower"): it was \(gap0) min too \(higher ? "low" : "high"). That need is the gate's **blame**: which way and how much it had to change. Blame is counted in blame points; the gate's share = \(WalkFormat.points(gateBlame)).",
                button: "Continue")
        case .routes(let l):
            let rule = routeRule(w, l, footnote: l == L - 1)
            if l == L - 1 {
                let closed = (0..<w.sizes[l]).filter { w.isClosed(layer: l, $0) }.map { w.name(layer: l, $0) }
                let callout: String
                switch closed.count {
                case 0: callout = ""
                case 1: callout = " \(closed[0])'s gate agent let nobody through, so its route carried no one: no change."
                default: callout = " \(Self.list(closed))'s gate agents let nobody through, so their routes carried no one: no change."
                }
                return PageCopy(
                    text: "Each route into the gate changes by how busy its hub was × the gate's blame.",
                    detail: "A busy hub sent lots of passengers down its route, so its route gets more of the change." + callout,
                    button: "Continue", rule: rule)
            }
            if l == 0 {
                let facts = w.before.architecture.inputs
                let byFar = facts.indices.sorted { abs(w.busy[0][$0]) > abs(w.busy[0][$1]) }
                let fl = ledger.map { $0.flight(showcase) }
                func say(_ i: Int) -> String { "\(facts[i].title) \(fl.map { facts[i].format($0.facts[facts[i].rawValue]) } ?? "")" }
                let unusual = Self.list(byFar.prefix(2).map(say))
                let ordinary = byFar.last.map(say) ?? ""
                var shoes = ""
                if let s = facts.firstIndex(of: .shoe) {
                    let out = w.sizes[1]
                    let top = (0..<out).map { w.change[0][s * out + $0] }.max { abs($0) < abs($1) } ?? 0
                    shoes = " Even the pilot's shoes routes moved (up to \(WalkFormat.points(top))): from one flight it cannot know shoes don't matter. Practice sorts that out."
                }
                let count = w.change[0].count, marked = min(Shots.chips(layer: 0), count)
                let more = marked < count ? " The \(marked) biggest changes are marked; the other \(count - marked) are smaller." : ""
                return PageCopy(
                    text: "Last, the routes from the clipboard, by the same rule.",
                    detail: "For a fact, how busy means how far it sits from an ordinary flight. This flight's unusual facts (\(unusual)) get their routes changed most; \(ordinary) is ordinary, so its routes barely move.\(shoes)\(more)",
                    button: "Continue", rule: rule)
            }
            let count = w.change[l].count, marked = min(Shots.chips(layer: l), count)
            return PageCopy(
                text: "The middle routes change by the same rule: how busy the start × the blame at the end.",
                detail: marked < count ? "The \(marked) biggest changes are marked; the other \(count - marked) are smaller." : nil,
                button: "Continue", rule: rule)
        case .hubs(let k):
            let row = w.blame[k - 1]
            let closed = (0..<w.sizes[k]).filter { w.isClosed(layer: k, $0) }.map { w.name(layer: k, $0) }
            if k == L - 1 {
                return PageCopy(
                    text: "Now the blame passes back to the row \(k) hubs.",
                    detail: "A hub's blame = the gate's blame passed back along its route: a jumbo-jet route passes back a lot, a relief route passes back the opposite, a closed hub takes none. ↑ means it should have been busier, ↓ quieter. Its **resting stress** (how busy it is before anyone arrives: the small dial) moves by its blame too.",
                    button: "Continue", rule: hubRule(w, k))
            }
            let top = row.indices.max { abs(row[$0]) < abs(row[$1]) } ?? 0
            let shut = closed.isEmpty ? "" : " \(Self.list(closed)) \(closed.count == 1 ? "was" : "were") closed and take none."
            return PageCopy(
                text: "Same again, one row back: the row \(k) hubs get their blame.",
                detail: "Each one adds up the blame passed back along all \(w.sizes[k + 1]) of its routes.\(shut) \(w.name(layer: k, top)) gets the most: \(WalkFormat.arrow(row[top])) \(WalkFormat.points(row[top])).",
                button: "Continue")
        case .together:
            let gap1 = Self.minutes(w.guessAfter - w.truth)
            let said = "Now it guesses **\(Self.phrase(w.guessAfter))**: off by \(gap1) instead of \(gap0). Every route moved by: how busy its start was × how much its end was to blame × a small step (\(String(format: "%.2f", Double(w.smallStep)))). Training repeats this thousands of times."
            let wait = "The same flight goes through again with every route at its new size."
            return PageCopy(
                text: "Together: every route and resting stress at its new size.",
                detail: reached(.together) ? said : wait, button: "Continue", reserveDetail: [said, wait])
        }
    }

    /// busy × blame = change, on the route of layer l that changes most.
    private func routeRule(_ w: WalkPlan, _ l: Int, footnote: Bool) -> RuleCard? {
        guard let idx = w.ranked[l].first else { return nil }
        let out = w.sizes[l + 1], i = idx / out, j = idx % out
        let busy = w.busy[l][i], blame = w.blame(layer: l + 1, j), change = w.change[l][idx]
        let top = max(w.busy[l].map(abs).max() ?? 1, 1e-6)
        let name = w.name(layer: l, i), dest = w.name(layer: l + 1, j)
        let before = w.before.weights[l][idx], after = w.after.weights[l][idx]
        var busyTile = RuleTile(label: "How busy \(name) was", value: WalkFormat.busy(busy, signed: l == 0), note: "passengers", bar: Double(busy / top))
        if l == 0, let fl = ledger?.flight(showcase) {
            let f = w.before.architecture.inputs[i]
            busyTile = RuleTile(label: "How unusual \(name) was", value: WalkFormat.busy(busy, signed: true),
                                note: f.format(fl.facts[f.rawValue]), bar: Double(busy / top))
        }
        let destLabel = l + 1 == w.routeLayers ? "The gate's blame" : "\(dest)'s blame"
        let foot = footnote
            ? "All in blame points. The real nudge is points × a small step (\(String(format: "%.2f", Double(w.smallStep)))): this route goes from \(Self.size(before)) to \(Self.size(after))."
            : "\(name) → \(dest) goes from \(Self.size(before)) to \(Self.size(after))."
        return RuleCard(tiles: [
            busyTile,
            RuleTile(label: destLabel, value: "\(WalkFormat.arrow(blame)) \(WalkFormat.points(blame))", blame: true),
            RuleTile(label: "Route change", value: "\(WalkFormat.chevron(change)) \(WalkFormat.points(change))"),
        ], footnote: foot)
    }

    /// route size × the gate's blame = the hub's blame, on the open hub with the most blame.
    private func hubRule(_ w: WalkPlan, _ k: Int) -> RuleCard? {
        let row = w.blame[k - 1]
        guard let j = row.indices.filter({ !w.isClosed(layer: k, $0) }).max(by: { abs(row[$0]) < abs(row[$1]) }) else { return nil }
        let gate = w.blame.last?.first ?? 1
        let size = w.before.weights[k][j * w.sizes[k + 1]]
        return RuleCard(tiles: [
            RuleTile(label: "\(w.name(layer: k, j))'s route to the gate", value: WalkFormat.busy(size, signed: true), note: size < 0 ? "a relief route" : "its size"),
            RuleTile(label: "The gate's blame", value: "\(WalkFormat.arrow(gate)) \(WalkFormat.points(gate))", blame: true),
            RuleTile(label: "\(w.name(layer: k, j))'s blame", value: "\(WalkFormat.arrow(row[j])) \(WalkFormat.points(row[j]))", blame: true),
        ], footnote: nil)
    }

    private static func size(_ w: Float) -> String { String(format: "%+.2f", Double(w)).replacingOccurrences(of: "-", with: "−") }
    private static func list(_ items: [String]) -> String {
        items.count <= 1 ? (items.first ?? "") : items.dropLast().joined(separator: ", ") + " and " + items.last!
    }

    /// The page's one button: skip a playing sequence to its end, else run the page's action if it has not run
    /// yet, otherwise move on. Returns true on the last page's Done.
    func primary() -> Bool {
        guard isReady else { return false }
        if chor.isPlaying {
            chor.skip()
            return false
        }
        switch page {
        case 2 where guess == nil:
            makeGuess(); return false
        case Self.page(of: .knob) where !knobMoved:
            return false
        case Self.practicePage where practice == .idle:
            startPractice(); return false
        case Self.practicePage where practice == .running:
            return false
        case Self.practicePage where practice == .tumbled:
            resetAfterTumble(); return false
        case Self.pageCount - 1:
            return true
        default:
            go(to: min(page + 1, Self.pageCount - 1))
            return false
        }
    }

    func back() {
        guard page > 0, practice != .running else { return }
        go(to: page - 1)
    }

    /// "Watch again" sits under the button on pages whose action animated, once its numbers are shown.
    var canWatchAgain: Bool {
        guard !reduceMotion, captionReady else { return false }
        if let st = stage { return st != .knob && walkPlan != nil }
        switch page {
        case 2: return guessNetwork != nil
        case Self.unseenPage: return true
        default: return false
        }
    }

    func watchAgain() {
        guard isReady else { return }
        if let st = stage { startWalkRun(st); return }
        switch page {
        case 2: startGuessRun()
        case Self.unseenPage: startUnseenRun()
        default: break
        }
    }

    /// The knob on the hotter-or-colder page: try a size for one route on a copy of the network.
    func setKnob(_ v: Float) {
        guard let w = walkPlan, let l = ledger else { return }
        knobWeight = v
        var net = w.before
        net.weights[w.knob.layer][w.knob.from * w.sizes[w.knob.layer + 1] + w.knob.to] = v
        knobGuess = net.predict(flight: showcase, in: l)
        if abs(v - w.knobStart) > 1e-3 { knobMoved = true }
    }

    // MARK: page arrivals

    private func go(to p: Int) {
        let from = page
        page = p
        if (from == 0) != (p == 0) { compactSince = Date() }
        if let st = stage {
            if walkPlan == nil { takeStep() }
            if st == .knob { knobWeight = nil; knobGuess = nil; knobMoved = false }
        }
        refreshRest()
        if let st = stage, st != .knob, !played.contains(p) {
            played.insert(p)
            startWalkRun(st)
            return
        }
        switch p {
        case 1 where !played.contains(1):
            played.insert(1); startDrawIn()
        case 3 where !played.contains(3) && guess != nil:
            played.insert(3); startReveal()
        case Self.unseenPage where !played.contains(p) && from == Self.practicePage:
            played.insert(p); startUnseenRun()
        case Self.pageCount - 1:
            startFrame(.wide, move: 1.2)
        default:
            startFrame(restFocus)
        }
    }

    /// Where the camera rests on each page once its sequence is over.
    private var restFocus: Focus {
        if let st = stage {
            switch st {
            case .knob: return .gateRoutes
            case .gateBlame, .together: return .gateScale
            case .routes(let l), .hubs(let l): return .between(l)
            }
        }
        switch page {
        case 0: return .clipboard
        case 3, Self.unseenPage: return .gateScale
        default: return .wide
        }
    }

    // MARK: runs

    private func startFrame(_ focus: Focus, move: Double = Shots.frameMove) { chor.frame(focus, move: move) }

    private func startRun(_ kind: AirlineRun.Kind, keys: [CameraKey], duration: Double, captionAt: Double,
                          configure: (inout AirlineRun) -> Void = { _ in }) {
        chor.start(kind, keys: keys, duration: duration, captionAt: captionAt, configure: configure)
    }

    private func keepAlive(_ seconds: Double) { chor.keepAlive(seconds) }

    private func startDrawIn() {
        startRun(.drawIn, keys: [CameraKey(at: 0, focus: .wide, move: 0.9)], duration: Shots.drawInDuration, captionAt: 0)
    }

    private func startGuessRun() {
        guard let net = guessNetwork, let l = ledger else { return }
        let plan = ForwardPlan(network: net, flight: showcase, ledger: l)
        let ft = ForwardTiming(start: 0, compress: 1, sizes: plan.sizes)
        var keys = ft.followKeys(fromClipboard: true)
        keys.append(CameraKey(at: ft.end + 0.45, focus: .wide, move: 0.9))
        startRun(.guess, keys: keys, duration: ft.end + 1.35, captionAt: ft.end) {
            $0.forward = plan
            $0.forwardTiming = ft
        }
    }

    private func startReveal() {
        startRun(.reveal, keys: [CameraKey(at: 0, focus: .gateScale, move: 0.8)],
                 duration: Shots.revealDuration, captionAt: Shots.revealBracket.1)
    }

    /// One walk page's sequence: the camera frames only the layer being explained.
    private func startWalkRun(_ st: WalkStage) {
        guard let w = walkPlan else { return }
        let duration: Double
        switch st {
        case .knob: duration = Shots.walkCamera
        case .gateBlame: duration = Shots.gateBlameDuration
        case .routes: duration = Shots.routesDuration
        case .hubs: duration = Shots.hubsDuration
        case .together: startTogetherRun(); return
        }
        startRun(.walk, keys: [CameraKey(at: 0, focus: restFocus, move: Shots.walkCamera)], duration: duration, captionAt: 0) {
            $0.walk = w
            $0.stage = st
        }
    }

    /// The walk's last page: the same flight again with every route at its new size; the gap shrinks.
    private func startTogetherRun() {
        guard let w = walkPlan, let l = ledger else { return }
        let ft = ForwardTiming(start: Shots.replayStart, compress: Shots.replayCompress, sizes: w.sizes)
        let keys = [CameraKey(at: 0, focus: .wide, move: 0.8), CameraKey(at: ft.end - 0.1, focus: .gateScale, move: 0.8)]
        let beforePlan = ForwardPlan(network: w.before, flight: showcase, ledger: l)
        let afterPlan = ForwardPlan(network: w.after, flight: showcase, ledger: l)
        startRun(.together, keys: keys, duration: ft.end + 0.8, captionAt: ft.end + 0.6) {
            $0.walk = w
            $0.stage = .together
            $0.before = beforePlan
            $0.forward = afterPlan
            $0.forwardTiming = ft
        }
    }

    private func startUnseenRun() {
        guard let net = network, let l = ledger else { return }
        let plan = ForwardPlan(network: net, flight: unseen, ledger: l)
        let ft = ForwardTiming(start: Shots.unseenForwardStart, compress: Shots.unseenCompress, sizes: plan.sizes)
        var keys = [CameraKey(at: 0, focus: .clipboard, move: 0.7)]
        keys += ft.followKeys(fromClipboard: false)
        keys.append(CameraKey(at: ft.end - 0.05, focus: .gateScale, move: 0.7))
        let shown = ft.countWindow.1
        let from = l.flight(showcase).facts
        startRun(.unseen, keys: keys, duration: shown + 1.5, captionAt: shown + 1.4) {
            $0.forward = plan
            $0.forwardTiming = ft
            $0.flipFrom = from
        }
    }

    /// One line for VoiceOver: what the sequence on screen showed (nil for plain camera moves).
    var sequenceSummary: String? {
        guard let r = run, captionReady else { return nil }
        switch r.kind {
        case .frame:
            if stage == .knob, let w = walkPlan, let kg = knobGuess {
                return "With that route at \(Self.size(knobWeight ?? w.knobStart)) the gate would guess \(Self.phrase(kg)), off by \(Self.minutes(kg - w.truth)) minutes."
            }
            return nil
        case .drawIn:
            guard let net = network else { return nil }
            let all = net.weights.flatMap { $0 }, up = all.filter { $0 > 0 }.count
            return "The \(all.count) routes were drawn in at random sizes: \(up) amber, \(all.count - up) dashed blue."
        case .guess:
            guard let p = r.forward else { return nil }
            let pre = p.pre.flatMap { $0 }, open = pre.filter { $0 > 0 }.count
            return "Passengers flowed from the clipboard through \(open) open hubs to the gate; gate agents held them at \(pre.count - open) hubs. It guessed \(Self.phrase(p.guess))."
        case .reveal:
            guard let g = guess, let l = ledger else { return nil }
            return "The hand lifted: it was really \(Self.phrase(l.delay(showcase))). The guess was off by \(Self.minutes(g - l.delay(showcase))) minutes."
        case .walk:
            guard let w = r.walk, let st = r.stage else { return nil }
            switch st {
            case .gateBlame:
                let b = w.blame.last?.first ?? 1
                return "The gate's blame is \(WalkFormat.points(b)): its guess needed to be \(w.truth > w.guessBefore ? "higher" : "lower") by \(Self.minutes(w.guessBefore - w.truth)) minutes."
            case .routes(let l):
                let c = w.change[l], up = c.filter { $0 > 0 }.count, down = c.filter { $0 < 0 }.count
                let top = w.ranked[l].first ?? 0, out = w.sizes[l + 1]
                return "\(c.count) routes changed by how busy their start was times the blame at their end: \(up) turned up, \(down) turned down, \(c.count - up - down) unchanged. The biggest, \(w.name(layer: l, top / out)) to \(w.name(layer: l + 1, top % out)), by \(WalkFormat.points(c[top])) blame points."
            case .hubs(let k):
                let row = w.blame[k - 1]
                let parts = row.indices.map { "\(w.name(layer: k, $0)) \(w.isClosed(layer: k, $0) ? "closed, none" : WalkFormat.points(row[$0]))" }
                return "Blame passed back to the row \(k) hubs: " + parts.joined(separator: ", ") + ". Their resting stress moved with it."
            default: return nil
            }
        case .together:
            guard let w = r.walk else { return nil }
            return "With every route at its new size the guess moved from \(Self.phrase(w.guessBefore)) to \(Self.phrase(w.guessAfter)): off by \(Self.minutes(w.guessAfter - w.truth)) instead of \(Self.minutes(w.guessBefore - w.truth)) minutes."
        case .unseen:
            guard let p = r.forward, let l = ledger else { return nil }
            return "A new flight's facts flowed through the hubs to the gate: it guesses \(Self.phrase(p.guess)); really \(Self.phrase(l.delay(unseen)))."
        }
    }

    // MARK: actions

    private func makeGuess() {
        guard let net = network, let l = ledger else { return }
        guessNetwork = net
        guess = net.predict(flight: showcase, in: l)
        refreshRest()
        startGuessRun()
    }

    /// The one nudge the walk teaches: one real step on the showcase flight, taken when the walk begins.
    private func takeStep() {
        guard let t = trainer, let l = ledger else { return }
        let before = t.network
        let stride = nudgeStride(for: before)
        let step = t.learn(fromFlight: showcase, stepSize: stride)
        learn = step
        walkPlan = WalkPlan(step: step, before: before, after: t.network, stride: stride, ledger: l)
        publish(t.network)
    }

    /// The biggest stride whose single step moves the guess toward the truth, closing no more than
    /// `nudgeMaxClose` of the gap. Tried on throwaway copies; the real trainer takes exactly one step.
    private func nudgeStride(for net: Network) -> Float {
        guard let l = ledger else { return Self.nudgeStrides.last! }
        let truth = l.delay(showcase)
        for stride in Self.nudgeStrides {
            let trial = Trainer(network: net, ledger: l).learn(fromFlight: showcase, stepSize: stride)
            let gap = truth - trial.guessBefore
            let moved = trial.guessAfter - trial.guessBefore
            if gap != 0, moved / gap > 0, moved / gap <= Self.nudgeMaxClose { return stride }
        }
        return Self.nudgeStrides.last!
    }

    /// Practice in chunks on a background task: after each chunk the map reshapes, the quiz score updates and the
    /// curve under the gate gains a point. The trainer is only ever touched by one task at a time (the buttons
    /// are gated while this runs).
    private func startPractice() {
        guard let t = trainer, let l = ledger else { return }
        practice = .running
        practiceStart = Date()
        lossHistory = [LossPoint(flights: 0, minutes: checkNow ?? 0, at: Date())]
        keepAlive(0.5)
        let batches = max(1, Self.practiceFlights / max(1, t.config.batchSize))
        let chunks = Self.practiceChunks
        practiceTask = Task { [weak self] in
            for c in 0..<chunks {
                if Task.isCancelled { return }
                let n = batches * (c + 1) / chunks - batches * c / chunks
                let (status, net, off, seen) = await Task.detached(priority: .userInitiated) { () -> (TrainStatus, Network, Float, Int) in
                    let s = t.train(batches: n)
                    let net = t.network
                    return (s, net, net.meanMinutesOff(.check, in: l), t.flightsSeen)
                }.value
                guard let self, !Task.isCancelled else { return }
                self.publish(net)
                self.flightsPractised = seen
                if status == .tumbled { self.practice = .tumbled; self.hotRoutes = []; self.keepAlive(0.5); return }
                self.checkNow = off
                self.lossHistory.append(LossPoint(flights: seen, minutes: off, at: Date()))
                try? await Task.sleep(for: .milliseconds(60))   // let each reshaping be seen
            }
            guard let self else { return }
            self.practice = .done
            self.hotRoutes = []
            self.keepAlive(0.6)
        }
    }

    private func resetAfterTumble() {
        guard let t = trainer, let l = ledger else { return }
        t.reset(to: Network(architecture: .lesson, ledger: l, seed: Architecture.lessonSeed))
        checkNow = checkStart
        flightsPractised = 0
        lossHistory = []
        practice = .idle
        publish(t.network)
    }

    // MARK: words for numbers

    /// "112 min late", "3 min early", "on time".
    nonisolated static func phrase(_ minutes: Float) -> String {
        let r = Int(minutes.rounded())
        if r > 0 { return "\(r) min late" }
        if r < 0 { return "\(-r) min early" }
        return "on time"
    }

    nonisolated static func minutes(_ gap: Float) -> Int { Int(abs(gap).rounded()) }
}
