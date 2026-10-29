import SwiftUI
import AirlineKit

/// Lessons 2–4 of the airline course (the clipboard, direct routes, layover hubs) and the Builder: the map for any
/// architecture, edits that rebuild from the lesson seed, and training on a background task. One sentence, one
/// button per page; every number is read from the engine.
@MainActor @Observable
final class AirlineNetLesson: AirlineLessonModel {
    let kind: AirlineLessonID
    let chor = Choreographer()

    private(set) var ledger: Ledger?
    private(set) var host: TrainerHost?
    private(set) var page = 0
    /// Lesson 4: the quiz score with one row of hubs, kept for the bars once the second row is added.
    private(set) var oneRowScore: Float?
    private(set) var rulebookScore: Float?
    /// Lesson 4: the row-1 hub most linked to the storm on the 2023 flights (index, signed match).
    private(set) var stormHub: (index: Int, r: Float)?
    private var findingStormHub = false
    private var played: Set<Int> = []

    init(kind: AirlineLessonID) {
        self.kind = kind
        chor.hold = { [weak self] in self?.host?.isRunning == true }
        chor.summary = { [weak self] in self?.sequenceSummary }
    }

    var title: String { kind.title }
    var pageCount: Int {
        switch kind {
        case .clipboard, .direct: 4
        case .hubs: 5
        default: 1
        }
    }
    var canGoBack: Bool { page > 0 && host?.isRunning != true }
    var isReady: Bool { host != nil }

    func load() async {
        guard ledger == nil else { return }
        let l = await AirlineLedger.load()
        ledger = l
        let inputs = AirlineProgress.clipboard
        let hidden: [Int] = kind == .builder ? [6] : []
        let h = TrainerHost(ledger: l, architecture: Architecture(inputs: inputs, hidden: hidden))
        h.onSnapshot = { [weak self] in self?.chor.keepAlive(0.5) }
        host = h
        rulebookScore = AirlineProgress.rulebook.meanMinutesOff(.check, in: l)
        arrive()
    }

    func stop() {
        host?.stop()
        chor.keepAlive(0)
    }

    // MARK: what is on screen

    /// The flight the card and map are about on this page.
    var displayFlight: Int? {
        guard let l = ledger else { return nil }
        if kind == .direct && page == 2 { return l.compoundFlight ?? l.showcaseFlight }
        if kind == .hubs && page == 3 { return l.unseenStormFlight }
        return l.showcaseFlight
    }
    var truthRevealed: Bool { kind != .clipboard }
    var networkVisible: Bool { kind != .clipboard }
    var showsTrainPanel: Bool {
        switch kind {
        case .direct: page >= 1
        case .hubs: page >= 1 && page <= 3
        case .builder: true
        default: false
        }
    }
    var editAllowed: EditStrip.Allowed? {
        switch kind {
        case .clipboard: page == 1 || page == 2 ? [.cards] : nil
        case .hubs: page == 0 ? [.hubsRow1] : nil
        case .builder: .all
        default: nil
        }
    }

    func edit(_ a: Architecture) {
        guard let h = host, a != h.architecture else { return }
        h.rebuild(a)
        if kind == .clipboard { AirlineProgress.clipboard = a.inputs }
        chor.keepAlive(0.3)
    }

    func toggle(_ f: Fact) {
        guard let h = host else { return }
        edit(ArchitectureEdit.toggling(f, in: h.architecture))
    }

    func mapScene(reduceMotion: Bool) -> MapScene? {
        guard let l = ledger, let h = host, let fi = displayFlight else { return nil }
        let net = h.network
        let f = l.flight(fi)
        var minutes: MinutesScale?
        if kind == .direct && page == 2 {
            let g = chor.run?.kind == .guess ? chor.run?.forward?.guess : net.predict(flight: fi, in: l)
            if let g { minutes = MinutesScale(guess: g, truth: f.delay) }
        }
        var z: [Float]?
        if kind == .clipboard && page == 3 {
            z = Fact.allCases.map { (l.fact($0, fi) - l.factMean[$0.rawValue]) / max(l.factSD[$0.rawValue], 1e-6) }
        }
        var point: (row: Int, index: Int)?
        if kind == .hubs && page == 3, let s = stormHub { point = (0, s.index) }
        return MapScene(
            network: net, weights: net.weights, routeOrder: h.routeOrder,
            previousWeights: h.isRunning ? h.previousWeights : nil, snapshotDate: h.snapshotDate,
            flight: f, rest: kind == .clipboard ? nil : net.forward(flight: fi, in: l),
            networkVisible: networkVisible, run: chor.run, minutes: minutes,
            hotRoutes: h.isRunning ? h.hotRoutes : [], reduceMotion: reduceMotion,
            allFacts: true, wobble: kind == .clipboard && page == 2 ? .shoe : nil, zScores: z, pointAt: point,
            compactCards: kind == .clipboard)
    }

    // MARK: the pages

    private var trainButton: (String, Bool) {
        guard let h = host else { return ("Train", false) }
        switch h.state {
        case .idle: return ("Train", true)
        case .running: return ("Pause", true)
        case .paused: return ("Resume", true)
        case .done: return ("Continue", true)
        case .tumbled: return ("Reset", true)
        }
    }

    private func trainingDetail(_ h: TrainerHost) -> String? {
        guard h.hasTrained || h.isRunning else { return nil }
        let passes = h.passesWords.map { " (\($0))" } ?? ""
        return "Studied \(h.flightsStudied.formatted()) flights\(passes). Quiz score: off by \(AirlineState.minutes(h.quizStart)) → \(AirlineState.minutes(h.quizNow)) min on average."
    }

    var copy: AirlinePageCopy {
        guard let l = ledger, let h = host, let fi = displayFlight else {
            return AirlinePageCopy(text: "Loading the logbook…", button: "Continue", enabled: false, working: true)
        }
        let a = h.architecture
        let routes = h.network.weights.reduce(0) { $0 + $1.count }
        switch (kind, page) {
        // the clipboard
        case (.clipboard, 0):
            let f = l.flight(fi)
            return AirlinePageCopy(
                text: "Everything the network knows about a flight is on the dispatcher's **clipboard**: the **inputs** (the facts it gets to see).",
                detail: "Here is one flight from the logbook: \(f.origin) → \(f.destination), \(FlightCardView.date(year: f.year, day: f.dayOfYear)).",
                button: "Continue")
        case (.clipboard, 1):
            let n = a.inputs.count
            return AirlinePageCopy(
                text: "Which facts would help predict a delay?",
                detail: "Tap a card to switch it off or on. \(n) of \(Fact.allCases.count) are on." + (n == 0 ? " With no cards it can only guess the average delay." : ""),
                button: "Continue", reserveDetail: ["Tap a card to switch it off or on. 8 of 8 are on. With no cards it can only guess the average delay."])
        case (.clipboard, 2):
            let on = a.inputs.contains(.shoe)
            return AirlinePageCopy(
                text: "Would the pilot's shoes matter? Keep it on; let's see what the network decides.",
                detail: on ? "Shoe colour is on the clipboard, like any other fact." : "It is off right now: tap the card to switch it back on and let the network decide.",
                button: "Continue")
        case (.clipboard, _):
            let ex = exampleFact(l, fi, a)
            var detail = "Then a busy morning and a big storm can be compared on one scale."
            if let (f, z) = ex {
                detail = "On this flight, \(f.title) is \(f.format(l.fact(f, fi))): \(MapPainter.zWords(z)) (\(WalkFormat.busy(z, signed: true)) steps from an ordinary flight). Your \(a.inputs.count) cards carry into the next lessons."
            }
            return AirlinePageCopy(
                text: "Each fact is put on the same scale first: how far it is from an ordinary flight.",
                detail: detail, button: "Done")

        // direct routes
        case (.direct, 0):
            return AirlinePageCopy(
                text: "Simplest network: a route straight from every fact to the gate. No layovers.",
                detail: "\(a.inputs.count) facts, \(routes) routes. Each route adds or takes away minutes on its own.",
                button: "Continue")
        case (.direct, 1):
            let (b, ok) = trainButton
            var detail = trainingDetail(h) ?? "It studies the \(l.indices(.practice).count.formatted()) flights from 2016–2022, six times over, and quizzes itself on \(l.indices(.check).count.formatted()) flights from 2023."
            if h.state == .done { detail += " It has flattened out: that is about as good as straight routes get." }
            return AirlinePageCopy(text: "Train it on the logbook.", detail: detail, button: b, enabled: ok)
        case (.direct, 2):
            let f = l.flight(fi)
            let said = "On this stormy, short-staffed flight (\(Self.facts(f, [.rain, .wind, .controllers, .holiday]))) it guesses **\(AirlineState.phrase(h.network.predict(flight: fi, in: l)))**; it was really **\(AirlineState.phrase(f.delay))**."
            let wait = "Watch each fact's route add its share."
            return AirlinePageCopy(
                text: "Direct routes add rain and wind separately. They can't see that rain AND wind together make a storm.",
                detail: chor.reached(.guess) ? said : wait, button: "Continue", reserveDetail: [said, wait])
        case (.direct, _):
            return AirlinePageCopy(
                text: "It needs places where facts meet.",
                detail: "Next: layover hubs, where several facts arrive at once.", button: "Done")

        // layover hubs
        case (.hubs, 0):
            let n = a.hidden.first ?? 0
            return AirlinePageCopy(
                text: "Add a row of **hubs**: layover cities where facts meet.",
                detail: n == 0 ? "Each hub gets a route from every fact and sends one route on to the gate."
                    : "\(n) hub\(n == 1 ? "" : "s"): \(routes) routes. Each hub gets a route from every fact and sends one on to the gate.",
                button: n < 6 ? "Add a hub" : "Continue")
        case (.hubs, 1):
            let (b, ok) = trainButton
            var detail = trainingDetail(h) ?? "Same logbook, same quiz as the direct routes."
            if h.state == .done, let d = AirlineProgress.directScore {
                detail += " Direct routes got to \(AirlineState.minutes(d)) min."
            }
            return AirlinePageCopy(text: "Train it.", detail: detail, button: b, enabled: ok)
        case (.hubs, 2):
            if a.hidden.count < 2 {
                return AirlinePageCopy(text: "Add a second row.", detail: "The second row's hubs meet the first row's hubs: combinations of combinations.",
                                       button: "Add a second row")
            }
            let (b, ok) = trainButton
            return AirlinePageCopy(text: "Add a second row.", detail: trainingDetail(h) ?? "Two rows of \(a.hidden[1]) and \(a.hidden[0]) hubs: \(routes) routes. Train it.",
                                   button: b, enabled: ok)
        case (.hubs, 3):
            guard let s = stormHub else {
                return AirlinePageCopy(text: "Which hub lights up on stormy days?", detail: "Checking every 2023 flight…", button: "Continue", enabled: false, working: true)
            }
            let name = MapModel.hubName(row: 0, index: s.index)
            let r = String(format: "%.2f", Double(s.r))
            if s.r >= 0.3 {
                return AirlinePageCopy(
                    text: "\(name) lights up on stormy days. Nobody told it to.",
                    detail: "On the 2023 flights, \(name)'s busyness rises and falls with the storm: a match of \(r) (1 would be perfect). Lesson 6 tests what each hub really means.",
                    button: "Continue")
            }
            return AirlinePageCopy(
                text: "\(name) leans toward stormy days. Nobody told it to.",
                detail: "On the 2023 flights its match with the storm is \(r) (1 would be perfect): a weak link so far. Lesson 6 tests what each hub really means.",
                button: "Continue")
        case (.hubs, _):
            return AirlinePageCopy(
                text: "That's **deep learning**: deep just means more rows of hubs.",
                detail: "Yours: \(a.inputs.count) facts, \(a.hidden.count) row\(a.hidden.count == 1 ? "" : "s") of hubs, \(routes) routes. The Builder is now open on Home.",
                button: "Done")

        // the Builder
        default:
            return AirlinePageCopy(
                text: "Build any network you like, then train it.",
                detail: "\(a.inputs.count) facts · \(a.hidden.isEmpty ? "no hubs" : a.hidden.map { "\($0)" }.joined(separator: " + ") + " hubs") · \(routes) routes. Quiz score: off by \(AirlineState.minutes(h.quizNow)) min.",
                button: "Done")
        }
    }

    /// The bars lesson 4 shows once there is something to compare.
    var bars: [ScoreBars.Bar]? {
        guard kind == .hubs, let h = host, page == 1 || page == 2, h.hasTrained || oneRowScore != nil else { return nil }
        var out = [ScoreBars.Bar(label: "Your rules", minutes: rulebookScore),
                   ScoreBars.Bar(label: "Direct routes", minutes: AirlineProgress.directScore)]
        if page == 1 {
            // back here after adding the second row: the one-row score is the remembered one
            out.append(ScoreBars.Bar(label: "One row of hubs", minutes: h.architecture.hidden.count >= 2 ? oneRowScore : h.quizNow, highlight: true))
        } else {
            out.append(ScoreBars.Bar(label: "One row of hubs", minutes: oneRowScore))
            out.append(ScoreBars.Bar(label: "Two rows", minutes: h.architecture.hidden.count >= 2 && h.hasTrained ? h.quizNow : nil, highlight: true))
        }
        return out
    }

    func primary() -> Bool {
        guard let h = host else { return false }
        if chor.isPlaying { chor.skip(); return false }
        switch (kind, page) {
        case (.direct, 1), (.hubs, 1):
            if trainStep(h) { return false }
        case (.hubs, 0):
            if (h.architecture.hidden.first ?? 0) < 6 { edit(ArchitectureEdit.hubs(1, row: 0, in: h.architecture)); return false }
        case (.hubs, 2):
            if h.architecture.hidden.count < 2 {
                oneRowScore = h.hasTrained ? h.quizNow : oneRowScore
                edit(ArchitectureEdit.addRow(h.architecture, hubs: h.architecture.hidden.first ?? 6))
                return false
            }
            if trainStep(h) { return false }
        case (.builder, _):
            return true
        default: break
        }
        if page == pageCount - 1 { return true }
        go(to: page + 1)
        return false
    }

    /// Train / Pause / Resume / Reset; false once training is done (the button moves on).
    private func trainStep(_ h: TrainerHost) -> Bool {
        switch h.state {
        case .idle, .paused: h.train(); return true
        case .running: h.pause(); return true
        case .tumbled: h.reset(); return true
        case .done:
            if kind == .direct { AirlineProgress.directScore = h.quizNow }
            if kind == .hubs && page == 1 { oneRowScore = h.quizNow }
            if kind == .hubs && page == 2 {
                AirlineProgress.networkScore = h.quizNow
                AirlineProgress.saveTrained(h.network)   // lessons 6–8 study this network
            }
            return false
        }
    }

    func back() {
        guard canGoBack else { return }
        go(to: page - 1)
    }

    var canWatchAgain: Bool { !chor.reduceMotion && chor.captionReady && kind == .direct && page == 2 }
    func watchAgain() { if kind == .direct && page == 2 { startGuess() } }

    private func go(to p: Int) {
        page = p
        arrive()
    }

    /// What each page does on arrival: a sequence the first time, otherwise a camera move to its framing.
    private func arrive() {
        let first = !played.contains(page)
        played.insert(page)
        switch (kind, page) {
        case (.clipboard, 2): chor.frame(.clipboard, move: 1.6)    // the shoe card wobbles for this long
        case (.clipboard, 3): chor.frame(.clipboardDials)
        case (.clipboard, _): chor.frame(.clipboard, move: first && page == 0 ? 0 : Shots.frameMove)
        case (.direct, 0) where first:
            chor.start(.drawIn, keys: [CameraKey(at: 0, focus: .wide, move: 0.9)], duration: Shots.drawInDuration, captionAt: 0)
        case (.direct, 2) where first: startGuess()
        case (.direct, 2): chor.frame(.gateScale)
        case (.hubs, 3):
            chor.frame(.between(0))
            findStormHub()
        default: chor.frame(.wide)
        }
    }

    /// The direct network's guess on the compound flight, traced: each fact's route adds its share.
    private func startGuess() {
        guard let h = host, let l = ledger, let fi = displayFlight else { return }
        let plan = ForwardPlan(network: h.network, flight: fi, ledger: l)
        let ft = ForwardTiming(start: 0, compress: 1, sizes: plan.sizes)
        var keys = ft.followKeys(fromClipboard: true)
        keys.append(CameraKey(at: ft.end, focus: .gateScale, move: 0.8))
        chor.start(.guess, keys: keys, duration: ft.end + 1.4, captionAt: ft.end + 0.6) {
            $0.forward = plan
            $0.forwardTiming = ft
        }
    }

    /// The row-1 hub whose busyness tracks the storm most closely (signed) on the 2023 flights.
    private func findStormHub() {
        guard stormHub == nil, !findingStormHub, let h = host, let l = ledger, (h.architecture.hidden.first ?? 0) > 0 else { return }
        findingStormHub = true
        let net = h.network
        Task { [weak self] in
            let found = await Task.detached(priority: .userInitiated) { Self.stormHub(net, l) }.value
            guard let self else { return }
            self.stormHub = found
            self.findingStormHub = false
            self.chor.keepAlive(0.2)
        }
    }

    nonisolated static func stormHub(_ net: Network, _ l: Ledger) -> (index: Int, r: Float)? {
        guard let n = net.architecture.hidden.first, n > 0 else { return nil }
        let idx = l.indices(.check)
        var acts = [[Float]](repeating: [], count: n), storm: [Float] = []
        for i in idx {
            let a = net.forward(flight: i, in: l)
            for k in 0..<n { acts[k].append(a.hubs[0][k]) }
            storm.append(l.concept(.storm, i))
        }
        var best: (Int, Float)?
        for k in 0..<n {
            let r = pearson(acts[k], storm)
            if best == nil || r > best!.1 { best = (k, r) }
        }
        return best
    }

    nonisolated private static func pearson(_ x: [Float], _ y: [Float]) -> Float {
        let n = Double(x.count)
        guard n > 1 else { return 0 }
        let mx = x.reduce(0.0) { $0 + Double($1) } / n, my = y.reduce(0.0) { $0 + Double($1) } / n
        var sxy = 0.0, sxx = 0.0, syy = 0.0
        for k in x.indices {
            let (a, b) = (Double(x[k]) - mx, Double(y[k]) - my)
            sxy += a * b; sxx += a * a; syy += b * b
        }
        return sxx > 0 && syy > 0 ? Float(sxy / (sxx * syy).squareRoot()) : 0
    }

    /// The on-card fact this flight is most unusual on (wind first if it is on and unusual).
    private func exampleFact(_ l: Ledger, _ fi: Int, _ a: Architecture) -> (Fact, Float)? {
        func z(_ f: Fact) -> Float { (l.fact(f, fi) - l.factMean[f.rawValue]) / max(l.factSD[f.rawValue], 1e-6) }
        if a.inputs.contains(.wind), abs(z(.wind)) >= 1.5 { return (.wind, z(.wind)) }
        return a.inputs.map { ($0, z($0)) }.max { abs($0.1) < abs($1.1) }
    }

    /// "Rain 2.2 in, Wind 39 mph, …" read from the flight.
    static func facts(_ f: Flight, _ which: [Fact]) -> String {
        which.map { "\($0.title.lowercased()) \($0.format(f.facts[$0.rawValue]))" }.joined(separator: ", ")
    }

    var sequenceSummary: String? {
        guard let r = chor.run, chor.captionReady, let l = ledger, let fi = displayFlight else { return nil }
        switch r.kind {
        case .guess:
            guard let p = r.forward else { return nil }
            return "Each fact's route added its share at the gate: it guessed \(AirlineState.phrase(p.guess)); really \(AirlineState.phrase(l.delay(fi)))."
        case .drawIn:
            guard let h = host else { return nil }
            return "The network's \(h.network.weights.reduce(0) { $0 + $1.count }) routes were drawn in at random sizes."
        default:
            return nil
        }
    }
}

// MARK: - the view: flight card, map, edit strip, train panel

struct AirlineNetLessonView: View {
    var onExit: (_ completed: Bool) -> Void
    @State private var model: AirlineNetLesson

    init(kind: AirlineLessonID, onExit: @escaping (_ completed: Bool) -> Void) {
        self.onExit = onExit
        _model = State(initialValue: AirlineNetLesson(kind: kind))
    }

    var body: some View {
        AirlineLessonShell(model: model, onExit: onExit) {
            if let h = model.host, let l = model.ledger, let fi = model.displayFlight {
                FlightCardView(flight: l.flight(fi), revealed: model.truthRevealed)
                    .id(fi)
                    .transition(.opacity)
                    .padding(.top, 6)
                // the strip shares the map's width, so each control sits under the column it edits
                VStack(spacing: 4) {
                    AirlineMapView(sceneFor: { model.mapScene(reduceMotion: $0) }, timelineActive: model.chor.timelineActive,
                                   summary: model.sequenceSummary,
                                   onToggleFact: model.editAllowed?.contains(.cards) == true ? { model.toggle($0) } : nil)
                        .frame(maxHeight: .infinity)
                    if let allowed = model.editAllowed, allowed != [.cards] {
                        EditStrip(architecture: h.architecture, allowed: allowed) { model.edit($0) }
                    }
                }
                .padding(.horizontal, 28)
                .padding(.vertical, 14)
                .frame(maxWidth: 1100, maxHeight: .infinity)
                if model.showsTrainPanel {
                    TrainPanel(host: h, showsControls: model.kind == .builder, showsStepSize: model.kind == .builder)
                        .padding(.bottom, 12)
                }
            } else {
                Spacer(minLength: 0)
            }
        } accessory: { _ in
            if let bars = model.bars { ScoreBars(bars: bars) }
        }
        .task { await model.load() }
    }
}
