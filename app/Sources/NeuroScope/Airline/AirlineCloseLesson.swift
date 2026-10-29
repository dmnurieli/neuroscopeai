import SwiftUI
import AirlineKit

/// Lesson 8, "Close a route": close any route or hub of the learner's trained network as an overlay (no rebuild;
/// `Network.closing`), and the quiz score and the flights that got worse update live. Guided first: the pilot's
/// shoe routes (barely matter: 5.78 → 5.76 min on the lesson network) versus the hub most linked to storms (its
/// storm flights go from 9.3 to 65 min off). Measured in HubCardTests.testShoesBarelyMatterStormHubDoes.
@MainActor @Observable
final class AirlineCloseLesson: AirlineLessonModel {
    let title = AirlineLessonID.close.title
    let pageCount = 5
    let chor = Choreographer()

    struct Worse: Identifiable { let flight: Int; let before: Float; let after: Float; var id: Int { flight } }

    private(set) var ledger: Ledger?
    private(set) var network: Network?
    private(set) var routeOrder: [[Int]] = []
    private(set) var page = 0
    private(set) var closedRoutes: Set<RouteID> = []
    private(set) var closedHubs: Set<HubRef> = []
    private(set) var quizOpen: Float?
    private(set) var quizNow: Float?
    private(set) var worse: [Worse] = []
    private(set) var stormHub: HubRef?
    private(set) var stormOpen: Float?
    private(set) var stormNow: Float?
    private var openErrors: [Float] = []
    private var stormFlights: [Int] = []
    private var task: Task<Void, Never>?

    var canGoBack: Bool { page > 0 }
    var canWatchAgain: Bool { false }
    var anyClosed: Bool { !closedRoutes.isEmpty || !closedHubs.isEmpty }

    init() { chor.summary = { [weak self] in self?.summary } }

    func load() async {
        guard ledger == nil else { return }
        let l = await AirlineLedger.load()
        ledger = l
        let net = await AirlineProgress.trainedNetwork(l)
        network = net
        routeOrder = MapModel.routeOrder(net.weights)
        chor.frame(.wide, move: 0)
        let found = await Task.detached(priority: .userInitiated) { () -> ([Float], Float, (index: Int, r: Float)?, [Int], Float) in
            let idx = l.indices(.check)
            let errs = idx.map { abs(net.predict(flight: $0, in: l) - l.delay($0)) }
            let storm = idx.filter { l.concept(.storm, $0) >= 0.5 }
            return (errs, errs.reduce(0, +) / Float(max(errs.count, 1)), AirlineNetLesson.stormHub(net, l),
                    storm, net.meanMinutesOff(indices: storm, in: l))
        }.value
        openErrors = found.0
        quizOpen = found.1
        quizNow = found.1
        stormHub = found.2.map { HubRef(row: 0, index: $0.index) }
        stormFlights = found.3
        stormOpen = found.4
        stormNow = found.4
    }

    func stop() { task?.cancel() }

    var shoeRoutes: Set<RouteID> {
        guard let net = network, let s = net.architecture.inputs.firstIndex(of: .shoe) else { return [] }
        return Set((0..<net.layerSizes[1]).map { RouteID(layer: 0, from: s, to: $0) })
    }

    // MARK: closing

    func set(routes: Set<RouteID>, hubs: Set<HubRef>) {
        closedRoutes = routes
        closedHubs = hubs
        rescore()
    }

    func openEverything() { set(routes: [], hubs: []) }

    /// The free page: a tap closes (or reopens) the hub or route under it.
    func tap(_ p: CGPoint, _ g: MapGeometry) {
        guard page == 3 else { return }
        if let h = g.hub(at: p) {
            var hubs = closedHubs
            if hubs.contains(h) { hubs.remove(h) } else { hubs.insert(h) }
            set(routes: closedRoutes, hubs: hubs)
        } else if let r = g.route(at: p) {
            var routes = closedRoutes
            if routes.contains(r) { routes.remove(r) } else { routes.insert(r) }
            set(routes: routes, hubs: closedHubs)
        }
    }

    /// The quiz score with the overlay, the storm flights, and the flights that got worse the most.
    private func rescore() {
        guard let l = ledger, let net = network else { return }
        let closed = net.closing(routes: closedRoutes, hubs: closedHubs)
        let open = openErrors, storm = stormFlights
        task?.cancel()
        task = Task { [weak self] in
            let r = await Task.detached(priority: .userInitiated) { () -> (Float, Float, [Worse]) in
                let idx = l.indices(.check)
                var total: Float = 0, deltas: [(Int, Float, Float)] = []
                deltas.reserveCapacity(idx.count)
                for (k, i) in idx.enumerated() {
                    let e = abs(closed.predict(flight: i, in: l) - l.delay(i))
                    total += e
                    if k < open.count { deltas.append((i, open[k], e)) }
                }
                let worst = deltas.sorted { ($0.2 - $0.1) > ($1.2 - $1.1) }.prefix(3)
                    .filter { $0.2 - $0.1 >= 0.5 }.map { Worse(flight: $0.0, before: $0.1, after: $0.2) }
                return (total / Float(max(idx.count, 1)), closed.meanMinutesOff(indices: storm, in: l), worst)
            }.value
            guard let self, !Task.isCancelled else { return }
            self.quizNow = r.0
            self.stormNow = r.1
            self.worse = r.2
        }
    }

    func mapScene(reduceMotion: Bool) -> MapScene? {
        guard let l = ledger, let net = network else { return nil }
        let fi = worse.first?.flight ?? l.showcaseFlight
        let closed = net.closing(routes: closedRoutes, hubs: closedHubs)
        return MapScene(network: net, weights: net.weights, routeOrder: routeOrder, flight: l.flight(fi),
                        rest: closed.forward(flight: fi, in: l), run: chor.run, reduceMotion: reduceMotion,
                        allFacts: true, pointAt: page == 2 ? stormHub.map { ($0.row, $0.index) } : nil,
                        closedRoutes: closedRoutes, closedHubs: closedHubs)
    }

    // MARK: the pages

    private func scoreLine() -> String {
        guard let a = quizOpen, let b = quizNow else { return "" }
        return "Quiz score: off by \(String(format: "%.1f", Double(a))) → \(String(format: "%.1f", Double(b))) min."
    }

    var copy: AirlinePageCopy {
        guard network != nil, let open = quizOpen else {
            return AirlinePageCopy(text: "Loading your network…", button: "Continue", enabled: false, working: true)
        }
        let hub = stormHub.map { MapModel.hubName(row: $0.row, index: $0.index) } ?? "the storm hub"
        switch page {
        case 0:
            return AirlinePageCopy(
                text: "Every route and hub carries part of the guess. What happens if you close one?",
                detail: "Your network is off by \(AirlineState.minutes(open)) min on the 2023 quiz. Closing a part is temporary: open it again any time.",
                button: "Continue")
        case 1:
            if shoeRoutes.isEmpty {
                return AirlinePageCopy(text: "Your clipboard has no pilot's-shoes card, so there are no shoe routes to close.",
                                       detail: nil, button: "Continue")
            }
            if closedRoutes == shoeRoutes {
                return AirlinePageCopy(
                    text: "The shoe routes are closed. " + scoreLine(),
                    detail: "The shoes barely mattered, and your network had already learned that: its shoe routes were thin.",
                    button: "Continue", reserveText: ["The shoe routes are closed. Quiz score: off by 00.0 → 00.0 min."])
            }
            return AirlinePageCopy(text: "Close the pilot's shoes routes.", detail: "\(shoeRoutes.count) routes, from the shoe card to every row-1 hub.",
                                   button: "Close them", reserveText: ["The shoe routes are closed. Quiz score: off by 00.0 → 00.0 min."])
        case 2:
            if let h = stormHub, closedHubs == [h] {
                let storm = (stormOpen, stormNow)
                let stormLine = storm.0.flatMap { a in storm.1.map { b in " On the \(stormFlights.count.formatted()) stormy flights: off by \(AirlineState.minutes(a)) → \(AirlineState.minutes(b)) min." } } ?? ""
                return AirlinePageCopy(
                    text: "\(hub) is closed. " + scoreLine(),
                    detail: "Stormy flights get much worse: \(hub) was carrying the storm.\(stormLine)",
                    button: "Continue", reserveText: ["\(hub) is closed. Quiz score: off by 00.0 → 00.0 min."])
            }
            return AirlinePageCopy(
                text: "Now close the hub most linked to storms: \(hub).",
                detail: "The shoe routes are open again, so only \(hub) changes.",
                button: "Close \(hub)", enabled: stormHub != nil, reserveText: ["\(hub) is closed. Quiz score: off by 00.0 → 00.0 min."])
        case 3:
            let n = closedRoutes.count + closedHubs.count
            return AirlinePageCopy(
                text: "Your turn: click any route or hub to close it; click it again to open it.",
                detail: n == 0 ? "Nothing closed. " + scoreLine() : "\(n) closed. " + scoreLine(),
                button: "Continue")
        default:
            return AirlinePageCopy(
                text: "That is how researchers look inside real AI: close one part, see what breaks.",
                detail: "It is called **ablation** (switching a part off). The next lessons do it to a real language model.",
                button: "Done")
        }
    }

    func primary() -> Bool {
        guard network != nil else { return false }
        switch page {
        case 1 where !shoeRoutes.isEmpty && closedRoutes != shoeRoutes:
            set(routes: shoeRoutes, hubs: []); return false
        case 2 where stormHub != nil && closedHubs != [stormHub!]:
            set(routes: [], hubs: [stormHub!]); return false
        case pageCount - 1:
            return true
        default: break
        }
        go(to: page + 1)
        return false
    }

    func back() { if page > 0 { go(to: page - 1) } }
    func watchAgain() {}

    private func go(to p: Int) {
        page = p
        // each guided page starts from the whole network; the free page keeps what is closed
        if p == 1 || p == 2 || p == 0 { openEverything() }
        chor.frame(p == 2 ? .between(0) : .wide)
    }

    var summary: String? {
        guard anyClosed, let a = quizOpen, let b = quizNow else { return nil }
        return "\(closedRoutes.count) routes and \(closedHubs.count) hubs closed. Quiz score off by \(AirlineState.minutes(a)) → \(AirlineState.minutes(b)) minutes."
    }
}

// MARK: - the view: the map, and beside it what got worse

struct AirlineCloseLessonView: View {
    var onExit: (_ completed: Bool) -> Void
    @State private var model = AirlineCloseLesson()

    var body: some View {
        AirlineLessonShell(model: model, onExit: onExit) {
            if let l = model.ledger, model.network != nil {
                HStack(alignment: .top, spacing: 18) {
                    AirlineMapView(sceneFor: { model.mapScene(reduceMotion: $0) }, timelineActive: model.chor.timelineActive,
                                   summary: model.summary,
                                   onTapMap: model.page == 3 ? { model.tap($0, $1) } : nil)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    if model.page >= 1 && model.page <= 3 {
                        sidePanel(l).frame(width: 320)
                    }
                }
                .padding(.horizontal, 28).padding(.vertical, 16)
                .frame(maxWidth: 1240, maxHeight: .infinity)
            } else {
                Spacer(minLength: 0)
            }
        } accessory: { _ in
            if model.page == 3 && model.anyClosed {
                Button("Open everything") { model.openEverything() }.buttonStyle(PillButtonStyle(quiet: true))
            }
        }
        .task { await model.load() }
    }

    private func sidePanel(_ l: Ledger) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("QUIZ SCORE (2023)").font(.system(size: 10, weight: .semibold)).kerning(0.8).foregroundStyle(.white.opacity(0.45))
            if let a = model.quizOpen, let b = model.quizNow {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(String(format: "%.1f", Double(a))).font(.system(size: 20, weight: .semibold, design: .monospaced)).foregroundStyle(.secondary)
                    Image(systemName: "arrow.right").foregroundStyle(.secondary)
                    Text(String(format: "%.1f min", Double(b))).font(.system(size: 20, weight: .semibold, design: .monospaced))
                        .foregroundStyle(b > a + 0.5 ? MapModel.blame : Theme.answer)
                }
            }
            Text("GOT WORSE THE MOST").font(.system(size: 10, weight: .semibold)).kerning(0.8).foregroundStyle(.white.opacity(0.45)).padding(.top, 6)
            if model.worse.isEmpty {
                Text(model.anyClosed ? "No flight got worse by half a minute or more." : "Close something to see which flights suffer.")
                    .font(.system(size: 13)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            ForEach(model.worse) { w in
                let f = l.flight(w.flight)
                VStack(alignment: .leading, spacing: 3) {
                    Text("\(f.origin) → \(f.destination) · \(FlightCardView.date(year: f.year, day: f.dayOfYear))")
                        .font(.system(size: 13, weight: .semibold))
                    Text(AirlineNetLesson.facts(f, [.rain, .wind, .controllers, .holiday, .load]))
                        .font(.system(size: 11)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    Text("off by \(AirlineState.minutes(w.before)) → \(AirlineState.minutes(w.after)) min")
                        .font(.system(size: 12, weight: .semibold, design: .monospaced)).foregroundStyle(MapModel.blame)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.white.opacity(0.04)))
                .accessibilityElement(children: .combine)
            }
            Spacer(minLength: 0)
        }
        .padding(16)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color.white.opacity(0.08)))
    }
}
