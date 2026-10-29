import SwiftUI
import AirlineKit

/// Lesson 6, "Name your hubs": every hub of the learner's trained network gets a card read from the 2023 flights
/// (what wakes it up, signed; flights it fires on and similar ones it ignores; the close-the-hub test). The learner
/// names one hub; the app then says honestly what the card supports. Labels follow HubLabel's rule: "mostly X"
/// only when the match is strong, clearly ahead of the runner-up, and closing the hub hurts X's flights.
@MainActor @Observable
final class AirlineHubNamesLesson: AirlineLessonModel {
    let title = AirlineLessonID.hubNames.title
    let pageCount = 5
    let chor = Choreographer()

    private(set) var ledger: Ledger?
    private(set) var network: Network?
    private(set) var routeOrder: [[Int]] = []
    private(set) var cards: [HubRef: HubCard] = [:]
    private(set) var featured: HubRef?
    private(set) var selected: HubRef?
    private(set) var page = 0
    private(set) var revealed = false
    var names: [String: String] = AirlineProgress.hubNames {
        didSet { AirlineProgress.hubNames = names }
    }
    var draft = ""

    var canGoBack: Bool { page > 0 && page != 2 && page != 4 }   // ← moves the cursor while typing a name
    var canWatchAgain: Bool { false }
    var ready: Bool { featured != nil }

    init() { chor.summary = { [weak self] in self?.summary } }

    func load() async {
        guard ledger == nil else { return }
        let l = await AirlineLedger.load()
        ledger = l
        let net = await AirlineProgress.trainedNetwork(l)
        network = net
        routeOrder = MapModel.routeOrder(net.weights)
        chor.frame(.wide, move: 0)
        let hubs = net.architecture.hidden.enumerated().flatMap { row, n in (0..<n).map { HubRef(row: row, index: $0) } }
        let found = await Task.detached(priority: .userInitiated) { () -> [HubRef: HubCard] in
            var out: [HubRef: HubCard] = [:]
            for h in hubs { out[h] = net.hubCard(h, in: l) }
            return out
        }.value
        cards = found
        // feature the clearest hub: a "mostly storm" one first (the hub lesson 4 pointed at),
        // then any "mostly" one, then the strongest match; row 1 before row 2 on ties
        func rank(_ c: HubCard) -> Int { c.label == .mostly(.storm) ? 2 : (Self.isMostly(c.label) ? 1 : 0) }
        let ranked = found.values.sorted { a, b in
            if rank(a) != rank(b) { return rank(a) > rank(b) }
            if rank(a) == 2, a.hub.row != b.hub.row { return a.hub.row < b.hub.row }
            return a.conceptMatch[a.topConcept.rawValue] > b.conceptMatch[b.topConcept.rawValue]
        }
        featured = ranked.first?.hub
        selected = featured
        draft = featured.map { names[code($0)] ?? "" } ?? ""
    }

    func stop() {}

    static func isMostly(_ l: HubLabel) -> Bool { if case .mostly = l { return true }; return false }
    func code(_ h: HubRef) -> String { MapModel.hubName(row: h.row, index: h.index) }
    var card: HubCard? { selected.flatMap { cards[$0] } }
    /// The featured hub's label stays hidden until the learner has named it.
    var showsLabel: Bool { revealed || (selected != nil && selected != featured) }

    func select(_ h: HubRef) {
        guard page == 4, cards[h] != nil else { return }
        selected = h
        draft = names[code(h)] ?? ""
        chor.frame(.between(h.row))
    }

    /// The map: tapping a hub on the last page selects it.
    func tap(_ p: CGPoint, _ g: MapGeometry) {
        if let h = g.hub(at: p) { select(h) }
    }

    func commitName() {
        guard let h = selected else { return }
        let name = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        if name.isEmpty { names.removeValue(forKey: code(h)) } else { names[code(h)] = name }
    }

    func mapScene(reduceMotion: Bool) -> MapScene? {
        guard let l = ledger, let net = network else { return nil }
        let fi = card?.firesOn.first ?? l.showcaseFlight
        return MapScene(network: net, weights: net.weights, routeOrder: routeOrder, flight: l.flight(fi),
                        rest: net.forward(flight: fi, in: l), run: chor.run, reduceMotion: reduceMotion,
                        allFacts: true, pointAt: selected.map { ($0.row, $0.index) })
    }

    // MARK: the pages

    var copy: AirlinePageCopy {
        guard ledger != nil, network != nil else {
            return AirlinePageCopy(text: "Loading your network…", button: "Continue", enabled: false, working: true)
        }
        guard let f = featured, let c = cards[f] else {
            return AirlinePageCopy(text: "Your network's hubs were never told what to look for.",
                                   detail: "Reading every hub's card on the 2023 flights…", button: "Continue", enabled: false, working: true)
        }
        let hub = code(f)
        switch page {
        case 0:
            return AirlinePageCopy(
                text: "Your network's hubs were never told what to look for. Each one picked something up by itself.",
                detail: "Every hub gets a **hub card**: what wakes it up, read from the 2023 flights it never studied.",
                button: "Continue")
        case 1:
            return AirlinePageCopy(
                text: "This is \(hub)'s card: what wakes it up.",
                detail: "Bars to the right: it gets busier when that is high. To the left: quieter. Below: flights it fires on, similar ones it ignores, and what breaks if it is closed.",
                button: "Continue")
        case 2:
            return AirlinePageCopy(
                text: "What would you call \(hub)?",
                detail: "Type a name from what the card shows. Your name stays, whatever the card says next.",
                button: "Reveal", enabled: !draft.trimmingCharacters(in: .whitespaces).isEmpty)
        case 3:
            let name = names[hub] ?? hub
            return AirlinePageCopy(text: revealSentence(name, c), detail: revealDetail(c), button: "Continue")
        default:
            return AirlinePageCopy(
                text: "Look at the others: hubs often mix ideas.",
                detail: "Click any hub on the map to see its card and name it. Some hubs are clean, many are mixes, and that is normal for real networks too.",
                button: "Done")
        }
    }

    private func revealSentence(_ name: String, _ c: HubCard) -> String {
        switch c.label {
        case .mostly(let x): return "“\(name)” it is. Your hub found the \(x.title.lowercased()) on its own."
        case .linked(let x): return "“\(name)” it is. Honestly, the card says: linked to \(x.title.lowercased()), not cleanly enough to say mostly."
        case .mix(let a, let b): return "“\(name)” it is. Honestly, the card says: a mix of \(a.title.lowercased()) and \(b.title.lowercased())."
        case .quiet: return "“\(name)” it is. Honestly, this hub is quiet: it fires on fewer than 1 in 100 flights."
        }
    }

    private func revealDetail(_ c: HubCard) -> String {
        let x = c.topConcept.title.lowercased()
        let m = String(format: "%.2f", Double(c.conceptMatch[c.topConcept.rawValue]))
        return "Its match with \(x) is \(m) (1 would be perfect), and closing it moves the \(c.conceptFlights.formatted()) \(x) flights from \(AirlineState.minutes(c.errorOpen)) to \(AirlineState.minutes(c.errorClosed)) min off. “Mostly” needs a match of at least 0.7, at least 1.5 times any other idea (narrower cases like a perfect storm, which is a storm too, don't count against it), and a real loss when closed." + related(c)
    }

    /// " It also fires on perfect storms, which are storms too." when a narrower case tracks along.
    private func related(_ c: HubCard) -> String {
        guard let n = HubCardView.relatedCase(c) else { return "" }
        return " It also fires on \(n.plural), which are \(c.topConcept.plural) too."
    }

    func primary() -> Bool {
        guard ready else { return false }
        switch page {
        case 2:
            guard !draft.trimmingCharacters(in: .whitespaces).isEmpty else { return false }
            commitName()
            revealed = true
        case 4:
            commitName()
            return true
        default: break
        }
        go(to: page + 1)
        return false
    }

    func back() {
        guard canGoBack else { return }
        if page == 4 { commitName() }
        go(to: page - 1)
    }

    func watchAgain() {}

    private func go(to p: Int) {
        page = p
        if p <= 3, let f = featured { selected = f; draft = names[code(f)] ?? draft }
        if let h = selected, p >= 1 { chor.frame(.between(h.row)) } else { chor.frame(.wide) }
    }

    var summary: String? {
        guard let c = card else { return nil }
        return "\(code(c.hub)): \(showsLabel ? c.label.words : "label hidden until you name it")."
    }
}

// MARK: - the view

struct AirlineHubNamesLessonView: View {
    var onExit: (_ completed: Bool) -> Void
    @State private var model = AirlineHubNamesLesson()

    var body: some View {
        AirlineLessonShell(model: model, onExit: onExit) {
            if let l = model.ledger, model.network != nil {
                HStack(alignment: .top, spacing: 18) {
                    AirlineMapView(sceneFor: { model.mapScene(reduceMotion: $0) }, timelineActive: model.chor.timelineActive,
                                   summary: model.summary,
                                   onTapMap: model.page == 4 ? { model.tap($0, $1) } : nil)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    if model.page >= 1, let c = model.card {
                        HubCardView(card: c, code: model.code(c.hub), name: model.names[model.code(c.hub)],
                                    showsLabel: model.showsLabel, ledger: l)
                            .frame(width: 360)
                            .transition(.opacity)
                    }
                }
                .padding(.horizontal, 28).padding(.vertical, 16)
                .frame(maxWidth: 1240, maxHeight: .infinity)
            } else {
                Spacer(minLength: 0)
            }
        } accessory: { _ in
            if model.page == 2 || model.page == 4 {
                TextField(model.page == 2 ? "A name for this hub" : "Name this hub (optional)", text: Bindable(model).draft)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 15))
                    .frame(width: 300)
                    .onSubmit { if model.page == 4 { model.commitName() } }
            }
        }
        .task { await model.load() }
    }
}

/// One hub's card: signed matches, contrasting flights, the close-the-hub test, and (once earned) its label.
struct HubCardView: View {
    let card: HubCard
    let code: String
    let name: String?
    let showsLabel: Bool
    let ledger: Ledger

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(code).font(.system(size: 22, weight: .bold, design: .monospaced))
                    if let name { Text("“\(name)”").font(.system(size: 17, weight: .semibold)).foregroundStyle(Theme.answer) }
                    Spacer()
                }
                Text(showsLabel ? card.label.words.capitalizedFirst : "Label hidden until you name it")
                    .font(.system(size: 13, weight: .semibold))
                    .padding(.horizontal, 10).padding(.vertical, 4)
                    .background(Capsule().fill(showsLabel && AirlineHubNamesLesson.isMostly(card.label) ? Theme.prompt.opacity(0.25) : Color.white.opacity(0.08)))
                Text("Fires on \(Int((card.firingShare * 100).rounded()))% of the 2023 flights")
                    .font(.system(size: 13)).foregroundStyle(.secondary)
                if showsLabel, let n = Self.relatedCase(card) {
                    Text("Also fires on \(n.plural), which are \(card.topConcept.plural) too.")
                        .font(.system(size: 12)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }

                section("What wakes it up")
                ForEach(Concept.allCases, id: \.self) { c in bar(c.title, card.conceptMatch[c.rawValue]) }
                ForEach(topFacts, id: \.self) { f in bar(f.title, card.factMatch[f.rawValue], fact: true) }

                section("Fires on")
                ForEach(card.firesOn, id: \.self) { i in flightRow(i, firing: true) }
                section("Ignores")
                ForEach(card.ignores, id: \.self) { i in flightRow(i, firing: false) }

                section("Close-the-hub test")
                Text("On the \(card.conceptFlights.formatted()) \(card.topConcept.title.lowercased()) flights: off by \(AirlineState.minutes(card.errorOpen)) min open → \(AirlineState.minutes(card.errorClosed)) min with \(code) closed.")
                    .font(.system(size: 13)).fixedSize(horizontal: false, vertical: true)
            }
            .padding(16)
        }
        .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(Color.white.opacity(0.08)))
        .accessibilityElement(children: .combine)
    }

    /// A narrower case of the hub's top idea that it also tracks (match ≥ 0.4): shown as related, not as a rival.
    static func relatedCase(_ c: HubCard) -> Concept? {
        c.topConcept.narrowerCases.filter { c.conceptMatch[$0.rawValue] >= 0.4 }.max { c.conceptMatch[$0.rawValue] < c.conceptMatch[$1.rawValue] }
    }

    /// The raw facts it tracks most (either way).
    private var topFacts: [Fact] {
        Fact.allCases.sorted { abs(card.factMatch[$0.rawValue]) > abs(card.factMatch[$1.rawValue]) }.prefix(3).map { $0 }
    }

    private func section(_ s: String) -> some View {
        Text(s.uppercased()).font(.system(size: 10, weight: .semibold)).kerning(0.8).foregroundStyle(.white.opacity(0.45)).padding(.top, 4)
    }

    /// A signed bar: right = busier when it is high, left = quieter.
    private func bar(_ label: String, _ r: Float, fact: Bool = false) -> some View {
        HStack(spacing: 8) {
            Text(label).font(.system(size: 12)).foregroundStyle(fact ? .secondary : .primary).frame(width: 110, alignment: .leading)
            GeometryReader { geo in
                let half = geo.size.width / 2, w = half * CGFloat(min(abs(r), 1))
                ZStack(alignment: .leading) {
                    Rectangle().fill(Color.white.opacity(0.15)).frame(width: 1).offset(x: half)
                    Capsule().fill(r >= 0 ? Theme.positive : Theme.negative)
                        .frame(width: max(w, 2), height: 7)
                        .offset(x: r >= 0 ? half : half - w)
                }
                .frame(maxHeight: .infinity)
            }
            .frame(height: 12)
            Text(String(format: "%+.2f", Double(r)).replacingOccurrences(of: "-", with: "−"))
                .font(.system(size: 12, design: .monospaced)).frame(width: 48, alignment: .trailing)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(label): match \(String(format: "%.2f", Double(r)))")
    }

    private func flightRow(_ i: Int, firing: Bool) -> some View {
        let f = ledger.flight(i)
        let facts = topFacts.prefix(2).map { "\($0.title.lowercased()) \($0.format(f.facts[$0.rawValue]))" }.joined(separator: ", ")
        return HStack(spacing: 8) {
            Circle().fill(firing ? Theme.positive : Color.white.opacity(0.25)).frame(width: 7, height: 7)
            VStack(alignment: .leading, spacing: 1) {
                Text("\(f.origin) → \(f.destination) · \(AirlineState.phrase(f.delay))").font(.system(size: 12, weight: .medium))
                Text(facts).font(.system(size: 11)).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            Text(firing ? "fires" : "quiet").font(.system(size: 11, weight: .semibold)).foregroundStyle(firing ? Theme.positive : .secondary)
        }
    }
}

extension String {
    var capitalizedFirst: String { prefix(1).uppercased() + dropFirst() }
}

extension Concept {
    /// "storms", "short-staffed days", "holiday rushes", "perfect storms".
    var plural: String {
        switch self {
        case .storm: "storms"
        case .shortStaff: "short-staffed days"
        case .holidayRush: "holiday rushes"
        case .perfectStorm: "perfect storms"
        }
    }
}
