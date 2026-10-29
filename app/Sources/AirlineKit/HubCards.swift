// The helpers behind Contract.swift's "Hub cards" section: closing routes and hubs as an overlay, the hub cards of
// lesson 6 (signed matches, contrasting flights, the close-the-hub test, the honest label), and a small study
// pile for exam day.

import Foundation

extension Network {
    func _closing(routes: Set<RouteID>, hubs: Set<HubRef>) -> Network {
        var net = self
        let sizes = layerSizes
        for r in routes where r.layer < weights.count && r.from < sizes[r.layer] && r.to < sizes[r.layer + 1] {
            net.weights[r.layer][r.from * sizes[r.layer + 1] + r.to] = 0
        }
        for h in hubs where h.row + 1 < weights.count && h.index < sizes[h.row + 1] {
            let l = h.row + 1, out = sizes[l + 1]
            for j in 0..<out { net.weights[l][h.index * out + j] = 0 }
        }
        return net
    }

    func _hubCard(_ hub: HubRef, in ledger: Ledger) -> HubCard {
        let idx = ledger.indices(.check)
        var act = [Float](), concepts = [[Float]](repeating: [], count: Concept.allCases.count)
        var facts = [[Float]](repeating: [], count: Fact.allCases.count)
        act.reserveCapacity(idx.count)
        for i in idx {
            let a = forward(flight: i, in: ledger)
            act.append(hub.row < a.hubs.count && hub.index < a.hubs[hub.row].count ? a.hubs[hub.row][hub.index] : 0)
            for c in Concept.allCases { concepts[c.rawValue].append(ledger.concept(c, i)) }
            for f in Fact.allCases { facts[f.rawValue].append(ledger.fact(f, i)) }
        }
        let firing = Float(act.filter { $0 > 0 }.count) / Float(max(act.count, 1))
        let cm = concepts.map { Self.pearson(act, $0) }
        let fm = facts.map { Self.pearson(act, $0) }
        let order = cm.indices.sorted { cm[$0] > cm[$1] }
        let top = Concept(rawValue: order[0])!
        // the runner-up is the best concept that is not a narrower case of the top one (perfect storm is a storm)
        let rival = order.dropFirst().map { Concept(rawValue: $0)! }.first { !top.narrowerCases.contains($0) }
        let runner = rival.map { cm[$0.rawValue] } ?? 0

        let firesOn = act.indices.sorted { act[$0] > act[$1] }.prefix(3).filter { act[$0] > 0 }.map { idx[$0] }
        let strongest = fm.indices.max { fm[$0] < fm[$1] } ?? 0
        let ignores = act.indices.filter { act[$0] <= 0 }
            .sorted { facts[strongest][$0] > facts[strongest][$1] }.prefix(3).map { idx[$0] }

        let flights = idx.filter { ledger.concept(top, $0) >= 0.5 }
        let open = meanMinutesOff(indices: flights, in: ledger)
        let closed = closing(hubs: [hub]).meanMinutesOff(indices: flights, in: ledger)

        let label: HubLabel
        if firing < 0.01 {
            label = .quiet
        } else if cm[top.rawValue] >= 0.7, cm[top.rawValue] >= 1.5 * runner, !flights.isEmpty,
                  closed >= open * 1.05, closed - open >= 0.5 {
            label = .mostly(top)
        } else if cm[top.rawValue] >= 1.5 * runner || runner < 0.2 {
            label = .linked(top)
        } else {
            label = .mix(top, rival ?? Concept(rawValue: order[1])!)
        }
        return HubCard(hub: hub, firingShare: firing, conceptMatch: cm, factMatch: fm, firesOn: firesOn, ignores: ignores,
                       topConcept: top, conceptFlights: flights.count, errorOpen: open, errorClosed: closed, label: label)
    }

    static func pearson(_ x: [Float], _ y: [Float]) -> Float {
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
}

extension Ledger {
    func _studyPile(count: Int, seed: UInt64) -> [Int] {
        var order = indices(.practice)
        var rng = SplitMix64(seed: seed)
        order.shuffle(using: &rng)
        return Array(order.prefix(max(count, 0))).sorted()
    }
}
