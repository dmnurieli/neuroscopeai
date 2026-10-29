// AirlineKit: the lesson's promises, checked on the real default ledger.
// Run: cd app && DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter AirlineKitTests
// Release timings: add `-c release -Xswiftc -enable-testing` (the gradient check needs @testable).

import Foundation
import XCTest
@testable import AirlineKit

/// Built once per test process: the default ledger and the two networks the premise compares.
enum Fixture {
    static let ledger = Ledger.generate()
    /// The lesson's network seed. A sweep of 10 seeds (Trainer seed 7) put a storm hub in row 1 (signed r ≥ 0.6)
    /// on 9; seed 2026 learned storm in row 2 instead (r 0.92) and has none in row 1.
    static let lessonSeed: UInt64 = Architecture.lessonSeed
    /// Six passes over practice.
    static let sixPasses = 6 * ((ledger.indices(.practice).count + 63) / 64)

    static func trained(_ arch: Architecture) -> Network {
        let trainer = Trainer(network: Network(architecture: arch, ledger: ledger, seed: lessonSeed), ledger: ledger)
        precondition(trainer.train(batches: sixPasses) == .ok)
        return trainer.network
    }
    static let lesson = trained(.lesson)
    static let direct = trained(Architecture(inputs: Fact.allCases, hidden: []))
}

func pearson(_ x: [Float], _ y: [Float]) -> Float {
    let n = Double(x.count)
    let mx = x.reduce(0.0) { $0 + Double($1) } / n, my = y.reduce(0.0) { $0 + Double($1) } / n
    var sxy = 0.0, sxx = 0.0, syy = 0.0
    for k in 0..<x.count {
        let (a, b) = (Double(x[k]) - mx, Double(y[k]) - my)
        sxy += a * b; sxx += a * a; syy += b * b
    }
    return sxx > 0 && syy > 0 ? Float(sxy / (sxx * syy).squareRoot()) : 0
}

final class AirlineKitTests: XCTestCase {

    // 1. Same seed → same world; different seed → different world.
    func testDeterminism() {
        let a = Ledger.generate(seed: 99, count: 20_000), b = Ledger.generate(seed: 99, count: 20_000)
        let c = Ledger.generate(seed: 100, count: 20_000)
        var differs = false
        for i in 0..<a.count {
            XCTAssertEqual(a.delay(i), b.delay(i))
            XCTAssertEqual(a.flight(i).facts, b.flight(i).facts)
            XCTAssertEqual(a.flight(i).concepts, b.flight(i).concepts)
            if a.delay(i) != c.delay(i) || a.flight(i).facts != c.flight(i).facts { differs = true }
        }
        let (fa, fb) = (a.flight(1234), b.flight(1234))
        XCTAssertEqual([fa.origin, fa.destination], [fb.origin, fb.destination])
        XCTAssertEqual([fa.scheduledDeparture, fa.actualArrival, fa.passengers, fa.seats, fa.dayOfYear],
                       [fb.scheduledDeparture, fb.actualArrival, fb.passengers, fb.seats, fb.dayOfYear])
        XCTAssertTrue(differs, "a different seed must give a different world")
        XCTAssertEqual(Network(architecture: .lesson, ledger: a, seed: 5), Network(architecture: .lesson, ledger: a, seed: 5))
    }

    // 2. Analytic gradients match central finite differences of an independent Double-precision loss.
    func testGradientCheck() {
        let ledger = Ledger.generate(seed: 11, count: 2_000)
        let arch = Architecture(inputs: [.rain, .wind, .controllers, .load, .hour], hidden: [4, 3])
        // Fresh stresses are exactly 0, so a row-2 hub whose row-1 inputs are all shut tallies exactly 0: a ReLU
        // kink, where a finite difference sees half a slope. Small resting stress moves every hub off the kink.
        var net = Network(architecture: arch, ledger: ledger, seed: 3)
        for l in 0..<(net.biases.count - 1) { net.biases[l] = net.biases[l].indices.map { 0.05 + 0.02 * Float($0) } }
        let batch = Array(ledger.indices(.practice).prefix(16))
        let analytic = net.gradient(indices: batch, in: ledger, upkeep: 0)

        var w = net.weights.map { $0.map(Double.init) }, b = net.biases.map { $0.map(Double.init) }
        let sizes = net.layerSizes
        func loss() -> Double {
            var total = 0.0
            for i in batch {
                var cur = arch.inputs.enumerated().map { k, f in
                    (Double(ledger.fact(f, i)) - Double(net.inputMean[k])) / Double(net.inputSD[k]) }
                for l in 0..<(sizes.count - 1) {
                    let (nin, nout) = (sizes[l], sizes[l + 1])
                    cur = (0..<nout).map { j in
                        let z = (0..<nin).reduce(b[l][j]) { $0 + cur[$1] * w[l][$1 * nout + j] }
                        return l == sizes.count - 2 ? z : max(0, z)
                    }
                }
                let e = cur[0] - (Double(ledger.delay(i)) - Double(net.delayMean)) / Double(net.delaySD)
                total += abs(e) <= 1 ? 0.5 * e * e : abs(e) - 0.5
            }
            return total / Double(batch.count)
        }
        let eps = 1e-4
        var worst = 0.0, checked = 0
        func compare(_ a: Float, _ numeric: Double, _ label: String) {
            let rel = abs(Double(a) - numeric) / max(abs(Double(a)), abs(numeric), 1e-3)
            worst = max(worst, rel); checked += 1
            XCTAssertLessThan(rel, 1e-2, "\(label): analytic \(a) vs numeric \(numeric)")
        }
        for l in 0..<w.count {
            for k in 0..<w[l].count {
                let v = w[l][k]
                w[l][k] = v + eps; let up = loss()
                w[l][k] = v - eps; let down = loss()
                w[l][k] = v
                compare(analytic.weights[l][k], (up - down) / (2 * eps), "w[\(l)][\(k)]")
            }
            for k in 0..<b[l].count {
                let v = b[l][k]
                b[l][k] = v + eps; let up = loss()
                b[l][k] = v - eps; let down = loss()
                b[l][k] = v
                compare(analytic.biases[l][k], (up - down) / (2 * eps), "b[\(l)][\(k)]")
            }
        }
        print("gradient check: \(checked) parameters, worst relative error \(worst)")
    }

    // 3. The lesson premise: hubs beat the direct routes, which beat the beginner's rulebook (CHECK year).
    func testLessonPremise() {
        let ledger = Fixture.ledger
        let rulebook = Rulebook.beginner.meanMinutesOff(.check, in: ledger)
        let direct = Fixture.direct.meanMinutesOff(.check, in: ledger)
        let lesson = Fixture.lesson.meanMinutesOff(.check, in: ledger)
        let average = ledger.indices(.check).reduce(0.0) { $0 + Double(abs(ledger.delay($1) - ledger.delayMean)) }
            / Double(ledger.indices(.check).count)
        let report = String(format: "check year, minutes off: average %.2f, rulebook %.2f, direct %.2f, 2x6 hubs %.2f (%d batches)",
                            average, rulebook, direct, lesson, Fixture.sixPasses)
        print(report)
        XCTAssertLessThan(lesson, direct, report)
        XCTAssertLessThan(lesson, rulebook, report)
        XCTAssertLessThan(direct, rulebook, report)
    }

    // 4. A row-1 hub finds the storm on its own (signed correlation on the CHECK year).
    func testConceptDiscovery() {
        let ledger = Fixture.ledger, net = Fixture.lesson
        let idx = ledger.indices(.check)
        let acts = idx.map { net.forward(flight: $0, in: ledger) }
        var lines: [String] = [], stormBest: Float = -1
        for c in Concept.allCases {
            let truth = idx.map { ledger.concept(c, $0) }
            for row in 0..<net.architecture.hidden.count {
                let best = (0..<net.architecture.hidden[row]).map { h in pearson(acts.map { $0.hubs[row][h] }, truth) }.max() ?? 0
                lines.append(String(format: "%@ row %d: %.2f", c.title, row + 1, best))
                if c == .storm && row == 0 { stormBest = best }
            }
        }
        let dead = (0..<net.architecture.hidden[0]).filter { h in !acts.contains { $0.hubs[0][h] > 0 } }.count
        let report = "best signed hub correlation: " + lines.joined(separator: "; ") + "; dead row-1 hubs: \(dead)"
        print(report)
        XCTAssertGreaterThanOrEqual(stormBest, 0.6, report)
    }

    // 5. Too big a stride: the hiker tumbles, training stops, every route stays a number.
    func testTumbleGuard() {
        let ledger = Fixture.ledger
        var config = TrainConfig()
        config.stepSize = 1e6
        let trainer = Trainer(network: Network(architecture: .lesson, ledger: ledger, seed: Fixture.lessonSeed),
                              ledger: ledger, config: config)
        XCTAssertEqual(trainer.train(batches: 500), .tumbled)
        XCTAssertTrue(trainer.network.weights.allSatisfy { $0.allSatisfy(\.isFinite) })
        XCTAssertTrue(trainer.network.biases.allSatisfy { $0.allSatisfy(\.isFinite) })
    }

    // 6. Sealed splits and practice-only normalisation.
    func testSealedSplit() {
        let ledger = Fixture.ledger
        let p = ledger.indices(.practice), c = ledger.indices(.check), e = ledger.indices(.exam)
        XCTAssertEqual(p.count + c.count + e.count, ledger.count)
        XCTAssertEqual(Set(p).union(c).union(e).count, ledger.count)
        XCTAssertTrue(Set(p).isDisjoint(with: c) && Set(p).isDisjoint(with: e) && Set(c).isDisjoint(with: e))
        XCTAssertTrue(p.allSatisfy { (0...6).contains(ledger.flight($0).year) })
        XCTAssertTrue(c.allSatisfy { ledger.flight($0).year == 7 })
        XCTAssertTrue(e.allSatisfy { (8...9).contains(ledger.flight($0).year) })
        XCTAssertEqual(p, p.sorted())
        for f in Fact.allCases {
            let v = p.map { Double(ledger.fact(f, $0)) }
            let mean = v.reduce(0, +) / Double(v.count)
            let sd = (v.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(v.count)).squareRoot()
            XCTAssertEqual(Double(ledger.factMean[f.rawValue]), mean, accuracy: 1e-4 * max(1, abs(mean)), "\(f) mean")
            XCTAssertEqual(Double(ledger.factSD[f.rawValue]), sd, accuracy: 1e-4 * max(1, sd), "\(f) sd")
        }
        let d = p.map { Double(ledger.delay($0)) }
        let dm = d.reduce(0, +) / Double(d.count)
        let dsd = (d.reduce(0) { $0 + ($1 - dm) * ($1 - dm) } / Double(d.count)).squareRoot()
        XCTAssertEqual(Double(ledger.delayMean), dm, accuracy: 1e-3)
        XCTAssertEqual(Double(ledger.delaySD), dsd, accuracy: 1e-3)
        let net = Network(architecture: Architecture(inputs: [.wind, .rain], hidden: [2]), ledger: ledger, seed: 1)
        XCTAssertEqual(net.inputMean, [ledger.factMean[Fact.wind.rawValue], ledger.factMean[Fact.rain.rawValue]])
    }

    // The minimal loop: one flight, one visible adjustment toward the truth, every route accounted for.
    func testLearnFromOneFlight() {
        let ledger = Fixture.ledger
        let trainer = Trainer(network: Network(architecture: .lesson, ledger: ledger, seed: Fixture.lessonSeed), ledger: ledger)
        let i = ledger.showcaseFlight
        XCTAssertEqual(trainer.network.predict(flight: i, in: ledger), trainer.network.forward(flight: i, in: ledger).gateMinutes)
        let step = trainer.learn(fromFlight: i, stepSize: 0.1)
        XCTAssertEqual(step.truth, ledger.delay(i))
        XCTAssertLessThan(abs(step.guessAfter - step.truth), abs(step.guessBefore - step.truth))
        XCTAssertEqual(step.guessAfter, trainer.network.predict(flight: i, in: ledger), accuracy: 1e-3)
        XCTAssertEqual(step.changes.count, trainer.network.weights.reduce(0) { $0 + $1.count })
        let moves = step.changes.map { abs($0.after - $0.before) }
        XCTAssertEqual(moves, moves.sorted(by: >))
        XCTAssertEqual(trainer.flightsSeen, 1)
        let c = step.changes[0]
        XCTAssertEqual(trainer.network.weights[c.route.layer][c.route.from * trainer.network.layerSizes[c.route.layer + 1] + c.route.to], c.after)
        print(String(format: "learn on flight %d: guess %.1f → %.1f, truth %.1f; biggest move %@ %.3f → %.3f",
                     i, step.guessBefore, step.guessAfter, step.truth, "\(c.route)", c.before, c.after))
    }

    // The nudge animation draws blame per hub (LearnStep.hubBlame) and per route (changes[].blame); they must be
    // the same backward walk: |route gradient| = |activation_from × hubBlame_to|, shut hubs pass back nothing.
    func testHubBlameMatchesRouteBlame() {
        let ledger = Fixture.ledger
        let net = Network(architecture: .lesson, ledger: ledger, seed: Fixture.lessonSeed)
        let i = ledger.showcaseFlight
        let acts = net.forward(flight: i, in: ledger)
        let step = Trainer(network: net, ledger: ledger).learn(fromFlight: i, stepSize: 0.1)
        let sizes = net.layerSizes
        XCTAssertEqual(step.hubBlame.map(\.count), Array(sizes.dropFirst()))
        // the gate's signal is the clamped Huber gradient: same sign as (guess − truth)
        XCTAssertEqual(step.hubBlame.last!.count, 1)
        XCTAssertEqual(step.hubBlame.last![0] > 0, step.guessBefore > step.truth)
        XCTAssertLessThanOrEqual(abs(step.hubBlame.last![0]), 1)
        for (row, pre) in acts.preActivation.enumerated() {
            for k in pre.indices where pre[k] <= 0 { XCTAssertEqual(step.hubBlame[row][k], 0, "a shut hub passes no blame back") }
        }
        XCTAssertTrue(step.hubBlame.dropLast().contains { $0.contains { $0 != 0 } }, "some hub must carry blame")
        var checked = 0
        for (n, c) in step.changes.enumerated() where n % 3 == 0 {
            let from: [Float] = c.route.layer == 0 ? acts.inputs : acts.hubs[c.route.layer - 1]
            let expected = abs(from[c.route.from] * step.hubBlame[c.route.layer][c.route.to])
            XCTAssertEqual(c.blame, expected, accuracy: 1e-5 + 1e-4 * expected, "\(c.route)")
            // the walk's rule card: the real nudge = −stride × activation_from × hubBlame_to (signed, exactly the step)
            let nudge = -0.1 * from[c.route.from] * step.hubBlame[c.route.layer][c.route.to]
            XCTAssertEqual(c.after - c.before, nudge, accuracy: 1e-5 + 1e-3 * abs(nudge), "\(c.route)")
            checked += 1
        }
        XCTAssertGreaterThan(checked, 25)
    }

    func testFlightsAreConsistent() {
        let ledger = Fixture.ledger
        let airports = Set(["MIA", "JFK", "BOS", "LAX", "SFO", "LAS", "MCO", "IAD", "SAN", "AUS"])
        for i in stride(from: 0, to: ledger.count, by: 97) {
            let f = ledger.flight(i)
            XCTAssertNotEqual(f.origin, f.destination)
            XCTAssertTrue(airports.contains(f.origin) && airports.contains(f.destination))
            XCTAssertEqual(f.scheduledDeparture / 60, Int(f.facts[Fact.hour.rawValue]))
            XCTAssertGreaterThan(f.scheduledArrival, f.scheduledDeparture)
            XCTAssertEqual(f.actualArrival, f.scheduledArrival + Int(f.delay.rounded()))
            XCTAssertGreaterThanOrEqual(f.actualDeparture, f.scheduledDeparture)
            XCTAssertEqual(Float(f.passengers) / Float(f.seats), f.facts[Fact.load.rawValue], accuracy: 1e-6)
            XCTAssertTrue((-15...300).contains(f.delay))
            XCTAssertTrue((1...365).contains(f.dayOfYear))
        }
    }

    func testRulebookAndCards() {
        let ledger = Fixture.ledger
        XCTAssertEqual(Rulebook().predict(flight: 0, in: ledger), 0)
        for i in 0..<500 {
            let expected: Float = (ledger.fact(.wind, i) > 30 ? 60 : 0) + (ledger.fact(.rain, i) > 2 ? 30 : 0)
            XCTAssertEqual(Rulebook.beginner.predict(flight: i, in: ledger), expected)
        }
        let both = Rulebook(rules: [Rule(conditions: [Condition(fact: .holiday, above: true, threshold: 0.5),
                                                      Condition(fact: .load, above: true, threshold: 0.9)], addMinutes: 40)])
        for i in 0..<500 {
            let fires = ledger.fact(.holiday, i) > 0.5 && ledger.fact(.load, i) > 0.9
            XCTAssertEqual(both.predict(flight: i, in: ledger), fires ? 40 : 0)
        }
        XCTAssertEqual(Fact.rain.format(1.83), "1.8 in")
        XCTAssertEqual(Fact.wind.format(31.4), "31 mph")
        XCTAssertEqual(Fact.controllers.format(7), "7")
        XCTAssertEqual(Fact.load.format(0.96), "96% full")
        XCTAssertEqual(Fact.holiday.format(1), "Yes")
        XCTAssertEqual(Fact.holiday.format(0), "No")
        XCTAssertEqual(Fact.hour.format(17), "5 pm")
        XCTAssertEqual(Fact.hour.format(12), "12 pm")
        XCTAssertEqual(Fact.hour.format(6), "6 am")
        XCTAssertEqual(Fact.crews.format(2), "2")
        XCTAssertEqual((0...4).map { Fact.shoe.format(Float($0)) }, ["Black", "Brown", "Red", "Blue", "White"])
        XCTAssertEqual(Fact.allCases.map(\.title),
                       ["Rain", "Wind", "Controllers on duty", "Plane", "Holiday week", "Departure", "Spare crews", "Pilot's shoes"])
        XCTAssertEqual(Concept.allCases.map(\.title), ["Storm", "Short-staffed", "Holiday rush", "Perfect storm"])
    }

    func testShowcaseFlights() {
        let ledger = Fixture.ledger
        let s = ledger.showcaseFlight, u = ledger.unseenStormFlight
        XCTAssertTrue(ledger.indices(.practice).contains(s))
        XCTAssertTrue(ledger.indices(.check).contains(u))
        XCTAssertGreaterThanOrEqual(ledger.concept(.storm, u), 0.8)
        let best = ledger.indices(.practice).map { ledger.concept(.perfectStorm, $0) }.max()!
        XCTAssertEqual(ledger.concept(.perfectStorm, s), best)
        for i in [s, u] {
            let f = ledger.flight(i)
            let card = Fact.allCases.map { "\($0.title) \($0.format(f.facts[$0.rawValue]))" }.joined(separator: ", ")
            print(String(format: "flight %d (%d, day %d, %@→%@): %@; delay %.1f min; lesson net guesses %.1f",
                         i, 2016 + f.year, f.dayOfYear, f.origin, f.destination, card, f.delay,
                         Fixture.lesson.predict(flight: i, in: ledger)))
        }
    }

    // Speed budget (asserted in release only): ledger < 0.3 s, 1,000 lesson batches < 0.5 s.
    func testSpeed() {
        var t = Date()
        let ledger = Ledger.generate()
        let generate = Date().timeIntervalSince(t)
        let trainer = Trainer(network: Network(architecture: .lesson, ledger: ledger, seed: 1), ledger: ledger)
        t = Date()
        XCTAssertEqual(trainer.train(batches: 1_000), .ok)
        let train = Date().timeIntervalSince(t)
        print(String(format: "Ledger.generate %.3f s; train 1,000 batches of 64 %.3f s", generate, train))
        #if !DEBUG
        XCTAssertLessThan(generate, 0.3)
        XCTAssertLessThan(train, 0.5)
        #endif
    }
}
