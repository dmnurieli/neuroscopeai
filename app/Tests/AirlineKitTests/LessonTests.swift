// Lesson helpers: firing rules, the lesson flights, and the foggy mountain with its hikes.
// Run with the rest: cd app && DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test --filter AirlineKitTests

import Foundation
import XCTest
@testable import AirlineKit

final class LessonTests: XCTestCase {

    private func describe(_ i: Int, _ ledger: Ledger) -> String {
        let f = ledger.flight(i)
        let card = Fact.allCases.map { "\($0.title) \($0.format(f.facts[$0.rawValue]))" }.joined(separator: ", ")
        let concepts = Concept.allCases.map { String(format: "%@ %.2f", $0.title, f.concepts[$0.rawValue]) }.joined(separator: ", ")
        let fired = Rulebook.beginner.firingRules(flight: i, in: ledger)
        return String(format: "flight %d (%d): %@; raw wind %.2f, rain %.2f; %@; delay %.1f min; beginner rules firing %@ → %.0f min",
                      i, 2016 + f.year, card, f.facts[Fact.wind.rawValue], f.facts[Fact.rain.rawValue], concepts, f.delay,
                      "\(fired)", Rulebook.beginner.predict(flight: i, in: ledger))
    }

    func testFiringRulesMatchPredict() {
        let ledger = Fixture.ledger
        let books = [Rulebook.beginner, Rulebook(),
                     Rulebook(rules: [Rule(conditions: [Condition(fact: .holiday, above: true, threshold: 0.5),
                                                        Condition(fact: .load, above: true, threshold: 0.9)], addMinutes: 40),
                                      Rule(conditions: [Condition(fact: .controllers, above: false, threshold: 9)], addMinutes: 15),
                                      Rule(conditions: [Condition(fact: .wind, above: true, threshold: 20)], addMinutes: 25)])]
        for book in books {
            for i in stride(from: 0, to: ledger.count, by: 37) {
                let fired = book.firingRules(flight: i, in: ledger)
                XCTAssertEqual(fired, fired.sorted())
                XCTAssertEqual(fired.reduce(Float(0)) { $0 + book.rules[$1].addMinutes }, book.predict(flight: i, in: ledger))
            }
        }
    }

    func testFirstFlight() {
        let ledger = Fixture.ledger
        let check = ledger.indices(.check)
        XCTAssertEqual(ledger.firstFlight(in: .check) { _ in true }, check.first)
        XCTAssertNil(ledger.firstFlight(in: .practice) { _ in false })
        let windy = ledger.firstFlight(in: .practice) { ledger.fact(.wind, $0) > 40 }!
        XCTAssertTrue(ledger.indices(.practice).contains(windy))
        XCTAssertFalse(ledger.indices(.practice).contains { $0 < windy && ledger.fact(.wind, $0) > 40 })
    }

    func testNearMissFlight() {
        let ledger = Fixture.ledger
        func qualifies(_ i: Int) -> Bool {
            let w = ledger.fact(.wind, i)
            return w >= 25 && w < 29.5 && ledger.fact(.rain, i) >= 1 && ledger.fact(.controllers, i) <= 8 && ledger.delay(i) >= 90
        }
        guard let i = ledger.nearMissFlight else { return XCTFail("the default ledger must have a 29-mph flight") }
        XCTAssertTrue(ledger.indices(.check).contains(i))
        XCTAssertTrue(qualifies(i))
        let windiest = ledger.indices(.check).filter(qualifies).map { ledger.fact(.wind, $0) }.max()!
        XCTAssertEqual(ledger.fact(.wind, i), windiest, "the wind closest to 30 mph wins")
        XCTAssertFalse(Rulebook.beginner.firingRules(flight: i, in: ledger).contains(0), "the wind rule must miss it")
        print("near miss: " + describe(i, ledger))
    }

    func testNearMissFollowsTheLearnersRules() {
        let ledger = Fixture.ledger
        XCTAssertEqual(ledger.nearMissFlight(for: .beginner), ledger.nearMissFlight, "the beginner keeps the 29-mph flight")
        // A learner who lowered the wind rule to 25 mph catches the 29-mph flight: pick another near miss.
        let book = Rulebook(rules: [
            Rule(conditions: [Condition(fact: .wind, above: true, threshold: 25)], addMinutes: 60),
            Rule(conditions: [Condition(fact: .rain, above: true, threshold: 2)], addMinutes: 30),
        ])
        guard let i = ledger.nearMissFlight(for: book) else { return XCTFail("expected a near miss") }
        XCTAssertNotEqual(i, ledger.nearMissFlight)
        XCTAssertTrue(book.firingRules(flight: i, in: ledger).isEmpty, "no rule may fire on the near-miss flight")
        XCTAssertTrue(ledger.indices(.check).contains(i))
        XCTAssertGreaterThanOrEqual(ledger.fact(.rain, i), 1)
        XCTAssertLessThanOrEqual(ledger.fact(.controllers, i), 8)
        XCTAssertGreaterThanOrEqual(ledger.delay(i), 90)
        XCTAssertNotNil(ledger.closestMiss(flight: i, of: book))
        print("near miss for a 25-mph rule: " + describe(i, ledger))
    }

    func testCompoundFlight() {
        let ledger = Fixture.ledger
        guard let i = ledger.compoundFlight else { return XCTFail("the default ledger must have a compound flight") }
        XCTAssertTrue(ledger.indices(.check).contains(i))
        XCTAssertGreaterThanOrEqual(ledger.concept(.perfectStorm, i), 0.7)
        let best = ledger.indices(.check).filter { ledger.concept(.perfectStorm, $0) >= 0.7 }.map { ledger.delay($0) }.max()!
        XCTAssertEqual(ledger.delay(i), best)
        print("compound: " + describe(i, ledger) + String(format: "; direct net %.1f, lesson net %.1f",
              Fixture.direct.predict(flight: i, in: ledger), Fixture.lesson.predict(flight: i, in: ledger)))
    }

    func testMountainRoutes() {
        let ledger = Fixture.ledger, net = Fixture.lesson
        XCTAssertNil(Fixture.direct.mountainRoutes(in: ledger))
        guard let (a, b) = net.mountainRoutes(in: ledger) else { return XCTFail("lesson net must have routes") }
        XCTAssertEqual(a.layer, 0); XCTAssertEqual(b.layer, 0)
        XCTAssertEqual(a.to, b.to); XCTAssertNotEqual(a.from, b.from)
        // the hub is the row-1 hub most (signed) correlated with storm on the check year
        let idx = ledger.indices(.check)
        let acts = idx.map { net.forward(flight: $0, in: ledger) }
        let storm = idx.map { ledger.concept(.storm, $0) }
        let rs = (0..<net.architecture.hidden[0]).map { h in pearson(acts.map { $0.hubs[0][h] }, storm) }
        XCTAssertEqual(a.to, rs.firstIndex(of: rs.max()!)!)
        let n = net.layerSizes[1]
        let w = (0..<net.architecture.inputs.count).map { abs(net.weights[0][$0 * n + a.to]) }
        let top = w.indices.sorted { w[$0] > w[$1] }
        XCTAssertEqual([a.from, b.from], Array(top.prefix(2)))
        print(String(format: "mountain routes: %@ (%.3f), %@ (%.3f) into row-1 hub %d (storm r %.2f)",
                     net.architecture.inputs[a.from].title, net.weights[0][a.from * n + a.to],
                     net.architecture.inputs[b.from].title, net.weights[0][b.from * n + b.to], a.to, rs[a.to]))
    }

    func testMountainCentreIsTheNetwork() {
        let ledger = Fixture.ledger, net = Fixture.lesson
        let (a, b) = net.mountainRoutes(in: ledger)!
        let slice = net.mountain(a: a, b: b, span: 1.5, resolution: 41, sampleCount: 2_000, seed: 3, in: ledger)
        XCTAssertEqual(slice.loss.count, 41 * 41)
        XCTAssertEqual(slice.sample.count, 2_000)
        XCTAssertTrue(slice.sample.allSatisfy { ledger.indices(.practice).contains($0) })
        let centre = slice.loss[20 * 41 + 20]
        XCTAssertEqual(centre, net.meanMinutesOff(indices: slice.sample, in: ledger), accuracy: 1e-3)
        XCTAssertEqual(slice.aRange.lowerBound + 1.5, net.weights[0][a.from * 6 + a.to], accuracy: 1e-5)
        XCTAssertEqual(slice.minLoss, slice.loss.min()); XCTAssertEqual(slice.maxLoss, slice.loss.max())
        XCTAssertLessThan(slice.minLoss, slice.maxLoss)
        // an off-centre cell is the network with those two routes moved, nothing else
        var moved = net
        moved.weights[0][a.from * 6 + a.to] = slice.aRange.lowerBound
        moved.weights[0][b.from * 6 + b.to] = slice.bRange.upperBound
        XCTAssertEqual(slice.loss[40 * 41 + 0], moved.meanMinutesOff(indices: slice.sample, in: ledger), accuracy: 1e-3)
        // general path: a route deeper in the network (row 1 → row 2) against a clipboard route
        let deep = RouteID(layer: 1, from: 2, to: 3)
        let s2 = net.mountain(a: a, b: deep, span: 1, resolution: 5, sampleCount: 300, seed: 9, in: ledger)
        var m2 = net
        m2.weights[0][a.from * 6 + a.to] = s2.aRange.upperBound
        m2.weights[1][2 * 6 + 3] = s2.bRange.lowerBound
        XCTAssertEqual(s2.loss[0 * 5 + 4], m2.meanMinutesOff(indices: s2.sample, in: ledger), accuracy: 1e-3)
        XCTAssertEqual(s2.loss[2 * 5 + 2], net.meanMinutesOff(indices: s2.sample, in: ledger), accuracy: 1e-3)
        print(String(format: "mountain: centre %.3f min, range %.2f…%.2f", centre, slice.minLoss, slice.maxLoss))
    }

    func testHike() {
        let ledger = Fixture.ledger, net = Fixture.lesson
        let (a, b) = net.mountainRoutes(in: ledger)!
        let slice = net.mountain(a: a, b: b, span: 1.5, resolution: 21, sampleCount: 2_000, seed: 3, in: ledger)
        let start = SIMD2<Float>(slice.aRange.lowerBound + 0.3, slice.bRange.upperBound - 0.3)
        let sane = net.hike(on: slice, from: start, stepSize: 0.5, steps: 60, seed: 3, in: ledger)
        XCTAssertFalse(sane.tumbled)
        XCTAssertEqual(sane.points.count, 61); XCTAssertEqual(sane.losses.count, 61)
        XCTAssertEqual(sane.points[0], start)
        XCTAssertLessThan(sane.losses.last!, sane.losses.first!)
        let rises = zip(sane.losses, sane.losses.dropFirst()).filter { $1 > $0 + 0.05 }.count
        print(String(format: "hike stride 0.5: %.2f → %.2f min over %d steps, %d rises; end (%.3f, %.3f)",
                     sane.losses.first!, sane.losses.last!, sane.points.count - 1, rises, sane.points.last!.x, sane.points.last!.y))
        let wild = net.hike(on: slice, from: start, stepSize: 1e4, steps: 60, seed: 3, in: ledger)
        XCTAssertTrue(wild.tumbled)
        XCTAssertLessThan(wild.points.count, 61)
        XCTAssertEqual(wild.points.count, wild.losses.count)
        print("hike stride 1e4: tumbled after \(wild.points.count - 1) steps at \(wild.points.last!)")
    }

    // Speed budget (asserted in release only): mountain(41, 2,000) < 0.3 s.
    func testMountainSpeed() {
        let ledger = Fixture.ledger, net = Fixture.lesson
        let (a, b) = net.mountainRoutes(in: ledger)!
        let t = Date()
        let slice = net.mountain(a: a, b: b, span: 1.5, resolution: 41, sampleCount: 2_000, seed: 3, in: ledger)
        let elapsed = Date().timeIntervalSince(t)
        let t2 = Date()
        _ = net.hike(on: slice, from: SIMD2(slice.aRange.lowerBound, slice.bRange.lowerBound), stepSize: 0.5, steps: 100, seed: 3, in: ledger)
        let hike = Date().timeIntervalSince(t2)
        print(String(format: "mountain 41×41 on 2,000 flights %.3f s; hike 100 steps %.3f s", elapsed, hike))
        #if !DEBUG
        XCTAssertLessThan(elapsed, 0.3)
        #endif
    }
}
