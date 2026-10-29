// Lessons 6–8: closing parts as an overlay, hub cards and their honest labels, exam day's crammer.
// Measured on the default ledger and the lesson network (Fixture.lesson, 2×6, six passes), 2026-10-28.

import XCTest
@testable import AirlineKit

final class HubCardTests: XCTestCase {

    // Closing is an overlay: the original is untouched; a closed hub contributes nothing downstream.
    func testClosingIsAnOverlay() {
        let l = Fixture.ledger, net = Fixture.lesson
        let i = l.showcaseFlight
        let hub = HubRef(row: 0, index: 0)
        let closed = net.closing(hubs: [hub])
        XCTAssertEqual(net, Fixture.lesson)
        XCTAssertNotEqual(closed, net)
        // by hand: zero hub 0's activation and run the rest of the network
        let a = net.forward(flight: i, in: l)
        var h1 = a.hubs[0]; h1[0] = 0
        let sizes = net.layerSizes
        var h2 = [Float](repeating: 0, count: sizes[2])
        for j in 0..<sizes[2] {
            var s = net.biases[1][j]
            for k in 0..<sizes[1] { s += h1[k] * net.weights[1][k * sizes[2] + j] }
            h2[j] = max(0, s)
        }
        var out = net.biases[2][0]
        for k in 0..<sizes[2] { out += h2[k] * net.weights[2][k] }
        XCTAssertEqual(closed.predict(flight: i, in: l), out * net.delaySD + net.delayMean, accuracy: 1e-3)
        let route = RouteID(layer: 0, from: 1, to: 2)
        XCTAssertEqual(net.closing(routes: [route]).weights[0][1 * sizes[1] + 2], 0)
    }

    // Lesson 8's guided moment: the shoes barely matter; the storm hub matters a great deal on storm flights.
    func testShoesBarelyMatterStormHubDoes() {
        let l = Fixture.ledger, net = Fixture.lesson
        let shoe = net.architecture.inputs.firstIndex(of: .shoe)!
        let shoeRoutes = Set((0..<net.layerSizes[1]).map { RouteID(layer: 0, from: shoe, to: $0) })
        let base = net.meanMinutesOff(.check, in: l)
        let noShoes = net.closing(routes: shoeRoutes).meanMinutesOff(.check, in: l)
        XCTAssertLessThan(abs(noShoes - base), 0.25, "closing the shoe routes barely moves the quiz score")
        let storm = l.indices(.check).filter { l.concept(.storm, $0) >= 0.5 }
        let cards = (0..<6).map { net.hubCard(HubRef(row: 0, index: $0), in: l) }
        let stormHub = cards.max { $0.conceptMatch[Concept.storm.rawValue] < $1.conceptMatch[Concept.storm.rawValue] }!
        let open = net.meanMinutesOff(indices: storm, in: l)
        let closed = net.closing(hubs: [stormHub.hub]).meanMinutesOff(indices: storm, in: l)
        XCTAssertGreaterThan(closed - open, 20, "closing the storm hub makes storm flights much worse")
        print(String(format: "close a route: quiz %.2f → %.2f without shoes; storm flights %.1f → %.1f without hub %d",
                     base, noShoes, open, closed, stormHub.hub.index))
    }

    // Lesson 6's label rule, checked on every hub of the lesson network.
    func testHubLabelsFollowTheRule() {
        let l = Fixture.ledger, net = Fixture.lesson
        var labels: [String] = []
        for row in 0..<2 { for k in 0..<6 {
            let c = net.hubCard(HubRef(row: row, index: k), in: l)
            let sorted = c.conceptMatch.sorted(by: >)
            XCTAssertEqual(c.conceptMatch[c.topConcept.rawValue], sorted[0])
            // the rival: the best concept that is not a narrower case of the top one
            let rival = Concept.allCases.filter { $0 != c.topConcept && !c.topConcept.narrowerCases.contains($0) }
                .map { c.conceptMatch[$0.rawValue] }.max() ?? 0
            XCTAssertLessThanOrEqual(c.firesOn.count, 3)
            XCTAssertLessThanOrEqual(c.ignores.count, 3)
            for i in c.ignores { XCTAssertLessThanOrEqual(net.forward(flight: i, in: l).hubs[row][k], 0) }
            switch c.label {
            case .mostly(let x):
                XCTAssertEqual(x, c.topConcept)
                XCTAssertGreaterThanOrEqual(sorted[0], 0.7)
                XCTAssertGreaterThanOrEqual(sorted[0], 1.5 * rival)
                XCTAssertGreaterThan(c.errorClosed, c.errorOpen)
            case .quiet:
                XCTAssertLessThan(c.firingShare, 0.01)
            default:
                XCTAssertGreaterThanOrEqual(c.firingShare, 0.01)
            }
            labels.append("\(row).\(k) \(c.label)")
        } }
        // measured: hub 0.1 (ORD) is mostly holiday rush (match 0.98, closing it 10.9 → 31 min on those flights)
        let ord = net.hubCard(HubRef(row: 0, index: 1), in: l)
        XCTAssertEqual(ord.label, .mostly(.holidayRush))
        // perfect storm is a storm, so it no longer counts against the storm hubs: ATL (row 1) and SLC (row 2)
        // read "mostly storm" (matches 0.81 and 0.94; closing them hurts the storm flights 9.3 → 65 and → 50 min)
        XCTAssertEqual(net.hubCard(HubRef(row: 0, index: 0), in: l).label, .mostly(.storm))
        XCTAssertEqual(net.hubCard(HubRef(row: 1, index: 4), in: l).label, .mostly(.storm))
        XCTAssertEqual(Concept.storm.narrowerCases, [.perfectStorm])
        XCTAssertEqual(Concept.shortStaff.narrowerCases, [.perfectStorm])
        XCTAssertTrue(Concept.holidayRush.narrowerCases.isEmpty && Concept.perfectStorm.narrowerCases.isEmpty)
        print("hub labels:", labels.joined(separator: "; "))
    }

    // Exam day: a big network cramming a tiny study pile aces its homework and fails the sealed exam; the
    // lesson network does about the same on both.
    func testExamDayCrammer() {
        let l = Fixture.ledger, net = Fixture.lesson
        let pile = l.studyPile(count: 200, seed: 3)
        XCTAssertEqual(pile.count, 200)
        XCTAssertTrue(Set(pile).isSubset(of: Set(l.indices(.practice))))
        let crammer = Trainer(network: Network(architecture: Architecture(inputs: Fact.allCases, hidden: [8, 8]), ledger: l,
                                               seed: Architecture.lessonSeed), ledger: l, studyPile: pile)
        XCTAssertEqual(crammer.train(batches: 4 * 3000), .ok)
        let cramStudy = crammer.network.meanMinutesOff(indices: pile, in: l)
        let cramExam = crammer.network.meanMinutesOff(.exam, in: l)
        let study = net.meanMinutesOff(.practice, in: l), exam = net.meanMinutesOff(.exam, in: l)
        XCTAssertLessThan(cramStudy, study - 1, "the crammer's homework looks better")
        XCTAssertGreaterThan(cramExam, exam + 3, "and its exam is much worse")
        XCTAssertLessThan(abs(exam - study), 1, "the lesson network does about as well on the exam as at home")
        // a study pile never reaches outside the practice years
        let leaky = Trainer(network: net, ledger: l, studyPile: [l.indices(.exam)[0], l.indices(.check)[0]])
        _ = leaky.train(batches: 1)
        XCTAssertEqual(leaky.flightsSeen, 0)
        print(String(format: "exam day: crammer homework %.2f exam %.2f; lesson network homework %.2f exam %.2f",
                     cramStudy, cramExam, study, exam))
    }
}
