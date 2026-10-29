// The synthetic flight ledger, drawn from a seeded SplitMix64 generator so every Mac sees the same 120,000 flights.
//
// The hidden rulebook: weather SYSTEMS make stormy days wet AND windy, while drizzly-calm and dry-breezy days
// happen on their own. Delay comes from the conjunctions (storm = rain AND wind, holiday rush = holiday AND a
// full plane, perfect storm = storm while short-staffed, worse in a rush), plus evening congestion, minus the
// relief spare crews bring, plus noise (σ 5 min), clamped to −15…300.

import Foundation

// MARK: - Draws (every random number in AirlineKit comes through here)

extension SplitMix64 {
    /// Uniform in [0, 1), 53 bits.
    @inline(__always) mutating func unit() -> Double { Double(next() >> 11) * 0x1p-53 }
    @inline(__always) mutating func uniform(_ lo: Double, _ hi: Double) -> Double { lo + (hi - lo) * unit() }
    /// Integer in lo…hi inclusive (modulo bias is below 1e-17 for these ranges).
    @inline(__always) mutating func int(_ lo: Int, _ hi: Int) -> Int { lo + Int(next() % UInt64(hi - lo + 1)) }
    @inline(__always) mutating func chance(_ p: Double) -> Bool { unit() < p }
    /// Exponential with the given mean, by inversion: −log(1 − u).
    @inline(__always) mutating func exponential(_ mean: Double) -> Double { -log(1 - unit()) * mean }
    /// Gamma(2, θ) as the sum of two exponentials with mean θ.
    @inline(__always) mutating func gamma2(_ theta: Double) -> Double { exponential(theta) + exponential(theta) }
    /// Normal by Box–Muller; one of the pair is used so every call costs exactly two draws.
    @inline(__always) mutating func normal(_ mean: Double, _ sd: Double) -> Double {
        let u1 = 1 - unit(), u2 = unit()
        return mean + sd * (-2 * log(u1)).squareRoot() * cos(2 * Double.pi * u2)
    }
    /// Fisher–Yates, in place.
    mutating func shuffle<T>(_ a: inout [T]) {
        guard a.count > 1 else { return }
        for k in stride(from: a.count - 1, to: 0, by: -1) {
            a.swapAt(k, Int(next() % UInt64(k + 1)))
        }
    }
}

// MARK: - The world's fixed scenery

enum World {
    static let factCount = Fact.allCases.count
    static let conceptCount = Concept.allCases.count

    static let airports = ["MIA", "JFK", "BOS", "LAX", "SFO", "LAS", "MCO", "IAD", "SAN", "AUS"]
    private static let coordinates: [(lat: Double, lon: Double)] = [
        (25.79, -80.29), (40.64, -73.78), (42.36, -71.01), (33.94, -118.41), (37.62, -122.38),
        (36.08, -115.15), (28.43, -81.31), (38.95, -77.46), (32.73, -117.19), (30.19, -97.67),
    ]
    /// Scheduled block time per city pair: 35 min of taxi and climb plus ~450 mph cruise, to the nearest 5.
    static let blockMinutes: [Int] = {
        var t = [Int](repeating: 0, count: 100)
        for a in 0..<10 { for b in 0..<10 where a != b {
            let (p, q) = (coordinates[a], coordinates[b])
            let r = Double.pi / 180
            let h = pow(sin((q.lat - p.lat) * r / 2), 2) + cos(p.lat * r) * cos(q.lat * r) * pow(sin((q.lon - p.lon) * r / 2), 2)
            let miles = 2 * 3958.8 * asin(h.squareRoot())
            t[a * 10 + b] = Int(((35 + miles / 7.5) / 5).rounded()) * 5
        } }
        return t
    }()
    /// A few aircraft sizes: regional jet, A220, 737-800, A321, 757.
    static let aircraftSeats = [76, 130, 160, 190, 200]
    /// Eight holiday weeks (New Year, Presidents' Day, spring break, Memorial Day, July 4th, Labor Day,
    /// Thanksgiving, Christmas): a holiday flight's date falls in one of them, an ordinary flight's never does.
    static let holidayDays: [Int] = [1, 46, 72, 144, 182, 244, 326, 355].flatMap { $0..<($0 + 7) }
    static let ordinaryDays: [Int] = { let h = Set(holidayDays); return (1...365).filter { !h.contains($0) } }()

    @inline(__always) static func sigmoid(_ x: Double) -> Double { 1 / (1 + exp(-x)) }

    /// Builds every column. Flights are in year order (year = i·10 / count), so splits are contiguous ranges.
    static func generate(seed: UInt64, count n: Int) -> WorldColumns {
        var c = WorldColumns(count: n)
        var rng = SplitMix64(seed: seed)
        let ordinary = ordinaryDays, holiday = holidayDays, seatsBySize = aircraftSeats, block = blockMinutes
        for i in 0..<n {
            // Weather: one system draw makes the day wet AND windy; drizzle and breeze come alone.
            let system = rng.chance(0.12)
            let showers = rng.exponential(0.35), showering = rng.chance(0.4), systemRain = rng.uniform(0.8, 2.5)
            var rain = (showering ? showers : 0) + (system ? systemRain : 0)
            var wind = rng.gamma2(6)
            let systemWind = rng.uniform(12, 30)
            if system { wind += systemWind }
            let drizzle = rng.chance(0.15), drizzleRain = rng.uniform(0.6, 2.0)
            if !system && drizzle { rain += drizzleRain }
            let breezy = rng.chance(0.15), breezeWind = rng.uniform(12, 30)
            if !system && breezy { wind += breezeWind }

            let controllers = Double(rng.int(6, 14))
            let seats = seatsBySize[rng.int(0, seatsBySize.count - 1)]
            let wantedLoad = min(max(rng.normal(0.82, 0.1), 0.4), 1.0)
            let passengers = Int((wantedLoad * Double(seats)).rounded())
            let load = Double(passengers) / Double(seats)
            let isHoliday = rng.chance(0.15)
            let hour = rng.int(6, 22), minute = 5 * rng.int(0, 11)
            let crews = Double(rng.int(0, 6))
            let shoe = Double(rng.int(0, 4))
            let origin = rng.int(0, 9)
            var destination = rng.int(0, 8)
            if destination >= origin { destination += 1 }
            let days = isHoliday ? holiday : ordinary
            let day = days[rng.int(0, days.count - 1)]

            // The hidden rulebook.
            let hol: Double = isHoliday ? 1 : 0
            let storm = sigmoid((rain - 0.6) * 8) * sigmoid((wind - 22) / 2.5)
            let short = sigmoid((9 - controllers) * 1.5)
            let rush = hol * sigmoid((load - 0.88) * 30)
            let perfect = storm * short * (0.5 + rush)
            var delay = 5 + 55 * storm + 18 * short + 40 * rush + 90 * perfect
                + 0.6 * max(0, Double(hour) - 15) - 3 * crews * (storm + short + 0.2) + rng.normal(0, 5)
            delay = min(max(delay, -15), 300)

            let f = i * factCount
            c.facts[f + Fact.rain.rawValue] = Float(rain)
            c.facts[f + Fact.wind.rawValue] = Float(wind)
            c.facts[f + Fact.controllers.rawValue] = Float(controllers)
            c.facts[f + Fact.load.rawValue] = Float(load)
            c.facts[f + Fact.holiday.rawValue] = Float(hol)
            c.facts[f + Fact.hour.rawValue] = Float(hour)
            c.facts[f + Fact.crews.rawValue] = Float(crews)
            c.facts[f + Fact.shoe.rawValue] = Float(shoe)
            let k = i * conceptCount
            c.concepts[k + Concept.storm.rawValue] = Float(storm)
            c.concepts[k + Concept.shortStaff.rawValue] = Float(short)
            c.concepts[k + Concept.holidayRush.rawValue] = Float(rush)
            c.concepts[k + Concept.perfectStorm.rawValue] = Float(perfect)
            c.delays[i] = Float(delay)
            c.years[i] = UInt8(i * 10 / n)
            c.days[i] = UInt16(day)
            c.origins[i] = UInt8(origin)
            c.destinations[i] = UInt8(destination)
            let departure = hour * 60 + minute
            c.scheduledDeparture[i] = Int32(departure)
            c.scheduledArrival[i] = Int32(departure + block[origin * 10 + destination])
            c.seats[i] = Int16(seats)
            c.passengers[i] = Int16(passengers)
        }
        return c
    }
}

/// The ledger's storage: flat, flight-major columns.
struct WorldColumns: Sendable {
    let count: Int
    var facts: [Float]            // count × Fact.count
    var concepts: [Float]         // count × Concept.count
    var delays: [Float]
    var years: [UInt8]
    var days: [UInt16]
    var origins: [UInt8]
    var destinations: [UInt8]
    var scheduledDeparture: [Int32]
    var scheduledArrival: [Int32]
    var seats: [Int16]
    var passengers: [Int16]

    init(count n: Int) {
        count = n
        facts = [Float](repeating: 0, count: n * World.factCount)
        concepts = [Float](repeating: 0, count: n * World.conceptCount)
        delays = [Float](repeating: 0, count: n)
        years = [UInt8](repeating: 0, count: n)
        days = [UInt16](repeating: 0, count: n)
        origins = [UInt8](repeating: 0, count: n)
        destinations = [UInt8](repeating: 0, count: n)
        scheduledDeparture = [Int32](repeating: 0, count: n)
        scheduledArrival = [Int32](repeating: 0, count: n)
        seats = [Int16](repeating: 0, count: n)
        passengers = [Int16](repeating: 0, count: n)
    }

    /// Practice = years 0…6, check = 7, exam = 8…9.
    func splitIndices() -> (practice: [Int], check: [Int], exam: [Int]) {
        var p: [Int] = [], c: [Int] = [], e: [Int] = []
        for i in 0..<count {
            switch years[i] {
            case 0...6: p.append(i)
            case 7: c.append(i)
            default: e.append(i)
            }
        }
        return (p, c, e)
    }

    /// Population mean and standard deviation of every fact and of delay over the given flights.
    /// A constant column gets SD 1 so standardising never divides by zero.
    func stats(over idx: [Int]) -> WorldStats {
        let m = World.factCount
        var sum = [Double](repeating: 0, count: m), sq = [Double](repeating: 0, count: m)
        var dSum = 0.0, dSq = 0.0
        for i in idx {
            for f in 0..<m { let v = Double(facts[i * m + f]); sum[f] += v; sq[f] += v * v }
            let d = Double(delays[i]); dSum += d; dSq += d * d
        }
        let n = Double(max(idx.count, 1))
        func sd(_ s: Double, _ q: Double) -> Float {
            let v = max(q / n - (s / n) * (s / n), 0).squareRoot()
            return v > 1e-9 ? Float(v) : 1
        }
        return WorldStats(factMean: sum.map { Float($0 / n) }, factSD: (0..<m).map { sd(sum[$0], sq[$0]) },
                          delayMean: Float(dSum / n), delaySD: sd(dSum, dSq))
    }

    func concept(_ c: Concept, _ i: Int) -> Float { concepts[i * World.conceptCount + c.rawValue] }

    /// Largest perfect-storm value among the given flights, ties → lowest index.
    func showcase(in idx: [Int]) -> Int {
        var best = idx.first ?? 0, bestValue = -Float.infinity
        for i in idx where concept(.perfectStorm, i) > bestValue { best = i; bestValue = concept(.perfectStorm, i) }
        return best
    }

    /// First flight with storm ≥ 0.8; if none, the stormiest one (ties → lowest index).
    func firstStorm(in idx: [Int]) -> Int {
        if let i = idx.first(where: { concept(.storm, $0) >= 0.8 }) { return i }
        var best = idx.first ?? 0, bestValue = -Float.infinity
        for i in idx where concept(.storm, i) > bestValue { best = i; bestValue = concept(.storm, i) }
        return best
    }
}

struct WorldStats: Sendable {
    var factMean: [Float], factSD: [Float], delayMean: Float, delaySD: Float
}
