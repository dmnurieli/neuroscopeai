// The lesson helpers behind Contract.swift's "Lesson helpers" section: firing rules, the lesson flights, and the
// foggy mountain (a two-route slice of the real training landscape) with its hikes.
//
// Mountain arithmetic goes through NetCore, so the slice's centre cell is the network's own meanMinutesOff.
// Speed: everything upstream of the routes' layer is computed once per sample flight; each grid cell re-runs
// only the layers the two routes can change.

import Foundation

// MARK: - Rulebook and ledger lookups

extension Rulebook {
    func _firingRules(flight i: Int, in ledger: Ledger) -> [Int] {
        rules.indices.filter { r in
            rules[r].conditions.allSatisfy { c in
                let v = ledger.fact(c.fact, i)
                return c.above ? v > c.threshold : v < c.threshold
            }
        }
    }
}

extension Ledger {
    func _firstFlight(in split: Split, where predicate: (Int) -> Bool) -> Int? {
        indices(split).first(where: predicate)
    }

    /// Largest delay among CHECK flights passing `predicate`; ties → lowest index.
    private func largestDelayCheckFlight(where predicate: (Int) -> Bool) -> Int? {
        var best: Int?, bestDelay = -Float.infinity
        for i in indices(.check) where predicate(i) && delay(i) > bestDelay { best = i; bestDelay = delay(i) }
        return best
    }

    /// Wind closest to (but not above) 30 mph wins, so the story is "29 mph: your 30-mph rule stayed quiet";
    /// ties → larger delay, then lower index.
    var _nearMissFlight: Int? {
        var best: Int?
        for i in indices(.check) {
            let w = fact(.wind, i)
            guard w >= 25, w < 29.5, fact(.rain, i) >= 1, fact(.controllers, i) <= 8, delay(i) >= 90 else { continue }
            if let b = best {
                let wb = fact(.wind, b)
                if w > wb || (w == wb && delay(i) > delay(b)) { best = i }
            } else { best = i }
        }
        return best
    }

    func _closestMiss(flight i: Int, of book: Rulebook) -> (rule: Int, condition: Condition, gap: Float)? {
        var best: (rule: Int, condition: Condition, gap: Float)?
        for (r, rule) in book.rules.enumerated() {
            for c in rule.conditions {
                let v = fact(c.fact, i)
                guard !(c.above ? v > c.threshold : v < c.threshold) else { continue }
                let gap = abs(v - c.threshold) / max(factSD[c.fact.rawValue], 1e-6)
                if best == nil || gap < best!.gap { best = (r, c, gap) }
            }
        }
        return best
    }

    func _nearMissFlight(for book: Rulebook) -> Int? {
        if let i = _nearMissFlight, book.firingRules(flight: i, in: self).isEmpty { return i }
        var best: (i: Int, gap: Float)?
        for i in indices(.check) where fact(.rain, i) >= 1 && fact(.controllers, i) <= 8 && delay(i) >= 90 {
            guard book.firingRules(flight: i, in: self).isEmpty, let m = _closestMiss(flight: i, of: book) else { continue }
            if let b = best, m.gap > b.gap || (m.gap == b.gap && delay(i) <= delay(b.i)) { continue }
            best = (i, m.gap)
        }
        return best?.i ?? _nearMissFlight
    }

    var _compoundFlight: Int? {
        largestDelayCheckFlight { i in concept(.perfectStorm, i) >= 0.7 }
    }
}

// MARK: - The foggy mountain

extension NetCore {
    /// Like `forward`, but starts at layer `l`: a[aOff[l]…] must already hold that layer's activations.
    @inline(__always) func forward(fromLayer start: Int) -> Float {
        for l in start..<layers {
            let nin = sizes[l], nout = sizes[l + 1]
            let al = a + aOff[l], an = a + aOff[l + 1], w = p + wOff[l], b = p + bOff[l]
            let last = l == layers - 1
            for j in 0..<nout {
                var s = b[j]
                for i in 0..<nin { s += al[i] * w[i * nout + j] }
                an[j] = last ? s : max(0, s)
            }
        }
        return a[aOff[layers]]
    }

    /// Flat parameter index of a route.
    func index(of r: RouteID) -> Int {
        precondition(r.layer >= 0 && r.layer < layers && r.from >= 0 && r.from < sizes[r.layer]
                     && r.to >= 0 && r.to < sizes[r.layer + 1], "route \(r) is not in this network")
        return wOff[r.layer] + r.from * sizes[r.layer + 1] + r.to
    }
}

extension Network {
    /// `count` practice flights from a SplitMix64(seed) shuffle of the practice years.
    func mountainSample(count: Int, seed: UInt64, in ledger: Ledger) -> [Int] {
        var order = ledger.indices(.practice)
        var rng = SplitMix64(seed: seed)
        rng.shuffle(&order)
        return Array(order.prefix(max(0, min(count, order.count))))
    }

    func _mountain(a: RouteID, b: RouteID, span: Float, resolution: Int, sampleCount: Int, seed: UInt64,
                   in ledger: Ledger) -> MountainSlice {
        let res = max(resolution, 1)
        let core = NetCore(self)
        let ka = core.index(of: a), kb = core.index(of: b)
        let a0 = core.p[ka], b0 = core.p[kb]
        // Cell k sits at centre + span·(2k/(res−1) − 1), so the middle cell of an odd grid is exactly the centre.
        func values(_ c: Float) -> [Float] {
            res == 1 ? [c] : (0..<res).map { c + span * (Float(2 * $0) / Float(res - 1) - 1) }
        }
        let aVals = values(a0), bVals = values(b0)
        let sample = mountainSample(count: sampleCount, seed: seed, in: ledger)
        let n = sample.count

        // Cache, per sample flight: layer L's activations (L = the earlier route's layer; nothing upstream moves)
        // and layer L+1's tally with the layer-L routes among a/b taken out.
        let L = min(a.layer, b.layer), nL = core.sizes[L], nN = core.sizes[L + 1]
        let aInL = a.layer == L, bInL = b.layer == L
        if aInL { core.p[ka] = 0 }
        if bInL { core.p[kb] = 0 }
        let cols = inputColumns
        let x = UnsafeMutablePointer<Float>.allocate(capacity: max(cols.count, 1))
        let actL = UnsafeMutablePointer<Float>.allocate(capacity: max(n * nL, 1))
        let base = UnsafeMutablePointer<Float>.allocate(capacity: max(n * nN, 1))
        let truth = UnsafeMutablePointer<Float>.allocate(capacity: max(n, 1))
        defer { x.deallocate(); actL.deallocate(); base.deallocate(); truth.deallocate() }
        let wL = core.p + core.wOff[L], bL = core.p + core.bOff[L]
        for (s, i) in sample.enumerated() {
            standardise(flight: i, in: ledger, columns: cols, into: x)
            _ = core.forward(x)
            let src = core.a + core.aOff[L], dst = actL + s * nL, z = base + s * nN
            dst.update(from: src, count: nL)
            for j in 0..<nN {
                var t = bL[j]
                for k in 0..<nL { t += dst[k] * wL[k * nN + j] }
                z[j] = t
            }
            truth[s] = ledger.delay(i)
        }

        let outputNext = L + 1 == core.layers
        let next = core.a + core.aOff[L + 1]
        let (sd, mean) = (delaySD, delayMean)
        var loss = [Float](repeating: 0, count: res * res)
        for (jb, vb) in bVals.enumerated() {
            for (ia, va) in aVals.enumerated() {
                if !aInL { core.p[ka] = va }
                if !bInL { core.p[kb] = vb }   // the same route on both axes: the y value wins
                var total = 0.0
                for s in 0..<n {
                    let z = base + s * nN, act = actL + s * nL
                    for j in 0..<nN { next[j] = z[j] }
                    if aInL && a != b { next[a.to] += act[a.from] * va }
                    if bInL { next[b.to] += act[b.from] * vb }
                    let out: Float
                    if outputNext {
                        out = next[0]
                    } else {
                        for j in 0..<nN { next[j] = max(0, next[j]) }
                        out = core.forward(fromLayer: L + 1)
                    }
                    total += Double(abs(out * sd + mean - truth[s]))
                }
                loss[jb * res + ia] = n > 0 ? Float(total / Double(n)) : 0
            }
        }
        return MountainSlice(routeA: a, routeB: b,
                             aRange: aVals.first!...aVals.last!, bRange: bVals.first!...bVals.last!,
                             resolution: res, loss: loss,
                             minLoss: loss.min() ?? 0, maxLoss: loss.max() ?? 0, sample: sample)
    }

    func _hike(on slice: MountainSlice, from start: SIMD2<Float>, stepSize: Float, steps: Int, seed: UInt64,
               in ledger: Ledger) -> MountainHike {
        // The slice's own sample; `seed` only rebuilds one (2,000 flights) for a slice that carries none.
        let sample = slice.sample.isEmpty ? mountainSample(count: 2_000, seed: seed, in: ledger) : slice.sample
        let core = NetCore(self)
        let ka = core.index(of: slice.routeA), kb = core.index(of: slice.routeB)
        let cols = inputColumns, nIn = cols.count, n = sample.count
        let xs = UnsafeMutablePointer<Float>.allocate(capacity: max(n * nIn, 1))
        let ys = UnsafeMutablePointer<Float>.allocate(capacity: max(n, 1))
        let minutesLate = sample.map { ledger.delay($0) }
        defer { xs.deallocate(); ys.deallocate() }
        for (s, i) in sample.enumerated() {
            standardise(flight: i, in: ledger, columns: cols, into: xs + s * nIn)
            ys[s] = standardisedTarget(flight: i, in: ledger)
        }
        let (sd, mean) = (delaySD, delayMean)
        let scale = 1 / Float(max(n, 1))
        // Minutes off at `pt`, and (if asked) the training loss's gradient on the two routes there.
        func evaluate(_ pt: SIMD2<Float>, gradient: Bool) -> (minutes: Float, grad: SIMD2<Float>) {
            core.p[ka] = pt.x
            core.p[kb] = pt.y
            core.zeroGradient()
            var total = 0.0
            for s in 0..<n {
                let out = core.forward(xs + s * nIn)
                total += Double(abs(out * sd + mean - minutesLate[s]))   // as the mountain measures it
                if gradient { _ = core.accumulate(target: ys[s], scale: scale) }
            }
            return (n > 0 ? Float(total / Double(n)) : 0, SIMD2(core.g[ka], core.g[kb]))
        }
        let spanA = (slice.aRange.upperBound - slice.aRange.lowerBound) / 2
        let spanB = (slice.bRange.upperBound - slice.bRange.lowerBound) / 2
        func outside(_ pt: SIMD2<Float>) -> Bool {
            pt.x < slice.aRange.lowerBound - spanA || pt.x > slice.aRange.upperBound + spanA
                || pt.y < slice.bRange.lowerBound - spanB || pt.y > slice.bRange.upperBound + spanB
        }
        func finite(_ v: SIMD2<Float>) -> Bool { v.x.isFinite && v.y.isFinite }

        var pt = start
        guard finite(pt) else { return MountainHike(points: [pt], losses: [.nan], tumbled: true) }
        var (minutes, grad) = evaluate(pt, gradient: steps > 0)
        var points = [pt], losses = [minutes], tumbled = outside(pt) || !minutes.isFinite
        var taken = 0
        while !tumbled && taken < steps {
            let next = pt - stepSize * grad
            taken += 1
            guard finite(next) else { tumbled = true; break }
            pt = next
            let away = outside(pt)
            (minutes, grad) = evaluate(pt, gradient: !away && taken < steps)
            points.append(pt); losses.append(minutes)
            if away || !minutes.isFinite || !finite(grad) { tumbled = true }
        }
        return MountainHike(points: points, losses: losses, tumbled: tumbled)
    }

    func _mountainRoutes(in ledger: Ledger) -> (RouteID, RouteID)? {
        guard let hubs = architecture.hidden.first, hubs > 0, architecture.inputs.count >= 2 else { return nil }
        let core = NetCore(self), cols = inputColumns
        let x = UnsafeMutablePointer<Float>.allocate(capacity: max(cols.count, 1))
        defer { x.deallocate() }
        // Streaming signed Pearson r of each row-1 hub with Concept.storm over the check year.
        var sx = [Double](repeating: 0, count: hubs), sxx = sx, sxy = sx
        var sy = 0.0, syy = 0.0
        let idx = ledger.indices(.check)
        let row1 = core.a + core.aOff[1]
        for i in idx {
            standardise(flight: i, in: ledger, columns: cols, into: x)
            _ = core.forward(x)
            let y = Double(ledger.concept(.storm, i))
            sy += y; syy += y * y
            for h in 0..<hubs {
                let v = Double(row1[h])
                sx[h] += v; sxx[h] += v * v; sxy[h] += v * y
            }
        }
        let n = Double(max(idx.count, 1))
        var bestHub = 0, bestR = -Double.infinity
        for h in 0..<hubs {
            let cov = sxy[h] - sx[h] * sy / n, vx = sxx[h] - sx[h] * sx[h] / n, vy = syy - sy * sy / n
            let r = vx > 0 && vy > 0 ? cov / (vx * vy).squareRoot() : 0
            if r > bestR { bestR = r; bestHub = h }
        }
        let routes = (0..<architecture.inputs.count).map { RouteID(layer: 0, from: $0, to: bestHub) }
        let ranked = routes.enumerated().sorted { l, r in
            let (wl, wr) = (abs(weights[0][l.element.from * hubs + bestHub]), abs(weights[0][r.element.from * hubs + bestHub]))
            return wl != wr ? wl > wr : l.offset < r.offset
        }.map(\.element)
        return (ranked[0], ranked[1])
    }
}
