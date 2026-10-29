// Pure functions over Runs. No UI, no engine calls — unit-testable.

import Accelerate
import Foundation
import simd

enum Analysis {
    /// Approximate k-th smallest (0-based) via quickselect; mutates a.
    private static func select(_ a: inout [Float], _ k: Int) -> Float {
        var lo = 0, hi = a.count - 1
        while lo < hi {
            let pivot = a[(lo + hi) / 2]
            var i = lo, j = hi
            while i <= j {
                while a[i] < pivot { i += 1 }
                while a[j] > pivot { j -= 1 }
                if i <= j { a.swapAt(i, j); i += 1; j -= 1 }
            }
            if k <= j { hi = j } else if k >= i { lo = i } else { break }
        }
        return a[k]
    }

    private static func pct(_ a: inout [Float]) -> Float {
        if a.isEmpty { return 1e-3 }
        let k = min(a.count - 1, Int(Double(a.count - 1) * 0.995))
        return max(select(&a, k), 1e-3)
    }

    /// Per-layer brightness normalizer: a robust high percentile (≈99.5th) of |activation| over the
    /// given tokens of the run, floored to avoid division blow-ups. Length numLayers.
    static func layerScales(_ run: Run, tokens: Range<Int>? = nil) -> [Float] {
        let g = run.geometry
        let r = (tokens ?? 0..<run.count).clamped(to: 0..<run.count)
        guard !r.isEmpty else { return [Float](repeating: 1e-3, count: g.numLayers) }
        let stride = max(1, (r.count * g.numNeurons + 59_999) / 60_000)
        var bufs = [[Float]](repeating: [], count: g.numLayers)
        for l in 0..<g.numLayers { bufs[l].reserveCapacity(r.count * g.numNeurons / stride + 1) }
        for t in r {
            run.withNeurons(at: t) { p in
                for l in 0..<g.numLayers {
                    var j = 0
                    let base = l * g.numNeurons
                    while j < g.numNeurons {
                        bufs[l].append(abs(Float(p[base + j])))
                        j += stride
                    }
                }
            }
        }
        return (0..<g.numLayers).map { pct(&bufs[$0]) }
    }

    /// Per-layer normalizer for head norms (same idea), length numLayers.
    static func headScales(_ run: Run) -> [Float] {
        let g = run.geometry
        guard run.count > 0 else { return [Float](repeating: 1e-3, count: g.numLayers) }
        var out = [Float]()
        for l in 0..<g.numLayers {
            var v = [Float]()
            v.reserveCapacity(run.count * g.numHeads)
            for t in 0..<run.count {
                run.withHeads(at: t) { p in
                    for h in 0..<g.numHeads { v.append(abs(p[l * g.numHeads + h])) }
                }
            }
            out.append(pct(&v))
        }
        return out
    }

    /// Keeps the k largest (value, key) pairs, descending.
    private static func topK<K>(_ k: Int, _ feed: ((K, Float) -> Void) -> Void) -> [(K, Float)] {
        guard k > 0 else { return [] }
        var heap = [(K, Float)]()  // min-heap by value
        heap.reserveCapacity(k)
        func down(_ start: Int) {
            var i = start
            while true {
                let l = 2 * i + 1, r = l + 1
                var m = i
                if l < heap.count && heap[l].1 < heap[m].1 { m = l }
                if r < heap.count && heap[r].1 < heap[m].1 { m = r }
                if m == i { return }
                heap.swapAt(i, m); i = m
            }
        }
        feed { key, v in
            if heap.count < k {
                heap.append((key, v))
                var i = heap.count - 1
                while i > 0 {
                    let p = (i - 1) / 2
                    if heap[i].1 < heap[p].1 { heap.swapAt(i, p); i = p } else { break }
                }
            } else if v > heap[0].1 {
                heap[0] = (key, v); down(0)
            }
        }
        return heap.sorted { $0.1 > $1.1 }
    }

    /// The k tokens of `run` where neuron `n` fired hardest (by |activation|), descending.
    static func topTokens(for n: NeuronID, in run: Run, k: Int) -> [(t: Int, value: Float)] {
        let res: [(Int, Float)] = topK(k) { emit in
            for t in 0..<run.count {
                let v = run.activation(n, at: t)
                emit(t, abs(v))
            }
        }
        return res.map { (t: $0.0, value: run.activation(n, at: $0.0)) }
    }

    /// The k most active neurons at token t (by |activation| / layer scale), descending.
    static func topNeurons(in run: Run, at t: Int, k: Int, scales: [Float]) -> [(id: NeuronID, value: Float)] {
        guard t >= 0, t < run.count else { return [] }
        let g = run.geometry
        let res: [(Int, Float)] = run.withNeurons(at: t) { p in
            topK(k) { emit in
                for l in 0..<g.numLayers {
                    let inv = 1 / max(l < scales.count ? scales[l] : 1, 1e-6)
                    let base = l * g.numNeurons
                    for j in 0..<g.numNeurons {
                        emit(base + j, abs(Float(p[base + j])) * inv)
                    }
                }
            }
        }
        return res.map { (id: NeuronID(layer: $0.0 / g.numNeurons, index: $0.0 % g.numNeurons), value: $0.1) }
    }

    /// First generated position where the two runs chose different tokens; nil if identical up to
    /// the shorter length (and same length).
    static func firstDivergence(_ a: Run, _ b: Run) -> Int? {
        let common = min(a.count, b.count)
        let start = min(a.promptCount, common)
        var i = start
        while i < common {
            if a.tokens[i].tokenID != b.tokens[i].tokenID { return i }
            i += 1
        }
        return a.count == b.count ? nil : common
    }

    /// Writes (edited − baseline) at token t into `out` (length neuronsPerFrame). Positions past
    /// either run's end write 0.
    static func diff(_ edited: Run, _ baseline: Run, at t: Int, into out: inout [Float]) {
        let n = edited.geometry.neuronsPerFrame
        if out.count != n { out = [Float](repeating: 0, count: n) }
        guard t >= 0, t < edited.count, t < baseline.count, baseline.geometry.neuronsPerFrame == n else {
            out.withUnsafeMutableBufferPointer { $0.update(repeating: 0) }
            return
        }
        edited.withNeurons(at: t) { e in
            baseline.withNeurons(at: t) { b in
                out.withUnsafeMutableBufferPointer { o in
                    for i in 0..<n { o[i] = Float(e[i]) - Float(b[i]) }
                }
            }
        }
    }

    /// Mean |edited − baseline| per layer at token t — how far each layer moved. Length numLayers.
    static func layerShift(_ edited: Run, _ baseline: Run, at t: Int) -> [Float] {
        let g = edited.geometry
        var out = [Float](repeating: 0, count: g.numLayers)
        guard t >= 0, t < edited.count, t < baseline.count else { return out }
        edited.withNeurons(at: t) { e in
            baseline.withNeurons(at: t) { b in
                for l in 0..<g.numLayers {
                    var s: Float = 0
                    let base = l * g.numNeurons
                    for j in 0..<g.numNeurons { s += abs(Float(e[base + j]) - Float(b[base + j])) }
                    out[l] = s / Float(g.numNeurons)
                }
            }
        }
        return out
    }

    /// Functional layout: within each layer, place neurons so that ones that co-activate across the run's
    /// tokens sit together (spherical k-means on z-scored activation profiles, clusters placed as
    /// non-overlapping islands, members on a sunflower spiral inside each island), returning per layer, per
    /// neuron, (x, y) in [-1, 1]². Dead (near-constant) neurons go to faint outer rings.
    static func clusterLayout(_ run: Run) -> [[SIMD2<Float>]] {
        clusterLayout(geometry: run.geometry, count: run.count, neurons: run.neurons)
    }

    /// Same, over a snapshot of a run's flat activation buffer (ContiguousArray is copy-on-write, so
    /// passing `run.neurons` to a detached task is a cheap, race-free snapshot).
    static func clusterLayout(geometry g: Geometry, count: Int,
                              neurons: ContiguousArray<Float16>) -> [[SIMD2<Float>]] {
        let L = g.numLayers, N = g.numNeurons
        guard count > 0, neurons.count >= count * g.neuronsPerFrame else {
            return [[SIMD2<Float>]](repeating: gridFallback(N), count: L)
        }
        let T = min(count, 256)
        let tokens = (0..<T).map { T == count ? $0 : $0 * count / T }
        var out = [[SIMD2<Float>]](repeating: [], count: L)
        let lock = NSLock()
        neurons.withUnsafeBufferPointer { src in
            let base = src
            DispatchQueue.concurrentPerform(iterations: L) { l in
                let pts = layerLayout(layer: l, g: g, tokens: tokens, src: base)
                lock.lock(); out[l] = pts; lock.unlock()
            }
        }
        return out
    }

    private static func gridFallback(_ n: Int) -> [SIMD2<Float>] {
        let w = max(1, Int(Double(n).squareRoot().rounded(.up)))
        return (0..<n).map { i in
            SIMD2<Float>(Float(i % w) / Float(max(w - 1, 1)) * 2 - 1, Float(i / w) / Float(max(w - 1, 1)) * 2 - 1)
        }
    }

    private struct LCG {
        var s: UInt64
        mutating func next() -> Double {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return Double(s >> 11) / Double(1 << 53)
        }
    }

    private static func layerLayout(layer l: Int, g: Geometry, tokens: [Int],
                                    src: UnsafeBufferPointer<Float16>) -> [SIMD2<Float>] {
        let N = g.numNeurons, T = tokens.count, npf = g.neuronsPerFrame
        // Profiles: row j = neuron j across the sampled tokens.
        var X = [Float](repeating: 0, count: N * T)
        X.withUnsafeMutableBufferPointer { x in
            for (ti, t) in tokens.enumerated() {
                let row = src.baseAddress! + t * npf + l * N
                for j in 0..<N { x[j * T + ti] = Float(row[j]) }
            }
        }
        // Center (when there is enough data to center), measure, unit-normalize.
        let center = T >= 4
        var norm = [Float](repeating: 0, count: N)
        X.withUnsafeMutableBufferPointer { x in
            for j in 0..<N {
                let r = x.baseAddress! + j * T
                if center {
                    var m: Float = 0
                    vDSP_meanv(r, 1, &m, vDSP_Length(T))
                    var nm = -m
                    vDSP_vsadd(r, 1, &nm, r, 1, vDSP_Length(T))
                }
                var ss: Float = 0
                vDSP_svesq(r, 1, &ss, vDSP_Length(T))
                norm[j] = ss.squareRoot()
            }
        }
        let meanNorm = norm.reduce(0, +) / Float(N)
        let thr = max(1e-6, 0.01 * meanNorm)
        var live = [Int](), dead = [Int]()
        for j in 0..<N { if norm[j] > thr { live.append(j) } else { dead.append(j) } }
        let n = live.count
        var P = [Float](repeating: 0, count: max(n, 1) * T)   // live profiles, unit rows
        for (i, j) in live.enumerated() {
            var inv = 1 / norm[j]
            X.withUnsafeBufferPointer { x in
                P.withUnsafeMutableBufferPointer { p in
                    vDSP_vsmul(x.baseAddress! + j * T, 1, &inv, p.baseAddress! + i * T, 1, vDSP_Length(T))
                }
            }
        }

        var assign = [Int](repeating: 0, count: n)
        var sim = [Float](repeating: 0, count: n)   // similarity to own center
        var k = 0
        if n > 0 {
            k = max(1, min(24, n / 8))
            var rng = LCG(s: 0x9E3779B97F4A7C15 &+ UInt64(l))
            var C = [Float](repeating: 0, count: k * T)
            // k-means++ seeding on cosine distance.
            var minDist = [Float](repeating: 1, count: n)
            var first = Int(rng.next() * Double(n)) % n
            for c in 0..<k {
                if c > 0 {
                    let total = minDist.reduce(0) { $0 + $1 * $1 }
                    var pick = rng.next() * Double(total), acc = 0.0
                    first = n - 1
                    for i in 0..<n { acc += Double(minDist[i] * minDist[i]); if acc >= pick { first = i; break } }
                    pick = 0
                }
                for t in 0..<T { C[c * T + t] = P[first * T + t] }
                for i in 0..<n {
                    var d: Float = 0
                    P.withUnsafeBufferPointer { p in
                        C.withUnsafeBufferPointer { cc in
                            vDSP_dotpr(p.baseAddress! + i * T, 1, cc.baseAddress! + c * T, 1, &d, vDSP_Length(T))
                        }
                    }
                    minDist[i] = min(minDist[i], max(0, 1 - d))
                }
            }
            var S = [Float](repeating: 0, count: n * k)
            for _ in 0..<10 {
                P.withUnsafeBufferPointer { p in
                    C.withUnsafeBufferPointer { cc in
                        S.withUnsafeMutableBufferPointer { s in
                            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(n), Int32(k), Int32(T),
                                        1, p.baseAddress!, Int32(T), cc.baseAddress!, Int32(T), 0, s.baseAddress!, Int32(k))
                        }
                    }
                }
                var changed = 0
                for i in 0..<n {
                    var bi = 0, bv = -Float.infinity
                    for c in 0..<k { let v = S[i * k + c]; if v > bv { bv = v; bi = c } }
                    if assign[i] != bi { changed += 1; assign[i] = bi }
                    sim[i] = bv
                }
                // Update centers (sum members, renormalize); reseed empties with the worst-fit neuron.
                var counts = [Int](repeating: 0, count: k)
                for c in 0..<(k * T) { C[c] = 0 }
                for i in 0..<n {
                    let c = assign[i]; counts[c] += 1
                    for t in 0..<T { C[c * T + t] += P[i * T + t] }
                }
                for c in 0..<k {
                    if counts[c] == 0 {
                        var wi = 0, wv = Float.infinity
                        for i in 0..<n where sim[i] < wv { wv = sim[i]; wi = i }
                        for t in 0..<T { C[c * T + t] = P[wi * T + t] }
                        sim[wi] = 1
                    } else {
                        var ss: Float = 0
                        for t in 0..<T { ss += C[c * T + t] * C[c * T + t] }
                        let inv = 1 / max(ss.squareRoot(), 1e-12)
                        for t in 0..<T { C[c * T + t] *= inv }
                    }
                }
                if changed == 0 { break }
            }
        }

        // Members per cluster, most central first.
        var members = [[Int]](repeating: [], count: k)   // indices into `live`
        for i in 0..<n { members[assign[i]].append(i) }
        for c in 0..<k { members[c].sort { sim[$0] > sim[$1] } }
        let order = (0..<k).filter { !members[$0].isEmpty }.sorted { members[$0].count > members[$1].count }

        // Unit spacing s = 1. Island radius from member count; islands packed greedily along a spiral.
        let golden = Float.pi * (3 - Float(5).squareRoot())
        let gap: Float = 3
        var centers = [SIMD2<Float>](), radii = [Float]()
        for c in order {
            let r = 1.05 * Float(members[c].count).squareRoot() + 1
            var pos = SIMD2<Float>(0, 0)
            if !centers.isEmpty {
                var a: Float = 0, rad: Float = 0
                while true {
                    pos = SIMD2<Float>(cos(a), sin(a)) * rad
                    var ok = true
                    for (q, cq) in centers.enumerated() where simd_length(pos - cq) < r + radii[q] + gap { ok = false; break }
                    if ok { break }
                    a += 0.35; rad += 0.35 * 0.8 / (2 * .pi) * 6
                }
            }
            centers.append(pos); radii.append(r)
        }
        var pts = [SIMD2<Float>](repeating: .zero, count: N)
        var extent: Float = 1
        for (oi, c) in order.enumerated() {
            for (m, i) in members[c].enumerated() {
                let r = 1.0 * (Float(m) + 0.5).squareRoot()
                let a = Float(m) * golden
                let p = centers[oi] + SIMD2<Float>(cos(a), sin(a)) * r
                pts[live[i]] = p
                extent = max(extent, simd_length(p))
            }
        }
        // Dead neurons: concentric rings outside everything, spacing 1.
        if !dead.isEmpty {
            var ring = extent + gap + 2
            var idx = 0
            while idx < dead.count {
                let cap = max(1, Int((2 * .pi * ring).rounded(.down)))
                let take = min(cap, dead.count - idx)
                for q in 0..<take {
                    let a = 2 * Float.pi * Float(q) / Float(take) + ring * 0.37
                    pts[dead[idx + q]] = SIMD2<Float>(cos(a), sin(a)) * ring
                    extent = max(extent, ring)
                }
                idx += take
                ring += 1.2
            }
        }
        let inv = 1 / max(extent, 1)
        for i in 0..<N { pts[i] *= inv }
        return pts
    }
}
