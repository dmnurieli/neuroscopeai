// The arithmetic every network shares: flat parameters, one flight at a time, nothing allocated in the loop.
// Network (display, scoring) and Trainer (descent) both go through here, so what the app draws and what it
// trains are the same numbers.
//
// Parameter layout, per layer l: weights (in × out, row-major, as `Network.weights[l]`), then biases (out).
// Training loss on one flight: Huber(1) on the standardised target; its gradient is clamp(error, −1, 1).
// Route upkeep: λ·|w| on clipboard routes (layer 0) only, gradient λ·sign(w).

import Foundation

final class NetCore {
    let sizes: [Int]
    let layers: Int
    let wOff: [Int], bOff: [Int]
    let paramCount: Int
    /// Offset of each layer's units in the activation and delta buffers.
    let aOff: [Int]
    let p: UnsafeMutablePointer<Float>       // parameters
    let g: UnsafeMutablePointer<Float>       // gradient accumulator
    let saved: UnsafeMutablePointer<Float>   // the network before the last step, for the tumble guard
    let a: UnsafeMutablePointer<Float>       // activations (layer 0 = standardised inputs; hubs after ReLU)
    let d: UnsafeMutablePointer<Float>       // ∂loss/∂(pre-activation) per unit

    init(sizes: [Int]) {
        self.sizes = sizes
        layers = sizes.count - 1
        var w: [Int] = [], b: [Int] = [], o = 0
        for l in 0..<layers {
            w.append(o); o += sizes[l] * sizes[l + 1]
            b.append(o); o += sizes[l + 1]
        }
        wOff = w; bOff = b; paramCount = o
        var ao: [Int] = [], u = 0
        for s in sizes { ao.append(u); u += s }
        aOff = ao
        p = .allocate(capacity: max(o, 1)); p.initialize(repeating: 0, count: max(o, 1))
        g = .allocate(capacity: max(o, 1)); g.initialize(repeating: 0, count: max(o, 1))
        saved = .allocate(capacity: max(o, 1)); saved.initialize(repeating: 0, count: max(o, 1))
        a = .allocate(capacity: max(u, 1)); a.initialize(repeating: 0, count: max(u, 1))
        d = .allocate(capacity: max(u, 1)); d.initialize(repeating: 0, count: max(u, 1))
    }

    convenience init(_ network: Network) {
        self.init(sizes: network.layerSizes)
        load(network)
    }

    deinit {
        p.deallocate(); g.deallocate(); saved.deallocate(); a.deallocate(); d.deallocate()
    }

    func load(_ n: Network) {
        for l in 0..<layers {
            n.weights[l].withUnsafeBufferPointer { src in
                if let base = src.baseAddress { (p + wOff[l]).update(from: base, count: src.count) }
            }
            n.biases[l].withUnsafeBufferPointer { src in
                if let base = src.baseAddress { (p + bOff[l]).update(from: base, count: src.count) }
            }
        }
    }

    func store(into n: inout Network) {
        for l in 0..<layers {
            n.weights[l] = Array(UnsafeBufferPointer(start: p + wOff[l], count: sizes[l] * sizes[l + 1]))
            n.biases[l] = Array(UnsafeBufferPointer(start: p + bOff[l], count: sizes[l + 1]))
        }
    }

    /// Standardised inputs in, standardised prediction out. Same summation order as `Network.forward`.
    @inline(__always) func forward(_ x: UnsafePointer<Float>) -> Float {
        if sizes[0] > 0 { a.update(from: x, count: sizes[0]) }
        for l in 0..<layers {
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

    /// After `forward`: adds scale × ∂Huber/∂params for target y into `g`. Returns the error (guess − truth).
    @inline(__always) func accumulate(target y: Float, scale: Float) -> Float {
        let err = a[aOff[layers]] - y
        d[aOff[layers]] = min(max(err, -1), 1) * scale
        var l = layers - 1
        while l >= 0 {
            let nin = sizes[l], nout = sizes[l + 1]
            let al = a + aOff[l], dn = d + aOff[l + 1], w = p + wOff[l]
            let gw = g + wOff[l], gb = g + bOff[l]
            for j in 0..<nout { gb[j] += dn[j] }
            for i in 0..<nin {
                let ai = al[i]
                if ai != 0 { for j in 0..<nout { gw[i * nout + j] += ai * dn[j] } }
            }
            if l > 0 {
                let dl = d + aOff[l]
                for i in 0..<nin {
                    if al[i] > 0 {      // the gate agent let this hub through, so blame flows back
                        var s: Float = 0
                        for j in 0..<nout { s += w[i * nout + j] * dn[j] }
                        dl[i] = s
                    } else {
                        dl[i] = 0
                    }
                }
            }
            l -= 1
        }
        return err
    }

    /// After `accumulate`: ∂loss/∂(pre-activation) of every unit past the inputs, one array per hidden row, then
    /// the output (zero for a hub its gate agent shut). Scaled by the `scale` that `accumulate` was given.
    func deltas() -> [[Float]] {
        (1...layers).map { l in Array(UnsafeBufferPointer(start: d + aOff[l], count: sizes[l])) }
    }

    func zeroGradient() { g.update(repeating: 0, count: paramCount) }

    /// Route upkeep on clipboard routes: λ·sign(w).
    func addUpkeep(_ lambda: Float) {
        guard lambda != 0, layers > 0 else { return }
        let w = p + wOff[0], gw = g + wOff[0]
        for k in 0..<(sizes[0] * sizes[1]) {
            let v = w[k]
            gw[k] += v > 0 ? lambda : (v < 0 ? -lambda : 0)
        }
    }

    func step(_ lr: Float) {
        for k in 0..<paramCount { p[k] -= lr * g[k] }
    }

    /// True when the parameters can no longer give a finite guess for inputs up to `maxInput` in size: a
    /// non-finite parameter, or a worst-case tally (|x|·max|w|·fan-in + max|b|, layer by layer) past Float's
    /// range. Sane networks sit tens of orders of magnitude below that line; a huge stride crosses it in a step
    /// or two, often by blowing every hub shut first, so checking finiteness alone would miss the fall.
    func hasTumbled(maxInput: Float) -> Bool {
        var bound = Double(max(maxInput, 1))
        for l in 0..<layers {
            var wMax: Float = 0, bMax: Float = 0
            for k in 0..<(sizes[l] * sizes[l + 1]) { wMax = max(wMax, abs(p[wOff[l] + k])) }
            for k in 0..<sizes[l + 1] { bMax = max(bMax, abs(p[bOff[l] + k])) }
            guard wMax.isFinite, bMax.isFinite else { return true }
            bound = bound * Double(wMax) * Double(max(sizes[l], 1)) + Double(bMax)
        }
        return !(bound < Double(Float.greatestFiniteMagnitude))
    }

    func save() { saved.update(from: p, count: paramCount) }
    func restore() { p.update(from: saved, count: paramCount) }
}

extension Network {
    /// Ledger columns this network reads, in input order.
    var inputColumns: [Int] { architecture.inputs.map(\.rawValue) }

    /// Writes flight i's standardised inputs (this network's own normalisation) into x.
    @inline(__always) func standardise(flight i: Int, in ledger: Ledger, columns: [Int], into x: UnsafeMutablePointer<Float>) {
        let base = i * World.factCount
        ledger.columns.facts.withUnsafeBufferPointer { facts in
            for k in 0..<columns.count { x[k] = (facts[base + columns[k]] - inputMean[k]) / inputSD[k] }
        }
    }

    @inline(__always) func standardisedTarget(flight i: Int, in ledger: Ledger) -> Float {
        (ledger.delay(i) - delayMean) / delaySD
    }

    /// Mean Huber(1) gradient over `indices` (standardised target), plus upkeep·sign(w) on clipboard routes:
    /// exactly the arithmetic `Trainer` steps with. Shaped like `weights` / `biases`. Exposed for the gradient check.
    func gradient(indices: [Int], in ledger: Ledger, upkeep: Float = 0) -> (weights: [[Float]], biases: [[Float]]) {
        let core = NetCore(self)
        let cols = inputColumns
        let x = UnsafeMutablePointer<Float>.allocate(capacity: max(cols.count, 1))
        defer { x.deallocate() }
        let scale = 1 / Float(max(indices.count, 1))
        for i in indices {
            standardise(flight: i, in: ledger, columns: cols, into: x)
            _ = core.forward(x)
            _ = core.accumulate(target: standardisedTarget(flight: i, in: ledger), scale: scale)
        }
        core.addUpkeep(upkeep)
        var w: [[Float]] = [], b: [[Float]] = []
        for l in 0..<core.layers {
            w.append(Array(UnsafeBufferPointer(start: core.g + core.wOff[l], count: core.sizes[l] * core.sizes[l + 1])))
            b.append(Array(UnsafeBufferPointer(start: core.g + core.bOff[l], count: core.sizes[l + 1])))
        }
        return (w, b)
    }
}
