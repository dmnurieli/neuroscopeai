// Which desks light up for one word: salience by rank, per floor and per word.
//
// Raw activations differ wildly between floors and between words, and the bulk of every floor sits a little above
// zero, so any fixed brightness scale either lights thousands of desks a little (a grey haze) or almost none. Instead
// each floor, at each word, colours only its busiest few: the top `litShare` of its desks by |activation| are fully
// saturated, the next `faintShare` get a faint tint, and every other desk stays grey. The shader turns the cutoffs
// computed here into a 0…1 salience per desk (see `salience` in Shaders.swift).

import simd

enum Salience {
    /// Share of a floor's desks drawn fully saturated at the default Intensity (about 82 of 8,192).
    static let litShare: Float = 0.01
    /// Share of a floor's desks just below those, drawn with a faint tint.
    static let faintShare: Float = 0.02
    /// The Intensity slider value that means "the default share" (App.swift starts there).
    static let referenceExposure: Float = 0.5

    /// Lit share for an Intensity setting: Intensity scales how many desks light, never how bright one gets.
    static func litShare(exposure: Float) -> Float {
        let e = exposure.isFinite ? max(exposure, 0) : referenceExposure
        return min(max(litShare * e / referenceExposure, 0.002), 0.05)
    }

    /// Per-floor cutoffs on |activation| for one word, layer-major `values` (numLayers × perLayer).
    /// x: where the faint tier starts, y: where the lit tier starts, z: the floor's largest |activation|, w: 0.
    /// `scratch` is reused between calls to avoid allocating per word.
    static func cutoffs(_ values: UnsafeBufferPointer<Float16>, perLayer: Int, layers: Int, litShare lit: Float,
                        faintShare faint: Float, scratch: inout [Float]) -> [SIMD4<Float>] {
        var out = [SIMD4<Float>](repeating: .zero, count: max(layers, 0))
        let n = max(perLayer, 1)
        if scratch.count < n { scratch = [Float](repeating: 0, count: n) }
        for l in 0..<layers {
            let lo = l * n
            guard lo < values.count else { break }
            let m = min(n, values.count - lo)
            var top: Float = 0
            for j in 0..<m {
                let a = abs(Float(values[lo + j]))
                let v = a.isFinite ? a : 0
                scratch[j] = v
                if v > top { top = v }
            }
            guard top > 0, m > 1 else { continue }
            // k-th largest = (m − k)-th smallest. At least one desk is lit on any floor that is not all zero.
            let kLit = max(1, Int((Float(m) * lit).rounded()))
            let kFaint = min(m, kLit + max(0, Int((Float(m) * faint).rounded())))
            let litCut = select(&scratch, count: m, m - kLit)
            let faintCut = kFaint > kLit ? select(&scratch, count: m, m - kFaint) : litCut
            out[l] = SIMD4(faintCut, litCut, top, 0)
        }
        return out
    }

    /// k-th smallest (0-based) of a[0..<count] by quickselect; reorders that prefix.
    private static func select(_ a: inout [Float], count: Int, _ k: Int) -> Float {
        var lo = 0, hi = count - 1
        let k = min(max(k, 0), count - 1)
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
}
