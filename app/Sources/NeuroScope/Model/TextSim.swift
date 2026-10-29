// Crude, language-agnostic text comparison used to judge candidate layers by behaviour.

import Foundation

enum TextSim {
    static func words(_ s: String) -> [String] {
        s.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init)
    }

    /// 0…1: half word overlap (Jaccard), half length agreement — so both vocabulary (French vs English) and
    /// shape (five words vs a paragraph) count.
    static func similarity(_ a: String, _ b: String) -> Float {
        let wa = words(a), wb = words(b)
        guard !wa.isEmpty || !wb.isEmpty else { return 1 }
        let sa = Set(wa), sb = Set(wb)
        let jac = Float(sa.intersection(sb).count) / Float(max(1, sa.union(sb).count))
        let len = Float(min(wa.count, wb.count)) / Float(max(1, max(wa.count, wb.count)))
        return 0.5 * jac + 0.5 * len
    }

    /// Echoing the question back, looping, or saying nothing all count as a broken answer.
    static func isBroken(_ out: String, question: String) -> Bool {
        let w = words(out)
        if w.count < 2 { return true }
        let q = Set(words(question))
        let echo = Float(Set(w).intersection(q).count) / Float(max(1, Set(w).count))
        if echo > 0.8 { return true }
        if w.count >= 8 && Float(Set(w).count) / Float(w.count) < 0.35 { return true }
        return false
    }
}
