import SwiftUI

/// Colors and fonts. Dark, instrument-like; stream colors used everywhere a token is drawn.
enum Theme {
    static let background = Color(red: 0.02, green: 0.027, blue: 0.05)
    static let panel = Color(red: 0.06, green: 0.07, blue: 0.10)
    static let thinking = Color(red: 0.55, green: 0.62, blue: 0.85)
    static let answer = Color(red: 0.95, green: 0.93, blue: 0.88)
    static let prompt = Color(red: 0.45, green: 0.80, blue: 0.70)
    static let special = Color(red: 0.6, green: 0.6, blue: 0.6)
    static let positive = Color(red: 1.0, green: 0.62, blue: 0.20)
    static let negative = Color(red: 0.25, green: 0.75, blue: 1.0)
    static let ablated = Color(red: 0.85, green: 0.15, blue: 0.2)
    static let scaled = Color(red: 0.7, green: 0.45, blue: 1.0)
    static let pinned = Color(red: 1.0, green: 0.82, blue: 0.3)
    static let mono = Font.system(.body, design: .monospaced)

    static func color(for s: StreamKind) -> Color {
        switch s { case .prompt: prompt; case .thinking: thinking; case .answer: answer; case .special: special }
    }
}

extension TokenRecord {
    /// What a beginner sees for this token: control tokens become plain words instead of "<|endoftext|>".
    /// OLMo 2 opens every prompt with <|endoftext|> (its start marker) and ends replies with it.
    var displayText: String {
        switch text {
        case "<think>": return "(starts thinking)"
        case "</think>": return "(done thinking)"
        case "<|endoftext|>" where stream == .prompt: return "(start)"
        case "<|im_end|>", "<|endoftext|>": return "(finished)"
        case "<|im_start|>": return "(turn)"
        default: return text
        }
    }
}

extension Run {
    private static let scaffoldHeads = ["<|endoftext|><|user|>\n", "<|im_start|>user\n"]
    private static let scaffoldTails = ["\n<|assistant|>\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                        "<|im_end|>\n<|im_start|>assistant\n"]

    /// Per token: true when it is chat-template scaffolding, not the person's words. OLMo 2 wraps a question as
    /// "<|endoftext|><|user|>\n … \n<|assistant|>\n", where <|user|> and <|assistant|> are ordinary pieces
    /// ("<" "|" "user" "|" ">\n"); Qwen3 as "<|im_start|>user\n … <|im_end|>\n<|im_start|>assistant\n".
    /// Leading prompt tokens that spell a template head, and trailing ones that fit wholly inside a template
    /// tail, are scaffolding; a piece that mixes the person's text with the template ("?\n") is kept.
    var scaffoldMask: [Bool] {
        let toks = tokens
        var mask = [Bool](repeating: false, count: toks.count)
        let pc = promptCount
        guard pc > 0 else { return mask }
        let longest = Self.scaffoldHeads.map(\.count).max() ?? 0
        var acc = ""
        for k in 0..<pc {
            acc += toks[k].text
            if Self.scaffoldHeads.contains(acc) { for i in 0...k { mask[i] = true }; break }
            if acc.count >= longest { break }
        }
        var tail = ""
        for k in stride(from: pc - 1, through: 0, by: -1) where !mask[k] {
            tail = toks[k].text + tail
            guard !tail.isEmpty, Self.scaffoldTails.contains(where: { $0.hasSuffix(tail) }) else { break }
            mask[k] = true
        }
        return mask
    }
}
