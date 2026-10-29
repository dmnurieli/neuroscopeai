// Ready-made experiments for the guided flow: a concept, two matched prompt lists that differ only in it,
// whether we remove or add it, and a question that shows the effect.

import Foundation

struct Experiment: Identifiable, Hashable, Sendable {
    let id: String
    let title: String            // "Remove French"
    let summary: String          // one plain sentence: what you will see
    let mode: DirectionMode      // .ablate = remove the concept, .add = push it in
    let withConcept: [String]    // list A
    let withoutConcept: [String] // list B, matched line by line
    let testQuestion: String     // a question that shows the effect (expresses the concept when removing)
    /// The two halves of every prompt: A = conceptPrefix + question, B = oppositePrefix + question. Matched
    /// opposites ("in French" vs "in English") isolate the concept far better than "with vs without" did.
    let conceptPrefix: String
    let oppositePrefix: String

    init(id: String, title: String, summary: String, mode: DirectionMode,
         conceptPrefix: String, oppositePrefix: String, testQuestion: String) {
        self.id = id; self.title = title; self.summary = summary; self.mode = mode
        self.conceptPrefix = conceptPrefix; self.oppositePrefix = oppositePrefix
        self.withConcept = Experiment.questions.map { conceptPrefix + $0 }
        self.withoutConcept = Experiment.questions.map { oppositePrefix + $0 }
        self.testQuestion = testQuestion
    }

    /// Held-out questions (never in the lists) used to TEST candidate layers before the user sees anything.
    static let heldOut = ["Why is the sky blue?", "What do cats eat?"]
    /// Remove: prompts that ask for the concept — the change should make them answer like the opposite.
    /// Add: plain prompts — the change should make them answer like the concept.
    var probes: [String] { Experiment.heldOut.map { mode == .ablate ? conceptPrefix + $0 : $0 } }
    /// What success looks like for each probe: the matched prompt whose UNEDITED answer we want to resemble.
    var targets: [String] { Experiment.heldOut.map { mode == .ablate ? oppositePrefix + $0 : conceptPrefix + $0 } }


    /// Ready-made experiments, in the order the Change step shows them (the first is the hero card). Only changes that
    /// give a coherent, recognisable reply on OLMo 2 1B are offered (measured with `lesson_numbers derive` / `add`
    /// and checked in the app): Remove French (15 of 16 held-out French requests answered in English) and Remove brevity (one-word
    /// replies become full sentences on 12 of 13). The Add cards were dropped: Add French and Add pirate garbled the
    /// reply, Add enthusiasm ran away into "ZZZZ…" on a full-length answer, and Make it brief emptied it.
    static let all: [Experiment] = [
        Experiment(id: "remove-french", title: "Remove French",
                   summary: "Erase the French highlighter on every floor, so a request for a French reply comes back in English.",
                   mode: .ablate,
                   conceptPrefix: "Answer in French, in one short sentence: ",
                   oppositePrefix: "Answer in English, in one short sentence: ",
                   testQuestion: "Answer in French, in one short sentence: What do dogs eat?"),
        Experiment(id: "remove-brevity", title: "Remove brevity",
                   summary: "Erase the short-answer highlighter on every floor, so a request for a short reply gets a longer one.",
                   mode: .ablate,
                   conceptPrefix: "Answer in at most five words: ",
                   oppositePrefix: "Answer in detail, in several sentences: ",
                   testQuestion: "Answer in at most five words: How does the internet work?"),
    ]

    /// Turns a phrase the learner typed ("answer like Shakespeare.") into a prompt prefix that reads naturally
    /// in front of a question ("Answer like Shakespeare: "): trims spaces, drops trailing punctuation,
    /// capitalizes the first letter and ends with ": ". Empty input gives "".
    static func prefix(from phrase: String) -> String {
        var s = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
        let trailing = CharacterSet(charactersIn: ".,:;!?\u{2026}-\u{2013}\u{2014} ")
        while let last = s.unicodeScalars.last, trailing.contains(last) { s.unicodeScalars.removeLast() }
        guard let first = s.first else { return "" }
        s = first.uppercased() + s.dropFirst()
        return s + ": "
    }

    /// A short phrase for titles: the prefix without its ": ".
    static func label(from phrase: String) -> String {
        let p = prefix(from: phrase)
        return p.hasSuffix(": ") ? String(p.dropLast(2)) : p
    }

    /// Stable id for a custom concept (FNV-1a over the inputs), so the derivation cache survives re-entry.
    static func customID(_ parts: [String]) -> String {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in parts.joined(separator: "\u{1F}").utf8 { h = (h ^ UInt64(b)) &* 0x0000_0100_0000_01B3 }
        return "custom-" + String(h, radix: 16)
    }

    static let questions: [String] = [
        "What is the capital of Japan?",
        "How do plants make their food?",
        "Why do we have seasons?",
        "What is a rainbow?",
        "How does a bicycle stay upright?",
        "What do bees do all day?",
        "Why is the ocean salty?",
        "How do airplanes fly?",
        "What is the moon made of?",
        "Why do cats purr?",
        "How does a refrigerator keep food cold?",
        "What causes thunder?",
        "How do fish breathe underwater?",
        "What is gravity?",
    ]


    /// How the layer (and, for Add, the strength) is derived — by BEHAVIOUR, measured on held-out questions:
    /// each candidate layer is applied, the model answers the probes (thinking off, greedy, short), and the
    /// layer whose answers move most toward the targets' answers wins — rejecting candidates that echo the
    /// question, loop, or (Remove only) change a plain unrelated answer. Representation-only scores were tried
    /// first and picked layers that did nothing (measured on Qwen3-0.6B).
    /// Remove tries the upper half of the building. OLMo 2 1B (2026-10-28, engine/examples/lesson_numbers.rs
    /// `derive`): French is removed only by the direction from floor 12 or higher (sites 13-16, 0.81 of the
    /// depth and up); every floor below leaves the reply in French. Answers that break are rejected anyway.
    /// (Qwen3 needed a 0.70 cap: its later ablations answered with nothing.)
    static let removeDepthCap: Float = 1.0
    /// Add (steering) stays available only through Make your own, with a warning: on OLMo 2 1B the short-answer
    /// highlighter went from no effect to an empty reply (every floor 3-13 at 0.25-2 × |d|, 2026-10-28).
    static let addDepthRange: ClosedRange<Float> = 0.55...0.90
    static let addStrengths: [Float] = [1.0, 1.5, 2.0]   // × the direction's norm at that layer
    static let addStrength: Float = 1.5
}
