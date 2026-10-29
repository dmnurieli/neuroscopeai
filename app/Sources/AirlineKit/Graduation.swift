import Foundation

// Lesson 9, "Graduation": recorded outputs of OLMo 2 1B (allenai, fully open) at its published stages, so the
// lesson ships text, not 5.9 GB of weights. The data is GENERATED (GraduationRecordings.swift, from
// scripts/stages/record.sh → recordings.jsonl → to_swift.py); this file is the API the lesson reads.

/// One of the model's top guesses for the next word piece, with its probability.
public struct StagePick: Sendable, Hashable {
    public let token: String
    public let prob: Float
    public init(token: String, prob: Float) { self.token = token; self.prob = prob }
}

/// One fixed prompt run greedily on one stage, with where the stage came from.
public struct StageRecording: Sendable, Hashable {
    public let stage: String
    /// "after 21 billion word pieces", "after flight school (SFT)".
    public let label: String
    public let repo: String
    public let branch: String
    public let commit: String
    /// Word pieces of pretraining text the checkpoint had seen (0 for the post-training stages: not recorded).
    public let trainedTokens: Int64
    public let promptID: String
    public let prompt: String
    /// "plain" (continued like a document) or "olmo2-chat" (the stage's chat template).
    public let format: String
    public let maxTokens: Int
    public let tokensGenerated: Int
    /// "eos" (it finished) or "maxtokens" (cut off at `maxTokens`).
    public let stop: String
    public let tokPerS: Double
    public let recordedAt: String
    public let device: String
    /// Its top guesses for the word piece right after the prompt.
    public let nextTop: [StagePick]
    public let output: String

    public init(stage: String, label: String, repo: String, branch: String, commit: String, trainedTokens: Int64,
                promptID: String, prompt: String, format: String, maxTokens: Int, tokensGenerated: Int, stop: String,
                tokPerS: Double, recordedAt: String, device: String, nextTop: [StagePick], output: String) {
        self.stage = stage; self.label = label; self.repo = repo; self.branch = branch; self.commit = commit
        self.trainedTokens = trainedTokens; self.promptID = promptID; self.prompt = prompt; self.format = format
        self.maxTokens = maxTokens; self.tokensGenerated = tokensGenerated; self.stop = stop; self.tokPerS = tokPerS
        self.recordedAt = recordedAt; self.device = device; self.nextTop = nextTop; self.output = output
    }

    /// True when it stopped on its own; false when cut off at the word-piece limit.
    public var finished: Bool { stop == "eos" }

    /// "recorded on this Mac from allenai/OLMo-2-0425-1B-SFT @0d85a3d0" (the branch named when it is not main).
    public var provenance: String {
        let at = branch == "main" ? "" : " \(branch)"
        return "recorded on this Mac from \(repo)\(at) @\(commit.prefix(8))"
    }

    /// "28 Oct 2026" from `recordedAt` (ISO 8601, UTC).
    public var recordedDay: String {
        let iso = ISO8601DateFormatter()
        guard let d = iso.date(from: recordedAt) else { return String(recordedAt.prefix(10)) }
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "d MMM yyyy"
        return f.string(from: d)
    }
}

public enum Graduation {
    /// The "watch it learn" checkpoints of the base model, earliest first (allenai/OLMo-2-0425-1B stage1 branches).
    public static let snapshotStages = ["step0", "step300", "step10000", "step100000", "step1000000", "step1907359"]
    /// The finished base model, then the post-training stages in the order they were trained.
    public static let baseStage = "base"
    public static let postStages = ["sft", "dpo", "rlvr1", "instruct"]

    public static func recording(_ stage: String, _ promptID: String) -> StageRecording? {
        recordings.last { $0.stage == stage && $0.promptID == promptID }
    }

    /// 0 → "0", 1e9 → "1 billion", 2.098e12 → "2.1 trillion", 4.001e12 → "4 trillion".
    public static func wordPieces(_ n: Int64) -> String {
        func short(_ v: Double, _ unit: String) -> String {
            let r = (v * 10).rounded() / 10
            let s = r == r.rounded() ? String(Int(r)) : String(format: "%.1f", r)
            return "\(s) \(unit)"
        }
        let v = Double(n)
        if v >= 1e12 { return short(v / 1e12, "trillion") }
        if v >= 1e9 { return short(v / 1e9, "billion") }
        if v >= 1e6 { return short(v / 1e6, "million") }
        return n.formatted()
    }

    // MARK: the maths prompt: a checkable answer

    /// The maths prompt's departure and trip, so the right answer is computed, never typed.
    public static let mathDeparture = (hour: 15, minute: 40)
    public static let mathTrip = (hours: 2, minutes: 35)

    /// "6:15 pm".
    public static var mathAnswer: String {
        let m = mathDeparture.hour * 60 + mathDeparture.minute + mathTrip.hours * 60 + mathTrip.minutes
        let h = (m / 60) % 24, mm = m % 60
        let h12 = h % 12 == 0 ? 12 : h % 12
        return String(format: "%d:%02d %@", h12, mm, h >= 12 ? "pm" : "am")
    }

    /// The landing time an answer settles on: the boxed answer when there is one, else the last "h:mm" it wrote.
    /// Returned as "h:mm" (no am/pm), or nil when it never wrote a time.
    public static func landingTime(in output: String) -> String? {
        if let r = output.range(of: #"\\boxed\{[^}]*\}"#, options: .regularExpression),
           let t = lastTime(in: String(output[r])) { return t }
        return lastTime(in: output)
    }

    /// Whether `output` lands on the right time.
    public static func landsRight(_ output: String) -> Bool {
        guard let t = landingTime(in: output) else { return false }
        return t == String(mathAnswer.split(separator: " ")[0])
    }

    // MARK: the false-premise prompt

    /// Whether an answer to the Great Wall prompt questions its premise (says it can't be seen, or calls it a myth).
    public static func pushesBack(_ output: String) -> Bool {
        let t = output.lowercased()
        let signs = ["myth", "misconception", "not visible", "isn't visible", "is not actually", "cannot be seen", "can't be seen",
                     "not possible to see", "cannot see", "can't see", "false premise", "actually not", "not true"]
        return signs.contains { t.contains($0) }
    }

    private static func lastTime(in s: String) -> String? {
        guard let re = try? NSRegularExpression(pattern: #"\b(\d{1,2}):(\d{2})\b"#) else { return nil }
        let ns = s as NSString
        guard let m = re.matches(in: s, range: NSRange(location: 0, length: ns.length)).last else { return nil }
        let h = Int(ns.substring(with: m.range(at: 1))) ?? 0
        return "\(h):\(ns.substring(with: m.range(at: 2)))"
    }

    // MARK: display helpers

    /// The output trimmed of leading/trailing blank space, with runs of 3+ newlines collapsed (display only; the
    /// record keeps every character).
    public static func tidy(_ s: String) -> String {
        var t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        while t.contains("\n\n\n") { t = t.replacingOccurrences(of: "\n\n\n", with: "\n\n") }
        return t
    }

    /// A word piece made readable: a leading space shown as nothing, a newline as "↵", an empty piece as "·".
    public static func pieceText(_ token: String) -> String {
        let t = token.replacingOccurrences(of: "\n", with: "↵")
        let trimmed = t.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? (t.isEmpty ? "·" : "␣") : trimmed
    }

    /// A probability as "61%", "<1%".
    public static func percent(_ p: Float) -> String {
        let v = Double(p) * 100
        if v < 1 { return "<1%" }
        return "\(Int(v.rounded()))%"
    }
}

// MARK: - the published training sets behind each stage (dataset cards on huggingface.co, read 2026-10-28)

public enum GraduationData {
    /// allenai/tulu-3-sft-olmo-2-mixture-0225 (train split): the SFT stage's example conversations.
    public static let sftExamples = 866_138
    public static let sftDataset = "allenai/tulu-3-sft-olmo-2-mixture-0225"
    /// allenai/olmo-2-0425-1b-preference-mix (train split): answer pairs, one chosen and one rejected (DPO).
    public static let preferencePairs = 378_301
    public static let preferenceDataset = "allenai/olmo-2-0425-1b-preference-mix"
    /// allenai/RLVR-GSM-MATH-IF-Mixed-Constraints: questions with an answer key (school maths, MATH, format rules), RLVR1.
    public static let checkablePrompts = 29_946
    public static let checkableDataset = "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"
    /// allenai/RLVR-MATH: the final round's maths problems (RLVR1 → Instruct).
    public static let finalMathPrompts = 7_500
    public static let finalMathDataset = "allenai/RLVR-MATH"
}
