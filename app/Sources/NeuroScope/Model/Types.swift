// Value types shared across the app.
// The Codable types mirror engine/src/types.rs; their JSON is the ns_generate request wire format.

import Foundation

struct Geometry: Equatable, Sendable {
    var numLayers: Int
    var numNeurons: Int
    var numHeads: Int
    var hiddenSize: Int
    var vocabSize: Int

    var neuronsPerFrame: Int { numLayers * numNeurons }
    var headsPerFrame: Int { numLayers * numHeads }

    /// What the app expects before the engine has loaded (the shipped model's shape); replaced by the engine's own.
    static let olmo2_1B = Geometry(numLayers: 16, numNeurons: 8192, numHeads: 16, hiddenSize: 2048, vocabSize: 100352)

    // Counts as copy shows them ("8,192"): every number of floors, desks or memo numbers on screen comes from here.
    var floorsText: String { Geometry.grouped(numLayers) }
    var desksPerFloorText: String { Geometry.grouped(numNeurons) }
    var allDesksText: String { Geometry.grouped(numLayers * numNeurons) }
    var memoText: String { Geometry.grouped(hiddenSize) }
    /// Residual sites: 0 = before floor 0, l+1 = after floor l.
    var sites: Int { numLayers + 1 }

    static func grouped(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.usesGroupingSeparator = true
        f.locale = Locale(identifier: "en_US")
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    /// Fills {floors} {topFloor} {desks} {allDesks} {memo} {sites} in lesson and help copy.
    func fill(_ s: String) -> String {
        guard s.contains("{") else { return s }
        return s.replacingOccurrences(of: "{floors}", with: floorsText)
            .replacingOccurrences(of: "{topFloor}", with: "\(max(numLayers - 1, 0))")
            .replacingOccurrences(of: "{desks}", with: desksPerFloorText)
            .replacingOccurrences(of: "{allDesks}", with: allDesksText)
            .replacingOccurrences(of: "{memo}", with: memoText)
            .replacingOccurrences(of: "{sites}", with: "\(sites)")
    }
}

enum StreamKind: Int32, Codable, Sendable {
    case prompt = 0, thinking = 1, answer = 2, special = 3
}

/// A neuron's address: MLP unit `index` of layer `layer`.
struct NeuronID: Hashable, Codable, Sendable {
    var layer: Int
    var index: Int
    /// Offset into a frame's layer-major neuron vector.
    func flat(_ g: Geometry) -> Int { layer * g.numNeurons + index }
}

struct HeadID: Hashable, Codable, Sendable {
    var layer: Int
    var head: Int
    func flat(_ g: Geometry) -> Int { layer * g.numHeads + head }
}

enum Selection: Hashable, Sendable {
    case neuron(NeuronID)
    case head(HeadID)
}

enum EditMode: String, Codable, CaseIterable, Sendable {
    case ablate, scale, pin
}

struct NeuronEdit: Codable, Hashable, Sendable {
    var layer: Int
    var index: Int
    var mode: EditMode
    var value: Float
    var id: NeuronID { NeuronID(layer: layer, index: index) }
}

struct HeadEdit: Codable, Hashable, Sendable {
    var layer: Int
    var head: Int
    var mode: EditMode   // .ablate or .scale only
    var value: Float
    var id: HeadID { HeadID(layer: layer, head: head) }
}

enum DirectionMode: String, Codable, Sendable {
    case ablate   // abliteration: remove the direction from the residual stream
    case add      // steering: push along it by `value`
}

/// Residual-stream edit along a direction. JSON keys match engine/src/types.rs (the engine ignores `id`).
struct DirectionEdit: Codable, Hashable, Identifiable, Sendable {
    var id = UUID()
    var label: String
    var vector: [Float]          // hiddenSize floats
    var sites: [Int] = []        // empty = every residual site (embedding + after each layer)
    var mode: DirectionMode
    var value: Float = 0
}

struct EditSet: Codable, Hashable, Sendable {
    var neurons: [NeuronEdit] = []
    var heads: [HeadEdit] = []
    var directions: [DirectionEdit] = []
    var isEmpty: Bool { neurons.isEmpty && heads.isEmpty && directions.isEmpty }
    var count: Int { neurons.count + heads.count + directions.count }

    init(neurons: [NeuronEdit] = [], heads: [HeadEdit] = [], directions: [DirectionEdit] = []) {
        self.neurons = neurons; self.heads = heads; self.directions = directions
    }
    init(from d: Decoder) throws {
        let c = try d.container(keyedBy: CodingKeys.self)
        neurons = try c.decodeIfPresent([NeuronEdit].self, forKey: .neurons) ?? []
        heads = try c.decodeIfPresent([HeadEdit].self, forKey: .heads) ?? []
        directions = try c.decodeIfPresent([DirectionEdit].self, forKey: .directions) ?? []
    }
}

struct ContrastRequest: Codable, Hashable, Sendable {
    var a: [String]
    var b: [String]
    var thinking: Bool = false
    var edits: EditSet = EditSet()
}

/// Mirrors engine ContrastResult. Sites: 0 = embedding output, l+1 = after layer l.
struct ContrastResult: Codable, Sendable {
    var numLayers: Int
    var hiddenSize: Int
    var directions: [[Float]]
    var separation: [Float]
    var norms: [Float]
    var neuronDiff: [Float]
    var countA: Int
    var countB: Int
    enum CodingKeys: String, CodingKey {
        case directions, separation, norms
        case numLayers = "num_layers", hiddenSize = "hidden_size", neuronDiff = "neuron_diff"
        case countA = "count_a", countB = "count_b"
    }
}

enum Sampling: Hashable, Sendable {
    case greedy
    case sample(temperature: Float, topP: Float, topK: Int, seed: UInt64)
}

extension Sampling: Codable {
    private enum K: String, CodingKey { case mode, temperature, top_p, top_k, seed }
    init(from d: Decoder) throws {
        let c = try d.container(keyedBy: K.self)
        if try c.decode(String.self, forKey: .mode) == "greedy" { self = .greedy; return }
        self = .sample(temperature: try c.decode(Float.self, forKey: .temperature),
                       topP: try c.decode(Float.self, forKey: .top_p),
                       topK: try c.decode(Int.self, forKey: .top_k),
                       seed: try c.decode(UInt64.self, forKey: .seed))
    }
    func encode(to e: Encoder) throws {
        var c = e.container(keyedBy: K.self)
        switch self {
        case .greedy: try c.encode("greedy", forKey: .mode)
        case let .sample(t, p, k, s):
            try c.encode("sample", forKey: .mode)
            try c.encode(t, forKey: .temperature); try c.encode(p, forKey: .top_p)
            try c.encode(k, forKey: .top_k); try c.encode(s, forKey: .seed)
        }
    }
}

/// The ns_generate request. CodingKeys = engine JSON keys.
struct GenerationRequest: Codable, Hashable, Sendable {
    var prompt: String
    var thinking: Bool = true
    var maxTokens: Int = 512
    var logitLens: Bool = true
    var topK: Int = 8
    var paceMs: Int = 0
    var sampling: Sampling = .greedy
    var edits: EditSet = EditSet()

    enum CodingKeys: String, CodingKey {
        case prompt, thinking, sampling, edits
        case maxTokens = "max_tokens", logitLens = "logit_lens", topK = "top_k", paceMs = "pace_ms"
    }
}

struct GenerationSummary: Codable, Hashable, Sendable {
    var ok: Bool
    var tokensGenerated: Int?
    var stop: String?
    var prefillMs: Double?
    var decodeTokPerS: Double?
    var error: String?
    enum CodingKeys: String, CodingKey {
        case ok, stop, error
        case tokensGenerated = "tokens_generated", prefillMs = "prefill_ms", decodeTokPerS = "decode_tok_per_s"
    }
}

struct TokenScore: Codable, Hashable, Sendable {
    var id: UInt32
    var token: String
    var score: Float
}

struct NeuronInfo: Codable, Hashable, Sendable {
    var layer: Int
    var index: Int
    var promotes: [TokenScore]
    var suppresses: [TokenScore]
    var outNorm: Float
    enum CodingKeys: String, CodingKey { case layer, index, promotes, suppresses; case outNorm = "out_norm" }
}

struct TokenCandidate: Hashable, Sendable {
    var id: UInt32
    var prob: Float
}

/// One frame as delivered by the engine bridge (already copied out of the C callback).
struct TokenFrame: Sendable {
    var index: Int
    var tokenID: UInt32
    var stream: StreamKind
    var text: String
    var neurons: [Float16]        // geometry.neuronsPerFrame, layer-major
    var heads: [Float]            // geometry.headsPerFrame
    var top: [TokenCandidate]
    var lens: [TokenCandidate]    // numLayers entries, or empty
    var logprob: Float            // NaN for prompt tokens
}

/// Per-token metadata a Run keeps (activations live in Run's flat buffers).
struct TokenRecord: Hashable, Sendable {
    var index: Int
    var tokenID: UInt32
    var stream: StreamKind
    var text: String
    var top: [TokenCandidate]
    var lens: [TokenCandidate]
    var logprob: Float
}

enum LayoutMode: String, CaseIterable, Sendable {
    case stack   // 3D: one plane per layer, stacked in depth
    case sheet   // 2D: all layer grids tiled flat
    case cluster // 3D stack, but within each layer neurons are placed by functional similarity (co-activation)
    case columns // 2D: floors as columns left → right (like a textbook network diagram) — used by slow motion
}

enum ColorMode: String, CaseIterable, Sendable {
    case activation  // brightness = |activation| normalized per layer; warm = +, cool = −
    case diff        // edited run minus baseline at the same position; red = up, blue = down
    case contrast    // ContrastResult.neuronDiff: which neurons respond to concept A vs B (static map)
}

struct ViewSettings: Hashable, Sendable {
    var layout: LayoutMode = .stack
    var colorMode: ColorMode = .activation
    var glowHalfLife: Float = 0.35   // seconds for a neuron's glow to halve after it fires
    var exposure: Float = 1.0        // brightness gain
    var threshold: Float = 0.08      // normalized activations below this stay dark
    var showHeads: Bool = true
    var pointSize: Float = 1.0
    /// Lessons: when set, only these floors are drawn at full strength; every other floor fades to dust.
    var highlightFloors: Set<Int>? = nil
}

/// ns_trace request (engine TraceRequest). focus encodes as [layer, index] or null.
struct TraceRequest: Codable, Sendable {
    var tokenIDs: [UInt32]
    var position: Int
    var edits: EditSet = EditSet()
    var topSources: Int = 12
    var topTargets: Int = 12
    var focus: [Int]? = nil
    var focusTop: Int = 24
    enum CodingKeys: String, CodingKey {
        case position, edits, focus
        case tokenIDs = "token_ids", topSources = "top_sources", topTargets = "top_targets", focusTop = "focus_top"
    }
}

/// One arrow: how much the source desk's note pushed (+) or held back (−) the target desk at this word.
struct TraceEdge: Codable, Hashable, Sendable {
    var fromLayer: Int
    var fromIndex: Int
    var toLayer: Int
    var toIndex: Int
    var weight: Float
    var from: NeuronID { NeuronID(layer: fromLayer, index: fromIndex) }
    var to: NeuronID { NeuronID(layer: toLayer, index: toIndex) }
    enum CodingKeys: String, CodingKey {
        case weight
        case fromLayer = "from_layer", fromIndex = "from_index", toLayer = "to_layer", toIndex = "to_index"
    }
}

struct TraceResult: Codable, Sendable {
    var position: Int
    var tokenID: UInt32
    var edges: [TraceEdge]          // floor l → l+1 for every l, strongest first within each step
    var focusEdges: [TraceEdge]     // every earlier floor → the focus desk
    var explained: [Float]          // per target floor: share of its busiest desks' input the arrows explain
    var forwardMs: Double           // real time of one pass at this position
    enum CodingKeys: String, CodingKey {
        case position, edges, explained
        case tokenID = "token_id", focusEdges = "focus_edges", forwardMs = "forward_ms"
    }
}

/// ns_project request: how much of one direction the memo carries at every floor and word of a run.
struct ProjectRequest: Codable, Sendable {
    var tokenIDs: [UInt32]
    var edits: EditSet = EditSet()
    var direction: [Float]
    var sampleSite: Int? = nil
    var samplePosition: Int? = nil
    var sampleDims: Int = 6
    enum CodingKeys: String, CodingKey {
        case edits, direction
        case tokenIDs = "token_ids", sampleSite = "sample_site", samplePosition = "sample_position", sampleDims = "sample_dims"
    }
}

/// Real numbers of one memo at one floor and word: x (before), d̂, x_new (after the floor's edit).
struct ProjectSample: Codable, Sendable {
    var site: Int
    var position: Int
    var dot: Float
    var dotAfter: Float
    var x: [Float]
    var dHat: [Float]
    var xNew: [Float]
    var xNorm: Float
    var xNewNorm: Float
    var dNorm: Float
    enum CodingKeys: String, CodingKey {
        case site, position, dot, x
        case dotAfter = "dot_after", dHat = "d_hat", xNew = "x_new", xNorm = "x_norm", xNewNorm = "x_new_norm", dNorm = "d_norm"
    }
}

struct ProjectResult: Codable, Sendable {
    var sites: Int
    var positions: Int
    var before: [Float]   // positions × sites, position-major: x·d̂ arriving at each site
    var after: [Float]    // positions × sites: x·d̂ after the site's edit
    var sample: ProjectSample?
}

/// One floor's tryout while the app derived where to apply a concept change.
struct Tryout: Hashable, Sendable {
    var site: Int          // residual site; floor = site - 1
    var value: Float       // steering amount (0 for erase)
    var score: Float       // how far the answers moved toward the target (−1 = rejected: echoed, looped or broke)
    var sampleAnswer: String
}

/// "How much highlighter is in the memo" per floor, averaged over the reply's words, for one run.
struct InkCurve: Sendable {
    var runID: UInt32
    var arriving: [Float]  // per site: mean x·d̂ arriving (before the site's edit)
    var leaving: [Float]   // per site: mean x·d̂ after the site's edit
}

/// ns_follow request: one desk's whole story at one word.
struct FollowRequest: Codable, Sendable {
    var tokenIDs: [UInt32]
    var position: Int
    var edits: EditSet = EditSet()
    var layer: Int
    var index: Int
    var top: Int = 8
    var hops: Int = 5
    enum CodingKeys: String, CodingKey { case position, edits, layer, index, top, hops; case tokenIDs = "token_ids" }
}

struct WordScore: Codable, Hashable, Sendable {
    var id: UInt32
    var token: String
    var score: Float
}

/// Every number of one desk's computation at one word (engine FollowResult; see engine/src/types.rs for the math).
struct FollowResult: Codable, Sendable {
    var layer: Int
    var index: Int
    var position: Int
    var tokenID: UInt32
    var memoNorm: Float
    var rms: Float
    var gate: Float
    var sigma: Float
    var silu: Float
    var up: Float
    var hRaw: Float
    var h: Float
    var wakesOn: [WordScore]
    var askedBy: [TraceEdge]
    var askedByExplained: Float
    var columnNorm: Float
    var writeNorm: Float
    var heardBy: [TraceEdge]
    var path: [TraceEdge]
    var nextToken: UInt32
    var nextEffect: Float
    var wordEffects: [WordScore]
    enum CodingKeys: String, CodingKey {
        case layer, index, position, rms, gate, sigma, silu, up, h, path
        case tokenID = "token_id", memoNorm = "memo_norm", hRaw = "h_raw", wakesOn = "wakes_on", askedBy = "asked_by"
        case askedByExplained = "asked_by_explained", columnNorm = "column_norm", writeNorm = "write_norm"
        case heardBy = "heard_by", nextToken = "next_token", nextEffect = "next_effect", wordEffects = "word_effects"
    }
}

// MARK: - Lessons (the default, beginner experience: one idea per page, one button per page)

/// What a lesson page makes the app DO. The page's Continue waits until the action finishes when `waits` is true.
enum LessonAction: Hashable, Sendable {
    case none
    case ask(String)                                 // run this question with thinking OFF; wait until done
    case slowMotion                                  // slow motion on the first reply word, paused at floor 0
    case slowMotionTo(Double)                        // glide the word's trip to this floor (0…numLayers + 1)
    case followDesk(layer: Int, index: Int, step: Int) // follow this desk at the first reply word, show story step 0…6
    case findConcept(String)                         // Experiment id: compute the highlighter (choose)
    case tryChange                                   // run Before/After with the current change
    case mathToy(Int)                                // the 3-number worked example, stage 0…4 (drawn by the page)
    case replay                                      // replay the last answer being written, word by word
    case explore                                     // leave the lessons for Explore
}

/// How the field looks on a page.
enum FieldPreset: Hashable, Sendable {
    case keep                    // leave it as the previous page left it
    case overview                // the whole building, 3/4 view, activity colors
    case floors([Int])           // overview with only these floors highlighted
    case columns                 // the wide floors-as-columns diagram (slow motion / follow)
    case concept                 // the concept map colors
    case compare                 // before→after change colors
    case hidden                  // no field (e.g. the math pages)
}

struct LessonPage: Identifiable, Hashable, Sendable {
    let id: String
    /// One idea, at most two short sentences, markdown allowed (bold a new word once).
    let text: String
    /// Optional single smaller line under it (a number, a tip). Never a paragraph.
    var detail: String? = nil
    var action: LessonAction = .none
    var field: FieldPreset = .keep
    var continueLabel: String = "Continue"
    /// Continue stays disabled until the action has finished (e.g. the answer is written).
    var waits: Bool = false
}

struct Lesson: Identifiable, Hashable, Sendable {
    let id: String
    let title: String
    let subtitle: String
    let minutes: Int
    let pages: [LessonPage]
}
