// The single source of truth for the UI.

import Foundation
import Observation

@MainActor
@Observable
final class AppState {
    enum EngineStatus: Equatable {
        case idle                     // nothing started: the app opens on Home without the model
        case checking
        case downloading(Double)      // 0…1
        case loading
        case ready
        case failed(String)
    }

    // MARK: where the person is: the lesson home, inside a lesson, or free Explore
    enum Mode: Equatable { case home, lesson, explore, airline }
    var mode: Mode = .home
    var lessonID: String?
    var pageIndex: Int = 0
    /// True once the current page's action has finished (or it has none) — Continue is enabled.
    var pageReady: Bool = true
    var completedLessons: Set<String> = []   // persisted in UserDefaults "neuroscope.completedLessons"
    /// Which airline-course lesson `.airline` mode shows.
    var airlineLesson: AirlineLessonID = .loop

    // MARK: engine
    var status: EngineStatus = .idle
    @ObservationIgnored var engine: NeuroEngine?
    var geometry: Geometry = ModelStore.spec.geometry
    /// The model's chat template has a thinking mode (Qwen3); OLMo 2 has none, so its thinking UI stays hidden.
    var hasThinking: Bool { ModelStore.spec.hasThinking }

    // MARK: prompt + decoding settings (the next request)
    var prompt: String = "Why is the sky blue? Answer in two sentences."
    var thinking: Bool = ModelStore.spec.hasThinking
    var maxTokens: Int = 512         // derived default for the guided flow (thinking + answer fit)
    var sampling: Sampling = .greedy
    var logitLens: Bool = true
    var paceMs: Int = 0              // slow motion: delay per generated token

    // MARK: runs
    var runs: [Run] = []
    var baselineID: UInt32?          // the run edits are compared against (first unedited run of a prompt)
    var activeRunID: UInt32?         // the run being shown in the field / streams
    var isGenerating: Bool = false
    @ObservationIgnored var nextRunID: UInt32 = 1

    // MARK: playback
    /// Token shown in the field. nil = follow the live/last token of the active run.
    var cursor: Int?
    var isPlaying: Bool = false      // auto-advance cursor through the active run
    var playbackTokensPerSecond: Double = 12

    // MARK: selection + edits
    var selection: Selection?
    var hover: Selection?
    /// Edits applied to the NEXT replay. Each finished run keeps its own copy in run.request.edits.
    var edits = EditSet()
    var neuronInfoCache: [NeuronID: NeuronInfo] = [:]

    // MARK: directions (contrast → abliterate / steer)
    /// One prompt per line. A = prompts WITH the concept, B = matched prompts WITHOUT it.
    var contrastA: String = ""
    var contrastB: String = ""
    var contrast: ContrastResult?
    var isContrasting: Bool = false
    var contrastError: String?
    /// The residual site whose direction is chosen (0 = embedding, l+1 = after layer l). Default: best separation.
    var contrastSite: Int?
    /// Per layer, per neuron (x, y) in [-1, 1]² from co-activation clustering; nil until computed.
    var clusterLayout: [[SIMD2<Float>]]?
    var isClustering: Bool = false
    var showGuide: Bool = false
    var rightPanel: RightPanel = .inspector
    enum RightPanel: String, CaseIterable { case inspector = "Inspector", directions = "Directions" }

    // MARK: guided flow (the beginner path; everything else lives behind showAdvanced)
    enum GuideStep: Int, CaseIterable, Identifiable {
        case ask, watch, change, compare
        var id: Int { rawValue }
        var title: String { ["Ask", "Watch", "Change", "Compare"][rawValue] }
    }
    enum ExperimentStatus: Equatable {
        case idle
        case preparing(String)   // plain-English progress, e.g. "Finding where the model keeps this idea…"
        case ready               // edits hold the derived change; tryIt() will run Before/After
        case failed(String)
    }
    var step: GuideStep = .ask
    var showAdvanced: Bool = false
    var experiment: Experiment?
    var experimentStatus: ExperimentStatus = .idle
    /// The derived residual site + strength for the chosen experiment, shown in Advanced.
    var experimentSite: Int?
    var experimentValue: Float?
    /// Plain caveat when the derivation found only a weak effect (shown under the experiment), else nil.
    var experimentNote: String?
    /// Concept changes are compared with thinking OFF: with thinking on the model re-reads the request in words
    /// and talks itself back into the idea (measured), which hides the change from a beginner.
    var comparisonThinking: Bool { edits.directions.isEmpty ? thinking : false }

    var beforeRun: Run? { baselineRun }
    /// The newest run that carries edits and shares the baseline's prompt.
    var afterRun: Run? {
        guard let b = baselineRun else { return nil }
        return runs.last { $0.id != b.id && !$0.edits.isEmpty && $0.request.prompt == b.request.prompt }
    }

    // MARK: slow motion (ultraslow trace of one word through the floors)
    /// The arrows for the word being slowed down; nil when slow motion is off.
    var trace: TraceResult?
    var isTracing: Bool = false            // computing a trace
    var slowMotion: Bool = false           // slow-motion view is showing (layout .columns, arrows drawn)
    /// Floors revealed so far, 0…numLayers (fractional while an arrow is mid-flight). Animated by AppState.
    var traceProgress: Double = 0
    var traceFloorsPerSecond: Double = 0.5
    var tracePaused: Bool = false
    /// The desk being explained ("why did this desk fire?"); its incoming arrows from every floor are drawn.
    var traceFocus: NeuronID?
    /// Real seconds one floor takes when the model WRITES a word: the run's measured decode speed spread over the
    /// floors (~0.3 ms at ~120 words/s). The trace's own forward_ms is NOT used: it re-reads the whole sentence and
    /// includes warm-up, which understated the slowdown about 100×.
    var realSecondsPerFloor: Double {
        let floors = Double(max(geometry.numLayers, 1))
        if let tps = activeRun?.summary?.decodeTokPerS, tps > 0 { return 1 / tps / floors }
        return 0.008 / floors
    }

    /// How many times slower than real the playback is.
    var slowdownFactor: Double {
        guard trace != nil, traceFloorsPerSecond > 0 else { return 0 }
        return (1 / traceFloorsPerSecond) / realSecondsPerFloor
    }

    // MARK: follow one desk (its whole story at one word, step by step)
    var follow: FollowResult?
    var isFollowing: Bool = false          // computing
    /// Which of the 7 story steps is shown (0 reads the memo … 6 effect on the final word).
    var followStep: Int = 0

    // MARK: the math (real numbers behind every change, for the "Show the math" panels)
    var showMath: Bool = false
    /// The floor tryouts from the last derivation (one row per candidate), so the math panel can show the scores.
    var tryouts: [Tryout] = []
    /// Ink per floor for the Before and After runs, measured along the current direction edit's d̂.
    var inkBefore: InkCurve?
    var inkAfter: InkCurve?
    /// Worked example with real numbers at the chosen floor, first reply word of the After run.
    var mathSample: ProjectSample?
    var isMeasuringMath: Bool = false

    // MARK: view
    var view = ViewSettings()
    /// Bumped whenever what the field should draw changes (new frame, cursor, run, mode, markers).
    /// The renderer re-uploads only when this differs from what it last drew.
    var displayVersion: Int = 0
    /// True for the instant a NEW live frame arrived — the renderer fires a glow pulse instead of snapping.
    @ObservationIgnored var pulsePending: Bool = false

    // MARK: derived
    var activeRun: Run? { runs.first { $0.id == activeRunID } }
    var baselineRun: Run? { runs.first { $0.id == baselineID } }
    /// The token index currently displayed (cursor, else the active run's last token). A finished run rests on its
    /// last real word, not the closing control token: OLMo 2 ends a reply with <|endoftext|>, the same token that
    /// opens every prompt, and the building barely works on it (every floor near grey), which reads as "nothing".
    var displayIndex: Int? {
        guard let r = activeRun, r.count > 0 else { return nil }
        return min(cursor ?? restingIndex(r), r.count - 1)
    }

    /// Last token worth showing at rest: the final non-special token of a finished run (the live token while running).
    func restingIndex(_ r: Run) -> Int {
        guard r.status != .running, let last = r.tokens.last, last.stream == .special, r.count > 1 else { return r.count - 1 }
        return r.tokens.indices.last { r.tokens[$0].stream != .special } ?? r.count - 1
    }

    init() {
        completedLessons = Set(UserDefaults.standard.stringArray(forKey: "neuroscope.completedLessons") ?? [])
        // Derived defaults (never asked on the beginner path): thinking on when the model has it, 512 tokens, greedy,
        // logit lens on, heads hidden. Advanced can change all of them.
        view.showHeads = false
    }

    // MARK: private state
    @ObservationIgnored private var baselineCounter = 0
    @ObservationIgnored private var playTask: Task<Void, Never>?
    @ObservationIgnored private var scaleCache: [UInt32: ScaleCacheEntry] = [:]
    /// Bumped by startOver(); async flow steps that resumed after it bail out instead of acting on a clean slate.
    @ObservationIgnored private var flowEpoch = 0
    /// Derived change per experiment id (the contrast depends only on the model and the fixed prompt lists).
    @ObservationIgnored private var experimentCache: [String: DerivedChange] = [:]
    // slow motion: playback loop, staleness counter, and the view to restore afterwards
    @ObservationIgnored private var slowTask: Task<Void, Never>?
    @ObservationIgnored private var followAnim: Task<Void, Never>?
    @ObservationIgnored private var followEpoch = 0
    @ObservationIgnored private var slowEpoch = 0
    @ObservationIgnored private var savedLayout: LayoutMode?
    @ObservationIgnored private var savedColorMode: ColorMode?
    /// Lesson pages: bumped on every page change so async work from a page that was left bails out.
    @ObservationIgnored private var pageEpoch = 0
    @ObservationIgnored private var pageTask: Task<Void, Never>?
    @ObservationIgnored private var lessonAnim: Task<Void, Never>?
    /// "lessonID/pageID" of pages whose heavy action has finished (Back does not re-run those).
    @ObservationIgnored private var lessonDonePages: Set<String> = []
    /// Bumped by clearMath(); a measureMath() that resumes after it drops its now-stale numbers.
    @ObservationIgnored private var mathEpoch = 0

    private struct DerivedChange {
        var contrast: ContrastResult
        var site: Int
        var value: Float
        var tryouts: [Tryout] = []
    }

    private struct ScaleCacheEntry {
        var count: Int
        var layer: [Float]
        var head: [Float]
        var prefillDone: Bool
    }

    // MARK: lifecycle
    /// Called at launch: if the model is already on disk, load it quietly; never download here.
    func launch() {
        guard status == .idle, ModelStore.isInstalled() else { return }
        Task { await bootstrap() }
    }

    /// Ensure the model is on disk (download with progress if not), then load it. Idempotent: does nothing while
    /// a bootstrap is running or once ready; after a failure it retries. Called when a real-AI lesson or Explore starts.
    func bootstrap() async {
        switch status {
        case .idle, .failed: break
        default: return
        }
        status = .checking
        do {
            if !ModelStore.isInstalled() {
                status = .downloading(0)
                try await ModelStore.download { [weak self] p in
                    self?.status = .downloading(max(0, min(1, p)))
                }
            }
            status = .loading
            let dir = ModelStore.modelDirectory
            let eng = try await Task.detached(priority: .userInitiated) {
                try NeuroEngine(modelDir: dir)
            }.value
            engine = eng
            geometry = eng.geometry
            status = .ready
            // A lesson started while the model was still coming up begins now.
            if mode == .lesson, lessonID != nil { applyPage() }
        } catch {
            status = .failed(error.localizedDescription)
        }
    }

    // MARK: running
    /// Runs the current prompt WITHOUT edits and makes it the baseline.
    func runBaseline() {
        guard status == .ready, !isGenerating, !isContrasting else { return }
        start(makeBaselineRun())
    }

    /// Creates (and registers as the baseline) an unedited run of the current prompt + settings.
    private func makeBaselineRun() -> Run {
        let req = GenerationRequest(prompt: prompt, thinking: comparisonThinking, maxTokens: maxTokens,
                                    logitLens: logitLens, paceMs: paceMs, sampling: sampling, edits: EditSet())
        baselineCounter += 1
        let label = baselineCounter == 1 ? "Baseline" : "Baseline \(baselineCounter)"
        let run = makeRun(label: label, request: req)
        baselineID = run.id
        return run
    }

    /// Re-runs the baseline's prompt + settings WITH the current edits (same seed), shows it, and keeps
    /// the baseline for comparison. Falls back to the current prompt when there is no baseline.
    func replayWithEdits() {
        guard status == .ready, !isGenerating, !isContrasting else { return }
        start(makeEditedRun())
    }

    /// Creates an edited run: the baseline's prompt + settings (or the current ones) with the current edits.
    private func makeEditedRun() -> Run {
        var req: GenerationRequest
        if let b = baselineRun {
            req = b.request
            req.logitLens = logitLens
            req.paceMs = paceMs
        } else {
            req = GenerationRequest(prompt: prompt, thinking: thinking, maxTokens: maxTokens,
                                    logitLens: logitLens, paceMs: paceMs, sampling: sampling, edits: EditSet())
        }
        req.edits = edits
        let n = edits.count
        return makeRun(label: "Edited - \(n) edit\(n == 1 ? "" : "s")", request: req)
    }

    private func makeRun(label: String, request: GenerationRequest) -> Run {
        let id = nextRunID
        nextRunID &+= 1
        let run = Run(id: id, label: label, request: request, geometry: geometry)
        runs.append(run)
        return run
    }

    private func start(_ run: Run) {
        guard engine != nil else { return }
        beginGenerating(run)
        Task { @MainActor [weak self] in await self?.finishGenerating(run) }
    }

    /// Synchronous half of a generation: shows the run and marks the engine busy, so every guard holds
    /// from this instant (no gap between a caller's check and the engine starting).
    private func beginGenerating(_ run: Run) {
        stopSlowMotion()
        stopPlayback()
        activeRunID = run.id
        cursor = nil
        isGenerating = true
        bumpDisplay()
    }

    /// Awaitable half: streams the run to completion and records its status. Returns when isGenerating is false.
    private func finishGenerating(_ run: Run) async {
        guard let engine else { isGenerating = false; run.status = .failed("engine not loaded"); return }
        do {
            let summary = try await engine.generate(runID: run.id, request: run.request) { [weak self] frame in
                run.append(frame)
                guard let self else { return }
                self.bumpDisplay(pulse: self.cursor == nil && self.activeRunID == run.id)
            }
            run.summary = summary
            if summary.stop == "cancelled" {
                run.status = .cancelled
            } else if !summary.ok {
                run.status = .failed(summary.error ?? "generation failed")
            } else {
                run.status = .finished
            }
        } catch {
            run.status = .failed(error.localizedDescription)
        }
        isGenerating = false
        bumpDisplay()
    }

    /// Generates `run` and waits for it (beginGenerating + finishGenerating).
    private func generate(_ run: Run) async {
        beginGenerating(run)
        await finishGenerating(run)
    }

    func cancel() {
        engine?.cancel()
    }

    func showRun(_ id: UInt32) {
        guard runs.contains(where: { $0.id == id }) else { return }
        if id != activeRunID { stopSlowMotion() }
        stopPlayback()
        activeRunID = id
        cursor = nil
        bumpDisplay()
    }

    func deleteRun(_ id: UInt32) {
        guard let i = runs.firstIndex(where: { $0.id == id }) else { return }
        if isGenerating && activeRunID == id && runs[i].status == .running { engine?.cancel() }
        runs.remove(at: i)
        scaleCache[id] = nil
        if baselineID == id { baselineID = nil }
        if activeRunID == id {
            stopSlowMotion()
            stopPlayback()
            activeRunID = runs.last?.id
            cursor = nil
        }
        bumpDisplay()
    }

    // MARK: playback
    func setCursor(_ t: Int?) {
        if let t, let r = activeRun, r.count > 0 {
            cursor = max(0, min(t, r.count - 1))
        } else {
            cursor = nil
        }
        bumpDisplay()
    }

    func togglePlay() {
        if isPlaying { stopPlayback(); return }
        guard let r = activeRun, r.count > 0 else { return }
        let end = restingIndex(r)
        if cursor == nil || cursor! >= end { cursor = 0; bumpDisplay(pulse: true) }
        isPlaying = true
        playTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                let tps = max(0.5, self?.playbackTokensPerSecond ?? 12)
                try? await Task.sleep(nanoseconds: UInt64(1_000_000_000 / tps))
                guard !Task.isCancelled, let self else { return }
                guard let run = self.activeRun, let c = self.cursor, c + 1 <= self.restingIndex(run) else {
                    self.isPlaying = false
                    self.playTask = nil
                    return
                }
                self.cursor = c + 1
                self.bumpDisplay(pulse: true)
            }
        }
    }

    private func stopPlayback() {
        playTask?.cancel()
        playTask = nil
        if isPlaying { isPlaying = false }
    }

    // MARK: selection
    func select(_ s: Selection?) {
        selection = s
        bumpDisplay()
    }

    /// Fetches (and caches) logit attribution for a neuron off the main thread.
    func neuronInfo(_ n: NeuronID) async -> NeuronInfo? {
        if let c = neuronInfoCache[n] { return c }
        guard let engine else { return nil }
        let fetch: () async -> NeuronInfo? = {
            await Task.detached(priority: .userInitiated) { engine.neuronInfo(n) }.value
        }
        var info = await fetch()
        if info == nil && isGenerating {
            var waited = 0
            while isGenerating && waited < 600 {
                try? await Task.sleep(nanoseconds: 100_000_000)
                waited += 1
            }
            info = await fetch()
        }
        if let info { neuronInfoCache[n] = info }
        return info
    }

    // MARK: edits (one edit per unit; setting a new one replaces the old)
    func edit(for n: NeuronID) -> NeuronEdit? {
        edits.neurons.first { $0.layer == n.layer && $0.index == n.index }
    }

    func edit(for h: HeadID) -> HeadEdit? {
        edits.heads.first { $0.layer == h.layer && $0.head == h.head }
    }

    func setEdit(_ n: NeuronID, mode: EditMode, value: Float) {
        let v: Float = mode == .ablate ? 0 : value
        let e = NeuronEdit(layer: n.layer, index: n.index, mode: mode, value: v)
        if let i = edits.neurons.firstIndex(where: { $0.layer == n.layer && $0.index == n.index }) {
            edits.neurons[i] = e
        } else {
            edits.neurons.append(e)
        }
        bumpDisplay()
    }

    func setEdit(_ h: HeadID, mode: EditMode, value: Float) {
        guard mode != .pin else { return }
        let v: Float = mode == .ablate ? 0 : value
        let e = HeadEdit(layer: h.layer, head: h.head, mode: mode, value: v)
        if let i = edits.heads.firstIndex(where: { $0.layer == h.layer && $0.head == h.head }) {
            edits.heads[i] = e
        } else {
            edits.heads.append(e)
        }
        bumpDisplay()
    }

    func removeEdit(_ s: Selection) {
        switch s {
        case .neuron(let n): edits.neurons.removeAll { $0.layer == n.layer && $0.index == n.index }
        case .head(let h): edits.heads.removeAll { $0.layer == h.layer && $0.head == h.head }
        }
        bumpDisplay()
    }

    func clearEdits() {
        edits = EditSet()
        bumpDisplay()
    }

    func saveEdits(to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(edits).write(to: url, options: .atomic)
    }

    func loadEdits(from url: URL) throws {
        let data = try Data(contentsOf: url)
        var loaded = try JSONDecoder().decode(EditSet.self, from: data)
        // Normalize: one edit per unit (last wins), no pinned heads, ablate value 0.
        var seenN = Set<NeuronID>(), seenH = Set<HeadID>()
        loaded.neurons = loaded.neurons.reversed().filter { seenN.insert($0.id).inserted }.reversed()
            .map { var e = $0; if e.mode == .ablate { e.value = 0 }; return e }
        loaded.heads = loaded.heads.reversed().filter { $0.mode != .pin && seenH.insert($0.id).inserted }.reversed()
            .map { var e = $0; if e.mode == .ablate { e.value = 0 }; return e }
        edits = loaded
        bumpDisplay()
    }

    /// Per-layer brightness normalizers for a run (Analysis.layerScales / headScales), cached and refreshed
    /// as the run grows (after prefill, then every ~32 tokens, and when it finishes). Cheap to call per frame.
    func scales(for run: Run) -> (layer: [Float], head: [Float]) {
        let count = run.count
        let finished = run.status != .running
        if let c = scaleCache[run.id] {
            let pc = count > 0 ? run.promptCount : 0
            let afterPrefill = !c.prefillDone && pc > 0 && count == pc
            let grew = count >= c.count + 32
            let finalize = finished && c.count != count
            if !(afterPrefill || grew || finalize) { return (c.layer, c.head) }
        }
        let fallbackL = [Float](repeating: 1, count: run.geometry.numLayers)
        guard count > 0 else {
            scaleCache[run.id] = ScaleCacheEntry(count: 0, layer: fallbackL, head: fallbackL, prefillDone: false)
            return (fallbackL, fallbackL)
        }
        let l = Analysis.layerScales(run)
        let h = Analysis.headScales(run)
        let pc = run.promptCount
        let prefillDone = (scaleCache[run.id]?.prefillDone ?? false) || (pc > 0 && count >= pc)
        scaleCache[run.id] = ScaleCacheEntry(count: count, layer: l, head: h, prefillDone: prefillDone)
        return (l, h)
    }

    /// Runs ns_contrast on the A/B prompt lists (non-empty lines), stores the result, picks the site with the
    /// highest separation as contrastSite, switches view.colorMode to .contrast and bumps the display.
    func computeContrast() async {
        guard status == .ready, !isGenerating, !isContrasting, let engine else { return }
        stopSlowMotion()
        func lines(_ s: String) -> [String] {
            s.split(whereSeparator: \.isNewline)
                .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        }
        let a = lines(contrastA), b = lines(contrastB)
        guard !a.isEmpty, !b.isEmpty else {
            contrastError = "Write at least one prompt in each list (one per line)."
            return
        }
        isContrasting = true
        contrastError = nil
        defer { isContrasting = false }
        do {
            let res = try await engine.contrast(ContrastRequest(a: a, b: b, thinking: false))
            contrast = res
            var best = 0
            for (i, s) in res.separation.enumerated() where s.isFinite && s > (res.separation[best].isFinite ? res.separation[best] : -.infinity) {
                best = i
            }
            contrastSite = best
            view.colorMode = .contrast
            bumpDisplay()
        } catch {
            contrastError = error.localizedDescription
        }
    }

    /// Adds a DirectionEdit from contrast.directions[site] (label e.g. "A-B @ L14", sites [] = all when
    /// allSites, else [site]); mode .ablate (abliteration) or .add with value alpha. Bumps display.
    func addDirectionEdit(site: Int, mode: DirectionMode, value: Float, allSites: Bool) {
        guard let c = contrast, site >= 0, site < c.directions.count else { return }
        let siteName = site == 0 ? "embed" : "L\(site - 1)"
        let verb = mode == .ablate ? "ablate" : "add \(value)x"
        let label = "\(verb) A\u{2212}B @ \(allSites ? "all sites (dir from \(siteName))" : siteName)"
        edits.directions.append(DirectionEdit(label: label, vector: c.directions[site],
                                              sites: allSites ? [] : [site], mode: mode,
                                              value: mode == .ablate ? 0 : value))
        bumpDisplay()
    }

    func removeDirectionEdit(_ id: UUID) {
        edits.directions.removeAll { $0.id == id }
        bumpDisplay()
    }

    /// Computes clusterLayout from the active run (Analysis.clusterLayout off the main actor), then
    /// switches view.layout to .cluster.
    func computeClusterLayout() async {
        guard let run = activeRun, run.count > 0, !isClustering else { return }
        isClustering = true
        defer { isClustering = false }
        // ContiguousArray is copy-on-write: this is a cheap snapshot the detached task can read safely
        // while the run keeps appending on the main actor.
        let g = run.geometry, count = run.count, buf = run.neurons
        let layout = await Task.detached(priority: .userInitiated) {
            Analysis.clusterLayout(geometry: g, count: count, neurons: buf)
        }.value
        clusterLayout = layout
        view.layout = .cluster
        bumpDisplay()
    }

    // MARK: guided flow actions
    /// Step 1 → 2: runs the prompt as the Before run (no edits) and moves to .watch as soon as it starts.
    /// After it finishes, computes the cluster layout automatically and switches the field to it.
    func ask() {
        guard status == .ready, !isGenerating, !isContrasting, !isPreparing else { return }
        stopFollow()
        let epoch = flowEpoch
        let run = makeBaselineRun()
        view.colorMode = .activation
        step = .watch
        beginGenerating(run)
        Task { @MainActor [weak self] in
            guard let self else { return }
            await self.finishGenerating(run)
            guard epoch == self.flowEpoch else { return }
            if run.status == .finished, self.clusterLayout == nil, self.activeRunID == run.id {
                await self.computeClusterLayout()
                guard epoch == self.flowEpoch else { return }
            }
            if self.clusterLayout != nil { self.view.layout = .cluster }
            if self.step == .watch { self.view.colorMode = .activation }
            self.bumpDisplay()
        }
    }

    private var isPreparing: Bool {
        if case .preparing = experimentStatus { return true }
        return false
    }

    func choose(_ e: Experiment) async {
        guard status == .ready, !isGenerating, !isContrasting, !isPreparing, let engine else { return }
        stopFollow()
        let epoch = flowEpoch
        experiment = e
        clearMath()
        experimentStatus = .preparing("Sending matched pairs of questions and comparing the memos to find the highlighter\u{2026}")
        isContrasting = true
        contrastError = nil
        defer { isContrasting = false }
        // Show the lists in Advanced (Directions) so an expert can see exactly what was compared.
        contrastA = e.withConcept.joined(separator: "\n")
        contrastB = e.withoutConcept.joined(separator: "\n")
        do {
            let derived: DerivedChange
            if let cached = experimentCache[e.id] {
                derived = cached
            } else {
                derived = try await derive(e, engine: engine, epoch: epoch)
                experimentCache[e.id] = derived
            }
            guard epoch == flowEpoch, experiment?.id == e.id else { return }
            let res = derived.contrast
            experimentSite = derived.site
            experimentValue = derived.value
            tryouts = derived.tryouts
            edits.directions = [DirectionEdit(label: e.title, vector: res.directions[derived.site],
                                              sites: e.mode == .ablate ? [] : [derived.site],
                                              mode: e.mode, value: derived.value)]
            contrast = res
            contrastSite = derived.site
            view.colorMode = .contrast
            step = .change
            experimentStatus = .ready
            bumpDisplay()
        } catch is CancellationError {
            if epoch == flowEpoch { experimentStatus = .idle }
        } catch {
            guard epoch == flowEpoch else { return }
            contrastError = error.localizedDescription
            experimentStatus = .failed("Something went wrong while finding this highlighter. Nothing was changed; please try again.")
        }
    }

    /// Derives the layer (and strength for Add) by BEHAVIOUR: apply each candidate, let the model answer the
    /// held-out probes (thinking off, greedy, short), and keep the candidate whose answers move most toward the
    /// targets' unedited answers. Candidates that echo the question, loop, or (Remove) change an unrelated plain
    /// answer are rejected. See Experiment's derivation note for why representation-only scores were dropped.
    private func derive(_ e: Experiment, engine: NeuroEngine, epoch: Int) async throws -> DerivedChange {
        let res = try await engine.contrast(ContrastRequest(a: e.withConcept, b: e.withoutConcept))
        let L = res.numLayers
        guard L > 0, res.separation.count == L + 1, res.directions.count == L + 1, res.norms.count == L + 1 else {
            throw EngineError.generate("unexpected contrast shape")
        }
        func check() throws { guard epoch == flowEpoch else { throw CancellationError() } }
        func answer(_ q: String, _ ed: EditSet) async throws -> String {
            try check()
            var text = ""
            let req = GenerationRequest(prompt: q, thinking: false, maxTokens: 40, logitLens: false, topK: 1,
                                        paceMs: 0, sampling: .greedy, edits: ed)
            _ = try await engine.generate(runID: 0, request: req) { f in if f.stream == .answer { text += f.text } }
            return text
        }
        experimentStatus = .preparing("Asking fresh questions first, to hear how the building normally replies\u{2026}")
        var probeRef: [String] = [], targetRef: [String] = []
        for (p, t) in zip(e.probes, e.targets) {
            probeRef.append(try await answer(p, EditSet()))
            targetRef.append(try await answer(t, EditSet()))
        }
        let plainQ = Experiment.heldOut[0]
        let plainRef = e.mode == .ablate ? try await answer(plainQ, EditSet()) : ""

        func edit(_ site: Int, _ value: Float) -> EditSet {
            EditSet(directions: [DirectionEdit(label: e.title, vector: res.directions[site],
                                               sites: e.mode == .ablate ? [] : [site], mode: e.mode, value: value)])
        }
        // Score one candidate: mean over probes of sim(edited, target) − sim(edited, probe's own normal answer).
        // Every candidate is recorded as a Tryout (with the first probe's edited answer) for the math panel.
        var tried: [Tryout] = []
        func score(_ site: Int, _ value: Float) async throws -> Float {
            let ed = edit(site, value)
            var total: Float = 0
            var sample = ""
            func record(_ s: Float) -> Float {
                tried.append(Tryout(site: site, value: value, score: s, sampleAnswer: sample))
                return s
            }
            for (i, p) in e.probes.enumerated() {
                let out = try await answer(p, ed)
                if i == 0 { sample = out }
                if TextSim.isBroken(out, question: p) { return record(-1) }
                total += TextSim.similarity(out, targetRef[i]) - TextSim.similarity(out, probeRef[i])
            }
            if e.mode == .ablate {
                let plain = try await answer(plainQ, ed)
                if TextSim.isBroken(plain, question: plainQ) || TextSim.similarity(plain, plainRef) < 0.25 { return record(-1) }
            }
            return record(total / Float(e.probes.count))
        }

        var best = (site: 1, value: Float(0), score: -Float.infinity)
        switch e.mode {
        case .ablate:
            let cap = max(1, min(L, Int((Experiment.removeDepthCap * Float(L)).rounded(.down))))
            let candidates = Array(max(1, cap / 2)...cap)
            for (k, c) in candidates.enumerated() {
                experimentStatus = .preparing("Testing floor \(c - 1) (\(k + 1) of \(candidates.count))\u{2026}")
                let sc = try await score(c, 0)
                if sc > best.score { best = (c, 0, sc) }
            }
        case .add:
            let r = Experiment.addDepthRange
            let lo = max(1, Int((r.lowerBound * Float(L)).rounded())), hi = min(L, Int((r.upperBound * Float(L)).rounded()))
            let sites = Array(lo...max(lo, hi))
            for (k, c) in sites.enumerated() {
                experimentStatus = .preparing("Testing floor \(c - 1) (\(k + 1) of \(sites.count))\u{2026}")
                let v = Experiment.addStrength * res.norms[c]
                let sc = try await score(c, v)
                if sc > best.score { best = (c, v, sc) }
            }
            for m in Experiment.addStrengths where m != Experiment.addStrength {
                experimentStatus = .preparing("Trying lighter and heavier ink on the best floor\u{2026}")
                let v = m * res.norms[best.site]
                let sc = try await score(best.site, v)
                if sc > best.score { best = (best.site, v, sc) }
            }
        }
        // 0.10: OLMo's Remove brevity scores 0.109 while turning one-word replies into clear sentences (the score's
        // length term undercounts a short-vs-long change), so the old 0.15 bar called a plain success "subtle".
        experimentNote = best.score > 0.10 ? nil
            : "Heads up: this small building holds on to this idea tightly. Even the best floor we found changes the reply only a little, because other desks and floors carry the same idea. The difference may be subtle."
        return DerivedChange(contrast: res, site: best.site, value: best.value, tryouts: tried)
    }

    func tryIt() {
        guard status == .ready, !isGenerating, !isContrasting, !isPreparing, !edits.isEmpty else { return }
        stopFollow()
        let epoch = flowEpoch
        let needsBefore = baselineRun?.request.prompt != prompt
            || baselineRun?.request.thinking != comparisonThinking
            || baselineRun.map { $0.status != .finished } ?? true
        let before = needsBefore ? makeBaselineRun() : nil
        if let before { beginGenerating(before) }
        Task { @MainActor [weak self] in
            guard let self else { return }
            if let before {
                await self.finishGenerating(before)
                guard epoch == self.flowEpoch, before.status == .finished else { return }
            }
            let after = self.makeEditedRun()
            self.view.colorMode = .diff
            self.step = .compare
            await self.generate(after)
            guard epoch == self.flowEpoch else { return }
            self.view.colorMode = .diff
            self.bumpDisplay()
            if after.status == .finished, after.edits.directions.contains(where: { !$0.vector.isEmpty }) {
                await self.measureMath()
            }
        }
    }

    func startOver() {
        stopFollow()
        flowEpoch += 1
        if isGenerating { engine?.cancel() }
        stopPlayback()
        edits = EditSet()
        experiment = nil
        experimentStatus = .idle
        experimentSite = nil
        experimentValue = nil
        experimentNote = nil
        clearMath()
        tryouts = []
        runs.removeAll()
        scaleCache.removeAll()
        baselineID = nil
        activeRunID = nil
        baselineCounter = 0
        cursor = nil
        selection = nil
        hover = nil
        contrast = nil
        contrastSite = nil
        contrastError = nil
        view.colorMode = .activation
        step = .ask
        bumpDisplay()
    }

    // MARK: slow motion actions
    /// Traces the active run at token t (default: displayIndex) with the run's own edits, switches the field to
    /// .columns, resets traceProgress to 0 and animates it to numLayers at traceFloorsPerSecond (honouring
    /// tracePaused). Refuses while generating/contrasting/preparing. Keeps the previous layout to restore.
    func startSlowMotion(at t: Int? = nil) async {
        guard status == .ready, !isGenerating, !isContrasting, !isPreparing, let engine,
              let run = activeRun, run.count > 0, let shown = displayIndex else { return }
        let position = max(0, min(t ?? (cursor == nil ? defaultSlowMotionPosition(run) ?? shown : shown), run.count - 1))
        slowTask?.cancel()
        slowTask = nil
        slowEpoch += 1
        let epoch = slowEpoch
        isTracing = true
        let request = TraceRequest(tokenIDs: run.tokens.map(\.tokenID), position: position, edits: run.edits)
        let result: TraceResult
        do {
            result = try await engine.trace(request)
        } catch {
            if epoch == slowEpoch { isTracing = false }
            return
        }
        guard epoch == slowEpoch else { return }   // stopped, restarted or started over meanwhile
        isTracing = false
        clearFollowState()
        if savedLayout == nil { savedLayout = view.layout; savedColorMode = view.colorMode }
        trace = result
        traceFocus = nil
        tracePaused = false
        traceProgress = 0
        cursor = position
        view.layout = .columns
        slowMotion = true
        bumpDisplay()
        runProgress(epoch: epoch)
    }

    /// Ends slow motion: stops the animation, clears trace/focus, restores the previous layout and color mode.
    func stopSlowMotion() {
        clearFollowState()
        guard slowMotion || isTracing || trace != nil || slowTask != nil || savedLayout != nil else { return }
        slowTask?.cancel()
        slowTask = nil
        slowEpoch += 1
        isTracing = false
        slowMotion = false
        trace = nil
        traceFocus = nil
        traceProgress = 0
        tracePaused = false
        if let l = savedLayout { view.layout = l }
        if let c = savedColorMode { view.colorMode = c }
        savedLayout = nil
        savedColorMode = nil
        bumpDisplay()
    }

    /// Re-traces the same word with `n` as focus (its incoming arrows from every earlier floor) and replays from
    /// the floor just below the first arrow's source. nil clears the focus.
    func explainDesk(_ n: NeuronID?) async {
        if let f = follow {   // following a desk: a click on another desk follows that one at the same word
            if let n, n != NeuronID(layer: f.layer, index: f.index) { await followDesk(n, at: f.position) }
            return
        }
        guard slowMotion, let tr = trace else { return }
        guard let n else {
            traceFocus = nil
            trace?.focusEdges = []
            bumpDisplay()
            return
        }
        guard let engine, let run = activeRun, run.count > 0 else { return }
        slowTask?.cancel()
        slowTask = nil
        slowEpoch += 1
        let epoch = slowEpoch
        isTracing = true
        traceFocus = n
        var request = TraceRequest(tokenIDs: run.tokens.map(\.tokenID), position: min(tr.position, run.count - 1),
                                   edits: run.edits)
        request.focus = [n.layer, n.index]
        let result: TraceResult
        do {
            result = try await engine.trace(request)
        } catch {
            if epoch == slowEpoch {
                isTracing = false
                traceFocus = nil
                runProgress(epoch: epoch)   // resume the playback that was interrupted
            }
            return
        }
        guard epoch == slowEpoch else { return }
        isTracing = false
        trace = result
        let firstSource = result.focusEdges.map(\.fromLayer).min() ?? 0
        traceProgress = Double(max(0, firstSource))
        bumpDisplay()
        runProgress(epoch: epoch)
    }

    /// Animates traceProgress at ~60 Hz up to numLayers + 1 (the last step is the final arrow to the answer).
    private func runProgress(epoch: Int) {
        slowTask?.cancel()
        slowTask = Task { @MainActor [weak self] in
            let clock = ContinuousClock()
            var last = clock.now
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 16_000_000)
                guard !Task.isCancelled, let self, self.slowEpoch == epoch, self.slowMotion else { return }
                let now = clock.now
                let d = last.duration(to: now)
                last = now
                if self.tracePaused { continue }
                let dt = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
                let end = Double(self.geometry.numLayers + 1)
                let next = min(end, self.traceProgress + min(dt, 0.25) * max(0.01, self.traceFloorsPerSecond))
                if next != self.traceProgress {
                    self.traceProgress = next
                    self.bumpDisplay()
                }
                if next >= end { self.slowTask = nil; return }
            }
        }
    }

    // MARK: math actions
    /// Drops the measured math (ink curves, worked sample, tryout rows). Called on a new experiment and startOver.
    private func clearMath() {
        mathEpoch += 1
        tryouts = []
        inkBefore = nil
        inkAfter = nil
        mathSample = nil
        isMeasuringMath = false
    }

    /// True for special tokens rendered literally ("<|im_end|>", "<think>").
    private func isSpecialText(_ s: String) -> Bool {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.count > 2 && t.hasPrefix("<") && t.hasSuffix(">") && !t.contains(" ")
    }

    /// Positions of the reply: tokens on the .answer stream that are not special tokens; falls back to every
    /// non-prompt, non-special token when the run has no answer stream.
    private func replyPositions(_ run: Run) -> [Int] {
        let toks = run.tokens
        let answer = toks.indices.filter { toks[$0].stream == .answer && !isSpecialText(toks[$0].text) }
        if !answer.isEmpty { return answer }
        return toks.indices.filter { toks[$0].stream != .prompt && toks[$0].stream != .special && !isSpecialText(toks[$0].text) }
    }

    /// Slow motion's default word: the first reply word that is real text (not whitespace, not a special token),
    /// else the last non-special token. Never <|im_end|>.
    private func defaultSlowMotionPosition(_ run: Run) -> Int? {
        let toks = run.tokens
        if let i = toks.indices.first(where: {
            toks[$0].stream == .answer && toks[$0].stream != .special && !isSpecialText(toks[$0].text)
                && !toks[$0].text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }) { return i }
        return toks.indices.last { toks[$0].stream != .special && !isSpecialText(toks[$0].text) }
    }

    /// Mean of x·d̂ over `positions` for every site (arriving = before the site's edit, leaving = after).
    private func inkCurve(_ r: ProjectResult, runID: UInt32, positions: [Int]) -> InkCurve {
        let S = r.sites
        let ps = positions.filter { $0 >= 0 && $0 < r.positions && ($0 + 1) * S <= min(r.before.count, r.after.count) }
        var arriving = [Float](repeating: 0, count: S), leaving = [Float](repeating: 0, count: S)
        if !ps.isEmpty {
            for p in ps {
                for s in 0..<S {
                    arriving[s] += r.before[p * S + s]
                    leaving[s] += r.after[p * S + s]
                }
            }
            let n = Float(ps.count)
            for s in 0..<S { arriving[s] /= n; leaving[s] /= n }
        }
        return InkCurve(runID: runID, arriving: arriving, leaving: leaving)
    }

    /// Measures inkBefore/inkAfter (ns_project of the Before and After runs along the current direction edit,
    /// averaged over each run's reply positions) and mathSample (After run, chosen site, first reply word).
    /// Called automatically when an After run finishes and whenever the Compare math panel opens without data.
    func measureMath() async {
        guard status == .ready, !isGenerating, !isMeasuringMath, let engine,
              let before = beforeRun, let after = afterRun,
              before.status == .finished, after.status == .finished else { return }
        // d = the direction edit the After run actually carried (the current edit, as of when it ran), so the
        // numbers always describe the runs on screen.
        guard let dir = after.edits.directions.first(where: { !$0.vector.isEmpty }) else { return }
        let epoch = flowEpoch
        mathEpoch += 1
        let mEpoch = mathEpoch
        isMeasuringMath = true
        defer { if mEpoch == mathEpoch { isMeasuringMath = false } }

        let beforePos = replyPositions(before), afterPos = replyPositions(after)
        let numSites = geometry.numLayers + 1
        let rawSite = experimentSite ?? dir.sites.first ?? contrastSite ?? geometry.numLayers / 2
        let site = max(0, min(rawSite, numSites - 1))
        let samplePos = afterPos.first ?? max(0, after.count - 1)

        let reqBefore = ProjectRequest(tokenIDs: before.tokens.map(\.tokenID), edits: before.edits, direction: dir.vector)
        var reqAfter = ProjectRequest(tokenIDs: after.tokens.map(\.tokenID), edits: after.edits, direction: dir.vector)
        reqAfter.sampleSite = site
        reqAfter.samplePosition = samplePos
        do {
            let rb = try await engine.project(reqBefore)
            guard epoch == flowEpoch, mEpoch == mathEpoch else { return }
            let ra = try await engine.project(reqAfter)
            guard epoch == flowEpoch, mEpoch == mathEpoch else { return }
            inkBefore = inkCurve(rb, runID: before.id, positions: beforePos)
            inkAfter = inkCurve(ra, runID: after.id, positions: afterPos)
            mathSample = ra.sample
        } catch {
            // Leave the math empty; the panel shows the formulas without this run's numbers.
            return
        }
    }

    /// A custom concept from two phrases: matched pairs = phrase + each of Experiment.questions vs opposite +
    /// the same question. Returns an Experiment the normal choose() pipeline can run (derivation, tryouts, math).
    func customExperiment(concept: String, opposite: String, mode: DirectionMode, testQuestion: String) -> Experiment {
        let cp = Experiment.prefix(from: concept)
        let op = Experiment.prefix(from: opposite)
        let name = Experiment.label(from: concept)
        let other = Experiment.label(from: opposite)
        var q = testQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        if q.isEmpty { q = "How do volcanoes erupt?" }
        let lname = name.isEmpty ? "your idea" : "\u{201C}\(name)\u{201D}"
        let lother = other.isEmpty ? "its opposite" : "\u{201C}\(other)\u{201D}"
        let title = (mode == .ablate ? "Remove: " : "Add: ") + (name.isEmpty ? "custom idea" : name)
        let summary = mode == .ablate
            ? "Erase the \(lname) highlighter on every floor, so a request for it gets an answer more like \(lother)."
            : "Ink the \(lname) highlighter on one floor, so a plain question gets an answer as if you had asked for it."
        return Experiment(id: Experiment.customID([mode.rawValue, cp, op, q]),
                          title: title, summary: summary, mode: mode,
                          conceptPrefix: cp, oppositePrefix: op,
                          testQuestion: mode == .ablate ? cp + q : q)
    }

    /// Manual override from the custom builder: apply the current experiment's direction at `site` (erase: every
    /// site; add: that site with `value`), skipping the tryouts. Updates edits, experimentSite/Value.
    func applyManual(site: Int, value: Float) {
        guard let c = contrast, site >= 0, site < c.directions.count else { return }
        let mode = experiment?.mode ?? edits.directions.first?.mode ?? .ablate
        let v: Float = mode == .ablate ? 0 : value
        let label = experiment?.title ?? (mode == .ablate ? "Erase" : "Add")
        edits.directions = [DirectionEdit(label: label, vector: c.directions[site],
                                          sites: mode == .ablate ? [] : [site], mode: mode, value: v)]
        experimentSite = site
        experimentValue = v
        contrastSite = site
        experimentStatus = .ready
        bumpDisplay()
    }

    // MARK: follow actions
    /// Computes the story of desk `n` at token t (default: displayIndex, else the first reply word) of the active run
    /// with the run's edits; enters the slow-motion diagram (layout .columns) showing the desk as the focus with
    /// trace = { edges: heardBy + path (outbound), focusEdges: askedBy (inbound) }; followStep = 0 and traceProgress
    /// is driven by followStep (steps 0–3 reveal up to the desk's floor, 4–5 beyond it along the path, 6 the top).
    func followDesk(_ n: NeuroScope.NeuronID, at t: Int? = nil) async {
        guard status == .ready, !isGenerating, !isContrasting, !isPreparing, let engine,
              let run = activeRun, run.count > 0 else { return }
        let fallback = defaultSlowMotionPosition(run) ?? displayIndex ?? 0
        let position = max(0, min(t ?? cursor ?? fallback, run.count - 1))
        slowTask?.cancel()
        slowTask = nil
        slowEpoch += 1
        followEpoch += 1
        let epoch = slowEpoch, fEpoch = followEpoch
        isTracing = true
        isFollowing = true
        let request = FollowRequest(tokenIDs: run.tokens.map(\.tokenID), position: position, edits: run.edits,
                                    layer: n.layer, index: n.index)
        let result: FollowResult
        do {
            result = try await engine.follow(request)
        } catch {
            if fEpoch == followEpoch { isFollowing = false }
            if epoch == slowEpoch { isTracing = false }
            return
        }
        guard epoch == slowEpoch, fEpoch == followEpoch else { return }   // stopped or superseded meanwhile
        isTracing = false
        isFollowing = false
        if savedLayout == nil { savedLayout = view.layout; savedColorMode = view.colorMode }
        follow = result
        followStep = 0
        trace = TraceResult(position: result.position, tokenID: result.tokenID, edges: result.heardBy + result.path,
                            focusEdges: result.askedBy, explained: [], forwardMs: 0)
        traceFocus = n
        tracePaused = true
        cursor = result.position
        view.layout = .columns
        slowMotion = true
        let start = followTarget(0, result)
        traceProgress = start
        bumpDisplay()
    }

    func setFollowStep(_ k: Int) {
        guard let f = follow else { return }
        let step = max(0, min(6, k))
        followStep = step
        animateProgress(to: followTarget(step, f))
    }

    func stopFollow() {
        clearFollowState()
        stopSlowMotion()
    }

    /// Drops the follow result and cancels its animation (does not touch the slow-motion view).
    private func clearFollowState() {
        followAnim?.cancel()
        followAnim = nil
        followEpoch += 1
        follow = nil
        isFollowing = false
        followStep = 0
    }

    /// Where traceProgress rests for each story step: 0-3 the desk's floor, 4 half a floor above, 5 one past the
    /// last path hop's target floor, 6 the very top (numLayers + 1).
    private func followTarget(_ step: Int, _ f: FollowResult) -> Double {
        let top = Double(geometry.numLayers + 1)
        let v: Double
        switch step {
        case 0...3: v = Double(f.layer)
        case 4: v = Double(f.layer) + 0.5
        case 5: v = Double((f.path.last?.toLayer ?? f.layer)) + 1
        default: v = top
        }
        return max(0, min(top, v))
    }

    /// Eases traceProgress to `target` over ~0.6 s (smoothstep) at ~60 Hz.
    private func animateProgress(to target: Double) {
        followAnim?.cancel()
        let from = traceProgress
        guard abs(target - from) > 1e-6 else { followAnim = nil; return }
        let epoch = followEpoch
        followAnim = Task { @MainActor [weak self] in
            let clock = ContinuousClock()
            let t0 = clock.now
            let duration = 0.6
            while !Task.isCancelled {
                guard let self, self.followEpoch == epoch, self.slowMotion else { return }
                let d = t0.duration(to: clock.now)
                let el = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
                let u = min(1, el / duration)
                let e = u * u * (3 - 2 * u)
                self.traceProgress = from + (target - from) * e
                self.bumpDisplay()
                if u >= 1 { self.followAnim = nil; return }
                try? await Task.sleep(nanoseconds: 16_000_000)
            }
        }
    }

    // MARK: lessons
    var currentLesson: Lesson? { Lessons.all.first { $0.id == lessonID } }
    var currentPage: LessonPage? {
        guard let l = currentLesson, l.pages.indices.contains(pageIndex) else { return nil }
        return l.pages[pageIndex]
    }

    /// Enters a lesson at page 0 (mode .lesson), applying the page's field preset and running its action.
    func startLesson(_ id: String) {
        guard Lessons.all.contains(where: { $0.id == id }) else { return }
        mode = .lesson
        lessonID = id
        pageIndex = 0
        if status == .ready { applyPage() } else { pageReady = true; Task { await bootstrap() } }
    }

    /// Next page (applies its preset, runs its action, sets pageReady); past the last page marks the lesson complete
    /// and returns home (or, for an .explore page, enters Explore).
    func nextPage() {
        guard mode == .lesson, let lesson = currentLesson else { return }
        if pageIndex + 1 < lesson.pages.count, currentPage?.action != .explore {
            pageIndex += 1
            applyPage()
        } else {
            finishLesson()
        }
    }

    func previousPage() {
        guard mode == .lesson, pageIndex > 0 else { return }
        pageIndex -= 1
        applyPage()
    }

    /// Back to the lesson list; stops anything running, clears highlights.
    func exitLesson() {
        leaveLessonField()
        mode = .home
        lessonID = nil
        pageIndex = 0
        pageReady = true
    }

    func enterExplore() {
        leaveLessonField()
        mode = .explore
        lessonID = nil
        pageIndex = 0
        pageReady = true
        step = .ask
        bumpDisplay()
        if status != .ready { Task { await bootstrap() } }
    }

    // MARK: lesson machinery
    private static let completedKey = "neuroscope.completedLessons"
    /// The question lesson 1 asks; slow-motion and desk pages need a run of it.
    private static let lessonQuestion = Lessons.skyQuestion

    private func markCompleted(_ id: String) {
        completedLessons.insert(id)
        UserDefaults.standard.set(completedLessons.sorted(), forKey: Self.completedKey)
    }

    /// The airline course's exit: a finished lesson is remembered (the Builder is never "finished"); either way
    /// the person lands on Home.
    func exitAirline(completed: Bool) {
        if completed && airlineLesson != .builder { markCompleted(airlineLesson.rawValue) }
        mode = .home
    }

    /// Opens an airline-course lesson (or the Builder). None of them needs the real model.
    func enterAirline(_ lesson: AirlineLessonID = .loop) {
        leaveLessonField()
        lessonID = nil
        pageIndex = 0
        pageReady = true
        airlineLesson = lesson
        mode = .airline
    }

    private func finishLesson() {
        if let id = lessonID {
            markCompleted(id)
        }
        if currentPage?.action == .explore { enterExplore() } else { exitLesson() }
    }

    /// Stops every lesson task, slow motion, follow and any running generation; clears highlights.
    private func leaveLessonField() {
        pageEpoch += 1
        pageTask?.cancel(); pageTask = nil
        lessonAnim?.cancel(); lessonAnim = nil
        if isGenerating { engine?.cancel() }
        stopFollow()
        view.highlightFloors = nil
        view.colorMode = .activation
        view.layout = clusterLayout != nil ? .cluster : .stack
        NotificationCenter.default.post(name: Notification.Name("NeuroScopeResetCamera"), object: nil)
        bumpDisplay()
    }

    private func isSlowMotionAction(_ a: LessonAction) -> Bool {
        switch a {
        case .slowMotion, .slowMotionTo, .followDesk: return true
        default: return false
        }
    }

    /// Applies the page's field preset, then starts its action. Stale async work from a previous page bails out
    /// through pageEpoch.
    private func applyPage() {
        pageEpoch += 1
        let epoch = pageEpoch
        pageTask?.cancel(); pageTask = nil
        lessonAnim?.cancel(); lessonAnim = nil
        guard let page = currentPage, let lid = lessonID else { pageReady = true; return }

        // Leave slow motion / follow unless this page keeps or builds on the columns view.
        switch page.field {
        case .keep, .columns: break
        default: stopFollow()
        }
        applyFieldPreset(page.field, resetCamera: true, action: page.action)

        let key = "\(lid)/\(page.id)"
        switch page.action {
        case .none, .mathToy, .explore:
            pageReady = true
        case .replay:
            if let run = activeRun, run.count > 0 {
                if isPlaying { togglePlay() }
                playbackTokensPerSecond = 6
                setCursor(run.promptCount)
                togglePlay()
            }
            pageReady = true
        default:
            pageReady = lessonDonePages.contains(key) && !isSlowMotionAction(page.action) && stateHolds(page.action)
            if pageReady { return }
            pageTask = Task { @MainActor [weak self] in
                await self?.runPageAction(page, key: key, epoch: epoch)
            }
        }
    }

    private func applyFieldPreset(_ p: FieldPreset, resetCamera: Bool, action: LessonAction) {
        func overview(_ floors: Set<Int>?) {
            view.layout = clusterLayout != nil ? .cluster : .stack
            view.colorMode = .activation
            view.highlightFloors = floors
            if resetCamera {
                NotificationCenter.default.post(name: Notification.Name("NeuroScopeResetCamera"), object: nil)
            }
        }
        switch p {
        case .keep, .hidden: break
        case .overview: overview(Set(0..<max(geometry.numLayers, 1)))   // whole building lit: resting desks visible
        case .floors(let fs): overview(Set(fs))
        case .columns:
            view.highlightFloors = nil
            // A slow-motion action sets the layout itself, so the layout it restores later is the overview.
            if !(isSlowMotionAction(action) && !slowMotion) { view.layout = .columns }
        case .concept:
            view.highlightFloors = nil
            view.colorMode = .contrast
        case .compare:
            view.highlightFloors = nil
            view.colorMode = .diff
        }
        bumpDisplay()
    }

    /// True when the result of a heavy action is still in memory, so Back need not re-run it.
    private func stateHolds(_ a: LessonAction) -> Bool {
        switch a {
        case .ask(let q): return findLessonRun(prompt: q) != nil
        case .findConcept(let id): return experiment?.id == id && experimentStatus == .ready && !edits.isEmpty
        case .tryChange: return afterRun?.status == .finished
        default: return true
        }
    }

    private func findLessonRun(prompt q: String) -> Run? {
        runs.last { $0.request.prompt == q && $0.request.edits.isEmpty && !$0.request.thinking && $0.status == .finished }
    }

    private func lessonPause(_ ms: Int) async {
        try? await Task.sleep(nanoseconds: UInt64(ms) * 1_000_000)
    }

    /// Waits for the engine to be loaded and idle. False when the page was left, or the engine failed.
    private func lessonSettle(_ epoch: Int) async -> Bool {
        while epoch == pageEpoch {
            switch status {
            case .failed: return false
            case .ready:
                if !isGenerating && !isContrasting && !isPreparing { return true }
            default: break
            }
            await lessonPause(80)
        }
        return false
    }

    /// Asks `q` with thinking off for this run only (the person's setting is restored), without changing the step.
    private func lessonAsk(_ q: String, epoch: Int) async -> Bool {
        if let r = findLessonRun(prompt: q) {
            prompt = q
            if activeRunID != r.id { showRun(r.id) }
            return true
        }
        guard await lessonSettle(epoch) else { return false }
        let before = runs.count
        let savedThinking = thinking, savedStep = step
        prompt = q
        thinking = false
        ask()
        thinking = savedThinking
        step = savedStep
        guard runs.count > before, let run = runs.last else { return false }
        while epoch == pageEpoch, run.status == .running || isGenerating { await lessonPause(50) }
        while epoch == pageEpoch, isClustering { await lessonPause(50) }
        return epoch == pageEpoch && run.status == .finished
    }

    /// A finished, unedited run to trace. Desk stories need the lesson question's run; slow motion takes whatever
    /// clean run is showing.
    private func ensureLessonRun(epoch: Int, needsLessonQuestion: Bool) async -> Bool {
        if !needsLessonQuestion, let r = activeRun, r.status == .finished, r.count > 0, r.request.edits.isEmpty { return true }
        return await lessonAsk(Self.lessonQuestion, epoch: epoch)
    }

    private func runPageAction(_ page: LessonPage, key: String, epoch: Int) async {
        func done(_ ok: Bool = true) {
            guard epoch == pageEpoch else { return }
            if ok { lessonDonePages.insert(key) }
            pageReady = true
        }
        switch page.action {
        case .none, .mathToy, .explore, .replay:
            done()

        case .ask(let q):
            let ok = await lessonAsk(q, epoch: epoch)
            guard epoch == pageEpoch else { return }
            // The cluster layout arrives after the run; the overview looks right once it has.
            if ok { applyFieldPreset(page.field, resetCamera: false, action: page.action) }
            done(ok)

        case .slowMotion:
            guard await ensureLessonRun(epoch: epoch, needsLessonQuestion: false), await lessonSettle(epoch) else { return done(false) }
            await beginSlowMotionPage(epoch: epoch)
            done()

        case .slowMotionTo(let floor):
            guard await ensureLessonRun(epoch: epoch, needsLessonQuestion: false), await lessonSettle(epoch) else { return done(false) }
            if !slowMotion || follow != nil { await beginSlowMotionPage(epoch: epoch) }
            guard epoch == pageEpoch, slowMotion else { return done(false) }
            tracePaused = true
            let target = max(0, min(Double(geometry.numLayers + 1), floor))
            await glideProgress(to: target, seconds: 1.5, epoch: epoch)
            done()

        case .followDesk(let layer, let index, let k):
            guard await ensureLessonRun(epoch: epoch, needsLessonQuestion: true), await lessonSettle(epoch) else { return done(false) }
            let desk = NeuronID(layer: layer, index: index)
            if follow.map({ NeuronID(layer: $0.layer, index: $0.index) }) != desk {
                let at = activeRun.flatMap { defaultSlowMotionPosition($0) }
                await followDesk(desk, at: at)
            }
            guard epoch == pageEpoch else { return }
            if follow != nil { setFollowStep(k) }
            done(follow != nil)

        case .findConcept(let id):
            guard let e = Experiment.all.first(where: { $0.id == id }), await lessonSettle(epoch) else { return done(false) }
            let savedStep = step
            await choose(e)
            step = savedStep
            guard epoch == pageEpoch else { return }
            applyFieldPreset(page.field, resetCamera: false, action: page.action)
            done(experimentStatus == .ready)

        case .tryChange:
            guard await lessonSettle(epoch) else { return done(false) }
            if edits.isEmpty, let e = experiment ?? Experiment.all.first {
                await choose(e)
                guard epoch == pageEpoch, await lessonSettle(epoch) else { return done(false) }
            }
            guard let e = experiment, !edits.isEmpty else { return done(false) }
            let savedStep = step
            let startCount = runs.count
            prompt = e.testQuestion
            tryIt()
            step = savedStep
            var waited = 0
            while epoch == pageEpoch {
                let fresh = runs.dropFirst(startCount).last { !$0.request.edits.isEmpty }
                if let fresh, fresh.status != .running, !isGenerating { break }
                if runs.count == startCount, !isGenerating { waited += 1; if waited > 40 { return done(false) } }
                await lessonPause(50)
            }
            guard epoch == pageEpoch else { return }
            step = savedStep
            done(afterRun?.status == .finished)
        }
    }

    /// Slow motion on the first reply word, paused at floor 0.
    private func beginSlowMotionPage(epoch: Int) async {
        let at = activeRun.flatMap { defaultSlowMotionPosition($0) }
        await startSlowMotion(at: at)
        guard epoch == pageEpoch, slowMotion else { return }
        tracePaused = true
        traceProgress = 0
        bumpDisplay()
    }

    /// Eases traceProgress to `target` (smoothstep) at ~60 Hz; stops early when the page is left.
    private func glideProgress(to target: Double, seconds: Double, epoch: Int) async {
        let from = traceProgress
        let clock = ContinuousClock()
        let t0 = clock.now
        while epoch == pageEpoch, slowMotion {
            let d = t0.duration(to: clock.now)
            let el = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
            let u = min(1, el / seconds)
            let e = u * u * (3 - 2 * u)
            traceProgress = from + (target - from) * e
            bumpDisplay()
            if u >= 1 { return }
            await lessonPause(16)
        }
    }

    func bumpDisplay(pulse: Bool = false) {
        pulsePending = pulsePending || pulse
        displayVersion &+= 1
    }
}
