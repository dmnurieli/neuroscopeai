import SwiftUI
import AirlineKit

/// Lesson 9, "Graduation": from the learner's one-gate airline network to a real, fully open language model,
/// OLMo 2 1B (allenai), following the same ideas at full scale. Every model output on pages 1–5 is a RECORDING made on
/// this Mac with the app's engine from a pinned published stage (GraduationRecordings.swift); page 6 asks the
/// final model live. Copy says "similar text appears in its training data", never "where it learned this".
@MainActor @Observable
final class AirlineGraduationLesson: AirlineLessonModel {
    let title = AirlineLessonID.graduation.title
    let pageCount = 7
    private(set) var page = 0
    /// Page 2: how many snapshot rows are showing.
    private(set) var snapshotsShown = 1
    /// Page 2: which sentence the snapshots continue.
    var snapshotPrompt = "blank"
    /// Page 5: which question and which stage are on screen.
    var checkPrompt = "math"
    var checkStage = "dpo"
    let live = GraduationLive()
    let baseDownload = GraduationDownload(model: .base)
    let finalDownload = GraduationDownload(model: .final)
    /// Bumped when a model folder appears or disappears (the views re-read `isInstalled`).
    private(set) var installedEpoch = 0

    var canGoBack: Bool { page > 0 }
    var canWatchAgain: Bool { page == 0 || (page == 1 && snapshotsShown > 1) }

    static let snapshots = Graduation.snapshotStages

    func stop() {
        live.release()
    }

    // MARK: pages

    var copy: AirlinePageCopy {
        switch page {
        case 0:
            let slots = Graduation.vocabSlots.formatted()
            return AirlinePageCopy(
                text: "Your network had one gate: how late. A language model's gate has \(slots) destinations, about one for every **word piece** (a word or part of one) it knows. It picks where the next word goes.",
                detail: "Each word so far comes in as an airport on a huge map, its **embedding** (words used alike sit close together); the hubs combine them, and the gate guesses the next piece. Shown: the busiest destinations after “…safely on the”.",
                button: "Continue")
        case 1:
            let total = Self.snapshots.count
            let r = Graduation.recording(Self.snapshots[min(snapshotsShown, total) - 1], snapshotPrompt)
            let detail: String
            if snapshotsShown < total {
                detail = "The same sentence, given to snapshots taken while OLMo trained. Showing \(snapshotsShown) of \(total): \(r?.label ?? "")."
            } else {
                detail = snapshotSummary
            }
            return AirlinePageCopy(
                text: "**Pretraining** is your hide-and-seek game, played on text: cover the next word, guess it, check, adjust. Trillions of times.",
                detail: detail,
                button: snapshotsShown < total ? "Next snapshot" : "Continue",
                reserveDetail: [snapshotSummary, "The same sentence, given to snapshots taken while OLMo trained. Showing 6 of 6: after 2.1 trillion word pieces."])
        case 2:
            return AirlinePageCopy(
                text: "Pretraining done, it can continue any text, but it isn't trying to help. Ask it a question and it carries on like a web page.",
                detail: baseDetail,
                button: "Continue")
        case 3:
            return AirlinePageCopy(
                text: "**Flight school** (supervised fine-tuning): it plays the same game on \(GraduationData.sftExamples.formatted()) example conversations, questions with good answers written out. Now it answers.",
                detail: sftDetail,
                button: "Continue")
        case 4:
            return AirlinePageCopy(
                text: "The **check ride**: shown \(GraduationData.preferencePairs.formatted()) pairs of answers, one marked better, it leans toward the better kind. Then maths with an answer key.",
                detail: checkDetail,
                button: "Continue",
                reserveDetail: [Self.premiseDetail, mathDetail])
        case 5:
            return liveCopy
        default:
            return AirlinePageCopy(
                text: "You built a network, trained it, tested it and took it apart. The same ideas, at scale, make a real AI.",
                detail: "Next: **Inside a real AI**. Those lessons run on this same model, OLMo 2 1B, and show its hubs lighting up word by word, and its control tower (**attention**) deciding which earlier words matter.",
                button: "Done")
        }
    }

    private var snapshotSummary: String {
        "Nobody wrote it a rule. It only played fill-in-the-blank: first noise, then the commonest little words, then sentences."
    }

    private var baseDetail: String {
        var s = "It reads your question as the top of a forum page and writes what tends to follow one."
        if let n = Graduation.trainingDataCounts["bake a chocolate cake"] {
            s += " Similar text appears in its training data: the words “bake a chocolate cake” appear \(n.formatted()) times in it."
        }
        return s
    }

    private var sftDetail: String {
        var s = "Same question, same settings. It already knew about ovens and cakes from pretraining; flight school taught it the format: you ask, it answers."
        if let r = Graduation.recording("sft", "cake"), !r.output.lowercased().contains("cocoa") {
            s += " Look closely, though: no cocoa in this chocolate cake. A small model sounds sure even when it's wrong."
        }
        return s
    }

    private var mathDetail: String {
        let parts = Graduation.postStages.compactMap { s -> String? in
            guard let r = Graduation.recording(s, "math") else { return nil }
            let t = Graduation.landingTime(in: r.output) ?? "no time"
            return "\(Self.shortName(s)) \(t)\(Graduation.landsRight(r.output) ? " ✓" : "")"
        }
        var s = "The right answer is \(Graduation.mathAnswer). They said: " + parts.joined(separator: " · ") + ". A computer can mark this, so training can reward right answers: **checkable rewards**."
        if let last = Graduation.postStages.last, let r = Graduation.recording(last, "math"), !Graduation.landsRight(r.output) {
            s += " Even so, the final model slips on this one: rewards raise the odds, they don't guarantee."
        }
        return s
    }

    static var premiseDetail: String {
        let stages = Graduation.postStages.compactMap { Graduation.recording($0, "premise") }
        let doubters = stages.filter { Graduation.pushesBack($0.output) }.count
        let verdict = doubters == 0 ? "None of the stages questions it; each one explains it instead."
            : "\(doubters) of \(stages.count) stages question it."
        return "Nobody can see the Great Wall from the Moon, but this question says you can. \(verdict) Rewarding what people like has a catch: a model can learn to agree instead of being right (**sycophancy**, a kind of **reward hacking**)."
    }

    private var checkDetail: String {
        switch checkPrompt {
        case "math": mathDetail
        case "premise": Self.premiseDetail
        default: "OLMo's check ride is **DPO** (direct preference training): it learns straight from the pairs, with no separate robot inspector (a **reward model**, a second model trained to score answers) in between. Each stage starts from the one before it."
        }
    }

    private var liveCopy: AirlinePageCopy {
        let installed = live.choice.isInstalled
        _ = installedEpoch
        let detail: String
        switch live.phase {
        case .loading: detail = "Opening the \(live.choice.name.lowercased()) on this Mac…"
        case .answering: detail = "Answering, one word piece at a time, on this Mac."
        case .failed(let m): detail = "Something went wrong: \(m)"
        default:
            detail = installed
                ? "Type anything. It runs here, on this Mac, nothing sent anywhere. It is small, so check what it says."
                : "The \(live.choice.name.lowercased()) isn't on this Mac yet (\(live.choice.sizeWords))."
        }
        let button: String, enabled: Bool, working: Bool
        if live.isBusy { button = "Answering…"; enabled = false; working = true }
        else if installed && live.hasNewQuestion { button = "Ask"; enabled = true; working = false }
        else { button = "Continue"; enabled = true; working = false }
        return AirlinePageCopy(
            text: "Your turn: ask the finished model a question of your own.",
            detail: detail, button: button, enabled: enabled, working: working,
            reserveDetail: ["Type anything. It runs here, on this Mac, nothing sent anywhere. It is small, so check what it says."])
    }

    static func shortName(_ stage: String) -> String {
        switch stage {
        case "sft": "flight school"
        case "dpo": "check ride"
        case "rlvr1": "checkable rewards"
        case "instruct": "final"
        case "base": "base model"
        default: stage
        }
    }

    // MARK: actions

    func primary() -> Bool {
        switch page {
        case 1 where snapshotsShown < Self.snapshots.count:
            snapshotsShown += 1
            return false
        case 5:
            if live.isBusy { return false }
            if live.choice.isInstalled && live.hasNewQuestion { live.ask(); return false }
        case pageCount - 1:
            return true
        default: break
        }
        page += 1
        if page == 1 { snapshotsShown = 1 }
        return false
    }

    func back() {
        guard page > 0 else { return }
        if page == 5 { live.cancel() }
        page -= 1
    }

    /// Page 1: the fan grows again. Page 2: the snapshots from the first again.
    func watchAgain() {
        if page == 1 { snapshotsShown = 1 }
        replay += 1
    }
    private(set) var replay = 0

    func downloadBase() {
        baseDownload.start { [weak self] in self?.installedEpoch += 1; self?.live.choice = .base }
    }
    func deleteBase() {
        live.release(.base)
        baseDownload.delete()
        if live.choice == .base { live.choice = .final }
        installedEpoch += 1
    }
    func downloadFinal() {
        finalDownload.start { [weak self] in self?.installedEpoch += 1 }
    }
}

// MARK: - the view

struct AirlineGraduationLessonView: View {
    var onExit: (_ completed: Bool) -> Void
    @State private var model = AirlineGraduationLesson()

    var body: some View {
        AirlineLessonShell(model: model, onExit: onExit) {
            Group {
                switch model.page {
                case 0: GateFanView(picks: Graduation.recording(Graduation.baseStage, "blank")?.nextTop ?? [], replay: model.replay)
                case 1: snapshots
                case 2: basePage
                case 3: sftPage
                case 4: checkRide
                case 5: livePage
                default: sendOff
                }
            }
            .padding(.horizontal, 28).padding(.vertical, 14)
            .frame(maxWidth: 1100, maxHeight: .infinity)
        } accessory: { _ in
            if model.page == 5 { liveControls }
        }
    }

    // page 2: watch it learn

    private var snapshots: some View {
        VStack(spacing: 12) {
            Picker("Sentence", selection: Bindable(model).snapshotPrompt) {
                Text("The pilot … landed safely on the").tag("blank")
                Text("The capital of France is").tag("capital")
                Text("How do I bake a chocolate cake?").tag("cake")
            }
            .pickerStyle(.segmented).labelsHidden().frame(maxWidth: 720)
            .accessibilityLabel("Which sentence the snapshots continue")
            ScrollView {
                VStack(spacing: 8) {
                    ForEach(Array(AirlineGraduationLesson.snapshots.enumerated()), id: \.element) { i, s in
                        if i < model.snapshotsShown, let r = Graduation.recording(s, model.snapshotPrompt) {
                            SnapshotRow(recording: r).transition(.opacity.combined(with: .move(edge: .top)))
                        }
                    }
                }
                .animation(.easeOut(duration: 0.35), value: model.snapshotsShown)
            }
            .frame(maxHeight: .infinity)
            Text("Snapshots of allenai/OLMo-2-0425-1B's first training stage (one branch each, pinned commits), recorded on this Mac, greedy, \(Graduation.recording(AirlineGraduationLesson.snapshots[0], "blank")?.maxTokens ?? 0) word pieces each.")
                .font(.system(size: 10)).foregroundStyle(.white.opacity(0.4))
        }
    }

    // page 3: the base model

    private var basePage: some View {
        VStack(spacing: 10) {
            if let r = Graduation.recording(Graduation.baseStage, "cake") {
                TranscriptCard(recording: r, style: .web).frame(maxWidth: 760)
            }
        }
        .frame(maxHeight: .infinity)
    }

    // page 4: flight school

    private var sftPage: some View {
        HStack(alignment: .top, spacing: 16) {
            if let b = Graduation.recording(Graduation.baseStage, "cake") {
                TranscriptCard(recording: b, style: .web, title: "Before: base model", compact: true)
            }
            if let s = Graduation.recording("sft", "cake") {
                TranscriptCard(recording: s, style: .chat, title: "After flight school")
            }
        }
        .frame(maxHeight: .infinity)
    }

    // page 5: the check ride

    private var checkRide: some View {
        VStack(spacing: 10) {
            HStack(spacing: 14) {
                Picker("Question", selection: Bindable(model).checkPrompt) {
                    Text("Maths").tag("math")
                    Text("A false premise").tag("premise")
                    Text("The cake").tag("cake")
                }
                .pickerStyle(.segmented).labelsHidden().frame(maxWidth: 360)
                .accessibilityLabel("Which question")
                Picker("Stage", selection: Bindable(model).checkStage) {
                    ForEach(Graduation.postStages, id: \.self) { s in Text(AirlineGraduationLesson.shortName(s).capitalized).tag(s) }
                }
                .pickerStyle(.segmented).labelsHidden().frame(maxWidth: 480)
                .accessibilityLabel("Which training stage")
            }
            if let r = Graduation.recording(model.checkStage, model.checkPrompt) {
                TranscriptCard(recording: r, style: .chat, title: Graduation.recordings.first { $0.stage == model.checkStage }?.label.capitalizedFirst,
                               verdict: model.checkPrompt == "math" ? MathVerdict(r.output) : nil)
                    .frame(maxWidth: 820)
                    .id(r.stage + r.promptID)
            }
        }
        .padding(.top, 24)
        .frame(maxHeight: .infinity, alignment: .top)
    }

    // page 6: live

    private var livePage: some View {
        let live = model.live
        _ = model.installedEpoch
        return VStack(spacing: 10) {
            if let q = live.askedText {
                LiveAnswerCard(question: q, answer: live.answer, model: live.askedModel ?? .final,
                               phase: live.phase, tokPerS: live.tokPerS, cutOff: live.cutOff)
                    .frame(maxWidth: 820)
            } else {
                Spacer(minLength: 0)
                Image(systemName: "text.bubble").font(.system(size: 40)).foregroundStyle(.white.opacity(0.25))
                Text("Its answer will appear here.").font(.system(size: 14)).foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
        }
        .frame(maxHeight: .infinity)
    }

    private var liveControls: some View {
        let live = model.live
        _ = model.installedEpoch
        return VStack(spacing: 10) {
            if live.choice.isInstalled {
                TextField("For example: Why do planes leave white trails in the sky?", text: Bindable(live).question)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 15))
                    .frame(maxWidth: 520)
                    .onSubmit { live.ask() }
                    .disabled(live.isBusy)
                    .accessibilityLabel("Your question")
            } else {
                downloadRow(model.finalDownload, label: "Download the final model (\(GraduationModel.final.sizeWords))", action: model.downloadFinal)
            }
            HStack(spacing: 10) {
                if GraduationModel.base.isInstalled {
                    Picker("Model", selection: Bindable(live).choice) {
                        Text("Final model").tag(GraduationModel.final)
                        Text("Base model").tag(GraduationModel.base)
                    }
                    .pickerStyle(.segmented).labelsHidden().frame(width: 220)
                    .disabled(live.isBusy)
                    .accessibilityLabel("Which model answers")
                    Button("Delete the base model (\(GraduationModel.base.sizeWords))") { model.deleteBase() }
                        .buttonStyle(PillButtonStyle(quiet: true)).disabled(live.isBusy)
                } else if GraduationModel.final.isInstalled {
                    downloadRow(model.baseDownload, label: "Optional: download the base model to try it yourself (\(GraduationModel.base.sizeWords))",
                                action: model.downloadBase)
                }
            }
        }
    }

    @ViewBuilder private func downloadRow(_ d: GraduationDownload, label: String, action: @escaping () -> Void) -> some View {
        switch d.state {
        case .idle:
            Button(label, action: action).buttonStyle(PillButtonStyle(quiet: true))
        case .downloading(let p):
            HStack(spacing: 10) {
                ProgressView(value: p).frame(width: 180)
                    .accessibilityLabel("Downloading the \(d.model.name.lowercased())")
                Text("\(Int((p * 100).rounded()))% of \(d.model.sizeWords)").font(.system(size: 12, design: .monospaced)).foregroundStyle(.secondary)
                Button("Cancel") { d.cancel() }.buttonStyle(PillButtonStyle(quiet: true))
                    .accessibilityLabel("Cancel the download")
            }
        case .failed(let m):
            HStack(spacing: 10) {
                Text("Download failed: \(m)").font(.system(size: 12)).foregroundStyle(.secondary)
                Button("Try again", action: action).buttonStyle(PillButtonStyle(quiet: true))
            }
        }
    }

    // page 7

    private var sendOff: some View {
        VStack(spacing: 18) {
            Spacer(minLength: 0)
            HStack(spacing: 18) {
                ForEach(["Rulebook", "Network", "Hubs", "Training", "Exam", "OLMo"], id: \.self) { s in
                    Text(s).font(.system(size: 13, weight: .semibold, design: .rounded))
                        .padding(.horizontal, 12).padding(.vertical, 7)
                        .background(Capsule().fill(s == "OLMo" ? Theme.answer.opacity(0.9) : Color.white.opacity(0.08)))
                        .foregroundStyle(s == "OLMo" ? Color.black : Theme.answer)
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Rulebook, network, hubs, training, exam, then OLMo")
            Image(systemName: "graduationcap.fill").font(.system(size: 54)).foregroundStyle(Theme.answer.opacity(0.85))
            Spacer(minLength: 0)
        }
    }
}

// MARK: - the gate fanning out into word-piece destinations

/// The airline map's single gate, then the same gate opening into many: the base model's top destinations after
/// "…safely on the", line thickness = recorded probability, plus faint lines for the rest of the vocabulary.
struct GateFanView: View {
    let picks: [StagePick]
    let replay: Int
    @State private var open = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height
            let gate = CGPoint(x: w * 0.40, y: h * 0.5)
            let top = picks.prefix(8)
            let maxP = CGFloat(top.map(\.prob).max() ?? 1)
            ZStack {
                // the tiny airline network: clipboard → two rows of hubs → the gate
                ForEach(0..<4, id: \.self) { i in
                    let a = CGPoint(x: w * 0.06, y: h * (0.26 + 0.16 * CGFloat(i)))
                    ForEach(0..<3, id: \.self) { j in
                        let b = CGPoint(x: w * 0.17, y: h * (0.32 + 0.18 * CGFloat(j)))
                        line(a, b).stroke(Color.white.opacity(0.12), lineWidth: 1)
                    }
                }
                ForEach(0..<3, id: \.self) { j in
                    let b = CGPoint(x: w * 0.17, y: h * (0.32 + 0.18 * CGFloat(j)))
                    ForEach(0..<3, id: \.self) { k in
                        let c = CGPoint(x: w * 0.28, y: h * (0.32 + 0.18 * CGFloat(k)))
                        line(b, c).stroke(Color.white.opacity(0.12), lineWidth: 1)
                    }
                    line(CGPoint(x: w * 0.28, y: h * (0.32 + 0.18 * CGFloat(j))), gate).stroke(Color.white.opacity(0.18), lineWidth: 1.2)
                }
                ForEach(0..<4, id: \.self) { i in
                    RoundedRectangle(cornerRadius: 3).fill(Theme.panel).overlay(RoundedRectangle(cornerRadius: 3).stroke(Color.white.opacity(0.25)))
                        .frame(width: 26, height: 16).position(x: w * 0.06, y: h * (0.26 + 0.16 * CGFloat(i)))
                }
                ForEach(0..<2, id: \.self) { r in
                    ForEach(0..<3, id: \.self) { j in
                        Circle().fill(Theme.panel).overlay(Circle().stroke(Color.white.opacity(0.35)))
                            .frame(width: 14, height: 14).position(x: w * (r == 0 ? 0.17 : 0.28), y: h * (0.32 + 0.18 * CGFloat(j)))
                    }
                }
                // the faint rest of the vocabulary
                ForEach(0..<64, id: \.self) { i in
                    let y = h * (0.04 + 0.92 * CGFloat(i) / 63)
                    line(gate, CGPoint(x: w * 0.80, y: y))
                        .trim(from: 0, to: open ? 1 : 0)
                        .stroke(Color.white.opacity(0.06), lineWidth: 0.6)
                }
                // the busiest destinations
                ForEach(Array(top.enumerated()), id: \.offset) { i, p in
                    let y = h * (0.14 + 0.72 * CGFloat(i) / CGFloat(max(top.count - 1, 1)))
                    let end = CGPoint(x: w * 0.80, y: y)
                    line(gate, end)
                        .trim(from: 0, to: open ? 1 : 0)
                        .stroke(Theme.positive.opacity(0.85), style: StrokeStyle(lineWidth: max(1, 14 * CGFloat(p.prob) / maxP), lineCap: .round))
                    HStack(spacing: 8) {
                        Text(Graduation.pieceText(p.token)).font(.system(size: 15, weight: .semibold, design: .monospaced)).foregroundStyle(Theme.answer)
                        Text(Graduation.percent(p.prob)).font(.system(size: 12, design: .monospaced)).foregroundStyle(.secondary)
                    }
                    .fixedSize()
                    .frame(width: 150, alignment: .leading)
                    .position(x: w * 0.80 + 85, y: y)
                    .opacity(open ? 1 : 0)
                }
                Text("+ \((Graduation.vocabSlots - top.count).formatted()) more")
                    .font(.system(size: 11)).foregroundStyle(.white.opacity(0.4))
                    .frame(width: 150, alignment: .leading)
                    .position(x: w * 0.80 + 85, y: h * 0.96).opacity(open ? 1 : 0)
                // what the words so far are
                if let r = Graduation.recording(Graduation.baseStage, "blank") {
                    (Text(r.prompt).foregroundColor(Theme.answer) + Text(" ___").foregroundColor(Theme.positive))
                        .font(.system(size: 15, design: .serif))
                        .frame(width: w * 0.36, alignment: .leading)
                        .position(x: w * 0.03 + w * 0.18, y: h * 0.08)
                }
                // the gate
                VStack(spacing: 4) {
                    RoundedRectangle(cornerRadius: 5).fill(Theme.answer).frame(width: 18, height: 34)
                    Text(open ? "next word?" : "how late?").font(.system(size: 12, weight: .semibold)).foregroundStyle(Theme.answer)
                }
                .position(x: gate.x, y: gate.y + 10)
                if let r = Graduation.recording(Graduation.baseStage, "blank") {
                    Text(r.provenance + ", " + r.recordedDay)
                        .font(.system(size: 10)).foregroundStyle(.white.opacity(0.4))
                        .position(x: w * 0.2, y: h - 6)
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("The airline gate becomes \(Graduation.vocabSlots.formatted()) word-piece destinations. Busiest after 'safely on the': "
                                + top.map { "\(Graduation.pieceText($0.token)) \(Graduation.percent($0.prob))" }.joined(separator: ", "))
        }
        .onAppear(perform: grow)
        .onChange(of: replay) { _, _ in open = false; grow() }
    }

    private func grow() {
        if reduceMotion { open = true; return }
        withAnimation(.easeInOut(duration: 1.4).delay(0.6)) { open = true }
    }

    private func line(_ a: CGPoint, _ b: CGPoint) -> Path {
        var p = Path()
        p.move(to: a)
        p.addCurve(to: b, control1: CGPoint(x: (a.x + b.x) / 2, y: a.y), control2: CGPoint(x: (a.x + b.x) / 2, y: b.y))
        return p
    }
}

// MARK: - a snapshot row: how far along, what it wrote

struct SnapshotRow: View {
    let recording: StageRecording

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text(recording.trainedTokens == 0 ? "before training" : "after \(Graduation.wordPieces(recording.trainedTokens))")
                    .font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.answer)
                Text(recording.trainedTokens == 0 ? "random dials" : "word pieces read")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
            .frame(width: 150, alignment: .leading)
            VStack(alignment: .leading, spacing: 4) {
                (Text(recording.prompt).foregroundColor(.white.opacity(0.45)) + Text(recording.output.replacingOccurrences(of: "\n", with: " ⏎ ")).foregroundColor(Theme.answer))
                    .font(.system(size: 14))
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                HStack(spacing: 10) {
                    if let t = recording.nextTop.first {
                        Text("top guess: \(Graduation.pieceText(t.token)) \(Graduation.percent(t.prob))")
                            .font(.system(size: 11, design: .monospaced)).foregroundStyle(.secondary)
                    }
                    Text("\(recording.branch) @\(recording.commit.prefix(8))")
                        .font(.system(size: 10, design: .monospaced)).foregroundStyle(.white.opacity(0.35))
                }
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 9)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Color.white.opacity(0.07)))
        .help(recording.provenance)
        .accessibilityElement(children: .combine)
    }
}

// MARK: - a recorded exchange

struct MathVerdict {
    let time: String?
    let right: Bool
    init(_ output: String) { time = Graduation.landingTime(in: output); right = Graduation.landsRight(output) }
}

struct TranscriptCard: View {
    enum Style { case web, chat }
    let recording: StageRecording
    let style: Style
    var title: String?
    var compact = false
    var verdict: MathVerdict?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                if let title { Text(title.uppercased()).font(.system(size: 10, weight: .semibold)).kerning(0.8).foregroundStyle(.white.opacity(0.5)) }
                Spacer()
                if let v = verdict {
                    Text(v.time.map { "lands at \($0): \(v.right ? "right" : "wrong")" } ?? "gives no time")
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(Capsule().fill(v.right ? Theme.prompt.opacity(0.25) : Theme.ablated.opacity(0.25)))
                        .foregroundStyle(v.right ? Theme.prompt : Color(red: 1, green: 0.55, blue: 0.55))
                }
            }
            FitOrScroll {
                VStack(alignment: .leading, spacing: 10) {
                    if style == .chat {
                        Text(recording.prompt)
                            .font(.system(size: 14, weight: .medium))
                            .padding(.horizontal, 12).padding(.vertical, 8)
                            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.prompt.opacity(0.18)))
                            .frame(maxWidth: .infinity, alignment: .trailing)
                        Text(Graduation.tidy(recording.output))
                            .font(.system(size: compact ? 13 : 14))
                            .foregroundStyle(Theme.answer)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        // a plain page: the question and what the model wrote after it, as one document
                        (Text(recording.prompt).bold() + Text(recording.output))
                            .font(.system(size: compact ? 13 : 15, design: .serif))
                            .foregroundStyle(Color(white: 0.12))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    if !recording.finished {
                        Text("(cut off at \(recording.maxTokens) word pieces; it would have kept going)")
                            .font(.system(size: 11)).foregroundStyle(style == .web ? Color(white: 0.4) : .secondary)
                    }
                }
                .padding(14)
            }
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(style == .web ? Color(white: 0.93) : Theme.panel))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Color.white.opacity(0.08)))
            Text("\(recording.provenance), \(recording.recordedDay) · greedy · \(Int(recording.tokPerS.rounded())) word pieces/s")
                .font(.system(size: 10)).foregroundStyle(.white.opacity(0.4))
        }
        .accessibilityElement(children: .combine)
    }
}

struct LiveAnswerCard: View {
    let question: String
    let answer: String
    let model: GraduationModel
    let phase: GraduationLive.Phase
    let tokPerS: Double?
    let cutOff: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(model.name.uppercased()).font(.system(size: 10, weight: .semibold)).kerning(0.8).foregroundStyle(.white.opacity(0.5))
            FitOrScroll {
                VStack(alignment: .leading, spacing: 10) {
                    Text(question)
                        .font(.system(size: 14, weight: .medium))
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.prompt.opacity(0.18)))
                        .frame(maxWidth: .infinity, alignment: .trailing)
                    Text(answer.isEmpty && phase != .done ? "…" : Graduation.tidy(answer))
                        .font(.system(size: 14)).foregroundStyle(Theme.answer)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if cutOff {
                        Text("(stopped at \(GraduationLive.maxTokens) word pieces)").font(.system(size: 11)).foregroundStyle(.secondary)
                    }
                }
                .padding(14)
            }
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.panel))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Color.white.opacity(0.08)))
            Text(model.provenance + " · greedy" + (tokPerS.map { " · \(Int($0.rounded())) word pieces/s" } ?? ""))
                .font(.system(size: 10)).foregroundStyle(.white.opacity(0.4))
        }
        .accessibilityElement(children: .combine)
    }
}

/// Content at its own height when it fits; a scroll view (same content) when it does not.
struct FitOrScroll<Content: View>: View {
    @ViewBuilder var content: () -> Content
    var body: some View {
        ViewThatFits(in: .vertical) {
            content().fixedSize(horizontal: false, vertical: true)
            ScrollView { content() }
        }
    }
}
