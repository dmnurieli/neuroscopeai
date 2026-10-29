import SwiftUI
import AirlineKit

/// Lesson 7, "Exam day": the sealed exam years (2024–2025) open for the first time. The learner's trained network
/// is compared with a crammer: a bigger network (two rows of 8) that studies only 200 flights, 3,000 times over.
/// Measured on the default ledger (HubCardTests.testExamDayCrammer): crammer homework 3.4 min off, exam 10.6; the
/// learner's 2×6 homework 5.9, exam 5.8. The exam is scored only when the learner opens it.
@MainActor @Observable
final class AirlineExamLesson: AirlineLessonModel {
    let title = AirlineLessonID.exam.title
    let pageCount = 5
    let chor = Choreographer()

    static let crammerPile = 200
    static let crammerPasses = 3_000
    static let crammerSeed: UInt64 = 3

    enum Cram: Equatable { case idle, running(Double), done }

    private(set) var ledger: Ledger?
    private(set) var network: Network?
    private(set) var crammer: Network?
    private(set) var pile: [Int] = []
    private(set) var cram: Cram = .idle
    private(set) var page = 0
    // scores (minutes off)
    private(set) var yoursHomework: Float?
    private(set) var crammerHomework: Float?
    private(set) var yoursExam: Float?
    private(set) var crammerExam: Float?
    private(set) var examOpened = false
    private var task: Task<Void, Never>?

    var canGoBack: Bool { page > 0 && !isCramming }
    var canWatchAgain: Bool { false }
    private var isCramming: Bool { if case .running = cram { return true }; return false }

    init() { chor.summary = { [weak self] in self?.summary } }

    func load() async {
        guard ledger == nil else { return }
        let l = await AirlineLedger.load()
        ledger = l
        let net = await AirlineProgress.trainedNetwork(l)
        network = net
        pile = l.studyPile(count: Self.crammerPile, seed: Self.crammerSeed)
        yoursHomework = await Task.detached(priority: .userInitiated) { net.meanMinutesOff(.practice, in: l) }.value
    }

    func stop() { task?.cancel() }

    /// The crammer studies its 200 flights over and over, in chunks so the button shows progress.
    private func startCram() {
        guard let l = ledger, let net = network else { return }
        let arch = Architecture(inputs: net.architecture.inputs, hidden: [8, 8])
        let pile = self.pile
        let batchesPerPass = (pile.count + 63) / 64
        let total = batchesPerPass * Self.crammerPasses, chunks = 30
        cram = .running(0)
        task = Task { [weak self] in
            let t = Trainer(network: Network(architecture: arch, ledger: l, seed: Architecture.lessonSeed), ledger: l, studyPile: pile)
            for c in 0..<chunks {
                let n = total * (c + 1) / chunks - total * c / chunks
                _ = await Task.detached(priority: .userInitiated) { t.train(batches: n) }.value
                guard let self, !Task.isCancelled else { return }
                self.cram = .running(Double(c + 1) / Double(chunks))
            }
            let crammed = t.network
            let homework = await Task.detached(priority: .userInitiated) { crammed.meanMinutesOff(indices: pile, in: l) }.value
            guard let self, !Task.isCancelled else { return }
            self.crammer = crammed
            self.crammerHomework = homework
            self.cram = .done
        }
    }

    /// The sealed years are scored here and nowhere earlier.
    private func openExam() {
        guard let l = ledger, let net = network, let cr = crammer, !examOpened else { return }
        examOpened = true
        Task { [weak self] in
            let (a, b) = await Task.detached(priority: .userInitiated) { (net.meanMinutesOff(.exam, in: l), cr.meanMinutesOff(.exam, in: l)) }.value
            self?.yoursExam = a
            self?.crammerExam = b
        }
    }

    // MARK: the pages

    var copy: AirlinePageCopy {
        guard let l = ledger, network != nil, yoursHomework != nil else {
            return AirlinePageCopy(text: "Loading your network…", button: "Continue", enabled: false, working: true)
        }
        let examCount = l.indices(.exam).count
        switch page {
        case 0:
            return AirlinePageCopy(
                text: "Until now it quizzed itself on 2023. The real exam, 2024–2025, has stayed sealed.",
                detail: "\(examCount.formatted()) flights it has never seen, not even once in a quiz. Every score so far came from the other two piles.",
                button: "Continue")
        case 1:
            switch cram {
            case .idle:
                return AirlinePageCopy(
                    text: "Meet a **crammer**: a bigger network that studies only \(Self.crammerPile) flights, \(Self.crammerPasses.formatted()) times over.",
                    detail: "Two rows of 8 hubs, same clipboard as yours. It sees the same few flights again and again.",
                    button: "Let it cram")
            case .running(let p):
                return AirlinePageCopy(
                    text: "Meet a **crammer**: a bigger network that studies only \(Self.crammerPile) flights, \(Self.crammerPasses.formatted()) times over.",
                    detail: "Cramming… \(Int((p * 100).rounded()))%", button: "Cramming…", enabled: false, working: true)
            case .done:
                return AirlinePageCopy(
                    text: "Meet a **crammer**: a bigger network that studies only \(Self.crammerPile) flights, \(Self.crammerPasses.formatted()) times over.",
                    detail: "Done: it has seen its \(Self.crammerPile) flights \(Self.crammerPasses.formatted()) times each.", button: "Continue")
            }
        case 2:
            let c = crammerHomework.map { AirlineState.minutes($0) } ?? 0, y = yoursHomework.map { AirlineState.minutes($0) } ?? 0
            return AirlinePageCopy(
                text: "Homework: the crammer looks brilliant.",
                detail: "On the flights each one studied, the crammer is off by \(c) min; yours by \(y).",
                button: "Continue")
        case 3:
            if let ye = yoursExam, let ce = crammerExam {
                return AirlinePageCopy(
                    text: "The exam: on flights neither has seen, the crammer is off by \(AirlineState.minutes(ce)) min, yours by \(AirlineState.minutes(ye)).",
                    detail: "The crammer memorised its \(Self.crammerPile) flights instead of learning how delays work.",
                    button: "Continue", reserveText: ["The exam: on flights neither has seen, the crammer is off by 00 min, yours by 00."])
            }
            return AirlinePageCopy(text: "Now open the sealed exam.", detail: "2024–2025, scored for the first time.",
                                   button: examOpened ? "Opening…" : "Open the exam", enabled: !examOpened, working: examOpened,
                                   reserveText: ["The exam: on flights neither has seen, the crammer is off by 00 min, yours by 00."])
        default:
            return AirlinePageCopy(
                text: "Memorising the homework is not understanding. That's why the exam stays sealed until the end.",
                detail: "Yours learned how delays work, so it does about as well on flights it never saw as on the ones it studied.",
                button: "Done")
        }
    }

    var bars: [ScoreBars.Bar] {
        switch page {
        case 2:
            return [.init(label: "Crammer, homework", minutes: crammerHomework), .init(label: "Yours, homework", minutes: yoursHomework, highlight: true)]
        case 3, 4:
            return [.init(label: "Crammer, homework", minutes: crammerHomework), .init(label: "Crammer, exam", minutes: crammerExam),
                    .init(label: "Yours, homework", minutes: yoursHomework, highlight: true), .init(label: "Yours, exam", minutes: yoursExam, highlight: true)]
        default:
            return []
        }
    }

    func primary() -> Bool {
        guard network != nil else { return false }
        switch page {
        case 1 where cram == .idle: startCram(); return false
        case 1 where isCramming: return false
        case 3 where yoursExam == nil: openExam(); return false
        case pageCount - 1: return true
        default: break
        }
        page += 1
        return false
    }

    func back() { if canGoBack { page -= 1 } }
    func watchAgain() {}

    var summary: String? {
        guard let ye = yoursExam, let ce = crammerExam, let ch = crammerHomework, let yh = yoursHomework else { return nil }
        return "Crammer: homework \(AirlineState.minutes(ch)), exam \(AirlineState.minutes(ce)) minutes off. Yours: homework \(AirlineState.minutes(yh)), exam \(AirlineState.minutes(ye))."
    }
}

// MARK: - the view: the three piles, then the bars

struct AirlineExamLessonView: View {
    var onExit: (_ completed: Bool) -> Void
    @State private var model = AirlineExamLesson()

    var body: some View {
        AirlineLessonShell(model: model, onExit: onExit) {
            VStack(spacing: 34) {
                Spacer(minLength: 0)
                if let l = model.ledger {
                    HStack(spacing: 44) {
                        pile("Study pile", "2016–2022", l.indices(.practice).count, sealed: false)
                        pile("Quiz pile", "2023", l.indices(.check).count, sealed: false)
                        pile("Exam", "2024–2025", l.indices(.exam).count, sealed: !model.examOpened)
                    }
                }
                if !model.bars.isEmpty {
                    ScoreBars(bars: model.bars).animation(.easeOut(duration: 0.5), value: model.yoursExam)
                }
                Spacer(minLength: 0)
            }
            .frame(maxWidth: 900, maxHeight: .infinity)
        } accessory: { _ in EmptyView() }
        .task { await model.load() }
    }

    private func pile(_ title: String, _ years: String, _ count: Int, sealed: Bool) -> some View {
        VStack(spacing: 8) {
            ZStack {
                ForEach(0..<3, id: \.self) { k in
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(Theme.panel)
                        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(Color.white.opacity(0.2)))
                        .frame(width: 92, height: 62)
                        .offset(x: CGFloat(k) * 4, y: CGFloat(-k) * 4)
                }
                Image(systemName: sealed ? "lock.fill" : "doc.text")
                    .font(.system(size: 22)).foregroundStyle(sealed ? Theme.answer : Color.white.opacity(0.5))
            }
            Text(title.uppercased()).font(.system(size: 11, weight: .semibold)).kerning(0.6).foregroundStyle(.white.opacity(0.55))
            Text("\(years) · \(count.formatted()) flights").font(.system(size: 13, design: .monospaced)).foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title), \(years), \(count) flights\(sealed ? ", sealed" : "")")
    }
}
