import SwiftUI

/// "Make your own": the bridge from ready-made toggles to custom edits. Two phrases become matched pairs, the
/// highlighter d is measured from them, and the learner either lets the app hold tryouts or picks the floor
/// (and, for Add, the ink amount α) by hand. Every control is labelled with its math symbol and plain meaning.
struct CustomConceptSheet: View {
    @Environment(AppState.self) private var state
    @Environment(\.dismiss) private var dismiss

    @State private var concept = "Answer in Spanish"
    @State private var opposite = "Answer in English"
    @State private var mode: DirectionMode = .ablate
    @State private var testQuestion = ""
    @State private var manual = false
    @State private var floor: Double = 14
    @State private var alphaMultiple: Double = 1.5
    @State private var measured: String?   // key of the experiment whose contrast is in state.contrast

    private var ready: Bool {
        !concept.trimmingCharacters(in: .whitespaces).isEmpty && !opposite.trimmingCharacters(in: .whitespaces).isEmpty
    }
    private var busy: Bool {
        if case .preparing = state.experimentStatus { return true }
        return state.isGenerating || state.isContrasting
    }
    private var key: String { "\(concept)|\(opposite)|\(mode.rawValue)" }

    private var question: String {
        let q = testQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        if !q.isEmpty { return q }
        return mode == .ablate ? "\(concept.trimmingCharacters(in: .whitespaces)): How do volcanoes erupt?" : "How do volcanoes erupt?"
    }

    private var experiment: Experiment? {
        guard ready else { return nil }
        return state.customExperiment(concept: concept, opposite: opposite, mode: mode, testQuestion: question)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Make your own change").font(.title2.weight(.semibold))
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            Text(markdown("Name an idea and its opposite. The app asks the same questions both ways (**matched pairs**: two questions that differ only in your idea), compares the memos, and finds the idea's highlighter **d**."))
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Label("On this small model, home-made changes often work only partly: expect mixed languages or a garbled reply, especially with Add.",
                  systemImage: "exclamationmark.triangle")
                .font(.caption).foregroundStyle(Theme.pinned).fixedSize(horizontal: false, vertical: true)

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    fields
                    preview
                    modePicker
                    testField
                    Divider().opacity(0.4)
                    Picker("How to place it", selection: $manual) {
                        Text("Find it for me").tag(false)
                        Text("I'll pick the floor").tag(true)
                    }
                    .pickerStyle(.segmented)
                    if manual { manualPanel } else { autoPanel }
                }
                .padding(.vertical, 4)
            }
        }
        .padding(20)
        .frame(width: 620, height: 680)
        .background(Theme.panel)
    }

    // MARK: pieces

    private var fields: some View {
        VStack(alignment: .leading, spacing: 8) {
            labelled("A", "The idea", "the questions that ask for it")
            TextField("e.g. Answer in Spanish", text: $concept).textFieldStyle(.roundedBorder)
            labelled("B", "Its opposite", "the same questions, asking for the opposite")
            TextField("e.g. Answer in English", text: $opposite).textFieldStyle(.roundedBorder)
        }
    }

    @ViewBuilder private var preview: some View {
        if let e = experiment {
            VStack(alignment: .leading, spacing: 4) {
                StepSection("First 3 of \(e.withConcept.count) matched pairs")
                ForEach(0..<min(3, e.withConcept.count, e.withoutConcept.count), id: \.self) { i in
                    VStack(alignment: .leading, spacing: 1) {
                        Text("A: \(e.withConcept[i])").font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.positive)
                        Text("B: \(e.withoutConcept[i])").font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.negative)
                    }
                }
                Text(markdown("**d = mean(A memos) \u{2212} mean(B memos)**, read at the last word of each question."))
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(8)
            .background(Theme.background, in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var modePicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("Change", selection: $mode) {
                Text("Remove it").tag(DirectionMode.ablate)
                Text("Add it").tag(DirectionMode.add)
            }
            .pickerStyle(.segmented)
            Text(mode == .ablate
                 ? "Remove: x_new = x \u{2212} (x\u{00B7}d\u{0302}) d\u{0302} on every floor. A question that asks for the idea should come back without it."
                 : "Add: x_new = x + \u{03B1} d\u{0302} on one floor. A plain question should come back with the idea.")
                .font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var testField: some View {
        VStack(alignment: .leading, spacing: 4) {
            labelled("q", "Test question", "what you will ask afterwards to see the change")
            TextField(question, text: $testQuestion).textFieldStyle(.roundedBorder)
        }
    }

    private var autoPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("The app measures d, then holds tryouts: it applies the change on several floors, asks fresh questions, and keeps the floor (and for Add, the \u{03B1}) that moves the answers most without breaking them. You will see every tryout's score in Show the math.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Button {
                guard let e = experiment else { return }
                dismiss()
                Task {
                    await state.choose(e)
                    if case .ready = state.experimentStatus { state.prompt = e.testQuestion }
                }
            } label: {
                Label("Find it for me", systemImage: "wand.and.stars").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
            .disabled(!ready || busy || state.status != .ready)
        }
    }

    @ViewBuilder private var manualPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            if measured != key || state.contrast == nil {
                Text("First, measure the highlighter d from your matched pairs (about a second, no tryouts).")
                    .font(.caption).foregroundStyle(.secondary)
                Button {
                    guard let e = experiment else { return }
                    Task { await measure(e) }
                } label: {
                    if state.isContrasting {
                        HStack { ProgressView().controlSize(.small); Text("Comparing the memos\u{2026}") }.frame(maxWidth: .infinity)
                    } else {
                        Label("Measure the highlighter d", systemImage: "ruler").frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.bordered).controlSize(.large)
                .disabled(!ready || busy || state.status != .ready)
                if let err = state.contrastError {
                    Text(err).font(.caption).foregroundStyle(Theme.ablated)
                }
            } else if let c = state.contrast {
                manualControls(c)
            }
        }
    }

    @ViewBuilder private func manualControls(_ c: ContrastResult) -> some View {
        let maxFloor = max(0, c.norms.count - 2)
        let site = min(Int(floor.rounded()) + 1, c.norms.count - 1)
        let norm = c.norms[site]
        let sep = site < c.separation.count ? c.separation[site] : 0
        let alpha = Float(alphaMultiple) * norm
        VStack(alignment: .leading, spacing: 10) {
            labelled("\u{2113}", "Floor", mode == .ablate
                     ? "where d is read; the erase itself then runs on every floor"
                     : "the one floor where the ink is added")
            HStack {
                Slider(value: $floor, in: 0...Double(maxFloor), step: 1)
                Text("floor \(Int(floor.rounded()))").font(.system(.callout, design: .monospaced)).frame(width: 80)
            }
            Text("At this floor: |d| = \(mathNum(norm)) (the highlighter's length), separation = \(mathNum(sep)) (how cleanly A and B memos split; higher = clearer).")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            if mode == .add {
                labelled("\u{03B1}", "Ink amount", "how many units of d\u{0302} to add, in multiples of |d|")
                HStack {
                    Slider(value: $alphaMultiple, in: 0.5...3.0, step: 0.25)
                    Text("\(mathNum(Float(alphaMultiple), 2))\u{00D7}|d|").font(.system(.callout, design: .monospaced)).frame(width: 80)
                }
                Text("\u{03B1} = \(mathNum(Float(alphaMultiple), 2)) \u{00D7} \(mathNum(norm)) = \(mathNum(alpha)). The ink x\u{00B7}d\u{0302} on floor \(Int(floor.rounded())) rises by exactly that. The app's tryouts use 1.0, 1.5 and 2.0; much more usually breaks the reply.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Text(markdown("Removal usually works at only one or two floors, and which ones shifts with the wording. Middle floors are a good first guess; if nothing changes, try a neighbour."))
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Button {
                state.applyManual(site: site, value: mode == .ablate ? 0 : alpha)
                state.prompt = question
                dismiss()
            } label: {
                Label(mode == .ablate ? "Erase d\u{0302} on every floor" : "Ink \u{03B1}d\u{0302} on floor \(Int(floor.rounded()))",
                      systemImage: "checkmark").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
            .disabled(busy)
        }
    }

    private func measure(_ e: Experiment) async {
        state.experiment = e
        state.contrastA = e.withConcept.joined(separator: "\n")
        state.contrastB = e.withoutConcept.joined(separator: "\n")
        await state.computeContrast()
        guard state.contrast != nil, state.contrastError == nil else { return }
        measured = key
        if let s = state.contrastSite, s > 0 { floor = Double(s - 1) }
    }

    private func labelled(_ symbol: String, _ title: String, _ meaning: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(symbol).font(.system(.caption, design: .monospaced).weight(.bold)).foregroundStyle(Theme.thinking)
                .frame(minWidth: 16)
            Text(title).font(.callout.weight(.semibold))
            Text("\u{2013} \(meaning)").font(.caption).foregroundStyle(.secondary)
        }
    }
}
