import SwiftUI

/// Step 3: remove or add an idea (an experiment card), or change one neuron; then Try it.
struct ChangeStep: View {
    @Environment(AppState.self) private var state
    @State private var showCustom = false

    private var busy: Bool {
        if case .preparing = state.experimentStatus { return true }
        return state.isGenerating || state.isContrasting
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            StepHeader(title: "Change something",
                       explainer: "Erase an idea from the building, or click a dot to turn one desk off or up. Then ask your question again.")
            LearnCard([
                "An idea like \"French\" works like one highlighter color running through the memo. The real word is a **direction** - one consistent mark the idea leaves in the memo. We find it with **matched pairs**: the same questions asked \"...in French\" and \"...in English\". Whatever is consistently different in the memos is the highlighter.",
                "**Remove** puts a filter in the elevator that erases that one color on every floor, so no floor can see it. Researchers call this **abliteration**, a ten-dollar word (ablation + obliteration) for a simple act: erase one highlighter color, everywhere.",
                "**Add** is **steering**: someone on one floor inks that highlighter onto every memo passing by, so the floors above read it as if you had asked for it. You can try it under Make your own, but on this small model it often garbles the reply, so the ready-made changes only erase.",
                "The ink shows up on many floors, but erasing it only works on a few. So before you see anything, the app holds tryouts on fresh questions and keeps the floor that works without breaking the reply." + (state.hasThinking ? " Scratch paper is off for this: with it on, the building re-reads \"in French\" and writes itself \"answer in French\", putting the ink back in plain words." : ""),
                "You can also turn off one desk. That often changes little, because of **superposition**: desks share jobs. One desk helps with several unrelated topics, and one topic is spread over many desks, so the others cover for it.",
            ], words: [
                ("direction", "an idea's highlighter color in the memo"),
                ("matched pairs", "the same question asked two ways, differing only in the idea"),
                ("abliteration", "Remove: erase one highlighter color on every floor"),
                ("steering", "Add: ink the highlighter onto the memo on one floor"),
                ("superposition", "desks share jobs, so one desk rarely matters alone"),
            ])
            cards
            status
            neuronCard
            yourChange
            Button { state.tryIt() } label: {
                Label("Try it", systemImage: "play.fill").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
            .disabled(state.edits.isEmpty || busy || state.status != .ready)
            .help("Ask the same question again with your change, then compare (⇧⌘↩)")
            mathSection
        }
    }

    // MARK: experiment cards

    /// The first ready-made change is the hero (full width, its whole summary shown); the rest, plus Make your own,
    /// sit two to a row underneath.
    private var cards: some View {
        VStack(spacing: 8) {
            if let hero = Experiment.all.first { experimentButton(hero, hero: true) }
            LazyVGrid(columns: [GridItem(.flexible(), spacing: 8), GridItem(.flexible(), spacing: 8)], spacing: 8) {
                ForEach(Experiment.all.dropFirst()) { e in experimentButton(e, hero: false) }
                Button { showCustom = true } label: {
                    card(title: "Make your own", summary: "Name any idea and its opposite.",
                         icon: "slider.horizontal.3", selected: false)
                }
                .buttonStyle(.plain)
                .disabled(busy)
            }
        }
        .sheet(isPresented: $showCustom) { CustomConceptSheet() }
    }

    private func experimentButton(_ e: Experiment, hero: Bool) -> some View {
        let selected = state.experiment?.id == e.id
        return Button { Task { await state.choose(e) } } label: {
            card(title: e.title, summary: e.summary,
                 icon: e.mode == .ablate ? "minus.circle" : "plus.circle", selected: selected, hero: hero)
        }
        .buttonStyle(.plain)
        .disabled(busy)
        .help(e.summary)
        .accessibilityLabel("\(e.title). \(e.summary)")
    }

    private func card(title: String, summary: String, icon: String, selected: Bool, hero: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(title, systemImage: icon).font(hero ? .headline : .callout.weight(.semibold))
                .foregroundStyle(selected ? Theme.positive : .primary)
            Text(summary).font(.caption).foregroundStyle(.secondary)
                .lineLimit(hero ? 3 : 1).truncationMode(.tail)
                .fixedSize(horizontal: false, vertical: hero)
            Spacer(minLength: 0)
        }
        .padding(10)
        .frame(maxWidth: .infinity, minHeight: 54, alignment: .topLeading)
        .background(Color.white.opacity(selected ? 0.1 : 0.04), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(selected ? Theme.positive.opacity(0.8) : Color.white.opacity(0.08)))
        .contentShape(RoundedRectangle(cornerRadius: 10))
    }

    // MARK: status of the chosen experiment

    @ViewBuilder private var status: some View {
        switch state.experimentStatus {
        case .idle:
            EmptyView()
        case .preparing(let msg):
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(msg).font(.callout).foregroundStyle(.secondary)
            }
        case .failed(let msg):
            Label(msg, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(Theme.ablated)
        case .ready:
            VStack(alignment: .leading, spacing: 8) {
                Text(foundSentence).font(.callout.weight(.medium))
                if let note = state.experimentNote {
                    Text(note).font(.caption).foregroundStyle(Theme.pinned)
                        .fixedSize(horizontal: false, vertical: true)
                }
                HStack(spacing: 14) {
                    legendDot(Theme.positive, "busier when the question asks for the idea")
                    legendDot(Theme.negative, "busier when it does not")
                }
                if let e = state.experiment, state.prompt != e.testQuestion {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("This question shows the change best: \(e.testQuestion)").font(.caption).foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Button("Use this question") { state.prompt = e.testQuestion }
                            .controlSize(.small)
                    }
                    .padding(8)
                    .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
    }

    private var foundSentence: String {
        guard let site = state.experimentSite ?? state.contrastSite else { return "Found it. The highlighter is ready." }
        let adding = state.experiment?.mode == .add
        if site == 0 {
            return adding
                ? "Ready. We tried the ink on several floors with fresh questions; it worked best right at the front door, before floor 0."
                : "Ready. We tried the filter on several floors with fresh questions; the highlighter read right at the front door, before floor 0, worked best. It is now erased on every floor."
        }
        return adding
            ? "Ready. We tried the ink on several floors with fresh questions; floor \(site - 1) worked best."
            : "Ready. We tried the filter on several floors with fresh questions; floor \(site - 1) worked best. Its highlighter is now erased on every floor."
    }

    private func legendDot(_ c: Color, _ text: String) -> some View {
        HStack(spacing: 5) {
            Circle().fill(c).frame(width: 9, height: 9)
            Text(text).font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: selected neuron

    @ViewBuilder private var neuronCard: some View {
        if state.selection != nil {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    StepSection("The desk you picked")
                    Spacer()
                    Button { state.select(nil) } label: { Image(systemName: "xmark") }
                        .buttonStyle(.borderless).foregroundStyle(.secondary)
                        .help("Put this desk back down")
                        .accessibilityLabel("Deselect this desk")
                }
                InspectorView()
                    .frame(height: 340)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        } else {
            Label("Or click any dot in the field to change one desk.", systemImage: "cursorarrow.click")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    // MARK: show the math

    private var selectedDesk: NeuronID? {
        if case .neuron(let n)? = state.selection { return n }
        return nil
    }

    @ViewBuilder private var mathSection: some View {
        let dir = state.edits.directions.first
        if dir != nil || selectedDesk != nil {
            MathDisclosure {
                VStack(alignment: .leading, spacing: 16) {
                    if let dir { directionMath(dir) }
                    if let n = selectedDesk {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Your desk: \(plainNeuronName(n))").font(.callout.weight(.semibold))
                            FormulaRow("turn off: h\u{2032} = 0    turn up: h\u{2032} = 3h    hold: h\u{2032} = v", [
                                ("h", "the desk's busyness on this word" + (deskH(n).map { ": **\(mathNum($0, 3))**" } ?? "")),
                                ("memo", "changes by \u{0394}h \u{00D7} the desk's column, size |\u{0394}h| \u{00D7} out_norm"),
                            ])
                            Text("The desk card above (\u{201C}What changes in the math\u{201D}) works this out with the desk's real numbers.")
                                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder private func directionMath(_ dir: DirectionEdit) -> some View {
        if let c = state.contrast, let site = state.experimentSite ?? state.contrastSite, site < c.norms.count {
            Text(markdown(dir.mode == .ablate
                ? "**In plain words:** the highlighter is one arrow in a space of \(state.geometry.memoText) numbers. Erasing it subtracts exactly the part of each memo that points along that arrow, on every floor, and leaves everything else alone."
                : "**In plain words:** the highlighter is one arrow in a space of \(state.geometry.memoText) numbers. Adding it pushes every memo on one floor a fixed distance along that arrow; nothing else moves."))
                .font(.callout).fixedSize(horizontal: false, vertical: true)
            DirectionMath(contrast: c, site: site, mode: dir.mode, alpha: state.experimentValue)
            MathStep(3, "Which floor? The tryouts",
                     plain: "The ink shows up on many floors, but changing it only works on a few, and which ones shifts with the wording. So the app tried each candidate on fresh questions:") {
                TryoutTable(tryouts: state.tryouts, mode: dir.mode, norms: c.norms,
                            chosenSite: state.experimentSite, chosenValue: state.experimentValue)
            }
            Text(markdown("**Build your own:** \u{201C}Make your own\u{201D} lets you choose the two phrases (which set **d**), the floor, and for Add the amount **\u{03B1}**: the same three numbers you just saw."))
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        } else {
            Text("This highlighter came from the Advanced panel. Its math is the same: x_new = x \u{2212} (x\u{00B7}d\u{0302}) d\u{0302} to erase, x_new = x + \u{03B1} d\u{0302} to add.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func deskH(_ n: NeuronID) -> Float? {
        guard let run = state.activeRun, let t = state.displayIndex, t < run.count else { return nil }
        return run.activation(n, at: t)
    }

    // MARK: your change

    @ViewBuilder private var yourChange: some View {
        let e = state.edits
        if !e.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    StepSection("Your change")
                    Spacer()
                    Button("Clear") { state.clearEdits() }.buttonStyle(.link).font(.caption)
                }
                GuideFlowLayout(spacing: 6) {
                    ForEach(e.directions) { d in
                        chip(directionName(d), d.mode == .ablate ? Theme.ablated : Theme.scaled) {
                            state.removeDirectionEdit(d.id)
                        }
                    }
                    ForEach(e.neurons, id: \.id) { n in
                        chip("\(plainNeuronName(n.id)) \(neuronVerb(n))", n.mode == .ablate ? Theme.ablated : Theme.scaled) {
                            state.removeEdit(.neuron(n.id))
                        }
                    }
                    ForEach(e.heads, id: \.id) { h in
                        chip("Floor \(h.layer) · head \(h.head) \(h.mode == .ablate ? "off" : "×\(String(format: "%g", h.value))")",
                             h.mode == .ablate ? Theme.ablated : Theme.scaled) {
                            state.removeEdit(.head(h.id))
                        }
                    }
                }
            }
        }
    }

    private func directionName(_ d: DirectionEdit) -> String {
        if let ex = state.experiment, state.edits.directions.count == 1 { return ex.title }
        return d.mode == .ablate ? "Erase a highlighter" : "Ink a highlighter"
    }

    private func neuronVerb(_ n: NeuronEdit) -> String {
        switch n.mode {
        case .ablate: return "off"
        case .scale: return "×\(String(format: "%g", n.value))"
        case .pin: return "held at \(String(format: "%g", n.value))"
        }
    }

    private func chip(_ text: String, _ color: Color, remove: @escaping () -> Void) -> some View {
        HStack(spacing: 5) {
            Text(text).font(.caption).lineLimit(1)
            Button(action: remove) { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).foregroundStyle(.secondary)
                .help("Undo this change")
                .accessibilityLabel("Undo \(text)")
        }
        .padding(.leading, 9).padding(.trailing, 5).padding(.vertical, 4)
        .background(color.opacity(0.22), in: Capsule())
        .overlay(Capsule().stroke(color.opacity(0.6)))
    }
}
