import SwiftUI

/// The four-step bar across the top: Ask -> Watch -> Change -> Compare. Completed steps are clickable to go
/// back; the Advanced toggle sits at the right.
struct StepBar: View {
    @Environment(AppState.self) private var state

    var body: some View {
        HStack(spacing: 6) {
            Text("NeuroScope").font(.system(size: 15, weight: .semibold, design: .rounded))
                .padding(.trailing, 14)
            ForEach(AppState.GuideStep.allCases) { s in
                if s != .ask {
                    Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.tertiary)
                }
                stepPill(s)
            }
            Spacer(minLength: 12)
            Button {
                state.showAdvanced.toggle()
            } label: {
                Label("Advanced", systemImage: "slider.horizontal.3")
                    .font(.callout)
                    .foregroundStyle(state.showAdvanced ? Theme.positive : .secondary)
            }
            .buttonStyle(.borderless)
            .help("Show every technical control, with the real names (⌥⌘A)")
        }
        .padding(.leading, 80)   // clear the window's traffic-light buttons (hidden title bar)
        .padding(.trailing, 16)
        .frame(height: 46)
        .background(Theme.panel)
    }

    private func stepPill(_ s: AppState.GuideStep) -> some View {
        let current = s == state.step
        let done = s.rawValue < state.step.rawValue
        return Button {
            if done { state.step = s }
        } label: {
            HStack(spacing: 7) {
                ZStack {
                    Circle().fill(current ? Theme.positive : (done ? Color.white.opacity(0.22) : Color.white.opacity(0.07)))
                    if done {
                        Image(systemName: "checkmark").font(.system(size: 9, weight: .bold))
                    } else {
                        Text("\(s.rawValue + 1)").font(.system(size: 11, weight: .bold, design: .rounded))
                    }
                }
                .frame(width: 20, height: 20)
                .foregroundStyle(current ? Color.black : .white)
                Text(s.title)
                    .font(.callout.weight(current ? .semibold : .regular))
                    .foregroundStyle(current ? Color.primary : (done ? Color.secondary : Color.secondary.opacity(0.6)))
            }
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(current ? Color.white.opacity(0.07) : Color.clear, in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .disabled(!done && !current)
        .help(done ? "Go back to \(s.title)" : "")
    }
}

// MARK: shared bits for the step views

/// A step's title + one-sentence explainer (markdown: **bold** terms) + "Learn more".
struct StepHeader: View {
    @Environment(AppState.self) private var state
    let title: String
    let explainer: String
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.title2.weight(.semibold))
            Text(markdown(explainer)).font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Button("Learn more") { state.showGuide = true }
                .buttonStyle(.link).font(.caption)
        }
    }
}

/// Inline markdown (bold, italics) for teaching copy; falls back to the plain string if parsing fails.
func markdown(_ s: String) -> AttributedString {
    (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
        ?? AttributedString(s)
}

/// The "What is happening" card under each step header: a few short paragraphs in the building scene
/// (markdown, real terms in **bold** and glossed where they stand) and a "New words" list of the terms this
/// step introduces, one plain line each. Collapsible so it can get out of the way once read.
public struct LearnCard: View {
    let paragraphs: [String]
    let words: [(term: String, meaning: String)]
    @State private var open: Bool
    private let key: String

    public init(_ paragraphs: [String], words: [(term: String, meaning: String)] = []) {
        self.paragraphs = paragraphs
        self.words = words
        let k = "learnCardOpen." + String((paragraphs.first ?? "").unicodeScalars.reduce(UInt64(5381)) { ($0 &* 33) &+ UInt64($1.value) })
        self.key = k
        _open = State(initialValue: UserDefaults.standard.bool(forKey: k))
    }

    private func firstSentence(_ p: String) -> String {
        if let r = p.range(of: ". ") { return String(p[p.startIndex..<r.upperBound]).trimmingCharacters(in: .whitespaces) }
        return p
    }

    private func setOpen(_ v: Bool) {
        withAnimation(.easeOut(duration: 0.15)) { open = v }
        UserDefaults.standard.set(v, forKey: key)
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if open {
                ForEach(Array(paragraphs.enumerated()), id: \.offset) { _, p in
                    Text(markdown(p)).font(.callout)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if !words.isEmpty {
                    Divider().opacity(0.4)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("NEW WORDS").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                        ForEach(Array(words.enumerated()), id: \.offset) { _, w in
                            Text(markdown("**\(w.term)**: \(w.meaning)"))
                                .font(.caption).foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                Button("Less") { setOpen(false) }
                    .buttonStyle(.link).font(.caption)
                    .accessibilityLabel("Hide the explanation of this step")
            } else {
                Text(markdown(firstSentence(paragraphs.first ?? ""))).font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
                Button("More") { setOpen(true) }
                    .buttonStyle(.link).font(.caption)
                    .accessibilityLabel("Show the explanation of this step")
            }
        }
        .padding(10)
        .background(Theme.pinned.opacity(0.06), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.pinned.opacity(0.18)))
    }
}

/// Small uppercase section label.
struct StepSection: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text.uppercased()).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
    }
}

/// Wrapping row layout for chips.
struct GuideFlowLayout: Layout {
    var spacing: CGFloat = 6
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        arrange(proposal.width ?? 320, subviews).size
    }
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let r = arrange(bounds.width, subviews)
        for (sv, f) in zip(subviews, r.frames) {
            sv.place(at: CGPoint(x: bounds.minX + f.minX, y: bounds.minY + f.minY), proposal: ProposedViewSize(f.size))
        }
    }
    private func arrange(_ width: CGFloat, _ subviews: Subviews) -> (size: CGSize, frames: [CGRect]) {
        var x: CGFloat = 0, y: CGFloat = 0, rowH: CGFloat = 0, maxW: CGFloat = 0
        var frames: [CGRect] = []
        for sv in subviews {
            var s = sv.sizeThatFits(.unspecified)
            s.width = min(s.width, width)
            if x > 0 && x + s.width > width { x = 0; y += rowH + spacing; rowH = 0 }
            frames.append(CGRect(origin: CGPoint(x: x, y: y), size: s))
            x += s.width + spacing
            rowH = max(rowH, s.height)
            maxW = max(maxW, x - spacing)
        }
        return (CGSize(width: maxW, height: y + rowH), frames)
    }
}

/// Plain name for a neuron on the beginner path: a desk on a floor (layer = floor, neuron = desk).
func plainNeuronName(_ n: NeuronID) -> String { "Floor \(n.layer) · desk #\(n.index)" }

/// Generated words (thinking + answer tokens) in a run.
func generatedWordCount(_ run: Run) -> Int {
    run.tokens.reduce(0) { $0 + (($1.stream == .thinking || $1.stream == .answer) ? 1 : 0) }
}

/// Text of a stream with the model's special markers trimmed for display.
func plainStreamText(_ run: Run, _ stream: StreamKind) -> String {
    run.text(of: stream).trimmingCharacters(in: .whitespacesAndNewlines)
}
