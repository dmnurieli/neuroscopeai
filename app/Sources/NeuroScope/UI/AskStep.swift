import SwiftUI

/// Step 1: type a question (or pick an example) and press Ask.
struct AskStep: View {
    @Environment(AppState.self) private var state

    static let examples = [
        "Why is the sky blue?",
        "How do I boil an egg?",
        "What is the capital of Japan?",
        "Explain how a bicycle stays upright.",
        "Why do cats purr?",
        "Give me a tip for learning guitar.",
    ]

    var body: some View {
        @Bindable var s = state
        VStack(alignment: .leading, spacing: 18) {
            StepHeader(title: "Ask a question",
                       explainer: "Type anything. When you press Ask, you will watch the inside of the model light up as it reads your question and writes its reply.")
            let g = state.geometry
            LearnCard([
                "Think of the **model** (the AI program answering you) as an office building with {floors} floors. Your question goes in at the bottom, and every floor works on it in turn.",
                "The building writes its reply **one word at a time**. After each word, all {floors} floors work again to pick the next one. That is why you can watch it think, word by word.",
                "Each floor is called a **layer** - think of it as a floor. They are numbered 0 at the bottom to {topFloor} at the top. Lower floors mostly read the words; higher floors decide what to say.",
            ].map(g.fill), words: [
                ("model", g.fill("the AI program, pictured here as a {floors}-floor office building")),
                ("layer", g.fill("one floor of the building, numbered 0 (bottom) to {topFloor} (top)")),
            ])

            TextEditor(text: $s.prompt)
                .font(.system(size: 14))
                .scrollContentBackground(.hidden)
                .padding(8)
                .background(Theme.background, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.1)))
                .frame(height: 120)
                .disabled(state.isGenerating)

            VStack(alignment: .leading, spacing: 8) {
                StepSection("Or try one of these")
                GuideFlowLayout(spacing: 6) {
                    ForEach(Self.examples, id: \.self) { q in
                        Button { state.prompt = q } label: {
                            Text(q).font(.caption).lineLimit(1)
                                .padding(.horizontal, 9).padding(.vertical, 4)
                                .background(Color.white.opacity(state.prompt == q ? 0.16 : 0.06), in: Capsule())
                        }
                        .buttonStyle(.plain)
                        .disabled(state.isGenerating)
                    }
                }
            }

            Spacer(minLength: 0)

            if state.isGenerating {
                Button(role: .destructive) { state.cancel() } label: {
                    Label("Stop", systemImage: "stop.fill").frame(maxWidth: .infinity)
                        .accessibilityLabel("Stop writing the reply")
                }
                .buttonStyle(.borderedProminent).controlSize(.large).tint(Theme.ablated)
            } else {
                Button { state.ask() } label: {
                    Label("Ask", systemImage: "sparkles").frame(maxWidth: .infinity)
                        .accessibilityLabel("Ask the model this question")
                }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(state.status != .ready || state.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Text("Shortcut: ⌘↩ (Command + Return)").font(.caption2).foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity)
            }
        }
    }
}
