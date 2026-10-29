import SwiftUI

/// One lesson page: the field fills the window (or the math toy, on the math pages), a tiny top bar, and one
/// caption card at the bottom with one idea and one button. Return continues, ← goes back, Esc leaves.
struct LessonPlayer: View {
    @Environment(AppState.self) private var state

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()
            if let page = state.currentPage {
                stage(for: page)
                    .transition(.opacity)
            }
            VStack(spacing: 0) {
                TopBar()
                Spacer(minLength: 0)
                if let page = state.currentPage {
                    CaptionCard(page: page)
                        .id(page.id)
                        .transition(.asymmetric(insertion: .opacity.combined(with: .offset(y: 10)), removal: .opacity))
                        .padding(.horizontal, 24)
                        .padding(.bottom, 28)
                }
            }
        }
        .animation(.easeInOut(duration: 0.35), value: state.currentPage?.id)
        .animation(.easeInOut(duration: 0.35), value: state.currentPage?.field == .hidden)
    }

    @ViewBuilder private func stage(for page: LessonPage) -> some View {
        if page.field == .hidden {
            if case .mathToy(let k) = page.action { MathToyView(stage: k) } else { Theme.background }
        } else {
            NeuronFieldView().frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

// MARK: - top bar: lesson title, page dots, exit

private struct TopBar: View {
    @Environment(AppState.self) private var state

    var body: some View {
        HStack(spacing: 14) {
            if let l = state.currentLesson {
                Text(l.title)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 12)
                HStack(spacing: 6) {
                    ForEach(l.pages.indices, id: \.self) { i in
                        Circle()
                            .fill(Color.white.opacity(i == state.pageIndex ? 0.85 : (i < state.pageIndex ? 0.35 : 0.14)))
                            .frame(width: 6, height: 6)
                    }
                }
                .accessibilityLabel("Page \(state.pageIndex + 1) of \(l.pages.count)")
                Spacer(minLength: 12)
            } else {
                Spacer()
            }
            Button { state.exitLesson() } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 26, height: 26)
                    .background(Circle().fill(Color.white.opacity(0.07)))
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.cancelAction)
            .help("Back to the lessons")
        }
        .padding(.leading, 84)   // clear the window's traffic-light buttons (hidden title bar)
        .padding(.trailing, 16)
        .frame(height: 44)
    }
}

// MARK: - the caption card: one idea, one button

private struct CaptionCard: View {
    @Environment(AppState.self) private var state
    let page: LessonPage

    var body: some View {
        VStack(spacing: 14) {
            Text(markdown(state.geometry.fill(page.text)))
                .font(.system(size: 21, weight: .regular))
                .lineSpacing(3)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            if let d = page.detail {
                Text(markdown(state.geometry.fill(d)))
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ReplyQuote()
            controls.padding(.top, 4)
        }
        .padding(.horizontal, 32)
        .padding(.top, 24)
        .padding(.bottom, 20)
        .frame(maxWidth: 640)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(.ultraThinMaterial)
                .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).fill(Theme.panel.opacity(0.55)))
        )
        .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Color.white.opacity(0.09)))
        .shadow(color: .black.opacity(0.35), radius: 24, y: 8)
    }

    private var controls: some View {
        ZStack {
            Button { state.nextPage() } label: {
                HStack(spacing: 8) {
                    if !state.pageReady { WaitingDot() }
                    Text(page.continueLabel)
                }
                .font(.system(size: 16, weight: .semibold))
                .padding(.horizontal, 22).padding(.vertical, 5)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .tint(Theme.answer.opacity(0.92))
            .foregroundStyle(Color.black)
            .disabled(!state.pageReady)
            .keyboardShortcut(.defaultAction)

            HStack {
                if state.pageIndex > 0 {
                    Button { state.previousPage() } label: {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 32, height: 32)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut(.leftArrow, modifiers: [])
                    .help("Back")
                }
                Spacer()
            }
        }
    }
}

/// A slow, quiet pulse while the page's action finishes (the AI is still writing, a change is being found…).
private struct WaitingDot: View {
    @State private var on = false
    var body: some View {
        Circle().fill(Color.black.opacity(0.6)).frame(width: 7, height: 7)
            .opacity(on ? 1 : 0.25)
            .onAppear { withAnimation(.easeInOut(duration: 0.7).repeatForever(autoreverses: true)) { on = true } }
            .accessibilityLabel("Working")
    }
}

// MARK: - the AI's reply, quoted under the text on the pages after Ask / Try the change

private struct ReplyQuote: View {
    @Environment(AppState.self) private var state

    /// The newest Ask or Try-the-change at or before this page decides what is quoted; math pages quote nothing.
    private var source: LessonAction? {
        guard let l = state.currentLesson, l.pages.indices.contains(state.pageIndex) else { return nil }
        if case .mathToy = l.pages[state.pageIndex].action { return nil }
        if l.pages[state.pageIndex].field == .hidden { return nil }
        for p in l.pages[...state.pageIndex].reversed() {
            switch p.action {
            case .ask, .tryChange: return p.action
            default: continue
            }
        }
        return nil
    }

    var body: some View {
        switch source {
        case .ask?:
            if let r = state.activeRun { quote(reply(r), writing: r.status == .running) }
        case .tryChange?:
            VStack(alignment: .leading, spacing: 6) {
                line("Before", state.beforeRun)
                line("After", state.afterRun)
            }
            .padding(.horizontal, 16).padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Color.white.opacity(0.05)))
        default:
            EmptyView()
        }
    }

    private func quote(_ text: String, writing: Bool) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Rectangle().fill(Theme.answer.opacity(0.5)).frame(width: 2)
            Text(text.isEmpty ? (writing ? "…" : "") : "“\(text)”" + (writing ? " …" : ""))
                .font(.system(size: 17, design: .serif).italic())
                .foregroundStyle(Theme.answer)
                .lineLimit(3)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .fixedSize(horizontal: false, vertical: true)
        .padding(.horizontal, 6)
    }

    private func line(_ label: String, _ run: Run?) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(label)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 50, alignment: .leading)
            if let r = run {
                let t = reply(r)
                Text(t.isEmpty ? "…" : "“\(t)”" + (r.status == .running ? " …" : ""))
                    .font(.system(size: 16, design: .serif).italic())
                    .foregroundStyle(label == "After" ? Theme.positive : Theme.answer)
                    .lineLimit(2)
            } else {
                Text("…").font(.system(size: 16)).foregroundStyle(.tertiary)
            }
        }
    }

    /// The written answer only: no control tokens, no thinking, trimmed.
    private func reply(_ r: Run) -> String {
        r.tokens.lazy
            .filter { $0.stream == .answer && !($0.text.hasPrefix("<|") && $0.text.hasSuffix("|>")) }
            .map(\.text).joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
