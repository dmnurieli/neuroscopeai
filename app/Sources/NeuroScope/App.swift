import SwiftUI
import AppKit

@main
struct NeuroScopeApp: App {
    @State private var state: AppState
    init() {
        let s = AppState()
        // Activation reads as saturation, not glare: start well below the old full-gain exposure.
        s.view.exposure = 0.5
        _state = State(initialValue: s)
    }
    var body: some Scene {
        WindowGroup("NeuroScope") {
            ContentView()
                .environment(state)
                .preferredColorScheme(.dark)
                .frame(minWidth: 1200, minHeight: 760)
                .background(WindowSizer(width: 1600, height: 1000))
                .task { state.launch() }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1600, height: 1000)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .help) {
                Button("NeuroScope Guide") { state.showGuide = true }.keyboardShortcut("?", modifiers: .command)
            }
            CommandGroup(after: .newItem) {
                Divider()
                Button("Start Over") { if !state.isGenerating { state.startOver() } }
                    .keyboardShortcut("n", modifiers: [.command, .shift])
                Divider()
                Button("Save Changes…") { saveEdits() }.keyboardShortcut("s", modifiers: .command)
                Button("Load Changes…") { loadEdits() }.keyboardShortcut("o", modifiers: .command)
            }
            CommandGroup(after: .toolbar) {
                Button("Lessons") { state.exitLesson() }
                    .keyboardShortcut("1", modifiers: .command)
                Button("Explore") { if state.status == .ready { state.enterExplore() } }
                    .keyboardShortcut("2", modifiers: .command)
                Divider()
                Button(state.showAdvanced ? "Hide Advanced" : "Show Advanced") { state.showAdvanced.toggle() }
                    .keyboardShortcut("a", modifiers: [.command, .option])
                Button("Toggle layout") {
                    state.view.layout = state.view.layout == .stack ? .sheet : .stack
                    state.bumpDisplay()
                }.keyboardShortcut("l", modifiers: .command)
                Divider()
            }
            CommandMenu("Run") {
                Button("Ask") { if canRun { state.ask() } }.keyboardShortcut(.return, modifiers: .command)
                Button("Try It") { if canRun && !state.edits.isEmpty { state.tryIt() } }
                    .keyboardShortcut(.return, modifiers: [.command, .shift])
                Button("Stop") { state.cancel() }.keyboardShortcut(".", modifiers: .command)
                Button(state.slowMotion ? "End Slow Motion" : "Slow Motion") {
                    if state.slowMotion { state.stopSlowMotion() }
                    else if canRun, state.activeRun?.status == .finished { Task { await state.startSlowMotion() } }
                }.keyboardShortcut("s", modifiers: [.command, .shift])
                Button("Follow Selected Desk") {
                    if canRun, case .neuron(let n)? = state.selection, state.activeRun != nil, !state.isFollowing {
                        Task { await state.followDesk(n) }
                    }
                }.keyboardShortcut("f", modifiers: [.command, .option])
                Divider()
                Button("Previous token") { step(-1) }.keyboardShortcut("[", modifiers: .command)
                Button("Next token") { step(1) }.keyboardShortcut("]", modifiers: .command)
                Button("Play / Pause") { state.togglePlay() }.keyboardShortcut("p", modifiers: .command)
            }
        }
    }

    private var canRun: Bool {
        guard state.status == .ready, !state.isGenerating, !state.isContrasting else { return false }
        if case .preparing = state.experimentStatus { return false }
        return true
    }

    private func step(_ d: Int) {
        guard let i = state.displayIndex, let r = state.activeRun else { return }
        state.setCursor(max(0, min(r.count - 1, i + d)))
    }

    private func saveEdits() {
        let p = NSSavePanel()
        p.allowedContentTypes = [.json]
        p.nameFieldStringValue = "changes.json"
        if p.runModal() == .OK, let u = p.url { try? state.saveEdits(to: u) }
    }

    private func loadEdits() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.json]
        if p.runModal() == .OK, let u = p.url { try? state.loadEdits(from: u) }
    }
}

/// Sets the first window's content size once (defaultSize is ignored when AppKit restores a saved frame).
private struct WindowSizer: NSViewRepresentable {
    let width: CGFloat
    let height: CGFloat
    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        DispatchQueue.main.async { [weak v] in
            guard let w = v?.window else { return }
            let vis = (w.screen ?? NSScreen.main)?.visibleFrame ?? NSRect(x: 0, y: 0, width: width, height: height)
            let cw = min(width, vis.width), ch = min(height, vis.height)
            w.setContentSize(NSSize(width: cw, height: ch))
            var f = w.frame
            f.origin.x = vis.midX - f.width / 2
            f.origin.y = vis.maxY - f.height - (vis.height - f.height) / 2
            w.setFrameOrigin(f.origin)
        }
        return v
    }
    func updateNSView(_ nsView: NSView, context: Context) {}
}
