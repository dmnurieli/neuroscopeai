import SwiftUI

/// Generation settings (Advanced panel). The prompt itself is typed in the Ask step.
struct PromptPanel: View {
    @Environment(AppState.self) private var state
    @State private var temperature: Double = 0.7
    @State private var topP: Double = 0.9
    @State private var topK: Int = 40
    @State private var seed: UInt64 = 42

    private var isSampling: Bool {
        if case .sample = state.sampling { return true }
        return false
    }

    private func pushSampling() {
        state.sampling = .sample(temperature: Float(temperature), topP: Float(topP), topK: topK, seed: seed)
    }

    var body: some View {
        @Bindable var s = state
        let modeBinding = Binding<Bool>(
            get: { isSampling },
            set: { on in
                if on { pushSampling() } else { state.sampling = .greedy }
            })
        let ready = state.status == .ready
        VStack(alignment: .leading, spacing: 10) {
            Group {
            if state.hasThinking { Toggle("Thinking", isOn: $s.thinking) }
            Stepper(value: $s.maxTokens, in: 16...4096, step: 16) {
                Text("Max tokens: \(state.maxTokens)")
            }
            Picker("Decoding", selection: modeBinding) {
                Text("Greedy").tag(false)
                Text("Seeded sample").tag(true)
            }
            .pickerStyle(.segmented)

            if isSampling {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Temperature \(temperature, specifier: "%.2f")").font(.caption)
                        Slider(value: $temperature, in: 0.1...1.5).onChange(of: temperature) { pushSampling() }
                    }
                    HStack {
                        Text("Top-p \(topP, specifier: "%.2f")").font(.caption)
                        Slider(value: $topP, in: 0.05...1.0).onChange(of: topP) { pushSampling() }
                    }
                    Stepper(value: $topK, in: 1...200) { Text("Top-k: \(topK)").font(.caption) }
                        .onChange(of: topK) { pushSampling() }
                    HStack {
                        Text("Seed").font(.caption)
                        TextField("seed", value: $seed, format: .number.grouping(.never))
                            .textFieldStyle(.roundedBorder)
                            .font(.system(.caption, design: .monospaced))
                            .onChange(of: seed) { pushSampling() }
                        Button {
                            seed = seed &* 6364136223846793005 &+ 1
                        } label: { Image(systemName: "dice") }
                        .help("Roll a new seed")
                    }
                }
            }

            Toggle("Logit lens", isOn: $s.logitLens)
            HStack {
                Text("Slow motion \(state.paceMs) ms").font(.caption)
                Slider(value: Binding(get: { Double(state.paceMs) }, set: { state.paceMs = Int($0) }),
                       in: 0...300, step: 10)
            }
            }
            .disabled(state.isGenerating)

            HStack {
                if state.isGenerating {
                    Button(role: .destructive) { state.cancel() } label: {
                        Label("Stop", systemImage: "stop.fill").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent).tint(Theme.ablated)
                } else {
                    Button { state.runBaseline() } label: {
                        Label("Run baseline (no edits)", systemImage: "play.fill").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!ready)
                }
            }
            Button { state.replayWithEdits() } label: {
                Label("Replay with \(state.edits.count) edits", systemImage: "arrow.triangle.2.circlepath")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent).tint(Theme.scaled)
            .disabled(state.edits.isEmpty || state.isGenerating || !ready)
        }
        .onAppear {
            if case let .sample(t, p, k, sd) = state.sampling {
                temperature = Double(t); topP = Double(p); topK = k; seed = sd
            }
        }
    }
}
