import SwiftUI

/// Contrast -> direction -> abliterate / steer. Advanced panel ("Custom directions"); scrolls with the panel.
struct DirectionsView: View {
    @Environment(AppState.self) private var state
    @State private var alpha: Float = 1.0
    @State private var top: (a: [(NeuronID, Float)], b: [(NeuronID, Float)]) = ([], [])
    @State private var topKey = ""

    private static let questions = [
        "Why is the sky blue?", "How do I boil an egg?", "What is the capital of Japan?",
        "Explain how a bicycle stays upright.", "What causes the tides?", "How does a refrigerator work?",
        "Give me a tip for learning guitar.", "What is photosynthesis?", "How do airplanes fly?",
        "Describe a good way to start the morning.", "What is a prime number?", "Why do cats purr?",
    ]
    private static func pairs(_ suffix: String) -> (String, String) {
        (questions.map { "\($0) \(suffix)" }.joined(separator: "\n"), questions.joined(separator: "\n"))
    }
    private static let presets: [(String, (String, String))] = [
        ("Answer in French", pairs("Answer in French.")),
        ("Formal tone", pairs("Use a very formal, professional tone.")),
        ("Talk like a pirate", pairs("Talk like a pirate.")),
        ("Be extremely brief", pairs("Be extremely brief, one short sentence.")),
    ]

    private func lines(_ s: String) -> Int { s.split(whereSeparator: \.isNewline).filter { !$0.allSatisfy(\.isWhitespace) }.count }

    var body: some View {
        @Bindable var s = state
        VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Contrast A/B → residual direction").font(.caption.bold()).foregroundStyle(.secondary)
                    Spacer()
                    Menu("Presets") {
                        ForEach(Self.presets, id: \.0) { p in
                            Button(p.0) { s.contrastA = p.1.0; s.contrastB = p.1.1 }
                        }
                    }.fixedSize().controlSize(.small)
                }
                editor("A - prompts WITH the concept", $s.contrastA)
                editor("B - matched prompts WITHOUT it", $s.contrastB)
                HStack {
                    Button {
                        Task { await state.computeContrast() }
                    } label: { Text("Compute direction") }
                        .buttonStyle(.borderedProminent)
                        .disabled(state.isContrasting || lines(state.contrastA) == 0 || lines(state.contrastB) == 0 || state.status != .ready)
                    if state.isContrasting { ProgressView().controlSize(.small) }
                    Spacer()
                    Button("Guide") { state.showGuide = true }.buttonStyle(.link).font(.caption)
                }
                if let e = state.contrastError {
                    Text(e).font(.caption).foregroundStyle(Theme.ablated).fixedSize(horizontal: false, vertical: true)
                }
                if let c = state.contrast { results(c) }
                else {
                    Text("Write prompts that differ only in the concept, compute the direction, then remove or add it. See the Guide for how this works.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
        }
    }

    private func editor(_ title: String, _ text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).font(.caption.bold()).foregroundStyle(.secondary)
                Spacer()
                Text("\(lines(text.wrappedValue)) lines").font(.caption2).foregroundStyle(.secondary)
            }
            TextEditor(text: text)
                .font(.system(size: 11, design: .monospaced))
                .scrollContentBackground(.hidden)
                .padding(4)
                .frame(height: 84)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.white.opacity(0.06)))
        }
    }

    @ViewBuilder private func results(_ c: ContrastResult) -> some View {
        let site = max(0, min(state.contrastSite ?? 0, c.directions.count - 1))
        let sep = site < c.separation.count ? c.separation[site] : 0
        let nrm = site < c.norms.count ? c.norms[site] : 0
        Divider()
        Text("Separation per site (A \(c.countA) vs B \(c.countB))").font(.caption.bold()).foregroundStyle(.secondary)
        SeparationChart(separation: c.separation, selected: site) { state.contrastSite = $0 }
            .frame(height: 70)
        Text("Site \(site) (\(site == 0 ? "embedding" : "after layer \(site - 1)"))  norm \(String(format: "%.2f", nrm))  separation d = \(String(format: "%.2f", sep))")
            .font(.caption.monospaced())
        VStack(alignment: .leading, spacing: 6) {
            Button { state.addDirectionEdit(site: site, mode: .ablate, value: 0, allSites: true) } label: {
                Text("Abliterate (remove at all layers)").frame(maxWidth: .infinity)
            }.buttonStyle(.bordered).tint(Theme.ablated)
            HStack {
                Slider(value: $alpha, in: -3...3)
                Text(String(format: "%+.1f x norm", alpha)).font(.caption.monospaced()).frame(width: 82, alignment: .trailing)
            }
            Button { state.addDirectionEdit(site: site, mode: .add, value: alpha * nrm, allSites: false) } label: {
                Text("Steer toward A").frame(maxWidth: .infinity)
            }.buttonStyle(.bordered).tint(Theme.scaled)
            HStack {
                Button("Show neuron map") { state.view.colorMode = .contrast; state.bumpDisplay() }
                Button(state.isClustering ? "Clustering..." : "Cluster layout") { Task { await state.computeClusterLayout() } }
                    .disabled(state.isClustering || state.activeRun == nil)
            }
        }.controlSize(.small)
        Text("Then press Replay with edits (Generation section).").font(.caption2).foregroundStyle(.secondary)
        experimentOverride(c)
        Divider()
        neuronLists(c)
    }

    /// The guided experiment's derived site/strength, and a way to override them with the selected site + alpha.
    @ViewBuilder private func experimentOverride(_ c: ContrastResult) -> some View {
        if let e = state.experiment {
            let site = max(0, min(state.contrastSite ?? 0, c.directions.count - 1))
            let nrm = site < c.norms.count ? c.norms[site] : 0
            Divider()
            Text("Experiment \"\(e.title)\" (\(e.mode == .ablate ? "ablate, all sites" : "add"))").font(.caption.bold()).foregroundStyle(.secondary)
            Text("Derived: site \(state.experimentSite.map(String.init) ?? "-")  value \(state.experimentValue.map { String(format: "%.2f", $0) } ?? "-")")
                .font(.caption.monospaced())
            Button("Override with site \(site)\(e.mode == .add ? String(format: ", %+.1f x norm", alpha) : "")") {
                for d in state.edits.directions { state.removeDirectionEdit(d.id) }
                let v: Float = e.mode == .add ? alpha * nrm : 0
                state.addDirectionEdit(site: site, mode: e.mode, value: v, allSites: e.mode == .ablate)
                state.experimentSite = site
                state.experimentValue = v
            }
            .controlSize(.small)
            .help("Replaces the direction edits with one from the selected site (chart above) and the alpha slider")
        }
    }

    private func neuronLists(_ c: ContrastResult) -> some View {
        let key = "\(c.countA)|\(c.countB)|\(c.neuronDiff.count)|\(c.neuronDiff.first ?? 0)|\(c.neuronDiff.last ?? 0)"
        return VStack(alignment: .leading, spacing: 6) {
            list("Top neurons for A", top.a)
            list("Top neurons for B", top.b)
        }
        .onAppear { refresh(c, key) }
        .onChange(of: key) { _, k in refresh(c, k) }
    }

    private func refresh(_ c: ContrastResult, _ key: String) {
        guard key != topKey || (top.a.isEmpty && top.b.isEmpty) else { return }
        topKey = key
        let n = state.geometry.numNeurons
        let idx = Array(c.neuronDiff.indices)
        func pick(_ order: (Float, Float) -> Bool) -> [(NeuronID, Float)] {
            // partial selection: sort indices by value, keep 12
            idx.sorted { order(c.neuronDiff[$0], c.neuronDiff[$1]) }.prefix(12).map {
                (NeuronID(layer: $0 / n, index: $0 % n), c.neuronDiff[$0])
            }
        }
        top = (pick(>), pick(<))
    }

    private func list(_ title: String, _ items: [(NeuronID, Float)]) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption.bold()).foregroundStyle(.secondary)
            ForEach(items, id: \.0) { id, v in
                HStack {
                    Text("L\(id.layer) · N\(id.index)").font(.system(.caption, design: .monospaced))
                    Spacer()
                    Text(String(format: "%+.3f", v)).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                }
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(RoundedRectangle(cornerRadius: 4).fill(Color.white.opacity(0.05)))
                .contentShape(Rectangle())
                .onTapGesture { state.select(.neuron(id)) }
            }
        }
    }
}

private struct SeparationChart: View {
    let separation: [Float]
    let selected: Int
    let onSelect: (Int) -> Void
    var body: some View {
        let best = separation.indices.max { separation[$0] < separation[$1] } ?? 0
        Canvas { ctx, size in
            let n = max(separation.count, 1)
            let mx = CGFloat(max(separation.max() ?? 1, 1e-6))
            let w = size.width / CGFloat(n)
            for i in separation.indices {
                let h = max(1, CGFloat(max(separation[i], 0)) / mx * (size.height - 4))
                let r = CGRect(x: CGFloat(i) * w + 0.5, y: size.height - h, width: max(w - 1, 1), height: h)
                let col: Color = i == best ? Theme.pinned : (i == selected ? Theme.positive : Theme.negative.opacity(0.6))
                ctx.fill(Path(r), with: .color(col))
            }
            let sx = CGFloat(selected) * w
            ctx.stroke(Path(CGRect(x: sx, y: 0, width: w, height: size.height)), with: .color(.white.opacity(0.6)), lineWidth: 1)
        }
        .background(Color.white.opacity(0.04))
        .overlay(alignment: .bottomLeading) { Text("L0").font(.system(size: 8)).foregroundStyle(.secondary).padding(1) }
        .overlay(alignment: .bottomTrailing) { Text("L\(max(separation.count - 1, 0))").font(.system(size: 8)).foregroundStyle(.secondary).padding(1) }
        .overlay(GeometryReader { geo in
            Color.clear.contentShape(Rectangle()).gesture(
                DragGesture(minimumDistance: 0).onChanged { g in
                    let i = Int(g.location.x / max(geo.size.width, 1) * CGFloat(separation.count))
                    onSelect(max(0, min(separation.count - 1, i)))
                })
        })
    }
}
