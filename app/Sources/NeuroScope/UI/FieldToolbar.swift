import SwiftUI

struct FieldToolbar: View {
    @Environment(AppState.self) private var state
    var body: some View {
        @Bindable var s = state
        let diffOK = state.baselineRun != nil && state.activeRunID != nil && state.activeRunID != state.baselineID
        let layout = Picker("Layout", selection: $s.view.layout) {
            Text("Stack").tag(LayoutMode.stack)
            Text("Sheet").tag(LayoutMode.sheet)
            Text("Cluster").tag(LayoutMode.cluster).disabled(state.clusterLayout == nil)
        }.pickerStyle(.segmented).labelsHidden()
        let clusterBtn = Button(state.isClustering ? "Clustering..." : "Compute cluster") {
            Task { await state.computeClusterLayout() }
        }.disabled(state.isClustering || state.activeRun == nil).fixedSize()
        let color = Picker("Color", selection: $s.view.colorMode) {
            Text("Activation").tag(ColorMode.activation)
            Text("Diff").tag(ColorMode.diff).disabled(!diffOK)
            Text("Contrast").tag(ColorMode.contrast).disabled(state.contrast == nil)
        }.pickerStyle(.segmented).labelsHidden()
        let heads = Toggle("Heads", isOn: $s.view.showHeads).toggleStyle(.checkbox).fixedSize()
        let reset = Button("Reset camera") {
            NotificationCenter.default.post(name: Notification.Name("NeuroScopeResetCamera"), object: nil)
        }.fixedSize()
        // Lives in the Advanced panel: one control per row.
        VStack(alignment: .leading, spacing: 8) {
            labeled("Layout") { layout }
            if state.clusterLayout == nil { clusterBtn }
            labeled("Color") { color }
            labeled("Intensity") { Slider(value: $s.view.exposure, in: 0.1...2) }
                .help("How many of each floor's busiest desks light up for the current word")
            labeled("Fade") { Slider(value: $s.view.glowHalfLife, in: 0.05...2) }
            labeled("Size") { Slider(value: $s.view.pointSize, in: 0.5...3) }
            HStack { heads; Spacer(minLength: 0); reset }
        }
        .font(.caption)
        .onChange(of: state.view.layout) { _, _ in state.bumpDisplay() }
        .onChange(of: diffOK) { _, ok in
            if !ok && state.view.colorMode == .diff { state.view.colorMode = .activation }
        }
    }

    private func labeled<C: View>(_ t: String, @ViewBuilder _ c: () -> C) -> some View {
        HStack(spacing: 6) { Text(t).foregroundStyle(.secondary).lineLimit(1).frame(width: 58, alignment: .leading); c() }
    }
}
