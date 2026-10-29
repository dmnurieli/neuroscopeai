// SwiftUI host for the renderer: an MTKView subclass handling mouse/trackpad, plus hover tooltip.
// Reads AppState each frame: when state.displayVersion changes it re-uploads the displayed token
// (active run at state.displayIndex; diff vs baseline when view.colorMode == .diff).
// Click selects (state.select), hover sets state.hover, double-click focuses the camera on a layer.
// Slow motion (state.slowMotion): pushes state.trace / traceProgress / traceFocus to the renderer every frame and
// keeps the display link running while playing; a click on a desk asks "why did this desk fire?"
// (state.explainDesk). Floor numbers sit above each column and an "explained" strip below, as a SwiftUI overlay
// re-projected only when the camera or view size changes.

import MetalKit
import SwiftUI

/// Mouse position (view points, origin top-left) shared between the MTKView and the SwiftUI overlay.
@MainActor @Observable
private final class FieldPointer {
    var point: CGPoint = .zero
    var inside = false
}

/// Screen anchors for the slow-motion floor labels, published by the coordinator only when they move.
@MainActor @Observable
private final class FieldLabels {
    struct Floor: Identifiable, Equatable {
        var id: Int { floor }
        var floor: Int
        var top: CGPoint?
        var bottom: CGPoint?
    }
    var floors: [Floor] = []
}

/// MTKView with orbit / pan / zoom / hover / click handling. Callbacks keep logic in the coordinator.
private final class FieldMTKView: MTKView {
    var onMove: ((CGPoint) -> Void)?
    var onExit: (() -> Void)?
    var onClick: ((CGPoint, Int) -> Void)?
    var onOrbit: ((Float, Float) -> Void)?
    var onPan: ((Float, Float) -> Void)?
    var onZoom: ((Float) -> Void)?
    var onActivity: (() -> Void)?

    private var dragDistance: CGFloat = 0
    private var tracking: NSTrackingArea?

    override var acceptsFirstResponder: Bool { true }
    override var isFlipped: Bool { true }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let t = tracking { removeTrackingArea(t) }
        let t = NSTrackingArea(rect: bounds,
                               options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
                               owner: self, userInfo: nil)
        addTrackingArea(t)
        tracking = t
    }

    private func local(_ e: NSEvent) -> CGPoint {
        let p = convert(e.locationInWindow, from: nil)
        return isFlipped ? p : CGPoint(x: p.x, y: bounds.height - p.y)
    }

    override func mouseMoved(with event: NSEvent) { onMove?(local(event)) }
    override func mouseExited(with event: NSEvent) { onExit?() }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        dragDistance = 0
    }

    override func mouseDragged(with event: NSEvent) {
        let dx = event.deltaX, dy = event.deltaY
        dragDistance += abs(dx) + abs(dy)
        if event.modifierFlags.contains(.shift) || event.modifierFlags.contains(.option) {
            onPan?(Float(dx), Float(dy))
        } else {
            onOrbit?(Float(dx), Float(dy))
        }
        onActivity?()
    }

    override func mouseUp(with event: NSEvent) {
        if dragDistance < 4 { onClick?(local(event), event.clickCount) }
    }

    override func scrollWheel(with event: NSEvent) {
        let d = Float(event.scrollingDeltaY) * (event.hasPreciseScrollingDeltas ? 0.01 : 0.05)
        if d != 0 { onZoom?(exp(-d)) }
    }

    override func magnify(with event: NSEvent) {
        onZoom?(exp(-Float(event.magnification)))
    }

    // Right-drag (and middle-drag) pans, same as shift/option + left-drag.
    override func rightMouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
    }

    override func rightMouseDragged(with event: NSEvent) {
        onPan?(Float(event.deltaX), Float(event.deltaY))
        onActivity?()
    }

    override func otherMouseDragged(with event: NSEvent) {
        onPan?(Float(event.deltaX), Float(event.deltaY))
        onActivity?()
    }
}

struct NeuronFieldView: View {
    @Environment(AppState.self) private var state
    @State private var pointer = FieldPointer()
    @State private var labels = FieldLabels()

    var body: some View {
        ZStack(alignment: .topLeading) {
            FieldRepresentable(state: state, pointer: pointer, labels: labels)
            if state.slowMotion && (state.mode != .lesson || state.view.layout == .columns) { FloorOverlay(labels: labels) }
            if state.mode != .lesson { HUD() }   // lessons: the caption card says what is shown
            Tooltip(pointer: pointer)
        }
        .background(Theme.background)
    }
}

private struct FieldRepresentable: NSViewRepresentable {
    let state: AppState
    let pointer: FieldPointer
    let labels: FieldLabels

    func makeCoordinator() -> Coordinator { Coordinator(state: state, pointer: pointer, labels: labels) }

    func makeNSView(context: Context) -> MTKView {
        let c = context.coordinator
        let v = FieldMTKView(frame: .zero, device: c.device)
        v.colorPixelFormat = .bgra8Unorm
        v.framebufferOnly = true
        v.preferredFramesPerSecond = 120
        v.clearColor = MTLClearColor(red: 0.02, green: 0.027, blue: 0.05, alpha: 1)
        v.delegate = c
        c.attach(v)
        return v
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        // Reading these registers observation so SwiftUI re-invokes us when they change.
        _ = state.displayVersion; _ = state.view; _ = state.selection; _ = state.hover; _ = state.edits
        _ = state.contrast; _ = state.clusterLayout
        _ = state.view.highlightFloors; _ = state.slowMotion; _ = state.trace?.position; _ = state.traceFocus; _ = state.tracePaused
        // Paused playback can still be scrubbed: redraw on progress changes then.
        if state.slowMotion && state.tracePaused { _ = state.traceProgress }
        nsView.isPaused = false
    }

    @MainActor
    final class Coordinator: NSObject, MTKViewDelegate {
        let state: AppState
        let pointer: FieldPointer
        let labels: FieldLabels
        let device: MTLDevice
        let renderer: NeuronRenderer?
        weak var view: FieldMTKView?
        private var lastDrawn = -1
        private var lastEdits = EditSet()
        private var lastSelection: Selection?
        private var lastHover: Selection?
        private var lastLayout: LayoutMode
        private var diffBuf = [Float]()
        private var lastDiffMode = false
        private var lastContrastMode = false
        private var lastCluster: [[SIMD2<Float>]]?
        private var lastPickTime: CFTimeInterval = 0
        private var pendingMove: CGPoint?
        private var moveTimer: Timer?
        private var observer: NSObjectProtocol?
        private var traceShown = false
        private var camTween: (from: Camera, to: Camera, start: CFTimeInterval)?
        private var lastHighlight: Set<Int>?
        private var labelKey: [Float] = []

        init(state: AppState, pointer: FieldPointer, labels: FieldLabels) {
            self.state = state
            self.pointer = pointer
            self.labels = labels
            self.device = MTLCreateSystemDefaultDevice()!
            self.renderer = NeuronRenderer(device: device, geometry: state.geometry)
            self.lastLayout = state.view.layout
            super.init()
            renderer?.camera = Camera.framing(for: lastLayout)
            observer = NotificationCenter.default.addObserver(
                forName: Notification.Name("NeuroScopeResetCamera"), object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated {
                    guard let self, let r = self.renderer else { return }
                    r.camera = Camera.framing(for: r.layout, aspect: self.viewAspect)
                    self.wake()
                }
            }
        }

        deinit { if let o = observer { NotificationCenter.default.removeObserver(o) } }

        func wake() { view?.isPaused = false }

        /// Width / height of the field, for fitting the camera to the view's real shape.
        var viewAspect: Float {
            guard let b = view?.bounds, b.height > 1 else { return 1.6 }
            return Float(b.width / b.height)
        }

        func attach(_ v: FieldMTKView) {
            view = v
            v.onActivity = { [weak self] in self?.wake() }
            v.onOrbit = { [weak self] dx, dy in self?.renderer?.camera.orbit(dx: dx, dy: dy) }
            v.onPan = { [weak self, weak v] dx, dy in
                guard let v else { return }
                self?.renderer?.camera.pan(dx: dx, dy: dy, viewSize: v.bounds.size)
                self?.wake()
            }
            v.onZoom = { [weak self] f in self?.renderer?.camera.zoom(by: f); self?.wake() }
            v.onMove = { [weak self, weak v] p in
                guard let self, let v else { return }
                self.pointer.point = p
                self.pointer.inside = true
                // Throttle the CPU pick (scans every unit) to ~30 Hz, with a trailing pick so the
                // final resting position is always resolved.
                let now = CACurrentMediaTime()
                if now - self.lastPickTime >= 0.033 {
                    self.runPick(p, v)
                } else {
                    self.pendingMove = p
                    if self.moveTimer == nil {
                        self.moveTimer = Timer.scheduledTimer(withTimeInterval: 0.035, repeats: false) { [weak self, weak v] _ in
                            MainActor.assumeIsolated {
                                guard let self else { return }
                                self.moveTimer = nil
                                if let p = self.pendingMove, let v, self.pointer.inside { self.runPick(p, v) }
                                self.pendingMove = nil
                            }
                        }
                    }
                }
                self.wake()
            }
            v.onExit = { [weak self] in
                guard let self else { return }
                self.pendingMove = nil
                self.pointer.inside = false
                if self.state.hover != nil { self.state.hover = nil }
                self.wake()
            }
            v.onClick = { [weak self, weak v] p, count in
                guard let self, let v, let r = self.renderer else { return }
                let hit = r.pick(at: p, viewSize: v.bounds.size)
                if count >= 2 {
                    let layer: Int?
                    switch hit {
                    case .neuron(let n)?: layer = n.layer
                    case .head(let h)?: layer = h.layer
                    case nil: layer = nil
                    }
                    if let l = layer {
                        r.camera.target = NeuronLayout.layerCenter(l, mode: r.layout, geometry: self.state.geometry)
                        r.camera.distance = min(r.camera.distance, 1.4)
                    }
                } else if self.state.slowMotion {
                    // Slow motion: a click on a desk asks why it fired; a click on empty space clears the focus.
                    switch hit {
                    case .neuron(let n)?:
                        let st = self.state
                        Task { await st.explainDesk(n) }
                    case nil where self.state.traceFocus != nil:
                        let st = self.state
                        Task { await st.explainDesk(nil) }
                    default: break
                    }
                } else {
                    self.state.select(hit)
                }
                self.wake()
            }
        }

        private func runPick(_ p: CGPoint, _ v: MTKView) {
            guard let r = renderer else { return }
            lastPickTime = CACurrentMediaTime()
            pendingMove = nil
            let h = r.pick(at: p, viewSize: v.bounds.size)
            if h != state.hover { state.hover = h }
        }

        /// Diff is only drawn when a different baseline run exists; otherwise the activation view is shown.
        private var wantsDiff: Bool {
            guard state.view.colorMode == .diff, let run = state.activeRun, let base = state.baselineRun
            else { return false }
            return base.id != run.id
        }

        /// Contrast colouring needs a computed ContrastResult; otherwise the activation view is shown.
        private var wantsContrast: Bool { state.view.colorMode == .contrast && state.contrast != nil }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            renderer?.mtkView(view, drawableSizeWillChange: size)
        }

        func draw(in view: MTKView) {
            guard let r = renderer else { return }
            // Cluster positions first, so a simultaneous switch to .cluster morphs straight to them.
            if state.clusterLayout != lastCluster {
                lastCluster = state.clusterLayout
                r.setClusterLayout(lastCluster)
            }
            if state.view.layout != lastLayout {
                let old = lastLayout
                lastLayout = state.view.layout
                r.setLayout(lastLayout, animated: true)
                // Into / out of the flat columns diagram: glide the camera to that layout's framing.
                if old == .columns || lastLayout == .columns {
                    camTween = (r.camera, Camera.framing(for: lastLayout, aspect: viewAspect), CACurrentMediaTime())
                }
            }
            if state.view.highlightFloors != lastHighlight {
                lastHighlight = state.view.highlightFloors
                if lastLayout != .columns {
                    var to = Camera.framing(for: lastLayout, aspect: viewAspect)
                    if let hl = lastHighlight, hl.count == 1, let l = hl.first {
                        to.target = NeuronLayout.layerCenter(l, mode: lastLayout, geometry: state.geometry)
                        to.distance = 2.6
                    }
                    camTween = (r.camera, to, CACurrentMediaTime())
                }
            }
            if let tw = camTween {
                let t = Float(min((CACurrentMediaTime() - tw.start) / 0.6, 1))
                let e = t < 0.5 ? 4 * t * t * t : 1 - pow(-2 * t + 2, 3) / 2
                var c = tw.to
                c.yaw = tw.from.yaw + (tw.to.yaw - tw.from.yaw) * e
                c.pitch = tw.from.pitch + (tw.to.pitch - tw.from.pitch) * e
                c.distance = tw.from.distance + (tw.to.distance - tw.from.distance) * e
                c.target = simd_mix(tw.from.target, tw.to.target, SIMD3(repeating: e))
                r.camera = c
                if t >= 1 { camTween = nil }
            }
            if state.slowMotion, let tr = state.trace {
                r.setTrace(tr.edges, focusEdges: tr.focusEdges, progress: Float(state.traceProgress), focus: state.traceFocus)
                traceShown = true
            } else if traceShown {
                r.clearTrace()
                traceShown = false
            }
            updateLabels(r, view)
            r.settings = state.view
            let diffNow = wantsDiff
            let contrastNow = wantsContrast
            r.settings.colorMode = contrastNow ? .contrast : (diffNow ? .diff : .activation)

            // A color-mode / baseline change alters what must be uploaded without bumping displayVersion.
            let changed = state.displayVersion != lastDrawn || diffNow != lastDiffMode || contrastNow != lastContrastMode
            lastDiffMode = diffNow
            lastContrastMode = contrastNow
            if changed {
                lastDrawn = state.displayVersion
                upload(r)
            }
            if changed || state.edits != lastEdits || state.selection != lastSelection || state.hover != lastHover {
                lastEdits = state.edits; lastSelection = state.selection; lastHover = state.hover
                r.setMarkers(edits: state.edits, selection: state.selection, hover: state.hover)
            }
            r.draw(in: view)
            // Idle: stop the display link; any change (event / state) un-pauses.
            let playing = state.slowMotion && !state.tracePaused
            if !changed && !r.isAnimating && state.displayVersion == lastDrawn && moveTimer == nil
                && camTween == nil && !playing {
                view.isPaused = true
            }
        }

        /// Re-projects the floor label anchors when the camera, view size or layout moved (not every frame).
        private func updateLabels(_ r: NeuronRenderer, _ view: MTKView) {
            guard state.slowMotion else {
                if !labels.floors.isEmpty { labels.floors = []; labelKey = [] }
                return
            }
            let size = view.bounds.size
            let c = r.camera
            let key: [Float] = [c.yaw, c.pitch, c.distance, c.target.x, c.target.y, c.target.z,
                                Float(size.width), Float(size.height), Float(LayoutMode.allCases.firstIndex(of: r.layout) ?? 0)]
            guard key != labelKey else { return }
            labelKey = key
            let g = state.geometry
            var out: [FieldLabels.Floor] = []
            out.reserveCapacity(g.numLayers)
            for l in 0..<g.numLayers {
                let a = NeuronLayout.layerLabelAnchors(l, mode: r.layout, geometry: g)
                out.append(FieldLabels.Floor(floor: l, top: c.project(a.top, viewSize: size)?.point,
                                             bottom: c.project(a.bottom, viewSize: size)?.point))
            }
            if out != labels.floors { labels.floors = out }
        }

        private func upload(_ r: NeuronRenderer) {
            if wantsContrast, let c = state.contrast {
                r.showContrast(c.neuronDiff)
                state.pulsePending = false
                return
            }
            guard let run = state.activeRun, let t = state.displayIndex else { return }
            let sc = state.scales(for: run)
            if wantsDiff, let base = state.baselineRun {
                Analysis.diff(run, base, at: t, into: &diffBuf)
                r.showDiff(diffBuf, layerScales: sc.layer)
            } else {
                let pulse = state.pulsePending
                run.withNeurons(at: t) { n in
                    run.withHeads(at: t) { h in
                        r.show(neurons: n, heads: h, layerScales: sc.layer, headScales: sc.head, pulse: pulse)
                    }
                }
                state.pulsePending = false
            }
        }
    }
}

/// Slow-motion overlay: a tiny floor number over each column, and under it a small bar for the share of that
/// floor's busiest desks' input that the drawn arrows explain (the rest comes via attention and the word itself).
private struct FloorOverlay: View {
    @Environment(AppState.self) private var state
    let labels: FieldLabels

    var body: some View {
        // Lessons show only the floor numbers; the explained-% bars belong to Explore.
        let explained = state.mode == .lesson ? [] : (state.trace?.explained ?? [])
        ZStack(alignment: .topLeading) {
            ForEach(labels.floors) { f in
                if let p = f.top {
                    Text("\(f.floor)")
                        .font(.system(size: 9, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.62))
                        .fixedSize()
                        .position(x: p.x, y: p.y - 7)
                }
                if let p = f.bottom, f.floor >= 1, f.floor < explained.count, explained[f.floor].isFinite {
                    ExplainedBar(value: explained[f.floor])
                        .position(x: p.x, y: p.y + 18)
                }
            }
            if !explained.isEmpty, let y = labels.floors.compactMap({ $0.bottom?.y }).max(),
               let x0 = labels.floors.compactMap({ $0.bottom?.x }).min() {
                Text("Bars: how much of each floor's busiest desks' input these arrows explain. The rest comes from attention (notes borrowed from earlier words) and the word itself.")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .frame(maxWidth: 520, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .offset(x: max(x0 - 4, 8), y: y + 40)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .allowsHitTesting(false)
    }
}

private struct ExplainedBar: View {
    let value: Float
    var body: some View {
        let v = CGFloat(min(max(value, 0), 1))
        VStack(spacing: 2) {
            ZStack(alignment: .bottom) {
                Rectangle().fill(.white.opacity(0.12)).frame(width: 5, height: 20)
                Rectangle().fill(.white.opacity(0.75)).frame(width: 5, height: 20 * v)
            }
            Text("\(Int((v * 100).rounded()))%")
                .font(.system(size: 8, design: .monospaced))
                .foregroundStyle(.white.opacity(0.6))
                .fixedSize()
        }
        .help("Share of this floor's busiest desks' input explained by the arrows")
    }
}

private struct HUD: View {
    @Environment(AppState.self) private var state
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let run = state.activeRun, let t = state.displayIndex, t < run.tokens.count {
                let tok = run.tokens[t]
                Text(tok.displayText.isEmpty ? "·" : tok.displayText.replacingOccurrences(of: "\n", with: "⏎"))
                    .font(.system(.title3, design: .monospaced).weight(.semibold))
                    .foregroundStyle(color(tok.stream))
                    .lineLimit(1)
                Text(phase(run, t, tok))
                    .font(.system(.caption, design: .monospaced).italic())
                    .foregroundStyle(.secondary)
                Text("word \(t + 1) of \(run.count)" + (state.showAdvanced ? (run.summary?.decodeTokPerS.map { String(format: " · %.1f tok/s", $0) } ?? "") : ""))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(8)
        .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 6))
        .padding(10)
        .opacity(state.activeRun == nil ? 0 : 1)
        .allowsHitTesting(false)
    }

    private func phase(_ run: Run, _ t: Int, _ tok: TokenRecord) -> String {
        if tok.text == "<|im_end|>" || tok.text == "<|endoftext|>", t >= run.promptCount { return "(finished)" }
        if t < run.promptCount { return "reading your question" }
        switch tok.stream {
        case .thinking: return "writing notes to itself"
        case .answer: return "writing the reply"
        default: return t == run.count - 1 && run.status == .finished ? "(finished)" : "writing the reply"
        }
    }

    private func color(_ s: StreamKind) -> Color {
        switch s {
        case .prompt: Theme.prompt
        case .thinking: Theme.thinking
        case .answer: Theme.answer
        case .special: Theme.special
        }
    }
}

private struct Tooltip: View {
    @Environment(AppState.self) private var state
    let pointer: FieldPointer

    var body: some View {
        if pointer.inside, let h = state.hover, let text = label(h) {
            Text(text)
                .font(.system(.caption, design: .monospaced))
                .padding(.horizontal, 6).padding(.vertical, 3)
                .background(.black.opacity(0.75), in: RoundedRectangle(cornerRadius: 4))
                .foregroundStyle(.white)
                .fixedSize()
                .offset(x: pointer.point.x + 14, y: pointer.point.y + 14)
                .allowsHitTesting(false)
        }
    }

    private func label(_ s: Selection) -> String? {
        let run = state.activeRun
        let t = state.displayIndex
        switch s {
        case .neuron(let n):
            var str = "L\(n.layer) · N\(n.index)"
            if let run, let t { str += String(format: "  %+.2f", run.activation(n, at: t)) }
            return str
        case .head(let h):
            var str = "L\(h.layer) · H\(h.head)"
            if let run, let t { str += String(format: "  %.2f", run.headNorm(h, at: t)) }
            return str
        }
    }
}
