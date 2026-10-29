// MTKViewDelegate that draws ~86k units as crisp round dots. Data path per displayed token:
// Float16 activations (172 KB) → per-floor rank cutoffs on the CPU (Salience.swift) → MTLBuffer → compute kernel
// (salience by rank per floor, persisted value = max(value·decay, new)) → instanced dot pass into a 4x MSAA bgra8
// target with a depth buffer (opaque, depth-tested: overlapping planes occlude, never sum) → resolved straight into
// the drawable. Activation is drawn as colour saturation and dot size (see `salience` and `saturation` in
// Shaders.swift); there is no bloom.
//
// MTKView requirements (set by the hosting field view):
//   view.colorPixelFormat = .bgra8Unorm, view.depthStencilPixelFormat = .invalid,
//   view.framebufferOnly = true, view.preferredFramesPerSecond = 120,
//   view.device = renderer.device, view.delegate = renderer.
//   The view may pause (isPaused / enableSetNeedsDisplay) whenever `isAnimating` is false.

import Metal
import MetalKit
import QuartzCore
import ImageIO
import simd

// Swift mirrors of the MSL structs in Shaders.swift.
private struct NSUniforms {
    var view: simd_float4x4
    var proj: simd_float4x4
    var viewport: SIMD2<Float>
    var morph: Float
    var exposure: Float
    var threshold: Float
    var flare: Float
    var pointSize: Float
    var colorMode: UInt32
    var numNeurons: UInt32
    var numUnits: UInt32
    var minPx: Float
    var fadeNear: Float
    var fadeFar: Float
    var traceOn: Float = 0
    var traceDim: Float = 0.25
    var traceMap: Float = 0        // 0…1: how far the trace-ordered column map has taken hold (dust shrinks, nodes grow)
    var hlOn: Float = 0            // 1 = lesson floor highlight in force (per-floor weights below)
    var neuronsPerLayer: UInt32 = 1
    var headsPerLayer: UInt32 = 1
    var hlPad: Float = 0
    var hl: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>,
             SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
        = (.one, .one, .one, .one, .one, .one, .one, .one, .one, .one, .one, .one, .one, .one, .one, .one)
}

private struct NSTargetParams {
    var numNeurons: UInt32, numHeads: UInt32, neuronsPerLayer: UInt32, headsPerLayer: UInt32
    var numLayers: UInt32, mode: UInt32, pad0: UInt32 = 0, pad1: UInt32 = 0
}

private struct NSGlowParams {
    var numUnits: UInt32
    var decay: Float
    var snap: UInt32
    var pulse: UInt32
}

private struct NSMarker {
    var unit: UInt32
    var kind: UInt32      // 0 ablated, 1 scaled, 2 pinned, 3 selection, 4 hover, 5 slow-motion source outline, 6/7 trace node ring (amber / blue)
}

private struct NSArrow {
    var from: UInt32
    var to: UInt32
    var weight: Float     // signed, normalized to its step's strongest (−1…1)
    var width: Float      // points: 0.5…2 (1…4 device px on Retina), by |weight| over the step's strongest
    var start: Float
    var end: Float
    var alpha: Float
    var focus: UInt32
}

private struct NSArrowParams {
    var progress: Float
    var scale: Float
    var pad0: Float = 0
    var pad1: Float = 0
}

/// CPU side of one arrow, for the per-frame "arriving sum" fill of its target desk.
private struct TraceArrival {
    var target: Int       // slot in traceTargets, or −1 (no fill contribution)
    var fill: Float       // weight / the target's normalizer
    var start: Float
    var end: Float
}

final class NeuronRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let geometry: Geometry
    var settings = ViewSettings()
    var camera = Camera()
    /// Animate the layout change (stack ⇄ sheet) over ~0.6 s.
    private(set) var layout: LayoutMode = .stack

    // Pipeline state
    private let queue: MTLCommandQueue
    private let spritePSO: MTLRenderPipelineState
    private let markerPSO: MTLRenderPipelineState
    private let arrowPSO: MTLRenderPipelineState
    private let targetsPSO: MTLComputePipelineState
    private let glowPSO: MTLComputePipelineState
    private let dotDepth: MTLDepthStencilState
    private let markerDepth: MTLDepthStencilState
    private static let sampleCount = 4
    private static let colorFormat: MTLPixelFormat = .bgra8Unorm
    private static let depthFormat: MTLPixelFormat = .depth32Float

    // Geometry
    private let numNeurons: Int
    private let numHeads: Int
    private let numUnits: Int
    private let posA: MTLBuffer
    private let posB: MTLBuffer
    private var cpuFrom: [SIMD3<Float>]
    private var cpuTo: [SIMD3<Float>]
    private var morphStart: CFTimeInterval = 0
    /// Per-floor strength (1 = full, 0 = dust) easing toward ViewSettings.highlightFloors over 0.4 s.
    private var hlWeights = [Float](repeating: 1, count: 64)
    private var hlEngaged = false

    private func hlTarget(_ f: Int) -> Float {
        guard let set = settings.highlightFloors else { return 1 }
        return set.contains(f) ? 1 : 0
    }
    private var hlSettled: Bool {
        for f in 0..<64 where abs(hlWeights[f] - hlTarget(f)) > 0.002 { return false }
        return true
    }
    private var morphing = false
    private var morphT: Float = 1
    private var morphFromScale: Float = 1
    private var clusterXY: [[SIMD2<Float>]]?
    /// Median nearest-neighbour desk distance in the .cluster layout (sets its dot radius); nil = grid fallback.
    private var clusterSpacing: Float?

    // Data (triple-buffered inputs, GPU-only targets/glow)
    private static let inflight = 3
    private let semaphore = DispatchSemaphore(value: NeuronRenderer.inflight)
    private var frameSlot = 0
    private var uniformBufs: [MTLBuffer] = []
    private var neuronBufs: [MTLBuffer] = []
    private var headBufs: [MTLBuffer] = []
    private var diffBufs: [MTLBuffer] = []
    private var layerScaleBufs: [MTLBuffer] = []
    private var headScaleBufs: [MTLBuffer] = []
    private var cutoffBufs: [MTLBuffer] = []
    private let targets: MTLBuffer
    private let glow: MTLBuffer

    // CPU staging written by show()/showDiff(); copied into the ring in draw() when dirty.
    private var pendingNeurons: [Float16]
    private var pendingHeads: [Float]
    private var pendingDiff: [Float]
    private var pendingLayerScales: [Float]
    private var pendingHeadScales: [Float]
    /// Per-floor rank cutoffs for the word on display (Salience.cutoffs), recomputed when the word or Intensity changes.
    private var pendingCutoffs: [SIMD4<Float>]
    private var cutoffScratch: [Float] = []
    private var cutoffExposure: Float = .nan
    private var pendingMode: UInt32 = 0
    private var dataDirty = false
    private var pendingSnap = true
    private var pendingPulse = false
    private var lastDataTime: CFTimeInterval = 0
    private var flareStart: CFTimeInterval = -10

    // Markers
    private var markerBufs: [MTLBuffer] = []
    private var markerCapacity = 4096
    private var pendingMarkers: [NSMarker] = []
    private var markerCounts = [Int](repeating: 0, count: NeuronRenderer.inflight)
    private var markerVersion = 0
    private var slotMarkerVersion = [Int](repeating: -1, count: NeuronRenderer.inflight)
    private var baseMarkers: [NSMarker] = []
    private var traceMarkers: [NSMarker] = []
    private var traceMarkerKey: (step: Int, focus: NeuronID?)?

    // Slow-motion trace (arrows). The arrow buffer is rebuilt only when the edges change; the reveal is on the GPU.
    private var traceOn = false
    private var traceEdges: [TraceEdge] = []
    private var traceFocusEdges: [TraceEdge] = []
    private var traceFocusID: NeuronID?
    private var traceProgress: Float = 0
    private var arrowBuf: MTLBuffer?
    private var arrowCount = 0
    private var arrivals: [TraceArrival] = []
    private var traceTargets: [UInt32] = []                     // units that receive arrows
    private var traceSourcesOnly: [(unit: UInt32, layer: Int)] = []   // members that only send
    private var traceStepSources: [Int: [UInt32]] = [:]         // floor → its outgoing step's source desks
    private var traceFills: [Float] = []
    // Per-unit trace info (float2: kind, value) — dense per ring slot, only touched entries rewritten.
    private var traceInfoBufs: [MTLBuffer] = []
    private var traceTouched = [[UInt32]](repeating: [], count: NeuronRenderer.inflight)
    private var pendingTraceUnits: [UInt32] = []
    private var pendingTraceValues: [SIMD2<Float>] = []
    private var traceVersion = 0
    // Trace-ordered column map: traced desks spread in the middle band of their column (see NeuronLayout.traceOrderedMap).
    private var traceMap: [Int: SIMD3<Float>]?
    private var traceMapUnits: [Int] = []
    private var traceRings: [(unit: UInt32, positive: Bool)] = []
    private var mapMix: Float = 0
    private var mapMixTarget: Float = 0
    private var traceMapActive: Bool { layout == .columns && traceOn && traceMap != nil }
    private var slotTraceVersion = [Int](repeating: -1, count: NeuronRenderer.inflight)

    // Offscreen MSAA targets (resolved into the drawable)
    private var msaaTex: MTLTexture?
    private var depthTex: MTLTexture?
    private var lastFrameTime: CFTimeInterval = 0


    init?(device: MTLDevice, geometry: Geometry) {
        self.device = device
        self.geometry = geometry
        numNeurons = geometry.neuronsPerFrame
        numHeads = geometry.headsPerFrame
        numUnits = numNeurons + numHeads

        guard let q = device.makeCommandQueue() else { return nil }
        queue = q
        let lib: MTLLibrary
        do { lib = try device.makeLibrary(source: Shaders.source, options: nil) } catch {
            NSLog("NeuroScope shader compile failed: \(error)")
            return nil
        }
        func fn(_ n: String) -> MTLFunction? { lib.makeFunction(name: n) }
        do {
            let sd = MTLRenderPipelineDescriptor()
            sd.vertexFunction = fn("spriteVertex"); sd.fragmentFunction = fn("spriteFragment")
            sd.rasterSampleCount = NeuronRenderer.sampleCount
            sd.colorAttachments[0].pixelFormat = NeuronRenderer.colorFormat
            sd.colorAttachments[0].isBlendingEnabled = false          // opaque dots: no additive pile-up
            sd.depthAttachmentPixelFormat = NeuronRenderer.depthFormat
            spritePSO = try device.makeRenderPipelineState(descriptor: sd)

            let md = MTLRenderPipelineDescriptor()
            md.vertexFunction = fn("markerVertex"); md.fragmentFunction = fn("markerFragment")
            md.rasterSampleCount = NeuronRenderer.sampleCount
            md.depthAttachmentPixelFormat = NeuronRenderer.depthFormat
            let ma = md.colorAttachments[0]!
            ma.pixelFormat = NeuronRenderer.colorFormat
            ma.isBlendingEnabled = true
            ma.sourceRGBBlendFactor = .sourceAlpha; ma.destinationRGBBlendFactor = .oneMinusSourceAlpha
            ma.sourceAlphaBlendFactor = .one; ma.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            markerPSO = try device.makeRenderPipelineState(descriptor: md)

            let ad = MTLRenderPipelineDescriptor()
            ad.vertexFunction = fn("arrowVertex"); ad.fragmentFunction = fn("arrowFragment")
            ad.rasterSampleCount = NeuronRenderer.sampleCount
            ad.depthAttachmentPixelFormat = NeuronRenderer.depthFormat
            let aa = ad.colorAttachments[0]!
            aa.pixelFormat = NeuronRenderer.colorFormat
            aa.isBlendingEnabled = true                               // crisp alpha-blended strokes, no additive glow
            aa.sourceRGBBlendFactor = .sourceAlpha; aa.destinationRGBBlendFactor = .oneMinusSourceAlpha
            aa.sourceAlphaBlendFactor = .one; aa.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            arrowPSO = try device.makeRenderPipelineState(descriptor: ad)

            guard let tf = fn("computeTargets"), let gf = fn("updateGlow") else { return nil }
            targetsPSO = try device.makeComputePipelineState(function: tf)
            glowPSO = try device.makeComputePipelineState(function: gf)
        } catch {
            NSLog("NeuroScope pipeline creation failed: \(error)")
            return nil
        }
        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = true
        let mdd = MTLDepthStencilDescriptor()
        mdd.depthCompareFunction = .always    // rings always on top, crisp
        mdd.isDepthWriteEnabled = false
        guard let ds = device.makeDepthStencilState(descriptor: dd),
              let mds = device.makeDepthStencilState(descriptor: mdd) else { return nil }
        dotDepth = ds; markerDepth = mds

        let from = NeuronLayout.positions(.stack, geometry: geometry)
        cpuFrom = from; cpuTo = from
        let posLen = MemoryLayout<SIMD4<Float>>.stride * max(numUnits, 1)
        guard let a = device.makeBuffer(length: posLen, options: .storageModeShared),
              let b = device.makeBuffer(length: posLen, options: .storageModeShared),
              let t = device.makeBuffer(length: 4 * max(numUnits, 1), options: .storageModePrivate),
              let g = device.makeBuffer(length: 4 * max(numUnits, 1), options: .storageModeShared)
        else { return nil }
        posA = a; posB = b; targets = t; glow = g
        memset(g.contents(), 0, g.length)

        let L = max(geometry.numLayers, 1)
        for _ in 0..<NeuronRenderer.inflight {
            guard let u = device.makeBuffer(length: MemoryLayout<NSUniforms>.stride, options: .storageModeShared),
                  let n = device.makeBuffer(length: 2 * max(numNeurons, 1), options: .storageModeShared),
                  let h = device.makeBuffer(length: 4 * max(numHeads, 1), options: .storageModeShared),
                  let d = device.makeBuffer(length: 4 * max(numNeurons, 1), options: .storageModeShared),
                  let ls = device.makeBuffer(length: 4 * L, options: .storageModeShared),
                  let hs = device.makeBuffer(length: 4 * L, options: .storageModeShared),
                  let co = device.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride * L, options: .storageModeShared),
                  let m = device.makeBuffer(length: MemoryLayout<NSMarker>.stride * markerCapacity, options: .storageModeShared),
                  let ti = device.makeBuffer(length: MemoryLayout<SIMD2<Float>>.stride * max(numUnits, 1), options: .storageModeShared)
            else { return nil }
            memset(ti.contents(), 0, ti.length)
            traceInfoBufs.append(ti)
            uniformBufs.append(u); neuronBufs.append(n); headBufs.append(h); diffBufs.append(d)
            layerScaleBufs.append(ls); headScaleBufs.append(hs); markerBufs.append(m); cutoffBufs.append(co)
        }
        pendingNeurons = [Float16](repeating: 0, count: numNeurons)
        pendingHeads = [Float](repeating: 0, count: numHeads)
        pendingDiff = [Float](repeating: 0, count: numNeurons)
        pendingLayerScales = [Float](repeating: 1, count: L)
        pendingHeadScales = [Float](repeating: 1, count: L)
        pendingCutoffs = [SIMD4<Float>](repeating: .zero, count: L)
        super.init()
        writePositions(posA, from, .stack)
        writePositions(posB, from, .stack)
        camera = Camera.framing(for: .stack)
        dataDirty = true
    }

    private func writePositions(_ buf: MTLBuffer, _ p: [SIMD3<Float>], _ mode: LayoutMode) {
        let r = NeuronLayout.dotRadius(mode, geometry: geometry, clusterSpacing: clusterSpacing)
        let dst = buf.contents().bindMemory(to: SIMD4<Float>.self, capacity: numUnits)
        for i in 0..<min(p.count, numUnits) { dst[i] = SIMD4(p[i], r) }
    }

    private static func ease(_ t: Float) -> Float {
        let x = min(max(t, 0), 1)
        return x < 0.5 ? 4 * x * x * x : 1 - pow(-2 * x + 2, 3) / 2
    }

    private func currentMorph(_ now: CFTimeInterval) -> Float {
        guard morphing else { return 1 }
        return NeuronRenderer.ease(Float((now - morphStart) / 0.6))
    }

    func setLayout(_ mode: LayoutMode, animated: Bool) {
        guard mode != layout || morphing else { return }
        retarget(to: mode, animated: animated)
    }

    /// Per-layer neuron (x, y) positions in [-1, 1]² for the .cluster layout (nil = grid fallback).
    /// Rebuilds the cluster targets and, when the layout is .cluster, morphs to them.
    func setClusterLayout(_ xy: [[SIMD2<Float>]]?) {
        clusterXY = xy
        clusterSpacing = NeuronLayout.clusterSpacing(xy, geometry: geometry)
        if layout == .cluster { retarget(to: .cluster, animated: true) }
    }

    /// The layout's positions, with the trace-ordered map laid over .columns while a trace is showing.
    private func targetPositions(_ mode: LayoutMode) -> [SIMD3<Float>] {
        var p = NeuronLayout.positions(mode, geometry: geometry, cluster: clusterXY)
        if mode == .columns, traceOn, let m = traceMap {
            for (u, pos) in m where u < p.count { p[u] = pos }
        }
        return p
    }

    /// Move every unit toward `targetPositions(mode)`, starting from wherever the units are now
    /// (handles a reversal mid-morph).
    private func retarget(to mode: LayoutMode, animated: Bool) {
        let target = targetPositions(mode)
        let now = CACurrentMediaTime()
        if animated {
            let t = currentMorph(now)
            var from = cpuTo
            let toScale = NeuronLayout.dotRadius(layout, geometry: geometry, clusterSpacing: clusterSpacing)
            var fromScale = toScale
            if morphing && t < 1 {
                for i in 0..<from.count { from[i] = simd_mix(cpuFrom[i], cpuTo[i], SIMD3(repeating: t)) }
                fromScale = toScale * t + morphFromScale * (1 - t)
            }
            cpuFrom = from; cpuTo = target
            let dst = posA.contents().bindMemory(to: SIMD4<Float>.self, capacity: numUnits)
            let r = fromScale
            for i in 0..<min(from.count, numUnits) { dst[i] = SIMD4(from[i], r) }
            writePositions(posB, target, mode)
            morphFromScale = fromScale
            morphStart = now; morphing = true
        } else {
            cpuFrom = target; cpuTo = target
            writePositions(posA, target, mode)
            writePositions(posB, target, mode)
            morphing = false
        }
        layout = mode
        settings.layout = mode
        mapMixTarget = traceMapActive ? 1 : 0
        if traceOn { traceMarkerKey = nil; updateTraceFrame() }   // rings follow the map in and out
    }

    /// Display one token's activations. pulse=true: live arrival — units flare and decay with the half-life.
    /// pulse=false: scrubbing — show the values as they are (no trail).
    func show(neurons: UnsafeBufferPointer<Float16>, heads: UnsafeBufferPointer<Float>,
              layerScales: [Float], headScales: [Float], pulse: Bool) {
        pendingNeurons.withUnsafeMutableBufferPointer { dst in
            let n = min(dst.count, neurons.count)
            if n > 0, let s = neurons.baseAddress { dst.baseAddress!.update(from: s, count: n) }
            if n < dst.count { for i in n..<dst.count { dst[i] = 0 } }
        }
        pendingHeads.withUnsafeMutableBufferPointer { dst in
            let n = min(dst.count, heads.count)
            if n > 0, let s = heads.baseAddress { dst.baseAddress!.update(from: s, count: n) }
            if n < dst.count { for i in n..<dst.count { dst[i] = 0 } }
        }
        copyScales(layerScales, into: &pendingLayerScales)
        copyScales(headScales, into: &pendingHeadScales)
        pendingMode = 0
        settings.colorMode = .activation
        updateCutoffs()
        markDirty(pulse: pulse)
    }

    /// Diff mode: signed (edited − baseline) per neuron, normalized by layerScales.
    func showDiff(_ diff: [Float], layerScales: [Float]) {
        pendingDiff.withUnsafeMutableBufferPointer { dst in
            let n = min(dst.count, diff.count)
            diff.withUnsafeBufferPointer { s in if n > 0 { dst.baseAddress!.update(from: s.baseAddress!, count: n) } }
            if n < dst.count { for i in n..<dst.count { dst[i] = 0 } }
        }
        copyScales(layerScales, into: &pendingLayerScales)
        pendingMode = 1
        settings.colorMode = .diff
        markDirty(pulse: false)
    }

    /// Contrast mode: a static map of ContrastResult.neuronDiff (layer-major, A − B mean activation).
    /// Each layer is normalized by a robust max of |diff| (its 99.5th percentile), then drawn with the usual
    /// saturation rule: amber = more active for A, cyan = more active for B. No pulse, no trail.
    func showContrast(_ neuronDiff: [Float]) {
        let per = max(geometry.numNeurons, 1)
        let layers = max(geometry.numLayers, 1)
        var scales = [Float](repeating: 1, count: layers)
        pendingDiff.withUnsafeMutableBufferPointer { dst in
            let n = min(dst.count, neuronDiff.count)
            if n > 0 { neuronDiff.withUnsafeBufferPointer { dst.baseAddress!.update(from: $0.baseAddress!, count: n) } }
            if n < dst.count { for i in n..<dst.count { dst[i] = 0 } }
            for i in 0..<dst.count where !dst[i].isFinite { dst[i] = 0 }
            var mags = [Float](repeating: 0, count: per)
            for l in 0..<layers {
                let lo = l * per
                guard lo < dst.count else { break }
                let hi = min(lo + per, dst.count)
                mags.removeAll(keepingCapacity: true)
                for i in lo..<hi { mags.append(abs(dst[i])) }
                mags.sort()
                let q = mags[min(mags.count - 1, Int(Float(mags.count) * 0.995))]
                scales[l] = max(q, mags.last.map { $0 * 0.05 } ?? 0, 1e-9)
            }
        }
        copyScales(scales, into: &pendingLayerScales)
        pendingMode = 1
        settings.colorMode = .contrast
        markDirty(pulse: false)
    }

    /// Per-floor rank cutoffs of the word in pendingNeurons at the current Intensity (activation mode only).
    private func updateCutoffs() {
        let lit = Salience.litShare(exposure: settings.exposure)
        cutoffExposure = settings.exposure
        pendingCutoffs = pendingNeurons.withUnsafeBufferPointer {
            Salience.cutoffs($0, perLayer: geometry.numNeurons, layers: max(geometry.numLayers, 1), litShare: lit,
                             faintShare: Salience.faintShare * lit / Salience.litShare, scratch: &cutoffScratch)
        }
    }

    private func copyScales(_ src: [Float], into dst: inout [Float]) {
        for i in 0..<dst.count {
            let v = i < src.count ? src[i] : 1
            dst[i] = (v.isFinite && v > 0) ? v : 1
        }
    }

    private func markDirty(pulse: Bool) {
        dataDirty = true
        pendingPulse = pulse
        pendingSnap = !pulse
        let now = CACurrentMediaTime()
        lastDataTime = now
        if pulse { flareStart = now }
    }

    /// Markers drawn on top: edited units (ablated = dark red cross-out, scaled = violet ring, pinned = gold ring),
    /// the selection (bright white ring) and the hover (thin ring).
    func setMarkers(edits: EditSet, selection: Selection?, hover: Selection?) {
        var m: [NSMarker] = []
        m.reserveCapacity(edits.count + 2)
        for e in edits.neurons {
            guard let u = unitIndex(.neuron(e.id)) else { continue }
            let k: UInt32 = e.mode == .ablate ? 0 : (e.mode == .scale ? 1 : 2)
            m.append(NSMarker(unit: UInt32(u), kind: k))
        }
        for e in edits.heads {
            guard let u = unitIndex(.head(e.id)) else { continue }
            let k: UInt32 = e.mode == .ablate ? 0 : (e.mode == .scale ? 1 : 2)
            m.append(NSMarker(unit: UInt32(u), kind: k))
        }
        if let h = hover, h != selection, let u = unitIndex(h) { m.append(NSMarker(unit: UInt32(u), kind: 4)) }
        if let s = selection, let u = unitIndex(s) { m.append(NSMarker(unit: UInt32(u), kind: 3)) }
        baseMarkers = m
        rebuildMarkers()
    }

    /// Base markers (edits, hover, selection) plus the slow-motion ones (source outlines, focus ring).
    private func rebuildMarkers() {
        let m = traceMarkers + baseMarkers
        if m.count > markerCapacity {
            // Grow (rare; never in draw()).
            let cap = max(m.count, markerCapacity * 2)
            var bufs: [MTLBuffer] = []
            for _ in 0..<NeuronRenderer.inflight {
                guard let b = device.makeBuffer(length: MemoryLayout<NSMarker>.stride * cap, options: .storageModeShared) else { return }
                bufs.append(b)
            }
            // Old buffers stay alive via in-flight command buffers' retains.
            markerBufs = bufs; markerCapacity = cap
            slotMarkerVersion = [Int](repeating: -1, count: NeuronRenderer.inflight)
        }
        pendingMarkers = m
        markerVersion &+= 1
    }

    // MARK: slow motion

    /// Slow-motion arrows for one word. `edges` are floor l → l+1 (any order; drawn strongest first within a step),
    /// `focusEdges` every earlier floor → the focus desk (drawn on top). `progress` is floors revealed (0…numLayers,
    /// fractional): an arrow from floor l draws between progress l and l+1, staggered by rank, growing from source to
    /// target with a dot travelling at its tip. Target desks fill with the arriving sum, the current step's source
    /// desks are outlined, the focus desk gets the selection ring, and desks not in the trace dim to ~25%.
    /// Cheap to call every frame: the arrow buffer is rebuilt only when the edges change.
    public func setTrace(_ edges: [TraceEdge], focusEdges: [TraceEdge], progress: Float, focus: NeuronID?) {
        let rebuild = !traceOn || edges != traceEdges || focusEdges != traceFocusEdges || focus != traceFocusID
        traceOn = true
        if rebuild { rebuildTrace(edges, focusEdges, focus: focus) }
        traceProgress = progress.isFinite ? max(progress, 0) : 0
        updateTraceFrame()
    }

    /// Ends the trace: no arrows, no dimming, no slow-motion markers.
    public func clearTrace() {
        guard traceOn || arrowCount > 0 || !traceMarkers.isEmpty else { return }
        let hadMap = traceMap != nil
        traceOn = false
        traceMap = nil; traceMapUnits = []; traceRings = []
        traceEdges = []; traceFocusEdges = []; traceFocusID = nil
        arrowBuf = nil; arrowCount = 0
        arrivals = []; traceTargets = []; traceSourcesOnly = []; traceStepSources = [:]; traceFills = []
        pendingTraceUnits = []; pendingTraceValues = []
        traceVersion &+= 1
        traceMarkerKey = nil
        if !traceMarkers.isEmpty { traceMarkers = []; rebuildMarkers() }
        mapMixTarget = 0
        if hadMap && layout == .columns { retarget(to: .columns, animated: true) }   // desks glide back to their grid slots
    }

    private func validUnit(_ layer: Int, _ index: Int) -> UInt32? {
        let g = geometry
        guard layer >= 0, layer < g.numLayers, index >= 0, index < g.numNeurons else { return nil }
        return UInt32(layer * g.numNeurons + index)
    }

    private func rebuildTrace(_ edges: [TraceEdge], _ focusEdges: [TraceEdge], focus: NeuronID?) {
        traceEdges = edges; traceFocusEdges = focusEdges; traceFocusID = focus
        traceMarkerKey = nil
        let focusUnit = focus.flatMap { validUnit($0.layer, $0.index) }
        var arrows: [NSArrow] = []
        arrows.reserveCapacity(edges.count + focusEdges.count)
        var arr: [TraceArrival] = []
        arr.reserveCapacity(edges.count + focusEdges.count)
        var targetSlot: [UInt32: Int] = [:]
        var targets: [UInt32] = []
        var sources: [UInt32: Int] = [:]          // unit → layer
        var stepSources: [Int: [UInt32]] = [:]
        func slot(_ u: UInt32) -> Int {
            if let s = targetSlot[u] { return s }
            targetSlot[u] = targets.count; targets.append(u)
            return targets.count - 1
        }

        // Main steps: group by source floor, strongest first; width and colour normalized per step.
        var byFloor: [Int: [(TraceEdge, UInt32, UInt32)]] = [:]
        for e in edges {
            guard e.weight.isFinite, e.toLayer > e.fromLayer,
                  let f = validUnit(e.fromLayer, e.fromIndex), let t = validUnit(e.toLayer, e.toIndex) else { continue }
            byFloor[e.fromLayer, default: []].append((e, f, t))
        }
        for l in byFloor.keys.sorted() {
            let group = byFloor[l]!.sorted { abs($0.0.weight) > abs($1.0.weight) }
            let maxW = max(group.map { abs($0.0.weight) }.max() ?? 0, 1e-12)
            var sums: [UInt32: Float] = [:]
            for (e, _, t) in group { sums[t, default: 0] += e.weight }
            let norm = max(sums.values.map { abs($0) }.max() ?? 0, maxW * 0.5, 1e-12)
            let n = group.count
            var seen = Set<UInt32>()
            for (r, (e, f, t)) in group.enumerated() {
                let start = Float(l) + (n > 1 ? 0.55 * Float(r) / Float(n - 1) : 0)
                let end = start + 0.45
                arrows.append(NSArrow(from: f, to: t, weight: e.weight / maxW, width: 0.5 + 1.5 * abs(e.weight) / maxW,
                                      start: start, end: end, alpha: 0.88, focus: 0))
                // The focus desk's fill comes from its own focus arrows only (no double count).
                let ts = (focusUnit != nil && t == focusUnit!) ? -1 : slot(t)
                arr.append(TraceArrival(target: ts, fill: e.weight / norm, start: start, end: end))
                if sources[f] == nil { sources[f] = e.fromLayer }
                if seen.insert(f).inserted { stepSources[l, default: []].append(f) }
            }
        }

        // Focus arrows: drawn last (on top), revealed by their source floor, normalized over the whole set.
        if let fu = focusUnit {
            var fe: [(TraceEdge, UInt32)] = []
            for e in focusEdges {
                guard e.weight.isFinite, let f = validUnit(e.fromLayer, e.fromIndex),
                      let t = validUnit(e.toLayer, e.toIndex), t == fu else { continue }
                fe.append((e, f))
            }
            fe.sort { $0.0.fromLayer != $1.0.fromLayer ? $0.0.fromLayer < $1.0.fromLayer : abs($0.0.weight) > abs($1.0.weight) }
            let maxW = max(fe.map { abs($0.0.weight) }.max() ?? 0, 1e-12)
            let total = fe.reduce(Float(0)) { $0 + $1.0.weight }
            let norm = max(abs(total), maxW, 1e-12)
            var perFloor: [Int: Int] = [:]
            for (e, _) in fe { perFloor[e.fromLayer, default: 0] += 1 }
            var rank: [Int: Int] = [:]
            let fs = fe.isEmpty ? -1 : slot(fu)
            for (e, f) in fe {
                let r = rank[e.fromLayer, default: 0]; rank[e.fromLayer] = r + 1
                let n = perFloor[e.fromLayer] ?? 1
                let start = Float(e.fromLayer) + (n > 1 ? 0.55 * Float(r) / Float(n - 1) : 0)
                let end = start + 0.45
                arrows.append(NSArrow(from: f, to: fu, weight: e.weight / maxW, width: 0.75 + 1.5 * abs(e.weight) / maxW,
                                      start: start, end: end, alpha: 1, focus: 1))
                arr.append(TraceArrival(target: fs, fill: e.weight / norm, start: start, end: end))
                if sources[f] == nil { sources[f] = e.fromLayer }
            }
        }

        arrivals = arr
        traceTargets = targets
        traceFills = [Float](repeating: 0, count: targets.count)
        traceSourcesOnly = sources.filter { targetSlot[$0.key] == nil }.map { (unit: $0.key, layer: $0.value) }
        traceStepSources = stepSources
        arrowCount = arrows.count
        if arrows.isEmpty {
            arrowBuf = nil
        } else {
            // A fresh buffer: frames still in flight keep the old one alive through their command buffers.
            arrowBuf = arrows.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
            if arrowBuf == nil { arrowCount = 0 }
        }

        // Trace-ordered map + ring colours (amber = net push, blue = net hold back), then glide into it.
        var netW: [UInt32: Float] = [:]
        var inW: [UInt32: Float] = [:]
        for e in edges + focusEdges {
            guard e.weight.isFinite, let f = validUnit(e.fromLayer, e.fromIndex), let t = validUnit(e.toLayer, e.toIndex) else { continue }
            netW[f, default: 0] += 0; inW[t, default: 0] += e.weight
            netW[t, default: 0] += 0
        }
        var outW: [UInt32: Float] = [:]
        for e in edges + focusEdges {
            guard e.weight.isFinite, let f = validUnit(e.fromLayer, e.fromIndex) else { continue }
            outW[f, default: 0] += e.weight
        }
        traceRings = netW.keys.sorted().map { u in (unit: u, positive: (inW[u] ?? outW[u] ?? 0) >= 0) }
        let map = NeuronLayout.traceOrderedMap(geometry: geometry, edges: edges, focusEdges: focusEdges)
        traceMap = map.isEmpty ? nil : map
        traceMapUnits = map.keys.sorted()
        mapMixTarget = traceMapActive ? 1 : 0
        if layout == .columns { retarget(to: .columns, animated: true) }
    }

    /// Per frame: target fills from the arrived part of each arrow, source reveal, outlines for the current step.
    private func updateTraceFrame() {
        let p = traceProgress
        for i in 0..<traceFills.count { traceFills[i] = 0 }
        for a in arrivals where a.target >= 0 {
            let f = min(max((p - a.start) / max(a.end - a.start, 1e-4), 0), 1)
            if f > 0 { traceFills[a.target] += a.fill * f }
        }
        var units = traceTargets
        var values = traceFills.map { SIMD2<Float>(1, min(max($0, -1), 1)) }
        units.reserveCapacity(units.count + traceSourcesOnly.count)
        values.reserveCapacity(units.count + traceSourcesOnly.count)
        for s in traceSourcesOnly {
            units.append(s.unit)
            values.append(SIMD2(2, min(max((p - Float(s.layer)) * 4 + 1, 0), 1)))
        }
        pendingTraceUnits = units
        pendingTraceValues = values
        traceVersion &+= 1

        let step = Int(p.rounded(.down))
        if traceMarkerKey == nil || traceMarkerKey!.step != step || traceMarkerKey!.focus != traceFocusID {
            traceMarkerKey = (step, traceFocusID)
            var m: [NSMarker] = []
            if step < geometry.numLayers, let src = traceStepSources[step] {
                for u in src { m.append(NSMarker(unit: u, kind: 5)) }
            }
            if traceMapActive { for r in traceRings { m.append(NSMarker(unit: r.unit, kind: r.positive ? 6 : 7)) } }
            if let f = traceFocusID, let u = validUnit(f.layer, f.index) { m.append(NSMarker(unit: u, kind: 3)) }
            traceMarkers = m
            rebuildMarkers()
        }
    }

    private func unitIndex(_ s: Selection) -> Int? {
        let g = geometry
        switch s {
        case .neuron(let n):
            guard n.layer >= 0, n.layer < g.numLayers, n.index >= 0, n.index < g.numNeurons else { return nil }
            return n.flat(g)
        case .head(let h):
            guard h.layer >= 0, h.layer < g.numLayers, h.head >= 0, h.head < g.numHeads else { return nil }
            return numNeurons + h.flat(g)
        }
    }

    private func selection(forUnit i: Int) -> Selection {
        let g = geometry
        if i < numNeurons { return .neuron(NeuronID(layer: i / g.numNeurons, index: i % g.numNeurons)) }
        let j = i - numNeurons
        return .head(HeadID(layer: j / g.numHeads, head: j % g.numHeads))
    }

    private func worldPosition(_ i: Int, morph t: Float) -> SIMD3<Float> {
        t >= 1 ? cpuTo[i] : simd_mix(cpuFrom[i], cpuTo[i], SIMD3(repeating: t))
    }

    /// Nearest unit to a view point (points, origin top-left) within ~8 pt, nearest-to-camera wins ties.
    func pick(at point: CGPoint, viewSize: CGSize) -> Selection? {
        let w = Float(viewSize.width), h = Float(viewSize.height)
        guard w > 0, h > 0, numUnits > 0 else { return nil }
        let vp = camera.viewProjection(aspect: w / h)
        let t = currentMorph(CACurrentMediaTime())
        let px = Float(point.x), py = Float(point.y)
        let radius: Float = 8
        var best = -1
        var bestD2 = radius * radius
        var bestDepth: Float = .infinity
        let limit = settings.showHeads ? numUnits : numNeurons
        // With the trace map showing, the big traced nodes win over the faint dust behind them.
        if traceMapActive {
            var tb = -1
            var tD2 = (radius + 3) * (radius + 3)
            for i in traceMapUnits where i < numNeurons {
                let c = vp * SIMD4(worldPosition(i, morph: t), 1)
                if c.w <= 0.02 { continue }
                let inv = 1 / c.w
                let dx = (c.x * inv + 1) * 0.5 * w - px, dy = (1 - c.y * inv) * 0.5 * h - py
                let d2 = dx * dx + dy * dy
                if d2 < tD2 { tb = i; tD2 = d2 }
            }
            if tb >= 0 { return selection(forUnit: tb) }
        }
        for i in 0..<limit {
            let c = vp * SIMD4(worldPosition(i, morph: t), 1)
            if c.w <= 0.02 { continue }
            let inv = 1 / c.w
            let sx = (c.x * inv + 1) * 0.5 * w
            let sy = (1 - c.y * inv) * 0.5 * h
            let dx = sx - px, dy = sy - py
            let d2 = dx * dx + dy * dy
            if d2 > radius * radius { continue }
            let depth = c.z * inv
            // Within ~1 pt of each other counts as a tie → nearest to camera wins.
            if best < 0 || d2 + 1 < bestD2 || (abs(d2 - bestD2) <= 1 && depth < bestDepth) {
                best = i; bestD2 = d2; bestDepth = depth
            }
        }
        return best >= 0 ? selection(forUnit: best) : nil
    }

    /// Screen position of a unit (for tooltips). nil when off-screen.
    func screenPosition(of s: Selection, viewSize: CGSize) -> CGPoint? {
        guard let i = unitIndex(s) else { return nil }
        let p = worldPosition(i, morph: currentMorph(CACurrentMediaTime()))
        guard let r = camera.project(p, viewSize: viewSize) else { return nil }
        let pt = r.point
        guard pt.x >= 0, pt.y >= 0, pt.x <= viewSize.width, pt.y <= viewSize.height, r.depth >= 0, r.depth <= 1 else { return nil }
        return pt
    }

    /// True while anything is still animating (glow decaying, layout morphing) — the view pauses otherwise.
    var isAnimating: Bool {
        let now = CACurrentMediaTime()
        if dataDirty { return true }
        if morphing && now - morphStart < 0.7 { return true }
        if mapMix != mapMixTarget { return true }
        if !hlSettled { return true }
        // Glow converges geometrically; ~10 half-lives ≈ 1/1000 of the initial gap.
        let settle = Double(max(settings.glowHalfLife, 0.01)) * 10 + 0.1
        return now - lastDataTime < settle
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        makeTargets(width: Int(size.width), height: Int(size.height))
    }

    private func makeTargets(width: Int, height: Int) {
        guard width > 0, height > 0 else { return }
        func tex(_ f: MTLPixelFormat) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: width, height: height, mipmapped: false)
            d.textureType = .type2DMultisample
            d.sampleCount = NeuronRenderer.sampleCount
            d.usage = [.renderTarget]
            d.storageMode = .memoryless
            return device.makeTexture(descriptor: d)
        }
        msaaTex = tex(NeuronRenderer.colorFormat)
        depthTex = tex(NeuronRenderer.depthFormat)
    }

    func draw(in view: MTKView) {
        let size = view.drawableSize
        guard size.width > 0, size.height > 0 else { return }
        if msaaTex == nil || msaaTex!.width != Int(size.width) || msaaTex!.height != Int(size.height) {
            makeTargets(width: Int(size.width), height: Int(size.height))
        }
        guard let msaa = msaaTex, let depth = depthTex else { return }

        semaphore.wait()
        guard let cb = queue.makeCommandBuffer() else { semaphore.signal(); return }
        let sem = semaphore
        cb.addCompletedHandler { _ in sem.signal() }

        let now = CACurrentMediaTime()
        let dt = lastFrameTime == 0 ? 1.0 / 120.0 : min(now - lastFrameTime, 0.25)
        lastFrameTime = now
        if let drawable = view.currentDrawable {
            encodeFrame(cb, msaa: msaa, depth: depth, resolve: drawable.texture, size: size, now: now, dt: dt,
                        scale: Float(view.window?.backingScaleFactor ?? 2))
            cb.present(drawable)
        }
        cb.commit()
    }

    /// One frame: compute (normalize + saturation persistence) then the dot + marker pass, resolved into `resolve`.
    private func encodeFrame(_ cb: MTLCommandBuffer, msaa: MTLTexture, depth: MTLTexture, resolve: MTLTexture,
                             size: CGSize, now: CFTimeInterval, dt: Double, scale: Float) {
        frameSlot = (frameSlot + 1) % NeuronRenderer.inflight
        let slot = frameSlot
        let morph = currentMorph(now)
        if morphing && now - morphStart >= 0.6 {
            // Morph finished: collapse to the target buffer.
            morphing = false
            memcpy(posA.contents(), posB.contents(), posA.length)
            cpuFrom = cpuTo
        }

        // ---- Uniforms
        let aspect = Float(size.width / size.height)
        // Depth fade across the model's extent around the orbit target (back planes slightly dimmer).
        let reach: Float
        switch layout {
        case .sheet, .columns: reach = 1.0
        case .stack, .cluster: reach = 2.6
        }
        var u = NSUniforms(
            view: camera.viewMatrix(), proj: camera.projectionMatrix(aspect: aspect),
            viewport: SIMD2(Float(size.width), Float(size.height)),
            morph: morphing ? morph : 1,
            exposure: settings.exposure, threshold: settings.threshold,
            flare: Float(max(0, 1 - (now - flareStart) / 0.25)),
            pointSize: max(settings.pointSize, 0.05),
            colorMode: settings.colorMode == .diff ? 1 : (settings.colorMode == .contrast ? 2 : 0),
            numNeurons: UInt32(numNeurons), numUnits: UInt32(numUnits),
            minPx: 0.75 * scale,
            fadeNear: max(camera.distance - reach, 0.01), fadeFar: camera.distance + reach)
        if !morphing { u.morph = 1 }
        let drawTrace = traceOn && (arrowCount > 0 || !pendingTraceUnits.isEmpty)
        u.traceOn = drawTrace ? 1 : 0
        let mixRate = Float(dt / 0.6)
        if mapMix < mapMixTarget { mapMix = min(mapMixTarget, mapMix + mixRate) }
        else if mapMix > mapMixTarget { mapMix = max(mapMixTarget, mapMix - mixRate) }
        u.traceMap = mapMix
        // Lesson highlight: ease each floor's weight toward its target; engaged until everything is back to full.
        let hlStep = Float(dt / 0.4)
        for f in 0..<64 {
            let t = hlTarget(f)
            if hlWeights[f] < t { hlWeights[f] = min(t, hlWeights[f] + hlStep) }
            else if hlWeights[f] > t { hlWeights[f] = max(t, hlWeights[f] - hlStep) }
        }
        hlEngaged = settings.highlightFloors != nil || !hlSettled
        u.hlOn = hlEngaged ? 1 : 0
        u.neuronsPerLayer = UInt32(max(geometry.numNeurons, 1))
        u.headsPerLayer = UInt32(max(geometry.numHeads, 1))
        let w = hlWeights
        func v4(_ i: Int) -> SIMD4<Float> { SIMD4(w[i * 4], w[i * 4 + 1], w[i * 4 + 2], w[i * 4 + 3]) }
        u.hl = (v4(0), v4(1), v4(2), v4(3), v4(4), v4(5), v4(6), v4(7),
                v4(8), v4(9), v4(10), v4(11), v4(12), v4(13), v4(14), v4(15))
        memcpy(uniformBufs[slot].contents(), &u, MemoryLayout<NSUniforms>.stride)

        // ---- Markers
        if slotMarkerVersion[slot] != markerVersion {
            let n = min(pendingMarkers.count, markerCapacity)
            pendingMarkers.withUnsafeBytes { raw in
                if n > 0 { memcpy(markerBufs[slot].contents(), raw.baseAddress!, n * MemoryLayout<NSMarker>.stride) }
            }
            markerCounts[slot] = n
            slotMarkerVersion[slot] = markerVersion
        }

        // ---- Trace info (sparse rewrite of this slot's dense per-unit buffer)
        if slotTraceVersion[slot] != traceVersion {
            let dst = traceInfoBufs[slot].contents().bindMemory(to: SIMD2<Float>.self, capacity: max(numUnits, 1))
            for i in traceTouched[slot] where Int(i) < numUnits { dst[Int(i)] = .zero }
            for (k, unit) in pendingTraceUnits.enumerated() where Int(unit) < numUnits { dst[Int(unit)] = pendingTraceValues[k] }
            traceTouched[slot] = pendingTraceUnits
            slotTraceVersion[slot] = traceVersion
        }

        // ---- Compute
        // Intensity sets how many desks light per floor: a change re-ranks the word on display (no trail).
        if pendingMode == 0, settings.exposure != cutoffExposure {
            updateCutoffs()
            if !dataDirty { dataDirty = true; pendingSnap = true; pendingPulse = false }
        }
        if let ce = cb.makeComputeCommandEncoder() {
            let total = numUnits
            if dataDirty {
                _ = pendingNeurons.withUnsafeBytes { memcpy(neuronBufs[slot].contents(), $0.baseAddress!, $0.count) }
                _ = pendingHeads.withUnsafeBytes { memcpy(headBufs[slot].contents(), $0.baseAddress!, $0.count) }
                if pendingMode == 1 {
                    _ = pendingDiff.withUnsafeBytes { memcpy(diffBufs[slot].contents(), $0.baseAddress!, $0.count) }
                }
                _ = pendingLayerScales.withUnsafeBytes { memcpy(layerScaleBufs[slot].contents(), $0.baseAddress!, $0.count) }
                _ = pendingHeadScales.withUnsafeBytes { memcpy(headScaleBufs[slot].contents(), $0.baseAddress!, $0.count) }
                _ = pendingCutoffs.withUnsafeBytes {
                    memcpy(cutoffBufs[slot].contents(), $0.baseAddress!, min($0.count, cutoffBufs[slot].length))
                }
                var tp = NSTargetParams(numNeurons: UInt32(numNeurons), numHeads: UInt32(numHeads),
                                        neuronsPerLayer: UInt32(geometry.numNeurons), headsPerLayer: UInt32(geometry.numHeads),
                                        numLayers: UInt32(max(geometry.numLayers, 1)), mode: pendingMode)
                ce.setComputePipelineState(targetsPSO)
                ce.setBuffer(neuronBufs[slot], offset: 0, index: 0)
                ce.setBuffer(headBufs[slot], offset: 0, index: 1)
                ce.setBuffer(diffBufs[slot], offset: 0, index: 2)
                ce.setBuffer(layerScaleBufs[slot], offset: 0, index: 3)
                ce.setBuffer(headScaleBufs[slot], offset: 0, index: 4)
                ce.setBuffer(targets, offset: 0, index: 5)
                ce.setBytes(&tp, length: MemoryLayout<NSTargetParams>.stride, index: 6)
                ce.setBuffer(cutoffBufs[slot], offset: 0, index: 7)
                dispatch(ce, targetsPSO, total)
            }
            let hl = Double(max(settings.glowHalfLife, 0.005))
            var gp = NSGlowParams(numUnits: UInt32(total), decay: Float(pow(0.5, dt / hl)),
                                  snap: (dataDirty && pendingSnap) ? 1 : 0,
                                  pulse: (dataDirty && pendingPulse) ? 1 : 0)
            dataDirty = false
            ce.setComputePipelineState(glowPSO)
            ce.setBuffer(targets, offset: 0, index: 0)
            ce.setBuffer(glow, offset: 0, index: 1)
            ce.setBytes(&gp, length: MemoryLayout<NSGlowParams>.stride, index: 2)
            dispatch(ce, glowPSO, total)
            ce.endEncoding()
        }

        // ---- Dots (opaque, depth-tested) + markers, MSAA resolved into the target
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = msaa
        rp.colorAttachments[0].resolveTexture = resolve
        rp.colorAttachments[0].loadAction = .clear
        rp.colorAttachments[0].storeAction = .multisampleResolve
        rp.colorAttachments[0].clearColor = MTLClearColor(red: 0.030, green: 0.036, blue: 0.058, alpha: 1)
        rp.depthAttachment.texture = depth
        rp.depthAttachment.loadAction = .clear
        rp.depthAttachment.storeAction = .dontCare
        rp.depthAttachment.clearDepth = 1
        if let re = cb.makeRenderCommandEncoder(descriptor: rp) {
            re.setRenderPipelineState(spritePSO)
            re.setDepthStencilState(dotDepth)
            re.setVertexBuffer(posA, offset: 0, index: 0)
            re.setVertexBuffer(posB, offset: 0, index: 1)
            re.setVertexBuffer(glow, offset: 0, index: 2)
            re.setVertexBuffer(uniformBufs[slot], offset: 0, index: 3)
            re.setVertexBuffer(traceInfoBufs[slot], offset: 0, index: 4)
            let count = settings.showHeads ? numUnits : numNeurons
            if count > 0 { re.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: count) }
            // Arrows over the dots (focus arrows are last in the buffer, so they land on top), rings over both.
            if drawTrace, arrowCount > 0, let ab = arrowBuf {
                var ap = NSArrowParams(progress: traceProgress, scale: scale)
                re.setRenderPipelineState(arrowPSO)
                re.setDepthStencilState(markerDepth)
                re.setVertexBuffer(ab, offset: 0, index: 2)
                re.setVertexBytes(&ap, length: MemoryLayout<NSArrowParams>.stride, index: 4)
                re.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 15, instanceCount: arrowCount)
            }
            let mc = markerCounts[slot]
            if mc > 0 {
                re.setRenderPipelineState(markerPSO)
                re.setDepthStencilState(markerDepth)
                re.setVertexBuffer(markerBufs[slot], offset: 0, index: 2)
                re.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: mc)
            }
            re.endEncoding()
        }
    }

#if DEBUG
    /// Debug-only offscreen snapshot: renders `frames` frames (fixed dt) of the current state into a
    /// width×height PNG at `url`. Never called by the app; used from scratch verification scripts.
    func debugSnapshot(to url: URL, width: Int, height: Int, frames: Int = 1, dt: Double = 1.0 / 60.0) -> Bool {
        func tex(_ f: MTLPixelFormat, ms: Bool) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: width, height: height, mipmapped: false)
            if ms { d.textureType = .type2DMultisample; d.sampleCount = NeuronRenderer.sampleCount }
            d.usage = [.renderTarget, .shaderRead]
            d.storageMode = ms ? .private : .shared
            return device.makeTexture(descriptor: d)
        }
        guard let msaa = tex(NeuronRenderer.colorFormat, ms: true), let depth = tex(NeuronRenderer.depthFormat, ms: true),
              let out = tex(NeuronRenderer.colorFormat, ms: false) else { return false }
        var now = CACurrentMediaTime()
        for _ in 0..<max(frames, 1) {
            guard let cb = queue.makeCommandBuffer() else { return false }
            encodeFrame(cb, msaa: msaa, depth: depth, resolve: out, size: CGSize(width: width, height: height),
                        now: now, dt: dt, scale: 2)
            cb.commit(); cb.waitUntilCompleted()
            now += dt
        }
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        out.getBytes(&bytes, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let ctx = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: info.rawValue),
              let img = ctx.makeImage(),
              let dst = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else { return false }
        CGImageDestinationAddImage(dst, img, nil)
        return CGImageDestinationFinalize(dst)
    }
#endif

    private func dispatch(_ ce: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, _ n: Int) {
        guard n > 0 else { return }
        let w = min(pso.maxTotalThreadsPerThreadgroup, 256)
        ce.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }
}
