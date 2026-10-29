// World-space positions for every drawable unit: all neurons (layer-major), then all heads (layer-major).
// Units are arranged so the whole model fits roughly in a [-1, 1]^3 box (the camera frames that box).

import simd

enum NeuronLayout {
    /// Neuron grid per layer: columns × rows with columns*rows >= numNeurons (105 × 79 for 8192, 64 × 48 for 3072).
    static func grid(for numNeurons: Int) -> (cols: Int, rows: Int) {
        let n = max(1, numNeurons)
        // 4:3 aspect grid: cols ≈ sqrt(n·4/3).
        let cols = max(1, Int((Double(n) * 4.0 / 3.0).squareRoot().rounded()))
        let rows = (n + cols - 1) / cols
        return (cols, rows)
    }

    // Local plane coordinates: u ∈ [-1, 1], v ∈ [-0.75, 0.75]; heads sit at v = headV.
    private static let headV: Float = 0.86
    private static let sheetCols = 7
    private static let sheetScale: Float = 0.19

    /// Half the stack's depth: generous spacing (0.32 per layer for 16 layers) so layers read as separate planes.
    private static let stackHalfDepth: Float = 2.4

    /// Plane depth for layer l in stack mode: layer 0 at +stackHalfDepth (nearest the default eye), last at −.
    private static func depth(_ l: Int, _ g: Geometry) -> Float {
        guard g.numLayers > 1 else { return 0 }
        return stackHalfDepth - 2 * stackHalfDepth * Float(l) / Float(g.numLayers - 1)
    }

    private static func sheetCenter(_ l: Int, _ g: Geometry) -> SIMD3<Float> {
        let cols = sheetCols
        let rows = max(1, (g.numLayers + cols - 1) / cols)
        let cw: Float = 3.2 / Float(cols), ch: Float = 2.0 / Float(rows)
        let c = l % cols, r = l / cols
        return SIMD3(-1.6 + (Float(c) + 0.5) * cw, 1.0 - (Float(r) + 0.5) * ch - 0.02, 0)
    }

    // ---- .columns: floors as thin vertical columns left → right on z = 0 (the textbook network diagram).
    /// The diagram is WIDE, not tall: x ∈ [-columnsHalfWidth, +columnsHalfWidth],
    /// y ∈ [-columnsHalfHeight, +columnsHalfHeight]; the camera fits this box to the view's own shape.
    static let columnsHalfWidth: Float = 2.6
    static let columnsHalfHeight: Float = 0.7
    /// Desks per row inside one column: chosen so a column fills ~63% of its floor's slot whatever the shape
    /// (16 × 192 for 28 floors of 3072, 35 × 235 for 16 floors of 8192), with square pitch.
    private static func columnWide(_ g: Geometry) -> Int {
        let slot = 2 * columnsHalfWidth / Float(max(g.numLayers, 1))
        let w = (0.63 * slot * Float(max(g.numNeurons, 1)) / (2 * columnsHalfHeight)).squareRoot()
        return max(1, min(max(g.numNeurons, 1), Int(w.rounded())))
    }
    /// Where the tucked head row sits under each column.
    private static let columnHeadY: Float = -0.76

    /// Centre x of floor l's column.
    static func columnX(_ l: Int, _ g: Geometry) -> Float {
        let n = max(g.numLayers, 1)
        return -columnsHalfWidth + (Float(l) + 0.5) * 2 * columnsHalfWidth / Float(n)
    }

    /// Rows per column and the square pitch between neighbouring desks.
    private static func columnGrid(_ g: Geometry) -> (wide: Int, rows: Int, pitch: Float) {
        let wide = columnWide(g)
        let rows = max(1, (g.numNeurons + wide - 1) / wide)
        return (wide, rows, 2 * columnsHalfHeight / Float(rows))
    }

    /// World position of desk i on floor l in .columns.
    private static func columnDesk(_ i: Int, layer l: Int, _ g: Geometry) -> SIMD3<Float> {
        let (wide, _, pitch) = columnGrid(g)
        let c = i % wide, r = i / wide
        return SIMD3(columnX(l, g) + (Float(c) + 0.5 - Float(wide) / 2) * pitch, columnsHalfHeight - (Float(r) + 0.5) * pitch, 0)
    }

    /// Map local plane coords of layer l to world space.
    private static func place(_ u: Float, _ v: Float, layer l: Int, mode: LayoutMode, g: Geometry) -> SIMD3<Float> {
        switch mode {
        case .stack, .cluster: return SIMD3(u, v, depth(l, g))
        case .sheet: return sheetCenter(l, g) + SIMD3(u, v, 0) * sheetScale
        case .columns:
            let (wide, _, pitch) = columnGrid(g)
            return SIMD3(columnX(l, g) + u * Float(wide) * pitch * 0.5, v / 0.75 * columnsHalfHeight, 0)
        }
    }

    /// World radius of a desk dot (stack, sheet, columns): 0.3 of the grid pitch (a diameter of 0.6 pitch leaves gaps through which the
    /// planes behind stay visible; 0.25 read as specks at 8,192 desks a floor): 2/cols × 0.3 on a plane (0.0057 for
    /// 105 columns), scaled with the tile in .sheet, and a quarter of the square column pitch in .columns.
    /// .cluster: 0.3 of the cluster's own desk spacing (`clusterSpacing`), never the grid's. The islands pack desks
    /// much tighter than the grid (8,192 desks on a sunflower spiral inside ~24 islands, the gaps between them empty),
    /// so the grid radius made each island's dots overlap into one solid grey disc.
    static func dotRadius(_ mode: LayoutMode, geometry g: Geometry, clusterSpacing: Float? = nil) -> Float {
        let plane = 0.6 / Float(grid(for: g.numNeurons).cols)
        switch mode {
        case .stack: return plane
        case .cluster:
            guard let s = clusterSpacing, s.isFinite, s > 0 else { return plane }
            return min(plane, 0.3 * s)
        case .sheet: return plane * sheetScale
        case .columns: return columnGrid(g).pitch * 0.255
        }
    }

    /// Typical nearest-neighbour distance between desks in the .cluster layout, in world units (the same mapping as
    /// `positions`: x → u, y → v·0.75). The median over a fixed sample of desks per floor (brute force against the
    /// whole floor: 48 × 8,192 × 16 ≈ 6M distance checks, a few ms once per cluster table).
    static func clusterSpacing(_ cluster: [[SIMD2<Float>]]?, geometry g: Geometry) -> Float? {
        guard let cl = cluster, !cl.isEmpty else { return nil }
        func world(_ p: SIMD2<Float>) -> SIMD2<Float>? {
            guard p.x.isFinite, p.y.isFinite else { return nil }
            return SIMD2(min(max(p.x, -1), 1), min(max(p.y, -1), 1) * 0.75)
        }
        var nn = [Float]()
        for layer in cl.prefix(g.numLayers) {
            let pts = layer.prefix(g.numNeurons).compactMap(world)
            guard pts.count > 1 else { continue }
            let samples = min(48, pts.count)
            for k in 0..<samples {
                let a = pts[k * pts.count / samples]
                var best = Float.infinity
                for b in pts {
                    let d = simd_length_squared(b - a)
                    if d > 0 && d < best { best = d }
                }
                if best.isFinite { nn.append(best.squareRoot()) }
            }
        }
        guard !nn.isEmpty else { return nil }
        nn.sort()
        return nn[nn.count / 2]
    }

    /// Count = neuronsPerFrame + headsPerFrame.
    /// .stack: layer l is a plane at z = depth(l) (layer 0 nearest the camera's default view), neurons on its
    ///   grid, that layer's heads as a row of larger nodes along the plane's top edge.
    /// .sheet: the layer grids tiled flat (z = 0) in reading order, 7 tiles per row.
    /// .columns: each floor a thin column (columnGrid), floors left → right across x ∈ ±columnsHalfWidth, z = 0;
    ///   heads tucked as a tiny row under their column.
    static func positions(_ mode: LayoutMode, geometry: Geometry) -> [SIMD3<Float>] {
        positions(mode, geometry: geometry, cluster: nil)
    }

    /// .cluster: the same stacked planes as .stack, but neuron i of layer l sits at cluster[l][i] (x, y in [-1, 1]²,
    /// scaled to the plane: x → u, y → v·0.75). A nil or short cluster table falls back to the grid position.
    static func positions(_ mode: LayoutMode, geometry: Geometry, cluster: [[SIMD2<Float>]]?) -> [SIMD3<Float>] {
        let g = geometry
        let (cols, rows) = grid(for: g.numNeurons)
        var out = [SIMD3<Float>]()
        out.reserveCapacity(g.neuronsPerFrame + g.headsPerFrame)
        for l in 0..<g.numLayers {
            if mode == .columns {
                for i in 0..<g.numNeurons { out.append(columnDesk(i, layer: l, g)) }
                continue
            }
            for i in 0..<g.numNeurons {
                let c = i % cols, r = i / cols
                let u = -1 + 2 * (Float(c) + 0.5) / Float(cols)
                let v = 0.75 - 1.5 * (Float(r) + 0.5) / Float(rows)
                if mode == .cluster, let cl = cluster, l < cl.count, i < cl[l].count {
                    let xy = cl[l][i]
                    let cx = xy.x.isFinite ? min(max(xy.x, -1), 1) : u
                    let cy = xy.y.isFinite ? min(max(xy.y, -1), 1) : v / 0.75
                    out.append(place(cx, cy * 0.75, layer: l, mode: mode, g: g))
                    continue
                }
                out.append(place(u, v, layer: l, mode: mode, g: g))
            }
        }
        for l in 0..<g.numLayers {
            if mode == .columns {
                let (wide, _, pitch) = columnGrid(g)
                let w = Float(wide) * pitch
                for h in 0..<g.numHeads {
                    let x = columnX(l, g) - w / 2 + w * (Float(h) + 0.5) / Float(max(1, g.numHeads))
                    out.append(SIMD3(x, columnHeadY, 0))
                }
                continue
            }
            for h in 0..<g.numHeads {
                let u = -0.9 + 1.8 * (Float(h) + 0.5) / Float(max(1, g.numHeads))
                out.append(place(u, headV, layer: l, mode: mode, g: g))
            }
        }
        return out
    }

    /// Slow-motion map for .columns: the desks that appear in a trace (edge sources and targets, the focus) are pulled
    /// out of their grid slots and spread evenly in the vertical middle band of their floor's column, so the arrows
    /// read like a textbook network diagram. Returns flat unit index → world position (traced desks only; every other
    /// desk keeps its grid slot and is drawn as faint dust by the renderer).
    /// Order: the first floor is sorted by the sign and size of its desks' arrow weight (pushes on top, holds back at
    /// the bottom); the other floors are then settled by barycentric sweeps (each desk goes to the weighted mean height
    /// of the desks it is wired to), which keeps arrows from crossing needlessly. The first floor is never reshuffled.
    static func traceOrderedMap(geometry g: Geometry, edges: [TraceEdge], focusEdges: [TraceEdge]) -> [Int: SIMD3<Float>] {
        let n = g.numNeurons
        func flat(_ l: Int, _ i: Int) -> Int? {
            (l >= 0 && l < g.numLayers && i >= 0 && i < n) ? l * n + i : nil
        }
        var incoming: [Int: [(Int, Float)]] = [:]
        var outgoing: [Int: [(Int, Float)]] = [:]
        var net: [Int: Float] = [:]
        var members: [Int: Set<Int>] = [:]            // floor → desk indices
        for e in edges + focusEdges {
            guard e.weight.isFinite, let f = flat(e.fromLayer, e.fromIndex), let t = flat(e.toLayer, e.toIndex) else { continue }
            outgoing[f, default: []].append((t, e.weight))
            incoming[t, default: []].append((f, e.weight))
            net[f, default: 0] += e.weight
            net[t, default: 0] += e.weight
            members[e.fromLayer, default: []].insert(e.fromIndex)
            members[e.toLayer, default: []].insert(e.toIndex)
        }
        let floors = members.keys.sorted()
        guard let first = floors.first else { return [:] }

        var order: [Int: [Int]] = [:]                 // floor → desk indices, top to bottom
        var y: [Int: Float] = [:]
        func assign(_ l: Int) {
            let o = order[l] ?? []
            let spacing: Float = o.count > 1 ? min(0.08, 2 * columnsHalfHeight * 0.85 / Float(o.count - 1)) : 0
            let top = Float(max(o.count - 1, 0)) * spacing / 2
            for (k, i) in o.enumerated() { y[l * n + i] = top - Float(k) * spacing }
        }
        for l in floors {
            order[l] = members[l]!.sorted { a, b in
                let na = net[l * n + a] ?? 0, nb = net[l * n + b] ?? 0
                return na != nb ? na > nb : a < b
            }
            assign(l)
        }
        func sweep(_ l: Int, _ nb: [Int: [(Int, Float)]]) {
            let cur = order[l] ?? []
            var keyed: [(i: Int, key: Float, rank: Int)] = []
            for (rank, i) in cur.enumerated() {
                let u = l * n + i
                var wsum: Float = 0, ysum: Float = 0
                for (v, w) in nb[u] ?? [] {
                    if let yv = y[v] { wsum += abs(w); ysum += abs(w) * yv }
                }
                keyed.append((i, wsum > 0 ? ysum / wsum : (y[u] ?? 0), rank))
            }
            keyed.sort { $0.key != $1.key ? $0.key > $1.key : $0.rank < $1.rank }
            order[l] = keyed.map { $0.i }
            assign(l)
        }
        for _ in 0..<2 {
            for l in floors.reversed() where l != first { sweep(l, outgoing) }
            for l in floors where l != first { sweep(l, incoming) }
        }
        var out: [Int: SIMD3<Float>] = [:]
        out.reserveCapacity(y.count)
        for (u, yy) in y { out[u] = SIMD3(columnX(u / n, g), yy, traceNodeZ) }
        return out
    }

    /// Traced desks sit a hair in front of the dust so they are never hidden by it.
    static let traceNodeZ: Float = 0.004

    /// Centre of layer l's plane/tile (for labels and camera focus).
    static func layerCenter(_ l: Int, mode: LayoutMode, geometry: Geometry) -> SIMD3<Float> {
        place(0, 0, layer: l, mode: mode, g: geometry)
    }

    /// Label anchors for floor l: just above and just below its plane / tile / column (world space).
    static func layerLabelAnchors(_ l: Int, mode: LayoutMode, geometry: Geometry) -> (top: SIMD3<Float>, bottom: SIMD3<Float>) {
        switch mode {
        case .columns:
            let x = columnX(l, geometry)
            return (SIMD3(x, columnsHalfHeight + 0.05, 0), SIMD3(x, columnHeadY - 0.04, 0))
        case .stack, .cluster, .sheet:
            return (place(0, 0.8, layer: l, mode: mode, g: geometry), place(0, -0.8, layer: l, mode: mode, g: geometry))
        }
    }
}
