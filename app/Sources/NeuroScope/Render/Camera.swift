// Orbit camera: drag orbits, scroll/pinch zooms, shift/option-drag pans. Right-handed, Metal clip space (z 0…1).

import CoreGraphics
import simd

struct Camera {
    var target = SIMD3<Float>(0, 0, 0)
    var yaw: Float = -1.15          // radians around +Y
    var pitch: Float = 0.42         // radians, clamped to ±1.45
    var distance: Float = 6.0
    var fovY: Float = 0.75          // radians

    private static let near: Float = 0.02
    private static let far: Float = 200

    func viewMatrix() -> simd_float4x4 {
        let e = eye
        let f = simd_normalize(target - e)
        var r = simd_cross(f, SIMD3<Float>(0, 1, 0))
        if simd_length_squared(r) < 1e-8 { r = SIMD3(1, 0, 0) }
        r = simd_normalize(r)
        let u = simd_cross(r, f)
        return simd_float4x4(columns: (
            SIMD4(r.x, u.x, -f.x, 0),
            SIMD4(r.y, u.y, -f.y, 0),
            SIMD4(r.z, u.z, -f.z, 0),
            SIMD4(-simd_dot(r, e), -simd_dot(u, e), simd_dot(f, e), 1)))
    }

    func projectionMatrix(aspect: Float) -> simd_float4x4 {
        let ys = 1 / tan(fovY * 0.5)
        let xs = ys / max(aspect, 1e-4)
        let n = Camera.near, fa = Camera.far
        let zs = fa / (n - fa)
        return simd_float4x4(columns: (
            SIMD4(xs, 0, 0, 0),
            SIMD4(0, ys, 0, 0),
            SIMD4(0, 0, zs, -1),
            SIMD4(0, 0, zs * n, 0)))
    }

    func viewProjection(aspect: Float) -> simd_float4x4 { projectionMatrix(aspect: aspect) * viewMatrix() }

    var eye: SIMD3<Float> {
        let cp = cos(pitch)
        return target + distance * SIMD3(cp * sin(yaw), sin(pitch), cp * cos(yaw))
    }

    mutating func orbit(dx: Float, dy: Float) {
        yaw -= dx * 0.008
        pitch = min(max(pitch + dy * 0.008, -1.45), 1.45)
    }

    mutating func zoom(by factor: Float) {
        guard factor.isFinite, factor > 0 else { return }
        distance = min(max(distance / factor, 0.4), 12)
    }

    /// Pan in the view plane by a screen-space delta (points) for a view of `size`.
    mutating func pan(dx: Float, dy: Float, viewSize: CGSize) {
        let h = Float(max(viewSize.height, 1))
        let worldPerPt = 2 * distance * tan(fovY * 0.5) / h
        let v = viewMatrix()
        let right = SIMD3(v.columns.0.x, v.columns.1.x, v.columns.2.x)
        let up = SIMD3(v.columns.0.y, v.columns.1.y, v.columns.2.y)
        target += (-right * dx + up * dy) * worldPerPt
    }

    /// World → view point (origin top-left, y down, in points). nil when behind the camera.
    func project(_ p: SIMD3<Float>, viewSize: CGSize) -> (point: CGPoint, depth: Float)? {
        let w = Float(viewSize.width), h = Float(viewSize.height)
        guard w > 0, h > 0 else { return nil }
        let c = viewProjection(aspect: w / h) * SIMD4(p, 1)
        guard c.w > Camera.near else { return nil }
        let nx = c.x / c.w, ny = c.y / c.w
        return (CGPoint(x: CGFloat((nx + 1) * 0.5 * w), y: CGFloat((1 - ny) * 0.5 * h)), c.z / c.w)
    }

    /// Default framing for a layout (stack: 3/4 view; sheet and columns: straight on).
    static func framing(for mode: LayoutMode, aspect: Float = 1.6) -> Camera {
        var c = Camera()
        switch mode {
        case .stack, .cluster:
            // Oblique 3/4 view so the deep stack reads as separate receding planes.
            c.yaw = -1.15; c.pitch = 0.42; c.distance = 6.0
        case .sheet:
            c.yaw = 0; c.pitch = 0; c.distance = 3.0
        case .columns:
            // Straight on, the whole diagram in view with room above for floor numbers and below for the strip.
            // Fit the wide diagram (plus room for floor numbers above and the explained strip below) to the view's
            // actual shape, so it fills the width instead of sitting tall in the middle.
            c.yaw = 0; c.pitch = 0; c.target = SIMD3(0, -0.02, 0)
            let halfW = NeuronLayout.columnsHalfWidth + 0.12
            let halfH = NeuronLayout.columnsHalfHeight + 0.2
            let t = tan(c.fovY / 2)
            c.distance = max(halfH / t, halfW / (t * max(aspect, 0.2))) * 1.02
        }
        return c
    }
}
