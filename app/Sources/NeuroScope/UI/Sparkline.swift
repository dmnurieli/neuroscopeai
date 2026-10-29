import SwiftUI

/// Tiny line chart of one neuron across a run's tokens; marks the cursor; tap to move the cursor.
struct Sparkline: View {
    var values: [Float]
    var cursor: Int?
    var promptCount: Int = 0          // shade the prompt region
    var onTap: ((Int) -> Void)? = nil
    var body: some View {
        Canvas { ctx, size in
            let n = values.count
            guard n > 0, size.width > 0 else { return }
            let step = n > 1 ? size.width / CGFloat(n - 1) : 0
            func x(_ i: Int) -> CGFloat { n > 1 ? CGFloat(i) * step : size.width / 2 }
            if promptCount > 0 {
                let w = n > 1 ? x(min(promptCount, n) - 1) : size.width
                ctx.fill(Path(CGRect(x: 0, y: 0, width: max(w, 1), height: size.height)), with: .color(Theme.prompt.opacity(0.12)))
            }
            var lo = min(values.min() ?? 0, 0), hi = max(values.max() ?? 0, 0)
            if hi - lo < 1e-6 { hi = lo + 1 }
            if lo == hi { lo -= 1 }
            let pad: CGFloat = 3
            func y(_ v: Float) -> CGFloat { pad + (1 - CGFloat((v - lo) / (hi - lo))) * (size.height - 2 * pad) }
            var zero = Path(); zero.move(to: CGPoint(x: 0, y: y(0))); zero.addLine(to: CGPoint(x: size.width, y: y(0)))
            ctx.stroke(zero, with: .color(.white.opacity(0.2)), style: StrokeStyle(lineWidth: 0.5, dash: [3, 3]))
            var p = Path()
            for i in 0..<n {
                let pt = CGPoint(x: x(i), y: y(values[i]))
                if i == 0 { p.move(to: pt) } else { p.addLine(to: pt) }
            }
            ctx.stroke(p, with: .color(Theme.positive), lineWidth: 1.2)
            if let c = cursor, c >= 0, c < n {
                var m = Path(); m.move(to: CGPoint(x: x(c), y: 0)); m.addLine(to: CGPoint(x: x(c), y: size.height))
                ctx.stroke(m, with: .color(.white.opacity(0.8)), lineWidth: 1)
                ctx.fill(Path(ellipseIn: CGRect(x: x(c) - 3, y: y(values[c]) - 3, width: 6, height: 6)), with: .color(.white))
            }
        }
        .frame(height: 56)
        .background(Color.black.opacity(0.25), in: RoundedRectangle(cornerRadius: 4))
        .overlay(GeometryReader { geo in
            Color.clear.contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onChanged { g in
                    guard let onTap, values.count > 0, geo.size.width > 0 else { return }
                    let f = min(max(g.location.x / geo.size.width, 0), 1)
                    onTap(Int((f * CGFloat(values.count - 1)).rounded()))
                })
        })
    }
}
