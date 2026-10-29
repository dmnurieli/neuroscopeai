import SwiftUI

struct TimelineView: View {
    @Environment(AppState.self) private var state

    private let chipGap: CGFloat = 2
    private func chipWidth(_ t: TokenRecord) -> CGFloat {
        min(max(CGFloat(t.displayText.count) * 7 + 10, 14), 120)
    }

    var body: some View {
        HStack(spacing: 10) {
            if let run = state.activeRun, run.count > 0 {
                controls
                lane(run)
            } else {
                Text("Each word the model writes will appear here. Click one to see the brain at that moment.")
                    .font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 10)
        .frame(height: 64)
        .background(Theme.panel)
    }

    private var controls: some View {
        HStack(spacing: 8) {
            Button { state.togglePlay() } label: {
                Image(systemName: state.isPlaying ? "pause.fill" : "play.fill").frame(width: 18)
            }
            .buttonStyle(.bordered).controlSize(.small)
            if state.showAdvanced { Menu {
                ForEach([3.0, 6, 12, 24, 48], id: \.self) { s in
                    Button { state.playbackTokensPerSecond = s } label: {
                        Text("\(Int(s)) tok/s" + (state.playbackTokensPerSecond == s ? "  ✓" : ""))
                    }
                }
            } label: {
                Text("\(Int(state.playbackTokensPerSecond))/s").font(.caption.monospacedDigit())
            }
            .menuStyle(.borderlessButton).fixedSize() }
            let live = state.cursor == nil
            if !live { Button { state.setCursor(nil) } label: {
                Text("Back to the end").font(.caption.bold())
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(Capsule().fill(live ? Theme.positive.opacity(0.85) : Color.white.opacity(0.1)))
                    .foregroundStyle(live ? Color.black : Color.secondary)
            }
            .buttonStyle(.plain) }
        }
    }

    /// Real token indices to show. Beginners lose the chat-template scaffolding; Advanced sees everything.
    private func visibleIndices(_ run: Run) -> [Int] {
        let toks = run.tokens
        if state.showAdvanced { return Array(toks.indices) }
        let pc = run.promptCount
        let scaffold = run.scaffoldMask
        var out = [Int]()
        for i in toks.indices {
            let t = toks[i]
            if i < pc {
                if t.stream == .special || scaffold[i] { continue }
                if t.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
            } else if t.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && t.stream != .special {
                continue
            }
            out.append(i)
        }
        return out.isEmpty ? Array(toks.indices) : out
    }

    private func lane(_ run: Run) -> some View {
        let all = run.tokens
        let vis = visibleIndices(run)
        let pc = run.promptCount
        let labelW: CGFloat = 78
        let showLabels = !state.showAdvanced
        var offsets = [CGFloat](); offsets.reserveCapacity(vis.count + 1)
        var labels = [(CGFloat, String)]()
        var x: CGFloat = 4
        for (k, ri) in vis.enumerated() {
            if showLabels {
                let first = k == 0 || (all[vis[k - 1]].stream == .prompt) != (all[ri].stream == .prompt) && ri >= pc && vis[k - 1] < pc
                if k == 0 || first {
                    labels.append((x, ri < pc ? "your question" : "the reply"))
                    x += labelW
                }
            }
            offsets.append(x); x += chipWidth(all[ri]) + chipGap
        }
        offsets.append(x)
        let total = x + 4
        let offs = offsets
        let lbls = labels
        let toks = vis.map { all[$0] }
        let realCur = state.displayIndex
        // visible slot of the cursor: nearest shown token at or before it
        let cur: Int? = realCur.map { rc in
            var best = 0
            for (k, ri) in vis.enumerated() where ri <= rc { best = k }
            return best
        }

        func index(at px: CGFloat) -> Int {
            var lo = 0, hi = toks.count - 1
            while lo < hi {
                let mid = (lo + hi + 1) / 2
                if offs[mid] <= px { lo = mid } else { hi = mid - 1 }
            }
            return vis[lo]
        }

        return ScrollViewReader { proxy in
            ScrollView(.horizontal, showsIndicators: false) {
                ZStack(alignment: .leading) {
                    HStack(spacing: 0) {
                        if let c = cur {
                            Color.clear.frame(width: offs[c], height: 1)
                            Color.clear.frame(width: chipWidth(toks[c]), height: 1).id("cursorAnchor")
                            Spacer(minLength: 0)
                        } else { Spacer(minLength: 0) }
                    }
                    Canvas { ctx, size in
                        let y: CGFloat = 12, h = size.height - 24
                        for (x0, name) in lbls {
                            ctx.draw(Text(name).font(.system(size: 9, weight: .medium)).foregroundColor(.white.opacity(0.5)),
                                     at: CGPoint(x: x0 + 2, y: size.height / 2), anchor: .leading)
                        }
                        for (i, t) in toks.enumerated() {
                            let quiet = showLabels && vis[i] < pc
                            let r = CGRect(x: offs[i], y: y, width: chipWidth(t), height: h)
                            let c = Theme.color(for: t.stream)
                            let isCur = i == cur
                            if isCur {
                                ctx.fill(Path(roundedRect: r.insetBy(dx: -3, dy: -3), cornerRadius: 7),
                                         with: .color(c.opacity(0.35)))
                            }
                            ctx.fill(Path(roundedRect: r, cornerRadius: 5), with: .color(c.opacity(isCur ? 0.95 : (quiet ? 0.2 : 0.4))))
                            let label = t.displayText.replacingOccurrences(of: "\n", with: "↵")
                            ctx.draw(Text(label).font(.system(size: 10, design: .monospaced))
                                .foregroundColor(isCur ? .black : .white.opacity(0.85)),
                                     in: r.insetBy(dx: 2, dy: 0))
                        }
                        if let c = cur {
                            let px = offs[c] + chipWidth(toks[c]) / 2
                            ctx.fill(Path(CGRect(x: px - 0.75, y: 2, width: 1.5, height: size.height - 4)),
                                     with: .color(.white))
                        }
                    }
                    .frame(width: total, height: 64)
                    .contentShape(Rectangle())
                    .gesture(DragGesture(minimumDistance: 0).onChanged { v in
                        state.setCursor(index(at: v.location.x))
                    })
                    .simultaneousGesture(TapGesture(count: 2).onEnded { state.setCursor(nil) })
                }
                .frame(width: total)
            }
            .onChange(of: realCur) { _, _ in
                if realCur != nil { proxy.scrollTo("cursorAnchor", anchor: .center) }
            }
        }
    }
}
