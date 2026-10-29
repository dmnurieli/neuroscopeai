import SwiftUI

struct RunsList: View {
    @Environment(AppState.self) private var state

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 4) {
                ForEach(state.runs.reversed()) { run in
                    RunRow(run: run,
                           active: run.id == state.activeRunID,
                           isBaseline: run.id == state.baselineID)
                        .contentShape(Rectangle())
                        .onTapGesture { state.showRun(run.id) }
                        .contextMenu {
                            Button("Set as baseline") { state.baselineID = run.id }
                            Button("Delete", role: .destructive) { state.deleteRun(run.id) }
                        }
                }
            }
            .padding(6)
        }
    }
}

private struct RunRow: View {
    let run: Run
    let active: Bool
    let isBaseline: Bool

    var body: some View {
        HStack(spacing: 8) {
            if run.edits.isEmpty {
                Image(systemName: isBaseline ? "star.fill" : "circle")
                    .foregroundStyle(isBaseline ? Theme.pinned : Theme.special)
            } else {
                Text("\(run.edits.count)")
                    .font(.caption.bold()).foregroundStyle(.white)
                    .padding(.horizontal, 6).padding(.vertical, 1)
                    .background(Theme.scaled, in: Capsule())
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(run.label).font(.callout).lineLimit(1)
                Text(detail).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 0)
            if run.status == .running { ProgressView().controlSize(.small) }
        }
        .padding(6)
        .background(active ? Theme.scaled.opacity(0.25) : Color.clear, in: RoundedRectangle(cornerRadius: 6))
    }

    private var detail: String {
        var parts = ["\(run.count) tok"]
        switch run.status {
        case .running: parts.append("running")
        case .cancelled: parts.append("cancelled")
        case .failed(let e): parts.append("failed: \(e)")
        case .finished: if let s = run.summary?.stop { parts.append(s) }
        }
        if let t = run.summary?.decodeTokPerS { parts.append(String(format: "%.1f tok/s", t)) }
        return parts.joined(separator: " · ")
    }
}
