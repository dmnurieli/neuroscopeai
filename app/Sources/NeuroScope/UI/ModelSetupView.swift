import SwiftUI

struct ModelSetupView: View {
    @Environment(AppState.self) private var state
    var body: some View {
        VStack(spacing: 14) {
            Text("NeuroScope").font(.system(size: 34, weight: .semibold, design: .rounded))
            let g = state.geometry
            Text("\(ModelStore.spec.displayName) · \(g.floorsText) layers · \(g.allDesksText) neurons")
                .font(.callout).foregroundStyle(.secondary)
            Text(markdown("A small, fully open AI model, pictured as a building with \(g.floorsText) floors (**layers**) and \(g.allDesksText) desks (**neurons**). The first time, it downloads once (\(ModelStore.spec.downloadText))."))
                .font(.caption).foregroundStyle(.secondary)
                .multilineTextAlignment(.center).frame(maxWidth: 360)
            Group {
                switch state.status {
                case .downloading(let p):
                    VStack(spacing: 6) {
                        ProgressView(value: p).frame(width: 260)
                        Text("Downloading the model… \(Int(p * 100))%").font(.caption).foregroundStyle(.secondary)
                    }
                case .idle:
                    VStack(spacing: 6) { ProgressView(); Text("Getting ready…").font(.caption).foregroundStyle(.secondary) }
                case .checking:
                    VStack(spacing: 6) { ProgressView(); Text("Checking the model files are complete…").font(.caption).foregroundStyle(.secondary) }
                case .loading:
                    VStack(spacing: 6) { ProgressView(); Text("Opening the building: loading the model into memory…").font(.caption).foregroundStyle(.secondary) }
                case .failed(let msg):
                    VStack(spacing: 8) {
                        Text(msg).font(.caption).foregroundStyle(Theme.ablated)
                            .multilineTextAlignment(.center).frame(maxWidth: 360)
                        Button("Try again") { Task { await state.bootstrap() } }
                    }
                case .ready:
                    EmptyView()
                }
            }
            .padding(.top, 6)
            Button("Back to home") { state.exitLesson() }
                .buttonStyle(.plain).font(.caption).foregroundStyle(.secondary)
                .padding(.top, 4)
        }
        .padding(36)
        .background(RoundedRectangle(cornerRadius: 14).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.white.opacity(0.08)))
    }
}
