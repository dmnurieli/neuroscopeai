import SwiftUI
import AirlineKit

/// The airline course's first lesson, "Your first network" (the minimal loop): one flight, a wrong guess, one
/// visible nudge, practice on the ledger, then a flight it has never seen. Same shell as LessonPlayer: a tiny
/// top bar, the stage, one caption card with one sentence and one button. Return continues, ← back, Esc leaves.
struct AirlineLoopView: View {
    /// Called when the learner finishes or leaves; `completed` is true only when they reached the last page.
    var onExit: (_ completed: Bool) -> Void

    @State private var model = AirlineState()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        AirlineLessonShell(model: model, onExit: onExit) {
            if model.network != nil, let flight = model.flight(model.displayFlight) {
                HStack(alignment: .center, spacing: 18) {
                    if model.page == AirlineState.practicePage {
                        PileView(title: "Study pile", years: "2016–2022", count: model.studyCount,
                                 flipping: model.practice == .running && !reduceMotion, start: model.practiceStart)
                    }
                    flightCard(flight)
                    if model.page == AirlineState.practicePage {
                        PileView(title: "Quiz pile", years: "2023", count: model.checkCount, flipping: false, start: model.practiceStart)
                    }
                }
                .padding(.top, 6)
                AirlineMapView(model: model)
                    .padding(.horizontal, 28)
                    .padding(.vertical, 18)
                    .frame(maxWidth: 1100, maxHeight: .infinity)
            } else {
                Spacer(minLength: 0)
            }
        } accessory: { copy in
            if let rule = copy.rule { RuleCardView(card: rule) }
            if copy.knob { KnobControl(model: model) }
        }
        .animation(.easeInOut(duration: 0.35), value: model.isReady)
        .task { await model.load() }
    }

    /// The flight on the card; while practising, it flips through real flights from the study pile instead.
    @ViewBuilder private func flightCard(_ flight: Flight) -> some View {
        if model.practice == .running && !reduceMotion {
            SwiftUI.TimelineView(.periodic(from: model.practiceStart, by: 1.0 / 12)) { tl in
                FlightCardView(flight: model.logbookFlight(at: tl.date) ?? flight, revealed: true, logbook: true)
            }
        } else {
            FlightCardView(flight: flight, revealed: model.truthRevealed)
                .id(model.displayFlight)
                .transition(.opacity)
        }
    }
}

// MARK: - the walk's rule card: A × B = C, every tile a live number

private struct RuleCardView: View {
    let card: RuleCard

    var body: some View {
        VStack(spacing: 8) {
            HStack(alignment: .center, spacing: 8) {
                ForEach(Array(card.tiles.enumerated()), id: \.offset) { n, tile in
                    if n == 1 { op("×") }
                    if n == 2 { op("=") }
                    tileView(tile)
                }
            }
            if let f = card.footnote {
                Text(f)
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(0.5))
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private func op(_ s: String) -> some View {
        Text(s).font(.system(size: 20, weight: .light)).foregroundStyle(.secondary)
    }

    private func tileView(_ tile: RuleTile) -> some View {
        VStack(spacing: 4) {
            Text(tile.label)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(.white.opacity(0.5))
                .lineLimit(2)
                .multilineTextAlignment(.center)
            Text(tile.value)
                .font(.system(size: 17, weight: .semibold, design: .monospaced))
                .foregroundStyle(tile.blame ? MapModel.blame : Theme.answer)
            if let bar = tile.bar {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.white.opacity(0.1))
                        Capsule().fill(Theme.positive.opacity(0.85))
                            .frame(width: geo.size.width * CGFloat(min(abs(bar), 1)))
                    }
                }
                .frame(width: 70, height: 4)
            }
            if let note = tile.note {
                Text(note).font(.system(size: 10)).foregroundStyle(.white.opacity(0.4))
            }
        }
        .padding(.horizontal, 10).padding(.vertical, 8)
        .frame(minWidth: 128)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(Color.white.opacity(0.05)))
        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous)
            .stroke(tile.blame ? MapModel.blame.opacity(0.5) : Color.white.opacity(0.1)))
    }
}

// MARK: - hotter or colder: one route's size on a knob, the gap answers live

private struct KnobControl: View {
    let model: AirlineState

    var body: some View {
        if let w = model.walkPlan {
            let range = w.knobRange
            let value = model.knobWeight ?? w.knobStart
            VStack(spacing: 6) {
                HStack(spacing: 10) {
                    Text("turn down").font(.system(size: 11)).foregroundStyle(.secondary)
                    Slider(value: Binding(get: { Double(value) }, set: { model.setKnob(Float($0)) }),
                           in: Double(range.lowerBound)...Double(range.upperBound))
                        .frame(width: 280)
                        .accessibilityLabel("Route from \(w.name(layer: w.knob.layer, w.knob.from)) to the gate")
                        .accessibilityValue(feedback(w) ?? "not moved yet")
                    Text("turn up").font(.system(size: 11)).foregroundStyle(.secondary)
                }
                Text(feedback(w) ?? "Drag the knob.")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(color(w))
                    .monospacedDigit()
            }
        }
    }

    private func feedback(_ w: WalkPlan) -> String? {
        guard let g = model.knobGuess else { return nil }
        let gap0 = AirlineState.minutes(w.guessBefore - w.truth), gap = AirlineState.minutes(g - w.truth)
        if gap < gap0 { return "Warmer: off by \(gap) min (was \(gap0))" }
        if gap > gap0 { return "Colder: off by \(gap) min (was \(gap0))" }
        return "Same gap: off by \(gap) min"
    }

    private func color(_ w: WalkPlan) -> Color {
        guard let g = model.knobGuess else { return .secondary }
        let gap0 = abs(w.guessBefore - w.truth), gap = abs(g - w.truth)
        return gap < gap0 - 0.5 ? MapModel.warmer : (gap > gap0 + 0.5 ? MapModel.colder : .secondary)
    }
}

// MARK: - practice: the two piles of the ledger

private struct PileView: View {
    let title: String
    let years: String
    let count: Int
    let flipping: Bool
    let start: Date

    var body: some View {
        VStack(spacing: 5) {
            ZStack {
                ForEach(0..<3, id: \.self) { k in
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .fill(Theme.panel)
                        .overlay(RoundedRectangle(cornerRadius: 4, style: .continuous).stroke(Color.white.opacity(0.18)))
                        .frame(width: 44, height: 30)
                        .offset(x: CGFloat(k) * 2.5, y: CGFloat(-k) * 2.5)
                }
                if flipping {
                    // the top sheet lifts off and is replaced, about 12 times a second
                    SwiftUI.TimelineView(.periodic(from: start, by: 1.0 / 12)) { tl in
                        let phase = tl.date.timeIntervalSince(start) * 12
                        let u = phase - phase.rounded(.down)
                        RoundedRectangle(cornerRadius: 4, style: .continuous)
                            .fill(Theme.answer.opacity(0.18))
                            .frame(width: 44, height: 30)
                            .offset(x: 5 + CGFloat(u) * 10, y: -5 - CGFloat(u) * 8)
                            .opacity(1 - u)
                    }
                }
            }
            .frame(width: 60, height: 42)
            Text(title.uppercased()).font(.system(size: 9, weight: .semibold)).kerning(0.6).foregroundStyle(.white.opacity(0.5))
            Text("\(years) · \(count.formatted())").font(.system(size: 10, design: .monospaced)).foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
    }
}
