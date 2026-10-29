import SwiftUI

/// The first thing a person sees: a calm list of short lessons (the airline course first, then the real-AI ones), the next one gently lit, and one quiet way out
/// to Explore. Nothing to configure.
struct HomeView: View {
    @Environment(AppState.self) private var state

    /// The airline course is done when every "Start here" lesson is (then the real-AI lessons light up).
    private var airlineDone: Bool { AirlineLessonID.course.allSatisfy { state.completedLessons.contains($0.rawValue) } }
    private var nextAirline: AirlineLessonID? { AirlineLessonID.course.first { !state.completedLessons.contains($0.rawValue) } }
    private func unlocked(_ i: Int) -> Bool {
        i == 0 || state.completedLessons.contains(AirlineLessonID.course[i - 1].rawValue)
    }
    private var nextID: String? { Lessons.all.first { !state.completedLessons.contains($0.id) }?.id }

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                Text("How an AI thinks")
                    .font(.system(size: 40, weight: .semibold, design: .rounded))
                Text("Short lessons. Build a tiny network, then watch a real AI write, one word at a time.")
                    .font(.system(size: 17))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.top, 10)

                sectionHeader("Start here").padding(.top, 44)
                VStack(spacing: 12) {
                    ForEach(Array(AirlineLessonID.course.enumerated()), id: \.element) { i, lesson in
                        LessonRow(label: i == 0 ? .symbol(lesson.symbol) : .number(i), title: lesson.title, subtitle: lesson.subtitle,
                                  minutes: lesson.minutes, done: state.completedLessons.contains(lesson.rawValue),
                                  isNext: lesson == nextAirline, locked: !unlocked(i)) {
                            state.enterAirline(lesson)
                        }
                    }
                    if state.completedLessons.contains(AirlineLessonID.hubs.rawValue) {
                        LessonRow(label: .symbol(AirlineLessonID.builder.symbol), title: AirlineLessonID.builder.title,
                                  subtitle: AirlineLessonID.builder.subtitle, minutes: AirlineLessonID.builder.minutes,
                                  done: false, isNext: false) {
                            state.enterAirline(.builder)
                        }
                    }
                }
                .padding(.top, 12)

                sectionHeader("Inside a real AI").padding(.top, 40)
                if !ModelStore.isInstalled() && state.status != .ready {
                    Text("These use a real AI model, \(ModelStore.spec.displayName) (a one-time \(ModelStore.spec.downloadText) download).")
                        .font(.system(size: 14)).foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 6)
                }
                VStack(spacing: 12) {
                    ForEach(Array(Lessons.all.enumerated()), id: \.element.id) { i, lesson in
                        LessonRow(label: .number(i + 1), title: lesson.title, subtitle: lesson.subtitle,
                                  minutes: lesson.minutes,
                                  done: state.completedLessons.contains(lesson.id),
                                  isNext: airlineDone && lesson.id == nextID) {
                            state.startLesson(lesson.id)
                        }
                    }
                }
                .padding(.top, 12)

                if nextID == nil && !Lessons.all.isEmpty {
                    Text("You finished every lesson.")
                        .font(.system(size: 15)).foregroundStyle(.secondary)
                        .padding(.top, 24)
                }

                Button("Explore freely") { state.enterExplore() }
                    .buttonStyle(.plain)
                    .font(.system(size: 15))
                    .foregroundStyle(.secondary)
                    .padding(.top, 36)
            }
            .frame(maxWidth: 600)
            .padding(.horizontal, 32)
            .padding(.vertical, 80)
            .frame(maxWidth: .infinity)
        }
        .background(Theme.background)
    }

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 13, weight: .semibold))
            .textCase(.uppercase)
            .tracking(0.8)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct LessonRow: View {
    enum Label { case number(Int), symbol(String) }
    let label: Label
    let title: String
    let subtitle: String
    let minutes: Int
    let done: Bool
    let isNext: Bool
    var locked = false
    let start: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: start) {
            HStack(spacing: 18) {
                ZStack {
                    Circle().fill(done ? Theme.prompt.opacity(0.18) : Color.white.opacity(isNext ? 0.12 : 0.06))
                    if locked {
                        Image(systemName: "lock.fill").font(.system(size: 13, weight: .semibold)).foregroundStyle(.tertiary)
                    } else if done {
                        Image(systemName: "checkmark").font(.system(size: 15, weight: .bold)).foregroundStyle(Theme.prompt)
                    } else {
                        switch label {
                        case .number(let n):
                            Text("\(n)").font(.system(size: 16, weight: .semibold, design: .rounded))
                                .foregroundStyle(isNext ? .primary : .secondary)
                        case .symbol(let name):
                            Image(systemName: name).font(.system(size: 15, weight: .semibold))
                                .foregroundStyle(isNext ? .primary : .secondary)
                        }
                    }
                }
                .frame(width: 38, height: 38)

                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.system(size: 19, weight: .semibold))
                        .foregroundStyle(done && !isNext ? .secondary : .primary)
                    Text(subtitle).font(.system(size: 14)).foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer(minLength: 12)
                if isNext {
                    Text("Start")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Color.black)
                        .padding(.horizontal, 18).padding(.vertical, 8)
                        .background(Capsule().fill(Theme.answer))
                } else {
                    Text("\(minutes) min").font(.system(size: 13)).foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal, 22).padding(.vertical, 18)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(Color.white.opacity(isNext ? 0.075 : (hovering ? 0.05 : 0.03)))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Color.white.opacity(isNext ? 0.16 : 0.06))
            )
            .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .buttonStyle(.plain)
        .disabled(locked)
        .opacity(locked ? 0.5 : 1)
        .help(locked ? "Finish the lesson above to open this one" : "")
        .accessibilityHint(locked ? "Locked until the lesson above is finished" : "")
        .onHover { hovering = $0 }
        .animation(.easeOut(duration: 0.15), value: hovering)
    }
}
