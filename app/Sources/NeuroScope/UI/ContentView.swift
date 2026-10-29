import SwiftUI

/// The window. Always opens on Home (no download). By mode: the home list, the airline course, or, for the real-AI
/// lessons and Explore, the setup screen until the model is ready, then one lesson page (lesson), or Explore: the step bar on top, the current step on the left, the neuron field (with its
/// timeline) in the middle, and the Advanced inspector on the right when shown.
struct ContentView: View {
    @Environment(AppState.self) private var state
    var body: some View {
        Group {
            switch state.mode {
            case .home: HomeView().transition(.opacity)
            case .airline:
                airlineLesson.id(state.airlineLesson).transition(.opacity)
            case .lesson, .explore:
                if state.status != .ready {
                    ModelSetupView().frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if state.mode == .lesson {
                    LessonPlayer().transition(.opacity)
                } else {
                    main.transition(.opacity)
                }
            }
        }
        .animation(.easeInOut(duration: 0.3), value: state.mode)
        .background(Theme.background)
        .sheet(isPresented: Bindable(state).showGuide) { GuideView().environment(state) }
    }

    /// The airline course: each lesson on the shared shell (AirlineShell).
    @ViewBuilder private var airlineLesson: some View {
        let exit: (Bool) -> Void = { completed in state.exitAirline(completed: completed) }
        switch state.airlineLesson {
        case .loop: AirlineLoopView(onExit: exit)
        case .rulebook: AirlineRulebookLessonView(onExit: exit)
        case .clipboard, .direct, .hubs, .builder: AirlineNetLessonView(kind: state.airlineLesson, onExit: exit)
        case .mountain: AirlineMountainLessonView(onExit: exit)
        case .hubNames: AirlineHubNamesLessonView(onExit: exit)
        case .exam: AirlineExamLessonView(onExit: exit)
        case .close: AirlineCloseLessonView(onExit: exit)
        case .graduation: AirlineGraduationLessonView(onExit: exit)
        }
    }

    private var main: some View {
        VStack(spacing: 0) {
            StepBar().overlay(alignment: .leading) { lessonsLink }
            hairline()
            HStack(spacing: 0) {
                ScrollView {
                    currentStep
                        .padding(20)
                        .frame(maxWidth: .infinity, alignment: .topLeading)
                }
                .frame(width: 360)
                .background(Theme.panel)
                vline()
                VStack(spacing: 0) {
                    NeuronFieldView().frame(minHeight: 320).frame(maxWidth: .infinity, maxHeight: .infinity)
                    hairline()
                    if state.follow != nil || state.isFollowing { FollowPanel(); hairline() }
                    else if state.slowMotion { SlowMotionBar(); hairline() }
                    TimelineView().frame(height: 64)
                }
                .frame(maxWidth: .infinity)
                if state.showAdvanced {
                    vline()
                    AdvancedPanel().frame(width: 360)
                        .transition(.move(edge: .trailing).combined(with: .opacity))
                }
            }
            .animation(.easeInOut(duration: 0.2), value: state.showAdvanced)
        }
        .onChange(of: state.step) { _, s in deriveColorMode(s) }
        .onChange(of: state.afterRun?.id) { _, _ in if state.step == .compare { deriveColorMode(.compare) } }
    }

    /// Sits over the step bar's title: the one way back to the lessons from Explore.
    private var lessonsLink: some View {
        Button { state.exitLesson() } label: {
            HStack(spacing: 4) {
                Image(systemName: "chevron.left").font(.system(size: 11, weight: .semibold))
                Text("Lessons").font(.system(size: 14, weight: .medium))
            }
            .foregroundStyle(.secondary)
            .padding(.horizontal, 8).frame(height: 30)
            .frame(minWidth: 104, alignment: .leading)
            .background(Theme.panel)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Back to the lessons (⌘1)")
        .padding(.leading, 72)
    }

    @ViewBuilder private var currentStep: some View {
        switch state.step {
        case .ask: AskStep()
        case .watch: WatchStep()
        case .change: ChangeStep()
        case .compare: CompareStep()
        }
    }

    /// The field's coloring follows the step: activation while asking/watching, the concept map while changing,
    /// before-vs-after while comparing. Advanced can still override it afterwards.
    private func deriveColorMode(_ s: AppState.GuideStep) {
        let want: ColorMode
        switch s {
        case .ask, .watch: want = .activation
        case .change: want = state.contrast != nil ? .contrast : .activation
        case .compare: want = state.afterRun != nil && state.beforeRun != nil ? .diff : .activation
        }
        if state.view.colorMode != want {
            state.view.colorMode = want
            state.bumpDisplay()
        }
    }

    private func hairline() -> some View { Rectangle().fill(Color.white.opacity(0.08)).frame(height: 1) }
    private func vline() -> some View { Rectangle().fill(Color.white.opacity(0.08)).frame(width: 1) }
}
