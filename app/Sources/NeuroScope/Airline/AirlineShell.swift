import SwiftUI
import AppKit
import AirlineKit

// The shared lesson shell for every airline lesson: a tiny top bar (title, page dots, exit),
// the lesson's stage, and one caption card (one sentence, one button). Return = the button, ← = back, Esc = leave,
// R = watch again. A lesson is page data (`AirlineLessonModel`) plus a stage view; its camera and sequences run on
// a `Choreographer`.

// MARK: - what a lesson provides

struct AirlinePageCopy {
    var text: String
    var detail: String?
    var button: String
    var enabled = true
    var working = false
    /// Longer versions the caption sizes itself to, so text that arrives mid-sequence does not reflow the stage.
    var reserveText: [String] = []
    var reserveDetail: [String] = []
    /// "Your first network": the walk's rule card and the hotter-or-colder knob.
    var rule: RuleCard?
    var knob = false
}

@MainActor protocol AirlineLessonModel: AnyObject {
    var title: String { get }
    var page: Int { get }
    var pageCount: Int { get }
    var copy: AirlinePageCopy { get }
    var canGoBack: Bool { get }
    var canWatchAgain: Bool { get }
    /// The page's button: skip a playing sequence, run the page's action, or move on. True on the last page's Done.
    func primary() -> Bool
    func back()
    func watchAgain()
    func stop()
}

// MARK: - the choreographer: the run on screen, skipping, and whether the map's timeline must tick

@MainActor @Observable
final class Choreographer {
    private(set) var run: AirlineRun?
    /// The run has reached the moment its numbers appear (always true for camera-only moves).
    private(set) var captionReady = true
    /// The map's TimelineView ticks only while something moves.
    private(set) var timelineActive = false

    /// Keeps the timeline ticking regardless of runs (practice, training).
    @ObservationIgnored var hold: () -> Bool = { false }
    /// One line for VoiceOver once a sequence reaches its numbers.
    @ObservationIgnored var summary: () -> String? = { nil }

    @ObservationIgnored private var serial = 0
    @ObservationIgnored private var announced = 0
    @ObservationIgnored private var aliveUntil = Date.distantPast

    var reduceMotion: Bool { NSWorkspace.shared.accessibilityDisplayShouldReduceMotion }

    /// A sequence (not a plain camera move) is playing: the button skips it.
    var isPlaying: Bool {
        guard let r = run, r.kind != .frame else { return false }
        return r.isPlaying(at: Date())
    }
    /// The current run has reached its numbers (true when the run is of another kind).
    func reached(_ kind: AirlineRun.Kind) -> Bool { run?.kind != kind || captionReady }

    func frame(_ focus: Focus, move: Double = Shots.frameMove) {
        let shot = Shots.frame(focus, move: move)
        start(.frame, keys: shot.keys, duration: shot.duration, captionAt: 0)
    }

    func start(_ kind: AirlineRun.Kind, keys: [CameraKey], duration: Double, captionAt: Double,
               configure: (inout AirlineRun) -> Void = { _ in }) {
        let now = Date()
        serial += 1
        var origin: AirlineRun.Origin = .focus(keys.first?.focus ?? .wide)
        if let prev = run {
            if prev.isPlaying(at: now) {
                // start from wherever the interrupted camera is (one level deep is enough)
                var flat = prev
                if case .midway(let earlier, let e) = prev.origin { flat.origin = .focus(earlier.focus(at: e)) }
                origin = .midway(flat, prev.elapsed(at: now))
            } else {
                origin = .focus(prev.endFocus)
            }
        }
        var r = AirlineRun(id: serial, kind: kind, start: now, duration: duration, captionAt: captionAt,
                           keys: keys, origin: origin, skipped: reduceMotion)
        configure(&r)
        run = r
        captionReady = r.skipped || captionAt <= 0
        keepAlive(r.skipped ? 0.05 : duration + 0.05)
        if captionReady {
            announce()
        } else {
            let id = r.id
            Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(captionAt))
                guard let self, self.run?.id == id, !self.captionReady else { return }
                self.captionReady = true
                self.announce()
            }
        }
    }

    /// The button pressed mid-sequence: jump to the end state now, never make the learner wait.
    func skip() {
        guard var r = run else { return }
        r.skipped = true
        run = r
        captionReady = true
        keepAlive(0.05)
        announce()
    }

    /// Keeps the map's timeline ticking for at least `seconds` more (`hold` keeps it ticking while true).
    func keepAlive(_ seconds: Double) {
        let until = Date().addingTimeInterval(seconds)
        if until > aliveUntil { aliveUntil = until }
        if !timelineActive { timelineActive = true }
        let wait = aliveUntil.timeIntervalSinceNow + 0.02
        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(max(wait, 0)))
            guard let self, Date() >= self.aliveUntil, !self.hold(), self.timelineActive else { return }
            self.timelineActive = false
        }
    }

    private func announce() {
        guard let r = run, r.id != announced, let line = summary() else { return }
        announced = r.id
        AccessibilityNotification.Announcement(line).post()
    }
}

// MARK: - the shell

struct AirlineLessonShell<Model: AirlineLessonModel, Stage: View, Accessory: View>: View {
    let model: Model
    /// Called when the learner finishes or leaves; `completed` is true only when they reached the last page.
    let onExit: (_ completed: Bool) -> Void
    @ViewBuilder var stage: () -> Stage
    @ViewBuilder var accessory: (AirlinePageCopy) -> Accessory

    var body: some View {
        ZStack {
            Theme.background.ignoresSafeArea()
            VStack(spacing: 0) {
                AirlineTopBar(title: model.title, page: model.page, pageCount: model.pageCount,
                              onClose: { model.stop(); onExit(false) })
                stage()
                AirlineCaptionCard(model: model, onDone: { model.stop(); onExit(true) }, accessory: accessory)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 28)
            }
        }
        .animation(.easeInOut(duration: 0.35), value: model.page)
        .onDisappear { model.stop() }
    }
}

// MARK: - top bar: lesson title, page dots, exit

struct AirlineTopBar: View {
    let title: String
    let page: Int
    let pageCount: Int
    let onClose: () -> Void

    var body: some View {
        HStack(spacing: 14) {
            Text(title)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
            Spacer(minLength: 12)
            if pageCount > 1 {
                HStack(spacing: 6) {
                    ForEach(0..<pageCount, id: \.self) { i in
                        Circle()
                            .fill(Color.white.opacity(i == page ? 0.85 : (i < page ? 0.35 : 0.14)))
                            .frame(width: 6, height: 6)
                    }
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Page \(page + 1) of \(pageCount)")
            }
            Spacer(minLength: 12)
            Button(action: onClose) {
                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .frame(width: 26, height: 26)
                    .background(Circle().fill(Color.white.opacity(0.07)))
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.cancelAction)
            .help("Back to the lessons")
            .accessibilityLabel("Leave the lesson")
        }
        .padding(.leading, 84)   // clear the window's traffic-light buttons (hidden title bar)
        .padding(.trailing, 16)
        .frame(height: 44)
    }
}

// MARK: - the caption card: one idea, one button

struct AirlineCaptionCard<Model: AirlineLessonModel, Accessory: View>: View {
    let model: Model
    let onDone: () -> Void
    @ViewBuilder var accessory: (AirlinePageCopy) -> Accessory

    var body: some View {
        let copy = model.copy
        VStack(spacing: 14) {
            // sized to the longest version, so numbers arriving mid-sequence never reflow (and move) the stage
            reserved(copy.reserveText) {
                Text(markdown($0))
                    .font(.system(size: 21, weight: .regular))
                    .lineSpacing(3)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            } shown: { copy.text }
            if let d = copy.detail {
                reserved(copy.reserveDetail) {
                    Text(markdown($0))
                        .font(.system(size: 14))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                } shown: { d }
            }
            accessory(copy)
            controls(copy).padding(.top, 4)
        }
        .padding(.horizontal, 32)
        .padding(.top, 24)
        .padding(.bottom, 20)
        .frame(maxWidth: 640)
        .background(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(.ultraThinMaterial)
                .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).fill(Theme.panel.opacity(0.55)))
        )
        .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Color.white.opacity(0.09)))
        .shadow(color: .black.opacity(0.35), radius: 24, y: 8)
        .id(model.page)
        // the old caption leaves fast, then the new one fades in: never two captions on top of each other
        .transition(.asymmetric(
            insertion: .opacity.combined(with: .offset(y: 10)).animation(.easeOut(duration: 0.28).delay(0.12)),
            removal: .opacity.animation(.easeOut(duration: 0.12))))
    }

    private func reserved<V: View>(_ versions: [String], @ViewBuilder _ text: @escaping (String) -> V,
                                   shown: () -> String) -> some View {
        let current = shown()
        return ZStack {
            ForEach(Array(versions.enumerated()), id: \.offset) { _, v in text(v).hidden().accessibilityHidden(true) }
            text(current)
        }
    }

    private func controls(_ copy: AirlinePageCopy) -> some View {
        ZStack {
            Button { if model.primary() { onDone() } } label: {
                HStack(spacing: 8) {
                    if copy.working { AirlineWaitingDot() }
                    Text(copy.button)
                }
                .font(.system(size: 16, weight: .semibold))
                .padding(.horizontal, 22).padding(.vertical, 5)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .tint(Theme.answer.opacity(0.92))
            .foregroundStyle(Color.black)
            .disabled(!copy.enabled)
            .keyboardShortcut(.defaultAction)

            HStack {
                if model.canGoBack {
                    Button { model.back() } label: {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 32, height: 32)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut(.leftArrow, modifiers: [])
                    .help("Back")
                    .accessibilityLabel("Back")
                }
                Spacer()
                if model.canWatchAgain {
                    Button { model.watchAgain() } label: {
                        Text("Watch again")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(.secondary)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut("r", modifiers: [])
                    .help("Replay what just happened on the map (R)")
                    .transition(.opacity)
                }
            }
        }
    }
}

/// A slow, quiet pulse while the page's work finishes (loading the logbook, practising).
struct AirlineWaitingDot: View {
    @State private var on = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    var body: some View {
        Circle().fill(Color.black.opacity(0.6)).frame(width: 7, height: 7)
            .opacity(on || reduceMotion ? 1 : 0.25)
            .onAppear {
                guard !reduceMotion else { return }
                withAnimation(.easeInOut(duration: 0.7).repeatForever(autoreverses: true)) { on = true }
            }
            .accessibilityLabel("Working")
    }
}

// MARK: - the flight card: route, date, scheduled times, and the real arrival (under a hand until page 4)

struct FlightCardView: View {
    let flight: Flight
    let revealed: Bool
    var logbook = false

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        HStack(spacing: 22) {
            VStack(alignment: .leading, spacing: 3) {
                Text("\(flight.origin) → \(flight.destination)")
                    .font(.system(size: 22, weight: .semibold, design: .rounded))
                    .foregroundStyle(Theme.answer)
                Text(Self.date(year: flight.year, day: flight.dayOfYear) + (logbook ? " · logbook" : ""))
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            divider
            column("Scheduled") {
                Text("\(Self.clock(flight.scheduledDeparture)) → \(Self.clock(flight.scheduledArrival))")
                    .font(.system(size: 15, weight: .medium, design: .monospaced))
                    .foregroundStyle(Theme.answer)
            }
            divider
            column("Arrived") {
                // the real arrival sits under the hand; on page 4 the hand lifts off it
                ZStack(alignment: .leading) {
                    Text("\(Self.clock(flight.actualArrival)) · \(AirlineState.phrase(flight.delay))")
                        .font(.system(size: 15, weight: .medium, design: .monospaced))
                        .foregroundStyle(Theme.answer)
                        .opacity(revealed ? 1 : 0)
                    HStack(spacing: 6) {
                        Image(systemName: "hand.raised.fill")
                        Text("hidden under a hand")
                    }
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(Theme.panel))
                    .offset(y: revealed ? -26 : 0)
                    .opacity(revealed ? 0 : 1)
                    .accessibilityHidden(revealed)
                }
            }
            .frame(minWidth: 170, alignment: .leading)
        }
        .padding(.horizontal, 22).padding(.vertical, 12)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Theme.panel))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Color.white.opacity(0.08)))
        .animation(reduceMotion ? nil : .easeOut(duration: 0.55), value: revealed)
        .accessibilityElement(children: .combine)
    }

    private var divider: some View {
        Rectangle().fill(Color.white.opacity(0.1)).frame(width: 1, height: 34)
    }

    private func column<C: View>(_ label: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label.uppercased())
                .font(.system(size: 10, weight: .semibold)).kerning(0.8)
                .foregroundStyle(.white.opacity(0.4))
            content()
        }
    }

    /// "Wed 22 Nov 2023" from the ledger's year (0 = 2016) and day of the year.
    static func date(year: Int, day: Int) -> String {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        guard let jan1 = cal.date(from: DateComponents(year: 2016 + year, month: 1, day: 1)),
              let d = cal.date(byAdding: .day, value: day - 1, to: jan1) else { return "\(2016 + year)" }
        let f = DateFormatter()
        f.calendar = cal
        f.timeZone = cal.timeZone
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "EEE d MMM yyyy"
        return f.string(from: d)
    }

    /// Minutes after midnight → "17:05" (a late arrival past midnight wraps to the next day's clock).
    static func clock(_ minutes: Int) -> String {
        let m = ((minutes % 1440) + 1440) % 1440
        return String(format: "%02d:%02d", m / 60, m % 60)
    }
}
