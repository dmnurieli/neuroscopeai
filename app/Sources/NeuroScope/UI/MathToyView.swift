import SwiftUI

/// The 3-number worked example, drawn large. Each stage adds one thing, so the eye only has one new thing to read.
///   0: the note (2, 5, 1) and the French color (0, 0.6, 0.8)
///   1: multiply each pair: 0, 3, 0.8
///   2: add them: 3.8 = how French the note is
///   3: erase: subtract 3.8 × color = (0, 2.28, 3.04), leaving (2, 2.72, −2.04)
///   4: check: the new note has no French left (0 + 1.632 − 1.632 = 0)
struct MathToyView: View {
    let stage: Int

    private let note: [Double] = [2, 5, 1]
    private let color: [Double] = [0, 0.6, 0.8]
    private var products: [Double] { zip(note, color).map { $0 * $1 } }      // 0, 3, 0.8
    private var amount: Double { products.reduce(0, +) }                    // 3.8
    private var removed: [Double] { color.map { $0 * amount } }             // 0, 2.28, 3.04
    private var result: [Double] { zip(note, removed).map { $0 - $1 } }     // 2, 2.72, −2.04
    private var check: [Double] { zip(result, color).map { $0 * $1 } }      // 0, 1.632, −1.632

    private let noteTint = Theme.answer
    private let frenchTint = Theme.negative

    var body: some View {
        VStack(spacing: 34) {
            if stage < 3 { measuring } else { erasing }
        }
        .padding(.bottom, 150)   // leave room for the caption card
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.background)
        .animation(.spring(response: 0.5, dampingFraction: 0.85), value: stage)
    }

    // MARK: stages 0–2: how French is the note?

    private var measuring: some View {
        VStack(spacing: 22) {
            row(label: "note", values: note, tint: noteTint)
            row(label: "French", values: color, tint: frenchTint)
            if stage >= 1 {
                Rectangle().fill(Color.white.opacity(0.15)).frame(width: 3 * cell + 2 * gap, height: 1)
                    .transition(.opacity)
                row(label: "×", values: products, tint: .white, joiner: "+")
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
            if stage >= 2 {
                total(label: "= how French", value: amount)
                    .transition(.scale(scale: 0.8).combined(with: .opacity))
            }
        }
    }

    // MARK: stages 3–4: remove the French

    private var erasing: some View {
        VStack(spacing: 22) {
            row(label: "note", values: note, tint: noteTint)
            row(label: "− 3.8 × French", values: removed, tint: frenchTint)
            Rectangle().fill(Color.white.opacity(0.15)).frame(width: 3 * cell + 2 * gap, height: 1)
            row(label: "new note", values: result, tint: noteTint)
                .transition(.move(edge: .top).combined(with: .opacity))
            if stage >= 4 {
                VStack(spacing: 14) {
                    row(label: "× French", values: check, tint: .white, joiner: "+")
                    total(label: "= how French", value: check.reduce(0, +))
                }
                .padding(.top, 10)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
    }

    // MARK: pieces

    private let cell: CGFloat = 132
    private let gap: CGFloat = 20
    private let labelWidth: CGFloat = 170

    private func row(label: String, values: [Double], tint: Color, joiner: String? = nil) -> some View {
        HStack(spacing: gap) {
            Text(label)
                .font(.system(size: 20, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)
                .lineLimit(1).minimumScaleFactor(0.7)
                .frame(width: labelWidth, alignment: .trailing)
            ForEach(values.indices, id: \.self) { i in
                Text(Self.format(values[i]))
                    .font(.system(size: 38, weight: .semibold, design: .rounded).monospacedDigit())
                    .foregroundStyle(tint)
                    .contentTransition(.numericText())
                    .frame(width: cell, height: 76)
                    .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(tint.opacity(0.10)))
                    .overlay(alignment: .trailing) {
                        if let j = joiner, i < values.count - 1 {
                            Text(j).font(.system(size: 24, weight: .medium)).foregroundStyle(.tertiary)
                                .offset(x: gap / 2 + 7)
                        }
                    }
            }
            Color.clear.frame(width: labelWidth)   // keeps the numbers centered in the window
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(label)
    }

    private func total(label: String, value: Double) -> some View {
        HStack(spacing: 16) {
            Text(label)
                .font(.system(size: 22, weight: .medium, design: .rounded))
                .foregroundStyle(.secondary)
            Text(Self.format(value))
                .font(.system(size: 56, weight: .bold, design: .rounded).monospacedDigit())
                .foregroundStyle(abs(value) < 1e-9 ? Theme.prompt : frenchTint)
                .contentTransition(.numericText())
        }
    }

    /// 3 → "3", 0.8 → "0.8", 2.28 → "2.28", −2.04 → "−2.04", 1.632 → "1.632"; tiny float noise → "0".
    static func format(_ v: Double) -> String {
        if abs(v) < 1e-9 { return "0" }
        var s = String(format: "%.3f", v)
        while s.hasSuffix("0") { s.removeLast() }
        if s.hasSuffix(".") { s.removeLast() }
        return s.replacingOccurrences(of: "-", with: "−")
    }
}
