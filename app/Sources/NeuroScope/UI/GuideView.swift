import SwiftUI

/// Renders docs/GUIDE.md block by block.
struct GuideView: View {
    @Environment(AppState.self) private var state
    @State private var blocks: [Block] = []

    struct Block: Identifiable { let id = UUID(); let kind: Kind; let text: String
        enum Kind { case h(Int), p, code, bullet, rule } }

    /// Development fallback: docs/GUIDE.md in the source tree (the app bundle carries its own copy).
    private static let repoPath = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("docs/GUIDE.md").path

    private static func load() -> String {
        if let u = Bundle.main.url(forResource: "GUIDE", withExtension: "md"), let s = try? String(contentsOf: u, encoding: .utf8) { return s }
        if let s = try? String(contentsOfFile: repoPath, encoding: .utf8) { return s }
        return "# Guide not found\n\nExpected docs/GUIDE.md at \(repoPath)."
    }

    static func parse(_ md: String) -> [Block] {
        var out: [Block] = []
        var para: [String] = []
        var code: [String]? = nil
        func flush() { if !para.isEmpty { out.append(Block(kind: .p, text: para.joined(separator: " "))); para = [] } }
        for line in md.components(separatedBy: "\n") {
            if line.hasPrefix("```") {
                if let c = code { out.append(Block(kind: .code, text: c.joined(separator: "\n"))); code = nil }
                else { flush(); code = [] }
                continue
            }
            if code != nil { code!.append(line); continue }
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.isEmpty { flush() }
            else if t.hasPrefix("#") {
                flush()
                let lvl = t.prefix(while: { $0 == "#" }).count
                out.append(Block(kind: .h(lvl), text: String(t.dropFirst(lvl)).trimmingCharacters(in: .whitespaces)))
            } else if t == "---" { flush(); out.append(Block(kind: .rule, text: "")) }
            else if t.hasPrefix("- ") || t.hasPrefix("* ") { flush(); out.append(Block(kind: .bullet, text: String(t.dropFirst(2)))) }
            else { para.append(t) }
        }
        flush()
        if let c = code { out.append(Block(kind: .code, text: c.joined(separator: "\n"))) }
        return out
    }

    private func inline(_ s: String) -> AttributedString {
        (try? AttributedString(markdown: s, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(s)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("NeuroScope Guide").font(.headline)
                Spacer()
                Button("Close") { state.showGuide = false }.keyboardShortcut(.cancelAction)
            }.padding(10)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach(blocks) { b in view(b) }
                }
                .frame(maxWidth: 720, alignment: .leading)
                .padding(24)
                .frame(maxWidth: .infinity)
            }
        }
        .frame(minWidth: 780, minHeight: 600)
        .background(Theme.background)
        .onAppear { if blocks.isEmpty { blocks = Self.parse(Self.load()) } }
    }

    @ViewBuilder private func view(_ b: Block) -> some View {
        switch b.kind {
        case .h(let l):
            Text(inline(b.text)).font(.system(size: l == 1 ? 26 : l == 2 ? 20 : 16, weight: .bold))
                .padding(.top, l <= 2 ? 10 : 4)
        case .p:
            Text(inline(b.text)).font(.system(size: 14)).lineSpacing(3).fixedSize(horizontal: false, vertical: true)
        case .bullet:
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text("•")
                Text(inline(b.text)).font(.system(size: 14)).lineSpacing(3).fixedSize(horizontal: false, vertical: true)
            }.padding(.leading, 8)
        case .code:
            ScrollView(.horizontal) {
                Text(b.text).font(.system(size: 12, design: .monospaced)).textSelection(.enabled).padding(10)
            }
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.white.opacity(0.07)))
        case .rule:
            Divider()
        }
    }
}
