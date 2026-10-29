import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

func render(_ s: Int) -> CGImage {
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let c = CGContext(data: nil, width: s, height: s, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    let f = CGFloat(s) / 1024
    c.scaleBy(x: f, y: f)
    let rect = CGRect(x: 40, y: 40, width: 944, height: 944)
    c.addPath(CGPath(roundedRect: rect, cornerWidth: 210, cornerHeight: 210, transform: nil))
    c.clip()
    let bg = CGGradient(colorsSpace: cs, colors: [CGColor(red: 0.07, green: 0.08, blue: 0.13, alpha: 1),
                                                  CGColor(red: 0.02, green: 0.02, blue: 0.04, alpha: 1)] as CFArray, locations: [0, 1])!
    c.drawRadialGradient(bg, startCenter: CGPoint(x: 512, y: 560), startRadius: 0, endCenter: CGPoint(x: 512, y: 512), endRadius: 700, options: .drawsAfterEndLocation)
    let layers = 5, per = 5
    var pts: [[CGPoint]] = []
    var seed: UInt64 = 7
    func rnd() -> CGFloat { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return CGFloat((seed >> 33) % 1000) / 1000 }
    for l in 0..<layers {
        let x = 200 + CGFloat(l) * 156
        pts.append((0..<per).map { i in CGPoint(x: x + (rnd() - 0.5) * 40, y: 230 + CGFloat(i) * 141 + (rnd() - 0.5) * 60) })
    }
    c.setLineWidth(3)
    for l in 0..<layers - 1 { for a in pts[l] { for b in pts[l + 1] where rnd() < 0.5 {
        c.setStrokeColor(CGColor(red: 0.3, green: 0.8, blue: 0.9, alpha: 0.22))
        c.move(to: a); c.addLine(to: b); c.strokePath() } } }
    for l in 0..<layers { for p in pts[l] {
        let hot = rnd() < 0.4
        let col = hot ? CGColor(red: 1, green: 0.7, blue: 0.2, alpha: 1) : CGColor(red: 0.3, green: 0.85, blue: 0.95, alpha: 1)
        let r: CGFloat = hot ? 26 : 18
        let g = CGGradient(colorsSpace: cs, colors: [col.copy(alpha: 0.7)!, col.copy(alpha: 0)!] as CFArray, locations: [0, 1])!
        c.drawRadialGradient(g, startCenter: p, startRadius: 0, endCenter: p, endRadius: r * 3, options: [])
        c.setFillColor(col); c.fillEllipse(in: CGRect(x: p.x - r / 2, y: p.y - r / 2, width: r, height: r))
    } }
    return c.makeImage()!
}

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let build = root.appendingPathComponent("build")
let set = build.appendingPathComponent("AppIcon.iconset")
try? FileManager.default.removeItem(at: set)
try FileManager.default.createDirectory(at: set, withIntermediateDirectories: true)
for base in [16, 32, 128, 256, 512] { for x in [1, 2] {
    let img = render(base * x)
    let name = "icon_\(base)x\(base)" + (x == 2 ? "@2x" : "") + ".png"
    let d = CGImageDestinationCreateWithURL(set.appendingPathComponent(name) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(d, img, nil); CGImageDestinationFinalize(d)
} }
let p = Process()
p.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
p.arguments = ["-c", "icns", set.path, "-o", build.appendingPathComponent("AppIcon.icns").path]
try p.run(); p.waitUntilExit()
print(build.appendingPathComponent("AppIcon.icns").path)
