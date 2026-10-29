// swift-tools-version: 6.0
import PackageDescription
import Foundation

let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let rustLib = root.appendingPathComponent("engine/target/release").path

let package = Package(
    name: "NeuroScope",
    platforms: [.macOS(.v15)],
    targets: [
        .target(
            name: "CNeuroScope",
            path: "Sources/CNeuroScope",
            publicHeadersPath: "include",
            linkerSettings: [
                .unsafeFlags(["-L\(rustLib)"]),
                .linkedLibrary("neuroscope_engine"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("Foundation"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("Accelerate"),
                .linkedFramework("CoreGraphics"),
            ]
        ),
        // The tiny teaching network + synthetic flight ledger: pure Swift, no Rust, testable on its own.
        .target(
            name: "AirlineKit",
            path: "Sources/AirlineKit",
            swiftSettings: [.unsafeFlags(["-Ounchecked"], .when(configuration: .release))]
        ),
        .testTarget(
            name: "AirlineKitTests",
            dependencies: ["AirlineKit"],
            path: "Tests/AirlineKitTests"
        ),
        .executableTarget(
            name: "NeuroScope",
            dependencies: ["CNeuroScope", "AirlineKit"],
            path: "Sources/NeuroScope",
            swiftSettings: [.swiftLanguageMode(.v5), .unsafeFlags(["-Ounchecked"], .when(configuration: .release))]
        ),
    ]
)
