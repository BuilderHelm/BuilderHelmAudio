// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "BuilderHelmAudio",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "BuilderHelmAudio", targets: ["BuilderHelmAudio"]),
        .executable(name: "bhaudio", targets: ["bhaudio"]),
    ],
    targets: [
        // CoreML + Accelerate Parakeet runtime, shared with the BuilderHelm model repositories.
        .target(name: "ParakeetRuntime", swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "BuilderHelmAudio", dependencies: ["ParakeetRuntime"]),
        .executableTarget(name: "bhaudio", dependencies: ["BuilderHelmAudio"]),
        .testTarget(name: "BuilderHelmAudioTests", dependencies: ["BuilderHelmAudio", "ParakeetRuntime"], resources: [.copy("Fixtures")]),
    ]
)
