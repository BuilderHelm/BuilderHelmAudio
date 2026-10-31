// swift-tools-version: 5.9

// Open this folder in Xcode, choose your iPhone and set your team under Signing & Capabilities.
import PackageDescription
import AppleProductTypes

let package = Package(
    name: "BuilderHelmAudioDemo",
    platforms: [.iOS("18.0")],
    products: [
        .iOSApplication(
            name: "BuilderHelm Audio",
            targets: ["AppModule"],
            bundleIdentifier: "ai.builderhelm.audio.demo",
            teamIdentifier: "",
            displayVersion: "0.1",
            bundleVersion: "1",
            appIcon: .placeholder(icon: .mic),
            accentColor: .presetColor(.green),
            supportedDeviceFamilies: [.phone, .pad],
            supportedInterfaceOrientations: [.portrait, .landscapeLeft, .landscapeRight],
            capabilities: [.microphone(purposeString: "Record speech to transcribe it on this device.")]
        ),
    ],
    dependencies: [.package(path: "../..")],
    targets: [
        .executableTarget(
            name: "AppModule",
            dependencies: [.product(name: "BuilderHelmAudio", package: "BuilderHelmAudio")],
            path: "Sources",
            resources: [.copy("Samples")]
        ),
    ]
)
