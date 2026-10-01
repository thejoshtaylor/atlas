// swift-tools-version: 6.0
import PackageDescription

// No third-party Swift package: the Mac app uses Apple frameworks only (D-05).
let package = Package(
    name: "AtlasDesktop",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "AtlasKit", targets: ["AtlasKit"]),
        .executable(name: "AtlasDesktop", targets: ["AtlasDesktop"]),
    ],
    targets: [
        .target(name: "AtlasKit"),
        .executableTarget(name: "AtlasDesktop", dependencies: ["AtlasKit"]),
        .testTarget(name: "AtlasKitTests", dependencies: ["AtlasKit"]),
    ],
    swiftLanguageModes: [.v6]
)
