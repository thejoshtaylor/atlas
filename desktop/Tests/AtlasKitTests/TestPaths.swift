import Foundation

/// The `desktop/` directory, found from this file's path:
/// desktop/Tests/AtlasKitTests/TestPaths.swift -> desktop/
func desktopRoot() -> URL {
    URL(filePath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}
