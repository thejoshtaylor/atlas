import Foundation

/// The real transport. It is exercised by the signing spike and the phase UAT,
/// not by `swift test`, because it needs a server.
///
/// It has no session delegate (RESEARCH Pitfall 7). `open` returns before the
/// handshake finishes. When the handshake fails, the first `send` or `receive`
/// throws, and the HTTP status and close code are read off the task then. A
/// token refused before accept shows up as status 403 with close code 0. A live
/// close shows up as a close code. Nothing here logs the request, its headers
/// or the token.
public struct URLSessionWebSocketTransport: WebSocketTransport {
    public init() {}

    public func open(url: URL, bearerToken: String) async throws -> any WebSocketChannel {
        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
        let session = URLSession(configuration: .ephemeral)
        let task = session.webSocketTask(with: request)
        task.resume()
        return URLSessionWebSocketChannel(task: task, session: session)
    }
}

final class URLSessionWebSocketChannel: WebSocketChannel, @unchecked Sendable {
    private let task: URLSessionWebSocketTask
    private let session: URLSession

    init(task: URLSessionWebSocketTask, session: URLSession) {
        self.task = task
        self.session = session
    }

    func send(text: String) async throws {
        do {
            try await task.send(.string(text))
        } catch {
            throw failure()
        }
    }

    func receive() async throws -> String {
        while true {
            do {
                switch try await task.receive() {
                case .string(let text): return text
                case .data: continue
                @unknown default: continue
                }
            } catch {
                throw failure()
            }
        }
    }

    func close(code: Int) async {
        let closeCode = URLSessionWebSocketTask.CloseCode(rawValue: code) ?? .goingAway
        task.cancel(with: closeCode, reason: nil)
        session.finishTasksAndInvalidate()
    }

    /// Reads both signals off the task, then drops the session.
    private func failure() -> DialFailure {
        let status = (task.response as? HTTPURLResponse)?.statusCode
        let result = DialFailure.classify(httpStatus: status, closeCode: task.closeCode.rawValue)
        session.invalidateAndCancel()
        return result
    }
}
