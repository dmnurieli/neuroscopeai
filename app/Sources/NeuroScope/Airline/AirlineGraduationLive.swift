import Foundation
import AirlineKit

// Lesson 9's live page: the learner's own question to OLMo 2 1B on this Mac, through the app's engine (its own
// handle, opened only on that page and closed when the lesson ends). The final model (Instruct) is what the
// "Inside a real AI" lessons use; the base model is an optional 5.9 GB download the learner can delete again.

/// A model folder the graduation lesson can use: where it lives, which pinned files make it complete.
struct GraduationModel: Sendable, Hashable {
    let id: String
    let name: String
    let repo: String
    let commit: String
    let folderName: String
    /// (file, exact bytes at `commit`)
    let files: [(String, Int64)]

    static func == (a: GraduationModel, b: GraduationModel) -> Bool { a.id == b.id }
    func hash(into h: inout Hasher) { h.combine(id) }

    var totalBytes: Int64 { files.reduce(0) { $0 + $1.1 } }
    var sizeWords: String { ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .decimal) }
    var directory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("NeuroScope/models/\(folderName)", isDirectory: true)
    }
    /// Every pinned file is there at its exact size.
    var isInstalled: Bool {
        let fm = FileManager.default
        for (f, size) in files {
            let path = directory.appendingPathComponent(f).path
            guard let n = (try? fm.attributesOfItem(atPath: path))?[.size] as? NSNumber, n.int64Value == size else { return false }
        }
        return true
    }
    /// "live on this Mac from allenai/OLMo-2-0425-1B-Instruct @48d788ec"
    var provenance: String { "live on this Mac from \(repo) @\(commit.prefix(8))" }

    /// The final model (the "Inside a real AI" lessons run on it). Pinned to the commit the engine was verified on.
    static let final = GraduationModel(
        id: "instruct", name: "Final model", repo: "allenai/OLMo-2-0425-1B-Instruct",
        commit: "48d788eca847d4d7548f375ad03d3c9312f6139e", folderName: "OLMo-2-0425-1B-Instruct",
        files: [("config.json", 625), ("generation_config.json", 121), ("tokenizer.json", 7_137_177),
                ("tokenizer_config.json", 4_884), ("special_tokens_map.json", 581), ("model.safetensors", 2_969_854_224)])

    /// The base model (finished pretraining, no flight school), F32 in two shards.
    static let base = GraduationModel(
        id: "base", name: "Base model", repo: "allenai/OLMo-2-0425-1B",
        commit: "a1847dff35000b4271fa70afc5db10fd29fedbdf", folderName: "OLMo-2-0425-1B",
        files: [("config.json", 623), ("tokenizer.json", 7_137_177), ("tokenizer_config.json", 4_345),
                ("special_tokens_map.json", 125), ("model.safetensors.index.json", 14_873),
                ("model-00001-of-00002.safetensors", 4_983_360_992), ("model-00002-of-00002.safetensors", 956_326_560)])
}

/// Downloads one GraduationModel's pinned files with progress; cancel removes what it fetched.
@MainActor @Observable
final class GraduationDownload {
    enum State: Equatable { case idle, downloading(Double), failed(String) }
    let model: GraduationModel
    private(set) var state: State = .idle
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var fetcher: FileFetcher?

    init(model: GraduationModel) { self.model = model }

    var isDownloading: Bool { if case .downloading = state { return true }; return false }

    func start(onDone: @escaping @MainActor () -> Void) {
        guard !isDownloading else { return }
        let m = model
        state = .downloading(0)
        let f = FileFetcher(total: m.totalBytes) { [weak self] p in self?.state = .downloading(p) }
        fetcher = f
        task = Task { [weak self] in
            do {
                try FileManager.default.createDirectory(at: m.directory, withIntermediateDirectories: true)
                for (file, size) in m.files {
                    let dest = m.directory.appendingPathComponent(file)
                    if let n = (try? FileManager.default.attributesOfItem(atPath: dest.path))?[.size] as? NSNumber,
                       n.int64Value == size { f.skip(size); continue }
                    guard let url = URL(string: "https://huggingface.co/\(m.repo)/resolve/\(m.commit)/\(file)") else { continue }
                    try await f.fetch(url: url, to: dest, expected: size)
                    try Task.checkCancellation()
                }
                guard let self else { return }
                self.state = .idle
                onDone()
            } catch {
                guard let self else { return }
                if Task.isCancelled || (error as? URLError)?.code == .cancelled {
                    self.state = .idle
                } else {
                    self.state = .failed(error.localizedDescription)
                }
            }
        }
    }

    func cancel() {
        task?.cancel()
        fetcher?.cancel()
        task = nil
        try? FileManager.default.removeItem(at: model.directory)
        state = .idle
    }

    /// Removes the model folder (the learner's "Delete" button).
    func delete() {
        cancel()
    }
}

/// URLSession download with byte progress across several files.
private final class FileFetcher: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let total: Int64
    private let report: @MainActor (Double) -> Void
    private var done: Int64 = 0
    private var last = Date.distantPast
    private var dest: URL?
    private var expected: Int64 = 0
    private var cont: CheckedContinuation<Void, Error>?
    private var moveError: Error?
    private var session: URLSession!
    private var current: URLSessionDownloadTask?

    init(total: Int64, report: @escaping @MainActor (Double) -> Void) {
        self.total = max(total, 1)
        self.report = report
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    func skip(_ bytes: Int64) { done += bytes }

    func cancel() { current?.cancel(); session.invalidateAndCancel() }

    func fetch(url: URL, to dest: URL, expected: Int64) async throws {
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
            self.dest = dest
            self.expected = expected
            self.moveError = nil
            self.cont = c
            let t = session.downloadTask(with: url)
            current = t
            t.resume()
        }
    }

    func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let now = Date()
        guard now.timeIntervalSince(last) >= 0.1 else { return }
        last = now
        let frac = min(0.999, Double(done + totalBytesWritten) / Double(total))
        let r = report
        Task { @MainActor in r(frac) }
    }

    func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
            moveError = URLError(.badServerResponse); return
        }
        guard let dest else { return }
        do {
            let fm = FileManager.default
            let size = ((try? fm.attributesOfItem(atPath: location.path))?[.size] as? NSNumber)?.int64Value ?? -1
            guard size == expected else { throw URLError(.cannotDecodeContentData) }
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            try fm.moveItem(at: location, to: dest)
            done += size
        } catch { moveError = error }
    }

    func urlSession(_ s: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let c = cont
        cont = nil
        if let error { c?.resume(throwing: error) }
        else if let moveError { c?.resume(throwing: moveError) }
        else { c?.resume() }
    }
}

/// The live question: opens the chosen model's engine on demand and streams its answer.
@MainActor @Observable
final class GraduationLive {
    enum Phase: Equatable { case idle, loading, answering, done, failed(String) }
    private(set) var phase: Phase = .idle
    private(set) var answer = ""
    private(set) var askedText: String?
    private(set) var askedModel: GraduationModel?
    private(set) var tokPerS: Double?
    private(set) var cutOff = false
    var question = ""
    var choice: GraduationModel = .final

    @ObservationIgnored private var engines: [String: NeuroEngine] = [:]
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var runID: UInt32 = 900_000
    static let maxTokens = 400

    var isBusy: Bool { phase == .loading || phase == .answering }
    var trimmedQuestion: String { question.trimmingCharacters(in: .whitespacesAndNewlines) }
    /// The field holds a question that has not been asked yet (of the chosen model).
    var hasNewQuestion: Bool {
        !trimmedQuestion.isEmpty && (trimmedQuestion != askedText || choice != askedModel)
    }

    func ask() {
        let q = trimmedQuestion
        guard !q.isEmpty, !isBusy, choice.isInstalled else { return }
        let model = choice
        askedText = q
        askedModel = model
        answer = ""
        tokPerS = nil
        cutOff = false
        runID &+= 1
        let id = runID
        task = Task { [weak self] in
            guard let self else { return }
            do {
                let engine: NeuroEngine
                if let e = self.engines[model.id] { engine = e } else {
                    self.phase = .loading
                    let dir = model.directory
                    engine = try await Task.detached(priority: .userInitiated) { try NeuroEngine(modelDir: dir) }.value
                    self.engines[model.id] = engine
                }
                guard !Task.isCancelled, self.runID == id else { return }
                self.phase = .answering
                let req = GenerationRequest(prompt: q, thinking: false, maxTokens: Self.maxTokens, logitLens: false, topK: 1)
                let summary = try await engine.generate(runID: id, request: req) { [weak self] frame in
                    guard let self, self.runID == id, frame.stream == .answer || frame.stream == .thinking else { return }
                    self.answer += frame.text
                }
                guard self.runID == id else { return }
                self.tokPerS = summary.decodeTokPerS
                self.cutOff = summary.stop == "max_tokens"
                self.phase = .done
            } catch {
                guard self.runID == id else { return }
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        for e in engines.values { e.cancel() }
        task?.cancel()
        runID &+= 1
        if isBusy { phase = answer.isEmpty ? .idle : .done }
    }

    /// Closes an engine (before its folder is deleted) or all of them (leaving the lesson).
    func release(_ model: GraduationModel? = nil) {
        cancel()
        if let m = model { engines[m.id] = nil } else { engines.removeAll() }
    }
}
