// Where the model lives and how it gets there: ~/Library/Application Support/NeuroScope/models/<folderName>,
// downloaded from huggingface.co/<repo>/resolve/<revision>/<file> with progress and resume.
// The app ships OLMo 2 1B Instruct (allenai, fully open: weights, data and training code). The engine reads the
// architecture from config.json (model_type "olmo2"). The engine also still opens Qwen3 folders ("qwen3"); the app
// no longer offers or downloads one (Qwen3-0.6B is no longer offered).

import Foundation

struct ModelSpec: Sendable {
    var repo: String
    var folderName: String
    /// The name shown to people ("OLMo 2 1B").
    var displayName: String
    var files: [String]
    /// Approximate total bytes, for the progress bar before Content-Length arrives (and the "3.0 GB" in copy).
    var approxBytes: Int64
    /// Branch or commit the files are fetched from (resolve/<revision>/<file>).
    var revision: String
    /// model.safetensors counts as installed only when LARGER than this (smaller = a partial download).
    var minWeightBytes: Int64
    /// True when the chat template has a thinking mode (scratch paper before the answer). Hides the thinking UI when false.
    var hasThinking: Bool
    /// The shape the engine will report, for copy and layout before it has loaded.
    var geometry: Geometry

    /// "3.0 GB": the one-time download, for copy.
    var downloadText: String { String(format: "%.1f GB", Double(approxBytes) / 1e9) }
}

extension ModelSpec {
    /// OLMo 2 1B Instruct (the final, RLVR stage of allenai's 1B post-training: SFT → DPO → RLVR1 → Instruct),
    /// bf16, one 2,969,854,224-byte model.safetensors (sha256 36d044c7…294f), pinned to the commit verified
    /// against HF transformers on 2026-10-28. No thinking mode: every generated token is an answer token.
    static let olmo2_1B_Instruct = ModelSpec(
        repo: "allenai/OLMo-2-0425-1B-Instruct",
        folderName: "OLMo-2-0425-1B-Instruct",
        displayName: "OLMo 2 1B",
        files: ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "model.safetensors"],
        approxBytes: 2_977_000_000,
        revision: "48d788eca847d4d7548f375ad03d3c9312f6139e",
        minWeightBytes: 2_960_000_000,
        hasThinking: false,
        geometry: .olmo2_1B)
}

enum ModelStore {
    static var spec = ModelSpec.olmo2_1B_Instruct

    static var modelDirectory: URL { directory(for: spec) }

    static func directory(for spec: ModelSpec) -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("NeuroScope/models/\(spec.folderName)", isDirectory: true)
    }

    /// True when every file exists and model.safetensors is non-trivially sized.
    static func isInstalled() -> Bool { isInstalled(spec) }

    static func isInstalled(_ spec: ModelSpec) -> Bool {
        let fm = FileManager.default
        let dir = directory(for: spec)
        for f in spec.files where !fm.fileExists(atPath: dir.appendingPathComponent(f).path) { return false }
        let attrs = try? fm.attributesOfItem(atPath: dir.appendingPathComponent("model.safetensors").path)
        return ((attrs?[.size] as? NSNumber)?.int64Value ?? 0) > spec.minWeightBytes
    }

    /// Downloads missing files; reports overall progress 0…1 on the main actor.
    static func download(progress: @escaping @MainActor (Double) -> Void) async throws {
        try await download(spec, progress: progress)
    }

    static func download(_ spec: ModelSpec, progress: @escaping @MainActor (Double) -> Void) async throws {
        let fm = FileManager.default
        let dir = directory(for: spec)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let missing = spec.files.filter { f in
            let path = dir.appendingPathComponent(f).path
            if !fm.fileExists(atPath: path) { return true }
            if f == "model.safetensors" {
                let sz = ((try? fm.attributesOfItem(atPath: path))?[.size] as? NSNumber)?.int64Value ?? 0
                return sz <= spec.minWeightBytes
            }
            return false
        }
        if missing.isEmpty { await progress(1); return }
        let tracker = DownloadTracker(total: spec.approxBytes, progress: progress)
        for f in missing {
            guard let url = URL(string: "https://huggingface.co/\(spec.repo)/resolve/\(spec.revision)/\(f)") else { continue }
            let dest = dir.appendingPathComponent(f)
            try await tracker.fetch(url: url, to: dest)
        }
        await progress(1)
    }
}

private final class DownloadTracker: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let progress: @MainActor (Double) -> Void
    private var total: Int64
    private var doneBytes: Int64 = 0
    private var lastReport = Date.distantPast
    private var dest: URL?
    private var cont: CheckedContinuation<Void, Error>?
    private var session: URLSession?
    private var moveError: Error?

    init(total: Int64, progress: @escaping @MainActor (Double) -> Void) {
        self.total = total
        self.progress = progress
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    func fetch(url: URL, to dest: URL) async throws {
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
            self.dest = dest
            self.moveError = nil
            self.cont = c
            session!.downloadTask(with: url).resume()
        }
    }

    func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let now = Date()
        guard now.timeIntervalSince(lastReport) >= 0.05 else { return }
        lastReport = now
        let frac = min(0.999, Double(doneBytes + totalBytesWritten) / Double(max(total, 1)))
        let p = progress
        Task { @MainActor in p(frac) }
    }

    func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
            moveError = URLError(.badServerResponse)
            return
        }
        guard let dest else { return }
        do {
            let fm = FileManager.default
            let tmp = dest.appendingPathExtension("part")
            try? fm.removeItem(at: tmp)
            try fm.moveItem(at: location, to: tmp)
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            try fm.moveItem(at: tmp, to: dest)
            let sz = ((try? fm.attributesOfItem(atPath: dest.path))?[.size] as? NSNumber)?.int64Value ?? 0
            if sz > 100_000_000 { total = max(total, 0) }
            doneBytes += sz
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
