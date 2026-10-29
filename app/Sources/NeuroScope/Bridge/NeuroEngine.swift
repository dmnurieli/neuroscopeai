// Swift wrapper over the C ABI (engine/include/neuroscope.h). Owns the NSEngine handle.
// generate() runs ns_generate on a dedicated background Thread (never the cooperative pool) and
// hops each copied TokenFrame to the main actor via `onFrame`.

import Accelerate
import CNeuroScope
import Foundation

enum EngineError: Error, LocalizedError {
    case open(String), generate(String)
    var errorDescription: String? {
        switch self { case .open(let m): "Could not load model: \(m)"; case .generate(let m): m }
    }
}

private final class GenContext {
    let geometry: Geometry
    let onFrame: @MainActor (TokenFrame) -> Void
    init(geometry: Geometry, onFrame: @escaping @MainActor (TokenFrame) -> Void) {
        self.geometry = geometry; self.onFrame = onFrame
    }
}

private func nsFrameCallback(_ user: UnsafeMutableRawPointer?, _ fp: UnsafePointer<NSTokenFrame>?) {
    guard let user, let fp else { return }
    let ctx = Unmanaged<GenContext>.fromOpaque(user).takeUnretainedValue()
    let f = fp.pointee
    let g = ctx.geometry
    let nN = g.neuronsPerFrame, nH = g.headsPerFrame
    var neurons = [Float16](repeating: 0, count: nN)
    if let src = f.neurons {
        neurons.withUnsafeMutableBufferPointer { dst in
            var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1,
                                  width: vImagePixelCount(nN), rowBytes: nN * 4)
            var d = vImage_Buffer(data: dst.baseAddress, height: 1,
                                  width: vImagePixelCount(nN), rowBytes: nN * 2)
            _ = vImageConvert_PlanarFtoPlanar16F(&s, &d, vImage_Flags(kvImageNoFlags))
        }
    }
    let heads = f.heads.map { Array(UnsafeBufferPointer(start: $0, count: nH)) } ?? [Float](repeating: 0, count: nH)
    var top: [TokenCandidate] = []
    if let ids = f.top_ids, let pr = f.top_probs, f.top_k > 0 {
        top = (0..<Int(f.top_k)).map { TokenCandidate(id: ids[$0], prob: pr[$0]) }
    }
    var lens: [TokenCandidate] = []
    if let ids = f.lens_ids, let pr = f.lens_probs {
        lens = (0..<g.numLayers).map { TokenCandidate(id: ids[$0], prob: pr[$0]) }
    }
    let frame = TokenFrame(index: Int(f.index), tokenID: f.token_id,
                           stream: StreamKind(rawValue: f.stream) ?? .special,
                           text: f.text.map { String(cString: $0) } ?? "",
                           neurons: neurons, heads: heads, top: top, lens: lens, logprob: f.logprob)
    let cb = ctx.onFrame
    DispatchQueue.main.async { MainActor.assumeIsolated { cb(frame) } }
}

private func takeString(_ p: UnsafeMutablePointer<CChar>?) -> String? {
    guard let p else { return nil }
    defer { ns_string_free(p) }
    return String(cString: p)
}

final class NeuroEngine: @unchecked Sendable {
    let geometry: Geometry
    private let handle: OpaquePointer
    private let lock = NSLock()
    private var memo: [UInt32: String] = [:]

    /// Loads the model (seconds; call off the main thread).
    init(modelDir: URL) throws {
        var err: UnsafeMutablePointer<CChar>? = nil
        guard let h = ns_engine_open(modelDir.path, &err) else {
            throw EngineError.open(takeString(err) ?? "unknown error")
        }
        if let err { ns_string_free(err) }
        handle = h
        let g = ns_engine_geometry(h)
        geometry = Geometry(numLayers: Int(g.num_layers), numNeurons: Int(g.num_neurons),
                            numHeads: Int(g.num_heads), hiddenSize: Int(g.hidden_size),
                            vocabSize: Int(g.vocab_size))
    }

    deinit { ns_engine_close(handle) }

    /// Runs one generation. `onFrame` is invoked on the MAIN actor, in order, once per token.
    /// Copies neurons to Float16 inside the C callback (the pointers die when it returns).
    func generate(runID: UInt32, request: GenerationRequest,
                  onFrame: @escaping @MainActor (TokenFrame) -> Void) async throws -> GenerationSummary {
        let json = String(decoding: try JSONEncoder().encode(request), as: UTF8.self)
        let ctx = GenContext(geometry: geometry, onFrame: onFrame)
        let h = handle
        let out: String? = await withCheckedContinuation { (cont: CheckedContinuation<String?, Never>) in
            let t = Thread {
                let box = Unmanaged.passRetained(ctx)
                let r = ns_generate(h, runID, json, nsFrameCallback, box.toOpaque())
                box.release()
                cont.resume(returning: takeString(r))
            }
            t.stackSize = 8 << 20
            t.qualityOfService = .userInitiated
            t.start()
        }
        guard let out, let data = out.data(using: .utf8) else { throw EngineError.generate("no summary") }
        let summary: GenerationSummary
        do { summary = try JSONDecoder().decode(GenerationSummary.self, from: data) }
        catch { throw EngineError.generate("bad summary: \(out)") }
        if !summary.ok { throw EngineError.generate(summary.error ?? "generation failed") }
        return summary
    }

    /// Thread-safe; makes the running generate() stop after the current token.
    func cancel() { ns_cancel(handle) }

    /// Direct logit attribution for one neuron (cheap: one matvec on the GPU). nil on failure.
    func neuronInfo(_ n: NeuronID, topK: Int = 12) -> NeuronInfo? {
        guard let s = takeString(ns_neuron_info(handle, UInt32(n.layer), UInt32(n.index), UInt32(topK))),
              let d = s.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(NeuronInfo.self, from: d)
    }

    /// Display text of a token id (memoized).
    func tokenText(_ id: UInt32) -> String {
        lock.lock()
        if let t = memo[id] { lock.unlock(); return t }
        lock.unlock()
        let t = takeString(ns_token_text(handle, id)) ?? ""
        lock.lock(); memo[id] = t; lock.unlock()
        return t
    }

    // MARK: contrast (ns_contrast) — blocking FFI, run on a dedicated Thread like generate().
    func contrast(_ request: ContrastRequest) async throws -> ContrastResult {
        let json = String(decoding: try JSONEncoder().encode(request), as: UTF8.self)
        let h = handle
        let out: String? = await withCheckedContinuation { (cont: CheckedContinuation<String?, Never>) in
            let t = Thread {
                cont.resume(returning: takeString(ns_contrast(h, json)))
            }
            t.stackSize = 8 << 20
            t.qualityOfService = .userInitiated
            t.start()
        }
        guard let out, let data = out.data(using: .utf8) else { throw EngineError.generate("no contrast result") }
        struct Err: Decodable { let error: String }
        if let e = try? JSONDecoder().decode(Err.self, from: data) { throw EngineError.generate(e.error) }
        do { return try JSONDecoder().decode(ContrastResult.self, from: data) }
        catch { throw EngineError.generate("bad contrast result: \(error.localizedDescription)") }
    }

    // MARK: trace (ns_trace) — blocking FFI, run on a dedicated Thread like contrast().
    func trace(_ request: TraceRequest) async throws -> TraceResult {
        let json = String(decoding: try JSONEncoder().encode(request), as: UTF8.self)
        let h = handle
        let out: String? = await withCheckedContinuation { (cont: CheckedContinuation<String?, Never>) in
            let t = Thread {
                cont.resume(returning: takeString(ns_trace(h, json)))
            }
            t.stackSize = 8 << 20
            t.qualityOfService = .userInitiated
            t.start()
        }
        guard let out, let data = out.data(using: .utf8) else { throw EngineError.generate("no trace result") }
        struct Err: Decodable { let error: String }
        if let e = try? JSONDecoder().decode(Err.self, from: data) { throw EngineError.generate(e.error) }
        do { return try JSONDecoder().decode(TraceResult.self, from: data) }
        catch { throw EngineError.generate("bad trace result: \(error.localizedDescription)") }
    }

    // MARK: project (ns_project) — blocking FFI on a dedicated Thread like trace().
    func project(_ request: ProjectRequest) async throws -> ProjectResult {
        let json = String(decoding: try JSONEncoder().encode(request), as: UTF8.self)
        let h = handle
        let out: String? = await withCheckedContinuation { (cont: CheckedContinuation<String?, Never>) in
            let t = Thread {
                cont.resume(returning: takeString(ns_project(h, json)))
            }
            t.stackSize = 8 << 20
            t.qualityOfService = .userInitiated
            t.start()
        }
        guard let out, let data = out.data(using: .utf8) else { throw EngineError.generate("no project result") }
        struct Err: Decodable { let error: String }
        if let e = try? JSONDecoder().decode(Err.self, from: data) { throw EngineError.generate(e.error) }
        do { return try JSONDecoder().decode(ProjectResult.self, from: data) }
        catch { throw EngineError.generate("bad project result: \(error.localizedDescription)") }
    }

    // MARK: follow (ns_follow) — blocking FFI on a dedicated Thread like trace().
    func follow(_ request: FollowRequest) async throws -> FollowResult {
        let json = String(decoding: try JSONEncoder().encode(request), as: UTF8.self)
        let h = handle
        let out: String? = await withCheckedContinuation { (cont: CheckedContinuation<String?, Never>) in
            let t = Thread {
                cont.resume(returning: takeString(ns_follow(h, json)))
            }
            t.stackSize = 8 << 20
            t.qualityOfService = .userInitiated
            t.start()
        }
        guard let out, let data = out.data(using: .utf8) else { throw EngineError.generate("no follow result") }
        struct Err: Decodable { let error: String }
        if let e = try? JSONDecoder().decode(Err.self, from: data) { throw EngineError.generate(e.error) }
        do { return try JSONDecoder().decode(FollowResult.self, from: data) }
        catch { throw EngineError.generate("bad follow result: \(error.localizedDescription)") }
    }
}
