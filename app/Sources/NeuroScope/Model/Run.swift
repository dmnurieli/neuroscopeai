// One generation: its request, every token, and every token's activations in flat buffers.
// Main-actor owned; frames arrive from the engine thread and are appended on the main actor.

import Foundation
import Observation

@Observable
final class Run: Identifiable {
    enum Status: Equatable { case running, finished, cancelled, failed(String) }

    let id: UInt32
    var label: String
    let request: GenerationRequest
    let geometry: Geometry
    var status: Status = .running
    var summary: GenerationSummary?

    private(set) var tokens: [TokenRecord] = []
    /// tokens.count * geometry.neuronsPerFrame, token-major then layer-major.
    @ObservationIgnored private(set) var neurons = ContiguousArray<Float16>()
    /// tokens.count * geometry.headsPerFrame.
    @ObservationIgnored private(set) var heads = ContiguousArray<Float>()

    init(id: UInt32, label: String, request: GenerationRequest, geometry: Geometry) {
        self.id = id
        self.label = label
        self.request = request
        self.geometry = geometry
        neurons.reserveCapacity(geometry.neuronsPerFrame * 640)
    }

    var count: Int { tokens.count }
    var promptCount: Int { tokens.prefix { $0.stream == .prompt }.count }
    var edits: EditSet { request.edits }

    func append(_ f: TokenFrame) {
        precondition(f.neurons.count == geometry.neuronsPerFrame && f.heads.count == geometry.headsPerFrame)
        neurons.append(contentsOf: f.neurons)
        heads.append(contentsOf: f.heads)
        tokens.append(TokenRecord(index: f.index, tokenID: f.tokenID, stream: f.stream, text: f.text,
                                  top: f.top, lens: f.lens, logprob: f.logprob))
    }

    /// Activation of one neuron at token t.
    func activation(_ n: NeuronID, at t: Int) -> Float {
        Float(neurons[t * geometry.neuronsPerFrame + n.flat(geometry)])
    }

    func headNorm(_ h: HeadID, at t: Int) -> Float {
        heads[t * geometry.headsPerFrame + h.flat(geometry)]
    }

    /// Zero-copy view of token t's neuron vector (layer-major). Valid until the next append.
    func withNeurons<R>(at t: Int, _ body: (UnsafeBufferPointer<Float16>) -> R) -> R {
        let n = geometry.neuronsPerFrame
        return neurons.withUnsafeBufferPointer { body(UnsafeBufferPointer(rebasing: $0[(t * n)..<((t + 1) * n)])) }
    }

    func withHeads<R>(at t: Int, _ body: (UnsafeBufferPointer<Float>) -> R) -> R {
        let n = geometry.headsPerFrame
        return heads.withUnsafeBufferPointer { body(UnsafeBufferPointer(rebasing: $0[(t * n)..<((t + 1) * n)])) }
    }

    /// One neuron's activation at every token (for sparklines).
    func trace(_ n: NeuronID) -> [Float] {
        (0..<count).map { activation(n, at: $0) }
    }

    func text(of stream: StreamKind) -> String {
        tokens.lazy.filter { $0.stream == stream }.map(\.text).joined()
    }
}
