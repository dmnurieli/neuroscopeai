// neuroscope.h — C ABI between the Rust engine (libneuroscope_engine.a) and the Swift app.
// Both sides build against this file. Change it only together with engine/src/ffi.rs
// and app/Sources/NeuroScope/Bridge/NeuroEngine.swift.
//
// Threading: ns_generate blocks and must be called from a background thread. Callbacks fire on
// that same thread. ns_cancel is safe from any thread. One generate at a time per engine.
// Memory: every pointer inside an NSTokenFrame is valid ONLY for the duration of the callback —
// copy what you keep. Every char* returned by a function must be released with ns_string_free.

#ifndef NEUROSCOPE_H
#define NEUROSCOPE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NSEngine NSEngine;

typedef struct {
    uint32_t num_layers;    // 28 for Qwen3-0.6B, 16 for OLMo 2 1B
    uint32_t num_neurons;   // MLP neurons per layer (intermediate_size): 3072 / 8192
    uint32_t num_heads;     // attention (query) heads per layer: 16 / 16
    uint32_t hidden_size;   // 1024 / 2048
    uint32_t vocab_size;    // 151936 / 100352 (OLMo's tokenizer has 100278 entries; the rest is padding)
} NSGeometry;

typedef enum {
    NS_STREAM_PROMPT = 0,    // a token of the rendered prompt (prefill)
    NS_STREAM_THINKING = 1,  // generated, inside <think> … </think> (Qwen3 only; OLMo 2 has no thinking mode)
    NS_STREAM_ANSWER = 2,    // generated, after </think> (or with thinking disabled)
    NS_STREAM_SPECIAL = 3    // generated control token: <think>, </think>, <|im_end|>, eos
} NSStream;

typedef struct {
    uint32_t run_id;          // echoed from ns_generate
    uint32_t index;           // position in the full sequence, 0-based (prompt tokens first)
    uint32_t token_id;
    int32_t  stream;          // NSStream
    const char* text;         // UTF-8 text this token contributes (may be "" while a multi-byte char is incomplete)
    const float* neurons;     // num_layers * num_neurons, layer-major: neurons[l*num_neurons + i]
                              // value = silu(gate(x))*up(x) at this position, AFTER edits are applied
    const float* heads;       // num_layers * num_heads: L2 norm of each head's attention output at this position
    const uint32_t* top_ids;  // top_k next-token candidates predicted at this position (descending prob)
    const float* top_probs;   // their probabilities (after temperature, before top-p)
    uint32_t top_k;           // 0 for prompt positions other than the last
    const uint32_t* lens_ids; // logit lens: top-1 token id predicted from the residual after each layer, num_layers entries
    const float* lens_probs;  // its probability; both NULL when the request has "logit_lens": false
    float logprob;            // log-prob the model gave THIS token when it was generated; NaN for prompt tokens
} NSTokenFrame;

typedef void (*NSFrameCallback)(void* user, const NSTokenFrame* frame);

// Opens the model directory (config.json, tokenizer.json, weights: model.safetensors, a sharded
// model.safetensors.index.json, or pytorch_model.bin). The architecture comes from config.json's
// model_type: "qwen3" (default) or "olmo2". Returns NULL on failure
// and, if err_out is non-NULL, sets *err_out to a message (free with ns_string_free).
NSEngine* ns_engine_open(const char* model_dir, char** err_out);
void ns_engine_close(NSEngine* engine);
NSGeometry ns_engine_geometry(const NSEngine* engine);

// Runs one prompt to completion (or cancel / max_tokens). Blocking.
// request_json (its shape is `Request` in engine/src/types.rs):
//   {"prompt":"…","thinking":true,"max_tokens":512,"logit_lens":true,"top_k":8,"pace_ms":0,
//    "sampling":{"mode":"greedy"} | {"mode":"sample","temperature":0.6,"top_p":0.95,"top_k":20,"seed":42},
//    "edits":{"neurons":[{"layer":3,"index":120,"mode":"ablate"|"scale"|"pin","value":0.0}],
//             "heads":[{"layer":5,"head":2,"mode":"ablate"|"scale","value":0.0}],
//             "directions":[{"label":"…","vector":[hidden floats],"sites":[] /*empty = all*/,
//                            "mode":"ablate"|"add","value":0.0}]}}
// Returns a JSON summary string {"ok":true,"tokens_generated":N,"stop":"eos"|"max_tokens"|"cancelled",
// "prefill_ms":…, "decode_tok_per_s":…} or {"ok":false,"error":"…"}. Free with ns_string_free.
char* ns_generate(NSEngine* engine, uint32_t run_id, const char* request_json,
                  NSFrameCallback on_frame, void* user);

void ns_cancel(NSEngine* engine);

// What a neuron writes: its down_proj column pushed through the final norm and the unembedding.
// {"layer":L,"index":i,"promotes":[{"id":…,"token":" Paris","score":…}],"suppresses":[…],"out_norm":…}
// Never blocks: while a generate holds the engine it returns {"error":"busy"} (callers retry after the run).
// Bad layer/index → {"error":"…"}.
char* ns_neuron_info(NSEngine* engine, uint32_t layer, uint32_t index, uint32_t top_k);

// Difference-of-means contrast between two prompt sets (prefill only, no generation). Blocking; waits
// for the engine (does not return busy). request_json: {"a":["…"],"b":["…"],"thinking":false,"edits":{…same shape as ns_generate…}}
// Returns ContrastResult JSON (engine/src/types.rs) or {"error":"…"}. Free with ns_string_free.
char* ns_contrast(NSEngine* engine, const char* request_json);

// Who-triggered-whom at one position of a finished run (direct-path contributions through the residual
// stream). Blocking; waits for the engine. request_json = TraceRequest (engine/src/types.rs):
// {"token_ids":[…],"position":N,"edits":{…},"top_sources":12,"top_targets":12,"focus":[layer,index]|null,"focus_top":24}
// Returns TraceResult JSON or {"error":"…"}. Free with ns_string_free.
char* ns_trace(NSEngine* engine, const char* request_json);

// How much of one direction the memo carries at every site/position of a run, before and after each site's
// edit, plus an optional worked example with real numbers. Blocking; waits for the engine.
// request_json = ProjectRequest; returns ProjectResult JSON or {"error":"…"}. Free with ns_string_free.
char* ns_project(NSEngine* engine, const char* request_json);

// One desk's whole computation at one word: what it reads, the question it asks (gate), its switch (silu),
// what it writes, who hears it (downstream + a greedy path) and its effect on the chosen word.
// Blocking; waits for the engine. request_json = FollowRequest; returns FollowResult JSON or {"error":"…"}.
char* ns_follow(NSEngine* engine, const char* request_json);

// Decodes one token id to its display text (special tokens rendered literally, e.g. "<think>").
char* ns_token_text(NSEngine* engine, uint32_t token_id);

void ns_string_free(char* s);

#ifdef __cplusplus
}
#endif

#endif
