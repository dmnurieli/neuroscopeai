//! Shared engine types.
//! JSON shapes here are the wire format of neuroscope.h (serde field names = JSON keys).

use serde::{Deserialize, Serialize};

/// Model geometry, mirrored 1:1 by `NSGeometry` in neuroscope.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Geometry {
    pub num_layers: usize,
    pub num_neurons: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl Geometry {
    /// Length of one frame's `neurons` vector (layer-major).
    pub fn neurons_per_frame(&self) -> usize {
        self.num_layers * self.num_neurons
    }
    /// Length of one frame's `heads` vector (layer-major).
    pub fn heads_per_frame(&self) -> usize {
        self.num_layers * self.num_heads
    }
}

/// Which stream a token belongs to; discriminants equal `NSStream`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i32)]
pub enum StreamKind {
    Prompt = 0,
    Thinking = 1,
    Answer = 2,
    Special = 3,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum Sampling {
    /// argmax — fully deterministic; the default for replays.
    Greedy,
    /// Seeded sampling: same seed + same edits ⇒ same tokens.
    Sample {
        #[serde(default = "default_temperature")]
        temperature: f32,
        #[serde(default = "default_top_p")]
        top_p: f32,
        #[serde(default = "default_sample_top_k")]
        top_k: usize,
        #[serde(default)]
        seed: u64,
    },
}

fn default_temperature() -> f32 {
    0.6
}
fn default_top_p() -> f32 {
    0.95
}
fn default_sample_top_k() -> usize {
    20
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling::Greedy
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EditMode {
    /// Delete the unit: its activation becomes 0.
    Ablate,
    /// Multiply the activation by `value` (2.0 doubles, -1.0 inverts).
    Scale,
    /// Pin the activation to the constant `value` regardless of input. Neurons only.
    Pin,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuronEdit {
    pub layer: usize,
    pub index: usize,
    pub mode: EditMode,
    #[serde(default)]
    pub value: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadEdit {
    pub layer: usize,
    pub head: usize,
    /// Ablate or Scale only; Pin on a head is rejected by validation.
    pub mode: EditMode,
    #[serde(default)]
    pub value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DirectionMode {
    /// Directional ablation ("abliteration"): x ← x − (x·d̂)d̂ — the residual stream can no longer
    /// represent the direction. Several ablated directions are orthonormalized and removed together.
    Ablate,
    /// Activation steering: x ← x + value·d̂.
    Add,
}

/// An edit to the residual stream along a direction (hidden_size floats; the engine normalizes it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionEdit {
    #[serde(default)]
    pub label: String,
    pub vector: Vec<f32>,
    /// Residual sites to edit: 0 = the embedding output, l+1 = the output of layer l.
    /// Empty = every site (0..=num_layers) — the standard abliteration setting.
    #[serde(default)]
    pub sites: Vec<usize>,
    pub mode: DirectionMode,
    #[serde(default)]
    pub value: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EditSet {
    #[serde(default)]
    pub neurons: Vec<NeuronEdit>,
    #[serde(default)]
    pub heads: Vec<HeadEdit>,
    #[serde(default)]
    pub directions: Vec<DirectionEdit>,
}

impl EditSet {
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty() && self.heads.is_empty() && self.directions.is_empty()
    }
}

/// The request_json of ns_contrast: two prompt sets that differ in one concept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContrastRequest {
    pub a: Vec<String>,
    pub b: Vec<String>,
    /// Render prompts with the chat template in thinking mode (default false = no-think template).
    #[serde(default)]
    pub thinking: bool,
    /// Edits applied during the contrast forward passes (e.g. a candidate ablation, to measure how much of
    /// the concept survives downstream). Default: none.
    #[serde(default)]
    pub edits: EditSet,
}

/// Difference-of-means analysis at the LAST prompt token (the position right before the model answers).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContrastResult {
    pub num_layers: usize,
    pub hidden_size: usize,
    /// num_layers+1 sites (0 = embedding output, l+1 = after layer l), each hidden_size floats:
    /// mean residual over A minus mean over B (NOT normalized).
    pub directions: Vec<Vec<f32>>,
    /// Per site: Cohen's d of the projections onto the unit direction (how cleanly it separates A from B).
    pub separation: Vec<f32>,
    /// Per site: L2 norm of the mean difference.
    pub norms: Vec<f32>,
    /// num_layers * num_neurons, layer-major: mean MLP activation over A minus over B.
    pub neuron_diff: Vec<f32>,
    pub count_a: usize,
    pub count_b: usize,
}

/// The `request_json` of `ns_generate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub prompt: String,
    #[serde(default = "default_true")]
    pub thinking: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_true")]
    pub logit_lens: bool,
    /// How many next-token candidates each frame reports.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Sleep this long after each generated token (slow-motion viewing). 0 = full speed.
    #[serde(default)]
    pub pace_ms: u64,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub edits: EditSet,
}

fn default_true() -> bool {
    true
}
fn default_max_tokens() -> usize {
    512
}
fn default_top_k() -> usize {
    8
}

/// One position's worth of data. Owned Rust-side twin of `NSTokenFrame`.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub index: usize,
    pub token_id: u32,
    pub stream: StreamKind,
    pub text: String,
    /// geometry.neurons_per_frame() values, layer-major, after edits.
    pub neurons: Vec<f32>,
    /// geometry.heads_per_frame() values.
    pub heads: Vec<f32>,
    /// (token_id, prob) descending; empty for prompt positions except the last.
    pub top: Vec<(u32, f32)>,
    /// num_layers entries when logit lens is on, else empty.
    pub lens: Vec<(u32, f32)>,
    /// NaN for prompt tokens.
    pub logprob: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Eos,
    MaxTokens,
    Cancelled,
}

/// The JSON returned by `ns_generate` when ok.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub ok: bool,
    pub tokens_generated: usize,
    pub stop: StopReason,
    pub prefill_ms: f64,
    pub decode_tok_per_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenScore {
    pub id: u32,
    pub token: String,
    pub score: f32,
}

/// The JSON returned by `ns_neuron_info`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuronInfo {
    pub layer: usize,
    pub index: usize,
    pub promotes: Vec<TokenScore>,
    pub suppresses: Vec<TokenScore>,
    /// L2 norm of the neuron's down_proj column (how loudly it can write).
    pub out_norm: f32,
}

/// Qwen3 special token ids (from tokenizer.json added_tokens; verified at load by tokenize.rs).
/// OLMo 2 ids are looked up by name at load instead (`Tok::stop_ids` / `special_ids`: <|endoftext|> 100257).
pub mod special {
    pub const ENDOFTEXT: u32 = 151643;
    pub const IM_START: u32 = 151644;
    pub const IM_END: u32 = 151645;
    pub const THINK: u32 = 151667;
    pub const THINK_END: u32 = 151668;
}

/// The request_json of ns_trace: replay ONE position of a finished run and measure who triggered whom.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceRequest {
    /// The run's full token ids (prompt incl. chat template, then generated), exactly as the frames reported them.
    pub token_ids: Vec<u32>,
    /// The position to explain (0-based index into token_ids); the engine prefills token_ids[..=position].
    pub position: usize,
    /// The edits the run was generated with (so a trace of an edited run explains the edited model).
    #[serde(default)]
    pub edits: EditSet,
    /// Per floor-to-next-floor step: how many of the busiest source desks and target desks to connect.
    #[serde(default = "default_trace_top")]
    pub top_sources: usize,
    #[serde(default = "default_trace_top")]
    pub top_targets: usize,
    /// Optional desk to explain: also return its strongest upstream contributors from EVERY earlier floor.
    #[serde(default)]
    pub focus: Option<(usize, usize)>,
    #[serde(default = "default_focus_top")]
    pub focus_top: usize,
}

fn default_trace_top() -> usize {
    12
}
fn default_focus_top() -> usize {
    24
}

/// One arrow: source desk (from_layer, from_index) → target desk (to_layer, to_index).
/// weight = the source's DIRECT-PATH contribution to the target's gate pre-activation at this position:
///   a_src · (w_norm[to] ⊙ W_gate[to][to_index]) · (w_out[from] ⊙ W_down[from][:, from_index]) · inv_rms[to] · ws[from]
/// Qwen3: w_out = 1, ws = 1 (pre-norm). OLMo 2: w_norm = 1, inv_rms = 1 (the MLP reads the raw residual),
/// w_out = post_feedforward_layernorm weight, ws = 1/rms of floor `from`'s MLP output at this position.
/// Positive = pushed the target to fire, negative = held it back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEdge {
    pub from_layer: usize,
    pub from_index: usize,
    pub to_layer: usize,
    pub to_index: usize,
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceResult {
    pub position: usize,
    pub token_id: u32,
    /// Floor l → floor l+1 arrows for every l, strongest |weight| first within each step.
    pub edges: Vec<TraceEdge>,
    /// Arrows from any earlier floor into `focus` (empty when no focus).
    pub focus_edges: Vec<TraceEdge>,
    /// Per target floor l (index = l, 0 for floor 0): share of the shown targets' total |gate pre-activation|
    /// that the shown arrows explain (0…1). The rest arrives via attention, earlier floors, the word itself.
    pub explained: Vec<f32>,
    /// Real time one forward pass of this position took, in ms (for "N× slower than real" labels).
    pub forward_ms: f64,
}

/// The request_json of ns_project: measure how much of one direction ("highlighter") the memo carries at
/// every site and position of a run, before and after that site's edits — the numbers the math panel teaches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectRequest {
    pub token_ids: Vec<u32>,
    /// The run's edits (the projections describe the run as it was generated).
    #[serde(default)]
    pub edits: EditSet,
    /// The direction to measure along (hidden_size floats; the engine normalizes it to d̂).
    pub direction: Vec<f32>,
    /// Optional worked example: return the actual memo numbers at (site, position).
    #[serde(default)]
    pub sample_site: Option<usize>,
    #[serde(default)]
    pub sample_position: Option<usize>,
    /// How many leading numbers of x, d̂ and x_new to return for the worked example.
    #[serde(default = "default_sample_dims")]
    pub sample_dims: usize,
}

fn default_sample_dims() -> usize {
    6
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectSample {
    pub site: usize,
    pub position: usize,
    /// x · d̂ of the memo arriving at the site (before its edit).
    pub dot: f32,
    /// x · d̂ after the site's edit (0 when the direction is ablated there).
    pub dot_after: f32,
    /// First sample_dims numbers of x (before), d̂, and x_new (after the site's edit).
    pub x: Vec<f32>,
    pub d_hat: Vec<f32>,
    pub x_new: Vec<f32>,
    /// |x| before and after, and |d| (the unnormalized direction's length).
    pub x_norm: f32,
    pub x_new_norm: f32,
    pub d_norm: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectResult {
    pub sites: usize,      // num_layers + 1
    pub positions: usize,  // token_ids.len()
    /// positions × sites, position-major: x·d̂ arriving at each site (before that site's edit).
    pub before: Vec<f32>,
    /// positions × sites: x·d̂ after that site's edit.
    pub after: Vec<f32>,
    pub sample: Option<ProjectSample>,
}

/// The request_json of ns_follow: one desk's whole story at one word of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FollowRequest {
    pub token_ids: Vec<u32>,
    pub position: usize,
    #[serde(default)]
    pub edits: EditSet,
    pub layer: usize,
    pub index: usize,
    /// How many upstream contributors, downstream listeners and words to report.
    #[serde(default = "default_follow_top")]
    pub top: usize,
    /// Hops of the greedy path (strongest listener, then its strongest listener, …).
    #[serde(default = "default_follow_hops")]
    pub hops: usize,
}

fn default_follow_top() -> usize {
    8
}
fn default_follow_hops() -> usize {
    5
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordScore {
    pub id: u32,
    pub token: String,
    pub score: f32,
}

/// Every number of one desk's computation at one word, in the order it happens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FollowResult {
    pub layer: usize,
    pub index: usize,
    pub position: usize,
    pub token_id: u32,
    // 1. Reads the memo: x̃ = x / rms(x) ⊙ w_norm (the MLP-input RMSNorm of this floor; OLMo 2 has none: x̃ = x)
    pub memo_norm: f32,          // |x| of the memo entering this floor's MLP
    pub rms: f32,                // rms(x) (Qwen3); exactly 1 for OLMo 2
    // gate / up / wakes_on below use w_norm = 1 for OLMo 2.
    // 2–4. Asks, switches, sizes:
    pub gate: f32,               // W_gate[j] · x̃
    pub sigma: f32,              // σ(gate) = 1 / (1 + e^−gate)
    pub silu: f32,               // gate × σ(gate)
    pub up: f32,                 // W_up[j] · x̃
    pub h_raw: f32,              // silu × up, before any edit
    pub h: f32,                  // after this run's edits (what was actually written)
    /// Words whose (normalized) embedding best matches the gate row: what wakes this desk up (static).
    pub wakes_on: Vec<WordScore>,
    /// Strongest direct-path contributors to `gate` from desks on earlier floors at this word.
    pub asked_by: Vec<TraceEdge>,
    /// Share of |gate| the listed contributors explain (the rest: attention, the word itself, other desks).
    pub asked_by_explained: f32,
    // 5. Writes: memo += h × c, c = W_down[:, j] (Qwen3) or (w_post_ff ⊙ W_down[:, j]) / rms(MLP output) (OLMo 2,
    //    the post_feedforward_layernorm at its actual scale for this word)
    pub column_norm: f32,        // |c| (Qwen3: |W_down[:, j]|)
    pub write_norm: f32,         // |h| × column_norm
    // 6. Who hears it: strongest direct-path pushes on later floors' desks' gates.
    pub heard_by: Vec<TraceEdge>,
    /// Greedy path: this desk → its strongest listener → that desk's strongest listener …
    pub path: Vec<TraceEdge>,
    // 7. Effect on the final word (direct path through the final norm and unembedding):
    pub next_token: u32,         // the word the model chose after this position
    pub next_effect: f32,        // this desk's direct contribution to that word's logit
    /// Its direct contribution to the top candidate words' logits (pushes toward / away).
    pub word_effects: Vec<WordScore>,
}
