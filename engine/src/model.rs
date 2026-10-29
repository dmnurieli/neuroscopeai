//! Decoder (Qwen3 or OLMo 2, see `config::Arch`) with activation capture and edits. Weights:
//! model.safetensors, a sharded model.safetensors.index.json, or pytorch_model.bin, loaded through
//! candle_nn::VarBuilder (mmap for safetensors). Runs in bf16 on Metal; captured values are returned as f32.
//!
//! Per layer, the two families differ only in where the RMSNorms sit:
//!   Qwen3:  x += o(attn(norm_in(x)))           x += down(h(norm_mlp(x)))
//!   OLMo 2: x += norm_attn_out(o(attn(x)))     x += norm_mlp_out(down(h(x)))
//! and in q_norm/k_norm (per head for Qwen3, over the whole projection for OLMo 2). Everything the
//! analyses need is expressed with a per-layer READ norm weight + scale (what the MLP input is multiplied
//! by: Qwen3 w_norm and 1/rms(x); OLMo 2 ones and 1) and a WRITE norm weight + scale (what a neuron's
//! down_proj output is multiplied by on its way into the residual: Qwen3 ones and 1; OLMo 2 w_post_ff
//! and 1/rms(mlp_out)), so trace/follow/neuron_info use one formula for both.

use crate::config::{Arch, ModelConfig};
use crate::edits::{DeviceEdits, ResidEdit};
use crate::types::Geometry;
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::{Linear, VarBuilder, linear_no_bias};
use std::path::Path;

/// What one forward call recorded, for every position it processed.
pub struct Capture {
    /// seq_len * num_layers * num_neurons, position-major then layer-major:
    /// neurons[(p * L + l) * N + i] — the SwiGLU activation silu(gate(x))*up(x) after edits.
    pub neurons: Vec<f32>,
    /// seq_len * num_layers * num_heads: L2 norm of each head's output (after head edits), same ordering.
    pub heads: Vec<f32>,
    /// When requested: [num_layers, hidden_size] residual stream after each layer at the LAST position.
    pub residuals: Option<Tensor>,
    /// When residuals were requested: the logit lens of them (per layer top-1 (id, prob)), read back
    /// in the same device→host sync as the logits so callers need not run `lens::logit_lens` again.
    pub(crate) lens: Option<Vec<(u32, f32)>>,
    /// When residuals were requested: [1, hidden_size] residual site 0 (embedding output, after
    /// site-0 direction edits) at the LAST position. Used by contrast.rs.
    pub(crate) embed_last: Option<Tensor>,
}

/// Last-position readout of one trace prefill (see `Model::forward_trace`), host f32.
pub(crate) struct TraceCapture {
    /// num_layers * num_neurons, layer-major: post-edit SwiGLU activations.
    pub act: Vec<f32>,
    /// num_layers * num_neurons: gate_proj pre-activations (before silu, before edits).
    pub gate: Vec<f32>,
    /// num_layers: the READ scale of each layer's MLP input. Qwen3: 1/rms of the residual entering the
    /// MLP (the post-attention RMSNorm input). OLMo 2: exactly 1 (the MLP reads the raw residual).
    pub inv_rms: Vec<f32>,
    /// num_layers: the WRITE scale of each layer's MLP output. Qwen3: exactly 1. OLMo 2: 1/rms of the
    /// MLP output down(h) (the post_feedforward_layernorm input), eps included.
    pub write_scale: Vec<f32>,
}

/// Extra last-position readout for follow.rs, filled by `Model::forward_follow` alongside the trace sink.
pub(crate) struct FollowCapture {
    /// The trace readout (act, gate, inv_rms for every layer).
    pub trace: TraceCapture,
    /// [N]: up_proj outputs W_up · x̃ of the followed layer (before silu product and edits).
    pub up: Vec<f32>,
    /// [hidden]: residual entering the followed layer's MLP (the post_attention_layernorm input).
    pub x: Vec<f32>,
    /// rms of the final residual (the model.norm input) at the last position, eps included like inv_rms.
    pub rms_final: f32,
    /// [vocab]: logits of the last position.
    pub logits: Vec<f32>,
}

/// Device-side collector for `forward_follow`.
struct FollowSink {
    layer: usize,
    up: Option<Tensor>,
    x: Option<Tensor>,
}

/// Collects, for `Model::forward_project`, x·d̂ at every residual site before and after the site's edit
/// (one [T] tensor each, on device), plus an optional worked sample.
pub(crate) struct ProjectSink {
    /// [hidden, 1] f32 unit direction.
    dhat: Tensor,
    /// (site, position, dims) of the worked sample.
    sample: Option<(usize, usize, usize)>,
    before: Vec<Tensor>,
    after: Vec<Tensor>,
    /// [2*dims + 2]: x[..dims], x_new[..dims], |x|, |x_new| (f32).
    sample_t: Option<Tensor>,
}

impl ProjectSink {
    pub(crate) fn new(dhat: Tensor, sample: Option<(usize, usize, usize)>) -> Self {
        Self { dhat, sample, before: Vec::new(), after: Vec::new(), sample_t: None }
    }

    fn dot(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (_, s, h) = x.dims3()?;
        x.to_dtype(DType::F32)?.reshape((s, h))?.matmul(&self.dhat)?.flatten_all()
    }

    /// `xb` = the memo arriving at `site`, `xa` = after the site's edit (`edited` false: xa is xb).
    fn record(&mut self, site: usize, xb: &Tensor, xa: &Tensor, edited: bool) -> candle_core::Result<()> {
        let b = self.dot(xb)?;
        let a = if edited { self.dot(xa)? } else { b.clone() };
        self.before.push(b);
        self.after.push(a);
        if let Some((ss, pos, dims)) = self.sample {
            if ss == site {
                let row = |x: &Tensor| -> candle_core::Result<Tensor> {
                    x.narrow(1, pos, 1)?.to_dtype(DType::F32)?.flatten_all()
                };
                let (rb, ra) = (row(xb)?, row(xa)?);
                let norm = |r: &Tensor| r.sqr()?.sum_all()?.sqrt()?.reshape(1);
                self.sample_t = Some(Tensor::cat(&[&rb.narrow(0, 0, dims)?, &ra.narrow(0, 0, dims)?, &norm(&rb)?, &norm(&ra)?], 0)?);
            }
        }
        Ok(())
    }
}

/// Host readout of `Model::forward_project`.
pub(crate) struct ProjectCapture {
    /// T * sites, position-major.
    pub before: Vec<f32>,
    pub after: Vec<f32>,
    /// [2*dims + 2] (see `ProjectSink`), when a sample was requested.
    pub sample: Option<Vec<f32>>,
}

/// Residual-stream direction edit on x [1, s, hidden]: x ← x − (x·Bᵀ)·B, then x ← x + add.
fn apply_resid(x: Tensor, e: &ResidEdit) -> candle_core::Result<Tensor> {
    let mut x = x;
    if let Some(b) = &e.ablate {
        let (one, s, h) = x.dims3()?;
        let x2 = x.reshape((one * s, h))?;
        let proj = x2.matmul(&b.t()?)?.matmul(b)?; // [s, h]
        x = (x2 - proj)?.reshape((one, s, h))?;
    }
    if let Some(a) = &e.add {
        x = x.broadcast_add(a)?;
    }
    Ok(x)
}

struct Layer {
    /// Qwen3 input_layernorm (before attention); None for OLMo 2.
    input_ln: Option<Tensor>,
    /// Qwen3 post_attention_layernorm (before the MLP); None for OLMo 2.
    mlp_in_ln: Option<Tensor>,
    /// OLMo 2 post_attention_layernorm (on o_proj's output); None for Qwen3.
    attn_out_ln: Option<Tensor>,
    /// OLMo 2 post_feedforward_layernorm (on down_proj's output); None for Qwen3.
    mlp_out_ln: Option<Tensor>,
    /// Analysis view [hidden]: the MLP-input norm weight (Qwen3 mlp_in_ln, OLMo 2 ones).
    read_w: Tensor,
    /// Analysis view [hidden]: the MLP-output norm weight (Qwen3 ones, OLMo 2 mlp_out_ln).
    write_w: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    /// Qwen3 [head_dim] applied per head; OLMo 2 [heads*head_dim] / [kv_heads*head_dim] over the projection.
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
    cache: KvCache,
}

pub struct Model {
    arch: Arch,
    geom: Geometry,
    eps: f32,
    device: Device,
    dtype: DType,
    embed: Tensor,
    /// [vocab, hidden] unembedding: lm_head.weight, or embed_tokens when tied.
    lm_head: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    cos: Tensor,
    sin: Tensor,
    max_pos: usize,
    /// Lazily computed [vocab] f32: 1 / |embed row| (follow.rs `wakes_on`). Never touched by forward.
    embed_inv_norm: std::sync::OnceLock<Tensor>,
}

/// Kept so older call sites read naturally; the struct serves every `Arch`.
pub type Qwen3 = Model;

const KV_CHUNK: usize = 512;

/// The weight files of a checkpoint directory: model.safetensors, the shards listed by
/// model.safetensors.index.json, or (when neither exists) pytorch_model.bin.
enum WeightFiles {
    Safetensors(Vec<std::path::PathBuf>),
    Pth(std::path::PathBuf),
}

fn weight_files(model_dir: &Path) -> anyhow::Result<WeightFiles> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(WeightFiles::Safetensors(vec![single]));
    }
    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
        let map = v["weight_map"].as_object().ok_or_else(|| anyhow::anyhow!("{}: no weight_map", index.display()))?;
        let mut files: Vec<String> = map.values().filter_map(|f| f.as_str().map(String::from)).collect();
        files.sort();
        files.dedup();
        return Ok(WeightFiles::Safetensors(files.into_iter().map(|f| model_dir.join(f)).collect()));
    }
    let pth = model_dir.join("pytorch_model.bin");
    if pth.exists() {
        return Ok(WeightFiles::Pth(pth));
    }
    anyhow::bail!("{}: no model.safetensors, model.safetensors.index.json or pytorch_model.bin", model_dir.display())
}

impl Model {
    /// bf16 on Metal, f32 on the CPU.
    pub fn load(cfg: &ModelConfig, model_dir: &Path, device: &Device) -> anyhow::Result<Self> {
        let dtype = if device.is_cpu() { DType::F32 } else { DType::BF16 };
        Self::load_with_dtype(cfg, model_dir, device, dtype)
    }

    /// As `load`, with an explicit compute dtype (F32 on Metal trades speed for exact small-edit effects).
    pub fn load_with_dtype(cfg: &ModelConfig, model_dir: &Path, device: &Device, dtype: DType) -> anyhow::Result<Self> {
        let vb = match weight_files(model_dir)? {
            // SAFETY: the weights files are not modified while mapped.
            WeightFiles::Safetensors(paths) => unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, device)? },
            WeightFiles::Pth(p) => VarBuilder::from_pth(p, dtype, device)?,
        };
        let arch = cfg.arch();
        let g = cfg.geometry();
        let (h, n, hd) = (g.hidden_size, g.num_neurons, g.head_dim);
        let vm = vb.pp("model");
        let embed = vm.get((g.vocab_size, h), "embed_tokens.weight")?;
        let lm_head = if cfg.tie_word_embeddings { embed.clone() } else { vb.get((g.vocab_size, h), "lm_head.weight")? };
        let ones = Tensor::ones(h, dtype, device)?;
        let mut layers = Vec::with_capacity(g.num_layers);
        for i in 0..g.num_layers {
            let lb = vm.pp(format!("layers.{i}"));
            let a = lb.pp("self_attn");
            let m = lb.pp("mlp");
            let (input_ln, mlp_in_ln, attn_out_ln, mlp_out_ln, q_norm, k_norm) = match arch {
                Arch::Qwen3 => (
                    Some(lb.get(h, "input_layernorm.weight")?),
                    Some(lb.get(h, "post_attention_layernorm.weight")?),
                    None,
                    None,
                    a.get(hd, "q_norm.weight")?,
                    a.get(hd, "k_norm.weight")?,
                ),
                Arch::Olmo2 => (
                    None,
                    None,
                    Some(lb.get(h, "post_attention_layernorm.weight")?),
                    Some(lb.get(h, "post_feedforward_layernorm.weight")?),
                    a.get(g.num_heads * hd, "q_norm.weight")?,
                    a.get(g.num_kv_heads * hd, "k_norm.weight")?,
                ),
            };
            let read_w = mlp_in_ln.clone().unwrap_or_else(|| ones.clone());
            let write_w = mlp_out_ln.clone().unwrap_or_else(|| ones.clone());
            layers.push(Layer {
                input_ln,
                mlp_in_ln,
                attn_out_ln,
                mlp_out_ln,
                read_w,
                write_w,
                q: linear_no_bias(h, g.num_heads * hd, a.pp("q_proj"))?,
                k: linear_no_bias(h, g.num_kv_heads * hd, a.pp("k_proj"))?,
                v: linear_no_bias(h, g.num_kv_heads * hd, a.pp("v_proj"))?,
                o: linear_no_bias(g.num_heads * hd, h, a.pp("o_proj"))?,
                q_norm,
                k_norm,
                gate: linear_no_bias(h, n, m.pp("gate_proj"))?,
                up: linear_no_bias(h, n, m.pp("up_proj"))?,
                down: linear_no_bias(n, h, m.pp("down_proj"))?,
                cache: KvCache::new(2, KV_CHUNK),
            });
        }
        let norm = vm.get(h, "norm.weight")?;
        // RoPE tables (non-interleaved), [max_pos, hd/2].
        let max_pos = cfg.max_position_embeddings.min(32768).max(1);
        let half = hd / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / (cfg.rope_theta.powf((2 * i) as f64 / hd as f64)) as f32)
            .collect();
        let inv = Tensor::from_vec(inv, (1, half), &Device::Cpu)?;
        let pos = Tensor::arange(0u32, max_pos as u32, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape((max_pos, 1))?;
        let freqs = pos.broadcast_mul(&inv)?;
        let cos = freqs.cos()?.to_dtype(dtype)?.to_device(device)?;
        let sin = freqs.sin()?.to_dtype(dtype)?.to_device(device)?;
        Ok(Self {
            arch,
            geom: g,
            eps: cfg.rms_norm_eps as f32,
            device: device.clone(),
            dtype,
            embed,
            lm_head,
            layers,
            norm,
            cos,
            sin,
            max_pos,
            embed_inv_norm: std::sync::OnceLock::new(),
        })
    }

    pub fn geometry(&self) -> Geometry {
        self.geom
    }

    pub fn arch(&self) -> Arch {
        self.arch
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Drops the KV cache (start of every run).
    pub fn clear_cache(&mut self) {
        for l in self.layers.iter_mut() {
            l.cache.reset();
        }
    }

    /// Processes `tokens` at positions offset..offset+len, appending to the KV cache.
    /// Returns the logits of the LAST position (f32, len vocab_size) and the capture.
    pub fn forward(
        &mut self,
        tokens: &[u32],
        offset: usize,
        edits: &DeviceEdits,
        want_residuals: bool,
    ) -> anyhow::Result<(Vec<f32>, Capture)> {
        self.forward_impl(tokens, offset, edits, want_residuals, None, None, None)
    }

    /// Prefill for project.rs: runs `tokens` from offset 0 (appending to the KV cache) and records x·d̂
    /// at every position and residual site before/after the site's edit (see `ProjectSink`). Skips the
    /// logits and all other capture. One device→host sync.
    pub(crate) fn forward_project(
        &mut self,
        tokens: &[u32],
        edits: &DeviceEdits,
        mut sink: ProjectSink,
    ) -> anyhow::Result<ProjectCapture> {
        self.forward_impl(tokens, 0, edits, false, None, Some(&mut sink), None)?;
        let sites = self.geom.num_layers + 1;
        anyhow::ensure!(sink.before.len() == sites && sink.after.len() == sites, "forward_project: missing sites");
        let t = tokens.len();
        let nb = t * sites;
        let before = Tensor::stack(&sink.before, 1)?.flatten_all()?; // [T, sites]
        let after = Tensor::stack(&sink.after, 1)?.flatten_all()?;
        let mut parts: Vec<&Tensor> = vec![&before, &after];
        if let Some(s) = &sink.sample_t {
            parts.push(s);
        }
        let mut all = Tensor::cat(&parts, 0)?.to_vec1::<f32>()?;
        let sample = if sink.sample_t.is_some() { Some(all.split_off(2 * nb)) } else { None };
        let after = all.split_off(nb);
        Ok(ProjectCapture { before: all, after, sample })
    }

    /// Prefill for trace.rs: runs `tokens` from offset 0 (appending to the KV cache) and records, at the
    /// LAST position only, per layer the post-edit MLP activations, the gate pre-activations and the
    /// MLP's read and write scales (see `TraceCapture`). Skips the logits and the all-position capture.
    /// One device→host sync.
    pub(crate) fn forward_trace(&mut self, tokens: &[u32], edits: &DeviceEdits) -> anyhow::Result<TraceCapture> {
        let g = self.geom;
        let mut sink: Vec<Tensor> = Vec::with_capacity(g.num_layers * 4);
        self.forward_impl(tokens, 0, edits, false, Some(&mut sink), None, None)?;
        let all = Tensor::cat(&sink, 0)?.to_vec1::<f32>()?;
        Self::unpack_trace(&g, &all)
    }

    fn unpack_trace(g: &Geometry, all: &[f32]) -> anyhow::Result<TraceCapture> {
        let (l, n) = (g.num_layers, g.num_neurons);
        anyhow::ensure!(all.len() == l * (2 * n + 2), "forward_trace: unexpected capture size");
        let mut cap = TraceCapture {
            act: Vec::with_capacity(l * n),
            gate: Vec::with_capacity(l * n),
            inv_rms: Vec::with_capacity(l),
            write_scale: Vec::with_capacity(l),
        };
        for chunk in all.chunks_exact(2 * n + 2) {
            cap.act.extend_from_slice(&chunk[..n]);
            cap.gate.extend_from_slice(&chunk[n..2 * n]);
            cap.inv_rms.push(chunk[2 * n]);
            cap.write_scale.push(chunk[2 * n + 1]);
        }
        Ok(cap)
    }

    /// Prefill for follow.rs: the trace readout plus, at the LAST position, the followed layer's up_proj
    /// outputs and MLP-input residual, the final residual's rms and the logits. One device→host sync.
    pub(crate) fn forward_follow(&mut self, tokens: &[u32], edits: &DeviceEdits, layer: usize) -> anyhow::Result<FollowCapture> {
        let g = self.geom;
        anyhow::ensure!(layer < g.num_layers, "forward_follow: layer {layer} out of range");
        let mut sink: Vec<Tensor> = Vec::with_capacity(g.num_layers * 4 + 4);
        let mut fs = FollowSink { layer, up: None, x: None };
        self.forward_impl(tokens, 0, edits, false, Some(&mut sink), None, Some(&mut fs))?;
        let nt = sink.len() - 2; // forward_impl appended [rms_final] and logits after the per-layer quads
        let (up, x) = (fs.up.ok_or_else(|| anyhow::anyhow!("forward_follow: up not captured"))?, fs.x.ok_or_else(|| anyhow::anyhow!("forward_follow: x not captured"))?);
        sink.push(up);
        sink.push(x);
        let all = Tensor::cat(&sink, 0)?.to_vec1::<f32>()?;
        let (l, n, h, v) = (g.num_layers, g.num_neurons, g.hidden_size, g.vocab_size);
        let tl = l * (2 * n + 2);
        anyhow::ensure!(nt == 4 * l && all.len() == tl + 1 + v + n + h, "forward_follow: unexpected capture size");
        let trace = Self::unpack_trace(&g, &all[..tl])?;
        let rms_final = all[tl];
        let logits = all[tl + 1..tl + 1 + v].to_vec();
        let up = all[tl + 1 + v..tl + 1 + v + n].to_vec();
        let x = all[tl + 1 + v + n..].to_vec();
        Ok(FollowCapture { trace, up, x, rms_final, logits })
    }

    /// [vocab] f32 1/|input-embedding row|, computed once (f32 accumulation) and cached on the device.
    pub(crate) fn embed_inv_norm(&self) -> anyhow::Result<&Tensor> {
        if let Some(t) = self.embed_inv_norm.get() {
            return Ok(t);
        }
        let v = self.geom.vocab_size;
        let chunk = 16384;
        let mut parts = Vec::new();
        let mut start = 0;
        while start < v {
            let len = chunk.min(v - start);
            let e = self.embed.narrow(0, start, len)?.to_dtype(DType::F32)?;
            parts.push((e.sqr()?.sum(1)? + 1e-12)?.sqrt()?.recip()?);
            start += len;
        }
        let t = Tensor::cat(&parts, 0)?;
        Ok(self.embed_inv_norm.get_or_init(|| t))
    }

    /// Layer `layer`'s up_proj weight [N, hidden], model dtype.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn up_weight(&self, layer: usize) -> &Tensor {
        self.layers[layer].up.weight()
    }

    /// The final RMSNorm weight [hidden], model dtype.
    pub(crate) fn final_norm_weight(&self) -> &Tensor {
        &self.norm
    }

    /// Layer `layer`'s MLP weights: (READ norm weight [hidden] — Qwen3 post_attention_layernorm, OLMo 2
    /// ones —, gate_proj [N, hidden], down_proj [hidden, N]), model dtype.
    pub(crate) fn mlp_weights(&self, layer: usize) -> (&Tensor, &Tensor, &Tensor) {
        let l = &self.layers[layer];
        (&l.read_w, l.gate.weight(), l.down.weight())
    }

    /// Layer `layer`'s WRITE norm weight [hidden], model dtype: OLMo 2 post_feedforward_layernorm, Qwen3
    /// ones. A neuron's direct write into the residual is a · (write_w ⊙ W_down[:, i]) · write_scale.
    pub(crate) fn mlp_write_weight(&self, layer: usize) -> &Tensor {
        &self.layers[layer].write_w
    }

    /// `trace` = Some: record the last position's (act [N], gate [N], inv_rms [1]) per layer as f32
    /// into the sink, skip the per-head/per-neuron capture and the logits (returns empty values).
    fn forward_impl(
        &mut self,
        tokens: &[u32],
        offset: usize,
        edits: &DeviceEdits,
        want_residuals: bool,
        mut trace: Option<&mut Vec<Tensor>>,
        mut project: Option<&mut ProjectSink>,
        mut follow: Option<&mut FollowSink>,
    ) -> anyhow::Result<(Vec<f32>, Capture)> {
        let s = tokens.len();
        anyhow::ensure!(s > 0, "forward: empty token slice");
        anyhow::ensure!(
            offset + s <= self.max_pos,
            "forward: position {} exceeds max {}",
            offset + s,
            self.max_pos
        );
        let g = self.geom;
        let (nh, nkv, hd) = (g.num_heads, g.num_kv_heads, g.head_dim);
        let n_rep = nh / nkv;
        let eps = self.eps;
        let dev = self.device.clone();
        let dtype = self.dtype;

        let arch = self.arch;
        let ids = Tensor::from_slice(tokens, (1, s), &dev)?;
        let mut x = self.embed.embedding(&ids.flatten_all()?)?.reshape((1, s, g.hidden_size))?;
        {
            let e = edits.resid.first().and_then(|r| r.as_ref());
            let xb = if project.is_some() { Some(x.clone()) } else { None };
            if let Some(e) = e {
                x = apply_resid(x, e)?;
            }
            if let (Some(p), Some(xb)) = (project.as_deref_mut(), &xb) {
                p.record(0, xb, &x, e.is_some())?;
            }
        }
        let embed_last = if want_residuals {
            Some(x.narrow(1, s - 1, 1)?.reshape((1, g.hidden_size))?)
        } else {
            None
        };

        let cos = self.cos.narrow(0, offset, s)?;
        let sin = self.sin.narrow(0, offset, s)?;
        let total = offset + s;
        let mask = if s > 1 {
            let m: Vec<f32> = (0..s)
                .flat_map(|i| {
                    (0..total).map(move |j| if j > i + offset { f32::NEG_INFINITY } else { 0.0 })
                })
                .collect();
            Some(Tensor::from_vec(m, (1, 1, s, total), &dev)?)
        } else {
            None
        };
        let scale = 1.0 / (hd as f64).sqrt();

        let mut neuron_caps: Vec<Tensor> = Vec::with_capacity(g.num_layers);
        let mut head_caps: Vec<Tensor> = Vec::with_capacity(g.num_layers);
        let mut resid: Vec<Tensor> = Vec::new();

        for (li, layer) in self.layers.iter_mut().enumerate() {
            // ---- attention ----
            let hx = match &layer.input_ln {
                Some(w) => candle_nn::ops::rms_norm(&x, w, eps)?,
                None => x.clone(),
            };
            let (q, k) = (layer.q.forward(&hx)?, layer.k.forward(&hx)?);
            let v = layer.v.forward(&hx)?.reshape((1, s, nkv, hd))?;
            let (q, k) = match arch {
                Arch::Qwen3 => (
                    candle_nn::ops::rms_norm(&q.reshape((1, s, nh, hd))?, &layer.q_norm, eps)?,
                    candle_nn::ops::rms_norm(&k.reshape((1, s, nkv, hd))?, &layer.k_norm, eps)?,
                ),
                Arch::Olmo2 => (
                    candle_nn::ops::rms_norm(&q, &layer.q_norm, eps)?.reshape((1, s, nh, hd))?,
                    candle_nn::ops::rms_norm(&k, &layer.k_norm, eps)?.reshape((1, s, nkv, hd))?,
                ),
            };
            let q = q.transpose(1, 2)?.contiguous()?;
            let k = k.transpose(1, 2)?.contiguous()?;
            let v = v.transpose(1, 2)?.contiguous()?;
            let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
            let (k, v) = layer.cache.append(&k, &v)?;
            let t = k.dim(2)?;
            let (k, v) = if n_rep > 1 {
                (
                    k.unsqueeze(2)?.expand((1, nkv, n_rep, t, hd))?.reshape((1, nh, t, hd))?,
                    v.unsqueeze(2)?.expand((1, nkv, n_rep, t, hd))?.reshape((1, nh, t, hd))?,
                )
            } else {
                (k.contiguous()?, v.contiguous()?)
            };
            let kt = k.transpose(2, 3)?.contiguous()?;
            let scores = (q.matmul(&kt)? * scale)?.to_dtype(DType::F32)?;
            let scores = match &mask {
                Some(m) => scores.broadcast_add(m)?,
                None => scores,
            };
            let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(dtype)?;
            let mut attn = probs.matmul(&v)?; // (1, nh, s, hd)
            if let Some(mult) = &edits.heads[li] {
                attn = attn.broadcast_mul(&mult.reshape((1, nh, 1, 1))?)?;
            }
            if trace.is_none() && project.is_none() {
                // per-head L2 norm over d -> [s, nh]
                let hn = attn
                    .to_dtype(DType::F32)?
                    .sqr()?
                    .sum(D::Minus1)?
                    .sqrt()?
                    .squeeze(0)?
                    .transpose(0, 1)?; // [s, nh]
                head_caps.push(hn);
            }
            let attn = attn.transpose(1, 2)?.reshape((1, s, nh * hd))?;
            let ao = layer.o.forward(&attn)?;
            let ao = match &layer.attn_out_ln {
                Some(w) => candle_nn::ops::rms_norm(&ao, w, eps)?,
                None => ao,
            };
            x = (x + ao)?;

            // ---- MLP ----
            let hx = match &layer.mlp_in_ln {
                Some(w) => candle_nn::ops::rms_norm(&x, w, eps)?,
                None => x.clone(),
            };
            let gate_pre = layer.gate.forward(&hx)?;
            let up = layer.up.forward(&hx)?;
            let mut h = (candle_nn::ops::silu(&gate_pre)? * &up)?;
            if let Some(e) = &edits.mlp[li] {
                h = h.broadcast_mul(&e.scale)?.broadcast_add(&e.offset)?;
            }
            if let Some(sink) = trace.as_deref_mut() {
                let last = |t: &Tensor| -> candle_core::Result<Tensor> {
                    t.narrow(1, s - 1, 1)?.to_dtype(DType::F32)?.flatten_all()
                };
                let xl = last(&x)?; // residual entering this MLP (after the attention add)
                let inv = match &layer.mlp_in_ln {
                    Some(_) => (xl.sqr()?.mean_all()? + eps as f64)?.sqrt()?.recip()?.reshape(1)?,
                    None => Tensor::ones(1, DType::F32, &dev)?,
                };
                sink.push(last(&h)?);
                sink.push(last(&gate_pre)?);
                sink.push(inv);
                if let Some(f) = follow.as_deref_mut() {
                    if f.layer == li {
                        f.up = Some(last(&up)?);
                        f.x = Some(xl);
                    }
                }
            } else if project.is_none() {
                neuron_caps.push(h.squeeze(0)?); // [s, N]
            }
            let mo = layer.down.forward(&h)?;
            let mo = match &layer.mlp_out_ln {
                Some(w) => {
                    if let Some(sink) = trace.as_deref_mut() {
                        let ml = mo.narrow(1, s - 1, 1)?.to_dtype(DType::F32)?.flatten_all()?;
                        sink.push((ml.sqr()?.mean_all()? + eps as f64)?.sqrt()?.recip()?.reshape(1)?);
                    }
                    candle_nn::ops::rms_norm(&mo, w, eps)?
                }
                None => {
                    if let Some(sink) = trace.as_deref_mut() {
                        sink.push(Tensor::ones(1, DType::F32, &dev)?);
                    }
                    mo
                }
            };
            x = (x + mo)?;
            let e = edits.resid.get(li + 1).and_then(|r| r.as_ref());
            let xb = if project.is_some() { Some(x.clone()) } else { None };
            if let Some(e) = e {
                x = apply_resid(x, e)?;
            }
            if let (Some(p), Some(xb)) = (project.as_deref_mut(), &xb) {
                p.record(li + 1, xb, &x, e.is_some())?;
            }

            if want_residuals {
                resid.push(x.narrow(1, s - 1, 1)?.reshape((1, g.hidden_size))?);
            }
        }

        if let (Some(sink), Some(_)) = (trace.as_deref_mut(), follow.as_deref()) {
            let xl = x.narrow(1, s - 1, 1)?.reshape((1, g.hidden_size))?;
            let xf = xl.to_dtype(DType::F32)?;
            let rms = (xf.sqr()?.mean_all()? + eps as f64)?.sqrt()?.reshape(1)?;
            let normed = candle_nn::ops::rms_norm(&xl, &self.norm, eps)?;
            let logits = normed.matmul(&self.lm_head.t()?)?.to_dtype(DType::F32)?.flatten_all()?;
            sink.push(rms);
            sink.push(logits);
        }
        if trace.is_some() || project.is_some() {
            let empty = Capture { neurons: Vec::new(), heads: Vec::new(), residuals: None, lens: None, embed_last: None };
            return Ok((Vec::new(), empty));
        }

        let last = x.narrow(1, s - 1, 1)?.reshape((1, g.hidden_size))?;
        let last = candle_nn::ops::rms_norm(&last, &self.norm, eps)?;
        let logits = last.matmul(&self.lm_head.t()?)?.to_dtype(DType::F32)?.flatten_all()?;

        let neurons = Tensor::stack(&neuron_caps, 1)?.to_dtype(DType::F32)?.flatten_all()?; // [s, L, N]
        let heads = Tensor::stack(&head_caps, 1)?.flatten_all()?; // [s, L, H]
        let n_len = neurons.dim(0)?;
        let h_len = heads.dim(0)?;
        let v_len = logits.dim(0)?;
        let residuals = if want_residuals {
            Some(Tensor::cat(&resid, 0)?)
        } else {
            None
        };
        let lens_t = match &residuals {
            Some(r) => Some(crate::lens::logit_lens_device(self, r)?),
            None => None,
        };
        // single device->host sync (logits, neurons, heads, and the lens when requested)
        let mut parts: Vec<&Tensor> = vec![&logits, &neurons, &heads];
        if let Some(l) = &lens_t {
            parts.push(l);
        }
        let mut all = Tensor::cat(&parts, 0)?.to_vec1::<f32>()?;
        let lens = lens_t.map(|_| crate::lens::unpack(&all[v_len + n_len + h_len..]));
        all.truncate(v_len + n_len + h_len);
        let heads_v = all.split_off(v_len + n_len);
        let neurons_v = all.split_off(v_len);
        let logits_v = all;
        Ok((
            logits_v,
            Capture {
                neurons: neurons_v,
                heads: heads_v,
                residuals,
                lens,
                embed_last,
            },
        ))
    }

    /// Final RMSNorm applied to x[..., hidden].
    pub fn final_norm(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = x.to_dtype(self.dtype)?.contiguous()?;
        Ok(candle_nn::ops::rms_norm(&x, &self.norm, self.eps)?)
    }

    /// Unembedding matrix [vocab_size, hidden_size]: lm_head (OLMo 2), tied to embed_tokens (Qwen3-0.6B).
    pub fn unembed(&self) -> &Tensor {
        &self.lm_head
    }

    /// Input embedding matrix [vocab_size, hidden_size] (embed_tokens).
    pub fn embedding(&self) -> &Tensor {
        &self.embed
    }

    /// Column `index` of layer `layer`'s down_proj weight: [hidden_size] — what the neuron writes before
    /// the write norm (OLMo 2 then multiplies it by `mlp_write_weight` and the write scale).
    pub fn down_proj_column(&self, layer: usize, index: usize) -> anyhow::Result<Tensor> {
        anyhow::ensure!(layer < self.geom.num_layers, "layer {layer} out of range");
        anyhow::ensure!(index < self.geom.num_neurons, "neuron {index} out of range");
        let w = self.layers[layer].down.weight(); // [hidden, N]
        Ok(w.narrow(1, index, 1)?.squeeze(1)?.contiguous()?)
    }
}

/// Random tiny checkpoints on CPU (f32) for unit tests, written to a temp dir and loaded the normal way.
#[cfg(test)]
pub(crate) mod test_util {
    use super::*;
    use std::collections::HashMap;

    /// `nl`-layer random model of either family (v 40, hidden 16, N 24, 2 heads × 8; Qwen3 1 kv head,
    /// OLMo 2 2 kv heads with head_dim left out of config.json, untied lm_head).
    pub(crate) fn tiny_model(arch: Arch, nl: usize, tag: &str) -> anyhow::Result<(Model, std::path::PathBuf)> {
        tiny_model_tied(arch, nl, tag, arch == Arch::Qwen3)
    }

    /// As `tiny_model`, choosing whether the unembedding is tied to embed_tokens (no lm_head.weight on disk).
    pub(crate) fn tiny_model_tied(arch: Arch, nl: usize, tag: &str, tied: bool) -> anyhow::Result<(Model, std::path::PathBuf)> {
        let (v, h, n, nh, hd) = (40usize, 16usize, 24usize, 2usize, 8usize);
        let nkv = if arch == Arch::Olmo2 { 2 } else { 1 };
        let dir = std::env::temp_dir().join(format!(
            "ns-{tag}-{arch:?}-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir)?;
        let cfg = match arch {
            Arch::Qwen3 => serde_json::json!({
                "vocab_size": v, "hidden_size": h, "intermediate_size": n, "num_hidden_layers": nl,
                "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hd,
                "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 64,
                "tie_word_embeddings": true
            }),
            Arch::Olmo2 => serde_json::json!({
                "model_type": "olmo2", "vocab_size": v, "hidden_size": h, "intermediate_size": n,
                "num_hidden_layers": nl, "num_attention_heads": nh, "num_key_value_heads": nkv,
                "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 64,
                "tie_word_embeddings": tied, "attention_bias": false, "hidden_act": "silu", "rope_scaling": null
            }),
        };
        std::fs::write(dir.join("config.json"), cfg.to_string())?;
        let cpu = Device::Cpu;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let rnd = |shape: &[usize], std: f64| Tensor::randn(0f32, std as f32, shape, &cpu);
        let norm = |len: usize| -> candle_core::Result<Tensor> { rnd(&[len], 0.3)? + 1.0 };
        t.insert("model.embed_tokens.weight".into(), rnd(&[v, h], 1.0)?);
        t.insert("model.norm.weight".into(), norm(h)?);
        if !tied {
            t.insert("lm_head.weight".into(), rnd(&[v, h], 1.0)?);
        }
        for i in 0..nl {
            let p = format!("model.layers.{i}");
            match arch {
                Arch::Qwen3 => {
                    t.insert(format!("{p}.input_layernorm.weight"), norm(h)?);
                    t.insert(format!("{p}.post_attention_layernorm.weight"), norm(h)?);
                    t.insert(format!("{p}.self_attn.q_norm.weight"), norm(hd)?);
                    t.insert(format!("{p}.self_attn.k_norm.weight"), norm(hd)?);
                }
                Arch::Olmo2 => {
                    t.insert(format!("{p}.post_attention_layernorm.weight"), norm(h)?);
                    t.insert(format!("{p}.post_feedforward_layernorm.weight"), norm(h)?);
                    t.insert(format!("{p}.self_attn.q_norm.weight"), norm(nh * hd)?);
                    t.insert(format!("{p}.self_attn.k_norm.weight"), norm(nkv * hd)?);
                }
            }
            t.insert(format!("{p}.self_attn.q_proj.weight"), rnd(&[nh * hd, h], 0.3)?);
            t.insert(format!("{p}.self_attn.k_proj.weight"), rnd(&[nkv * hd, h], 0.3)?);
            t.insert(format!("{p}.self_attn.v_proj.weight"), rnd(&[nkv * hd, h], 0.3)?);
            t.insert(format!("{p}.self_attn.o_proj.weight"), rnd(&[h, nh * hd], 0.3)?);
            t.insert(format!("{p}.mlp.gate_proj.weight"), rnd(&[n, h], 0.5)?);
            t.insert(format!("{p}.mlp.up_proj.weight"), rnd(&[n, h], 0.5)?);
            t.insert(format!("{p}.mlp.down_proj.weight"), rnd(&[h, n], 0.5)?);
        }
        candle_core::safetensors::save(&t, dir.join("model.safetensors"))?;
        let cfg = ModelConfig::load(&dir)?;
        assert_eq!(cfg.arch(), arch);
        Ok((Model::load(&cfg, &dir, &cpu)?, dir))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OLMo 2 with tie_word_embeddings (what the trainer in trainer/ exports): loads without lm_head.weight and
    /// gives the same logits as the untied model whose lm_head is a copy of embed_tokens.
    #[test]
    fn olmo2_tied_embeddings_match_an_explicit_copy() -> anyhow::Result<()> {
        let (mut tied, dir) = test_util::tiny_model_tied(Arch::Olmo2, 2, "tied", true)?;
        let mut t: std::collections::HashMap<String, Tensor> = candle_core::safetensors::load(dir.join("model.safetensors"), &Device::Cpu)?;
        assert!(!t.contains_key("lm_head.weight"), "a tied export has no lm_head.weight");
        t.insert("lm_head.weight".into(), t["model.embed_tokens.weight"].clone());
        let udir = dir.with_file_name(format!("{}-untied", dir.file_name().unwrap().to_string_lossy()));
        std::fs::create_dir_all(&udir)?;
        candle_core::safetensors::save(&t, udir.join("model.safetensors"))?;
        let cfg_raw = std::fs::read_to_string(dir.join("config.json"))?.replace("\"tie_word_embeddings\":true", "\"tie_word_embeddings\":false");
        std::fs::write(udir.join("config.json"), &cfg_raw)?;
        let ucfg = ModelConfig::load(&udir)?;
        assert!(!ucfg.tie_word_embeddings);
        let mut untied = Model::load(&ucfg, &udir, &Device::Cpu)?;
        let ids = [3u32, 17, 5, 29, 11];
        let edits = DeviceEdits::none(2);
        let (a, _) = tied.forward(&ids, 0, &edits, false)?;
        let (b, _) = untied.forward(&ids, 0, &edits, false)?;
        let max = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(max < 1e-6, "tied vs untied logits differ by {max}");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&udir);
        Ok(())
    }

    #[test]
    #[ignore]
    fn paris_on_metal() -> anyhow::Result<()> {
        let home = std::env::var("HOME")?;
        let dir = std::path::PathBuf::from(home)
            .join("Library/Application Support/NeuroScope/models/Qwen3-0.6B");
        let cfg = ModelConfig::load(&dir)?;
        let dev = Device::new_metal(0)?;
        let mut m = Model::load(&cfg, &dir, &dev)?;
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let enc = tok.encode("The capital of France is", false).map_err(anyhow::Error::msg)?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        let edits = DeviceEdits::none(cfg.num_hidden_layers);
        let argmax = |v: &[f32]| {
            v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
        };
        let t0 = std::time::Instant::now();
        let (logits, cap) = m.forward(&ids, 0, &edits, true)?;
        let prefill = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(cap.neurons.len(), ids.len() * 28 * 3072);
        assert_eq!(cap.heads.len(), ids.len() * 28 * 16);
        let mut next = argmax(&logits);
        let text = tok.decode(&[next], false).map_err(anyhow::Error::msg)?;
        println!("prefill {} tokens: {prefill:.1} ms, next = {text:?}", ids.len());
        assert_eq!(text, " Paris");
        let mut pos = ids.len();
        let mut out = vec![next];
        let t1 = std::time::Instant::now();
        for _ in 0..30 {
            let (lg, _c) = m.forward(&[next], pos, &edits, false)?;
            next = argmax(&lg);
            out.push(next);
            pos += 1;
        }
        let tps = 30.0 / t1.elapsed().as_secs_f64();
        println!("decode: {tps:.1} tok/s; text: {:?}", tok.decode(&out, false).map_err(anyhow::Error::msg)?);
        Ok(())
    }
}
