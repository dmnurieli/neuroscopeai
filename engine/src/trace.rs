//! Who triggered whom: direct-path contributions of MLP desks to later desks' gate pre-activations at
//! ONE position (see `TraceEdge` in types.rs for the exact formula).
//!
//! One prefill of token_ids[..=position] (with the run's edits) records, at the last position, every
//! layer's post-edit activations a, gate pre-activations g and the MLP read/write scales (model.rs).
//! Source desk i on floor L wrote a_i · (w_out[L] ⊙ W_down[L][:, i]) · ws[L] into the residual; target
//! desk j on floor L' reads it through g_j = (w_norm[L'] ⊙ W_gate[L'][j]) · x · inv_rms[L'], so the
//! direct-path share is
//!   a_i · ((w_norm[L'] ⊙ W_gate[L'][j]) · (w_out[L] ⊙ W_down[L][:, i])) · inv_rms[L'] · ws[L].
//! Qwen3 (pre-norm): w_out = 1, ws = 1. OLMo 2 (post-norm): w_norm = 1, inv_rms = 1, w_out = the
//! post_feedforward_layernorm weight and ws = 1/rms of the MLP output at this position (the norm's
//! actual scale, held fixed — exact for the sum over a floor's desks, linear per desk).
//! Per step L → L+1 the |T|×|S| matrix is one device matmul of gathered rows/columns; focus contributions
//! are one matvec per earlier floor. Host work is only top-k selection and sorting.

use crate::edits::DeviceEdits;
use crate::engine::Engine;
use crate::model::{Model, TraceCapture};
use crate::types::{TraceEdge, TraceRequest, TraceResult};
use candle_core::{DType, Tensor};
use std::time::Instant;

const MAX_TOP: usize = 256;

pub fn trace(engine: &mut Engine, req: &TraceRequest) -> anyhow::Result<TraceResult> {
    trace_model(&mut engine.model, req)
}

/// Indices of the `k` largest |v|, strongest first (ties by lower index).
fn top_abs(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    let k = k.min(v.len());
    if k < idx.len() {
        idx.select_nth_unstable_by(k, |&a, &b| v[b].abs().total_cmp(&v[a].abs()).then(a.cmp(&b)));
        idx.truncate(k);
    }
    idx.sort_by(|&a, &b| v[b].abs().total_cmp(&v[a].abs()).then(a.cmp(&b)));
    idx
}

fn validate(req: &TraceRequest, g: &crate::types::Geometry) -> anyhow::Result<()> {
    anyhow::ensure!(!req.token_ids.is_empty(), "trace: token_ids is empty");
    anyhow::ensure!(
        req.position < req.token_ids.len(),
        "trace: position {} out of range (token_ids has {})",
        req.position,
        req.token_ids.len()
    );
    if let Some(&bad) = req.token_ids[..=req.position].iter().find(|&&t| t as usize >= g.vocab_size) {
        anyhow::bail!("trace: token id {bad} out of range (vocab {})", g.vocab_size);
    }
    for (name, v) in [("top_sources", req.top_sources), ("top_targets", req.top_targets), ("focus_top", req.focus_top)] {
        anyhow::ensure!((1..=MAX_TOP).contains(&v), "trace: {name} = {v} must be in 1..={MAX_TOP}");
    }
    if let Some((l, i)) = req.focus {
        anyhow::ensure!(l < g.num_layers, "trace: focus layer {l} out of range (0..{})", g.num_layers);
        anyhow::ensure!(i < g.num_neurons, "trace: focus index {i} out of range (0..{})", g.num_neurons);
    }
    Ok(())
}

pub(crate) fn trace_model(model: &mut Model, req: &TraceRequest) -> anyhow::Result<TraceResult> {
    let g = model.geometry();
    validate(req, &g)?;
    let edits = if req.edits.is_empty() {
        DeviceEdits::none(g.num_layers)
    } else {
        DeviceEdits::build(&req.edits, &g, model.dtype(), model.device())?
    };
    let tokens = &req.token_ids[..=req.position];

    model.clear_cache();
    let t0 = Instant::now();
    let cap = model.forward_trace(tokens, &edits);
    let forward_ms = t0.elapsed().as_secs_f64() * 1000.0;
    model.clear_cache();
    let cap = cap?;

    let (edges, explained) = step_edges(model, &cap, req.top_sources, req.top_targets)?;
    let focus_edges = match req.focus {
        Some((fl, fi)) => focus_edges(model, &cap, fl, fi, req.focus_top)?,
        None => Vec::new(),
    };
    Ok(TraceResult {
        position: req.position,
        token_id: req.token_ids[req.position],
        edges,
        focus_edges,
        explained,
        forward_ms,
    })
}

/// Floor L → L+1 arrows for every L, plus `explained` per target floor.
fn step_edges(
    model: &Model,
    cap: &TraceCapture,
    top_sources: usize,
    top_targets: usize,
) -> anyhow::Result<(Vec<TraceEdge>, Vec<f32>)> {
    let g = model.geometry();
    let (nl, n) = (g.num_layers, g.num_neurons);
    let dev = model.device();
    let mut explained = vec![0f32; nl];
    if nl < 2 {
        return Ok((Vec::new(), explained));
    }
    let act = |l: usize| &cap.act[l * n..(l + 1) * n];
    let gate = |l: usize| &cap.gate[l * n..(l + 1) * n];

    let mut sel: Vec<(Vec<usize>, Vec<usize>)> = Vec::with_capacity(nl - 1);
    let mut mats: Vec<Tensor> = Vec::with_capacity(nl - 1);
    for l in 0..nl - 1 {
        let src = top_abs(act(l), top_sources);
        let tgt = top_abs(gate(l + 1), top_targets);
        let si = Tensor::from_vec(src.iter().map(|&i| i as u32).collect::<Vec<_>>(), src.len(), dev)?;
        let ti = Tensor::from_vec(tgt.iter().map(|&i| i as u32).collect::<Vec<_>>(), tgt.len(), dev)?;
        let (wn, gw, _) = model.mlp_weights(l + 1);
        let (_, _, dw) = model.mlp_weights(l);
        let wn = wn.to_dtype(DType::F32)?.unsqueeze(0)?; // [1, h]
        let wo = model.mlp_write_weight(l).to_dtype(DType::F32)?.unsqueeze(1)?; // [h, 1]
        let rows = gw.index_select(&ti, 0)?.to_dtype(DType::F32)?.broadcast_mul(&wn)?; // [T, h]
        let cols = dw.index_select(&si, 1)?.to_dtype(DType::F32)?.broadcast_mul(&wo)?; // [h, S]
        let a_s: Vec<f32> = src.iter().map(|&i| act(l)[i]).collect();
        let a_s = Tensor::from_vec(a_s, (1, src.len()), dev)?;
        let m = ((rows.matmul(&cols)?.broadcast_mul(&a_s)? * cap.inv_rms[l + 1] as f64)? * cap.write_scale[l] as f64)?; // [T, S]
        mats.push(m.flatten_all()?);
        sel.push((src, tgt));
    }
    let all = Tensor::cat(&mats, 0)?.to_vec1::<f32>()?;

    let mut edges = Vec::new();
    let mut off = 0;
    for (l, (src, tgt)) in sel.iter().enumerate() {
        let (ns, nt) = (src.len(), tgt.len());
        let m = &all[off..off + ns * nt];
        off += ns * nt;
        let mut step: Vec<TraceEdge> = Vec::with_capacity(ns * nt);
        let (mut num, mut den) = (0f64, 0f64);
        for (t, &j) in tgt.iter().enumerate() {
            let row = &m[t * ns..(t + 1) * ns];
            num += row.iter().map(|&w| w as f64).sum::<f64>().abs();
            den += gate(l + 1)[j].abs() as f64;
            for (s, &i) in src.iter().enumerate() {
                step.push(TraceEdge { from_layer: l, from_index: i, to_layer: l + 1, to_index: j, weight: row[s] });
            }
        }
        step.sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));
        edges.extend(step);
        explained[l + 1] = if den > 0.0 { (num / den).clamp(0.0, 1.0) as f32 } else { 0.0 };
    }
    Ok((edges, explained))
}

/// Strongest direct contributions from every desk of every floor below `fl` into desk (fl, fi).
fn focus_edges(model: &Model, cap: &TraceCapture, fl: usize, fi: usize, top: usize) -> anyhow::Result<Vec<TraceEdge>> {
    if fl == 0 {
        return Ok(Vec::new());
    }
    let g = model.geometry();
    let n = g.num_neurons;
    let dev = model.device();
    let (wn, gw, _) = model.mlp_weights(fl);
    let row = (gw.narrow(0, fi, 1)?.to_dtype(DType::F32)? * wn.to_dtype(DType::F32)?.unsqueeze(0)?)?; // [1, h]
    let mut per_floor = Vec::with_capacity(fl);
    for l in 0..fl {
        let (_, _, dw) = model.mlp_weights(l);
        let wo = model.mlp_write_weight(l).to_dtype(DType::F32)?.unsqueeze(0)?; // [1, h]
        let a = Tensor::from_slice(&cap.act[l * n..(l + 1) * n], (1, n), dev)?;
        let c = (row.broadcast_mul(&wo)?.matmul(&dw.to_dtype(DType::F32)?)?.broadcast_mul(&a)? * cap.write_scale[l] as f64)?; // [1, N]
        per_floor.push(c.flatten_all()?);
    }
    let all = (Tensor::cat(&per_floor, 0)? * cap.inv_rms[fl] as f64)?.to_vec1::<f32>()?;
    Ok(top_abs(&all, top)
        .into_iter()
        .map(|k| TraceEdge { from_layer: k / n, from_index: k % n, to_layer: fl, to_index: fi, weight: all[k] })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Arch;
    use crate::model::test_util::tiny_model;
    use crate::types::{EditMode, EditSet, NeuronEdit};

    fn mat(t: &Tensor) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(t.to_dtype(DType::F32)?.to_vec2::<f32>()?)
    }

    /// Brute force: a_i · Σ_k w_norm[L'][k] · W_gate[L'][j][k] · w_out[L][k] · W_down[L][k][i] · inv_rms[L'] · ws[L], f64.
    fn brute(m: &Model, cap: &TraceCapture, l: usize, i: usize, lt: usize, j: usize) -> anyhow::Result<f64> {
        let n = m.geometry().num_neurons;
        let (wn, gw, _) = m.mlp_weights(lt);
        let (_, _, dw) = m.mlp_weights(l);
        let wn = wn.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let wo = m.mlp_write_weight(l).to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let (gw, dw) = (mat(gw)?, mat(dw)?);
        let dot: f64 = (0..wn.len()).map(|k| wn[k] as f64 * gw[j][k] as f64 * wo[k] as f64 * dw[k][i] as f64).sum();
        Ok(cap.act[l * n + i] as f64 * dot * cap.inv_rms[lt] as f64 * cap.write_scale[l] as f64)
    }

    #[test]
    fn contrib_matches_brute_force() -> anyhow::Result<()> {
        contrib_matches_brute_force_for(Arch::Qwen3)
    }

    #[test]
    fn contrib_matches_brute_force_olmo2() -> anyhow::Result<()> {
        contrib_matches_brute_force_for(Arch::Olmo2)
    }

    /// The scales the trace uses are the real ones: Qwen3 read = 1/rms(x), write = 1; OLMo 2 read = 1,
    /// write = 1/rms(W_down · a) computed here on the host from the captured activations (f64).
    fn check_scales(m: &Model, cap: &TraceCapture) -> anyhow::Result<()> {
        let g = m.geometry();
        let (n, eps) = (g.num_neurons, 1e-6f64);
        for l in 0..g.num_layers {
            let (_, _, dw) = m.mlp_weights(l);
            let dw = mat(dw)?;
            let a = &cap.act[l * n..(l + 1) * n];
            let mo: Vec<f64> = dw.iter().map(|row| row.iter().zip(a).map(|(&w, &x)| w as f64 * x as f64).sum()).collect();
            let inv = 1.0 / (mo.iter().map(|v| v * v).sum::<f64>() / mo.len() as f64 + eps).sqrt();
            match m.arch() {
                Arch::Qwen3 => assert_eq!(cap.write_scale[l], 1.0),
                Arch::Olmo2 => {
                    assert_eq!(cap.inv_rms[l], 1.0);
                    let err = (cap.write_scale[l] as f64 - inv).abs() / inv;
                    assert!(err < 1e-4, "layer {l} write scale {} vs host {inv}", cap.write_scale[l]);
                }
            }
        }
        Ok(())
    }

    fn contrib_matches_brute_force_for(arch: Arch) -> anyhow::Result<()> {
        let (mut m, dir) = tiny_model(arch, 3, "trace")?;
        let g = m.geometry();
        let req = TraceRequest {
            token_ids: vec![3, 17, 5, 29, 11, 8],
            position: 4,
            edits: EditSet {
                neurons: vec![NeuronEdit { layer: 0, index: 2, mode: EditMode::Scale, value: 3.0 }],
                ..Default::default()
            },
            top_sources: 7,
            top_targets: 5,
            focus: Some((2, 9)),
            focus_top: 30,
        };
        let res = trace_model(&mut m, &req)?;
        let edits = DeviceEdits::build(&req.edits, &g, m.dtype(), m.device())?;
        m.clear_cache();
        let cap = m.forward_trace(&req.token_ids[..=req.position], &edits)?;
        m.clear_cache();

        // the capture is the same activations the normal forward reports at that position
        let (_, full) = m.forward(&req.token_ids[..=req.position], 0, &edits, false)?;
        m.clear_cache();
        check_scales(&m, &cap)?;
        let per = g.neurons_per_frame();
        let p = req.position;
        for (x, y) in full.neurons[p * per..(p + 1) * per].iter().zip(&cap.act) {
            assert!((x - y).abs() <= 1e-5 * (1.0 + x.abs()), "capture mismatch {x} vs {y}");
        }

        let check = |w: f32, bf: f64| {
            let err = (w as f64 - bf).abs() / bf.abs().max(1e-6);
            assert!(err < 1e-3, "contrib {w} vs brute {bf} (rel err {err})");
        };
        assert_eq!(res.edges.len(), (g.num_layers - 1) * 7 * 5);
        for e in &res.edges {
            assert_eq!(e.to_layer, e.from_layer + 1);
            check(e.weight, brute(&m, &cap, e.from_layer, e.from_index, e.to_layer, e.to_index)?);
        }
        // sorted by |weight| within each step
        for step in res.edges.chunks(35) {
            assert!(step.windows(2).all(|w| w[0].weight.abs() >= w[1].weight.abs()));
        }
        // explained = Σ_T |Σ_S c| / Σ_T |g|
        for lt in 1..g.num_layers {
            let step: Vec<&TraceEdge> = res.edges.iter().filter(|e| e.to_layer == lt).collect();
            let mut tg: Vec<usize> = step.iter().map(|e| e.to_index).collect();
            tg.sort();
            tg.dedup();
            let num: f64 = tg.iter().map(|&j| step.iter().filter(|e| e.to_index == j).map(|e| e.weight as f64).sum::<f64>().abs()).sum();
            let den: f64 = tg.iter().map(|&j| cap.gate[lt * g.num_neurons + j].abs() as f64).sum();
            assert!(((num / den).clamp(0.0, 1.0) as f32 - res.explained[lt]).abs() < 1e-4);
        }
        assert_eq!(res.explained[0], 0.0);

        // focus: every desk of floors 0 and 1 → (2, 9); top 30 of 48, strongest first, exact values
        assert_eq!(res.focus_edges.len(), 30);
        for e in &res.focus_edges {
            assert_eq!((e.to_layer, e.to_index), (2, 9));
            check(e.weight, brute(&m, &cap, e.from_layer, e.from_index, 2, 9)?);
        }
        let mut all = Vec::new();
        for l in 0..2 {
            for i in 0..g.num_neurons {
                all.push(brute(&m, &cap, l, i, 2, 9)?.abs());
            }
        }
        all.sort_by(|a, b| b.total_cmp(a));
        assert!((res.focus_edges[29].weight.abs() as f64 - all[29]).abs() <= 1e-3 * all[29]);

        // validation
        let mut bad = req.clone();
        bad.position = 6;
        assert!(trace_model(&mut m, &bad).is_err());
        let mut bad = req.clone();
        bad.top_sources = 0;
        assert!(trace_model(&mut m, &bad).is_err());
        let mut bad = req.clone();
        bad.focus = Some((3, 0));
        assert!(trace_model(&mut m, &bad).is_err());
        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }
}
