//! One desk's whole story at one word (see `FollowResult` in types.rs for every field's math).
//!
//! One prefill of token_ids[..=position] with the run's edits (KV cache cleared before and after) reads
//! back, at the last position, every layer's post-edit activations a, gate pre-activations and 1/rms
//! (the trace capture), plus the followed floor's up_proj outputs and MLP-input residual x, the final
//! residual's rms and the logits. Everything else is device matvecs over the weights, with each floor's
//! write vector c[l][:, i] = (w_out[l] ⊙ W_down[l][:, i]) · ws[l] (Qwen3: w_out = 1, ws = 1; OLMo 2:
//! post_feedforward_layernorm weight and 1/rms of the MLP output, see trace.rs) and read row
//! (w_norm ⊙ W_gate[j]) · inv_rms (OLMo 2: w_norm = 1, inv_rms = 1):
//!   • wakes_on   — (E / |E_row|) · (w_norm ⊙ W_gate[j]) over the whole [vocab, hidden] input embedding;
//!   • asked_by   — a_i · ((w_norm ⊙ W_gate[j]) · c[l][:, i]) · inv_rms[layer] for every desk of every
//!                  earlier floor (as trace.rs focus edges);
//!   • heard_by   — h · ((w_norm' ⊙ W_gate'[k]) · c[layer][:, j]) · inv_rms[L'] for every desk of every
//!                  later floor: one W_gate'[N, hidden] @ (w_norm' ⊙ c) per later floor;
//!   • word effect — h · (U[t] · (w_final ⊙ c[layer][:, j])) / rms_final, U = the unembedding (lm_head).
//! asked_by_explained = |Σ listed| / |gate|: the remainder arrives through attention, the word's own
//! embedding and the unlisted desks.
//! The greedy path re-runs the heard_by matvecs once per hop from the previous hop's listener, with that
//! listener's actual activation as the source strength (one small readback per hop after the first).

use crate::edits::DeviceEdits;
use crate::engine::Engine;
use crate::model::{FollowCapture, Model};
use crate::types::{FollowRequest, FollowResult, TraceEdge, WordScore};
use candle_core::{DType, Tensor};

pub const MAX_TOP: usize = 64;
pub const MAX_HOPS: usize = 27;

pub fn follow(engine: &mut Engine, req: &FollowRequest) -> anyhow::Result<FollowResult> {
    let tok = &engine.tok;
    follow_model(&mut engine.model, &|id| tok.token_text(id), req)
}

/// Indices of the `k` largest |v|, strongest first (ties by lower index).
fn top_abs(v: &[f32], k: usize) -> Vec<usize> {
    top_by(v, k, |x| x.abs())
}

/// Indices of the `k` largest key(v), largest first (ties by lower index).
fn top_by(v: &[f32], k: usize, key: impl Fn(f32) -> f32) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    let k = k.min(v.len());
    let cmp = |a: &usize, b: &usize| key(v[*b]).total_cmp(&key(v[*a])).then(a.cmp(b));
    if k < idx.len() && k > 0 {
        idx.select_nth_unstable_by(k - 1, cmp);
    }
    idx.truncate(k);
    idx.sort_by(cmp);
    idx
}

fn validate(req: &FollowRequest, g: &crate::types::Geometry) -> anyhow::Result<()> {
    anyhow::ensure!(!req.token_ids.is_empty(), "follow: token_ids is empty");
    anyhow::ensure!(
        req.position < req.token_ids.len(),
        "follow: position {} out of range (token_ids has {})",
        req.position,
        req.token_ids.len()
    );
    let upto = (req.position + 2).min(req.token_ids.len());
    if let Some(&bad) = req.token_ids[..upto].iter().find(|&&t| t as usize >= g.vocab_size) {
        anyhow::bail!("follow: token id {bad} out of range (vocab {})", g.vocab_size);
    }
    anyhow::ensure!(req.layer < g.num_layers, "follow: layer {} out of range (0..{})", req.layer, g.num_layers);
    anyhow::ensure!(req.index < g.num_neurons, "follow: index {} out of range (0..{})", req.index, g.num_neurons);
    anyhow::ensure!((1..=MAX_TOP).contains(&req.top), "follow: top = {} must be in 1..={MAX_TOP}", req.top);
    anyhow::ensure!(req.hops <= MAX_HOPS, "follow: hops = {} must be in 0..={MAX_HOPS}", req.hops);
    Ok(())
}

/// Device [h] f32: the write vector c = (w_out ⊙ W_down[layer][:, i]) · ws[layer] — what one unit of
/// activation of desk (layer, i) adds to the residual at this position.
fn write_vec(model: &Model, cap: &FollowCapture, layer: usize, i: usize) -> anyhow::Result<Tensor> {
    let col = model.down_proj_column(layer, i)?.to_dtype(DType::F32)?;
    let wo = model.mlp_write_weight(layer).to_dtype(DType::F32)?;
    Ok(((col * wo)? * cap.trace.write_scale[layer] as f64)?)
}

/// Device [nl_later * N] f32: direct push of `strength` × the write vector of (src_layer, src) on every
/// desk of every floor above src_layer (floor-major), each floor scaled by its inv_rms.
fn listeners(model: &Model, cap: &FollowCapture, src_layer: usize, src: usize, strength: f32) -> anyhow::Result<Option<Tensor>> {
    let g = model.geometry();
    if src_layer + 1 >= g.num_layers {
        return Ok(None);
    }
    let col = write_vec(model, cap, src_layer, src)?; // [h]
    let mut per = Vec::with_capacity(g.num_layers - src_layer - 1);
    for lt in src_layer + 1..g.num_layers {
        let (wn, gw, _) = model.mlp_weights(lt);
        let v = (wn.to_dtype(DType::F32)? * &col)?.to_dtype(model.dtype())?.unsqueeze(1)?; // [h, 1]
        let c = gw.matmul(&v)?.to_dtype(DType::F32)?.flatten_all()?; // [N]
        per.push((c * (strength as f64 * cap.trace.inv_rms[lt] as f64))?);
    }
    Ok(Some(Tensor::cat(&per, 0)?))
}

/// `text` decodes a token id for display (`Tok::token_text`).
pub(crate) fn follow_model(model: &mut Model, text: &dyn Fn(u32) -> String, req: &FollowRequest) -> anyhow::Result<FollowResult> {
    let g = model.geometry();
    validate(req, &g)?;
    let (nl, n, (layer, j)) = (g.num_layers, g.num_neurons, (req.layer, req.index));
    let edits = if req.edits.is_empty() {
        DeviceEdits::none(nl)
    } else {
        DeviceEdits::build(&req.edits, &g, model.dtype(), model.device())?
    };
    let tokens = &req.token_ids[..=req.position];
    model.clear_cache();
    let cap = model.forward_follow(tokens, &edits, layer);
    model.clear_cache();
    let cap = cap?;
    let dev = model.device().clone();
    let dt = model.dtype();

    // 1–4: the desk's own numbers, from the readback.
    let inv_rms = cap.trace.inv_rms[layer];
    let rms = 1.0 / inv_rms;
    let memo_norm = cap.x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt() as f32;
    let gate = cap.trace.gate[layer * n + j];
    let sigma = 1.0 / (1.0 + (-gate).exp());
    let silu = gate * sigma;
    let up = cap.up[j];
    let h_raw = silu * up;
    let h = cap.trace.act[layer * n + j];

    let next_token = match req.token_ids.get(req.position + 1) {
        Some(&t) => t,
        None => top_by(&cap.logits, 1, |x| x)[0] as u32,
    };
    let cands: Vec<u32> = top_by(&cap.logits, req.top, |x| x).into_iter().map(|i| i as u32).collect();

    // Device work, one readback: [wakes V | asked_by layer*N | heard_by (nl-layer-1)*N | effects 1+top | |col|].
    let (wn, gw, _) = model.mlp_weights(layer);
    let row = (gw.narrow(0, j, 1)?.to_dtype(DType::F32)? * wn.to_dtype(DType::F32)?.unsqueeze(0)?)?; // [1, h]
    let wakes = model
        .embedding()
        .matmul(&row.to_dtype(dt)?.t()?)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .mul(model.embed_inv_norm()?)?; // [V]
    let mut parts: Vec<Tensor> = vec![wakes];
    if layer > 0 {
        let row_m = row.to_dtype(dt)?;
        for l in 0..layer {
            let (_, _, dw) = model.mlp_weights(l);
            let wo = model.mlp_write_weight(l).to_dtype(dt)?.unsqueeze(0)?; // [1, h]
            let a = Tensor::from_slice(&cap.trace.act[l * n..(l + 1) * n], n, &dev)?;
            let c = row_m.broadcast_mul(&wo)?.matmul(dw)?.to_dtype(DType::F32)?.flatten_all()?.mul(&a)?; // [N]
            parts.push(((c * inv_rms as f64)? * cap.trace.write_scale[l] as f64)?);
        }
    }
    let heard = listeners(model, &cap, layer, j, h)?;
    if let Some(t) = &heard {
        parts.push(t.clone());
    }
    let col = write_vec(model, &cap, layer, j)?; // [h]
    let wcol = (model.final_norm_weight().to_dtype(DType::F32)? * &col)?.unsqueeze(1)?; // [h, 1]
    let mut ids = vec![next_token];
    ids.extend_from_slice(&cands);
    let ids_t = Tensor::from_vec(ids.clone(), ids.len(), &dev)?;
    let rows = model.unembed().index_select(&ids_t, 0)?.to_dtype(DType::F32)?; // [k, h]
    let eff = (rows.matmul(&wcol)?.flatten_all()? * (h as f64 / cap.rms_final as f64))?;
    parts.push(eff);
    parts.push(col.sqr()?.sum_all()?.sqrt()?.reshape(1)?);
    let all = Tensor::cat(&parts, 0)?.to_vec1::<f32>()?;

    let v = g.vocab_size;
    let (wakes, rest) = all.split_at(v);
    let (asked, rest) = rest.split_at(layer * n);
    let (heard_v, rest) = rest.split_at((nl - layer - 1) * n);
    let (eff, rest) = rest.split_at(ids.len());
    let column_norm = rest[0];

    let wakes_on = top_by(wakes, req.top, |x| x)
        .into_iter()
        .map(|i| WordScore { id: i as u32, token: text(i as u32), score: wakes[i] })
        .collect();

    let asked_by: Vec<TraceEdge> = top_abs(asked, req.top)
        .into_iter()
        .map(|k| TraceEdge { from_layer: k / n, from_index: k % n, to_layer: layer, to_index: j, weight: asked[k] })
        .collect();
    let listed: f64 = asked_by.iter().map(|e| e.weight as f64).sum();
    let asked_by_explained = if gate != 0.0 { (listed.abs() / gate.abs() as f64).clamp(0.0, 1.0) as f32 } else { 0.0 };

    let heard_edge = |src_l: usize, src: usize, vals: &[f32], k: usize| TraceEdge {
        from_layer: src_l,
        from_index: src,
        to_layer: src_l + 1 + k / n,
        to_index: k % n,
        weight: vals[k],
    };
    let heard_by: Vec<TraceEdge> = top_abs(heard_v, req.top).into_iter().map(|k| heard_edge(layer, j, heard_v, k)).collect();

    // Greedy path: hop 1 is the strongest listener overall; each further hop starts at the previous
    // listener with its actual activation.
    let mut path = Vec::with_capacity(req.hops);
    if req.hops > 0 && !heard_v.is_empty() {
        let mut best = heard_edge(layer, j, heard_v, top_abs(heard_v, 1)[0]);
        while best.weight != 0.0 {
            path.push(best.clone());
            if path.len() == req.hops || best.to_layer + 1 >= nl {
                break;
            }
            let (sl, si) = (best.to_layer, best.to_index);
            let a = cap.trace.act[sl * n + si];
            let Some(t) = listeners(model, &cap, sl, si, a)? else { break };
            let vals = t.to_vec1::<f32>()?;
            best = heard_edge(sl, si, &vals, top_abs(&vals, 1)[0]);
        }
    }

    let next_effect = eff[0];
    let mut word_effects: Vec<WordScore> = cands
        .iter()
        .zip(&eff[1..])
        .map(|(&id, &score)| WordScore { id, token: text(id), score })
        .collect();
    word_effects.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));

    Ok(FollowResult {
        layer,
        index: j,
        position: req.position,
        token_id: req.token_ids[req.position],
        memo_norm,
        rms,
        gate,
        sigma,
        silu,
        up,
        h_raw,
        h,
        wakes_on,
        asked_by,
        asked_by_explained,
        column_norm,
        write_norm: h.abs() * column_norm,
        heard_by,
        path,
        next_token,
        next_effect,
        word_effects,
    })
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
    fn vec1(t: &Tensor) -> anyhow::Result<Vec<f32>> {
        Ok(t.to_dtype(DType::F32)?.to_vec1::<f32>()?)
    }

    /// Brute force direct push: a · Σ_k w_norm[lt][k] · W_gate[lt][tj][k] · w_out[l][k] · W_down[l][k][i]
    /// · inv_rms[lt] · ws[l], f64.
    fn brute(m: &Model, tc: &crate::model::TraceCapture, a: f32, l: usize, i: usize, lt: usize, tj: usize) -> anyhow::Result<f64> {
        let (wn, gw, _) = m.mlp_weights(lt);
        let (_, _, dw) = m.mlp_weights(l);
        let (wn, gw, dw) = (vec1(wn)?, mat(gw)?, mat(dw)?);
        let wo = vec1(m.mlp_write_weight(l))?;
        let dot: f64 = (0..wn.len()).map(|k| wn[k] as f64 * gw[tj][k] as f64 * wo[k] as f64 * dw[k][i] as f64).sum();
        Ok(a as f64 * dot * tc.inv_rms[lt] as f64 * tc.write_scale[l] as f64)
    }

    fn close(w: f32, bf: f64, what: &str) {
        let err = (w as f64 - bf).abs() / bf.abs().max(1e-4);
        assert!(err < 1e-3, "{what}: {w} vs brute {bf} (rel err {err})");
    }

    #[test]
    fn follow_matches_brute_force() -> anyhow::Result<()> {
        follow_matches_brute_force_for(Arch::Qwen3)
    }

    #[test]
    fn follow_matches_brute_force_olmo2() -> anyhow::Result<()> {
        follow_matches_brute_force_for(Arch::Olmo2)
    }

    fn follow_matches_brute_force_for(arch: Arch) -> anyhow::Result<()> {
        let (mut m, dir) = tiny_model(arch, 4, "follow")?;
        let g = m.geometry();
        let (nl, n) = (g.num_layers, g.num_neurons);
        let text = |id: u32| format!("t{id}");
        let tokens = vec![3u32, 17, 5, 29, 11, 8];
        let req = FollowRequest {
            token_ids: tokens.clone(),
            position: 3,
            edits: EditSet::default(),
            layer: 1,
            index: 7,
            top: 10,
            hops: 5,
        };
        let r = follow_model(&mut m, &text, &req)?;
        let edits = DeviceEdits::none(nl);
        m.clear_cache();
        let (logits, full) = m.forward(&tokens[..=req.position], 0, &edits, true)?;
        m.clear_cache();
        let cap = m.forward_follow(&tokens[..=req.position], &edits, req.layer)?;
        m.clear_cache();
        let per = g.neurons_per_frame();
        let act = |l: usize, i: usize| full.neurons[req.position * per + l * n + i];
        let inv = &cap.trace;

        // 1–4: own numbers
        assert!((r.h_raw - r.silu * r.up).abs() <= 1e-5 * (1.0 + r.h_raw.abs()));
        assert!((r.silu - r.gate / (1.0 + (-r.gate).exp())).abs() <= 1e-6);
        assert!((r.h_raw - act(1, 7)).abs() <= 1e-5 * (1.0 + r.h_raw.abs()), "h_raw {} vs capture {}", r.h_raw, act(1, 7));
        assert!((r.h - r.h_raw).abs() <= 1e-5 * (1.0 + r.h.abs()), "unedited h {} vs h_raw {}", r.h, r.h_raw);
        let x = &cap.x;
        let ms: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len() as f64;
        match arch {
            Arch::Qwen3 => close(r.rms, (ms + 1e-6).sqrt(), "rms"),
            Arch::Olmo2 => assert_eq!(r.rms, 1.0, "OLMo 2 MLPs read the raw residual"),
        }
        close(r.memo_norm, (ms * x.len() as f64).sqrt(), "memo_norm");
        let (wn, gw, dw) = m.mlp_weights(1);
        let (wn, gw, dw) = (vec1(wn)?, mat(gw)?, mat(dw)?);
        let uw = mat(m.up_weight(1))?;
        let xt: Vec<f64> = (0..x.len()).map(|k| x[k] as f64 / r.rms as f64 * wn[k] as f64).collect();
        close(r.gate, (0..x.len()).map(|k| gw[7][k] as f64 * xt[k]).sum(), "gate");
        close(r.up, (0..x.len()).map(|k| uw[7][k] as f64 * xt[k]).sum(), "up");
        let wo1 = vec1(m.mlp_write_weight(1))?;
        let ws1 = cap.trace.write_scale[1] as f64;
        let cn: f64 = (0..x.len()).map(|k| (wo1[k] as f64 * dw[k][7] as f64).powi(2)).sum::<f64>().sqrt() * ws1;
        close(r.column_norm, cn, "column_norm");
        close(r.write_norm, r.h.abs() as f64 * cn, "write_norm");

        // wakes_on: normalized embedding · (w_norm ⊙ W_gate[j]), descending
        let emb = mat(m.embedding())?;
        let wake = |t: usize| -> f64 {
            let e = &emb[t];
            let en: f64 = e.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
            (0..e.len()).map(|k| e[k] as f64 * wn[k] as f64 * gw[7][k] as f64).sum::<f64>() / en
        };
        let mut ws: Vec<f64> = (0..g.vocab_size).map(wake).collect();
        assert_eq!(r.wakes_on.len(), 10);
        for w in &r.wakes_on {
            close(w.score, ws[w.id as usize], "wakes_on");
            assert_eq!(w.token, format!("t{}", w.id));
        }
        ws.sort_by(|a, b| b.total_cmp(a));
        close(r.wakes_on[9].score, ws[9], "wakes_on 10th");
        assert!(r.wakes_on.windows(2).all(|w| w[0].score >= w[1].score));

        // asked_by: every desk of floor 0 into (1, 7)
        assert_eq!(r.asked_by.len(), 10);
        let mut sum = 0f64;
        for e in &r.asked_by {
            assert_eq!((e.from_layer, e.to_layer, e.to_index), (0, 1, 7));
            close(e.weight, brute(&m, inv, act(0, e.from_index), 0, e.from_index, 1, 7)?, "asked_by");
            sum += e.weight as f64;
        }
        let mut all: Vec<f64> = (0..n).map(|i| brute(&m, inv, act(0, i), 0, i, 1, 7).unwrap().abs()).collect();
        all.sort_by(|a, b| b.total_cmp(a));
        close(r.asked_by[9].weight.abs(), all[9], "asked_by 10th");
        assert!((r.asked_by_explained - (sum.abs() / r.gate.abs() as f64).clamp(0.0, 1.0) as f32).abs() < 1e-5);

        // heard_by: every desk of floors 2, 3
        assert_eq!(r.heard_by.len(), 10);
        for e in &r.heard_by {
            assert_eq!((e.from_layer, e.from_index), (1, 7));
            assert!(e.to_layer >= 2 && e.to_layer < nl);
            close(e.weight, brute(&m, inv, r.h, 1, 7, e.to_layer, e.to_index)?, "heard_by");
        }
        let mut all = Vec::new();
        for lt in 2..nl {
            for k in 0..n {
                all.push(brute(&m, inv, r.h, 1, 7, lt, k)?.abs());
            }
        }
        all.sort_by(|a, b| b.total_cmp(a));
        close(r.heard_by[0].weight.abs(), all[0], "heard_by 1st");
        close(r.heard_by[9].weight.abs(), all[9], "heard_by 10th");

        // path: starts at the desk, hop 1 = strongest listener, each hop starts where the previous ended
        assert!(!r.path.is_empty() && r.path.len() <= 5);
        assert_eq!(r.path[0], r.heard_by[0]);
        assert_eq!(*r.path.last().map(|e| &e.to_layer).unwrap(), nl - 1, "path runs to the top floor (no zero hop here)");
        for w in r.path.windows(2) {
            assert_eq!((w[1].from_layer, w[1].from_index), (w[0].to_layer, w[0].to_index));
            assert!(w[1].to_layer > w[1].from_layer);
        }
        for e in &r.path[1..] {
            let a = act(e.from_layer, e.from_index);
            close(e.weight, brute(&m, inv, a, e.from_layer, e.from_index, e.to_layer, e.to_index)?, "path hop");
            let best = (e.from_layer + 1..nl)
                .flat_map(|lt| (0..n).map(move |k| (lt, k)))
                .map(|(lt, k)| brute(&m, inv, a, e.from_layer, e.from_index, lt, k).unwrap().abs())
                .fold(0f64, f64::max);
            close(e.weight.abs(), best, "path hop is the strongest");
        }

        // word effects: h · (E[t] · (w_final ⊙ col)) / rms_final
        let resid = mat(full.residuals.as_ref().unwrap())?;
        let xf = &resid[nl - 1];
        let rf = (xf.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / xf.len() as f64 + 1e-6).sqrt();
        let wf = vec1(m.final_norm_weight())?;
        let un = mat(m.unembed())?;
        let eff = |t: u32| -> f64 {
            let e = &un[t as usize];
            r.h as f64 * ws1 * (0..e.len()).map(|k| e[k] as f64 * wf[k] as f64 * wo1[k] as f64 * dw[k][7] as f64).sum::<f64>() / rf
        };
        assert_eq!(r.next_token, tokens[req.position + 1]);
        close(r.next_effect, eff(r.next_token), "next_effect");
        assert_eq!(r.word_effects.len(), 10);
        let mut cand: Vec<usize> = (0..logits.len()).collect();
        cand.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let mut ids: Vec<u32> = r.word_effects.iter().map(|w| w.id).collect();
        ids.sort();
        let mut want: Vec<u32> = cand[..10].iter().map(|&i| i as u32).collect();
        want.sort();
        assert_eq!(ids, want);
        for w in &r.word_effects {
            close(w.score, eff(w.id), "word_effect");
        }
        assert!(r.word_effects.windows(2).all(|w| w[0].score >= w[1].score));

        // last position: next_token = argmax; edits scale h; last floor has no listeners
        let mut q = req.clone();
        q.position = tokens.len() - 1;
        q.layer = nl - 1;
        q.edits = EditSet { neurons: vec![NeuronEdit { layer: nl - 1, index: 7, mode: EditMode::Scale, value: 3.0 }], ..Default::default() };
        let r2 = follow_model(&mut m, &text, &q)?;
        assert!((r2.h - 3.0 * r2.h_raw).abs() <= 1e-5 * (1.0 + r2.h.abs()));
        assert!(r2.heard_by.is_empty() && r2.path.is_empty());
        let (lg, _) = m.forward(&tokens, 0, &DeviceEdits::build(&q.edits, &g, m.dtype(), m.device())?, false)?;
        m.clear_cache();
        let am = (0..lg.len()).max_by(|&a, &b| lg[a].total_cmp(&lg[b]).then(b.cmp(&a))).unwrap() as u32;
        assert_eq!(r2.next_token, am);
        q.layer = 0;
        q.hops = 0;
        let r3 = follow_model(&mut m, &text, &q)?;
        assert!(r3.asked_by.is_empty() && r3.asked_by_explained == 0.0 && r3.path.is_empty());

        // validation
        for bad in [
            FollowRequest { position: 6, ..req.clone() },
            FollowRequest { layer: nl, ..req.clone() },
            FollowRequest { index: n, ..req.clone() },
            FollowRequest { top: 0, ..req.clone() },
            FollowRequest { top: 65, ..req.clone() },
            FollowRequest { hops: 28, ..req.clone() },
            FollowRequest { token_ids: vec![3, 99], position: 0, ..req.clone() },
        ] {
            assert!(follow_model(&mut m, &text, &bad).is_err(), "accepted {bad:?}");
        }
        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }
}
