//! Difference-of-means contrast between two prompt sets (Arditi et al. 2024, "Refusal in language
//! models is mediated by a single direction"). Each prompt is prefilled through the chat template (no
//! generation); the residual stream at the LAST prompt token is read at every site (0 = embedding
//! output, l+1 = output of layer l) together with the MLP activations there. Per site the direction is
//! mean(A) − mean(B); the separation is Cohen's d of the per-prompt projections onto its unit vector.

use crate::edits::DeviceEdits;
use crate::engine::Engine;
use crate::types::{ContrastRequest, ContrastResult};

/// Per-prompt readout: residuals [(L+1) * hidden] (site-major) and neurons [L * N] (layer-major).
fn readout(engine: &mut Engine, prompt: &str, thinking: bool, none: &DeviceEdits) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let g = engine.geometry;
    engine.model.clear_cache();
    let ids = engine.tok.encode_chat(prompt, thinking)?;
    anyhow::ensure!(!ids.is_empty(), "empty prompt");
    let (_logits, cap) = engine.model.forward(&ids, 0, none, true)?;
    let r = cap.residuals.ok_or_else(|| anyhow::anyhow!("forward returned no residuals"))?;
    let e0 = cap.embed_last.ok_or_else(|| anyhow::anyhow!("forward returned no embedding site"))?;
    let resid = candle_core::Tensor::cat(&[&e0, &r], 0)?
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    anyhow::ensure!(resid.len() == (g.num_layers + 1) * g.hidden_size, "unexpected residual size");
    let per = g.neurons_per_frame();
    let last = ids.len() - 1;
    let neurons = cap.neurons[last * per..(last + 1) * per].to_vec();
    Ok((resid, neurons))
}

pub fn contrast(engine: &mut Engine, req: &ContrastRequest) -> anyhow::Result<ContrastResult> {
    let a: Vec<&str> = req.a.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    let b: Vec<&str> = req.b.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    anyhow::ensure!(!a.is_empty(), "contrast: prompt set A is empty");
    anyhow::ensure!(!b.is_empty(), "contrast: prompt set B is empty");
    let g = engine.geometry;
    let (sites, h, per) = (g.num_layers + 1, g.hidden_size, g.neurons_per_frame());
    let none = if req.edits.is_empty() {
        DeviceEdits::none(g.num_layers)
    } else {
        DeviceEdits::build(&req.edits, &g, engine.model.dtype(), engine.model.device())?
    };

    let mut run = |set: &[&str]| -> anyhow::Result<(Vec<Vec<f32>>, Vec<f64>, Vec<f64>)> {
        let mut rs = Vec::with_capacity(set.len());
        let mut sum_r = vec![0f64; sites * h];
        let mut sum_n = vec![0f64; per];
        for p in set {
            let (r, n) = readout(engine, p, req.thinking, &none)?;
            sum_r.iter_mut().zip(&r).for_each(|(s, &v)| *s += v as f64);
            sum_n.iter_mut().zip(&n).for_each(|(s, &v)| *s += v as f64);
            rs.push(r);
        }
        let k = set.len() as f64;
        sum_r.iter_mut().for_each(|v| *v /= k);
        sum_n.iter_mut().for_each(|v| *v /= k);
        Ok((rs, sum_r, sum_n))
    };
    let (ra, mean_ra, mean_na) = run(&a)?;
    let (rb, mean_rb, mean_nb) = run(&b)?;
    engine.model.clear_cache();

    let mut directions = Vec::with_capacity(sites);
    let mut separation = Vec::with_capacity(sites);
    let mut norms = Vec::with_capacity(sites);
    for s in 0..sites {
        let span = s * h..(s + 1) * h;
        let diff: Vec<f64> = mean_ra[span.clone()].iter().zip(&mean_rb[span.clone()]).map(|(x, y)| x - y).collect();
        let norm = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
        let sep = if norm > 1e-9 {
            let u: Vec<f64> = diff.iter().map(|v| v / norm).collect();
            let proj = |rs: &[Vec<f32>]| -> Vec<f64> {
                rs.iter().map(|r| r[span.clone()].iter().zip(&u).map(|(&x, y)| x as f64 * y).sum()).collect()
            };
            cohen_d(&proj(&ra), &proj(&rb))
        } else {
            0.0
        };
        directions.push(diff.iter().map(|&v| v as f32).collect());
        separation.push(sep as f32);
        norms.push(norm as f32);
    }
    let neuron_diff = mean_na.iter().zip(&mean_nb).map(|(x, y)| (x - y) as f32).collect();
    Ok(ContrastResult {
        num_layers: g.num_layers,
        hidden_size: h,
        directions,
        separation,
        norms,
        neuron_diff,
        count_a: a.len(),
        count_b: b.len(),
    })
}

/// Cohen's d with pooled standard deviation. Finite always (JSON has no NaN): a zero spread with a
/// nonzero gap is reported as the cap 1000.
pub(crate) fn cohen_d(pa: &[f64], pb: &[f64]) -> f64 {
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (ma, mb) = (mean(pa), mean(pb));
    let ss = |v: &[f64], m: f64| v.iter().map(|x| (x - m) * (x - m)).sum::<f64>();
    let dof = (pa.len() + pb.len()).saturating_sub(2).max(1) as f64;
    let sd = ((ss(pa, ma) + ss(pb, mb)) / dof).sqrt();
    let gap = ma - mb;
    if sd > 1e-9 {
        (gap / sd).clamp(-1000.0, 1000.0)
    } else if gap.abs() > 1e-9 {
        1000.0 * gap.signum()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohen_d_basic() {
        let d = cohen_d(&[2.0, 3.0, 4.0], &[0.0, 1.0, 2.0]);
        assert!((d - 2.0).abs() < 1e-12);
        assert_eq!(cohen_d(&[1.0], &[1.0]), 0.0);
        assert_eq!(cohen_d(&[2.0], &[1.0]), 1000.0);
    }

    #[test]
    #[ignore]
    fn ablating_direction_lowers_downstream_separation() -> anyhow::Result<()> {
        use crate::types::{DirectionEdit, DirectionMode, EditSet};
        let dir = std::path::PathBuf::from(std::env::var("HOME")?)
            .join("Library/Application Support/NeuroScope/models/Qwen3-0.6B");
        let mut eng = Engine::open(&dir)?;
        let a: Vec<String> = ["Reply in French: what is the sun?", "Reply in French: name a fruit.", "Reply in French: how do birds fly?", "Reply in French: what is rain?"].iter().map(|s| s.to_string()).collect();
        let b: Vec<String> = ["Reply in English: what is the sun?", "Reply in English: name a fruit.", "Reply in English: how do birds fly?", "Reply in English: what is rain?"].iter().map(|s| s.to_string()).collect();
        let mut req = ContrastRequest { a, b, thinking: false, edits: EditSet::default() };
        let base = contrast(&mut eng, &req)?;
        let mid = base.separation.len() / 2;
        req.edits.directions.push(DirectionEdit {
            label: "A-B".into(),
            vector: base.directions[mid].clone(),
            sites: vec![],
            mode: DirectionMode::Ablate,
            value: 0.0,
        });
        let after = contrast(&mut eng, &req)?;
        let down = mid + 1..base.separation.len();
        let mean = |v: &[f32]| v[down.clone()].iter().map(|x| x.abs()).sum::<f32>() / down.len() as f32;
        let (b0, a0) = (mean(&base.separation), mean(&after.separation));
        println!("mid site {mid}: separation at mid before={:.3} after={:.3}; mean |d| downstream before={b0:.3} after={a0:.3}", base.separation[mid], after.separation[mid]);
        assert!(a0 < b0, "ablation must lower downstream separation");
        Ok(())
    }
}
