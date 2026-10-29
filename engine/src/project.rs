//! How much of one direction d̂ ("highlighter") the memo carries at every residual site and position of
//! a run, before and after that site's edits — the numbers behind the math panel.
//!
//! One prefill of token_ids with the run's edits. At each site (0 = embedding output, l+1 = output of
//! layer l) and every position, `before` = x·d̂ of the memo arriving at the site, `after` = x·d̂ once the
//! site's resid edit has been applied (identical when the site has no edit). The dots are taken on the
//! device (f32 accumulation of the model-dtype residual), stacked to [T, sites] and read back once.
//! The worked sample returns the actual leading numbers of x, d̂ and x_new at one (site, position).

use crate::edits::DeviceEdits;
use crate::engine::Engine;
use crate::model::{Model, ProjectSink};
use crate::types::{ProjectRequest, ProjectResult, ProjectSample};
use candle_core::{DType, Tensor};

pub fn project(engine: &mut Engine, req: &ProjectRequest) -> anyhow::Result<ProjectResult> {
    project_model(&mut engine.model, req)
}

/// Unit vector of `d` (f64 norm) and the unnormalized length.
fn normalize(d: &[f32], hidden: usize) -> anyhow::Result<(Vec<f32>, f32)> {
    anyhow::ensure!(d.len() == hidden, "project: direction has {} floats, expected hidden_size {hidden}", d.len());
    anyhow::ensure!(d.iter().all(|v| v.is_finite()), "project: direction must be finite");
    let n = d.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    anyhow::ensure!(n > 1e-12, "project: direction has zero norm");
    Ok((d.iter().map(|&x| (x as f64 / n) as f32).collect(), n as f32))
}

pub(crate) fn project_model(model: &mut Model, req: &ProjectRequest) -> anyhow::Result<ProjectResult> {
    let g = model.geometry();
    anyhow::ensure!(!req.token_ids.is_empty(), "project: token_ids is empty");
    if let Some(&bad) = req.token_ids.iter().find(|&&t| t as usize >= g.vocab_size) {
        anyhow::bail!("project: token id {bad} out of range (vocab {})", g.vocab_size);
    }
    let (dhat, d_norm) = normalize(&req.direction, g.hidden_size)?;
    let t = req.token_ids.len();
    let sites = g.num_layers + 1;
    let sample = match (req.sample_site, req.sample_position) {
        (Some(s), Some(p)) => {
            anyhow::ensure!(s < sites, "project: sample_site {s} out of range (0..{sites})");
            anyhow::ensure!(p < t, "project: sample_position {p} out of range (token_ids has {t})");
            anyhow::ensure!(req.sample_dims > 0, "project: sample_dims must be at least 1");
            Some((s, p, req.sample_dims.min(g.hidden_size)))
        }
        (None, None) => None,
        _ => anyhow::bail!("project: sample_site and sample_position must be given together"),
    };
    let edits = if req.edits.is_empty() {
        DeviceEdits::none(g.num_layers)
    } else {
        DeviceEdits::build(&req.edits, &g, model.dtype(), model.device())?
    };
    let dev_d = Tensor::from_vec(dhat.clone(), (g.hidden_size, 1), model.device())?.to_dtype(DType::F32)?;

    model.clear_cache();
    let cap = model.forward_project(&req.token_ids, &edits, ProjectSink::new(dev_d, sample));
    model.clear_cache();
    let cap = cap?;
    anyhow::ensure!(cap.before.len() == t * sites && cap.after.len() == t * sites, "project: unexpected capture size");

    let sample = match (sample, cap.sample) {
        (Some((site, position, dims)), Some(v)) => {
            anyhow::ensure!(v.len() == 2 * dims + 2, "project: unexpected sample size");
            Some(ProjectSample {
                site,
                position,
                dot: cap.before[position * sites + site],
                dot_after: cap.after[position * sites + site],
                x: v[..dims].to_vec(),
                d_hat: dhat[..dims].to_vec(),
                x_new: v[dims..2 * dims].to_vec(),
                x_norm: v[2 * dims],
                x_new_norm: v[2 * dims + 1],
                d_norm,
            })
        }
        (None, None) => None,
        _ => anyhow::bail!("project: sample was not captured"),
    };
    Ok(ProjectResult { sites, positions: t, before: cap.before, after: cap.after, sample })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Arch;
    use crate::model::test_util::tiny_model;
    use crate::types::{DirectionEdit, DirectionMode, EditSet};

    fn dir_vec(h: usize, seed: f32) -> Vec<f32> {
        (0..h).map(|i| ((i as f32 + 1.0) * seed).sin() + 0.1 * i as f32 - 0.5).collect()
    }

    fn ablate(v: &[f32], sites: Vec<usize>) -> DirectionEdit {
        DirectionEdit { label: "t".into(), vector: v.to_vec(), sites, mode: DirectionMode::Ablate, value: 0.0 }
    }

    fn base(dir: &[f32]) -> ProjectRequest {
        ProjectRequest {
            token_ids: vec![3, 17, 5, 29, 11, 8],
            edits: EditSet::default(),
            direction: dir.to_vec(),
            sample_site: None,
            sample_position: None,
            sample_dims: 6,
        }
    }

    #[test]
    fn project_semantics() -> anyhow::Result<()> {
        project_semantics_for(Arch::Qwen3)
    }

    #[test]
    fn project_semantics_olmo2() -> anyhow::Result<()> {
        project_semantics_for(Arch::Olmo2)
    }

    fn project_semantics_for(arch: Arch) -> anyhow::Result<()> {
        let (mut m, dir) = tiny_model(arch, 3, "project")?;
        let g = m.geometry();
        let (sites, t, h) = (g.num_layers + 1, 6usize, g.hidden_size);
        let d = dir_vec(h, 0.7);
        let dn = d.iter().map(|x| x * x).sum::<f32>().sqrt();

        // no edits: before == after everywhere; scaling d changes only d_norm and nothing else
        let mut req = base(&d);
        req.sample_site = Some(2);
        req.sample_position = Some(4);
        let r0 = project_model(&mut m, &req)?;
        assert_eq!((r0.sites, r0.positions, r0.before.len(), r0.after.len()), (sites, t, t * sites, t * sites));
        assert_eq!(r0.before, r0.after);
        assert!(r0.before.iter().any(|v| v.abs() > 1e-3));
        let s0 = r0.sample.as_ref().unwrap();
        assert_eq!((s0.site, s0.position), (2, 4));
        assert!((s0.d_norm - dn).abs() < 1e-5);
        assert_eq!(s0.dot, r0.before[4 * sites + 2]);
        assert_eq!(s0.dot_after, s0.dot);
        assert_eq!(s0.x, s0.x_new);
        assert!((s0.x_norm - s0.x_new_norm).abs() < 1e-6);
        let mut req5 = req.clone();
        req5.direction = d.iter().map(|x| x * 5.0).collect();
        let r5 = project_model(&mut m, &req5)?;
        for (a, b) in r5.before.iter().zip(&r0.before) {
            assert!((a - b).abs() < 1e-4 * (1.0 + b.abs()));
        }
        assert!((r5.sample.as_ref().unwrap().d_norm - 5.0 * dn).abs() < 1e-4);

        // ablated at a subset of sites: before/after equal on the unedited ones, after == 0 on the edited ones;
        // the first edited site's `before` still matches the unedited run
        let mut req = base(&d);
        req.edits = EditSet { directions: vec![ablate(&d, vec![1, 3])], ..Default::default() };
        let r1 = project_model(&mut m, &req)?;
        for p in 0..t {
            for s in 0..sites {
                let (b, a) = (r1.before[p * sites + s], r1.after[p * sites + s]);
                if s == 1 || s == 3 {
                    assert!(a.abs() < 1e-4, "site {s} pos {p}: after = {a}");
                } else {
                    assert_eq!(b, a, "site {s} pos {p}");
                }
            }
        }
        for p in 0..t {
            assert!((r1.before[p * sites] - r0.before[p * sites]).abs() < 1e-6);
            assert!((r1.before[p * sites + 1] - r0.before[p * sites + 1]).abs() < 1e-4);
        }
        // a downstream site's `before` really is changed by the upstream ablation
        assert!(r1.before[5 * sites + 2] != r0.before[5 * sites + 2]);

        // ablated at ALL sites (empty sites list): after ≈ 0 everywhere, and the worked sample is
        // x_new = x − (x·d̂)d̂ on the returned dims
        let mut req = base(&d);
        req.edits = EditSet { directions: vec![ablate(&d, vec![])], ..Default::default() };
        req.sample_site = Some(3);
        req.sample_position = Some(2);
        req.sample_dims = 10;
        let ra = project_model(&mut m, &req)?;
        let worst = ra.after.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!(worst < 1e-4, "max |after| = {worst}");
        assert!(ra.before.iter().any(|v| v.abs() > 1e-3));
        let s = ra.sample.as_ref().unwrap();
        assert_eq!((s.x.len(), s.d_hat.len(), s.x_new.len()), (10, 10, 10));
        assert_eq!(s.dot, ra.before[2 * sites + 3]);
        for i in 0..10 {
            let want = s.x[i] - s.dot * s.d_hat[i];
            assert!((s.x_new[i] - want).abs() < 1e-4, "dim {i}: {} vs {want}", s.x_new[i]);
        }
        assert!((s.d_hat[0] - d[0] / dn).abs() < 1e-6);
        assert!(s.dot_after.abs() < 1e-4);
        assert!(s.x_new_norm <= s.x_norm + 1e-5);
        assert!((s.x_norm * s.x_norm - s.x_new_norm * s.x_new_norm - s.dot * s.dot).abs() < 1e-3 * s.x_norm * s.x_norm);

        // the cache is clean afterwards: a repeat gives identical numbers
        let again = project_model(&mut m, &req)?;
        assert_eq!(again.before, ra.before);

        // validation
        let mut bad = |f: &dyn Fn(&mut ProjectRequest)| {
            let mut r = base(&d);
            f(&mut r);
            project_model(&mut m, &r).is_err()
        };
        assert!(bad(&|r| r.direction.pop().map(|_| ()).unwrap_or(())));
        assert!(bad(&|r| r.direction = vec![0.0; h]));
        assert!(bad(&|r| r.direction[3] = f32::NAN));
        assert!(bad(&|r| r.token_ids.clear()));
        assert!(bad(&|r| r.token_ids[0] = 40));
        assert!(bad(&|r| { r.sample_site = Some(sites); r.sample_position = Some(0); }));
        assert!(bad(&|r| { r.sample_site = Some(0); r.sample_position = Some(t); }));
        assert!(bad(&|r| r.sample_site = Some(0)));
        let _ = std::fs::remove_dir_all(dir);
        Ok(())
    }
}
