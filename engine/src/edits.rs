//! Turns an EditSet into per-layer device tensors the forward pass applies.
//! MLP: h' = h * scale + offset  (ablate: scale 0, offset 0 · scale: scale v · pin: scale 0, offset v)
//! Heads: per-head multiplier on the attention output before o_proj (ablate 0 · scale v).
//! Directions: per residual site, ablated unit vectors are orthonormalized (Gram-Schmidt in f64) into
//! B [k, hidden] and steering vectors are summed into one [hidden] add (see `ResidEdit`).

use crate::types::{DirectionMode, EditMode, EditSet, Geometry};
use candle_core::{DType, Device, Tensor};

pub struct MlpEdit {
    /// [num_neurons], model dtype
    pub scale: Tensor,
    /// [num_neurons], model dtype
    pub offset: Tensor,
}

/// Residual-stream edit at one site. Applied as x ← x − (x·Bᵀ)·B then x ← x + add.
pub struct ResidEdit {
    /// [k, hidden] orthonormal rows (model dtype) to project out; None = no ablation here.
    pub ablate: Option<Tensor>,
    /// [hidden] vector to add (sum of value·d̂ over Add edits); None = no steering here.
    pub add: Option<Tensor>,
}

pub struct DeviceEdits {
    /// len num_layers; None = layer untouched (no extra ops in the forward pass)
    pub mlp: Vec<Option<MlpEdit>>,
    /// len num_layers; Some([num_heads] multiplier, model dtype)
    pub heads: Vec<Option<Tensor>>,
    /// len num_layers + 1; index 0 = embedding output, l+1 = output of layer l.
    pub resid: Vec<Option<ResidEdit>>,
}

impl DeviceEdits {
    pub fn none(num_layers: usize) -> Self {
        Self {
            mlp: (0..num_layers).map(|_| None).collect(),
            heads: (0..num_layers).map(|_| None).collect(),
            resid: (0..=num_layers).map(|_| None).collect(),
        }
    }

    /// Validates (indices in range, no Pin on heads, finite values) then builds the tensors.
    /// Later edits to the same unit override earlier ones.
    pub fn build(set: &EditSet, g: &Geometry, dtype: DType, device: &Device) -> anyhow::Result<Self> {
        validate(set, g)?;
        let mut out = Self::none(g.num_layers);
        let mut mlp: Vec<Option<(Vec<f32>, Vec<f32>)>> = (0..g.num_layers).map(|_| None).collect();
        for e in &set.neurons {
            let (sc, of) = mlp[e.layer]
                .get_or_insert_with(|| (vec![1.0; g.num_neurons], vec![0.0; g.num_neurons]));
            let (s, o) = match e.mode {
                EditMode::Ablate => (0.0, 0.0),
                EditMode::Scale => (e.value, 0.0),
                EditMode::Pin => (0.0, e.value),
            };
            sc[e.index] = s;
            of[e.index] = o;
        }
        for (l, m) in mlp.into_iter().enumerate() {
            if let Some((sc, of)) = m {
                let n = sc.len();
                out.mlp[l] = Some(MlpEdit {
                    scale: Tensor::from_vec(sc, n, device)?.to_dtype(dtype)?,
                    offset: Tensor::from_vec(of, n, device)?.to_dtype(dtype)?,
                });
            }
        }
        let mut heads: Vec<Option<Vec<f32>>> = (0..g.num_layers).map(|_| None).collect();
        for e in &set.heads {
            let v = heads[e.layer].get_or_insert_with(|| vec![1.0; g.num_heads]);
            v[e.head] = match e.mode {
                EditMode::Ablate => 0.0,
                _ => e.value,
            };
        }
        for (l, h) in heads.into_iter().enumerate() {
            if let Some(v) = h {
                let n = v.len();
                out.heads[l] = Some(Tensor::from_vec(v, n, device)?.to_dtype(dtype)?);
            }
        }
        let sites = resid_sites(set, g);
        for (site, (basis, add)) in sites.into_iter().enumerate() {
            let ablate = if basis.is_empty() {
                None
            } else {
                let k = basis.len();
                let flat: Vec<f32> = basis.iter().flat_map(|r| r.iter().map(|&v| v as f32)).collect();
                Some(Tensor::from_vec(flat, (k, g.hidden_size), device)?.to_dtype(dtype)?)
            };
            let add = match add {
                Some(a) => {
                    let v: Vec<f32> = a.iter().map(|&v| v as f32).collect();
                    Some(Tensor::from_vec(v, g.hidden_size, device)?.to_dtype(dtype)?)
                }
                None => None,
            };
            if ablate.is_some() || add.is_some() {
                out.resid[site] = Some(ResidEdit { ablate, add });
            }
        }
        Ok(out)
    }
}

/// Unit vector of `v` in f64 (caller has validated a nonzero finite norm).
fn unit(v: &[f32]) -> Vec<f64> {
    let n = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    v.iter().map(|&x| x as f64 / n).collect()
}

/// Per residual site (num_layers + 1): orthonormal ablation basis rows and the summed steering vector.
/// Ablated directions that are (nearly) in the span of earlier ones are dropped (Gram-Schmidt, twice
/// for numerical stability).
fn resid_sites(set: &EditSet, g: &Geometry) -> Vec<(Vec<Vec<f64>>, Option<Vec<f64>>)> {
    let n_sites = g.num_layers + 1;
    let mut out: Vec<(Vec<Vec<f64>>, Option<Vec<f64>>)> = (0..n_sites).map(|_| (Vec::new(), None)).collect();
    for e in &set.directions {
        let u = unit(&e.vector);
        let sites: Vec<usize> = if e.sites.is_empty() { (0..n_sites).collect() } else { e.sites.clone() };
        for s in sites {
            let (basis, add) = &mut out[s];
            match e.mode {
                DirectionMode::Ablate => {
                    let mut r = u.clone();
                    for _ in 0..2 {
                        for b in basis.iter() {
                            let d: f64 = r.iter().zip(b).map(|(x, y)| x * y).sum();
                            for (x, y) in r.iter_mut().zip(b) {
                                *x -= d * y;
                            }
                        }
                    }
                    let n = r.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if n > 1e-6 {
                        r.iter_mut().for_each(|x| *x /= n);
                        basis.push(r);
                    }
                }
                DirectionMode::Add => {
                    let a = add.get_or_insert_with(|| vec![0.0; g.hidden_size]);
                    for (x, y) in a.iter_mut().zip(&u) {
                        *x += e.value as f64 * y;
                    }
                }
            }
        }
    }
    out
}

pub fn validate(set: &EditSet, g: &Geometry) -> anyhow::Result<()> {
    for (i, e) in set.neurons.iter().enumerate() {
        if e.layer >= g.num_layers {
            anyhow::bail!("neuron edit {i}: layer {} out of range (0..{})", e.layer, g.num_layers);
        }
        if e.index >= g.num_neurons {
            anyhow::bail!("neuron edit {i}: index {} out of range (0..{})", e.index, g.num_neurons);
        }
        if !e.value.is_finite() {
            anyhow::bail!("neuron edit {i}: value must be finite");
        }
    }
    for (i, e) in set.heads.iter().enumerate() {
        if e.layer >= g.num_layers {
            anyhow::bail!("head edit {i}: layer {} out of range (0..{})", e.layer, g.num_layers);
        }
        if e.head >= g.num_heads {
            anyhow::bail!("head edit {i}: head {} out of range (0..{})", e.head, g.num_heads);
        }
        if e.mode == EditMode::Pin {
            anyhow::bail!("head edit {i}: Pin is not supported on attention heads (use ablate or scale)");
        }
        if !e.value.is_finite() {
            anyhow::bail!("head edit {i}: value must be finite");
        }
    }
    for (i, e) in set.directions.iter().enumerate() {
        if e.vector.len() != g.hidden_size {
            anyhow::bail!("direction edit {i}: vector has {} floats, expected hidden_size {}", e.vector.len(), g.hidden_size);
        }
        if !e.vector.iter().all(|v| v.is_finite()) {
            anyhow::bail!("direction edit {i}: vector must be finite");
        }
        let n2: f64 = e.vector.iter().map(|&x| (x as f64) * (x as f64)).sum();
        if !(n2.sqrt() > 1e-12) {
            anyhow::bail!("direction edit {i}: vector has zero norm");
        }
        if !e.value.is_finite() {
            anyhow::bail!("direction edit {i}: value must be finite");
        }
        if let Some(&s) = e.sites.iter().find(|&&s| s > g.num_layers) {
            anyhow::bail!("direction edit {i}: site {s} out of range (0..={})", g.num_layers);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DirectionEdit, HeadEdit, NeuronEdit};

    fn dir(v: Vec<f32>, sites: Vec<usize>, mode: DirectionMode, value: f32) -> DirectionEdit {
        DirectionEdit { label: String::new(), vector: v, sites, mode, value }
    }

    /// x ← x − (x·Bᵀ)·B then + add, on CPU f32 — the same ops model.rs runs.
    fn apply(x: &Tensor, e: &ResidEdit) -> Tensor {
        let mut x = x.clone();
        if let Some(b) = &e.ablate {
            x = (&x - x.matmul(&b.t().unwrap()).unwrap().matmul(b).unwrap()).unwrap();
        }
        if let Some(a) = &e.add {
            x = x.broadcast_add(a).unwrap();
        }
        x
    }

    #[test]
    fn direction_validation() {
        let gg = g();
        for d in [
            dir(vec![1.0; 7], vec![], DirectionMode::Ablate, 0.0),
            dir(vec![0.0; 8], vec![], DirectionMode::Ablate, 0.0),
            dir(vec![f32::NAN; 8], vec![], DirectionMode::Ablate, 0.0),
            dir(vec![1.0; 8], vec![4], DirectionMode::Ablate, 0.0),
            dir(vec![1.0; 8], vec![], DirectionMode::Add, f32::INFINITY),
        ] {
            let mut s = EditSet::default();
            s.directions.push(d);
            assert!(validate(&s, &gg).is_err());
        }
        let mut s = EditSet::default();
        s.directions.push(dir(vec![1.0; 8], vec![3], DirectionMode::Ablate, 0.0));
        assert!(validate(&s, &gg).is_ok());
    }

    #[test]
    fn ablation_removes_component_exactly() {
        let mut s = EditSet::default();
        let v: Vec<f32> = (0..8).map(|i| (i as f32) - 3.5).collect();
        let w: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 1.0 } else { 0.3 }).collect();
        s.directions.push(dir(v.clone(), vec![], DirectionMode::Ablate, 0.0));
        s.directions.push(dir(w.clone(), vec![], DirectionMode::Ablate, 0.0));
        let d = DeviceEdits::build(&s, &g(), DType::F32, &Device::Cpu).unwrap();
        assert_eq!(d.resid.len(), 4);
        let e = d.resid[2].as_ref().unwrap();
        assert_eq!(e.ablate.as_ref().unwrap().dims(), &[2, 8]);
        assert!(e.add.is_none());
        let xs: Vec<f32> = (0..16).map(|i| ((i * 7 % 11) as f32) - 5.0).collect();
        let x = Tensor::from_vec(xs, (2, 8), &Device::Cpu).unwrap();
        let y = apply(&x, e).to_vec2::<f32>().unwrap();
        for row in &y {
            for dv in [&v, &w] {
                let p: f32 = row.iter().zip(dv.iter()).map(|(a, b)| a * b).sum();
                assert!(p.abs() < 1e-4, "component left: {p}");
            }
        }
        // idempotent: ablating again changes nothing
        let y2 = apply(&Tensor::new(y.clone(), &Device::Cpu).unwrap(), e).to_vec2::<f32>().unwrap();
        for (a, b) in y.iter().flatten().zip(y2.iter().flatten()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn parallel_directions_collapse_and_sites_apply() {
        let mut s = EditSet::default();
        let v: Vec<f32> = vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0];
        let v2: Vec<f32> = v.iter().map(|x| -3.0 * x).collect();
        s.directions.push(dir(v.clone(), vec![1], DirectionMode::Ablate, 0.0));
        s.directions.push(dir(v2, vec![1], DirectionMode::Ablate, 0.0));
        s.directions.push(dir(vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![0, 3], DirectionMode::Add, 1.5));
        s.directions.push(dir(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], vec![3], DirectionMode::Add, 0.5));
        let d = DeviceEdits::build(&s, &g(), DType::F32, &Device::Cpu).unwrap();
        assert!(d.resid[2].is_none());
        assert_eq!(d.resid[1].as_ref().unwrap().ablate.as_ref().unwrap().dims(), &[1, 8]);
        let a0 = d.resid[0].as_ref().unwrap().add.as_ref().unwrap().to_vec1::<f32>().unwrap();
        assert!((a0[2] - 1.5).abs() < 1e-6 && a0[0] == 0.0);
        let a3 = d.resid[3].as_ref().unwrap().add.as_ref().unwrap().to_vec1::<f32>().unwrap();
        assert!((a3[2] - 2.0).abs() < 1e-6);
        assert!(d.resid[0].as_ref().unwrap().ablate.is_none());
    }

    fn g() -> Geometry {
        Geometry { num_layers: 3, num_neurons: 8, num_heads: 4, num_kv_heads: 2, head_dim: 4, hidden_size: 8, vocab_size: 10 }
    }

    #[test]
    fn validate_rejects() {
        let mut s = EditSet::default();
        s.heads.push(HeadEdit { layer: 0, head: 0, mode: EditMode::Pin, value: 1.0 });
        assert!(validate(&s, &g()).is_err());
        let mut s = EditSet::default();
        s.neurons.push(NeuronEdit { layer: 3, index: 0, mode: EditMode::Ablate, value: 0.0 });
        assert!(validate(&s, &g()).is_err());
        let mut s = EditSet::default();
        s.neurons.push(NeuronEdit { layer: 0, index: 0, mode: EditMode::Scale, value: f32::NAN });
        assert!(validate(&s, &g()).is_err());
    }

    #[test]
    fn build_overrides() {
        let mut s = EditSet::default();
        s.neurons.push(NeuronEdit { layer: 1, index: 2, mode: EditMode::Ablate, value: 0.0 });
        s.neurons.push(NeuronEdit { layer: 1, index: 2, mode: EditMode::Pin, value: 5.0 });
        s.heads.push(HeadEdit { layer: 2, head: 1, mode: EditMode::Scale, value: 2.0 });
        let d = DeviceEdits::build(&s, &g(), DType::F32, &Device::Cpu).unwrap();
        assert!(d.mlp[0].is_none() && d.heads[0].is_none());
        let m = d.mlp[1].as_ref().unwrap();
        assert_eq!(m.scale.to_vec1::<f32>().unwrap()[2], 0.0);
        assert_eq!(m.offset.to_vec1::<f32>().unwrap()[2], 5.0);
        assert_eq!(m.scale.to_vec1::<f32>().unwrap()[0], 1.0);
        assert_eq!(d.heads[2].as_ref().unwrap().to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 1.0, 1.0]);
    }
}
