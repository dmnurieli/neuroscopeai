//! Token selection. Greedy = argmax. Sample = temperature → top-k → top-p → seeded draw.

use crate::types::Sampling;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

pub struct Sampler {
    mode: Sampling,
    rng: StdRng,
}

impl Sampler {
    pub fn new(sampling: &Sampling) -> Self {
        let seed = match sampling {
            Sampling::Sample { seed, .. } => *seed,
            Sampling::Greedy => 0,
        };
        Self { mode: sampling.clone(), rng: StdRng::seed_from_u64(seed) }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        match self.mode {
            Sampling::Greedy => argmax(logits),
            Sampling::Sample { temperature, top_p, top_k, .. } => {
                if temperature <= 0.0 {
                    return argmax(logits);
                }
                let k = if top_k == 0 { logits.len() } else { top_k };
                let mut cand = top_k_probs(logits, temperature, k);
                if cand.is_empty() {
                    return 0;
                }
                // top-p: keep the smallest prefix whose mass reaches top_p
                if top_p > 0.0 && top_p < 1.0 {
                    let mut acc = 0.0;
                    let mut keep = cand.len();
                    for (i, (_, p)) in cand.iter().enumerate() {
                        acc += p;
                        if acc >= top_p {
                            keep = i + 1;
                            break;
                        }
                    }
                    cand.truncate(keep);
                }
                let total: f32 = cand.iter().map(|c| c.1).sum();
                let mut r: f32 = self.rng.random::<f32>() * total;
                for (id, p) in &cand {
                    if r < *p {
                        return *id;
                    }
                    r -= p;
                }
                cand.last().unwrap().0
            }
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

/// Softmax(logits / temperature) then the k most probable (id, prob), descending. Numerically stable.
pub fn top_k_probs(logits: &[f32], temperature: f32, k: usize) -> Vec<(u32, f32)> {
    top_k_probs_stats(logits, temperature, k).0
}

/// Ordering of the result: probability descending, then id ascending.
fn prob_cmp(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
}

/// top_k_probs plus the (max, Σexp((l-max)/t)) it computed, so a caller at temperature 1 can reuse
/// them for log_prob without another pass over the vocabulary.
///
/// Only O(vocab) work: one max pass, one exp-sum pass (same arithmetic and order as a full softmax,
/// so every probability is bit-identical to it), one selection pass over the raw logits. Because
/// prob is monotone in the logit, the k best by prob are among the k best by logit plus anything
/// whose logit is within float-rounding distance of the k-th (which could tie it in prob); those
/// few candidates get exact probs and the exact (prob desc, id asc) ordering.
pub(crate) fn top_k_probs_stats(logits: &[f32], temperature: f32, k: usize) -> (Vec<(u32, f32)>, f32, f64) {
    if logits.is_empty() {
        return (Vec::new(), f32::NEG_INFINITY, 0.0);
    }
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &l in logits {
        sum += (((l - max) / t) as f64).exp();
    }
    if k == 0 {
        return (Vec::new(), max, sum);
    }
    let prob = |l: f32| ((((l - max) / t) as f64).exp() / sum) as f32;
    let k = k.min(logits.len());
    let fast = k * 8 < logits.len() && logits.iter().all(|l| !l.is_nan());
    if !fast {
        let mut v: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, prob(l))).collect();
        if k < v.len() {
            v.select_nth_unstable_by(k - 1, prob_cmp);
            v.truncate(k);
        }
        v.sort_by(prob_cmp);
        return (v, max, sum);
    }
    // k-th largest logit via a small descending buffer.
    let mut best: Vec<f32> = Vec::with_capacity(k + 1);
    for &l in logits {
        if best.len() < k {
            let pos = best.partition_point(|&b| b >= l);
            best.insert(pos, l);
        } else if l > best[k - 1] {
            let pos = best.partition_point(|&b| b >= l);
            best.insert(pos, l);
            best.pop();
        }
    }
    let tau = best[k - 1];
    // generous slack: equal f32 probs need |Δl|/t ≲ 2.4e-7 relative to 1; 1e-4 covers any scale here.
    let slack = (tau.abs().max(1.0) * 1e-4) * t.max(1.0);
    let lo = tau - slack;
    let mut v: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l >= lo)
        .map(|(i, &l)| (i as u32, prob(l)))
        .collect();
    v.sort_by(prob_cmp);
    v.truncate(k);
    (v, max, sum)
}

/// log softmax(logits)[id] at temperature 1.
pub fn log_prob(logits: &[f32], id: u32) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    log_prob_from(logits, id, max, sum)
}

/// log_prob with the max and temperature-1 exp-sum already known (bit-identical to log_prob).
pub(crate) fn log_prob_from(logits: &[f32], id: u32, max: f32, sum: f64) -> f32 {
    (logits[id as usize] - max) - sum.ln() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_topk() {
        let l = [0.1, 3.0, 1.0, 2.0];
        let mut s = Sampler::new(&Sampling::Greedy);
        assert_eq!(s.sample(&l), 1);
        let t = top_k_probs(&l, 1.0, 2);
        assert_eq!(t.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 3]);
        assert!(t[0].1 > t[1].1);
        assert!((log_prob(&l, 1).exp() - t[0].1).abs() < 1e-5);
    }

    fn reference_top_k(logits: &[f32], temperature: f32, k: usize) -> Vec<(u32, f32)> {
        let t = if temperature > 0.0 { temperature } else { 1.0 };
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for &l in logits {
            sum += (((l - max) / t) as f64).exp();
        }
        let mut v: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as u32, ((((l - max) / t) as f64).exp() / sum) as f32))
            .collect();
        v.sort_by(prob_cmp);
        v.truncate(k);
        v
    }

    #[test]
    fn fast_top_k_matches_reference() {
        let mut x: u64 = 0x9e3779b97f4a7c15;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32
        };
        for case in 0..6 {
            let n = 20000;
            let mut l: Vec<f32> = (0..n).map(|_| rnd() * 30.0 - 15.0).collect();
            if case % 2 == 1 {
                // heavy ties around the top
                for i in 0..n {
                    if i % 97 == 0 {
                        l[i] = 14.5;
                    }
                }
            }
            for &(t, k) in &[(1.0f32, 8usize), (0.6, 20), (1.0, 1), (2.0, 50)] {
                let a = top_k_probs(&l, t, k);
                let b = reference_top_k(&l, t, k);
                assert_eq!(a, b, "case {case} t {t} k {k}");
            }
            let (_, mx, sum) = top_k_probs_stats(&l, 1.0, 8);
            assert_eq!(log_prob_from(&l, 7, mx, sum).to_bits(), log_prob(&l, 7).to_bits());
        }
    }

    #[test]
    fn seeded_deterministic() {
        let l: Vec<f32> = (0..50).map(|i| (i % 7) as f32).collect();
        let m = Sampling::Sample { temperature: 0.8, top_p: 0.9, top_k: 10, seed: 42 };
        let mut a = Sampler::new(&m);
        let mut b = Sampler::new(&m);
        for _ in 0..20 {
            assert_eq!(a.sample(&l), b.sample(&l));
        }
    }
}
