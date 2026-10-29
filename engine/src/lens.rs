//! Logit lens: decode the residual stream after every layer through final norm + unembedding.

use crate::model::Model;
use candle_core::{DType, Tensor, D};

/// `residuals`: [num_layers, hidden]. Returns, per layer, the top-1 (token_id, probability) —
/// one batched matmul so the unembedding is read once.
///
/// The top-1 softmax probability is exp(max - logsumexp) = 1 / Σ exp(l - max), so the full
/// [L, vocab] softmax is never materialised; argmax over logits equals argmax over probabilities.
/// The unembedding is used through a transposed view (no per-call copy of the vocab matrix), and
/// ids + probabilities come back in a single device→host read.
pub fn logit_lens(model: &Model, residuals: &Tensor) -> anyhow::Result<Vec<(u32, f32)>> {
    let packed = logit_lens_device(model, residuals)?.to_vec1::<f32>()?;
    Ok(unpack(&packed))
}

/// The lens computed on device, not read back: flat f32 [L * 2] = (id as f32, prob) per layer.
/// `Model::forward` appends this to its single device→host read.
pub(crate) fn logit_lens_device(model: &Model, residuals: &Tensor) -> anyhow::Result<Tensor> {
    let x = residuals.to_dtype(model.dtype())?.contiguous()?;
    let normed = model.final_norm(&x)?;
    let logits = normed.matmul(&model.unembed().t()?)?.to_dtype(DType::F32)?; // [L, vocab]
    let mx = logits.max_keepdim(D::Minus1)?; // [L, 1]
    let ids = logits.argmax_keepdim(D::Minus1)?.to_dtype(DType::F32)?; // [L, 1], ids < 2^24 exact
    let sum = logits.broadcast_sub(&mx)?.exp()?.sum_keepdim(D::Minus1)?; // [L, 1]
    let prob = sum.recip()?;
    Ok(Tensor::cat(&[&ids, &prob], 1)?.flatten_all()?)
}

pub(crate) fn unpack(packed: &[f32]) -> Vec<(u32, f32)> {
    packed.chunks_exact(2).map(|r| (r[0] as u32, r[1])).collect()
}
