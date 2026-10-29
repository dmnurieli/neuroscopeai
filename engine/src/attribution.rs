//! Direct logit attribution of a single MLP neuron.

use crate::model::Model;
use crate::tokenize::Tok;
use crate::types::{NeuronInfo, TokenScore};
use candle_core::DType;

/// Scores every vocab token by unembed · (final_norm_weight ⊙ w_out ⊙ down_proj[:, index]) and returns the
/// top_k most promoted (highest) and most suppressed (lowest) tokens, plus the down_proj column's L2 norm.
/// w_out is the MLP write norm weight (OLMo 2 post_feedforward_layernorm; ones for Qwen3): the direction
/// the neuron actually adds to the residual. Its positive per-position scale does not change the ranking.
pub fn neuron_info(model: &Model, tok: &Tok, layer: usize, index: usize, top_k: usize) -> anyhow::Result<NeuronInfo> {
    let g = model.geometry();
    if layer >= g.num_layers {
        anyhow::bail!("layer {layer} out of range (0..{})", g.num_layers);
    }
    if index >= g.num_neurons {
        anyhow::bail!("neuron index {index} out of range (0..{})", g.num_neurons);
    }
    let col = model.down_proj_column(layer, index)?.to_dtype(DType::F32)?;
    let out_norm = col.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let col = (col * model.mlp_write_weight(layer).to_dtype(DType::F32)?)?;
    // Approximation: no norm-weight accessor, so apply model.final_norm to the column. This rescales
    // the column by 1/rms and applies the norm weight, preserving ranking by the weighted direction.
    let weighted = model.final_norm(&col.to_dtype(model.dtype())?.unsqueeze(0)?)?.to_dtype(DType::F32)?; // [1, hidden]
    let unembed = model.unembed().to_dtype(DType::F32)?;
    let scores = unembed.matmul(&weighted.t()?.contiguous()?)?.flatten_all()?.to_vec1::<f32>()?;
    let k = top_k.min(scores.len());
    let mut idx: Vec<u32> = (0..scores.len() as u32).collect();
    let desc = |a: &u32, b: &u32| scores[*b as usize].partial_cmp(&scores[*a as usize]).unwrap_or(std::cmp::Ordering::Equal);
    let mut promotes = Vec::new();
    let mut suppresses = Vec::new();
    if k > 0 {
        idx.select_nth_unstable_by(k - 1, desc);
        let mut top: Vec<u32> = idx[..k].to_vec();
        top.sort_by(desc);
        promotes = top;
        idx.select_nth_unstable_by(scores.len() - k, desc);
        let mut bot: Vec<u32> = idx[scores.len() - k..].to_vec();
        bot.sort_by(|a, b| desc(b, a));
        suppresses = bot;
    }
    let mk = |ids: Vec<u32>| -> Vec<TokenScore> {
        ids.into_iter().map(|id| TokenScore { id, token: tok.token_text(id), score: scores[id as usize] }).collect()
    };
    Ok(NeuronInfo { layer, index, promotes: mk(promotes), suppresses: mk(suppresses), out_norm })
}
