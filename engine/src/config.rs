//! config.json of a supported checkpoint: Qwen3 (`model_type` "qwen3", the default when absent) or
//! OLMo 2 (`model_type` "olmo2").

use crate::types::Geometry;
use serde::Deserialize;
use std::path::Path;

/// Decoder family, chosen from config.json's `model_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// Pre-norm (input_layernorm before attention, post_attention_layernorm before the MLP),
    /// per-head q_norm/k_norm [head_dim], tied embeddings (0.6B).
    Qwen3,
    /// OLMo 2: no pre-norms — attention and MLP read the raw residual; post_attention_layernorm and
    /// post_feedforward_layernorm normalize each sublayer's OUTPUT before the residual add; q_norm/k_norm
    /// over the full projected q [heads*head_dim] / k [kv_heads*head_dim]; untied lm_head.
    Olmo2,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub model_type: Option<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// Absent in OLMo 2 configs: hidden_size / num_attention_heads (as HF does).
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

/// Kept so older call sites read naturally.
pub type Qwen3Config = ModelConfig;

impl ModelConfig {
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(model_dir.join("config.json"))?;
        let cfg: Self = serde_json::from_str(&raw)?;
        cfg.check()?;
        Ok(cfg)
    }

    pub fn arch(&self) -> Arch {
        match self.model_type.as_deref() {
            Some("olmo2") => Arch::Olmo2,
            _ => Arch::Qwen3,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Refuses configs this engine would silently compute wrong.
    fn check(&self) -> anyhow::Result<()> {
        if let Some(t) = self.model_type.as_deref() {
            anyhow::ensure!(t == "qwen3" || t == "olmo2", "unsupported model_type {t:?} (supported: qwen3, olmo2)");
        }
        anyhow::ensure!(!self.attention_bias, "attention_bias = true is not supported");
        if let Some(a) = self.hidden_act.as_deref() {
            anyhow::ensure!(a == "silu", "hidden_act {a:?} is not supported (silu only)");
        }
        if let Some(r) = &self.rope_scaling {
            anyhow::ensure!(r.is_null(), "rope_scaling is not supported: {r}");
        }
        anyhow::ensure!(self.num_attention_heads % self.kv_heads() == 0, "num_attention_heads must be a multiple of num_key_value_heads");
        Ok(())
    }

    pub fn geometry(&self) -> Geometry {
        Geometry {
            num_layers: self.num_hidden_layers,
            num_neurons: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.kv_heads(),
            head_dim: self.head_dim(),
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
        }
    }
}
