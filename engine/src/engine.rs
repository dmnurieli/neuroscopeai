//! The loaded model + tokenizer, owned by one FFI handle.

use crate::config::{Arch, ModelConfig};
use crate::model::Model;
use crate::tokenize::Tok;
use crate::types::Geometry;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub struct Engine {
    pub model: Model,
    pub tok: Tok,
    pub config: ModelConfig,
    pub geometry: Geometry,
    /// Set by ns_cancel; generate() polls it between tokens and clears it at start.
    pub cancel: Arc<AtomicBool>,
}

impl Engine {
    /// Metal device 0 (falls back to CPU only if Metal is unavailable), config, tokenizer, weights.
    /// The architecture (Qwen3 or OLMo 2) comes from config.json's model_type.
    pub fn open(model_dir: &Path) -> anyhow::Result<Self> {
        let device = match candle_core::Device::new_metal(0) {
            Ok(d) => d,
            Err(_) => candle_core::Device::Cpu,
        };
        Self::open_on(model_dir, &device)
    }


    /// As `open`, on an explicit device (tests use the CPU for f32 comparisons). Compute dtype: bf16 on
    /// Metal, f32 on the CPU, unless NS_DTYPE=f32|bf16 says otherwise.
    pub fn open_on(model_dir: &Path, device: &candle_core::Device) -> anyhow::Result<Self> {
        let dtype = match std::env::var("NS_DTYPE").ok().as_deref() {
            Some("f32") => candle_core::DType::F32,
            Some("bf16") => candle_core::DType::BF16,
            Some(other) => anyhow::bail!("NS_DTYPE={other:?}: expected f32 or bf16"),
            None if device.is_cpu() => candle_core::DType::F32,
            None => candle_core::DType::BF16,
        };
        Self::open_with(model_dir, device, dtype)
    }

    pub fn open_with(model_dir: &Path, device: &candle_core::Device, dtype: candle_core::DType) -> anyhow::Result<Self> {
        let config = ModelConfig::load(model_dir)?;
        let tok = Tok::load_for(model_dir, config.arch())?;
        let model = Model::load_with_dtype(&config, model_dir, device, dtype)?;
        let geometry = model.geometry();
        Ok(Engine { model, tok, config, geometry, cancel: Arc::new(AtomicBool::new(false)) })
    }

    pub fn arch(&self) -> Arch {
        self.config.arch()
    }
}
