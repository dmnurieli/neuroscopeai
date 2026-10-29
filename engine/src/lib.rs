//! NeuroScope engine: Qwen3 / OLMo 2 inference on Metal with per-neuron capture and editing,
//! exposed to the Swift app through the C ABI in include/neuroscope.h.

pub mod attribution;
pub mod config;
pub mod contrast;
pub mod edits;
pub mod engine;
pub mod ffi;
pub mod follow;
pub mod generate;
pub mod lens;
pub mod model;
pub mod project;
pub mod sampler;
pub mod streams;
pub mod tokenize;
pub mod trace;
pub mod types;

pub use engine::Engine;
pub use types::*;
