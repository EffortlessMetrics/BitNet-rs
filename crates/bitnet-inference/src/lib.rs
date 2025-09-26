//! # BitNet Inference Engine
//!
//! High-performance inference engine for BitNet models with streaming support,
//! CPU/GPU backends, and comprehensive sampling strategies.

pub mod backends;
pub mod cache;
pub mod config;
pub mod engine;
pub mod generation;
pub mod gguf;
pub mod layers;
pub mod production_engine; // always available (sync parser)

// Re-export GGUF types for easy access
pub use gguf::{GGUF_HEADER_LEN, GgufError, GgufHeader, GgufKv, GgufValue, read_kv_pairs};
pub mod parity;
pub mod rt;
pub mod runtime_utils;
pub mod sampling;
pub mod simple_forward;
pub mod streaming;
// Only compile the shim when tests or a GPU feature need it
#[cfg(any(test, feature = "gpu"))]
mod tensor_ext;
#[cfg(any(test, feature = "gpu"))]
#[allow(unused_imports)]
pub(crate) use tensor_ext::TensorDeviceExt;

pub use backends::{Backend, CpuBackend, GpuBackend};
pub use cache::{CacheConfig, KVCache};
pub use config::{GenerationConfig, InferenceConfig};
pub use engine::{InferenceEngine, InferenceResult};
pub use generation::{
    AutoregressiveGenerator, GenConfig, SampleConfig, SamplingStrategy as GenSamplingStrategy,
};
pub use layers::{BitNetAttention, QuantizedLinear};
pub use parity::{
    eval_logits_incremental, eval_logits_once, get_model_config, get_model_vocab_size,
};
pub use production_engine::{
    GenerationResult, PerformanceMetricsCollector, ProductionInferenceEngine, ThroughputMetrics,
    TimingMetrics,
};
pub use sampling::{SamplingConfig, SamplingStrategy};
pub use streaming::{GenerationStream, StreamingConfig};

// Module-level imports (removed - unused)

/// Re-export commonly used types
pub mod prelude {
    pub use super::{
        Backend, CacheConfig, CpuBackend, GenerationConfig, GenerationStream, GpuBackend,
        InferenceConfig, InferenceEngine, InferenceResult, KVCache, SamplingConfig,
        SamplingStrategy, StreamingConfig,
    };
    pub use anyhow::Result;
    pub use futures_util::{Stream, StreamExt};
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_inference_engine_creation() {
        // This would require actual model and tokenizer implementations
        // For now, just test that the module compiles
    }
}
