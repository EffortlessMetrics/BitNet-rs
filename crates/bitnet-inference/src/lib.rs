//! # BitNet Inference Engine
//!
//! High-performance inference engine for BitNet models with streaming support,
//! CPU/GPU backends, and comprehensive sampling strategies.

pub mod backends;
pub mod cache;
pub mod config;
pub mod cpu_opt;
pub mod engine;
pub mod generation;
pub mod gguf;
pub mod kernel_recorder;
pub mod layers;
pub mod production_engine; // always available (sync parser)
pub mod prompt_template; // Chat and instruct format templates
pub mod receipts; // AC4: Inference receipt generation

// Re-export GGUF types for easy access
pub use gguf::{GGUF_HEADER_LEN, GgufError, GgufHeader, GgufKv, GgufValue, read_kv_pairs};
#[cfg(all(feature = "ffi", not(bitnet_sys_stub)))]
pub mod ffi_session; // FFI session wrapper for validation-only parity checking
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
pub use kernel_recorder::KernelRecorder;
pub use layers::{BitNetAttention, LookupTable, QuantizedLinear};
pub use parity::{
    eval_logits_all_positions, eval_logits_incremental, eval_logits_once,
    eval_logits_once_for_parity, get_model_config, get_model_vocab_size,
};
pub use production_engine::{
    GenerationResult, PerformanceMetricsCollector, PrefillStrategy, ProductionInferenceConfig,
    ProductionInferenceEngine, ThroughputMetrics, TimingMetrics,
};
pub use prompt_template::{ChatRole, ChatTurn, PromptTemplate, TemplateType};
pub use receipts::{
    AccuracyMetric, AccuracyTestResults, CrossValidation, DeterminismTestResults, InferenceReceipt,
    KVCacheTestResults, ModelInfo, PerformanceBaseline, RECEIPT_SCHEMA_VERSION, TestResults,
};
// Re-export CorrectionRecord from bitnet-common for convenience
pub use bitnet_common::CorrectionRecord;
pub use sampling::{SamplingConfig, SamplingStrategy};
pub use streaming::{GenerationStream, StreamingConfig};

// Re-export SRP-extracted orchestration contracts from bitnet-engine-core.
pub use bitnet_engine_core::{BackendInfo, InferenceSession, SessionConfig, SessionMetrics};

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
# retrigger
