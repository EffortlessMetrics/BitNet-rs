//! # BitNet Inference Engine
//!
//! High-performance inference engine for BitNet models with streaming support,
//! CPU/GPU backends, and comprehensive sampling strategies.

pub mod attention_mask;
pub mod backends;
pub mod batch;
pub mod batch_engine;
pub mod batch_scheduler;
pub mod cache;
pub mod compute_cost;
pub mod compute_graph;
pub mod config;
pub mod config_builder;
pub mod context_window;
pub mod cpu_opt;
pub mod decode_strategy;
pub mod dense_forward;
pub mod dense_generation;
pub mod dense_integration_tests;
pub mod embedding_utils;
pub mod engine;
pub mod generation;
pub mod generation_budget;
pub mod generation_output;
pub mod generation_pipeline;
pub mod gguf;
pub mod gpu_streaming;
pub mod inference_cache;
pub mod kernel_recorder;
pub mod kv_cache_manager;
pub mod kv_cache_optimized;
pub mod kv_cache_stats;
pub mod layer_executor;
pub mod layers;
pub mod memory_estimation;
pub mod memory_pool;
pub mod metrics;
pub mod metrics_collector;
pub mod model_profiler;
pub mod npu;
pub mod production_engine; // always available (sync parser)
pub mod prompt_template; // Chat and instruct format templates
pub mod receipts; // AC4: Inference receipt generation
pub mod request_types;
pub mod rope_config;
pub mod run_metrics;
pub mod template_registry;
pub mod thread_pool;
pub mod tool_templates; // Tool-use / function-calling prompt templates
pub mod warmup;

// Re-export GGUF types for easy access
pub use gguf::{GGUF_HEADER_LEN, GgufError, GgufHeader, GgufKv, GgufValue, read_kv_pairs};
#[cfg(all(feature = "ffi", not(bitnet_sys_stub)))]
pub mod ffi_session; // FFI session wrapper for validation-only parity checking
pub mod parity;
pub mod pipeline_builder;
pub mod prefix_cache;
pub mod profiler;
pub mod rt;
pub mod runtime_utils;
pub mod sampling;
pub mod sampling_presets;
pub mod sampling_utils;
pub mod session_manager;
pub mod simple_forward;
pub mod streaming;
pub mod streaming_render;
pub mod tensor_parallel;
pub mod token_stream;
// Only compile the shim when tests or a GPU feature need it
#[cfg(any(test, feature = "gpu"))]
mod tensor_ext;
#[cfg(any(test, feature = "gpu"))]
#[allow(unused_imports)]
pub(crate) use tensor_ext::TensorDeviceExt;

pub use backends::{Backend, CpuBackend, GpuBackend};
pub use batch::{BatchConfig, BatchRequest, BatchResult, BatchScheduler, SingleResult};
pub use bitnet_inference_metrics_core::{ThroughputMetrics, TimingMetrics};
pub use cache::{CacheConfig, KVCache};
pub use config::{GenerationConfig, InferenceConfig};
pub use dense_forward::{
    DenseAttention, DenseAttentionConfig, DenseFFN, DenseLinear, DenseModel, DenseTransformerBlock,
    dense_block_forward_tensor, rms_norm, silu,
};
pub use engine::{InferenceEngine, InferenceResult};
pub use generation::{
    AutoregressiveGenerator, GenConfig, SampleConfig, SamplingStrategy as GenSamplingStrategy,
};
pub use kernel_recorder::KernelRecorder;
pub use kv_cache_optimized::{
    CacheEvictionPolicy, CacheMetrics, EvictionConfig, Page, PageId, PagedKvCache,
};
pub use layers::{BitNetAttention, LookupTable, QuantizedLinear};
pub use metrics::{
    InferenceMetrics, LatencyHistogram, MemoryProfiler, MetricsCollector, MetricsReport,
    ThroughputTracker,
};
pub use npu::{BITNET_ENABLE_NPU, map_device_token, npu_requested};
pub use parity::{
    eval_logits_all_positions, eval_logits_incremental, eval_logits_once,
    eval_logits_once_for_parity, get_model_config, get_model_vocab_size,
};
pub use prefix_cache::{
    CacheEntry as PrefixCacheEntry, EvictionPolicy as PrefixEvictionPolicy, PrefixCache,
    PrefixCacheConfig, PrefixCacheStats,
};
pub use production_engine::{
    GenerationResult, PerformanceMetricsCollector, PrefillStrategy, ProductionInferenceConfig,
    ProductionInferenceEngine,
};
pub use prompt_template::{ChatRole, ChatTurn, PromptTemplate, TemplateType};
pub use receipts::{
    AccuracyMetric, AccuracyTestResults, CrossValidation, DeterminismTestResults, InferenceReceipt,
    KVCacheTestResults, ModelInfo, PerformanceBaseline, RECEIPT_SCHEMA_VERSION, TestResults,
};
// Re-export CorrectionRecord from bitnet-common for convenience
pub use bitnet_common::CorrectionRecord;
pub use gpu_streaming::{GpuGenerationStream, GpuStreamingConfig, GpuTokenEvent};
pub use sampling::{SamplingConfig, SamplingStrategy};
pub use streaming::{GenerationStream, StreamingConfig};
pub use thread_pool::{InferenceThreadPool, ThreadPoolConfig, ThreadPoolMetrics};
pub use token_stream::{StreamConfig, StreamEvent, StreamStats, TokenBuffer, TokenStream};

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

// retrigger-ci-placeholder: remove if needed
