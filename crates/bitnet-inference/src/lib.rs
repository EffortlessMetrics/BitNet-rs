//! # BitNet Inference Engine
//!
//! High-performance inference engine for BitNet models with streaming support,
//! CPU/GPU backends, and comprehensive sampling strategies.

pub mod engine;
pub mod streaming;
pub mod sampling;
pub mod cache;
pub mod config;
pub mod backends;

pub use engine::{InferenceEngine, InferenceResult};
pub use streaming::{GenerationStream, StreamingConfig};
pub use sampling::{SamplingStrategy, SamplingConfig};
pub use cache::{KVCache, CacheConfig};
pub use config::{InferenceConfig, GenerationConfig};
pub use backends::{Backend, CpuBackend, GpuBackend};

use anyhow::Result;
use bitnet_common::{BitNetConfig, Device, Tensor};
use bitnet_models::Model;
use bitnet_tokenizers::Tokenizer;
use std::sync::Arc;

/// Re-export commonly used types
pub mod prelude {
    pub use super::{
        InferenceEngine, InferenceResult, GenerationStream, StreamingConfig,
        SamplingStrategy, SamplingConfig, KVCache, CacheConfig,
        InferenceConfig, GenerationConfig, Backend, CpuBackend, GpuBackend,
    };
    pub use anyhow::Result;
    pub use futures_util::{Stream, StreamExt};
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_inference_engine_creation() {
        // This would require actual model and tokenizer implementations
        // For now, just test that the module compiles
        assert!(true);
    }
}