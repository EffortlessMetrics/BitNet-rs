//! OpenCL backend for BitNet GPU inference.
//!
//! Provides optimized OpenCL primitives for Intel Arc GPU inference,
//! including context pooling, local memory optimizations, prefetch
//! pipelines, KV cache, paged attention, and multi-backend GPU dispatch
//! with automatic selection.
//! pipelines, KV cache, paged attention, and CPU reference implementations
//! with OpenCL kernel sources for I2_S dequantization, QK256 block
//! dequantization, and ternary matrix multiply.
//! pipelines, KV cache, paged attention, SPIR-V compilation, and kernel registry.

pub mod backend_dispatcher;
pub mod backend_registry;
pub mod context_pool;
pub mod kv_cache;
pub mod paged_attention;

pub use backend_dispatcher::{
    BackendCapabilityMatrix, BackendDispatcher, BackendStatus, DispatchDecision, DispatchError,
    DispatchLog, DispatchStrategy, Operation,
};
pub use backend_registry::{BackendInfo, BackendProvider, BackendRegistry};
pub mod quantized_kernels;
pub mod quantized_ops;
pub mod spirv;
pub mod spirv_kernels;

// Re-exports for convenience.
pub use spirv::{
    CompileOptions, CompilerBackend, OptimizationLevel, SPIRV_MAGIC, SpirVCache, SpirVCompiler,
    SpirVError, SpirVModule, SpirVValidator,
};
pub use spirv_kernels::{KernelSource, SpirvKernelRegistry};
//! `OpenCL`/GPU text generation for `BitNet` inference.
//!
//! This crate provides [`GenerationEngine`], which orchestrates the
//! full text generation loop: tokenisation → transformer forward pass →
//! sampling → KV-cache append → stopping-criteria check → decode.
//!
//! The current implementation is **CPU-only** (MVP).  A GPU dispatch
//! interface ([`ModelBackend`] trait) is ready so that an `OpenCL` or CUDA
//! backend can be plugged in without changing the loop logic.
//!
//! ## Quick start
//!
//! ```rust
//! use bitnet_opencl::{
//!     GenerationConfig, GenerationEngine, MockModelBackend,
//!     MockTokenizer, StoppingCriteria,
//! };
//!
//! let mut engine = GenerationEngine::new(
//!     MockModelBackend::new(256),
//!     MockTokenizer,
//!     GenerationConfig { max_tokens: 8, temperature: 0.0, ..Default::default() },
//!     StoppingCriteria { max_length: 8, ..Default::default() },
//! );
//! let result = engine.generate("hello").unwrap();
//! assert!(!result.tokens.is_empty());
//! ```

pub mod generation;
pub mod generation_stats;

// Re-exports for ergonomic access.
pub use generation::{
    GenerationConfig, GenerationEngine, GenerationError, GenerationResult, GenerationStream,
    MockModelBackend, MockTokenizer, ModelBackend, StoppingCriteria, StreamToken, Tokenizer,
};
pub use generation_stats::{GenerationStats, StatsCollector};
