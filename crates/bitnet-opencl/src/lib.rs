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
//! `OpenCL` backend with unified model format support for `BitNet` inference.
//!
//! This crate provides:
//! - [`model_format`]: Auto-detection of GGUF, `SafeTensors`, and ONNX formats
//! - [`model_loader_unified`]: Format-agnostic model loading trait and registry
//! - [`error`]: Structured error types for format and loading failures

pub mod error;
pub mod model_format;
pub mod model_loader_unified;

// Re-export primary public types.
pub use error::ModelFormatError;
pub use model_format::{
    ModelFormat, ModelFormatDetector, ModelMetadata, QuantizationHint, detect_format,
};
pub use model_loader_unified::{
    DeviceId, GgufLoader, ModelLoaderRegistry, ModelWeights, SafeTensorsLoader, UnifiedModelLoader,
};
