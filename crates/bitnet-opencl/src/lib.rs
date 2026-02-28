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
//! Async GPU execution engine with pipeline parallelism for `BitNet` inference.
//!
//! This crate provides:
//!
//! - [`AsyncGpuExecutor`] — multi-queue GPU command management with
//!   double-buffering support for overlapped host↔device transfers.
//! - [`PipelineScheduler`] — DAG-based scheduler that overlaps transfer,
//!   compute, and readback stages across successive tokens.
//! - [`GpuFuture`] / [`GpuEvent`] — lightweight async completion primitives.
//!
//! All `OpenCL` interactions are gated behind the `oneapi` feature flag.
//! Without it, a **mock backend** is compiled that completes every operation
//! synchronously, enabling the full test suite to run without GPU hardware.
//!
//! # Feature flags
//!
//! | Feature  | Effect |
//! |----------|--------|
//! | `cpu`    | CPU-only mock backend (default path) |
//! | `gpu`    | Enables `cuda` + `oneapi` |
//! | `cuda`   | CUDA backend (placeholder) |
//! | `oneapi` | Intel oneAPI / OpenCL backend |

pub mod async_executor;
pub mod error;
pub mod pipeline_scheduler;

// Re-export primary types at crate root for convenience.
pub use async_executor::{
    AsyncGpuExecutor, ExecutionPlan, GpuEvent, GpuFuture, OpId, PipelineStage,
};
pub use error::{GpuError, GpuResult};
pub use pipeline_scheduler::PipelineScheduler;
