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
//! Provides optimized OpenCL kernel sources and a hardware-aware
//! [`KernelSelector`] that picks the best variant for each operation.
//!
//! # Kernel Sources
//!
//! The `.cl` files under [`kernels`] are embedded at compile time via
//! `include_str!` so the binary is self-contained.
//!
//! # Example
//!
//! ```rust
//! use bitnet_opencl::kernel_selector::{KernelSelector, KernelVariant};
//!
//! let sel = KernelSelector::new(256, true, 4);
//! assert_eq!(sel.select_matmul(512, 512, 512), KernelVariant::Vectorized);
//! ```

pub mod kernel_selector;
pub mod kernels;
