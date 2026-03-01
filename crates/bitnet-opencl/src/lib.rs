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
//! OpenCL backend for BitNet inference.
//!
//! This crate provides OpenCL 3.0 compute backend support for BitNet-rs,
//! including:
//!
//! - **WASM-compatible kernel validation** ([`wasm_shim`]) — parse and validate
//!   OpenCL kernel source on any target (including `wasm32`) without FFI
//! - **Unified Shared Memory (USM)** ([`usm`]) — zero-copy host↔device data
//!   access via `clSVMAlloc`, with automatic fallback to explicit buffer copies
//! - **Peer-to-peer transfers** ([`p2p`]) — GPU-to-GPU memory transfers with
//!   bandwidth measurement and automatic fallback to host-staged copies
//!
//! # Feature Gates
//!
//! Real OpenCL FFI calls are gated behind `#[cfg(not(target_arch = "wasm32"))]`.
//! The [`wasm_shim`] module provides pure-Rust validation that works everywhere.

pub mod wasm_shim;
pub mod usm;
pub mod p2p;

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
