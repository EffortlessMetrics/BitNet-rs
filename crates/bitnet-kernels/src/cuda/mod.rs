//! CUDA kernel scaffolding for BitNet inference operations.
//!
//! This module provides specialized CUDA kernel launch configurations and stubs
//! for high-performance GPU inference. Each submodule targets a specific operation
//! in the BitNet transformer pipeline:
//!
//! - [`qk256_gemv`]: QK256 2-bit dequantization fused with GEMV
//! - [`attention`]: Scaled dot-product attention with causal masking
//! - [`rmsnorm`]: RMSNorm layer normalization
//! - [`rope`]: Rotary Position Embedding (RoPE)
//! - [`crate::reduction`]: Parallel reductions (sum, max, min, mean, L2 norm)
//! - [`softmax`]: Numerically stable row-wise softmax with temperature scaling
//! - [`crate::scatter_gather`]: Scatter/gather indexed tensor operations with reductions
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! These stubs define launch configurations and function signatures; actual PTX
//! compilation and kernel dispatch are handled by the parent [`super::gpu::cuda`]
//! module via `cudarc`.

pub mod attention;
pub mod kv_cache;
pub mod qk256_gemv;
pub mod rmsnorm;
pub mod rope;
pub mod softmax;

pub use attention::{AttentionKernelConfig, launch_attention};
pub use kv_cache::{CacheDtype, CacheStats, KvCacheBuffer, KvCacheConfig, launch_append_kv};
pub use qk256_gemv::{Qk256GemvConfig, launch_qk256_gemv};
pub use rmsnorm::{RmsNormConfig, launch_rmsnorm};
pub use rope::{RopeConfig, launch_rope, rope_forward, rope_forward_cpu};

// Re-export scatter/gather types from the crate-level module (always compiled).
pub use crate::scatter_gather::{
    GatherConfig, ScatterGatherKernel, ScatterMode, gather_cpu, gather_forward, index_select_cpu,
    scatter_cpu, scatter_forward,
};

// Re-export reduction types from the crate-level module (always compiled).
pub use crate::reduction::{
    ReductionConfig, ReductionOp, launch_reduce_cols_f32, launch_reduce_f32,
    launch_reduce_rows_f32, reduce_cols_f32, reduce_f32, reduce_rows_f32,
};
pub use softmax::{SoftmaxConfig, launch_softmax, softmax_cpu, softmax_forward};
