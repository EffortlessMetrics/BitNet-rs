//! CUDA kernel scaffolding for BitNet inference operations.
//!
//! This module provides specialized CUDA kernel launch configurations and stubs
//! for high-performance GPU inference. Each submodule targets a specific operation
//! in the BitNet transformer pipeline:
//!
//! - [`gemm`]: General matrix multiply (tiled GEMM) for transformer layers
//! - [`qk256_gemv`]: QK256 2-bit dequantization fused with GEMV
//! - [`attention`]: Scaled dot-product attention with causal masking
//! - [`rmsnorm`]: RMSNorm layer normalization
//! - [`rope`]: Rotary Position Embedding (RoPE)
//! - [`crate::reduction`]: Parallel reductions (sum, max, min, mean, L2 norm)
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! These stubs define launch configurations and function signatures; actual PTX
//! compilation and kernel dispatch are handled by the parent [`super::gpu::cuda`]
//! module via `cudarc`.

pub mod attention;
pub mod gemm;
pub mod qk256_gemv;
pub mod rmsnorm;
pub mod rope;

pub use attention::{AttentionKernelConfig, launch_attention};
pub use gemm::{DType, GemmConfig, Layout, TileSize, gemm_cpu_reference, launch_gemm};
pub use qk256_gemv::{Qk256GemvConfig, launch_qk256_gemv};
pub use rmsnorm::{RmsNormConfig, launch_rmsnorm};
pub use rope::{RopeConfig, launch_rope, rope_forward, rope_forward_cpu};

// Re-export reduction types from the crate-level module (always compiled).
pub use crate::reduction::{
    ReductionConfig, ReductionOp, launch_reduce_cols_f32, launch_reduce_f32,
    launch_reduce_rows_f32, reduce_cols_f32, reduce_f32, reduce_rows_f32,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use gemm::{GEMM_KERNEL_SRC, launch_gemm_cuda};
