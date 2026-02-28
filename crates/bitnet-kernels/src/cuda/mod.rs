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
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! These stubs define launch configurations and function signatures; actual PTX
//! compilation and kernel dispatch are handled by the parent [`super::gpu::cuda`]
//! module via `cudarc`.

pub mod attention;
pub mod qk256_gemv;
pub mod rmsnorm;
pub mod rope;

pub use attention::{AttentionKernelConfig, launch_attention};
pub use qk256_gemv::{Qk256GemvConfig, launch_qk256_gemv};
pub use rmsnorm::{RmsNormConfig, launch_rmsnorm};
pub use rope::{RopeConfig, launch_rope, rope_forward, rope_forward_cpu};
