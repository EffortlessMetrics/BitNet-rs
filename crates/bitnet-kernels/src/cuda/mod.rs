//! CUDA kernel scaffolding for BitNet inference operations.
//!
//! This module provides specialized CUDA kernel launch configurations and stubs
//! for high-performance GPU inference. Each submodule targets a specific operation
//! in the BitNet transformer pipeline:
//!
//! - [`qk256_gemv`]: QK256 2-bit dequantization fused with GEMV
//! - [`attention`]: Scaled dot-product attention with causal masking
//! - [`rmsnorm`]: RMSNorm layer normalization
//! - [`softmax`]: Numerically stable row-wise softmax with temperature scaling
//! - [`embedding`]: Token-to-vector embedding lookup
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! These stubs define launch configurations and function signatures; actual PTX
//! compilation and kernel dispatch are handled by the parent [`super::gpu::cuda`]
//! module via `cudarc`.

pub mod attention;
pub mod embedding;
pub mod qk256_gemv;
pub mod rmsnorm;
pub mod softmax;

pub use attention::{
    AttentionKernelConfig, HeadDim, KvCacheConfig, MhaConfig, MhaKvCache, launch_attention,
    launch_mha_decode, launch_mha_prefill,
};
pub use embedding::{
    EmbeddingConfig, EmbeddingTied, batched_embedding_lookup, embedding_lookup_cpu,
    launch_embedding_lookup,
};
pub use qk256_gemv::{Qk256GemvConfig, launch_qk256_gemv};

pub use rmsnorm::{RmsNormConfig, launch_rmsnorm, rmsnorm_cpu_reference};
pub use softmax::{SoftmaxConfig, launch_softmax, softmax_cpu};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use rmsnorm::{RMSNORM_KERNEL_SRC, launch_rmsnorm_cuda};
