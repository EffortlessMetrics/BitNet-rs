//! CUDA kernel scaffolding for BitNet inference operations.
//!
//! This module provides specialized CUDA kernel launch configurations and stubs
//! for high-performance GPU inference. Each submodule targets a specific operation
//! in the BitNet transformer pipeline:
//!
//! - [`activations`]: SiLU, GELU, ReLU, and fused SiLU-gate activations
//! - [`qk256_gemv`]: QK256 2-bit dequantization fused with GEMV
//! - [`attention`]: Scaled dot-product attention with causal masking
//! - [`batch_norm`]: Batch normalization with training/eval mode support
//! - [`rmsnorm`]: RMSNorm layer normalization
//! - [`rope`]: Rotary Position Embedding (RoPE)
//! - [`crate::reduction`]: Parallel reductions (sum, max, min, mean, L2 norm)
//! - [`softmax`]: Numerically stable row-wise softmax with temperature scaling
//! - [`quantized_matmul`]: I2_S quantized matrix multiplication with CPU fallback
//! - [`crate::scatter_gather`]: Scatter/gather indexed tensor operations with reductions
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! These stubs define launch configurations and function signatures; actual PTX
//! compilation and kernel dispatch are handled by the parent [`super::gpu::cuda`]
//! module via `cudarc`.

pub mod activations;
pub mod attention;
pub mod batch_norm;
pub mod kv_cache;
pub mod pooling;
pub mod qk256_gemv;
pub mod quantized_matmul;
pub mod rmsnorm;
pub mod rope;
pub mod softmax;

pub use activations::{
    ActivationConfig, ActivationType, SiluGateConfig, activation_cpu, launch_activation,
    launch_silu_gate, silu_gate_cpu,
};
pub use attention::{AttentionKernelConfig, launch_attention};
pub use batch_norm::{BatchNormConfig, BatchNormKernel, BatchNormState, batch_norm_cpu};
pub use kv_cache::{CacheDtype, CacheStats, KvCacheBuffer, KvCacheConfig, launch_append_kv};
pub use qk256_gemv::{Qk256GemvConfig, launch_qk256_gemv};
pub use rmsnorm::{RmsNormConfig, launch_rmsnorm};
pub use rope::{RopeConfig, compute_sincos_table, launch_rope, rope_forward, rope_forward_cpu};

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
// Re-export shaped reduction from the crate-level module.
pub use crate::shaped_reduction::reduce_f32 as shaped_reduce_f32;
pub use crate::shaped_reduction::{ShapedReductionConfig, reduction_output_shape};
pub use pooling::{CudaPoolType, PoolingConfig, pooling_cpu, pooling_forward};
pub use softmax::{SoftmaxConfig, launch_softmax, softmax_cpu, softmax_forward};

pub use quantized_matmul::{I2sMatmulConfig, i2s_matmul_cpu, i2s_matmul_forward, pack_i2s};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use activations::{ACTIVATION_KERNEL_SRC, launch_activation_cuda, launch_silu_gate_cuda};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use quantized_matmul::launch_i2s_matmul;
