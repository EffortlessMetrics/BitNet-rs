//! CUDA kernel scaffolding for BitNet inference operations.
//!
//! This module provides specialized CUDA kernel launch configurations and stubs
//! for high-performance GPU inference. Each submodule targets a specific operation
//! in the BitNet transformer pipeline:
//!
//! - [`activations`]: SiLU, GELU, ReLU, and fused SiLU-gate activations
//! - [`fusion`]: Fused operation pairs (RMSNorm+Linear, GELU+Linear, etc.)
//! - [`qk256_gemv`]: QK256 2-bit dequantization fused with GEMV
//! - [`attention`]: Scaled dot-product attention with causal masking
//! - [`batch_norm`]: Batch normalization with training/eval mode support
//! - [`conv1d`]: 1-D convolution with stride, padding, dilation, groups
//! - [`rmsnorm`]: RMSNorm layer normalization
//! - [`rope`]: Rotary Position Embedding (RoPE)
//! - [`crate::reduction`]: Parallel reductions (sum, max, min, mean, L2 norm)
//! - [`softmax`]: Numerically stable row-wise softmax with temperature scaling,
//!   causal masking, log-softmax, in-place mode, and batched multi-head support
//! - [`quantized_matmul`]: I2_S quantized matrix multiplication with CPU fallback
//! - [`transpose`]: 2D/ND transpose and reshape with tiled shared-memory CUDA kernels
//! - [`crate::scatter_gather`]: Scatter/gather indexed tensor operations with reductions
//! - [`elementwise`]: Element-wise arithmetic (add/mul/sub/div) and activations with fused ops
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! These stubs define launch configurations and function signatures; actual PTX
//! compilation and kernel dispatch are handled by the parent [`super::gpu::cuda`]
//! module via `cudarc`.

pub mod activations;
pub mod attention;
pub mod batch_norm;
pub mod conv1d;
pub mod elementwise;
pub mod fusion;
pub mod kv_cache;
pub mod pooling;
pub mod qk256_gemv;
pub mod quantized_matmul;
pub mod rmsnorm;
pub mod rope;
pub mod softmax;
pub mod transpose;

pub use activations::{
    ActivationConfig, ActivationType, SiluGateConfig, activation_cpu, launch_activation,
    launch_silu_gate, silu_gate_cpu,
};
pub use attention::{
    AttentionConfig, AttentionKernelConfig, attention_cpu_fallback, attention_forward,
    launch_attention, masked_attention_cpu_fallback, multi_head_attention_cpu_fallback,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use attention::ATTENTION_KERNEL_SRC;
pub use batch_norm::{BatchNormConfig, BatchNormKernel, BatchNormState, batch_norm_cpu};
pub use conv1d::{Conv1dConfig, PaddingMode, conv1d_cpu, conv1d_forward, launch_conv1d};
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
pub use fusion::{
    FusedElementwiseLaunchConfig, FusedMatmulLaunchConfig, FusedOp, FusionConfig, FusionError,
    fused_add_rmsnorm, fused_add_rmsnorm_cpu, fused_gelu_linear, fused_gelu_linear_cpu,
    fused_rmsnorm_linear, fused_rmsnorm_linear_cpu, fused_scale_add, fused_scale_add_cpu,
    fused_softmax_mask, fused_softmax_mask_cpu,
};
pub use pooling::{CudaPoolType, PoolingConfig, pooling_cpu, pooling_forward};
pub use softmax::{SoftmaxConfig, launch_softmax, softmax_cpu, softmax_forward};

pub use quantized_matmul::{I2sMatmulConfig, i2s_matmul_cpu, i2s_matmul_forward, pack_i2s};
pub use transpose::{
    CudaTransposeConfig, reshape_cpu, transpose_2d_cpu_fallback, transpose_2d_forward,
    transpose_nd_cpu_fallback,
};

pub use elementwise::{
    ElementwiseConfig, ElementwiseOp, elementwise_cpu_fallback, elementwise_unary_cpu,
    fused_elementwise_cpu, launch_elementwise_binary, launch_elementwise_unary,
    launch_fused_add_mul,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use activations::{ACTIVATION_KERNEL_SRC, launch_activation_cuda, launch_silu_gate_cuda};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use elementwise::{ELEMENTWISE_BINARY_KERNEL_SRC, ELEMENTWISE_UNARY_KERNEL_SRC};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use quantized_matmul::launch_i2s_matmul;

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use fusion::{
    FUSION_KERNEL_SRC, launch_fused_add_rmsnorm_cuda, launch_fused_gelu_linear_cuda,
    launch_fused_rmsnorm_linear_cuda, launch_fused_scale_add_cuda, launch_fused_softmax_mask_cuda,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use transpose::{TRANSPOSE_2D_KERNEL_SRC, TRANSPOSE_ND_KERNEL_SRC, launch_transpose_2d};
