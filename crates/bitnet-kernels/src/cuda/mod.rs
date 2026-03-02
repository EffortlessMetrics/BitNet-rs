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
//! - [`attention_mask`]: Attention mask generation (causal, padding, sliding window,
//!   block-sparse, ALiBi, prefix LM) and application (additive, multiplicative)
//! - [`batch_norm`]: Batch normalization with training/eval mode support
//! - [`conv1d`]: 1-D convolution with stride, padding, dilation, groups
//! - [`layernorm`]: Full LayerNorm and RMSNorm with CPU fallback and GPU dispatch
//! - [`rmsnorm`]: RMSNorm layer normalization
//! - [`rope`]: Rotary Position Embedding (RoPE)
//! - [`crate::reduction`]: Parallel reductions (sum, max, min, mean, L2 norm)
//! - [`softmax`]: Numerically stable row-wise softmax with temperature scaling,
//!   causal masking, log-softmax, in-place mode, and batched multi-head support
//! - [`matmul`]: Dense f32/f16 matrix multiplication (tiled GEMM) with batched and
//!   transpose support
//! - [`linear`]: Linear projection (y = xW^T + bias) CUDA kernel and launch stub
//! - [`quantized_matmul`]: I2_S quantized matrix multiplication with CPU fallback
//! - [`transpose`]: 2D/ND transpose and reshape with tiled shared-memory CUDA kernels
//! - [`embedding`]: Token and positional embedding lookup with padding support
//! - [`crate::scatter_gather`]: Scatter/gather indexed tensor operations with reductions
//! - [`elementwise`]: Element-wise arithmetic (add/mul/sub/div) and activations with fused ops
//! - [`warp_ops`]: Warp-level primitives (reduce, shuffle, ballot, scan, cooperative softmax)
//!
//! All code is feature-gated behind `#[cfg(any(feature = "gpu", feature = "cuda"))]`.
//! These stubs define launch configurations and function signatures; actual PTX
//! compilation and kernel dispatch are handled by the parent `super::gpu::cuda`
//! module via `cudarc`.

pub mod activations;
pub mod attention;
pub mod attention_mask;
pub mod batch_norm;
pub mod conv1d;
pub mod dequant;
pub mod elementwise;
pub mod embedding;
pub mod fused_attention;
pub mod fusion;
pub mod gating;
pub mod graph_exec;
pub mod kernel_fusion;
pub mod kv_cache;
pub mod kv_cache_gpu;
pub mod layernorm;
pub mod linear;
pub mod loss;
pub mod matmul;
pub mod memory_pool;
pub mod multi_head_attention;
pub mod pooling;
pub mod profiling;
pub mod qk256_gemv;
pub mod quantize;
pub mod quantized_gemm;
pub mod quantized_matmul;
pub mod residual;
pub mod rmsnorm;
pub mod rope;
pub mod softmax;
pub mod stream_mgmt;
pub mod transpose;
pub mod warp_ops;

pub use activations::{
    ActivationConfig, ActivationType, SiluGateConfig, activation_cpu, launch_activation,
    launch_silu_gate, silu_gate_cpu,
};
pub use attention::{
    AttentionConfig, AttentionKernelConfig, CudaAttentionConfig, attention_cpu_fallback,
    attention_forward, attention_forward_cpu, batch_attention_cpu, chunked_attention_cpu,
    launch_attention, masked_attention_cpu_fallback, multi_head_attention_cpu_fallback,
};

pub use attention_mask::{
    AlibiConfig, AttentionMaskConfig, BlockSparseConfig, NEG_INF, PrefixMaskConfig,
    SlidingWindowConfig, alibi_mask, apply_mask_additive, apply_mask_multiplicative,
    apply_mask_to_scores, block_sparse_mask, causal_mask, combined_mask, compute_alibi_slopes,
    create_prefix_mask, padding_mask, sliding_window_mask,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use attention::ATTENTION_KERNEL_SRC;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use attention_mask::{
    ATTENTION_MASK_KERNEL_SRC, launch_alibi_mask, launch_apply_mask_additive,
    launch_apply_mask_multiplicative, launch_causal_mask, launch_sliding_window_mask,
};
pub use batch_norm::{
    BatchNormConfig, BatchNormKernel, BatchNormState, CudaBatchNormConfig, batch_norm_cpu,
    batch_norm_cpu_fallback, batch_norm_inference_cpu_fallback,
};
pub use conv1d::{Conv1dConfig, PaddingMode, conv1d_cpu, conv1d_forward, launch_conv1d};
pub use kv_cache::{CacheDtype, CacheStats, KvCacheBuffer, KvCacheConfig, launch_append_kv};
pub use kv_cache_gpu::{
    KvCacheGpuConfig, KvCacheGpuError, KvCacheGpuMetrics, KvCacheGpuState, PageTable,
    QuantizedKvResult, kv_cache_append, kv_cache_copy_on_write, kv_cache_cow_materialize,
    kv_cache_defrag, kv_cache_dequantize, kv_cache_evict, kv_cache_gpu_metrics,
    kv_cache_paged_lookup, kv_cache_prefetch, kv_cache_quantize, kv_cache_rotate,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use kv_cache_gpu::{
    KV_CACHE_GPU_KERNEL_SRC, launch_kv_cache_append_gpu, launch_kv_cache_gather_gpu,
};
pub use layernorm::{
    LayerNormConfig, batch_layer_norm_cpu, layer_norm_cpu_fallback, layer_norm_forward,
    rms_norm_cpu_fallback, rms_norm_forward,
};
pub use linear::{LINEAR_KERNEL_SRC, launch_linear};
pub use qk256_gemv::{Qk256GemvConfig, launch_qk256_gemv};
pub use rmsnorm::{RmsNormConfig, launch_rmsnorm};
pub use rope::{
    RopeConfig, apply_rope, apply_rope_batched, build_rope_freqs, compute_sincos_table,
    launch_rope, launch_rope_backward, rope_backward, rope_backward_cpu, rope_forward,
    rope_forward_cpu,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use rope::{ROPE_BACKWARD_KERNEL_SRC, ROPE_FORWARD_KERNEL_SRC};

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
pub use fused_attention::{
    AttentionMetrics, AttentionPattern, FusedAttentionConfig, FusedAttentionError,
    apply_alibi_bias, apply_attention_mask, compute_attention_scores, flash_attention_forward,
    fused_attention_forward, grouped_query_attention, multi_head_attention,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use fused_attention::{FUSED_ATTENTION_KERNEL_SRC, launch_fused_attention};
pub use fusion::{
    FusedElementwiseLaunchConfig, FusedMatmulLaunchConfig, FusedOp, FusionConfig, FusionError,
    fused_add_rmsnorm, fused_add_rmsnorm_cpu, fused_gelu_linear, fused_gelu_linear_cpu,
    fused_rmsnorm_linear, fused_rmsnorm_linear_cpu, fused_scale_add, fused_scale_add_cpu,
    fused_softmax_mask, fused_softmax_mask_cpu,
};
pub use pooling::{
    AdaptivePool2dConfig, CudaPoolType, Pool2dConfig, PoolingConfig, adaptive_avg_pool2d_cpu,
    adaptive_avg_pool2d_forward, avg_pool2d_cpu, avg_pool2d_forward, launch_adaptive_avg_pool2d,
    launch_avg_pool2d, launch_max_pool2d, max_pool2d_cpu, max_pool2d_forward, pooling_cpu,
    pooling_forward,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use pooling::{ADAPTIVE_AVG_POOL2D_KERNEL_SRC, AVG_POOL2D_KERNEL_SRC, MAX_POOL2D_KERNEL_SRC};
pub use softmax::{
    SoftmaxConfig, launch_softmax, online_softmax_cpu, softmax_backward_cpu, softmax_cpu,
    softmax_forward,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use softmax::SOFTMAX_KERNEL_SRC;

pub use dequant::{
    DequantConfig, DequantPrecision, QK256_BLOCK_SIZE, QuantBitWidth, ScaleMode,
    batch_dequantize_int2_to_f32, dequantize_int2_per_channel_f32, dequantize_int2_to_f16,
    dequantize_int2_to_f32, dequantize_int2_uniform_f32, dequantize_int4_to_f16,
    dequantize_int4_to_f32, dequantize_int8_to_f16, dequantize_int8_to_f32,
    dequantize_int8_uniform_f32, dequantize_qk256_to_f16, dequantize_qk256_to_f32,
};
pub use matmul::{
    GemmConfig, MatmulConfig, MatmulDtype, matmul_cpu, matmul_f16_cpu, matmul_f16_forward,
    matmul_forward, matmul_tiled_cpu,
};
pub use quantize::{
    QuantMethod, QuantizeConfig, calibrate_scales, dequantize_i2s_cpu, dequantize_ternary_cpu,
    quantize_i2s_cpu, quantize_ternary_cpu,
};
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

pub use embedding::{
    EmbeddingKernelConfig, PositionEmbeddingConfig, embedding_forward, embedding_lookup_cpu,
    embedding_with_position_cpu, launch_embedding_lookup, launch_position_embedding,
    position_embedding_forward,
};

pub use gating::{GatingConfig, GatingType, gating_cpu, launch_gating};
pub use loss::{
    LossConfig, LossReduction, binary_cross_entropy, contrastive_loss, cross_entropy_loss,
    cross_entropy_loss_forward, cross_entropy_with_logits, focal_loss, huber_loss,
    huber_loss_forward, kl_divergence, label_smoothing_ce, mse_loss, mse_loss_forward,
    perplexity_from_logits, triplet_loss,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use loss::{LOSS_KERNEL_SRC, launch_cross_entropy_loss, launch_huber_loss, launch_mse_loss};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use warp_ops::WARP_OPS_KERNEL_SRC;
pub use warp_ops::{
    DEFAULT_WARP_SIZE, WarpConfig, block_reduce_max, block_reduce_sum, cooperative_softmax,
    warp_all, warp_any, warp_ballot, warp_broadcast, warp_exclusive_scan, warp_match,
    warp_prefix_sum, warp_reduce_max, warp_reduce_min, warp_reduce_sum, warp_shuffle,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use gating::{GATING_KERNEL_SRC, launch_gating_cuda};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use embedding::{EMBEDDING_LOOKUP_KERNEL_SRC, EMBEDDING_WITH_POSITION_KERNEL_SRC};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use activations::{ACTIVATION_KERNEL_SRC, launch_activation_cuda, launch_silu_gate_cuda};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use elementwise::{ELEMENTWISE_BINARY_KERNEL_SRC, ELEMENTWISE_UNARY_KERNEL_SRC};
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use layernorm::LAYERNORM_KERNEL_SRC;

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use batch_norm::{BATCH_NORM_INFERENCE_KERNEL_SRC, BATCH_NORM_TRAIN_KERNEL_SRC};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use matmul::{launch_matmul, launch_matmul_f16};
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use quantized_matmul::{I2S_MATMUL_KERNEL_SRC, launch_i2s_matmul};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use quantize::{
    DEQUANTIZE_I2S_KERNEL_SRC, DEQUANTIZE_TERNARY_KERNEL_SRC, QUANTIZE_I2S_KERNEL_SRC,
    QUANTIZE_TERNARY_KERNEL_SRC,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use dequant::{
    DEQUANT_INT2_F32_KERNEL_SRC, DEQUANT_INT4_F32_KERNEL_SRC, DEQUANT_INT8_F32_KERNEL_SRC,
    DEQUANT_QK256_F32_KERNEL_SRC,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use fusion::{
    FUSION_KERNEL_SRC, launch_fused_add_rmsnorm_cuda, launch_fused_gelu_linear_cuda,
    launch_fused_rmsnorm_linear_cuda, launch_fused_scale_add_cuda, launch_fused_softmax_mask_cuda,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use transpose::{TRANSPOSE_2D_KERNEL_SRC, TRANSPOSE_ND_KERNEL_SRC, launch_transpose_2d};

pub use stream_mgmt::{
    DefaultStreamBehavior, DepNode, DispatchResult, PipelineSchedule, PipelineStage,
    PipelineStageKind, ProfileRecord, ScheduleStrategy, ScheduledTask, StreamAssignment,
    StreamConfig, StreamEvent, StreamHandle, StreamOp, StreamPool, StreamPriority,
    StreamPriorityManager, StreamProfiler, StreamScheduler, StreamUtilization,
    dependency_graph_to_streams, event_record, event_wait, multi_stream_dispatch, pipeline_stages,
    stream_sync,
};

pub use kernel_fusion::{
    FusedKernel, FusionGraph, FusionNode, FusionPattern, KernelFusionConfig, KernelFusionError,
    LaunchConfig as FusionLaunchConfig, OpType, apply_fusion, detect_fusion_opportunities,
    estimate_fusion_speedup, estimate_register_pressure, estimate_shared_memory,
    fused_attention_score_softmax, fused_layer_norm_residual, fused_matmul_bias,
    fused_matmul_bias_relu,
};

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use kernel_fusion::{
    launch_fused_attention_score_softmax_cuda, launch_fused_layer_norm_residual_cuda,
    launch_fused_matmul_bias_cuda, launch_fused_matmul_bias_relu_cuda,
};
