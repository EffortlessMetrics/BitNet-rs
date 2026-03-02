//! CPU kernel implementations

pub mod activations;
pub mod batch;
pub use batch::{batched_add, batched_layer_norm, batched_matmul, batched_softmax};
pub mod attention;
pub mod attention_mask;
pub use attention::{
    AttentionConfig, AttentionKernel, CpuAttention, CpuAttentionConfig, GqaConfig,
    apply_rotary_embedding, attention_with_kv_cache, causal_attention, causal_mask,
    masked_attention, multi_head_attention_cpu, scaled_dot_product_attention,
};
pub mod batch_norm;
pub mod batch_normalization;
pub use batch_normalization::*;
pub mod concat;
pub use concat::ConcatKernel;
pub mod conv2d;
pub mod convolution;
pub mod dequant;
pub use conv2d::{Conv2dConfig, compute_output_size, conv2d, depthwise_conv2d, im2col};
pub use convolution::{
    Conv1dConfig, PaddingMode, apply_padding, col2im as col2im_1d, compute_output_length, conv1d,
    conv1d_avx2, conv1d_depthwise, conv1d_f32, conv1d_grouped, conv1d_pointwise, conv1d_transposed,
    im2col as im2col_1d,
};
pub mod embedding;
pub mod fallback;
pub mod ffn;
pub mod fusion;
pub mod gating;
pub mod kv_cache;
pub mod layer_norm;
pub use layer_norm::{
    GroupNormConfig, LayerNormConfig, batch_group_norm, batch_instance_norm, batch_layer_norm,
    batch_rms_norm, group_norm, instance_norm, layer_norm as cpu_layer_norm, rms_norm,
};
pub mod linear;
pub use linear::{LinearConfig, linear_cpu, linear_forward};
pub mod loss;
pub mod pooling;
pub use pooling::{
    PoolConfig, PoolType, PoolingConfig, PoolingKernel, adaptive_avg_pool_1d, adaptive_avg_pool_2d,
    adaptive_max_pool1d, avg_pool1d, avg_pool1d_avx2, global_avg_pool, global_max_pool, lp_pool1d,
    max_pool1d, max_pool1d_avx2, max_unpool1d, pool_1d, pool_2d,
};
pub mod quantize;
pub mod quantized_matmul;
pub mod reduction;
pub mod residual;
pub use residual::{add_residual, add_residual_scaled, add_residual_with_dropout};
pub mod cache_matmul;
pub mod rope;
pub mod rope_simd;
pub use rope_simd::*;
pub mod scatter_gather;
pub mod simd_math;
pub mod simd_matmul;
pub mod transpose;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

#[cfg(target_arch = "aarch64")]
pub mod neon_activations;

#[cfg(target_arch = "aarch64")]
pub mod neon_rope;

#[cfg(target_arch = "aarch64")]
pub mod neon_elementwise;

#[cfg(target_arch = "aarch64")]
pub mod neon_kv_cache;

#[cfg(target_arch = "aarch64")]
pub mod neon_layernorm;

#[cfg(target_arch = "aarch64")]
pub mod neon_pooling;

#[cfg(target_arch = "aarch64")]
pub mod neon_batch_norm;

#[cfg(target_arch = "aarch64")]
pub mod neon_quantized_matmul;

#[cfg(target_arch = "aarch64")]
pub mod neon_reductions;

#[cfg(target_arch = "aarch64")]
pub mod neon_scatter_gather;

#[cfg(target_arch = "aarch64")]
pub mod neon_softmax;

#[cfg(target_arch = "aarch64")]
pub mod neon_transpose;

#[cfg(target_arch = "aarch64")]
pub mod neon_convolution;

#[cfg(target_arch = "aarch64")]
pub mod neon_padding_clipping;

#[cfg(target_arch = "aarch64")]
pub mod neon_activation_suite;

#[cfg(target_arch = "aarch64")]
pub mod neon_inference_bridge;

pub use activations::ActivationType;
pub use activations::{
    apply_activation, elu_vec, gelu_approx_vec, gelu_inplace, gelu_vec, hard_sigmoid_vec,
    hard_swish_vec, leaky_relu_vec, mish_vec, relu_inplace, silu_inplace, silu_vec, softplus_beta,
    softplus_vec,
};
pub use batch_norm::BatchNormConfig;
pub use fallback::*;
pub use ffn::{FfnActivation, FfnConfig, ffn_forward, ffn_forward_batched, gated_ffn_forward};
pub use gating::{GatingType, apply_gating, geglu, reglu, swiglu};
pub use scatter_gather::{
    ScatterGatherConfig, ScatterReduce, gather_1d, gather_2d, index_select, scatter_1d, scatter_2d,
    scatter_add, scatter_max,
};
pub use simd_math::*;

// Re-export position-encoding embedding types.
pub use embedding::{CpuEmbeddingConfig, PackedEmbeddingTable};
pub use loss::*;

// Re-export KV cache types and operations.
pub use kv_cache::{
    KvCache, KvCacheBlock, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear,
    kv_cache_memory_usage, kv_cache_slice, paged_kv_cache_alloc,
};

// Re-export new embedding operations.
pub use embedding::{
    add_positional_encoding, embedding_bag_mean, embedding_bag_sum, embedding_lookup_batched,
    embedding_lookup_with_padding, positional_embedding, positional_encoding,
};

#[cfg(target_arch = "x86_64")]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub use arm::*;
pub mod gather;
pub use gather::{gather_rows, index_select_dim, scatter_add_rows};
pub mod tensor_parallel;
