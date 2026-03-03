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
pub mod concat;
pub use concat::ConcatKernel;
pub mod conv2d;
pub mod dequant;
pub use conv2d::{Conv2dConfig, compute_output_size, conv2d, depthwise_conv2d, im2col};
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
pub mod layer_norm_simd;
pub mod linear;
pub use linear::{LinearConfig, linear_cpu, linear_forward};
pub mod loss;
pub mod pooling;
pub use pooling::{
    PoolConfig, PoolType, PoolingConfig, PoolingKernel, adaptive_avg_pool_1d, adaptive_avg_pool_2d,
    global_avg_pool, global_max_pool, pool_1d, pool_2d,
};
pub mod quantize;
pub mod quantized_attention;
pub mod quantized_matmul;
pub mod reduction;
pub mod residual;
pub use residual::{add_residual, add_residual_scaled, add_residual_with_dropout};
pub mod rope;
pub mod scatter_gather;
pub mod simd_attention_mask;
pub mod simd_math;
pub mod simd_matmul;
pub use simd_attention_mask::*;
pub mod transpose;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

#[cfg(target_arch = "aarch64")]
pub mod neon_activation_functions;

#[cfg(target_arch = "aarch64")]
pub mod neon_activations;

#[cfg(target_arch = "aarch64")]
pub mod neon_rope;
pub mod neon_rope_v4;

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
pub mod neon_quantized_attention;

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
pub mod neon_batch_norm_v2;

#[cfg(target_arch = "aarch64")]
pub mod neon_padding_clipping;

#[cfg(target_arch = "aarch64")]
pub mod neon_fma_ops;

#[cfg(target_arch = "aarch64")]
pub mod neon_inference_bridge;

#[cfg(target_arch = "aarch64")]
pub mod neon_weight_packing;

#[cfg(target_arch = "aarch64")]
pub mod neon_batch_scheduler;

#[cfg(target_arch = "aarch64")]
pub mod neon_quant_calibration;

#[cfg(target_arch = "aarch64")]
pub mod neon_fused_mlp;

#[cfg(target_arch = "aarch64")]
pub mod neon_attention_masking;

#[cfg(target_arch = "aarch64")]
pub mod neon_flash_attention;

#[cfg(target_arch = "aarch64")]
pub mod neon_gemv;

#[cfg(target_arch = "aarch64")]
pub mod neon_instruction_scheduler;

#[cfg(target_arch = "aarch64")]
pub mod neon_kv_cache_v4;

#[cfg(target_arch = "aarch64")]
pub mod neon_rope_v3;

pub mod neon_simd_utils;
#[cfg(target_arch = "aarch64")]
pub mod neon_weight_dequantize;

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
pub use loss::LossReduction;

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
pub mod batch_normalization;
pub mod cache_aware_matmul;
pub mod cache_matmul;
pub mod convolution;
pub mod elementwise_ops;
pub mod kv_cache_simd;
pub mod layer_fusion;
pub mod matrix_ops;
pub mod mixed_precision;
pub mod numa_aware_ops;
pub mod pipeline_parallel;
pub mod quantized_layer_norm;
pub mod quantized_pipeline;
pub mod rope_simd;
pub mod simd_activation_functions;
pub mod simd_embedding;
pub mod simd_mixed_precision;
pub mod simd_quantized_attention;
pub mod simd_quantized_matmul;
pub mod simd_reduction;
pub mod simd_rope_extended;
pub mod simd_softmax;
pub mod simd_tensor_parallel;
pub mod softmax;
pub mod tensor_parallel;
pub mod x86_qk256_property_tests;
pub use pipeline_parallel::*;
pub mod neon_conv1d;
pub mod neon_gather_scatter;
#[cfg(target_arch = "aarch64")]
pub mod neon_quantized_ffn;

#[cfg(target_arch = "aarch64")]
pub mod neon_attention_mask;
#[cfg(target_arch = "aarch64")]
pub mod neon_attention_scoring;
#[cfg(target_arch = "aarch64")]
pub mod neon_batch_matmul;
#[cfg(target_arch = "aarch64")]
pub mod neon_beam_search;
#[cfg(target_arch = "aarch64")]
pub mod neon_continuous_batching;
#[cfg(target_arch = "aarch64")]
pub mod neon_dynamic_quant;
#[cfg(target_arch = "aarch64")]
pub mod neon_dynamic_quantization;
#[cfg(target_arch = "aarch64")]
pub mod neon_embedding_ops;
#[cfg(target_arch = "aarch64")]
pub mod neon_flash_attn_v2;
#[cfg(target_arch = "aarch64")]
pub mod neon_fused_ops;
#[cfg(target_arch = "aarch64")]
pub mod neon_gelu_ops;
#[cfg(target_arch = "aarch64")]
pub mod neon_group_query_attention;
#[cfg(target_arch = "aarch64")]
pub mod neon_int8_quantization;
#[cfg(target_arch = "aarch64")]
pub mod neon_int8_quantize;
#[cfg(target_arch = "aarch64")]
pub mod neon_kv_cache_paged;
#[cfg(target_arch = "aarch64")]
pub mod neon_kv_cache_v3;
#[cfg(target_arch = "aarch64")]
pub mod neon_layer_norm;
#[cfg(target_arch = "aarch64")]
pub mod neon_layer_norm_v3;
#[cfg(target_arch = "aarch64")]
pub mod neon_matmul_tiling;
#[cfg(target_arch = "aarch64")]
pub mod neon_memory_layout;
#[cfg(target_arch = "aarch64")]
pub mod neon_memory_pool;
#[cfg(target_arch = "aarch64")]
pub mod neon_mixed_precision;
#[cfg(target_arch = "aarch64")]
pub mod neon_model_sharding;
#[cfg(target_arch = "aarch64")]
pub mod neon_pipeline_scheduler;
#[cfg(target_arch = "aarch64")]
pub mod neon_prefix_caching;
#[cfg(target_arch = "aarch64")]
pub mod neon_quant_embed;
#[cfg(target_arch = "aarch64")]
pub mod neon_quantized_activation;
#[cfg(target_arch = "aarch64")]
pub mod neon_quantized_matmul_v2;
#[cfg(target_arch = "aarch64")]
pub mod neon_quantized_softmax;
#[cfg(target_arch = "aarch64")]
pub mod neon_residual;
#[cfg(target_arch = "aarch64")]
pub mod neon_residual_ops;
#[cfg(target_arch = "aarch64")]
pub mod neon_rope_interleaved;
#[cfg(target_arch = "aarch64")]
pub mod neon_rotary_embedding_v2;
#[cfg(target_arch = "aarch64")]
pub mod neon_sliding_window_attn;
#[cfg(target_arch = "aarch64")]
pub mod neon_softmax_stable;
#[cfg(target_arch = "aarch64")]
pub mod neon_sparse_matmul;
#[cfg(target_arch = "aarch64")]
pub mod neon_speculative_decoding;
#[cfg(target_arch = "aarch64")]
pub mod neon_tensor_concat;
#[cfg(target_arch = "aarch64")]
pub mod neon_tensor_parallel;
#[cfg(target_arch = "aarch64")]
pub mod neon_tensor_reshape;
#[cfg(target_arch = "aarch64")]
pub mod neon_token_embedding;
#[cfg(target_arch = "aarch64")]
pub mod neon_vector_reduce;
#[cfg(target_arch = "aarch64")]
pub mod neon_vectorized_search;
#[cfg(target_arch = "aarch64")]
pub mod neon_weight_dequant;
#[cfg(target_arch = "aarch64")]
pub mod neon_weight_dequant_v2;
#[cfg(target_arch = "aarch64")]
pub mod neon_weight_pack;
pub mod wgpu_metal_runner;
