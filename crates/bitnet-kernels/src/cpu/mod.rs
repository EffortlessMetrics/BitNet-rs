//! CPU kernel implementations

pub mod activations;
pub mod attention;
pub use attention::{
    AttentionConfig, AttentionKernel, CpuAttentionConfig, GqaConfig, apply_rotary_embedding,
    attention_with_kv_cache, causal_attention, causal_mask, masked_attention,
    multi_head_attention_cpu, scaled_dot_product_attention,
};
pub mod batch_norm;
pub mod embedding;
pub mod fallback;
pub mod fusion;
pub mod kv_cache;
pub mod layer_norm;
pub mod loss;
pub mod pooling;
pub use pooling::{
    PoolConfig, PoolType, PoolingConfig, PoolingKernel, adaptive_avg_pool_1d, adaptive_avg_pool_2d,
    global_avg_pool, global_max_pool, pool_1d, pool_2d,
};
pub mod quantize;
pub mod quantized_matmul;
pub mod reduction;
pub mod rope;
pub mod scatter_gather;
pub mod simd_math;
pub mod simd_matmul;
pub mod transpose;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

pub use batch_norm::BatchNormConfig;
pub use fallback::*;
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

#[cfg(target_arch = "x86_64")]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub use arm::*;
