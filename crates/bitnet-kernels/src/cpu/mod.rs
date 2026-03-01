//! CPU kernel implementations

pub mod activations;
pub mod attention;
pub mod embedding;
pub mod fallback;
pub mod fusion;
pub mod kv_cache;
pub mod layer_norm;
pub mod pooling;
pub mod quantized_matmul;
pub mod reduction;
pub mod rope;
pub mod scatter_gather;
pub mod simd_math;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

pub use fallback::*;
pub use simd_math::*;

// Re-export position-encoding embedding types.
pub use embedding::{CpuEmbeddingConfig, PackedEmbeddingTable};

// Re-export KV cache types and operations.
pub use kv_cache::{
    KvCache, KvCacheBlock, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear,
    kv_cache_memory_usage, kv_cache_slice, paged_kv_cache_alloc,
};

#[cfg(target_arch = "x86_64")]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub use arm::*;
