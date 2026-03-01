//! CPU kernel implementations

pub mod activations;
pub mod attention;
pub mod batch_norm;
pub mod embedding;
pub mod fallback;
pub mod fusion;
pub mod layer_norm;
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

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

pub use batch_norm::BatchNormConfig;
pub use fallback::*;
pub use simd_math::*;

// Re-export position-encoding embedding types.
pub use embedding::{CpuEmbeddingConfig, PackedEmbeddingTable};

#[cfg(target_arch = "x86_64")]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub use arm::*;
