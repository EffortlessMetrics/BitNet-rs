//! CPU kernel implementations

pub mod attention;
pub mod conv1d;
pub mod embedding;
pub mod fallback;
pub mod fusion;
pub mod pooling;
pub mod quantized_matmul;
pub mod rope;
pub mod simd_math;
pub mod softmax;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

pub use fallback::*;
pub use simd_math::*;

// Re-export position-encoding embedding types.
pub use embedding::{CpuEmbeddingConfig, PackedEmbeddingTable};

#[cfg(target_arch = "x86_64")]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub use arm::*;
