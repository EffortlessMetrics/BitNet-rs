//! CPU kernel implementations

pub mod activations;
pub mod attention;
pub mod batch_norm;
pub mod embedding;
pub mod fallback;
pub mod fusion;
pub mod layer_norm;
pub mod pooling;
pub mod quantized_matmul;
pub mod reduction;
pub mod rope;
pub mod simd_math;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

pub use batch_norm::BatchNormConfig;
pub use fallback::*;
pub use simd_math::*;

#[cfg(target_arch = "x86_64")]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub use arm::*;
