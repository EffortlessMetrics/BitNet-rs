//! AVX2-optimized `LayerNorm`, `RMSNorm`, and `BatchNorm` operations.
//!
//! Provides fused normalize+scale+bias in a single pass for cache efficiency.
//! Falls back to scalar implementation when AVX2 is unavailable at runtime.

mod norm;
mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;

pub use norm::{BatchNorm, BatchNormParams, LayerNorm, NormError, NormResult, RmsNorm};

/// Returns `true` if AVX2 acceleration is available at runtime.
#[must_use]
pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests;
