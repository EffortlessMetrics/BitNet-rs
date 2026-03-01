//! CPU capability probing for `BitNet` inference.
//!
//! Focuses on runtime SIMD detection (including Intel x86 AVX levels)
//! and logical core availability.

pub use bitnet_common::kernel_registry::SimdLevel;

/// CPU capabilities detected at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuCapabilities {
    /// Number of logical CPU cores available to the process (always ≥ 1).
    pub core_count: usize,
    /// AVX2 SIMD extension available on this CPU (`x86_64` only).
    pub has_avx2: bool,
    /// AVX-512 SIMD extension available on this CPU (`x86_64` only).
    pub has_avx512: bool,
    /// NEON SIMD extension available (always `true` on `AArch64`, `false` elsewhere).
    pub has_neon: bool,
}

/// Probe the current CPU and return its capabilities.
pub fn probe_cpu() -> CpuCapabilities {
    let core_count = std::thread::available_parallelism().map(std::num::NonZero::get).unwrap_or(1);

    #[cfg(target_arch = "x86_64")]
    let (has_avx2, has_avx512, has_neon) =
        (is_x86_feature_detected!("avx2"), is_x86_feature_detected!("avx512f"), false);

    #[cfg(target_arch = "aarch64")]
    let (has_avx2, has_avx512, has_neon) = (false, false, true);

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let (has_avx2, has_avx512, has_neon) = (false, false, false);

    CpuCapabilities { core_count, has_avx2, has_avx512, has_neon }
}

/// Detect the best SIMD instruction-set level available at runtime.
#[allow(clippy::missing_const_for_fn)]
pub fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            SimdLevel::Avx512
        } else if is_x86_feature_detected!("avx2") {
            SimdLevel::Avx2
        } else if is_x86_feature_detected!("sse4.2") {
            SimdLevel::Sse42
        } else {
            SimdLevel::Scalar
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        SimdLevel::Neon
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdLevel::Scalar
    }
}

/// Return a numeric rank for a [`SimdLevel`] so callers can compare levels.
pub const fn simd_level_rank(level: &SimdLevel) -> u32 {
    match level {
        SimdLevel::Scalar => 0,
        SimdLevel::Sse42 => 1,
        SimdLevel::Avx2 => 2,
        SimdLevel::Avx512 => 3,
        SimdLevel::Neon => 4,
        _ => u32::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_cpu_core_count_at_least_one() {
        assert!(probe_cpu().core_count >= 1);
    }

    #[test]
    fn detect_simd_level_is_debuggable() {
        let _ = format!("{:?}", detect_simd_level());
    }

    #[test]
    fn simd_level_rank_orders_x86_levels() {
        assert!(simd_level_rank(&SimdLevel::Avx512) > simd_level_rank(&SimdLevel::Avx2));
        assert!(simd_level_rank(&SimdLevel::Avx2) > simd_level_rank(&SimdLevel::Sse42));
        assert!(simd_level_rank(&SimdLevel::Sse42) > simd_level_rank(&SimdLevel::Scalar));
    }
}
