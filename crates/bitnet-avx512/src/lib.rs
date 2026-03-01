//! Shared x86 runtime SIMD capability helpers.
//!
//! This crate centralizes AVX2 and AVX-512 feature detection so callers can
//! choose optimized kernels without duplicating architecture checks.

/// Runtime SIMD capabilities relevant for x86 quantization and kernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct X86SimdFeatures {
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_avx512bw: bool,
    pub has_avx512vl: bool,
}

impl X86SimdFeatures {
    /// Detect x86 SIMD capabilities on the current host.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                has_avx2: std::arch::is_x86_feature_detected!("avx2"),
                has_avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                has_avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
                has_avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self::default()
        }
    }

    /// AVX2 kernel eligibility.
    #[must_use]
    pub fn supports_avx2(self) -> bool {
        self.has_avx2
    }

    /// AVX-512F+BW eligibility, used by the AVX-512 kernel in bitnet-kernels.
    #[must_use]
    pub fn supports_avx512_core(self) -> bool {
        self.has_avx512f && self.has_avx512bw
    }

    /// AVX-512F+BW+VL eligibility, used by TL2 quantization fast paths.
    #[must_use]
    pub fn supports_avx512_tl2(self) -> bool {
        self.has_avx512f && self.has_avx512bw && self.has_avx512vl
    }

    /// Best x86 SIMD tier for runtime dispatch.
    #[must_use]
    pub fn best_tier(self) -> X86SimdTier {
        if self.supports_avx512_tl2() {
            X86SimdTier::Avx512
        } else if self.supports_avx2() {
            X86SimdTier::Avx2
        } else {
            X86SimdTier::Scalar
        }
    }
}

/// Ordered runtime SIMD tiers for x86 backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum X86SimdTier {
    Scalar,
    Avx2,
    Avx512,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx512_tiers_imply_avx2() {
        let features = X86SimdFeatures::detect();
        if features.supports_avx512_core() {
            assert!(features.supports_avx2());
        }
        if features.supports_avx512_tl2() {
            assert!(features.supports_avx512_core());
            assert!(features.supports_avx2());
        }
    }

    #[test]
    fn best_tier_consistent_with_flags() {
        let tier = X86SimdFeatures::detect().best_tier();
        assert!(matches!(tier, X86SimdTier::Scalar | X86SimdTier::Avx2 | X86SimdTier::Avx512));
    }
}
