//! SIMD feature detection and capability reporting.
//!
//! Runtime detection of CPU SIMD capabilities, feature level
//! classification, and capability summaries.

/// SIMD feature level (ordered from lowest to highest capability).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdFeatureLevel {
    /// No SIMD (scalar only).
    Scalar,
    /// SSE2 (128-bit, baseline x86-64).
    Sse2,
    /// SSE4.1/4.2 (improved 128-bit).
    Sse4,
    /// AVX (256-bit float).
    Avx,
    /// AVX2 + FMA (256-bit integer + fused multiply-add).
    Avx2,
    /// AVX-512F (512-bit).
    Avx512,
    /// ARM NEON (128-bit).
    Neon,
    /// ARM NEON + dotprod.
    NeonDotprod,
}

impl SimdFeatureLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Sse2 => "sse2",
            Self::Sse4 => "sse4",
            Self::Avx => "avx",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Neon => "neon",
            Self::NeonDotprod => "neon+dotprod",
        }
    }

    /// Register width in bits.
    pub fn register_bits(&self) -> usize {
        match self {
            Self::Scalar => 64,
            Self::Sse2 | Self::Sse4 => 128,
            Self::Avx | Self::Avx2 => 256,
            Self::Avx512 => 512,
            Self::Neon | Self::NeonDotprod => 128,
        }
    }

    /// Floats per register (f32).
    pub fn f32_lanes(&self) -> usize {
        self.register_bits() / 32
    }

    /// Whether FMA (fused multiply-add) is likely available.
    pub fn has_fma(&self) -> bool {
        matches!(self, Self::Avx2 | Self::Avx512)
    }
}

/// Detected SIMD capabilities on the current CPU.
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub features: Vec<SimdFeatureLevel>,
    pub best: SimdFeatureLevel,
}

impl SimdCapabilities {
    /// Detect available SIMD features at runtime.
    pub fn detect() -> Self {
        let mut features = vec![SimdFeatureLevel::Scalar];

        #[cfg(target_arch = "x86_64")]
        {
            // SSE2 is baseline on x86-64
            features.push(SimdFeatureLevel::Sse2);

            if std::arch::is_x86_feature_detected!("sse4.1") {
                features.push(SimdFeatureLevel::Sse4);
            }
            if std::arch::is_x86_feature_detected!("avx") {
                features.push(SimdFeatureLevel::Avx);
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                features.push(SimdFeatureLevel::Avx2);
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                features.push(SimdFeatureLevel::Avx512);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is baseline on aarch64
            features.push(SimdFeatureLevel::Neon);
        }

        let best = features
            .iter()
            .copied()
            .max()
            .unwrap_or(SimdFeatureLevel::Scalar);
        Self { features, best }
    }

    pub fn has(&self, level: SimdFeatureLevel) -> bool {
        self.features.contains(&level)
    }

    pub fn summary(&self) -> String {
        format!(
            "best={}, available=[{}]",
            self.best.as_str(),
            self.features
                .iter()
                .map(|f| f.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Check if we're running on a Kaby Lake class CPU (AVX2 without AVX-512).
pub fn is_kaby_lake_class() -> bool {
    let caps = SimdCapabilities::detect();
    caps.has(SimdFeatureLevel::Avx2) && !caps.has(SimdFeatureLevel::Avx512)
}

/// Recommended vector width for the current CPU (in f32 elements).
pub fn recommended_vector_width() -> usize {
    SimdCapabilities::detect().best.f32_lanes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_level_ordering() {
        assert!(SimdFeatureLevel::Avx2 > SimdFeatureLevel::Avx);
        assert!(SimdFeatureLevel::Avx > SimdFeatureLevel::Sse4);
        assert!(SimdFeatureLevel::Scalar < SimdFeatureLevel::Sse2);
    }

    #[test]
    fn test_register_bits() {
        assert_eq!(SimdFeatureLevel::Scalar.register_bits(), 64);
        assert_eq!(SimdFeatureLevel::Sse2.register_bits(), 128);
        assert_eq!(SimdFeatureLevel::Avx2.register_bits(), 256);
        assert_eq!(SimdFeatureLevel::Avx512.register_bits(), 512);
    }

    #[test]
    fn test_f32_lanes() {
        assert_eq!(SimdFeatureLevel::Scalar.f32_lanes(), 2);
        assert_eq!(SimdFeatureLevel::Sse2.f32_lanes(), 4);
        assert_eq!(SimdFeatureLevel::Avx2.f32_lanes(), 8);
        assert_eq!(SimdFeatureLevel::Avx512.f32_lanes(), 16);
    }

    #[test]
    fn test_has_fma() {
        assert!(SimdFeatureLevel::Avx2.has_fma());
        assert!(SimdFeatureLevel::Avx512.has_fma());
        assert!(!SimdFeatureLevel::Avx.has_fma());
        assert!(!SimdFeatureLevel::Sse4.has_fma());
    }

    #[test]
    fn test_detect() {
        let caps = SimdCapabilities::detect();
        assert!(caps.has(SimdFeatureLevel::Scalar));
        assert!(caps.best >= SimdFeatureLevel::Scalar);
    }

    #[test]
    fn test_summary() {
        let caps = SimdCapabilities::detect();
        let s = caps.summary();
        assert!(s.contains("best="));
        assert!(s.contains("scalar"));
    }

    #[test]
    fn test_recommended_width() {
        let w = recommended_vector_width();
        assert!(w >= 2);
    }

    #[test]
    fn test_as_str() {
        assert_eq!(SimdFeatureLevel::Avx2.as_str(), "avx2");
        assert_eq!(SimdFeatureLevel::Neon.as_str(), "neon");
        assert_eq!(SimdFeatureLevel::Scalar.as_str(), "scalar");
    }

    #[test]
    fn test_neon_register() {
        assert_eq!(SimdFeatureLevel::Neon.register_bits(), 128);
        assert_eq!(SimdFeatureLevel::NeonDotprod.register_bits(), 128);
    }

    #[test]
    fn test_detect_has_sse2_on_x86() {
        let caps = SimdCapabilities::detect();
        #[cfg(target_arch = "x86_64")]
        assert!(caps.has(SimdFeatureLevel::Sse2));
        let _ = caps; // use on non-x86
    }

    #[test]
    fn test_kaby_lake_class() {
        // Just verify it doesn't panic
        let _ = is_kaby_lake_class();
    }

    #[test]
    fn test_feature_level_copy() {
        let level = SimdFeatureLevel::Avx2;
        let copy = level;
        assert_eq!(level, copy);
    }
}
