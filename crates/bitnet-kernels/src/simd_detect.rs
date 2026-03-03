//! Runtime SIMD feature detection.
//!
//! Detect available SIMD instruction sets at runtime on x86_64 and aarch64.

/// SIMD instruction set level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SimdLevel {
    None,
    Sse2,
    Sse42,
    Avx,
    Avx2,
    Avx512,
    Neon,
}

impl SimdLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Sse2 => "sse2",
            Self::Sse42 => "sse4.2",
            Self::Avx => "avx",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Neon => "neon",
        }
    }

    pub fn vector_width_bits(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Sse2 | Self::Sse42 => 128,
            Self::Avx | Self::Avx2 => 256,
            Self::Avx512 => 512,
            Self::Neon => 128,
        }
    }

    pub fn vector_width_f32(&self) -> usize {
        self.vector_width_bits() / 32
    }

    pub fn is_x86(&self) -> bool {
        matches!(self, Self::Sse2 | Self::Sse42 | Self::Avx | Self::Avx2 | Self::Avx512)
    }

    pub fn is_arm(&self) -> bool {
        matches!(self, Self::Neon)
    }
}

/// All SIMD features detected on this machine.
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub has_sse2: bool,
    pub has_sse42: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_fma: bool,
    pub has_avx512f: bool,
    pub has_avx512bw: bool,
    pub has_avx512vnni: bool,
    pub has_neon: bool,
}

impl SimdCapabilities {
    /// Detect available SIMD features at runtime.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_sse2: std::arch::is_x86_feature_detected!("sse2"),
                has_sse42: std::arch::is_x86_feature_detected!("sse4.2"),
                has_avx: std::arch::is_x86_feature_detected!("avx"),
                has_avx2: std::arch::is_x86_feature_detected!("avx2"),
                has_fma: std::arch::is_x86_feature_detected!("fma"),
                has_avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                has_avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
                has_avx512vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
                has_neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                has_sse2: false,
                has_sse42: false,
                has_avx: false,
                has_avx2: false,
                has_fma: false,
                has_avx512f: false,
                has_avx512bw: false,
                has_avx512vnni: false,
                has_neon: true, // NEON is mandatory on aarch64
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                has_sse2: false,
                has_sse42: false,
                has_avx: false,
                has_avx2: false,
                has_fma: false,
                has_avx512f: false,
                has_avx512bw: false,
                has_avx512vnni: false,
                has_neon: false,
            }
        }
    }

    /// Best available SIMD level.
    pub fn best_level(&self) -> SimdLevel {
        if self.has_avx512f {
            SimdLevel::Avx512
        } else if self.has_avx2 {
            SimdLevel::Avx2
        } else if self.has_avx {
            SimdLevel::Avx
        } else if self.has_sse42 {
            SimdLevel::Sse42
        } else if self.has_sse2 {
            SimdLevel::Sse2
        } else if self.has_neon {
            SimdLevel::Neon
        } else {
            SimdLevel::None
        }
    }

    /// All levels supported (from lowest to highest).
    pub fn supported_levels(&self) -> Vec<SimdLevel> {
        let mut levels = Vec::new();
        if self.has_sse2 {
            levels.push(SimdLevel::Sse2);
        }
        if self.has_sse42 {
            levels.push(SimdLevel::Sse42);
        }
        if self.has_avx {
            levels.push(SimdLevel::Avx);
        }
        if self.has_avx2 {
            levels.push(SimdLevel::Avx2);
        }
        if self.has_avx512f {
            levels.push(SimdLevel::Avx512);
        }
        if self.has_neon {
            levels.push(SimdLevel::Neon);
        }
        if levels.is_empty() {
            levels.push(SimdLevel::None);
        }
        levels
    }

    /// Can use FMA instructions (Kaby Lake+).
    pub fn has_fma_support(&self) -> bool {
        self.has_fma
    }

    /// Report string for diagnostics.
    pub fn report(&self) -> String {
        let levels: Vec<&str> = self.supported_levels().iter().map(|l| l.as_str()).collect();
        format!(
            "SIMD: best={}, available=[{}], fma={}",
            self.best_level().as_str(),
            levels.join(", "),
            self.has_fma
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_level_ordering() {
        assert!(SimdLevel::Avx2 > SimdLevel::Avx);
        assert!(SimdLevel::Avx512 > SimdLevel::Avx2);
        assert!(SimdLevel::None < SimdLevel::Sse2);
    }

    #[test]
    fn test_vector_width() {
        assert_eq!(SimdLevel::Avx2.vector_width_bits(), 256);
        assert_eq!(SimdLevel::Avx2.vector_width_f32(), 8);
        assert_eq!(SimdLevel::Avx512.vector_width_f32(), 16);
    }

    #[test]
    fn test_is_x86() {
        assert!(SimdLevel::Avx2.is_x86());
        assert!(!SimdLevel::Neon.is_x86());
        assert!(!SimdLevel::None.is_x86());
    }

    #[test]
    fn test_is_arm() {
        assert!(SimdLevel::Neon.is_arm());
        assert!(!SimdLevel::Avx2.is_arm());
    }

    #[test]
    fn test_detect() {
        let caps = SimdCapabilities::detect();
        // On any modern x86_64, SSE2 should be available
        #[cfg(target_arch = "x86_64")]
        assert!(caps.has_sse2);
        let _ = caps; // use on non-x86
    }

    #[test]
    fn test_best_level() {
        let caps = SimdCapabilities::detect();
        let best = caps.best_level();
        // Should be at least SSE2 on x86_64
        #[cfg(target_arch = "x86_64")]
        assert!(best >= SimdLevel::Sse2);
        let _ = best;
    }

    #[test]
    fn test_supported_levels() {
        let caps = SimdCapabilities::detect();
        let levels = caps.supported_levels();
        assert!(!levels.is_empty());
    }

    #[test]
    fn test_report() {
        let caps = SimdCapabilities::detect();
        let report = caps.report();
        assert!(report.contains("SIMD:"));
        assert!(report.contains("best="));
    }

    #[test]
    fn test_level_str() {
        assert_eq!(SimdLevel::Avx2.as_str(), "avx2");
        assert_eq!(SimdLevel::None.as_str(), "none");
        assert_eq!(SimdLevel::Neon.as_str(), "neon");
    }

    #[test]
    fn test_none_width() {
        assert_eq!(SimdLevel::None.vector_width_bits(), 0);
        assert_eq!(SimdLevel::None.vector_width_f32(), 0);
    }

    #[test]
    fn test_sse_widths() {
        assert_eq!(SimdLevel::Sse2.vector_width_bits(), 128);
        assert_eq!(SimdLevel::Sse42.vector_width_f32(), 4);
    }

    #[test]
    fn test_avx512_width() {
        assert_eq!(SimdLevel::Avx512.vector_width_bits(), 512);
    }
}
