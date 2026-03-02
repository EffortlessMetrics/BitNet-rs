//! CPU SIMD feature detection and dispatch diagnostics.

/// Detected SIMD capabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct SimdCapabilities {
    pub sse2: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,
    pub fma: bool,
    pub neon: bool,
    pub arch: String,
}

impl SimdCapabilities {
    /// Detect runtime SIMD capabilities.
    pub fn detect() -> Self {
        Self {
            sse2: cfg!(target_feature = "sse2") || is_x86_feature_detected_safe("sse2"),
            sse4_1: cfg!(target_feature = "sse4.1") || is_x86_feature_detected_safe("sse4.1"),
            sse4_2: cfg!(target_feature = "sse4.2") || is_x86_feature_detected_safe("sse4.2"),
            avx: cfg!(target_feature = "avx") || is_x86_feature_detected_safe("avx"),
            avx2: cfg!(target_feature = "avx2") || is_x86_feature_detected_safe("avx2"),
            avx512f: cfg!(target_feature = "avx512f") || is_x86_feature_detected_safe("avx512f"),
            avx512bw: cfg!(target_feature = "avx512bw") || is_x86_feature_detected_safe("avx512bw"),
            avx512vnni: is_x86_feature_detected_safe("avx512vnni"),
            fma: cfg!(target_feature = "fma") || is_x86_feature_detected_safe("fma"),
            neon: cfg!(target_arch = "aarch64"),
            arch: std::env::consts::ARCH.to_string(),
        }
    }

    /// Best available SIMD level.
    pub fn best_level(&self) -> SimdLevel {
        if self.avx512f {
            SimdLevel::Avx512
        } else if self.avx2 {
            SimdLevel::Avx2
        } else if self.avx {
            SimdLevel::Avx
        } else if self.sse4_2 {
            SimdLevel::Sse42
        } else if self.sse2 {
            SimdLevel::Sse2
        } else if self.neon {
            SimdLevel::Neon
        } else {
            SimdLevel::Scalar
        }
    }

    /// Vector width in bits for the best available SIMD level.
    pub fn vector_width_bits(&self) -> usize {
        match self.best_level() {
            SimdLevel::Avx512 => 512,
            SimdLevel::Avx2 | SimdLevel::Avx => 256,
            SimdLevel::Sse42 | SimdLevel::Sse2 => 128,
            SimdLevel::Neon => 128,
            SimdLevel::Scalar => 64,
        }
    }

    /// Number of f32 elements that can be processed in one SIMD operation.
    pub fn f32_lanes(&self) -> usize {
        self.vector_width_bits() / 32
    }

    /// Feature summary string.
    pub fn summary(&self) -> String {
        let mut features = Vec::new();
        if self.avx512f {
            features.push("AVX-512F");
        }
        if self.avx512bw {
            features.push("AVX-512BW");
        }
        if self.avx512vnni {
            features.push("AVX-512VNNI");
        }
        if self.avx2 {
            features.push("AVX2");
        }
        if self.avx {
            features.push("AVX");
        }
        if self.fma {
            features.push("FMA");
        }
        if self.sse4_2 {
            features.push("SSE4.2");
        }
        if self.sse4_1 {
            features.push("SSE4.1");
        }
        if self.sse2 {
            features.push("SSE2");
        }
        if self.neon {
            features.push("NEON");
        }
        if features.is_empty() {
            features.push("Scalar only");
        }
        format!(
            "arch={}, features=[{}], level={:?}, lanes={}",
            self.arch,
            features.join(", "),
            self.best_level(),
            self.f32_lanes()
        )
    }
}

/// SIMD level hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    Scalar,
    Sse2,
    Sse42,
    Neon,
    Avx,
    Avx2,
    Avx512,
}

impl SimdLevel {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Scalar => "Scalar",
            Self::Sse2 => "SSE2",
            Self::Sse42 => "SSE4.2",
            Self::Neon => "NEON",
            Self::Avx => "AVX",
            Self::Avx2 => "AVX2",
            Self::Avx512 => "AVX-512",
        }
    }
}

impl std::fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Dispatch recommendation for a given operation.
#[derive(Debug, Clone)]
pub struct DispatchRecommendation {
    pub operation: String,
    pub selected_level: SimdLevel,
    pub reason: String,
    pub estimated_speedup: Option<f32>,
}

/// Get dispatch recommendations based on detected capabilities.
pub fn recommend_dispatch(caps: &SimdCapabilities) -> Vec<DispatchRecommendation> {
    let level = caps.best_level();
    let mut recs = vec![
        DispatchRecommendation {
            operation: "matmul".to_string(),
            selected_level: level,
            reason: format!("Using {} for matrix multiplication", level),
            estimated_speedup: speedup_estimate(level, "matmul"),
        },
        DispatchRecommendation {
            operation: "softmax".to_string(),
            selected_level: level,
            reason: format!("Using {} for softmax", level),
            estimated_speedup: speedup_estimate(level, "softmax"),
        },
        DispatchRecommendation {
            operation: "layer_norm".to_string(),
            selected_level: level,
            reason: format!("Using {} for layer normalization", level),
            estimated_speedup: speedup_estimate(level, "layer_norm"),
        },
        DispatchRecommendation {
            operation: "dequantize".to_string(),
            selected_level: if caps.avx2 { SimdLevel::Avx2 } else { SimdLevel::Scalar },
            reason: if caps.avx2 {
                "AVX2 nibble-LUT dequantization".to_string()
            } else {
                "Scalar dequantization (AVX2 recommended for speedup)".to_string()
            },
            estimated_speedup: if caps.avx2 { Some(3.0) } else { Some(1.0) },
        },
    ];

    // Add FMA note if available
    if caps.fma {
        recs.push(DispatchRecommendation {
            operation: "fma_tiling".to_string(),
            selected_level: SimdLevel::Avx2,
            reason: "FMA available for fused multiply-add tiling".to_string(),
            estimated_speedup: Some(1.5),
        });
    }

    recs
}

fn speedup_estimate(level: SimdLevel, _op: &str) -> Option<f32> {
    match level {
        SimdLevel::Avx512 => Some(8.0),
        SimdLevel::Avx2 => Some(4.0),
        SimdLevel::Avx => Some(2.0),
        SimdLevel::Sse42 | SimdLevel::Sse2 => Some(2.0),
        SimdLevel::Neon => Some(2.0),
        SimdLevel::Scalar => Some(1.0),
    }
}

/// Format dispatch recommendations.
pub fn format_diagnostics(caps: &SimdCapabilities) -> String {
    let mut out = format!("=== SIMD Diagnostics ===\n{}\n\n", caps.summary());
    let recs = recommend_dispatch(caps);
    out.push_str("Dispatch plan:\n");
    for rec in &recs {
        out.push_str(&format!("  {:<15} → {:<8}", rec.operation, rec.selected_level));
        if let Some(speedup) = rec.estimated_speedup {
            out.push_str(&format!(" (~{:.1}x)", speedup));
        }
        out.push('\n');
    }
    out
}

// Helper for safe x86 feature detection
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn is_x86_feature_detected_safe(feature: &str) -> bool {
    match feature {
        "sse2" => std::arch::is_x86_feature_detected!("sse2"),
        "sse4.1" => std::arch::is_x86_feature_detected!("sse4.1"),
        "sse4.2" => std::arch::is_x86_feature_detected!("sse4.2"),
        "avx" => std::arch::is_x86_feature_detected!("avx"),
        "avx2" => std::arch::is_x86_feature_detected!("avx2"),
        "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
        "avx512bw" => std::arch::is_x86_feature_detected!("avx512bw"),
        "avx512vnni" => std::arch::is_x86_feature_detected!("avx512vnni"),
        "fma" => std::arch::is_x86_feature_detected!("fma"),
        _ => false,
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn is_x86_feature_detected_safe(_feature: &str) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect() {
        let caps = SimdCapabilities::detect();
        assert!(!caps.arch.is_empty());
    }

    #[test]
    fn test_best_level_ordering() {
        assert!(SimdLevel::Avx512 > SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 > SimdLevel::Avx);
        assert!(SimdLevel::Avx > SimdLevel::Sse42);
        assert!(SimdLevel::Sse42 > SimdLevel::Scalar);
    }

    #[test]
    fn test_simd_level_display() {
        assert_eq!(format!("{}", SimdLevel::Avx2), "AVX2");
        assert_eq!(format!("{}", SimdLevel::Scalar), "Scalar");
        assert_eq!(format!("{}", SimdLevel::Avx512), "AVX-512");
    }

    #[test]
    fn test_vector_width_avx2() {
        let mut caps = SimdCapabilities::detect();
        // Force AVX2 for testing
        caps.avx2 = true;
        caps.avx512f = false;
        assert_eq!(caps.vector_width_bits(), 256);
        assert_eq!(caps.f32_lanes(), 8);
    }

    #[test]
    fn test_vector_width_avx512() {
        let mut caps = SimdCapabilities::detect();
        caps.avx512f = true;
        assert_eq!(caps.vector_width_bits(), 512);
        assert_eq!(caps.f32_lanes(), 16);
    }

    #[test]
    fn test_vector_width_scalar() {
        let caps = SimdCapabilities {
            sse2: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512f: false,
            avx512bw: false,
            avx512vnni: false,
            fma: false,
            neon: false,
            arch: "test".to_string(),
        };
        assert_eq!(caps.best_level(), SimdLevel::Scalar);
        assert_eq!(caps.f32_lanes(), 2);
    }

    #[test]
    fn test_summary_not_empty() {
        let caps = SimdCapabilities::detect();
        let s = caps.summary();
        assert!(s.contains("arch="));
        assert!(s.contains("features="));
    }

    #[test]
    fn test_recommend_dispatch() {
        let caps = SimdCapabilities::detect();
        let recs = recommend_dispatch(&caps);
        assert!(recs.len() >= 4);
        assert!(recs.iter().any(|r| r.operation == "matmul"));
        assert!(recs.iter().any(|r| r.operation == "softmax"));
    }

    #[test]
    fn test_dispatch_has_speedup() {
        let caps = SimdCapabilities::detect();
        let recs = recommend_dispatch(&caps);
        for rec in &recs {
            assert!(rec.estimated_speedup.is_some());
        }
    }

    #[test]
    fn test_format_diagnostics() {
        let caps = SimdCapabilities::detect();
        let out = format_diagnostics(&caps);
        assert!(out.contains("SIMD Diagnostics"));
        assert!(out.contains("Dispatch plan:"));
    }

    #[test]
    fn test_neon_detection() {
        let caps = SimdCapabilities {
            sse2: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512f: false,
            avx512bw: false,
            avx512vnni: false,
            fma: false,
            neon: true,
            arch: "aarch64".to_string(),
        };
        assert_eq!(caps.best_level(), SimdLevel::Neon);
        assert_eq!(caps.vector_width_bits(), 128);
    }

    #[test]
    fn test_caps_clone() {
        let caps = SimdCapabilities::detect();
        let cloned = caps.clone();
        assert_eq!(caps, cloned);
    }

    #[test]
    fn test_speedup_scalar() {
        assert_eq!(speedup_estimate(SimdLevel::Scalar, "matmul"), Some(1.0));
    }

    #[test]
    fn test_speedup_avx2() {
        assert_eq!(speedup_estimate(SimdLevel::Avx2, "matmul"), Some(4.0));
    }

    #[test]
    fn test_dispatch_recommendation_fields() {
        let rec = DispatchRecommendation {
            operation: "test_op".to_string(),
            selected_level: SimdLevel::Avx2,
            reason: "testing".to_string(),
            estimated_speedup: Some(4.0),
        };
        assert_eq!(rec.operation, "test_op");
        assert_eq!(rec.selected_level, SimdLevel::Avx2);
    }

    #[test]
    fn test_simd_level_display_name() {
        assert_eq!(SimdLevel::Sse2.display_name(), "SSE2");
        assert_eq!(SimdLevel::Sse42.display_name(), "SSE4.2");
        assert_eq!(SimdLevel::Neon.display_name(), "NEON");
    }
}
