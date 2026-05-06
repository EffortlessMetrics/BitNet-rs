//! x86 CPU feature probing used by validation lanes.

use serde::{Deserialize, Serialize};

/// Runtime-visible x86 CPU feature facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct X86CpuFeatureProbe {
    /// Whether AVX2 is available to the current process.
    pub has_avx2: bool,
    /// Whether AVX-512F is available to the current process.
    pub has_avx512: bool,
    /// Whether FMA is available to the current process.
    pub has_fma: bool,
    /// Whether SSE4.2 is available to the current process.
    pub has_sse42: bool,
}

/// Probe x86 SIMD features without panicking on non-x86 targets.
#[allow(clippy::missing_const_for_fn)]
pub fn probe_x86_cpu_features() -> X86CpuFeatureProbe {
    #[cfg(target_arch = "x86_64")]
    {
        X86CpuFeatureProbe {
            has_avx2: is_x86_feature_detected!("avx2"),
            has_avx512: is_x86_feature_detected!("avx512f"),
            has_fma: is_x86_feature_detected!("fma"),
            has_sse42: is_x86_feature_detected!("sse4.2"),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        X86CpuFeatureProbe { has_avx2: false, has_avx512: false, has_fma: false, has_sse42: false }
    }
}
