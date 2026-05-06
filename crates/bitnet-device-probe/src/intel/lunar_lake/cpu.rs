//! Lunar Lake CPU visibility facts.
#![allow(clippy::doc_markdown)]

use serde::{Deserialize, Serialize};

use crate::cpu::{probe_cpu_topology, probe_x86_cpu_features};

/// CPU facts for the Core Ultra 7 258V validation lane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct Lnl258vCpuProbe {
    /// CPU brand string when the OS exposes one.
    pub brand: Option<String>,
    /// Best-effort physical core count.
    pub cores: usize,
    /// Logical thread count visible to the process.
    pub threads: usize,
    /// Expected 258V P-core count when the machine identity matches Lunar Lake.
    pub p_core_count: Option<usize>,
    /// Expected 258V low-power E-core count when the machine identity matches Lunar Lake.
    pub lp_e_core_count: Option<usize>,
    /// Whether AVX2 is available at runtime.
    pub has_avx2: bool,
    /// Whether AVX-512F is available at runtime.
    pub has_avx512: bool,
    /// Whether FMA is available at runtime.
    pub has_fma: bool,
    /// Whether SSE4.2 is available at runtime.
    pub has_sse42: bool,
    /// Human-readable scheduler or topology note.
    pub scheduler_hint: Option<String>,
}

/// Probe Lunar Lake CPU facts without making a BitNet inference claim.
pub fn probe_lnl258v_cpu() -> Lnl258vCpuProbe {
    let topology = probe_cpu_topology();
    let features = probe_x86_cpu_features();
    let is_258v = topology
        .brand
        .as_deref()
        .is_some_and(|brand| brand.contains("258V") || brand.contains("Core Ultra 7"));

    Lnl258vCpuProbe {
        brand: topology.brand,
        cores: topology.cores,
        threads: topology.threads,
        p_core_count: is_258v.then_some(4),
        lp_e_core_count: is_258v.then_some(4),
        has_avx2: features.has_avx2,
        has_avx512: features.has_avx512,
        has_fma: features.has_fma,
        has_sse42: features.has_sse42,
        scheduler_hint: is_258v.then_some(
            "Lunar Lake validation should record P-core / low-power E-core and power context"
                .to_owned(),
        ),
    }
}
