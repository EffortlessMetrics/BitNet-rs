//! Snapshot tests for `bitnet-common` public API surface.
//!
//! These tests pin the Display / Debug formats of key types so that
//! unintentional changes are caught at review time.

use bitnet_common::BitNetConfig;
use bitnet_common::kernel_registry::{KernelBackend, KernelCapabilities, SimdLevel};

// ---------------------------------------------------------------------------
// BitNetConfig
// ---------------------------------------------------------------------------

#[test]
fn bitnet_config_default_json_snapshot() {
    // Pin the serialized default config so that any schema change is visible at review time.
    let cfg = BitNetConfig::default();
    insta::assert_json_snapshot!("bitnet_config_default", cfg);
}

// ---------------------------------------------------------------------------
// SimdLevel
// ---------------------------------------------------------------------------

#[test]
fn simd_level_display_all_variants() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    let displays: Vec<String> = levels.iter().map(|l| l.to_string()).collect();
    insta::assert_debug_snapshot!("simd_level_display_variants", displays);
}

#[test]
fn simd_level_ordering_is_ascending() {
    // Snapshot the sorted order to document the contract: Scalar < Neon < SSE4.2 < AVX2 < AVX512
    let ordered = {
        let mut v = [
            SimdLevel::Avx512,
            SimdLevel::Scalar,
            SimdLevel::Neon,
            SimdLevel::Avx2,
            SimdLevel::Sse42,
        ];
        v.sort();
        v.iter().map(|l| format!("{l}")).collect::<Vec<_>>()
    };
    insta::assert_debug_snapshot!("simd_level_sorted_order", ordered);
}

// ---------------------------------------------------------------------------
// KernelBackend
// ---------------------------------------------------------------------------

#[test]
fn kernel_backend_display_all_variants() {
    let backends = [KernelBackend::CpuRust, KernelBackend::Cuda, KernelBackend::CppFfi];
    let displays: Vec<String> = backends.iter().map(|b| b.to_string()).collect();
    insta::assert_debug_snapshot!("kernel_backend_display_variants", displays);
}

// ---------------------------------------------------------------------------
// KernelCapabilities
// ---------------------------------------------------------------------------

#[test]
fn kernel_capabilities_cpu_only_snapshot() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: SimdLevel::Avx2,
    };
    insta::assert_debug_snapshot!("kernel_capabilities_cpu_avx2", caps);
}

#[test]
fn kernel_capabilities_from_compile_time_is_deterministic() {
    // Two calls must return identical results.
    let a = KernelCapabilities::from_compile_time();
    let b = KernelCapabilities::from_compile_time();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}
