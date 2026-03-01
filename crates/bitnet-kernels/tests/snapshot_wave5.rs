//! Wave 5 snapshot tests for `bitnet-kernels` public API surface.
//!
//! Pins the kernel provider listing, SIMD detection, and capability summary
//! formats so that unintentional changes are caught at review time.

use bitnet_kernels::KernelManager;
use bitnet_kernels::device_features;

// ---------------------------------------------------------------------------
// Kernel provider listing
// ---------------------------------------------------------------------------

#[test]
fn kernel_provider_list_non_empty() {
    let mgr = KernelManager::new();
    let providers = mgr.list_available_providers();
    insta::assert_snapshot!(format!(
        "provider_count={} providers={:?}",
        providers.len(),
        providers
    ));
}

#[test]
fn kernel_provider_list_always_has_cpu_fallback() {
    let mgr = KernelManager::new();
    let providers = mgr.list_available_providers();
    let has_cpu = providers
        .iter()
        .any(|p| p.contains("cpu") || p.contains("fallback") || p.contains("Fallback"));
    insta::assert_snapshot!(format!("has_cpu_fallback={has_cpu}"));
}

// ---------------------------------------------------------------------------
// SIMD level detection
// ---------------------------------------------------------------------------

#[test]
fn detect_simd_level_is_stable() {
    let level_a = device_features::detect_simd_level();
    let level_b = device_features::detect_simd_level();
    // Two consecutive calls must return the same level
    assert_eq!(level_a, level_b);
    insta::with_settings!({filters => vec![(r"(?:avx512|avx2|sse4\.2|neon|scalar)", "[SIMD]")]}, {
        insta::assert_snapshot!(format!("simd_level={level_a}"));
    });
}

// ---------------------------------------------------------------------------
// Device capability summary (with SIMD + CUDA version filter)
// ---------------------------------------------------------------------------

#[test]
fn device_capability_summary_format() {
    let summary = device_features::device_capability_summary();
    insta::with_settings!({filters => vec![
        (r"(?:avx512|avx2|sse4\.2|neon|scalar)", "[SIMD]"),
        (r"CUDA \d+\.\d+", "CUDA [VERSION]"),
    ]}, {
        insta::assert_snapshot!(summary);
    });
}

// ---------------------------------------------------------------------------
// KernelCapabilities via current_kernel_capabilities
// ---------------------------------------------------------------------------

#[test]
fn current_kernel_capabilities_summary() {
    let caps = device_features::current_kernel_capabilities();
    insta::with_settings!({filters => vec![(r"simd=\w+", "simd=[SIMD]")]}, {
        insta::assert_snapshot!(caps.summary());
    });
}

#[test]
fn current_kernel_capabilities_cpu_is_compiled() {
    let caps = device_features::current_kernel_capabilities();
    // When built with --features cpu, cpu_rust must be true
    insta::assert_snapshot!(format!("cpu_rust={}", caps.cpu_rust));
}
