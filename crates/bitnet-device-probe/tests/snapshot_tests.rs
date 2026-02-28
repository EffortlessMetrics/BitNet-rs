//! Snapshot tests for `bitnet-device-probe` public API surface.
//!
//! Pins invariants of device capability detection that must hold across
//! all supported build configurations.

use bitnet_device_probe::{DeviceCapabilities, SimdLevel, detect_simd_level, gpu_compiled};

#[test]
fn gpu_compiled_is_bool() {
    assert_eq!(gpu_compiled(), cfg!(any(feature = "gpu", feature = "cuda", feature = "rocm")));
    // gpu_compiled reflects whether any GPU backend feature is compiled.
    assert_eq!(
        gpu_compiled(),
        cfg!(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi")),
    );
}

#[test]
fn detect_simd_level_returns_valid_variant() {
    let level = detect_simd_level();
    assert!(!level.to_string().is_empty());
}

#[test]
fn device_capabilities_detect_cpu_rust_always_true() {
    let caps = DeviceCapabilities::detect();
    assert!(caps.cpu_rust, "cpu_rust should always be true");
    insta::assert_snapshot!("device_capabilities_cpu_rust", format!("cpu_rust={}", caps.cpu_rust));
}

#[test]
fn simd_level_all_variants_debug() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    let debug: Vec<String> = levels.iter().map(|l| format!("{l:?}")).collect();
    insta::assert_debug_snapshot!("simd_level_debug_variants", debug);
}

// -- Wave 3: device capability invariants ------------------------------------

#[test]
fn device_capabilities_gpu_compiled_consistent() {
    let caps = DeviceCapabilities::detect();
    let gpu_feat = cfg!(any(feature = "gpu", feature = "cuda"));
    assert_eq!(caps.cuda_compiled, gpu_feat, "cuda_compiled should match gpu/cuda feature gate");
    insta::assert_snapshot!(
        "device_capabilities_gpu_consistent",
        format!("cuda_compiled={} gpu_feature={}", caps.cuda_compiled, gpu_feat)
    );
}

#[test]
fn device_capabilities_npu_compiled_matches_feature() {
    let caps = DeviceCapabilities::detect();
    let npu_feat = cfg!(feature = "npu");
    assert_eq!(caps.npu_compiled, npu_feat);
    insta::assert_snapshot!(
        "device_capabilities_npu",
        format!("npu_compiled={} npu_feature={}", caps.npu_compiled, npu_feat)
    );
}

#[test]
fn device_probe_full_summary() {
    let caps = DeviceCapabilities::detect();
    // SIMD level is machine-dependent (Avx512 vs Avx2 vs Scalar), so redact it
    // to keep the snapshot stable across CI runners and developer machines.
    let summary = format!(
        "cpu_rust={} cuda_compiled={} rocm_compiled={} npu_compiled={} simd={:?}",
        caps.cpu_rust, caps.cuda_compiled, caps.rocm_compiled, caps.npu_compiled, caps.simd_level
    );
    insta::with_settings!({filters => vec![
        (r"simd=\w+", "simd=[SIMD]"),
    ]}, {
        insta::assert_snapshot!("device_probe_summary", summary);
    });
}
