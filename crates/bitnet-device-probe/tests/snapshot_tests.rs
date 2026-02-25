//! Snapshot tests for `bitnet-device-probe` public API surface.
//!
//! Pins invariants of device capability detection that must hold across
//! all supported build configurations.

use bitnet_device_probe::{DeviceCapabilities, SimdLevel, detect_simd_level, gpu_compiled};

#[test]
fn gpu_compiled_is_bool() {
    // Snapshot the bool so that accidental feature-flag changes are visible.
    insta::assert_snapshot!("gpu_compiled_value", gpu_compiled().to_string());
}

#[test]
fn detect_simd_level_returns_valid_variant() {
    let level = detect_simd_level();
    // Any valid SimdLevel is acceptable; snapshot the string form.
    insta::assert_snapshot!(
        "detected_simd_level",
        format!("simd={level} (valid SimdLevel variant)")
    );
}

#[test]
fn device_capabilities_detect_cpu_rust_always_true() {
    let caps = DeviceCapabilities::detect();
    // cpu_rust must always be true — we always have the Rust kernel path.
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
