//! Snapshot tests for `bitnet-device-probe` public API surface.
//!
//! Pins invariants of device capability detection that must hold across
//! all supported build configurations.

use bitnet_device_probe::{DeviceCapabilities, SimdLevel, detect_simd_level, gpu_compiled};

#[test]
fn gpu_compiled_is_bool() {
    // gpu_compiled reflects whether any GPU backend feature is compiled.
    assert_eq!(gpu_compiled(), cfg!(any(feature = "gpu", feature = "cuda", feature = "rocm")));
}

#[test]
fn detect_simd_level_returns_valid_variant() {
    // Just confirm the function returns without panicking — the actual level
    // is hardware-dependent and cannot be snapshotted portably.
    let level = detect_simd_level();
    // Document the invariant: every detected level has a non-empty Display.
    assert!(!level.to_string().is_empty());
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
