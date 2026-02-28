//! Integration tests for `bitnet-device-probe` public API.
//!
//! These tests exercise the full device-probing pipeline: `probe_device`,
//! `probe_cpu`, `detect_simd_level`, and `simd_level_rank`.

use bitnet_device_probe::{
    DeviceCapabilities, SimdLevel, detect_simd_level, probe_cpu, probe_device, simd_level_rank,
};

/// `probe_device()` must return a well-formed `DeviceProbe` without panicking.
#[test]
fn probe_device_returns_valid_struct() {
    let probe = probe_device();
    assert!(probe.cpu.cores >= 1, "cpu.cores must be >= 1, got {}", probe.cpu.cores);
    assert!(probe.cpu.threads >= 1, "cpu.threads must be >= 1, got {}", probe.cpu.threads);
    // Verify cuda_available field is accessible (bool type assertion via identity).
    let _: bool = probe.cuda_available;
}

/// `SimdLevel` rank is strictly monotonic on the x86 path:
/// Scalar < Sse42 < Avx2 < Avx512.
#[test]
fn simd_level_rank_is_monotonic() {
    assert!(
        simd_level_rank(&SimdLevel::Scalar) < simd_level_rank(&SimdLevel::Sse42),
        "Scalar rank must be less than Sse42"
    );
    assert!(
        simd_level_rank(&SimdLevel::Sse42) < simd_level_rank(&SimdLevel::Avx2),
        "Sse42 rank must be less than Avx2"
    );
    assert!(
        simd_level_rank(&SimdLevel::Avx2) < simd_level_rank(&SimdLevel::Avx512),
        "Avx2 rank must be less than Avx512"
    );
}

/// The SIMD level of the probed CPU must be a recognised variant (rank < `u32::MAX`).
///
/// `u32::MAX` is the catch-all for unknown future variants; every named variant
/// must map to a concrete rank.
#[test]
fn probed_cpu_simd_level_is_valid() {
    let probe = probe_device();
    let rank = simd_level_rank(&probe.cpu.simd_level);
    assert!(
        rank < u32::MAX,
        "unexpected SimdLevel variant; rank == u32::MAX is reserved for unknown variants"
    );
}

/// The `Display` representation of the detected SIMD level must be non-empty.
///
/// This is the closest equivalent to a `cpu_name()` accessor in the public API:
/// it provides a human-readable string describing the CPU's SIMD capability.
#[test]
fn simd_level_display_is_non_empty() {
    let level = detect_simd_level();
    let display = level.to_string();
    assert!(!display.is_empty(), "SimdLevel::to_string() must produce a non-empty string");
}

/// `probe_cpu()` must return a `CpuCapabilities` struct with at least one logical
/// core and mutually-exclusive SIMD flags (AVX and NEON cannot coexist).
#[test]
fn probe_cpu_returns_consistent_capabilities() {
    let caps = probe_cpu();
    assert!(caps.core_count >= 1, "core_count must be >= 1, got {}", caps.core_count);
    // On x86_64 NEON is never set; on AArch64 AVX flags are never set.
    assert!(!(caps.has_avx2 && caps.has_neon), "avx2 and neon are mutually exclusive");
    assert!(!(caps.has_avx512 && caps.has_neon), "avx512 and neon are mutually exclusive");
}

/// `DeviceCapabilities::detect()` is idempotent: two consecutive calls return
/// identical values (assuming no hardware changes between calls).
#[test]
fn device_capabilities_detect_is_idempotent() {
    let first = DeviceCapabilities::detect();
    let second = DeviceCapabilities::detect();
    assert_eq!(first, second, "DeviceCapabilities::detect() must be idempotent");
}

/// `probe_device()` and `DeviceCapabilities::detect()` must agree on the SIMD level.
#[test]
fn probe_device_simd_consistent_with_device_capabilities() {
    let probe = probe_device();
    let caps = DeviceCapabilities::detect();
    assert_eq!(
        probe.cpu.simd_level, caps.simd_level,
        "probe_device and DeviceCapabilities::detect must agree on SIMD level"
    );
}
