//! Tests for the `CpuProbe`, `DeviceProbe`, and `probe_device()` API.

use bitnet_device_probe::{SimdLevel, detect_simd_level, probe_device, simd_level_rank};

// ── probe_device_never_panics ─────────────────────────────────────────────────

/// Calling `probe_device()` must never panic on any supported platform.
#[test]
fn probe_device_never_panics() {
    let _ = probe_device();
}

// ── simd_level_ordering_is_consistent ────────────────────────────────────────

/// A platform that supports a higher SIMD level must also rank higher via
/// `simd_level_rank`. On x86_64 AVX512 ≥ AVX2 ≥ SSE4.2 ≥ Scalar.
/// On AArch64 NEON is the only level.
#[test]
fn simd_level_ordering_is_consistent() {
    // Static ordering: Scalar < Sse42 < Avx2 < Avx512
    assert!(simd_level_rank(&SimdLevel::Scalar) < simd_level_rank(&SimdLevel::Sse42));
    assert!(simd_level_rank(&SimdLevel::Sse42) < simd_level_rank(&SimdLevel::Avx2));
    assert!(simd_level_rank(&SimdLevel::Avx2) < simd_level_rank(&SimdLevel::Avx512));

    // The detected level on this machine must have a valid (non-panicking) rank.
    let detected = detect_simd_level();
    let _ = simd_level_rank(&detected);
}

// ── cpu_probe_cores_positive ──────────────────────────────────────────────────

/// `probe_device().cpu.cores` must always be ≥ 1.
#[test]
fn cpu_probe_cores_positive() {
    let result = probe_device();
    assert!(result.cpu.cores >= 1, "cores must be >= 1, got {}", result.cpu.cores);
    assert!(result.cpu.threads >= 1, "threads must be >= 1, got {}", result.cpu.threads);
}

// ── cuda_probe_is_deterministic ───────────────────────────────────────────────

/// Two consecutive calls to `probe_device()` must return the same
/// `cuda_available` value (assuming no hardware changes between calls).
#[test]
fn cuda_probe_is_deterministic() {
    let first = probe_device();
    let second = probe_device();
    assert_eq!(
        first.cuda_available, second.cuda_available,
        "cuda_available must be deterministic across consecutive calls"
    );
}

// ── simd_level_from_str_roundtrip ─────────────────────────────────────────────

/// `SimdLevel`'s `Display` impl must produce a non-empty string for every
/// variant, and every variant must produce a distinct string.
#[test]
fn simd_level_display_roundtrip() {
    let all_levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];

    let strings: Vec<String> = all_levels.iter().map(|l| l.to_string()).collect();

    // Every string must be non-empty.
    for s in &strings {
        assert!(!s.is_empty(), "SimdLevel Display must produce a non-empty string");
    }

    // Every string must be unique (no two levels share the same display text).
    let mut seen = std::collections::HashSet::new();
    for s in &strings {
        assert!(seen.insert(s.as_str()), "Duplicate SimdLevel display string: {s}");
    }
}
