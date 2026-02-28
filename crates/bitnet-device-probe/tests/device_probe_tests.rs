//! Comprehensive tests for the `bitnet-device-probe` public API.
//!
//! Covers `SimdLevel` ordering and rank, `CpuCapabilities`, `GpuCapabilities`,
//! `DeviceCapabilities`, `CpuProbe`, `DeviceProbe`, and property-based tests
//! for determinism invariants using `proptest`.

use bitnet_device_probe::{
    DeviceCapabilities, SimdLevel, detect_simd_level, gpu_compiled, npu_compiled, probe_cpu,
    probe_device, probe_gpu, probe_npu, simd_level_rank,
};
use proptest::prelude::*;

// ── probe_device_never_panics ─────────────────────────────────────────────────

/// Calling `probe_device()` must never panic on any supported platform.
#[test]
fn probe_device_never_panics() {
    let _ = probe_device();
}

// ── simd_level_ordering_is_consistent ────────────────────────────────────────

/// A platform that supports a higher SIMD level must also rank higher via
/// `simd_level_rank`. On `x86_64` AVX512 ≥ AVX2 ≥ SSE4.2 ≥ Scalar.
/// On `AArch64` NEON is the only level.
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

// ── simd_level_display_roundtrip ─────────────────────────────────────────────

/// `SimdLevel`'s `Display` impl must produce a non-empty string for every
/// variant, and every variant must produce a distinct string.
#[test]
fn simd_level_display_roundtrip() {
    let all_levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];

    let strings: Vec<String> = all_levels.iter().map(std::string::ToString::to_string).collect();

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

// ── simd_level_rank explicit values ──────────────────────────────────────────

/// `simd_level_rank` must return the exact documented numeric value for Scalar.
#[test]
fn simd_level_rank_scalar_is_zero() {
    assert_eq!(simd_level_rank(&SimdLevel::Scalar), 0, "Scalar rank must be 0");
}

/// `simd_level_rank` must return the exact documented numeric value for Sse42.
#[test]
fn simd_level_rank_sse42_is_one() {
    assert_eq!(simd_level_rank(&SimdLevel::Sse42), 1, "Sse42 rank must be 1");
}

/// `simd_level_rank` must return the exact documented numeric value for Avx2.
#[test]
fn simd_level_rank_avx2_is_two() {
    assert_eq!(simd_level_rank(&SimdLevel::Avx2), 2, "Avx2 rank must be 2");
}

/// `simd_level_rank` must return the exact documented numeric value for Avx512.
#[test]
fn simd_level_rank_avx512_is_three() {
    assert_eq!(simd_level_rank(&SimdLevel::Avx512), 3, "Avx512 rank must be 3");
}

/// `simd_level_rank` must return the exact documented numeric value for Neon.
#[test]
fn simd_level_rank_neon_is_four() {
    assert_eq!(simd_level_rank(&SimdLevel::Neon), 4, "Neon rank must be 4");
}

/// Every named `SimdLevel` variant must map to a concrete rank (< `u32::MAX`).
/// `u32::MAX` is reserved for unknown future variants.
#[test]
fn all_simd_levels_have_valid_rank() {
    let all =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    for level in &all {
        let rank = simd_level_rank(level);
        assert!(rank < u32::MAX, "rank must not be u32::MAX for known variant {level:?}");
    }
}

// ── SimdLevel type properties ─────────────────────────────────────────────────

/// `SimdLevel` is `Copy`; a copy must compare equal to the original via `PartialEq`.
#[test]
fn simd_level_copy_equals_original() {
    let level = detect_simd_level();
    let copied = level;
    assert_eq!(level, copied, "Copy of SimdLevel must equal the original");
}

/// The derived `Ord` chain on x86-style levels must hold:
/// `Scalar < Sse42 < Avx2 < Avx512`.
#[test]
fn simd_level_ord_chain_x86_style() {
    assert!(SimdLevel::Scalar < SimdLevel::Sse42);
    assert!(SimdLevel::Sse42 < SimdLevel::Avx2);
    assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
}

/// The derived `Ord` chain places `Neon` between `Scalar` and `Sse42`.
#[test]
fn simd_level_ord_neon_placement() {
    assert!(SimdLevel::Scalar < SimdLevel::Neon);
    assert!(SimdLevel::Neon < SimdLevel::Sse42);
}

/// Every `SimdLevel` variant must produce a non-empty `Debug` representation.
#[test]
fn simd_level_debug_non_empty_for_all_variants() {
    let all =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    for level in &all {
        assert!(!format!("{level:?}").is_empty(), "{level:?} Debug repr must be non-empty");
    }
}

/// `SimdLevel::Hash`: two equal values must produce the same hash.
#[test]
fn simd_level_hash_consistent_with_eq() {
    use std::collections::HashSet;
    let a = SimdLevel::Avx2;
    let b = SimdLevel::Avx2;
    assert_eq!(a, b);
    // If they're equal they must be in the same HashSet bucket.
    let mut set = HashSet::new();
    set.insert(a);
    assert!(set.contains(&b), "equal SimdLevel values must produce the same hash");
}

// ── probe_cpu tests ───────────────────────────────────────────────────────────

/// `probe_cpu()` must return without panicking.
#[test]
fn probe_cpu_never_panics() {
    let _ = probe_cpu();
}

/// `probe_cpu().core_count` is always ≥ 1.
#[test]
fn probe_cpu_core_count_at_least_one() {
    let caps = probe_cpu();
    assert!(caps.core_count >= 1, "core_count must be >= 1, got {}", caps.core_count);
}

/// AVX2 and NEON are mutually exclusive across CPU architectures.
#[test]
fn probe_cpu_avx2_and_neon_mutually_exclusive() {
    let caps = probe_cpu();
    assert!(!(caps.has_avx2 && caps.has_neon), "avx2 and neon cannot both be true");
}

/// AVX-512 and NEON are mutually exclusive across CPU architectures.
#[test]
fn probe_cpu_avx512_and_neon_mutually_exclusive() {
    let caps = probe_cpu();
    assert!(!(caps.has_avx512 && caps.has_neon), "avx512 and neon cannot both be true");
}

/// `CpuCapabilities` is `Clone + PartialEq`; a clone must compare equal.
#[test]
fn cpu_capabilities_clone_equals_original() {
    let caps = probe_cpu();
    assert_eq!(caps.clone(), caps);
}

/// Two consecutive calls to `probe_cpu()` must return the same SIMD flags.
#[test]
fn probe_cpu_simd_flags_are_deterministic() {
    let a = probe_cpu();
    let b = probe_cpu();
    assert_eq!(a.has_avx2, b.has_avx2, "has_avx2 must be deterministic");
    assert_eq!(a.has_avx512, b.has_avx512, "has_avx512 must be deterministic");
    assert_eq!(a.has_neon, b.has_neon, "has_neon must be deterministic");
}

/// On `x86_64`, NEON must always be `false`.
#[cfg(target_arch = "x86_64")]
#[test]
fn probe_cpu_neon_is_false_on_x86_64() {
    assert!(!probe_cpu().has_neon, "NEON must be false on x86_64");
}

/// On `aarch64`, NEON is mandatory and must always be `true`.
#[cfg(target_arch = "aarch64")]
#[test]
fn probe_cpu_neon_is_true_on_aarch64() {
    assert!(probe_cpu().has_neon, "NEON must be true on AArch64");
}

/// On `x86_64`, `detect_simd_level()` must never return `Neon`.
#[cfg(target_arch = "x86_64")]
#[test]
fn detect_simd_level_not_neon_on_x86_64() {
    assert_ne!(detect_simd_level(), SimdLevel::Neon, "SIMD level must not be Neon on x86_64");
}

/// On `aarch64`, `detect_simd_level()` must return `Neon`.
#[cfg(target_arch = "aarch64")]
#[test]
fn detect_simd_level_is_neon_on_aarch64() {
    assert_eq!(detect_simd_level(), SimdLevel::Neon, "SIMD level must be Neon on AArch64");
}

/// `probe_npu()` must never panic and aligns with `npu_compiled()` when disabled.
#[test]
fn probe_npu_never_panics() {
    let caps = probe_npu();
    if !npu_compiled() {
        assert!(!caps.available);
    }
}

// ── probe_gpu tests ───────────────────────────────────────────────────────────

/// `probe_gpu()` must never panic, regardless of whether GPU hardware is present.
#[test]
fn probe_gpu_never_panics() {
    let _ = probe_gpu();
}

/// `GpuCapabilities::available` must be the OR of CUDA and `ROCm` availability.
#[test]
fn probe_gpu_available_consistent_with_backend_flags() {
    let caps = probe_gpu();
    assert_eq!(
        caps.available,
        caps.cuda_available || caps.rocm_available,
        "available must equal cuda_available || rocm_available"
    );
}

/// `GpuCapabilities` is `Clone + PartialEq`; a clone must compare equal.
#[test]
fn gpu_capabilities_clone_equals_original() {
    let caps = probe_gpu();
    assert_eq!(caps.clone(), caps);
}

/// Without GPU feature, `probe_gpu()` must return all-`false` fields.
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
#[test]
fn probe_gpu_returns_false_without_gpu_feature() {
    let caps = probe_gpu();
    assert!(!caps.available, "available must be false without GPU feature");
    assert!(!caps.cuda_available, "cuda_available must be false without GPU feature");
    assert!(!caps.rocm_available, "rocm_available must be false without GPU feature");
}

/// Without GPU feature, `gpu_compiled()` must return `false`.
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
#[test]
fn gpu_compiled_is_false_with_cpu_feature_only() {
    assert!(!gpu_compiled(), "gpu_compiled() must be false when built with --features cpu only");
}

// ── DeviceCapabilities tests ──────────────────────────────────────────────────

/// `DeviceCapabilities::detect()` must never panic.
#[test]
fn device_capabilities_detect_never_panics() {
    let _ = DeviceCapabilities::detect();
}

/// `cpu_rust` is always `true`; the CPU-Rust kernel path is always available.
#[test]
fn device_capabilities_cpu_rust_always_true() {
    assert!(DeviceCapabilities::detect().cpu_rust, "cpu_rust must always be true");
}

/// Combined compiled GPU flags must agree with `gpu_compiled()`.
#[test]
fn device_capabilities_compiled_flags_match_gpu_compiled() {
    let caps = DeviceCapabilities::detect();
    assert_eq!(caps.cuda_compiled || caps.rocm_compiled, gpu_compiled());
}

/// `DeviceCapabilities` is `Clone + PartialEq`; a clone must compare equal.
#[test]
fn device_capabilities_clone_equals_original() {
    let caps = DeviceCapabilities::detect();
    assert_eq!(caps.clone(), caps);
}

/// `DeviceCapabilities::simd_level` must agree with `detect_simd_level()`.
#[test]
fn device_capabilities_simd_level_matches_detect_simd_level() {
    let caps = DeviceCapabilities::detect();
    assert_eq!(caps.simd_level, detect_simd_level());
}

// ── DeviceProbe / CpuProbe tests ──────────────────────────────────────────────

/// `probe_device().cpu.threads` must always be ≥ 1.
#[test]
fn device_probe_threads_at_least_one() {
    let probe = probe_device();
    assert!(probe.cpu.threads >= 1, "threads must be >= 1, got {}", probe.cpu.threads);
}

/// `DeviceProbe` is `Clone + PartialEq`; a clone must compare equal.
#[test]
fn device_probe_clone_equals_original() {
    let probe = probe_device();
    assert_eq!(probe.clone(), probe);
}

/// `probe_device().cpu.simd_level` must agree with `detect_simd_level()`.
#[test]
fn device_probe_simd_level_matches_detect_simd_level() {
    let probe = probe_device();
    assert_eq!(probe.cpu.simd_level, detect_simd_level());
}

/// The `simd_level` reported by `probe_device` must agree with `DeviceCapabilities`.
#[test]
fn probe_device_and_device_capabilities_agree_on_simd() {
    let probe = probe_device();
    let caps = DeviceCapabilities::detect();
    assert_eq!(
        probe.cpu.simd_level, caps.simd_level,
        "probe_device and DeviceCapabilities::detect must agree on simd_level"
    );
}

// ── proptest: determinism properties ─────────────────────────────────────────

proptest! {
    /// `simd_level_rank` is a pure function: repeated calls with the same
    /// input must always return the same output.
    #[test]
    fn simd_rank_is_deterministic(_n in 0u8..=10) {
        let level = detect_simd_level();
        prop_assert_eq!(simd_level_rank(&level), simd_level_rank(&level));
    }

    /// `probe_device()` returns the same SIMD level and `cuda_available` flag on
    /// every call within a single test run (hardware does not change mid-test).
    #[test]
    fn probe_device_is_deterministic(_n in 0u8..=10) {
        let a = probe_device();
        let b = probe_device();
        prop_assert_eq!(a.cpu.simd_level, b.cpu.simd_level);
        prop_assert_eq!(a.cuda_available, b.cuda_available);
    }

    /// `DeviceCapabilities::detect()` is stable: two consecutive calls return
    /// identical snapshots (no hardware changes within a test run).
    #[test]
    fn device_capabilities_is_stable(_n in 0u8..=10) {
        let a = DeviceCapabilities::detect();
        let b = DeviceCapabilities::detect();
        prop_assert_eq!(a, b);
    }

    /// `gpu_compiled()` is a compile-time constant; it must be the same on
    /// every call regardless of any runtime state.
    #[test]
    fn gpu_compiled_stable_across_calls(_n in 0u8..=10) {
        prop_assert_eq!(gpu_compiled(), gpu_compiled());
    }

    /// `detect_simd_level()` must always return a level whose rank is < `u32::MAX`
    /// (i.e., the detected level is always one of the known named variants).
    #[test]
    fn detected_simd_level_has_bounded_rank(_n in 0u8..=10) {
        let level = detect_simd_level();
        prop_assert!(
            simd_level_rank(&level) < u32::MAX,
            "detected level {level:?} must have a bounded rank"
        );
    }

    /// `probe_cpu().core_count` must always be ≥ 1, across any number of calls.
    #[test]
    fn probe_cpu_core_count_always_positive(_n in 0u8..=10) {
        let caps = probe_cpu();
        prop_assert!(caps.core_count >= 1, "core_count must always be >= 1, got {}", caps.core_count);
    }

    /// `probe_gpu().available` must equal CUDA/ROCm ORed availability.
    #[test]
    fn probe_gpu_fields_always_consistent(_n in 0u8..=10) {
        let caps = probe_gpu();
        prop_assert_eq!(
            caps.available, caps.cuda_available || caps.rocm_available,
            "available must equal cuda_available || rocm_available"
        );
    }
}

// ── Cross-consistency tests ───────────────────────────────────────────────────

/// `probe_cpu().has_avx2` must agree with `detect_simd_level() >= Avx2`.
#[cfg(target_arch = "x86_64")]
#[test]
fn probe_cpu_has_avx2_consistent_with_simd_level() {
    let caps = probe_cpu();
    let level = detect_simd_level();
    // If has_avx2 is true, simd_level must be at least Avx2.
    if caps.has_avx2 {
        assert!(
            simd_level_rank(&level) >= simd_level_rank(&SimdLevel::Avx2),
            "has_avx2=true requires simd_level rank >= Avx2 rank"
        );
    }
}

/// `probe_cpu().has_avx512` must agree with `detect_simd_level() >= Avx512`.
#[cfg(target_arch = "x86_64")]
#[test]
fn probe_cpu_has_avx512_consistent_with_simd_level() {
    let caps = probe_cpu();
    let level = detect_simd_level();
    if caps.has_avx512 {
        assert!(
            simd_level_rank(&level) >= simd_level_rank(&SimdLevel::Avx512),
            "has_avx512=true requires simd_level rank >= Avx512 rank"
        );
    }
}

/// `probe_device().cpu.cores` must equal `probe_device().cpu.threads`
/// (the implementation uses `available_parallelism` for both).
#[test]
fn probe_device_cores_equal_threads() {
    let probe = probe_device();
    assert_eq!(
        probe.cpu.cores, probe.cpu.threads,
        "cores and threads must be equal (both derived from available_parallelism)"
    );
}

/// `probe_cpu().core_count` must match `probe_device().cpu.threads`.
#[test]
fn probe_cpu_core_count_matches_probe_device_threads() {
    let cpu = probe_cpu();
    let device = probe_device();
    assert_eq!(
        cpu.core_count, device.cpu.threads,
        "probe_cpu().core_count must match probe_device().cpu.threads"
    );
}

/// `DeviceCapabilities.cuda_runtime` must match `probe_device().cuda_available`.
#[test]
fn device_capabilities_cuda_runtime_matches_probe_device() {
    let caps = DeviceCapabilities::detect();
    let probe = probe_device();
    assert_eq!(
        caps.cuda_runtime || caps.rocm_runtime,
        probe.cuda_available,
        "DeviceCapabilities combined GPU runtime must match probe_device().cuda_available"
    );
}

/// `DeviceCapabilities.simd_level` rank must be >= Scalar rank (always some SIMD level).
#[test]
fn device_capabilities_simd_rank_at_least_scalar() {
    let caps = DeviceCapabilities::detect();
    assert!(
        simd_level_rank(&caps.simd_level) >= simd_level_rank(&SimdLevel::Scalar),
        "simd_level rank must be >= Scalar rank"
    );
}

/// `GpuCapabilities` has a non-empty Debug representation.
#[test]
fn gpu_capabilities_debug_non_empty() {
    let caps = probe_gpu();
    assert!(!format!("{caps:?}").is_empty(), "GpuCapabilities Debug must be non-empty");
}

/// `CpuCapabilities` has a non-empty Debug representation.
#[test]
fn cpu_capabilities_debug_non_empty() {
    let caps = probe_cpu();
    assert!(!format!("{caps:?}").is_empty(), "CpuCapabilities Debug must be non-empty");
}

/// `DeviceProbe` and `CpuProbe` both have non-empty Debug representations.
#[test]
fn device_probe_and_cpu_probe_debug_non_empty() {
    let probe = probe_device();
    assert!(!format!("{probe:?}").is_empty(), "DeviceProbe Debug must be non-empty");
    assert!(!format!("{:?}", probe.cpu).is_empty(), "CpuProbe Debug must be non-empty");
}

/// `simd_level_rank` ordering: Avx512 must strictly outrank Avx2.
#[test]
fn simd_level_rank_avx512_greater_than_avx2() {
    assert!(
        simd_level_rank(&SimdLevel::Avx512) > simd_level_rank(&SimdLevel::Avx2),
        "Avx512 rank must exceed Avx2 rank"
    );
}

/// `simd_level_rank` ordering: Avx2 must strictly outrank Sse42.
#[test]
fn simd_level_rank_avx2_greater_than_sse42() {
    assert!(
        simd_level_rank(&SimdLevel::Avx2) > simd_level_rank(&SimdLevel::Sse42),
        "Avx2 rank must exceed Sse42 rank"
    );
}

/// `simd_level_rank` ordering: Sse42 must strictly outrank Scalar.
#[test]
fn simd_level_rank_sse42_greater_than_scalar() {
    assert!(
        simd_level_rank(&SimdLevel::Sse42) > simd_level_rank(&SimdLevel::Scalar),
        "Sse42 rank must exceed Scalar rank"
    );
}

/// All five named SIMD level ranks must be pairwise distinct.
#[test]
fn simd_level_ranks_are_pairwise_distinct() {
    let all =
        [SimdLevel::Scalar, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512, SimdLevel::Neon];
    let ranks: Vec<u32> = all.iter().map(simd_level_rank).collect();
    let unique: std::collections::HashSet<u32> = ranks.iter().copied().collect();
    assert_eq!(
        unique.len(),
        all.len(),
        "all SimdLevel ranks must be pairwise distinct, got: {ranks:?}"
    );
}

/// `probe_device()` returns a `DeviceProbe` that can be round-tripped through Debug.
#[test]
fn probe_device_debug_roundtrip_contains_simd() {
    let probe = probe_device();
    let debug_str = format!("{probe:?}");
    // The Debug output must mention "simd_level" (derived Debug includes field names).
    assert!(
        debug_str.contains("simd_level"),
        "DeviceProbe Debug must contain 'simd_level', got: {debug_str}"
    );
}

/// `DeviceCapabilities::detect()` round-trips through Debug and mentions `cpu_rust`.
#[test]
fn device_capabilities_debug_mentions_cpu_rust() {
    let caps = DeviceCapabilities::detect();
    let debug_str = format!("{caps:?}");
    assert!(
        debug_str.contains("cpu_rust"),
        "DeviceCapabilities Debug must mention 'cpu_rust', got: {debug_str}"
    );
}
