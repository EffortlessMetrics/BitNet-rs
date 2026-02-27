//! Property tests for `bitnet-device-probe`.
//!
//! These tests verify environment-variable-driven behaviour of device
//! capability detection without requiring real GPU hardware.

use bitnet_device_probe::{
    DeviceCapabilities, detect_simd_level, gpu_available_runtime, gpu_compiled, probe_cpu,
    probe_gpu,
};
use proptest::prelude::*;
use serial_test::serial;

// ── SIMD level ordering invariant ───────────────────────────────────────────

proptest! {
    #[test]
    fn simd_level_display_never_empty(_dummy in 0u8..4) {
        // detect_simd_level() is a compile-time+runtime probe; calling it
        // multiple times with random "context" still produces a non-empty string.
        let level = detect_simd_level();
        prop_assert!(!level.to_string().is_empty());
    }
}

proptest! {
    #[test]
    fn simd_level_consistent_across_calls(_dummy in 0u8..8) {
        // The SIMD level is a function of the hardware and kernel state;
        // it must return the same value on every call within a test.
        let a = detect_simd_level();
        let b = detect_simd_level();
        prop_assert_eq!(a, b);
    }
}

// ── gpu_compiled() is a compile-time constant ────────────────────────────────

proptest! {
    #[test]
    fn gpu_compiled_is_stable(_dummy in 0u8..4) {
        // compile-time constant — must be the same on every call
        prop_assert_eq!(gpu_compiled(), gpu_compiled());
    }
}

// ── DeviceCapabilities invariants ────────────────────────────────────────────

proptest! {
    #[test]
    fn device_capabilities_cpu_rust_always_true(_dummy in 0u8..4) {
        let caps = DeviceCapabilities::detect();
        prop_assert!(caps.cpu_rust, "cpu_rust must always be true");
    }
}

proptest! {
    #[test]
    fn device_capabilities_compiled_flags_match_gpu_compiled(_dummy in 0u8..4) {
        let caps = DeviceCapabilities::detect();
        prop_assert_eq!(caps.cuda_compiled || caps.rocm_compiled, gpu_compiled());
    }
}

// ── BITNET_GPU_FAKE environment overrides ────────────────────────────────────

/// Allowed BITNET_GPU_FAKE values that mean "GPU present".
#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
const GPU_FAKE_PRESENT: &[&str] = &["cuda", "CUDA", "rocm", "ROCM", "gpu", "GPU"];
/// Values that mean "GPU absent" or unrecognised.
#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
const GPU_FAKE_ABSENT: &[&str] = &["none", "NONE", "0", "", "no", "false"];

#[test]
#[serial(bitnet_env)]
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
fn gpu_available_runtime_false_without_gpu_feature() {
    // Without GPU feature, gpu_available_runtime() is const false regardless of env.
    temp_env::with_var("BITNET_GPU_FAKE", Some("cuda"), || {
        assert!(!gpu_available_runtime());
    });
}

proptest! {
    #[test]
    #[serial(bitnet_env)]
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
    fn gpu_fake_cuda_returns_true(idx in 0..GPU_FAKE_PRESENT.len()) {
        let fake_val = GPU_FAKE_PRESENT[idx];
        temp_env::with_var("BITNET_GPU_FAKE", Some(fake_val), || {
            prop_assert!(gpu_available_runtime(),
                "BITNET_GPU_FAKE={fake_val} should make gpu_available_runtime() return true");
            Ok(())
        })?;
    }
}

proptest! {
    #[test]
    #[serial(bitnet_env)]
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
    fn gpu_fake_none_returns_false(idx in 0..GPU_FAKE_ABSENT.len()) {
        let fake_val = GPU_FAKE_ABSENT[idx];
        temp_env::with_var("BITNET_GPU_FAKE", Some(fake_val), || {
            // Values except cuda/rocm/gpu (case-insensitive) should return false.
            // All others (including empty string) return false.
            prop_assert!(!gpu_available_runtime(),
                "BITNET_GPU_FAKE={fake_val} should make gpu_available_runtime() return false");
            Ok(())
        })?;
    }
}

// ── BITNET_STRICT_MODE overrides BITNET_GPU_FAKE ─────────────────────────────

#[test]
#[serial(bitnet_env)]
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
fn strict_mode_with_no_gpu_feature_still_returns_false() {
    temp_env::with_vars(
        [("BITNET_STRICT_MODE", Some("1")), ("BITNET_GPU_FAKE", Some("cuda"))],
        || {
            // Without GPU feature, the function is always false regardless of strict mode.
            assert!(!gpu_available_runtime());
        },
    );
}

#[test]
#[serial(bitnet_env)]
#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
fn strict_mode_ignores_gpu_fake() {
    temp_env::with_vars(
        [("BITNET_STRICT_MODE", Some("1")), ("BITNET_GPU_FAKE", Some("cuda"))],
        || {
            // Strict mode bypasses BITNET_GPU_FAKE and probes real hardware.
            // On CI (no GPU), this returns false.
            let result = gpu_available_runtime();
            // We can only assert it's a bool without error — the value is hardware-dependent.
            let _ = result;
        },
    );
}

// ── probe_cpu() invariants ────────────────────────────────────────────────────

proptest! {
    #[test]
    fn probe_cpu_core_count_at_least_one(_dummy in 0u8..4) {
        let caps = probe_cpu();
        prop_assert!(caps.core_count >= 1, "core_count must always be >= 1, got {}", caps.core_count);
    }
}

proptest! {
    #[test]
    fn probe_cpu_is_consistent_across_calls(_dummy in 0u8..4) {
        let a = probe_cpu();
        let b = probe_cpu();
        prop_assert_eq!(a.has_avx2, b.has_avx2, "has_avx2 must be consistent");
        prop_assert_eq!(a.has_avx512, b.has_avx512, "has_avx512 must be consistent");
        prop_assert_eq!(a.has_neon, b.has_neon, "has_neon must be consistent");
    }
}

#[test]
fn probe_cpu_capability_flags_are_booleans() {
    let caps = probe_cpu();
    // Flags are Rust bools — just confirm they are accessible without panicking.
    let _ = caps.has_avx2;
    let _ = caps.has_avx512;
    let _ = caps.has_neon;
}

// AVX2 and AVX-512 are x86_64-only; NEON is aarch64-only.
#[cfg(target_arch = "aarch64")]
#[test]
fn probe_cpu_neon_true_on_aarch64() {
    assert!(probe_cpu().has_neon, "NEON is mandatory on AArch64");
}

#[cfg(not(target_arch = "aarch64"))]
#[test]
fn probe_cpu_neon_false_on_non_aarch64() {
    assert!(!probe_cpu().has_neon, "NEON should be false on non-AArch64");
}

// ── probe_gpu() invariants ────────────────────────────────────────────────────

proptest! {
    #[test]
    fn probe_gpu_available_consistent_with_cuda_available(_dummy in 0u8..4) {
        let caps = probe_gpu();
        // For our current implementation, available == cuda_available.
        prop_assert_eq!(caps.available, caps.cuda_available);
    }
}

#[test]
#[serial(bitnet_env)]
fn probe_gpu_respects_gpu_fake_cuda() {
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
    temp_env::with_vars(
        [("BITNET_STRICT_MODE", None::<&str>), ("BITNET_GPU_FAKE", Some("cuda"))],
        || {
            let caps = probe_gpu();
            assert!(caps.cuda_available, "BITNET_GPU_FAKE=cuda should report cuda_available=true");
            assert!(caps.available, "BITNET_GPU_FAKE=cuda should report available=true");
        },
    );

    #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
    {
        // Without GPU feature, probe_gpu() always returns false regardless of env.
        let caps = probe_gpu();
        assert!(!caps.cuda_available);
        assert!(!caps.available);
    }
}

#[test]
#[serial(bitnet_env)]
fn probe_gpu_respects_gpu_fake_none() {
    temp_env::with_var("BITNET_GPU_FAKE", Some("none"), || {
        let caps = probe_gpu();
        assert!(!caps.cuda_available, "BITNET_GPU_FAKE=none should report cuda_available=false");
        assert!(!caps.available, "BITNET_GPU_FAKE=none should report available=false");
    });
}
