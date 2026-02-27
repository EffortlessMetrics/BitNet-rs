//! Property and unit tests for recently-added backend capabilities.
//!
//! Covers:
//! - ROCm availability field in [`GpuCapabilities`] and [`DeviceProbe`]
//! - Vulkan compile-time and runtime probes (`vulkan_compiled`, `vulkan_available_runtime`)
//! - oneAPI feature flag (compile-time gate; no runtime function yet)
//! - GPU feature flag interactions (`gpu` implies `cuda` OR `rocm`)
//! - [`DeviceProbe`] struct clone/roundtrip validation
//! - Cross-consistency between [`probe_gpu`] and [`probe_device`]
//!
//! Tests run under `--no-default-features --features cpu` unless gated with
//! `#[cfg(any(feature = "gpu", ...))]`.

use bitnet_device_probe::{
    DeviceCapabilities, gpu_available_runtime, gpu_compiled, probe_device, probe_gpu,
    vulkan_available_runtime, vulkan_compiled,
};
use proptest::prelude::*;
use serial_test::serial;

// ─────────────────────────────────────────────────────────────────────────────
// ROCm: GpuCapabilities.rocm_available
// ─────────────────────────────────────────────────────────────────────────────

/// `GpuCapabilities.rocm_available` must be a valid `bool` (field access never panics).
#[test]
fn probe_gpu_rocm_available_field_accessible() {
    let caps = probe_gpu();
    let _: bool = caps.rocm_available;
}

/// Without GPU feature, `rocm_available` must always be `false`.
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
#[test]
fn probe_gpu_rocm_available_false_without_gpu_feature() {
    assert!(!probe_gpu().rocm_available, "rocm_available must be false without GPU feature");
}

/// `GpuCapabilities.available` must equal `cuda_available || rocm_available`.
///
/// This invariant must hold regardless of the GPU feature set.
proptest! {
    #[test]
    fn probe_gpu_available_equals_cuda_or_rocm(_n in 0u8..=8) {
        let caps = probe_gpu();
        prop_assert_eq!(
            caps.available,
            caps.cuda_available || caps.rocm_available,
            "available must equal cuda_available || rocm_available, got: available={}, cuda={}, rocm={}",
            caps.available, caps.cuda_available, caps.rocm_available
        );
    }
}

/// `GpuCapabilities` clone must equal the original (derived `Clone + PartialEq`).
proptest! {
    #[test]
    fn probe_gpu_capabilities_clone_roundtrip(_n in 0u8..=8) {
        let caps = probe_gpu();
        prop_assert_eq!(caps.clone(), caps);
    }
}

/// `rocm_available` is deterministic across consecutive calls.
proptest! {
    #[test]
    fn probe_gpu_rocm_available_is_deterministic(_n in 0u8..=8) {
        let a = probe_gpu();
        let b = probe_gpu();
        prop_assert_eq!(
            a.rocm_available, b.rocm_available,
            "rocm_available must be deterministic"
        );
    }
}

/// BITNET_GPU_FAKE=rocm must set `rocm_available=true` (with ROCm/GPU feature).
#[cfg(any(feature = "gpu", feature = "rocm"))]
#[test]
#[serial(bitnet_env)]
fn rocm_fake_env_sets_rocm_available_true() {
    temp_env::with_vars(
        [("BITNET_STRICT_MODE", None::<&str>), ("BITNET_GPU_FAKE", Some("rocm"))],
        || {
            let caps = probe_gpu();
            assert!(caps.rocm_available, "BITNET_GPU_FAKE=rocm must set rocm_available=true");
            assert!(caps.available, "BITNET_GPU_FAKE=rocm must set available=true");
        },
    );
}

/// BITNET_GPU_FAKE=gpu must set both `cuda_available` and `rocm_available` (with GPU feature).
#[cfg(any(feature = "gpu", feature = "rocm", feature = "cuda"))]
#[test]
#[serial(bitnet_env)]
fn gpu_fake_gpu_sets_all_available() {
    temp_env::with_vars(
        [("BITNET_STRICT_MODE", None::<&str>), ("BITNET_GPU_FAKE", Some("gpu"))],
        || {
            let caps = probe_gpu();
            assert!(caps.available, "BITNET_GPU_FAKE=gpu must set available=true");
        },
    );
}

/// BITNET_GPU_FAKE=none must set all availability flags to `false` (with GPU feature).
#[cfg(any(feature = "gpu", feature = "rocm", feature = "cuda"))]
#[test]
#[serial(bitnet_env)]
fn gpu_fake_none_clears_rocm_available() {
    temp_env::with_vars(
        [("BITNET_STRICT_MODE", None::<&str>), ("BITNET_GPU_FAKE", Some("none"))],
        || {
            let caps = probe_gpu();
            assert!(!caps.rocm_available, "BITNET_GPU_FAKE=none must clear rocm_available");
            assert!(!caps.cuda_available, "BITNET_GPU_FAKE=none must clear cuda_available");
            assert!(!caps.available, "BITNET_GPU_FAKE=none must clear available");
        },
    );
}

/// Known GPU-fake values that should enable ROCm.
#[cfg(any(feature = "gpu", feature = "rocm"))]
const ROCM_FAKE_PRESENT: &[&str] = &["rocm", "ROCM", "gpu", "GPU"];

/// Known GPU-fake values that should disable ROCm.
#[cfg(any(feature = "gpu", feature = "rocm"))]
const ROCM_FAKE_ABSENT: &[&str] = &["none", "NONE", "cuda", "CUDA"];

proptest! {
    #[test]
    #[serial(bitnet_env)]
    #[cfg(any(feature = "gpu", feature = "rocm"))]
    fn rocm_fake_present_values_enable_rocm(idx in 0..ROCM_FAKE_PRESENT.len()) {
        let val = ROCM_FAKE_PRESENT[idx];
        temp_env::with_vars(
            [("BITNET_STRICT_MODE", None::<&str>), ("BITNET_GPU_FAKE", Some(val))],
            || {
                let caps = probe_gpu();
                prop_assert!(
                    caps.rocm_available,
                    "BITNET_GPU_FAKE={val} should set rocm_available=true"
                );
                Ok(())
            },
        )?;
    }
}

proptest! {
    #[test]
    #[serial(bitnet_env)]
    #[cfg(any(feature = "gpu", feature = "rocm"))]
    fn rocm_fake_absent_values_disable_rocm(idx in 0..ROCM_FAKE_ABSENT.len()) {
        let val = ROCM_FAKE_ABSENT[idx];
        temp_env::with_vars(
            [("BITNET_STRICT_MODE", None::<&str>), ("BITNET_GPU_FAKE", Some(val))],
            || {
                let caps = probe_gpu();
                prop_assert!(
                    !caps.rocm_available,
                    "BITNET_GPU_FAKE={val} should leave rocm_available=false"
                );
                Ok(())
            },
        )?;
    }
}

/// CUDA fake value must NOT set `rocm_available` (only `cuda_available`).
#[cfg(any(feature = "gpu", feature = "cuda", feature = "rocm"))]
#[test]
#[serial(bitnet_env)]
fn rocm_not_set_when_fake_is_cuda_only() {
    temp_env::with_vars(
        [("BITNET_STRICT_MODE", None::<&str>), ("BITNET_GPU_FAKE", Some("cuda"))],
        || {
            let caps = probe_gpu();
            assert!(!caps.rocm_available, "BITNET_GPU_FAKE=cuda must NOT set rocm_available");
        },
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ROCm: DeviceProbe.rocm_available
// ─────────────────────────────────────────────────────────────────────────────

/// `DeviceProbe.rocm_available` field must be accessible without panic.
#[test]
fn device_probe_rocm_available_field_accessible() {
    let probe = probe_device();
    let _: bool = probe.rocm_available;
}

/// Without GPU feature, `DeviceProbe.rocm_available` must be `false`.
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
#[test]
fn device_probe_rocm_available_false_without_gpu_feature() {
    assert!(!probe_device().rocm_available, "rocm_available must be false without GPU feature");
}

/// `DeviceProbe.rocm_available` must be deterministic across consecutive calls.
proptest! {
    #[test]
    fn device_probe_rocm_available_is_deterministic(_n in 0u8..=8) {
        let a = probe_device();
        let b = probe_device();
        prop_assert_eq!(
            a.rocm_available, b.rocm_available,
            "DeviceProbe.rocm_available must be deterministic"
        );
    }
}

/// `DeviceProbe.rocm_available` must agree with `GpuCapabilities.rocm_available`.
#[test]
fn device_probe_rocm_consistent_with_probe_gpu() {
    let probe = probe_device();
    let gpu = probe_gpu();
    assert_eq!(
        probe.rocm_available, gpu.rocm_available,
        "DeviceProbe.rocm_available must match GpuCapabilities.rocm_available"
    );
}

/// `DeviceProbe.cuda_available` must agree with `GpuCapabilities.cuda_available`.
#[test]
fn device_probe_cuda_consistent_with_probe_gpu() {
    let probe = probe_device();
    let gpu = probe_gpu();
    assert_eq!(
        probe.cuda_available, gpu.cuda_available,
        "DeviceProbe.cuda_available must match GpuCapabilities.cuda_available"
    );
}

/// `DeviceProbe` clone must equal the original (roundtrip validation).
#[test]
fn device_probe_clone_roundtrip() {
    let probe = probe_device();
    assert_eq!(probe.clone(), probe, "DeviceProbe clone must equal original");
}

/// `DeviceProbe` Debug output must contain both availability field names.
#[test]
fn device_probe_debug_contains_rocm_and_cuda_fields() {
    let probe = probe_device();
    let debug_str = format!("{probe:?}");
    assert!(debug_str.contains("rocm_available"), "Debug must contain 'rocm_available'");
    assert!(debug_str.contains("cuda_available"), "Debug must contain 'cuda_available'");
}

/// `DeviceProbe` fields are all consistent: the probe never panics.
proptest! {
    #[test]
    fn device_probe_roundtrip_never_panics(_n in 0u8..=8) {
        let a = probe_device();
        let b = a.clone();
        prop_assert_eq!(a, b);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vulkan backend: vulkan_compiled / vulkan_available_runtime
// ─────────────────────────────────────────────────────────────────────────────

/// `vulkan_compiled()` must equal `cfg!(feature = "vulkan")` at compile time.
#[test]
fn vulkan_compiled_reflects_feature_flag() {
    assert_eq!(vulkan_compiled(), cfg!(feature = "vulkan"));
}

/// Without the `vulkan` feature, `vulkan_compiled()` must return `false`.
#[cfg(not(feature = "vulkan"))]
#[test]
fn vulkan_compiled_false_without_vulkan_feature() {
    assert!(!vulkan_compiled(), "vulkan_compiled() must be false without 'vulkan' feature");
}

/// `vulkan_compiled()` is a compile-time constant; repeated calls must agree.
proptest! {
    #[test]
    fn vulkan_compiled_is_stable_across_calls(_n in 0u8..=8) {
        prop_assert_eq!(vulkan_compiled(), vulkan_compiled());
    }
}

/// Without the `vulkan` feature, `vulkan_available_runtime()` must return `false`.
#[cfg(not(feature = "vulkan"))]
#[test]
fn vulkan_available_runtime_false_without_vulkan_feature() {
    assert!(
        !vulkan_available_runtime(),
        "vulkan_available_runtime() must be false without 'vulkan' feature"
    );
}

/// `vulkan_available_runtime()` is deterministic across consecutive calls.
proptest! {
    #[test]
    fn vulkan_available_runtime_is_deterministic(_n in 0u8..=8) {
        prop_assert_eq!(vulkan_available_runtime(), vulkan_available_runtime());
    }
}

/// When Vulkan is not compiled, `vulkan_available_runtime()` must be `false`.
///
/// Even if a Vulkan driver happens to be installed on the test machine, the
/// function must return `false` if the crate was not compiled with the `vulkan`
/// feature.
#[test]
fn vulkan_available_implies_vulkan_compiled() {
    if vulkan_available_runtime() {
        assert!(
            vulkan_compiled(),
            "vulkan_available_runtime()=true requires vulkan_compiled()=true"
        );
    }
}

/// `vulkan_compiled()` returns `false` in the cpu-only build.
#[cfg(not(feature = "vulkan"))]
#[test]
fn vulkan_compiled_is_false_with_cpu_only_features() {
    // Document the invariant: cpu-only builds never include Vulkan.
    assert!(!vulkan_compiled());
}

// ─────────────────────────────────────────────────────────────────────────────
// oneAPI feature flag (compile-time gate only; no runtime function yet)
// ─────────────────────────────────────────────────────────────────────────────

/// The `oneapi` feature flag can be queried at compile time without panicking.
///
/// This test documents that the feature exists in `Cargo.toml` as a future
/// extension point. When a `oneapi_compiled()` function is added to the public
/// API, this test should be updated to call it.
#[test]
fn oneapi_feature_flag_is_accessible() {
    // oneAPI feature exists as a Cargo gate but has no runtime function yet.
    // Verify the feature flag can be evaluated and produces a bool.
    let _oneapi_enabled: bool = cfg!(feature = "oneapi");
}

/// Without the `oneapi` feature, it must evaluate to `false`.
#[cfg(not(feature = "oneapi"))]
#[test]
fn oneapi_feature_disabled_in_cpu_build() {
    assert!(!cfg!(feature = "oneapi"), "oneapi feature must be false in cpu-only builds");
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU feature flag interactions
// ─────────────────────────────────────────────────────────────────────────────

/// `gpu_compiled()` must reflect whether ANY GPU backend was compiled.
///
/// `gpu_compiled()` returns `true` iff `feature="gpu"`, `feature="cuda"`, or
/// `feature="rocm"` is active.
#[test]
fn gpu_compiled_reflects_any_gpu_backend() {
    let expected = cfg!(any(feature = "gpu", feature = "cuda", feature = "rocm"));
    assert_eq!(gpu_compiled(), expected, "gpu_compiled() must reflect gpu/cuda/rocm features");
}

/// When `gpu_compiled()` is `true`, at least one of `cuda_compiled` or
/// `rocm_compiled` in `DeviceCapabilities` must also be `true`.
#[test]
fn gpu_compiled_implies_cuda_or_rocm_compiled_in_device_caps() {
    if gpu_compiled() {
        let caps = DeviceCapabilities::detect();
        assert!(
            caps.cuda_compiled || caps.rocm_compiled,
            "gpu_compiled()=true must imply cuda_compiled || rocm_compiled in DeviceCapabilities"
        );
    }
}

/// `DeviceCapabilities.rocm_compiled` must match the `rocm`/`gpu` feature flag.
#[test]
fn device_caps_rocm_compiled_reflects_feature_flag() {
    let caps = DeviceCapabilities::detect();
    assert_eq!(
        caps.rocm_compiled,
        cfg!(any(feature = "gpu", feature = "rocm")),
        "rocm_compiled must equal cfg!(any(feature=\"gpu\", feature=\"rocm\"))"
    );
}

/// `DeviceCapabilities.cuda_compiled` must match the `cuda`/`gpu` feature flag.
#[test]
fn device_caps_cuda_compiled_reflects_feature_flag() {
    let caps = DeviceCapabilities::detect();
    assert_eq!(
        caps.cuda_compiled,
        cfg!(any(feature = "gpu", feature = "cuda")),
        "cuda_compiled must equal cfg!(any(feature=\"gpu\", feature=\"cuda\"))"
    );
}

/// `cuda_compiled || rocm_compiled` must equal `gpu_compiled()`.
///
/// This is the core invariant: the combined compiled-backend flags must agree
/// with the unified `gpu_compiled()` predicate.
#[test]
fn device_caps_or_of_backends_equals_gpu_compiled() {
    let caps = DeviceCapabilities::detect();
    assert_eq!(
        caps.cuda_compiled || caps.rocm_compiled,
        gpu_compiled(),
        "(cuda_compiled || rocm_compiled) must equal gpu_compiled()"
    );
}

/// Without GPU feature, both runtime availability flags must be `false` and
/// `gpu_available_runtime()` must be `false`.
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
#[test]
fn no_gpu_feature_means_no_runtime_availability() {
    assert!(!gpu_available_runtime(), "gpu_available_runtime() must be false without GPU feature");
    let caps = DeviceCapabilities::detect();
    assert!(!caps.cuda_runtime, "cuda_runtime must be false without GPU feature");
    assert!(!caps.rocm_runtime, "rocm_runtime must be false without GPU feature");
}

/// Without GPU feature, `probe_gpu().available` must be `false`.
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm")))]
#[test]
fn probe_gpu_available_false_without_gpu_feature() {
    assert!(!probe_gpu().available, "available must be false without GPU feature");
}

/// `DeviceCapabilities` clone roundtrip must be identity.
proptest! {
    #[test]
    fn device_capabilities_clone_roundtrip(_n in 0u8..=8) {
        let caps = DeviceCapabilities::detect();
        prop_assert_eq!(caps.clone(), caps);
    }
}

/// `DeviceCapabilities.rocm_compiled` is a compile-time constant; stable across calls.
proptest! {
    #[test]
    fn device_caps_rocm_compiled_is_stable(_n in 0u8..=8) {
        let a = DeviceCapabilities::detect();
        let b = DeviceCapabilities::detect();
        prop_assert_eq!(a.rocm_compiled, b.rocm_compiled);
    }
}

/// `DeviceCapabilities.cuda_compiled` is a compile-time constant; stable across calls.
proptest! {
    #[test]
    fn device_caps_cuda_compiled_is_stable(_n in 0u8..=8) {
        let a = DeviceCapabilities::detect();
        let b = DeviceCapabilities::detect();
        prop_assert_eq!(a.cuda_compiled, b.cuda_compiled);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeviceCapabilities: Debug output mentions new fields
// ─────────────────────────────────────────────────────────────────────────────

/// `DeviceCapabilities` Debug output must mention `rocm_compiled` and `rocm_runtime`.
#[test]
fn device_capabilities_debug_mentions_rocm_fields() {
    let caps = DeviceCapabilities::detect();
    let debug_str = format!("{caps:?}");
    assert!(debug_str.contains("rocm_compiled"), "Debug must contain 'rocm_compiled'");
    assert!(debug_str.contains("rocm_runtime"), "Debug must contain 'rocm_runtime'");
}

/// `GpuCapabilities` Debug output must mention `rocm_available`.
#[test]
fn gpu_capabilities_debug_mentions_rocm_available() {
    let caps = probe_gpu();
    let debug_str = format!("{caps:?}");
    assert!(debug_str.contains("rocm_available"), "Debug must contain 'rocm_available'");
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot tests for new-backend Debug output
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot: `GpuCapabilities` Debug with cpu-only feature (all fields false).
#[test]
fn snapshot_gpu_capabilities_debug_no_gpu_feature() {
    let caps = probe_gpu();
    insta::assert_debug_snapshot!("gpu_capabilities_debug_no_gpu", caps);
}

/// Snapshot: `DeviceCapabilities` GPU-related fields with cpu-only feature.
#[test]
fn snapshot_device_capabilities_gpu_fields_no_gpu() {
    let caps = DeviceCapabilities::detect();
    let fields = format!(
        "cuda_compiled={} rocm_compiled={} cuda_runtime={} rocm_runtime={}",
        caps.cuda_compiled, caps.rocm_compiled, caps.cuda_runtime, caps.rocm_runtime
    );
    insta::assert_snapshot!("device_capabilities_gpu_fields_no_gpu", fields);
}

/// Snapshot: `vulkan_compiled()` with cpu-only feature (expected: false).
#[test]
fn snapshot_vulkan_compiled_no_vulkan_feature() {
    insta::assert_snapshot!("vulkan_compiled_no_feature", vulkan_compiled().to_string());
}

/// Snapshot: `DeviceProbe` availability fields with cpu-only feature.
#[test]
fn snapshot_device_probe_availability_no_gpu() {
    let probe = probe_device();
    let fields =
        format!("cuda_available={} rocm_available={}", probe.cuda_available, probe.rocm_available);
    insta::assert_snapshot!("device_probe_availability_no_gpu", fields);
}
