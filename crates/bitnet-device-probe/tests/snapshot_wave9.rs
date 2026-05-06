//! Snapshot wave 9 — device probe types and formatting.
//!
//! Pins the Debug output and invariant properties of device probe types
//! to catch accidental regressions. Machine-specific values (core count,
//! SIMD level) are redacted or tested as invariants rather than exact values.

#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi")))]
use bitnet_device_probe::{DeviceCapabilities, probe_gpu};
use bitnet_device_probe::{SimdLevel, probe_cpu, probe_device, probe_npu};

// ── CpuCapabilities ────────────────────────────────────────────────

#[test]
fn cpu_capabilities_core_count_positive() {
    let caps = probe_cpu();
    insta::assert_snapshot!(format!("core_count_positive={}", caps.core_count >= 1));
}

#[test]
fn cpu_capabilities_simd_flags_exclusive() {
    let caps = probe_cpu();
    // AVX and NEON are mutually exclusive across architectures
    let exclusive = !(caps.has_avx2 && caps.has_neon);
    insta::assert_snapshot!(format!("avx_neon_exclusive={exclusive}"));
}

#[test]
fn cpu_capabilities_debug_shape() {
    let caps = probe_cpu();
    let debug = format!("{caps:?}");
    insta::with_settings!({
        filters => vec![
            (r"core_count: \d+", "core_count: [N]"),
            (r"has_avx2: (true|false)", "has_avx2: [ARCH]"),
            (r"has_avx512: (true|false)", "has_avx512: [ARCH]"),
            (r"has_neon: (true|false)", "has_neon: [ARCH]"),
        ]
    }, {
        insta::assert_snapshot!(debug);
    });
}

// ── GpuCapabilities ────────────────────────────────────────────────

#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi")))]
fn gpu_capabilities_cpu_only_build() {
    let caps = probe_gpu();
    insta::assert_snapshot!(format!(
        "available={} cuda={} rocm={} oneapi={}",
        caps.available, caps.cuda_available, caps.rocm_available, caps.oneapi_available
    ));
}

#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi")))]
fn gpu_capabilities_debug() {
    let caps = probe_gpu();
    insta::assert_debug_snapshot!(caps);
}

// ── NpuCapabilities ────────────────────────────────────────────────

#[test]
fn npu_capabilities_without_feature() {
    let caps = probe_npu();
    insta::assert_snapshot!(format!(
        "available={} accel_device={}",
        caps.available, caps.accel_device_present
    ));
}

#[test]
fn npu_capabilities_debug() {
    let caps = probe_npu();
    insta::assert_debug_snapshot!(caps);
}

// ── SimdLevel Display ──────────────────────────────────────────────

#[test]
fn simd_level_display_all() {
    let levels =
        [SimdLevel::Scalar, SimdLevel::Neon, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512];
    let output: Vec<String> = levels.iter().map(|l| format!("{l}")).collect();
    insta::assert_snapshot!(output.join("\n"));
}

// ── DeviceProbe ────────────────────────────────────────────────────

#[test]
fn device_probe_cpu_cores_positive() {
    let probe = probe_device();
    insta::assert_snapshot!(format!(
        "cores_positive={} threads_positive={}",
        probe.cpu.cores >= 1,
        probe.cpu.threads >= 1
    ));
}

#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi")))]
fn device_probe_debug_shape() {
    let probe = probe_device();
    let debug = format!("{probe:?}");
    insta::with_settings!({
        filters => vec![
            (r"cores: \d+", "cores: [N]"),
            (r"threads: \d+", "threads: [N]"),
            (r"simd_level: \w+", "simd_level: [SIMD]"),
        ]
    }, {
        insta::assert_snapshot!(debug);
    });
}

// ── DeviceCapabilities ─────────────────────────────────────────────

#[test]
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "rocm", feature = "oneapi")))]
fn device_capabilities_detect_invariants() {
    let caps = DeviceCapabilities::detect();
    insta::with_settings!({
        filters => vec![
            (r"simd_level: \w+", "simd_level: [SIMD]"),
        ]
    }, {
        insta::assert_debug_snapshot!(caps);
    });
}
