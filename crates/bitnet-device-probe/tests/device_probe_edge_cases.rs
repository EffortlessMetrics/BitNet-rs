//! Edge-case tests for the device-probe crate covering CPU/GPU capability
//! detection, SIMD level ranking, DeviceCapabilities snapshots, and
//! the full DeviceProbe path.

use bitnet_device_probe::{
    DeviceCapabilities, SimdLevel, detect_simd_level, gpu_available_runtime, gpu_compiled,
    probe_cpu, probe_device, probe_gpu, simd_level_rank,
};

// ── CPU Capabilities ─────────────────────────────────────────────────────

#[test]
fn probe_cpu_has_at_least_one_core() {
    let caps = probe_cpu();
    assert!(caps.core_count >= 1);
}

#[test]
fn probe_cpu_neon_and_avx_mutually_exclusive() {
    let caps = probe_cpu();
    // On x86_64 neon is always false; on aarch64 avx is always false
    assert!(!(caps.has_avx2 && caps.has_neon));
    assert!(!(caps.has_avx512 && caps.has_neon));
}

#[test]
fn probe_cpu_avx512_implies_avx2() {
    let caps = probe_cpu();
    if caps.has_avx512 {
        assert!(caps.has_avx2, "AVX-512 implies AVX2 support");
    }
}

#[test]
fn probe_cpu_is_deterministic() {
    let a = probe_cpu();
    let b = probe_cpu();
    assert_eq!(a, b);
}

#[test]
fn cpu_capabilities_debug_format() {
    let caps = probe_cpu();
    let dbg = format!("{:?}", caps);
    assert!(dbg.contains("core_count"));
    assert!(dbg.contains("has_avx2"));
}

#[test]
fn cpu_capabilities_clone_eq() {
    let caps = probe_cpu();
    let clone = caps.clone();
    assert_eq!(caps, clone);
}

// ── GPU Capabilities (CPU-only build) ────────────────────────────────────

#[test]
fn probe_gpu_no_gpu_when_cpu_only() {
    let caps = probe_gpu();
    // Without GPU features, all should be false
    if !gpu_compiled() {
        assert!(!caps.available);
        assert!(!caps.cuda_available);
        assert!(!caps.rocm_available);
        assert!(!caps.oneapi_available);
    }
}

#[test]
fn gpu_compiled_matches_feature_gates() {
    // In CPU-only builds, this should be false
    let compiled = gpu_compiled();
    // Can't assert true/false portably, but verify it's a bool
    let _: bool = compiled;
}

#[test]
fn gpu_available_runtime_consistent_with_compiled() {
    if !gpu_compiled() {
        assert!(!gpu_available_runtime());
    }
}

#[test]
fn gpu_capabilities_debug_format() {
    let caps = probe_gpu();
    let dbg = format!("{:?}", caps);
    assert!(dbg.contains("available"));
    assert!(dbg.contains("cuda_available"));
}

#[test]
fn gpu_capabilities_clone_eq() {
    let caps = probe_gpu();
    let clone = caps.clone();
    assert_eq!(caps, clone);
}

// ── SIMD Level Detection ─────────────────────────────────────────────────

#[test]
fn detect_simd_returns_valid_level() {
    let level = detect_simd_level();
    // Must be one of the valid variants
    let rank = simd_level_rank(&level);
    assert!(rank <= 4);
}

#[test]
fn detect_simd_is_deterministic() {
    let a = detect_simd_level();
    let b = detect_simd_level();
    assert_eq!(a, b);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn detect_simd_x86_not_neon() {
    let level = detect_simd_level();
    assert_ne!(level, SimdLevel::Neon);
}

// ── SimdLevel Ranking ────────────────────────────────────────────────────

#[test]
fn simd_rank_scalar_is_zero() {
    assert_eq!(simd_level_rank(&SimdLevel::Scalar), 0);
}

#[test]
fn simd_rank_ordering_x86() {
    assert!(simd_level_rank(&SimdLevel::Avx512) > simd_level_rank(&SimdLevel::Avx2));
    assert!(simd_level_rank(&SimdLevel::Avx2) > simd_level_rank(&SimdLevel::Sse42));
    assert!(simd_level_rank(&SimdLevel::Sse42) > simd_level_rank(&SimdLevel::Scalar));
}

#[test]
fn simd_rank_neon_highest() {
    assert_eq!(simd_level_rank(&SimdLevel::Neon), 4);
}

#[test]
fn simd_rank_all_variants_unique() {
    let ranks: Vec<u32> =
        [SimdLevel::Scalar, SimdLevel::Sse42, SimdLevel::Avx2, SimdLevel::Avx512, SimdLevel::Neon]
            .iter()
            .map(simd_level_rank)
            .collect();

    // All ranks should be unique
    let mut sorted = ranks.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(ranks.len(), sorted.len());
}

// ── DeviceCapabilities ───────────────────────────────────────────────────

#[test]
fn device_capabilities_cpu_always_available() {
    let caps = DeviceCapabilities::detect();
    assert!(caps.cpu_rust, "CPU backend is always available");
}

#[test]
fn device_capabilities_gpu_compiled_consistent() {
    let caps = DeviceCapabilities::detect();
    assert_eq!(caps.cuda_compiled || caps.rocm_compiled || caps.oneapi_compiled, gpu_compiled(),);
}

#[test]
fn device_capabilities_no_cuda_runtime_without_compile() {
    let caps = DeviceCapabilities::detect();
    if !caps.cuda_compiled {
        assert!(!caps.cuda_runtime);
    }
}

#[test]
fn device_capabilities_simd_matches_detect() {
    let caps = DeviceCapabilities::detect();
    assert_eq!(caps.simd_level, detect_simd_level());
}

#[test]
fn device_capabilities_is_deterministic() {
    let a = DeviceCapabilities::detect();
    let b = DeviceCapabilities::detect();
    assert_eq!(a, b);
}

#[test]
fn device_capabilities_debug_format() {
    let caps = DeviceCapabilities::detect();
    let dbg = format!("{:?}", caps);
    assert!(dbg.contains("cpu_rust"));
    assert!(dbg.contains("simd_level"));
}

// ── DeviceProbe ──────────────────────────────────────────────────────────

#[test]
fn probe_device_cores_at_least_one() {
    let probe = probe_device();
    assert!(probe.cpu.cores >= 1);
    assert!(probe.cpu.threads >= 1);
}

#[test]
fn probe_device_threads_gte_cores() {
    let probe = probe_device();
    assert!(probe.cpu.threads >= probe.cpu.cores);
}

#[test]
fn probe_device_simd_matches_detect() {
    let probe = probe_device();
    assert_eq!(probe.cpu.simd_level, detect_simd_level());
}

#[test]
fn probe_device_gpu_consistent() {
    let probe = probe_device();
    if !gpu_compiled() {
        assert!(!probe.cuda_available);
        assert!(!probe.rocm_available);
        assert!(!probe.oneapi_available);
    }
}

#[test]
fn probe_device_is_deterministic() {
    let a = probe_device();
    let b = probe_device();
    assert_eq!(a, b);
}

#[test]
fn probe_device_debug_format() {
    let probe = probe_device();
    let dbg = format!("{:?}", probe);
    assert!(dbg.contains("cpu"));
    assert!(dbg.contains("cuda_available"));
}

// ── NPU (CPU-only build) ────────────────────────────────────────────────

#[test]
fn npu_compiled_is_bool() {
    let _: bool = bitnet_device_probe::npu_compiled();
}

#[test]
fn probe_npu_not_available_without_feature() {
    if !bitnet_device_probe::npu_compiled() {
        let npu = bitnet_device_probe::probe_npu();
        assert!(!npu.available);
    }
}

// ── Vulkan (CPU-only build) ──────────────────────────────────────────────

#[test]
fn vulkan_compiled_is_bool() {
    let _: bool = bitnet_device_probe::vulkan_compiled();
}

#[test]
fn vulkan_runtime_not_available_without_feature() {
    if !bitnet_device_probe::vulkan_compiled() {
        assert!(!bitnet_device_probe::vulkan_available_runtime());
    }
}

// ── oneAPI (CPU-only build) ──────────────────────────────────────────────

#[test]
fn oneapi_compiled_is_bool() {
    let _: bool = bitnet_device_probe::oneapi_compiled();
}

#[test]
fn oneapi_runtime_not_available_without_feature() {
    if !bitnet_device_probe::oneapi_compiled() {
        assert!(!bitnet_device_probe::oneapi_available_runtime());
    }
}
