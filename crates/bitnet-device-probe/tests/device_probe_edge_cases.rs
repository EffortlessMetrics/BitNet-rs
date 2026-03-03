//! Edge-case tests for bitnet-device-probe public API.

use bitnet_device_probe::{
    DeviceCapabilities, SimdLevel, detect_simd_level, gpu_available_runtime, gpu_compiled,
    npu_compiled, oneapi_available_runtime, oneapi_compiled, probe_cpu, probe_device, probe_gpu,
    probe_npu, simd_level_rank, vulkan_available_runtime, vulkan_compiled,
};

// ---------------------------------------------------------------------------
// probe_cpu
// ---------------------------------------------------------------------------

#[test]
fn probe_cpu_core_count_at_least_one() {
    let cpu = probe_cpu();
    assert!(cpu.core_count >= 1);
}

#[test]
fn probe_cpu_avx_neon_mutually_exclusive() {
    let cpu = probe_cpu();
    assert!(!(cpu.has_avx2 && cpu.has_neon));
    assert!(!(cpu.has_avx512 && cpu.has_neon));
}

#[test]
fn probe_cpu_avx512_implies_avx2() {
    let cpu = probe_cpu();
    if cpu.has_avx512 {
        assert!(cpu.has_avx2, "AVX-512 implies AVX2");
    }
}

#[test]
fn probe_cpu_eq_and_clone() {
    let a = probe_cpu();
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn probe_cpu_debug() {
    let cpu = probe_cpu();
    let d = format!("{:?}", cpu);
    assert!(d.contains("CpuCapabilities"));
}

// ---------------------------------------------------------------------------
// probe_gpu (without GPU features, always unavailable)
// ---------------------------------------------------------------------------

#[test]
fn probe_gpu_no_features() {
    let gpu = probe_gpu();
    if !gpu_compiled() {
        assert!(!gpu.available);
        assert!(!gpu.cuda_available);
        assert!(!gpu.rocm_available);
        assert!(!gpu.oneapi_available);
    }
}

#[test]
fn probe_gpu_eq_and_clone() {
    let a = probe_gpu();
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn probe_gpu_debug() {
    let gpu = probe_gpu();
    let d = format!("{:?}", gpu);
    assert!(d.contains("GpuCapabilities"));
}

// ---------------------------------------------------------------------------
// probe_npu
// ---------------------------------------------------------------------------

#[test]
fn probe_npu_no_features() {
    let npu = probe_npu();
    if !npu_compiled() {
        assert!(!npu.available);
        assert!(!npu.accel_device_present);
    }
}

#[test]
fn probe_npu_eq_and_clone() {
    let a = probe_npu();
    let b = a.clone();
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// detect_simd_level
// ---------------------------------------------------------------------------

#[test]
fn detect_simd_level_deterministic() {
    let a = detect_simd_level();
    let b = detect_simd_level();
    assert_eq!(a, b);
}

#[test]
fn detect_simd_level_debug() {
    let level = detect_simd_level();
    let d = format!("{:?}", level);
    assert!(!d.is_empty());
}

// ---------------------------------------------------------------------------
// simd_level_rank
// ---------------------------------------------------------------------------

#[test]
fn simd_level_rank_scalar_is_zero() {
    assert_eq!(simd_level_rank(&SimdLevel::Scalar), 0);
}

#[test]
fn simd_level_rank_ordering() {
    assert!(simd_level_rank(&SimdLevel::Sse42) > simd_level_rank(&SimdLevel::Scalar));
    assert!(simd_level_rank(&SimdLevel::Avx2) > simd_level_rank(&SimdLevel::Sse42));
    assert!(simd_level_rank(&SimdLevel::Avx512) > simd_level_rank(&SimdLevel::Avx2));
    assert!(simd_level_rank(&SimdLevel::Neon) > simd_level_rank(&SimdLevel::Avx512));
}

#[test]
fn simd_level_rank_neon_is_four() {
    assert_eq!(simd_level_rank(&SimdLevel::Neon), 4);
}

// ---------------------------------------------------------------------------
// gpu_compiled / gpu_available_runtime
// ---------------------------------------------------------------------------

#[test]
fn gpu_compiled_is_bool() {
    let _: bool = gpu_compiled();
}

#[test]
fn gpu_available_runtime_is_bool() {
    let _: bool = gpu_available_runtime();
}

#[test]
fn gpu_runtime_implies_compiled() {
    if gpu_available_runtime() {
        assert!(gpu_compiled(), "GPU runtime available but not compiled");
    }
}

// ---------------------------------------------------------------------------
// oneapi_compiled / oneapi_available_runtime
// ---------------------------------------------------------------------------

#[test]
fn oneapi_compiled_reflects_feature() {
    assert_eq!(oneapi_compiled(), cfg!(feature = "oneapi"));
}

#[test]
fn oneapi_runtime_implies_compiled() {
    if oneapi_available_runtime() {
        assert!(oneapi_compiled(), "oneAPI runtime available but not compiled");
    }
}

// ---------------------------------------------------------------------------
// npu_compiled
// ---------------------------------------------------------------------------

#[test]
fn npu_compiled_reflects_feature() {
    assert_eq!(npu_compiled(), cfg!(feature = "npu"));
}

// ---------------------------------------------------------------------------
// vulkan
// ---------------------------------------------------------------------------

#[test]
fn vulkan_compiled_reflects_feature() {
    assert_eq!(vulkan_compiled(), cfg!(feature = "vulkan"));
}

#[test]
fn vulkan_runtime_implies_compiled() {
    if vulkan_available_runtime() {
        assert!(vulkan_compiled(), "Vulkan runtime available but not compiled");
    }
}

// ---------------------------------------------------------------------------
// DeviceCapabilities::detect
// ---------------------------------------------------------------------------

#[test]
fn device_capabilities_cpu_always_true() {
    let caps = DeviceCapabilities::detect();
    assert!(caps.cpu_rust);
}

#[test]
fn device_capabilities_compiled_consistency() {
    let caps = DeviceCapabilities::detect();
    assert_eq!(caps.cuda_compiled || caps.rocm_compiled || caps.oneapi_compiled, gpu_compiled());
}

#[test]
fn device_capabilities_simd_matches_detect() {
    let caps = DeviceCapabilities::detect();
    assert_eq!(caps.simd_level, detect_simd_level());
}

#[test]
fn device_capabilities_eq_and_clone() {
    let a = DeviceCapabilities::detect();
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn device_capabilities_debug() {
    let caps = DeviceCapabilities::detect();
    let d = format!("{:?}", caps);
    assert!(d.contains("DeviceCapabilities"));
}

// ---------------------------------------------------------------------------
// probe_device
// ---------------------------------------------------------------------------

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
fn probe_device_eq_and_clone() {
    let a = probe_device();
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn probe_device_debug() {
    let probe = probe_device();
    let d = format!("{:?}", probe);
    assert!(d.contains("DeviceProbe"));
}

// ---------------------------------------------------------------------------
// Consistency across probes
// ---------------------------------------------------------------------------

#[test]
fn probe_cpu_consistent_with_probe_device() {
    let cpu = probe_cpu();
    let device = probe_device();
    assert_eq!(cpu.core_count, device.cpu.cores);
}
