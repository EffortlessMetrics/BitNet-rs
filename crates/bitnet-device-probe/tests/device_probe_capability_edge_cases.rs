//! Edge-case tests for `bitnet-device-probe` capability detection.

use bitnet_device_probe::{
    CpuCapabilities, CpuProbe, DeviceCapabilities, DeviceProbe, GpuCapabilities, NpuCapabilities,
    SimdLevel, detect_simd_level, gpu_available_runtime, gpu_compiled, npu_compiled,
    oneapi_available_runtime, oneapi_compiled, probe_cpu, probe_device, probe_gpu, probe_npu,
    simd_level_rank, vulkan_available_runtime, vulkan_compiled,
};

// ═══════════════════════════════════════════════════════════════════════════════
// 1. CpuCapabilities — field access, defaults, validation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn cpu_capabilities_manual_construction_zero_cores() {
    let cpu =
        CpuCapabilities { core_count: 0, has_avx2: false, has_avx512: false, has_neon: false };
    assert_eq!(cpu.core_count, 0);
}

#[test]
fn cpu_capabilities_all_simd_flags_true() {
    // Architecturally impossible but struct allows it — tests that fields are independent.
    let cpu = CpuCapabilities { core_count: 1, has_avx2: true, has_avx512: true, has_neon: true };
    assert!(cpu.has_avx2);
    assert!(cpu.has_avx512);
    assert!(cpu.has_neon);
}

#[test]
fn cpu_capabilities_max_cores() {
    let cpu = CpuCapabilities {
        core_count: usize::MAX,
        has_avx2: false,
        has_avx512: false,
        has_neon: false,
    };
    assert_eq!(cpu.core_count, usize::MAX);
}

#[test]
fn cpu_capabilities_clone_independence() {
    let a = CpuCapabilities { core_count: 4, has_avx2: true, has_avx512: false, has_neon: false };
    let mut b = a.clone();
    b.core_count = 8;
    assert_eq!(a.core_count, 4);
    assert_eq!(b.core_count, 8);
}

#[test]
fn cpu_capabilities_equality_field_sensitive() {
    let base =
        CpuCapabilities { core_count: 4, has_avx2: true, has_avx512: false, has_neon: false };
    let diff_cores =
        CpuCapabilities { core_count: 8, has_avx2: true, has_avx512: false, has_neon: false };
    let diff_avx2 =
        CpuCapabilities { core_count: 4, has_avx2: false, has_avx512: false, has_neon: false };
    assert_ne!(base, diff_cores);
    assert_ne!(base, diff_avx2);
    assert_eq!(base, base.clone());
}

#[test]
fn cpu_capabilities_debug_contains_field_names() {
    let cpu = CpuCapabilities { core_count: 2, has_avx2: true, has_avx512: false, has_neon: false };
    let d = format!("{cpu:?}");
    assert!(d.contains("core_count"));
    assert!(d.contains("has_avx2"));
    assert!(d.contains("has_avx512"));
    assert!(d.contains("has_neon"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. GpuCapabilities — available flag logic, default values
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn gpu_capabilities_all_false() {
    let gpu = GpuCapabilities {
        available: false,
        cuda_available: false,
        rocm_available: false,
        oneapi_available: false,
    };
    assert!(!gpu.available);
    assert!(!gpu.cuda_available);
}

#[test]
fn gpu_capabilities_available_without_backends() {
    // Edge case: `available` is true but no specific backend flagged.
    let gpu = GpuCapabilities {
        available: true,
        cuda_available: false,
        rocm_available: false,
        oneapi_available: false,
    };
    assert!(gpu.available);
}

#[test]
fn gpu_capabilities_single_backend_cuda() {
    let gpu = GpuCapabilities {
        available: true,
        cuda_available: true,
        rocm_available: false,
        oneapi_available: false,
    };
    assert!(gpu.cuda_available);
    assert!(!gpu.rocm_available);
    assert!(!gpu.oneapi_available);
}

#[test]
fn gpu_capabilities_all_backends() {
    let gpu = GpuCapabilities {
        available: true,
        cuda_available: true,
        rocm_available: true,
        oneapi_available: true,
    };
    assert!(gpu.cuda_available && gpu.rocm_available && gpu.oneapi_available);
}

#[test]
fn gpu_capabilities_equality_field_sensitive() {
    let a = GpuCapabilities {
        available: true,
        cuda_available: true,
        rocm_available: false,
        oneapi_available: false,
    };
    let b = GpuCapabilities {
        available: true,
        cuda_available: false,
        rocm_available: true,
        oneapi_available: false,
    };
    assert_ne!(a, b);
}

#[test]
fn gpu_capabilities_debug_contains_fields() {
    let gpu = GpuCapabilities {
        available: false,
        cuda_available: false,
        rocm_available: false,
        oneapi_available: false,
    };
    let d = format!("{gpu:?}");
    assert!(d.contains("GpuCapabilities"));
    assert!(d.contains("cuda_available"));
    assert!(d.contains("rocm_available"));
    assert!(d.contains("oneapi_available"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. NpuCapabilities — construction, defaults
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn npu_capabilities_both_false() {
    let npu = NpuCapabilities { available: false, accel_device_present: false };
    assert!(!npu.available);
    assert!(!npu.accel_device_present);
}

#[test]
fn npu_capabilities_available_without_accel() {
    let npu = NpuCapabilities { available: true, accel_device_present: false };
    assert!(npu.available);
    assert!(!npu.accel_device_present);
}

#[test]
fn npu_capabilities_accel_without_available() {
    let npu = NpuCapabilities { available: false, accel_device_present: true };
    assert!(!npu.available);
    assert!(npu.accel_device_present);
}

#[test]
fn npu_capabilities_clone_and_eq() {
    let a = NpuCapabilities { available: true, accel_device_present: true };
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn npu_capabilities_debug_contains_fields() {
    let npu = NpuCapabilities { available: false, accel_device_present: false };
    let d = format!("{npu:?}");
    assert!(d.contains("NpuCapabilities"));
    assert!(d.contains("accel_device_present"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. probe_cpu() — returns valid CPU info on current machine
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn probe_cpu_core_count_is_positive() {
    assert!(probe_cpu().core_count >= 1, "must have at least 1 core");
}

#[test]
fn probe_cpu_simd_flags_arch_consistent() {
    let cpu = probe_cpu();
    // On x86_64 NEON must be false; on aarch64 AVX must be false.
    if cfg!(target_arch = "x86_64") {
        assert!(!cpu.has_neon);
    }
    if cfg!(target_arch = "aarch64") {
        assert!(!cpu.has_avx2);
        assert!(!cpu.has_avx512);
    }
}

#[test]
fn probe_cpu_avx512_implies_avx2_on_real_hw() {
    let cpu = probe_cpu();
    if cpu.has_avx512 {
        assert!(cpu.has_avx2, "AVX-512 hardware always supports AVX2");
    }
}

#[test]
fn probe_cpu_is_idempotent() {
    let a = probe_cpu();
    let b = probe_cpu();
    assert_eq!(a, b, "successive calls must agree");
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. probe_gpu() — returns GPU info
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn probe_gpu_returns_without_panic() {
    let _gpu = probe_gpu();
}

#[test]
fn probe_gpu_available_consistency() {
    let gpu = probe_gpu();
    // If no backend is available, `available` must be false (when not using FAKE).
    if !gpu_compiled() {
        assert!(!gpu.available);
    }
}

#[test]
fn probe_gpu_no_backend_implies_not_available_without_gpu_feature() {
    if !gpu_compiled() {
        let gpu = probe_gpu();
        assert!(!gpu.cuda_available);
        assert!(!gpu.rocm_available);
        assert!(!gpu.oneapi_available);
        assert!(!gpu.available);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. detect_simd_level() — returns a valid SimdLevel
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn detect_simd_level_is_known_variant() {
    let level = detect_simd_level();
    // Must match one of the known variants.
    let valid = matches!(
        level,
        SimdLevel::Scalar
            | SimdLevel::Sse42
            | SimdLevel::Avx2
            | SimdLevel::Avx512
            | SimdLevel::Neon
    );
    assert!(valid, "unexpected SimdLevel variant: {level:?}");
}

#[test]
fn detect_simd_level_arch_consistent() {
    let level = detect_simd_level();
    if cfg!(target_arch = "x86_64") {
        assert_ne!(level, SimdLevel::Neon, "x86_64 must not detect NEON");
    }
    if cfg!(target_arch = "aarch64") {
        assert_eq!(level, SimdLevel::Neon, "aarch64 must detect NEON");
    }
}

#[test]
fn detect_simd_level_repeated_is_stable() {
    let results: Vec<_> = (0..10).map(|_| detect_simd_level()).collect();
    assert!(results.windows(2).all(|w| w[0] == w[1]));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. simd_level_rank() — ranking comparison
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn simd_rank_scalar_is_minimum() {
    assert_eq!(simd_level_rank(&SimdLevel::Scalar), 0);
}

#[test]
fn simd_rank_strict_ordering_x86() {
    let scalar = simd_level_rank(&SimdLevel::Scalar);
    let sse42 = simd_level_rank(&SimdLevel::Sse42);
    let avx2 = simd_level_rank(&SimdLevel::Avx2);
    let avx512 = simd_level_rank(&SimdLevel::Avx512);
    assert!(scalar < sse42, "Scalar < Sse42");
    assert!(sse42 < avx2, "Sse42 < Avx2");
    assert!(avx2 < avx512, "Avx2 < Avx512");
}

#[test]
fn simd_rank_neon_above_avx512() {
    assert!(
        simd_level_rank(&SimdLevel::Neon) > simd_level_rank(&SimdLevel::Avx512),
        "Neon rank must exceed Avx512"
    );
}

#[test]
fn simd_rank_exact_values() {
    assert_eq!(simd_level_rank(&SimdLevel::Scalar), 0);
    assert_eq!(simd_level_rank(&SimdLevel::Sse42), 1);
    assert_eq!(simd_level_rank(&SimdLevel::Avx2), 2);
    assert_eq!(simd_level_rank(&SimdLevel::Avx512), 3);
    assert_eq!(simd_level_rank(&SimdLevel::Neon), 4);
}

#[test]
fn simd_rank_same_level_equal() {
    assert_eq!(simd_level_rank(&SimdLevel::Avx2), simd_level_rank(&SimdLevel::Avx2));
}

#[test]
fn simd_rank_detected_level_is_nonzero_on_modern_hw() {
    let level = detect_simd_level();
    let rank = simd_level_rank(&level);
    // On any modern x86_64 or aarch64, we expect at least SSE4.2 or NEON.
    if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
        assert!(rank >= 1, "modern hardware should have rank >= 1, got {rank}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 8. ProbeResult aggregation — combining CPU + GPU + NPU
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn device_probe_aggregates_cpu_gpu_npu() {
    let probe = probe_device();
    // CPU fields must be populated.
    assert!(probe.cpu.cores >= 1);
    assert!(probe.cpu.threads >= 1);
    // GPU and NPU flags are booleans — just ensure accessible.
    let _ = probe.cuda_available;
    let _ = probe.rocm_available;
    let _ = probe.npu_available;
    let _ = probe.oneapi_available;
}

#[test]
fn device_probe_manual_construction_minimal() {
    let probe = DeviceProbe {
        cpu: CpuProbe { simd_level: SimdLevel::Scalar, cores: 1, threads: 1 },
        cuda_available: false,
        rocm_available: false,
        npu_available: false,
        oneapi_available: false,
    };
    assert_eq!(probe.cpu.cores, 1);
    assert!(!probe.cuda_available);
}

#[test]
fn device_probe_manual_construction_all_available() {
    let probe = DeviceProbe {
        cpu: CpuProbe { simd_level: SimdLevel::Avx512, cores: 128, threads: 256 },
        cuda_available: true,
        rocm_available: true,
        npu_available: true,
        oneapi_available: true,
    };
    assert!(probe.cuda_available);
    assert!(probe.rocm_available);
    assert!(probe.npu_available);
    assert!(probe.oneapi_available);
    assert_eq!(probe.cpu.simd_level, SimdLevel::Avx512);
}

#[test]
fn cpu_probe_clone_independence() {
    let a = CpuProbe { simd_level: SimdLevel::Avx2, cores: 4, threads: 8 };
    let mut b = a.clone();
    b.cores = 16;
    assert_eq!(a.cores, 4);
    assert_eq!(b.cores, 16);
}

#[test]
fn cpu_probe_equality() {
    let a = CpuProbe { simd_level: SimdLevel::Neon, cores: 8, threads: 8 };
    let b = CpuProbe { simd_level: SimdLevel::Neon, cores: 8, threads: 8 };
    let c = CpuProbe { simd_level: SimdLevel::Scalar, cores: 8, threads: 8 };
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn device_probe_gpu_mirrors_probe_gpu() {
    let dp = probe_device();
    let gpu = probe_gpu();
    assert_eq!(dp.cuda_available, gpu.cuda_available);
    assert_eq!(dp.rocm_available, gpu.rocm_available);
    assert_eq!(dp.oneapi_available, gpu.oneapi_available);
}

#[test]
fn device_probe_npu_mirrors_probe_npu() {
    let dp = probe_device();
    let npu = probe_npu();
    assert_eq!(dp.npu_available, npu.available);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. Display implementations — SimdLevel and Debug for all types
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn simd_level_display_scalar() {
    assert_eq!(format!("{}", SimdLevel::Scalar), "scalar");
}

#[test]
fn simd_level_display_sse42() {
    assert_eq!(format!("{}", SimdLevel::Sse42), "sse4.2");
}

#[test]
fn simd_level_display_avx2() {
    assert_eq!(format!("{}", SimdLevel::Avx2), "avx2");
}

#[test]
fn simd_level_display_avx512() {
    assert_eq!(format!("{}", SimdLevel::Avx512), "avx512");
}

#[test]
fn simd_level_display_neon() {
    assert_eq!(format!("{}", SimdLevel::Neon), "neon");
}

#[test]
fn simd_level_display_matches_debug_lowercase() {
    // Display uses lowercase, Debug uses variant name — they should differ.
    let level = SimdLevel::Avx2;
    let display = format!("{level}");
    let debug = format!("{level:?}");
    assert_eq!(display, "avx2");
    assert_eq!(debug, "Avx2");
}

#[test]
fn all_types_implement_debug() {
    // Ensure Debug formatting doesn't panic for any type.
    let _ = format!("{:?}", probe_cpu());
    let _ = format!("{:?}", probe_gpu());
    let _ = format!("{:?}", probe_npu());
    let _ = format!("{:?}", probe_device());
    let _ = format!("{:?}", DeviceCapabilities::detect());
    let _ = format!("{:?}", detect_simd_level());
}

#[test]
fn device_capabilities_debug_has_simd_level() {
    let d = format!("{:?}", DeviceCapabilities::detect());
    assert!(d.contains("simd_level"));
    assert!(d.contains("cpu_rust"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. DeviceCapabilities::detect consistency
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn device_capabilities_cpu_rust_always_true() {
    assert!(DeviceCapabilities::detect().cpu_rust);
}

#[test]
fn device_capabilities_compiled_flags_match_feature_cfg() {
    let caps = DeviceCapabilities::detect();
    assert_eq!(caps.cuda_compiled, cfg!(any(feature = "gpu", feature = "cuda")));
    assert_eq!(caps.rocm_compiled, cfg!(any(feature = "gpu", feature = "rocm")));
    assert_eq!(caps.oneapi_compiled, cfg!(feature = "oneapi"));
    assert_eq!(caps.npu_compiled, cfg!(feature = "npu"));
}

#[test]
fn device_capabilities_simd_level_matches_detect() {
    assert_eq!(DeviceCapabilities::detect().simd_level, detect_simd_level());
}

#[test]
fn device_capabilities_runtime_without_compiled_impossible() {
    let caps = DeviceCapabilities::detect();
    if !caps.cuda_compiled {
        assert!(!caps.cuda_runtime, "CUDA runtime without compilation");
    }
    if !caps.rocm_compiled {
        assert!(!caps.rocm_runtime, "ROCm runtime without compilation");
    }
    if !caps.oneapi_compiled {
        assert!(!caps.oneapi_runtime, "oneAPI runtime without compilation");
    }
    if !caps.npu_compiled {
        assert!(!caps.npu_runtime, "NPU runtime without compilation");
    }
}

#[test]
fn device_capabilities_idempotent() {
    let a = DeviceCapabilities::detect();
    let b = DeviceCapabilities::detect();
    assert_eq!(a, b);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compile-time feature flag functions
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn compiled_functions_return_bools() {
    let _: bool = gpu_compiled();
    let _: bool = npu_compiled();
    let _: bool = oneapi_compiled();
    let _: bool = vulkan_compiled();
}

#[test]
fn runtime_functions_return_bools() {
    let _: bool = gpu_available_runtime();
    let _: bool = oneapi_available_runtime();
    let _: bool = vulkan_available_runtime();
}

#[test]
fn runtime_implies_compiled_for_all_backends() {
    if gpu_available_runtime() {
        assert!(gpu_compiled());
    }
    if oneapi_available_runtime() {
        assert!(oneapi_compiled());
    }
    if vulkan_available_runtime() {
        assert!(vulkan_compiled());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SimdLevel trait impls (from bitnet-common)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn simd_level_copy_semantics() {
    let a = SimdLevel::Avx2;
    let b = a; // Copy
    assert_eq!(a, b);
}

#[test]
fn simd_level_hash_consistent_with_eq() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(SimdLevel::Scalar);
    set.insert(SimdLevel::Avx2);
    set.insert(SimdLevel::Avx2); // duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn simd_level_ord_consistent_with_variant_order() {
    // The derive(Ord) uses variant declaration order: Scalar, Neon, Sse42, Avx2, Avx512.
    assert!(SimdLevel::Scalar < SimdLevel::Neon);
    assert!(SimdLevel::Neon < SimdLevel::Sse42);
    assert!(SimdLevel::Sse42 < SimdLevel::Avx2);
    assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
}
