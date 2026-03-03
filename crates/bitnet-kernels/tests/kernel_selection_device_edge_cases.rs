//! Edge-case tests for kernel selection, device features, and Intel GPU detection.
//!
//! Tests cover: KernelManager (creation, selection, listing, reset),
//! select_cpu_kernel, compile-time feature checks (gpu_compiled, hip_compiled,
//! oneapi_compiled, opencl_compiled), gpu_available_runtime on CPU-only builds,
//! device_capability_summary, detect_simd_level, current_kernel_capabilities,
//! IntelGpuInfo defaults, and intel_gpu_status_string.

use bitnet_kernels::KernelProvider;
use bitnet_kernels::device_features;

// ---------------------------------------------------------------------------
// KernelManager
// ---------------------------------------------------------------------------

#[test]
fn kernel_manager_new() {
    let km = bitnet_kernels::KernelManager::new();
    // Should have at least the fallback provider
    let providers = km.list_available_providers();
    assert!(!providers.is_empty());
}

#[test]
fn kernel_manager_has_fallback() {
    let km = bitnet_kernels::KernelManager::new();
    let providers = km.list_available_providers();
    assert!(
        providers.iter().any(|p| p.to_lowercase().contains("fallback")),
        "Fallback kernel should always be available, got: {:?}",
        providers
    );
}

#[test]
fn kernel_manager_select_best() {
    let km = bitnet_kernels::KernelManager::new();
    let best = km.select_best();
    assert!(best.is_ok());
    let provider = best.unwrap();
    assert!(!provider.name().is_empty());
}

#[test]
fn kernel_manager_selected_provider_name_before_select() {
    let km = bitnet_kernels::KernelManager::new();
    // Before selecting, the name may or may not be set (auto-selects in new())
    let _name = km.selected_provider_name();
    // Just shouldn't panic
}

#[test]
fn kernel_manager_list_providers_nonempty() {
    let km = bitnet_kernels::KernelManager::new();
    let providers = km.list_available_providers();
    assert!(providers.len() >= 1, "Should have at least fallback");
}

#[test]
fn kernel_manager_select_best_twice_consistent() {
    let km = bitnet_kernels::KernelManager::new();
    let first = km.select_best().unwrap().name();
    let second = km.select_best().unwrap().name();
    assert_eq!(first, second, "Multiple selections should be consistent");
}

// ---------------------------------------------------------------------------
// select_cpu_kernel
// ---------------------------------------------------------------------------

#[test]
fn select_cpu_kernel_returns_provider() {
    let kernel = bitnet_kernels::select_cpu_kernel();
    assert!(kernel.is_ok());
    let provider = kernel.unwrap();
    assert!(provider.is_available());
    assert!(!provider.name().is_empty());
}

// ---------------------------------------------------------------------------
// Compile-time feature checks
// ---------------------------------------------------------------------------

#[test]
fn gpu_compiled_is_bool() {
    // On CPU-only build, should be false; on GPU build, true
    let _val: bool = device_features::gpu_compiled();
}

#[test]
fn hip_compiled_is_bool() {
    let _val: bool = device_features::hip_compiled();
}

#[test]
fn oneapi_compiled_is_bool() {
    let _val: bool = device_features::oneapi_compiled();
}

#[test]
fn opencl_compiled_is_bool() {
    let _val: bool = device_features::opencl_compiled();
}

// ---------------------------------------------------------------------------
// gpu_available_runtime (CPU-only build → always false)
// ---------------------------------------------------------------------------

#[test]
fn gpu_available_runtime_cpu_only() {
    // When built with --features cpu (no gpu), this should be false
    if !device_features::gpu_compiled() {
        assert!(!device_features::gpu_available_runtime());
    }
}

#[test]
fn oneapi_available_runtime_no_feature() {
    // If oneapi not compiled, should be false
    if !device_features::oneapi_compiled() {
        assert!(!device_features::oneapi_available_runtime());
    }
}

#[test]
fn opencl_available_runtime_no_feature() {
    // If opencl not compiled, should be false
    if !device_features::opencl_compiled() {
        assert!(!device_features::opencl_available_runtime());
    }
}

// ---------------------------------------------------------------------------
// device_capability_summary
// ---------------------------------------------------------------------------

#[test]
fn device_capability_summary_contains_cpu() {
    let summary = device_features::device_capability_summary();
    assert!(summary.contains("CPU"));
    assert!(summary.contains("Device Capabilities"));
}

#[test]
fn device_capability_summary_not_empty() {
    let summary = device_features::device_capability_summary();
    assert!(summary.len() > 20);
}

// ---------------------------------------------------------------------------
// detect_simd_level
// ---------------------------------------------------------------------------

#[test]
fn detect_simd_level_returns_valid() {
    let level = device_features::detect_simd_level();
    let dbg = format!("{level:?}");
    assert!(!dbg.is_empty());
}

// ---------------------------------------------------------------------------
// current_kernel_capabilities
// ---------------------------------------------------------------------------

#[test]
fn current_kernel_capabilities_cpu_rust() {
    let caps = device_features::current_kernel_capabilities();
    // When built with cpu feature, cpu_rust should be true
    if cfg!(feature = "cpu") {
        assert!(caps.cpu_rust);
    }
}

#[test]
fn current_kernel_capabilities_consistency() {
    let caps = device_features::current_kernel_capabilities();
    // cuda_compiled should match gpu_compiled()
    assert_eq!(caps.cuda_compiled, device_features::gpu_compiled());
    assert_eq!(caps.hip_compiled, device_features::hip_compiled());
    assert_eq!(caps.oneapi_compiled, device_features::oneapi_compiled());
    assert_eq!(caps.opencl_compiled, device_features::opencl_compiled());
}

#[test]
fn current_kernel_capabilities_debug() {
    let caps = device_features::current_kernel_capabilities();
    let dbg = format!("{caps:?}");
    assert!(dbg.contains("cpu_rust"));
}

// ---------------------------------------------------------------------------
// IntelGpuInfo
// ---------------------------------------------------------------------------

#[test]
fn intel_gpu_info_default() {
    let info = device_features::IntelGpuInfo::default();
    assert!(!info.detected);
    assert!(info.device_name.is_empty());
    assert_eq!(info.memory_bytes, 0);
    assert_eq!(info.compute_units, 0);
}

#[test]
fn intel_gpu_info_clone() {
    let info = device_features::IntelGpuInfo {
        detected: true,
        device_name: "Test GPU".into(),
        driver_version: "1.0".into(),
        opencl_version: "3.0".into(),
        memory_bytes: 1024,
        compute_units: 8,
        max_work_group_size: 256,
        level_zero_available: false,
    };
    let cloned = info.clone();
    assert_eq!(cloned.device_name, "Test GPU");
    assert_eq!(cloned.compute_units, 8);
}

#[test]
fn intel_gpu_info_debug() {
    let info = device_features::IntelGpuInfo::default();
    let dbg = format!("{info:?}");
    assert!(dbg.contains("IntelGpuInfo"));
}

// ---------------------------------------------------------------------------
// intel_gpu_status_string
// ---------------------------------------------------------------------------

#[test]
fn intel_gpu_status_string_not_empty() {
    let status = device_features::intel_gpu_status_string();
    assert!(!status.is_empty());
    // On most machines without Intel GPU, should say "not detected"
    // (Unless BITNET_GPU_FAKE is set, which we don't set here)
}

// ---------------------------------------------------------------------------
// FallbackKernel
// ---------------------------------------------------------------------------

#[test]
fn fallback_kernel_is_available() {
    let kernel = bitnet_kernels::FallbackKernel;
    assert!(kernel.is_available());
}

#[test]
fn fallback_kernel_name() {
    let kernel = bitnet_kernels::FallbackKernel;
    let name = kernel.name();
    assert!(!name.is_empty());
    assert!(
        name.to_lowercase().contains("fallback") || name.to_lowercase().contains("scalar"),
        "Expected fallback/scalar in name, got: {}",
        name
    );
}

// ---------------------------------------------------------------------------
// select_gpu_kernel (CPU-only: should fail)
// ---------------------------------------------------------------------------

#[test]
fn select_gpu_kernel_without_gpu_feature() {
    if !device_features::gpu_compiled() {
        let result = bitnet_kernels::select_gpu_kernel(0);
        assert!(result.is_err());
    }
}

// ---------------------------------------------------------------------------
// select_npu_kernel (no NPU: should fail)
// ---------------------------------------------------------------------------

#[test]
fn select_npu_kernel_without_npu() {
    let result = bitnet_kernels::select_npu_kernel();
    // NPU is not available on standard hardware
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// select_rocm_kernel (no ROCm: should fail)
// ---------------------------------------------------------------------------

#[test]
fn select_rocm_kernel_without_hip() {
    if !device_features::hip_compiled() {
        let result = bitnet_kernels::select_rocm_kernel();
        assert!(result.is_err());
    }
}
