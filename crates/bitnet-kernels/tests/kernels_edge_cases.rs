//! Edge-case tests for bitnet-kernels public API (CPU-only, no feature gates).

use bitnet_kernels::cpu::FallbackKernel;
use bitnet_kernels::device_features;
use bitnet_kernels::{KernelManager, KernelProvider, select_cpu_kernel, select_gpu_kernel};

// ---------------------------------------------------------------------------
// KernelManager
// ---------------------------------------------------------------------------

#[test]
fn kernel_manager_new_returns_instance() {
    let _km = KernelManager::new();
}

#[test]
fn kernel_manager_default_returns_instance() {
    let _km = KernelManager::default();
}

#[test]
fn kernel_manager_select_best_succeeds() {
    let km = KernelManager::new();
    let provider = km.select_best();
    assert!(provider.is_ok(), "should always have at least fallback");
}

#[test]
fn kernel_manager_selected_provider_name_after_select() {
    let km = KernelManager::new();
    let _ = km.select_best();
    let name = km.selected_provider_name();
    assert!(name.is_some());
    assert!(!name.unwrap().is_empty());
}

#[test]
fn kernel_manager_selected_provider_name_before_select() {
    let km = KernelManager::new();
    // Before select_best, no selection has been made
    let name = km.selected_provider_name();
    assert!(name.is_none());
}

#[test]
fn kernel_manager_list_available_providers_non_empty() {
    let km = KernelManager::new();
    let providers = km.list_available_providers();
    assert!(!providers.is_empty(), "should have at least FallbackKernel");
    assert!(providers.contains(&"FallbackCPU"));
}

#[test]
fn kernel_manager_select_best_is_cached() {
    let km = KernelManager::new();
    let p1 = km.select_best().unwrap().name();
    let p2 = km.select_best().unwrap().name();
    assert_eq!(p1, p2, "select_best should be cached and deterministic");
}

// ---------------------------------------------------------------------------
// FallbackKernel
// ---------------------------------------------------------------------------

#[test]
fn fallback_kernel_name() {
    let fk = FallbackKernel;
    assert_eq!(fk.name(), "FallbackCPU");
}

#[test]
fn fallback_kernel_is_available() {
    let fk = FallbackKernel;
    assert!(fk.is_available(), "FallbackCPU should always be available");
}

#[test]
fn fallback_kernel_matmul_trivial() {
    let fk = FallbackKernel;
    let a: Vec<i8> = vec![1, 0, -1, 1];
    let b: Vec<u8> = vec![1, 2, 3, 4];
    let mut c = vec![0.0f32; 4];
    // 2x2 * 2x2 matmul
    let result = fk.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    assert!(result.is_ok());
}

#[test]
fn fallback_kernel_matmul_zero_dimensions() {
    let fk = FallbackKernel;
    let a: Vec<i8> = vec![];
    let b: Vec<u8> = vec![];
    let mut c: Vec<f32> = vec![];
    let result = fk.matmul_i2s(&a, &b, &mut c, 0, 0, 0);
    // Should handle gracefully (either Ok or error, not panic)
    let _ = result;
}

#[test]
fn fallback_kernel_matmul_single_element() {
    let fk = FallbackKernel;
    let a: Vec<i8> = vec![2];
    let b: Vec<u8> = vec![3];
    let mut c = vec![0.0f32; 1];
    let result = fk.matmul_i2s(&a, &b, &mut c, 1, 1, 1);
    assert!(result.is_ok());
}

// ---------------------------------------------------------------------------
// select_cpu_kernel
// ---------------------------------------------------------------------------

#[test]
fn select_cpu_kernel_returns_provider() {
    let kernel = select_cpu_kernel();
    assert!(kernel.is_ok(), "CPU kernel selection should always succeed");
}

#[test]
fn select_cpu_kernel_provider_is_available() {
    let kernel = select_cpu_kernel().unwrap();
    assert!(kernel.is_available());
}

#[test]
fn select_cpu_kernel_provider_has_name() {
    let kernel = select_cpu_kernel().unwrap();
    assert!(!kernel.name().is_empty());
}

// ---------------------------------------------------------------------------
// select_gpu_kernel (without GPU features)
// ---------------------------------------------------------------------------

#[test]
fn select_gpu_kernel_returns_error_without_gpu() {
    // Without GPU features, this should return an error
    let result = select_gpu_kernel(0);
    if !device_features::gpu_compiled() {
        assert!(result.is_err());
    }
}

// ---------------------------------------------------------------------------
// device_features
// ---------------------------------------------------------------------------

#[test]
fn device_features_gpu_compiled_consistent() {
    // Should return the same value every time
    let a = device_features::gpu_compiled();
    let b = device_features::gpu_compiled();
    assert_eq!(a, b);
}

#[test]
fn device_features_hip_compiled_consistent() {
    let a = device_features::hip_compiled();
    let b = device_features::hip_compiled();
    assert_eq!(a, b);
}

#[test]
fn device_features_current_kernel_capabilities() {
    let caps = device_features::current_kernel_capabilities();
    // Should always have cpu_rust
    assert!(caps.cpu_rust);
}

// ---------------------------------------------------------------------------
// KernelProvider trait on FallbackKernel
// ---------------------------------------------------------------------------

#[test]
fn fallback_kernel_quantize_i2s() {
    use bitnet_common::QuantizationType;
    let fk = FallbackKernel;
    let input = vec![0.5f32, -0.3, 0.8, -0.1];
    let mut output = vec![0u8; 4];
    let mut scales = vec![0.0f32; 1];
    let result = fk.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    // May succeed or return unsupported — just shouldn't panic
    let _ = result;
}

// ---------------------------------------------------------------------------
// Multiple KernelManagers are independent
// ---------------------------------------------------------------------------

#[test]
fn multiple_kernel_managers_independent() {
    let km1 = KernelManager::new();
    let km2 = KernelManager::new();
    let _ = km1.select_best();
    // km2 should not be affected by km1's selection
    assert!(km2.selected_provider_name().is_none());
}
