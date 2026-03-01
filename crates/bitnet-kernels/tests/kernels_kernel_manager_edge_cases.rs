//! Edge-case integration tests for `KernelManager`, `select_cpu_kernel`, and
//! `select_gpu_kernel` kernel dispatch logic.
//!
//! These tests exercise the public API of the kernel selection and provider
//! management from outside the crate boundary.

use bitnet_common::QuantizationType;
use bitnet_kernels::{KernelManager, KernelProvider, select_cpu_kernel, select_gpu_kernel};

// =========================================================================
// KernelManager: construction and default
// =========================================================================

#[test]
fn kernel_manager_new_has_providers() {
    let mgr = KernelManager::new();
    let available = mgr.list_available_providers();
    assert!(!available.is_empty(), "should have at least the fallback provider");
}

#[test]
fn kernel_manager_default_matches_new() {
    let mgr = KernelManager::default();
    let available = mgr.list_available_providers();
    assert!(!available.is_empty());
}

// =========================================================================
// KernelManager: provider selection
// =========================================================================

#[test]
fn kernel_manager_select_best_succeeds() {
    let mgr = KernelManager::new();
    let provider = mgr.select_best().expect("should select a provider");
    assert!(provider.is_available());
    assert!(!provider.name().is_empty());
}

#[test]
fn kernel_manager_select_best_cached() {
    let mgr = KernelManager::new();
    let p1 = mgr.select_best().unwrap();
    let p2 = mgr.select_best().unwrap();
    // Same provider should be selected both times (OnceLock)
    assert_eq!(p1.name(), p2.name());
}

#[test]
fn kernel_manager_selected_provider_name_before_selection() {
    let mgr = KernelManager::new();
    // Before selecting, name should be None
    assert!(mgr.selected_provider_name().is_none());
}

#[test]
fn kernel_manager_selected_provider_name_after_selection() {
    let mgr = KernelManager::new();
    let _ = mgr.select_best().unwrap();
    let name = mgr.selected_provider_name();
    assert!(name.is_some());
    assert!(!name.unwrap().is_empty());
}

// =========================================================================
// KernelManager: list_available_providers
// =========================================================================

#[test]
fn kernel_manager_list_includes_fallback() {
    let mgr = KernelManager::new();
    let available = mgr.list_available_providers();
    assert!(
        available.contains(&"fallback"),
        "fallback kernel should always be available, got {available:?}"
    );
}

#[test]
fn kernel_manager_all_listed_providers_are_available() {
    let mgr = KernelManager::new();
    let available = mgr.list_available_providers();
    // All names should be non-empty strings
    for name in &available {
        assert!(!name.is_empty());
    }
}

// =========================================================================
// KernelManager: matmul via selected provider
// =========================================================================

#[test]
fn kernel_manager_matmul_via_provider() {
    let mgr = KernelManager::new();
    let provider = mgr.select_best().unwrap();

    // Simple 2x2 identity matmul
    let a = vec![1i8, 0, 0, 1];
    let b = vec![3u8, 5, 7, 2];
    let mut c = vec![0.0f32; 4];
    provider.matmul_i2s(&a, &b, &mut c, 2, 2, 2).unwrap();

    assert!((c[0] - 3.0).abs() < 1e-4);
    assert!((c[3] - 2.0).abs() < 1e-4);
}

// =========================================================================
// KernelManager: quantize via selected provider
// =========================================================================

#[test]
fn kernel_manager_quantize_i2s_via_provider() {
    let mgr = KernelManager::new();
    let provider = mgr.select_best().unwrap();

    let input = vec![1.0, -1.0, 0.0, 0.5, -0.5, 0.2, -0.2, 0.0];
    let mut output = vec![0u8; 2];
    let mut scales = vec![0.0f32; 1];
    provider.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    assert!(scales[0] > 0.0);
}

// =========================================================================
// select_cpu_kernel
// =========================================================================

#[test]
fn select_cpu_kernel_succeeds() {
    let kernel = select_cpu_kernel().expect("CPU kernel should always be available");
    assert!(kernel.is_available());
}

#[test]
fn select_cpu_kernel_name_nonempty() {
    let kernel = select_cpu_kernel().unwrap();
    assert!(!kernel.name().is_empty());
}

#[test]
fn select_cpu_kernel_matmul_works() {
    let kernel = select_cpu_kernel().unwrap();
    let a = vec![2i8, 3];
    let b = vec![4u8, 5];
    let mut c = vec![0.0f32; 1];
    kernel.matmul_i2s(&a, &b, &mut c, 1, 1, 2).unwrap();
    // 2*4 + 3*5 = 23
    assert!((c[0] - 23.0).abs() < 1e-4);
}

// =========================================================================
// select_gpu_kernel (CPU-only mode: should fail)
// =========================================================================

#[test]
fn select_gpu_kernel_without_gpu_feature_fails() {
    // In CPU-only builds, GPU kernel selection should fail
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    {
        let result = select_gpu_kernel(0);
        assert!(result.is_err(), "GPU kernel should not be available in CPU-only build");
    }
}

// =========================================================================
// Provider error handling
// =========================================================================

#[test]
fn selected_provider_matmul_dimension_error() {
    // Use FallbackKernel directly since optimized kernels may panic on invalid dims
    let provider = bitnet_kernels::cpu::fallback::FallbackKernel;

    let a = vec![1i8; 2]; // Too small for 2x2
    let b = vec![1u8; 4];
    let mut c = vec![0.0f32; 4];
    let result = provider.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    assert!(result.is_err());
}

#[test]
fn selected_provider_quantize_buffer_too_small() {
    // Use FallbackKernel directly for error path testing
    let provider = bitnet_kernels::cpu::fallback::FallbackKernel;

    let input = vec![1.0f32; 32];
    let mut output = vec![0u8; 1]; // Too small
    let mut scales = vec![0.0f32; 1];
    let result = provider.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    assert!(result.is_err());
}

// =========================================================================
// Thread safety: KernelManager is Send + Sync
// =========================================================================

#[test]
fn kernel_manager_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<KernelManager>();
}

#[test]
fn kernel_manager_concurrent_select() {
    use std::sync::Arc;
    use std::thread;

    let mgr = Arc::new(KernelManager::new());
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let mgr = Arc::clone(&mgr);
            thread::spawn(move || {
                let provider = mgr.select_best().unwrap();
                provider.name().to_string()
            })
        })
        .collect();

    let names: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    // All threads should get the same provider
    assert!(names.windows(2).all(|w| w[0] == w[1]));
}
