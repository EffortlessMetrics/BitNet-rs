//! Integration tests for kernel dispatch and provider selection.
//!
//! Validates cross-crate interactions between `bitnet-kernels` and
//! `bitnet-common` for kernel management, provider selection, SIMD
//! capability reporting, and error propagation.

use bitnet_common::{BitNetError, KernelError, QuantizationType};
use bitnet_kernels::{FallbackKernel, KernelManager, KernelProvider};

// ── Provider enumeration ─────────────────────────────────────────────

#[test]
fn integration_kernel_manager_has_at_least_one_provider() {
    let manager = KernelManager::new();
    let providers = manager.list_available_providers();
    assert!(!providers.is_empty(), "KernelManager must always have at least one provider");
}

#[test]
fn integration_fallback_kernel_always_available() {
    let providers = KernelManager::new().list_available_providers();
    assert!(
        providers.contains(&"fallback"),
        "fallback kernel must always be listed; got: {providers:?}"
    );
}

#[test]
fn integration_select_best_returns_ok() {
    let manager = KernelManager::new();
    let best = manager.select_best();
    assert!(best.is_ok(), "select_best() should always succeed: {:?}", best.err());
}

#[test]
fn integration_selected_provider_name_after_select() {
    let manager = KernelManager::new();
    // Before selection, name may be None
    let _ = manager.select_best().unwrap();
    let name = manager.selected_provider_name();
    assert!(name.is_some(), "selected_provider_name must be Some after select_best()");
    assert!(!name.unwrap().is_empty(), "selected provider name must not be empty");
}

#[test]
fn integration_select_best_is_cached() {
    let manager = KernelManager::new();
    let first = manager.select_best().unwrap().name();
    let second = manager.select_best().unwrap().name();
    assert_eq!(first, second, "select_best() must return the same provider on repeated calls");
}

// ── Provider capabilities ────────────────────────────────────────────

#[test]
fn integration_provider_names_are_non_empty() {
    let manager = KernelManager::new();
    for name in manager.list_available_providers() {
        assert!(!name.is_empty(), "provider name must not be empty");
        assert!(name.is_ascii(), "provider name must be ASCII: {name}");
    }
}

#[test]
fn integration_available_providers_report_available() {
    let fallback = FallbackKernel;
    assert!(fallback.is_available(), "FallbackKernel::is_available() must return true");
}

#[test]
fn integration_fallback_kernel_name_is_fallback() {
    let fallback = FallbackKernel;
    assert_eq!(fallback.name(), "fallback");
}

#[test]
fn integration_provider_names_are_unique() {
    let providers = KernelManager::new().list_available_providers();
    let mut seen = std::collections::HashSet::new();
    for name in &providers {
        assert!(seen.insert(name), "duplicate provider name: {name}");
    }
}

// ── Kernel dispatch (matmul) ─────────────────────────────────────────

#[test]
fn integration_fallback_matmul_basic() {
    let kernel = FallbackKernel;
    // 1×1 matmul: [2] * [3] = [6.0]
    let a: Vec<i8> = vec![2];
    let b: Vec<u8> = vec![3];
    let mut c: Vec<f32> = vec![0.0];
    kernel.matmul_i2s(&a, &b, &mut c, 1, 1, 1).unwrap();
    assert!((c[0] - 6.0).abs() < f32::EPSILON);
}

#[test]
fn integration_fallback_matmul_2x2() {
    let kernel = FallbackKernel;
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // C = [[19, 22], [43, 50]]
    let a: Vec<i8> = vec![1, 2, 3, 4];
    let b: Vec<u8> = vec![5, 6, 7, 8];
    let mut c = vec![0.0f32; 4];
    kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2).unwrap();
    assert!((c[0] - 19.0).abs() < f32::EPSILON);
    assert!((c[1] - 22.0).abs() < f32::EPSILON);
    assert!((c[2] - 43.0).abs() < f32::EPSILON);
    assert!((c[3] - 50.0).abs() < f32::EPSILON);
}

#[test]
fn integration_dispatch_matmul_via_manager() {
    let manager = KernelManager::new();
    let provider = manager.select_best().unwrap();
    let a: Vec<i8> = vec![1, 0, 0, 1];
    let b: Vec<u8> = vec![2, 3, 4, 5];
    let mut c = vec![0.0f32; 4];
    provider.matmul_i2s(&a, &b, &mut c, 2, 2, 2).unwrap();
    // Identity-like A => C ≈ B rows rearranged
    assert!((c[0] - 2.0).abs() < f32::EPSILON);
    assert!((c[3] - 5.0).abs() < f32::EPSILON);
}

// ── Error handling ───────────────────────────────────────────────────

#[test]
fn integration_matmul_dimension_mismatch_a() {
    let kernel = FallbackKernel;
    let a: Vec<i8> = vec![1]; // too small for 2×2
    let b: Vec<u8> = vec![1, 2, 3, 4];
    let mut c = vec![0.0f32; 4];
    let err = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    assert!(err.is_err(), "dimension mismatch on A should fail");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("dimension") || msg.contains("mismatch"),
        "error should mention dimension: {msg}"
    );
}

#[test]
fn integration_matmul_dimension_mismatch_b() {
    let kernel = FallbackKernel;
    let a: Vec<i8> = vec![1, 2, 3, 4];
    let b: Vec<u8> = vec![1]; // too small
    let mut c = vec![0.0f32; 4];
    let err = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    assert!(err.is_err(), "dimension mismatch on B should fail");
}

#[test]
fn integration_matmul_dimension_mismatch_c() {
    let kernel = FallbackKernel;
    let a: Vec<i8> = vec![1, 2, 3, 4];
    let b: Vec<u8> = vec![5, 6, 7, 8];
    let mut c = vec![0.0f32; 1]; // too small
    let err = kernel.matmul_i2s(&a, &b, &mut c, 2, 2, 2);
    assert!(err.is_err(), "dimension mismatch on C should fail");
}

#[test]
fn integration_kernel_error_converts_to_bitnet_error() {
    let kernel_err = KernelError::NoProvider;
    let bitnet_err: BitNetError = kernel_err.into();
    let msg = format!("{bitnet_err}");
    assert!(
        msg.contains("Kernel") || msg.contains("kernel") || msg.contains("provider"),
        "BitNetError from KernelError should mention kernel: {msg}"
    );
}

#[test]
fn integration_unsupported_architecture_error_has_context() {
    let err = KernelError::UnsupportedArchitecture { arch: "mips64".to_string() };
    let msg = format!("{err}");
    assert!(msg.contains("mips64"), "error message should contain the architecture name: {msg}");
}

// ── Quantization dispatch ────────────────────────────────────────────

#[test]
fn integration_fallback_quantize_i2s() {
    let kernel = FallbackKernel;
    let input = vec![1.0f32, -1.0, 0.5, -0.5];
    let mut output = vec![0u8; 4];
    let mut scales = vec![0.0f32; 1];
    let result = kernel.quantize(&input, &mut output, &mut scales, QuantizationType::I2S);
    // Either succeeds or returns a well-formed error
    match result {
        Ok(()) => {
            // At least one byte should be non-zero after quantization
            assert!(output.iter().any(|&b| b != 0), "quantized output should have non-zero bytes");
        }
        Err(e) => {
            let msg = format!("{e}");
            assert!(!msg.is_empty(), "error from quantize must have a message");
        }
    }
}

#[test]
fn integration_select_cpu_kernel_returns_provider() {
    let provider = bitnet_kernels::select_cpu_kernel();
    assert!(provider.is_ok(), "select_cpu_kernel() must succeed: {:?}", provider.err());
    let p = provider.unwrap();
    assert!(!p.name().is_empty());
    assert!(p.is_available());
}
