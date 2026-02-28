//! Backend auto-selection integration tests.
//!
//! Validates that the backend selection logic correctly prioritises
//! GPU → CPU fallback and that `BackendStartupSummary` reports accurately.

use bitnet_common::{BackendStartupSummary, Device, KernelBackend, KernelCapabilities};
use bitnet_inference::backends::BackendCapabilities;

// ── Fallback / auto-select ─────────────────────────────────────────────

#[test]
fn test_auto_select_falls_back_to_cpu() {
    // Given: capabilities with no GPU runtime
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: bitnet_common::SimdLevel::Scalar,
    };

    // When: selecting best available backend
    let best = caps.best_available();

    // Then: CPU is selected
    assert_eq!(best, Some(KernelBackend::CpuRust));
}

#[test]
fn test_explicit_backend_selection() {
    // When: user requests specific device
    let cpu = Device::Cpu;
    let cuda0 = Device::Cuda(0);

    // Then: device variants are distinct
    assert!(cpu.is_cpu());
    assert!(!cpu.is_cuda());
    assert!(cuda0.is_cuda());
    assert!(!cuda0.is_cpu());
}

#[test]
fn test_backend_priority_ordering() {
    // Given: CUDA + HIP + OneAPI + CPU all available
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: true,
        hip_compiled: true,
        hip_runtime: true,
        oneapi_compiled: true,
        oneapi_runtime: true,
        cpp_ffi: true,
        simd_level: bitnet_common::SimdLevel::Avx2,
    };

    // When: selecting best
    let best = caps.best_available().unwrap();

    // Then: CUDA wins (highest priority)
    assert_eq!(best, KernelBackend::Cuda, "CUDA must have highest priority");
}

#[test]
fn test_hip_selected_when_cuda_unavailable() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: true,
        hip_runtime: true,
        oneapi_compiled: true,
        oneapi_runtime: true,
        cpp_ffi: false,
        simd_level: bitnet_common::SimdLevel::Scalar,
    };

    let best = caps.best_available().unwrap();
    assert_eq!(best, KernelBackend::Hip, "HIP should be selected over OneAPI");
}

#[test]
fn test_oneapi_selected_when_no_cuda_or_hip() {
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: true,
        oneapi_runtime: true,
        cpp_ffi: false,
        simd_level: bitnet_common::SimdLevel::Scalar,
    };

    let best = caps.best_available().unwrap();
    assert_eq!(best, KernelBackend::OneApi);
}

#[test]
fn test_backend_capabilities_reporting() {
    // When: querying default capabilities
    let caps = BackendCapabilities::default();

    // Then: sensible defaults
    assert!(!caps.supports_mixed_precision, "default has no mixed precision");
    assert!(caps.supports_batching, "batching should be on by default");
    assert_eq!(caps.max_batch_size, 1, "default batch size is 1");
    assert!(caps.memory_efficient, "default is memory efficient");
}

#[test]
fn test_backend_startup_summary_format() {
    // Given: a summary
    let summary = BackendStartupSummary::new("auto", vec!["cpu-rust".to_string()], "cpu-rust");

    // When: formatting for logs
    let line = summary.log_line();

    // Then: contains required fields
    assert!(line.contains("requested=auto"), "missing requested: {line}");
    assert!(line.contains("cpu-rust"), "missing detected backend: {line}");
    assert!(line.contains("selected=cpu-rust"), "missing selected: {line}");
}

#[test]
fn test_backend_startup_summary_multiple_detected() {
    let summary =
        BackendStartupSummary::new("gpu", vec!["cuda".to_string(), "cpu-rust".to_string()], "cuda");

    let line = summary.log_line();
    assert!(line.contains("cuda, cpu-rust"), "multi-backend: {line}");
}

#[test]
fn test_no_backends_available_returns_none() {
    let caps = KernelCapabilities {
        cpu_rust: false,
        cuda_compiled: false,
        cuda_runtime: false,
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: bitnet_common::SimdLevel::Scalar,
    };

    assert_eq!(caps.best_available(), None, "no backends → None");
}
