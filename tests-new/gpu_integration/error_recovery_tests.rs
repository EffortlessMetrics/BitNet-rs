//! Graceful degradation integration tests.
//!
//! Validates that GPU errors, kernel failures, and device-lost scenarios
//! are handled gracefully without panics and with informative errors.

use std::sync::Arc;

use bitnet_common::{ConcreteTensor, Device, KernelBackend, KernelCapabilities};
use bitnet_inference::{Backend, CpuBackend, GpuBackend, KVCache};
use bitnet_models::Model;

use super::MockModel;

// ── GPU → CPU fallback ─────────────────────────────────────────────────

#[test]
fn test_gpu_backend_unavailable_falls_back() {
    // Given: GPU is not compiled in (feature gate)
    let model = Arc::new(MockModel::new());

    // When: trying to create GPU backend
    let gpu_result = GpuBackend::new(model.clone(), Device::Cuda(0));

    // Then: on CPU-only builds this must fail; CPU fallback works
    if !GpuBackend::is_available() {
        assert!(gpu_result.is_err(), "GpuBackend should fail without GPU");
        // CPU fallback must succeed
        let cpu = CpuBackend::new(model).unwrap();
        assert_eq!(cpu.backend_type(), "cpu");
    }
    // If GPU *is* compiled, both paths are valid — no assertion needed.
}

#[tokio::test(flavor = "multi_thread")]
async fn test_model_forward_failure_propagation() {
    // Given: a model that always fails
    let model = Arc::new(MockModel::with_failure());
    let backend = CpuBackend::new(model).unwrap();
    let input = ConcreteTensor::mock(vec![1, 8]);
    let mut cache = KVCache::new(Default::default()).unwrap();

    // When: running forward
    let result = backend.forward(&input, &mut cache).await;

    // Then: error propagates cleanly (no panic)
    assert!(result.is_err(), "forward should propagate model error");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Mock model failure"), "error should contain root cause: {err_msg}",);
}

#[test]
fn test_kernel_compilation_failure_recovery() {
    // Given: capabilities where nothing is available
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

    // When: requesting the best kernel backend
    let best = caps.best_available();

    // Then: returns None (no panic)
    assert_eq!(best, None, "no providers → None, not a panic");
}

#[test]
fn test_device_lost_recovery() {
    // Given: a device that reports as unavailable at runtime
    let caps = KernelCapabilities {
        cpu_rust: true,
        cuda_compiled: true,
        cuda_runtime: false, // "device lost" — compiled but runtime gone
        hip_compiled: false,
        hip_runtime: false,
        oneapi_compiled: false,
        oneapi_runtime: false,
        cpp_ffi: false,
        simd_level: bitnet_common::SimdLevel::Scalar,
    };

    // When: selecting best
    let best = caps.best_available();

    // Then: falls back to CPU instead of panicking
    assert_eq!(best, Some(KernelBackend::CpuRust), "device-lost should fall back to CPU");
}

#[test]
fn test_invalid_device_selection_error() {
    // When: creating a GPU backend with CPU device
    let model = Arc::new(MockModel::new());
    let result = GpuBackend::new(model, Device::Cpu);

    // Then: returns error (not panic)
    assert!(result.is_err(), "GPU backend with CPU device should return an error");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_warmup_failure_does_not_crash() {
    // Given: a failing model used in warmup
    let model = Arc::new(MockModel::with_failure());
    let backend = CpuBackend::new(model).unwrap();

    // When: warming up
    let result = backend.warmup().await;

    // Then: failure surfaces as error, not panic
    assert!(result.is_err(), "warmup with failing model should error");
}

#[test]
fn test_embed_failure_propagation() {
    // Given: a model that fails embed
    let model = MockModel::with_failure();

    // When: embedding tokens
    let result = model.embed(&[1, 2, 3]);

    // Then: error propagates
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Mock embed failure"), "unexpected: {msg}");
}

#[test]
fn test_logits_failure_propagation() {
    // Given: a model that fails logits
    let model = MockModel::with_failure();
    let hidden = ConcreteTensor::mock(vec![1, 4096]);

    // When: computing logits
    let result = model.logits(&hidden);

    // Then: error propagates
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Mock logits failure"), "unexpected: {msg}");
}
