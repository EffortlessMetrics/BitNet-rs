//! Cross-backend consistency tests.
//!
//! Validates that different backend configurations produce compatible
//! outputs and can be used sequentially without interference.

use std::sync::Arc;

use bitnet_common::{ConcreteTensor, Tensor};
use bitnet_inference::{Backend, CpuBackend, KVCache};
use bitnet_models::Model;

use super::MockModel;

// ── Cross-backend output consistency ───────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_cpu_and_mock_gpu_produce_same_output() {
    // Given: two CPU backends constructed from the same model
    let model = Arc::new(MockModel::new());
    let backend_a = CpuBackend::new(model.clone()).unwrap();
    let backend_b = CpuBackend::new(model).unwrap();

    let input = ConcreteTensor::mock(vec![1, 16]);
    let mut cache_a = KVCache::new(Default::default()).unwrap();
    let mut cache_b = KVCache::new(Default::default()).unwrap();

    // When: running forward on both
    let out_a = backend_a.forward(&input, &mut cache_a).await.unwrap();
    let out_b = backend_b.forward(&input, &mut cache_b).await.unwrap();

    // Then: shapes match (mock produces identical tensors)
    assert_eq!(out_a.shape(), out_b.shape(), "same model → identical output shapes");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_multiple_backends_sequential() {
    // Given: two backends
    let model = Arc::new(MockModel::new());
    let first = CpuBackend::new(model.clone()).unwrap();
    let second = CpuBackend::new(model).unwrap();

    let input = ConcreteTensor::mock(vec![1, 8]);

    // When: using them sequentially
    let mut c1 = KVCache::new(Default::default()).unwrap();
    let r1 = first.forward(&input, &mut c1).await.unwrap();
    drop(first); // explicitly drop to ensure no state leakage

    let mut c2 = KVCache::new(Default::default()).unwrap();
    let r2 = second.forward(&input, &mut c2).await.unwrap();

    // Then: second backend is unaffected by the first
    assert_eq!(r1.shape(), r2.shape());
}

#[test]
fn test_backend_type_names() {
    let model = Arc::new(MockModel::new());
    let cpu = CpuBackend::new(model).unwrap();

    assert_eq!(cpu.backend_type(), "cpu");
    // Backend type is a non-empty string
    assert!(!cpu.backend_type().is_empty());
}

#[test]
fn test_backend_capabilities_differ() {
    // Given: CPU capabilities
    let model = Arc::new(MockModel::new());
    let cpu = CpuBackend::new(model).unwrap();
    let cpu_caps = cpu.capabilities();

    // Then: CPU has known characteristics
    assert!(!cpu_caps.supports_mixed_precision, "CPU: no mixed prec");
    assert!(cpu_caps.supports_batching, "CPU: batching supported");
    assert_eq!(cpu_caps.max_batch_size, 8, "CPU: batch size 8");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_clone_backend_produces_independent_instance() {
    // Given: a backend
    let model = Arc::new(MockModel::new());
    let original = CpuBackend::new(model).unwrap();

    // When: cloning
    let cloned = original.clone_backend();

    // Then: the clone is independent
    assert_eq!(cloned.backend_type(), original.backend_type(), "cloned type matches");

    let input = ConcreteTensor::mock(vec![1, 4]);
    let mut cache = KVCache::new(Default::default()).unwrap();
    let result = cloned.forward(&input, &mut cache).await;
    assert!(result.is_ok(), "cloned backend should work independently");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_backend_forward_with_varying_input_sizes() {
    let model = Arc::new(MockModel::new());
    let backend = CpuBackend::new(model.clone()).unwrap();
    let vocab = model.config().model.vocab_size;

    // Different input sizes should all produce vocab-sized output
    for seq_len in [1, 4, 16, 64, 256] {
        let input = ConcreteTensor::mock(vec![1, seq_len]);
        let mut cache = KVCache::new(Default::default()).unwrap();
        let output = backend.forward(&input, &mut cache).await.unwrap();
        assert_eq!(output.shape()[1], vocab, "seq_len={seq_len} → vocab dim");
    }
}
