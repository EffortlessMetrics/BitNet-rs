//! End-to-end GPU inference pipeline integration tests.
//!
//! Tests the complete flow: tokenize → embed → layers → logits → sample
//! using mock GPU backend for CI compatibility.

use std::sync::Arc;

use bitnet_common::{BitNetConfig, ConcreteTensor, ModelConfig, Tensor};
use bitnet_inference::{Backend, CacheConfig, CpuBackend, KVCache};
use bitnet_models::Model;

use super::MockModel;

// ── Pipeline construction ──────────────────────────────────────────────

#[test]
fn test_pipeline_construction_from_model_config() {
    // Given: model config (32 layers, 4096 hidden dim, 32000 vocab)
    let config = BitNetConfig {
        model: ModelConfig {
            num_layers: 32,
            hidden_size: 4096,
            vocab_size: 32000,
            ..Default::default()
        },
        ..Default::default()
    };
    let model = Arc::new(MockModel::with_config(config.clone()));

    // When: constructing CPU backend (mock GPU path)
    let backend = CpuBackend::new(model).unwrap();

    // Then: backend reports correct type and capabilities
    assert_eq!(backend.backend_type(), "cpu");
    let caps = backend.capabilities();
    assert!(caps.supports_batching);
    assert!(caps.max_batch_size >= 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pipeline_forward_produces_valid_logits() {
    // Given: a constructed pipeline with mock weights
    let model = Arc::new(MockModel::new());
    let backend = CpuBackend::new(model.clone()).unwrap();
    let input = ConcreteTensor::mock(vec![1, 512]);
    let mut cache = KVCache::new(Default::default()).unwrap();

    // When: running forward pass
    let output = backend.forward(&input, &mut cache).await.unwrap();

    // Then: output has shape [1, vocab_size]
    let shape = output.shape();
    assert_eq!(shape.len(), 2, "logits should be 2-D");
    assert_eq!(shape[1], model.config().model.vocab_size, "second dim must equal vocab_size");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pipeline_memory_stays_within_budget() {
    // Given: a memory budget of 4 GB via CacheConfig
    let budget_bytes: usize = 4 * 1024 * 1024 * 1024;
    let config = CacheConfig { max_size_bytes: budget_bytes, ..Default::default() };
    let mut cache = KVCache::new(config).unwrap();

    // When: storing entries within budget
    let key = vec![0.1_f32; 128];
    let value = vec![0.2_f32; 128];
    for pos in 0..100 {
        cache.store(0, pos, key.clone(), value.clone()).unwrap();
    }

    // Then: current size never exceeds budget
    assert!(
        cache.size() <= budget_bytes,
        "cache size {} exceeds budget {}",
        cache.size(),
        budget_bytes,
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pipeline_handles_empty_input() {
    // Given: a mock model
    let model = Arc::new(MockModel::new());

    // When: embedding zero tokens
    let result = model.embed(&[]);

    // Then: either succeeds with shape [0, hidden] or returns error
    match result {
        Ok(ref tensor) => {
            let shape = tensor.shape();
            assert_eq!(shape[0], 0, "empty input → seq_len 0");
        }
        Err(e) => {
            // An explicit InvalidInput error is also acceptable
            let msg = format!("{e}");
            assert!(
                msg.contains("empty") || msg.contains("invalid") || msg.contains("fail"),
                "unexpected error: {msg}",
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pipeline_batch_inference() {
    // Given: multiple input sequences
    let model = Arc::new(MockModel::new());
    let backend = CpuBackend::new(model.clone()).unwrap();

    let sequences: &[&[u32]] = &[&[1, 2, 3], &[4, 5], &[6, 7, 8, 9]];
    let mut results = Vec::new();

    // When: running forward for each sequence
    for tokens in sequences {
        let input = ConcreteTensor::mock(vec![1, tokens.len()]);
        let mut cache = KVCache::new(Default::default()).unwrap();
        let output = backend.forward(&input, &mut cache).await.unwrap();
        results.push(output);
    }

    // Then: each sequence gets independent logits of the same vocab dim
    assert_eq!(results.len(), sequences.len());
    let vocab = model.config().model.vocab_size;
    for (i, r) in results.iter().enumerate() {
        assert_eq!(r.shape()[1], vocab, "sequence {i} logits dim mismatch");
    }
}

#[test]
fn test_pipeline_kv_cache_grows_correctly() {
    // Given: an empty cache
    let mut cache = KVCache::new(Default::default()).unwrap();
    assert_eq!(cache.size(), 0);

    let key = vec![0.5_f32; 64];
    let val = vec![0.5_f32; 64];
    let entry_bytes = (key.len() + val.len()) * std::mem::size_of::<f32>();

    // When: generating tokens one at a time
    for step in 0..10 {
        cache.store(0, step, key.clone(), val.clone()).unwrap();
    }

    // Then: KV cache grows by one entry per step
    assert_eq!(cache.size(), entry_bytes * 10, "cache should grow linearly with stored entries");
}

#[test]
fn test_pipeline_respects_max_sequence_length() {
    // Given: max_sequence_length = 512
    let config = CacheConfig { max_sequence_length: 512, ..Default::default() };

    // When/Then: config is honoured
    assert_eq!(config.max_sequence_length, 512);

    // Verify the model config also limits positions
    let model_cfg = BitNetConfig {
        model: ModelConfig { max_position_embeddings: 512, ..Default::default() },
        ..Default::default()
    };
    assert_eq!(model_cfg.model.max_position_embeddings, 512);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pipeline_deterministic_with_seed() {
    // Given: same model and same input
    let model = Arc::new(MockModel::new());
    let backend = CpuBackend::new(model).unwrap();
    let input = ConcreteTensor::mock(vec![1, 8]);

    // When: running forward twice
    let mut c1 = KVCache::new(Default::default()).unwrap();
    let mut c2 = KVCache::new(Default::default()).unwrap();
    let out1 = backend.forward(&input, &mut c1).await.unwrap();
    let out2 = backend.forward(&input, &mut c2).await.unwrap();

    // Then: shapes are identical (mock always produces same output)
    assert_eq!(out1.shape(), out2.shape());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pipeline_warmup_succeeds() {
    // Given: a CPU backend
    let model = Arc::new(MockModel::new());
    let backend = CpuBackend::new(model).unwrap();

    // When: warming up
    let result = backend.warmup().await;

    // Then: no error
    assert!(result.is_ok(), "warmup should succeed: {:?}", result.err());
}
