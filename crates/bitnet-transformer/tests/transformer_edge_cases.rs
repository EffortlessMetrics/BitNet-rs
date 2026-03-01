//! Edge-case tests for transformer layer components.
//!
//! Tests cover KVCache operations, RotaryEmbedding, config validation,
//! and TransformerModel construction on CPU device.

#![cfg(feature = "cpu")]

use bitnet_common::BitNetConfig;
use bitnet_transformer::{KVCache, LayerKVCache, RotaryEmbedding};
use candle_core::{DType, Device, Tensor};

fn small_config(num_heads: usize, num_kv_heads: usize, max_seq_len: usize) -> BitNetConfig {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_layers = 2;
    cfg.model.num_heads = num_heads;
    cfg.model.num_key_value_heads = num_kv_heads;
    cfg.model.hidden_size = num_heads * 4; // head_dim = 4
    cfg.model.max_position_embeddings = max_seq_len;
    cfg
}

// ── LayerKVCache ─────────────────────────────────────────────────────

#[test]
fn layer_kv_cache_new_empty() {
    // batch_size, n_kv_heads, max_seq_len, head_dim, device
    let cache = LayerKVCache::new(1, 4, 16, 8, &Device::Cpu);
    assert!(cache.is_ok());
}

#[test]
fn layer_kv_cache_append_and_clear() {
    let mut cache = LayerKVCache::new(1, 4, 16, 8, &Device::Cpu).unwrap();
    let k = Tensor::zeros(&[1, 4, 1, 8], DType::F32, &Device::Cpu).unwrap();
    let v = Tensor::zeros(&[1, 4, 1, 8], DType::F32, &Device::Cpu).unwrap();
    cache.append(&k, &v).unwrap();
    cache.clear();
}

#[test]
fn layer_kv_cache_multiple_appends() {
    let mut cache = LayerKVCache::new(1, 2, 64, 4, &Device::Cpu).unwrap();
    for _ in 0..5 {
        let k = Tensor::zeros(&[1, 2, 1, 4], DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros(&[1, 2, 1, 4], DType::F32, &Device::Cpu).unwrap();
        cache.append(&k, &v).unwrap();
    }
}

#[test]
fn layer_kv_cache_single_head() {
    let mut cache = LayerKVCache::new(1, 1, 32, 16, &Device::Cpu).unwrap();
    let k = Tensor::ones(&[1, 1, 1, 16], DType::F32, &Device::Cpu).unwrap();
    let v = Tensor::ones(&[1, 1, 1, 16], DType::F32, &Device::Cpu).unwrap();
    cache.append(&k, &v).unwrap();
}

// ── RotaryEmbedding ──────────────────────────────────────────────────

#[test]
fn rope_preserves_shape() {
    // dim, max_seq_len, rope_theta (Option<f32>), device
    let rope = RotaryEmbedding::new(64, 128, Some(10000.0), &Device::Cpu).unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[1, 8, 1, 64], &Device::Cpu).unwrap();
    let result = rope.apply(&x, 0).unwrap();
    assert_eq!(result.dims(), &[1, 8, 1, 64]);
}

#[test]
fn rope_different_positions() {
    let rope = RotaryEmbedding::new(32, 64, Some(10000.0), &Device::Cpu).unwrap();
    let x = Tensor::ones(&[1, 4, 1, 32], DType::F32, &Device::Cpu).unwrap();
    let pos0 = rope.apply(&x, 0).unwrap();
    let pos1 = rope.apply(&x, 1).unwrap();
    let diff =
        pos0.sub(&pos1).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(diff > 0.0, "Different positions should produce different embeddings");
}

#[test]
fn rope_position_zero_preserves_magnitude() {
    let rope = RotaryEmbedding::new(16, 32, Some(10000.0), &Device::Cpu).unwrap();
    let x = Tensor::ones(&[1, 2, 1, 16], DType::F32, &Device::Cpu).unwrap();
    let result = rope.apply(&x, 0).unwrap();
    let input_norm = x.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
    let result_norm = result.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
    assert!(
        (input_norm - result_norm).abs() < 0.1,
        "RoPE at pos 0 should preserve magnitude: in={input_norm}, out={result_norm}"
    );
}

#[test]
fn rope_small_head_dim() {
    let rope = RotaryEmbedding::new(4, 16, Some(10000.0), &Device::Cpu).unwrap();
    let x = Tensor::randn(0.0f32, 1.0, &[1, 1, 1, 4], &Device::Cpu).unwrap();
    let result = rope.apply(&x, 5).unwrap();
    assert_eq!(result.dims(), &[1, 1, 1, 4]);
}

#[test]
fn rope_default_theta() {
    // Pass None for theta to test default behavior
    let rope = RotaryEmbedding::new(32, 64, None, &Device::Cpu).unwrap();
    let x = Tensor::ones(&[1, 4, 1, 32], DType::F32, &Device::Cpu).unwrap();
    let result = rope.apply(&x, 0).unwrap();
    assert_eq!(result.dims(), &[1, 4, 1, 32]);
}

#[test]
fn rope_large_position() {
    let rope = RotaryEmbedding::new(16, 2048, Some(10000.0), &Device::Cpu).unwrap();
    let x = Tensor::ones(&[1, 2, 1, 16], DType::F32, &Device::Cpu).unwrap();
    // Should work at position near max_seq_len
    let result = rope.apply(&x, 2000).unwrap();
    assert_eq!(result.dims(), &[1, 2, 1, 16]);
}

// ── KVCache (full model cache) ───────────────────────────────────────

#[test]
fn kv_cache_clear_all_layers() {
    let config = small_config(4, 2, 32);
    let mut cache = KVCache::new(&config, 1, &Device::Cpu).unwrap();
    cache.clear();
}

#[test]
fn kv_cache_layer_access() {
    let config = small_config(4, 2, 32);
    let mut cache = KVCache::new(&config, 1, &Device::Cpu).unwrap();
    assert!(cache.layer_mut(0).is_some());
    assert!(cache.layer_mut(1).is_some());
    assert!(cache.layer_mut(2).is_none()); // only 2 layers
}

#[test]
fn kv_cache_gqa_config() {
    // Grouped query attention: fewer KV heads than query heads
    let config = small_config(8, 2, 64);
    let cache = KVCache::new(&config, 1, &Device::Cpu);
    assert!(cache.is_ok(), "GQA config (8 heads, 2 KV heads) should work");
}

#[test]
fn kv_cache_equal_heads() {
    // Standard MHA: same number of KV and query heads
    let config = small_config(4, 4, 32);
    let cache = KVCache::new(&config, 1, &Device::Cpu);
    assert!(cache.is_ok(), "MHA config (equal heads) should work");
}

#[test]
fn kv_cache_single_kv_head() {
    // Multi-query attention: single KV head
    let config = small_config(8, 1, 32);
    let cache = KVCache::new(&config, 1, &Device::Cpu);
    assert!(cache.is_ok(), "MQA config (1 KV head) should work");
}
