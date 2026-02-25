//! Tests for `LayerKVCache` and `KVCache` — the autoregressive generation state.
//!
//! Covers: initialization, single/multi-step append, overflow rejection,
//! head-count mismatch rejection, clear semantics, and GQA configurations.
#![cfg(feature = "cpu")]

use bitnet_common::BitNetConfig;
use bitnet_transformer::{KVCache, LayerKVCache};
use candle_core::{DType, Device, Tensor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn small_config(num_heads: usize, num_kv_heads: usize, max_seq_len: usize) -> BitNetConfig {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_layers = 2;
    cfg.model.num_heads = num_heads;
    cfg.model.num_key_value_heads = num_kv_heads;
    cfg.model.hidden_size = num_heads * 4; // head_dim = 4
    cfg.model.max_position_embeddings = max_seq_len;
    cfg
}

fn zeros_kv(batch: usize, heads: usize, seq: usize, dim: usize) -> Tensor {
    Tensor::zeros(&[batch, heads, seq, dim], DType::F32, &Device::Cpu).unwrap()
}

fn ones_kv(batch: usize, heads: usize, seq: usize, dim: usize) -> Tensor {
    Tensor::ones(&[batch, heads, seq, dim], DType::F32, &Device::Cpu).unwrap()
}

// ---------------------------------------------------------------------------
// LayerKVCache
// ---------------------------------------------------------------------------

#[test]
fn layer_kv_cache_initial_seq_len_is_zero() {
    let cache = LayerKVCache::new(1, 4, 128, 64, &Device::Cpu).unwrap();
    assert_eq!(cache.seq_len, 0);
}

#[test]
fn layer_kv_cache_first_append_updates_seq_len() {
    let mut cache = LayerKVCache::new(1, 2, 64, 8, &Device::Cpu).unwrap();
    let k = zeros_kv(1, 2, 3, 8);
    let v = zeros_kv(1, 2, 3, 8);
    cache.append(&k, &v).unwrap();
    assert_eq!(cache.seq_len, 3);
}

#[test]
fn layer_kv_cache_multiple_appends_accumulate_seq_len() {
    let mut cache = LayerKVCache::new(1, 2, 64, 8, &Device::Cpu).unwrap();
    for step in 1..=4 {
        let k = zeros_kv(1, 2, 1, 8);
        let v = zeros_kv(1, 2, 1, 8);
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.seq_len, step, "seq_len after step {step}");
    }
}

#[test]
fn layer_kv_cache_clear_resets_seq_len() {
    let mut cache = LayerKVCache::new(1, 2, 64, 8, &Device::Cpu).unwrap();
    let k = zeros_kv(1, 2, 5, 8);
    let v = zeros_kv(1, 2, 5, 8);
    cache.append(&k, &v).unwrap();
    assert_eq!(cache.seq_len, 5);
    cache.clear();
    assert_eq!(cache.seq_len, 0, "clear() must reset seq_len to 0");
}

#[test]
fn layer_kv_cache_overflow_returns_error() {
    let mut cache = LayerKVCache::new(1, 2, 4, 8, &Device::Cpu).unwrap();
    // Fill to capacity
    let k = zeros_kv(1, 2, 4, 8);
    let v = zeros_kv(1, 2, 4, 8);
    cache.append(&k, &v).unwrap();
    // One more token should overflow
    let k2 = zeros_kv(1, 2, 1, 8);
    let v2 = zeros_kv(1, 2, 1, 8);
    let result = cache.append(&k2, &v2);
    assert!(result.is_err(), "append past max_seq_len must return an error");
}

#[test]
fn layer_kv_cache_head_mismatch_returns_error() {
    let mut cache = LayerKVCache::new(1, 4, 64, 8, &Device::Cpu).unwrap();
    // Supply tensors with wrong head count
    let k_bad = zeros_kv(1, 2, 1, 8); // 2 heads, but cache expects 4
    let v_bad = zeros_kv(1, 2, 1, 8);
    let result = cache.append(&k_bad, &v_bad);
    assert!(result.is_err(), "head count mismatch must return an error");
}

#[test]
fn layer_kv_cache_concatenates_values_correctly() {
    let mut cache = LayerKVCache::new(1, 1, 8, 2, &Device::Cpu).unwrap();

    // Append ones first
    let k1 = ones_kv(1, 1, 2, 2);
    let v1 = ones_kv(1, 1, 2, 2);
    cache.append(&k1, &v1).unwrap();

    // Append zeros second
    let k2 = zeros_kv(1, 1, 2, 2);
    let v2 = zeros_kv(1, 1, 2, 2);
    cache.append(&k2, &v2).unwrap();

    assert_eq!(cache.seq_len, 4);
    // First 2 positions in K should be 1.0, last 2 should be 0.0
    let k_flat: Vec<f32> = cache.k.flatten_all().unwrap().to_vec1().unwrap();
    assert!(k_flat[0..4].iter().all(|&x| x == 1.0), "first 2 positions must be ones");
    assert!(k_flat[4..8].iter().all(|&x| x == 0.0), "last 2 positions must be zeros");
}

// ---------------------------------------------------------------------------
// KVCache (full model)
// ---------------------------------------------------------------------------

#[test]
fn kv_cache_creates_correct_number_of_layers() {
    let cfg = small_config(4, 4, 32);
    let cache = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();
    assert_eq!(cache.layers.len(), cfg.model.num_layers);
}

#[test]
fn kv_cache_all_layers_start_with_zero_seq_len() {
    let cfg = small_config(4, 4, 32);
    let cache = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();
    for (i, layer) in cache.layers.iter().enumerate() {
        assert_eq!(layer.seq_len, 0, "layer {i} must start at seq_len=0");
    }
}

#[test]
fn kv_cache_clear_resets_all_layers() {
    let cfg = small_config(4, 4, 32);
    let mut cache = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();

    // Append one token to each layer
    let head_dim = cfg.model.hidden_size / cfg.model.num_heads;
    for layer in &mut cache.layers {
        let k = zeros_kv(1, 4, 1, head_dim);
        let v = zeros_kv(1, 4, 1, head_dim);
        layer.append(&k, &v).unwrap();
    }

    cache.clear();

    for (i, layer) in cache.layers.iter().enumerate() {
        assert_eq!(layer.seq_len, 0, "layer {i} must be cleared");
    }
}

#[test]
fn kv_cache_layer_mut_returns_correct_layer() {
    let cfg = small_config(4, 4, 32);
    let mut cache = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();
    assert!(cache.layer_mut(0).is_some(), "layer 0 must exist");
    assert!(cache.layer_mut(cfg.model.num_layers - 1).is_some(), "last layer must exist");
    assert!(cache.layer_mut(cfg.model.num_layers).is_none(), "out-of-bounds must be None");
}

#[test]
fn kv_cache_rejects_unaligned_hidden_size() {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_layers = 2;
    cfg.model.num_heads = 3;
    cfg.model.hidden_size = 10; // 10 / 3 = not integer
    cfg.model.max_position_embeddings = 32;
    let result = KVCache::new(&cfg, 1, &Device::Cpu);
    assert!(result.is_err(), "hidden_size not divisible by num_heads must fail");
}

#[test]
fn kv_cache_gqa_config_creates_correct_kv_heads() {
    // GQA: 8 Q-heads, 2 KV-heads
    let cfg = small_config(8, 2, 32);
    let cache = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();
    // Each layer's k/v cache should have 2 KV-heads
    for (i, layer) in cache.layers.iter().enumerate() {
        assert_eq!(layer.n_kv_heads, 2, "layer {i} must have 2 KV-heads for GQA config");
    }
}
