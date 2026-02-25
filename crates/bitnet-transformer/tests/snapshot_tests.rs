//! Snapshot tests for `bitnet-transformer` public API surface.
//!
//! Pins the structural invariants of `KVCache` and `LayerKVCache` that must
//! remain stable across refactors — capacity, initial seq_len, and the
//! result of appending tokens.
#![cfg(feature = "cpu")]

use bitnet_common::BitNetConfig;
use bitnet_transformer::{KVCache, LayerKVCache};
use candle_core::{DType, Device, Tensor};

// ── helpers ───────────────────────────────────────────────────────────────────

fn small_config() -> BitNetConfig {
    let mut cfg = BitNetConfig::default();
    cfg.model.num_layers = 2;
    cfg.model.num_heads = 4;
    cfg.model.num_key_value_heads = 4;
    cfg.model.hidden_size = 16; // head_dim = 4
    cfg.model.max_position_embeddings = 8;
    cfg
}

fn zeros_kv(batch: usize, heads: usize, seq: usize, dim: usize) -> Tensor {
    Tensor::zeros(&[batch, heads, seq, dim], DType::F32, &Device::Cpu).unwrap()
}

// ── KVCache structural snapshots ─────────────────────────────────────────────

#[test]
fn kvcache_initial_state_snapshot() {
    let cfg = small_config();
    let cache = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();

    let summary = format!(
        "num_layers={} layer[0].seq_len={} layer[0].max_seq_len={} layer[0].n_kv_heads={}",
        cache.layers.len(),
        cache.layers[0].seq_len,
        cache.layers[0].max_seq_len,
        cache.layers[0].n_kv_heads,
    );
    insta::assert_snapshot!("kvcache_initial_state", summary);
}

#[test]
fn kvcache_after_append_snapshot() {
    let cfg = small_config();
    let mut cache = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();

    // Append 1 token to layer 0
    let kv = zeros_kv(1, 4, 1, 4);
    cache.layers[0].append(&kv, &kv).unwrap();

    let summary = format!(
        "layer[0].seq_len={} layer[1].seq_len={}",
        cache.layers[0].seq_len, cache.layers[1].seq_len,
    );
    insta::assert_snapshot!("kvcache_after_single_append", summary);
}

#[test]
fn kvcache_after_clear_snapshot() {
    let cfg = small_config();
    let mut cache = KVCache::new(&cfg, 1, &Device::Cpu).unwrap();

    // Append then clear
    let kv = zeros_kv(1, 4, 1, 4);
    cache.layers[0].append(&kv, &kv).unwrap();
    cache.clear();

    let summary = format!(
        "layer[0].seq_len={} layer[1].seq_len={}",
        cache.layers[0].seq_len, cache.layers[1].seq_len,
    );
    insta::assert_snapshot!("kvcache_after_clear", summary);
}

// ── LayerKVCache structural snapshots ────────────────────────────────────────

#[test]
fn layer_kvcache_initial_state_snapshot() {
    let layer = LayerKVCache::new(1, 2, 16, 8, &Device::Cpu).unwrap();
    let summary = format!(
        "seq_len={} max_seq_len={} n_kv_heads={}",
        layer.seq_len, layer.max_seq_len, layer.n_kv_heads,
    );
    insta::assert_snapshot!("layer_kvcache_initial_state", summary);
}

#[test]
fn layer_kvcache_append_increments_seqlen_snapshot() {
    let mut layer = LayerKVCache::new(1, 2, 16, 8, &Device::Cpu).unwrap();

    let kv = zeros_kv(1, 2, 1, 8);
    layer.append(&kv, &kv).unwrap();
    layer.append(&kv, &kv).unwrap();
    layer.append(&kv, &kv).unwrap();

    insta::assert_snapshot!("layer_kvcache_after_3_appends", layer.seq_len.to_string());
}
