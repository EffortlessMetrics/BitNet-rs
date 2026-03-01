//! Edge-case tests for CPU KV cache operations.
//!
//! Tests cover cache creation, append, slice, clear,
//! memory usage tracking, and paged allocation.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice,
};

fn make_config(
    num_layers: usize,
    max_seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> KvCacheConfig {
    KvCacheConfig { num_layers, max_seq_len: max_seq, num_heads, head_dim, dtype: KvDtype::F32 }
}

// ── Cache creation ───────────────────────────────────────────────────

#[test]
fn create_cache_basic() {
    let config = make_config(2, 64, 4, 16);
    assert!(config.validate().is_ok());
    let cache = KvCache::new(config).unwrap();
    assert_eq!(cache.num_layers(), 2);
}

#[test]
fn create_cache_single_layer() {
    let config = make_config(1, 32, 1, 8);
    let cache = KvCache::new(config).unwrap();
    assert_eq!(cache.num_layers(), 1);
    assert_eq!(cache.seq_len(0).unwrap(), 0);
}

// ── Append and read back ─────────────────────────────────────────────

#[test]
fn append_and_seq_len() {
    let config = make_config(1, 64, 2, 4);
    let mut cache = KvCache::new(config).unwrap();
    assert_eq!(cache.seq_len(0).unwrap(), 0);

    // Append 1 token: num_heads * head_dim = 2*4 = 8 floats
    let keys = vec![1.0; 8];
    let values = vec![2.0; 8];
    kv_cache_append(&mut cache, 0, &keys, &values).unwrap();
    assert_eq!(cache.seq_len(0).unwrap(), 1);
}

#[test]
fn append_multiple_tokens() {
    let config = make_config(1, 64, 2, 4);
    let mut cache = KvCache::new(config).unwrap();

    let kv_size = 8; // 2 heads * 4 dim
    for i in 0..5 {
        let keys = vec![i as f32; kv_size];
        let values = vec![(i * 10) as f32; kv_size];
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();
    }
    assert_eq!(cache.seq_len(0).unwrap(), 5);
}

#[test]
fn append_to_multiple_layers() {
    let config = make_config(3, 64, 2, 4);
    let mut cache = KvCache::new(config).unwrap();
    let kv_size = 8;

    for layer in 0..3 {
        let keys = vec![layer as f32; kv_size];
        let values = vec![(layer * 10) as f32; kv_size];
        kv_cache_append(&mut cache, layer, &keys, &values).unwrap();
    }

    for layer in 0..3 {
        assert_eq!(cache.seq_len(layer).unwrap(), 1);
    }
}

// ── Slice operations ─────────────────────────────────────────────────

#[test]
fn slice_after_append() {
    let config = make_config(1, 64, 1, 4);
    let mut cache = KvCache::new(config).unwrap();

    let keys = vec![1.0, 2.0, 3.0, 4.0];
    let values = vec![10.0, 20.0, 30.0, 40.0];
    kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

    let (k_slice, v_slice) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
    assert!(!k_slice.is_empty());
    assert!(!v_slice.is_empty());
}

// ── Clear operations ─────────────────────────────────────────────────

#[test]
fn clear_resets_seq_len() {
    let config = make_config(2, 64, 2, 4);
    let mut cache = KvCache::new(config).unwrap();
    let kv_size = 8;

    let keys = vec![1.0; kv_size];
    let values = vec![2.0; kv_size];
    kv_cache_append(&mut cache, 0, &keys, &values).unwrap();
    kv_cache_append(&mut cache, 1, &keys, &values).unwrap();

    kv_cache_clear(&mut cache);

    assert_eq!(cache.seq_len(0).unwrap(), 0);
    assert_eq!(cache.seq_len(1).unwrap(), 0);
}

// ── Memory usage ─────────────────────────────────────────────────────

#[test]
fn memory_usage_empty() {
    let config = make_config(1, 64, 2, 4);
    let cache = KvCache::new(config).unwrap();
    let usage = kv_cache_memory_usage(&cache);
    // Should be non-zero even for empty cache (allocated capacity)
    assert!(usage > 0);
}

#[test]
fn memory_usage_grows_with_layers() {
    let config_1 = make_config(1, 64, 2, 4);
    let config_4 = make_config(4, 64, 2, 4);
    let cache_1 = KvCache::new(config_1).unwrap();
    let cache_4 = KvCache::new(config_4).unwrap();
    assert!(kv_cache_memory_usage(&cache_4) > kv_cache_memory_usage(&cache_1));
}

// ── KvDtype ──────────────────────────────────────────────────────────

#[test]
fn kv_dtype_element_bytes() {
    assert_eq!(KvDtype::F32.element_bytes(), 4);
    assert_eq!(KvDtype::F16.element_bytes(), 2);
}
