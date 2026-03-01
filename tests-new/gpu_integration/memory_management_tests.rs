//! GPU memory lifecycle integration tests.
//!
//! Validates buffer allocation, reuse, eviction, and graceful OOM handling
//! using the KV cache and memory pool subsystems.

use bitnet_inference::{CacheConfig, KVCache};

// ── Buffer lifecycle ───────────────────────────────────────────────────

#[test]
fn test_buffers_freed_on_pipeline_drop() {
    // Given: a cache with stored entries
    let mut cache = KVCache::new(Default::default()).unwrap();
    let key = vec![1.0_f32; 256];
    let val = vec![2.0_f32; 256];
    for pos in 0..50 {
        cache.store(0, pos, key.clone(), val.clone()).unwrap();
    }
    let size_before = cache.size();
    assert!(size_before > 0, "cache should be non-empty before drop");

    // When: dropping the cache (explicit to verify no panic)
    drop(cache);

    // Then: no panic — Rust's RAII frees buffers automatically.
    //       (We cannot measure freed memory, but absence of panic is the gate.)
}

#[test]
fn test_memory_pool_reuse() {
    // Given: a cache
    let mut cache = KVCache::new(Default::default()).unwrap();
    let key = vec![0.5_f32; 64];
    let val = vec![0.5_f32; 64];

    // When: store, clear, re-store
    for pos in 0..20 {
        cache.store(0, pos, key.clone(), val.clone()).unwrap();
    }
    let size_after_first = cache.size();
    cache.clear();
    assert_eq!(cache.size(), 0, "clear should reset size");

    for pos in 0..20 {
        cache.store(0, pos, key.clone(), val.clone()).unwrap();
    }

    // Then: same size after re-populating
    assert_eq!(
        cache.size(),
        size_after_first,
        "re-storing the same entries should yield identical size"
    );
}

#[test]
fn test_oom_handling_graceful() {
    // Given: a very small cache budget (1 KB)
    let config = CacheConfig { max_size_bytes: 1024, ..Default::default() };
    let mut cache = KVCache::new(config).unwrap();

    let key = vec![0.1_f32; 128]; // 512 bytes per key
    let val = vec![0.2_f32; 128]; // 512 bytes per val — entry = 1024 bytes

    // When: filling beyond budget, the cache should evict old entries
    for pos in 0..10 {
        let result = cache.store(0, pos, key.clone(), val.clone());
        // Then: either succeeds (eviction) or returns a bounded error
        assert!(
            result.is_ok(),
            "store at pos {pos} should succeed via eviction: {:?}",
            result.err()
        );
    }
    assert!(cache.size() <= 1024, "cache size {} exceeds 1 KB budget", cache.size());
}

#[test]
fn test_cache_config_defaults() {
    let config = CacheConfig::default();
    assert_eq!(config.max_size_bytes, 1024 * 1024 * 1024, "default 1 GB");
    assert_eq!(config.max_sequence_length, 2048);
    assert!(!config.enable_compression);
    assert_eq!(config.block_size, 64);
}

#[test]
fn test_cache_eviction_policy_lru() {
    // Given: a cache with LRU eviction and small budget
    let config = CacheConfig { max_size_bytes: 2048, ..Default::default() };
    let mut cache = KVCache::new(config).unwrap();

    let key = vec![0.1_f32; 128]; // 512 B
    let val = vec![0.1_f32; 128]; // 512 B → entry = 1024 B

    // Store two entries (fills 2048 B budget exactly)
    cache.store(0, 0, key.clone(), val.clone()).unwrap();
    cache.store(0, 1, key.clone(), val.clone()).unwrap();
    assert!(cache.contains(0, 0));
    assert!(cache.contains(0, 1));

    // When: storing a third entry → LRU eviction of entry 0
    cache.store(0, 2, key.clone(), val.clone()).unwrap();

    // Then: oldest entry is evicted, newest is present
    assert!(!cache.contains(0, 0), "oldest entry should be evicted");
    assert!(cache.contains(0, 2), "newest entry should be present");
}

#[test]
fn test_cache_clear_layer() {
    let mut cache = KVCache::new(Default::default()).unwrap();
    let key = vec![0.1_f32; 32];
    let val = vec![0.1_f32; 32];

    // Store entries in two layers
    for pos in 0..5 {
        cache.store(0, pos, key.clone(), val.clone()).unwrap();
        cache.store(1, pos, key.clone(), val.clone()).unwrap();
    }
    let size_all = cache.size();

    // When: clearing only layer 0
    cache.clear_layer(0);

    // Then: layer 0 entries are gone, layer 1 entries remain
    for pos in 0..5 {
        assert!(!cache.contains(0, pos), "layer 0 pos {pos} should be gone");
        assert!(cache.contains(1, pos), "layer 1 pos {pos} should remain");
    }
    assert!(cache.size() < size_all, "size should decrease after clear_layer");
}

#[test]
fn test_cache_stats() {
    let mut cache = KVCache::new(Default::default()).unwrap();
    let key = vec![0.1_f32; 16];
    let val = vec![0.1_f32; 16];

    cache.store(0, 0, key.clone(), val.clone()).unwrap();

    // Hit
    let _ = cache.get(0, 0);
    // Miss
    let _ = cache.get(0, 999);

    let stats = cache.stats();
    assert_eq!(stats.total_entries, 1);
    assert!(stats.hit_rate > 0.0, "should have non-zero hit rate");
    assert!(stats.current_size_bytes > 0);
}
