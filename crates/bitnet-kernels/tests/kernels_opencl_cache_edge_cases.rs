#![allow(clippy::field_reassign_with_default)]
//! Edge-case integration tests for `bitnet_kernels::opencl_cache` module.
//!
//! Covers:
//! - SourceHasher: deterministic hashing
//! - KernelCacheKey: construction, from_source, Display, cache_filename
//! - KernelCacheEntry: construction, binary_size, touch, is_expired
//! - CacheConfig: default, validate
//! - CacheStats: hit_rate, Display
//! - CacheSerializer: serialize/deserialize roundtrip
//! - KernelCacheManager: store/lookup, eviction, invalidate, clear, NoCache policy

use bitnet_kernels::opencl_cache::{
    CacheConfig, CacheEvictionStrategy, CachePolicy, CacheSerializer, CacheStats, KernelCacheEntry,
    KernelCacheKey, KernelCacheManager, SourceHasher,
};
use std::time::Duration;

// =========================================================================
// SourceHasher
// =========================================================================

#[test]
fn source_hasher_deterministic() {
    let h1 = SourceHasher::hash_source("kernel void foo() {}");
    let h2 = SourceHasher::hash_source("kernel void foo() {}");
    assert_eq!(h1, h2);
}

#[test]
fn source_hasher_different_sources() {
    let h1 = SourceHasher::hash_source("kernel void foo() {}");
    let h2 = SourceHasher::hash_source("kernel void bar() {}");
    assert_ne!(h1, h2);
}

#[test]
fn source_hasher_empty_string() {
    let h = SourceHasher::hash_source("");
    // Should produce a valid hash (not panic)
    let _ = h;
}

#[test]
fn hash_bytes_deterministic() {
    let h1 = SourceHasher::hash_bytes(b"hello");
    let h2 = SourceHasher::hash_bytes(b"hello");
    assert_eq!(h1, h2);
}

#[test]
fn hash_bytes_vs_source_differs() {
    // bytes and string hash use same hasher but different input types
    let h_str = SourceHasher::hash_source("hello");
    let h_bytes = SourceHasher::hash_bytes(b"hello");
    // They may or may not be equal (implementation detail); just check no panic
    let _ = (h_str, h_bytes);
}

// =========================================================================
// KernelCacheKey
// =========================================================================

#[test]
fn cache_key_new() {
    let key = KernelCacheKey::new(42, "device", "opts", "CL 3.0");
    assert_eq!(key.source_hash, 42);
    assert_eq!(key.device_name, "device");
    assert_eq!(key.build_options, "opts");
    assert_eq!(key.opencl_version, "CL 3.0");
}

#[test]
fn cache_key_from_source() {
    let key = KernelCacheKey::from_source("kernel void f() {}", "GPU", "-O2", "CL 2.0");
    assert_eq!(key.device_name, "GPU");
    assert_eq!(key.build_options, "-O2");
    assert_eq!(key.source_hash, SourceHasher::hash_source("kernel void f() {}"));
}

#[test]
fn cache_key_display() {
    let key = KernelCacheKey::new(0xFF, "dev", "opt", "3.0");
    let s = format!("{key}");
    assert!(s.contains("00000000000000ff"));
    assert!(s.contains("dev"));
}

#[test]
fn cache_key_filename() {
    let key = KernelCacheKey::new(0xDEAD_BEEF, "", "", "");
    let filename = key.cache_filename();
    assert!(filename.ends_with(".bin"));
    assert!(filename.contains("deadbeef"));
}

#[test]
fn cache_key_equality() {
    let k1 = KernelCacheKey::new(1, "a", "b", "c");
    let k2 = KernelCacheKey::new(1, "a", "b", "c");
    let k3 = KernelCacheKey::new(2, "a", "b", "c");
    assert_eq!(k1, k2);
    assert_ne!(k1, k3);
}

// =========================================================================
// KernelCacheEntry
// =========================================================================

#[test]
fn cache_entry_new() {
    let entry = KernelCacheEntry::new(
        vec![1, 2, 3],
        "log".into(),
        Duration::from_millis(100),
        "GPU".into(),
    );
    assert_eq!(entry.binary, vec![1, 2, 3]);
    assert_eq!(entry.build_log, "log");
    assert_eq!(entry.device_info, "GPU");
    assert_eq!(entry.access_count, 0);
}

#[test]
fn cache_entry_binary_size() {
    let entry =
        KernelCacheEntry::new(vec![0u8; 1024], String::new(), Duration::ZERO, String::new());
    assert_eq!(entry.binary_size(), 1024);
}

#[test]
fn cache_entry_touch() {
    let mut entry = KernelCacheEntry::new(vec![], String::new(), Duration::ZERO, String::new());
    assert_eq!(entry.access_count, 0);
    entry.touch();
    assert_eq!(entry.access_count, 1);
    entry.touch();
    assert_eq!(entry.access_count, 2);
}

#[test]
fn cache_entry_not_expired_without_ttl() {
    let entry = KernelCacheEntry::new(vec![], String::new(), Duration::ZERO, String::new());
    // With a very long TTL, should not be expired
    assert!(!entry.is_expired(Duration::from_hours(1)));
}

// =========================================================================
// CacheConfig
// =========================================================================

#[test]
fn cache_config_default() {
    let cfg = CacheConfig::default();
    assert_eq!(cfg.max_entries, 256);
    assert_eq!(cfg.policy, CachePolicy::MemoryOnly);
    assert_eq!(cfg.eviction_strategy, CacheEvictionStrategy::Lru);
    assert!(cfg.ttl.is_none());
}

#[test]
fn cache_config_validate_ok() {
    let cfg = CacheConfig::default();
    assert!(cfg.validate().is_ok());
}

#[test]
fn cache_config_validate_zero_entries() {
    let mut cfg = CacheConfig::default();
    cfg.max_entries = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn cache_config_validate_zero_size() {
    let mut cfg = CacheConfig::default();
    cfg.max_total_size = 0;
    assert!(cfg.validate().is_err());
}

// =========================================================================
// CacheStats
// =========================================================================

#[test]
fn cache_stats_default() {
    let stats = CacheStats::default();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.evictions, 0);
}

#[test]
fn cache_stats_hit_rate_zero() {
    let stats = CacheStats::default();
    assert!((stats.hit_rate() - 0.0).abs() < f64::EPSILON);
}

#[test]
fn cache_stats_hit_rate_calculated() {
    let stats = CacheStats { hits: 3, misses: 1, ..Default::default() };
    assert!((stats.hit_rate() - 0.75).abs() < 1e-10);
}

#[test]
fn cache_stats_display() {
    let stats = CacheStats { hits: 10, misses: 5, ..Default::default() };
    let s = format!("{stats}");
    assert!(s.contains("hits=10"));
    assert!(s.contains("misses=5"));
}

// =========================================================================
// CacheSerializer: roundtrip
// =========================================================================

#[test]
fn serializer_roundtrip() {
    let entry = KernelCacheEntry::new(
        vec![0xDE, 0xAD, 0xBE, 0xEF],
        "build ok".into(),
        Duration::from_millis(42),
        "TestGPU".into(),
    );
    let data = CacheSerializer::serialize(&entry);
    let recovered = CacheSerializer::deserialize(&data).expect("deserialize should succeed");
    assert_eq!(recovered.binary, entry.binary);
    assert_eq!(recovered.build_log, entry.build_log);
    assert_eq!(recovered.device_info, entry.device_info);
}

#[test]
fn serializer_roundtrip_empty_binary() {
    let entry = KernelCacheEntry::new(vec![], String::new(), Duration::ZERO, String::new());
    let data = CacheSerializer::serialize(&entry);
    let recovered = CacheSerializer::deserialize(&data).unwrap();
    assert!(recovered.binary.is_empty());
}

#[test]
fn serializer_bad_magic() {
    let mut data = CacheSerializer::serialize(&KernelCacheEntry::new(
        vec![1],
        String::new(),
        Duration::ZERO,
        String::new(),
    ));
    data[0] = 0xFF; // corrupt magic
    assert!(CacheSerializer::deserialize(&data).is_err());
}

#[test]
fn serializer_truncated() {
    let data = vec![0x42, 0x4B, 0x43, 0x4C, 0x01, 0x00, 0x00, 0x00]; // just magic + version
    assert!(CacheSerializer::deserialize(&data).is_err());
}

// =========================================================================
// KernelCacheManager: basic store/lookup
// =========================================================================

fn test_key(id: u64) -> KernelCacheKey {
    KernelCacheKey::new(id, "test-device", "", "CL 3.0")
}

#[test]
fn cache_manager_store_and_lookup() {
    let mgr = KernelCacheManager::new(CacheConfig::default());
    let key = test_key(1);
    mgr.store(key.clone(), vec![1, 2, 3], "ok".into());
    let entry = mgr.lookup(&key).expect("should find stored entry");
    assert_eq!(entry.binary, vec![1, 2, 3]);
}

#[test]
fn cache_manager_lookup_miss() {
    let mgr = KernelCacheManager::new(CacheConfig::default());
    assert!(mgr.lookup(&test_key(999)).is_none());
}

#[test]
fn cache_manager_stats_after_operations() {
    let mgr = KernelCacheManager::new(CacheConfig::default());
    mgr.store(test_key(1), vec![1], String::new());
    let _ = mgr.lookup(&test_key(1)); // hit
    let _ = mgr.lookup(&test_key(2)); // miss

    let stats = mgr.stats();
    assert_eq!(stats.stores, 1);
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.entry_count, 1);
}

// =========================================================================
// KernelCacheManager: NoCache policy
// =========================================================================

#[test]
fn cache_manager_no_cache_policy() {
    let mut cfg = CacheConfig::default();
    cfg.policy = CachePolicy::NoCache;
    let mgr = KernelCacheManager::new(cfg);

    mgr.store(test_key(1), vec![1], String::new());
    assert!(mgr.lookup(&test_key(1)).is_none(), "NoCache should not store");

    let stats = mgr.stats();
    assert_eq!(stats.stores, 0);
    assert_eq!(stats.misses, 1);
}

// =========================================================================
// KernelCacheManager: eviction
// =========================================================================

#[test]
fn cache_manager_eviction_by_max_entries() {
    let mut cfg = CacheConfig::default();
    cfg.max_entries = 2;
    cfg.max_total_size = 1024 * 1024;
    let mgr = KernelCacheManager::new(cfg);

    mgr.store(test_key(1), vec![1], String::new());
    mgr.store(test_key(2), vec![2], String::new());
    // Third store should evict one
    mgr.store(test_key(3), vec![3], String::new());

    let stats = mgr.stats();
    assert!(stats.evictions >= 1, "should have evicted at least one entry");
    assert!(stats.entry_count <= 2, "should not exceed max_entries");
}

#[test]
fn cache_manager_eviction_by_total_size() {
    let mut cfg = CacheConfig::default();
    cfg.max_entries = 100;
    cfg.max_total_size = 10; // Only 10 bytes total
    let mgr = KernelCacheManager::new(cfg);

    mgr.store(test_key(1), vec![0u8; 5], String::new()); // 5 bytes
    mgr.store(test_key(2), vec![0u8; 5], String::new()); // 5 bytes → 10 total
    mgr.store(test_key(3), vec![0u8; 5], String::new()); // Should evict to fit

    let stats = mgr.stats();
    assert!(stats.evictions >= 1);
}

// =========================================================================
// KernelCacheManager: invalidate and clear
// =========================================================================

#[test]
fn cache_manager_invalidate() {
    let mgr = KernelCacheManager::new(CacheConfig::default());
    let key = test_key(1);
    mgr.store(key.clone(), vec![1], String::new());
    assert!(mgr.lookup(&key).is_some());

    mgr.invalidate(&key);
    assert!(mgr.lookup(&key).is_none());
}

#[test]
fn cache_manager_invalidate_nonexistent_key() {
    let mgr = KernelCacheManager::new(CacheConfig::default());
    // Should not panic
    mgr.invalidate(&test_key(42));
}

#[test]
fn cache_manager_clear() {
    let mgr = KernelCacheManager::new(CacheConfig::default());
    mgr.store(test_key(1), vec![1], String::new());
    mgr.store(test_key(2), vec![2], String::new());

    mgr.clear();
    let stats = mgr.stats();
    assert_eq!(stats.entry_count, 0);
    assert_eq!(stats.total_binary_size, 0);
}

// =========================================================================
// KernelCacheManager: store_with_metadata
// =========================================================================

#[test]
fn cache_manager_store_with_metadata() {
    let mgr = KernelCacheManager::new(CacheConfig::default());
    let key = test_key(1);
    mgr.store_with_metadata(
        key.clone(),
        vec![0xAB, 0xCD],
        "compiled ok".into(),
        Duration::from_millis(50),
        "Intel Arc".into(),
    );
    let entry = mgr.lookup(&key).unwrap();
    assert_eq!(entry.binary, vec![0xAB, 0xCD]);
    assert_eq!(entry.build_log, "compiled ok");
}

// =========================================================================
// Thread safety
// =========================================================================

#[test]
fn cache_manager_concurrent_access() {
    use std::sync::Arc;
    use std::thread;

    let mgr = Arc::new(KernelCacheManager::new(CacheConfig::default()));

    // Pre-populate
    mgr.store(test_key(1), vec![1, 2, 3], String::new());

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let mgr = Arc::clone(&mgr);
            thread::spawn(move || {
                // Some threads read, some write
                if i % 2 == 0 {
                    mgr.lookup(&test_key(1));
                } else {
                    mgr.store(test_key(100 + i as u64), vec![i as u8], String::new());
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Should not have panicked; stats should be consistent
    let stats = mgr.stats();
    assert!(stats.stores >= 1);
}
