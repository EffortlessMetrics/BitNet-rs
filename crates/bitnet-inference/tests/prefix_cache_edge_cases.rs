//! Edge-case tests for the prefix cache subsystem.
//!
//! Tests cover:
//! - PrefixCacheConfig defaults and custom configuration
//! - EvictionPolicy enum variants
//! - PrefixCache: insert, lookup, eviction, invalidation, clear
//! - Trie-based prefix matching (longest match, partial match, no match)
//! - Memory accounting and capacity limits
//! - Statistics tracking (hit rate, miss rate, eviction count)
//! - Min prefix length enforcement
//! - Multiple eviction policies (LRU, LFU, FIFO, TTL)

use bitnet_inference::prefix_cache::{
    EvictionPolicy, PrefixCache, PrefixCacheConfig, PrefixCacheStats,
};

// ---------------------------------------------------------------------------
// PrefixCacheConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn config_defaults() {
    let cfg = PrefixCacheConfig::default();
    assert_eq!(cfg.max_entries, 1024);
    assert_eq!(cfg.max_memory_bytes, 512 * 1024 * 1024);
    assert_eq!(cfg.eviction_policy, EvictionPolicy::LRU);
    assert_eq!(cfg.min_prefix_length, 4);
    assert_eq!(cfg.ttl_seconds, 3600);
}

#[test]
fn config_custom() {
    let cfg = PrefixCacheConfig {
        max_entries: 10,
        max_memory_bytes: 1024,
        eviction_policy: EvictionPolicy::LFU,
        min_prefix_length: 2,
        ttl_seconds: 60,
    };
    assert_eq!(cfg.max_entries, 10);
    assert_eq!(cfg.max_memory_bytes, 1024);
    assert_eq!(cfg.eviction_policy, EvictionPolicy::LFU);
    assert_eq!(cfg.min_prefix_length, 2);
    assert_eq!(cfg.ttl_seconds, 60);
}

// ---------------------------------------------------------------------------
// EvictionPolicy
// ---------------------------------------------------------------------------

#[test]
fn eviction_policy_variants_are_distinct() {
    let policies = [
        EvictionPolicy::LRU,
        EvictionPolicy::LFU,
        EvictionPolicy::FIFO,
        EvictionPolicy::TTL,
    ];
    for (i, a) in policies.iter().enumerate() {
        for (j, b) in policies.iter().enumerate() {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b);
            }
        }
    }
}

#[test]
fn eviction_policy_debug() {
    let lru = EvictionPolicy::LRU;
    let dbg = format!("{:?}", lru);
    assert!(dbg.contains("LRU"));
}

#[test]
fn eviction_policy_clone() {
    let original = EvictionPolicy::FIFO;
    let cloned = original;
    assert_eq!(original, cloned);
}

// ---------------------------------------------------------------------------
// PrefixCache — empty state
// ---------------------------------------------------------------------------

#[test]
fn empty_cache() {
    let cache = PrefixCache::new(PrefixCacheConfig::default());
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn empty_cache_stats() {
    let cache = PrefixCache::new(PrefixCacheConfig::default());
    let stats = cache.stats();
    assert_eq!(stats.hit_rate, 0.0);
    assert_eq!(stats.miss_rate, 0.0);
    assert_eq!(stats.eviction_count, 0);
    assert_eq!(stats.memory_usage, 0);
    assert_eq!(stats.avg_prefix_match_length, 0.0);
}

#[test]
fn empty_cache_lookup_returns_none() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    assert!(cache.lookup(&[1, 2, 3, 4, 5]).is_none());
}

#[test]
fn empty_cache_evict_returns_false() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    assert!(!cache.evict());
}

// ---------------------------------------------------------------------------
// Insert and lookup
// ---------------------------------------------------------------------------

#[test]
fn insert_and_lookup_exact_match() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    let tokens = vec![10, 20, 30, 40];
    let state = vec![0xAAu8; 32];
    cache.insert(&tokens, state.clone()).unwrap();

    let result = cache.lookup(&tokens);
    assert!(result.is_some());
    let (matched_len, entry) = result.unwrap();
    assert_eq!(matched_len, 4);
    assert_eq!(entry.cached_state, state);
    assert_eq!(entry.token_prefix, tokens);
}

#[test]
fn insert_and_lookup_prefix_match() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    let prefix = vec![1, 2, 3, 4];
    let state = vec![0xBBu8; 16];
    cache.insert(&prefix, state.clone()).unwrap();

    // Query with longer sequence — should still match the cached prefix
    let result = cache.lookup(&[1, 2, 3, 4, 5, 6, 7]);
    assert!(result.is_some());
    let (matched_len, entry) = result.unwrap();
    assert_eq!(matched_len, 4);
    assert_eq!(entry.cached_state, state);
}

#[test]
fn lookup_no_match_different_tokens() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    cache.insert(&[1, 2, 3, 4], vec![0u8; 8]).unwrap();

    // Completely different token sequence
    assert!(cache.lookup(&[5, 6, 7, 8]).is_none());
}

#[test]
fn lookup_partial_prefix_no_entry() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    // Insert a prefix [1,2,3,4] — the entry is at the terminal node
    cache.insert(&[1, 2, 3, 4], vec![0u8; 8]).unwrap();

    // Query [1,2] — trie path exists but no entry_id at depth 2
    assert!(cache.lookup(&[1, 2]).is_none());
}

#[test]
fn insert_updates_len_and_memory() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    assert_eq!(cache.len(), 0);

    cache.insert(&[1, 2, 3, 4], vec![0u8; 100]).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.stats().memory_usage, 100);

    cache.insert(&[5, 6, 7, 8], vec![0u8; 200]).unwrap();
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.stats().memory_usage, 300);
}

// ---------------------------------------------------------------------------
// Min prefix length enforcement
// ---------------------------------------------------------------------------

#[test]
fn insert_too_short_prefix_fails() {
    let cfg = PrefixCacheConfig { min_prefix_length: 4, ..PrefixCacheConfig::default() };
    let mut cache = PrefixCache::new(cfg);

    // 3 tokens < min 4
    let result = cache.insert(&[1, 2, 3], vec![0u8; 8]);
    assert!(result.is_err());
    assert!(cache.is_empty());
}

#[test]
fn insert_exact_min_length_succeeds() {
    let cfg = PrefixCacheConfig { min_prefix_length: 4, ..PrefixCacheConfig::default() };
    let mut cache = PrefixCache::new(cfg);

    cache.insert(&[1, 2, 3, 4], vec![0u8; 8]).unwrap();
    assert_eq!(cache.len(), 1);
}

#[test]
fn insert_empty_prefix_fails() {
    let cfg = PrefixCacheConfig { min_prefix_length: 1, ..PrefixCacheConfig::default() };
    let mut cache = PrefixCache::new(cfg);

    let result = cache.insert(&[], vec![0u8; 8]);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Overwriting existing prefix
// ---------------------------------------------------------------------------

#[test]
fn insert_same_prefix_replaces_entry() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    let prefix = vec![1, 2, 3, 4];

    cache.insert(&prefix, vec![0xAAu8; 32]).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.stats().memory_usage, 32);

    // Re-insert same prefix with different state
    cache.insert(&prefix, vec![0xBBu8; 64]).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.stats().memory_usage, 64);

    let (_, entry) = cache.lookup(&prefix).unwrap();
    assert_eq!(entry.cached_state, vec![0xBBu8; 64]);
}

// ---------------------------------------------------------------------------
// Multiple prefixes — longest match
// ---------------------------------------------------------------------------

#[test]
fn longest_prefix_match() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());

    // Insert two prefixes that share a common start
    cache.insert(&[1, 2, 3, 4], vec![0xAAu8; 8]).unwrap();
    cache.insert(&[1, 2, 3, 4, 5, 6], vec![0xBBu8; 16]).unwrap();

    // Query should match the longer prefix
    let (matched_len, entry) = cache.lookup(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    assert_eq!(matched_len, 6);
    assert_eq!(entry.cached_state, vec![0xBBu8; 16]);
}

#[test]
fn shorter_prefix_returned_when_longer_not_cached() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());

    cache.insert(&[1, 2, 3, 4], vec![0xAAu8; 8]).unwrap();
    // No entry for [1,2,3,4,5]

    let (matched_len, entry) = cache.lookup(&[1, 2, 3, 4, 5]).unwrap();
    assert_eq!(matched_len, 4);
    assert_eq!(entry.cached_state, vec![0xAAu8; 8]);
}

// ---------------------------------------------------------------------------
// Invalidation
// ---------------------------------------------------------------------------

#[test]
fn invalidate_removes_entry() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    let prefix = vec![1, 2, 3, 4];

    cache.insert(&prefix, vec![0u8; 32]).unwrap();
    assert_eq!(cache.len(), 1);

    cache.invalidate(&prefix);
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert!(cache.lookup(&prefix).is_none());
}

#[test]
fn invalidate_nonexistent_is_noop() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    cache.insert(&[1, 2, 3, 4], vec![0u8; 8]).unwrap();

    // Invalidating a prefix that doesn't exist
    cache.invalidate(&[9, 8, 7, 6]);
    assert_eq!(cache.len(), 1);
}

#[test]
fn invalidate_frees_memory() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    cache.insert(&[1, 2, 3, 4], vec![0u8; 100]).unwrap();
    assert_eq!(cache.stats().memory_usage, 100);

    cache.invalidate(&[1, 2, 3, 4]);
    assert_eq!(cache.stats().memory_usage, 0);
}

// ---------------------------------------------------------------------------
// Clear
// ---------------------------------------------------------------------------

#[test]
fn clear_removes_all() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    cache.insert(&[1, 2, 3, 4], vec![0u8; 16]).unwrap();
    cache.insert(&[5, 6, 7, 8], vec![0u8; 32]).unwrap();
    assert_eq!(cache.len(), 2);

    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.stats().memory_usage, 0);
    assert_eq!(cache.stats().eviction_count, 0);
}

// ---------------------------------------------------------------------------
// Statistics tracking
// ---------------------------------------------------------------------------

#[test]
fn stats_track_hit_and_miss() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    cache.insert(&[1, 2, 3, 4], vec![0u8; 8]).unwrap();

    // One hit
    let _ = cache.lookup(&[1, 2, 3, 4]);
    // One miss
    let _ = cache.lookup(&[9, 8, 7, 6]);

    let stats = cache.stats();
    assert!((stats.hit_rate - 0.5).abs() < 1e-6);
    assert!((stats.miss_rate - 0.5).abs() < 1e-6);
}

#[test]
fn stats_avg_prefix_match_length() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    cache.insert(&[1, 2, 3, 4], vec![0u8; 8]).unwrap();
    cache.insert(&[1, 2, 3, 4, 5, 6], vec![0u8; 16]).unwrap();

    // Hit 4-token prefix
    let _ = cache.lookup(&[1, 2, 3, 4]);
    // Hit 6-token prefix
    let _ = cache.lookup(&[1, 2, 3, 4, 5, 6]);

    let stats = cache.stats();
    // avg = (4+6)/2 = 5.0
    assert!((stats.avg_prefix_match_length - 5.0).abs() < 1e-6);
}

#[test]
fn cache_entry_hits_increment() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    cache.insert(&[1, 2, 3, 4], vec![0u8; 8]).unwrap();

    // First lookup
    let (_, entry1) = cache.lookup(&[1, 2, 3, 4]).unwrap();
    assert_eq!(entry1.hits, 1);

    // Second lookup
    let (_, entry2) = cache.lookup(&[1, 2, 3, 4]).unwrap();
    assert_eq!(entry2.hits, 2);
}

// ---------------------------------------------------------------------------
// Eviction — max entries
// ---------------------------------------------------------------------------

#[test]
fn eviction_on_max_entries() {
    let cfg = PrefixCacheConfig {
        max_entries: 2,
        min_prefix_length: 1,
        ..PrefixCacheConfig::default()
    };
    let mut cache = PrefixCache::new(cfg);

    cache.insert(&[1, 2, 3, 4], vec![0u8; 8]).unwrap();
    cache.insert(&[5, 6, 7, 8], vec![0u8; 8]).unwrap();
    assert_eq!(cache.len(), 2);

    // Third insert triggers eviction
    cache.insert(&[9, 10, 11, 12], vec![0u8; 8]).unwrap();
    assert_eq!(cache.len(), 2); // still at max
    assert!(cache.stats().eviction_count >= 1);
}

// ---------------------------------------------------------------------------
// Eviction — max memory
// ---------------------------------------------------------------------------

#[test]
fn eviction_on_max_memory() {
    let cfg = PrefixCacheConfig {
        max_entries: 100,
        max_memory_bytes: 100,
        min_prefix_length: 1,
        ..PrefixCacheConfig::default()
    };
    let mut cache = PrefixCache::new(cfg);

    cache.insert(&[1, 2, 3, 4], vec![0u8; 60]).unwrap();
    cache.insert(&[5, 6, 7, 8], vec![0u8; 60]).unwrap();
    // Second insert should trigger eviction of first (60 + 60 > 100)
    assert!(cache.stats().eviction_count >= 1);
    assert!(cache.stats().memory_usage <= 100);
}

// ---------------------------------------------------------------------------
// Eviction policies
// ---------------------------------------------------------------------------

#[test]
fn lru_eviction_removes_least_recently_used() {
    let cfg = PrefixCacheConfig {
        max_entries: 2,
        eviction_policy: EvictionPolicy::LRU,
        min_prefix_length: 1,
        ..PrefixCacheConfig::default()
    };
    let mut cache = PrefixCache::new(cfg);

    cache.insert(&[1, 2, 3, 4], vec![0xAAu8; 8]).unwrap();
    cache.insert(&[5, 6, 7, 8], vec![0xBBu8; 8]).unwrap();

    // Access the first entry to make it more recent
    let _ = cache.lookup(&[1, 2, 3, 4]);

    // Third insert should evict [5,6,7,8] (LRU)
    cache.insert(&[9, 10, 11, 12], vec![0xCCu8; 8]).unwrap();

    assert!(cache.lookup(&[1, 2, 3, 4]).is_some());
    assert!(cache.lookup(&[5, 6, 7, 8]).is_none()); // evicted
}

#[test]
fn lfu_eviction_removes_least_frequently_used() {
    let cfg = PrefixCacheConfig {
        max_entries: 2,
        eviction_policy: EvictionPolicy::LFU,
        min_prefix_length: 1,
        ..PrefixCacheConfig::default()
    };
    let mut cache = PrefixCache::new(cfg);

    cache.insert(&[1, 2, 3, 4], vec![0xAAu8; 8]).unwrap();
    cache.insert(&[5, 6, 7, 8], vec![0xBBu8; 8]).unwrap();

    // Access first entry 3 times to boost its frequency
    let _ = cache.lookup(&[1, 2, 3, 4]);
    let _ = cache.lookup(&[1, 2, 3, 4]);
    let _ = cache.lookup(&[1, 2, 3, 4]);

    // Access second entry only 1 time
    let _ = cache.lookup(&[5, 6, 7, 8]);

    // Third insert should evict [5,6,7,8] (fewer hits)
    cache.insert(&[9, 10, 11, 12], vec![0xCCu8; 8]).unwrap();

    assert!(cache.lookup(&[1, 2, 3, 4]).is_some());
    assert!(cache.lookup(&[5, 6, 7, 8]).is_none()); // evicted
}

#[test]
fn fifo_eviction_removes_oldest() {
    let cfg = PrefixCacheConfig {
        max_entries: 2,
        eviction_policy: EvictionPolicy::FIFO,
        min_prefix_length: 1,
        ..PrefixCacheConfig::default()
    };
    let mut cache = PrefixCache::new(cfg);

    cache.insert(&[1, 2, 3, 4], vec![0xAAu8; 8]).unwrap();
    cache.insert(&[5, 6, 7, 8], vec![0xBBu8; 8]).unwrap();

    // Even if we access the first entry, FIFO should still evict it
    let _ = cache.lookup(&[1, 2, 3, 4]);

    cache.insert(&[9, 10, 11, 12], vec![0xCCu8; 8]).unwrap();

    // First entry (oldest by creation) should be evicted
    assert!(cache.lookup(&[1, 2, 3, 4]).is_none());
    assert!(cache.lookup(&[5, 6, 7, 8]).is_some());
}

// ---------------------------------------------------------------------------
// CacheEntry fields
// ---------------------------------------------------------------------------

#[test]
fn cache_entry_size_bytes_matches_state() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    let state = vec![0xFFu8; 256];
    cache.insert(&[1, 2, 3, 4], state.clone()).unwrap();

    let (_, entry) = cache.lookup(&[1, 2, 3, 4]).unwrap();
    assert_eq!(entry.size_bytes, 256);
    assert_eq!(entry.cached_state.len(), 256);
}

#[test]
fn cache_entry_token_prefix_preserved() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    let tokens = vec![100, 200, 300, 400, 500];
    cache.insert(&tokens, vec![0u8; 16]).unwrap();

    let (_, entry) = cache.lookup(&tokens).unwrap();
    assert_eq!(entry.token_prefix, tokens);
}

// ---------------------------------------------------------------------------
// PrefixCacheStats
// ---------------------------------------------------------------------------

#[test]
fn stats_debug_impl() {
    let stats = PrefixCacheStats {
        hit_rate: 0.75,
        miss_rate: 0.25,
        eviction_count: 5,
        memory_usage: 1024,
        avg_prefix_match_length: 8.5,
    };
    let dbg = format!("{:?}", stats);
    assert!(dbg.contains("hit_rate"));
    assert!(dbg.contains("memory_usage"));
}

#[test]
fn stats_clone() {
    let stats = PrefixCacheStats {
        hit_rate: 0.5,
        miss_rate: 0.5,
        eviction_count: 0,
        memory_usage: 0,
        avg_prefix_match_length: 0.0,
    };
    let cloned = stats.clone();
    assert!((stats.hit_rate - cloned.hit_rate).abs() < 1e-10);
    assert_eq!(stats.eviction_count, cloned.eviction_count);
}

// ---------------------------------------------------------------------------
// Stress test — many entries
// ---------------------------------------------------------------------------

#[test]
fn many_entries_with_eviction() {
    let cfg = PrefixCacheConfig {
        max_entries: 10,
        min_prefix_length: 1,
        ..PrefixCacheConfig::default()
    };
    let mut cache = PrefixCache::new(cfg);

    // Insert 20 entries — should trigger eviction after 10
    for i in 0u32..20 {
        let tokens: Vec<u32> = vec![i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3];
        cache.insert(&tokens, vec![0u8; 16]).unwrap();
    }

    assert_eq!(cache.len(), 10);
    assert!(cache.stats().eviction_count >= 10);
}

// ---------------------------------------------------------------------------
// Edge case: zero-length state
// ---------------------------------------------------------------------------

#[test]
fn insert_empty_state() {
    let mut cache = PrefixCache::new(PrefixCacheConfig::default());
    cache.insert(&[1, 2, 3, 4], vec![]).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.stats().memory_usage, 0);

    let (_, entry) = cache.lookup(&[1, 2, 3, 4]).unwrap();
    assert!(entry.cached_state.is_empty());
    assert_eq!(entry.size_bytes, 0);
}
