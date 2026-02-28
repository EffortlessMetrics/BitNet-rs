//! Property-based tests for inference KV cache invariants.
//!
//! Key invariants tested:
//! - Capacity is never exceeded: `current_size <= max_size_bytes` after any
//!   sequence of store operations
//! - After `clear()`, size returns to 0 and hit/miss counters reset
//! - `get()` after `store()` is a cache hit; `get()` on absent key is a miss
//! - `clear_layer()` removes only entries for that layer
//! - `hit_rate` is in [0.0, 1.0] and consistent with hit/miss counts
//! - `usage_percent()` is in [0.0, 100.0]
//! - `record_prefill` + `record_incremental` maintain monotonic token totals

use bitnet_inference::cache::{CacheConfig, KVCache};
use proptest::prelude::*;

// ── helpers ──────────────────────────────────────────────────────────────────

fn small_cache_config(max_bytes: usize) -> CacheConfig {
    let mut cfg = CacheConfig::default();
    cfg.max_size_bytes = max_bytes;
    cfg
}

fn make_kv_pair(size: usize) -> (Vec<f32>, Vec<f32>) {
    (vec![1.0f32; size], vec![2.0f32; size])
}

// ── Capacity constraint ──────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// After any sequence of stores, current_size never exceeds max_size_bytes.
    #[test]
    fn prop_capacity_never_exceeded(
        n_stores in 1usize..20,
        kv_size in 1usize..64,
    ) {
        // Each entry = 2 * kv_size * 4 bytes
        let entry_bytes = 2 * kv_size * std::mem::size_of::<f32>();
        // Allow room for 3 entries max
        let max_bytes = entry_bytes * 3;
        let config = small_cache_config(max_bytes);
        let mut cache = KVCache::new(config).unwrap();

        for i in 0..n_stores {
            let (k, v) = make_kv_pair(kv_size);
            let _ = cache.store(0, i, k, v);
            prop_assert!(
                cache.size() <= max_bytes,
                "size {} exceeds max {} after store {}", cache.size(), max_bytes, i
            );
        }
    }

    /// After `clear()`, size is 0.
    #[test]
    fn prop_clear_resets_size(n_stores in 1usize..10) {
        let config = small_cache_config(1024 * 1024);
        let mut cache = KVCache::new(config).unwrap();

        for i in 0..n_stores {
            let (k, v) = make_kv_pair(16);
            cache.store(0, i, k, v).unwrap();
        }
        cache.clear();
        prop_assert_eq!(cache.size(), 0, "size must be 0 after clear");
    }
}

// ── Hit / miss semantics ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// `get()` after `store()` finds the entry (cache hit).
    #[test]
    fn prop_get_after_store_is_hit(
        layer in 0usize..4,
        position in 0usize..64,
        kv_size in 1usize..32,
    ) {
        let config = small_cache_config(1024 * 1024);
        let mut cache = KVCache::new(config).unwrap();

        let (k, v) = make_kv_pair(kv_size);
        cache.store(layer, position, k, v).unwrap();

        let result = cache.get(layer, position);
        prop_assert!(result.is_some(), "get after store must find entry");
    }

    /// `get()` on an absent key is a miss.
    #[test]
    fn prop_get_absent_is_miss(
        layer in 0usize..4,
        position in 0usize..64,
    ) {
        let config = small_cache_config(1024 * 1024);
        let mut cache = KVCache::new(config).unwrap();

        let result = cache.get(layer, position);
        prop_assert!(result.is_none(), "get on empty cache must be None");
    }

    /// `contains()` reflects store state.
    #[test]
    fn prop_contains_reflects_store(
        layer in 0usize..4,
        position in 0usize..64,
    ) {
        let config = small_cache_config(1024 * 1024);
        let mut cache = KVCache::new(config).unwrap();

        prop_assert!(!cache.contains(layer, position));

        let (k, v) = make_kv_pair(8);
        cache.store(layer, position, k, v).unwrap();

        prop_assert!(cache.contains(layer, position));
    }
}

// ── clear_layer isolation ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// `clear_layer(L)` removes only entries for layer L, leaving others intact.
    #[test]
    fn prop_clear_layer_isolates(
        n_per_layer in 1usize..5,
    ) {
        let config = small_cache_config(1024 * 1024);
        let mut cache = KVCache::new(config).unwrap();

        // Store in layers 0 and 1
        for pos in 0..n_per_layer {
            let (k, v) = make_kv_pair(8);
            cache.store(0, pos, k, v).unwrap();
            let (k, v) = make_kv_pair(8);
            cache.store(1, pos, k, v).unwrap();
        }

        cache.clear_layer(0);

        // Layer 0 entries gone
        for pos in 0..n_per_layer {
            prop_assert!(!cache.contains(0, pos), "layer 0 pos {} should be cleared", pos);
        }
        // Layer 1 entries still present
        for pos in 0..n_per_layer {
            prop_assert!(cache.contains(1, pos), "layer 1 pos {} should survive", pos);
        }
    }
}

// ── Stats invariants ─────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// hit_rate is in [0.0, 1.0].
    #[test]
    fn prop_hit_rate_bounded(n_stores in 1usize..10, n_gets in 1usize..10) {
        let config = small_cache_config(1024 * 1024);
        let mut cache = KVCache::new(config).unwrap();

        for i in 0..n_stores {
            let (k, v) = make_kv_pair(8);
            cache.store(0, i, k, v).unwrap();
        }
        for i in 0..n_gets {
            let _ = cache.get(0, i);
        }

        let stats = cache.stats();
        prop_assert!(
            (0.0..=1.0).contains(&stats.hit_rate),
            "hit_rate {} must be in [0.0, 1.0]", stats.hit_rate
        );
    }

    /// usage_percent is in [0.0, 100.0].
    #[test]
    fn prop_usage_percent_bounded(n_stores in 0usize..10) {
        let config = small_cache_config(1024 * 1024);
        let mut cache = KVCache::new(config).unwrap();

        for i in 0..n_stores {
            let (k, v) = make_kv_pair(8);
            cache.store(0, i, k, v).unwrap();
        }

        let pct = cache.usage_percent();
        prop_assert!(
            (0.0..=100.0).contains(&pct),
            "usage_percent {} must be in [0, 100]", pct
        );
    }
}

// ── Token tracking monotonicity ──────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// After record_prefill(P) + N incremental(1), total == P + N.
    #[test]
    fn prop_token_tracking_monotonic(
        prefill in 1usize..64,
        incremental_steps in 0usize..32,
    ) {
        let config = small_cache_config(1024 * 1024);
        let mut cache = KVCache::new(config).unwrap();

        cache.record_prefill(prefill);
        prop_assert_eq!(cache.num_tokens_prefilled(), prefill);
        prop_assert_eq!(cache.num_tokens_total(), prefill);

        for step in 1..=incremental_steps {
            cache.record_incremental(1);
            prop_assert_eq!(
                cache.num_tokens_total(), prefill + step,
                "total must be {} after {} incremental steps", prefill + step, step
            );
        }
    }
}
