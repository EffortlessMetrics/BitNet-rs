//! Edge-case tests for PageAllocator, GpuKvCache, PagedAttentionEngine,
//! GqaConfig, KvCacheConfig, PageTable, and CacheMemoryStats.
//!
//! All tests are pure-CPU math — no GPU device needed.

use bitnet_opencl::kv_cache::{CacheMemoryStats, GpuKvCache, KvCacheConfig};
use bitnet_opencl::paged_attention::{
    GqaConfig, PageAllocator, PagedAttentionEngine, kv_config_for_gqa,
};

// ---------------------------------------------------------------------------
// PageAllocator
// ---------------------------------------------------------------------------

#[test]
fn allocator_initial_state() {
    let alloc = PageAllocator::new(8);
    assert_eq!(alloc.total_pages(), 8);
    assert_eq!(alloc.free_count(), 8);
    assert_eq!(alloc.used_count(), 0);
}

#[test]
fn allocator_allocate_and_free() {
    let mut alloc = PageAllocator::new(4);
    let p0 = alloc.allocate().unwrap();
    assert_eq!(alloc.used_count(), 1);
    assert_eq!(alloc.free_count(), 3);

    assert!(alloc.free(p0));
    assert_eq!(alloc.used_count(), 0);
    assert_eq!(alloc.free_count(), 4);
}

#[test]
fn allocator_allocate_all_pages() {
    let mut alloc = PageAllocator::new(3);
    alloc.allocate().unwrap();
    alloc.allocate().unwrap();
    alloc.allocate().unwrap();
    assert_eq!(alloc.free_count(), 0);
    assert!(alloc.allocate().is_none());
}

#[test]
fn allocator_free_nonexistent() {
    let mut alloc = PageAllocator::new(4);
    assert!(!alloc.free(99));
}

#[test]
fn allocator_free_already_freed() {
    let mut alloc = PageAllocator::new(4);
    let p = alloc.allocate().unwrap();
    assert!(alloc.free(p));
    // Double free returns false
    assert!(!alloc.free(p));
}

#[test]
fn allocator_defragment_no_gaps() {
    let mut alloc = PageAllocator::new(4);
    alloc.allocate().unwrap(); // 0
    alloc.allocate().unwrap(); // 1
    let mapping = alloc.defragment();
    // Pages 0,1 are contiguous — no moves needed
    assert!(mapping.is_empty());
}

#[test]
fn allocator_defragment_with_gaps() {
    let mut alloc = PageAllocator::new(4);
    let p0 = alloc.allocate().unwrap();
    let _p1 = alloc.allocate().unwrap();
    let _p2 = alloc.allocate().unwrap();
    alloc.free(p0); // Creates a gap at the front

    let mapping = alloc.defragment();
    // Should have moved something to fill the gap
    assert!(!mapping.is_empty());
    assert_eq!(alloc.used_count(), 2);
}

#[test]
fn allocator_zero_pages() {
    let alloc = PageAllocator::new(0);
    assert_eq!(alloc.total_pages(), 0);
    assert_eq!(alloc.free_count(), 0);
}

// ---------------------------------------------------------------------------
// KvCacheConfig
// ---------------------------------------------------------------------------

#[test]
fn kv_cache_config_debug_clone() {
    let cfg =
        KvCacheConfig { num_layers: 2, num_heads: 4, head_dim: 8, max_seq_len: 32, page_size: 4 };
    let cfg2 = cfg.clone();
    assert_eq!(cfg2.num_layers, 2);
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("KvCacheConfig"));
}

// ---------------------------------------------------------------------------
// GpuKvCache — basic operations
// ---------------------------------------------------------------------------

fn small_cache() -> GpuKvCache {
    GpuKvCache::new(KvCacheConfig {
        num_layers: 2,
        num_heads: 2,
        head_dim: 4,
        max_seq_len: 8,
        page_size: 2,
    })
}

#[test]
fn kv_cache_initial_empty() {
    let cache = small_cache();
    assert_eq!(cache.seq_len(0), 0);
    assert_eq!(cache.seq_len(1), 0);
}

#[test]
fn kv_cache_append_and_get() {
    let mut cache = small_cache();
    // stride = num_heads * head_dim = 2 * 4 = 8
    let k = vec![1.0f32; 8];
    let v = vec![2.0f32; 8];
    cache.append(0, &k, &v);
    assert_eq!(cache.seq_len(0), 1);

    let (keys, vals) = cache.get(0, 0..1);
    assert_eq!(keys.len(), 8);
    assert_eq!(vals.len(), 8);
    assert!(keys.iter().all(|&x| x == 1.0));
    assert!(vals.iter().all(|&x| x == 2.0));
}

#[test]
fn kv_cache_append_multiple_positions() {
    let mut cache = small_cache();
    let stride = 8;
    for i in 0..4 {
        let k = vec![i as f32; stride];
        let v = vec![(i + 10) as f32; stride];
        cache.append(0, &k, &v);
    }
    assert_eq!(cache.seq_len(0), 4);

    let (keys, vals) = cache.get(0, 0..4);
    assert_eq!(keys.len(), 4 * stride);
    // First position should be all 0.0
    assert!(keys[..stride].iter().all(|&x| x == 0.0));
    // Last position should be all 3.0
    assert!(keys[3 * stride..4 * stride].iter().all(|&x| x == 3.0));
    assert!(vals[3 * stride..4 * stride].iter().all(|&x| x == 13.0));
}

#[test]
fn kv_cache_layers_independent() {
    let mut cache = small_cache();
    let stride = 8;
    cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    cache.append(1, &vec![2.0; stride], &vec![2.0; stride]);
    cache.append(1, &vec![3.0; stride], &vec![3.0; stride]);

    assert_eq!(cache.seq_len(0), 1);
    assert_eq!(cache.seq_len(1), 2);
}

#[test]
fn kv_cache_clear() {
    let mut cache = small_cache();
    let stride = 8;
    cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    cache.append(1, &vec![1.0; stride], &vec![1.0; stride]);
    cache.clear();
    assert_eq!(cache.seq_len(0), 0);
    assert_eq!(cache.seq_len(1), 0);
}

#[test]
fn kv_cache_trim() {
    let mut cache = small_cache();
    let stride = 8;
    for _ in 0..4 {
        cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    }
    assert_eq!(cache.seq_len(0), 4);
    cache.trim(2);
    assert_eq!(cache.seq_len(0), 2);
}

#[test]
fn kv_cache_trim_to_zero() {
    let mut cache = small_cache();
    let stride = 8;
    cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    cache.trim(0);
    assert_eq!(cache.seq_len(0), 0);
}

#[test]
fn kv_cache_trim_larger_than_len() {
    let mut cache = small_cache();
    let stride = 8;
    cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    cache.trim(100);
    // No change — trim only shrinks
    assert_eq!(cache.seq_len(0), 1);
}

#[test]
fn kv_cache_memory_usage_initial() {
    let cache = small_cache();
    let stats = cache.memory_usage();
    assert!(stats.total_bytes > 0);
    assert_eq!(stats.used_bytes, 0);
    assert_eq!(stats.utilization_pct, 0.0);
}

#[test]
fn kv_cache_memory_usage_after_append() {
    let mut cache = small_cache();
    let stride = 8;
    cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    let stats = cache.memory_usage();
    assert!(stats.used_bytes > 0);
    assert!(stats.utilization_pct > 0.0);
}

#[test]
fn kv_cache_page_table_access() {
    let mut cache = small_cache();
    let stride = 8;
    // Append 3 positions to layer 0 (page_size=2, so 2 pages needed)
    for _ in 0..3 {
        cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    }
    let pt = cache.page_table();
    let pages = pt.pages_for_layer(0);
    assert_eq!(pages.len(), 2); // 3 positions / page_size=2 = 2 pages
}

#[test]
fn kv_cache_config_accessor() {
    let cache = small_cache();
    let cfg = cache.config();
    assert_eq!(cfg.num_layers, 2);
    assert_eq!(cfg.page_size, 2);
}

// ---------------------------------------------------------------------------
// CacheMemoryStats
// ---------------------------------------------------------------------------

#[test]
fn cache_memory_stats_debug_clone_eq() {
    let stats = CacheMemoryStats {
        total_bytes: 1024,
        used_bytes: 512,
        page_count: 4,
        utilization_pct: 50.0,
    };
    let stats2 = stats.clone();
    assert_eq!(stats, stats2);
    let dbg = format!("{stats:?}");
    assert!(dbg.contains("CacheMemoryStats"));
}

// ---------------------------------------------------------------------------
// GqaConfig
// ---------------------------------------------------------------------------

#[test]
fn gqa_config_group_size() {
    let gqa = GqaConfig { num_q_heads: 32, num_kv_heads: 8, head_dim: 64 };
    assert_eq!(gqa.group_size(), 4);
}

#[test]
fn gqa_config_group_size_no_gqa() {
    // All heads are KV heads (MHA)
    let gqa = GqaConfig { num_q_heads: 8, num_kv_heads: 8, head_dim: 64 };
    assert_eq!(gqa.group_size(), 1);
}

#[test]
fn gqa_config_debug_clone() {
    let gqa = GqaConfig { num_q_heads: 4, num_kv_heads: 2, head_dim: 8 };
    let gqa2 = gqa.clone();
    assert_eq!(gqa2.num_q_heads, 4);
    let dbg = format!("{gqa:?}");
    assert!(dbg.contains("GqaConfig"));
}

// ---------------------------------------------------------------------------
// kv_config_for_gqa helper
// ---------------------------------------------------------------------------

#[test]
fn kv_config_for_gqa_helper() {
    let gqa = GqaConfig { num_q_heads: 8, num_kv_heads: 2, head_dim: 4 };
    let cfg = kv_config_for_gqa(&gqa, 4, 32, 8);
    assert_eq!(cfg.num_layers, 4);
    assert_eq!(cfg.num_heads, 2); // Uses KV heads
    assert_eq!(cfg.head_dim, 4);
    assert_eq!(cfg.max_seq_len, 32);
    assert_eq!(cfg.page_size, 8);
}

// ---------------------------------------------------------------------------
// PagedAttentionEngine — single-head MHA
// ---------------------------------------------------------------------------

fn single_head_engine() -> (PagedAttentionEngine, GpuKvCache) {
    let gqa = GqaConfig { num_q_heads: 1, num_kv_heads: 1, head_dim: 4 };
    let cfg = kv_config_for_gqa(&gqa, 1, 16, 4);
    let cache = GpuKvCache::new(cfg);
    let engine = PagedAttentionEngine::new(gqa);
    (engine, cache)
}

#[test]
fn attention_empty_cache_returns_zeros() {
    let (engine, cache) = single_head_engine();
    let q = vec![1.0f32; 4];
    let out = engine.compute_attention(&q, &cache, 0, &[]);
    assert_eq!(out.len(), 4);
    assert!(out.iter().all(|&x| x == 0.0));
}

#[test]
fn attention_single_position_identity() {
    let (engine, mut cache) = single_head_engine();
    // With a single KV entry, attention should return V (softmax is 1.0)
    let k = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![3.0, 4.0, 5.0, 6.0];
    cache.append(0, &k, &v);

    let q = vec![1.0, 0.0, 0.0, 0.0];
    let out = engine.compute_attention(&q, &cache, 0, &[]);
    // Single entry — softmax(score) = 1.0 → output = v
    for (a, b) in out.iter().zip(v.iter()) {
        assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
    }
}

#[test]
fn attention_two_positions_uniform() {
    let (engine, mut cache) = single_head_engine();
    // Two identical keys → uniform attention → output = average of values
    let k = vec![1.0, 0.0, 0.0, 0.0];
    let v1 = vec![2.0, 0.0, 0.0, 0.0];
    let v2 = vec![0.0, 2.0, 0.0, 0.0];
    cache.append(0, &k, &v1);
    cache.append(0, &k, &v2);

    let q = vec![1.0, 0.0, 0.0, 0.0];
    let out = engine.compute_attention(&q, &cache, 0, &[]);
    // Equal scores → 50/50 → average
    assert!((out[0] - 1.0).abs() < 1e-5);
    assert!((out[1] - 1.0).abs() < 1e-5);
}

#[test]
fn attention_mask_excludes_position() {
    let (engine, mut cache) = single_head_engine();
    let k = vec![1.0, 0.0, 0.0, 0.0];
    let v1 = vec![10.0, 0.0, 0.0, 0.0];
    let v2 = vec![0.0, 20.0, 0.0, 0.0];
    cache.append(0, &k, &v1);
    cache.append(0, &k, &v2);

    let q = vec![1.0, 0.0, 0.0, 0.0];
    // Mask out position 0
    let mask = vec![0u8, 1];
    let out = engine.compute_attention(&q, &cache, 0, &mask);
    // Only position 1 attended — output = v2
    assert!((out[0] - 0.0).abs() < 1e-5);
    assert!((out[1] - 20.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// PagedAttentionEngine — GQA (multi-head with grouped keys)
// ---------------------------------------------------------------------------

#[test]
fn attention_gqa_two_q_one_kv() {
    let gqa = GqaConfig { num_q_heads: 2, num_kv_heads: 1, head_dim: 2 };
    let cfg = kv_config_for_gqa(&gqa, 1, 16, 4);
    let mut cache = GpuKvCache::new(cfg);
    let engine = PagedAttentionEngine::new(gqa);

    // stride = num_kv_heads * head_dim = 1 * 2 = 2
    let k = vec![1.0, 0.0];
    let v = vec![5.0, 7.0];
    cache.append(0, &k, &v);

    // q has 2 heads * 2 head_dim = 4 elements
    let q = vec![1.0, 0.0, 0.0, 1.0];
    let out = engine.compute_attention(&q, &cache, 0, &[]);
    // Both heads attend the same single KV → both output v
    assert_eq!(out.len(), 4);
    assert!((out[0] - 5.0).abs() < 1e-5); // head 0
    assert!((out[1] - 7.0).abs() < 1e-5);
    assert!((out[2] - 5.0).abs() < 1e-5); // head 1
    assert!((out[3] - 7.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// PagedAttentionEngine — blocked attention
// ---------------------------------------------------------------------------

#[test]
fn blocked_attention_empty_cache() {
    let (engine, cache) = single_head_engine();
    let q = vec![1.0f32; 4];
    let out = engine.compute_attention_blocked(&q, &cache, 0, &[], 2);
    assert!(out.iter().all(|&x| x == 0.0));
}

#[test]
fn blocked_attention_matches_regular() {
    let (engine, mut cache) = single_head_engine();
    let k1 = vec![1.0, 0.0, 0.0, 0.0];
    let v1 = vec![2.0, 3.0, 4.0, 5.0];
    let k2 = vec![0.0, 1.0, 0.0, 0.0];
    let v2 = vec![6.0, 7.0, 8.0, 9.0];
    cache.append(0, &k1, &v1);
    cache.append(0, &k2, &v2);

    let q = vec![1.0, 0.5, 0.0, 0.0];

    let regular = engine.compute_attention(&q, &cache, 0, &[]);
    let blocked = engine.compute_attention_blocked(&q, &cache, 0, &[], 1);

    for (a, b) in regular.iter().zip(blocked.iter()) {
        assert!((a - b).abs() < 1e-4, "regular={a}, blocked={b}");
    }
}

#[test]
fn blocked_attention_zero_block_size() {
    let (engine, cache) = single_head_engine();
    let q = vec![1.0f32; 4];
    let out = engine.compute_attention_blocked(&q, &cache, 0, &[], 0);
    assert!(out.iter().all(|&x| x == 0.0));
}

// ---------------------------------------------------------------------------
// PagedAttentionEngine — accessors
// ---------------------------------------------------------------------------

#[test]
fn engine_gqa_config_accessor() {
    let gqa = GqaConfig { num_q_heads: 32, num_kv_heads: 8, head_dim: 64 };
    let engine = PagedAttentionEngine::new(gqa);
    let cfg = engine.gqa_config();
    assert_eq!(cfg.num_q_heads, 32);
    assert_eq!(cfg.num_kv_heads, 8);
    assert_eq!(cfg.head_dim, 64);
}

// ---------------------------------------------------------------------------
// KV cache — page boundary crossing
// ---------------------------------------------------------------------------

#[test]
fn kv_cache_cross_page_boundary() {
    // page_size=2, append 5 positions → 3 pages
    let mut cache = GpuKvCache::new(KvCacheConfig {
        num_layers: 1,
        num_heads: 1,
        head_dim: 2,
        max_seq_len: 8,
        page_size: 2,
    });

    for i in 0..5 {
        cache.append(0, &[i as f32, (i * 10) as f32], &[i as f32, i as f32]);
    }
    assert_eq!(cache.seq_len(0), 5);

    // Read across page boundaries
    let (keys, _) = cache.get(0, 1..4);
    assert_eq!(keys.len(), 6); // 3 positions * stride=2
    assert!((keys[0] - 1.0).abs() < 1e-5); // position 1
    assert!((keys[2] - 2.0).abs() < 1e-5); // position 2
    assert!((keys[4] - 3.0).abs() < 1e-5); // position 3
}

// ---------------------------------------------------------------------------
// KV cache — max_seq_len panic
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "page_size must be > 0")]
fn kv_cache_zero_page_size_panics() {
    GpuKvCache::new(KvCacheConfig {
        num_layers: 1,
        num_heads: 1,
        head_dim: 1,
        max_seq_len: 4,
        page_size: 0,
    });
}
