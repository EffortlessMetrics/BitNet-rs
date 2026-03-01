//! Integration tests for the GPU KV cache and paged attention engine.

use bitnet_opencl::kv_cache::{GpuKvCache, KvCacheConfig};
use bitnet_opencl::paged_attention::{
    GqaConfig, PageAllocator, PagedAttentionEngine, kv_config_for_gqa,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn small_config() -> KvCacheConfig {
    KvCacheConfig { num_layers: 2, num_heads: 4, head_dim: 8, max_seq_len: 16, page_size: 4 }
}

fn make_kv(val: f32, stride: usize) -> (Vec<f32>, Vec<f32>) {
    (vec![val; stride], vec![val + 100.0; stride])
}

// ---------------------------------------------------------------------------
// GpuKvCache — basic operations
// ---------------------------------------------------------------------------

#[test]
fn append_and_retrieve_single_position() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    let (k, v) = make_kv(1.0, stride);
    cache.append(0, &k, &v);

    let (got_k, got_v) = cache.get(0, 0..1);
    assert_eq!(got_k, k);
    assert_eq!(got_v, v);
}

#[test]
fn append_multiple_positions() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    for i in 0..5 {
        let (k, v) = make_kv(i as f32, stride);
        cache.append(0, &k, &v);
    }
    assert_eq!(cache.seq_len(0), 5);

    let (keys, vals) = cache.get(0, 2..4);
    assert_eq!(keys.len(), 2 * stride);
    assert_eq!(keys[0], 2.0);
    assert_eq!(keys[stride], 3.0);
    assert_eq!(vals[0], 102.0);
}

#[test]
fn sequential_growth_to_max_seq_len() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg.clone());

    for i in 0..cfg.max_seq_len {
        let (k, v) = make_kv(i as f32, stride);
        cache.append(0, &k, &v);
    }
    assert_eq!(cache.seq_len(0), cfg.max_seq_len);
}

#[test]
#[should_panic(expected = "sequence length exceeds max_seq_len")]
fn append_beyond_max_seq_len_panics() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg.clone());

    for i in 0..=cfg.max_seq_len {
        let (k, v) = make_kv(i as f32, stride);
        cache.append(0, &k, &v);
    }
}

#[test]
fn trim_reduces_sequence_length() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    for i in 0..10 {
        let (k, v) = make_kv(i as f32, stride);
        cache.append(0, &k, &v);
    }

    cache.trim(6);
    assert_eq!(cache.seq_len(0), 6);

    // Verify data of kept positions.
    let (keys, _) = cache.get(0, 0..6);
    assert_eq!(keys[0], 0.0);
    assert_eq!(keys[5 * stride], 5.0);
}

#[test]
fn trim_to_zero() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    for i in 0..4 {
        let (k, v) = make_kv(i as f32, stride);
        cache.append(0, &k, &v);
    }

    cache.trim(0);
    assert_eq!(cache.seq_len(0), 0);
}

#[test]
fn trim_larger_than_seq_len_is_noop() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    for i in 0..3 {
        let (k, v) = make_kv(i as f32, stride);
        cache.append(0, &k, &v);
    }

    cache.trim(100);
    assert_eq!(cache.seq_len(0), 3);
}

#[test]
fn clear_resets_all_state() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg.clone());

    for layer in 0..cfg.num_layers {
        for i in 0..4 {
            let (k, v) = make_kv(i as f32, stride);
            cache.append(layer, &k, &v);
        }
    }

    cache.clear();

    for layer in 0..cfg.num_layers {
        assert_eq!(cache.seq_len(layer), 0);
    }
}

#[test]
fn clear_then_reuse() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    cache.append(0, &vec![1.0; stride], &vec![2.0; stride]);
    cache.clear();

    cache.append(0, &vec![9.0; stride], &vec![10.0; stride]);
    let (k, v) = cache.get(0, 0..1);
    assert_eq!(k[0], 9.0);
    assert_eq!(v[0], 10.0);
}

// ---------------------------------------------------------------------------
// Page allocation / deallocation
// ---------------------------------------------------------------------------

#[test]
fn page_allocation_on_demand() {
    let cfg = small_config(); // page_size=4
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    // 4 appends should use 1 page.
    for i in 0..4 {
        cache.append(0, &vec![i as f32; stride], &vec![0.0; stride]);
    }
    assert_eq!(cache.page_table().pages_for_layer(0).len(), 1);

    // 5th append should allocate a second page.
    cache.append(0, &vec![4.0; stride], &vec![0.0; stride]);
    assert_eq!(cache.page_table().pages_for_layer(0).len(), 2);
}

#[test]
fn page_boundary_crossing_retrieval() {
    let cfg =
        KvCacheConfig { num_layers: 1, num_heads: 2, head_dim: 4, max_seq_len: 8, page_size: 3 };
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    for i in 0..7 {
        cache.append(0, &vec![i as f32; stride], &vec![(i * 10) as f32; stride]);
    }

    // Retrieve across page boundary (page 0: [0,1,2], page 1: [3,4,5]).
    let (keys, vals) = cache.get(0, 2..5);
    assert_eq!(keys[0], 2.0);
    assert_eq!(keys[stride], 3.0);
    assert_eq!(keys[2 * stride], 4.0);
    assert_eq!(vals[0], 20.0);
}

// ---------------------------------------------------------------------------
// Memory stats
// ---------------------------------------------------------------------------

#[test]
fn memory_stats_initial() {
    let cfg = small_config();
    let cache = GpuKvCache::new(cfg);
    let stats = cache.memory_usage();

    assert!(stats.total_bytes > 0);
    assert_eq!(stats.used_bytes, 0);
    assert_eq!(stats.utilization_pct, 0.0);
    assert!(stats.page_count > 0);
}

#[test]
fn memory_stats_after_appends() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    cache.append(0, &vec![1.0; stride], &vec![2.0; stride]);

    let stats = cache.memory_usage();
    assert!(stats.used_bytes > 0);
    assert!(stats.utilization_pct > 0.0);
    assert!(stats.utilization_pct <= 100.0);
}

#[test]
fn memory_stats_after_clear() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    for i in 0..8 {
        cache.append(0, &vec![i as f32; stride], &vec![0.0; stride]);
    }
    cache.clear();

    let stats = cache.memory_usage();
    assert_eq!(stats.used_bytes, 0);
    assert_eq!(stats.utilization_pct, 0.0);
}

// ---------------------------------------------------------------------------
// Multiple layers
// ---------------------------------------------------------------------------

#[test]
fn layers_are_independent() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    cache.append(0, &vec![1.0; stride], &vec![10.0; stride]);
    cache.append(0, &vec![2.0; stride], &vec![20.0; stride]);
    cache.append(1, &vec![7.0; stride], &vec![70.0; stride]);

    assert_eq!(cache.seq_len(0), 2);
    assert_eq!(cache.seq_len(1), 1);

    let (k0, _) = cache.get(0, 0..2);
    let (k1, _) = cache.get(1, 0..1);
    assert_eq!(k0[0], 1.0);
    assert_eq!(k0[stride], 2.0);
    assert_eq!(k1[0], 7.0);
}

#[test]
fn trim_only_affects_layers_above_max_len() {
    let cfg = small_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    for i in 0..8 {
        cache.append(0, &vec![i as f32; stride], &vec![0.0; stride]);
    }
    for i in 0..3 {
        cache.append(1, &vec![i as f32; stride], &vec![0.0; stride]);
    }

    cache.trim(5);
    assert_eq!(cache.seq_len(0), 5);
    assert_eq!(cache.seq_len(1), 3); // untouched
}

// ---------------------------------------------------------------------------
// PageAllocator
// ---------------------------------------------------------------------------

#[test]
fn page_allocator_allocate_and_free() {
    let mut alloc = PageAllocator::new(10);
    assert_eq!(alloc.free_count(), 10);

    let p0 = alloc.allocate().unwrap();
    let p1 = alloc.allocate().unwrap();
    assert_eq!(alloc.used_count(), 2);
    assert_eq!(alloc.free_count(), 8);

    assert!(alloc.free(p0));
    assert_eq!(alloc.used_count(), 1);
    assert_eq!(alloc.free_count(), 9);

    // Freeing same page again returns false.
    assert!(!alloc.free(p0));
    let _ = p1;
}

#[test]
fn page_allocator_exhaustion() {
    let mut alloc = PageAllocator::new(2);
    assert!(alloc.allocate().is_some());
    assert!(alloc.allocate().is_some());
    assert!(alloc.allocate().is_none());
}

#[test]
fn page_allocator_defragment() {
    let mut alloc = PageAllocator::new(6);

    let pages: Vec<usize> = (0..6).map(|_| alloc.allocate().unwrap()).collect();
    // Free pages 1 and 3 to create gaps.
    alloc.free(pages[1]);
    alloc.free(pages[3]);
    assert_eq!(alloc.used_count(), 4);

    let mapping = alloc.defragment();
    // After defrag, used pages should be contiguous 0..4.
    assert_eq!(alloc.used_count(), 4);
    assert_eq!(alloc.free_count(), 2);
    // Mapping contains any pages that moved.
    for &(old, new) in &mapping {
        assert_ne!(old, new);
        assert!(new < 4);
    }
}

#[test]
fn page_allocator_defragment_already_compact() {
    let mut alloc = PageAllocator::new(4);
    let _p0 = alloc.allocate().unwrap();
    let _p1 = alloc.allocate().unwrap();

    let mapping = alloc.defragment();
    assert!(mapping.is_empty(), "no moves needed");
}

// ---------------------------------------------------------------------------
// PagedAttentionEngine — basic attention
// ---------------------------------------------------------------------------

fn simple_gqa() -> GqaConfig {
    GqaConfig { num_q_heads: 4, num_kv_heads: 4, head_dim: 8 }
}

fn setup_attention() -> (PagedAttentionEngine, GpuKvCache) {
    let gqa = simple_gqa();
    let kv_cfg = kv_config_for_gqa(&gqa, 1, 32, 4);
    let cache = GpuKvCache::new(kv_cfg);
    let engine = PagedAttentionEngine::new(gqa);
    (engine, cache)
}

#[test]
fn attention_empty_cache_returns_zeros() {
    let (engine, cache) = setup_attention();
    let gqa = engine.gqa_config();
    let q = vec![1.0; gqa.num_q_heads * gqa.head_dim];
    let out = engine.compute_attention(&q, &cache, 0, &[]);
    assert!(out.iter().all(|&x| x == 0.0));
}

#[test]
fn attention_single_kv_copies_value() {
    let (engine, mut cache) = setup_attention();
    let gqa = engine.gqa_config();
    let head_dim = gqa.head_dim;
    let stride = gqa.num_kv_heads * head_dim;

    // Insert a single KV pair.
    let mut k = vec![0.0; stride];
    let mut v = vec![0.0; stride];
    for h in 0..gqa.num_kv_heads {
        for d in 0..head_dim {
            k[h * head_dim + d] = 1.0;
            v[h * head_dim + d] = (d + 1) as f32;
        }
    }
    cache.append(0, &k, &v);

    let q = vec![1.0; gqa.num_q_heads * head_dim];
    let out = engine.compute_attention(&q, &cache, 0, &[]);

    // With one KV entry softmax weight is 1.0, so output == v for each head.
    for h in 0..gqa.num_q_heads {
        for d in 0..head_dim {
            let expected = (d + 1) as f32;
            let actual = out[h * head_dim + d];
            assert!(
                (actual - expected).abs() < 1e-5,
                "head {h} dim {d}: expected {expected}, got {actual}"
            );
        }
    }
}

#[test]
fn attention_with_mask() {
    let (engine, mut cache) = setup_attention();
    let gqa = engine.gqa_config();
    let head_dim = gqa.head_dim;
    let stride = gqa.num_kv_heads * head_dim;

    // Two KV entries: v0 = all-1s, v1 = all-2s.
    let k0 = vec![1.0; stride];
    let v0 = vec![1.0; stride];
    let k1 = vec![1.0; stride];
    let v1 = vec![2.0; stride];
    cache.append(0, &k0, &v0);
    cache.append(0, &k1, &v1);

    // Mask out position 0; only position 1 should contribute.
    let mask = vec![0u8, 1u8];
    let q = vec![1.0; gqa.num_q_heads * head_dim];
    let out = engine.compute_attention(&q, &cache, 0, &mask);

    for &val in &out {
        assert!((val - 2.0).abs() < 1e-5, "expected ~2.0, got {val}");
    }
}

#[test]
fn attention_scores_sum_to_one() {
    // Verify softmax property: attention weights sum to 1.
    let gqa = simple_gqa();
    let kv_cfg = kv_config_for_gqa(&gqa, 1, 64, 8);
    let mut cache = GpuKvCache::new(kv_cfg);
    let head_dim = gqa.head_dim;
    let stride = gqa.num_kv_heads * head_dim;

    for i in 0..10 {
        let k: Vec<f32> = (0..stride).map(|d| (i * d) as f32 * 0.1).collect();
        let v = vec![1.0; stride]; // all-ones values
        cache.append(0, &k, &v);
    }

    let q = vec![0.5; gqa.num_q_heads * head_dim];
    let engine = PagedAttentionEngine::new(gqa.clone());
    let out = engine.compute_attention(&q, &cache, 0, &[]);

    // If all values are 1.0, output should be ~1.0 per dim (weighted average of 1s).
    for &val in &out {
        assert!((val - 1.0).abs() < 1e-4, "expected ~1.0, got {val}");
    }
}

// ---------------------------------------------------------------------------
// GQA — grouped query attention
// ---------------------------------------------------------------------------

#[test]
fn gqa_fewer_kv_heads() {
    let gqa = GqaConfig { num_q_heads: 8, num_kv_heads: 2, head_dim: 4 };
    assert_eq!(gqa.group_size(), 4);

    let kv_cfg = kv_config_for_gqa(&gqa, 1, 16, 4);
    let mut cache = GpuKvCache::new(kv_cfg);
    let stride = gqa.num_kv_heads * gqa.head_dim;

    // One KV entry.
    let mut v = vec![0.0; stride];
    for h in 0..gqa.num_kv_heads {
        for d in 0..gqa.head_dim {
            v[h * gqa.head_dim + d] = (h * 10 + d) as f32;
        }
    }
    cache.append(0, &vec![1.0; stride], &v);

    let q = vec![1.0; gqa.num_q_heads * gqa.head_dim];
    let engine = PagedAttentionEngine::new(gqa.clone());
    let out = engine.compute_attention(&q, &cache, 0, &[]);

    // Q heads 0-3 share KV head 0, Q heads 4-7 share KV head 1.
    for qh in 0..4 {
        for d in 0..gqa.head_dim {
            let expected = d as f32; // KV head 0 values
            let actual = out[qh * gqa.head_dim + d];
            assert!(
                (actual - expected).abs() < 1e-5,
                "q_head {qh} dim {d}: {actual} != {expected}"
            );
        }
    }
    for qh in 4..8 {
        for d in 0..gqa.head_dim {
            let expected = (10 + d) as f32; // KV head 1 values
            let actual = out[qh * gqa.head_dim + d];
            assert!(
                (actual - expected).abs() < 1e-5,
                "q_head {qh} dim {d}: {actual} != {expected}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Block-wise attention
// ---------------------------------------------------------------------------

#[test]
fn blocked_attention_matches_standard() {
    let gqa = simple_gqa();
    let kv_cfg = kv_config_for_gqa(&gqa, 1, 32, 4);
    let mut cache = GpuKvCache::new(kv_cfg);
    let head_dim = gqa.head_dim;
    let stride = gqa.num_kv_heads * head_dim;

    for i in 0..12 {
        let k: Vec<f32> = (0..stride).map(|d| ((i + d) as f32) * 0.1).collect();
        let v: Vec<f32> = (0..stride).map(|d| ((i * 2 + d) as f32) * 0.05).collect();
        cache.append(0, &k, &v);
    }

    let q: Vec<f32> = (0..gqa.num_q_heads * head_dim).map(|i| i as f32 * 0.1).collect();
    let engine = PagedAttentionEngine::new(gqa);

    let standard = engine.compute_attention(&q, &cache, 0, &[]);
    let blocked = engine.compute_attention_blocked(&q, &cache, 0, &[], 5);

    for (i, (&a, &b)) in standard.iter().zip(blocked.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "mismatch at {i}: standard={a}, blocked={b}");
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "page_size must be > 0")]
fn zero_page_size_panics() {
    let cfg =
        KvCacheConfig { num_layers: 1, num_heads: 1, head_dim: 1, max_seq_len: 4, page_size: 0 };
    let _ = GpuKvCache::new(cfg);
}

#[test]
#[should_panic(expected = "k length mismatch")]
fn wrong_k_length_panics() {
    let cfg = small_config();
    let mut cache = GpuKvCache::new(cfg.clone());
    let stride = cfg.num_heads * cfg.head_dim;
    cache.append(0, &vec![0.0; stride - 1], &vec![0.0; stride]);
}

#[test]
#[should_panic(expected = "v length mismatch")]
fn wrong_v_length_panics() {
    let cfg = small_config();
    let mut cache = GpuKvCache::new(cfg.clone());
    let stride = cfg.num_heads * cfg.head_dim;
    cache.append(0, &vec![0.0; stride], &vec![0.0; stride + 1]);
}

#[test]
#[should_panic(expected = "range out of bounds")]
fn get_out_of_bounds_panics() {
    let cfg = small_config();
    let cache = GpuKvCache::new(cfg);
    let _ = cache.get(0, 0..1);
}

#[test]
fn config_accessor() {
    let cfg = small_config();
    let cache = GpuKvCache::new(cfg);
    let c = cache.config();
    assert_eq!(c.num_layers, 2);
    assert_eq!(c.page_size, 4);
}

#[test]
fn kv_config_for_gqa_helper() {
    let gqa = GqaConfig { num_q_heads: 16, num_kv_heads: 4, head_dim: 64 };
    let cfg = kv_config_for_gqa(&gqa, 12, 2048, 64);
    assert_eq!(cfg.num_heads, 4);
    assert_eq!(cfg.head_dim, 64);
    assert_eq!(cfg.num_layers, 12);
    assert_eq!(cfg.max_seq_len, 2048);
    assert_eq!(cfg.page_size, 64);
}
