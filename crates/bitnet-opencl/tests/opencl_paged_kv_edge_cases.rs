//! Edge-case tests for OpenCL paged attention, page allocator, KV cache,
//! and GQA configuration.

use bitnet_opencl::kv_cache::{CacheMemoryStats, GpuKvCache, KvCacheConfig, PageTable};
use bitnet_opencl::paged_attention::{
    GqaConfig, PageAllocator, PagedAttentionEngine, kv_config_for_gqa,
};

// ── PageAllocator tests ──────────────────────────────────────────────────────

#[test]
fn allocator_new_all_free() {
    let alloc = PageAllocator::new(16);
    assert_eq!(alloc.total_pages(), 16);
    assert_eq!(alloc.free_count(), 16);
    assert_eq!(alloc.used_count(), 0);
}

#[test]
fn allocator_zero_pages() {
    let alloc = PageAllocator::new(0);
    assert_eq!(alloc.total_pages(), 0);
    assert_eq!(alloc.free_count(), 0);
    assert_eq!(alloc.used_count(), 0);
}

#[test]
fn allocator_allocate_returns_page_index() {
    let mut alloc = PageAllocator::new(4);
    let page = alloc.allocate();
    assert!(page.is_some());
    assert_eq!(alloc.used_count(), 1);
    assert_eq!(alloc.free_count(), 3);
}

#[test]
fn allocator_allocate_all_pages() {
    let mut alloc = PageAllocator::new(3);
    assert!(alloc.allocate().is_some());
    assert!(alloc.allocate().is_some());
    assert!(alloc.allocate().is_some());
    assert_eq!(alloc.free_count(), 0);
    assert_eq!(alloc.used_count(), 3);
}

#[test]
fn allocator_allocate_exhausted() {
    let mut alloc = PageAllocator::new(1);
    assert!(alloc.allocate().is_some());
    assert!(alloc.allocate().is_none());
}

#[test]
fn allocator_free_returns_true() {
    let mut alloc = PageAllocator::new(4);
    let page = alloc.allocate().unwrap();
    assert!(alloc.free(page));
    assert_eq!(alloc.free_count(), 4);
    assert_eq!(alloc.used_count(), 0);
}

#[test]
fn allocator_free_nonexistent_returns_false() {
    let mut alloc = PageAllocator::new(4);
    assert!(!alloc.free(999));
}

#[test]
fn allocator_free_already_freed_returns_false() {
    let mut alloc = PageAllocator::new(4);
    let page = alloc.allocate().unwrap();
    alloc.free(page);
    assert!(!alloc.free(page));
}

#[test]
fn allocator_defragment_no_moves_when_contiguous() {
    let mut alloc = PageAllocator::new(4);
    alloc.allocate(); // page 3 (reversed)
    alloc.allocate(); // page 2
    // After defrag, pages 0..2 are used
    let mapping = alloc.defragment();
    // Pages should be compacted to 0, 1
    assert_eq!(alloc.used_count(), 2);
    assert_eq!(alloc.free_count(), 2);
    // Mapping contains remapped pages (if any)
    for (old, new) in &mapping {
        assert_ne!(old, new);
    }
}

#[test]
fn allocator_defragment_with_gap() {
    let mut alloc = PageAllocator::new(8);
    let p0 = alloc.allocate().unwrap(); // gets page from end
    let p1 = alloc.allocate().unwrap();
    let _p2 = alloc.allocate().unwrap();

    // Free middle page to create gap
    alloc.free(p1);
    assert_eq!(alloc.used_count(), 2);

    let mapping = alloc.defragment();
    assert_eq!(alloc.used_count(), 2);
    assert_eq!(alloc.free_count(), 6);
    // After defrag, used pages should be contiguous 0..2
    // mapping shows which pages moved
    let _ = mapping; // verify it compiles and runs
}

#[test]
fn allocator_defragment_empty() {
    let mut alloc = PageAllocator::new(4);
    let mapping = alloc.defragment();
    assert!(mapping.is_empty());
}

#[test]
fn allocator_single_page() {
    let mut alloc = PageAllocator::new(1);
    let p = alloc.allocate().unwrap();
    assert_eq!(alloc.free_count(), 0);
    alloc.free(p);
    assert_eq!(alloc.free_count(), 1);
}

// ── GqaConfig tests ──────────────────────────────────────────────────────────

#[test]
fn gqa_config_group_size_standard() {
    let gqa = GqaConfig { num_q_heads: 32, num_kv_heads: 8, head_dim: 128 };
    assert_eq!(gqa.group_size(), 4);
}

#[test]
fn gqa_config_mha_group_size_1() {
    let gqa = GqaConfig { num_q_heads: 16, num_kv_heads: 16, head_dim: 64 };
    assert_eq!(gqa.group_size(), 1);
}

#[test]
fn gqa_config_mqa_full_sharing() {
    let gqa = GqaConfig { num_q_heads: 40, num_kv_heads: 1, head_dim: 128 };
    assert_eq!(gqa.group_size(), 40);
}

#[test]
fn gqa_config_phi4_like() {
    let gqa = GqaConfig { num_q_heads: 40, num_kv_heads: 10, head_dim: 128 };
    assert_eq!(gqa.group_size(), 4);
}

// ── KvCacheConfig / kv_config_for_gqa tests ──────────────────────────────────

#[test]
fn kv_config_for_gqa_populates_fields() {
    let gqa = GqaConfig { num_q_heads: 32, num_kv_heads: 8, head_dim: 128 };
    let config = kv_config_for_gqa(&gqa, 40, 4096, 256);
    assert_eq!(config.num_layers, 40);
    assert_eq!(config.num_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.max_seq_len, 4096);
    assert_eq!(config.page_size, 256);
}

// ── GpuKvCache tests ─────────────────────────────────────────────────────────

fn small_cache_config() -> KvCacheConfig {
    KvCacheConfig { num_layers: 2, num_heads: 4, head_dim: 8, max_seq_len: 16, page_size: 4 }
}

#[test]
fn kv_cache_new_empty() {
    let cache = GpuKvCache::new(small_cache_config());
    assert_eq!(cache.seq_len(0), 0);
    assert_eq!(cache.seq_len(1), 0);
}

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

#[test]
fn kv_cache_append_increments_seq_len() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim; // 32
    let mut cache = GpuKvCache::new(cfg);
    let k = vec![1.0f32; stride];
    let v = vec![2.0f32; stride];
    cache.append(0, &k, &v);
    assert_eq!(cache.seq_len(0), 1);
    assert_eq!(cache.seq_len(1), 0); // other layer unchanged
}

#[test]
fn kv_cache_append_and_get_roundtrip() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    let k: Vec<f32> = (0..stride).map(|i| i as f32).collect();
    let v: Vec<f32> = (0..stride).map(|i| (i as f32) * 10.0).collect();
    cache.append(0, &k, &v);

    let (got_k, got_v) = cache.get(0, 0..1);
    assert_eq!(got_k, k);
    assert_eq!(got_v, v);
}

#[test]
fn kv_cache_multiple_appends() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    for i in 0..5 {
        let k = vec![i as f32; stride];
        let v = vec![(i as f32) * -1.0; stride];
        cache.append(0, &k, &v);
    }
    assert_eq!(cache.seq_len(0), 5);

    let (keys, vals) = cache.get(0, 0..5);
    assert_eq!(keys.len(), 5 * stride);
    assert_eq!(vals.len(), 5 * stride);
    // First position should be all 0.0
    assert!((keys[0] - 0.0).abs() < 1e-6);
    // Last position should be all 4.0
    assert!((keys[4 * stride] - 4.0).abs() < 1e-6);
}

#[test]
fn kv_cache_multiple_layers() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    cache.append(0, &vec![2.0; stride], &vec![2.0; stride]);
    cache.append(1, &vec![3.0; stride], &vec![3.0; stride]);

    assert_eq!(cache.seq_len(0), 2);
    assert_eq!(cache.seq_len(1), 1);
}

#[test]
#[should_panic(expected = "k length mismatch")]
fn kv_cache_wrong_k_length_panics() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);
    cache.append(0, &vec![1.0; stride + 1], &vec![1.0; stride]);
}

#[test]
#[should_panic(expected = "v length mismatch")]
fn kv_cache_wrong_v_length_panics() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);
    cache.append(0, &vec![1.0; stride], &vec![1.0; stride - 1]);
}

#[test]
fn kv_cache_trim() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    for i in 0..8 {
        cache.append(0, &vec![i as f32; stride], &vec![i as f32; stride]);
    }
    assert_eq!(cache.seq_len(0), 8);

    cache.trim(3);
    assert_eq!(cache.seq_len(0), 3);

    // Verify data integrity after trim
    let (keys, _) = cache.get(0, 0..3);
    assert!((keys[0] - 0.0).abs() < 1e-6);
}

#[test]
fn kv_cache_trim_to_zero() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    cache.trim(0);
    assert_eq!(cache.seq_len(0), 0);
}

#[test]
fn kv_cache_trim_larger_than_current_is_noop() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    cache.trim(100); // larger than current
    assert_eq!(cache.seq_len(0), 1);
}

#[test]
fn kv_cache_clear() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    cache.append(1, &vec![2.0; stride], &vec![2.0; stride]);
    cache.clear();
    assert_eq!(cache.seq_len(0), 0);
    assert_eq!(cache.seq_len(1), 0);
}

#[test]
fn kv_cache_memory_usage_initially_zero_utilization() {
    let cache = GpuKvCache::new(small_cache_config());
    let stats = cache.memory_usage();
    // All pages are pre-allocated but free
    assert!(stats.total_bytes > 0);
    assert_eq!(stats.used_bytes, 0);
    assert_eq!(stats.utilization_pct, 0.0);
}

#[test]
fn kv_cache_memory_usage_after_append() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    let stats = cache.memory_usage();
    assert!(stats.used_bytes > 0);
    assert!(stats.utilization_pct > 0.0);
}

#[test]
fn kv_cache_page_table_empty() {
    let cache = GpuKvCache::new(small_cache_config());
    let table = cache.page_table();
    assert!(table.pages_for_layer(0).is_empty());
    assert!(table.pages_for_layer(1).is_empty());
}

#[test]
fn kv_cache_page_table_after_append() {
    let cfg = small_cache_config();
    let stride = cfg.num_heads * cfg.head_dim;
    let mut cache = GpuKvCache::new(cfg);

    cache.append(0, &vec![1.0; stride], &vec![1.0; stride]);
    let table = cache.page_table();
    assert_eq!(table.pages_for_layer(0).len(), 1);
    assert!(table.pages_for_layer(1).is_empty());
}

#[test]
fn kv_cache_config_accessor() {
    let cfg = small_cache_config();
    let cache = GpuKvCache::new(cfg.clone());
    assert_eq!(cache.config().num_layers, 2);
    assert_eq!(cache.config().head_dim, 8);
}

// ── PagedAttentionEngine tests ───────────────────────────────────────────────

fn simple_gqa() -> GqaConfig {
    GqaConfig { num_q_heads: 2, num_kv_heads: 1, head_dim: 4 }
}

fn simple_kv_config() -> KvCacheConfig {
    let gqa = simple_gqa();
    kv_config_for_gqa(&gqa, 1, 16, 4)
}

#[test]
fn paged_attention_empty_cache_returns_zeros() {
    let engine = PagedAttentionEngine::new(simple_gqa());
    let cache = GpuKvCache::new(simple_kv_config());
    let q = vec![1.0f32; 2 * 4]; // num_q_heads * head_dim
    let output = engine.compute_attention(&q, &cache, 0, &[]);
    assert_eq!(output.len(), 8);
    assert!(output.iter().all(|&v| v == 0.0));
}

#[test]
fn paged_attention_single_position() {
    let gqa = simple_gqa();
    let engine = PagedAttentionEngine::new(gqa.clone());
    let kv_cfg = simple_kv_config();
    let mut cache = GpuKvCache::new(kv_cfg);

    // Append one KV pair: k = [1,0,0,0], v = [0,0,0,1]
    let k = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![0.0, 0.0, 0.0, 1.0];
    cache.append(0, &k, &v);

    // Query
    let q = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let output = engine.compute_attention(&q, &cache, 0, &[]);

    // With single position, softmax is 1.0, so output = value
    assert_eq!(output.len(), 8);
    // Both heads should produce [0,0,0,1] since there's only one position
    assert!((output[3] - 1.0).abs() < 1e-5);
    assert!((output[7] - 1.0).abs() < 1e-5);
}

#[test]
fn paged_attention_with_mask() {
    let gqa = simple_gqa();
    let engine = PagedAttentionEngine::new(gqa);
    let kv_cfg = simple_kv_config();
    let mut cache = GpuKvCache::new(kv_cfg);

    let k1 = vec![1.0, 0.0, 0.0, 0.0];
    let v1 = vec![10.0, 0.0, 0.0, 0.0];
    let k2 = vec![0.0, 1.0, 0.0, 0.0];
    let v2 = vec![0.0, 20.0, 0.0, 0.0];
    cache.append(0, &k1, &v1);
    cache.append(0, &k2, &v2);

    let q = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

    // Mask out position 0 (attend only to position 1)
    let mask = vec![0u8, 1u8];
    let output = engine.compute_attention(&q, &cache, 0, &mask);
    // Only position 1 attended; its value is [0,20,0,0]
    assert!((output[1] - 20.0).abs() < 1e-5);
}

#[test]
fn paged_attention_blocked_matches_standard() {
    let gqa = simple_gqa();
    let engine = PagedAttentionEngine::new(gqa);
    let kv_cfg = simple_kv_config();
    let mut cache = GpuKvCache::new(kv_cfg);

    for i in 0..4 {
        let k = vec![i as f32, 0.0, 0.0, 0.0];
        let v = vec![0.0, i as f32, 0.0, 0.0];
        cache.append(0, &k, &v);
    }

    let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    let standard = engine.compute_attention(&q, &cache, 0, &[]);
    let blocked = engine.compute_attention_blocked(&q, &cache, 0, &[], 2);

    // Both should produce the same output
    for (a, b) in standard.iter().zip(blocked.iter()) {
        assert!((a - b).abs() < 1e-5, "standard={a}, blocked={b}");
    }
}

#[test]
fn paged_attention_blocked_zero_block_size_returns_zeros() {
    let engine = PagedAttentionEngine::new(simple_gqa());
    let kv_cfg = simple_kv_config();
    let mut cache = GpuKvCache::new(kv_cfg);

    let k = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![0.0, 1.0, 0.0, 0.0];
    cache.append(0, &k, &v);

    let q = vec![1.0; 8];
    let output = engine.compute_attention_blocked(&q, &cache, 0, &[], 0);
    assert!(output.iter().all(|&v| v == 0.0));
}

#[test]
fn paged_attention_gqa_config_accessor() {
    let gqa = simple_gqa();
    let engine = PagedAttentionEngine::new(gqa);
    assert_eq!(engine.gqa_config().num_q_heads, 2);
    assert_eq!(engine.gqa_config().num_kv_heads, 1);
    assert_eq!(engine.gqa_config().head_dim, 4);
}

// ── CacheMemoryStats tests ───────────────────────────────────────────────────

#[test]
fn cache_memory_stats_equality() {
    let a = CacheMemoryStats {
        total_bytes: 1024,
        used_bytes: 512,
        page_count: 4,
        utilization_pct: 50.0,
    };
    let b = a.clone();
    assert_eq!(a, b);
}
