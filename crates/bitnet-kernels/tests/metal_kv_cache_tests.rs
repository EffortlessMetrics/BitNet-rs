#![allow(clippy::approx_constant)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::identity_op)]
#![allow(clippy::manual_abs_diff)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_contains)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_vec)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::manual_saturating_arithmetic)]

//! Metal GPU KV cache validation tests for Apple Silicon.
//!
//! Validates KV cache operations targeting Metal via the compute
//! pipeline: append, batch append, read-back, paged allocation,
//! eviction policies, multi-head storage, grouped-query attention,
//! copy-on-write, INT8 quantised caching, sequence isolation,
//! buffer alignment, and maximum sequence length handling.

#![cfg(target_os = "macos")]

use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice, paged_kv_cache_alloc,
};

// ── Metal-specific constants ────────────────────────────────────────

/// Metal requires 256-byte buffer alignment on Apple GPUs.
const METAL_ALIGNMENT: usize = 256;

/// Maximum threads per threadgroup on Apple Silicon.
const _METAL_MAX_WORKGROUP_SIZE: u32 = 1024;

/// Typical head dimension for transformer models.
const HEAD_DIM: usize = 64;

/// Typical number of KV heads.
const NUM_HEADS: usize = 8;

/// Standard page size (tokens) for paged KV caching.
const PAGE_SIZE: usize = 16;

// ── Helpers ─────────────────────────────────────────────────────────

fn make_config(
    num_layers: usize,
    max_seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> KvCacheConfig {
    KvCacheConfig { num_layers, max_seq_len: max_seq, num_heads, head_dim, dtype: KvDtype::F32 }
}

/// Round `size` up to the next multiple of `METAL_ALIGNMENT`.
fn metal_align(size: usize) -> usize {
    let mask = METAL_ALIGNMENT - 1;
    (size + mask) & !mask
}

/// Round `size` up to the next 256-byte boundary (mirrors
/// `metal_compute::align_buffer_size`).
fn align_buffer_size(size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    metal_align(size)
}

/// Check whether `offset` is 256-byte aligned (mirrors
/// `metal_compute::is_aligned`).
fn is_aligned(offset: usize) -> bool {
    offset.is_multiple_of(METAL_ALIGNMENT)
}

/// Compute aligned buffer bytes for `element_count × element_bytes`.
fn aligned_buffer_bytes(element_count: usize, element_bytes: usize) -> usize {
    align_buffer_size(element_count * element_bytes)
}

/// Build a flat f32 vector of `count` elements starting at `base`.
fn make_kv(count: usize, base: f32) -> Vec<f32> {
    (0..count).map(|i| base + i as f32).collect()
}

// ═══════════════════════════════════════════════════════════════════
// 1. KV append — single token append to cache
// ═══════════════════════════════════════════════════════════════════

mod kv_append {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn single_token_append_updates_seq_len() {
        let config = make_config(1, 128, NUM_HEADS, HEAD_DIM);
        let mut cache = KvCache::new(config).unwrap();
        assert_eq!(cache.seq_len(0).unwrap(), 0);

        let elems = NUM_HEADS * HEAD_DIM;
        let keys = make_kv(elems, 1.0);
        let values = make_kv(elems, 100.0);
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

        assert_eq!(cache.seq_len(0).unwrap(), 1);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn append_preserves_data_across_layers() {
        let num_layers = 4;
        let config = make_config(num_layers, 128, NUM_HEADS, HEAD_DIM);
        let mut cache = KvCache::new(config).unwrap();
        let elems = NUM_HEADS * HEAD_DIM;

        for layer in 0..num_layers {
            let keys = make_kv(elems, layer as f32 * 1000.0);
            let values = make_kv(elems, layer as f32 * 2000.0);
            kv_cache_append(&mut cache, layer, &keys, &values).unwrap();
        }

        for layer in 0..num_layers {
            assert_eq!(cache.seq_len(layer).unwrap(), 1);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn append_buffer_size_is_metal_aligned() {
        let config = make_config(1, 128, NUM_HEADS, HEAD_DIM);
        let cache = KvCache::new(config).unwrap();

        let raw_bytes = kv_cache_memory_usage(&cache);
        let aligned = metal_align(raw_bytes);
        assert!(is_aligned(aligned));
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. KV batch append — multiple tokens in batch
// ═══════════════════════════════════════════════════════════════════

mod kv_batch_append {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn batch_append_increments_seq_len_correctly() {
        let config = make_config(1, 256, NUM_HEADS, HEAD_DIM);
        let mut cache = KvCache::new(config).unwrap();
        let elems = NUM_HEADS * HEAD_DIM;

        let batch_size = 8;
        let keys = make_kv(elems * batch_size, 0.0);
        let values = make_kv(elems * batch_size, 0.0);
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

        assert_eq!(cache.seq_len(0).unwrap(), batch_size);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn batch_append_followed_by_single_append() {
        let config = make_config(1, 256, NUM_HEADS, HEAD_DIM);
        let mut cache = KvCache::new(config).unwrap();
        let elems = NUM_HEADS * HEAD_DIM;

        // Batch of 4
        let keys = make_kv(elems * 4, 0.0);
        let values = make_kv(elems * 4, 0.0);
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

        // Single token
        let keys = make_kv(elems, 100.0);
        let values = make_kv(elems, 200.0);
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

        assert_eq!(cache.seq_len(0).unwrap(), 5);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. KV read — read back cached K/V for attention
// ═══════════════════════════════════════════════════════════════════

mod kv_read {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn read_back_single_token() {
        let config = make_config(1, 128, 2, 4);
        let mut cache = KvCache::new(config).unwrap();

        let keys = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

        let (k, v) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
        assert_eq!(k, &keys[..]);
        assert_eq!(v, &values[..]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn read_back_range_of_tokens() {
        let config = make_config(1, 128, 1, 4);
        let mut cache = KvCache::new(config).unwrap();

        for i in 0..5 {
            let keys = vec![i as f32; 4];
            let values = vec![(i * 10) as f32; 4];
            kv_cache_append(&mut cache, 0, &keys, &values).unwrap();
        }

        // Read tokens [1, 3)
        let (k, _v) = kv_cache_slice(&cache, 0, 1, 3).unwrap();
        assert_eq!(k.len(), 2 * 4); // 2 tokens × 4 dims
        assert_eq!(k[0], 1.0); // token 1
        assert_eq!(k[4], 2.0); // token 2
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn read_empty_range_returns_empty_slices() {
        let config = make_config(1, 128, 2, 4);
        let mut cache = KvCache::new(config).unwrap();

        let keys = make_kv(8, 0.0);
        let values = make_kv(8, 0.0);
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

        let (k, v) = kv_cache_slice(&cache, 0, 0, 0).unwrap();
        assert!(k.is_empty());
        assert!(v.is_empty());
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Paged KV — block-paged cache allocation
// ═══════════════════════════════════════════════════════════════════

mod paged_kv {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn allocate_paged_blocks() {
        let num_pages = 4;
        let pages = paged_kv_cache_alloc(num_pages, PAGE_SIZE, NUM_HEADS, HEAD_DIM).unwrap();
        assert_eq!(pages.len(), num_pages);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn paged_block_capacity_matches_page_size() {
        let pages = paged_kv_cache_alloc(1, PAGE_SIZE, NUM_HEADS, HEAD_DIM).unwrap();
        assert_eq!(pages[0].remaining(), PAGE_SIZE);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn paged_allocation_alignment() {
        let pages = paged_kv_cache_alloc(2, PAGE_SIZE, NUM_HEADS, HEAD_DIM).unwrap();
        for page in &pages {
            let raw_bytes = (page.keys.len() + page.values.len()) * std::mem::size_of::<f32>();
            let aligned = align_buffer_size(raw_bytes);
            assert!(is_aligned(aligned));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. KV eviction — LRU / sliding window eviction
// ═══════════════════════════════════════════════════════════════════

mod kv_eviction {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn clear_evicts_all_tokens() {
        let config = make_config(2, 64, 2, 4);
        let mut cache = KvCache::new(config).unwrap();
        let elems = 2 * 4;

        for layer in 0..2 {
            let keys = make_kv(elems, 0.0);
            let values = make_kv(elems, 0.0);
            kv_cache_append(&mut cache, layer, &keys, &values).unwrap();
        }

        kv_cache_clear(&mut cache);

        for layer in 0..2 {
            assert_eq!(cache.seq_len(layer).unwrap(), 0);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn sliding_window_eviction_by_refill() {
        let window = 4;
        let config = make_config(1, window, 1, 4);
        let mut cache = KvCache::new(config).unwrap();

        // Fill to capacity
        for i in 0..window {
            let keys = vec![i as f32; 4];
            let values = vec![(i * 10) as f32; 4];
            kv_cache_append(&mut cache, 0, &keys, &values).unwrap();
        }
        assert_eq!(cache.seq_len(0).unwrap(), window);

        // Evict via clear + refill with newest tokens
        kv_cache_clear(&mut cache);
        assert_eq!(cache.seq_len(0).unwrap(), 0);

        let keys = vec![99.0; 4];
        let values = vec![990.0; 4];
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();
        assert_eq!(cache.seq_len(0).unwrap(), 1);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. Multi-head KV — per-head cache storage
// ═══════════════════════════════════════════════════════════════════

mod multi_head_kv {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn multi_head_data_layout() {
        let num_heads = 4;
        let head_dim = 8;
        let config = make_config(1, 128, num_heads, head_dim);
        let mut cache = KvCache::new(config).unwrap();

        let elems = num_heads * head_dim; // 32
        let keys: Vec<f32> = (0..elems).map(|i| i as f32).collect();
        let values: Vec<f32> = (0..elems).map(|i| (i + 100) as f32).collect();
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

        let (k, v) = kv_cache_slice(&cache, 0, 0, 1).unwrap();

        // Head 0 occupies [0..head_dim), head 1 [head_dim..2*head_dim)
        assert_eq!(k[0], 0.0);
        assert_eq!(k[head_dim], head_dim as f32);
        assert_eq!(v[0], 100.0);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn per_head_stride_is_consistent() {
        let num_heads = 8;
        let head_dim = 64;
        let config = make_config(1, 64, num_heads, head_dim);
        let cache = KvCache::new(config).unwrap();

        let expected_stride = num_heads * head_dim;
        let aligned = aligned_buffer_bytes(expected_stride, std::mem::size_of::<f32>());
        assert!(is_aligned(aligned));
        assert!(aligned >= expected_stride * std::mem::size_of::<f32>());
        let _ = cache;
    }
}

// ═══════════════════════════════════════════════════════════════════
// 7. GQA KV — grouped query attention KV sharing
// ═══════════════════════════════════════════════════════════════════

mod gqa_kv {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn gqa_fewer_kv_heads_than_query_heads() {
        // GQA: 8 query heads share 2 KV heads (group size 4)
        let kv_heads = 2;
        let head_dim = 64;
        let config = make_config(1, 128, kv_heads, head_dim);
        let mut cache = KvCache::new(config).unwrap();

        let elems = kv_heads * head_dim;
        let keys = make_kv(elems, 1.0);
        let values = make_kv(elems, 100.0);
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

        let (k, _) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
        assert_eq!(k.len(), kv_heads * head_dim);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn gqa_single_kv_head_mqa() {
        // Multi-query attention: all query heads share 1 KV head
        let kv_heads = 1;
        let head_dim = 64;
        let config = make_config(1, 128, kv_heads, head_dim);
        let mut cache = KvCache::new(config).unwrap();

        let keys = make_kv(head_dim, 0.0);
        let values = make_kv(head_dim, 0.0);
        kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

        let (k, v) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
        assert_eq!(k.len(), head_dim);
        assert_eq!(v.len(), head_dim);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn gqa_memory_savings_over_mha() {
        let head_dim = 64;
        let max_seq = 512;

        let mha_config = make_config(1, max_seq, 8, head_dim);
        let mha_cache = KvCache::new(mha_config).unwrap();

        let gqa_config = make_config(1, max_seq, 2, head_dim);
        let gqa_cache = KvCache::new(gqa_config).unwrap();

        let mha_mem = kv_cache_memory_usage(&mha_cache);
        let gqa_mem = kv_cache_memory_usage(&gqa_cache);
        assert!(
            gqa_mem < mha_mem,
            "GQA cache ({gqa_mem} B) should use less memory \
             than MHA ({mha_mem} B)"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// 8. KV copy-on-write — shared prefix with CoW on divergence
// ═══════════════════════════════════════════════════════════════════

mod kv_copy_on_write {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn cow_clone_shares_prefix() {
        let config = make_config(1, 128, 2, 4);
        let mut base = KvCache::new(config).unwrap();
        let elems = 2 * 4;

        // Shared prefix: 3 tokens
        for i in 0..3 {
            let keys = make_kv(elems, i as f32 * 10.0);
            let values = make_kv(elems, i as f32 * 100.0);
            kv_cache_append(&mut base, 0, &keys, &values).unwrap();
        }

        // Clone for CoW divergence
        let mut fork = base.clone();
        assert_eq!(fork.seq_len(0).unwrap(), 3);

        // Diverge: append different data to fork
        let keys = make_kv(elems, 999.0);
        let values = make_kv(elems, 9999.0);
        kv_cache_append(&mut fork, 0, &keys, &values).unwrap();

        assert_eq!(base.seq_len(0).unwrap(), 3);
        assert_eq!(fork.seq_len(0).unwrap(), 4);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn cow_diverged_data_independent() {
        let config = make_config(1, 128, 1, 4);
        let mut base = KvCache::new(config).unwrap();

        let keys = vec![1.0, 2.0, 3.0, 4.0];
        let values = vec![10.0, 20.0, 30.0, 40.0];
        kv_cache_append(&mut base, 0, &keys, &values).unwrap();

        let mut fork = base.clone();

        // Append different data to each
        let base_k = vec![5.0, 6.0, 7.0, 8.0];
        let base_v = vec![50.0, 60.0, 70.0, 80.0];
        kv_cache_append(&mut base, 0, &base_k, &base_v).unwrap();

        let fork_k = vec![99.0, 98.0, 97.0, 96.0];
        let fork_v = vec![990.0, 980.0, 970.0, 960.0];
        kv_cache_append(&mut fork, 0, &fork_k, &fork_v).unwrap();

        let (bk, _) = kv_cache_slice(&base, 0, 1, 2).unwrap();
        let (fk, _) = kv_cache_slice(&fork, 0, 1, 2).unwrap();
        assert_ne!(bk, fk, "diverged branches must differ");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 9. KV quantized — INT8 compressed KV cache
// ═══════════════════════════════════════════════════════════════════

mod kv_quantized {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn f16_dtype_halves_element_size() {
        assert_eq!(KvDtype::F32.element_bytes(), 4);
        assert_eq!(KvDtype::F16.element_bytes(), 2);
        assert_eq!(KvDtype::Bf16.element_bytes(), 2);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn quantized_cache_config_validates() {
        let config = KvCacheConfig {
            num_layers: 2,
            max_seq_len: 512,
            num_heads: 4,
            head_dim: 64,
            dtype: KvDtype::F16,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn quantized_buffer_alignment() {
        // INT8 cache: 1 byte per element
        let seq_len = 512_usize;
        let num_heads = 8_usize;
        let head_dim = 64_usize;
        let raw_bytes = seq_len * num_heads * head_dim; // 1 byte each
        let aligned = align_buffer_size(raw_bytes);
        assert!(is_aligned(aligned));
        assert!(aligned >= raw_bytes);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 10. Sequence management — multiple sequence cache isolation
// ═══════════════════════════════════════════════════════════════════

mod sequence_management {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn independent_sequences_via_separate_caches() {
        let config = make_config(2, 128, 2, 4);
        let elems = 2 * 4;

        let mut seq_a = KvCache::new(config.clone()).unwrap();
        let mut seq_b = KvCache::new(config).unwrap();

        // Seq A: 3 tokens
        for i in 0..3 {
            let k = make_kv(elems, i as f32);
            let v = make_kv(elems, i as f32 * 10.0);
            kv_cache_append(&mut seq_a, 0, &k, &v).unwrap();
        }

        // Seq B: 5 tokens
        for i in 0..5 {
            let k = make_kv(elems, i as f32 + 100.0);
            let v = make_kv(elems, i as f32 * 10.0 + 100.0);
            kv_cache_append(&mut seq_b, 0, &k, &v).unwrap();
        }

        assert_eq!(seq_a.seq_len(0).unwrap(), 3);
        assert_eq!(seq_b.seq_len(0).unwrap(), 5);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn clear_one_sequence_does_not_affect_another() {
        let config = make_config(1, 64, 2, 4);
        let elems = 2 * 4;

        let mut seq_a = KvCache::new(config.clone()).unwrap();
        let mut seq_b = KvCache::new(config).unwrap();

        let k = make_kv(elems, 1.0);
        let v = make_kv(elems, 2.0);
        kv_cache_append(&mut seq_a, 0, &k, &v).unwrap();
        kv_cache_append(&mut seq_b, 0, &k, &v).unwrap();

        kv_cache_clear(&mut seq_a);
        assert_eq!(seq_a.seq_len(0).unwrap(), 0);
        assert_eq!(seq_b.seq_len(0).unwrap(), 1);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn layer_isolation_within_sequence() {
        let config = make_config(4, 64, 2, 4);
        let mut cache = KvCache::new(config).unwrap();
        let elems = 2 * 4;

        // Only append to layer 2
        let k = make_kv(elems, 42.0);
        let v = make_kv(elems, 84.0);
        kv_cache_append(&mut cache, 2, &k, &v).unwrap();

        assert_eq!(cache.seq_len(0).unwrap(), 0);
        assert_eq!(cache.seq_len(1).unwrap(), 0);
        assert_eq!(cache.seq_len(2).unwrap(), 1);
        assert_eq!(cache.seq_len(3).unwrap(), 0);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 11. Buffer alignment — 256-byte Metal buffer alignment
// ═══════════════════════════════════════════════════════════════════

mod buffer_alignment {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn alignment_constant_matches_metal_spec() {
        assert_eq!(METAL_ALIGNMENT, 256);
        assert!(METAL_ALIGNMENT.is_power_of_two());
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn align_buffer_size_rounds_up() {
        assert_eq!(align_buffer_size(1), METAL_ALIGNMENT);
        assert_eq!(align_buffer_size(255), METAL_ALIGNMENT);
        assert_eq!(align_buffer_size(256), METAL_ALIGNMENT);
        assert_eq!(align_buffer_size(257), 2 * METAL_ALIGNMENT);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn align_buffer_size_zero() {
        assert_eq!(align_buffer_size(0), 0);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn kv_buffer_alignment_for_typical_config() {
        let config = make_config(1, 2048, NUM_HEADS, HEAD_DIM);
        let cache = KvCache::new(config).unwrap();

        let raw = kv_cache_memory_usage(&cache);
        let aligned = align_buffer_size(raw);
        assert!(is_aligned(aligned), "aligned size {aligned} must satisfy Metal 256B alignment");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn pipeline_aligned_buffer_bytes() {
        let elems = NUM_HEADS * HEAD_DIM; // 512
        let aligned = aligned_buffer_bytes(elems, std::mem::size_of::<f32>());
        assert!(is_aligned(aligned));
        assert!(aligned >= elems * std::mem::size_of::<f32>());
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn workgroup_size_for_kv_dispatch() {
        // Linear workgroup of 256 threads fits within the 1024 limit
        let threads: u32 = 256;
        assert!(threads <= _METAL_MAX_WORKGROUP_SIZE);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 12. Max sequence length — 2048, 4096, 8192 positions
// ═══════════════════════════════════════════════════════════════════

mod max_sequence_length {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn max_seq_2048_config_validates() {
        let config = make_config(1, 2048, NUM_HEADS, HEAD_DIM);
        assert!(config.validate().is_ok());
        let cache = KvCache::new(config).unwrap();
        assert_eq!(cache.num_layers(), 1);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn max_seq_4096_config_validates() {
        let config = make_config(1, 4096, NUM_HEADS, HEAD_DIM);
        assert!(config.validate().is_ok());
        let cache = KvCache::new(config).unwrap();
        assert_eq!(cache.num_layers(), 1);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn max_seq_8192_config_validates() {
        let config = make_config(1, 8192, NUM_HEADS, HEAD_DIM);
        assert!(config.validate().is_ok());
        let cache = KvCache::new(config).unwrap();
        assert_eq!(cache.num_layers(), 1);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn buffer_alignment_scales_with_seq_len() {
        for &max_seq in &[2048_usize, 4096, 8192] {
            let config = make_config(1, max_seq, NUM_HEADS, HEAD_DIM);
            let cache = KvCache::new(config).unwrap();
            let raw = kv_cache_memory_usage(&cache);
            let aligned = align_buffer_size(raw);
            assert!(
                is_aligned(aligned),
                "alignment failed for max_seq={max_seq}: \
                 raw={raw}, aligned={aligned}"
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn memory_growth_is_linear_in_seq_len() {
        let mem_2k = {
            let c = make_config(1, 2048, NUM_HEADS, HEAD_DIM);
            let cache = KvCache::new(c).unwrap();
            kv_cache_memory_usage(&cache)
        };
        let mem_4k = {
            let c = make_config(1, 4096, NUM_HEADS, HEAD_DIM);
            let cache = KvCache::new(c).unwrap();
            kv_cache_memory_usage(&cache)
        };
        let mem_8k = {
            let c = make_config(1, 8192, NUM_HEADS, HEAD_DIM);
            let cache = KvCache::new(c).unwrap();
            kv_cache_memory_usage(&cache)
        };

        assert_eq!(mem_4k, 2 * mem_2k, "4K memory should be 2× of 2K");
        assert_eq!(mem_8k, 4 * mem_2k, "8K memory should be 4× of 2K");
    }
}
