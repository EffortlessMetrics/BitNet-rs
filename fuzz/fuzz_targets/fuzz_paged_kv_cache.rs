#![no_main]

use arbitrary::Arbitrary;
use bitnet_inference::kv_cache_optimized::{CacheEvictionPolicy, EvictionConfig, PagedKvCache};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct PagedCacheInput {
    tokens_per_page: u8,
    head_dim: u8,
    max_pages: u8,
    eviction_policy: u8,
    ops: Vec<CacheOp>,
}

#[derive(Arbitrary, Debug)]
enum CacheOp {
    Allocate { layer: u8 },
    Free { page_idx: u8 },
    MapPage { layer: u8, virt_idx: u8, page_idx: u8 },
    Resolve { layer: u8, virt_idx: u8 },
    RecordAttention { page_idx: u8, score: f32 },
    Clear,
}

fuzz_target!(|input: PagedCacheInput| {
    let tokens_per_page = (input.tokens_per_page as usize % 8) + 1;
    let head_dim = (input.head_dim as usize % 16) + 1;
    let max_pages = (input.max_pages as usize % 32) + 4;
    let n_layers = 2;

    let policy = match input.eviction_policy % 3 {
        0 => CacheEvictionPolicy::LRU,
        1 => CacheEvictionPolicy::SlidingWindow,
        _ => CacheEvictionPolicy::AttentionBased,
    };

    let eviction_cfg = EvictionConfig { policy, max_pages, window_size: max_pages / 2 + 1 };

    let mut cache = PagedKvCache::new(tokens_per_page, head_dim, eviction_cfg);

    // Track allocated page IDs for use in later ops
    let mut allocated_pages = Vec::new();

    for op in input.ops.into_iter().take(256) {
        match op {
            CacheOp::Allocate { layer } => {
                let layer_idx = layer as usize % n_layers;
                if let Some(page_id) = cache.allocate_page(layer_idx) {
                    allocated_pages.push(page_id);
                    // Invariant: allocated count increases
                    assert!(cache.allocated_pages() > 0);
                }
            }
            CacheOp::Free { page_idx } => {
                if !allocated_pages.is_empty() {
                    let idx = page_idx as usize % allocated_pages.len();
                    let page_id = allocated_pages.remove(idx);
                    cache.free_page(page_id);
                }
            }
            CacheOp::MapPage { layer, virt_idx, page_idx } => {
                if !allocated_pages.is_empty() {
                    let layer_idx = layer as usize % n_layers;
                    let virt = virt_idx as usize % 16;
                    let idx = page_idx as usize % allocated_pages.len();
                    let page_id = allocated_pages[idx];
                    cache.map_page(layer_idx, virt, page_id);
                }
            }
            CacheOp::Resolve { layer, virt_idx } => {
                let layer_idx = layer as usize % n_layers;
                let virt = virt_idx as usize % 16;
                // Resolve may return None — that's fine
                let _ = cache.resolve(layer_idx, virt);
            }
            CacheOp::RecordAttention { page_idx, score } => {
                if !allocated_pages.is_empty() {
                    let idx = page_idx as usize % allocated_pages.len();
                    let page_id = allocated_pages[idx];
                    let clamped_score =
                        if score.is_finite() { score.clamp(-1e6, 1e6) as f64 } else { 0.0 };
                    cache.record_attention(page_id, clamped_score);
                }
            }
            CacheOp::Clear => {
                cache.clear();
                allocated_pages.clear();
                assert_eq!(cache.allocated_pages(), 0);
            }
        }
    }

    // Invariant: capacity is consistent
    assert!(cache.allocated_pages() + cache.free_pages() == cache.capacity());
});
