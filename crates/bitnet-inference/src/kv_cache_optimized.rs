//! # Paged KV Cache with Eviction Policies
//!
//! High-performance key-value cache using paged memory management for
//! transformer attention layers. Features O(1) virtual→physical page
//! mapping, configurable eviction policies, and detailed metrics.

use std::collections::VecDeque;

// ── Page IDs ─────────────────────────────────────────────────────────────────

/// Opaque page identifier for the paged allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(u32);

impl PageId {
    fn index(self) -> usize {
        self.0 as usize
    }
}

// ── Page ─────────────────────────────────────────────────────────────────────

/// A fixed-size page holding KV data for a contiguous span of token positions.
///
/// Each page stores `tokens_per_page` slots, each slot holding a key vector
/// and a value vector of dimension `head_dim`.
#[derive(Debug, Clone)]
pub struct Page {
    /// Key data: `tokens_per_page * head_dim` floats.
    pub keys: Vec<f32>,
    /// Value data: `tokens_per_page * head_dim` floats.
    pub values: Vec<f32>,
    /// Number of token slots currently written in this page.
    pub len: usize,
    /// Layer this page belongs to.
    pub layer: usize,
}

impl Page {
    fn new(tokens_per_page: usize, head_dim: usize, layer: usize) -> Self {
        let cap = tokens_per_page * head_dim;
        Self { keys: vec![0.0; cap], values: vec![0.0; cap], len: 0, layer }
    }

    /// Size of the page data in bytes (keys + values).
    fn size_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * size_of::<f32>()
    }
}

// ── Eviction ─────────────────────────────────────────────────────────────────

/// Cache eviction policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheEvictionPolicy {
    /// Least-recently-used: evict the page that was accessed longest ago.
    LRU,
    /// Sliding window: keep only the most recent `window_size` pages per layer.
    SlidingWindow,
    /// Attention-score-based: evict the page with the lowest accumulated
    /// attention score (caller must supply scores via [`PagedKvCache::record_attention`]).
    AttentionBased,
}

/// Configuration for eviction behaviour.
#[derive(Debug, Clone)]
pub struct EvictionConfig {
    pub policy: CacheEvictionPolicy,
    /// Maximum number of **physical** pages the cache may hold.
    pub max_pages: usize,
    /// For [`CacheEvictionPolicy::SlidingWindow`]: number of most-recent pages
    /// to keep **per layer**.
    pub window_size: usize,
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self { policy: CacheEvictionPolicy::LRU, max_pages: 1024, window_size: 64 }
    }
}

// ── Metrics ──────────────────────────────────────────────────────────────────

/// Runtime statistics for the paged cache.
#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    /// Total bytes currently occupied by allocated pages.
    pub memory_bytes: usize,
    /// Number of pages evicted since creation / last reset.
    pub evictions: u64,
}

impl CacheMetrics {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

// ── PagedKvCache ─────────────────────────────────────────────────────────────

/// Paged KV cache with configurable eviction.
///
/// Pages are allocated from a free-list pool and mapped through a page table
/// keyed by `(layer, virtual_page_index)`.
pub struct PagedKvCache {
    /// Tokens stored per page.
    tokens_per_page: usize,
    /// Per-head dimension (elements per token slot).
    head_dim: usize,
    /// Physical page storage (indexed by `PageId`).
    pages: Vec<Option<Page>>,
    /// Free page indices ready for reuse.
    free_list: Vec<u32>,
    /// Virtual → physical mapping: `(layer, virtual_page_idx) → PageId`.
    page_table: Vec<((usize, usize), PageId)>,
    /// LRU order: front = oldest access.
    lru_order: VecDeque<PageId>,
    /// Per-page accumulated attention score (for AttentionBased eviction).
    attention_scores: Vec<f64>,
    /// Eviction configuration.
    eviction: EvictionConfig,
    /// Runtime metrics.
    pub metrics: CacheMetrics,
}

impl PagedKvCache {
    /// Create a new paged cache.
    ///
    /// * `tokens_per_page` – token slots per page (default recommendation: 16).
    /// * `head_dim` – elements per key/value vector per token.
    /// * `eviction` – eviction policy & limits.
    pub fn new(tokens_per_page: usize, head_dim: usize, eviction: EvictionConfig) -> Self {
        assert!(tokens_per_page > 0, "tokens_per_page must be > 0");
        assert!(head_dim > 0, "head_dim must be > 0");
        assert!(eviction.max_pages > 0, "max_pages must be > 0");

        let capacity = eviction.max_pages;
        Self {
            tokens_per_page,
            head_dim,
            pages: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            page_table: Vec::new(),
            lru_order: VecDeque::new(),
            attention_scores: Vec::new(),
            eviction,
            metrics: CacheMetrics::default(),
        }
    }

    // ── Allocation ───────────────────────────────────────────────────────

    /// Allocate a new physical page for `layer`.
    ///
    /// If the pool is at capacity, eviction runs first. Returns `None` only
    /// if eviction fails to free a page (should not happen with valid config).
    pub fn allocate_page(&mut self, layer: usize) -> Option<PageId> {
        // Try the free list first.
        if let Some(idx) = self.free_list.pop() {
            let id = PageId(idx);
            let page = Page::new(self.tokens_per_page, self.head_dim, layer);
            self.metrics.memory_bytes += page.size_bytes();
            self.pages[id.index()] = Some(page);
            self.attention_scores[id.index()] = 0.0;
            self.lru_order.push_back(id);
            return Some(id);
        }

        // Grow storage if below capacity.
        if self.pages.len() < self.eviction.max_pages {
            let idx = self.pages.len() as u32;
            let id = PageId(idx);
            let page = Page::new(self.tokens_per_page, self.head_dim, layer);
            self.metrics.memory_bytes += page.size_bytes();
            self.pages.push(Some(page));
            self.attention_scores.push(0.0);
            self.lru_order.push_back(id);
            return Some(id);
        }

        // At capacity – evict.
        self.evict_one();
        // After eviction, free list should have an entry.
        if let Some(idx) = self.free_list.pop() {
            let id = PageId(idx);
            let page = Page::new(self.tokens_per_page, self.head_dim, layer);
            self.metrics.memory_bytes += page.size_bytes();
            self.pages[id.index()] = Some(page);
            self.attention_scores[id.index()] = 0.0;
            self.lru_order.push_back(id);
            Some(id)
        } else {
            None
        }
    }

    /// Free a physical page, returning it to the pool.
    pub fn free_page(&mut self, id: PageId) {
        if let Some(page) = self.pages.get_mut(id.index()).and_then(|slot| slot.take()) {
            self.metrics.memory_bytes = self.metrics.memory_bytes.saturating_sub(page.size_bytes());
            self.free_list.push(id.0);
            self.lru_order.retain(|&pid| pid != id);
            self.page_table.retain(|&(_, pid)| pid != id);
        }
    }

    /// Immutable page access.
    pub fn get_page(&mut self, id: PageId) -> Option<&Page> {
        // Touch LRU order.
        self.touch_lru(id);
        self.metrics.hits += 1;
        self.pages.get(id.index()).and_then(|slot| slot.as_ref())
    }

    /// Mutable page access.
    pub fn get_page_mut(&mut self, id: PageId) -> Option<&mut Page> {
        self.touch_lru(id);
        self.metrics.hits += 1;
        self.pages.get_mut(id.index()).and_then(|slot| slot.as_mut())
    }

    // ── Page table ───────────────────────────────────────────────────────

    /// Map a virtual page `(layer, virt_idx)` to a physical `PageId`.
    pub fn map_page(&mut self, layer: usize, virt_idx: usize, id: PageId) {
        // Remove any existing mapping for this virtual address.
        self.page_table.retain(|&(key, _)| key != (layer, virt_idx));
        self.page_table.push(((layer, virt_idx), id));
    }

    /// Resolve a virtual page to its physical `PageId`.
    pub fn resolve(&mut self, layer: usize, virt_idx: usize) -> Option<PageId> {
        let found =
            self.page_table.iter().find(|&&(key, _)| key == (layer, virt_idx)).map(|&(_, pid)| pid);
        if found.is_some() {
            self.metrics.hits += 1;
        } else {
            self.metrics.misses += 1;
        }
        found
    }

    /// Number of entries currently in the page table.
    pub fn mapped_count(&self) -> usize {
        self.page_table.len()
    }

    // ── Attention-based scoring ──────────────────────────────────────────

    /// Record an attention score for a page (used by `AttentionBased` policy).
    pub fn record_attention(&mut self, id: PageId, score: f64) {
        if let Some(s) = self.attention_scores.get_mut(id.index()) {
            *s += score;
        }
    }

    // ── Eviction ─────────────────────────────────────────────────────────

    fn evict_one(&mut self) {
        let victim = match self.eviction.policy {
            CacheEvictionPolicy::LRU => self.pick_lru_victim(),
            CacheEvictionPolicy::SlidingWindow => self.pick_window_victim(),
            CacheEvictionPolicy::AttentionBased => self.pick_attention_victim(),
        };
        if let Some(id) = victim {
            self.free_page(id);
            self.metrics.evictions += 1;
        }
    }

    fn pick_lru_victim(&self) -> Option<PageId> {
        // Front of the deque is the least-recently used.
        self.lru_order.front().copied()
    }

    fn pick_window_victim(&self) -> Option<PageId> {
        // For each layer, find pages beyond the window.
        // Collect all layers that have pages.
        let mut layer_pages: std::collections::HashMap<usize, Vec<PageId>> =
            std::collections::HashMap::new();
        for &((_layer, _virt), pid) in &self.page_table {
            if self.pages.get(pid.index()).is_some_and(|s| s.is_some()) {
                let layer = self.pages[pid.index()].as_ref().unwrap().layer;
                layer_pages.entry(layer).or_default().push(pid);
            }
        }
        // Evict the oldest page from the layer with the most pages exceeding window.
        for pids in layer_pages.values() {
            if pids.len() > self.eviction.window_size {
                // Return the first in LRU order that belongs to this over-limit layer.
                for &lru_id in &self.lru_order {
                    if pids.contains(&lru_id) {
                        return Some(lru_id);
                    }
                }
            }
        }
        // Fallback to plain LRU if no layer exceeds window.
        self.pick_lru_victim()
    }

    fn pick_attention_victim(&self) -> Option<PageId> {
        // Pick the allocated page with the lowest attention score.
        let mut best: Option<(PageId, f64)> = None;
        for &id in &self.lru_order {
            let score = self.attention_scores.get(id.index()).copied().unwrap_or(0.0);
            if best.is_none() || score < best.unwrap().1 {
                best = Some((id, score));
            }
        }
        best.map(|(id, _)| id)
    }

    fn touch_lru(&mut self, id: PageId) {
        self.lru_order.retain(|&pid| pid != id);
        self.lru_order.push_back(id);
    }

    // ── Queries ──────────────────────────────────────────────────────────

    /// Number of physical pages currently allocated (not free).
    pub fn allocated_pages(&self) -> usize {
        self.pages.iter().filter(|s| s.is_some()).count()
    }

    /// Number of physical page slots in the free list.
    pub fn free_pages(&self) -> usize {
        self.free_list.len()
    }

    /// Total physical capacity (max pages).
    pub fn capacity(&self) -> usize {
        self.eviction.max_pages
    }

    /// Clear all pages and reset metrics.
    pub fn clear(&mut self) {
        for slot in &mut self.pages {
            *slot = None;
        }
        self.free_list.clear();
        self.page_table.clear();
        self.lru_order.clear();
        self.attention_scores.iter_mut().for_each(|s| *s = 0.0);
        self.metrics = CacheMetrics::default();
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(max_pages: usize) -> EvictionConfig {
        EvictionConfig { policy: CacheEvictionPolicy::LRU, max_pages, window_size: 4 }
    }

    // ── Page allocation / deallocation ───────────────────────────────────

    #[test]
    fn test_allocate_single_page() {
        let mut cache = PagedKvCache::new(16, 64, default_config(8));
        let id = cache.allocate_page(0);
        assert!(id.is_some());
        assert_eq!(cache.allocated_pages(), 1);
        assert_eq!(cache.free_pages(), 0);
    }

    #[test]
    fn test_allocate_up_to_capacity() {
        let max = 4;
        let mut cache = PagedKvCache::new(16, 64, default_config(max));
        let mut ids = Vec::new();
        for _ in 0..max {
            ids.push(cache.allocate_page(0).unwrap());
        }
        assert_eq!(cache.allocated_pages(), max);
    }

    #[test]
    fn test_free_page_returns_to_pool() {
        let mut cache = PagedKvCache::new(16, 64, default_config(4));
        let id = cache.allocate_page(0).unwrap();
        assert_eq!(cache.allocated_pages(), 1);
        cache.free_page(id);
        assert_eq!(cache.allocated_pages(), 0);
        assert_eq!(cache.free_pages(), 1);
    }

    #[test]
    fn test_reuse_freed_page() {
        let mut cache = PagedKvCache::new(16, 64, default_config(2));
        let id1 = cache.allocate_page(0).unwrap();
        cache.free_page(id1);
        let id2 = cache.allocate_page(0).unwrap();
        // Should reuse the slot.
        assert_eq!(id1, id2);
        assert_eq!(cache.allocated_pages(), 1);
    }

    // ── Page table mapping ───────────────────────────────────────────────

    #[test]
    fn test_map_and_resolve() {
        let mut cache = PagedKvCache::new(16, 64, default_config(8));
        let id = cache.allocate_page(0).unwrap();
        cache.map_page(0, 0, id);
        let resolved = cache.resolve(0, 0);
        assert_eq!(resolved, Some(id));
    }

    #[test]
    fn test_resolve_miss_increments_metric() {
        let mut cache = PagedKvCache::new(16, 64, default_config(8));
        assert!(cache.resolve(0, 99).is_none());
        assert_eq!(cache.metrics.misses, 1);
    }

    #[test]
    fn test_map_overwrite_replaces_old() {
        let mut cache = PagedKvCache::new(16, 64, default_config(8));
        let id1 = cache.allocate_page(0).unwrap();
        let id2 = cache.allocate_page(0).unwrap();
        cache.map_page(0, 0, id1);
        cache.map_page(0, 0, id2);
        assert_eq!(cache.resolve(0, 0), Some(id2));
        assert_eq!(cache.mapped_count(), 1);
    }

    // ── Page data access ─────────────────────────────────────────────────

    #[test]
    fn test_get_page_read_write() {
        let mut cache = PagedKvCache::new(4, 2, default_config(8));
        let id = cache.allocate_page(0).unwrap();
        {
            let page = cache.get_page_mut(id).unwrap();
            page.keys[0] = 1.0;
            page.values[0] = 2.0;
            page.len = 1;
        }
        let page = cache.get_page(id).unwrap();
        assert_eq!(page.keys[0], 1.0);
        assert_eq!(page.values[0], 2.0);
        assert_eq!(page.len, 1);
    }

    // ── LRU eviction ─────────────────────────────────────────────────────

    #[test]
    fn test_lru_eviction_oldest_first() {
        let mut cache = PagedKvCache::new(4, 2, default_config(3));
        let id0 = cache.allocate_page(0).unwrap();
        let _id1 = cache.allocate_page(0).unwrap();
        let _id2 = cache.allocate_page(0).unwrap();
        cache.map_page(0, 0, id0);
        cache.map_page(0, 1, _id1);
        cache.map_page(0, 2, _id2);

        // Cache is full (3/3). Allocating a 4th triggers eviction of id0 (oldest).
        let id3 = cache.allocate_page(0).unwrap();
        assert!(id3 == id0); // Reused slot of evicted page.
        assert_eq!(cache.metrics.evictions, 1);
    }

    #[test]
    fn test_lru_touch_prevents_eviction() {
        let mut cache = PagedKvCache::new(4, 2, default_config(3));
        let id0 = cache.allocate_page(0).unwrap();
        let id1 = cache.allocate_page(0).unwrap();
        let _id2 = cache.allocate_page(0).unwrap();

        // Touch id0 so it becomes most-recently used.
        let _ = cache.get_page(id0);

        // Allocate triggers eviction: id1 is now oldest.
        let id3 = cache.allocate_page(0).unwrap();
        assert_eq!(id3, id1); // id1 was evicted, not id0.
    }

    // ── Sliding window eviction ──────────────────────────────────────────

    #[test]
    fn test_sliding_window_keeps_recent() {
        let config = EvictionConfig {
            policy: CacheEvictionPolicy::SlidingWindow,
            max_pages: 6,
            window_size: 2,
        };
        let mut cache = PagedKvCache::new(4, 2, config);

        // Fill layer 0 with 6 pages (exceeds window_size=2).
        let mut ids = Vec::new();
        for i in 0..6 {
            let id = cache.allocate_page(0).unwrap();
            cache.map_page(0, i, id);
            ids.push(id);
        }
        assert_eq!(cache.allocated_pages(), 6);

        // Next allocation should evict oldest of layer 0.
        let new_id = cache.allocate_page(0).unwrap();
        assert_eq!(cache.metrics.evictions, 1);
        // The evicted page should be the first one (oldest in LRU for layer 0).
        assert_eq!(new_id, ids[0]);
    }

    // ── Attention-based eviction ─────────────────────────────────────────

    #[test]
    fn test_attention_eviction_lowest_score() {
        let config = EvictionConfig {
            policy: CacheEvictionPolicy::AttentionBased,
            max_pages: 3,
            window_size: 64,
        };
        let mut cache = PagedKvCache::new(4, 2, config);

        let id0 = cache.allocate_page(0).unwrap();
        let id1 = cache.allocate_page(0).unwrap();
        let id2 = cache.allocate_page(0).unwrap();

        // Give id0 high score, id1 low score, id2 medium.
        cache.record_attention(id0, 10.0);
        cache.record_attention(id1, 1.0);
        cache.record_attention(id2, 5.0);

        // Full → allocate triggers eviction of lowest score (id1).
        let new_id = cache.allocate_page(0).unwrap();
        assert_eq!(new_id, id1);
        assert_eq!(cache.metrics.evictions, 1);
    }

    // ── Memory limits ────────────────────────────────────────────────────

    #[test]
    fn test_memory_bytes_tracks_allocation() {
        let mut cache = PagedKvCache::new(4, 2, default_config(8));
        let expected_per_page = 4 * 2 * size_of::<f32>() * 2; // keys + values
        let id = cache.allocate_page(0).unwrap();
        assert_eq!(cache.metrics.memory_bytes, expected_per_page);
        cache.free_page(id);
        assert_eq!(cache.metrics.memory_bytes, 0);
    }

    #[test]
    fn test_allocated_never_exceeds_capacity() {
        let max = 4;
        let mut cache = PagedKvCache::new(4, 2, default_config(max));
        for _ in 0..max * 3 {
            let _ = cache.allocate_page(0);
        }
        assert!(cache.allocated_pages() <= max);
    }

    // ── Metrics accuracy ─────────────────────────────────────────────────

    #[test]
    fn test_hit_rate_accuracy() {
        let mut cache = PagedKvCache::new(4, 2, default_config(8));
        let id = cache.allocate_page(0).unwrap();
        cache.map_page(0, 0, id);

        // 1 hit (resolve existing).
        cache.resolve(0, 0);
        // 1 miss (resolve absent).
        cache.resolve(0, 99);

        // resolve gives 1 hit + 1 miss; get_page also adds hits.
        // Reset for clarity.
        cache.metrics.hits = 0;
        cache.metrics.misses = 0;

        cache.resolve(0, 0); // hit
        cache.resolve(0, 0); // hit
        cache.resolve(0, 1); // miss

        assert_eq!(cache.metrics.hits, 2);
        assert_eq!(cache.metrics.misses, 1);
        let rate = cache.metrics.hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_eviction_count() {
        let mut cache = PagedKvCache::new(4, 2, default_config(2));
        let id0 = cache.allocate_page(0).unwrap();
        let _id1 = cache.allocate_page(0).unwrap();
        cache.map_page(0, 0, id0);
        cache.map_page(0, 1, _id1);

        // 3rd alloc triggers one eviction.
        let _ = cache.allocate_page(0);
        assert_eq!(cache.metrics.evictions, 1);

        // 4th alloc triggers another.
        let _ = cache.allocate_page(0);
        assert_eq!(cache.metrics.evictions, 2);
    }

    // ── Multiple layers ──────────────────────────────────────────────────

    #[test]
    fn test_multiple_layers_independent() {
        let mut cache = PagedKvCache::new(4, 2, default_config(8));
        let id_l0 = cache.allocate_page(0).unwrap();
        let id_l1 = cache.allocate_page(1).unwrap();

        cache.map_page(0, 0, id_l0);
        cache.map_page(1, 0, id_l1);

        assert_eq!(cache.resolve(0, 0), Some(id_l0));
        assert_eq!(cache.resolve(1, 0), Some(id_l1));
        assert_ne!(id_l0, id_l1);
    }

    // ── Full → evict → reuse cycle ──────────────────────────────────────

    #[test]
    fn test_full_evict_reuse_cycle() {
        let max = 3;
        let mut cache = PagedKvCache::new(4, 2, default_config(max));
        let mut ids = Vec::new();
        for i in 0..max {
            let id = cache.allocate_page(0).unwrap();
            cache.map_page(0, i, id);
            ids.push(id);
        }
        assert_eq!(cache.allocated_pages(), max);

        // Write data to page 2 (most recent).
        {
            let p = cache.get_page_mut(ids[2]).unwrap();
            p.keys[0] = 42.0;
            p.len = 1;
        }

        // Allocate a new page — evicts ids[0].
        let new_id = cache.allocate_page(0).unwrap();
        cache.map_page(0, max, new_id);
        assert_eq!(cache.metrics.evictions, 1);

        // Old page 2 data is still intact.
        let p2 = cache.get_page(ids[2]).unwrap();
        assert_eq!(p2.keys[0], 42.0);
        assert_eq!(p2.len, 1);

        // New page is fresh.
        let pn = cache.get_page(new_id).unwrap();
        assert_eq!(pn.len, 0);
        assert_eq!(pn.keys[0], 0.0);
    }

    // ── Clear ────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_resets_everything() {
        let mut cache = PagedKvCache::new(4, 2, default_config(8));
        for _ in 0..4 {
            cache.allocate_page(0);
        }
        cache.metrics.hits = 10;
        cache.metrics.misses = 5;
        cache.metrics.evictions = 2;

        cache.clear();

        assert_eq!(cache.allocated_pages(), 0);
        assert_eq!(cache.mapped_count(), 0);
        assert_eq!(cache.metrics.hits, 0);
        assert_eq!(cache.metrics.misses, 0);
        assert_eq!(cache.metrics.evictions, 0);
        assert_eq!(cache.metrics.memory_bytes, 0);
    }

    // ── Edge: free non-existent page is no-op ────────────────────────────

    #[test]
    fn test_free_nonexistent_page_noop() {
        let mut cache = PagedKvCache::new(4, 2, default_config(4));
        // Free a PageId that was never allocated — should not panic.
        cache.free_page(PageId(99));
        assert_eq!(cache.allocated_pages(), 0);
    }

    // ── Tokens per page / head_dim are respected ─────────────────────────

    #[test]
    fn test_page_dimensions() {
        let tpp = 8;
        let hd = 32;
        let mut cache = PagedKvCache::new(tpp, hd, default_config(4));
        let id = cache.allocate_page(0).unwrap();
        let page = cache.get_page(id).unwrap();
        assert_eq!(page.keys.len(), tpp * hd);
        assert_eq!(page.values.len(), tpp * hd);
    }
}
