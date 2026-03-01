//! GPU-optimized key-value cache with page-based memory allocation.
//!
//! Provides [`GpuKvCache`] for efficient KV storage during autoregressive
//! inference, using fixed-size pages to reduce fragmentation and support
//! dynamic sequence growth.

use std::ops::Range;

/// Configuration for the KV cache.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    /// Number of sequence positions per page.
    pub page_size: usize,
}

/// Memory usage statistics for the KV cache.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheMemoryStats {
    pub total_bytes: usize,
    pub used_bytes: usize,
    pub page_count: usize,
    pub utilization_pct: f64,
}

/// Maps logical sequence positions to physical page indices.
#[derive(Debug, Clone)]
pub struct PageTable {
    /// For each layer: ordered list of page indices backing the sequence.
    layer_pages: Vec<Vec<usize>>,
}

impl PageTable {
    fn new(num_layers: usize) -> Self {
        Self { layer_pages: vec![Vec::new(); num_layers] }
    }

    /// Returns the page indices for the given layer.
    pub fn pages_for_layer(&self, layer: usize) -> &[usize] {
        &self.layer_pages[layer]
    }

    fn push_page(&mut self, layer: usize, page_idx: usize) {
        self.layer_pages[layer].push(page_idx);
    }

    fn trim_to(&mut self, layer: usize, page_count: usize) {
        self.layer_pages[layer].truncate(page_count);
    }

    fn clear_layer(&mut self, layer: usize) {
        self.layer_pages[layer].clear();
    }
}

/// A single page of KV storage for one layer.
#[derive(Debug, Clone)]
struct KvPage {
    /// Key data: up to `page_size * num_heads * head_dim` elements.
    k_data: Vec<f32>,
    /// Value data: same shape as `k_data`.
    v_data: Vec<f32>,
    /// Number of sequence positions actually written.
    len: usize,
    /// Maximum positions this page can hold.
    capacity: usize,
}

impl KvPage {
    fn new(page_size: usize, stride: usize) -> Self {
        let cap = page_size * stride;
        Self { k_data: vec![0.0; cap], v_data: vec![0.0; cap], len: 0, capacity: page_size }
    }

    const fn remaining(&self) -> usize {
        self.capacity - self.len
    }
}

/// GPU-optimized key-value cache using page-based allocation.
///
/// Memory is divided into fixed-size pages to reduce fragmentation and
/// support efficient trimming. Each layer maintains its own sequence of
/// pages via a [`PageTable`].
#[derive(Debug)]
pub struct GpuKvCache {
    config: KvCacheConfig,
    /// All pages across all layers: `pages[page_idx]`.
    pages: Vec<KvPage>,
    /// Tracks which pages are free for reuse.
    free_pages: Vec<usize>,
    /// Maps sequence positions to pages per layer.
    page_table: PageTable,
    /// Per-layer current sequence length.
    seq_lens: Vec<usize>,
    /// `stride` = `num_heads` × `head_dim` (elements per position).
    stride: usize,
}

impl GpuKvCache {
    /// Create a new KV cache with the given configuration.
    ///
    /// Pre-allocates pages for the full `max_seq_len` across all layers.
    ///
    /// # Panics
    ///
    /// Panics if `page_size` is zero.
    pub fn new(config: KvCacheConfig) -> Self {
        assert!(config.page_size > 0, "page_size must be > 0");

        let stride = config.num_heads * config.head_dim;
        let pages_per_layer = config.max_seq_len.div_ceil(config.page_size);
        let total_pages = config.num_layers * pages_per_layer;

        let pages: Vec<KvPage> =
            (0..total_pages).map(|_| KvPage::new(config.page_size, stride)).collect();

        let free_pages: Vec<usize> = (0..total_pages).rev().collect();

        let num_layers = config.num_layers;

        Self {
            config,
            pages,
            free_pages,
            page_table: PageTable::new(num_layers),
            seq_lens: vec![0; num_layers],
            stride,
        }
    }

    /// Append key-value data for one new position at the given layer.
    ///
    /// `k` and `v` must each have exactly `num_heads * head_dim` elements.
    ///
    /// # Panics
    ///
    /// Panics if the cache is full (no free pages and current page is full),
    /// or if `k`/`v` have the wrong length.
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.stride, "k length mismatch");
        assert_eq!(v.len(), self.stride, "v length mismatch");
        assert!(
            self.seq_lens[layer] < self.config.max_seq_len,
            "sequence length exceeds max_seq_len"
        );

        // Find or allocate a page with remaining capacity.
        let page_idx = self.ensure_page_for_layer(layer);
        let page = &mut self.pages[page_idx];
        let offset = page.len * self.stride;
        page.k_data[offset..offset + self.stride].copy_from_slice(k);
        page.v_data[offset..offset + self.stride].copy_from_slice(v);
        page.len += 1;

        self.seq_lens[layer] += 1;
    }

    /// Retrieve key and value slices for a range of positions at the given
    /// layer.
    ///
    /// Returns `(keys, values)` as contiguous `Vec<f32>`.
    pub fn get(&self, layer: usize, range: Range<usize>) -> (Vec<f32>, Vec<f32>) {
        assert!(range.end <= self.seq_lens[layer], "range out of bounds");

        let len = range.end - range.start;
        let mut keys = Vec::with_capacity(len * self.stride);
        let mut vals = Vec::with_capacity(len * self.stride);

        let pages = self.page_table.pages_for_layer(layer);
        for pos in range {
            let page_local = pos / self.config.page_size;
            let offset_in_page = pos % self.config.page_size;
            let page_idx = pages[page_local];
            let page = &self.pages[page_idx];
            let start = offset_in_page * self.stride;
            let end = start + self.stride;
            keys.extend_from_slice(&page.k_data[start..end]);
            vals.extend_from_slice(&page.v_data[start..end]);
        }

        (keys, vals)
    }

    /// Trim the cache at every layer to at most `max_len` positions,
    /// discarding the newest entries beyond that limit.
    pub fn trim(&mut self, max_len: usize) {
        for layer in 0..self.config.num_layers {
            if self.seq_lens[layer] <= max_len {
                continue;
            }

            let keep_pages = if max_len == 0 { 0 } else { max_len.div_ceil(self.config.page_size) };

            let pages = self.page_table.pages_for_layer(layer).to_vec();
            // Free pages beyond what we keep.
            for &page_idx in &pages[keep_pages..] {
                self.pages[page_idx].len = 0;
                self.free_pages.push(page_idx);
            }
            self.page_table.trim_to(layer, keep_pages);

            // Adjust the last kept page's length.
            if keep_pages > 0 {
                let remainder = max_len % self.config.page_size;
                if remainder != 0 {
                    let last_page_idx = self.page_table.pages_for_layer(layer)[keep_pages - 1];
                    self.pages[last_page_idx].len = remainder;
                }
            }

            self.seq_lens[layer] = max_len;
        }
    }

    /// Reset the cache, freeing all pages.
    pub fn clear(&mut self) {
        for layer in 0..self.config.num_layers {
            let pages = self.page_table.pages_for_layer(layer).to_vec();
            for page_idx in pages {
                self.pages[page_idx].len = 0;
                self.free_pages.push(page_idx);
            }
            self.page_table.clear_layer(layer);
            self.seq_lens[layer] = 0;
        }
    }

    /// Returns memory usage statistics.
    #[allow(clippy::cast_precision_loss)]
    pub fn memory_usage(&self) -> CacheMemoryStats {
        let bytes_per_page = self.config.page_size * self.stride * 2 * size_of::<f32>();
        let total_bytes = self.pages.len() * bytes_per_page;
        let used_pages = self.pages.len() - self.free_pages.len();
        let used_bytes = used_pages * bytes_per_page;
        let utilization_pct =
            if total_bytes == 0 { 0.0 } else { (used_bytes as f64 / total_bytes as f64) * 100.0 };

        CacheMemoryStats { total_bytes, used_bytes, page_count: self.pages.len(), utilization_pct }
    }

    /// Returns the current sequence length for the given layer.
    pub fn seq_len(&self, layer: usize) -> usize {
        self.seq_lens[layer]
    }

    /// Returns a reference to the page table.
    pub const fn page_table(&self) -> &PageTable {
        &self.page_table
    }

    /// Returns the cache configuration.
    pub const fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Ensure a page with remaining capacity exists for the layer,
    /// allocating one from the free list if needed. Returns the page index.
    fn ensure_page_for_layer(&mut self, layer: usize) -> usize {
        let pages = self.page_table.pages_for_layer(layer);
        if let Some(&last_idx) = pages.last()
            && self.pages[last_idx].remaining() > 0
        {
            return last_idx;
        }
        // Allocate a new page.
        let page_idx = self.free_pages.pop().expect("KV cache exhausted: no free pages");
        self.page_table.push_page(layer, page_idx);
        page_idx
    }
}
