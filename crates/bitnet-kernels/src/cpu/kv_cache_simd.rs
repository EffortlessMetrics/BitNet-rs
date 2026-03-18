//! SIMD-optimized paged KV cache with eviction policies.
//!
//! Extends the base [`super::kv_cache`] with page-table backed storage,
//! configurable eviction, and AVX2-accelerated gather/scatter/copy
//! operations for high-throughput incremental inference.
#![allow(unsafe_op_in_unsafe_fn)]

use bitnet_common::{BitNetError, KernelError, Result};

// ── Helpers ────────────────────────────────────────────────────────

fn invalid_arg(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for a SIMD-optimized paged KV cache.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub num_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Maximum sequence length the cache can hold.
    pub max_seq_len: usize,
    /// Number of tokens per page (must be > 0).
    pub page_size: usize,
}

impl KVCacheConfig {
    /// Validate configuration, returning an error on nonsensical values.
    pub fn validate(&self) -> Result<()> {
        if self.num_layers == 0 {
            return Err(invalid_arg("num_layers must be > 0"));
        }
        if self.num_heads == 0 {
            return Err(invalid_arg("num_heads must be > 0"));
        }
        if self.head_dim == 0 {
            return Err(invalid_arg("head_dim must be > 0"));
        }
        if self.max_seq_len == 0 {
            return Err(invalid_arg("max_seq_len must be > 0"));
        }
        if self.page_size == 0 {
            return Err(invalid_arg("page_size must be > 0"));
        }
        Ok(())
    }

    /// Elements per token across all heads.
    #[inline]
    fn token_elements(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Number of pages needed to cover `max_seq_len`.
    #[inline]
    fn pages_needed(&self) -> usize {
        self.max_seq_len.div_ceil(self.page_size)
    }
}

// ── Eviction policy ────────────────────────────────────────────────

/// Strategy used when the cache reaches capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictionPolicy {
    /// Least Recently Used — evict pages not accessed for the longest time.
    LRU,
    /// First In, First Out — evict the oldest pages by insertion order.
    FIFO,
    /// Sliding Window — keep only the most recent `window_size` tokens.
    SlidingWindow,
}

// ── Page ───────────────────────────────────────────────────────────

/// A fixed-size page holding keys and values for up to `capacity` tokens.
#[derive(Debug, Clone)]
struct Page {
    keys: Vec<f32>,
    values: Vec<f32>,
    /// Number of tokens currently stored in this page.
    used: usize,
    /// Maximum tokens this page can hold.
    capacity: usize,
    /// Elements per token (`num_heads * head_dim`).
    token_elements: usize,
    /// Monotonic counter set on every access (for LRU).
    last_access: u64,
    /// Monotonic counter set at creation (for FIFO).
    creation_order: u64,
}

impl Page {
    fn new(capacity: usize, token_elements: usize, creation_order: u64) -> Self {
        let total = capacity * token_elements;
        Self {
            keys: vec![0.0; total],
            values: vec![0.0; total],
            used: 0,
            capacity,
            token_elements,
            last_access: creation_order,
            creation_order,
        }
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.capacity - self.used
    }

    fn append(&mut self, new_keys: &[f32], new_values: &[f32]) -> Result<usize> {
        let new_tokens = new_keys.len() / self.token_elements;
        if new_keys.len() != new_tokens * self.token_elements {
            return Err(invalid_arg("key length not a multiple of token_elements"));
        }
        if new_values.len() != new_keys.len() {
            return Err(invalid_arg("key and value lengths must match"));
        }
        let to_store = new_tokens.min(self.remaining());
        if to_store == 0 {
            return Ok(0);
        }
        let n = to_store * self.token_elements;
        let off = self.used * self.token_elements;
        self.keys[off..off + n].copy_from_slice(&new_keys[..n]);
        self.values[off..off + n].copy_from_slice(&new_values[..n]);
        self.used += to_store;
        Ok(to_store)
    }

    fn clear(&mut self) {
        self.used = 0;
    }
}

// ── Per-layer state ────────────────────────────────────────────────

/// Page-table entry mapping a logical page index to a physical page.
#[derive(Debug, Clone)]
struct PageTableEntry {
    page_idx: usize,
}

/// Per-layer cache state with a page table.
#[derive(Debug, Clone)]
struct LayerCache {
    /// Physical page storage.
    pages: Vec<Page>,
    /// Logical-to-physical page mapping.
    page_table: Vec<PageTableEntry>,
    /// Current total number of cached tokens across all pages.
    seq_len: usize,
    /// Global access counter for LRU tracking.
    access_counter: u64,
    /// Global creation counter for FIFO ordering.
    creation_counter: u64,
    /// Eviction history: indices of evicted page table entries (for stats).
    eviction_count: usize,
}

impl LayerCache {
    fn new(num_pages: usize, page_size: usize, token_elements: usize) -> Self {
        let mut pages = Vec::with_capacity(num_pages);
        let mut page_table = Vec::with_capacity(num_pages);
        for i in 0..num_pages {
            pages.push(Page::new(page_size, token_elements, i as u64));
            page_table.push(PageTableEntry { page_idx: i });
        }
        Self {
            pages,
            page_table,
            seq_len: 0,
            access_counter: num_pages as u64,
            creation_counter: num_pages as u64,
            eviction_count: 0,
        }
    }

    /// Find the page and offset within page for a given token index.
    fn locate_token(&self, token_idx: usize, page_size: usize) -> Option<(usize, usize)> {
        if token_idx >= self.seq_len {
            return None;
        }
        let logical_page = token_idx / page_size;
        let offset_in_page = token_idx % page_size;
        if logical_page >= self.page_table.len() {
            return None;
        }
        Some((self.page_table[logical_page].page_idx, offset_in_page))
    }

    fn touch(&mut self, phys_page: usize) {
        self.access_counter += 1;
        if phys_page < self.pages.len() {
            self.pages[phys_page].last_access = self.access_counter;
        }
    }
}

// ── Main cache ─────────────────────────────────────────────────────

/// SIMD-optimized paged KV cache.
///
/// Uses a page-table backed design where each layer independently
/// manages pages of cached key/value pairs. AVX2 intrinsics accelerate
/// gather, scatter, and bulk copy when available at runtime.
#[derive(Debug, Clone)]
pub struct PagedKVCache {
    /// Per-layer caches.
    layers: Vec<LayerCache>,
    /// Configuration snapshot.
    config: KVCacheConfig,
    /// Active eviction policy.
    eviction_policy: EvictionPolicy,
}

// ── Public kernel functions ────────────────────────────────────────

/// Create a new SIMD-optimized paged KV cache.
pub fn create_kv_cache(config: KVCacheConfig, policy: EvictionPolicy) -> Result<PagedKVCache> {
    config.validate()?;
    let num_pages = config.pages_needed();
    let te = config.token_elements();
    let layers =
        (0..config.num_layers).map(|_| LayerCache::new(num_pages, config.page_size, te)).collect();
    Ok(PagedKVCache { layers, config, eviction_policy: policy })
}

/// Append key/value pairs to a single layer.
///
/// If the active page is full, new tokens spill into subsequent pages.
/// Returns the number of tokens actually appended.
pub fn append_kv(
    cache: &mut PagedKVCache,
    layer: usize,
    new_keys: &[f32],
    new_values: &[f32],
) -> Result<usize> {
    let te = cache.config.token_elements();
    let lc = cache.layers.get_mut(layer).ok_or_else(|| invalid_arg("layer index out of range"))?;

    let total_new = new_keys.len() / te;
    if new_keys.len() != total_new * te {
        return Err(invalid_arg("key length not a multiple of token_elements"));
    }
    if new_values.len() != new_keys.len() {
        return Err(invalid_arg("key and value lengths must match"));
    }

    let mut appended = 0usize;
    while appended < total_new {
        // Find the current active page (the one that still has room).
        let active_logical = lc.seq_len / cache.config.page_size;
        if active_logical >= lc.page_table.len() {
            break; // No more pages available.
        }
        let phys = lc.page_table[active_logical].page_idx;
        let start = appended * te;
        let remaining_tokens = total_new - appended;
        let end = start + remaining_tokens * te;
        let stored = lc.pages[phys].append(&new_keys[start..end], &new_values[start..end])?;
        if stored == 0 {
            break;
        }
        lc.touch(phys);
        appended += stored;
        lc.seq_len += stored;
    }
    Ok(appended)
}

/// Look up keys and values for tokens `[start, end)` at `layer`.
///
/// Copies data out of pages into contiguous output vectors.
pub fn lookup_kv(
    cache: &mut PagedKVCache,
    layer: usize,
    start: usize,
    end: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if start > end {
        return Err(invalid_arg("start must be <= end"));
    }
    let te = cache.config.token_elements();
    let page_size = cache.config.page_size;
    let lc = cache.layers.get_mut(layer).ok_or_else(|| invalid_arg("layer index out of range"))?;
    if end > lc.seq_len {
        return Err(invalid_arg("end exceeds current seq_len"));
    }

    let num_tokens = end - start;
    let mut out_keys = vec![0.0f32; num_tokens * te];
    let mut out_values = vec![0.0f32; num_tokens * te];

    for i in 0..num_tokens {
        let token_idx = start + i;
        let (phys, off) = lc
            .locate_token(token_idx, page_size)
            .ok_or_else(|| invalid_arg("token index out of range"))?;
        let src_off = off * te;
        let dst_off = i * te;
        out_keys[dst_off..dst_off + te]
            .copy_from_slice(&lc.pages[phys].keys[src_off..src_off + te]);
        out_values[dst_off..dst_off + te]
            .copy_from_slice(&lc.pages[phys].values[src_off..src_off + te]);
        lc.touch(phys);
    }
    Ok((out_keys, out_values))
}

/// Evict the oldest page from a layer according to the active policy.
///
/// Returns `true` if a page was evicted, `false` if no evictable pages remain.
pub fn evict_oldest(cache: &mut PagedKVCache, layer: usize) -> Result<bool> {
    let page_size = cache.config.page_size;
    let policy = cache.eviction_policy;
    let lc = cache.layers.get_mut(layer).ok_or_else(|| invalid_arg("layer index out of range"))?;

    if lc.seq_len == 0 {
        return Ok(false);
    }

    // Find the logical page to evict.
    let num_active = lc.seq_len.div_ceil(page_size);
    if num_active == 0 {
        return Ok(false);
    }

    let victim_logical = match policy {
        EvictionPolicy::LRU => {
            let mut min_access = u64::MAX;
            let mut victim = 0;
            for i in 0..num_active {
                let phys = lc.page_table[i].page_idx;
                if lc.pages[phys].last_access < min_access {
                    min_access = lc.pages[phys].last_access;
                    victim = i;
                }
            }
            victim
        }
        EvictionPolicy::FIFO => {
            let mut min_creation = u64::MAX;
            let mut victim = 0;
            for i in 0..num_active {
                let phys = lc.page_table[i].page_idx;
                if lc.pages[phys].creation_order < min_creation {
                    min_creation = lc.pages[phys].creation_order;
                    victim = i;
                }
            }
            victim
        }
        EvictionPolicy::SlidingWindow => {
            // Always evict the first (oldest) logical page.
            0
        }
    };

    let phys = lc.page_table[victim_logical].page_idx;
    let tokens_freed = lc.pages[phys].used;
    lc.pages[phys].clear();
    lc.pages[phys].creation_order = lc.creation_counter;
    lc.creation_counter += 1;
    lc.seq_len = lc.seq_len.saturating_sub(tokens_freed);
    lc.eviction_count += 1;

    // Rotate the evicted logical page entry to the end of active range.
    let entry = lc.page_table.remove(victim_logical);
    lc.page_table.push(entry);

    Ok(true)
}

/// Compact a layer's cache by eliminating gaps left by eviction.
///
/// After compaction, all cached tokens are packed into the fewest
/// possible pages and `seq_len` accurately reflects the stored count.
pub fn compact_cache(cache: &mut PagedKVCache, layer: usize) -> Result<()> {
    let te = cache.config.token_elements();
    let page_size = cache.config.page_size;
    let lc = cache.layers.get_mut(layer).ok_or_else(|| invalid_arg("layer index out of range"))?;

    // Collect all live tokens in order.
    let mut all_keys = Vec::new();
    let mut all_values = Vec::new();
    for entry in &lc.page_table {
        let page = &lc.pages[entry.page_idx];
        if page.used > 0 {
            let n = page.used * te;
            all_keys.extend_from_slice(&page.keys[..n]);
            all_values.extend_from_slice(&page.values[..n]);
        }
    }

    // Clear all pages.
    for page in &mut lc.pages {
        page.clear();
    }
    lc.seq_len = 0;

    // Re-insert tokens into pages sequentially.
    let total_tokens = all_keys.len() / te;
    let mut written = 0;
    for entry in &lc.page_table {
        if written >= total_tokens {
            break;
        }
        let phys = entry.page_idx;
        let to_write = (total_tokens - written).min(page_size);
        let start = written * te;
        let end = start + to_write * te;
        lc.pages[phys]
            .append(&all_keys[start..end], &all_values[start..end])
            .expect("compaction re-insert should not fail");
        written += to_write;
    }
    lc.seq_len = written;
    Ok(())
}

/// Append key/value pairs across all layers simultaneously.
///
/// `keys_per_layer` and `values_per_layer` must have one entry per layer.
pub fn batch_append_kv(
    cache: &mut PagedKVCache,
    keys_per_layer: &[&[f32]],
    values_per_layer: &[&[f32]],
) -> Result<Vec<usize>> {
    let nl = cache.config.num_layers;
    if keys_per_layer.len() != nl {
        return Err(invalid_arg("keys_per_layer length != num_layers"));
    }
    if values_per_layer.len() != nl {
        return Err(invalid_arg("values_per_layer length != num_layers"));
    }
    let mut counts = Vec::with_capacity(nl);
    for layer in 0..nl {
        counts.push(append_kv(cache, layer, keys_per_layer[layer], values_per_layer[layer])?);
    }
    Ok(counts)
}

/// Scatter values from a contiguous buffer into non-contiguous cache
/// positions specified by `indices`.
///
/// Uses AVX2 when available for the copy inner loop.
pub fn scatter_kv(
    cache: &mut PagedKVCache,
    layer: usize,
    indices: &[usize],
    keys: &[f32],
    values: &[f32],
) -> Result<()> {
    let te = cache.config.token_elements();
    let page_size = cache.config.page_size;
    if keys.len() != indices.len() * te {
        return Err(invalid_arg("keys length != indices.len() * token_elements"));
    }
    if values.len() != keys.len() {
        return Err(invalid_arg("keys and values lengths must match"));
    }
    let lc = cache.layers.get_mut(layer).ok_or_else(|| invalid_arg("layer index out of range"))?;

    for (i, &token_idx) in indices.iter().enumerate() {
        let (phys, off) = lc
            .locate_token(token_idx, page_size)
            .ok_or_else(|| invalid_arg("scatter index out of range"))?;
        let src_off = i * te;
        let dst_off = off * te;
        simd_copy_f32(
            &keys[src_off..src_off + te],
            &mut lc.pages[phys].keys[dst_off..dst_off + te],
        );
        simd_copy_f32(
            &values[src_off..src_off + te],
            &mut lc.pages[phys].values[dst_off..dst_off + te],
        );
        lc.touch(phys);
    }
    Ok(())
}

/// Gather values from non-contiguous cache positions into a contiguous
/// output buffer.
///
/// Uses AVX2 when available for the copy inner loop.
pub fn gather_kv(
    cache: &mut PagedKVCache,
    layer: usize,
    indices: &[usize],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let te = cache.config.token_elements();
    let page_size = cache.config.page_size;
    let lc = cache.layers.get_mut(layer).ok_or_else(|| invalid_arg("layer index out of range"))?;

    let n = indices.len();
    let mut out_keys = vec![0.0f32; n * te];
    let mut out_values = vec![0.0f32; n * te];

    for (i, &token_idx) in indices.iter().enumerate() {
        let (phys, off) = lc
            .locate_token(token_idx, page_size)
            .ok_or_else(|| invalid_arg("gather index out of range"))?;
        let src_off = off * te;
        let dst_off = i * te;
        simd_copy_f32(
            &lc.pages[phys].keys[src_off..src_off + te],
            &mut out_keys[dst_off..dst_off + te],
        );
        simd_copy_f32(
            &lc.pages[phys].values[src_off..src_off + te],
            &mut out_values[dst_off..dst_off + te],
        );
        lc.touch(phys);
    }
    Ok((out_keys, out_values))
}

/// Return cache utilisation statistics.
pub fn cache_stats(cache: &PagedKVCache) -> CacheStats {
    let mut total_pages = 0usize;
    let mut used_pages = 0usize;
    let mut total_tokens = 0usize;
    let mut max_seq_len = 0usize;
    let mut total_evictions = 0usize;
    let mut total_bytes = 0usize;

    for lc in &cache.layers {
        for entry in &lc.page_table {
            total_pages += 1;
            let p = &lc.pages[entry.page_idx];
            if p.used > 0 {
                used_pages += 1;
            }
            total_bytes += (p.keys.len() + p.values.len()) * size_of::<f32>();
        }
        total_tokens += lc.seq_len;
        max_seq_len = max_seq_len.max(lc.seq_len);
        total_evictions += lc.eviction_count;
    }

    CacheStats {
        total_pages,
        used_pages,
        total_tokens,
        max_seq_len,
        total_evictions,
        memory_bytes: total_bytes,
        num_layers: cache.layers.len(),
    }
}

/// Clear all cached tokens in every layer, resetting to empty.
pub fn clear_cache(cache: &mut PagedKVCache) {
    for lc in &mut cache.layers {
        for page in &mut lc.pages {
            page.clear();
        }
        lc.seq_len = 0;
    }
}

// ── Stats struct ───────────────────────────────────────────────────

/// Snapshot of cache utilisation metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStats {
    /// Total allocated pages across all layers.
    pub total_pages: usize,
    /// Pages with at least one stored token.
    pub used_pages: usize,
    /// Total cached tokens summed across layers.
    pub total_tokens: usize,
    /// Maximum single-layer sequence length.
    pub max_seq_len: usize,
    /// Total evictions performed across all layers.
    pub total_evictions: usize,
    /// Total memory occupied by page buffers in bytes.
    pub memory_bytes: usize,
    /// Number of layers.
    pub num_layers: usize,
}

// ── SIMD-accelerated copy ──────────────────────────────────────────

/// Copy `src` into `dst` using AVX2 when available, scalar fallback otherwise.
#[inline]
fn simd_copy_f32(src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    let _len = src.len();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && _len >= 8 {
            // Safety: we checked AVX2 availability and length >= 8.
            unsafe {
                avx2_copy_f32(src, dst);
            }
            return;
        }
    }

    dst.copy_from_slice(src);
}

/// AVX2-accelerated f32 copy: 8-wide loads/stores with scalar tail.
///
/// # Safety
/// Caller must ensure AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_copy_f32(src: &[f32], dst: &mut [f32]) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let len = src.len();
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        let v = _mm256_loadu_ps(src.as_ptr().add(base));
        _mm256_storeu_ps(dst.as_mut_ptr().add(base), v);
    }

    let tail_start = chunks * 8;
    dst[tail_start..tail_start + remainder]
        .copy_from_slice(&src[tail_start..tail_start + remainder]);
}

/// AVX2-accelerated dot product of two f32 slices.
///
/// Used for attention-score computation over cached keys.
#[inline]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() >= 8 {
            // Safety: we checked AVX2 availability and length >= 8.
            unsafe {
                return avx2_dot_f32(a, b);
            }
        }
    }

    scalar_dot_f32(a, b)
}

/// Scalar fallback dot product.
#[inline]
pub fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// AVX2 dot product: 8-wide FMA with horizontal sum.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;
    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb = _mm256_loadu_ps(b.as_ptr().add(base));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    // Horizontal sum of 256-bit accumulator.
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    let mut result = _mm_cvtss_f32(sum32);

    // Scalar tail.
    let tail_start = chunks * 8;
    for i in tail_start..len {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    result
}

/// AVX2-accelerated bulk scale: multiply every element by `scale`.
#[inline]
pub fn simd_scale_f32(data: &mut [f32], scale: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && data.len() >= 8 {
            unsafe {
                avx2_scale_f32(data, scale);
            }
            return;
        }
    }

    for v in data.iter_mut() {
        *v *= scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_scale_f32(data: &mut [f32], scale: f32) {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    let len = data.len();
    let chunks = len / 8;
    let scale_vec = _mm256_set1_ps(scale);

    for i in 0..chunks {
        let base = i * 8;
        let v = _mm256_loadu_ps(data.as_ptr().add(base));
        let scaled = _mm256_mul_ps(v, scale_vec);
        _mm256_storeu_ps(data.as_mut_ptr().add(base), scaled);
    }

    let tail_start = chunks * 8;
    for i in tail_start..len {
        *data.get_unchecked_mut(i) *= scale;
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────

    fn small_config() -> KVCacheConfig {
        KVCacheConfig { num_layers: 2, num_heads: 4, head_dim: 8, max_seq_len: 16, page_size: 4 }
    }

    fn tiny_config() -> KVCacheConfig {
        KVCacheConfig { num_layers: 1, num_heads: 1, head_dim: 2, max_seq_len: 8, page_size: 2 }
    }

    fn make_token(cfg: &KVCacheConfig, val: f32) -> (Vec<f32>, Vec<f32>) {
        let te = cfg.token_elements();
        (vec![val; te], vec![val * 10.0; te])
    }

    fn make_tokens(cfg: &KVCacheConfig, n: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
        let te = cfg.token_elements();
        let keys: Vec<f32> = (0..n).flat_map(|i| vec![base + i as f32; te]).collect();
        let values: Vec<f32> = (0..n).flat_map(|i| vec![(base + i as f32) * 10.0; te]).collect();
        (keys, values)
    }

    // ── Config validation ─────────────────────────────────────────

    #[test]
    fn test_config_valid() {
        assert!(small_config().validate().is_ok());
    }

    #[test]
    fn test_config_zero_layers() {
        let mut c = small_config();
        c.num_layers = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_heads() {
        let mut c = small_config();
        c.num_heads = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_head_dim() {
        let mut c = small_config();
        c.head_dim = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_max_seq_len() {
        let mut c = small_config();
        c.max_seq_len = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_page_size() {
        let mut c = small_config();
        c.page_size = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_token_elements() {
        let c = small_config();
        assert_eq!(c.token_elements(), 32); // 4 * 8
    }

    #[test]
    fn test_config_pages_needed() {
        let c = small_config(); // max_seq_len=16, page_size=4
        assert_eq!(c.pages_needed(), 4);
    }

    #[test]
    fn test_config_pages_needed_non_divisible() {
        let mut c = small_config();
        c.max_seq_len = 15; // 15 / 4 = 3.75 → 4
        assert_eq!(c.pages_needed(), 4);
    }

    // ── Cache creation ────────────────────────────────────────────

    #[test]
    fn test_create_cache() {
        let cache = create_kv_cache(small_config(), EvictionPolicy::LRU).unwrap();
        assert_eq!(cache.layers.len(), 2);
    }

    #[test]
    fn test_create_cache_initial_empty() {
        let cache = create_kv_cache(small_config(), EvictionPolicy::LRU).unwrap();
        let stats = cache_stats(&cache);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.used_pages, 0);
        assert_eq!(stats.num_layers, 2);
    }

    #[test]
    fn test_create_cache_page_count() {
        let cache = create_kv_cache(small_config(), EvictionPolicy::LRU).unwrap();
        let stats = cache_stats(&cache);
        // 2 layers × 4 pages each = 8
        assert_eq!(stats.total_pages, 8);
    }

    #[test]
    fn test_create_invalid_config() {
        let mut c = small_config();
        c.num_layers = 0;
        assert!(create_kv_cache(c, EvictionPolicy::LRU).is_err());
    }

    // ── Append ────────────────────────────────────────────────────

    #[test]
    fn test_append_single_token() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_token(&cfg, 1.0);
        let n = append_kv(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(n, 1);
        let stats = cache_stats(&cache);
        assert_eq!(stats.total_tokens, 1);
    }

    #[test]
    fn test_append_multiple_tokens() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 5, 1.0);
        let n = append_kv(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(n, 5);
    }

    #[test]
    fn test_append_fills_pages() {
        let cfg = small_config(); // page_size=4, max_seq_len=16
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 16, 1.0);
        let n = append_kv(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(n, 16);
        let stats = cache_stats(&cache);
        assert_eq!(stats.used_pages, 4 + 0); // 4 pages used in layer 0
    }

    #[test]
    fn test_append_exceeds_capacity() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 20, 1.0); // 20 > max_seq_len=16
        let n = append_kv(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(n, 16); // Capped at capacity.
    }

    #[test]
    fn test_append_bad_key_alignment() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let k = [0.0; 5]; // Not a multiple of 32.
        let v = [0.0; 5];
        assert!(append_kv(&mut cache, 0, &k, &v).is_err());
    }

    #[test]
    fn test_append_mismatched_kv() {
        let cfg = small_config();
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        let k = vec![0.0; te];
        let v = vec![0.0; te * 2];
        assert!(append_kv(&mut cache, 0, &k, &v).is_err());
    }

    #[test]
    fn test_append_invalid_layer() {
        let cfg = small_config();
        let (k, v) = make_token(&cfg, 1.0);
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        assert!(append_kv(&mut cache, 99, &k, &v).is_err());
    }

    #[test]
    fn test_append_incremental() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        for i in 0..6 {
            let (k, v) = make_token(&cfg, i as f32);
            append_kv(&mut cache, 0, &k, &v).unwrap();
        }
        let stats = cache_stats(&cache);
        assert_eq!(stats.total_tokens, 6);
    }

    #[test]
    fn test_append_spans_page_boundary() {
        let cfg = tiny_config(); // page_size=2
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 3, 1.0); // 3 tokens → pages 0 and 1
        let n = append_kv(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(n, 3);
    }

    // ── Lookup ────────────────────────────────────────────────────

    #[test]
    fn test_lookup_single_token() {
        let cfg = small_config();
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_token(&cfg, 42.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        let (rk, rv) = lookup_kv(&mut cache, 0, 0, 1).unwrap();
        assert_eq!(rk.len(), te);
        assert!((rk[0] - 42.0).abs() < f32::EPSILON);
        assert!((rv[0] - 420.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lookup_range() {
        let cfg = small_config();
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        let (rk, _rv) = lookup_kv(&mut cache, 0, 1, 3).unwrap();
        assert_eq!(rk.len(), 2 * te);
        // Token 1 should have base value 2.0.
        assert!((rk[0] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lookup_across_pages() {
        let cfg = tiny_config(); // page_size=2
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 10.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        let (rk, _) = lookup_kv(&mut cache, 0, 0, 4).unwrap();
        assert_eq!(rk.len(), 4 * te);
        // Token 2 is on page 1, value = 12.0
        assert!((rk[2 * te] - 12.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_lookup_empty_range() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_token(&cfg, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        let (rk, rv) = lookup_kv(&mut cache, 0, 0, 0).unwrap();
        assert!(rk.is_empty());
        assert!(rv.is_empty());
    }

    #[test]
    fn test_lookup_start_gt_end() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_token(&cfg, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        assert!(lookup_kv(&mut cache, 0, 1, 0).is_err());
    }

    #[test]
    fn test_lookup_exceeds_seq_len() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_token(&cfg, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        assert!(lookup_kv(&mut cache, 0, 0, 5).is_err());
    }

    #[test]
    fn test_lookup_invalid_layer() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        assert!(lookup_kv(&mut cache, 99, 0, 0).is_err());
    }

    // ── Eviction ──────────────────────────────────────────────────

    #[test]
    fn test_evict_lru_frees_tokens() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        assert!(evict_oldest(&mut cache, 0).unwrap());
        let stats = cache_stats(&cache);
        assert!(stats.total_tokens < 4);
        assert_eq!(stats.total_evictions, 1);
    }

    #[test]
    fn test_evict_fifo() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::FIFO).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        assert!(evict_oldest(&mut cache, 0).unwrap());
        let stats = cache_stats(&cache);
        assert!(stats.total_tokens < 4);
    }

    #[test]
    fn test_evict_sliding_window() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::SlidingWindow).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        assert!(evict_oldest(&mut cache, 0).unwrap());
        // Sliding window always evicts first logical page.
        let stats = cache_stats(&cache);
        assert!(stats.total_tokens < 4);
    }

    #[test]
    fn test_evict_empty_cache() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        assert!(!evict_oldest(&mut cache, 0).unwrap());
    }

    #[test]
    fn test_evict_invalid_layer() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        assert!(evict_oldest(&mut cache, 99).is_err());
    }

    #[test]
    fn test_evict_multiple_rounds() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 8, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        for _ in 0..3 {
            evict_oldest(&mut cache, 0).unwrap();
        }
        let stats = cache_stats(&cache);
        assert_eq!(stats.total_evictions, 3);
    }

    #[test]
    fn test_evict_lru_respects_access() {
        let cfg = tiny_config(); // page_size=2
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        // Fill 4 tokens → 2 pages.
        let (k, v) = make_tokens(&cfg, 4, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        // Access page containing token 0 to make it "recent".
        lookup_kv(&mut cache, 0, 0, 1).unwrap();
        // Evict — should evict page 1 (less recently accessed).
        evict_oldest(&mut cache, 0).unwrap();
        // Token 0 should still be available after compaction.
        compact_cache(&mut cache, 0).unwrap();
        let (rk, _) = lookup_kv(&mut cache, 0, 0, 1).unwrap();
        assert!((rk[0] - 1.0).abs() < f32::EPSILON);
    }

    // ── Compact ───────────────────────────────────────────────────

    #[test]
    fn test_compact_after_evict() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        evict_oldest(&mut cache, 0).unwrap();
        compact_cache(&mut cache, 0).unwrap();
        let stats = cache_stats(&cache);
        // After eviction of 1 page (2 tokens) from 4 tokens → 2 remain.
        assert_eq!(stats.total_tokens, 2);
    }

    #[test]
    fn test_compact_empty() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        compact_cache(&mut cache, 0).unwrap();
        let stats = cache_stats(&cache);
        assert_eq!(stats.total_tokens, 0);
    }

    #[test]
    fn test_compact_invalid_layer() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        assert!(compact_cache(&mut cache, 99).is_err());
    }

    #[test]
    fn test_compact_preserves_data() {
        let cfg = tiny_config();
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::FIFO).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        evict_oldest(&mut cache, 0).unwrap();
        compact_cache(&mut cache, 0).unwrap();
        // After evicting oldest page (tokens 0,1), tokens 2,3 remain.
        let (rk, _) = lookup_kv(&mut cache, 0, 0, 2).unwrap();
        // Compacted tokens should have values 3.0, 4.0.
        assert!((rk[0] - 3.0).abs() < f32::EPSILON);
        assert!((rk[te] - 4.0).abs() < f32::EPSILON);
    }

    // ── Batch append ──────────────────────────────────────────────

    #[test]
    fn test_batch_append() {
        let cfg = small_config(); // 2 layers
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k0, v0) = make_tokens(&cfg, 3, 1.0);
        let (k1, v1) = make_tokens(&cfg, 2, 10.0);
        let counts = batch_append_kv(&mut cache, &[&k0, &k1], &[&v0, &v1]).unwrap();
        assert_eq!(counts, vec![3, 2]);
        let stats = cache_stats(&cache);
        assert_eq!(stats.total_tokens, 5);
    }

    #[test]
    fn test_batch_append_wrong_layer_count() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_token(&cfg, 1.0);
        assert!(batch_append_kv(&mut cache, &[&k], &[&v]).is_err()); // 1 != 2
    }

    // ── Scatter/Gather ────────────────────────────────────────────

    #[test]
    fn test_scatter_overwrites() {
        let cfg = tiny_config();
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        // Overwrite token 1 with value 99.0.
        let new_k = vec![99.0; te];
        let new_v = vec![990.0; te];
        scatter_kv(&mut cache, 0, &[1], &new_k, &new_v).unwrap();
        let (rk, rv) = lookup_kv(&mut cache, 0, 1, 2).unwrap();
        assert!((rk[0] - 99.0).abs() < f32::EPSILON);
        assert!((rv[0] - 990.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scatter_multiple_indices() {
        let cfg = tiny_config();
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        let new_k = vec![50.0; te * 2];
        let new_v = vec![500.0; te * 2];
        scatter_kv(&mut cache, 0, &[0, 3], &new_k, &new_v).unwrap();
        let (rk0, _) = lookup_kv(&mut cache, 0, 0, 1).unwrap();
        let (rk3, _) = lookup_kv(&mut cache, 0, 3, 4).unwrap();
        assert!((rk0[0] - 50.0).abs() < f32::EPSILON);
        assert!((rk3[0] - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_scatter_bad_key_length() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 2, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        assert!(scatter_kv(&mut cache, 0, &[0], &[1.0; 5], &[1.0; 5]).is_err());
    }

    #[test]
    fn test_scatter_out_of_range() {
        let cfg = tiny_config();
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 2, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        let new_k = vec![0.0; te];
        let new_v = vec![0.0; te];
        assert!(scatter_kv(&mut cache, 0, &[99], &new_k, &new_v).is_err());
    }

    #[test]
    fn test_scatter_invalid_layer() {
        let cfg = tiny_config();
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        assert!(scatter_kv(&mut cache, 99, &[0], &vec![0.0; te], &vec![0.0; te]).is_err());
    }

    #[test]
    fn test_gather_basic() {
        let cfg = tiny_config();
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 10.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        let (gk, _gv) = gather_kv(&mut cache, 0, &[1, 3]).unwrap();
        assert_eq!(gk.len(), 2 * te);
        assert!((gk[0] - 11.0).abs() < f32::EPSILON); // token 1
        assert!((gk[te] - 13.0).abs() < f32::EPSILON); // token 3
    }

    #[test]
    fn test_gather_empty_indices() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 2, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        let (gk, gv) = gather_kv(&mut cache, 0, &[]).unwrap();
        assert!(gk.is_empty());
        assert!(gv.is_empty());
    }

    #[test]
    fn test_gather_out_of_range() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 2, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        assert!(gather_kv(&mut cache, 0, &[99]).is_err());
    }

    #[test]
    fn test_gather_invalid_layer() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        assert!(gather_kv(&mut cache, 99, &[0]).is_err());
    }

    // ── Cache stats ───────────────────────────────────────────────

    #[test]
    fn test_stats_after_operations() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 5, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        let stats = cache_stats(&cache);
        assert_eq!(stats.total_tokens, 5);
        assert_eq!(stats.num_layers, 2);
        assert!(stats.memory_bytes > 0);
        assert_eq!(stats.total_evictions, 0);
        assert_eq!(stats.max_seq_len, 5);
    }

    #[test]
    fn test_stats_memory_positive() {
        let cache = create_kv_cache(small_config(), EvictionPolicy::LRU).unwrap();
        let stats = cache_stats(&cache);
        assert!(stats.memory_bytes > 0);
    }

    #[test]
    fn test_stats_used_pages() {
        let cfg = tiny_config(); // page_size=2
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 3, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        let stats = cache_stats(&cache);
        assert_eq!(stats.used_pages, 2); // 3 tokens, 2 per page → 2 pages
    }

    // ── Clear ─────────────────────────────────────────────────────

    #[test]
    fn test_clear_resets_all() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 5, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        append_kv(&mut cache, 1, &k, &v).unwrap();
        clear_cache(&mut cache);
        let stats = cache_stats(&cache);
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.used_pages, 0);
    }

    #[test]
    fn test_clear_then_reuse() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_token(&cfg, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        clear_cache(&mut cache);
        let (k2, v2) = make_token(&cfg, 2.0);
        append_kv(&mut cache, 0, &k2, &v2).unwrap();
        let (rk, _) = lookup_kv(&mut cache, 0, 0, 1).unwrap();
        assert!((rk[0] - 2.0).abs() < f32::EPSILON);
    }

    // ── Multi-layer independence ──────────────────────────────────

    #[test]
    fn test_layer_independence() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k0, v0) = make_token(&cfg, 1.0);
        let (k1, v1) = make_tokens(&cfg, 3, 10.0);
        append_kv(&mut cache, 0, &k0, &v0).unwrap();
        append_kv(&mut cache, 1, &k1, &v1).unwrap();
        let (rk0, _) = lookup_kv(&mut cache, 0, 0, 1).unwrap();
        let (rk1, _) = lookup_kv(&mut cache, 1, 0, 1).unwrap();
        assert!((rk0[0] - 1.0).abs() < f32::EPSILON);
        assert!((rk1[0] - 10.0).abs() < f32::EPSILON);
    }

    // ── SIMD correctness ──────────────────────────────────────────

    #[test]
    fn test_simd_copy_small() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = [0.0; 5];
        simd_copy_f32(&src, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_simd_copy_avx2_aligned() {
        let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut dst = [0.0; 16];
        simd_copy_f32(&src, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_simd_copy_avx2_with_tail() {
        let src: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let mut dst = [0.0; 19];
        simd_copy_f32(&src, &mut dst);
        assert_eq!(src, dst);
    }

    #[test]
    fn test_simd_copy_empty() {
        let src: Vec<f32> = vec![];
        let mut dst: Vec<f32> = vec![];
        simd_copy_f32(&src, &mut dst);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_simd_copy_single() {
        let src = [42.0];
        let mut dst = [0.0];
        simd_copy_f32(&src, &mut dst);
        assert_eq!(dst[0], 42.0);
    }

    #[test]
    fn test_simd_dot_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = simd_dot_f32(&a, &b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!((result - 70.0).abs() < 1e-4);
    }

    #[test]
    fn test_simd_dot_avx2_length() {
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
        let result = simd_dot_f32(&a, &b);
        let expected = scalar_dot_f32(&a, &b);
        assert!((result - expected).abs() < 1e-3);
    }

    #[test]
    fn test_simd_dot_with_tail() {
        let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..19).map(|i| 1.0 + i as f32 * 0.1).collect();
        let result = simd_dot_f32(&a, &b);
        let expected = scalar_dot_f32(&a, &b);
        assert!((result - expected).abs() < 1e-2);
    }

    #[test]
    fn test_simd_dot_zeros() {
        let a = [0.0; 32];
        let b = [1.0; 32];
        assert!((simd_dot_f32(&a, &b)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_simd_scale_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        simd_scale_f32(&mut data, 2.0);
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_simd_scale_avx2_length() {
        let mut data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let expected: Vec<f32> = (0..16).map(|i| i as f32 * 3.0).collect();
        simd_scale_f32(&mut data, 3.0);
        for (a, e) in data.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5);
        }
    }

    #[test]
    fn test_simd_scale_with_tail() {
        let mut data: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let expected: Vec<f32> = (0..19).map(|i| i as f32 * 0.5).collect();
        simd_scale_f32(&mut data, 0.5);
        for (a, e) in data.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5);
        }
    }

    #[test]
    fn test_simd_scale_zero() {
        let mut data = vec![1.0, 2.0, 3.0];
        simd_scale_f32(&mut data, 0.0);
        assert!(data.iter().all(|&v| v == 0.0));
    }

    // ── Edge cases ────────────────────────────────────────────────

    #[test]
    fn test_single_head_single_dim() {
        let cfg = KVCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 1,
            max_seq_len: 4,
            page_size: 2,
        };
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        append_kv(&mut cache, 0, &[42.0], &[84.0]).unwrap();
        let (rk, rv) = lookup_kv(&mut cache, 0, 0, 1).unwrap();
        assert!((rk[0] - 42.0).abs() < f32::EPSILON);
        assert!((rv[0] - 84.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_page_size_one() {
        let cfg = KVCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
            max_seq_len: 4,
            page_size: 1,
        };
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        for i in 0..4 {
            append_kv(&mut cache, 0, &[i as f32; 2], &[0.0; 2]).unwrap();
        }
        let (rk, _) = lookup_kv(&mut cache, 0, 2, 3).unwrap();
        assert!((rk[0] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_page_size_equals_max_seq_len() {
        let cfg = KVCacheConfig {
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 8,
            page_size: 8,
        };
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        let k = vec![1.0; te * 8];
        let v = vec![2.0; te * 8];
        let n = append_kv(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(n, 8);
        let stats = cache_stats(&cache);
        assert_eq!(stats.used_pages, 1);
    }

    #[test]
    fn test_all_eviction_policies_create() {
        let cfg = tiny_config();
        for policy in &[EvictionPolicy::LRU, EvictionPolicy::FIFO, EvictionPolicy::SlidingWindow] {
            let cache = create_kv_cache(cfg.clone(), *policy).unwrap();
            assert_eq!(cache.eviction_policy, *policy);
        }
    }

    #[test]
    fn test_eviction_policy_equality() {
        assert_eq!(EvictionPolicy::LRU, EvictionPolicy::LRU);
        assert_ne!(EvictionPolicy::LRU, EvictionPolicy::FIFO);
        assert_ne!(EvictionPolicy::FIFO, EvictionPolicy::SlidingWindow);
    }

    #[test]
    fn test_append_zero_tokens() {
        let cfg = small_config();
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        let n = append_kv(&mut cache, 0, &[], &[]).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_scatter_gather_roundtrip() {
        let cfg = tiny_config();
        let te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();

        // Gather tokens [0, 2].
        let (gk, gv) = gather_kv(&mut cache, 0, &[0, 2]).unwrap();
        assert_eq!(gk.len(), 2 * te);

        // Scatter them back to positions [1, 3].
        scatter_kv(&mut cache, 0, &[1, 3], &gk, &gv).unwrap();

        // Verify positions 1 and 3 now match originals at 0 and 2.
        let (rk1, _) = lookup_kv(&mut cache, 0, 1, 2).unwrap();
        assert!((rk1[0] - 1.0).abs() < f32::EPSILON); // was token 0
        let (rk3, _) = lookup_kv(&mut cache, 0, 3, 4).unwrap();
        assert!((rk3[0] - 3.0).abs() < f32::EPSILON); // was token 2
    }

    #[test]
    fn test_large_head_dim() {
        let cfg = KVCacheConfig {
            num_layers: 1,
            num_heads: 8,
            head_dim: 128,
            max_seq_len: 32,
            page_size: 8,
        };
        let te = cfg.token_elements(); // 1024
        let mut cache = create_kv_cache(cfg, EvictionPolicy::LRU).unwrap();
        let k: Vec<f32> = (0..te).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..te).map(|i| -(i as f32)).collect();
        append_kv(&mut cache, 0, &k, &v).unwrap();
        let (rk, rv) = lookup_kv(&mut cache, 0, 0, 1).unwrap();
        assert_eq!(rk, k);
        assert_eq!(rv, v);
    }

    #[test]
    fn test_stats_eviction_counter() {
        let cfg = tiny_config();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::FIFO).unwrap();
        let (k, v) = make_tokens(&cfg, 4, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        evict_oldest(&mut cache, 0).unwrap();
        evict_oldest(&mut cache, 0).unwrap();
        let stats = cache_stats(&cache);
        assert_eq!(stats.total_evictions, 2);
    }

    #[test]
    fn test_clear_preserves_capacity() {
        let cfg = small_config();
        let cache_before = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let mem_before = cache_stats(&cache_before).memory_bytes;

        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_tokens(&cfg, 10, 1.0);
        append_kv(&mut cache, 0, &k, &v).unwrap();
        clear_cache(&mut cache);
        let mem_after = cache_stats(&cache).memory_bytes;
        assert_eq!(mem_before, mem_after);
    }

    #[test]
    fn test_batch_append_all_layers() {
        let cfg = small_config(); // 2 layers
        let _te = cfg.token_elements();
        let mut cache = create_kv_cache(cfg.clone(), EvictionPolicy::LRU).unwrap();
        let (k, v) = make_token(&cfg, 5.0);
        let counts = batch_append_kv(&mut cache, &[&k, &k], &[&v, &v]).unwrap();
        assert_eq!(counts, vec![1, 1]);
        // Both layers should have 1 token.
        let (rk0, _) = lookup_kv(&mut cache, 0, 0, 1).unwrap();
        let (rk1, _) = lookup_kv(&mut cache, 1, 0, 1).unwrap();
        assert!((rk0[0] - 5.0).abs() < f32::EPSILON);
        assert!((rk1[0] - 5.0).abs() < f32::EPSILON);
    }
}
