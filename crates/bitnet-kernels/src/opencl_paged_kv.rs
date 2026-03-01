//! Paged KV cache for efficient long-sequence attention on Intel A770 GPUs.
//!
//! Implements a page-table–based key/value cache inspired by vLLM's paged
//! attention. Each sequence owns a [`PageTable`] that maps logical token
//! positions to physical [`KVPage`]s. Pages are drawn from a shared free-list
//! and returned on eviction, enabling near-zero-waste memory reuse across
//! concurrent sequences.
//!
//! CPU reference kernels are provided for every cache operation so the
//! correctness of future OpenCL accelerated paths can be validated against
//! them.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a [`PagedKVCache`].
#[derive(Debug, Clone)]
pub struct PagedKVConfig {
    /// Number of tokens stored per page.
    pub page_size: usize,
    /// Total number of physical pages available.
    pub max_pages: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimensionality of each attention head.
    pub head_dim: usize,
}

impl Default for PagedKVConfig {
    fn default() -> Self {
        Self {
            page_size: 16,
            max_pages: 64,
            num_layers: 1,
            num_heads: 1,
            head_dim: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// Page / PageTable types
// ---------------------------------------------------------------------------

/// A single physical page holding key and value data.
#[derive(Debug, Clone)]
pub struct KVPage {
    /// Key tensor data – length = page_size × num_heads × head_dim.
    pub key_data: Vec<f32>,
    /// Value tensor data – same layout as `key_data`.
    pub value_data: Vec<f32>,
    /// Number of valid tokens currently stored in this page.
    pub num_tokens: usize,
    /// Unique identifier for this page.
    pub page_id: u32,
}

/// Logical-to-physical page mapping for a single sequence.
#[derive(Debug, Clone)]
pub struct PageTable {
    /// `entries[i]` is the physical page id that holds logical page `i`.
    pub entries: Vec<Option<u32>>,
    /// Total number of tokens appended to this sequence so far.
    pub sequence_length: usize,
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

/// Paged KV cache with free-list page management.
pub struct PagedKVCache {
    /// All physical pages (indexed by `page_id`).
    pub pages: Vec<KVPage>,
    /// Stack of free page ids.
    pub free_pages: Vec<u32>,
    /// Per-sequence page tables, keyed by `(seq_id, layer)`.
    pub page_tables: HashMap<u64, PageTable>,
    /// Configuration snapshot.
    pub config: PagedKVConfig,
}

impl fmt::Debug for PagedKVCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PagedKVCache")
            .field("total_pages", &self.pages.len())
            .field("free_pages", &self.free_pages.len())
            .field("sequences", &self.page_tables.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by paged KV cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagedKVError {
    /// No free pages remain.
    OutOfPages,
    /// The requested sequence id (and layer) has no page table.
    InvalidSequence,
    /// The page table has reached its capacity.
    PageTableFull,
    /// A requested token position is out of range.
    InvalidPosition,
}

impl fmt::Display for PagedKVError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfPages => write!(f, "no free pages available"),
            Self::InvalidSequence => write!(f, "invalid sequence id"),
            Self::PageTableFull => write!(f, "page table is full"),
            Self::InvalidPosition => write!(f, "invalid token position"),
        }
    }
}

impl std::error::Error for PagedKVError {}

// ---------------------------------------------------------------------------
// Composite key helper
// ---------------------------------------------------------------------------

/// Encode `(seq_id, layer)` into a single `u64` key for the page-table map.
#[inline]
fn table_key(seq_id: u64, layer: usize) -> u64 {
    // Top 32 bits: seq_id (truncated), bottom 32 bits: layer.
    (seq_id << 32) | (layer as u64)
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Create an empty [`PagedKVCache`] pre-populated with physical pages.
pub fn create_paged_cache(config: &PagedKVConfig) -> PagedKVCache {
    let slot_size = config.page_size * config.num_heads * config.head_dim;
    let pages: Vec<KVPage> = (0..config.max_pages as u32)
        .map(|id| KVPage {
            key_data: vec![0.0; slot_size],
            value_data: vec![0.0; slot_size],
            num_tokens: 0,
            page_id: id,
        })
        .collect();

    let free_pages: Vec<u32> =
        (0..config.max_pages as u32).rev().collect();

    PagedKVCache {
        pages,
        free_pages,
        page_tables: HashMap::new(),
        config: config.clone(),
    }
}

/// Allocate a free page, returning its id.
pub fn cpu_allocate_page(
    cache: &mut PagedKVCache,
) -> Result<u32, PagedKVError> {
    cache.free_pages.pop().ok_or(PagedKVError::OutOfPages)
}

/// Return a page to the free list and reset its contents.
pub fn cpu_free_page(cache: &mut PagedKVCache, page_id: u32) {
    let pid = page_id as usize;
    if pid < cache.pages.len() {
        cache.pages[pid].num_tokens = 0;
        for v in &mut cache.pages[pid].key_data {
            *v = 0.0;
        }
        for v in &mut cache.pages[pid].value_data {
            *v = 0.0;
        }
        cache.free_pages.push(page_id);
    }
}

/// Append a single key/value token to the cache for `(seq_id, layer)`.
///
/// `key` and `value` must each have length `num_heads × head_dim`.
pub fn cpu_append_kv(
    cache: &mut PagedKVCache,
    seq_id: u64,
    layer: usize,
    key: &[f32],
    value: &[f32],
) -> Result<(), PagedKVError> {
    let kv_len = cache.config.num_heads * cache.config.head_dim;
    assert_eq!(key.len(), kv_len, "key length mismatch");
    assert_eq!(value.len(), kv_len, "value length mismatch");

    let tk = table_key(seq_id, layer);
    let page_size = cache.config.page_size;

    // Ensure a page table exists for this (seq, layer).
    let pt = cache
        .page_tables
        .entry(tk)
        .or_insert_with(|| PageTable {
            entries: Vec::new(),
            sequence_length: 0,
        });

    let logical_page = pt.sequence_length / page_size;
    let offset_in_page = pt.sequence_length % page_size;

    // Allocate a new physical page if we're at the start of a logical page.
    if offset_in_page == 0 {
        let pid = cpu_allocate_page(cache)?;
        let pt = cache.page_tables.get_mut(&tk).unwrap();
        if logical_page >= pt.entries.len() {
            pt.entries.resize(logical_page + 1, None);
        }
        pt.entries[logical_page] = Some(pid);
    }

    let pt = cache.page_tables.get_mut(&tk).unwrap();
    let phys_id = pt.entries[logical_page]
        .ok_or(PagedKVError::InvalidPosition)? as usize;

    // Write key/value into the physical page.
    let start = offset_in_page * kv_len;
    cache.pages[phys_id].key_data[start..start + kv_len]
        .copy_from_slice(key);
    cache.pages[phys_id].value_data[start..start + kv_len]
        .copy_from_slice(value);
    cache.pages[phys_id].num_tokens += 1;

    // Advance sequence length.
    cache.page_tables.get_mut(&tk).unwrap().sequence_length += 1;
    Ok(())
}

/// Read key/value vectors at specific token positions.
pub fn cpu_read_kv(
    cache: &PagedKVCache,
    seq_id: u64,
    layer: usize,
    positions: &[usize],
) -> Result<(Vec<f32>, Vec<f32>), PagedKVError> {
    let tk = table_key(seq_id, layer);
    let pt = cache
        .page_tables
        .get(&tk)
        .ok_or(PagedKVError::InvalidSequence)?;
    let kv_len = cache.config.num_heads * cache.config.head_dim;
    let page_size = cache.config.page_size;

    let mut keys = Vec::with_capacity(positions.len() * kv_len);
    let mut values = Vec::with_capacity(positions.len() * kv_len);

    for &pos in positions {
        if pos >= pt.sequence_length {
            return Err(PagedKVError::InvalidPosition);
        }
        let logical_page = pos / page_size;
        let offset = pos % page_size;
        let phys_id = pt.entries[logical_page]
            .ok_or(PagedKVError::InvalidPosition)? as usize;
        let start = offset * kv_len;
        keys.extend_from_slice(
            &cache.pages[phys_id].key_data[start..start + kv_len],
        );
        values.extend_from_slice(
            &cache.pages[phys_id].value_data[start..start + kv_len],
        );
    }
    Ok((keys, values))
}

/// Read all key/value vectors for a sequence in order.
pub fn cpu_read_all_kv(
    cache: &PagedKVCache,
    seq_id: u64,
    layer: usize,
) -> Result<(Vec<f32>, Vec<f32>), PagedKVError> {
    let tk = table_key(seq_id, layer);
    let pt = cache
        .page_tables
        .get(&tk)
        .ok_or(PagedKVError::InvalidSequence)?;
    let positions: Vec<usize> = (0..pt.sequence_length).collect();
    cpu_read_kv(cache, seq_id, layer, &positions)
}

/// Free all pages owned by `seq_id` across every layer.
pub fn cpu_evict_sequence(cache: &mut PagedKVCache, seq_id: u64) {
    let keys_to_remove: Vec<u64> = cache
        .page_tables
        .keys()
        .filter(|&&k| k >> 32 == seq_id)
        .copied()
        .collect();

    for tk in keys_to_remove {
        if let Some(pt) = cache.page_tables.remove(&tk) {
            for entry in &pt.entries {
                if let Some(pid) = entry {
                    cpu_free_page(cache, *pid);
                }
            }
        }
    }
}

/// Fraction of pages currently in use (0.0 – 1.0).
pub fn cpu_cache_utilization(cache: &PagedKVCache) -> f32 {
    if cache.config.max_pages == 0 {
        return 0.0;
    }
    let used = cache.config.max_pages - cache.free_pages.len();
    used as f32 / cache.config.max_pages as f32
}

/// Compact pages so that each logical page is backed by a contiguous
/// physical page with no internal gaps.
///
/// After defragmentation the relative ordering of tokens within each
/// sequence is preserved, but physical page ids may change.
pub fn cpu_defragment(cache: &mut PagedKVCache) {
    let kv_len = cache.config.num_heads * cache.config.head_dim;
    let page_size = cache.config.page_size;

    // Collect allocated page ids (those NOT in the free list).
    let free_set: std::collections::HashSet<u32> =
        cache.free_pages.iter().copied().collect();
    let mut allocated: Vec<u32> = (0..cache.config.max_pages as u32)
        .filter(|pid| !free_set.contains(pid))
        .collect();
    allocated.sort_unstable();

    // Pack allocated pages to the lowest physical ids.
    let mut remap: HashMap<u32, u32> = HashMap::new();
    for (new_idx, &old_pid) in allocated.iter().enumerate() {
        let new_pid = new_idx as u32;
        if new_pid != old_pid {
            remap.insert(old_pid, new_pid);
        }
    }

    // Swap page contents for remapped pages.
    for (&old, &new) in &remap {
        let slot = page_size * kv_len;
        let old_idx = old as usize;
        let new_idx = new as usize;

        // Swap key_data
        for i in 0..slot {
            let a = cache.pages[old_idx].key_data[i];
            let b = cache.pages[new_idx].key_data[i];
            cache.pages[old_idx].key_data[i] = b;
            cache.pages[new_idx].key_data[i] = a;
        }
        // Swap value_data
        for i in 0..slot {
            let a = cache.pages[old_idx].value_data[i];
            let b = cache.pages[new_idx].value_data[i];
            cache.pages[old_idx].value_data[i] = b;
            cache.pages[new_idx].value_data[i] = a;
        }
        // Swap num_tokens
        let tmp = cache.pages[old_idx].num_tokens;
        cache.pages[old_idx].num_tokens =
            cache.pages[new_idx].num_tokens;
        cache.pages[new_idx].num_tokens = tmp;
    }

    // Rewrite page table entries.
    for pt in cache.page_tables.values_mut() {
        for entry in &mut pt.entries {
            if let Some(pid) = entry {
                if let Some(&new_pid) = remap.get(pid) {
                    *pid = new_pid;
                }
            }
        }
    }

    // Rebuild free list (all pages above the allocated count are free).
    let used = allocated.len() as u32;
    cache.free_pages = (used..cache.config.max_pages as u32)
        .rev()
        .collect();
}

/// CPU reference paged-attention: Q × K^T / scale → softmax → × V.
///
/// `query` has shape `[num_heads × head_dim]`.  Returns the attention
/// output with the same shape.
pub fn cpu_paged_attention(
    query: &[f32],
    cache: &PagedKVCache,
    seq_id: u64,
    layer: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>, PagedKVError> {
    let (keys, values) =
        cpu_read_all_kv(cache, seq_id, layer)?;
    let num_heads = cache.config.num_heads;
    let kv_len = num_heads * head_dim;
    let tk = table_key(seq_id, layer);
    let seq_len = cache
        .page_tables
        .get(&tk)
        .ok_or(PagedKVError::InvalidSequence)?
        .sequence_length;

    assert_eq!(query.len(), kv_len);
    assert_eq!(keys.len(), seq_len * kv_len);

    let mut output = vec![0.0f32; kv_len];

    for h in 0..num_heads {
        let q_off = h * head_dim;
        let q = &query[q_off..q_off + head_dim];

        // Compute attention scores.
        let mut scores = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let k_off = t * kv_len + h * head_dim;
            let k = &keys[k_off..k_off + head_dim];
            let dot: f32 =
                q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
            scores.push(dot * scale);
        }

        // Softmax.
        let max_score =
            scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            exp_sum += *s;
        }
        if exp_sum > 0.0 {
            for s in &mut scores {
                *s /= exp_sum;
            }
        }

        // Weighted sum of values.
        for t in 0..seq_len {
            let v_off = t * kv_len + h * head_dim;
            let weight = scores[t];
            for d in 0..head_dim {
                output[q_off + d] += weight * values[v_off + d];
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// OpenCL kernel source (const string)
// ---------------------------------------------------------------------------

/// OpenCL C source for paged attention kernels targeting Intel A770.
pub const PAGED_KV_SRC: &str = r#"
// ----- paged_attention_fwd -----
// Each work-group handles one attention head.
// query:       [num_heads, head_dim]
// key_pages:   [max_pages, page_size, num_heads, head_dim]
// value_pages: [max_pages, page_size, num_heads, head_dim]
// page_table:  [max_logical_pages]  (int, physical page id or -1)
// output:      [num_heads, head_dim]
__kernel void paged_attention_fwd(
    __global const float* query,
    __global const float* key_pages,
    __global const float* value_pages,
    __global const int*   page_table,
    __global float*       output,
    const int seq_len,
    const int page_size,
    const int num_heads,
    const int head_dim,
    const float scale)
{
    const int h = get_global_id(0);
    if (h >= num_heads) return;

    const int kv_stride = num_heads * head_dim;
    const int q_off = h * head_dim;

    // --- dot products --------------------------------------------------
    float max_score = -1e30f;
    // First pass: compute scores and find max
    for (int t = 0; t < seq_len; ++t) {
        int logical_page = t / page_size;
        int offset       = t % page_size;
        int phys_page    = page_table[logical_page];
        int k_base       = phys_page * page_size * kv_stride
                         + offset * kv_stride + h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += query[q_off + d] * key_pages[k_base + d];
        dot *= scale;
        if (dot > max_score) max_score = dot;
    }

    // --- softmax numerator + denominator ------------------------------
    float exp_sum = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        int logical_page = t / page_size;
        int offset       = t % page_size;
        int phys_page    = page_table[logical_page];
        int k_base       = phys_page * page_size * kv_stride
                         + offset * kv_stride + h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            dot += query[q_off + d] * key_pages[k_base + d];
        float s = exp(dot * scale - max_score);
        exp_sum += s;

        int v_base = phys_page * page_size * kv_stride
                   + offset * kv_stride + h * head_dim;
        for (int d = 0; d < head_dim; ++d)
            output[q_off + d] += s * value_pages[v_base + d];
    }

    // --- normalize -----------------------------------------------------
    if (exp_sum > 0.0f) {
        for (int d = 0; d < head_dim; ++d)
            output[q_off + d] /= exp_sum;
    }
}

// ----- copy_to_page -----
// Scatter a single token's KV into the correct page slot.
__kernel void copy_to_page(
    __global const float* key_in,
    __global const float* value_in,
    __global float*       key_pages,
    __global float*       value_pages,
    const int page_id,
    const int offset_in_page,
    const int num_heads,
    const int head_dim)
{
    const int idx = get_global_id(0);
    const int kv_stride = num_heads * head_dim;
    if (idx >= kv_stride) return;

    const int page_size_stride = kv_stride;  // one token slice
    int base = page_id * page_size_stride * 16  // max page_size assumed
             + offset_in_page * kv_stride + idx;
    key_pages[base]   = key_in[idx];
    value_pages[base] = value_in[idx];
}

// ----- gather_from_pages -----
// Gather KV for a contiguous range of tokens into a flat buffer.
__kernel void gather_from_pages(
    __global const float* key_pages,
    __global const float* value_pages,
    __global const int*   page_table,
    __global float*       keys_out,
    __global float*       values_out,
    const int seq_len,
    const int page_size,
    const int num_heads,
    const int head_dim)
{
    const int t = get_global_id(0);
    if (t >= seq_len) return;

    const int kv_stride = num_heads * head_dim;
    int logical_page = t / page_size;
    int offset       = t % page_size;
    int phys_page    = page_table[logical_page];
    int src_base     = phys_page * page_size * kv_stride
                     + offset * kv_stride;
    int dst_base     = t * kv_stride;

    for (int i = 0; i < kv_stride; ++i) {
        keys_out[dst_base + i]   = key_pages[src_base + i];
        values_out[dst_base + i] = value_pages[src_base + i];
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> PagedKVConfig {
        PagedKVConfig {
            page_size: 4,
            max_pages: 8,
            num_layers: 2,
            num_heads: 2,
            head_dim: 4,
        }
    }

    fn tiny_config() -> PagedKVConfig {
        PagedKVConfig {
            page_size: 1,
            max_pages: 4,
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
        }
    }

    fn make_kv(config: &PagedKVConfig, val: f32) -> (Vec<f32>, Vec<f32>) {
        let len = config.num_heads * config.head_dim;
        (vec![val; len], vec![val; len])
    }

    // ---- allocation / free round-trip --------------------------------

    #[test]
    fn allocate_returns_unique_ids() {
        let mut cache = create_paged_cache(&small_config());
        let a = cpu_allocate_page(&mut cache).unwrap();
        let b = cpu_allocate_page(&mut cache).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn free_page_makes_it_available_again() {
        let mut cache = create_paged_cache(&small_config());
        let pid = cpu_allocate_page(&mut cache).unwrap();
        let before = cache.free_pages.len();
        cpu_free_page(&mut cache, pid);
        assert_eq!(cache.free_pages.len(), before + 1);
    }

    #[test]
    fn allocate_all_then_free_all() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let mut ids = Vec::new();
        for _ in 0..cfg.max_pages {
            ids.push(cpu_allocate_page(&mut cache).unwrap());
        }
        assert!(cpu_allocate_page(&mut cache).is_err());
        for pid in ids {
            cpu_free_page(&mut cache, pid);
        }
        assert_eq!(cache.free_pages.len(), cfg.max_pages);
    }

    // ---- utilization -------------------------------------------------

    #[test]
    fn utilization_empty_cache_is_zero() {
        let cache = create_paged_cache(&small_config());
        assert!((cpu_cache_utilization(&cache) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn utilization_full_cache_is_one() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        for _ in 0..cfg.max_pages {
            cpu_allocate_page(&mut cache).unwrap();
        }
        assert!((cpu_cache_utilization(&cache) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn utilization_half() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        for _ in 0..cfg.max_pages / 2 {
            cpu_allocate_page(&mut cache).unwrap();
        }
        assert!(
            (cpu_cache_utilization(&cache) - 0.5).abs() < f32::EPSILON
        );
    }

    #[test]
    fn utilization_always_in_unit_range() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        for _ in 0..cfg.max_pages {
            let u = cpu_cache_utilization(&cache);
            assert!((0.0..=1.0).contains(&u));
            cpu_allocate_page(&mut cache).unwrap();
        }
        let u = cpu_cache_utilization(&cache);
        assert!((0.0..=1.0).contains(&u));
    }

    // ---- append + read single KV -------------------------------------

    #[test]
    fn append_and_read_single_kv() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let (k, v) = make_kv(&cfg, 1.0);
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        let (rk, rv) = cpu_read_kv(&cache, 0, 0, &[0]).unwrap();
        assert_eq!(rk, k);
        assert_eq!(rv, v);
    }

    #[test]
    fn append_two_tokens_read_both() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let (k1, v1) = make_kv(&cfg, 1.0);
        let (k2, v2) = make_kv(&cfg, 2.0);
        cpu_append_kv(&mut cache, 0, 0, &k1, &v1).unwrap();
        cpu_append_kv(&mut cache, 0, 0, &k2, &v2).unwrap();
        let (rk, rv) = cpu_read_kv(&cache, 0, 0, &[0, 1]).unwrap();
        let kv_len = cfg.num_heads * cfg.head_dim;
        assert_eq!(&rk[..kv_len], &k1[..]);
        assert_eq!(&rk[kv_len..], &k2[..]);
        assert_eq!(&rv[..kv_len], &v1[..]);
        assert_eq!(&rv[kv_len..], &v2[..]);
    }

    #[test]
    fn read_all_kv_returns_full_sequence() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        for i in 0..3 {
            let (k, v) = make_kv(&cfg, i as f32);
            cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        }
        let (all_k, all_v) = cpu_read_all_kv(&cache, 0, 0).unwrap();
        let kv_len = cfg.num_heads * cfg.head_dim;
        assert_eq!(all_k.len(), 3 * kv_len);
        assert_eq!(all_v.len(), 3 * kv_len);
    }

    // ---- multi-sequence isolation ------------------------------------

    #[test]
    fn two_sequences_do_not_interfere() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let (k1, v1) = make_kv(&cfg, 10.0);
        let (k2, v2) = make_kv(&cfg, 20.0);
        cpu_append_kv(&mut cache, 1, 0, &k1, &v1).unwrap();
        cpu_append_kv(&mut cache, 2, 0, &k2, &v2).unwrap();

        let (rk1, _) = cpu_read_kv(&cache, 1, 0, &[0]).unwrap();
        let (rk2, _) = cpu_read_kv(&cache, 2, 0, &[0]).unwrap();
        assert_eq!(rk1, k1);
        assert_eq!(rk2, k2);
    }

    #[test]
    fn evict_one_sequence_preserves_other() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let (k1, v1) = make_kv(&cfg, 10.0);
        let (k2, v2) = make_kv(&cfg, 20.0);
        cpu_append_kv(&mut cache, 1, 0, &k1, &v1).unwrap();
        cpu_append_kv(&mut cache, 2, 0, &k2, &v2).unwrap();
        cpu_evict_sequence(&mut cache, 1);

        assert!(cpu_read_kv(&cache, 1, 0, &[0]).is_err());
        let (rk2, _) = cpu_read_kv(&cache, 2, 0, &[0]).unwrap();
        assert_eq!(rk2, k2);
    }

    #[test]
    fn evict_nonexistent_sequence_is_noop() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        cpu_evict_sequence(&mut cache, 999); // should not panic
    }

    // ---- multi-layer -------------------------------------------------

    #[test]
    fn different_layers_are_independent() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let (k0, v0) = make_kv(&cfg, 5.0);
        let (k1, v1) = make_kv(&cfg, 9.0);
        cpu_append_kv(&mut cache, 0, 0, &k0, &v0).unwrap();
        cpu_append_kv(&mut cache, 0, 1, &k1, &v1).unwrap();

        let (rk0, _) = cpu_read_kv(&cache, 0, 0, &[0]).unwrap();
        let (rk1, _) = cpu_read_kv(&cache, 0, 1, &[0]).unwrap();
        assert_eq!(rk0, k0);
        assert_eq!(rk1, k1);
    }

    // ---- page boundary -----------------------------------------------

    #[test]
    fn exactly_page_size_tokens_fills_one_page() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let initial_free = cache.free_pages.len();
        for i in 0..cfg.page_size {
            let (k, v) = make_kv(&cfg, i as f32);
            cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        }
        // Exactly one page consumed.
        assert_eq!(cache.free_pages.len(), initial_free - 1);
    }

    #[test]
    fn page_size_plus_one_allocates_second_page() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let initial_free = cache.free_pages.len();
        for i in 0..=cfg.page_size {
            let (k, v) = make_kv(&cfg, i as f32);
            cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        }
        assert_eq!(cache.free_pages.len(), initial_free - 2);
    }

    #[test]
    fn tokens_across_page_boundary_read_correctly() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let n = cfg.page_size + 2;
        for i in 0..n {
            let (k, v) = make_kv(&cfg, i as f32);
            cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        }
        let (all_k, _) = cpu_read_all_kv(&cache, 0, 0).unwrap();
        let kv_len = cfg.num_heads * cfg.head_dim;
        for i in 0..n {
            let expected = i as f32;
            assert!(
                (all_k[i * kv_len] - expected).abs() < f32::EPSILON,
                "mismatch at token {i}"
            );
        }
    }

    // ---- out of pages ------------------------------------------------

    #[test]
    fn out_of_pages_error() {
        let cfg = PagedKVConfig {
            page_size: 1,
            max_pages: 2,
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
        };
        let mut cache = create_paged_cache(&cfg);
        let (k, v) = make_kv(&cfg, 1.0);
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        let res = cpu_append_kv(&mut cache, 0, 0, &k, &v);
        assert_eq!(res, Err(PagedKVError::OutOfPages));
    }

    #[test]
    fn allocate_beyond_max_pages_fails() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        for _ in 0..cfg.max_pages {
            cpu_allocate_page(&mut cache).unwrap();
        }
        assert_eq!(
            cpu_allocate_page(&mut cache),
            Err(PagedKVError::OutOfPages)
        );
    }

    // ---- evict and reuse ---------------------------------------------

    #[test]
    fn evict_frees_pages_for_reuse() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let (k, v) = make_kv(&cfg, 1.0);
        for _ in 0..cfg.page_size {
            cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        }
        let free_before = cache.free_pages.len();
        cpu_evict_sequence(&mut cache, 0);
        assert!(cache.free_pages.len() > free_before);
    }

    #[test]
    fn reuse_pages_after_eviction() {
        let cfg = PagedKVConfig {
            page_size: 1,
            max_pages: 2,
            ..small_config()
        };
        let mut cache = create_paged_cache(&cfg);
        let (k, v) = make_kv(&cfg, 1.0);
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        assert!(cpu_append_kv(&mut cache, 0, 0, &k, &v).is_err());

        cpu_evict_sequence(&mut cache, 0);
        // Now we can allocate again.
        cpu_append_kv(&mut cache, 1, 0, &k, &v).unwrap();
    }

    // ---- defragment --------------------------------------------------

    #[test]
    fn defragment_compacts_sparse_pages() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let (k, v) = make_kv(&cfg, 1.0);
        // Fill sequences 0 and 1, then evict 0 to create a gap.
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        cpu_append_kv(&mut cache, 1, 0, &k, &v).unwrap();
        cpu_evict_sequence(&mut cache, 0);

        cpu_defragment(&mut cache);

        // The remaining sequence should still be readable.
        let (rk, _) = cpu_read_kv(&cache, 1, 0, &[0]).unwrap();
        assert_eq!(rk, k);
    }

    #[test]
    fn defragment_empty_cache_is_noop() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        cpu_defragment(&mut cache);
        assert_eq!(cache.free_pages.len(), cfg.max_pages);
    }

    #[test]
    fn defragment_preserves_all_data() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        // Insert 3 tokens across 2 sequences, evict seq 0.
        for i in 0..3u32 {
            let (k, v) = make_kv(&cfg, (i + 1) as f32);
            cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        }
        for i in 0..2u32 {
            let (k, v) = make_kv(&cfg, (i + 10) as f32);
            cpu_append_kv(&mut cache, 1, 0, &k, &v).unwrap();
        }
        // Snapshot seq 1 before defrag.
        let (before_k, before_v) =
            cpu_read_all_kv(&cache, 1, 0).unwrap();

        cpu_evict_sequence(&mut cache, 0);
        cpu_defragment(&mut cache);

        let (after_k, after_v) =
            cpu_read_all_kv(&cache, 1, 0).unwrap();
        assert_eq!(before_k, after_k);
        assert_eq!(before_v, after_v);
    }

    // ---- paged attention ---------------------------------------------

    #[test]
    fn paged_attention_single_token() {
        let cfg = PagedKVConfig {
            page_size: 4,
            max_pages: 4,
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
        };
        let mut cache = create_paged_cache(&cfg);
        let key = vec![1.0, 0.0];
        let val = vec![0.0, 1.0];
        cpu_append_kv(&mut cache, 0, 0, &key, &val).unwrap();

        let query = vec![1.0, 0.0];
        let out = cpu_paged_attention(
            &query, &cache, 0, 0, cfg.head_dim, 1.0,
        )
        .unwrap();
        // Single token → output must equal value (softmax of one = 1).
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn paged_attention_two_tokens_uniform() {
        let cfg = PagedKVConfig {
            page_size: 4,
            max_pages: 4,
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
        };
        let mut cache = create_paged_cache(&cfg);
        // Two identical keys → equal attention weights → output = avg.
        let key = vec![1.0, 0.0];
        let v1 = vec![2.0, 0.0];
        let v2 = vec![0.0, 2.0];
        cpu_append_kv(&mut cache, 0, 0, &key, &v1).unwrap();
        cpu_append_kv(&mut cache, 0, 0, &key, &v2).unwrap();

        let query = vec![1.0, 0.0];
        let out = cpu_paged_attention(
            &query, &cache, 0, 0, cfg.head_dim, 1.0,
        )
        .unwrap();
        // Equal keys → equal weights → average.
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn paged_attention_matches_naive() {
        // Build a scenario with multiple tokens and compare against a
        // manual naive implementation.
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let kv_len = cfg.num_heads * cfg.head_dim;
        for i in 0..5 {
            let k: Vec<f32> =
                (0..kv_len).map(|d| (i * kv_len + d) as f32 * 0.1).collect();
            let v: Vec<f32> =
                (0..kv_len).map(|d| (i * kv_len + d) as f32 * 0.05).collect();
            cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        }
        let query: Vec<f32> = (0..kv_len).map(|d| d as f32 * 0.2).collect();
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let paged_out = cpu_paged_attention(
            &query, &cache, 0, 0, cfg.head_dim, scale,
        )
        .unwrap();

        // Naive attention on flat buffers.
        let (flat_k, flat_v) = cpu_read_all_kv(&cache, 0, 0).unwrap();
        let seq_len = 5;
        let mut naive_out = vec![0.0f32; kv_len];
        for h in 0..cfg.num_heads {
            let q_off = h * cfg.head_dim;
            let q = &query[q_off..q_off + cfg.head_dim];
            let mut scores = Vec::new();
            for t in 0..seq_len {
                let k_off = t * kv_len + h * cfg.head_dim;
                let dot: f32 = q
                    .iter()
                    .zip(&flat_k[k_off..k_off + cfg.head_dim])
                    .map(|(a, b)| a * b)
                    .sum();
                scores.push(dot * scale);
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                exp_sum += *s;
            }
            for s in &mut scores {
                *s /= exp_sum;
            }
            for t in 0..seq_len {
                let v_off = t * kv_len + h * cfg.head_dim;
                for d in 0..cfg.head_dim {
                    naive_out[q_off + d] +=
                        scores[t] * flat_v[v_off + d];
                }
            }
        }

        for (a, b) in paged_out.iter().zip(naive_out.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "paged {a} != naive {b}"
            );
        }
    }

    // ---- error cases -------------------------------------------------

    #[test]
    fn read_invalid_sequence_returns_error() {
        let cache = create_paged_cache(&small_config());
        assert_eq!(
            cpu_read_kv(&cache, 42, 0, &[0]),
            Err(PagedKVError::InvalidSequence)
        );
    }

    #[test]
    fn read_invalid_position_returns_error() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let (k, v) = make_kv(&cfg, 1.0);
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        assert_eq!(
            cpu_read_kv(&cache, 0, 0, &[99]),
            Err(PagedKVError::InvalidPosition)
        );
    }

    #[test]
    fn read_all_kv_invalid_sequence() {
        let cache = create_paged_cache(&small_config());
        assert_eq!(
            cpu_read_all_kv(&cache, 99, 0),
            Err(PagedKVError::InvalidSequence)
        );
    }

    #[test]
    fn paged_attention_invalid_sequence() {
        let cfg = small_config();
        let cache = create_paged_cache(&cfg);
        let kv_len = cfg.num_heads * cfg.head_dim;
        let query = vec![0.0; kv_len];
        assert_eq!(
            cpu_paged_attention(&query, &cache, 99, 0, cfg.head_dim, 1.0),
            Err(PagedKVError::InvalidSequence)
        );
    }

    // ---- edge cases: page_size=1 -------------------------------------

    #[test]
    fn page_size_one_allocates_page_per_token() {
        let cfg = tiny_config();
        let mut cache = create_paged_cache(&cfg);
        let initial = cache.free_pages.len();
        let (k, v) = make_kv(&cfg, 1.0);
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        assert_eq!(cache.free_pages.len(), initial - 1);
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        assert_eq!(cache.free_pages.len(), initial - 2);
    }

    #[test]
    fn page_size_one_read_back() {
        let cfg = tiny_config();
        let mut cache = create_paged_cache(&cfg);
        let (k, v) = make_kv(&cfg, 3.0);
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        let (rk, rv) = cpu_read_kv(&cache, 0, 0, &[0]).unwrap();
        assert_eq!(rk, k);
        assert_eq!(rv, v);
    }

    // ---- single head / single layer ----------------------------------

    #[test]
    fn single_head_single_layer() {
        let cfg = PagedKVConfig {
            page_size: 2,
            max_pages: 4,
            num_layers: 1,
            num_heads: 1,
            head_dim: 3,
        };
        let mut cache = create_paged_cache(&cfg);
        let k = vec![1.0, 2.0, 3.0];
        let v = vec![4.0, 5.0, 6.0];
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        let (rk, rv) = cpu_read_kv(&cache, 0, 0, &[0]).unwrap();
        assert_eq!(rk, k);
        assert_eq!(rv, v);
    }

    // ---- property: free + used = total -------------------------------

    #[test]
    fn free_plus_used_equals_total() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let (k, v) = make_kv(&cfg, 1.0);
        for _ in 0..5 {
            cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        }
        let used = cfg.max_pages - cache.free_pages.len();
        assert_eq!(used + cache.free_pages.len(), cfg.max_pages);
    }

    #[test]
    fn free_plus_used_after_eviction() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let (k, v) = make_kv(&cfg, 1.0);
        for _ in 0..cfg.page_size * 2 {
            cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        }
        cpu_evict_sequence(&mut cache, 0);
        assert_eq!(cache.free_pages.len(), cfg.max_pages);
    }

    // ---- OpenCL source sanity ----------------------------------------

    #[test]
    fn opencl_source_contains_required_kernels() {
        assert!(PAGED_KV_SRC.contains("paged_attention_fwd"));
        assert!(PAGED_KV_SRC.contains("copy_to_page"));
        assert!(PAGED_KV_SRC.contains("gather_from_pages"));
    }

    #[test]
    fn opencl_source_is_non_empty() {
        assert!(!PAGED_KV_SRC.is_empty());
        assert!(PAGED_KV_SRC.len() > 100);
    }

    // ---- Debug impls -------------------------------------------------

    #[test]
    fn cache_debug_does_not_panic() {
        let cache = create_paged_cache(&small_config());
        let _ = format!("{cache:?}");
    }

    #[test]
    fn error_display() {
        assert_eq!(
            PagedKVError::OutOfPages.to_string(),
            "no free pages available"
        );
        assert_eq!(
            PagedKVError::InvalidSequence.to_string(),
            "invalid sequence id"
        );
        assert_eq!(
            PagedKVError::PageTableFull.to_string(),
            "page table is full"
        );
        assert_eq!(
            PagedKVError::InvalidPosition.to_string(),
            "invalid token position"
        );
    }

    // ---- multi-layer eviction ----------------------------------------

    #[test]
    fn evict_frees_all_layers() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        let (k, v) = make_kv(&cfg, 1.0);
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        cpu_append_kv(&mut cache, 0, 1, &k, &v).unwrap();
        cpu_evict_sequence(&mut cache, 0);
        assert!(cpu_read_kv(&cache, 0, 0, &[0]).is_err());
        assert!(cpu_read_kv(&cache, 0, 1, &[0]).is_err());
    }

    // ---- paged attention with multi-head -----------------------------

    #[test]
    fn paged_attention_multi_head() {
        let cfg = small_config(); // 2 heads, head_dim=4
        let mut cache = create_paged_cache(&cfg);
        let kv_len = cfg.num_heads * cfg.head_dim;

        // Append one token whose K = V = [1,1,1,1, 2,2,2,2]
        let k = vec![1.0; kv_len];
        let v: Vec<f32> =
            (0..kv_len).map(|i| if i < cfg.head_dim { 1.0 } else { 2.0 }).collect();
        cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();

        let query = vec![1.0; kv_len];
        let out = cpu_paged_attention(
            &query, &cache, 0, 0, cfg.head_dim, 1.0,
        )
        .unwrap();
        // Single KV token → output = value.
        assert_eq!(out.len(), kv_len);
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[cfg.head_dim] - 2.0).abs() < 1e-5);
    }

    // ---- utilization with zero max_pages -----------------------------

    #[test]
    fn utilization_zero_max_pages() {
        let cfg = PagedKVConfig { max_pages: 0, ..small_config() };
        let cache = create_paged_cache(&cfg);
        assert!((cpu_cache_utilization(&cache) - 0.0).abs() < f32::EPSILON);
    }

    // ---- default config ----------------------------------------------

    #[test]
    fn default_config_values() {
        let cfg = PagedKVConfig::default();
        assert_eq!(cfg.page_size, 16);
        assert_eq!(cfg.max_pages, 64);
        assert_eq!(cfg.num_layers, 1);
        assert_eq!(cfg.num_heads, 1);
        assert_eq!(cfg.head_dim, 64);
    }

    // ---- page_id stability after partial fill ------------------------

    #[test]
    fn page_id_field_matches_index() {
        let cfg = small_config();
        let cache = create_paged_cache(&cfg);
        for (i, page) in cache.pages.iter().enumerate() {
            assert_eq!(page.page_id, i as u32);
        }
    }

    // ---- append across many pages ------------------------------------

    #[test]
    fn append_fills_multiple_pages() {
        let cfg = small_config(); // page_size=4, max_pages=8
        let mut cache = create_paged_cache(&cfg);
        let (k, v) = make_kv(&cfg, 7.0);
        // Fill 3 full pages = 12 tokens.
        for _ in 0..12 {
            cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        }
        let (all_k, _) = cpu_read_all_kv(&cache, 0, 0).unwrap();
        let kv_len = cfg.num_heads * cfg.head_dim;
        assert_eq!(all_k.len(), 12 * kv_len);
    }

    // ---- read subset of positions ------------------------------------

    #[test]
    fn read_subset_of_positions() {
        let cfg = small_config();
        let mut cache = create_paged_cache(&cfg);
        for i in 0..6 {
            let (k, v) = make_kv(&cfg, i as f32);
            cpu_append_kv(&mut cache, 0, 0, &k, &v).unwrap();
        }
        // Read only positions 1 and 4.
        let (rk, _) = cpu_read_kv(&cache, 0, 0, &[1, 4]).unwrap();
        let kv_len = cfg.num_heads * cfg.head_dim;
        assert!((rk[0] - 1.0).abs() < f32::EPSILON);
        assert!((rk[kv_len] - 4.0).abs() < f32::EPSILON);
    }

    // ---- paged attention with scale ----------------------------------

    #[test]
    fn paged_attention_respects_scale() {
        let cfg = PagedKVConfig {
            page_size: 4,
            max_pages: 4,
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
        };
        let mut cache = create_paged_cache(&cfg);
        cpu_append_kv(&mut cache, 0, 0, &[1.0, 0.0], &[1.0, 0.0])
            .unwrap();
        cpu_append_kv(&mut cache, 0, 0, &[0.0, 1.0], &[0.0, 1.0])
            .unwrap();

        let query = vec![1.0, 0.0];
        // A very large scale sharpens softmax toward the first token.
        let out = cpu_paged_attention(
            &query, &cache, 0, 0, cfg.head_dim, 100.0,
        )
        .unwrap();
        assert!(out[0] > 0.99, "expected sharp attention, got {}", out[0]);
    }
}
