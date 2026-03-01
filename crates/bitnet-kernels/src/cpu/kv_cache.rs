//! CPU KV cache operations kernel.
//!
//! Provides allocation, append, slice, clear, and memory-tracking
//! operations for key/value caches used during autoregressive inference.
//! Supports both contiguous per-layer caches and paged allocation.

use bitnet_common::{BitNetError, KernelError, Result};

// ── Helper ─────────────────────────────────────────────────────────

fn invalid_arg(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Data-type descriptor ───────────────────────────────────────────

/// Supported element data types for cache storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvDtype {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 half-precision float.
    F16,
    /// 16-bit brain floating-point.
    Bf16,
}

impl KvDtype {
    /// Size in bytes of a single element.
    #[inline]
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

// ── Configuration ──────────────────────────────────────────────────

/// Fully describes the shape of a KV cache.
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub num_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Maximum sequence length the cache can hold.
    pub max_seq_len: usize,
    /// Element data type.
    pub dtype: KvDtype,
}

impl KvCacheConfig {
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
        Ok(())
    }

    /// Number of f32 elements per token across all heads.
    #[inline]
    fn token_elements(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

// ── Per-layer block ────────────────────────────────────────────────

/// Key and value tensors for a single transformer layer.
///
/// Storage is `[num_heads * head_dim]` per token, appended sequentially.
/// `seq_len` tracks the current number of cached tokens.
#[derive(Debug, Clone)]
pub struct KvCacheBlock {
    /// Cached key vectors: `[seq_len, num_heads * head_dim]` flattened.
    pub keys: Vec<f32>,
    /// Cached value vectors: `[seq_len, num_heads * head_dim]` flattened.
    pub values: Vec<f32>,
    /// Current number of tokens stored.
    pub seq_len: usize,
    /// Number of elements per token (`num_heads * head_dim`).
    token_elements: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
}

impl KvCacheBlock {
    /// Allocate a new block pre-sized for `max_seq_len` tokens.
    fn new(token_elements: usize, max_seq_len: usize) -> Self {
        let cap = max_seq_len * token_elements;
        Self {
            keys: vec![0.0; cap],
            values: vec![0.0; cap],
            seq_len: 0,
            token_elements,
            max_seq_len,
        }
    }

    /// Remaining token capacity.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.max_seq_len - self.seq_len
    }

    /// Append `new_tokens` key/value pairs.
    fn append(&mut self, new_keys: &[f32], new_values: &[f32]) -> Result<()> {
        let new_tokens = new_keys.len() / self.token_elements;
        if new_keys.len() != new_tokens * self.token_elements {
            return Err(invalid_arg("new_keys length is not a multiple of token_elements"));
        }
        if new_values.len() != new_keys.len() {
            return Err(invalid_arg("new_keys and new_values must have the same length"));
        }
        if new_tokens > self.remaining() {
            return Err(invalid_arg("append would exceed max_seq_len"));
        }
        let offset = self.seq_len * self.token_elements;
        let n = new_keys.len();
        self.keys[offset..offset + n].copy_from_slice(new_keys);
        self.values[offset..offset + n].copy_from_slice(new_values);
        self.seq_len += new_tokens;
        Ok(())
    }

    /// Return slices `(keys, values)` for tokens in `[start, end)`.
    fn slice(&self, start: usize, end: usize) -> Result<(&[f32], &[f32])> {
        if start > end {
            return Err(invalid_arg("start must be <= end"));
        }
        if end > self.seq_len {
            return Err(invalid_arg("end exceeds current seq_len"));
        }
        let s = start * self.token_elements;
        let e = end * self.token_elements;
        Ok((&self.keys[s..e], &self.values[s..e]))
    }

    /// Reset the block to empty (zero-length) without de-allocating.
    fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Memory occupied by key + value buffers in bytes.
    fn memory_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * size_of::<f32>()
    }
}

// ── Multi-layer cache ──────────────────────────────────────────────

/// Multi-layer KV cache wrapping one [`KvCacheBlock`] per transformer layer.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Per-layer cache blocks.
    pub blocks: Vec<KvCacheBlock>,
    /// Configuration snapshot.
    pub config: KvCacheConfig,
}

impl KvCache {
    /// Allocate a new cache from a validated configuration.
    pub fn new(config: KvCacheConfig) -> Result<Self> {
        config.validate()?;
        let te = config.token_elements();
        let blocks =
            (0..config.num_layers).map(|_| KvCacheBlock::new(te, config.max_seq_len)).collect();
        Ok(Self { blocks, config })
    }

    /// Number of layers in the cache.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Current sequence length for a given layer.
    pub fn seq_len(&self, layer: usize) -> Result<usize> {
        self.block(layer).map(|b| b.seq_len)
    }

    fn block(&self, layer: usize) -> Result<&KvCacheBlock> {
        self.blocks.get(layer).ok_or_else(|| invalid_arg("layer index out of range"))
    }

    fn block_mut(&mut self, layer: usize) -> Result<&mut KvCacheBlock> {
        let n = self.blocks.len();
        self.blocks
            .get_mut(layer)
            .ok_or_else(|| invalid_arg(&format!("layer {layer} out of range (num_layers={n})")))
    }
}

// ── Public kernel functions ────────────────────────────────────────

/// Append new key/value pairs at `layer`.
///
/// `new_keys` and `new_values` must have length `num_tokens * num_heads * head_dim`.
pub fn kv_cache_append(
    cache: &mut KvCache,
    layer: usize,
    new_keys: &[f32],
    new_values: &[f32],
) -> Result<()> {
    cache.block_mut(layer)?.append(new_keys, new_values)
}

/// Return `(keys, values)` slices for tokens `[start, end)` at `layer`.
pub fn kv_cache_slice(
    cache: &KvCache,
    layer: usize,
    start: usize,
    end: usize,
) -> Result<(&[f32], &[f32])> {
    cache.block(layer)?.slice(start, end)
}

/// Reset every layer's cache to empty.
pub fn kv_cache_clear(cache: &mut KvCache) {
    for block in &mut cache.blocks {
        block.clear();
    }
}

/// Total memory used by the cache in bytes.
pub fn kv_cache_memory_usage(cache: &KvCache) -> usize {
    cache.blocks.iter().map(KvCacheBlock::memory_bytes).sum()
}

/// Allocate a vector of independent [`KvCacheBlock`]s for paged KV caching.
///
/// Each page holds `page_size` tokens with `num_heads * head_dim` elements each.
pub fn paged_kv_cache_alloc(
    num_pages: usize,
    page_size: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<KvCacheBlock>> {
    if num_pages == 0 {
        return Err(invalid_arg("num_pages must be > 0"));
    }
    if page_size == 0 {
        return Err(invalid_arg("page_size must be > 0"));
    }
    if num_heads == 0 {
        return Err(invalid_arg("num_heads must be > 0"));
    }
    if head_dim == 0 {
        return Err(invalid_arg("head_dim must be > 0"));
    }
    let te = num_heads * head_dim;
    Ok((0..num_pages).map(|_| KvCacheBlock::new(te, page_size)).collect())
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> KvCacheConfig {
        KvCacheConfig {
            num_layers: 2,
            num_heads: 4,
            head_dim: 8,
            max_seq_len: 16,
            dtype: KvDtype::F32,
        }
    }

    // -- Config validation ---------------------------------------------------

    #[test]
    fn test_config_valid() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn test_config_zero_layers() {
        let mut c = default_config();
        c.num_layers = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_heads() {
        let mut c = default_config();
        c.num_heads = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_head_dim() {
        let mut c = default_config();
        c.head_dim = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_max_seq_len() {
        let mut c = default_config();
        c.max_seq_len = 0;
        assert!(c.validate().is_err());
    }

    // -- KvDtype -------------------------------------------------------------

    #[test]
    fn test_dtype_element_bytes() {
        assert_eq!(KvDtype::F32.element_bytes(), 4);
        assert_eq!(KvDtype::F16.element_bytes(), 2);
        assert_eq!(KvDtype::Bf16.element_bytes(), 2);
    }

    // -- Cache construction --------------------------------------------------

    #[test]
    fn test_new_cache_layers() {
        let cache = KvCache::new(default_config()).unwrap();
        assert_eq!(cache.num_layers(), 2);
    }

    #[test]
    fn test_new_cache_initial_seq_len() {
        let cache = KvCache::new(default_config()).unwrap();
        assert_eq!(cache.seq_len(0).unwrap(), 0);
        assert_eq!(cache.seq_len(1).unwrap(), 0);
    }

    #[test]
    fn test_layer_out_of_range() {
        let cache = KvCache::new(default_config()).unwrap();
        assert!(cache.seq_len(99).is_err());
    }

    // -- Append --------------------------------------------------------------

    #[test]
    fn test_append_single_token() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim; // 32
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![1.0_f32; te];
        let v = vec![2.0_f32; te];
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(cache.seq_len(0).unwrap(), 1);
        assert_eq!(cache.seq_len(1).unwrap(), 0);
    }

    #[test]
    fn test_append_multiple_tokens() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![1.0; te * 3];
        let v = vec![2.0; te * 3];
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(cache.seq_len(0).unwrap(), 3);
    }

    #[test]
    fn test_append_incremental() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        for i in 0..4 {
            let k = vec![i as f32; te];
            let v = vec![(i as f32) * 10.0; te];
            kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        }
        assert_eq!(cache.seq_len(0).unwrap(), 4);
    }

    #[test]
    fn test_append_exceeds_capacity() {
        let cfg = default_config(); // max_seq_len = 16
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![0.0; te * 17];
        let v = vec![0.0; te * 17];
        assert!(kv_cache_append(&mut cache, 0, &k, &v).is_err());
    }

    #[test]
    fn test_append_mismatched_kv_lengths() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![0.0; te];
        let v = vec![0.0; te * 2];
        assert!(kv_cache_append(&mut cache, 0, &k, &v).is_err());
    }

    #[test]
    fn test_append_bad_alignment() {
        let cfg = default_config();
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![0.0; 5]; // not a multiple of token_elements (32)
        let v = vec![0.0; 5];
        assert!(kv_cache_append(&mut cache, 0, &k, &v).is_err());
    }

    #[test]
    fn test_append_invalid_layer() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![0.0; te];
        let v = vec![0.0; te];
        assert!(kv_cache_append(&mut cache, 99, &k, &v).is_err());
    }

    // -- Slice ---------------------------------------------------------------

    #[test]
    fn test_slice_after_append() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let k: Vec<f32> = (0..te as u32).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..te as u32).map(|i| (i as f32) + 100.0).collect();
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();

        let (sk, sv) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
        assert_eq!(sk, &k[..]);
        assert_eq!(sv, &v[..]);
    }

    #[test]
    fn test_slice_partial_range() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        // Append 4 tokens with ascending values.
        for t in 0..4 {
            let k = vec![(t + 1) as f32; te];
            let v = vec![((t + 1) * 10) as f32; te];
            kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        }
        let (sk, _sv) = kv_cache_slice(&cache, 0, 1, 3).unwrap();
        // Tokens 1..3 → 2 tokens.
        assert_eq!(sk.len(), 2 * te);
        // First element of token-1 should be 2.0.
        assert!((sk[0] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_slice_empty_range() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![1.0; te];
        let v = vec![2.0; te];
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        let (sk, sv) = kv_cache_slice(&cache, 0, 0, 0).unwrap();
        assert!(sk.is_empty());
        assert!(sv.is_empty());
    }

    #[test]
    fn test_slice_start_greater_than_end() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![1.0; te];
        let v = vec![2.0; te];
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        assert!(kv_cache_slice(&cache, 0, 1, 0).is_err());
    }

    #[test]
    fn test_slice_end_exceeds_seq_len() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![1.0; te];
        let v = vec![2.0; te];
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        assert!(kv_cache_slice(&cache, 0, 0, 5).is_err());
    }

    // -- Clear ---------------------------------------------------------------

    #[test]
    fn test_clear_resets_all_layers() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![1.0; te * 3];
        let v = vec![2.0; te * 3];
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        kv_cache_append(&mut cache, 1, &k, &v).unwrap();
        kv_cache_clear(&mut cache);
        assert_eq!(cache.seq_len(0).unwrap(), 0);
        assert_eq!(cache.seq_len(1).unwrap(), 0);
    }

    #[test]
    fn test_clear_then_reuse() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let k = vec![1.0; te];
        let v = vec![2.0; te];
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        kv_cache_clear(&mut cache);
        // Re-append after clear.
        kv_cache_append(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(cache.seq_len(0).unwrap(), 1);
    }

    // -- Memory usage --------------------------------------------------------

    #[test]
    fn test_memory_usage() {
        let cfg = default_config();
        let cache = KvCache::new(cfg.clone()).unwrap();
        let expected =
            cfg.num_layers * 2 * cfg.max_seq_len * cfg.num_heads * cfg.head_dim * size_of::<f32>();
        assert_eq!(kv_cache_memory_usage(&cache), expected);
    }

    #[test]
    fn test_memory_usage_unchanged_after_append() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let before = kv_cache_memory_usage(&cache);
        kv_cache_append(&mut cache, 0, &vec![0.0; te], &vec![0.0; te]).unwrap();
        assert_eq!(kv_cache_memory_usage(&cache), before);
    }

    // -- Multi-layer ---------------------------------------------------------

    #[test]
    fn test_multi_layer_independence() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        let k0 = vec![1.0; te];
        let v0 = vec![10.0; te];
        let k1 = vec![2.0; te * 2];
        let v1 = vec![20.0; te * 2];
        kv_cache_append(&mut cache, 0, &k0, &v0).unwrap();
        kv_cache_append(&mut cache, 1, &k1, &v1).unwrap();
        assert_eq!(cache.seq_len(0).unwrap(), 1);
        assert_eq!(cache.seq_len(1).unwrap(), 2);
        let (sk0, _) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
        assert!((sk0[0] - 1.0).abs() < f32::EPSILON);
        let (sk1, _) = kv_cache_slice(&cache, 1, 0, 1).unwrap();
        assert!((sk1[0] - 2.0).abs() < f32::EPSILON);
    }

    // -- Paged allocation ----------------------------------------------------

    #[test]
    fn test_paged_alloc_basic() {
        let pages = paged_kv_cache_alloc(4, 32, 8, 64).unwrap();
        assert_eq!(pages.len(), 4);
        for p in &pages {
            assert_eq!(p.max_seq_len, 32);
            assert_eq!(p.token_elements, 8 * 64);
            assert_eq!(p.seq_len, 0);
        }
    }

    #[test]
    fn test_paged_alloc_append_per_page() {
        let mut pages = paged_kv_cache_alloc(2, 8, 2, 4).unwrap();
        let te = 2 * 4;
        let k = vec![1.0; te];
        let v = vec![2.0; te];
        pages[0].append(&k, &v).unwrap();
        assert_eq!(pages[0].seq_len, 1);
        assert_eq!(pages[1].seq_len, 0);
    }

    #[test]
    fn test_paged_alloc_zero_pages() {
        assert!(paged_kv_cache_alloc(0, 8, 2, 4).is_err());
    }

    #[test]
    fn test_paged_alloc_zero_page_size() {
        assert!(paged_kv_cache_alloc(4, 0, 2, 4).is_err());
    }

    #[test]
    fn test_paged_alloc_zero_heads() {
        assert!(paged_kv_cache_alloc(4, 8, 0, 4).is_err());
    }

    #[test]
    fn test_paged_alloc_zero_head_dim() {
        assert!(paged_kv_cache_alloc(4, 8, 2, 0).is_err());
    }

    // -- Edge cases ----------------------------------------------------------

    #[test]
    fn test_append_fill_to_capacity() {
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
            max_seq_len: 4,
            dtype: KvDtype::F32,
        };
        let te = 2;
        let mut cache = KvCache::new(cfg).unwrap();
        for _ in 0..4 {
            kv_cache_append(&mut cache, 0, &[1.0; 2], &[2.0; 2]).unwrap();
        }
        assert_eq!(cache.seq_len(0).unwrap(), 4);
        // One more should fail.
        assert!(kv_cache_append(&mut cache, 0, &[1.0; 2], &[2.0; 2]).is_err());
        // Full slice succeeds.
        let (sk, _) = kv_cache_slice(&cache, 0, 0, 4).unwrap();
        assert_eq!(sk.len(), 4 * te);
    }

    #[test]
    fn test_remaining_capacity() {
        let cfg = default_config();
        let te = cfg.num_heads * cfg.head_dim;
        let mut cache = KvCache::new(cfg).unwrap();
        assert_eq!(cache.blocks[0].remaining(), 16);
        kv_cache_append(&mut cache, 0, &vec![0.0; te * 3], &vec![0.0; te * 3]).unwrap();
        assert_eq!(cache.blocks[0].remaining(), 13);
    }

    #[test]
    fn test_single_head_single_dim() {
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 1,
            max_seq_len: 8,
            dtype: KvDtype::F32,
        };
        let mut cache = KvCache::new(cfg).unwrap();
        kv_cache_append(&mut cache, 0, &[42.0], &[84.0]).unwrap();
        let (sk, sv) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
        assert!((sk[0] - 42.0).abs() < f32::EPSILON);
        assert!((sv[0] - 84.0).abs() < f32::EPSILON);
    }
}
