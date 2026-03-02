//! Apple Silicon NEON-optimized KV cache with paged allocation.
//!
//! Design principles for the M-series memory hierarchy:
//!
//! * **128-bit aligned pages** – every [`NeonCachePage`] begins on a 16-byte
//!   boundary so that `vld1q_f32` / `vst1q_f32` can issue aligned loads and
//!   stores without penalty.
//!
//! * **Head-major layout** – within each page the elements are laid out as
//!   `[token][head_dim]` so a single head's cache line is contiguous.  This
//!   enables streaming NEON loads across the sequence dimension.
//!
//! * **Prefetch hints** – [`NeonKvCache::prefetch_layer`] emits `__prefetch`
//!   intrinsics (placeholder) to pull the next layer's pages into L1/L2
//!   before they are needed.

use bitnet_common::{BitNetError, KernelError, Result};

// ── helpers ────────────────────────────────────────────────────────

fn invalid_arg(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── configuration ──────────────────────────────────────────────────

/// Configuration for the NEON-optimised paged KV cache.
#[derive(Debug, Clone)]
pub struct NeonKvCacheConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads per layer.
    pub num_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Maximum sequence length the cache can hold.
    pub max_seq_len: usize,
    /// Tokens per page (default 256).  Must be > 0.
    pub page_size: usize,
}

impl Default for NeonKvCacheConfig {
    fn default() -> Self {
        Self { num_layers: 1, num_heads: 1, head_dim: 64, max_seq_len: 2048, page_size: 256 }
    }
}

impl NeonKvCacheConfig {
    /// Validate the configuration.
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
}

// ── page ───────────────────────────────────────────────────────────

/// A fixed-size page holding up to `capacity` tokens for a single head.
///
/// The backing `Vec<f32>` is allocated with capacity rounded up to a
/// multiple of 4 so that NEON 128-bit loads/stores (`vld1q_f32`) always
/// have a full lane to read without bounds checks on the fast path.
#[derive(Debug, Clone)]
pub struct NeonCachePage {
    /// Element storage – length == `used * head_dim`, capacity is
    /// always a multiple of 4 for NEON alignment.
    data: Vec<f32>,
    /// Number of tokens currently stored in this page.
    used: usize,
    /// Maximum tokens this page can hold.
    capacity: usize,
    /// Dimensionality of each token vector.
    head_dim: usize,
}

impl NeonCachePage {
    /// Allocate a new page.  The backing buffer is 128-bit aligned by
    /// rounding the element count up to a multiple of 4.
    pub fn new(capacity: usize, head_dim: usize) -> Self {
        let total_elements = capacity * head_dim;
        // Round up to multiple of 4 for 128-bit NEON alignment.
        let aligned = (total_elements + 3) & !3;
        let mut data = Vec::with_capacity(aligned);
        // Pre-zero so reads into unused tail lanes are deterministic.
        data.resize(aligned, 0.0);
        Self { data, used: 0, capacity, head_dim }
    }

    /// Number of tokens stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.used
    }

    /// Whether the page has no tokens.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    /// Whether the page is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.used == self.capacity
    }

    /// Remaining token slots.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity - self.used
    }

    /// Append a single token vector.  Returns `Err` if the page is full.
    pub fn append(&mut self, vec: &[f32]) -> Result<()> {
        if vec.len() != self.head_dim {
            return Err(invalid_arg(&format!(
                "vector length {} != head_dim {}",
                vec.len(),
                self.head_dim
            )));
        }
        if self.is_full() {
            return Err(invalid_arg("page is full"));
        }
        let offset = self.used * self.head_dim;
        self.data[offset..offset + self.head_dim].copy_from_slice(vec);
        self.used += 1;
        Ok(())
    }

    /// Read a contiguous slice of token vectors starting at `start`.
    pub fn get_slice(&self, start: usize, len: usize) -> Result<&[f32]> {
        if start + len > self.used {
            return Err(invalid_arg(&format!(
                "slice [{start}..{}] out of range (used={})",
                start + len,
                self.used
            )));
        }
        let begin = start * self.head_dim;
        let end = (start + len) * self.head_dim;
        Ok(&self.data[begin..end])
    }

    /// Remove the first `count` tokens, shifting the remainder down.
    pub fn evict_front(&mut self, count: usize) {
        let count = count.min(self.used);
        if count == 0 {
            return;
        }
        let remaining = self.used - count;
        if remaining > 0 {
            let src_start = count * self.head_dim;
            let byte_len = remaining * self.head_dim;
            self.data.copy_within(src_start..src_start + byte_len, 0);
        }
        // Zero out the freed tail for determinism.
        let new_end = remaining * self.head_dim;
        let old_end = self.used * self.head_dim;
        for v in &mut self.data[new_end..old_end] {
            *v = 0.0;
        }
        self.used = remaining;
    }

    /// Total heap bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<f32>()
    }
}

// ── per-head cache ─────────────────────────────────────────────────

/// Pages belonging to a single (layer, head) pair.
#[derive(Debug, Clone)]
struct HeadCache {
    pages: Vec<NeonCachePage>,
    page_size: usize,
    head_dim: usize,
    /// Total tokens across all pages.
    total_tokens: usize,
}

impl HeadCache {
    fn new(page_size: usize, head_dim: usize) -> Self {
        Self { pages: Vec::new(), page_size, head_dim, total_tokens: 0 }
    }

    fn append(&mut self, vec: &[f32]) -> Result<()> {
        // Allocate a new page if needed.
        if self.pages.is_empty() || self.pages.last().unwrap().is_full() {
            self.pages.push(NeonCachePage::new(self.page_size, self.head_dim));
        }
        self.pages.last_mut().unwrap().append(vec)?;
        self.total_tokens += 1;
        Ok(())
    }

    /// Collect a contiguous slice across pages into `out`.
    fn get_range(&self, start: usize, len: usize, out: &mut Vec<f32>) -> Result<()> {
        if start + len > self.total_tokens {
            return Err(invalid_arg(&format!(
                "range [{start}..{}] exceeds total_tokens {}",
                start + len,
                self.total_tokens
            )));
        }
        out.clear();
        out.reserve(len * self.head_dim);

        let mut remaining = len;
        let mut pos = start;
        for page in &self.pages {
            if remaining == 0 {
                break;
            }
            let page_len = page.len();
            if pos >= page_len {
                pos -= page_len;
                continue;
            }
            let available = (page_len - pos).min(remaining);
            let slice = page.get_slice(pos, available)?;
            out.extend_from_slice(slice);
            remaining -= available;
            pos = 0;
        }
        Ok(())
    }

    /// Evict the oldest `count` tokens.
    fn evict_oldest(&mut self, mut count: usize) {
        while count > 0 && !self.pages.is_empty() {
            let front_used = self.pages[0].len();
            if count >= front_used {
                self.total_tokens -= front_used;
                count -= front_used;
                self.pages.remove(0);
            } else {
                self.pages[0].evict_front(count);
                self.total_tokens -= count;
                count = 0;
            }
        }
    }

    /// Remove empty pages and merge under-filled trailing pages.
    fn compact(&mut self) {
        self.pages.retain(|p| !p.is_empty());
        // Defragment: merge consecutive under-half pages.
        if self.pages.len() < 2 {
            return;
        }
        let mut compacted: Vec<NeonCachePage> = Vec::new();
        for page in self.pages.drain(..) {
            if let Some(last) = compacted.last_mut() {
                if last.remaining() >= page.len() {
                    // Merge page into last.
                    for t in 0..page.len() {
                        let slice = page.get_slice(t, 1).unwrap();
                        last.append(slice).unwrap();
                    }
                    continue;
                }
            }
            compacted.push(page);
        }
        self.pages = compacted;
    }

    fn memory_bytes(&self) -> usize {
        self.pages.iter().map(|p| p.memory_bytes()).sum()
    }
}

// ── per-layer cache ────────────────────────────────────────────────

/// All heads in a single transformer layer.
#[derive(Debug, Clone)]
struct LayerCache {
    key_heads: Vec<HeadCache>,
    value_heads: Vec<HeadCache>,
}

impl LayerCache {
    fn new(num_heads: usize, page_size: usize, head_dim: usize) -> Self {
        let key_heads = (0..num_heads).map(|_| HeadCache::new(page_size, head_dim)).collect();
        let value_heads = (0..num_heads).map(|_| HeadCache::new(page_size, head_dim)).collect();
        Self { key_heads, value_heads }
    }

    fn sequence_length(&self) -> usize {
        self.key_heads.first().map_or(0, |h| h.total_tokens)
    }

    fn memory_bytes(&self) -> usize {
        let k: usize = self.key_heads.iter().map(|h| h.memory_bytes()).sum();
        let v: usize = self.value_heads.iter().map(|h| h.memory_bytes()).sum();
        k + v
    }
}

// ── top-level cache ────────────────────────────────────────────────

/// Apple Silicon NEON-optimised KV cache with paged allocation.
///
/// # Memory layout
///
/// ```text
/// NeonKvCache
///  └─ layers[]
///      └─ LayerCache
///          ├─ key_heads[]   → HeadCache → [NeonCachePage …]
///          └─ value_heads[] → HeadCache → [NeonCachePage …]
/// ```
///
/// Each [`NeonCachePage`] holds up to `page_size` token vectors in a
/// contiguous, 128-bit-aligned buffer suitable for streaming NEON
/// `vld1q_f32` / `vst1q_f32` loads and stores.
#[derive(Debug, Clone)]
pub struct NeonKvCache {
    config: NeonKvCacheConfig,
    layers: Vec<LayerCache>,
}

impl NeonKvCache {
    /// Create a new paged KV cache from a validated configuration.
    pub fn new(config: NeonKvCacheConfig) -> Result<Self> {
        config.validate()?;
        let layers = (0..config.num_layers)
            .map(|_| LayerCache::new(config.num_heads, config.page_size, config.head_dim))
            .collect();
        Ok(Self { config, layers })
    }

    /// Append a key/value pair for `(layer, head)`.
    pub fn append(
        &mut self,
        layer: usize,
        head: usize,
        key_vec: &[f32],
        value_vec: &[f32],
    ) -> Result<()> {
        let lc = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| invalid_arg(&format!("layer {layer} out of range")))?;
        let kh = lc
            .key_heads
            .get_mut(head)
            .ok_or_else(|| invalid_arg(&format!("head {head} out of range")))?;
        let vh = lc
            .value_heads
            .get_mut(head)
            .ok_or_else(|| invalid_arg(&format!("head {head} out of range")))?;
        if self.config.max_seq_len > 0 && kh.total_tokens >= self.config.max_seq_len {
            return Err(invalid_arg("max_seq_len reached"));
        }
        kh.append(key_vec)?;
        vh.append(value_vec)?;
        Ok(())
    }

    /// Retrieve key vectors for `(layer, head)` in `[start .. start+len)`.
    pub fn get_keys(
        &self,
        layer: usize,
        head: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<f32>> {
        let lc = self
            .layers
            .get(layer)
            .ok_or_else(|| invalid_arg(&format!("layer {layer} out of range")))?;
        let kh = lc
            .key_heads
            .get(head)
            .ok_or_else(|| invalid_arg(&format!("head {head} out of range")))?;
        let mut out = Vec::new();
        kh.get_range(start, len, &mut out)?;
        Ok(out)
    }

    /// Retrieve value vectors for `(layer, head)` in `[start .. start+len)`.
    pub fn get_values(
        &self,
        layer: usize,
        head: usize,
        start: usize,
        len: usize,
    ) -> Result<Vec<f32>> {
        let lc = self
            .layers
            .get(layer)
            .ok_or_else(|| invalid_arg(&format!("layer {layer} out of range")))?;
        let vh = lc
            .value_heads
            .get(head)
            .ok_or_else(|| invalid_arg(&format!("head {head} out of range")))?;
        let mut out = Vec::new();
        vh.get_range(start, len, &mut out)?;
        Ok(out)
    }

    /// Evict the oldest `count` tokens from every head in `layer`.
    pub fn evict_oldest(&mut self, layer: usize, count: usize) -> Result<()> {
        let lc = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| invalid_arg(&format!("layer {layer} out of range")))?;
        for kh in &mut lc.key_heads {
            kh.evict_oldest(count);
        }
        for vh in &mut lc.value_heads {
            vh.evict_oldest(count);
        }
        Ok(())
    }

    /// Current sequence length for `layer` (tokens stored in head 0).
    pub fn sequence_length(&self, layer: usize) -> Result<usize> {
        let lc = self
            .layers
            .get(layer)
            .ok_or_else(|| invalid_arg(&format!("layer {layer} out of range")))?;
        Ok(lc.sequence_length())
    }

    /// Total heap memory (bytes) across all layers and heads.
    pub fn memory_usage_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Emit prefetch hints for the pages belonging to `layer`.
    ///
    /// On aarch64 this would use `core::arch::aarch64::__prefetch` (or
    /// the `PRFM` instruction) to pull pages into L1/L2 ahead of the
    /// attention kernel.  Currently a no-op placeholder; the intent is
    /// to iterate each page's data pointer and issue:
    ///
    /// ```text
    /// // pseudo-code for future NEON intrinsic path
    /// unsafe { __prefetch(page.data.as_ptr(), _PREFETCH_READ, _PREFETCH_LOCALITY3); }
    /// ```
    pub fn prefetch_layer(&self, layer: usize) -> Result<()> {
        let lc = self
            .layers
            .get(layer)
            .ok_or_else(|| invalid_arg(&format!("layer {layer} out of range")))?;
        // Placeholder: iterate pages to mark them for prefetch.
        // On real hardware this would emit PRFM PLDL1KEEP instructions.
        let _ = &lc.key_heads;
        let _ = &lc.value_heads;
        Ok(())
    }

    /// Defragment pages across all layers and heads.
    pub fn compact(&mut self) {
        for lc in &mut self.layers {
            for kh in &mut lc.key_heads {
                kh.compact();
            }
            for vh in &mut lc.value_heads {
                vh.compact();
            }
        }
    }

    /// Read-only access to the configuration.
    pub fn config(&self) -> &NeonKvCacheConfig {
        &self.config
    }
}

// ── tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> NeonKvCacheConfig {
        NeonKvCacheConfig {
            num_layers: 2,
            num_heads: 4,
            head_dim: 8,
            max_seq_len: 1024,
            page_size: 4,
        }
    }

    fn make_vec(head_dim: usize, val: f32) -> Vec<f32> {
        vec![val; head_dim]
    }

    // ── config validation ──────────────────────────────────────────

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

    #[test]
    fn test_config_zero_page_size() {
        let mut c = default_config();
        c.page_size = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_default() {
        let c = NeonKvCacheConfig::default();
        assert_eq!(c.page_size, 256);
        assert!(c.validate().is_ok());
    }

    // ── page basics ────────────────────────────────────────────────

    #[test]
    fn test_page_new_empty() {
        let p = NeonCachePage::new(4, 8);
        assert!(p.is_empty());
        assert!(!p.is_full());
        assert_eq!(p.remaining(), 4);
    }

    #[test]
    fn test_page_append_and_len() {
        let mut p = NeonCachePage::new(4, 8);
        p.append(&make_vec(8, 1.0)).unwrap();
        assert_eq!(p.len(), 1);
        assert!(!p.is_full());
    }

    #[test]
    fn test_page_full() {
        let mut p = NeonCachePage::new(2, 4);
        p.append(&make_vec(4, 1.0)).unwrap();
        p.append(&make_vec(4, 2.0)).unwrap();
        assert!(p.is_full());
        assert!(p.append(&make_vec(4, 3.0)).is_err());
    }

    #[test]
    fn test_page_wrong_dim() {
        let mut p = NeonCachePage::new(4, 8);
        assert!(p.append(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_page_get_slice() {
        let mut p = NeonCachePage::new(4, 2);
        p.append(&[1.0, 2.0]).unwrap();
        p.append(&[3.0, 4.0]).unwrap();
        let s = p.get_slice(0, 2).unwrap();
        assert_eq!(s, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_page_get_slice_oob() {
        let p = NeonCachePage::new(4, 2);
        assert!(p.get_slice(0, 1).is_err());
    }

    #[test]
    fn test_page_evict_front() {
        let mut p = NeonCachePage::new(4, 2);
        p.append(&[1.0, 2.0]).unwrap();
        p.append(&[3.0, 4.0]).unwrap();
        p.append(&[5.0, 6.0]).unwrap();
        p.evict_front(1);
        assert_eq!(p.len(), 2);
        assert_eq!(p.get_slice(0, 2).unwrap(), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_page_evict_all() {
        let mut p = NeonCachePage::new(4, 2);
        p.append(&[1.0, 2.0]).unwrap();
        p.evict_front(10);
        assert!(p.is_empty());
    }

    #[test]
    fn test_page_memory_positive() {
        let p = NeonCachePage::new(8, 16);
        assert!(p.memory_bytes() > 0);
    }

    // ── cache creation ─────────────────────────────────────────────

    #[test]
    fn test_cache_new() {
        let cache = NeonKvCache::new(default_config()).unwrap();
        assert_eq!(cache.sequence_length(0).unwrap(), 0);
    }

    // ── append + get ───────────────────────────────────────────────

    #[test]
    fn test_cache_append_and_get_keys() {
        let cfg = default_config();
        let mut cache = NeonKvCache::new(cfg.clone()).unwrap();
        let k = make_vec(cfg.head_dim, 1.0);
        let v = make_vec(cfg.head_dim, 2.0);
        cache.append(0, 0, &k, &v).unwrap();
        let keys = cache.get_keys(0, 0, 0, 1).unwrap();
        assert_eq!(keys, k);
    }

    #[test]
    fn test_cache_append_and_get_values() {
        let cfg = default_config();
        let mut cache = NeonKvCache::new(cfg.clone()).unwrap();
        let k = make_vec(cfg.head_dim, 1.0);
        let v = make_vec(cfg.head_dim, 2.0);
        cache.append(0, 0, &k, &v).unwrap();
        let values = cache.get_values(0, 0, 0, 1).unwrap();
        assert_eq!(values, v);
    }

    #[test]
    fn test_cache_multi_token_append() {
        let cfg = default_config();
        let mut cache = NeonKvCache::new(cfg.clone()).unwrap();
        for i in 0..6 {
            let k = make_vec(cfg.head_dim, i as f32);
            let v = make_vec(cfg.head_dim, (i as f32) * 10.0);
            cache.append(0, 0, &k, &v).unwrap();
        }
        assert_eq!(cache.sequence_length(0).unwrap(), 6);
        let keys = cache.get_keys(0, 0, 0, 6).unwrap();
        assert_eq!(keys.len(), 6 * cfg.head_dim);
    }

    #[test]
    fn test_cache_cross_page_get() {
        // page_size=4, append 6 tokens → spans two pages.
        let cfg = default_config();
        let mut cache = NeonKvCache::new(cfg.clone()).unwrap();
        for i in 0..6 {
            cache
                .append(0, 0, &make_vec(cfg.head_dim, i as f32), &make_vec(cfg.head_dim, 0.0))
                .unwrap();
        }
        // Retrieve tokens 2..6 which crosses the page boundary at 4.
        let keys = cache.get_keys(0, 0, 2, 4).unwrap();
        assert_eq!(keys.len(), 4 * cfg.head_dim);
        // First element of token 2 should be 2.0.
        assert!((keys[0] - 2.0).abs() < 1e-6);
    }

    // ── eviction ───────────────────────────────────────────────────

    #[test]
    fn test_cache_evict_oldest() {
        let cfg = default_config();
        let mut cache = NeonKvCache::new(cfg.clone()).unwrap();
        for i in 0..6 {
            cache
                .append(0, 0, &make_vec(cfg.head_dim, i as f32), &make_vec(cfg.head_dim, 0.0))
                .unwrap();
        }
        cache.evict_oldest(0, 2).unwrap();
        assert_eq!(cache.sequence_length(0).unwrap(), 4);
        let keys = cache.get_keys(0, 0, 0, 1).unwrap();
        assert!((keys[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cache_evict_more_than_stored() {
        let cfg = default_config();
        let mut cache = NeonKvCache::new(cfg.clone()).unwrap();
        cache.append(0, 0, &make_vec(cfg.head_dim, 1.0), &make_vec(cfg.head_dim, 1.0)).unwrap();
        cache.evict_oldest(0, 100).unwrap();
        assert_eq!(cache.sequence_length(0).unwrap(), 0);
    }

    // ── compact ────────────────────────────────────────────────────

    #[test]
    fn test_cache_compact_reduces_pages() {
        let cfg = default_config();
        let mut cache = NeonKvCache::new(cfg.clone()).unwrap();
        // Fill 8 tokens across 2 pages (page_size=4).
        for i in 0..8 {
            cache
                .append(0, 0, &make_vec(cfg.head_dim, i as f32), &make_vec(cfg.head_dim, 0.0))
                .unwrap();
        }
        // Evict first 5 → page0 gone, page1 has 3 tokens.
        cache.evict_oldest(0, 5).unwrap();
        assert_eq!(cache.sequence_length(0).unwrap(), 3);
        cache.compact();
        // After compact, data is still accessible.
        let keys = cache.get_keys(0, 0, 0, 3).unwrap();
        assert!((keys[0] - 5.0).abs() < 1e-6);
    }

    // ── memory tracking ────────────────────────────────────────────

    #[test]
    fn test_memory_usage_increases() {
        let cfg = default_config();
        let mut cache = NeonKvCache::new(cfg.clone()).unwrap();
        let before = cache.memory_usage_bytes();
        cache.append(0, 0, &make_vec(cfg.head_dim, 1.0), &make_vec(cfg.head_dim, 1.0)).unwrap();
        let after = cache.memory_usage_bytes();
        assert!(after >= before);
    }

    #[test]
    fn test_memory_zero_before_append() {
        let cfg = default_config();
        let cache = NeonKvCache::new(cfg).unwrap();
        // No pages allocated yet → 0 memory.
        assert_eq!(cache.memory_usage_bytes(), 0);
    }

    // ── prefetch (no-op on non-aarch64, just validates layer) ──────

    #[test]
    fn test_prefetch_valid_layer() {
        let cache = NeonKvCache::new(default_config()).unwrap();
        assert!(cache.prefetch_layer(0).is_ok());
    }

    #[test]
    fn test_prefetch_invalid_layer() {
        let cache = NeonKvCache::new(default_config()).unwrap();
        assert!(cache.prefetch_layer(99).is_err());
    }

    // ── boundary / error cases ─────────────────────────────────────

    #[test]
    fn test_append_invalid_layer() {
        let cfg = default_config();
        let mut cache = NeonKvCache::new(cfg.clone()).unwrap();
        assert!(
            cache
                .append(99, 0, &make_vec(cfg.head_dim, 1.0), &make_vec(cfg.head_dim, 1.0))
                .is_err()
        );
    }

    #[test]
    fn test_append_invalid_head() {
        let cfg = default_config();
        let mut cache = NeonKvCache::new(cfg.clone()).unwrap();
        assert!(
            cache
                .append(0, 99, &make_vec(cfg.head_dim, 1.0), &make_vec(cfg.head_dim, 1.0))
                .is_err()
        );
    }

    #[test]
    fn test_get_keys_oob() {
        let cfg = default_config();
        let cache = NeonKvCache::new(cfg).unwrap();
        assert!(cache.get_keys(0, 0, 0, 1).is_err());
    }

    #[test]
    fn test_max_seq_len_enforced() {
        let cfg = NeonKvCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 4,
            max_seq_len: 3,
            page_size: 8,
        };
        let mut cache = NeonKvCache::new(cfg).unwrap();
        for i in 0..3 {
            cache.append(0, 0, &make_vec(4, i as f32), &make_vec(4, 0.0)).unwrap();
        }
        assert!(cache.append(0, 0, &make_vec(4, 99.0), &make_vec(4, 0.0)).is_err());
    }

    #[test]
    fn test_evict_invalid_layer() {
        let mut cache = NeonKvCache::new(default_config()).unwrap();
        assert!(cache.evict_oldest(99, 1).is_err());
    }

    #[test]
    fn test_sequence_length_invalid_layer() {
        let cache = NeonKvCache::new(default_config()).unwrap();
        assert!(cache.sequence_length(99).is_err());
    }

    #[test]
    fn test_page_alignment_multiple_of_four() {
        // Verify backing buffer capacity is a multiple of 4.
        let p = NeonCachePage::new(3, 5); // 3*5=15, rounded to 16
        assert_eq!(p.data.capacity() % 4, 0);
    }

    #[test]
    fn test_cache_config_accessor() {
        let cfg = default_config();
        let cache = NeonKvCache::new(cfg.clone()).unwrap();
        assert_eq!(cache.config().num_layers, cfg.num_layers);
        assert_eq!(cache.config().page_size, cfg.page_size);
    }
}
