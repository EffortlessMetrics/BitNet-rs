//! Optimized key-value cache for transformer attention in GPU inference.
//!
//! Supports multiple eviction strategies (FIFO, LRU, Sliding Window,
//! `StreamingLLM`) and optional cache quantization for memory reduction.

use std::fmt;

/// Key/value pair slices returned from cache lookups.
pub type KvSlices<'a> = (&'a [Vec<f32>], &'a [Vec<f32>]);

// ── Configuration ───────────────────────────────────────────────────────────

/// Data type used for cached key/value tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheDtype {
    F32,
    F16,
    BF16,
    I8,
}

impl CacheDtype {
    /// Bytes consumed per scalar element.
    #[must_use]
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
        }
    }
}

/// Cache eviction strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictionStrategy {
    /// First-in, first-out — drops the oldest entries.
    Fifo,
    /// Least-recently-used — drops entries with the oldest access time.
    Lru,
    /// Keeps only the most recent `window_size` tokens.
    SlidingWindow { window_size: usize },
    /// Keeps `attention_sinks` initial tokens plus `recent_tokens` latest.
    StreamingLlm { attention_sinks: usize, recent_tokens: usize },
}

/// Full configuration for a [`KvCache`].
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub dtype: CacheDtype,
    pub strategy: EvictionStrategy,
    pub quantize_cache: bool,
}

// ── Per-layer cache ─────────────────────────────────────────────────────────

/// Cache for a single transformer layer.
#[derive(Debug, Clone)]
pub struct LayerCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    seq_len: usize,
    capacity: usize,
    /// Per-position access counter for LRU eviction.
    access_order: Vec<u64>,
    access_counter: u64,
    /// Current dtype (may change after quantization).
    dtype: CacheDtype,
}

impl LayerCache {
    fn new(capacity: usize) -> Self {
        Self {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            seq_len: 0,
            capacity,
            access_order: Vec::with_capacity(capacity),
            access_counter: 0,
            dtype: CacheDtype::F32,
        }
    }

    fn append(&mut self, key: Vec<f32>, value: Vec<f32>) {
        self.keys.push(key);
        self.values.push(value);
        self.access_counter += 1;
        self.access_order.push(self.access_counter);
        self.seq_len += 1;
    }

    fn get(&mut self, start: usize, len: usize) -> Option<KvSlices<'_>> {
        let end = start + len;
        if end > self.seq_len {
            return None;
        }
        // Update LRU access timestamps for the retrieved range.
        for idx in start..end {
            self.access_counter += 1;
            self.access_order[idx] = self.access_counter;
        }
        Some((&self.keys[start..end], &self.values[start..end]))
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.access_order.clear();
        self.seq_len = 0;
        self.access_counter = 0;
    }

    const fn memory_bytes(&self, num_heads: usize, head_dim: usize) -> u64 {
        let elements_per_pos = num_heads * head_dim;
        let per_pos_bytes = 2 * elements_per_pos * self.dtype.bytes_per_element();
        self.seq_len as u64 * per_pos_bytes as u64
    }

    fn evict_fifo(&mut self, count: usize) -> u64 {
        let count = count.min(self.seq_len);
        self.keys.drain(..count);
        self.values.drain(..count);
        self.access_order.drain(..count);
        self.seq_len -= count;
        count as u64
    }

    fn evict_lru(&mut self, count: usize) -> u64 {
        let count = count.min(self.seq_len);
        let mut evicted = 0u64;
        for _ in 0..count {
            if self.seq_len == 0 {
                break;
            }
            // Find position with the smallest (oldest) access timestamp.
            let min_idx = self
                .access_order
                .iter()
                .enumerate()
                .min_by_key(|&(_, ts)| *ts)
                .map_or(0, |(i, _)| i);
            self.keys.remove(min_idx);
            self.values.remove(min_idx);
            self.access_order.remove(min_idx);
            self.seq_len -= 1;
            evicted += 1;
        }
        evicted
    }

    /// Keep only the last `window_size` positions.
    fn sliding_window_trim(&mut self, window_size: usize) -> u64 {
        if self.seq_len <= window_size {
            return 0;
        }
        let drop = self.seq_len - window_size;
        self.evict_fifo(drop)
    }

    /// Keep `sinks` initial positions plus the last `recent` positions.
    fn streaming_llm_trim(&mut self, sinks: usize, recent: usize) -> u64 {
        let keep = sinks + recent;
        if self.seq_len <= keep {
            return 0;
        }
        let remove_start = sinks;
        let remove_end = self.seq_len - recent;
        let remove_count = remove_end - remove_start;
        self.keys.drain(remove_start..remove_end);
        self.values.drain(remove_start..remove_end);
        self.access_order.drain(remove_start..remove_end);
        self.seq_len -= remove_count;
        remove_count as u64
    }

    fn resize(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;
        if self.seq_len > new_capacity {
            let drop = self.seq_len - new_capacity;
            self.evict_fifo(drop);
        }
    }

    fn prefill(&mut self, keys: Vec<Vec<f32>>, values: Vec<Vec<f32>>) {
        let count = keys.len();
        self.keys.extend(keys);
        self.values.extend(values);
        for _ in 0..count {
            self.access_counter += 1;
            self.access_order.push(self.access_counter);
        }
        self.seq_len += count;
    }
}

// ── Cache statistics ────────────────────────────────────────────────────────

/// Aggregate statistics for the entire [`KvCache`].
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub total_entries: u64,
    pub evictions: u64,
    pub memory_bytes: u64,
    pub hit_rate: f64,
    pub avg_seq_len: f64,
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "entries={} evictions={} mem={}B hit_rate={:.2} avg_seq={:.1}",
            self.total_entries, self.evictions, self.memory_bytes, self.hit_rate, self.avg_seq_len,
        )
    }
}

// ── Slice descriptor ────────────────────────────────────────────────────────

/// Identifies a contiguous slice within a single layer's cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheSlice {
    pub layer: usize,
    pub start_pos: usize,
    pub length: usize,
}

// ── Main KvCache ────────────────────────────────────────────────────────────

/// Multi-layer key-value cache for transformer attention.
pub struct KvCache {
    config: KvCacheConfig,
    layers: Vec<LayerCache>,
    total_evictions: u64,
    hits: u64,
    misses: u64,
}

impl KvCache {
    /// Create a new cache with the given configuration.
    #[must_use]
    pub fn new(config: KvCacheConfig) -> Self {
        let layers = (0..config.num_layers).map(|_| LayerCache::new(config.max_seq_len)).collect();
        Self { config, layers, total_evictions: 0, hits: 0, misses: 0 }
    }

    /// Append a single key/value pair to the given layer, applying eviction
    /// when the layer is at capacity.
    pub fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) {
        let lc = &mut self.layers[layer];
        if lc.seq_len >= lc.capacity {
            let evicted = match &self.config.strategy {
                EvictionStrategy::Fifo => lc.evict_fifo(1),
                EvictionStrategy::Lru => lc.evict_lru(1),
                EvictionStrategy::SlidingWindow { window_size } => {
                    lc.sliding_window_trim(*window_size)
                }
                EvictionStrategy::StreamingLlm { attention_sinks, recent_tokens } => {
                    lc.streaming_llm_trim(*attention_sinks, *recent_tokens)
                }
            };
            self.total_evictions += evicted;
        }
        lc.append(key, value);
    }

    /// Retrieve a range of cached key/value pairs from a layer.
    pub fn get(&mut self, layer: usize, start: usize, len: usize) -> Option<KvSlices<'_>> {
        let result = self.layers[layer].get(start, len);
        if result.is_some() {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        result
    }

    /// Current sequence length of the first layer (all layers grow equally
    /// in typical usage).
    #[must_use]
    pub fn current_seq_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.seq_len)
    }

    /// Total memory consumed across all layers.
    #[must_use]
    pub fn memory_usage(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| l.memory_bytes(self.config.num_heads, self.config.head_dim))
            .sum()
    }

    /// Force eviction of `count` entries from the specified layer.
    pub fn evict(&mut self, layer: usize, count: usize) {
        let evicted = match &self.config.strategy {
            EvictionStrategy::Lru => self.layers[layer].evict_lru(count),
            EvictionStrategy::Fifo
            | EvictionStrategy::SlidingWindow { .. }
            | EvictionStrategy::StreamingLlm { .. } => self.layers[layer].evict_fifo(count),
        };
        self.total_evictions += evicted;
    }

    /// Clear all layers.
    pub fn clear(&mut self) {
        for l in &mut self.layers {
            l.clear();
        }
    }

    /// Resize capacity of every layer.
    pub fn resize(&mut self, new_max_len: usize) {
        self.config.max_seq_len = new_max_len;
        for l in &mut self.layers {
            l.resize(new_max_len);
        }
    }

    /// Simulate quantization of a layer's cache to a smaller dtype,
    /// reducing reported memory but keeping f32 data for correctness.
    pub fn quantize_layer(&mut self, layer: usize, dtype: CacheDtype) {
        self.layers[layer].dtype = dtype;
    }

    /// Apply sliding-window trim to a specific layer.
    pub fn sliding_window_trim(&mut self, layer: usize) {
        if let EvictionStrategy::SlidingWindow { window_size } = &self.config.strategy {
            let ws = *window_size;
            let evicted = self.layers[layer].sliding_window_trim(ws);
            self.total_evictions += evicted;
        }
    }

    /// Apply `StreamingLLM` trim to a specific layer.
    pub fn streaming_llm_trim(&mut self, layer: usize) {
        if let EvictionStrategy::StreamingLlm { attention_sinks, recent_tokens } =
            &self.config.strategy
        {
            let (s, r) = (*attention_sinks, *recent_tokens);
            let evicted = self.layers[layer].streaming_llm_trim(s, r);
            self.total_evictions += evicted;
        }
    }

    /// Aggregate statistics across all layers.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn stats(&self) -> CacheStats {
        let total_entries: u64 = self.layers.iter().map(|l| l.seq_len as u64).sum();
        let num_layers = self.layers.len().max(1) as f64;
        let avg_seq_len = self.layers.iter().map(|l| l.seq_len as f64).sum::<f64>() / num_layers;
        let total_attempts = self.hits + self.misses;
        let hit_rate =
            if total_attempts == 0 { 0.0 } else { self.hits as f64 / total_attempts as f64 };
        CacheStats {
            total_entries,
            evictions: self.total_evictions,
            memory_bytes: self.memory_usage(),
            hit_rate,
            avg_seq_len,
        }
    }

    /// Bulk insert pre-computed key/value pairs into a layer.
    pub fn prefill(&mut self, layer: usize, keys: Vec<Vec<f32>>, values: Vec<Vec<f32>>) {
        self.layers[layer].prefill(keys, values);
    }

    /// Read-only access to the configuration.
    #[must_use]
    pub const fn config(&self) -> &KvCacheConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;

    // Helpers ────────────────────────────────────────────────────────────

    fn default_config(max_seq: usize) -> KvCacheConfig {
        KvCacheConfig {
            num_layers: 2,
            num_heads: 4,
            head_dim: 8,
            max_seq_len: max_seq,
            dtype: CacheDtype::F32,
            strategy: EvictionStrategy::Fifo,
            quantize_cache: false,
        }
    }

    fn make_vec(val: f32, len: usize) -> Vec<f32> {
        vec![val; len]
    }

    fn dim(cfg: &KvCacheConfig) -> usize {
        cfg.num_heads * cfg.head_dim
    }

    // ── Basic append / retrieve ────────────────────────────────────────

    #[test]
    fn append_and_retrieve_single() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));

        let (k, v) = cache.get(0, 0, 1).unwrap();
        assert_eq!(k.len(), 1);
        assert_eq!(v.len(), 1);
        assert_eq!(k[0][0], 1.0);
        assert_eq!(v[0][0], 2.0);
    }

    #[test]
    fn append_multiple_retrieve_range() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..5 {
            cache.append(0, make_vec(i as f32, d), make_vec((i * 10) as f32, d));
        }
        let (k, v) = cache.get(0, 1, 3).unwrap();
        assert_eq!(k.len(), 3);
        assert_eq!(k[0][0], 1.0);
        assert_eq!(k[2][0], 3.0);
        assert_eq!(v[0][0], 10.0);
    }

    #[test]
    fn get_out_of_range_returns_none() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        assert!(cache.get(0, 0, 5).is_none());
    }

    #[test]
    fn empty_cache_returns_none() {
        let cfg = default_config(16);
        let mut cache = KvCache::new(cfg);
        assert!(cache.get(0, 0, 1).is_none());
    }

    #[test]
    fn current_seq_len_tracks_appends() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        assert_eq!(cache.current_seq_len(), 0);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        assert_eq!(cache.current_seq_len(), 1);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        assert_eq!(cache.current_seq_len(), 2);
    }

    #[test]
    fn sequential_append_grows_correctly() {
        let cfg = default_config(100);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..50 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        assert_eq!(cache.current_seq_len(), 50);
        let (k, _) = cache.get(0, 49, 1).unwrap();
        assert_eq!(k[0][0], 49.0);
    }

    // ── FIFO eviction ──────────────────────────────────────────────────

    #[test]
    fn fifo_evicts_oldest_when_full() {
        let cfg = default_config(4);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..4 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        // This append should evict position 0.
        cache.append(0, make_vec(99.0, d), make_vec(0.0, d));
        assert_eq!(cache.current_seq_len(), 4);
        let (k, _) = cache.get(0, 0, 1).unwrap();
        assert_eq!(k[0][0], 1.0); // old pos 0 (val 0.0) is gone
    }

    #[test]
    fn fifo_eviction_count_tracked() {
        let cfg = default_config(2);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(0.0, d), make_vec(0.0, d));
        cache.append(0, make_vec(1.0, d), make_vec(0.0, d));
        cache.append(0, make_vec(2.0, d), make_vec(0.0, d));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn fifo_multiple_evictions() {
        let cfg = default_config(3);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..6 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        assert_eq!(cache.current_seq_len(), 3);
        assert_eq!(cache.stats().evictions, 3);
        let (k, _) = cache.get(0, 0, 1).unwrap();
        assert_eq!(k[0][0], 3.0);
    }

    #[test]
    fn explicit_evict_fifo() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..5 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.evict(0, 2);
        let (k, _) = cache.get(0, 0, 1).unwrap();
        assert_eq!(k[0][0], 2.0);
        assert_eq!(cache.stats().evictions, 2);
    }

    // ── LRU eviction ───────────────────────────────────────────────────

    #[test]
    fn lru_evicts_least_recently_used() {
        let mut cfg = default_config(4);
        cfg.strategy = EvictionStrategy::Lru;
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);

        for i in 0..4 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        // Access positions 1 and 3 to make them recent.
        cache.get(0, 1, 1);
        cache.get(0, 3, 1);
        // Append triggers eviction of LRU entry (position 0).
        cache.append(0, make_vec(99.0, d), make_vec(0.0, d));
        assert_eq!(cache.current_seq_len(), 4);
        let (k, _) = cache.get(0, 0, cache.current_seq_len()).unwrap();
        // Position 0 (val 0.0) was evicted; remaining should not contain it.
        assert!(!k.iter().map(|v| v[0]).any(|x| x == 0.0));
    }

    #[test]
    fn lru_evicts_second_oldest_after_access() {
        let mut cfg = default_config(3);
        cfg.strategy = EvictionStrategy::Lru;
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);

        cache.append(0, make_vec(0.0, d), make_vec(0.0, d));
        cache.append(0, make_vec(1.0, d), make_vec(0.0, d));
        cache.append(0, make_vec(2.0, d), make_vec(0.0, d));
        // Touch position 0, making position 1 the LRU.
        cache.get(0, 0, 1);
        cache.append(0, make_vec(3.0, d), make_vec(0.0, d));
        let (k, _) = cache.get(0, 0, cache.current_seq_len()).unwrap();
        assert!(!k.iter().map(|v| v[0]).any(|x| x == 1.0));
        assert!(k.iter().map(|v| v[0]).any(|x| x == 0.0));
    }

    #[test]
    fn lru_eviction_count_tracked() {
        let mut cfg = default_config(2);
        cfg.strategy = EvictionStrategy::Lru;
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(0.0, d), make_vec(0.0, d));
        cache.append(0, make_vec(1.0, d), make_vec(0.0, d));
        cache.append(0, make_vec(2.0, d), make_vec(0.0, d));
        assert_eq!(cache.stats().evictions, 1);
    }

    // ── Sliding window ─────────────────────────────────────────────────

    #[test]
    fn sliding_window_keeps_recent_n() {
        let mut cfg = default_config(16);
        cfg.strategy = EvictionStrategy::SlidingWindow { window_size: 3 };
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..8 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.sliding_window_trim(0);
        let (k, _) = cache.get(0, 0, 3).unwrap();
        assert_eq!(k[0][0], 5.0);
        assert_eq!(k[2][0], 7.0);
    }

    #[test]
    fn sliding_window_noop_under_window() {
        let mut cfg = default_config(16);
        cfg.strategy = EvictionStrategy::SlidingWindow { window_size: 10 };
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..5 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.sliding_window_trim(0);
        assert_eq!(cache.current_seq_len(), 5);
    }

    #[test]
    fn sliding_window_eviction_on_full() {
        let mut cfg = default_config(4);
        cfg.strategy = EvictionStrategy::SlidingWindow { window_size: 3 };
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..4 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        // One more append should trigger SlidingWindow eviction.
        cache.append(0, make_vec(4.0, d), make_vec(0.0, d));
        assert!(cache.current_seq_len() <= 4);
    }

    #[test]
    fn sliding_window_eviction_count() {
        let mut cfg = default_config(16);
        cfg.strategy = EvictionStrategy::SlidingWindow { window_size: 2 };
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..6 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.sliding_window_trim(0);
        assert!(cache.stats().evictions > 0);
    }

    // ── StreamingLLM ───────────────────────────────────────────────────

    #[test]
    fn streaming_llm_keeps_sinks_and_recent() {
        let mut cfg = default_config(16);
        cfg.strategy = EvictionStrategy::StreamingLlm { attention_sinks: 2, recent_tokens: 3 };
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..10 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.streaming_llm_trim(0);
        let len = cache.current_seq_len();
        assert_eq!(len, 5); // 2 sinks + 3 recent
        let (k, _) = cache.get(0, 0, len).unwrap();
        // First two should be the original sinks.
        assert_eq!(k[0][0], 0.0);
        assert_eq!(k[1][0], 1.0);
        // Last three should be the most recent.
        assert_eq!(k[2][0], 7.0);
        assert_eq!(k[3][0], 8.0);
        assert_eq!(k[4][0], 9.0);
    }

    #[test]
    fn streaming_llm_noop_under_threshold() {
        let mut cfg = default_config(16);
        cfg.strategy = EvictionStrategy::StreamingLlm { attention_sinks: 2, recent_tokens: 5 };
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..6 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.streaming_llm_trim(0);
        assert_eq!(cache.current_seq_len(), 6);
    }

    #[test]
    fn streaming_llm_eviction_count() {
        let mut cfg = default_config(16);
        cfg.strategy = EvictionStrategy::StreamingLlm { attention_sinks: 1, recent_tokens: 1 };
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..5 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.streaming_llm_trim(0);
        assert_eq!(cache.stats().evictions, 3);
    }

    #[test]
    fn streaming_llm_eviction_on_capacity() {
        let mut cfg = default_config(5);
        cfg.strategy = EvictionStrategy::StreamingLlm { attention_sinks: 1, recent_tokens: 3 };
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..5 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.append(0, make_vec(5.0, d), make_vec(0.0, d));
        assert!(cache.current_seq_len() <= 5);
    }

    // ── Memory ─────────────────────────────────────────────────────────

    #[test]
    fn memory_usage_zero_when_empty() {
        let cfg = default_config(16);
        let cache = KvCache::new(cfg);
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn memory_usage_correct_f32() {
        let cfg = default_config(16);
        let d = dim(&cfg); // 4 * 8 = 32
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        // 1 position * 2 (k+v) * 32 elements * 4 bytes = 256
        assert_eq!(cache.memory_usage(), 256);
    }

    #[test]
    fn memory_usage_grows_with_appends() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        let m1 = cache.memory_usage();
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        assert_eq!(cache.memory_usage(), m1 * 2);
    }

    #[test]
    fn memory_usage_both_layers() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        cache.append(1, make_vec(1.0, d), make_vec(2.0, d));
        assert_eq!(cache.memory_usage(), 512); // 256 * 2 layers
    }

    // ── Multi-layer independence ───────────────────────────────────────

    #[test]
    fn layers_are_independent() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(1.0, d));
        cache.append(1, make_vec(2.0, d), make_vec(2.0, d));

        let (k0, _) = cache.get(0, 0, 1).unwrap();
        assert_eq!(k0[0][0], 1.0);
        let (k1, _) = cache.get(1, 0, 1).unwrap();
        assert_eq!(k1[0][0], 2.0);
    }

    #[test]
    fn eviction_on_one_layer_does_not_affect_other() {
        let cfg = default_config(3);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..3 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.append(1, make_vec(99.0, d), make_vec(0.0, d));
        // Trigger eviction on layer 0.
        cache.append(0, make_vec(10.0, d), make_vec(0.0, d));
        // Layer 1 unaffected.
        let (k1, _) = cache.get(1, 0, 1).unwrap();
        assert_eq!(k1[0][0], 99.0);
    }

    #[test]
    fn layer_clear_independence() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(1.0, d));
        cache.append(1, make_vec(2.0, d), make_vec(2.0, d));
        cache.layers[0].clear();
        assert!(cache.get(0, 0, 1).is_none());
        assert!(cache.get(1, 0, 1).is_some());
    }

    // ── Clear ──────────────────────────────────────────────────────────

    #[test]
    fn clear_resets_all_layers() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(1.0, d));
        cache.append(1, make_vec(2.0, d), make_vec(2.0, d));
        cache.clear();
        assert_eq!(cache.current_seq_len(), 0);
        assert!(cache.get(0, 0, 1).is_none());
        assert!(cache.get(1, 0, 1).is_none());
    }

    #[test]
    fn clear_resets_memory() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(1.0, d));
        cache.clear();
        assert_eq!(cache.memory_usage(), 0);
    }

    // ── Resize ─────────────────────────────────────────────────────────

    #[test]
    fn resize_increases_capacity() {
        let cfg = default_config(4);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..4 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.resize(8);
        cache.append(0, make_vec(10.0, d), make_vec(0.0, d));
        assert_eq!(cache.current_seq_len(), 5);
    }

    #[test]
    fn resize_decreases_drops_excess() {
        let cfg = default_config(8);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..6 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.resize(3);
        assert_eq!(cache.current_seq_len(), 3);
        let (k, _) = cache.get(0, 0, 1).unwrap();
        assert_eq!(k[0][0], 3.0); // oldest 3 evicted
    }

    #[test]
    fn resize_noop_when_equal() {
        let cfg = default_config(4);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..4 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        cache.resize(4);
        assert_eq!(cache.current_seq_len(), 4);
    }

    // ── Prefill ────────────────────────────────────────────────────────

    #[test]
    fn prefill_bulk_insert() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        let keys: Vec<Vec<f32>> = (0..5).map(|i| make_vec(i as f32, d)).collect();
        let values: Vec<Vec<f32>> = (0..5).map(|i| make_vec((i * 10) as f32, d)).collect();
        cache.prefill(0, keys, values);
        assert_eq!(cache.current_seq_len(), 5);
        let (k, v) = cache.get(0, 2, 1).unwrap();
        assert_eq!(k[0][0], 2.0);
        assert_eq!(v[0][0], 20.0);
    }

    #[test]
    fn prefill_then_append() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        let keys: Vec<Vec<f32>> = (0..3).map(|i| make_vec(i as f32, d)).collect();
        let values = keys.clone();
        cache.prefill(0, keys, values);
        cache.append(0, make_vec(99.0, d), make_vec(99.0, d));
        assert_eq!(cache.current_seq_len(), 4);
        let (k, _) = cache.get(0, 3, 1).unwrap();
        assert_eq!(k[0][0], 99.0);
    }

    #[test]
    fn prefill_empty_is_noop() {
        let cfg = default_config(16);
        let mut cache = KvCache::new(cfg);
        cache.prefill(0, vec![], vec![]);
        assert_eq!(cache.current_seq_len(), 0);
    }

    // ── Quantization ───────────────────────────────────────────────────

    #[test]
    fn quantize_reduces_reported_memory() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        let before = cache.memory_usage();
        cache.quantize_layer(0, CacheDtype::I8);
        let after = cache.memory_usage();
        assert!(after < before);
    }

    #[test]
    fn quantize_f16_halves_memory() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        let before = cache.memory_usage();
        cache.quantize_layer(0, CacheDtype::F16);
        let after = cache.memory_usage();
        assert_eq!(after, before / 2);
    }

    #[test]
    fn quantize_i8_quarters_memory() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        let before = cache.memory_usage();
        cache.quantize_layer(0, CacheDtype::I8);
        let after = cache.memory_usage();
        assert_eq!(after, before / 4);
    }

    #[test]
    fn quantize_preserves_data_access() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(42.0, d), make_vec(7.0, d));
        cache.quantize_layer(0, CacheDtype::I8);
        let (k, v) = cache.get(0, 0, 1).unwrap();
        assert_eq!(k[0][0], 42.0);
        assert_eq!(v[0][0], 7.0);
    }

    // ── Stats ──────────────────────────────────────────────────────────

    #[test]
    fn stats_initial() {
        let cfg = default_config(16);
        let cache = KvCache::new(cfg);
        let s = cache.stats();
        assert_eq!(s.total_entries, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.memory_bytes, 0);
    }

    #[test]
    fn stats_after_appends() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(1.0, d));
        cache.append(1, make_vec(1.0, d), make_vec(1.0, d));
        let s = cache.stats();
        assert_eq!(s.total_entries, 2);
        assert!(s.memory_bytes > 0);
    }

    #[test]
    fn stats_hit_rate() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(1.0, d));
        cache.get(0, 0, 1); // hit
        cache.get(0, 0, 5); // miss
        let s = cache.stats();
        assert!((s.hit_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn stats_avg_seq_len() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(1.0, d));
        cache.append(0, make_vec(1.0, d), make_vec(1.0, d));
        // Layer 0: 2, Layer 1: 0 → avg = 1.0
        let s = cache.stats();
        assert!((s.avg_seq_len - 1.0).abs() < 0.01);
    }

    #[test]
    fn stats_display() {
        let cfg = default_config(16);
        let cache = KvCache::new(cfg);
        let s = cache.stats();
        let display = format!("{s}");
        assert!(display.contains("entries="));
        assert!(display.contains("evictions="));
    }

    #[test]
    fn stats_eviction_count_accumulates() {
        let cfg = default_config(2);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        for i in 0..10 {
            cache.append(0, make_vec(i as f32, d), make_vec(0.0, d));
        }
        assert_eq!(cache.stats().evictions, 8);
    }

    // ── Head dim / num_heads interaction ────────────────────────────────

    #[test]
    fn head_dim_and_num_heads_memory() {
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_heads: 8,
            head_dim: 16,
            max_seq_len: 16,
            dtype: CacheDtype::F32,
            strategy: EvictionStrategy::Fifo,
            quantize_cache: false,
        };
        let d = cfg.num_heads * cfg.head_dim; // 128
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        // 1 pos * 2 * 128 * 4 = 1024
        assert_eq!(cache.memory_usage(), 1024);
    }

    #[test]
    fn single_head_single_dim() {
        let cfg = KvCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 1,
            max_seq_len: 16,
            dtype: CacheDtype::F32,
            strategy: EvictionStrategy::Fifo,
            quantize_cache: false,
        };
        let mut cache = KvCache::new(cfg);
        cache.append(0, vec![1.0], vec![2.0]);
        // 1 pos * 2 * 1 * 4 = 8
        assert_eq!(cache.memory_usage(), 8);
        let (k, v) = cache.get(0, 0, 1).unwrap();
        assert_eq!(k[0], &[1.0]);
        assert_eq!(v[0], &[2.0]);
    }

    // ── CacheDtype ─────────────────────────────────────────────────────

    #[test]
    fn dtype_bytes_per_element() {
        assert_eq!(CacheDtype::F32.bytes_per_element(), 4);
        assert_eq!(CacheDtype::F16.bytes_per_element(), 2);
        assert_eq!(CacheDtype::BF16.bytes_per_element(), 2);
        assert_eq!(CacheDtype::I8.bytes_per_element(), 1);
    }

    // ── CacheSlice ─────────────────────────────────────────────────────

    #[test]
    fn cache_slice_equality() {
        let a = CacheSlice { layer: 0, start_pos: 1, length: 5 };
        let b = CacheSlice { layer: 0, start_pos: 1, length: 5 };
        assert_eq!(a, b);
    }

    #[test]
    fn cache_slice_debug() {
        let s = CacheSlice { layer: 1, start_pos: 2, length: 3 };
        let dbg = format!("{s:?}");
        assert!(dbg.contains("layer: 1"));
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn get_zero_length_returns_empty_slice() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        let (k, v) = cache.get(0, 0, 0).unwrap();
        assert!(k.is_empty());
        assert!(v.is_empty());
    }

    #[test]
    fn evict_more_than_exists() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(2.0, d));
        cache.evict(0, 100);
        assert_eq!(cache.current_seq_len(), 0);
    }

    #[test]
    fn config_accessor() {
        let cfg = default_config(16);
        let cache = KvCache::new(cfg);
        assert_eq!(cache.config().num_layers, 2);
        assert_eq!(cache.config().max_seq_len, 16);
    }

    #[test]
    fn large_batch_prefill() {
        let cfg = default_config(1024);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        let keys: Vec<Vec<f32>> = (0..512).map(|i| make_vec(i as f32, d)).collect();
        let values = keys.clone();
        cache.prefill(0, keys, values);
        assert_eq!(cache.current_seq_len(), 512);
    }

    #[test]
    fn multiple_layers_different_lengths() {
        let cfg = default_config(16);
        let d = dim(&cfg);
        let mut cache = KvCache::new(cfg);
        cache.append(0, make_vec(1.0, d), make_vec(1.0, d));
        cache.append(0, make_vec(2.0, d), make_vec(2.0, d));
        cache.append(1, make_vec(3.0, d), make_vec(3.0, d));
        let s = cache.stats();
        assert_eq!(s.total_entries, 3);
        assert!((s.avg_seq_len - 1.5).abs() < 0.01);
    }
}
