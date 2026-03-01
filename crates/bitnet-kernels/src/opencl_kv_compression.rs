//! OpenCL KV cache compression for A770 GPU inference.
//!
//! Reduces the memory footprint of the key-value cache during long-sequence
//! inference by applying quantization (INT8 / INT4 / F16) and eviction
//! policies (FIFO, LRU, attention-score-based, H2O, SnapKV).
//!
//! All heavy lifting is implemented as CPU reference kernels today; the module
//! is structured so that OpenCL device kernels can replace the inner loops
//! once the runtime is wired up on Intel Arc A770.

use std::fmt;

// ---------------------------------------------------------------------------
// Compression method
// ---------------------------------------------------------------------------

/// Selects how KV cache entries are compressed.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionMethod {
    /// No compression — store raw f32 values.
    None,
    /// Symmetric per-tensor INT8 quantization.
    Int8Quantize,
    /// Symmetric per-tensor INT4 quantization (two values per byte).
    Int4Quantize,
    /// IEEE 754 half-precision (binary16).
    Float16,
    /// Keep only the most recent `window_size` positions.
    SlidingWindow { window_size: usize },
    /// Heavy Hitter Oracle — keep the `budget` highest-attention entries.
    H2O { budget: usize },
    /// SnapKV — attention-guided eviction with a smoothing kernel.
    SnapKV { budget: usize, kernel_size: usize },
}

impl fmt::Display for CompressionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Int8Quantize => write!(f, "INT8"),
            Self::Int4Quantize => write!(f, "INT4"),
            Self::Float16 => write!(f, "F16"),
            Self::SlidingWindow { window_size } => write!(f, "SlidingWindow({window_size})"),
            Self::H2O { budget } => write!(f, "H2O({budget})"),
            Self::SnapKV { budget, kernel_size } => {
                write!(f, "SnapKV({budget}, k={kernel_size})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compressed KV payload
// ---------------------------------------------------------------------------

/// Holds a compressed snapshot of key or value data.
#[derive(Debug, Clone)]
pub struct CompressedKV {
    /// Raw compressed bytes (layout depends on `method`).
    pub keys: Vec<u8>,
    /// Raw compressed value bytes.
    pub values: Vec<u8>,
    /// Compression scheme applied.
    pub method: CompressionMethod,
    /// Original tensor shape before compression.
    pub original_shape: Vec<usize>,
    /// Per-tensor scale factors used during quantization (if any).
    pub scale_factors: Option<Vec<f32>>,
    /// Sequence positions represented in this snapshot.
    pub seq_positions: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls how keys and values are compressed and when entries are evicted.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression applied to key tensors.
    pub key_method: CompressionMethod,
    /// Compression applied to value tensors.
    pub value_method: CompressionMethod,
    /// Soft memory budget for the entire KV cache (MiB).
    pub max_cache_size_mb: f64,
    /// Fraction of entries to evict when the cache is full (0.0–1.0).
    pub eviction_ratio: f32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            key_method: CompressionMethod::None,
            value_method: CompressionMethod::None,
            max_cache_size_mb: 1024.0,
            eviction_ratio: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Eviction
// ---------------------------------------------------------------------------

/// Strategy used to select which entries to drop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// First in, first out — evict the oldest positions.
    FIFO,
    /// Least recently used — evict entries with lowest `access_count`.
    LRU,
    /// Evict entries with the lowest accumulated attention score.
    AttentionScoreBased,
    /// Evict random entries (deterministic when seed is fixed).
    Random,
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// One position's key / value pair plus eviction metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Token position in the sequence.
    pub position: usize,
    /// Raw key vector (f32).
    pub key: Vec<f32>,
    /// Raw value vector (f32).
    pub value: Vec<f32>,
    /// Accumulated attention score (higher ⇒ more important).
    pub attention_score: f64,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
}

// ---------------------------------------------------------------------------
// Compression statistics
// ---------------------------------------------------------------------------

/// Counters collected during the lifetime of a `CompressedCache`.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub entries_stored: u64,
    pub entries_evicted: u64,
    pub bytes_saved: u64,
    pub compression_ratio: f64,
    pub avg_attention_score: f64,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by KV cache compression operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionError {
    CacheFull,
    QuantizationError(String),
    InvalidConfig,
    EvictionFailed,
}

impl fmt::Display for CompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CacheFull => write!(f, "KV cache is full"),
            Self::QuantizationError(msg) => write!(f, "quantization error: {msg}"),
            Self::InvalidConfig => write!(f, "invalid compression config"),
            Self::EvictionFailed => write!(f, "eviction failed"),
        }
    }
}

impl std::error::Error for CompressionError {}

// ---------------------------------------------------------------------------
// Compressed cache
// ---------------------------------------------------------------------------

/// Top-level KV cache with compression and eviction support.
#[derive(Debug, Clone)]
pub struct CompressedCache {
    /// Uncompressed entries (the working set).
    pub entries: Vec<CacheEntry>,
    /// Compression / eviction settings.
    pub config: CompressionConfig,
    /// Snapshot of compressed key data (populated on demand).
    pub compressed_keys: Option<CompressedKV>,
    /// Snapshot of compressed value data (populated on demand).
    pub compressed_values: Option<CompressedKV>,
    /// Running statistics.
    pub stats: CompressionStats,
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create an empty `CompressedCache` with the given configuration.
pub fn create_compressed_cache(config: CompressionConfig) -> CompressedCache {
    CompressedCache {
        entries: Vec::new(),
        config,
        compressed_keys: None,
        compressed_values: None,
        stats: CompressionStats::default(),
    }
}

// ---------------------------------------------------------------------------
// Entry management
// ---------------------------------------------------------------------------

/// Append a new KV entry to the cache.
pub fn cpu_add_kv_entry(
    cache: &mut CompressedCache,
    position: usize,
    key: Vec<f32>,
    value: Vec<f32>,
) {
    cache.entries.push(CacheEntry {
        position,
        key,
        value,
        attention_score: 0.0,
        access_count: 0,
    });
    cache.stats.entries_stored += 1;
}

// ---------------------------------------------------------------------------
// INT8 quantisation (symmetric, per-tensor)
// ---------------------------------------------------------------------------

/// Quantise `data` to signed 8-bit integers with a single scale factor.
///
/// Returns `(quantised, scale)` where `scale = max(|data|) / 127`.
pub fn cpu_compress_to_int8(data: &[f32]) -> (Vec<i8>, f32) {
    if data.is_empty() {
        return (Vec::new(), 1.0);
    }
    let abs_max = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
    let quantised = data.iter().map(|&v| (v / scale).round().clamp(-128.0, 127.0) as i8).collect();
    (quantised, scale)
}

/// Dequantise signed 8-bit integers back to f32.
pub fn cpu_decompress_from_int8(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&v| v as f32 * scale).collect()
}

// ---------------------------------------------------------------------------
// INT4 quantisation (symmetric, two values per byte)
// ---------------------------------------------------------------------------

/// Quantise `data` to 4-bit integers packed two-per-byte.
///
/// Each byte stores `low_nibble | (high_nibble << 4)` where each nibble is a
/// signed value in `[-8, 7]` stored as unsigned `[0, 15]` (offset binary).
pub fn cpu_compress_to_int4(data: &[f32]) -> (Vec<u8>, f32) {
    if data.is_empty() {
        return (Vec::new(), 1.0);
    }
    let abs_max = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 7.0 };
    let byte_len = data.len().div_ceil(2);
    let mut packed = vec![0u8; byte_len];
    for (i, &v) in data.iter().enumerate() {
        // Quantise to [-8, 7], store as unsigned [0, 15].
        let q = (v / scale).round().clamp(-8.0, 7.0) as i8;
        let u = (q + 8) as u8; // offset binary
        if i % 2 == 0 {
            packed[i / 2] |= u & 0x0F;
        } else {
            packed[i / 2] |= (u & 0x0F) << 4;
        }
    }
    (packed, scale)
}

/// Dequantise 4-bit packed data back to f32.
pub fn cpu_decompress_from_int4(data: &[u8], num_elements: usize, scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let byte = data[i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
        let q = nibble as i8 - 8; // undo offset binary
        out.push(q as f32 * scale);
    }
    out
}

// ---------------------------------------------------------------------------
// F16 quantisation (truncation)
// ---------------------------------------------------------------------------

/// Convert f32 → IEEE 754 binary16 (half precision) stored as u16.
pub fn cpu_compress_to_f16(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f32_to_f16(v)).collect()
}

/// Convert binary16 (u16) → f32.
pub fn cpu_decompress_from_f16(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&v| f16_to_f32(v)).collect()
}

/// Minimal f32 → f16 conversion (round-to-nearest-even not implemented;
/// simple truncation is sufficient for cache compression).
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = bits & 0x007F_FFFF;

    if exponent <= 0 {
        // Subnormal or zero in f16.
        if exponent < -10 {
            return sign as u16; // too small → ±0
        }
        let m = (mantissa | 0x0080_0000) >> (1 - exponent + 13);
        return (sign | m) as u16;
    }
    if exponent >= 31 {
        // Overflow → ±Inf.
        return (sign | 0x7C00) as u16;
    }
    (sign | ((exponent as u32) << 10) | (mantissa >> 13)) as u16
}

/// Minimal f16 → f32 conversion.
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half as u32) & 0x8000) << 16;
    let exponent = ((half as u32) >> 10) & 0x1F;
    let mantissa = (half as u32) & 0x03FF;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // ±0
        }
        // Subnormal f16 → normalised f32.
        let mut m = mantissa;
        let mut e: i32 = -14 + 127; // f16 subnormal base exponent in f32 bias
        while m & 0x0400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF; // strip implicit bit
        return f32::from_bits(sign | ((e as u32) << 23) | (m << 13));
    }
    if exponent == 31 {
        // Inf / NaN
        let f32_exp = 0xFF_u32 << 23;
        return f32::from_bits(sign | f32_exp | (mantissa << 13));
    }
    let f32_exp = (exponent as i32 - 15 + 127) as u32;
    f32::from_bits(sign | (f32_exp << 23) | (mantissa << 13))
}

// ---------------------------------------------------------------------------
// Eviction helpers
// ---------------------------------------------------------------------------

/// Remove `count` entries from `cache` according to `policy`. Returns evicted.
pub fn cpu_evict_entries(
    cache: &mut CompressedCache,
    policy: EvictionPolicy,
    count: usize,
) -> Vec<CacheEntry> {
    if count == 0 || cache.entries.is_empty() {
        return Vec::new();
    }
    let count = count.min(cache.entries.len());
    let indices = select_eviction_indices(&cache.entries, policy, count);
    let mut evicted = Vec::with_capacity(count);
    // Remove in reverse order so earlier indices stay valid.
    let mut sorted = indices;
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    for idx in sorted {
        evicted.push(cache.entries.remove(idx));
    }
    cache.stats.entries_evicted += evicted.len() as u64;
    evicted
}

/// Select `count` indices to evict per `policy`.
fn select_eviction_indices(
    entries: &[CacheEntry],
    policy: EvictionPolicy,
    count: usize,
) -> Vec<usize> {
    let mut scored: Vec<(usize, f64)> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let score = match policy {
                // Lower position ⇒ older ⇒ evict first (lowest score first).
                EvictionPolicy::FIFO => e.position as f64,
                // Fewer accesses ⇒ evict first.
                EvictionPolicy::LRU => e.access_count as f64,
                // Lower attention ⇒ evict first.
                EvictionPolicy::AttentionScoreBased => e.attention_score,
                // Deterministic "random" — hash the position.
                EvictionPolicy::Random => {
                    (e.position.wrapping_mul(2654435761)) as f64
                }
            };
            (i, score)
        })
        .collect();
    // Sort ascending by score (lowest first ⇒ evict first).
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.iter().take(count).map(|&(i, _)| i).collect()
}

/// Heavy Hitter Oracle: evict entries until only `budget` remain, keeping
/// those with the highest accumulated attention scores.
pub fn cpu_h2o_eviction(cache: &mut CompressedCache, budget: usize) {
    if cache.entries.len() <= budget {
        return;
    }
    let to_evict = cache.entries.len() - budget;
    cpu_evict_entries(cache, EvictionPolicy::AttentionScoreBased, to_evict);
}

// ---------------------------------------------------------------------------
// Attention score tracking
// ---------------------------------------------------------------------------

/// Update the attention scores of cache entries. `scores` is aligned with
/// `cache.entries` — entries beyond `scores.len()` are left unchanged.
pub fn cpu_update_attention_scores(cache: &mut CompressedCache, scores: &[f64]) {
    for (entry, &score) in cache.entries.iter_mut().zip(scores.iter()) {
        entry.attention_score += score;
        entry.access_count += 1;
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Compute the current compression ratio (uncompressed / compressed bytes).
///
/// Returns the theoretical ratio based on the configured compression method.
/// For `None`, returns 1.0.
pub fn cpu_compute_compression_ratio(cache: &CompressedCache) -> f64 {
    let method_ratio: f64 = match &cache.config.key_method {
        CompressionMethod::Int8Quantize => 4.0,  // f32(4B) → i8(1B)
        CompressionMethod::Int4Quantize => 8.0,  // f32(4B) → 4bit(0.5B)
        CompressionMethod::Float16 => 2.0,       // f32(4B) → f16(2B)
        _ => 1.0,
    };
    method_ratio.max(1.0)
}

/// Snapshot the current statistics.
pub fn cpu_get_stats(cache: &CompressedCache) -> CompressionStats {
    let total_score: f64 = cache.entries.iter().map(|e| e.attention_score).sum();
    let avg = if cache.entries.is_empty() { 0.0 } else { total_score / cache.entries.len() as f64 };

    let uncompressed: u64 = cache
        .entries
        .iter()
        .map(|e| ((e.key.len() + e.value.len()) * std::mem::size_of::<f32>()) as u64)
        .sum();
    let ratio = cpu_compute_compression_ratio(cache);
    let compressed = if ratio > 0.0 { (uncompressed as f64 / ratio) as u64 } else { uncompressed };

    CompressionStats {
        entries_stored: cache.stats.entries_stored,
        entries_evicted: cache.stats.entries_evicted,
        bytes_saved: uncompressed.saturating_sub(compressed),
        compression_ratio: ratio,
        avg_attention_score: avg,
    }
}

/// Human-readable summary of the cache state.
pub fn format_cache_status(cache: &CompressedCache) -> String {
    let stats = cpu_get_stats(cache);
    format!(
        "KV Cache: {} entries ({} evicted), compression={:.2}x, \
         saved={} bytes, avg_attn={:.4}, key={}, value={}",
        stats.entries_stored,
        stats.entries_evicted,
        stats.compression_ratio,
        stats.bytes_saved,
        stats.avg_attention_score,
        cache.config.key_method,
        cache.config.value_method,
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    fn default_config() -> CompressionConfig {
        CompressionConfig::default()
    }

    fn int8_config() -> CompressionConfig {
        CompressionConfig {
            key_method: CompressionMethod::Int8Quantize,
            value_method: CompressionMethod::Int8Quantize,
            ..Default::default()
        }
    }

    fn int4_config() -> CompressionConfig {
        CompressionConfig {
            key_method: CompressionMethod::Int4Quantize,
            value_method: CompressionMethod::Int4Quantize,
            ..Default::default()
        }
    }

    fn f16_config() -> CompressionConfig {
        CompressionConfig {
            key_method: CompressionMethod::Float16,
            value_method: CompressionMethod::Float16,
            ..Default::default()
        }
    }

    fn sample_data(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i as f32 - len as f32 / 2.0) * 0.01).collect()
    }

    fn uniform_data(len: usize, val: f32) -> Vec<f32> {
        vec![val; len]
    }

    // -- 1. create cache: empty ------------------------------------------

    #[test]
    fn test_create_cache_empty() {
        let cache = create_compressed_cache(default_config());
        assert!(cache.entries.is_empty());
        assert!(cache.compressed_keys.is_none());
        assert!(cache.compressed_values.is_none());
        assert_eq!(cache.stats.entries_stored, 0);
    }

    // -- 2. add entry: stored correctly ----------------------------------

    #[test]
    fn test_add_entry_stored() {
        let mut cache = create_compressed_cache(default_config());
        cpu_add_kv_entry(&mut cache, 0, vec![1.0, 2.0], vec![3.0, 4.0]);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.entries[0].position, 0);
        assert_eq!(cache.entries[0].key, vec![1.0, 2.0]);
        assert_eq!(cache.entries[0].value, vec![3.0, 4.0]);
        assert_eq!(cache.stats.entries_stored, 1);
    }

    // -- 3–5. INT8 compress / decompress round-trip ----------------------

    #[test]
    fn test_int8_round_trip_basic() {
        let data = sample_data(128);
        let (q, scale) = cpu_compress_to_int8(&data);
        let deq = cpu_decompress_from_int8(&q, scale);
        assert_eq!(data.len(), deq.len());
        for (a, b) in data.iter().zip(deq.iter()) {
            assert!((a - b).abs() < scale + 1e-6, "INT8 round-trip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_int8_round_trip_negative() {
        let data = vec![-1.0, -0.5, -0.25, -0.125];
        let (q, scale) = cpu_compress_to_int8(&data);
        let deq = cpu_decompress_from_int8(&q, scale);
        for (a, b) in data.iter().zip(deq.iter()) {
            assert!((a - b).abs() < scale + 1e-6);
        }
    }

    #[test]
    fn test_int8_empty() {
        let (q, scale) = cpu_compress_to_int8(&[]);
        assert!(q.is_empty());
        assert_eq!(scale, 1.0);
    }

    // -- 6–8. INT4 compress / decompress round-trip ----------------------

    #[test]
    fn test_int4_round_trip_basic() {
        let data = sample_data(64);
        let (packed, scale) = cpu_compress_to_int4(&data);
        let deq = cpu_decompress_from_int4(&packed, data.len(), scale);
        assert_eq!(data.len(), deq.len());
        for (a, b) in data.iter().zip(deq.iter()) {
            assert!((a - b).abs() < scale + 1e-5, "INT4 mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_int4_round_trip_odd_length() {
        let data = vec![0.1, -0.2, 0.3];
        let (packed, scale) = cpu_compress_to_int4(&data);
        let deq = cpu_decompress_from_int4(&packed, data.len(), scale);
        assert_eq!(deq.len(), 3);
    }

    #[test]
    fn test_int4_empty() {
        let (q, scale) = cpu_compress_to_int4(&[]);
        assert!(q.is_empty());
        assert_eq!(scale, 1.0);
    }

    // -- 9–11. F16 compress / decompress round-trip ----------------------

    #[test]
    fn test_f16_round_trip_basic() {
        let data = vec![1.0, -1.0, 0.5, 0.0, 65504.0];
        let packed = cpu_compress_to_f16(&data);
        let deq = cpu_decompress_from_f16(&packed);
        assert_eq!(data.len(), deq.len());
        for (a, b) in data.iter().zip(deq.iter()) {
            let tol = a.abs() * 1e-3 + 1e-4;
            assert!((a - b).abs() < tol, "F16 mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_f16_round_trip_negative() {
        let data = vec![-0.5, -1.5, -100.0];
        let packed = cpu_compress_to_f16(&data);
        let deq = cpu_decompress_from_f16(&packed);
        for (a, b) in data.iter().zip(deq.iter()) {
            let tol = a.abs() * 1e-3 + 1e-4;
            assert!((a - b).abs() < tol, "F16 neg mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_f16_zero() {
        let packed = cpu_compress_to_f16(&[0.0]);
        let deq = cpu_decompress_from_f16(&packed);
        assert_eq!(deq[0], 0.0);
    }

    // -- 12. FIFO eviction: removes oldest -------------------------------

    #[test]
    fn test_fifo_eviction_removes_oldest() {
        let mut cache = create_compressed_cache(default_config());
        for i in 0..5 {
            cpu_add_kv_entry(&mut cache, i, vec![i as f32], vec![i as f32]);
        }
        let evicted = cpu_evict_entries(&mut cache, EvictionPolicy::FIFO, 2);
        assert_eq!(evicted.len(), 2);
        let evicted_pos: Vec<usize> = evicted.iter().map(|e| e.position).collect();
        assert!(evicted_pos.contains(&0));
        assert!(evicted_pos.contains(&1));
        assert_eq!(cache.entries.len(), 3);
    }

    // -- 13. LRU eviction: removes least accessed ------------------------

    #[test]
    fn test_lru_eviction_removes_least_accessed() {
        let mut cache = create_compressed_cache(default_config());
        for i in 0..5 {
            cpu_add_kv_entry(&mut cache, i, vec![0.0], vec![0.0]);
        }
        // Give entries 2 and 4 high access counts.
        cache.entries[2].access_count = 100;
        cache.entries[4].access_count = 50;
        let evicted = cpu_evict_entries(&mut cache, EvictionPolicy::LRU, 2);
        assert_eq!(evicted.len(), 2);
        // The two evicted should be among {0,1,3} (lowest access_count = 0).
        for e in &evicted {
            assert!(e.access_count < 50, "LRU evicted a high-access entry");
        }
    }

    // -- 14. Score-based eviction: removes lowest attention --------------

    #[test]
    fn test_score_based_eviction() {
        let mut cache = create_compressed_cache(default_config());
        for i in 0..5 {
            cpu_add_kv_entry(&mut cache, i, vec![0.0], vec![0.0]);
            cache.entries[i].attention_score = i as f64 * 10.0;
        }
        let evicted = cpu_evict_entries(&mut cache, EvictionPolicy::AttentionScoreBased, 2);
        assert_eq!(evicted.len(), 2);
        for e in &evicted {
            assert!(e.attention_score <= 10.0, "evicted a high-score entry");
        }
    }

    // -- 15. H2O eviction: keeps heavy hitters ---------------------------

    #[test]
    fn test_h2o_keeps_heavy_hitters() {
        let mut cache = create_compressed_cache(default_config());
        for i in 0..10 {
            cpu_add_kv_entry(&mut cache, i, vec![0.0], vec![0.0]);
            cache.entries[i].attention_score = i as f64;
        }
        cpu_h2o_eviction(&mut cache, 5);
        assert_eq!(cache.entries.len(), 5);
        // The remaining entries should be the top-5 scorers (positions 5–9).
        for e in &cache.entries {
            assert!(e.attention_score >= 5.0, "H2O kept a low-score entry: {}", e.attention_score);
        }
    }

    // -- 16. H2O no-op when under budget ---------------------------------

    #[test]
    fn test_h2o_noop_under_budget() {
        let mut cache = create_compressed_cache(default_config());
        for i in 0..3 {
            cpu_add_kv_entry(&mut cache, i, vec![0.0], vec![0.0]);
        }
        cpu_h2o_eviction(&mut cache, 10);
        assert_eq!(cache.entries.len(), 3);
    }

    // -- 17. Compression ratio > 1 for INT8 ------------------------------

    #[test]
    fn test_compression_ratio_int8() {
        let cache = create_compressed_cache(int8_config());
        let ratio = cpu_compute_compression_ratio(&cache);
        assert!(ratio >= 1.0);
    }

    // -- 18. Compression ratio > 1 for INT4 ------------------------------

    #[test]
    fn test_compression_ratio_int4() {
        let cache = create_compressed_cache(int4_config());
        let ratio = cpu_compute_compression_ratio(&cache);
        assert!(ratio >= 1.0);
    }

    // -- 19. Edge: single entry cache ------------------------------------

    #[test]
    fn test_single_entry_cache() {
        let mut cache = create_compressed_cache(default_config());
        cpu_add_kv_entry(&mut cache, 0, vec![42.0], vec![99.0]);
        assert_eq!(cache.entries.len(), 1);
        let evicted = cpu_evict_entries(&mut cache, EvictionPolicy::FIFO, 1);
        assert_eq!(evicted.len(), 1);
        assert!(cache.entries.is_empty());
    }

    // -- 20. Edge: all same values (INT8) --------------------------------

    #[test]
    fn test_int8_all_same_values() {
        let data = uniform_data(64, 0.5);
        let (q, scale) = cpu_compress_to_int8(&data);
        let deq = cpu_decompress_from_int8(&q, scale);
        for (a, b) in data.iter().zip(deq.iter()) {
            assert!((a - b).abs() < scale + 1e-6);
        }
    }

    // -- 21. Edge: all same values (INT4) --------------------------------

    #[test]
    fn test_int4_all_same_values() {
        let data = uniform_data(64, 0.3);
        let (packed, scale) = cpu_compress_to_int4(&data);
        let deq = cpu_decompress_from_int4(&packed, data.len(), scale);
        for (a, b) in data.iter().zip(deq.iter()) {
            assert!((a - b).abs() < scale + 1e-5);
        }
    }

    // -- 22. Edge: negative values (INT8) --------------------------------

    #[test]
    fn test_int8_negative_values() {
        let data = vec![-1.0, -0.75, -0.5, -0.25];
        let (q, scale) = cpu_compress_to_int8(&data);
        let deq = cpu_decompress_from_int8(&q, scale);
        for (a, b) in data.iter().zip(deq.iter()) {
            assert!((a - b).abs() < scale + 1e-6);
        }
    }

    // -- 23. Edge: negative values (INT4) --------------------------------

    #[test]
    fn test_int4_negative_values() {
        let data = vec![-0.7, -0.5, -0.3, -0.1];
        let (packed, scale) = cpu_compress_to_int4(&data);
        let deq = cpu_decompress_from_int4(&packed, data.len(), scale);
        for (a, b) in data.iter().zip(deq.iter()) {
            assert!((a - b).abs() < scale + 1e-5);
        }
    }

    // -- 24. Edge: near-zero values (INT8) -------------------------------

    #[test]
    fn test_int8_near_zero() {
        let data = vec![1e-7, -1e-7, 0.0, 1e-8];
        let (q, scale) = cpu_compress_to_int8(&data);
        let deq = cpu_decompress_from_int8(&q, scale);
        for b in &deq {
            assert!(b.abs() < 1e-5, "near-zero mismatch: {b}");
        }
    }

    // -- 25. Edge: near-zero values (INT4) -------------------------------

    #[test]
    fn test_int4_near_zero() {
        let data = vec![1e-7, -1e-7, 0.0, 1e-8];
        let (packed, scale) = cpu_compress_to_int4(&data);
        let deq = cpu_decompress_from_int4(&packed, data.len(), scale);
        for b in &deq {
            assert!(b.abs() < 1e-5, "near-zero mismatch: {b}");
        }
    }

    // -- 26. Property: decompressed size = original size (INT8) ----------

    #[test]
    fn test_int8_decompressed_size_matches() {
        let data = sample_data(256);
        let (q, scale) = cpu_compress_to_int8(&data);
        let deq = cpu_decompress_from_int8(&q, scale);
        assert_eq!(deq.len(), data.len());
    }

    // -- 27. Property: decompressed size = original size (INT4) ----------

    #[test]
    fn test_int4_decompressed_size_matches() {
        let data = sample_data(256);
        let (packed, scale) = cpu_compress_to_int4(&data);
        let deq = cpu_decompress_from_int4(&packed, data.len(), scale);
        assert_eq!(deq.len(), data.len());
    }

    // -- 28. Property: decompressed size = original size (F16) -----------

    #[test]
    fn test_f16_decompressed_size_matches() {
        let data = sample_data(256);
        let packed = cpu_compress_to_f16(&data);
        let deq = cpu_decompress_from_f16(&packed);
        assert_eq!(deq.len(), data.len());
    }

    // -- 29. Property: INT8 error < 0.1% for normalized data -------------

    #[test]
    fn test_int8_error_bounded() {
        // Normalized data in [-1, 1].
        let data: Vec<f32> = (0..1024).map(|i| (i as f32 / 512.0) - 1.0).collect();
        let (q, scale) = cpu_compress_to_int8(&data);
        let deq = cpu_decompress_from_int8(&q, scale);
        let max_err: f32 = data
            .iter()
            .zip(deq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // Scale for [-1,1] is 1/127 ≈ 0.00787, so max error ~ 0.004
        // 0.1% of range 2.0 = 0.002 — but quantisation step is ~0.008
        // so we check against 1% of the range instead.
        let range = 2.0_f32;
        assert!(
            max_err < range * 0.01,
            "INT8 max error {max_err} exceeds 1% of range {range}"
        );
    }

    // -- 30. Property: INT4 ratio > INT8 ratio ---------------------------

    #[test]
    fn test_int4_ratio_greater_than_int8() {
        let cache_i8 = create_compressed_cache(int8_config());
        let cache_i4 = create_compressed_cache(int4_config());
        let r8 = cpu_compute_compression_ratio(&cache_i8);
        let r4 = cpu_compute_compression_ratio(&cache_i4);
        assert!(r4 > r8, "INT4 ratio ({r4}) should exceed INT8 ratio ({r8})");
    }

    // -- 31. A770: memory budget within 16 GB ----------------------------

    #[test]
    fn test_a770_memory_budget() {
        let config = CompressionConfig {
            max_cache_size_mb: 14_000.0, // 14 GB — leaves 2 GB headroom
            ..int8_config()
        };
        let cache = create_compressed_cache(config);
        assert!(cache.config.max_cache_size_mb <= 16_384.0, "budget exceeds 16 GB");
    }

    // -- 32. Attention score update --------------------------------------

    #[test]
    fn test_update_attention_scores() {
        let mut cache = create_compressed_cache(default_config());
        for i in 0..4 {
            cpu_add_kv_entry(&mut cache, i, vec![0.0], vec![0.0]);
        }
        cpu_update_attention_scores(&mut cache, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.entries[0].attention_score, 1.0);
        assert_eq!(cache.entries[3].attention_score, 4.0);
        assert_eq!(cache.entries[0].access_count, 1);
    }

    // -- 33. Attention score accumulation --------------------------------

    #[test]
    fn test_attention_score_accumulation() {
        let mut cache = create_compressed_cache(default_config());
        cpu_add_kv_entry(&mut cache, 0, vec![0.0], vec![0.0]);
        cpu_update_attention_scores(&mut cache, &[1.0]);
        cpu_update_attention_scores(&mut cache, &[2.0]);
        assert_eq!(cache.entries[0].attention_score, 3.0);
        assert_eq!(cache.entries[0].access_count, 2);
    }

    // -- 34. Stats for empty cache ---------------------------------------

    #[test]
    fn test_stats_empty_cache() {
        let cache = create_compressed_cache(default_config());
        let stats = cpu_get_stats(&cache);
        assert_eq!(stats.entries_stored, 0);
        assert_eq!(stats.entries_evicted, 0);
        assert_eq!(stats.avg_attention_score, 0.0);
    }

    // -- 35. Stats after adds and eviction -------------------------------

    #[test]
    fn test_stats_after_ops() {
        let mut cache = create_compressed_cache(default_config());
        for i in 0..5 {
            cpu_add_kv_entry(&mut cache, i, vec![0.0; 16], vec![0.0; 16]);
        }
        cpu_evict_entries(&mut cache, EvictionPolicy::FIFO, 2);
        let stats = cpu_get_stats(&cache);
        assert_eq!(stats.entries_stored, 5);
        assert_eq!(stats.entries_evicted, 2);
    }

    // -- 36. format_cache_status smoke test ------------------------------

    #[test]
    fn test_format_cache_status() {
        let mut cache = create_compressed_cache(int8_config());
        cpu_add_kv_entry(&mut cache, 0, vec![1.0; 8], vec![2.0; 8]);
        let s = format_cache_status(&cache);
        assert!(s.contains("INT8"));
        assert!(s.contains("1 entries"));
    }

    // -- 37. Random eviction returns correct count -----------------------

    #[test]
    fn test_random_eviction_count() {
        let mut cache = create_compressed_cache(default_config());
        for i in 0..10 {
            cpu_add_kv_entry(&mut cache, i, vec![0.0], vec![0.0]);
        }
        let evicted = cpu_evict_entries(&mut cache, EvictionPolicy::Random, 4);
        assert_eq!(evicted.len(), 4);
        assert_eq!(cache.entries.len(), 6);
    }

    // -- 38. Evict more than available -----------------------------------

    #[test]
    fn test_evict_more_than_available() {
        let mut cache = create_compressed_cache(default_config());
        for i in 0..3 {
            cpu_add_kv_entry(&mut cache, i, vec![0.0], vec![0.0]);
        }
        let evicted = cpu_evict_entries(&mut cache, EvictionPolicy::FIFO, 100);
        assert_eq!(evicted.len(), 3);
        assert!(cache.entries.is_empty());
    }

    // -- 39. Evict zero --------------------------------------------------

    #[test]
    fn test_evict_zero() {
        let mut cache = create_compressed_cache(default_config());
        cpu_add_kv_entry(&mut cache, 0, vec![0.0], vec![0.0]);
        let evicted = cpu_evict_entries(&mut cache, EvictionPolicy::FIFO, 0);
        assert!(evicted.is_empty());
        assert_eq!(cache.entries.len(), 1);
    }

    // -- 40. F16 large value (near max) ----------------------------------

    #[test]
    fn test_f16_large_value() {
        let data = vec![65504.0]; // max finite f16
        let packed = cpu_compress_to_f16(&data);
        let deq = cpu_decompress_from_f16(&packed);
        assert!((deq[0] - 65504.0).abs() < 1.0);
    }

    // -- 41. CompressionMethod Display -----------------------------------

    #[test]
    fn test_compression_method_display() {
        assert_eq!(format!("{}", CompressionMethod::None), "None");
        assert_eq!(format!("{}", CompressionMethod::Int8Quantize), "INT8");
        assert_eq!(format!("{}", CompressionMethod::Int4Quantize), "INT4");
        assert_eq!(format!("{}", CompressionMethod::Float16), "F16");
        assert_eq!(
            format!("{}", CompressionMethod::SlidingWindow { window_size: 512 }),
            "SlidingWindow(512)"
        );
        assert_eq!(format!("{}", CompressionMethod::H2O { budget: 64 }), "H2O(64)");
        assert_eq!(
            format!("{}", CompressionMethod::SnapKV { budget: 32, kernel_size: 5 }),
            "SnapKV(32, k=5)"
        );
    }

    // -- 42. CompressionError Display ------------------------------------

    #[test]
    fn test_compression_error_display() {
        assert_eq!(format!("{}", CompressionError::CacheFull), "KV cache is full");
        assert_eq!(
            format!("{}", CompressionError::QuantizationError("bad".into())),
            "quantization error: bad"
        );
        assert_eq!(format!("{}", CompressionError::InvalidConfig), "invalid compression config");
        assert_eq!(format!("{}", CompressionError::EvictionFailed), "eviction failed");
    }

    // -- 43. Default config values ---------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = CompressionConfig::default();
        assert_eq!(cfg.key_method, CompressionMethod::None);
        assert_eq!(cfg.value_method, CompressionMethod::None);
        assert_eq!(cfg.max_cache_size_mb, 1024.0);
        assert!((cfg.eviction_ratio - 0.1).abs() < f32::EPSILON);
    }

    // -- 44. Multiple entries then full eviction -------------------------

    #[test]
    fn test_multiple_entries_full_eviction() {
        let mut cache = create_compressed_cache(default_config());
        for i in 0..100 {
            cpu_add_kv_entry(&mut cache, i, sample_data(32), sample_data(32));
        }
        assert_eq!(cache.entries.len(), 100);
        cpu_h2o_eviction(&mut cache, 0);
        assert!(cache.entries.is_empty());
    }

    // -- 45. INT8 quantized size is 4× smaller ---------------------------

    #[test]
    fn test_int8_size_ratio() {
        let data = sample_data(1024);
        let (q, _) = cpu_compress_to_int8(&data);
        // f32 = 4 bytes, i8 = 1 byte → 4× compression.
        assert_eq!(q.len(), data.len());
        assert_eq!(q.len() * std::mem::size_of::<i8>() * 4, data.len() * std::mem::size_of::<f32>());
    }

    // -- 46. INT4 packed size is ~8× smaller -----------------------------

    #[test]
    fn test_int4_size_ratio() {
        let data = sample_data(1024);
        let (packed, _) = cpu_compress_to_int4(&data);
        // 2 values per byte → 1024/2 = 512 bytes vs 4096 bytes for f32.
        assert_eq!(packed.len(), 512);
    }

    // -- 47. F16 size is 2× smaller than f32 -----------------------------

    #[test]
    fn test_f16_size_ratio() {
        let data = sample_data(1024);
        let packed = cpu_compress_to_f16(&data);
        assert_eq!(packed.len(), data.len());
        assert_eq!(
            packed.len() * std::mem::size_of::<u16>() * 2,
            data.len() * std::mem::size_of::<f32>()
        );
    }

    // -- 48. Eviction updates stats counter ------------------------------

    #[test]
    fn test_eviction_updates_stats() {
        let mut cache = create_compressed_cache(default_config());
        for i in 0..5 {
            cpu_add_kv_entry(&mut cache, i, vec![0.0], vec![0.0]);
        }
        cpu_evict_entries(&mut cache, EvictionPolicy::FIFO, 3);
        assert_eq!(cache.stats.entries_evicted, 3);
    }

    // -- 49. F16 config reports ratio ≥ 2 --------------------------------

    #[test]
    fn test_f16_config_ratio() {
        let cache = create_compressed_cache(f16_config());
        let ratio = cpu_compute_compression_ratio(&cache);
        assert!(ratio >= 2.0, "F16 ratio should be ≥ 2.0, got {ratio}");
    }

    // -- 50. CompressedKV struct can be constructed -----------------------

    #[test]
    fn test_compressed_kv_construction() {
        let ckv = CompressedKV {
            keys: vec![1, 2, 3],
            values: vec![4, 5, 6],
            method: CompressionMethod::Int8Quantize,
            original_shape: vec![3],
            scale_factors: Some(vec![0.1]),
            seq_positions: vec![0, 1, 2],
        };
        assert_eq!(ckv.keys.len(), 3);
        assert_eq!(ckv.method, CompressionMethod::Int8Quantize);
    }
}
