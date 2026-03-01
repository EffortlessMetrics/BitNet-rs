//! OpenCL KV (Key-Value) cache management for A770 GPU inference.
//!
//! # Overview
//!
//! The KV cache stores computed key and value tensors from past tokens during
//! autoregressive LLM inference, avoiding redundant recomputation. This module
//! provides:
//!
//! - **`KvCacheConfig`** — configuration describing cache geometry (heads, dims,
//!   layers, sequence length).
//! - **`KvCacheEntry`** — single-layer cache with append / read / clear ops.
//! - **`KvCache`** — multi-layer cache manager.
//! - **`KvCacheError`** — typed errors for bounds, capacity, and shape
//!   mismatches.
//! - **OpenCL kernel source** — embedded CL C kernels for future GPU offload
//!   on Intel Arc A770 and other OpenCL 3.0 devices.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by KV cache operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheError {
    /// Requested layer index exceeds available layers.
    LayerOutOfBounds {
        requested: usize,
        available: usize,
    },
    /// Cache has reached its maximum sequence length.
    CacheFull {
        max_len: usize,
    },
    /// Input slice length does not match expected row size.
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayerOutOfBounds { requested, available } => {
                write!(
                    f,
                    "layer index {requested} out of bounds (available: {available})"
                )
            }
            Self::CacheFull { max_len } => {
                write!(f, "KV cache is full (max_len={max_len})")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "dimension mismatch: expected {expected} elements, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for KvCacheError {}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, KvCacheError>;

// ---------------------------------------------------------------------------
// KvCacheConfig
// ---------------------------------------------------------------------------

/// Describes the geometry of a KV cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// Maximum sequence length (token positions).
    pub max_seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Bytes per element (4 for f32, 2 for f16).
    pub dtype_bytes: usize,
}

impl KvCacheConfig {
    /// Number of bytes required for one key or value cache per layer.
    ///
    /// Layout: `[max_seq_len, num_heads, head_dim]`.
    #[inline]
    pub fn memory_per_layer(&self) -> usize {
        // key + value → ×2
        2 * self.max_seq_len * self.num_heads * self.head_dim * self.dtype_bytes
    }

    /// Total bytes for all layers.
    #[inline]
    pub fn total_memory(&self) -> usize {
        self.memory_per_layer() * self.num_layers
    }

    /// Number of f32 elements in a single row (one token, all heads).
    #[inline]
    fn row_len(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

// ---------------------------------------------------------------------------
// KvCacheEntry — single-layer cache
// ---------------------------------------------------------------------------

/// KV cache for a single transformer layer.
///
/// Key and value tensors are stored in flattened row-major order with shape
/// `[seq_len, num_heads, head_dim]`.
#[derive(Debug, Clone)]
pub struct KvCacheEntry {
    /// Flattened key cache.
    pub key_cache: Vec<f32>,
    /// Flattened value cache.
    pub value_cache: Vec<f32>,
    /// Number of token positions currently stored.
    pub current_len: usize,
    /// Maximum token positions this entry can hold.
    pub max_len: usize,
    /// Elements per row (`num_heads * head_dim`).
    row_len: usize,
}

impl KvCacheEntry {
    /// Create a new zeroed entry.
    pub fn new(max_len: usize, row_len: usize) -> Self {
        let capacity = max_len * row_len;
        Self {
            key_cache: vec![0.0; capacity],
            value_cache: vec![0.0; capacity],
            current_len: 0,
            max_len,
            row_len,
        }
    }

    /// Append one token's key and value row to the cache.
    pub fn append(&mut self, key_row: &[f32], value_row: &[f32]) -> Result<()> {
        if self.current_len >= self.max_len {
            return Err(KvCacheError::CacheFull { max_len: self.max_len });
        }
        if key_row.len() != self.row_len {
            return Err(KvCacheError::DimensionMismatch {
                expected: self.row_len,
                got: key_row.len(),
            });
        }
        if value_row.len() != self.row_len {
            return Err(KvCacheError::DimensionMismatch {
                expected: self.row_len,
                got: value_row.len(),
            });
        }

        let offset = self.current_len * self.row_len;
        self.key_cache[offset..offset + self.row_len].copy_from_slice(key_row);
        self.value_cache[offset..offset + self.row_len]
            .copy_from_slice(value_row);
        self.current_len += 1;
        Ok(())
    }

    /// Return the cached keys for positions `[0, up_to)`.
    ///
    /// `up_to` is clamped to `current_len`.
    pub fn get_keys(&self, up_to: usize) -> &[f32] {
        let end = up_to.min(self.current_len) * self.row_len;
        &self.key_cache[..end]
    }

    /// Return the cached values for positions `[0, up_to)`.
    pub fn get_values(&self, up_to: usize) -> &[f32] {
        let end = up_to.min(self.current_len) * self.row_len;
        &self.value_cache[..end]
    }

    /// Reset the cache to empty (zero-length, data untouched for perf).
    pub fn clear(&mut self) {
        self.current_len = 0;
    }

    /// Returns `true` when no more tokens can be appended.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.current_len >= self.max_len
    }
}

// ---------------------------------------------------------------------------
// KvCache — multi-layer manager
// ---------------------------------------------------------------------------

/// Multi-layer KV cache manager.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Per-layer cache entries.
    pub layers: Vec<KvCacheEntry>,
    /// Geometry configuration.
    pub config: KvCacheConfig,
}

impl KvCache {
    /// Create a new multi-layer KV cache from a config.
    pub fn new(config: KvCacheConfig) -> Self {
        let row_len = config.row_len();
        let layers = (0..config.num_layers)
            .map(|_| KvCacheEntry::new(config.max_seq_len, row_len))
            .collect();
        Self { layers, config }
    }

    /// Append one token's key/value to a specific layer.
    pub fn append_layer(
        &mut self,
        layer_idx: usize,
        keys: &[f32],
        values: &[f32],
    ) -> Result<()> {
        let available = self.layers.len();
        let entry = self.layers.get_mut(layer_idx).ok_or(
            KvCacheError::LayerOutOfBounds { requested: layer_idx, available },
        )?;
        entry.append(keys, values)
    }

    /// Get an immutable reference to a layer's cache entry.
    pub fn get_layer(&self, layer_idx: usize) -> Result<&KvCacheEntry> {
        let available = self.layers.len();
        self.layers.get(layer_idx).ok_or(
            KvCacheError::LayerOutOfBounds { requested: layer_idx, available },
        )
    }

    /// Clear all layers.
    pub fn clear_all(&mut self) {
        for entry in &mut self.layers {
            entry.clear();
        }
    }

    /// Current sequence length (taken from layer 0; all layers stay in sync
    /// during normal autoregressive decoding).
    pub fn current_sequence_length(&self) -> usize {
        self.layers.first().map_or(0, |e| e.current_len)
    }

    /// Approximate memory usage in bytes (f32 storage only).
    pub fn memory_usage_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|e| {
                (e.key_cache.len() + e.value_cache.len())
                    * std::mem::size_of::<f32>()
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C kernel source for KV cache operations on Intel Arc A770 and
/// other OpenCL 3.0 devices.
///
/// Contains two kernels:
/// - `kv_cache_append` — writes a new K/V row at a given position.
/// - `kv_cache_gather` — gathers cached K/V rows for attention computation.
pub const KV_CACHE_CL: &str = r#"
// kv_cache_append: copy a new K/V row into cache at `pos`.
//
// Global work size: (row_len,)
// Arguments:
//   cache    – [max_seq_len * row_len] buffer (key or value)
//   new_row  – [row_len] input for the current token
//   pos      – token position (0-based)
//   row_len  – num_heads * head_dim
__kernel void kv_cache_append(
    __global float *cache,
    __global const float *new_row,
    const int pos,
    const int row_len)
{
    int gid = get_global_id(0);
    if (gid < row_len) {
        cache[pos * row_len + gid] = new_row[gid];
    }
}

// kv_cache_gather: gather cached rows [0, seq_len) into a contiguous output.
//
// Global work size: (seq_len * row_len,)
// Arguments:
//   cache   – source cache buffer
//   output  – destination contiguous buffer [seq_len, row_len]
//   seq_len – number of rows to gather
//   row_len – elements per row
__kernel void kv_cache_gather(
    __global const float *cache,
    __global float *output,
    const int seq_len,
    const int row_len)
{
    int gid = get_global_id(0);
    int total = seq_len * row_len;
    if (gid < total) {
        output[gid] = cache[gid];
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> KvCacheConfig {
        KvCacheConfig {
            max_seq_len: 2048,
            num_heads: 8,
            head_dim: 64,
            num_layers: 4,
            dtype_bytes: 4,
        }
    }

    // -- KvCacheConfig memory calculations ----------------------------------

    #[test]
    fn config_memory_per_layer() {
        let cfg = default_config();
        // 2 * 2048 * 8 * 64 * 4 = 8_388_608
        assert_eq!(cfg.memory_per_layer(), 8_388_608);
    }

    #[test]
    fn config_total_memory() {
        let cfg = default_config();
        assert_eq!(cfg.total_memory(), 8_388_608 * 4);
    }

    #[test]
    fn config_memory_f16() {
        let cfg = KvCacheConfig { dtype_bytes: 2, ..default_config() };
        assert_eq!(cfg.memory_per_layer(), 8_388_608 / 2);
    }

    #[test]
    fn config_row_len() {
        let cfg = default_config();
        assert_eq!(cfg.row_len(), 512);
    }

    #[test]
    fn config_minimal_dimensions() {
        let cfg = KvCacheConfig {
            max_seq_len: 1,
            num_heads: 1,
            head_dim: 1,
            num_layers: 1,
            dtype_bytes: 4,
        };
        assert_eq!(cfg.memory_per_layer(), 8); // 2 * 1 * 1 * 1 * 4
        assert_eq!(cfg.total_memory(), 8);
    }

    // -- KvCacheEntry basic operations --------------------------------------

    #[test]
    fn entry_new_is_empty() {
        let entry = KvCacheEntry::new(4, 2);
        assert_eq!(entry.current_len, 0);
        assert!(!entry.is_full());
    }

    #[test]
    fn entry_append_and_read() {
        let mut entry = KvCacheEntry::new(4, 2);
        entry.append(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_eq!(entry.current_len, 1);
        assert_eq!(entry.get_keys(1), &[1.0, 2.0]);
        assert_eq!(entry.get_values(1), &[3.0, 4.0]);
    }

    #[test]
    fn entry_append_multiple() {
        let mut entry = KvCacheEntry::new(4, 2);
        entry.append(&[1.0, 2.0], &[10.0, 20.0]).unwrap();
        entry.append(&[3.0, 4.0], &[30.0, 40.0]).unwrap();
        assert_eq!(entry.current_len, 2);
        assert_eq!(entry.get_keys(2), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(entry.get_values(2), &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn entry_get_keys_clamped() {
        let mut entry = KvCacheEntry::new(4, 2);
        entry.append(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        // Requesting more than stored should clamp.
        assert_eq!(entry.get_keys(100), &[1.0, 2.0]);
    }

    #[test]
    fn entry_get_values_clamped() {
        let mut entry = KvCacheEntry::new(4, 2);
        entry.append(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_eq!(entry.get_values(100), &[3.0, 4.0]);
    }

    #[test]
    fn entry_clear() {
        let mut entry = KvCacheEntry::new(4, 2);
        entry.append(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        entry.clear();
        assert_eq!(entry.current_len, 0);
        assert_eq!(entry.get_keys(10), &[] as &[f32]);
    }

    #[test]
    fn entry_is_full() {
        let mut entry = KvCacheEntry::new(2, 1);
        entry.append(&[1.0], &[2.0]).unwrap();
        assert!(!entry.is_full());
        entry.append(&[3.0], &[4.0]).unwrap();
        assert!(entry.is_full());
    }

    #[test]
    fn entry_cache_full_error() {
        let mut entry = KvCacheEntry::new(1, 1);
        entry.append(&[1.0], &[2.0]).unwrap();
        let err = entry.append(&[3.0], &[4.0]).unwrap_err();
        assert_eq!(err, KvCacheError::CacheFull { max_len: 1 });
    }

    #[test]
    fn entry_key_dimension_mismatch() {
        let mut entry = KvCacheEntry::new(4, 2);
        let err = entry.append(&[1.0], &[3.0, 4.0]).unwrap_err();
        assert_eq!(
            err,
            KvCacheError::DimensionMismatch { expected: 2, got: 1 }
        );
    }

    #[test]
    fn entry_value_dimension_mismatch() {
        let mut entry = KvCacheEntry::new(4, 2);
        let err = entry.append(&[1.0, 2.0], &[3.0]).unwrap_err();
        assert_eq!(
            err,
            KvCacheError::DimensionMismatch { expected: 2, got: 1 }
        );
    }

    #[test]
    fn entry_max_len_one() {
        let mut entry = KvCacheEntry::new(1, 3);
        entry.append(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert!(entry.is_full());
        assert_eq!(entry.get_keys(1), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn entry_head_dim_one() {
        let mut entry = KvCacheEntry::new(4, 1);
        entry.append(&[42.0], &[99.0]).unwrap();
        assert_eq!(entry.get_keys(1), &[42.0]);
        assert_eq!(entry.get_values(1), &[99.0]);
    }

    // -- KvCache multi-layer ------------------------------------------------

    #[test]
    fn cache_new_layers_count() {
        let cache = KvCache::new(default_config());
        assert_eq!(cache.layers.len(), 4);
    }

    #[test]
    fn cache_append_and_get_layer() {
        let cfg = KvCacheConfig {
            max_seq_len: 4,
            num_heads: 1,
            head_dim: 2,
            num_layers: 2,
            dtype_bytes: 4,
        };
        let mut cache = KvCache::new(cfg);
        cache.append_layer(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        cache.append_layer(1, &[5.0, 6.0], &[7.0, 8.0]).unwrap();

        let l0 = cache.get_layer(0).unwrap();
        assert_eq!(l0.get_keys(1), &[1.0, 2.0]);
        let l1 = cache.get_layer(1).unwrap();
        assert_eq!(l1.get_values(1), &[7.0, 8.0]);
    }

    #[test]
    fn cache_layer_out_of_bounds_append() {
        let cfg = KvCacheConfig {
            max_seq_len: 4,
            num_heads: 1,
            head_dim: 2,
            num_layers: 2,
            dtype_bytes: 4,
        };
        let mut cache = KvCache::new(cfg);
        let err =
            cache.append_layer(5, &[1.0, 2.0], &[3.0, 4.0]).unwrap_err();
        assert_eq!(
            err,
            KvCacheError::LayerOutOfBounds { requested: 5, available: 2 }
        );
    }

    #[test]
    fn cache_layer_out_of_bounds_get() {
        let cache = KvCache::new(default_config());
        let err = cache.get_layer(100).unwrap_err();
        assert_eq!(
            err,
            KvCacheError::LayerOutOfBounds { requested: 100, available: 4 }
        );
    }

    #[test]
    fn cache_clear_all() {
        let cfg = KvCacheConfig {
            max_seq_len: 4,
            num_heads: 1,
            head_dim: 1,
            num_layers: 2,
            dtype_bytes: 4,
        };
        let mut cache = KvCache::new(cfg);
        cache.append_layer(0, &[1.0], &[2.0]).unwrap();
        cache.append_layer(1, &[3.0], &[4.0]).unwrap();
        cache.clear_all();
        assert_eq!(cache.current_sequence_length(), 0);
    }

    #[test]
    fn cache_current_sequence_length() {
        let cfg = KvCacheConfig {
            max_seq_len: 8,
            num_heads: 1,
            head_dim: 1,
            num_layers: 1,
            dtype_bytes: 4,
        };
        let mut cache = KvCache::new(cfg);
        assert_eq!(cache.current_sequence_length(), 0);
        cache.append_layer(0, &[1.0], &[2.0]).unwrap();
        assert_eq!(cache.current_sequence_length(), 1);
        cache.append_layer(0, &[3.0], &[4.0]).unwrap();
        assert_eq!(cache.current_sequence_length(), 2);
    }

    #[test]
    fn cache_memory_usage_bytes() {
        let cfg = KvCacheConfig {
            max_seq_len: 4,
            num_heads: 1,
            head_dim: 2,
            num_layers: 2,
            dtype_bytes: 4,
        };
        let cache = KvCache::new(cfg);
        // Each layer: key(4*2) + value(4*2) = 8 f32 = 32 bytes; ×2 layers
        assert_eq!(cache.memory_usage_bytes(), 128);
    }

    #[test]
    fn cache_sequential_append_full_sequence() {
        let cfg = KvCacheConfig {
            max_seq_len: 3,
            num_heads: 1,
            head_dim: 2,
            num_layers: 1,
            dtype_bytes: 4,
        };
        let mut cache = KvCache::new(cfg);
        for i in 0..3 {
            let v = i as f32;
            cache.append_layer(0, &[v, v + 0.5], &[v + 10.0, v + 10.5]).unwrap();
        }
        let layer = cache.get_layer(0).unwrap();
        assert!(layer.is_full());
        assert_eq!(
            layer.get_keys(3),
            &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        );
    }

    #[test]
    fn cache_dimension_mismatch_propagates() {
        let cfg = KvCacheConfig {
            max_seq_len: 4,
            num_heads: 2,
            head_dim: 3,
            num_layers: 1,
            dtype_bytes: 4,
        };
        let mut cache = KvCache::new(cfg);
        let err =
            cache.append_layer(0, &[1.0, 2.0], &[3.0, 4.0]).unwrap_err();
        assert_eq!(
            err,
            KvCacheError::DimensionMismatch { expected: 6, got: 2 }
        );
    }

    #[test]
    fn cache_empty_sequence_length() {
        let cfg = KvCacheConfig {
            max_seq_len: 4,
            num_heads: 1,
            head_dim: 1,
            num_layers: 0,
            dtype_bytes: 4,
        };
        let cache = KvCache::new(cfg);
        assert_eq!(cache.current_sequence_length(), 0);
        assert_eq!(cache.memory_usage_bytes(), 0);
    }

    // -- Error Display ------------------------------------------------------

    #[test]
    fn error_display_layer_out_of_bounds() {
        let e = KvCacheError::LayerOutOfBounds { requested: 5, available: 4 };
        assert_eq!(
            e.to_string(),
            "layer index 5 out of bounds (available: 4)"
        );
    }

    #[test]
    fn error_display_cache_full() {
        let e = KvCacheError::CacheFull { max_len: 2048 };
        assert_eq!(e.to_string(), "KV cache is full (max_len=2048)");
    }

    #[test]
    fn error_display_dimension_mismatch() {
        let e = KvCacheError::DimensionMismatch { expected: 64, got: 32 };
        assert_eq!(
            e.to_string(),
            "dimension mismatch: expected 64 elements, got 32"
        );
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> =
            Box::new(KvCacheError::CacheFull { max_len: 1 });
        assert!(e.to_string().contains("full"));
    }

    // -- OpenCL kernel source validation ------------------------------------

    #[test]
    fn kernel_source_non_empty() {
        assert!(!KV_CACHE_CL.is_empty());
    }

    #[test]
    fn kernel_source_contains_append() {
        assert!(KV_CACHE_CL.contains("kv_cache_append"));
    }

    #[test]
    fn kernel_source_contains_gather() {
        assert!(KV_CACHE_CL.contains("kv_cache_gather"));
    }

    #[test]
    fn kernel_source_contains_kernel_keyword() {
        assert!(KV_CACHE_CL.contains("__kernel"));
    }

    #[test]
    fn kernel_source_contains_global_qualifier() {
        assert!(KV_CACHE_CL.contains("__global"));
    }
}
