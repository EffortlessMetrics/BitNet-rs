//! Advanced CPU KV cache operations kernel.
//!
//! Extends the base KV cache with gather, rotate, copy, quantize/dequantize,
//! sliding-window extraction, and beam-search reorder operations used by
//! paged-attention and speculative-decoding inference paths.

use std::fmt;

// ── Error type ─────────────────────────────────────────────────────

/// Errors returned by advanced KV cache operations.
#[derive(Debug, Clone, PartialEq)]
pub enum KvCacheError {
    /// A positional or layer index was out of range.
    OutOfBounds { index: usize, limit: usize },
    /// Operation would exceed the configured capacity.
    CapacityOverflow { requested: usize, capacity: usize },
    /// Dimension / length mismatch between operands.
    DimensionMismatch { expected: usize, got: usize },
    /// An argument was invalid or nonsensical.
    InvalidArgument(String),
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds { index, limit } => {
                write!(f, "index {index} out of bounds (limit {limit})")
            }
            Self::CapacityOverflow { requested, capacity } => {
                write!(f, "capacity overflow: requested {requested}, capacity {capacity}")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for KvCacheError {}

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for advanced KV cache operations.
#[derive(Debug, Clone)]
pub struct KvCacheOpsConfig {
    /// Maximum sequence length the cache can hold.
    pub max_seq_len: usize,
    /// Number of KV heads per layer.
    pub num_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Page size for paged operations (in tokens).
    pub page_size: usize,
}

impl KvCacheOpsConfig {
    /// Elements per single token across all heads.
    #[inline]
    pub fn token_elements(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), KvCacheError> {
        if self.max_seq_len == 0 {
            return Err(KvCacheError::InvalidArgument("max_seq_len must be > 0".into()));
        }
        if self.num_heads == 0 {
            return Err(KvCacheError::InvalidArgument("num_heads must be > 0".into()));
        }
        if self.head_dim == 0 {
            return Err(KvCacheError::InvalidArgument("head_dim must be > 0".into()));
        }
        if self.num_layers == 0 {
            return Err(KvCacheError::InvalidArgument("num_layers must be > 0".into()));
        }
        if self.page_size == 0 {
            return Err(KvCacheError::InvalidArgument("page_size must be > 0".into()));
        }
        Ok(())
    }
}

// ── Per-layer cache block ──────────────────────────────────────────

/// Key/value storage for a single transformer layer.
///
/// Layout: `[max_seq_len, num_heads * head_dim]` flattened in row-major order.
/// `seq_len` tracks how many positions are occupied.
#[derive(Debug, Clone)]
pub struct KvCacheOpsBlock {
    /// Cached key vectors.
    pub keys: Vec<f32>,
    /// Cached value vectors.
    pub values: Vec<f32>,
    /// Number of occupied token positions.
    pub seq_len: usize,
    /// Elements per token (`num_heads * head_dim`).
    token_elements: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
}

impl KvCacheOpsBlock {
    /// Create a new block pre-allocated for `max_seq_len` tokens.
    pub fn new(token_elements: usize, max_seq_len: usize) -> Self {
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
        self.max_seq_len.saturating_sub(self.seq_len)
    }
}

// ── Multi-layer cache ──────────────────────────────────────────────

/// Multi-layer cache wrapping one [`KvCacheOpsBlock`] per layer.
#[derive(Debug, Clone)]
pub struct KvCacheOps {
    /// Per-layer blocks.
    pub blocks: Vec<KvCacheOpsBlock>,
    /// Configuration snapshot.
    pub config: KvCacheOpsConfig,
}

impl KvCacheOps {
    /// Allocate a new cache from a validated configuration.
    pub fn new(config: KvCacheOpsConfig) -> Result<Self, KvCacheError> {
        config.validate()?;
        let te = config.token_elements();
        let blocks =
            (0..config.num_layers).map(|_| KvCacheOpsBlock::new(te, config.max_seq_len)).collect();
        Ok(Self { blocks, config })
    }

    fn block(&self, layer: usize) -> Result<&KvCacheOpsBlock, KvCacheError> {
        self.blocks
            .get(layer)
            .ok_or(KvCacheError::OutOfBounds { index: layer, limit: self.blocks.len() })
    }

    fn block_mut(&mut self, layer: usize) -> Result<&mut KvCacheOpsBlock, KvCacheError> {
        let n = self.blocks.len();
        self.blocks.get_mut(layer).ok_or(KvCacheError::OutOfBounds { index: layer, limit: n })
    }
}

// ── Kernel operations ──────────────────────────────────────────────

/// Append new key/value entries at a position in `layer`.
///
/// `new_keys` / `new_values` must each have length
/// `num_tokens * num_heads * head_dim`.
pub fn kv_cache_ops_append(
    cache: &mut KvCacheOps,
    layer: usize,
    new_keys: &[f32],
    new_values: &[f32],
) -> Result<(), KvCacheError> {
    let blk = cache.block_mut(layer)?;
    let te = blk.token_elements;
    if te == 0 {
        return Err(KvCacheError::InvalidArgument("token_elements is 0".into()));
    }
    if !new_keys.len().is_multiple_of(te) {
        return Err(KvCacheError::DimensionMismatch { expected: te, got: new_keys.len() % te });
    }
    if new_keys.len() != new_values.len() {
        return Err(KvCacheError::DimensionMismatch {
            expected: new_keys.len(),
            got: new_values.len(),
        });
    }
    let new_tokens = new_keys.len() / te;
    if new_tokens > blk.remaining() {
        return Err(KvCacheError::CapacityOverflow {
            requested: blk.seq_len + new_tokens,
            capacity: blk.max_seq_len,
        });
    }
    let offset = blk.seq_len * te;
    let n = new_keys.len();
    blk.keys[offset..offset + n].copy_from_slice(new_keys);
    blk.values[offset..offset + n].copy_from_slice(new_values);
    blk.seq_len += new_tokens;
    Ok(())
}

/// Gather key/value vectors from scattered positions for paged attention.
///
/// `positions` contains token indices to gather from.  Returns
/// `(gathered_keys, gathered_values)` each of length
/// `positions.len() * token_elements`.
pub fn kv_cache_ops_gather(
    cache: &KvCacheOps,
    layer: usize,
    positions: &[usize],
) -> Result<(Vec<f32>, Vec<f32>), KvCacheError> {
    let blk = cache.block(layer)?;
    let te = blk.token_elements;
    let mut gathered_keys = Vec::with_capacity(positions.len() * te);
    let mut gathered_values = Vec::with_capacity(positions.len() * te);
    for &pos in positions {
        if pos >= blk.seq_len {
            return Err(KvCacheError::OutOfBounds { index: pos, limit: blk.seq_len });
        }
        let start = pos * te;
        let end = start + te;
        gathered_keys.extend_from_slice(&blk.keys[start..end]);
        gathered_values.extend_from_slice(&blk.values[start..end]);
    }
    Ok((gathered_keys, gathered_values))
}

/// Circular-buffer rotation: evict the oldest `evict_count` tokens by
/// shifting remaining entries to the front, then reset `seq_len`.
///
/// This is a rotating-buffer eviction strategy used when the cache is
/// full and new tokens need to be appended.
pub fn kv_cache_ops_rotate(
    cache: &mut KvCacheOps,
    layer: usize,
    evict_count: usize,
) -> Result<(), KvCacheError> {
    let blk = cache.block_mut(layer)?;
    if evict_count > blk.seq_len {
        return Err(KvCacheError::OutOfBounds { index: evict_count, limit: blk.seq_len });
    }
    if evict_count == 0 {
        return Ok(());
    }
    let te = blk.token_elements;
    let src_start = evict_count * te;
    let remaining_elems = (blk.seq_len - evict_count) * te;
    blk.keys.copy_within(src_start..src_start + remaining_elems, 0);
    blk.values.copy_within(src_start..src_start + remaining_elems, 0);
    blk.seq_len -= evict_count;
    Ok(())
}

/// Copy cache entries between layers and/or positions.
///
/// Copies `count` tokens starting at `src_pos` in `src_layer` to
/// `dst_pos` in `dst_layer`.  Source and destination layers may be the
/// same only when ranges do not overlap.
pub fn kv_cache_ops_copy(
    cache: &mut KvCacheOps,
    src_layer: usize,
    src_pos: usize,
    dst_layer: usize,
    dst_pos: usize,
    count: usize,
) -> Result<(), KvCacheError> {
    if count == 0 {
        return Ok(());
    }
    let num_layers = cache.blocks.len();
    if src_layer >= num_layers {
        return Err(KvCacheError::OutOfBounds { index: src_layer, limit: num_layers });
    }
    if dst_layer >= num_layers {
        return Err(KvCacheError::OutOfBounds { index: dst_layer, limit: num_layers });
    }
    let te = cache.blocks[0].token_elements;

    // Validate source range.
    let src_seq = cache.blocks[src_layer].seq_len;
    if src_pos + count > src_seq {
        return Err(KvCacheError::OutOfBounds { index: src_pos + count, limit: src_seq });
    }

    // Validate destination range.
    let dst_max = cache.blocks[dst_layer].max_seq_len;
    if dst_pos + count > dst_max {
        return Err(KvCacheError::CapacityOverflow {
            requested: dst_pos + count,
            capacity: dst_max,
        });
    }

    let src_start = src_pos * te;
    let dst_start = dst_pos * te;
    let len = count * te;

    if src_layer == dst_layer {
        // In-place copy within the same block.
        let blk = &mut cache.blocks[src_layer];
        blk.keys.copy_within(src_start..src_start + len, dst_start);
        blk.values.copy_within(src_start..src_start + len, dst_start);
        // Update seq_len if we wrote past it.
        if dst_pos + count > blk.seq_len {
            blk.seq_len = dst_pos + count;
        }
    } else {
        // Cross-layer copy: extract source data then write to destination.
        let src_keys: Vec<f32> = cache.blocks[src_layer].keys[src_start..src_start + len].to_vec();
        let src_vals: Vec<f32> =
            cache.blocks[src_layer].values[src_start..src_start + len].to_vec();
        let dst = &mut cache.blocks[dst_layer];
        dst.keys[dst_start..dst_start + len].copy_from_slice(&src_keys);
        dst.values[dst_start..dst_start + len].copy_from_slice(&src_vals);
        if dst_pos + count > dst.seq_len {
            dst.seq_len = dst_pos + count;
        }
    }
    Ok(())
}

/// Result of int8 quantization: quantized keys/values and their per-token scales.
pub type QuantizedKvData = (Vec<i8>, Vec<i8>, Vec<f32>, Vec<f32>);

/// Quantize cached key/value data to int8 with per-token absmax scaling.
///
/// Returns `(quantized_keys, quantized_values, key_scales, value_scales)`.
/// Each scale is the absmax of the corresponding token vector.
pub fn kv_cache_ops_quantize_i8(
    cache: &KvCacheOps,
    layer: usize,
) -> Result<QuantizedKvData, KvCacheError> {
    let blk = cache.block(layer)?;
    let te = blk.token_elements;
    let n = blk.seq_len;
    let total = n * te;

    let mut q_keys = vec![0i8; total];
    let mut q_values = vec![0i8; total];
    let mut key_scales = Vec::with_capacity(n);
    let mut value_scales = Vec::with_capacity(n);

    for t in 0..n {
        let start = t * te;
        let end = start + te;

        // Keys
        let k_slice = &blk.keys[start..end];
        let k_absmax = k_slice.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let k_scale = if k_absmax == 0.0 { 1.0 } else { k_absmax };
        let inv_k = 127.0 / k_scale;
        for (i, &val) in k_slice.iter().enumerate() {
            q_keys[start + i] = (val * inv_k).round().clamp(-128.0, 127.0) as i8;
        }
        key_scales.push(k_scale);

        // Values
        let v_slice = &blk.values[start..end];
        let v_absmax = v_slice.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let v_scale = if v_absmax == 0.0 { 1.0 } else { v_absmax };
        let inv_v = 127.0 / v_scale;
        for (i, &val) in v_slice.iter().enumerate() {
            q_values[start + i] = (val * inv_v).round().clamp(-128.0, 127.0) as i8;
        }
        value_scales.push(v_scale);
    }

    Ok((q_keys, q_values, key_scales, value_scales))
}

/// Dequantize int8 key/value data back to f32 using per-token scales.
///
/// `quantized_keys` / `quantized_values` are `[num_tokens * token_elements]`.
/// `key_scales` / `value_scales` have one entry per token.
pub fn kv_cache_ops_dequantize_i8(
    quantized_keys: &[i8],
    quantized_values: &[i8],
    key_scales: &[f32],
    value_scales: &[f32],
    token_elements: usize,
) -> Result<(Vec<f32>, Vec<f32>), KvCacheError> {
    if token_elements == 0 {
        return Err(KvCacheError::InvalidArgument("token_elements must be > 0".into()));
    }
    let num_tokens = key_scales.len();
    if value_scales.len() != num_tokens {
        return Err(KvCacheError::DimensionMismatch {
            expected: num_tokens,
            got: value_scales.len(),
        });
    }
    let expected_len = num_tokens * token_elements;
    if quantized_keys.len() != expected_len {
        return Err(KvCacheError::DimensionMismatch {
            expected: expected_len,
            got: quantized_keys.len(),
        });
    }
    if quantized_values.len() != expected_len {
        return Err(KvCacheError::DimensionMismatch {
            expected: expected_len,
            got: quantized_values.len(),
        });
    }

    let mut keys = vec![0.0f32; expected_len];
    let mut values = vec![0.0f32; expected_len];

    for t in 0..num_tokens {
        let start = t * token_elements;
        let end = start + token_elements;
        let k_denom = key_scales[t] / 127.0;
        let v_denom = value_scales[t] / 127.0;
        for i in start..end {
            keys[i] = quantized_keys[i] as f32 * k_denom;
            values[i] = quantized_values[i] as f32 * v_denom;
        }
    }

    Ok((keys, values))
}

/// Extract a sliding window of the most recent `window_size` tokens.
///
/// Returns `(keys, values)` each of length `window_size * token_elements`
/// (or fewer if `seq_len < window_size`).
pub fn kv_cache_ops_sliding_window(
    cache: &KvCacheOps,
    layer: usize,
    window_size: usize,
) -> Result<(Vec<f32>, Vec<f32>), KvCacheError> {
    if window_size == 0 {
        return Err(KvCacheError::InvalidArgument("window_size must be > 0".into()));
    }
    let blk = cache.block(layer)?;
    let te = blk.token_elements;
    let actual_window = window_size.min(blk.seq_len);
    let start_pos = blk.seq_len - actual_window;
    let start = start_pos * te;
    let end = blk.seq_len * te;
    Ok((blk.keys[start..end].to_vec(), blk.values[start..end].to_vec()))
}

/// Reorder cache entries for beam search.
///
/// `beam_indices[i]` is the source beam index for output position `i`.
/// Reorders the first `seq_len` tokens according to the permutation,
/// allowing beam hypotheses to be rearranged in-place.
pub fn kv_cache_ops_reorder_beam(
    cache: &mut KvCacheOps,
    layer: usize,
    beam_indices: &[usize],
) -> Result<(), KvCacheError> {
    let blk = cache.block_mut(layer)?;
    let beam_count = beam_indices.len();
    if beam_count == 0 {
        return Err(KvCacheError::InvalidArgument("beam_indices must not be empty".into()));
    }
    // Validate all indices.
    for &idx in beam_indices {
        if idx >= beam_count {
            return Err(KvCacheError::OutOfBounds { index: idx, limit: beam_count });
        }
    }
    let te = blk.token_elements;
    let seq = blk.seq_len;

    // Reorder all token positions: for each position, copy from source beam.
    // We operate on head-stripes: each beam owns `seq * te / beam_count`
    // elements… but the simpler model: cache is `[seq_len, token_elements]`
    // and beams correspond to different hypotheses that share the same
    // positional layout.  We treat the first dimension as beam_count blocks
    // of `tokens_per_beam` tokens.
    //
    // Simpler approach: reorder per-position across all heads.
    // beam_indices maps output-beam → source-beam.  Each beam's tokens
    // span `[beam_idx * tokens_per_beam .. (beam_idx+1) * tokens_per_beam]`.
    let tokens_per_beam = seq / beam_count;
    if tokens_per_beam == 0 || seq % beam_count != 0 {
        // Fallback: treat as identity if layout doesn't divide evenly.
        return if seq == 0 {
            Ok(())
        } else {
            Err(KvCacheError::DimensionMismatch { expected: beam_count, got: seq })
        };
    }

    let block_elems = tokens_per_beam * te;
    let old_keys = blk.keys[..seq * te].to_vec();
    let old_values = blk.values[..seq * te].to_vec();

    for (dst_beam, &src_beam) in beam_indices.iter().enumerate() {
        let dst_off = dst_beam * block_elems;
        let src_off = src_beam * block_elems;
        blk.keys[dst_off..dst_off + block_elems]
            .copy_from_slice(&old_keys[src_off..src_off + block_elems]);
        blk.values[dst_off..dst_off + block_elems]
            .copy_from_slice(&old_values[src_off..src_off + block_elems]);
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::too_many_lines)]
mod tests {
    use super::*;

    fn default_cfg() -> KvCacheOpsConfig {
        KvCacheOpsConfig { max_seq_len: 16, num_heads: 4, head_dim: 8, num_layers: 2, page_size: 4 }
    }

    fn make_cache() -> KvCacheOps {
        KvCacheOps::new(default_cfg()).unwrap()
    }

    fn te(cfg: &KvCacheOpsConfig) -> usize {
        cfg.num_heads * cfg.head_dim
    }

    // ── Config validation ──────────────────────────────────────────

    #[test]
    fn test_config_valid() {
        assert!(default_cfg().validate().is_ok());
    }

    #[test]
    fn test_config_zero_max_seq_len() {
        let mut c = default_cfg();
        c.max_seq_len = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_num_heads() {
        let mut c = default_cfg();
        c.num_heads = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_head_dim() {
        let mut c = default_cfg();
        c.head_dim = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_num_layers() {
        let mut c = default_cfg();
        c.num_layers = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_page_size() {
        let mut c = default_cfg();
        c.page_size = 0;
        assert!(c.validate().is_err());
    }

    // ── Error Display ──────────────────────────────────────────────

    #[test]
    fn test_error_display_out_of_bounds() {
        let e = KvCacheError::OutOfBounds { index: 5, limit: 3 };
        assert!(e.to_string().contains("5"));
        assert!(e.to_string().contains("3"));
    }

    #[test]
    fn test_error_display_capacity_overflow() {
        let e = KvCacheError::CapacityOverflow { requested: 20, capacity: 16 };
        assert!(e.to_string().contains("20"));
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = KvCacheError::DimensionMismatch { expected: 32, got: 5 };
        assert!(e.to_string().contains("32"));
    }

    #[test]
    fn test_error_display_invalid_argument() {
        let e = KvCacheError::InvalidArgument("bad".into());
        assert!(e.to_string().contains("bad"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(KvCacheError::InvalidArgument("test".into()));
        assert!(!e.to_string().is_empty());
    }

    // ── Cache construction ─────────────────────────────────────────

    #[test]
    fn test_new_cache_num_layers() {
        let cache = make_cache();
        assert_eq!(cache.blocks.len(), 2);
    }

    #[test]
    fn test_new_cache_initial_seq_len() {
        let cache = make_cache();
        assert_eq!(cache.blocks[0].seq_len, 0);
        assert_eq!(cache.blocks[1].seq_len, 0);
    }

    #[test]
    fn test_new_cache_remaining() {
        let cache = make_cache();
        assert_eq!(cache.blocks[0].remaining(), 16);
    }

    // ── Append ─────────────────────────────────────────────────────

    #[test]
    fn test_append_single_token() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        let k = vec![1.0f32; t];
        let v = vec![2.0f32; t];
        kv_cache_ops_append(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(cache.blocks[0].seq_len, 1);
        assert_eq!(cache.blocks[1].seq_len, 0);
    }

    #[test]
    fn test_append_multiple_tokens() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        let k = vec![1.0; t * 3];
        let v = vec![2.0; t * 3];
        kv_cache_ops_append(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(cache.blocks[0].seq_len, 3);
    }

    #[test]
    fn test_append_read_back() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        let k: Vec<f32> = (0..t).map(|i| i as f32).collect();
        let v: Vec<f32> = (0..t).map(|i| i as f32 + 100.0).collect();
        kv_cache_ops_append(&mut cache, 0, &k, &v).unwrap();
        assert_eq!(&cache.blocks[0].keys[..t], &k[..]);
        assert_eq!(&cache.blocks[0].values[..t], &v[..]);
    }

    #[test]
    fn test_append_capacity_overflow() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        let k = vec![0.0; t * 17];
        let v = vec![0.0; t * 17];
        assert!(kv_cache_ops_append(&mut cache, 0, &k, &v).is_err());
    }

    #[test]
    fn test_append_mismatched_lengths() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        let k = vec![0.0; t];
        let v = vec![0.0; t * 2];
        assert!(kv_cache_ops_append(&mut cache, 0, &k, &v).is_err());
    }

    #[test]
    fn test_append_bad_alignment() {
        let mut cache = make_cache();
        let k = vec![0.0; 5];
        let v = vec![0.0; 5];
        assert!(kv_cache_ops_append(&mut cache, 0, &k, &v).is_err());
    }

    #[test]
    fn test_append_invalid_layer() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        let k = vec![0.0; t];
        let v = vec![0.0; t];
        assert!(kv_cache_ops_append(&mut cache, 99, &k, &v).is_err());
    }

    // ── Gather ─────────────────────────────────────────────────────

    #[test]
    fn test_gather_basic() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        // Append 4 tokens with distinct values.
        for i in 0..4u32 {
            let k = vec![i as f32; t];
            let v = vec![(i as f32) * 10.0; t];
            kv_cache_ops_append(&mut cache, 0, &k, &v).unwrap();
        }
        let (gk, gv) = kv_cache_ops_gather(&cache, 0, &[2, 0]).unwrap();
        assert_eq!(gk.len(), 2 * t);
        assert_eq!(gk[0], 2.0);
        assert_eq!(gk[t], 0.0);
        assert_eq!(gv[0], 20.0);
        assert_eq!(gv[t], 0.0);
    }

    #[test]
    fn test_gather_empty_positions() {
        let mut cache = make_cache();
        let cfg = default_cfg();
        let t = te(&cfg);
        kv_cache_ops_append(&mut cache, 0, &vec![1.0; t], &vec![2.0; t]).unwrap();
        let (gk, gv) = kv_cache_ops_gather(&cache, 0, &[]).unwrap();
        assert!(gk.is_empty());
        assert!(gv.is_empty());
    }

    #[test]
    fn test_gather_out_of_bounds() {
        let cache = make_cache();
        assert!(kv_cache_ops_gather(&cache, 0, &[0]).is_err());
    }

    #[test]
    fn test_gather_duplicate_positions() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![7.0; t], &vec![8.0; t]).unwrap();
        let (gk, _) = kv_cache_ops_gather(&cache, 0, &[0, 0, 0]).unwrap();
        assert_eq!(gk.len(), 3 * t);
        assert_eq!(gk[0], 7.0);
        assert_eq!(gk[t], 7.0);
        assert_eq!(gk[2 * t], 7.0);
    }

    // ── Rotate ─────────────────────────────────────────────────────

    #[test]
    fn test_rotate_evict_oldest() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        for i in 0..4u32 {
            kv_cache_ops_append(&mut cache, 0, &vec![i as f32; t], &vec![0.0; t]).unwrap();
        }
        kv_cache_ops_rotate(&mut cache, 0, 2).unwrap();
        assert_eq!(cache.blocks[0].seq_len, 2);
        // Remaining tokens should be [2.0, 3.0].
        assert_eq!(cache.blocks[0].keys[0], 2.0);
        assert_eq!(cache.blocks[0].keys[t], 3.0);
    }

    #[test]
    fn test_rotate_evict_all() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![1.0; t * 3], &vec![2.0; t * 3]).unwrap();
        kv_cache_ops_rotate(&mut cache, 0, 3).unwrap();
        assert_eq!(cache.blocks[0].seq_len, 0);
    }

    #[test]
    fn test_rotate_zero_evict() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![1.0; t], &vec![2.0; t]).unwrap();
        kv_cache_ops_rotate(&mut cache, 0, 0).unwrap();
        assert_eq!(cache.blocks[0].seq_len, 1);
    }

    #[test]
    fn test_rotate_exceeds_seq_len() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![1.0; t], &vec![2.0; t]).unwrap();
        assert!(kv_cache_ops_rotate(&mut cache, 0, 5).is_err());
    }

    // ── Copy ───────────────────────────────────────────────────────

    #[test]
    fn test_copy_same_layer() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        for i in 0..4u32 {
            kv_cache_ops_append(&mut cache, 0, &vec![i as f32; t], &vec![0.0; t]).unwrap();
        }
        // Copy token 0 to position 4 within layer 0.
        kv_cache_ops_copy(&mut cache, 0, 0, 0, 4, 1).unwrap();
        assert_eq!(cache.blocks[0].seq_len, 5);
        assert_eq!(cache.blocks[0].keys[4 * t], 0.0);
    }

    #[test]
    fn test_copy_cross_layer() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![5.0; t * 2], &vec![6.0; t * 2]).unwrap();
        kv_cache_ops_copy(&mut cache, 0, 0, 1, 0, 2).unwrap();
        assert_eq!(cache.blocks[1].seq_len, 2);
        assert_eq!(cache.blocks[1].keys[0], 5.0);
        assert_eq!(cache.blocks[1].values[0], 6.0);
    }

    #[test]
    fn test_copy_zero_count() {
        let mut cache = make_cache();
        kv_cache_ops_copy(&mut cache, 0, 0, 1, 0, 0).unwrap();
    }

    #[test]
    fn test_copy_src_out_of_bounds() {
        let mut cache = make_cache();
        assert!(kv_cache_ops_copy(&mut cache, 0, 0, 1, 0, 1).is_err());
    }

    #[test]
    fn test_copy_dst_capacity_overflow() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![1.0; t], &vec![2.0; t]).unwrap();
        // Destination position would exceed max_seq_len.
        assert!(kv_cache_ops_copy(&mut cache, 0, 0, 1, 16, 1).is_err());
    }

    #[test]
    fn test_copy_invalid_layer() {
        let mut cache = make_cache();
        assert!(kv_cache_ops_copy(&mut cache, 99, 0, 0, 0, 1).is_err());
    }

    // ── Quantize / Dequantize ──────────────────────────────────────

    #[test]
    fn test_quantize_dequantize_round_trip() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        let k: Vec<f32> = (0..t).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let v: Vec<f32> = (0..t).map(|i| (i as f32) * 0.05).collect();
        kv_cache_ops_append(&mut cache, 0, &k, &v).unwrap();

        let (qk, qv, ks, vs) = kv_cache_ops_quantize_i8(&cache, 0).unwrap();
        let (dk, dv) = kv_cache_ops_dequantize_i8(&qk, &qv, &ks, &vs, t).unwrap();

        for i in 0..t {
            let k_err = (dk[i] - k[i]).abs();
            let v_err = (dv[i] - v[i]).abs();
            // Per-token absmax quantization: error bounded by scale / 127.
            assert!(k_err < ks[0] / 127.0 + 1e-6, "key error {k_err} at {i}");
            assert!(v_err < vs[0] / 127.0 + 1e-6, "value error {v_err} at {i}");
        }
    }

    #[test]
    fn test_quantize_empty_cache() {
        let cache = make_cache();
        let (qk, qv, ks, vs) = kv_cache_ops_quantize_i8(&cache, 0).unwrap();
        assert!(qk.is_empty());
        assert!(qv.is_empty());
        assert!(ks.is_empty());
        assert!(vs.is_empty());
    }

    #[test]
    fn test_quantize_zeros() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![0.0; t], &vec![0.0; t]).unwrap();
        let (qk, qv, ks, vs) = kv_cache_ops_quantize_i8(&cache, 0).unwrap();
        assert!(qk.iter().all(|&x| x == 0));
        assert!(qv.iter().all(|&x| x == 0));
        assert_eq!(ks[0], 1.0); // scale defaults to 1.0 for zero inputs
        assert_eq!(vs[0], 1.0);
    }

    #[test]
    fn test_quantize_multi_token() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![1.0; t], &vec![2.0; t]).unwrap();
        kv_cache_ops_append(&mut cache, 0, &vec![3.0; t], &vec![4.0; t]).unwrap();
        let (qk, _qv, ks, _vs) = kv_cache_ops_quantize_i8(&cache, 0).unwrap();
        assert_eq!(ks.len(), 2);
        assert_eq!(qk.len(), 2 * t);
    }

    #[test]
    fn test_dequantize_mismatched_scales() {
        let qk = vec![0i8; 32];
        let qv = vec![0i8; 32];
        let ks = vec![1.0f32; 1];
        let vs = vec![1.0f32; 2]; // Mismatch
        assert!(kv_cache_ops_dequantize_i8(&qk, &qv, &ks, &vs, 32).is_err());
    }

    #[test]
    fn test_dequantize_wrong_length() {
        let qk = vec![0i8; 10]; // Wrong length
        let qv = vec![0i8; 32];
        let ks = vec![1.0f32; 1];
        let vs = vec![1.0f32; 1];
        assert!(kv_cache_ops_dequantize_i8(&qk, &qv, &ks, &vs, 32).is_err());
    }

    #[test]
    fn test_dequantize_zero_token_elements() {
        assert!(kv_cache_ops_dequantize_i8(&[], &[], &[], &[], 0).is_err());
    }

    // ── Sliding window ─────────────────────────────────────────────

    #[test]
    fn test_sliding_window_full() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        for i in 0..8u32 {
            kv_cache_ops_append(&mut cache, 0, &vec![i as f32; t], &vec![0.0; t]).unwrap();
        }
        let (wk, _) = kv_cache_ops_sliding_window(&cache, 0, 4).unwrap();
        assert_eq!(wk.len(), 4 * t);
        // Last 4 tokens: 4, 5, 6, 7.
        assert_eq!(wk[0], 4.0);
        assert_eq!(wk[t], 5.0);
        assert_eq!(wk[2 * t], 6.0);
        assert_eq!(wk[3 * t], 7.0);
    }

    #[test]
    fn test_sliding_window_larger_than_seq() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![1.0; t * 2], &vec![2.0; t * 2]).unwrap();
        let (wk, _) = kv_cache_ops_sliding_window(&cache, 0, 100).unwrap();
        // Should return all 2 tokens.
        assert_eq!(wk.len(), 2 * t);
    }

    #[test]
    fn test_sliding_window_size_one() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        for i in 0..5u32 {
            kv_cache_ops_append(&mut cache, 0, &vec![i as f32; t], &vec![0.0; t]).unwrap();
        }
        let (wk, _) = kv_cache_ops_sliding_window(&cache, 0, 1).unwrap();
        assert_eq!(wk.len(), t);
        assert_eq!(wk[0], 4.0); // Last token
    }

    #[test]
    fn test_sliding_window_empty_cache() {
        let cache = make_cache();
        let (wk, wv) = kv_cache_ops_sliding_window(&cache, 0, 4).unwrap();
        assert!(wk.is_empty());
        assert!(wv.is_empty());
    }

    #[test]
    fn test_sliding_window_zero_size() {
        let cache = make_cache();
        assert!(kv_cache_ops_sliding_window(&cache, 0, 0).is_err());
    }

    // ── Beam reorder ───────────────────────────────────────────────

    #[test]
    fn test_beam_reorder_identity() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        // 2 beams, 2 tokens each = 4 tokens total.
        for i in 0..4u32 {
            kv_cache_ops_append(&mut cache, 0, &vec![i as f32; t], &vec![0.0; t]).unwrap();
        }
        // Identity permutation.
        kv_cache_ops_reorder_beam(&mut cache, 0, &[0, 1]).unwrap();
        assert_eq!(cache.blocks[0].keys[0], 0.0);
        assert_eq!(cache.blocks[0].keys[2 * t], 2.0);
    }

    #[test]
    fn test_beam_reorder_swap() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        // 2 beams × 2 tokens = 4 tokens.
        for i in 0..4u32 {
            kv_cache_ops_append(&mut cache, 0, &vec![i as f32; t], &vec![0.0; t]).unwrap();
        }
        // Swap beam 0 ↔ beam 1.
        kv_cache_ops_reorder_beam(&mut cache, 0, &[1, 0]).unwrap();
        // Beam 0 now has tokens [2.0, 3.0], beam 1 has [0.0, 1.0].
        assert_eq!(cache.blocks[0].keys[0], 2.0);
        assert_eq!(cache.blocks[0].keys[t], 3.0);
        assert_eq!(cache.blocks[0].keys[2 * t], 0.0);
        assert_eq!(cache.blocks[0].keys[3 * t], 1.0);
    }

    #[test]
    fn test_beam_reorder_broadcast() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        // 2 beams × 2 tokens = 4 tokens.
        for i in 0..4u32 {
            kv_cache_ops_append(&mut cache, 0, &vec![i as f32; t], &vec![0.0; t]).unwrap();
        }
        // Both beams copy from beam 0.
        kv_cache_ops_reorder_beam(&mut cache, 0, &[0, 0]).unwrap();
        assert_eq!(cache.blocks[0].keys[0], 0.0);
        assert_eq!(cache.blocks[0].keys[t], 1.0);
        assert_eq!(cache.blocks[0].keys[2 * t], 0.0);
        assert_eq!(cache.blocks[0].keys[3 * t], 1.0);
    }

    #[test]
    fn test_beam_reorder_empty_indices() {
        let mut cache = make_cache();
        assert!(kv_cache_ops_reorder_beam(&mut cache, 0, &[]).is_err());
    }

    #[test]
    fn test_beam_reorder_invalid_index() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![1.0; t * 2], &vec![2.0; t * 2]).unwrap();
        assert!(kv_cache_ops_reorder_beam(&mut cache, 0, &[5, 0]).is_err());
    }

    #[test]
    fn test_beam_reorder_uneven_split() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        // 3 tokens doesn't divide evenly into 2 beams.
        kv_cache_ops_append(&mut cache, 0, &vec![1.0; t * 3], &vec![2.0; t * 3]).unwrap();
        assert!(kv_cache_ops_reorder_beam(&mut cache, 0, &[0, 1]).is_err());
    }

    // ── Multi-layer operations ─────────────────────────────────────

    #[test]
    fn test_multi_layer_append_independence() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![1.0; t], &vec![2.0; t]).unwrap();
        kv_cache_ops_append(&mut cache, 1, &vec![3.0; t * 3], &vec![4.0; t * 3]).unwrap();
        assert_eq!(cache.blocks[0].seq_len, 1);
        assert_eq!(cache.blocks[1].seq_len, 3);
    }

    #[test]
    fn test_multi_layer_gather_independence() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![10.0; t], &vec![20.0; t]).unwrap();
        kv_cache_ops_append(&mut cache, 1, &vec![30.0; t], &vec![40.0; t]).unwrap();
        let (gk0, _) = kv_cache_ops_gather(&cache, 0, &[0]).unwrap();
        let (gk1, _) = kv_cache_ops_gather(&cache, 1, &[0]).unwrap();
        assert_eq!(gk0[0], 10.0);
        assert_eq!(gk1[0], 30.0);
    }

    // ── Integration scenarios ──────────────────────────────────────

    #[test]
    fn test_append_rotate_append_cycle() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        // Fill to capacity.
        for i in 0..16u32 {
            kv_cache_ops_append(&mut cache, 0, &vec![i as f32; t], &vec![0.0; t]).unwrap();
        }
        assert_eq!(cache.blocks[0].seq_len, 16);
        // Evict oldest 4, then append 4 new.
        kv_cache_ops_rotate(&mut cache, 0, 4).unwrap();
        assert_eq!(cache.blocks[0].seq_len, 12);
        for i in 16..20u32 {
            kv_cache_ops_append(&mut cache, 0, &vec![i as f32; t], &vec![0.0; t]).unwrap();
        }
        assert_eq!(cache.blocks[0].seq_len, 16);
        // First token should now be 4.0.
        assert_eq!(cache.blocks[0].keys[0], 4.0);
    }

    #[test]
    fn test_sliding_window_after_rotate() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        for i in 0..8u32 {
            kv_cache_ops_append(&mut cache, 0, &vec![i as f32; t], &vec![0.0; t]).unwrap();
        }
        kv_cache_ops_rotate(&mut cache, 0, 3).unwrap();
        // seq_len = 5, tokens are [3, 4, 5, 6, 7].
        let (wk, _) = kv_cache_ops_sliding_window(&cache, 0, 3).unwrap();
        assert_eq!(wk.len(), 3 * t);
        assert_eq!(wk[0], 5.0);
        assert_eq!(wk[t], 6.0);
        assert_eq!(wk[2 * t], 7.0);
    }

    #[test]
    fn test_quantize_after_copy() {
        let cfg = default_cfg();
        let t = te(&cfg);
        let mut cache = make_cache();
        kv_cache_ops_append(&mut cache, 0, &vec![1.5; t], &vec![2.5; t]).unwrap();
        kv_cache_ops_copy(&mut cache, 0, 0, 1, 0, 1).unwrap();
        let (qk0, _, _, _) = kv_cache_ops_quantize_i8(&cache, 0).unwrap();
        let (qk1, _, _, _) = kv_cache_ops_quantize_i8(&cache, 1).unwrap();
        assert_eq!(qk0, qk1);
    }
}
