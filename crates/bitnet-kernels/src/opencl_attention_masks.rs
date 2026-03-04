//! Comprehensive attention mask generation and manipulation for OpenCL kernels.
//!
//! Provides CPU reference implementations of common attention mask patterns used
//! in transformer-based LLMs:
//!
//! - [`CausalMask`] — lower-triangular mask for autoregressive decoding
//! - [`PaddingMask`] — variable-length sequence masking (batch-aware)
//! - [`SlidingWindowMask`] — bounded local attention window
//! - [`PrefixMask`] — full attention on a prefix, causal on the remainder
//! - [`BlockSparseMask`] — block-diagonal sparse attention pattern
//! - [`MaskCombiner`] — bitwise AND/OR/XOR composition of masks
//! - [`MaskExpander`] — expand 2D masks to 4D `[batch, heads, seq, seq]`
//! - [`MaskFormat`] / [`MaskConverter`] — conversion between bool, float, additive,
//!   and multiplicative representations
//! - [`MaskStats`] — sparsity ratio and memory savings estimates
//!
//! All implementations are pure CPU (no `opencl3` dependency) so the module
//! compiles and tests without an OpenCL runtime.  The embedded OpenCL C kernel
//! source ([`MASK_GENERATION_CL`]) is provided for future GPU dispatch.

use std::fmt;

// ---------------------------------------------------------------------------
// Mask format
// ---------------------------------------------------------------------------

/// Representation format for attention masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaskFormat {
    /// `true` = attend, `false` = masked.
    Bool,
    /// `0.0` = attend, `-inf` = masked.  Added to raw scores before softmax.
    Float,
    /// `0.0` = attend, large negative value = masked (same semantics as Float
    /// but uses a finite sentinel instead of `-inf`).
    Additive,
    /// `1.0` = attend, `0.0` = masked.  Multiplied with scores.
    Multiplicative,
}

impl fmt::Display for MaskFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool => write!(f, "Bool"),
            Self::Float => write!(f, "Float"),
            Self::Additive => write!(f, "Additive"),
            Self::Multiplicative => write!(f, "Multiplicative"),
        }
    }
}

// ---------------------------------------------------------------------------
// Mask data — the common mask payload
// ---------------------------------------------------------------------------

/// A 2-D attention mask stored row-major as `[seq_len, kv_len]` booleans.
///
/// `true` at position `(i, j)` means query position `i` **may** attend to key
/// position `j`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskData {
    data: Vec<bool>,
    pub seq_len: usize,
    pub kv_len: usize,
}

impl MaskData {
    /// Create a mask from raw data.  Panics if `data.len() != seq_len * kv_len`.
    pub fn new(data: Vec<bool>, seq_len: usize, kv_len: usize) -> Self {
        assert_eq!(data.len(), seq_len * kv_len, "mask size mismatch");
        Self { data, seq_len, kv_len }
    }

    /// Create a fully-permissive mask (all `true`).
    pub fn all_true(seq_len: usize, kv_len: usize) -> Self {
        Self { data: vec![true; seq_len * kv_len], seq_len, kv_len }
    }

    /// Create a fully-masked mask (all `false`).
    pub fn all_false(seq_len: usize, kv_len: usize) -> Self {
        Self { data: vec![false; seq_len * kv_len], seq_len, kv_len }
    }

    /// Check whether query `i` may attend to key `j`.
    #[inline]
    pub fn allows(&self, i: usize, j: usize) -> bool {
        self.data[i * self.kv_len + j]
    }

    /// Set mask value at `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: bool) {
        self.data[i * self.kv_len + j] = value;
    }

    /// Raw boolean slice.
    pub fn as_slice(&self) -> &[bool] {
        &self.data
    }

    /// Count the number of `true` entries.
    pub fn count_true(&self) -> usize {
        self.data.iter().filter(|&&v| v).count()
    }

    /// Count the number of `false` entries.
    pub fn count_false(&self) -> usize {
        self.data.iter().filter(|&&v| !v).count()
    }

    /// Total number of entries.
    pub fn total(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// CausalMask
// ---------------------------------------------------------------------------

/// Lower-triangular causal mask for autoregressive decoding.
///
/// For a sequence of length `n`, position `i` may attend to positions
/// `0..=i` (with optional offset for KV-cache prefill).
#[derive(Debug, Clone)]
pub struct CausalMask;

impl CausalMask {
    /// Generate a causal mask of shape `[seq_len, kv_len]`.
    ///
    /// `offset` accounts for KV-cache positions preceding the current query
    /// window: query position `i` corresponds to absolute position `offset + i`.
    pub fn generate(seq_len: usize, kv_len: usize, offset: usize) -> MaskData {
        let mut data = vec![false; seq_len * kv_len];
        for i in 0..seq_len {
            let query_pos = offset + i;
            for j in 0..kv_len {
                data[i * kv_len + j] = j <= query_pos;
            }
        }
        MaskData::new(data, seq_len, kv_len)
    }
}

// ---------------------------------------------------------------------------
// PaddingMask
// ---------------------------------------------------------------------------

/// Padding mask for batched sequences of variable length.
///
/// Each sequence in the batch has a true length; positions beyond that length
/// are masked out in both query and key dimensions.
#[derive(Debug, Clone)]
pub struct PaddingMask;

impl PaddingMask {
    /// Generate a padding mask for a single sequence.
    ///
    /// Positions `0..actual_len` are valid; the rest are masked.
    pub fn generate(seq_len: usize, kv_len: usize, actual_len: usize) -> MaskData {
        let mut data = vec![false; seq_len * kv_len];
        let q_end = actual_len.min(seq_len);
        let k_end = actual_len.min(kv_len);
        for i in 0..q_end {
            for j in 0..k_end {
                data[i * kv_len + j] = true;
            }
        }
        MaskData::new(data, seq_len, kv_len)
    }

    /// Generate padding masks for a whole batch, returning one [`MaskData`] per
    /// sequence.
    pub fn generate_batch(seq_len: usize, kv_len: usize, lengths: &[usize]) -> Vec<MaskData> {
        lengths.iter().map(|&len| Self::generate(seq_len, kv_len, len)).collect()
    }
}

// ---------------------------------------------------------------------------
// SlidingWindowMask
// ---------------------------------------------------------------------------

/// Sliding-window mask that restricts attention to a local neighbourhood.
///
/// Position `i` may attend to positions `[max(0, i - window + 1) ..= i]`.
#[derive(Debug, Clone)]
pub struct SlidingWindowMask;

impl SlidingWindowMask {
    /// Generate a sliding-window mask with the given `window_size`.
    ///
    /// If `window_size >= kv_len` the mask is fully causal.
    pub fn generate(seq_len: usize, kv_len: usize, window_size: usize) -> MaskData {
        let mut data = vec![false; seq_len * kv_len];
        for i in 0..seq_len {
            let start = if i >= window_size { i - window_size + 1 } else { 0 };
            for j in start..=i.min(kv_len - 1) {
                data[i * kv_len + j] = true;
            }
        }
        MaskData::new(data, seq_len, kv_len)
    }

    /// Generate with an offset (for incremental decoding with KV cache).
    pub fn generate_with_offset(
        seq_len: usize,
        kv_len: usize,
        window_size: usize,
        offset: usize,
    ) -> MaskData {
        let mut data = vec![false; seq_len * kv_len];
        for i in 0..seq_len {
            let abs_pos = offset + i;
            let start = abs_pos.saturating_sub(window_size - 1);
            for j in 0..kv_len {
                data[i * kv_len + j] = j >= start && j <= abs_pos;
            }
        }
        MaskData::new(data, seq_len, kv_len)
    }
}

// ---------------------------------------------------------------------------
// PrefixMask
// ---------------------------------------------------------------------------

/// Prefix mask: full (bidirectional) attention within a prefix region, causal
/// attention for the remaining positions.
///
/// Used in prefix-LM architectures where the prompt tokens attend to each
/// other freely, while generated tokens are autoregressively masked.
#[derive(Debug, Clone)]
pub struct PrefixMask;

impl PrefixMask {
    /// Generate a prefix mask.
    ///
    /// - Positions `0..prefix_len` may attend to all positions `0..prefix_len`.
    /// - Positions `prefix_len..seq_len` may attend to `0..=i` (causal).
    pub fn generate(seq_len: usize, kv_len: usize, prefix_len: usize) -> MaskData {
        let prefix = prefix_len.min(seq_len).min(kv_len);
        let mut data = vec![false; seq_len * kv_len];
        for i in 0..seq_len {
            if i < prefix {
                // Prefix rows: attend to entire prefix
                for j in 0..prefix.min(kv_len) {
                    data[i * kv_len + j] = true;
                }
            } else {
                // Causal rows: attend to positions 0..=i
                for j in 0..=i.min(kv_len - 1) {
                    data[i * kv_len + j] = true;
                }
            }
        }
        MaskData::new(data, seq_len, kv_len)
    }
}

// ---------------------------------------------------------------------------
// BlockSparseMask
// ---------------------------------------------------------------------------

/// Block-diagonal sparse attention pattern.
///
/// The sequence is divided into non-overlapping blocks of `block_size`.
/// Positions within the same block may attend to each other; cross-block
/// attention is masked.  An optional set of "global" positions may attend to
/// (and be attended from) all positions.
#[derive(Debug, Clone)]
pub struct BlockSparseMask;

impl BlockSparseMask {
    /// Generate a block-sparse mask with uniform `block_size`.
    pub fn generate(seq_len: usize, kv_len: usize, block_size: usize) -> MaskData {
        assert!(block_size > 0, "block_size must be > 0");
        let mut data = vec![false; seq_len * kv_len];
        for i in 0..seq_len {
            let block_i = i / block_size;
            for j in 0..kv_len {
                let block_j = j / block_size;
                data[i * kv_len + j] = block_i == block_j;
            }
        }
        MaskData::new(data, seq_len, kv_len)
    }

    /// Generate a block-sparse mask with global token positions.
    ///
    /// Global tokens attend to all positions and are attended to by all
    /// positions.  Remaining positions use block-diagonal attention.
    pub fn generate_with_global(
        seq_len: usize,
        kv_len: usize,
        block_size: usize,
        global_positions: &[usize],
    ) -> MaskData {
        assert!(block_size > 0, "block_size must be > 0");
        let mut data = vec![false; seq_len * kv_len];
        for i in 0..seq_len {
            let is_global_i = global_positions.contains(&i);
            let block_i = i / block_size;
            for j in 0..kv_len {
                let is_global_j = global_positions.contains(&j);
                let block_j = j / block_size;
                data[i * kv_len + j] = is_global_i || is_global_j || block_i == block_j;
            }
        }
        MaskData::new(data, seq_len, kv_len)
    }
}

// ---------------------------------------------------------------------------
// MaskCombiner
// ---------------------------------------------------------------------------

/// Logical operation for combining two masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CombineOp {
    And,
    Or,
    Xor,
}

impl fmt::Display for CombineOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::And => write!(f, "AND"),
            Self::Or => write!(f, "OR"),
            Self::Xor => write!(f, "XOR"),
        }
    }
}

/// Combines two [`MaskData`] instances element-wise using a logical operation.
#[derive(Debug, Clone)]
pub struct MaskCombiner;

impl MaskCombiner {
    /// Combine `a` and `b` using the given operation.
    ///
    /// # Panics
    ///
    /// Panics if `a` and `b` have different dimensions.
    pub fn combine(a: &MaskData, b: &MaskData, op: CombineOp) -> MaskData {
        assert_eq!(a.seq_len, b.seq_len, "seq_len mismatch");
        assert_eq!(a.kv_len, b.kv_len, "kv_len mismatch");
        let data: Vec<bool> = a
            .as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .map(|(&x, &y)| match op {
                CombineOp::And => x && y,
                CombineOp::Or => x || y,
                CombineOp::Xor => x ^ y,
            })
            .collect();
        MaskData::new(data, a.seq_len, a.kv_len)
    }

    /// Negate a mask (element-wise NOT).
    pub fn negate(mask: &MaskData) -> MaskData {
        let data: Vec<bool> = mask.as_slice().iter().map(|&v| !v).collect();
        MaskData::new(data, mask.seq_len, mask.kv_len)
    }
}

// ---------------------------------------------------------------------------
// MaskExpander
// ---------------------------------------------------------------------------

/// Expands a 2D `[seq_len, kv_len]` mask to a 4D
/// `[batch, heads, seq_len, kv_len]` tensor representation.
///
/// The 4D tensor is stored as a flat `Vec<f32>` in row-major order.
#[derive(Debug, Clone)]
pub struct MaskExpander;

impl MaskExpander {
    /// Expand a single [`MaskData`] to 4D float tensor.
    ///
    /// Returns a flat `Vec<f32>` of length `batch * heads * seq_len * kv_len`.
    /// Uses the given [`MaskFormat`] to determine the fill values.
    pub fn expand(mask: &MaskData, batch: usize, heads: usize, format: MaskFormat) -> Vec<f32> {
        let inner_size = mask.seq_len * mask.kv_len;
        let total = batch * heads * inner_size;
        let mut out = vec![0.0f32; total];

        // Build one 2D plane
        let plane: Vec<f32> =
            mask.as_slice().iter().map(|&v| MaskConverter::bool_to_float(v, format)).collect();

        // Tile across batch × heads
        for bh in 0..(batch * heads) {
            let offset = bh * inner_size;
            out[offset..offset + inner_size].copy_from_slice(&plane);
        }
        out
    }

    /// Expand per-batch masks: one [`MaskData`] per batch element, tiled across
    /// `heads`.
    ///
    /// # Panics
    ///
    /// Panics if `masks.len() != batch`.
    pub fn expand_batch(masks: &[MaskData], heads: usize, format: MaskFormat) -> Vec<f32> {
        let batch = masks.len();
        assert!(batch > 0, "need at least one mask");
        let seq_len = masks[0].seq_len;
        let kv_len = masks[0].kv_len;
        let inner_size = seq_len * kv_len;
        let total = batch * heads * inner_size;
        let mut out = vec![0.0f32; total];

        for (b, mask) in masks.iter().enumerate() {
            assert_eq!(mask.seq_len, seq_len, "all masks must have same seq_len");
            assert_eq!(mask.kv_len, kv_len, "all masks must have same kv_len");
            let plane: Vec<f32> =
                mask.as_slice().iter().map(|&v| MaskConverter::bool_to_float(v, format)).collect();
            for h in 0..heads {
                let offset = (b * heads + h) * inner_size;
                out[offset..offset + inner_size].copy_from_slice(&plane);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// MaskConverter
// ---------------------------------------------------------------------------

/// Sentinel value used for `MaskFormat::Additive` (large negative, but finite).
pub const ADDITIVE_MASK_VALUE: f32 = -1e9;

/// Converts masks between different [`MaskFormat`] representations.
#[derive(Debug, Clone)]
pub struct MaskConverter;

impl MaskConverter {
    /// Convert a single boolean to the target format's float value.
    #[inline]
    pub fn bool_to_float(value: bool, format: MaskFormat) -> f32 {
        match format {
            MaskFormat::Bool => {
                if value {
                    1.0
                } else {
                    0.0
                }
            }
            MaskFormat::Float => {
                if value {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            }
            MaskFormat::Additive => {
                if value {
                    0.0
                } else {
                    ADDITIVE_MASK_VALUE
                }
            }
            MaskFormat::Multiplicative => {
                if value {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Convert a float value back to boolean using the given source format.
    #[inline]
    pub fn float_to_bool(value: f32, format: MaskFormat) -> bool {
        match format {
            MaskFormat::Bool => value > 0.5,
            MaskFormat::Float => value > f32::NEG_INFINITY,
            MaskFormat::Additive => value > ADDITIVE_MASK_VALUE / 2.0,
            MaskFormat::Multiplicative => value > 0.5,
        }
    }

    /// Convert a boolean [`MaskData`] to a flat float vector in the target
    /// format.
    pub fn to_float(mask: &MaskData, format: MaskFormat) -> Vec<f32> {
        mask.as_slice().iter().map(|&v| Self::bool_to_float(v, format)).collect()
    }

    /// Convert a flat float vector back to [`MaskData`] using the source
    /// format.
    pub fn from_float(
        values: &[f32],
        seq_len: usize,
        kv_len: usize,
        format: MaskFormat,
    ) -> MaskData {
        let data: Vec<bool> = values.iter().map(|&v| Self::float_to_bool(v, format)).collect();
        MaskData::new(data, seq_len, kv_len)
    }

    /// Convert between two float formats.
    pub fn convert(values: &[f32], from: MaskFormat, to: MaskFormat) -> Vec<f32> {
        values
            .iter()
            .map(|&v| {
                let b = Self::float_to_bool(v, from);
                Self::bool_to_float(b, to)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// MaskStats
// ---------------------------------------------------------------------------

/// Statistics about an attention mask's sparsity and memory characteristics.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskStats {
    /// Number of unmasked (true) entries.
    pub active_entries: usize,
    /// Total entries in the mask.
    pub total_entries: usize,
    /// Fraction of entries that are masked (false).
    pub sparsity_ratio: f64,
    /// Estimated memory usage for the dense bool mask (bytes).
    pub dense_memory_bytes: usize,
    /// Estimated memory savings if a sparse representation were used.
    pub estimated_sparse_memory_bytes: usize,
    /// Savings ratio: `1.0 - sparse / dense`.
    pub savings_ratio: f64,
}

impl MaskStats {
    /// Compute statistics for a mask.
    pub fn compute(mask: &MaskData) -> Self {
        let total = mask.total();
        let active = mask.count_true();
        let sparsity = if total == 0 { 0.0 } else { 1.0 - (active as f64 / total as f64) };

        // Dense: 1 byte per bool (Rust Vec<bool>)
        let dense_memory = total;
        // Sparse estimate: store (row, col) as u32 pairs for each active entry
        let sparse_memory = active * 8; // 2 × u32 per entry
        let savings = if dense_memory == 0 {
            0.0
        } else {
            1.0 - (sparse_memory as f64 / dense_memory as f64)
        };

        Self {
            active_entries: active,
            total_entries: total,
            sparsity_ratio: sparsity,
            dense_memory_bytes: dense_memory,
            estimated_sparse_memory_bytes: sparse_memory,
            savings_ratio: savings,
        }
    }
}

impl fmt::Display for MaskStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MaskStats {{ active: {}/{}, sparsity: {:.1}%, savings: {:.1}% }}",
            self.active_entries,
            self.total_entries,
            self.sparsity_ratio * 100.0,
            self.savings_ratio * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source (embedded, for future GPU dispatch)
// ---------------------------------------------------------------------------

/// OpenCL C kernel source for mask generation.
///
/// Contains kernels for generating causal, padding, sliding-window, prefix,
/// and block-sparse masks directly on the GPU.
pub const MASK_GENERATION_CL: &str = r#"
// --- Causal mask kernel ---
__kernel void generate_causal_mask(
    __global uchar* mask,     // output: [seq_len, kv_len]
    const uint seq_len,
    const uint kv_len,
    const uint offset
) {
    uint i = get_global_id(0); // query position
    uint j = get_global_id(1); // key position
    if (i >= seq_len || j >= kv_len) return;
    uint query_pos = offset + i;
    mask[i * kv_len + j] = (j <= query_pos) ? 1 : 0;
}

// --- Padding mask kernel ---
__kernel void generate_padding_mask(
    __global uchar* mask,     // output: [seq_len, kv_len]
    const uint seq_len,
    const uint kv_len,
    const uint actual_len
) {
    uint i = get_global_id(0);
    uint j = get_global_id(1);
    if (i >= seq_len || j >= kv_len) return;
    mask[i * kv_len + j] = (i < actual_len && j < actual_len) ? 1 : 0;
}

// --- Sliding window mask kernel ---
__kernel void generate_sliding_window_mask(
    __global uchar* mask,
    const uint seq_len,
    const uint kv_len,
    const uint window_size,
    const uint offset
) {
    uint i = get_global_id(0);
    uint j = get_global_id(1);
    if (i >= seq_len || j >= kv_len) return;
    uint abs_pos = offset + i;
    uint start = (abs_pos >= window_size - 1) ? (abs_pos - window_size + 1) : 0;
    mask[i * kv_len + j] = (j >= start && j <= abs_pos) ? 1 : 0;
}

// --- Prefix mask kernel ---
__kernel void generate_prefix_mask(
    __global uchar* mask,
    const uint seq_len,
    const uint kv_len,
    const uint prefix_len
) {
    uint i = get_global_id(0);
    uint j = get_global_id(1);
    if (i >= seq_len || j >= kv_len) return;
    uint plen = min(prefix_len, min(seq_len, kv_len));
    if (i < plen) {
        mask[i * kv_len + j] = (j < plen) ? 1 : 0;
    } else {
        mask[i * kv_len + j] = (j <= i) ? 1 : 0;
    }
}

// --- Block-sparse mask kernel ---
__kernel void generate_block_sparse_mask(
    __global uchar* mask,
    const uint seq_len,
    const uint kv_len,
    const uint block_size
) {
    uint i = get_global_id(0);
    uint j = get_global_id(1);
    if (i >= seq_len || j >= kv_len) return;
    mask[i * kv_len + j] = (i / block_size == j / block_size) ? 1 : 0;
}

// --- Bool-to-float mask conversion kernel ---
__kernel void convert_mask_to_float(
    __global const uchar* bool_mask,
    __global float* float_mask,
    const uint total,
    const float true_val,
    const float false_val
) {
    uint idx = get_global_id(0);
    if (idx >= total) return;
    float_mask[idx] = (bool_mask[idx] != 0) ? true_val : false_val;
}

// --- 4D mask expansion kernel ---
// Tiles a 2D [seq_len, kv_len] float mask across [batch, heads].
__kernel void expand_mask_4d(
    __global const float* mask_2d,  // [seq_len, kv_len]
    __global float* mask_4d,        // [batch, heads, seq_len, kv_len]
    const uint inner_size,          // seq_len * kv_len
    const uint heads,
    const uint batch
) {
    uint bh = get_global_id(0);     // batch * heads index
    uint idx = get_global_id(1);    // position within 2D mask
    if (bh >= batch * heads || idx >= inner_size) return;
    mask_4d[bh * inner_size + idx] = mask_2d[idx];
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ===================================================================
    // CausalMask tests
    // ===================================================================

    #[test]
    fn causal_mask_4x4() {
        let m = CausalMask::generate(4, 4, 0);
        // Row 0: [T F F F]
        // Row 1: [T T F F]
        // Row 2: [T T T F]
        // Row 3: [T T T T]
        assert!(m.allows(0, 0));
        assert!(!m.allows(0, 1));
        assert!(m.allows(1, 0));
        assert!(m.allows(1, 1));
        assert!(!m.allows(1, 2));
        assert!(m.allows(3, 3));
        assert!(m.allows(3, 0));
    }

    #[test]
    fn causal_mask_is_lower_triangular() {
        let n = 8;
        let m = CausalMask::generate(n, n, 0);
        for i in 0..n {
            for j in 0..n {
                if j <= i {
                    assert!(m.allows(i, j), "should allow ({i},{j})");
                } else {
                    assert!(!m.allows(i, j), "should mask ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn causal_mask_entry_count() {
        // A causal mask of size n×n has exactly n*(n+1)/2 true entries.
        for n in 1..=16 {
            let m = CausalMask::generate(n, n, 0);
            assert_eq!(m.count_true(), n * (n + 1) / 2, "n={n}");
        }
    }

    #[test]
    fn causal_mask_with_offset() {
        // offset=3, seq_len=2, kv_len=6
        // query 0 → abs pos 3 → attend to 0..=3
        // query 1 → abs pos 4 → attend to 0..=4
        let m = CausalMask::generate(2, 6, 3);
        assert!(m.allows(0, 0));
        assert!(m.allows(0, 3));
        assert!(!m.allows(0, 4));
        assert!(m.allows(1, 4));
        assert!(!m.allows(1, 5));
    }

    #[test]
    fn causal_mask_seq_len_1() {
        let m = CausalMask::generate(1, 1, 0);
        assert_eq!(m.count_true(), 1);
        assert!(m.allows(0, 0));
    }

    #[test]
    fn causal_mask_single_token_with_offset() {
        // Single new token decoding with 5 cached KV positions.
        let m = CausalMask::generate(1, 6, 5);
        // query at abs pos 5 can attend to all 6 KV positions.
        for j in 0..6 {
            assert!(m.allows(0, j));
        }
    }

    #[test]
    fn causal_mask_rectangular() {
        // seq_len=3, kv_len=5, offset=0
        let m = CausalMask::generate(3, 5, 0);
        assert!(m.allows(0, 0));
        assert!(!m.allows(0, 1));
        assert!(m.allows(2, 2));
        assert!(!m.allows(2, 3));
    }

    #[test]
    fn causal_mask_large() {
        let n = 128;
        let m = CausalMask::generate(n, n, 0);
        assert_eq!(m.count_true(), n * (n + 1) / 2);
    }

    // ===================================================================
    // PaddingMask tests
    // ===================================================================

    #[test]
    fn padding_mask_basic() {
        let m = PaddingMask::generate(4, 4, 2);
        // Only the top-left 2×2 block is true.
        assert!(m.allows(0, 0));
        assert!(m.allows(0, 1));
        assert!(!m.allows(0, 2));
        assert!(m.allows(1, 0));
        assert!(m.allows(1, 1));
        assert!(!m.allows(2, 0));
    }

    #[test]
    fn padding_mask_full_length() {
        let m = PaddingMask::generate(4, 4, 4);
        assert_eq!(m.count_true(), 16);
    }

    #[test]
    fn padding_mask_zero_length() {
        let m = PaddingMask::generate(4, 4, 0);
        assert_eq!(m.count_true(), 0);
    }

    #[test]
    fn padding_mask_exceeds_seq_len() {
        let m = PaddingMask::generate(3, 3, 100);
        // Clamped to the actual dimensions.
        assert_eq!(m.count_true(), 9);
    }

    #[test]
    fn padding_mask_batch() {
        let masks = PaddingMask::generate_batch(4, 4, &[1, 3, 4]);
        assert_eq!(masks.len(), 3);
        assert_eq!(masks[0].count_true(), 1); // 1×1 block
        assert_eq!(masks[1].count_true(), 9); // 3×3 block
        assert_eq!(masks[2].count_true(), 16); // full
    }

    #[test]
    fn padding_mask_rectangular() {
        // seq_len=3, kv_len=5, actual_len=2
        let m = PaddingMask::generate(3, 5, 2);
        assert!(m.allows(0, 0));
        assert!(m.allows(0, 1));
        assert!(!m.allows(0, 2));
        assert!(m.allows(1, 0));
        assert!(!m.allows(2, 0));
    }

    #[test]
    fn padding_mask_seq_len_1() {
        let m = PaddingMask::generate(1, 1, 1);
        assert_eq!(m.count_true(), 1);
    }

    // ===================================================================
    // SlidingWindowMask tests
    // ===================================================================

    #[test]
    fn sliding_window_mask_basic() {
        // window=2, 4×4
        let m = SlidingWindowMask::generate(4, 4, 2);
        // Row 0: attend [0]         (window covers 0..=0)
        // Row 1: attend [0,1]       (window covers 0..=1)
        // Row 2: attend [1,2]       (window covers 1..=2)
        // Row 3: attend [2,3]       (window covers 2..=3)
        assert!(m.allows(0, 0));
        assert!(!m.allows(0, 1));
        assert!(m.allows(1, 0));
        assert!(m.allows(1, 1));
        assert!(!m.allows(2, 0));
        assert!(m.allows(2, 1));
        assert!(m.allows(2, 2));
        assert!(m.allows(3, 2));
        assert!(m.allows(3, 3));
        assert!(!m.allows(3, 0));
    }

    #[test]
    fn sliding_window_equals_causal_when_large() {
        // When window_size >= seq_len, sliding window degenerates to causal.
        let n = 6;
        let causal = CausalMask::generate(n, n, 0);
        let sw = SlidingWindowMask::generate(n, n, n);
        assert_eq!(causal.as_slice(), sw.as_slice());
    }

    #[test]
    fn sliding_window_mask_window_1() {
        // Window=1: only attend to self.
        let m = SlidingWindowMask::generate(4, 4, 1);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m.allows(i, j), i == j, "({i},{j})");
            }
        }
    }

    #[test]
    fn sliding_window_active_count() {
        // Window of size w on an n×n grid with w <= n:
        // Row 0..w-1 have 1..w true entries, rows w-1..n-1 have w entries.
        let n = 8;
        let w = 3;
        let m = SlidingWindowMask::generate(n, n, w);
        let expected: usize = (1..=w.min(n)).sum::<usize>() + (n.saturating_sub(w)) * w;
        assert_eq!(m.count_true(), expected);
    }

    #[test]
    fn sliding_window_seq_len_1() {
        let m = SlidingWindowMask::generate(1, 1, 4);
        assert_eq!(m.count_true(), 1);
    }

    #[test]
    fn sliding_window_with_offset() {
        // offset=2, seq_len=2, kv_len=4, window=2
        // query 0 → abs pos 2 → window [1,2]
        // query 1 → abs pos 3 → window [2,3]
        let m = SlidingWindowMask::generate_with_offset(2, 4, 2, 2);
        assert!(!m.allows(0, 0));
        assert!(m.allows(0, 1));
        assert!(m.allows(0, 2));
        assert!(!m.allows(0, 3));
        assert!(!m.allows(1, 0));
        assert!(!m.allows(1, 1));
        assert!(m.allows(1, 2));
        assert!(m.allows(1, 3));
    }

    #[test]
    fn sliding_window_window_exceeds_seq_len() {
        // Window larger than seq_len → acts like causal
        let m = SlidingWindowMask::generate(3, 3, 100);
        let causal = CausalMask::generate(3, 3, 0);
        assert_eq!(m.as_slice(), causal.as_slice());
    }

    // ===================================================================
    // PrefixMask tests
    // ===================================================================

    #[test]
    fn prefix_mask_basic() {
        // prefix=2, 4×4
        let m = PrefixMask::generate(4, 4, 2);
        // Row 0 (prefix): [T T F F]
        // Row 1 (prefix): [T T F F]
        // Row 2 (causal): [T T T F]
        // Row 3 (causal): [T T T T]
        assert!(m.allows(0, 0));
        assert!(m.allows(0, 1));
        assert!(!m.allows(0, 2));
        assert!(m.allows(1, 0));
        assert!(m.allows(1, 1));
        assert!(!m.allows(1, 2));
        assert!(m.allows(2, 0));
        assert!(m.allows(2, 2));
        assert!(!m.allows(2, 3));
        assert!(m.allows(3, 3));
    }

    #[test]
    fn prefix_mask_zero_prefix() {
        // prefix=0 → pure causal
        let n = 4;
        let prefix = PrefixMask::generate(n, n, 0);
        let causal = CausalMask::generate(n, n, 0);
        assert_eq!(prefix.as_slice(), causal.as_slice());
    }

    #[test]
    fn prefix_mask_full_prefix() {
        // prefix=n → full attention
        let n = 4;
        let m = PrefixMask::generate(n, n, n);
        // All rows are prefix rows attending to 0..n
        assert_eq!(m.count_true(), n * n);
    }

    #[test]
    fn prefix_mask_prefix_exceeds_seq_len() {
        let m = PrefixMask::generate(3, 3, 100);
        assert_eq!(m.count_true(), 9);
    }

    #[test]
    fn prefix_mask_seq_len_1() {
        let m = PrefixMask::generate(1, 1, 1);
        assert_eq!(m.count_true(), 1);
    }

    #[test]
    fn prefix_mask_single_prefix_token() {
        let m = PrefixMask::generate(4, 4, 1);
        // Row 0 (prefix): [T F F F]
        // Row 1 (causal): [T T F F]
        // Row 2 (causal): [T T T F]
        // Row 3 (causal): [T T T T]
        // Same as causal:
        let causal = CausalMask::generate(4, 4, 0);
        assert_eq!(m.as_slice(), causal.as_slice());
    }

    // ===================================================================
    // BlockSparseMask tests
    // ===================================================================

    #[test]
    fn block_sparse_basic() {
        // block_size=2, 4×4 → two 2×2 blocks on diagonal
        let m = BlockSparseMask::generate(4, 4, 2);
        // Block 0: rows 0-1, cols 0-1
        assert!(m.allows(0, 0));
        assert!(m.allows(0, 1));
        assert!(m.allows(1, 0));
        assert!(m.allows(1, 1));
        // Block 1: rows 2-3, cols 2-3
        assert!(m.allows(2, 2));
        assert!(m.allows(2, 3));
        assert!(m.allows(3, 2));
        assert!(m.allows(3, 3));
        // Cross-block
        assert!(!m.allows(0, 2));
        assert!(!m.allows(2, 0));
    }

    #[test]
    fn block_sparse_entry_count() {
        // n=6, block_size=3 → 2 blocks of 3×3 = 18 entries
        let m = BlockSparseMask::generate(6, 6, 3);
        assert_eq!(m.count_true(), 18);
    }

    #[test]
    fn block_sparse_single_block() {
        // block_size >= seq_len → entire matrix is one block (all true)
        let m = BlockSparseMask::generate(4, 4, 4);
        assert_eq!(m.count_true(), 16);
    }

    #[test]
    fn block_sparse_block_size_1() {
        // block_size=1 → diagonal only
        let n = 5;
        let m = BlockSparseMask::generate(n, n, 1);
        assert_eq!(m.count_true(), n);
        for i in 0..n {
            assert!(m.allows(i, i));
        }
    }

    #[test]
    fn block_sparse_with_global_tokens() {
        // 6×6, block_size=3, global=[0]
        let m = BlockSparseMask::generate_with_global(6, 6, 3, &[0]);
        // Position 0 is global: it can attend to all, and all can attend to it.
        for j in 0..6 {
            assert!(m.allows(0, j), "global row 0 should attend to {j}");
        }
        for i in 0..6 {
            assert!(m.allows(i, 0), "all rows should attend to global col 0");
        }
        // Non-global cross-block should be masked.
        assert!(!m.allows(1, 3), "non-global cross-block");
    }

    #[test]
    fn block_sparse_partial_last_block() {
        // n=5, block_size=3 → blocks: [0..2] and [3..4]
        let m = BlockSparseMask::generate(5, 5, 3);
        assert!(m.allows(3, 3));
        assert!(m.allows(3, 4));
        assert!(m.allows(4, 3));
        assert!(m.allows(4, 4));
        assert!(!m.allows(3, 2));
        assert!(!m.allows(2, 3));
    }

    #[test]
    fn block_sparse_rectangular() {
        // seq_len=4, kv_len=6, block_size=2
        let m = BlockSparseMask::generate(4, 6, 2);
        assert!(m.allows(0, 0));
        assert!(m.allows(0, 1));
        assert!(!m.allows(0, 2));
        assert!(m.allows(2, 2));
        assert!(m.allows(2, 3));
        assert!(!m.allows(2, 4));
    }

    // ===================================================================
    // MaskCombiner tests
    // ===================================================================

    #[test]
    fn combine_and() {
        let a = CausalMask::generate(4, 4, 0);
        let b = PaddingMask::generate(4, 4, 3);
        let c = MaskCombiner::combine(&a, &b, CombineOp::And);
        // AND of causal and padding(3): must be both causal AND within length 3.
        assert!(c.allows(0, 0));
        assert!(!c.allows(0, 1)); // causal blocks
        assert!(c.allows(2, 2));
        assert!(!c.allows(3, 3)); // padding blocks row 3
    }

    #[test]
    fn combine_or() {
        let a = MaskData::all_false(4, 4);
        let b = CausalMask::generate(4, 4, 0);
        let c = MaskCombiner::combine(&a, &b, CombineOp::Or);
        // OR with all-false → same as b
        assert_eq!(c.as_slice(), b.as_slice());
    }

    #[test]
    fn combine_xor() {
        let a = MaskData::all_true(2, 2);
        let b = MaskData::all_true(2, 2);
        let c = MaskCombiner::combine(&a, &b, CombineOp::Xor);
        assert_eq!(c.count_true(), 0);
    }

    #[test]
    fn combine_and_identity() {
        // AND with all_true is identity.
        let a = CausalMask::generate(4, 4, 0);
        let b = MaskData::all_true(4, 4);
        let c = MaskCombiner::combine(&a, &b, CombineOp::And);
        assert_eq!(c.as_slice(), a.as_slice());
    }

    #[test]
    fn combine_or_identity() {
        // OR with all_false is identity.
        let a = CausalMask::generate(4, 4, 0);
        let b = MaskData::all_false(4, 4);
        let c = MaskCombiner::combine(&a, &b, CombineOp::Or);
        assert_eq!(c.as_slice(), a.as_slice());
    }

    #[test]
    fn negate_mask() {
        let m = CausalMask::generate(3, 3, 0);
        let neg = MaskCombiner::negate(&m);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(neg.allows(i, j), !m.allows(i, j));
            }
        }
    }

    #[test]
    fn negate_all_true() {
        let m = MaskData::all_true(4, 4);
        let neg = MaskCombiner::negate(&m);
        assert_eq!(neg.count_true(), 0);
    }

    #[test]
    fn double_negate_roundtrip() {
        let m = CausalMask::generate(5, 5, 0);
        let double_neg = MaskCombiner::negate(&MaskCombiner::negate(&m));
        assert_eq!(double_neg.as_slice(), m.as_slice());
    }

    #[test]
    fn combine_causal_and_sliding_window() {
        // AND of causal + sliding window → just sliding window (it's stricter).
        let n = 6;
        let w = 3;
        let causal = CausalMask::generate(n, n, 0);
        let sw = SlidingWindowMask::generate(n, n, w);
        let combined = MaskCombiner::combine(&causal, &sw, CombineOp::And);
        assert_eq!(combined.as_slice(), sw.as_slice());
    }

    // ===================================================================
    // MaskExpander tests
    // ===================================================================

    #[test]
    fn expand_single_mask_bool() {
        let m = CausalMask::generate(2, 2, 0);
        let expanded = MaskExpander::expand(&m, 1, 1, MaskFormat::Bool);
        assert_eq!(expanded.len(), 4);
        assert_eq!(expanded[0], 1.0); // (0,0) true
        assert_eq!(expanded[1], 0.0); // (0,1) false
        assert_eq!(expanded[2], 1.0); // (1,0) true
        assert_eq!(expanded[3], 1.0); // (1,1) true
    }

    #[test]
    fn expand_tiling_batch_heads() {
        let m = CausalMask::generate(2, 2, 0);
        let expanded = MaskExpander::expand(&m, 2, 3, MaskFormat::Multiplicative);
        // 2 batches × 3 heads × 4 entries = 24
        assert_eq!(expanded.len(), 24);
        // All planes should be identical.
        for bh in 0..6 {
            let offset = bh * 4;
            assert_eq!(expanded[offset], 1.0);
            assert_eq!(expanded[offset + 1], 0.0);
            assert_eq!(expanded[offset + 2], 1.0);
            assert_eq!(expanded[offset + 3], 1.0);
        }
    }

    #[test]
    fn expand_float_format() {
        let m = CausalMask::generate(2, 2, 0);
        let expanded = MaskExpander::expand(&m, 1, 1, MaskFormat::Float);
        assert_eq!(expanded[0], 0.0);
        assert!(expanded[1].is_infinite() && expanded[1] < 0.0);
    }

    #[test]
    fn expand_additive_format() {
        let m = CausalMask::generate(2, 2, 0);
        let expanded = MaskExpander::expand(&m, 1, 1, MaskFormat::Additive);
        assert_eq!(expanded[0], 0.0);
        assert_eq!(expanded[1], ADDITIVE_MASK_VALUE);
    }

    #[test]
    fn expand_batch_per_sequence() {
        let masks = PaddingMask::generate_batch(3, 3, &[1, 3]);
        let expanded = MaskExpander::expand_batch(&masks, 2, MaskFormat::Bool);
        // 2 batches × 2 heads × 9 entries = 36
        assert_eq!(expanded.len(), 36);
        // First batch (len=1): only (0,0) is 1.0
        assert_eq!(expanded[0], 1.0);
        assert_eq!(expanded[1], 0.0);
    }

    #[test]
    fn expand_empty_head_1x1() {
        let m = CausalMask::generate(1, 1, 0);
        let expanded = MaskExpander::expand(&m, 1, 1, MaskFormat::Multiplicative);
        assert_eq!(expanded, vec![1.0]);
    }

    // ===================================================================
    // MaskConverter tests
    // ===================================================================

    #[test]
    fn convert_bool_to_float_roundtrip() {
        let m = CausalMask::generate(4, 4, 0);
        let floats = MaskConverter::to_float(&m, MaskFormat::Float);
        let back = MaskConverter::from_float(&floats, 4, 4, MaskFormat::Float);
        assert_eq!(back.as_slice(), m.as_slice());
    }

    #[test]
    fn convert_bool_to_additive_roundtrip() {
        let m = CausalMask::generate(4, 4, 0);
        let additive = MaskConverter::to_float(&m, MaskFormat::Additive);
        let back = MaskConverter::from_float(&additive, 4, 4, MaskFormat::Additive);
        assert_eq!(back.as_slice(), m.as_slice());
    }

    #[test]
    fn convert_bool_to_multiplicative_roundtrip() {
        let m = CausalMask::generate(4, 4, 0);
        let mult = MaskConverter::to_float(&m, MaskFormat::Multiplicative);
        let back = MaskConverter::from_float(&mult, 4, 4, MaskFormat::Multiplicative);
        assert_eq!(back.as_slice(), m.as_slice());
    }

    #[test]
    fn convert_float_to_additive() {
        let m = CausalMask::generate(3, 3, 0);
        let float_vals = MaskConverter::to_float(&m, MaskFormat::Float);
        let additive = MaskConverter::convert(&float_vals, MaskFormat::Float, MaskFormat::Additive);
        for (f, a) in float_vals.iter().zip(additive.iter()) {
            if *f == 0.0 {
                assert_eq!(*a, 0.0);
            } else {
                assert_eq!(*a, ADDITIVE_MASK_VALUE);
            }
        }
    }

    #[test]
    fn convert_multiplicative_to_bool() {
        let m = CausalMask::generate(3, 3, 0);
        let mult = MaskConverter::to_float(&m, MaskFormat::Multiplicative);
        let bool_vals = MaskConverter::convert(&mult, MaskFormat::Multiplicative, MaskFormat::Bool);
        // In Bool format: true → 1.0, false → 0.0 (same as Multiplicative)
        assert_eq!(mult, bool_vals);
    }

    #[test]
    fn convert_values_true() {
        assert_eq!(MaskConverter::bool_to_float(true, MaskFormat::Bool), 1.0);
        assert_eq!(MaskConverter::bool_to_float(true, MaskFormat::Float), 0.0);
        assert_eq!(MaskConverter::bool_to_float(true, MaskFormat::Additive), 0.0);
        assert_eq!(MaskConverter::bool_to_float(true, MaskFormat::Multiplicative), 1.0);
    }

    #[test]
    fn convert_values_false() {
        assert_eq!(MaskConverter::bool_to_float(false, MaskFormat::Bool), 0.0);
        assert_eq!(MaskConverter::bool_to_float(false, MaskFormat::Float), f32::NEG_INFINITY);
        assert_eq!(MaskConverter::bool_to_float(false, MaskFormat::Additive), ADDITIVE_MASK_VALUE);
        assert_eq!(MaskConverter::bool_to_float(false, MaskFormat::Multiplicative), 0.0);
    }

    #[test]
    fn float_to_bool_threshold() {
        assert!(MaskConverter::float_to_bool(0.0, MaskFormat::Float));
        assert!(!MaskConverter::float_to_bool(f32::NEG_INFINITY, MaskFormat::Float));
        assert!(MaskConverter::float_to_bool(0.0, MaskFormat::Additive));
        assert!(!MaskConverter::float_to_bool(ADDITIVE_MASK_VALUE, MaskFormat::Additive));
        assert!(MaskConverter::float_to_bool(1.0, MaskFormat::Multiplicative));
        assert!(!MaskConverter::float_to_bool(0.0, MaskFormat::Multiplicative));
    }

    // ===================================================================
    // MaskStats tests
    // ===================================================================

    #[test]
    fn stats_causal_mask() {
        let m = CausalMask::generate(8, 8, 0);
        let stats = MaskStats::compute(&m);
        assert_eq!(stats.active_entries, 36); // 8*9/2
        assert_eq!(stats.total_entries, 64);
        let expected_sparsity = 1.0 - 36.0 / 64.0;
        assert!((stats.sparsity_ratio - expected_sparsity).abs() < 1e-10);
    }

    #[test]
    fn stats_full_mask() {
        let m = MaskData::all_true(4, 4);
        let stats = MaskStats::compute(&m);
        assert_eq!(stats.sparsity_ratio, 0.0);
        assert_eq!(stats.active_entries, 16);
    }

    #[test]
    fn stats_empty_mask() {
        let m = MaskData::all_false(4, 4);
        let stats = MaskStats::compute(&m);
        assert_eq!(stats.sparsity_ratio, 1.0);
        assert_eq!(stats.active_entries, 0);
    }

    #[test]
    fn stats_display() {
        let m = CausalMask::generate(4, 4, 0);
        let stats = MaskStats::compute(&m);
        let s = format!("{stats}");
        assert!(s.contains("active: 10/16"));
        assert!(s.contains("sparsity:"));
    }

    #[test]
    fn stats_dense_memory() {
        let m = CausalMask::generate(8, 8, 0);
        let stats = MaskStats::compute(&m);
        assert_eq!(stats.dense_memory_bytes, 64);
    }

    #[test]
    fn stats_sparse_savings_positive_for_sparse() {
        // A very sparse mask should have positive savings.
        let m = BlockSparseMask::generate(64, 64, 4);
        let stats = MaskStats::compute(&m);
        assert!(stats.savings_ratio > 0.0, "savings should be positive");
    }

    // ===================================================================
    // MaskData utility tests
    // ===================================================================

    #[test]
    fn mask_data_all_true() {
        let m = MaskData::all_true(3, 3);
        assert_eq!(m.count_true(), 9);
        assert_eq!(m.count_false(), 0);
    }

    #[test]
    fn mask_data_all_false() {
        let m = MaskData::all_false(3, 3);
        assert_eq!(m.count_true(), 0);
        assert_eq!(m.count_false(), 9);
    }

    #[test]
    fn mask_data_set() {
        let mut m = MaskData::all_false(2, 2);
        m.set(0, 0, true);
        m.set(1, 1, true);
        assert!(m.allows(0, 0));
        assert!(!m.allows(0, 1));
        assert!(!m.allows(1, 0));
        assert!(m.allows(1, 1));
    }

    #[test]
    fn mask_data_total() {
        let m = MaskData::all_true(5, 7);
        assert_eq!(m.total(), 35);
    }

    // ===================================================================
    // MaskFormat tests
    // ===================================================================

    #[test]
    fn mask_format_display() {
        assert_eq!(format!("{}", MaskFormat::Bool), "Bool");
        assert_eq!(format!("{}", MaskFormat::Float), "Float");
        assert_eq!(format!("{}", MaskFormat::Additive), "Additive");
        assert_eq!(format!("{}", MaskFormat::Multiplicative), "Multiplicative");
    }

    #[test]
    fn combine_op_display() {
        assert_eq!(format!("{}", CombineOp::And), "AND");
        assert_eq!(format!("{}", CombineOp::Or), "OR");
        assert_eq!(format!("{}", CombineOp::Xor), "XOR");
    }

    // ===================================================================
    // OpenCL kernel source tests
    // ===================================================================

    #[test]
    fn kernel_source_not_empty() {
        assert!(!MASK_GENERATION_CL.is_empty());
    }

    #[test]
    fn kernel_source_contains_causal() {
        assert!(MASK_GENERATION_CL.contains("generate_causal_mask"));
    }

    #[test]
    fn kernel_source_contains_padding() {
        assert!(MASK_GENERATION_CL.contains("generate_padding_mask"));
    }

    #[test]
    fn kernel_source_contains_sliding_window() {
        assert!(MASK_GENERATION_CL.contains("generate_sliding_window_mask"));
    }

    #[test]
    fn kernel_source_contains_prefix() {
        assert!(MASK_GENERATION_CL.contains("generate_prefix_mask"));
    }

    #[test]
    fn kernel_source_contains_block_sparse() {
        assert!(MASK_GENERATION_CL.contains("generate_block_sparse_mask"));
    }

    #[test]
    fn kernel_source_contains_convert() {
        assert!(MASK_GENERATION_CL.contains("convert_mask_to_float"));
    }

    #[test]
    fn kernel_source_contains_expand_4d() {
        assert!(MASK_GENERATION_CL.contains("expand_mask_4d"));
    }

    // ===================================================================
    // Property-style tests
    // ===================================================================

    #[test]
    fn property_causal_mask_true_count() {
        for n in 1..=20 {
            let m = CausalMask::generate(n, n, 0);
            assert_eq!(m.count_true(), n * (n + 1) / 2, "n={n}: expected n*(n+1)/2 true entries");
        }
    }

    #[test]
    fn property_causal_diagonal_always_true() {
        for n in 1..=16 {
            let m = CausalMask::generate(n, n, 0);
            for i in 0..n {
                assert!(m.allows(i, i), "diagonal must be true at ({i},{i})");
            }
        }
    }

    #[test]
    fn property_sliding_window_respects_window() {
        for n in 2..=12 {
            for w in 1..=n {
                let m = SlidingWindowMask::generate(n, n, w);
                for i in 0..n {
                    for j in 0..n {
                        if j > i {
                            // Future positions always masked.
                            assert!(!m.allows(i, j), "future pos ({i},{j}) w={w}");
                        } else if i - j >= w {
                            // Outside window: masked.
                            assert!(!m.allows(i, j), "outside window ({i},{j}) w={w}");
                        } else {
                            // Within window: allowed.
                            assert!(m.allows(i, j), "within window ({i},{j}) w={w}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn property_prefix_subsumes_causal() {
        // Prefix mask with prefix=0 equals causal mask.
        for n in 1..=10 {
            let prefix = PrefixMask::generate(n, n, 0);
            let causal = CausalMask::generate(n, n, 0);
            assert_eq!(prefix.as_slice(), causal.as_slice(), "n={n}");
        }
    }

    #[test]
    fn property_block_sparse_symmetric() {
        // Block-sparse mask is symmetric when seq_len == kv_len.
        for n in 1..=12 {
            for bs in 1..=n {
                let m = BlockSparseMask::generate(n, n, bs);
                for i in 0..n {
                    for j in 0..n {
                        assert_eq!(
                            m.allows(i, j),
                            m.allows(j, i),
                            "symmetry at ({i},{j}) n={n} bs={bs}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn property_combine_and_commutative() {
        let a = CausalMask::generate(4, 4, 0);
        let b = SlidingWindowMask::generate(4, 4, 2);
        let ab = MaskCombiner::combine(&a, &b, CombineOp::And);
        let ba = MaskCombiner::combine(&b, &a, CombineOp::And);
        assert_eq!(ab.as_slice(), ba.as_slice());
    }

    #[test]
    fn property_combine_or_commutative() {
        let a = CausalMask::generate(4, 4, 0);
        let b = BlockSparseMask::generate(4, 4, 2);
        let ab = MaskCombiner::combine(&a, &b, CombineOp::Or);
        let ba = MaskCombiner::combine(&b, &a, CombineOp::Or);
        assert_eq!(ab.as_slice(), ba.as_slice());
    }

    #[test]
    fn property_negate_inverts_count() {
        let m = CausalMask::generate(6, 6, 0);
        let neg = MaskCombiner::negate(&m);
        assert_eq!(m.count_true() + neg.count_true(), m.total());
    }

    #[test]
    fn property_stats_consistency() {
        // active + masked = total
        for n in 1..=10 {
            let m = CausalMask::generate(n, n, 0);
            let stats = MaskStats::compute(&m);
            assert_eq!(
                stats.active_entries + (stats.total_entries - stats.active_entries),
                stats.total_entries
            );
        }
    }
}
