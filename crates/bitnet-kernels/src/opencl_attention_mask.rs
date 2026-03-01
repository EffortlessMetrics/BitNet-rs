//! Attention mask generation for transformer inference on OpenCL devices.
//!
//! Provides CPU reference implementations for causal, padding, sliding-window,
//! prefix-causal, and custom mask patterns. A [`MaskGenerator`] with an LRU
//! cache avoids redundant recomputation during autoregressive decoding.
//!
//! # Mask semantics
//!
//! - `1.0` = **unmasked** (attend).
//! - `0.0` = **masked** (do not attend / apply `-inf` penalty).

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Kind of attention mask to generate.
#[derive(Debug, Clone, PartialEq)]
pub enum MaskType {
    /// Standard lower-triangular causal mask.
    Causal,
    /// Masks padding tokens at the end of each sequence in a batch.
    Padding { pad_lengths: Vec<usize> },
    /// Causal mask limited to the last `window_size` tokens.
    SlidingWindow { window_size: usize },
    /// User-supplied boolean mask (`true` = attend).
    Custom { mask: Vec<Vec<bool>> },
    /// Causal mask where the first `prefix_len` tokens are fully visible
    /// to each other.
    CausalWithPrefix { prefix_len: usize },
}

/// Data type used to represent mask values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaskDType {
    Bool,
    Float32,
    Float16,
}

/// Full configuration for mask generation.
#[derive(Debug, Clone)]
pub struct MaskConfig {
    pub seq_len: usize,
    pub batch_size: usize,
    pub num_heads: usize,
    pub mask_type: MaskType,
    pub dtype: MaskDType,
}

impl MaskConfig {
    /// Produce a deterministic cache key.
    fn cache_key(&self) -> String {
        format!(
            "s{}_b{}_h{}_{:?}_d{:?}",
            self.seq_len, self.batch_size, self.num_heads, self.mask_type, self.dtype,
        )
    }
}

/// A generated attention mask stored as flat `f32` data.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionMask {
    /// Flat mask values (`1.0` = attend, `0.0` = mask).
    pub data: Vec<f32>,
    /// Shape: typically `[batch_size, num_heads, seq_len, seq_len]` or
    /// `[seq_len, seq_len]` for single-head masks.
    pub shape: Vec<usize>,
    /// The type of mask that produced this data.
    pub mask_type: MaskType,
    /// Number of masked (0.0) positions.
    pub num_masked: usize,
    /// Number of unmasked (1.0) positions.
    pub num_unmasked: usize,
}

/// Simple LRU-ish cache for generated masks.
#[derive(Debug)]
pub struct MaskCache {
    cached_masks: HashMap<String, AttentionMask>,
    max_entries: usize,
}

impl MaskCache {
    fn new(max_entries: usize) -> Self {
        Self { cached_masks: HashMap::new(), max_entries }
    }

    fn get(&self, key: &str) -> Option<&AttentionMask> {
        self.cached_masks.get(key)
    }

    fn insert(&mut self, key: String, mask: AttentionMask) {
        if self.cached_masks.len() >= self.max_entries {
            // Evict the first key (arbitrary but deterministic).
            if let Some(oldest) = self.cached_masks.keys().next().cloned() {
                self.cached_masks.remove(&oldest);
            }
        }
        self.cached_masks.insert(key, mask);
    }
}

/// Cumulative statistics for a [`MaskGenerator`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MaskStats {
    pub masks_generated: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Cache-aware mask generator.
#[derive(Debug)]
pub struct MaskGenerator {
    cache: MaskCache,
    stats: MaskStats,
}

/// Errors specific to mask generation.
#[derive(Debug, Clone, PartialEq)]
pub enum MaskError {
    /// Sequence length is zero or otherwise invalid.
    InvalidSeqLen,
    /// Sliding-window size exceeds or is incompatible with sequence length.
    InvalidWindowSize { window: usize, seq_len: usize },
    /// Two masks have incompatible shapes.
    ShapeMismatch,
}

impl fmt::Display for MaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSeqLen => write!(f, "invalid sequence length (must be > 0)"),
            Self::InvalidWindowSize { window, seq_len } => {
                write!(f, "window size {window} invalid for seq_len {seq_len}")
            }
            Self::ShapeMismatch => write!(f, "mask shapes do not match"),
        }
    }
}

impl std::error::Error for MaskError {}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Create a new [`MaskGenerator`] with the given cache capacity.
pub fn create_mask_generator(cache_size: usize) -> MaskGenerator {
    MaskGenerator { cache: MaskCache::new(cache_size), stats: MaskStats::default() }
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Generate a lower-triangular causal mask of shape `[seq_len, seq_len]`.
///
/// Position `(i, j)` is `1.0` when `j <= i` (token *i* may attend to token
/// *j*), and `0.0` otherwise.
pub fn cpu_generate_causal_mask(seq_len: usize) -> AttentionMask {
    let mut data = vec![0.0f32; seq_len * seq_len];
    let mut num_unmasked = 0usize;
    for i in 0..seq_len {
        for j in 0..=i {
            data[i * seq_len + j] = 1.0;
            num_unmasked += 1;
        }
    }
    let total = seq_len * seq_len;
    AttentionMask {
        data,
        shape: vec![seq_len, seq_len],
        mask_type: MaskType::Causal,
        num_masked: total - num_unmasked,
        num_unmasked,
    }
}

/// Generate a padding mask of shape `[batch_size, seq_len]`.
///
/// For each batch element, the last `pad_lengths[b]` positions are masked.
pub fn cpu_generate_padding_mask(
    seq_len: usize,
    pad_lengths: &[usize],
    batch_size: usize,
) -> AttentionMask {
    let mut data = vec![1.0f32; batch_size * seq_len];
    let mut num_masked = 0usize;
    for b in 0..batch_size {
        let pad = if b < pad_lengths.len() { pad_lengths[b] } else { 0 };
        let pad = pad.min(seq_len);
        for j in (seq_len - pad)..seq_len {
            data[b * seq_len + j] = 0.0;
            num_masked += 1;
        }
    }
    let total = batch_size * seq_len;
    AttentionMask {
        data,
        shape: vec![batch_size, seq_len],
        mask_type: MaskType::Padding { pad_lengths: pad_lengths.to_vec() },
        num_masked,
        num_unmasked: total - num_masked,
    }
}

/// Generate a sliding-window causal mask of shape `[seq_len, seq_len]`.
///
/// Position `(i, j)` is unmasked when `j <= i` **and** `i - j < window_size`.
pub fn cpu_generate_sliding_window_mask(
    seq_len: usize,
    window_size: usize,
) -> AttentionMask {
    let mut data = vec![0.0f32; seq_len * seq_len];
    let mut num_unmasked = 0usize;
    for i in 0..seq_len {
        for j in 0..=i {
            if i - j < window_size {
                data[i * seq_len + j] = 1.0;
                num_unmasked += 1;
            }
        }
    }
    let total = seq_len * seq_len;
    AttentionMask {
        data,
        shape: vec![seq_len, seq_len],
        mask_type: MaskType::SlidingWindow { window_size },
        num_masked: total - num_unmasked,
        num_unmasked,
    }
}

/// Generate a prefix-causal mask of shape `[seq_len, seq_len]`.
///
/// The first `prefix_len` tokens form a fully-visible bidirectional block;
/// tokens after the prefix follow standard causal masking.
pub fn cpu_generate_prefix_causal_mask(
    seq_len: usize,
    prefix_len: usize,
) -> AttentionMask {
    let prefix_len = prefix_len.min(seq_len);
    let mut data = vec![0.0f32; seq_len * seq_len];
    let mut num_unmasked = 0usize;
    for i in 0..seq_len {
        for j in 0..seq_len {
            let visible = if i < prefix_len && j < prefix_len {
                // Within the prefix: fully bidirectional.
                true
            } else if i >= prefix_len {
                // After the prefix: causal — may see prefix + past.
                j <= i
            } else {
                // Prefix token looking beyond prefix: not allowed.
                false
            };
            if visible {
                data[i * seq_len + j] = 1.0;
                num_unmasked += 1;
            }
        }
    }
    let total = seq_len * seq_len;
    AttentionMask {
        data,
        shape: vec![seq_len, seq_len],
        mask_type: MaskType::CausalWithPrefix { prefix_len },
        num_masked: total - num_unmasked,
        num_unmasked,
    }
}

/// Combine two masks with AND semantics (element-wise minimum).
///
/// Both masks must have identical `data` lengths.
pub fn cpu_combine_masks(
    mask_a: &AttentionMask,
    mask_b: &AttentionMask,
) -> Result<AttentionMask, MaskError> {
    if mask_a.data.len() != mask_b.data.len() {
        return Err(MaskError::ShapeMismatch);
    }
    let data: Vec<f32> =
        mask_a.data.iter().zip(&mask_b.data).map(|(a, b)| a.min(*b)).collect();
    let num_unmasked = data.iter().filter(|&&v| v > 0.0).count();
    let num_masked = data.len() - num_unmasked;
    Ok(AttentionMask {
        data,
        shape: mask_a.shape.clone(),
        mask_type: MaskType::Causal, // combined
        num_masked,
        num_unmasked,
    })
}

/// Apply a mask to attention scores in-place.
///
/// Wherever the mask is `0.0`, the corresponding score is set to `mask_value`
/// (typically `f32::NEG_INFINITY`).
pub fn cpu_apply_mask_to_scores(
    scores: &mut [f32],
    mask: &AttentionMask,
    mask_value: f32,
) {
    let len = scores.len().min(mask.data.len());
    for (score, &m) in scores[..len].iter_mut().zip(&mask.data[..len]) {
        if m == 0.0 {
            *score = mask_value;
        }
    }
}

/// Cache-aware mask generation.
///
/// Returns a cached mask if one exists for the given config, otherwise
/// generates and caches it.
pub fn cpu_generate_mask(
    mgen: &mut MaskGenerator,
    config: MaskConfig,
) -> Result<AttentionMask, MaskError> {
    if config.seq_len == 0 {
        return Err(MaskError::InvalidSeqLen);
    }

    let key = config.cache_key();
    if let Some(cached) = mgen.cache.get(&key) {
        mgen.stats.cache_hits += 1;
        return Ok(cached.clone());
    }
    mgen.stats.cache_misses += 1;

    let mask = match &config.mask_type {
        MaskType::Causal => cpu_generate_causal_mask(config.seq_len),
        MaskType::Padding { pad_lengths } => {
            cpu_generate_padding_mask(config.seq_len, pad_lengths, config.batch_size)
        }
        MaskType::SlidingWindow { window_size } => {
            let ws = *window_size;
            if ws == 0 {
                return Err(MaskError::InvalidWindowSize {
                    window: ws,
                    seq_len: config.seq_len,
                });
            }
            cpu_generate_sliding_window_mask(config.seq_len, ws)
        }
        MaskType::Custom { mask: user_mask } => {
            let rows = user_mask.len();
            let cols = if rows > 0 { user_mask[0].len() } else { 0 };
            let mut data = Vec::with_capacity(rows * cols);
            let mut num_unmasked = 0usize;
            for row in user_mask {
                for &cell in row {
                    let val = if cell { 1.0 } else { 0.0 };
                    if cell {
                        num_unmasked += 1;
                    }
                    data.push(val);
                }
            }
            let total = data.len();
            AttentionMask {
                data,
                shape: vec![rows, cols],
                mask_type: config.mask_type.clone(),
                num_masked: total - num_unmasked,
                num_unmasked,
            }
        }
        MaskType::CausalWithPrefix { prefix_len } => {
            cpu_generate_prefix_causal_mask(config.seq_len, *prefix_len)
        }
    };

    mgen.stats.masks_generated += 1;
    mgen.cache.insert(key, mask.clone());
    Ok(mask)
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Convert a mask to a flat boolean array (`true` = attend).
pub fn cpu_mask_to_bool(mask: &AttentionMask) -> Vec<bool> {
    mask.data.iter().map(|&v| v > 0.0).collect()
}

/// Count the number of unmasked positions.
pub fn cpu_count_unmasked(mask: &AttentionMask) -> usize {
    mask.data.iter().filter(|&&v| v > 0.0).count()
}

/// Basic validity check: all values are `0.0` or `1.0`, and shape product
/// matches data length.
pub fn cpu_is_valid_mask(mask: &AttentionMask) -> bool {
    let shape_product: usize = mask.shape.iter().product();
    if shape_product != mask.data.len() {
        return false;
    }
    mask.data.iter().all(|&v| v == 0.0 || v == 1.0)
}

/// Return a clone of the generator's statistics.
pub fn cpu_get_stats(mgen: &MaskGenerator) -> MaskStats {
    mgen.stats
}

/// Pretty-print a 2-D slice of the mask (first `seq_len × seq_len` block).
///
/// `█` = attend, `·` = masked.
pub fn format_mask_2d(mask: &AttentionMask, seq_len: usize) -> String {
    let mut out = String::new();
    for i in 0..seq_len {
        for j in 0..seq_len {
            let idx = i * seq_len + j;
            if idx < mask.data.len() && mask.data[idx] > 0.0 {
                out.push('█');
            } else {
                out.push('·');
            }
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------
// OpenCL kernel source (for future GPU dispatch)
// ---------------------------------------------------------------------------

/// OpenCL C source for causal mask generation.
///
/// Each work-item fills one row of the output mask buffer.
#[allow(dead_code)]
pub const CAUSAL_MASK_CL: &str = r#"
__kernel void generate_causal_mask(
    __global float* mask,
    const int seq_len)
{
    int row = get_global_id(0);
    if (row >= seq_len) return;
    for (int col = 0; col < seq_len; col++) {
        mask[row * seq_len + col] = (col <= row) ? 1.0f : 0.0f;
    }
}

__kernel void generate_sliding_window_mask(
    __global float* mask,
    const int seq_len,
    const int window_size)
{
    int row = get_global_id(0);
    if (row >= seq_len) return;
    for (int col = 0; col < seq_len; col++) {
        int causal  = (col <= row) ? 1 : 0;
        int in_win  = (row - col < window_size) ? 1 : 0;
        mask[row * seq_len + col] = (float)(causal & in_win);
    }
}

__kernel void apply_mask_to_scores(
    __global float* scores,
    __global const float* mask,
    const float mask_value,
    const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    if (mask[idx] == 0.0f) {
        scores[idx] = mask_value;
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Causal mask tests -------------------------------------------------

    #[test]
    fn causal_mask_seq4_lower_triangular() {
        let m = cpu_generate_causal_mask(4);
        let expected = vec![
            1.0, 0.0, 0.0, 0.0, // row 0
            1.0, 1.0, 0.0, 0.0, // row 1
            1.0, 1.0, 1.0, 0.0, // row 2
            1.0, 1.0, 1.0, 1.0, // row 3
        ];
        assert_eq!(m.data, expected);
    }

    #[test]
    fn causal_mask_seq8_lower_triangular() {
        let m = cpu_generate_causal_mask(8);
        assert_eq!(m.data.len(), 64);
        for i in 0..8 {
            for j in 0..8 {
                let expected = if j <= i { 1.0 } else { 0.0 };
                assert_eq!(m.data[i * 8 + j], expected, "({i},{j})");
            }
        }
    }

    #[test]
    fn causal_mask_seq16_lower_triangular() {
        let m = cpu_generate_causal_mask(16);
        assert_eq!(m.data.len(), 256);
        for i in 0..16 {
            for j in 0..16 {
                let expected = if j <= i { 1.0 } else { 0.0 };
                assert_eq!(m.data[i * 16 + j], expected, "({i},{j})");
            }
        }
    }

    #[test]
    fn causal_mask_first_token_sees_only_itself() {
        let m = cpu_generate_causal_mask(8);
        // Row 0: only column 0 should be 1.0
        assert_eq!(m.data[0], 1.0);
        for j in 1..8 {
            assert_eq!(m.data[j], 0.0);
        }
    }

    #[test]
    fn causal_mask_last_token_sees_all() {
        let m = cpu_generate_causal_mask(8);
        let last_row = &m.data[7 * 8..8 * 8];
        assert!(last_row.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn causal_mask_shape() {
        let m = cpu_generate_causal_mask(5);
        assert_eq!(m.shape, vec![5, 5]);
    }

    #[test]
    fn causal_mask_counts() {
        let m = cpu_generate_causal_mask(4);
        assert_eq!(m.num_unmasked, 10); // 4*5/2
        assert_eq!(m.num_masked, 6);
    }

    // -- Padding mask tests ------------------------------------------------

    #[test]
    fn padding_mask_correct_positions_masked() {
        let m = cpu_generate_padding_mask(6, &[2, 0, 3], 3);
        // batch 0: last 2 masked
        assert_eq!(m.data[0 * 6 + 4], 0.0);
        assert_eq!(m.data[0 * 6 + 5], 0.0);
        assert_eq!(m.data[0 * 6 + 3], 1.0);
        // batch 1: none masked
        for j in 0..6 {
            assert_eq!(m.data[1 * 6 + j], 1.0);
        }
        // batch 2: last 3 masked
        for j in 3..6 {
            assert_eq!(m.data[2 * 6 + j], 0.0);
        }
    }

    #[test]
    fn padding_mask_all_padded() {
        let m = cpu_generate_padding_mask(4, &[4], 1);
        assert!(m.data.iter().all(|&v| v == 0.0));
        assert_eq!(m.num_masked, 4);
    }

    #[test]
    fn padding_mask_none_padded() {
        let m = cpu_generate_padding_mask(4, &[0], 1);
        assert!(m.data.iter().all(|&v| v == 1.0));
        assert_eq!(m.num_unmasked, 4);
    }

    #[test]
    fn padding_mask_shape() {
        let m = cpu_generate_padding_mask(5, &[1, 2], 2);
        assert_eq!(m.shape, vec![2, 5]);
    }

    // -- Sliding window tests ----------------------------------------------

    #[test]
    fn sliding_window_correct_window_size() {
        let m = cpu_generate_sliding_window_mask(6, 3);
        // Row 5: columns 3,4,5 should be unmasked (window of 3 before row).
        assert_eq!(m.data[5 * 6 + 2], 0.0);
        assert_eq!(m.data[5 * 6 + 3], 1.0);
        assert_eq!(m.data[5 * 6 + 4], 1.0);
        assert_eq!(m.data[5 * 6 + 5], 1.0);
    }

    #[test]
    fn sliding_window_ge_seq_len_equals_causal() {
        let sw = cpu_generate_sliding_window_mask(4, 4);
        let causal = cpu_generate_causal_mask(4);
        assert_eq!(sw.data, causal.data);
    }

    #[test]
    fn sliding_window_larger_than_seq_equals_causal() {
        let sw = cpu_generate_sliding_window_mask(4, 100);
        let causal = cpu_generate_causal_mask(4);
        assert_eq!(sw.data, causal.data);
    }

    #[test]
    fn sliding_window_size_1() {
        let m = cpu_generate_sliding_window_mask(4, 1);
        // Only the diagonal should be 1.0
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(m.data[i * 4 + j], expected, "({i},{j})");
            }
        }
    }

    #[test]
    fn sliding_window_shape() {
        let m = cpu_generate_sliding_window_mask(5, 2);
        assert_eq!(m.shape, vec![5, 5]);
    }

    // -- Prefix causal tests -----------------------------------------------

    #[test]
    fn prefix_causal_prefix_fully_visible() {
        let m = cpu_generate_prefix_causal_mask(6, 3);
        // The 3×3 prefix block should be fully 1.0.
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(m.data[i * 6 + j], 1.0, "prefix({i},{j})");
            }
            // Prefix tokens must NOT see beyond the prefix.
            for j in 3..6 {
                assert_eq!(m.data[i * 6 + j], 0.0, "prefix-future({i},{j})");
            }
        }
    }

    #[test]
    fn prefix_causal_after_prefix_is_causal() {
        let m = cpu_generate_prefix_causal_mask(6, 2);
        // Row 4 (after prefix): columns 0..=4 visible.
        for j in 0..=4 {
            assert_eq!(m.data[4 * 6 + j], 1.0, "row4 col{j}");
        }
        assert_eq!(m.data[4 * 6 + 5], 0.0);
    }

    #[test]
    fn prefix_causal_prefix_equals_seq_len() {
        let m = cpu_generate_prefix_causal_mask(4, 4);
        // All positions visible (bidirectional within full prefix).
        assert!(m.data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn prefix_causal_prefix_zero_equals_causal() {
        let prefix = cpu_generate_prefix_causal_mask(4, 0);
        let causal = cpu_generate_causal_mask(4);
        assert_eq!(prefix.data, causal.data);
    }

    // -- Combine masks tests -----------------------------------------------

    #[test]
    fn combine_masks_and_semantics() {
        let a = cpu_generate_causal_mask(4);
        let b = cpu_generate_sliding_window_mask(4, 2);
        let combined = cpu_combine_masks(&a, &b).unwrap();
        // AND of causal and sliding-window(2) = sliding-window(2).
        assert_eq!(combined.data, b.data);
    }

    #[test]
    fn combine_masks_shape_mismatch_error() {
        let a = cpu_generate_causal_mask(3);
        let b = cpu_generate_causal_mask(4);
        assert_eq!(cpu_combine_masks(&a, &b), Err(MaskError::ShapeMismatch));
    }

    #[test]
    fn combine_masks_identity_with_all_ones() {
        let causal = cpu_generate_causal_mask(4);
        let ones = AttentionMask {
            data: vec![1.0; 16],
            shape: vec![4, 4],
            mask_type: MaskType::Causal,
            num_masked: 0,
            num_unmasked: 16,
        };
        let combined = cpu_combine_masks(&causal, &ones).unwrap();
        assert_eq!(combined.data, causal.data);
    }

    // -- Apply mask to scores tests ----------------------------------------

    #[test]
    fn apply_mask_masked_positions_neg_inf() {
        let mask = cpu_generate_causal_mask(3);
        let mut scores = vec![1.0; 9];
        cpu_apply_mask_to_scores(&mut scores, &mask, f32::NEG_INFINITY);
        // (0,1), (0,2), (1,2) should be -inf.
        assert_eq!(scores[1], f32::NEG_INFINITY);
        assert_eq!(scores[2], f32::NEG_INFINITY);
        assert_eq!(scores[5], f32::NEG_INFINITY);
        // Unmasked positions remain 1.0.
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[3], 1.0);
        assert_eq!(scores[4], 1.0);
    }

    #[test]
    fn apply_mask_custom_mask_value() {
        let mask = cpu_generate_causal_mask(2);
        let mut scores = vec![5.0; 4];
        cpu_apply_mask_to_scores(&mut scores, &mask, -1e9);
        assert_eq!(scores[0], 5.0); // (0,0) unmasked
        assert_eq!(scores[1], -1e9); // (0,1) masked
        assert_eq!(scores[2], 5.0); // (1,0) unmasked
        assert_eq!(scores[3], 5.0); // (1,1) unmasked
    }

    // -- Cache tests -------------------------------------------------------

    #[test]
    fn cache_hit_same_config_returns_cached() {
        let mut mgen = create_mask_generator(8);
        let config = MaskConfig {
            seq_len: 4,
            batch_size: 1,
            num_heads: 1,
            mask_type: MaskType::Causal,
            dtype: MaskDType::Float32,
        };
        let m1 = cpu_generate_mask(&mut mgen, config.clone()).unwrap();
        let m2 = cpu_generate_mask(&mut mgen, config).unwrap();
        assert_eq!(m1.data, m2.data);
        assert_eq!(mgen.stats.cache_hits, 1);
    }

    #[test]
    fn cache_miss_new_config_generates() {
        let mut mgen = create_mask_generator(8);
        let c1 = MaskConfig {
            seq_len: 4,
            batch_size: 1,
            num_heads: 1,
            mask_type: MaskType::Causal,
            dtype: MaskDType::Float32,
        };
        let c2 = MaskConfig {
            seq_len: 8,
            batch_size: 1,
            num_heads: 1,
            mask_type: MaskType::Causal,
            dtype: MaskDType::Float32,
        };
        let _ = cpu_generate_mask(&mut mgen, c1).unwrap();
        let _ = cpu_generate_mask(&mut mgen, c2).unwrap();
        assert_eq!(mgen.stats.cache_misses, 2);
        assert_eq!(mgen.stats.masks_generated, 2);
    }

    #[test]
    fn cache_eviction_when_full() {
        let mut mgen = create_mask_generator(2);
        for s in [4, 8, 16] {
            let cfg = MaskConfig {
                seq_len: s,
                batch_size: 1,
                num_heads: 1,
                mask_type: MaskType::Causal,
                dtype: MaskDType::Float32,
            };
            cpu_generate_mask(&mut mgen, cfg).unwrap();
        }
        assert_eq!(mgen.stats.masks_generated, 3);
        assert!(mgen.cache.cached_masks.len() <= 2);
    }

    #[test]
    fn cache_invalid_seq_len_zero() {
        let mut mgen = create_mask_generator(4);
        let cfg = MaskConfig {
            seq_len: 0,
            batch_size: 1,
            num_heads: 1,
            mask_type: MaskType::Causal,
            dtype: MaskDType::Float32,
        };
        assert_eq!(cpu_generate_mask(&mut mgen, cfg), Err(MaskError::InvalidSeqLen));
    }

    #[test]
    fn cache_invalid_window_size_zero() {
        let mut mgen = create_mask_generator(4);
        let cfg = MaskConfig {
            seq_len: 8,
            batch_size: 1,
            num_heads: 1,
            mask_type: MaskType::SlidingWindow { window_size: 0 },
            dtype: MaskDType::Float32,
        };
        assert!(matches!(
            cpu_generate_mask(&mut mgen, cfg),
            Err(MaskError::InvalidWindowSize { .. })
        ));
    }

    #[test]
    fn cache_custom_mask() {
        let mut mgen = create_mask_generator(4);
        let cfg = MaskConfig {
            seq_len: 2,
            batch_size: 1,
            num_heads: 1,
            mask_type: MaskType::Custom {
                mask: vec![vec![true, false], vec![true, true]],
            },
            dtype: MaskDType::Float32,
        };
        let m = cpu_generate_mask(&mut mgen, cfg).unwrap();
        assert_eq!(m.data, vec![1.0, 0.0, 1.0, 1.0]);
    }

    // -- Bool conversion ---------------------------------------------------

    #[test]
    fn bool_conversion_correct_mapping() {
        let m = cpu_generate_causal_mask(3);
        let bools = cpu_mask_to_bool(&m);
        let expected = vec![
            true, false, false, // row 0
            true, true, false, // row 1
            true, true, true, // row 2
        ];
        assert_eq!(bools, expected);
    }

    // -- Count unmasked ----------------------------------------------------

    #[test]
    fn count_unmasked_causal() {
        let m = cpu_generate_causal_mask(5);
        assert_eq!(cpu_count_unmasked(&m), 15); // 5*6/2
    }

    #[test]
    fn count_unmasked_matches_field() {
        let m = cpu_generate_causal_mask(7);
        assert_eq!(cpu_count_unmasked(&m), m.num_unmasked);
    }

    // -- Edge cases --------------------------------------------------------

    #[test]
    fn edge_seq_len_1() {
        let m = cpu_generate_causal_mask(1);
        assert_eq!(m.data, vec![1.0]);
        assert_eq!(m.num_unmasked, 1);
        assert_eq!(m.num_masked, 0);
    }

    #[test]
    fn edge_window_size_1() {
        let m = cpu_generate_sliding_window_mask(5, 1);
        // Only diagonal
        assert_eq!(cpu_count_unmasked(&m), 5);
    }

    #[test]
    fn edge_all_padding() {
        let m = cpu_generate_padding_mask(3, &[3, 3], 2);
        assert!(m.data.iter().all(|&v| v == 0.0));
    }

    // -- Properties --------------------------------------------------------

    #[test]
    fn property_causal_unmasked_count_formula() {
        for n in 1..=20 {
            let m = cpu_generate_causal_mask(n);
            assert_eq!(m.num_unmasked, n * (n + 1) / 2, "n={n}");
        }
    }

    #[test]
    fn property_sliding_window_subset_of_causal() {
        for n in [4, 8, 12] {
            let causal = cpu_generate_causal_mask(n);
            for w in 1..=n {
                let sw = cpu_generate_sliding_window_mask(n, w);
                for (i, (&s, &c)) in sw.data.iter().zip(&causal.data).enumerate() {
                    if s > 0.0 {
                        assert!(
                            c > 0.0,
                            "sw unmasked but causal masked at {i} (n={n}, w={w})"
                        );
                    }
                }
            }
        }
    }

    // -- Validity ----------------------------------------------------------

    #[test]
    fn is_valid_mask_true_for_generated() {
        assert!(cpu_is_valid_mask(&cpu_generate_causal_mask(4)));
        assert!(cpu_is_valid_mask(&cpu_generate_sliding_window_mask(4, 2)));
    }

    #[test]
    fn is_valid_mask_false_for_bad_shape() {
        let mut m = cpu_generate_causal_mask(3);
        m.shape = vec![2, 2]; // mismatch: 4 != 9
        assert!(!cpu_is_valid_mask(&m));
    }

    #[test]
    fn is_valid_mask_false_for_non_binary() {
        let mut m = cpu_generate_causal_mask(2);
        m.data[0] = 0.5;
        assert!(!cpu_is_valid_mask(&m));
    }

    // -- Format ------------------------------------------------------------

    #[test]
    fn format_mask_visual_representation() {
        let m = cpu_generate_causal_mask(3);
        let s = format_mask_2d(&m, 3);
        assert_eq!(s, "█··\n██·\n███\n");
    }

    #[test]
    fn format_mask_sliding_window() {
        let m = cpu_generate_sliding_window_mask(4, 2);
        let s = format_mask_2d(&m, 4);
        // Row 0: [1,0,0,0] → "█···"
        // Row 1: [1,1,0,0] → "██··"
        // Row 2: [0,1,1,0] → "·██·"
        // Row 3: [0,0,1,1] → "··██"
        assert_eq!(s, "█···\n██··\n·██·\n··██\n");
    }

    // -- Stats -------------------------------------------------------------

    #[test]
    fn stats_initial_zero() {
        let mgen = create_mask_generator(4);
        let stats = cpu_get_stats(&mgen);
        assert_eq!(stats.masks_generated, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }

    #[test]
    fn stats_track_generation_and_hits() {
        let mut mgen = create_mask_generator(4);
        let cfg = MaskConfig {
            seq_len: 4,
            batch_size: 1,
            num_heads: 1,
            mask_type: MaskType::Causal,
            dtype: MaskDType::Float32,
        };
        cpu_generate_mask(&mut mgen, cfg.clone()).unwrap();
        cpu_generate_mask(&mut mgen, cfg).unwrap();
        let stats = cpu_get_stats(&mgen);
        assert_eq!(stats.masks_generated, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    // -- Mask type identity ------------------------------------------------

    #[test]
    fn mask_type_preserved() {
        let m = cpu_generate_sliding_window_mask(4, 2);
        assert_eq!(m.mask_type, MaskType::SlidingWindow { window_size: 2 });
    }

    // -- OpenCL source presence --------------------------------------------

    #[test]
    fn opencl_kernel_source_not_empty() {
        assert!(!CAUSAL_MASK_CL.is_empty());
        assert!(CAUSAL_MASK_CL.contains("generate_causal_mask"));
        assert!(CAUSAL_MASK_CL.contains("generate_sliding_window_mask"));
        assert!(CAUSAL_MASK_CL.contains("apply_mask_to_scores"));
    }
}
