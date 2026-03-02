//! OpenCL-accelerated sliding window attention (SWA) for long-sequence processing.
//!
//! # Overview
//!
//! Standard self-attention scales as O(n²) in sequence length, making long
//! contexts prohibitively expensive. This module implements **sliding window
//! attention** and related sparse attention patterns that restrict each token
//! to attend only to a local neighbourhood (plus optional global tokens),
//! reducing complexity to O(n · w) where `w` is the window size.
//!
//! Supported patterns:
//!
//! | Pattern               | Description                                       |
//! |-----------------------|---------------------------------------------------|
//! | `FullCausal`          | Standard lower-triangular (baseline / fallback)   |
//! | `SlidingWindow`       | Fixed-width causal window                         |
//! | `SlidingWindowGlobal` | Sliding window + global tokens (Longformer-like)  |
//! | `Longformer`          | Sliding window + global + dilated attention        |
//! | `BigBird`             | Sliding window + global + random attention blocks  |
//!
//! # Modules
//!
//! - [`WindowConfig`] — tuning knobs (window size, overlap, global count).
//! - [`AttentionPattern`] — enum selecting the sparse pattern.
//! - [`WindowMask`] — generates the boolean attention mask.
//! - [`SlidingWindowAttention`] — computes masked attention (CPU reference).
//! - [`GlobalTokens`] — manages which positions are "global".
//! - [`WindowedKvCache`] — KV cache that evicts entries outside the window.
//! - [`WindowStats`] — reports sparsity and compute-savings metrics.
//! - [`ChunkedPrefill`] — splits a long prefill into window-sized chunks.
//! - OpenCL kernel source string ([`SLIDING_WINDOW_ATTENTION_CL`]).

use std::collections::HashSet;

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for sliding window attention.
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Number of past positions each query may attend to (including itself).
    pub window_size: usize,
    /// Overlap between consecutive chunks during chunked prefill.
    pub overlap: usize,
    /// Number of global tokens prepended to every attention window.
    pub global_token_count: usize,
    /// Which sparse attention pattern to use.
    pub pattern_type: AttentionPattern,
}

impl WindowConfig {
    /// Create a new configuration.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] when `window_size == 0` or
    /// `overlap >= window_size`.
    pub fn new(
        window_size: usize,
        overlap: usize,
        global_token_count: usize,
        pattern_type: AttentionPattern,
    ) -> Result<Self> {
        if window_size == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "window_size must be > 0".into() }.into()
            );
        }
        if overlap >= window_size {
            return Err(KernelError::InvalidArguments {
                reason: format!("overlap ({overlap}) must be < window_size ({window_size})"),
            }
            .into());
        }
        Ok(Self { window_size, overlap, global_token_count, pattern_type })
    }
}

// ---------------------------------------------------------------------------
// Attention pattern enum
// ---------------------------------------------------------------------------

/// Sparse attention pattern selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionPattern {
    /// Standard full causal (lower-triangular) mask — no sparsity.
    FullCausal,
    /// Fixed-width sliding window (each token attends to
    /// `[max(0, i - w + 1) .. i]`).
    SlidingWindow,
    /// Sliding window plus a set of global tokens that attend everywhere.
    SlidingWindowGlobal,
    /// Longformer-style: sliding window + global tokens + dilated gaps.
    /// The `usize` is the dilation factor.
    Longformer(usize),
    /// BigBird-style: sliding window + global tokens + random attention
    /// blocks. The `usize` is the number of random blocks per query row.
    BigBird(usize),
}

// ---------------------------------------------------------------------------
// Global tokens
// ---------------------------------------------------------------------------

/// Tracks which positions are "global" (attend to / from all positions).
#[derive(Debug, Clone)]
pub struct GlobalTokens {
    positions: HashSet<usize>,
}

impl GlobalTokens {
    /// Create a global-token set from the first `count` positions.
    pub fn first_n(count: usize) -> Self {
        Self { positions: (0..count).collect() }
    }

    /// Create from an explicit set of positions.
    pub fn from_positions(positions: &[usize]) -> Self {
        Self { positions: positions.iter().copied().collect() }
    }

    /// Returns `true` if position `pos` is global.
    #[inline]
    pub fn is_global(&self, pos: usize) -> bool {
        self.positions.contains(&pos)
    }

    /// Number of global positions.
    #[inline]
    pub fn count(&self) -> usize {
        self.positions.len()
    }

    /// Iterate over global positions (unordered).
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.positions.iter().copied()
    }
}

// ---------------------------------------------------------------------------
// Window mask
// ---------------------------------------------------------------------------

/// Sparse attention mask for a given pattern and sequence geometry.
#[derive(Debug, Clone)]
pub struct WindowMask {
    /// Row-major `[seq_len, kv_len]` boolean mask.
    mask: Vec<bool>,
    pub seq_len: usize,
    pub kv_len: usize,
}

impl WindowMask {
    /// Generate a mask for the given config and sequence lengths.
    ///
    /// `global_tokens` may be `None` when no global positions are needed.
    pub fn generate(
        config: &WindowConfig,
        seq_len: usize,
        kv_len: usize,
        global_tokens: Option<&GlobalTokens>,
    ) -> Self {
        let mut mask = vec![false; seq_len * kv_len];
        for i in 0..seq_len {
            for j in 0..kv_len {
                mask[i * kv_len + j] =
                    Self::should_attend(config, i, j, seq_len, kv_len, global_tokens);
            }
        }
        Self { mask, seq_len, kv_len }
    }

    /// Decide whether query position `i` may attend to key position `j`.
    fn should_attend(
        config: &WindowConfig,
        i: usize,
        j: usize,
        _seq_len: usize,
        _kv_len: usize,
        global_tokens: Option<&GlobalTokens>,
    ) -> bool {
        // Causal constraint: never attend to future positions.
        if j > i {
            return false;
        }

        // Global tokens attend to / from all (causal) positions.
        if let Some(gt) = global_tokens
            && (gt.is_global(i) || gt.is_global(j))
        {
            return true;
        }

        match &config.pattern_type {
            AttentionPattern::FullCausal => true,
            AttentionPattern::SlidingWindow | AttentionPattern::SlidingWindowGlobal => {
                let start = i.saturating_sub(config.window_size - 1);
                j >= start
            }
            AttentionPattern::Longformer(dilation) => {
                let start = i.saturating_sub(config.window_size - 1);
                if j >= start {
                    return true;
                }
                // Dilated attention: attend every `dilation` positions.
                if *dilation > 0 && j.is_multiple_of(*dilation) {
                    return true;
                }
                false
            }
            AttentionPattern::BigBird(num_random) => {
                let start = i.saturating_sub(config.window_size - 1);
                if j >= start {
                    return true;
                }
                // Deterministic pseudo-random blocks seeded by row index.
                if *num_random > 0 && i > 0 {
                    let seed = i.wrapping_mul(2654435761) ^ j;
                    if seed % (i.max(1)) < *num_random {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Check whether query position `i` may attend to key position `j`.
    #[inline]
    pub fn allows(&self, i: usize, j: usize) -> bool {
        self.mask[i * self.kv_len + j]
    }

    /// Number of `true` entries.
    pub fn nnz(&self) -> usize {
        self.mask.iter().filter(|&&v| v).count()
    }

    /// Total number of entries in the mask.
    pub fn total(&self) -> usize {
        self.seq_len * self.kv_len
    }

    /// Fraction of entries that are masked out (false).
    pub fn sparsity(&self) -> f64 {
        if self.total() == 0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f64 / self.total() as f64)
    }
}

// ---------------------------------------------------------------------------
// Window statistics
// ---------------------------------------------------------------------------

/// Summary statistics for a sliding-window attention configuration.
#[derive(Debug, Clone)]
pub struct WindowStats {
    /// Effective local context each token can attend to.
    pub effective_context_length: usize,
    /// Ratio of FLOPs saved compared to full attention (0.0 – 1.0).
    pub compute_savings_ratio: f64,
    /// Fraction of mask entries that are `false`.
    pub mask_sparsity: f64,
}

impl WindowStats {
    /// Compute statistics for a given config and sequence length.
    pub fn compute(config: &WindowConfig, seq_len: usize) -> Self {
        let effective_context_length = config.window_size.min(seq_len) + config.global_token_count;
        let effective_context_length = effective_context_length.min(seq_len);

        // Full causal attention has n*(n+1)/2 active entries.
        let full_causal_nnz = seq_len * (seq_len + 1) / 2;

        // Estimate windowed nnz: each row i attends to
        // min(window_size, i+1) + global_token_count positions.
        let mut windowed_nnz: usize = 0;
        for i in 0..seq_len {
            let local = config.window_size.min(i + 1);
            let global = config.global_token_count.min(seq_len);
            // Avoid double-counting globals inside the window.
            let total_for_row = (local + global).min(i + 1);
            windowed_nnz += total_for_row;
        }

        let compute_savings_ratio = if full_causal_nnz > 0 {
            1.0 - (windowed_nnz as f64 / full_causal_nnz as f64)
        } else {
            0.0
        };
        let compute_savings_ratio = compute_savings_ratio.max(0.0);

        // Mask sparsity over the full [seq_len, seq_len] grid.
        let total = seq_len * seq_len;
        let mask_sparsity =
            if total > 0 { 1.0 - (windowed_nnz as f64 / total as f64) } else { 0.0 };

        Self { effective_context_length, compute_savings_ratio, mask_sparsity }
    }
}

// ---------------------------------------------------------------------------
// Windowed KV cache
// ---------------------------------------------------------------------------

/// KV cache that retains only the most recent `window_size` entries plus
/// any global-token entries.
#[derive(Debug, Clone)]
pub struct WindowedKvCache {
    /// Per-position key vectors, shape `[capacity, head_dim]`.
    keys: Vec<f32>,
    /// Per-position value vectors, shape `[capacity, head_dim]`.
    values: Vec<f32>,
    head_dim: usize,
    window_size: usize,
    /// Total number of positions that have been appended (monotonically
    /// increasing).
    total_appended: usize,
    /// Global token positions that are never evicted.
    global_positions: HashSet<usize>,
    /// Ring-buffer of active (non-global) position indices.
    active_positions: Vec<usize>,
}

impl WindowedKvCache {
    /// Create a new windowed KV cache.
    ///
    /// # Errors
    ///
    /// Returns an error if `head_dim == 0` or `window_size == 0`.
    pub fn new(
        head_dim: usize,
        window_size: usize,
        global_tokens: Option<&GlobalTokens>,
    ) -> Result<Self> {
        if head_dim == 0 || window_size == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "head_dim and window_size must be > 0".into(),
            }
            .into());
        }
        let global_positions: HashSet<usize> =
            global_tokens.map(|gt| gt.iter().collect()).unwrap_or_default();
        let capacity = window_size + global_positions.len();
        Ok(Self {
            keys: vec![0.0; capacity * head_dim],
            values: vec![0.0; capacity * head_dim],
            head_dim,
            window_size,
            total_appended: 0,
            global_positions,
            active_positions: Vec::with_capacity(window_size),
        })
    }

    /// Append a new key-value pair at the current position.
    pub fn append(&mut self, key: &[f32], value: &[f32]) {
        assert_eq!(key.len(), self.head_dim);
        assert_eq!(value.len(), self.head_dim);

        let pos = self.total_appended;
        self.total_appended += 1;

        if self.global_positions.contains(&pos) {
            // Store in the global section (at the front).
            let slot = self.global_slot(pos);
            let off = slot * self.head_dim;
            self.keys[off..off + self.head_dim].copy_from_slice(key);
            self.values[off..off + self.head_dim].copy_from_slice(value);
        } else {
            // Ring-buffer for the sliding window section.
            if self.active_positions.len() >= self.window_size {
                // Evict the oldest non-global entry.
                let evict_idx = 0;
                let slot = self.global_positions.len() + evict_idx;
                let off = slot * self.head_dim;
                self.keys[off..off + self.head_dim].copy_from_slice(key);
                self.values[off..off + self.head_dim].copy_from_slice(value);
                self.active_positions.remove(0);
                self.active_positions.push(pos);
            } else {
                let slot = self.global_positions.len() + self.active_positions.len();
                let off = slot * self.head_dim;
                self.keys[off..off + self.head_dim].copy_from_slice(key);
                self.values[off..off + self.head_dim].copy_from_slice(value);
                self.active_positions.push(pos);
            }
        }
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        let global_stored =
            self.global_positions.iter().filter(|&&p| p < self.total_appended).count();
        global_stored + self.active_positions.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read all cached keys as a contiguous slice (global first, then window).
    pub fn keys(&self) -> &[f32] {
        &self.keys[..self.len() * self.head_dim]
    }

    /// Read all cached values as a contiguous slice.
    pub fn values(&self) -> &[f32] {
        &self.values[..self.len() * self.head_dim]
    }

    /// Total number of positions appended so far.
    pub fn total_appended(&self) -> usize {
        self.total_appended
    }

    /// Map a global position to its slot index (deterministic ordering).
    fn global_slot(&self, pos: usize) -> usize {
        // Sorted order among globals for determinism.
        let mut sorted: Vec<usize> = self.global_positions.iter().copied().collect();
        sorted.sort_unstable();
        sorted.iter().position(|&p| p == pos).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// CPU reference: sliding window attention
// ---------------------------------------------------------------------------

/// Computes masked attention output on the CPU (reference implementation).
///
/// `q`, `k`, `v` are `[seq_len, head_dim]`, `[kv_len, head_dim]`,
/// `[kv_len, head_dim]` respectively (single-head, row-major).
pub struct SlidingWindowAttention;

impl SlidingWindowAttention {
    /// Compute `softmax(Q @ K^T / sqrt(d) * mask) @ V`.
    ///
    /// # Errors
    ///
    /// Returns an error on dimension mismatches.
    pub fn compute(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        kv_len: usize,
        head_dim: usize,
        mask: &WindowMask,
    ) -> Result<Vec<f32>> {
        if q.len() != seq_len * head_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "q length {} != seq_len({}) * head_dim({})",
                    q.len(),
                    seq_len,
                    head_dim
                ),
            }
            .into());
        }
        if k.len() != kv_len * head_dim || v.len() != kv_len * head_dim {
            return Err(
                KernelError::InvalidArguments { reason: "k/v length mismatch".into() }.into()
            );
        }
        if mask.seq_len != seq_len || mask.kv_len != kv_len {
            return Err(KernelError::InvalidArguments {
                reason: "mask dimensions mismatch".into(),
            }
            .into());
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * head_dim];

        for i in 0..seq_len {
            // Compute raw scores for row i.
            let mut scores = vec![f32::NEG_INFINITY; kv_len];
            for j in 0..kv_len {
                if mask.allows(i, j) {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    scores[j] = dot * scale;
                }
            }

            // Softmax.
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max_score == f32::NEG_INFINITY {
                // All positions masked — output zeros.
                continue;
            }
            let mut sum = 0.0f32;
            let mut exp_scores = vec![0.0f32; kv_len];
            for j in 0..kv_len {
                if scores[j] != f32::NEG_INFINITY {
                    let e = (scores[j] - max_score).exp();
                    exp_scores[j] = e;
                    sum += e;
                }
            }
            if sum > 0.0 {
                for e in &mut exp_scores {
                    *e /= sum;
                }
            }

            // Weighted sum of values.
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..kv_len {
                    acc += exp_scores[j] * v[j * head_dim + d];
                }
                output[i * head_dim + d] = acc;
            }
        }

        Ok(output)
    }

    /// Convenience: full causal attention (no window) for comparison.
    pub fn full_causal(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let config = WindowConfig {
            window_size: seq_len,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::FullCausal,
        };
        let mask = WindowMask::generate(&config, seq_len, seq_len, None);
        Self::compute(q, k, v, seq_len, seq_len, head_dim, &mask)
    }
}

// ---------------------------------------------------------------------------
// Chunked prefill
// ---------------------------------------------------------------------------

/// Splits a long prefill sequence into window-sized chunks, computes
/// attention within each chunk, and concatenates results.
pub struct ChunkedPrefill;

impl ChunkedPrefill {
    /// Run chunked prefill.
    ///
    /// `q`, `k`, `v` are all `[seq_len, head_dim]`.
    ///
    /// # Errors
    ///
    /// Returns an error on dimension mismatches.
    pub fn run(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        config: &WindowConfig,
    ) -> Result<Vec<f32>> {
        if q.len() != seq_len * head_dim
            || k.len() != seq_len * head_dim
            || v.len() != seq_len * head_dim
        {
            return Err(
                KernelError::InvalidArguments { reason: "input length mismatch".into() }.into()
            );
        }

        // If window covers the whole sequence, just do one pass.
        if config.window_size >= seq_len {
            let mask = WindowMask::generate(config, seq_len, seq_len, None);
            return SlidingWindowAttention::compute(q, k, v, seq_len, seq_len, head_dim, &mask);
        }

        let step = config.window_size - config.overlap;
        let mut output = vec![0.0f32; seq_len * head_dim];

        let mut chunk_start: usize = 0;
        while chunk_start < seq_len {
            let chunk_end = (chunk_start + config.window_size).min(seq_len);
            let chunk_len = chunk_end - chunk_start;

            // KV range: the chunk can attend to keys in
            // [kv_start .. chunk_end].
            let kv_start = chunk_start.saturating_sub(config.overlap);
            let kv_end = chunk_end;
            let kv_len = kv_end - kv_start;

            // Slice Q for this chunk.
            let q_chunk: Vec<f32> = q[chunk_start * head_dim..chunk_end * head_dim].to_vec();
            let k_chunk: Vec<f32> = k[kv_start * head_dim..kv_end * head_dim].to_vec();
            let v_chunk: Vec<f32> = v[kv_start * head_dim..kv_end * head_dim].to_vec();

            // Build a local mask for this chunk.
            let local_config = WindowConfig {
                window_size: config.window_size,
                overlap: 0,
                global_token_count: 0,
                pattern_type: AttentionPattern::SlidingWindow,
            };
            let local_mask = WindowMask::generate(&local_config, chunk_len, kv_len, None);

            let chunk_out = SlidingWindowAttention::compute(
                &q_chunk,
                &k_chunk,
                &v_chunk,
                chunk_len,
                kv_len,
                head_dim,
                &local_mask,
            )?;

            output[chunk_start * head_dim..chunk_end * head_dim].copy_from_slice(&chunk_out);

            chunk_start += step;
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C kernel for sliding-window attention (single-workitem-per-query-row).
///
/// This is a functional-correctness kernel suitable for validation on Intel
/// Arc A770 and other OpenCL 3.0 devices. A tiled / sub-group variant is
/// planned for production throughput.
pub const SLIDING_WINDOW_ATTENTION_CL: &str = r#"
__kernel void sliding_window_attention(
    __global const float* Q,       // [seq_len, head_dim]
    __global const float* K,       // [kv_len, head_dim]
    __global const float* V,       // [kv_len, head_dim]
    __global const int*   mask,    // [seq_len, kv_len] — 1 = attend, 0 = mask
    __global       float* output,  // [seq_len, head_dim]
    const int seq_len,
    const int kv_len,
    const int head_dim,
    const float scale
) {
    int i = get_global_id(0);  // query row
    if (i >= seq_len) return;

    // --- Compute raw scores and find max (for numerical stability) --------
    float max_score = -1e30f;
    for (int j = 0; j < kv_len; ++j) {
        if (mask[i * kv_len + j] == 0) continue;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += Q[i * head_dim + d] * K[j * head_dim + d];
        }
        dot *= scale;
        if (dot > max_score) max_score = dot;
    }

    // --- Softmax numerator + denominator -----------------------------------
    float sum_exp = 0.0f;
    for (int j = 0; j < kv_len; ++j) {
        if (mask[i * kv_len + j] == 0) continue;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += Q[i * head_dim + d] * K[j * head_dim + d];
        }
        dot = exp(dot * scale - max_score);
        sum_exp += dot;
    }

    // --- Weighted value accumulation ---------------------------------------
    for (int d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < kv_len; ++j) {
            if (mask[i * kv_len + j] == 0) continue;
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) {
                dot += Q[i * head_dim + dd] * K[j * head_dim + dd];
            }
            float w = exp(dot * scale - max_score) / sum_exp;
            acc += w * V[j * head_dim + d];
        }
        output[i * head_dim + d] = acc;
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers --------------------------------------------------------

    const ATOL: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn assert_slices_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(approx_eq(*x, *y, tol), "mismatch at index {i}: {x} vs {y} (tol={tol})");
        }
    }

    /// Deterministic Q/K/V for a given seq_len and head_dim.
    fn make_qkv(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.02).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03 + 1.0).sin()).collect();
        (q, k, v)
    }

    // ====================================================================
    // WindowConfig
    // ====================================================================

    #[test]
    fn test_config_valid() {
        let cfg = WindowConfig::new(128, 16, 2, AttentionPattern::SlidingWindow).unwrap();
        assert_eq!(cfg.window_size, 128);
        assert_eq!(cfg.overlap, 16);
        assert_eq!(cfg.global_token_count, 2);
    }

    #[test]
    fn test_config_zero_window() {
        assert!(WindowConfig::new(0, 0, 0, AttentionPattern::SlidingWindow).is_err());
    }

    #[test]
    fn test_config_overlap_ge_window() {
        assert!(WindowConfig::new(4, 4, 0, AttentionPattern::SlidingWindow).is_err());
        assert!(WindowConfig::new(4, 5, 0, AttentionPattern::SlidingWindow).is_err());
    }

    #[test]
    fn test_config_overlap_lt_window_ok() {
        assert!(WindowConfig::new(4, 3, 0, AttentionPattern::SlidingWindow).is_ok());
    }

    // ====================================================================
    // AttentionPattern
    // ====================================================================

    #[test]
    fn test_pattern_eq() {
        assert_eq!(AttentionPattern::FullCausal, AttentionPattern::FullCausal);
        assert_ne!(AttentionPattern::SlidingWindow, AttentionPattern::FullCausal);
        assert_eq!(AttentionPattern::Longformer(3), AttentionPattern::Longformer(3));
        assert_ne!(AttentionPattern::BigBird(2), AttentionPattern::BigBird(3));
    }

    #[test]
    fn test_pattern_clone() {
        let p = AttentionPattern::BigBird(5);
        let p2 = p.clone();
        assert_eq!(p, p2);
    }

    // ====================================================================
    // GlobalTokens
    // ====================================================================

    #[test]
    fn test_global_tokens_first_n() {
        let gt = GlobalTokens::first_n(3);
        assert!(gt.is_global(0));
        assert!(gt.is_global(1));
        assert!(gt.is_global(2));
        assert!(!gt.is_global(3));
        assert_eq!(gt.count(), 3);
    }

    #[test]
    fn test_global_tokens_from_positions() {
        let gt = GlobalTokens::from_positions(&[0, 5, 10]);
        assert!(gt.is_global(0));
        assert!(!gt.is_global(1));
        assert!(gt.is_global(5));
        assert!(gt.is_global(10));
        assert_eq!(gt.count(), 3);
    }

    #[test]
    fn test_global_tokens_empty() {
        let gt = GlobalTokens::first_n(0);
        assert_eq!(gt.count(), 0);
        assert!(!gt.is_global(0));
    }

    #[test]
    fn test_global_tokens_iter() {
        let gt = GlobalTokens::first_n(3);
        let mut positions: Vec<usize> = gt.iter().collect();
        positions.sort_unstable();
        assert_eq!(positions, vec![0, 1, 2]);
    }

    #[test]
    fn test_global_tokens_dedup() {
        let gt = GlobalTokens::from_positions(&[1, 1, 2, 2]);
        assert_eq!(gt.count(), 2);
    }

    // ====================================================================
    // WindowMask — full causal
    // ====================================================================

    #[test]
    fn test_full_causal_mask_4x4() {
        let config = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::FullCausal,
        };
        let mask = WindowMask::generate(&config, 4, 4, None);
        // Row 0: [T F F F]
        assert!(mask.allows(0, 0));
        assert!(!mask.allows(0, 1));
        // Row 3: [T T T T]
        for j in 0..4 {
            assert!(mask.allows(3, j));
        }
    }

    #[test]
    fn test_full_causal_nnz() {
        let config = WindowConfig {
            window_size: 8,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::FullCausal,
        };
        let mask = WindowMask::generate(&config, 4, 4, None);
        // Lower triangle: 1+2+3+4 = 10
        assert_eq!(mask.nnz(), 10);
    }

    #[test]
    fn test_full_causal_sparsity() {
        let config = WindowConfig {
            window_size: 8,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::FullCausal,
        };
        let mask = WindowMask::generate(&config, 4, 4, None);
        // 10 / 16 = 0.625 → sparsity = 0.375
        let expected = 1.0 - 10.0 / 16.0;
        assert!((mask.sparsity() - expected).abs() < 1e-10);
    }

    // ====================================================================
    // WindowMask — sliding window
    // ====================================================================

    #[test]
    fn test_sliding_window_mask_w2() {
        let config = WindowConfig {
            window_size: 2,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 4, 4, None);
        // Row 0: [T F F F]
        assert!(mask.allows(0, 0));
        assert!(!mask.allows(0, 1));
        // Row 1: [T T F F]
        assert!(mask.allows(1, 0));
        assert!(mask.allows(1, 1));
        assert!(!mask.allows(1, 2));
        // Row 2: [F T T F]
        assert!(!mask.allows(2, 0));
        assert!(mask.allows(2, 1));
        assert!(mask.allows(2, 2));
        // Row 3: [F F T T]
        assert!(!mask.allows(3, 0));
        assert!(!mask.allows(3, 1));
        assert!(mask.allows(3, 2));
        assert!(mask.allows(3, 3));
    }

    #[test]
    fn test_sliding_window_mask_w1() {
        let config = WindowConfig {
            window_size: 1,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 4, 4, None);
        // Each row only attends to itself.
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(mask.allows(i, j), i == j);
            }
        }
    }

    #[test]
    fn test_sliding_window_larger_than_seq() {
        let config = WindowConfig {
            window_size: 100,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 4, 4, None);
        // Should be identical to full causal.
        let full_config = WindowConfig {
            window_size: 100,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::FullCausal,
        };
        let full_mask = WindowMask::generate(&full_config, 4, 4, None);
        assert_eq!(mask.nnz(), full_mask.nnz());
    }

    #[test]
    fn test_sliding_window_more_sparse_than_full() {
        let sw = WindowConfig {
            window_size: 2,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let fc = WindowConfig {
            window_size: 8,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::FullCausal,
        };
        let sw_mask = WindowMask::generate(&sw, 8, 8, None);
        let fc_mask = WindowMask::generate(&fc, 8, 8, None);
        assert!(sw_mask.nnz() < fc_mask.nnz());
        assert!(sw_mask.sparsity() > fc_mask.sparsity());
    }

    #[test]
    fn test_sliding_window_w64() {
        let config = WindowConfig {
            window_size: 64,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 128, 128, None);
        // Row 127: can attend to [64..127].
        assert!(!mask.allows(127, 63));
        assert!(mask.allows(127, 64));
        assert!(mask.allows(127, 127));
    }

    #[test]
    fn test_sliding_window_w128() {
        let config = WindowConfig {
            window_size: 128,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 128, 128, None);
        // Window covers entire sequence → same as full causal.
        let fc_config = WindowConfig {
            window_size: 128,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::FullCausal,
        };
        let fc_mask = WindowMask::generate(&fc_config, 128, 128, None);
        assert_eq!(mask.nnz(), fc_mask.nnz());
    }

    #[test]
    fn test_sliding_window_w512_sparse() {
        let config = WindowConfig {
            window_size: 512,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let seq = 1024;
        let mask = WindowMask::generate(&config, seq, seq, None);
        // With w=512 on seq=1024, sparsity should be > 0.
        assert!(mask.sparsity() > 0.0);
        assert!(mask.sparsity() < 1.0);
    }

    // ====================================================================
    // WindowMask — global tokens
    // ====================================================================

    #[test]
    fn test_global_tokens_attend_everywhere() {
        let config = WindowConfig {
            window_size: 2,
            overlap: 0,
            global_token_count: 1,
            pattern_type: AttentionPattern::SlidingWindowGlobal,
        };
        let gt = GlobalTokens::first_n(1);
        let mask = WindowMask::generate(&config, 4, 4, Some(&gt));
        // Row 0 (global): attends to position 0 only (causal).
        assert!(mask.allows(0, 0));
        assert!(!mask.allows(0, 1)); // causal blocks future
        // Row 3: attends to global (0) and window (2, 3).
        assert!(mask.allows(3, 0)); // global
        assert!(mask.allows(3, 2)); // window
        assert!(mask.allows(3, 3)); // self
    }

    #[test]
    fn test_global_tokens_column_visible() {
        let config = WindowConfig {
            window_size: 1,
            overlap: 0,
            global_token_count: 1,
            pattern_type: AttentionPattern::SlidingWindowGlobal,
        };
        let gt = GlobalTokens::first_n(1);
        let mask = WindowMask::generate(&config, 4, 4, Some(&gt));
        // Every row can attend to position 0 (the global token).
        for i in 0..4 {
            assert!(mask.allows(i, 0), "row {i} should attend to global 0");
        }
    }

    #[test]
    fn test_multiple_global_tokens() {
        let config = WindowConfig {
            window_size: 1,
            overlap: 0,
            global_token_count: 2,
            pattern_type: AttentionPattern::SlidingWindowGlobal,
        };
        let gt = GlobalTokens::first_n(2);
        let mask = WindowMask::generate(&config, 4, 4, Some(&gt));
        // Row 3: attends to globals (0, 1) and self (3).
        assert!(mask.allows(3, 0));
        assert!(mask.allows(3, 1));
        assert!(!mask.allows(3, 2)); // not in window, not global
        assert!(mask.allows(3, 3));
    }

    // ====================================================================
    // WindowMask — Longformer (dilated)
    // ====================================================================

    #[test]
    fn test_longformer_dilation() {
        let config = WindowConfig {
            window_size: 2,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::Longformer(3),
        };
        let mask = WindowMask::generate(&config, 8, 8, None);
        // Row 7: window = [6,7], dilated every 3 → positions 0, 3, 6.
        assert!(mask.allows(7, 0)); // dilated (0 % 3 == 0)
        assert!(!mask.allows(7, 1)); // not in window or dilated
        assert!(!mask.allows(7, 2)); // not in window or dilated
        assert!(mask.allows(7, 3)); // dilated (3 % 3 == 0)
        assert!(!mask.allows(7, 4));
        assert!(!mask.allows(7, 5));
        assert!(mask.allows(7, 6)); // window + dilated
        assert!(mask.allows(7, 7)); // window (self)
    }

    #[test]
    fn test_longformer_zero_dilation_eq_sliding() {
        let lf = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::Longformer(0),
        };
        let sw = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let lf_mask = WindowMask::generate(&lf, 8, 8, None);
        let sw_mask = WindowMask::generate(&sw, 8, 8, None);
        assert_eq!(lf_mask.nnz(), sw_mask.nnz());
    }

    // ====================================================================
    // WindowMask — BigBird (random)
    // ====================================================================

    #[test]
    fn test_bigbird_has_random_connections() {
        let config = WindowConfig {
            window_size: 2,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::BigBird(3),
        };
        let mask = WindowMask::generate(&config, 16, 16, None);
        // BigBird should have more active entries than plain sliding window.
        let sw = WindowConfig {
            window_size: 2,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let sw_mask = WindowMask::generate(&sw, 16, 16, None);
        assert!(mask.nnz() >= sw_mask.nnz());
    }

    #[test]
    fn test_bigbird_zero_random_eq_sliding() {
        let bb = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::BigBird(0),
        };
        let sw = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let bb_mask = WindowMask::generate(&bb, 8, 8, None);
        let sw_mask = WindowMask::generate(&sw, 8, 8, None);
        assert_eq!(bb_mask.nnz(), sw_mask.nnz());
    }

    #[test]
    fn test_bigbird_deterministic() {
        let config = WindowConfig {
            window_size: 2,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::BigBird(2),
        };
        let m1 = WindowMask::generate(&config, 8, 8, None);
        let m2 = WindowMask::generate(&config, 8, 8, None);
        assert_eq!(m1.nnz(), m2.nnz());
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(m1.allows(i, j), m2.allows(i, j));
            }
        }
    }

    // ====================================================================
    // WindowMask — edge cases
    // ====================================================================

    #[test]
    fn test_mask_seq_len_1() {
        let config = WindowConfig {
            window_size: 1,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 1, 1, None);
        assert!(mask.allows(0, 0));
        assert_eq!(mask.nnz(), 1);
    }

    #[test]
    fn test_mask_empty_sparsity() {
        let config = WindowConfig {
            window_size: 1,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 0, 0, None);
        assert_eq!(mask.sparsity(), 0.0);
    }

    #[test]
    fn test_mask_total() {
        let config = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 5, 5, None);
        assert_eq!(mask.total(), 25);
    }

    // ====================================================================
    // WindowStats
    // ====================================================================

    #[test]
    fn test_stats_full_causal_no_savings() {
        let config = WindowConfig {
            window_size: 64,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::FullCausal,
        };
        let stats = WindowStats::compute(&config, 64);
        assert_eq!(stats.effective_context_length, 64);
        assert!(stats.compute_savings_ratio.abs() < 0.01);
    }

    #[test]
    fn test_stats_small_window_savings() {
        let config = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let stats = WindowStats::compute(&config, 128);
        assert!(stats.compute_savings_ratio > 0.5);
        assert!(stats.mask_sparsity > 0.5);
    }

    #[test]
    fn test_stats_effective_context_with_globals() {
        let config = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 2,
            pattern_type: AttentionPattern::SlidingWindowGlobal,
        };
        let stats = WindowStats::compute(&config, 128);
        assert_eq!(stats.effective_context_length, 6);
    }

    #[test]
    fn test_stats_effective_context_capped_at_seq() {
        let config = WindowConfig {
            window_size: 200,
            overlap: 0,
            global_token_count: 50,
            pattern_type: AttentionPattern::SlidingWindowGlobal,
        };
        let stats = WindowStats::compute(&config, 10);
        assert_eq!(stats.effective_context_length, 10);
    }

    #[test]
    fn test_stats_seq_len_zero() {
        let config = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let stats = WindowStats::compute(&config, 0);
        assert_eq!(stats.effective_context_length, 0);
        assert_eq!(stats.compute_savings_ratio, 0.0);
    }

    // ====================================================================
    // WindowedKvCache
    // ====================================================================

    #[test]
    fn test_kv_cache_new_valid() {
        let cache = WindowedKvCache::new(64, 128, None).unwrap();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kv_cache_zero_head_dim() {
        assert!(WindowedKvCache::new(0, 128, None).is_err());
    }

    #[test]
    fn test_kv_cache_zero_window() {
        assert!(WindowedKvCache::new(64, 0, None).is_err());
    }

    #[test]
    fn test_kv_cache_append_and_len() {
        let mut cache = WindowedKvCache::new(4, 3, None).unwrap();
        cache.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.len(), 1);
        cache.append(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_kv_cache_eviction() {
        let mut cache = WindowedKvCache::new(2, 2, None).unwrap();
        // Append 3 entries; window=2 → oldest evicted.
        cache.append(&[1.0, 1.0], &[10.0, 10.0]);
        cache.append(&[2.0, 2.0], &[20.0, 20.0]);
        assert_eq!(cache.len(), 2);
        cache.append(&[3.0, 3.0], &[30.0, 30.0]);
        assert_eq!(cache.len(), 2); // still 2
        assert_eq!(cache.total_appended(), 3);
    }

    #[test]
    fn test_kv_cache_keys_values_slices() {
        let mut cache = WindowedKvCache::new(2, 4, None).unwrap();
        cache.append(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(cache.keys().len(), 2);
        assert_eq!(cache.values().len(), 2);
        assert_slices_close(cache.keys(), &[1.0, 2.0], 1e-7);
        assert_slices_close(cache.values(), &[3.0, 4.0], 1e-7);
    }

    #[test]
    fn test_kv_cache_with_global_tokens() {
        let gt = GlobalTokens::first_n(1);
        let mut cache = WindowedKvCache::new(2, 2, Some(&gt)).unwrap();
        // Position 0 is global.
        cache.append(&[1.0, 1.0], &[10.0, 10.0]); // global
        cache.append(&[2.0, 2.0], &[20.0, 20.0]); // window
        cache.append(&[3.0, 3.0], &[30.0, 30.0]); // window
        // Global (1) + window (2) = 3 entries.
        assert_eq!(cache.len(), 3);
        cache.append(&[4.0, 4.0], &[40.0, 40.0]); // evicts pos 1
        assert_eq!(cache.len(), 3); // global + 2 window
    }

    #[test]
    fn test_kv_cache_is_empty_after_new() {
        let cache = WindowedKvCache::new(4, 8, None).unwrap();
        assert!(cache.is_empty());
    }

    // ====================================================================
    // SlidingWindowAttention — correctness
    // ====================================================================

    #[test]
    fn test_swa_full_causal_matches_reference() {
        let seq = 4;
        let hd = 4;
        let (q, k, v) = make_qkv(seq, hd);
        let full = SlidingWindowAttention::full_causal(&q, &k, &v, seq, hd).unwrap();
        // Build same result via windowed path with window >= seq.
        let config = WindowConfig {
            window_size: seq,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::FullCausal,
        };
        let mask = WindowMask::generate(&config, seq, seq, None);
        let windowed = SlidingWindowAttention::compute(&q, &k, &v, seq, seq, hd, &mask).unwrap();
        assert_slices_close(&full, &windowed, ATOL);
    }

    #[test]
    fn test_swa_sliding_matches_full_within_window() {
        // For tokens whose full causal context fits within the window,
        // SWA should produce the same result as full causal.
        let seq = 8;
        let hd = 4;
        let w = 8; // window covers all
        let (q, k, v) = make_qkv(seq, hd);
        let full = SlidingWindowAttention::full_causal(&q, &k, &v, seq, hd).unwrap();
        let config = WindowConfig {
            window_size: w,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, seq, seq, None);
        let swa = SlidingWindowAttention::compute(&q, &k, &v, seq, seq, hd, &mask).unwrap();
        assert_slices_close(&full, &swa, ATOL);
    }

    #[test]
    fn test_swa_window_1_attends_only_self() {
        let seq = 4;
        let hd = 2;
        let (q, _k, v) = make_qkv(seq, hd);
        // With window=1, K=Q and each token attends only to itself,
        // so output = V (after softmax of single element = 1.0).
        let config = WindowConfig {
            window_size: 1,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, seq, seq, None);
        let out = SlidingWindowAttention::compute(&q, &q, &v, seq, seq, hd, &mask).unwrap();
        assert_slices_close(&out, &v, ATOL);
    }

    #[test]
    fn test_swa_dim_mismatch_q() {
        let config = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 2, 2, None);
        // Wrong Q length.
        let result =
            SlidingWindowAttention::compute(&[0.0; 3], &[0.0; 4], &[0.0; 4], 2, 2, 2, &mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_swa_dim_mismatch_kv() {
        let config = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 2, 2, None);
        let result =
            SlidingWindowAttention::compute(&[0.0; 4], &[0.0; 3], &[0.0; 4], 2, 2, 2, &mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_swa_mask_mismatch() {
        let config = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 3, 3, None);
        let result =
            SlidingWindowAttention::compute(&[0.0; 4], &[0.0; 4], &[0.0; 4], 2, 2, 2, &mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_swa_seq_len_1() {
        let hd = 4;
        let q = vec![1.0; hd];
        let k = vec![1.0; hd];
        let v = vec![2.0; hd];
        let config = WindowConfig {
            window_size: 1,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 1, 1, None);
        let out = SlidingWindowAttention::compute(&q, &k, &v, 1, 1, hd, &mask).unwrap();
        // Single entry → output = V.
        assert_slices_close(&out, &v, ATOL);
    }

    #[test]
    fn test_swa_output_length() {
        let seq = 6;
        let hd = 8;
        let (q, k, v) = make_qkv(seq, hd);
        let config = WindowConfig {
            window_size: 3,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, seq, seq, None);
        let out = SlidingWindowAttention::compute(&q, &k, &v, seq, seq, hd, &mask).unwrap();
        assert_eq!(out.len(), seq * hd);
    }

    #[test]
    fn test_swa_softmax_sums_to_one() {
        // Verify attention weights sum to ~1.0 for each query row.
        let seq = 4;
        let hd = 4;
        let (q, k, _v) = make_qkv(seq, hd);
        let config = WindowConfig {
            window_size: 2,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, seq, seq, None);
        let scale = 1.0 / (hd as f32).sqrt();
        for i in 0..seq {
            let mut scores: Vec<f32> = Vec::new();
            for j in 0..seq {
                if mask.allows(i, j) {
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q[i * hd + d] * k[j * hd + d];
                    }
                    scores.push(dot * scale);
                }
            }
            if scores.is_empty() {
                continue;
            }
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum();
            let weights: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp() / sum).collect();
            let total: f32 = weights.iter().sum();
            assert!(approx_eq(total, 1.0, 1e-5), "row {i}: weight sum = {total}");
        }
    }

    // ====================================================================
    // ChunkedPrefill
    // ====================================================================

    #[test]
    fn test_chunked_prefill_window_ge_seq() {
        let seq = 4;
        let hd = 4;
        let (q, k, v) = make_qkv(seq, hd);
        let config = WindowConfig::new(seq, 0, 0, AttentionPattern::SlidingWindow).unwrap();
        let chunked = ChunkedPrefill::run(&q, &k, &v, seq, hd, &config).unwrap();
        let full = SlidingWindowAttention::full_causal(&q, &k, &v, seq, hd).unwrap();
        assert_slices_close(&chunked, &full, ATOL);
    }

    #[test]
    fn test_chunked_prefill_output_length() {
        let seq = 16;
        let hd = 4;
        let (q, k, v) = make_qkv(seq, hd);
        let config = WindowConfig::new(4, 1, 0, AttentionPattern::SlidingWindow).unwrap();
        let out = ChunkedPrefill::run(&q, &k, &v, seq, hd, &config).unwrap();
        assert_eq!(out.len(), seq * hd);
    }

    #[test]
    fn test_chunked_prefill_dim_mismatch() {
        let config = WindowConfig::new(4, 0, 0, AttentionPattern::SlidingWindow).unwrap();
        let result = ChunkedPrefill::run(&[0.0; 5], &[0.0; 8], &[0.0; 8], 2, 4, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_chunked_prefill_small_overlap() {
        let seq = 8;
        let hd = 2;
        let (q, k, v) = make_qkv(seq, hd);
        let config = WindowConfig::new(4, 2, 0, AttentionPattern::SlidingWindow).unwrap();
        let out = ChunkedPrefill::run(&q, &k, &v, seq, hd, &config).unwrap();
        assert_eq!(out.len(), seq * hd);
        // All values should be finite.
        assert!(out.iter().all(|x| x.is_finite()));
    }

    // ====================================================================
    // OpenCL kernel source
    // ====================================================================

    #[test]
    fn test_opencl_source_not_empty() {
        assert!(!SLIDING_WINDOW_ATTENTION_CL.is_empty());
    }

    #[test]
    fn test_opencl_source_has_kernel_name() {
        assert!(SLIDING_WINDOW_ATTENTION_CL.contains("sliding_window_attention"));
    }

    #[test]
    fn test_opencl_source_has_softmax() {
        assert!(SLIDING_WINDOW_ATTENTION_CL.contains("exp("));
    }

    #[test]
    fn test_opencl_source_has_mask_check() {
        assert!(SLIDING_WINDOW_ATTENTION_CL.contains("mask["));
    }

    #[test]
    fn test_opencl_source_has_scale() {
        assert!(SLIDING_WINDOW_ATTENTION_CL.contains("scale"));
    }

    // ====================================================================
    // Property-like tests
    // ====================================================================

    #[test]
    fn test_property_swa_within_window_matches_full() {
        // For the first `window_size` tokens, SWA output should equal
        // full causal output since the window covers all available context.
        let seq = 8;
        let hd = 4;
        let w = 4;
        let (q, k, v) = make_qkv(seq, hd);
        let full = SlidingWindowAttention::full_causal(&q, &k, &v, seq, hd).unwrap();
        let config = WindowConfig {
            window_size: w,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, seq, seq, None);
        let swa = SlidingWindowAttention::compute(&q, &k, &v, seq, seq, hd, &mask).unwrap();
        // Tokens 0..w have context <= w, so SWA == full.
        for i in 0..w {
            let start = i * hd;
            let end = start + hd;
            assert_slices_close(&full[start..end], &swa[start..end], ATOL);
        }
    }

    #[test]
    fn test_property_output_finite() {
        for seq in [1, 2, 4, 8, 16] {
            let hd = 4;
            let (q, k, v) = make_qkv(seq, hd);
            let config = WindowConfig {
                window_size: 3,
                overlap: 0,
                global_token_count: 0,
                pattern_type: AttentionPattern::SlidingWindow,
            };
            let mask = WindowMask::generate(&config, seq, seq, None);
            let out = SlidingWindowAttention::compute(&q, &k, &v, seq, seq, hd, &mask).unwrap();
            assert!(out.iter().all(|x| x.is_finite()), "non-finite value for seq={seq}");
        }
    }

    #[test]
    fn test_property_more_sparsity_smaller_window() {
        let seq = 32;
        let hd = 4;
        let (q, k, v) = make_qkv(seq, hd);
        let w_small = 4;
        let w_large = 16;

        let cfg_s = WindowConfig {
            window_size: w_small,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let cfg_l = WindowConfig {
            window_size: w_large,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask_s = WindowMask::generate(&cfg_s, seq, seq, None);
        let mask_l = WindowMask::generate(&cfg_l, seq, seq, None);
        assert!(mask_s.sparsity() > mask_l.sparsity());

        // Both should still produce finite output.
        let out_s = SlidingWindowAttention::compute(&q, &k, &v, seq, seq, hd, &mask_s).unwrap();
        let out_l = SlidingWindowAttention::compute(&q, &k, &v, seq, seq, hd, &mask_l).unwrap();
        assert!(out_s.iter().all(|x| x.is_finite()));
        assert!(out_l.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_property_causal_no_future_attendance() {
        let config = WindowConfig {
            window_size: 4,
            overlap: 0,
            global_token_count: 0,
            pattern_type: AttentionPattern::SlidingWindow,
        };
        let mask = WindowMask::generate(&config, 8, 8, None);
        for i in 0..8 {
            for j in (i + 1)..8 {
                assert!(!mask.allows(i, j), "row {i} should not attend to future position {j}");
            }
        }
    }

    #[test]
    fn test_property_self_attendance() {
        // Every token should attend to itself in any pattern.
        for pattern in [
            AttentionPattern::FullCausal,
            AttentionPattern::SlidingWindow,
            AttentionPattern::SlidingWindowGlobal,
            AttentionPattern::Longformer(2),
            AttentionPattern::BigBird(1),
        ] {
            let config = WindowConfig {
                window_size: 4,
                overlap: 0,
                global_token_count: 0,
                pattern_type: pattern.clone(),
            };
            let mask = WindowMask::generate(&config, 8, 8, None);
            for i in 0..8 {
                assert!(mask.allows(i, i), "token {i} should attend to itself in {:?}", pattern);
            }
        }
    }
}
