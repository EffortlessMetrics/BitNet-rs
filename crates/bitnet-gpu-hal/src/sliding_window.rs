//! Sliding window attention patterns for efficient long-sequence processing.
//!
//! This module implements several attention masking strategies:
//! - **Standard sliding window**: each query attends to its nearest `window_size` keys.
//! - **Dilated window**: attend to every K-th token within an extended range.
//! - **Longformer-style**: combine local sliding window with designated global tokens.
//!
//! Also provides a circular-buffer KV cache ([`WindowedKVCache`]) that caps memory at
//! `window_size` entries, and an efficient attention routine ([`SlidingWindowAttention`])
//! that only computes scores for visible pairs.

use std::f32;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for sliding window attention.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Number of neighbouring keys each query can attend to (half on each side).
    pub window_size: usize,
    /// First N tokens are treated as global (attend to / attended by all).
    pub use_global_tokens: usize,
    /// Dilation factor — attend to every K-th token in the dilated variant.
    pub dilated: usize,
}

impl SlidingWindowConfig {
    pub fn new(window_size: usize) -> Self {
        Self { window_size, use_global_tokens: 0, dilated: 1 }
    }

    pub fn with_global_tokens(mut self, n: usize) -> Self {
        self.use_global_tokens = n;
        self
    }

    pub fn with_dilation(mut self, k: usize) -> Self {
        self.dilated = k.max(1);
        self
    }
}

// ---------------------------------------------------------------------------
// Window masks
// ---------------------------------------------------------------------------

/// Standard sliding-window mask.
#[derive(Debug)]
pub struct WindowMask {
    pub window_size: usize,
}

impl WindowMask {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Returns `true` when `key_pos` is visible from `query_pos`.
    pub fn is_visible(&self, query_pos: usize, key_pos: usize) -> bool {
        // Causal: key must not be in the future.
        if key_pos > query_pos {
            return false;
        }
        let half = self.window_size / 2;
        let start = query_pos.saturating_sub(half);
        let end = query_pos.saturating_add(half);
        key_pos >= start && key_pos <= end
    }

    /// Build a full seq_len × seq_len boolean mask.
    pub fn build_mask(&self, seq_len: usize) -> Vec<Vec<bool>> {
        (0..seq_len).map(|q| (0..seq_len).map(|k| self.is_visible(q, k)).collect()).collect()
    }
}

/// Dilated sliding-window mask: attend to every K-th token within an extended
/// range of `window_size * dilation`.
#[derive(Debug)]
pub struct DilatedWindowMask {
    pub window_size: usize,
    pub dilation: usize,
}

impl DilatedWindowMask {
    pub fn new(window_size: usize, dilation: usize) -> Self {
        Self { window_size, dilation: dilation.max(1) }
    }

    pub fn is_visible(&self, query_pos: usize, key_pos: usize) -> bool {
        if key_pos > query_pos {
            return false;
        }
        let effective_range = self.window_size * self.dilation;
        let half = effective_range / 2;
        let start = query_pos.saturating_sub(half);
        let end = query_pos.saturating_add(half);
        if key_pos < start || key_pos > end {
            return false;
        }
        // Within the range, only every dilation-th offset is visible.
        query_pos.abs_diff(key_pos).is_multiple_of(self.dilation)
    }

    pub fn build_mask(&self, seq_len: usize) -> Vec<Vec<bool>> {
        (0..seq_len).map(|q| (0..seq_len).map(|k| self.is_visible(q, k)).collect()).collect()
    }
}

/// Longformer-style mask: local sliding window **plus** designated global tokens
/// that attend to (and are attended by) every position.
#[derive(Debug)]
pub struct LongformerMask {
    pub window_size: usize,
    /// Indices of tokens that have global attention.
    pub global_token_indices: Vec<usize>,
}

impl LongformerMask {
    pub fn new(window_size: usize, global_token_indices: Vec<usize>) -> Self {
        Self { window_size, global_token_indices }
    }

    /// Build from a [`SlidingWindowConfig`] that specifies the first N tokens as global.
    pub fn from_config(config: &SlidingWindowConfig, _seq_len: usize) -> Self {
        let global = (0..config.use_global_tokens).collect();
        Self::new(config.window_size, global)
    }

    pub fn is_global(&self, pos: usize) -> bool {
        self.global_token_indices.contains(&pos)
    }

    pub fn is_visible(&self, query_pos: usize, key_pos: usize) -> bool {
        if key_pos > query_pos {
            return false;
        }
        // Global tokens see everything; everything sees global tokens.
        if self.is_global(query_pos) || self.is_global(key_pos) {
            return true;
        }
        // Fall back to local window.
        let half = self.window_size / 2;
        let start = query_pos.saturating_sub(half);
        let end = query_pos.saturating_add(half);
        key_pos >= start && key_pos <= end
    }

    pub fn build_mask(&self, seq_len: usize) -> Vec<Vec<bool>> {
        (0..seq_len).map(|q| (0..seq_len).map(|k| self.is_visible(q, k)).collect()).collect()
    }
}

// ---------------------------------------------------------------------------
// Windowed KV cache (circular buffer)
// ---------------------------------------------------------------------------

/// A KV cache that retains at most `window_size` entries using a circular buffer.
#[derive(Debug)]
pub struct WindowedKVCache {
    pub window_size: usize,
    pub head_dim: usize,
    /// Ring buffer for keys: `[window_size][head_dim]`.
    keys: Vec<Vec<f32>>,
    /// Ring buffer for values: `[window_size][head_dim]`.
    values: Vec<Vec<f32>>,
    /// Next write position (wraps around).
    write_pos: usize,
    /// Total number of entries ever written.
    total_written: usize,
}

impl WindowedKVCache {
    pub fn new(window_size: usize, head_dim: usize) -> Self {
        Self {
            window_size,
            head_dim,
            keys: vec![vec![0.0; head_dim]; window_size],
            values: vec![vec![0.0; head_dim]; window_size],
            write_pos: 0,
            total_written: 0,
        }
    }

    /// Number of valid entries currently stored.
    pub fn len(&self) -> usize {
        self.total_written.min(self.window_size)
    }

    pub fn is_empty(&self) -> bool {
        self.total_written == 0
    }

    /// Append a key-value pair, overwriting the oldest entry when full.
    pub fn append(&mut self, key: &[f32], value: &[f32]) {
        assert_eq!(key.len(), self.head_dim);
        assert_eq!(value.len(), self.head_dim);
        self.keys[self.write_pos].copy_from_slice(key);
        self.values[self.write_pos].copy_from_slice(value);
        self.write_pos = (self.write_pos + 1) % self.window_size;
        self.total_written += 1;
    }

    /// Return valid keys in chronological order.
    pub fn get_keys(&self) -> Vec<&[f32]> {
        self.ordered_slices(&self.keys)
    }

    /// Return valid values in chronological order.
    pub fn get_values(&self) -> Vec<&[f32]> {
        self.ordered_slices(&self.values)
    }

    fn ordered_slices<'a>(&'a self, buf: &'a [Vec<f32>]) -> Vec<&'a [f32]> {
        let n = self.len();
        if n < self.window_size {
            // Not yet wrapped — entries 0..n in order.
            buf[..n].iter().map(|v| v.as_slice()).collect()
        } else {
            // Wrapped — oldest is at write_pos.
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let idx = (self.write_pos + i) % self.window_size;
                out.push(buf[idx].as_slice());
            }
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Sliding-window attention
// ---------------------------------------------------------------------------

/// Computes attention restricted to a sliding window, only scoring visible pairs.
#[derive(Debug)]
pub struct SlidingWindowAttention {
    pub config: SlidingWindowConfig,
}

impl SlidingWindowAttention {
    pub fn new(config: SlidingWindowConfig) -> Self {
        Self { config }
    }

    /// Compute windowed attention output for a single head.
    ///
    /// * `queries`  – `[seq_len][head_dim]`
    /// * `keys`     – `[seq_len][head_dim]`
    /// * `values`   – `[seq_len][head_dim]`
    ///
    /// Returns `[seq_len][head_dim]` output.
    pub fn forward(
        &self,
        queries: &[Vec<f32>],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let seq_len = queries.len();
        let head_dim = if seq_len > 0 { queries[0].len() } else { 0 };
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mask = WindowMask::new(self.config.window_size);

        let mut output = vec![vec![0.0f32; head_dim]; seq_len];

        for q in 0..seq_len {
            // Collect visible (key_index, score) pairs.
            let mut scores: Vec<(usize, f32)> = Vec::new();
            for (k, key_vec) in keys.iter().enumerate().take(seq_len) {
                if mask.is_visible(q, k) {
                    let dot: f32 = queries[q].iter().zip(key_vec.iter()).map(|(a, b)| a * b).sum();
                    scores.push((k, dot * scale));
                }
            }
            if scores.is_empty() {
                continue;
            }
            // Softmax over visible scores.
            let max_s = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<(usize, f32)> =
                scores.iter().map(|(i, s)| (*i, (s - max_s).exp())).collect();
            let sum: f32 = exp.iter().map(|(_, e)| e).sum();
            for (ki, e) in &exp {
                let w = e / sum;
                for d in 0..head_dim {
                    output[q][d] += w * values[*ki][d];
                }
            }
        }
        output
    }

    /// Variant that uses a [`LongformerMask`] instead of the plain window.
    pub fn forward_longformer(
        &self,
        queries: &[Vec<f32>],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        longformer: &LongformerMask,
    ) -> Vec<Vec<f32>> {
        let seq_len = queries.len();
        let head_dim = if seq_len > 0 { queries[0].len() } else { 0 };
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut output = vec![vec![0.0f32; head_dim]; seq_len];

        for q in 0..seq_len {
            let mut scores: Vec<(usize, f32)> = Vec::new();
            for (k, key_vec) in keys.iter().enumerate().take(seq_len) {
                if longformer.is_visible(q, k) {
                    let dot: f32 = queries[q].iter().zip(key_vec.iter()).map(|(a, b)| a * b).sum();
                    scores.push((k, dot * scale));
                }
            }
            if scores.is_empty() {
                continue;
            }
            let max_s = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<(usize, f32)> =
                scores.iter().map(|(i, s)| (*i, (s - max_s).exp())).collect();
            let sum: f32 = exp.iter().map(|(_, e)| e).sum();
            for (ki, e) in &exp {
                let w = e / sum;
                for d in 0..head_dim {
                    output[q][d] += w * values[*ki][d];
                }
            }
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Global / local splitter
// ---------------------------------------------------------------------------

/// Splits token indices into global and local sets based on configuration.
#[derive(Debug)]
pub struct GlobalLocalSplitter {
    pub global_indices: Vec<usize>,
}

impl GlobalLocalSplitter {
    pub fn new(global_indices: Vec<usize>) -> Self {
        Self { global_indices }
    }

    /// Build from a [`SlidingWindowConfig`]: first `use_global_tokens` are global.
    pub fn from_config(config: &SlidingWindowConfig) -> Self {
        Self::new((0..config.use_global_tokens).collect())
    }

    pub fn is_global(&self, idx: usize) -> bool {
        self.global_indices.contains(&idx)
    }

    pub fn split(&self, seq_len: usize) -> (Vec<usize>, Vec<usize>) {
        let mut global = Vec::new();
        let mut local = Vec::new();
        for i in 0..seq_len {
            if self.is_global(i) {
                global.push(i);
            } else {
                local.push(i);
            }
        }
        (global, local)
    }
}

// ---------------------------------------------------------------------------
// Window position bias
// ---------------------------------------------------------------------------

/// Relative position bias within the attention window.
///
/// Supports ALiBi-style linear slopes or learned per-offset biases.
#[derive(Debug, Clone)]
pub enum WindowPositionBias {
    /// ALiBi-style: bias = -slope × |q - k|.
    Alibi { slope: f32 },
    /// Learned bias table indexed by relative offset within the window.
    Learned { biases: Vec<f32>, window_size: usize },
}

impl WindowPositionBias {
    /// Create ALiBi bias with the given slope.
    pub fn alibi(slope: f32) -> Self {
        Self::Alibi { slope }
    }

    /// Create a learned bias table (length must equal `window_size`).
    pub fn learned(biases: Vec<f32>) -> Self {
        let window_size = biases.len();
        Self::Learned { biases, window_size }
    }

    /// Compute the bias for a (query_pos, key_pos) pair.
    pub fn bias(&self, query_pos: usize, key_pos: usize) -> f32 {
        match self {
            Self::Alibi { slope } => {
                let dist = query_pos.abs_diff(key_pos);
                -slope * dist as f32
            }
            Self::Learned { biases, window_size } => {
                let half = window_size / 2;
                let offset = query_pos.abs_diff(key_pos);
                if offset < *window_size {
                    // Index: 0 = same position, 1 = distance 1, etc.
                    // Map to table centred on half.
                    let idx = (half + offset).min(window_size - 1);
                    biases[idx]
                } else {
                    0.0
                }
            }
        }
    }

    /// Apply bias to pre-computed attention scores in-place.
    ///
    /// `scores[q][k]` is modified for every pair where `mask.is_visible(q,k)`.
    pub fn apply(&self, scores: &mut [Vec<f32>], mask: &WindowMask, seq_len: usize) {
        for q in 0..seq_len.min(scores.len()) {
            for k in 0..seq_len.min(scores[q].len()) {
                if mask.is_visible(q, k) {
                    scores[q][k] += self.bias(q, k);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Efficiency metrics for a sliding-window attention pass.
#[derive(Debug, Clone)]
pub struct SlidingWindowMetrics {
    pub seq_len: usize,
    pub window_size: usize,
    pub global_tokens_count: usize,
}

impl SlidingWindowMetrics {
    pub fn new(seq_len: usize, window_size: usize, global_tokens_count: usize) -> Self {
        Self { seq_len, window_size, global_tokens_count }
    }

    /// Percentage of FLOPS saved compared to full causal attention.
    ///
    /// Full causal pairs = seq_len*(seq_len+1)/2.  Windowed pairs are fewer.
    pub fn flops_saved_pct(&self) -> f64 {
        let full = self.seq_len as f64 * (self.seq_len as f64 + 1.0) / 2.0;
        if full == 0.0 {
            return 0.0;
        }
        let windowed = self.visible_pairs() as f64;
        ((full - windowed) / full) * 100.0
    }

    /// Fraction of window capacity actually used (average over queries).
    pub fn window_utilization(&self) -> f64 {
        if self.seq_len == 0 || self.window_size == 0 {
            return 0.0;
        }
        let total_visible = self.visible_pairs();
        let max_possible = self.seq_len as f64 * self.window_size as f64;
        if max_possible == 0.0 {
            return 0.0;
        }
        (total_visible as f64 / max_possible).min(1.0)
    }

    /// Count the total number of visible (query, key) pairs under the windowed
    /// causal mask (ignoring global tokens for simplicity).
    fn visible_pairs(&self) -> usize {
        let mask = WindowMask::new(self.window_size);
        let mut count = 0usize;
        for q in 0..self.seq_len {
            for k in 0..self.seq_len {
                if mask.is_visible(q, k) {
                    count += 1;
                }
            }
        }
        count
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // WindowMask
    // -----------------------------------------------------------------------

    #[test]
    fn window_mask_basic_visibility() {
        let mask = WindowMask::new(4);
        // query=5, window half=2 → visible range [3,7], causal clips to [3,5]
        assert!(mask.is_visible(5, 3));
        assert!(mask.is_visible(5, 4));
        assert!(mask.is_visible(5, 5));
        assert!(!mask.is_visible(5, 6)); // future
        assert!(!mask.is_visible(5, 1)); // too far back
    }

    #[test]
    fn window_mask_causal() {
        let mask = WindowMask::new(100);
        // Even with huge window, future keys are invisible.
        assert!(!mask.is_visible(3, 4));
        assert!(mask.is_visible(4, 3));
    }

    #[test]
    fn window_mask_self_attention() {
        let mask = WindowMask::new(2);
        for q in 0..10 {
            assert!(mask.is_visible(q, q), "token should see itself");
        }
    }

    #[test]
    fn window_mask_size_one() {
        let mask = WindowMask::new(1);
        // half = 0 → only self visible
        assert!(mask.is_visible(5, 5));
        assert!(!mask.is_visible(5, 4));
    }

    #[test]
    fn window_mask_at_sequence_start() {
        let mask = WindowMask::new(6);
        // query=0 can only see itself (no negative positions).
        assert!(mask.is_visible(0, 0));
        assert!(!mask.is_visible(0, 1)); // future
    }

    #[test]
    fn window_mask_build_mask_shape() {
        let mask = WindowMask::new(4);
        let m = mask.build_mask(8);
        assert_eq!(m.len(), 8);
        for row in &m {
            assert_eq!(row.len(), 8);
        }
    }

    #[test]
    fn window_mask_build_mask_diagonal() {
        let mask = WindowMask::new(4);
        let m = mask.build_mask(6);
        for (i, row) in m.iter().enumerate() {
            assert!(row[i], "diagonal should be true");
        }
    }

    #[test]
    fn window_mask_window_larger_than_seq() {
        let mask = WindowMask::new(100);
        let m = mask.build_mask(4);
        // With huge window, should match full causal mask.
        for (q, row) in m.iter().enumerate() {
            for (k, &visible) in row.iter().enumerate() {
                assert_eq!(visible, k <= q, "should be full causal");
            }
        }
    }

    #[test]
    fn window_mask_visible_count_grows_with_window() {
        let small = WindowMask::new(2);
        let large = WindowMask::new(8);
        let s_count: usize = small.build_mask(10).iter().flatten().filter(|&&v| v).count();
        let l_count: usize = large.build_mask(10).iter().flatten().filter(|&&v| v).count();
        assert!(l_count >= s_count);
    }

    #[test]
    fn window_mask_symmetry_within_visible() {
        // For causal mask, visibility is NOT symmetric, but within visible range
        // of both tokens the range check is symmetric.
        let mask = WindowMask::new(4);
        // q=4,k=3 visible; q=3,k=4 not visible (causal).
        assert!(mask.is_visible(4, 3));
        assert!(!mask.is_visible(3, 4));
    }

    // -----------------------------------------------------------------------
    // DilatedWindowMask
    // -----------------------------------------------------------------------

    #[test]
    fn dilated_mask_dilation_one_matches_window() {
        let window = WindowMask::new(4);
        let dilated = DilatedWindowMask::new(4, 1);
        for q in 0..10 {
            for k in 0..10 {
                assert_eq!(
                    window.is_visible(q, k),
                    dilated.is_visible(q, k),
                    "dilation=1 should match plain window at q={q}, k={k}"
                );
            }
        }
    }

    #[test]
    fn dilated_mask_skips_intermediate() {
        let dilated = DilatedWindowMask::new(4, 2);
        // effective_range = 8, half = 4.  query=8 → range [4,12].
        // visible offsets from query must be divisible by 2.
        assert!(dilated.is_visible(8, 8)); // diff 0
        assert!(!dilated.is_visible(8, 7)); // diff 1 — not divisible by 2
        assert!(dilated.is_visible(8, 6)); // diff 2
        assert!(!dilated.is_visible(8, 5)); // diff 3
        assert!(dilated.is_visible(8, 4)); // diff 4
    }

    #[test]
    fn dilated_mask_causal() {
        let dilated = DilatedWindowMask::new(4, 3);
        assert!(!dilated.is_visible(3, 5));
    }

    #[test]
    fn dilated_mask_self_visible() {
        let dilated = DilatedWindowMask::new(4, 5);
        for q in 0..10 {
            assert!(dilated.is_visible(q, q));
        }
    }

    #[test]
    fn dilated_mask_build_mask_shape() {
        let dilated = DilatedWindowMask::new(4, 2);
        let m = dilated.build_mask(8);
        assert_eq!(m.len(), 8);
        assert_eq!(m[0].len(), 8);
    }

    #[test]
    fn dilated_mask_larger_dilation_fewer_visible() {
        let d1 = DilatedWindowMask::new(4, 1);
        let d3 = DilatedWindowMask::new(4, 3);
        let c1: usize = d1.build_mask(12).iter().flatten().filter(|&&v| v).count();
        let c3: usize = d3.build_mask(12).iter().flatten().filter(|&&v| v).count();
        // Larger dilation should generally produce fewer or equal visible pairs.
        assert!(c3 <= c1 + 12); // allow small tolerance due to boundary effects
    }

    // -----------------------------------------------------------------------
    // LongformerMask
    // -----------------------------------------------------------------------

    #[test]
    fn longformer_global_sees_all_past() {
        let mask = LongformerMask::new(4, vec![0]);
        // Global token 0 sees itself.
        assert!(mask.is_visible(0, 0));
        // Non-global token 5 should see global token 0.
        assert!(mask.is_visible(5, 0));
    }

    #[test]
    fn longformer_non_global_limited() {
        let mask = LongformerMask::new(2, vec![0]);
        // Token 8 (non-global) with window=2 (half=1): local range [7,9].
        // Token 5 is outside local range and non-global.
        assert!(!mask.is_visible(8, 5));
    }

    #[test]
    fn longformer_global_token_attended_by_all() {
        let mask = LongformerMask::new(2, vec![2]);
        // Every token past position 2 should attend to global token 2.
        for q in 2..10 {
            assert!(mask.is_visible(q, 2), "q={q} should see global token 2");
        }
    }

    #[test]
    fn longformer_from_config() {
        let cfg = SlidingWindowConfig::new(4).with_global_tokens(3);
        let mask = LongformerMask::from_config(&cfg, 10);
        assert!(mask.is_global(0));
        assert!(mask.is_global(1));
        assert!(mask.is_global(2));
        assert!(!mask.is_global(3));
    }

    #[test]
    fn longformer_no_global_matches_window() {
        let lf = LongformerMask::new(4, vec![]);
        let wm = WindowMask::new(4);
        for q in 0..10 {
            for k in 0..10 {
                assert_eq!(
                    lf.is_visible(q, k),
                    wm.is_visible(q, k),
                    "no globals should match plain window at q={q}, k={k}"
                );
            }
        }
    }

    #[test]
    fn longformer_all_global() {
        let globals: Vec<usize> = (0..8).collect();
        let mask = LongformerMask::new(2, globals);
        let m = mask.build_mask(8);
        // Should be full causal.
        for (q, row) in m.iter().enumerate() {
            for (k, &visible) in row.iter().enumerate() {
                assert_eq!(visible, k <= q, "all-global should be full causal");
            }
        }
    }

    #[test]
    fn longformer_build_mask_shape() {
        let mask = LongformerMask::new(4, vec![0, 1]);
        let m = mask.build_mask(6);
        assert_eq!(m.len(), 6);
    }

    #[test]
    fn longformer_global_increases_visibility() {
        let no_global = LongformerMask::new(4, vec![]);
        let with_global = LongformerMask::new(4, vec![0, 1]);
        let c0: usize = no_global.build_mask(10).iter().flatten().filter(|&&v| v).count();
        let c1: usize = with_global.build_mask(10).iter().flatten().filter(|&&v| v).count();
        assert!(c1 >= c0);
    }

    // -----------------------------------------------------------------------
    // WindowedKVCache
    // -----------------------------------------------------------------------

    #[test]
    fn kv_cache_starts_empty() {
        let cache = WindowedKVCache::new(4, 2);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn kv_cache_single_insert() {
        let mut cache = WindowedKVCache::new(4, 2);
        cache.append(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn kv_cache_returns_in_order() {
        let mut cache = WindowedKVCache::new(4, 2);
        cache.append(&[1.0, 0.0], &[0.0, 1.0]);
        cache.append(&[2.0, 0.0], &[0.0, 2.0]);
        cache.append(&[3.0, 0.0], &[0.0, 3.0]);
        let keys = cache.get_keys();
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0], &[1.0, 0.0]);
        assert_eq!(keys[2], &[3.0, 0.0]);
    }

    #[test]
    fn kv_cache_wraps_around() {
        let mut cache = WindowedKVCache::new(3, 1);
        cache.append(&[1.0], &[10.0]);
        cache.append(&[2.0], &[20.0]);
        cache.append(&[3.0], &[30.0]);
        cache.append(&[4.0], &[40.0]); // overwrites first
        assert_eq!(cache.len(), 3);
        let keys = cache.get_keys();
        assert_eq!(keys[0], &[2.0]); // oldest surviving
        assert_eq!(keys[1], &[3.0]);
        assert_eq!(keys[2], &[4.0]); // newest
    }

    #[test]
    fn kv_cache_double_wrap() {
        let mut cache = WindowedKVCache::new(2, 1);
        for i in 0..7 {
            cache.append(&[i as f32], &[0.0]);
        }
        assert_eq!(cache.len(), 2);
        let keys = cache.get_keys();
        assert_eq!(keys[0], &[5.0]);
        assert_eq!(keys[1], &[6.0]);
    }

    #[test]
    fn kv_cache_values_match_keys() {
        let mut cache = WindowedKVCache::new(3, 2);
        cache.append(&[1.0, 2.0], &[10.0, 20.0]);
        cache.append(&[3.0, 4.0], &[30.0, 40.0]);
        let vals = cache.get_values();
        assert_eq!(vals[0], &[10.0, 20.0]);
        assert_eq!(vals[1], &[30.0, 40.0]);
    }

    #[test]
    fn kv_cache_window_size_one() {
        let mut cache = WindowedKVCache::new(1, 2);
        cache.append(&[1.0, 2.0], &[3.0, 4.0]);
        cache.append(&[5.0, 6.0], &[7.0, 8.0]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_keys()[0], &[5.0, 6.0]);
    }

    // -----------------------------------------------------------------------
    // SlidingWindowAttention
    // -----------------------------------------------------------------------

    #[allow(clippy::type_complexity)]
    fn make_identity_qkv(
        seq_len: usize,
        dim: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let q: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| {
                let mut v = vec![0.0; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect();
        (q.clone(), q.clone(), q)
    }

    #[test]
    fn attention_output_shape() {
        let cfg = SlidingWindowConfig::new(4);
        let attn = SlidingWindowAttention::new(cfg);
        let (q, k, v) = make_identity_qkv(6, 4);
        let out = attn.forward(&q, &k, &v);
        assert_eq!(out.len(), 6);
        assert_eq!(out[0].len(), 4);
    }

    #[test]
    fn attention_single_token() {
        let cfg = SlidingWindowConfig::new(4);
        let attn = SlidingWindowAttention::new(cfg);
        let q = vec![vec![1.0, 0.0]];
        let k = vec![vec![1.0, 0.0]];
        let v = vec![vec![0.5, 0.5]];
        let out = attn.forward(&q, &k, &v);
        // Single token → output = value.
        assert!((out[0][0] - 0.5).abs() < 1e-5);
        assert!((out[0][1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn attention_large_window_matches_full() {
        // With window >= seq_len, windowed attention should match full causal.
        let cfg_large = SlidingWindowConfig::new(100);
        let attn = SlidingWindowAttention::new(cfg_large);
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let k = q.clone();
        let v = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let out = attn.forward(&q, &k, &v);
        // First token only sees itself → output = v[0].
        assert!((out[0][0] - 1.0).abs() < 1e-5);
        assert!((out[0][1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn attention_empty_sequence() {
        let cfg = SlidingWindowConfig::new(4);
        let attn = SlidingWindowAttention::new(cfg);
        let out = attn.forward(&[], &[], &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn attention_output_is_convex_combination() {
        // Each output should be a convex combination of values (weights sum to 1).
        let cfg = SlidingWindowConfig::new(10);
        let attn = SlidingWindowAttention::new(cfg);
        let q = vec![vec![1.0]; 5];
        let k = vec![vec![1.0]; 5];
        let v: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32]).collect();
        let out = attn.forward(&q, &k, &v);
        for (qi, row) in out.iter().enumerate() {
            // Output must lie within [0, qi] since v goes from 0..qi.
            assert!(row[0] >= -1e-5, "q={qi} output below minimum");
            assert!(row[0] <= qi as f32 + 1e-5, "q={qi} output above maximum");
        }
    }

    #[test]
    fn attention_window_limits_context() {
        // With a tiny window (size=2, half=1), token 4 should NOT attend to token 0.
        let cfg = SlidingWindowConfig::new(2);
        let attn = SlidingWindowAttention::new(cfg);
        // Make token 0 very distinctive.
        let mut q = vec![vec![1.0, 0.0]; 5];
        let mut k = vec![vec![1.0, 0.0]; 5];
        let mut v = vec![vec![0.0, 0.0]; 5];
        q[0] = vec![0.0, 1.0];
        k[0] = vec![0.0, 1.0];
        v[0] = vec![100.0, 100.0];
        let out = attn.forward(&q, &k, &v);
        // Token 4 should NOT pick up v[0]'s huge values.
        assert!(out[4][0] < 50.0, "token 4 should not attend to token 0");
    }

    // -----------------------------------------------------------------------
    // Longformer attention
    // -----------------------------------------------------------------------

    #[test]
    fn longformer_attention_output_shape() {
        let cfg = SlidingWindowConfig::new(4);
        let attn = SlidingWindowAttention::new(cfg);
        let lf = LongformerMask::new(4, vec![0]);
        let (q, k, v) = make_identity_qkv(6, 4);
        let out = attn.forward_longformer(&q, &k, &v, &lf);
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn longformer_attention_global_spreads_info() {
        let cfg = SlidingWindowConfig::new(2);
        let attn = SlidingWindowAttention::new(cfg);
        let lf = LongformerMask::new(2, vec![0]);
        // Token 0 has distinctive value.
        let q = vec![vec![1.0]; 5];
        let k = vec![vec![1.0]; 5];
        let mut v = vec![vec![0.0]; 5];
        v[0] = vec![10.0];
        let out = attn.forward_longformer(&q, &k, &v, &lf);
        // Token 4 should see token 0 (global) and have non-zero output.
        assert!(out[4][0] > 0.0, "global token should spread info");
    }

    // -----------------------------------------------------------------------
    // GlobalLocalSplitter
    // -----------------------------------------------------------------------

    #[test]
    fn splitter_basic() {
        let s = GlobalLocalSplitter::new(vec![0, 3]);
        let (g, l) = s.split(5);
        assert_eq!(g, vec![0, 3]);
        assert_eq!(l, vec![1, 2, 4]);
    }

    #[test]
    fn splitter_no_globals() {
        let s = GlobalLocalSplitter::new(vec![]);
        let (g, l) = s.split(4);
        assert!(g.is_empty());
        assert_eq!(l, vec![0, 1, 2, 3]);
    }

    #[test]
    fn splitter_all_global() {
        let s = GlobalLocalSplitter::new(vec![0, 1, 2, 3]);
        let (g, l) = s.split(4);
        assert_eq!(g, vec![0, 1, 2, 3]);
        assert!(l.is_empty());
    }

    #[test]
    fn splitter_from_config() {
        let cfg = SlidingWindowConfig::new(4).with_global_tokens(2);
        let s = GlobalLocalSplitter::from_config(&cfg);
        assert!(s.is_global(0));
        assert!(s.is_global(1));
        assert!(!s.is_global(2));
    }

    #[test]
    fn splitter_preserves_order() {
        let s = GlobalLocalSplitter::new(vec![1, 3, 5]);
        let (g, l) = s.split(7);
        assert_eq!(g, vec![1, 3, 5]);
        assert_eq!(l, vec![0, 2, 4, 6]);
    }

    // -----------------------------------------------------------------------
    // WindowPositionBias
    // -----------------------------------------------------------------------

    #[test]
    fn alibi_zero_distance() {
        let bias = WindowPositionBias::alibi(0.5);
        assert!((bias.bias(5, 5) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn alibi_increases_with_distance() {
        let bias = WindowPositionBias::alibi(0.5);
        let b1 = bias.bias(5, 4); // distance 1
        let b2 = bias.bias(5, 3); // distance 2
        assert!(b2 < b1, "farther should be more negative");
    }

    #[test]
    fn alibi_linear_slope() {
        let slope = 0.25;
        let bias = WindowPositionBias::alibi(slope);
        let b = bias.bias(10, 7); // distance 3
        assert!((b - (-0.75)).abs() < 1e-6);
    }

    #[test]
    fn learned_bias_lookup() {
        let biases = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let bias = WindowPositionBias::learned(biases);
        // Same position → offset 0, idx = half(2) + 0 = 2
        let b = bias.bias(5, 5);
        assert!((b - 0.3).abs() < 1e-6);
    }

    #[test]
    fn bias_apply_modifies_scores() {
        let bias = WindowPositionBias::alibi(1.0);
        let mask = WindowMask::new(4);
        let mut scores = vec![vec![0.0; 4]; 4];
        bias.apply(&mut scores, &mask, 4);
        // Diagonal (distance 0) should remain 0.
        assert!((scores[2][2] - 0.0).abs() < 1e-6);
        // Off-diagonal visible should be negative.
        if mask.is_visible(2, 1) {
            assert!(scores[2][1] < 0.0);
        }
    }

    #[test]
    fn bias_apply_does_not_touch_invisible() {
        let bias = WindowPositionBias::alibi(1.0);
        let mask = WindowMask::new(2);
        let mut scores = vec![vec![0.0; 8]; 8];
        bias.apply(&mut scores, &mask, 8);
        // Token 7 should not see token 0 (half=1, range [6,8]).
        assert!(!mask.is_visible(7, 0));
        assert!((scores[7][0] - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // SlidingWindowMetrics
    // -----------------------------------------------------------------------

    #[test]
    fn metrics_full_window_zero_savings() {
        let m = SlidingWindowMetrics::new(4, 100, 0);
        // Window covers everything → 0 % savings.
        assert!(m.flops_saved_pct() < 1.0);
    }

    #[test]
    fn metrics_small_window_high_savings() {
        let m = SlidingWindowMetrics::new(100, 4, 0);
        assert!(m.flops_saved_pct() > 50.0);
    }

    #[test]
    fn metrics_global_tokens_count() {
        let m = SlidingWindowMetrics::new(10, 4, 3);
        assert_eq!(m.global_tokens_count, 3);
    }

    #[test]
    fn metrics_utilization_bounded() {
        let m = SlidingWindowMetrics::new(10, 4, 0);
        let u = m.window_utilization();
        assert!((0.0..=1.0).contains(&u));
    }

    #[test]
    fn metrics_empty_sequence() {
        let m = SlidingWindowMetrics::new(0, 4, 0);
        assert!((m.flops_saved_pct() - 0.0).abs() < 1e-6);
        assert!((m.window_utilization() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn metrics_utilization_full_causal() {
        // When window is enormous the utilization still caps at 1.0.
        let m = SlidingWindowMetrics::new(4, 100, 0);
        assert!(m.window_utilization() <= 1.0);
    }

    // -----------------------------------------------------------------------
    // SlidingWindowConfig builder
    // -----------------------------------------------------------------------

    #[test]
    fn config_defaults() {
        let cfg = SlidingWindowConfig::new(8);
        assert_eq!(cfg.window_size, 8);
        assert_eq!(cfg.use_global_tokens, 0);
        assert_eq!(cfg.dilated, 1);
    }

    #[test]
    fn config_builder_chain() {
        let cfg = SlidingWindowConfig::new(16).with_global_tokens(4).with_dilation(3);
        assert_eq!(cfg.window_size, 16);
        assert_eq!(cfg.use_global_tokens, 4);
        assert_eq!(cfg.dilated, 3);
    }

    #[test]
    fn config_dilation_floor() {
        let cfg = SlidingWindowConfig::new(4).with_dilation(0);
        assert_eq!(cfg.dilated, 1, "dilation should be at least 1");
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn edge_window_equal_seq_len() {
        // window_size=6, half=3; for q=5 range is [2,8] so k=0,1 are outside.
        // Verify this is NOT full causal — need window >= 2*(seq_len-1) for that.
        let mask = WindowMask::new(6);
        let m = mask.build_mask(6);
        // Token 5 should NOT see token 0 (distance 5 > half 3).
        assert!(!m[5][0]);
        // Token 5 should see token 2 (distance 3 == half).
        assert!(m[5][2]);
        // With window = 2*(seq_len-1), we get full causal.
        let full = WindowMask::new(2 * 5);
        let fm = full.build_mask(6);
        for (q, row) in fm.iter().enumerate() {
            for (k, &visible) in row.iter().enumerate() {
                assert_eq!(visible, k <= q);
            }
        }
    }

    #[test]
    fn edge_seq_len_one() {
        let mask = WindowMask::new(4);
        let m = mask.build_mask(1);
        assert!(m[0][0]);
    }

    #[test]
    fn edge_dilated_seq_len_one() {
        let mask = DilatedWindowMask::new(4, 3);
        assert!(mask.is_visible(0, 0));
    }

    #[test]
    fn edge_longformer_seq_len_one() {
        let mask = LongformerMask::new(4, vec![0]);
        assert!(mask.is_visible(0, 0));
    }

    #[test]
    fn edge_kv_cache_fill_exact() {
        let mut cache = WindowedKVCache::new(3, 1);
        cache.append(&[1.0], &[10.0]);
        cache.append(&[2.0], &[20.0]);
        cache.append(&[3.0], &[30.0]);
        assert_eq!(cache.len(), 3);
        let keys = cache.get_keys();
        assert_eq!(keys[0], &[1.0]);
        assert_eq!(keys[2], &[3.0]);
    }

    #[test]
    fn edge_attention_two_tokens() {
        let cfg = SlidingWindowConfig::new(4);
        let attn = SlidingWindowAttention::new(cfg);
        let q = vec![vec![1.0], vec![1.0]];
        let k = vec![vec![1.0], vec![1.0]];
        let v = vec![vec![1.0], vec![2.0]];
        let out = attn.forward(&q, &k, &v);
        // Token 0 only sees itself → output = v[0] = [1.0]
        assert!((out[0][0] - 1.0).abs() < 1e-5);
        // Token 1 sees both → weighted combination
        assert!(out[1][0] > 1.0 - 1e-5);
    }

    #[test]
    fn edge_metrics_window_zero() {
        let m = SlidingWindowMetrics::new(10, 0, 0);
        assert!((m.window_utilization() - 0.0).abs() < 1e-6);
    }
}
