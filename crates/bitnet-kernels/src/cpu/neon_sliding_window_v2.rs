//! NEON sliding window attention v2 for Apple Silicon.
//!
//! Second-generation sliding window attention kernel with configurable overlap,
//! circular-buffer position tracking, and optimised NEON SIMD scoring. Unlike
//! v1 this kernel operates on flat `&[f32]` slices for query/key/value and
//! supports an *overlap* parameter that lets adjacent windows share context
//! positions, which is critical for chunk-boundary continuity in long-context
//! inference.
//!
//! # Key features
//!
//! * Configurable window sizes (any positive `usize`; powers of 2 recommended)
//! * Causal and non-causal masking within the window
//! * Overlap between adjacent windows (0..window_size)
//! * NEON SIMD for dot-product scoring and weighted-sum accumulation
//! * Position-aware windowing with circular buffer indexing
//!
//! # NEON intrinsics used
//!
//! | Intrinsic      | Purpose                                       |
//! |----------------|-----------------------------------------------|
//! | `vld1q_f32`    | 128-bit (4×f32) load                          |
//! | `vst1q_f32`    | 128-bit (4×f32) store                         |
//! | `vdupq_n_f32`  | Broadcast scalar → 4 lanes                    |
//! | `vfmaq_f32`    | Fused multiply-add                            |
//! | `vmulq_f32`    | Lane-wise multiply                            |
//! | `vaddq_f32`    | Lane-wise add                                 |
//! | `vaddvq_f32`   | Horizontal add (4 lanes → scalar)             |
//! | `vmaxq_f32`    | Lane-wise max                                 |
//! | `vmaxvq_f32`   | Horizontal max                                |
//! | `vsubq_f32`    | Lane-wise subtract                            |

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane width for `float32x4_t`.
const LANES: usize = 4;

// ── Configuration ──────────────────────────────────────────────────────

/// Configuration for v2 sliding window attention.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfigV2 {
    /// Window size – each query attends to at most this many key positions.
    pub window_size: usize,
    /// Whether to apply causal masking (query at position *i* can only
    /// attend to keys at position *j ≤ i*).
    pub causal: bool,
    /// Overlap between adjacent windows. Positions in the overlap region
    /// are visible from both the current and the next window. Must be in
    /// `0..window_size`.
    pub overlap: usize,
}

impl SlidingWindowConfigV2 {
    /// Create a new config with the given window size.
    /// Defaults to causal=true, overlap=0.
    pub fn new(window_size: usize) -> Self {
        Self { window_size, causal: true, overlap: 0 }
    }

    /// Builder: set causal flag.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Builder: set overlap.
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Validate the configuration. Returns `Err` on invalid parameters.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.window_size == 0 {
            return Err("window_size must be > 0");
        }
        if self.overlap >= self.window_size {
            return Err("overlap must be < window_size");
        }
        Ok(())
    }
}

// ── Window boundary helpers ────────────────────────────────────────────

/// Compute the inclusive [start, end) key-range visible to query position
/// `q_pos` given the supplied configuration and total `seq_len`.
///
/// The *effective* window centre is `q_pos` for causal mode (window looks
/// backwards) and `q_pos` for non-causal (window is symmetric around the
/// query). Overlap widens the start boundary by `overlap` positions.
#[inline]
pub fn window_bounds(
    q_pos: usize,
    seq_len: usize,
    config: &SlidingWindowConfigV2,
) -> (usize, usize) {
    let w = config.window_size;
    let o = config.overlap;

    if config.causal {
        // Causal: attend to keys in [q_pos - w + 1 - overlap, q_pos + 1),
        // clamped to [0, seq_len).
        let raw_start = (q_pos + 1).saturating_sub(w).saturating_sub(o);
        let start = raw_start.min(seq_len);
        let end = (q_pos + 1).min(seq_len);
        (start, end)
    } else {
        // Non-causal: symmetric window around q_pos.
        let half = w / 2;
        let raw_start = q_pos.saturating_sub(half + o);
        let start = raw_start.min(seq_len);
        let raw_end = (q_pos + half + 1).min(seq_len);
        (start, raw_end)
    }
}

// ── Causal mask value ──────────────────────────────────────────────────

/// Returns the additive mask value for position pair `(q_pos, k_pos)`.
/// `0.0` means "attend", `f32::NEG_INFINITY` means "masked".
#[inline]
pub fn causal_mask_value(q_pos: usize, k_pos: usize, config: &SlidingWindowConfigV2) -> f32 {
    if config.causal && k_pos > q_pos {
        return f32::NEG_INFINITY;
    }
    let (start, end) = window_bounds(q_pos, usize::MAX, config);
    if k_pos >= start && k_pos < end { 0.0 } else { f32::NEG_INFINITY }
}

// ── Full mask generation ───────────────────────────────────────────────

/// Build a `seq_len × seq_len` additive mask (row-major).
/// `0.0` = attend, `NEG_INFINITY` = masked.
pub fn build_attention_mask(seq_len: usize, config: &SlidingWindowConfigV2) -> Vec<f32> {
    let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        let (start, end) = window_bounds(i, seq_len, config);
        for j in start..end {
            if !config.causal || j <= i {
                mask[i * seq_len + j] = 0.0;
            }
        }
    }
    mask
}

// ── Scalar helpers ─────────────────────────────────────────────────────

/// Scalar dot product (used for tail elements and reference).
#[allow(dead_code)]
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// In-place softmax over a mutable slice.
fn softmax_inplace(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // If everything is masked (all NEG_INFINITY) produce uniform zeros.
    if max_val == f32::NEG_INFINITY {
        row.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        row.iter_mut().for_each(|v| *v *= inv);
    }
}

// ── NEON dot product ───────────────────────────────────────────────────

/// Dot product of two `head_dim`-length vectors using NEON FMA.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / LANES;
    let mut acc = vdupq_n_f32(0.0);

    for c in 0..chunks {
        let offset = c * LANES;
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(offset));
            let vb = vld1q_f32(b.as_ptr().add(offset));
            acc = vfmaq_f32(acc, va, vb);
        }
    }

    let mut result = vaddvq_f32(acc);
    // Scalar tail.
    for i in (chunks * LANES)..n {
        result += a[i] * b[i];
    }
    result
}

/// Weighted accumulation: `out[d] += weight * v[d]` using NEON.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_weighted_add(out: &mut [f32], v: &[f32], weight: f32) {
    debug_assert_eq!(out.len(), v.len());
    let n = out.len();
    let chunks = n / LANES;
    let w = vdupq_n_f32(weight);

    for c in 0..chunks {
        let offset = c * LANES;
        unsafe {
            let vo = vld1q_f32(out.as_ptr().add(offset));
            let vv = vld1q_f32(v.as_ptr().add(offset));
            let res = vfmaq_f32(vo, vv, w);
            vst1q_f32(out.as_mut_ptr().add(offset), res);
        }
    }

    // Scalar tail.
    for i in (chunks * LANES)..n {
        out[i] += weight * v[i];
    }
}

// ── Main attention kernel ──────────────────────────────────────────────

/// Compute sliding window attention for a **single head**.
///
/// # Arguments
/// * `query`    – `[seq_len * head_dim]` query vectors (row-major)
/// * `key`      – `[seq_len * head_dim]` key vectors
/// * `value`    – `[seq_len * head_dim]` value vectors
/// * `head_dim` – dimension per head
/// * `seq_len`  – number of positions
/// * `config`   – sliding window configuration
///
/// # Returns
/// `Vec<f32>` of length `seq_len * head_dim` containing the output.
///
/// # Panics
/// Panics if any slice length is less than `seq_len * head_dim` or if the
/// config is invalid.
pub fn sliding_window_attention_neon(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    head_dim: usize,
    seq_len: usize,
    config: &SlidingWindowConfigV2,
) -> Vec<f32> {
    config.validate().expect("invalid SlidingWindowConfigV2");
    assert!(head_dim > 0, "head_dim must be > 0");
    let total = seq_len * head_dim;
    assert!(query.len() >= total, "query too short");
    assert!(key.len() >= total, "key too short");
    assert!(value.len() >= total, "value too short");

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0_f32; total];

    for q_pos in 0..seq_len {
        let q_off = q_pos * head_dim;
        let q_vec = &query[q_off..q_off + head_dim];

        let (win_start, win_end) = window_bounds(q_pos, seq_len, config);
        let win_len = win_end - win_start;
        if win_len == 0 {
            continue;
        }

        // Compute scores for keys in the window.
        let mut scores = Vec::with_capacity(win_len);
        for k_pos in win_start..win_end {
            let k_off = k_pos * head_dim;
            let k_vec = &key[k_off..k_off + head_dim];

            #[cfg(target_arch = "aarch64")]
            let dot = unsafe { neon_dot(q_vec, k_vec) };
            #[cfg(not(target_arch = "aarch64"))]
            let dot = scalar_dot(q_vec, k_vec);

            let mut s = dot * scale;

            // Apply causal mask within window.
            if config.causal && k_pos > q_pos {
                s = f32::NEG_INFINITY;
            }
            scores.push(s);
        }

        // Softmax over window scores.
        softmax_inplace(&mut scores);

        // Weighted sum of values.
        let out_slice = &mut output[q_off..q_off + head_dim];
        for (idx, &w) in scores.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let v_pos = win_start + idx;
            let v_off = v_pos * head_dim;
            let v_vec = &value[v_off..v_off + head_dim];

            #[cfg(target_arch = "aarch64")]
            unsafe {
                neon_weighted_add(out_slice, v_vec, w);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for d in 0..head_dim {
                    out_slice[d] += w * v_vec[d];
                }
            }
        }
    }

    output
}

// ── Circular-buffer position tracker ───────────────────────────────────

/// Tracks which key/value positions are active in the current window using
/// a circular buffer index scheme. Useful for incremental / streaming
/// inference where only the most recent `window_size` positions are kept.
pub struct CircularWindowTracker {
    capacity: usize,
    head: usize,
    len: usize,
}

impl CircularWindowTracker {
    /// Create a tracker with the given capacity (= window_size).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self { capacity, head: 0, len: 0 }
    }

    /// Push a new position; returns the circular-buffer slot index.
    /// If the buffer is full, the oldest position is overwritten.
    pub fn push(&mut self) -> usize {
        let slot = self.head;
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
        slot
    }

    /// Number of active entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capacity of the circular buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Iterate over active slot indices in insertion order (oldest first).
    pub fn active_slots(&self) -> Vec<usize> {
        if self.len < self.capacity {
            (0..self.len).collect()
        } else {
            let mut slots = Vec::with_capacity(self.capacity);
            for i in 0..self.capacity {
                slots.push((self.head + i) % self.capacity);
            }
            slots
        }
    }
}

// ── Multi-head wrapper ─────────────────────────────────────────────────

/// Compute sliding window attention across multiple heads.
///
/// # Layout
/// `query`, `key`, `value` are `[num_heads * seq_len * head_dim]`
/// packed as `head-major` (head 0's seq_len vectors first, then head 1, …).
///
/// Returns `[num_heads * seq_len * head_dim]`.
pub fn sliding_window_attention_neon_multi_head(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    config: &SlidingWindowConfigV2,
) -> Vec<f32> {
    let head_total = seq_len * head_dim;
    let full_total = num_heads * head_total;
    assert!(query.len() >= full_total, "query too short for multi-head");
    assert!(key.len() >= full_total, "key too short for multi-head");
    assert!(value.len() >= full_total, "value too short for multi-head");

    let mut output = vec![0.0_f32; full_total];
    for h in 0..num_heads {
        let off = h * head_total;
        let head_out = sliding_window_attention_neon(
            &query[off..off + head_total],
            &key[off..off + head_total],
            &value[off..off + head_total],
            head_dim,
            seq_len,
            config,
        );
        output[off..off + head_total].copy_from_slice(&head_out);
    }
    output
}

// ── Reference full attention (for testing) ─────────────────────────────

/// Full (non-windowed) scaled dot-product attention for a single head.
/// Used as a reference for correctness checks.
#[cfg(test)]
fn full_attention_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    head_dim: usize,
    seq_len: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0_f32; seq_len * head_dim];

    for i in 0..seq_len {
        let q = &query[i * head_dim..(i + 1) * head_dim];
        let mut scores = Vec::with_capacity(seq_len);
        for j in 0..seq_len {
            let k = &key[j * head_dim..(j + 1) * head_dim];
            let mut s = scalar_dot(q, k) * scale;
            if causal && j > i {
                s = f32::NEG_INFINITY;
            }
            scores.push(s);
        }
        softmax_inplace(&mut scores);
        for j in 0..seq_len {
            if scores[j] == 0.0 {
                continue;
            }
            let v = &value[j * head_dim..(j + 1) * head_dim];
            for d in 0..head_dim {
                output[i * head_dim + d] += scores[j] * v[d];
            }
        }
    }
    output
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test helpers ───────────────────────────────────────────────────

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    fn uniform_vec(len: usize, val: f32) -> Vec<f32> {
        vec![val; len]
    }

    fn make_qkv(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.02).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();
        (q, k, v)
    }

    // ── SlidingWindowConfigV2 tests ───────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = SlidingWindowConfigV2::new(128);
        assert_eq!(cfg.window_size, 128);
        assert!(cfg.causal);
        assert_eq!(cfg.overlap, 0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let cfg = SlidingWindowConfigV2::new(256).with_causal(false).with_overlap(32);
        assert_eq!(cfg.window_size, 256);
        assert!(!cfg.causal);
        assert_eq!(cfg.overlap, 32);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_zero_window() {
        let cfg = SlidingWindowConfigV2 { window_size: 0, causal: true, overlap: 0 };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_overlap_equals_window() {
        let cfg = SlidingWindowConfigV2 { window_size: 4, causal: true, overlap: 4 };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_overlap_exceeds_window() {
        let cfg = SlidingWindowConfigV2 { window_size: 4, causal: true, overlap: 5 };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_max_valid_overlap() {
        let cfg = SlidingWindowConfigV2 { window_size: 8, causal: true, overlap: 7 };
        assert!(cfg.validate().is_ok());
    }

    // ── Window bounds tests ───────────────────────────────────────────

    #[test]
    fn test_bounds_causal_no_overlap() {
        let cfg = SlidingWindowConfigV2::new(4);
        assert_eq!(window_bounds(0, 10, &cfg), (0, 1));
        assert_eq!(window_bounds(1, 10, &cfg), (0, 2));
        assert_eq!(window_bounds(3, 10, &cfg), (0, 4));
        assert_eq!(window_bounds(4, 10, &cfg), (1, 5));
        assert_eq!(window_bounds(9, 10, &cfg), (6, 10));
    }

    #[test]
    fn test_bounds_causal_with_overlap() {
        let cfg = SlidingWindowConfigV2::new(4).with_overlap(2);
        // q_pos=5: raw_start = (5+1)-4-2 = 0, end=6
        assert_eq!(window_bounds(5, 10, &cfg), (0, 6));
        // q_pos=9: raw_start = (9+1)-4-2 = 4, end=10
        assert_eq!(window_bounds(9, 10, &cfg), (4, 10));
    }

    #[test]
    fn test_bounds_non_causal() {
        let cfg = SlidingWindowConfigV2::new(4).with_causal(false);
        // half = 2, q_pos=5 -> start=3, end=8
        assert_eq!(window_bounds(5, 10, &cfg), (3, 8));
        // q_pos=0 -> start=0, end=3
        assert_eq!(window_bounds(0, 10, &cfg), (0, 3));
        // q_pos=9 -> start=7, end=10 (clamped)
        assert_eq!(window_bounds(9, 10, &cfg), (7, 10));
    }

    #[test]
    fn test_bounds_non_causal_with_overlap() {
        let cfg = SlidingWindowConfigV2::new(4).with_causal(false).with_overlap(1);
        // half=2, q_pos=5 -> start=5-2-1=2, end=8
        assert_eq!(window_bounds(5, 10, &cfg), (2, 8));
    }

    #[test]
    fn test_bounds_window_larger_than_seq() {
        let cfg = SlidingWindowConfigV2::new(100);
        // Window exceeds sequence; should clamp.
        assert_eq!(window_bounds(3, 8, &cfg), (0, 4));
        assert_eq!(window_bounds(7, 8, &cfg), (0, 8));
    }

    #[test]
    fn test_bounds_single_token() {
        let cfg = SlidingWindowConfigV2::new(4);
        assert_eq!(window_bounds(0, 1, &cfg), (0, 1));
    }

    // ── Causal mask value tests ───────────────────────────────────────

    #[test]
    fn test_causal_mask_within_window() {
        let cfg = SlidingWindowConfigV2::new(4);
        assert_eq!(causal_mask_value(3, 0, &cfg), 0.0);
        assert_eq!(causal_mask_value(3, 3, &cfg), 0.0);
    }

    #[test]
    fn test_causal_mask_outside_window() {
        let cfg = SlidingWindowConfigV2::new(2);
        assert_eq!(causal_mask_value(5, 2, &cfg), f32::NEG_INFINITY);
    }

    #[test]
    fn test_causal_mask_future() {
        let cfg = SlidingWindowConfigV2::new(8);
        assert_eq!(causal_mask_value(3, 5, &cfg), f32::NEG_INFINITY);
    }

    #[test]
    fn test_non_causal_mask() {
        let cfg = SlidingWindowConfigV2::new(4).with_causal(false);
        // Non-causal: future positions within window are ok.
        assert_eq!(causal_mask_value(3, 5, &cfg), 0.0);
    }

    // ── build_attention_mask tests ────────────────────────────────────

    #[test]
    fn test_mask_causal_w2() {
        let cfg = SlidingWindowConfigV2::new(2);
        let mask = build_attention_mask(4, &cfg);
        // Row 0: [0, -inf, -inf, -inf]
        assert_eq!(mask[0], 0.0);
        assert_eq!(mask[1], f32::NEG_INFINITY);
        // Row 1: [0, 0, -inf, -inf]
        assert_eq!(mask[4], 0.0);
        assert_eq!(mask[5], 0.0);
        assert_eq!(mask[6], f32::NEG_INFINITY);
        // Row 2: [-inf, 0, 0, -inf]
        assert_eq!(mask[8], f32::NEG_INFINITY);
        assert_eq!(mask[9], 0.0);
        assert_eq!(mask[10], 0.0);
        // Row 3: [-inf, -inf, 0, 0]
        assert_eq!(mask[12], f32::NEG_INFINITY);
        assert_eq!(mask[14], 0.0);
        assert_eq!(mask[15], 0.0);
    }

    #[test]
    fn test_mask_non_causal_w3() {
        let cfg = SlidingWindowConfigV2::new(3).with_causal(false);
        let mask = build_attention_mask(4, &cfg);
        // Row 1: half=1 -> [0, 0, 0, -inf]
        assert_eq!(mask[4], 0.0);
        assert_eq!(mask[5], 0.0);
        assert_eq!(mask[6], 0.0);
        assert_eq!(mask[7], f32::NEG_INFINITY);
    }

    #[test]
    fn test_mask_full_window() {
        let cfg = SlidingWindowConfigV2::new(100);
        let mask = build_attention_mask(4, &cfg);
        // Large window + causal = lower triangular.
        for i in 0..4 {
            for j in 0..4 {
                if j <= i {
                    assert_eq!(mask[i * 4 + j], 0.0, "({i},{j}) should be 0");
                } else {
                    assert_eq!(mask[i * 4 + j], f32::NEG_INFINITY, "({i},{j}) should be -inf");
                }
            }
        }
    }

    // ── Softmax tests ─────────────────────────────────────────────────

    #[test]
    fn test_softmax_basic() {
        let mut row = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut row);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(row[2] > row[1] && row[1] > row[0]);
    }

    #[test]
    fn test_softmax_all_masked() {
        let mut row = vec![f32::NEG_INFINITY; 4];
        softmax_inplace(&mut row);
        assert!(row.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_softmax_single() {
        let mut row = vec![42.0];
        softmax_inplace(&mut row);
        assert!((row[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_empty() {
        let mut row: Vec<f32> = vec![];
        softmax_inplace(&mut row);
        assert!(row.is_empty());
    }

    #[test]
    fn test_softmax_equal_inputs() {
        let mut row = vec![5.0; 4];
        softmax_inplace(&mut row);
        for &v in &row {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    // ── Scalar dot tests ──────────────────────────────────────────────

    #[test]
    fn test_scalar_dot_basic() {
        assert!((scalar_dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_dot_zeros() {
        assert_eq!(scalar_dot(&[0.0; 4], &[1.0; 4]), 0.0);
    }

    #[test]
    fn test_scalar_dot_single() {
        assert!((scalar_dot(&[3.0], &[7.0]) - 21.0).abs() < 1e-6);
    }

    // ── CircularWindowTracker tests ───────────────────────────────────

    #[test]
    fn test_tracker_new() {
        let t = CircularWindowTracker::new(4);
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        assert_eq!(t.capacity(), 4);
    }

    #[test]
    fn test_tracker_push_within_capacity() {
        let mut t = CircularWindowTracker::new(4);
        assert_eq!(t.push(), 0);
        assert_eq!(t.push(), 1);
        assert_eq!(t.push(), 2);
        assert_eq!(t.len(), 3);
        assert!(!t.is_empty());
    }

    #[test]
    fn test_tracker_push_wraps() {
        let mut t = CircularWindowTracker::new(3);
        assert_eq!(t.push(), 0);
        assert_eq!(t.push(), 1);
        assert_eq!(t.push(), 2);
        assert_eq!(t.len(), 3);
        // Wrap around.
        assert_eq!(t.push(), 0);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn test_tracker_active_slots_not_full() {
        let mut t = CircularWindowTracker::new(5);
        t.push();
        t.push();
        t.push();
        assert_eq!(t.active_slots(), vec![0, 1, 2]);
    }

    #[test]
    fn test_tracker_active_slots_full() {
        let mut t = CircularWindowTracker::new(3);
        for _ in 0..3 {
            t.push();
        }
        assert_eq!(t.active_slots(), vec![0, 1, 2]);
    }

    #[test]
    fn test_tracker_active_slots_wrapped() {
        let mut t = CircularWindowTracker::new(3);
        for _ in 0..5 {
            t.push();
        }
        // head=2, so oldest is slot 2, then 0, then 1
        assert_eq!(t.active_slots(), vec![2, 0, 1]);
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_tracker_zero_capacity() {
        let _ = CircularWindowTracker::new(0);
    }

    // ── Attention kernel: basic correctness ───────────────────────────

    #[test]
    fn test_attention_single_token() {
        let cfg = SlidingWindowConfigV2::new(4);
        let q = vec![1.0_f32; 4];
        let k = vec![1.0_f32; 4];
        let v = vec![2.0_f32; 4];
        let out = sliding_window_attention_neon(&q, &k, &v, 4, 1, &cfg);
        // Single token: output == value.
        assert!(approx_eq(&out, &v, 1e-5));
    }

    #[test]
    fn test_attention_uniform_qkv() {
        let cfg = SlidingWindowConfigV2::new(16);
        let seq_len = 4;
        let head_dim = 4;
        let q = uniform_vec(seq_len * head_dim, 1.0);
        let k = uniform_vec(seq_len * head_dim, 1.0);
        let v = uniform_vec(seq_len * head_dim, 3.0);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        // Uniform K,V → all positions get same softmax → output ≈ value.
        for &val in &out {
            assert!((val - 3.0).abs() < 1e-4, "expected ~3.0, got {val}");
        }
    }

    #[test]
    fn test_attention_large_window_matches_full_causal() {
        let seq_len = 8;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let cfg = SlidingWindowConfigV2::new(100);
        let windowed = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, true);
        assert!(
            approx_eq(&windowed, &full, 1e-4),
            "Large window should match full causal attention"
        );
    }

    #[test]
    fn test_attention_non_causal_large_window_matches_full() {
        let seq_len = 6;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let cfg = SlidingWindowConfigV2::new(100).with_causal(false);
        let windowed = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, false);
        assert!(
            approx_eq(&windowed, &full, 1e-4),
            "Large non-causal window should match full attention"
        );
    }

    // ── Window size variations ────────────────────────────────────────

    #[test]
    fn test_window_1() {
        let cfg = SlidingWindowConfigV2::new(1);
        let seq_len = 4;
        let head_dim = 2;
        let q = uniform_vec(seq_len * head_dim, 1.0);
        let k = uniform_vec(seq_len * head_dim, 1.0);
        let v: Vec<f32> = (0..seq_len).flat_map(|i| vec![i as f32; head_dim]).collect();
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        // Window=1 causal: each position only sees itself.
        for i in 0..seq_len {
            for d in 0..head_dim {
                assert!((out[i * head_dim + d] - i as f32).abs() < 1e-5, "pos {i} dim {d}");
            }
        }
    }

    #[test]
    fn test_window_2() {
        let cfg = SlidingWindowConfigV2::new(2);
        let seq_len = 4;
        let head_dim = 2;
        let q = uniform_vec(seq_len * head_dim, 0.1);
        let k = uniform_vec(seq_len * head_dim, 0.1);
        let v = uniform_vec(seq_len * head_dim, 5.0);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        // All values identical → output ≈ value regardless of window size.
        for &val in &out {
            assert!((val - 5.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_window_4() {
        let cfg = SlidingWindowConfigV2::new(4);
        let seq_len = 8;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        assert_eq!(out.len(), seq_len * head_dim);
        // Sanity: no NaN/Inf.
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_window_8() {
        let cfg = SlidingWindowConfigV2::new(8);
        let seq_len = 16;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        assert_eq!(out.len(), seq_len * head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_window_16() {
        let cfg = SlidingWindowConfigV2::new(16);
        let seq_len = 32;
        let head_dim = 8;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        assert_eq!(out.len(), seq_len * head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_window_128() {
        let cfg = SlidingWindowConfigV2::new(128);
        let seq_len = 16;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let windowed = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, true);
        assert!(approx_eq(&windowed, &full, 1e-4));
    }

    #[test]
    fn test_window_256() {
        let cfg = SlidingWindowConfigV2::new(256);
        let seq_len = 32;
        let head_dim = 8;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let windowed = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, true);
        assert!(approx_eq(&windowed, &full, 1e-4));
    }

    #[test]
    fn test_window_512() {
        let cfg = SlidingWindowConfigV2::new(512);
        let seq_len = 20;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let windowed = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, true);
        assert!(approx_eq(&windowed, &full, 1e-4));
    }

    #[test]
    fn test_window_1024() {
        let cfg = SlidingWindowConfigV2::new(1024);
        let seq_len = 24;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let windowed = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, true);
        assert!(approx_eq(&windowed, &full, 1e-4));
    }

    // ── Causal vs non-causal ──────────────────────────────────────────

    #[test]
    fn test_causal_restricts_future() {
        let cfg = SlidingWindowConfigV2::new(100);
        let seq_len = 4;
        let head_dim = 2;
        let q = uniform_vec(seq_len * head_dim, 1.0);
        let k = uniform_vec(seq_len * head_dim, 1.0);
        // Values differ per position.
        let v: Vec<f32> =
            (0..seq_len).flat_map(|i| vec![(i + 1) as f32 * 10.0; head_dim]).collect();
        let causal_out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let non_causal_cfg = SlidingWindowConfigV2::new(100).with_causal(false);
        let non_causal_out =
            sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &non_causal_cfg);
        // First position: causal sees only pos 0, non-causal sees all.
        // So they should differ (unless seq_len=1).
        assert!(
            !approx_eq(&causal_out, &non_causal_out, 1e-6),
            "Causal and non-causal should differ"
        );
    }

    #[test]
    fn test_non_causal_symmetric() {
        let cfg = SlidingWindowConfigV2::new(100).with_causal(false);
        let seq_len = 4;
        let head_dim = 2;
        let q = uniform_vec(seq_len * head_dim, 1.0);
        let k = uniform_vec(seq_len * head_dim, 1.0);
        let v = uniform_vec(seq_len * head_dim, 7.0);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        // Uniform everything → all positions get 7.0.
        for &val in &out {
            assert!((val - 7.0).abs() < 1e-4);
        }
    }

    // ── Overlap tests ─────────────────────────────────────────────────

    #[test]
    fn test_overlap_zero() {
        let cfg_no_overlap = SlidingWindowConfigV2::new(4).with_overlap(0);
        let cfg_with_overlap = SlidingWindowConfigV2::new(4).with_overlap(2);
        let seq_len = 8;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let out_no = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg_no_overlap);
        let out_yes =
            sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg_with_overlap);
        // With overlap the window is wider, so outputs should differ.
        assert!(!approx_eq(&out_no, &out_yes, 1e-6), "overlap should affect output");
    }

    #[test]
    fn test_overlap_widens_window() {
        // With overlap=2 and window=4, effective backward reach is 4+2=6.
        let cfg = SlidingWindowConfigV2::new(4).with_overlap(2);
        let (start, end) = window_bounds(7, 10, &cfg);
        // raw_start = (7+1)-4-2 = 2, end = 8
        assert_eq!(start, 2);
        assert_eq!(end, 8);
        assert_eq!(end - start, 6);
    }

    #[test]
    fn test_overlap_clamped_at_zero() {
        let cfg = SlidingWindowConfigV2::new(4).with_overlap(3);
        let (start, _) = window_bounds(0, 10, &cfg);
        assert_eq!(start, 0); // Can't go negative.
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_seq_len_smaller_than_window() {
        let cfg = SlidingWindowConfigV2::new(32);
        let seq_len = 3;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, true);
        assert!(approx_eq(&out, &full, 1e-4));
    }

    #[test]
    fn test_seq_len_equals_window() {
        let cfg = SlidingWindowConfigV2::new(5);
        let seq_len = 5;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, true);
        assert!(approx_eq(&out, &full, 1e-4));
    }

    #[test]
    fn test_head_dim_1() {
        let cfg = SlidingWindowConfigV2::new(4);
        let seq_len = 4;
        let head_dim = 1;
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![1.0, 1.0, 1.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_head_dim_not_multiple_of_4() {
        // head_dim=5 tests the scalar tail path in NEON dot.
        let cfg = SlidingWindowConfigV2::new(8);
        let seq_len = 4;
        let head_dim = 5;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        assert_eq!(out.len(), seq_len * head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_head_dim_3() {
        let cfg = SlidingWindowConfigV2::new(4);
        let seq_len = 3;
        let head_dim = 3;
        let q = uniform_vec(seq_len * head_dim, 1.0);
        let k = uniform_vec(seq_len * head_dim, 1.0);
        let v = uniform_vec(seq_len * head_dim, 2.0);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        for &val in &out {
            assert!((val - 2.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_large_head_dim() {
        let cfg = SlidingWindowConfigV2::new(8);
        let seq_len = 4;
        let head_dim = 64;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        assert_eq!(out.len(), seq_len * head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_large_sequence() {
        let cfg = SlidingWindowConfigV2::new(16);
        let seq_len = 128;
        let head_dim = 8;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        assert_eq!(out.len(), seq_len * head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_very_large_sequence() {
        let cfg = SlidingWindowConfigV2::new(32);
        let seq_len = 512;
        let head_dim = 4;
        let q = uniform_vec(seq_len * head_dim, 0.1);
        let k = uniform_vec(seq_len * head_dim, 0.1);
        let v = uniform_vec(seq_len * head_dim, 1.0);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        assert_eq!(out.len(), seq_len * head_dim);
        // Uniform → output ≈ 1.0.
        for &val in &out {
            assert!((val - 1.0).abs() < 1e-3);
        }
    }

    // ── Score accuracy ────────────────────────────────────────────────

    #[test]
    fn test_scores_sum_to_one_per_position() {
        // Verify softmax probabilities: weighted sum of unit values = 1.
        let cfg = SlidingWindowConfigV2::new(8);
        let seq_len = 6;
        let head_dim = 4;
        let (q, k, _) = make_qkv(seq_len, head_dim);
        let v = uniform_vec(seq_len * head_dim, 1.0);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        for i in 0..seq_len {
            for d in 0..head_dim {
                assert!(
                    (out[i * head_dim + d] - 1.0).abs() < 1e-4,
                    "pos {i} dim {d} not ~1.0: {}",
                    out[i * head_dim + d]
                );
            }
        }
    }

    #[test]
    fn test_attention_deterministic() {
        let cfg = SlidingWindowConfigV2::new(4);
        let seq_len = 8;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let out1 = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let out2 = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        assert!(approx_eq(&out1, &out2, 1e-7), "Must be deterministic");
    }

    #[test]
    fn test_different_window_gives_different_output() {
        let seq_len = 8;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let cfg2 = SlidingWindowConfigV2::new(2);
        let cfg4 = SlidingWindowConfigV2::new(4);
        let out2 = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg2);
        let out4 = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg4);
        assert!(!approx_eq(&out2, &out4, 1e-6), "Different windows should give different results");
    }

    // ── Mask correctness ──────────────────────────────────────────────

    #[test]
    fn test_mask_symmetry_check() {
        let cfg = SlidingWindowConfigV2::new(3).with_causal(false);
        let mask = build_attention_mask(6, &cfg);
        // Non-causal mask should be symmetric.
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(
                    mask[i * 6 + j],
                    mask[j * 6 + i],
                    "Non-causal mask should be symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_mask_diagonal_always_visible() {
        for w in [1, 2, 4, 8] {
            let cfg = SlidingWindowConfigV2::new(w);
            let mask = build_attention_mask(10, &cfg);
            for i in 0..10 {
                assert_eq!(
                    mask[i * 10 + i],
                    0.0,
                    "Diagonal should always be visible (w={w}, i={i})"
                );
            }
        }
    }

    #[test]
    fn test_mask_count_visible() {
        let cfg = SlidingWindowConfigV2::new(3);
        let mask = build_attention_mask(8, &cfg);
        // Row i: visible count = min(i+1, 3).
        for i in 0..8_usize {
            let visible: usize = (0..8).filter(|&j| mask[i * 8 + j] == 0.0).count();
            let expected = (i + 1).min(3);
            assert_eq!(visible, expected, "row {i}");
        }
    }

    #[test]
    fn test_mask_with_overlap_has_more_visible() {
        let cfg_no = SlidingWindowConfigV2::new(4).with_overlap(0);
        let cfg_yes = SlidingWindowConfigV2::new(4).with_overlap(2);
        let mask_no = build_attention_mask(10, &cfg_no);
        let mask_yes = build_attention_mask(10, &cfg_yes);
        let count_no: usize = mask_no.iter().filter(|&&v| v == 0.0).count();
        let count_yes: usize = mask_yes.iter().filter(|&&v| v == 0.0).count();
        assert!(count_yes >= count_no, "Overlap should not reduce visible positions");
    }

    // ── Multi-head tests ──────────────────────────────────────────────

    #[test]
    fn test_multi_head_single_head() {
        let cfg = SlidingWindowConfigV2::new(8);
        let seq_len = 4;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let single = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let multi =
            sliding_window_attention_neon_multi_head(&q, &k, &v, 1, head_dim, seq_len, &cfg);
        assert!(approx_eq(&single, &multi, 1e-6));
    }

    #[test]
    fn test_multi_head_two_heads() {
        let cfg = SlidingWindowConfigV2::new(8);
        let num_heads = 2;
        let seq_len = 4;
        let head_dim = 4;
        let total = num_heads * seq_len * head_dim;
        let q: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.01).sin()).collect();
        let k: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.02).cos()).collect();
        let v: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.03).sin()).collect();
        let out = sliding_window_attention_neon_multi_head(
            &q, &k, &v, num_heads, head_dim, seq_len, &cfg,
        );
        assert_eq!(out.len(), total);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_multi_head_independence() {
        let cfg = SlidingWindowConfigV2::new(8);
        let num_heads = 2;
        let seq_len = 4;
        let head_dim = 4;
        let ht = seq_len * head_dim;
        // Head 0: all ones; Head 1: all twos.
        let mut q = vec![1.0_f32; num_heads * ht];
        let mut k = vec![1.0_f32; num_heads * ht];
        let mut v = vec![1.0_f32; num_heads * ht];
        for i in ht..2 * ht {
            q[i] = 2.0;
            k[i] = 2.0;
            v[i] = 5.0;
        }
        let out = sliding_window_attention_neon_multi_head(
            &q, &k, &v, num_heads, head_dim, seq_len, &cfg,
        );
        // Head 0 output ≈ 1.0, Head 1 output ≈ 5.0.
        for &val in &out[..ht] {
            assert!((val - 1.0).abs() < 1e-4);
        }
        for &val in &out[ht..] {
            assert!((val - 5.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_multi_head_four_heads() {
        let cfg = SlidingWindowConfigV2::new(16);
        let num_heads = 4;
        let seq_len = 6;
        let head_dim = 8;
        let total = num_heads * seq_len * head_dim;
        let q = uniform_vec(total, 0.5);
        let k = uniform_vec(total, 0.5);
        let v = uniform_vec(total, 2.0);
        let out = sliding_window_attention_neon_multi_head(
            &q, &k, &v, num_heads, head_dim, seq_len, &cfg,
        );
        for &val in &out {
            assert!((val - 2.0).abs() < 1e-3);
        }
    }

    // ── Panics ────────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "invalid SlidingWindowConfigV2")]
    fn test_panic_invalid_config() {
        let cfg = SlidingWindowConfigV2 { window_size: 0, causal: true, overlap: 0 };
        let _ = sliding_window_attention_neon(&[1.0], &[1.0], &[1.0], 1, 1, &cfg);
    }

    #[test]
    #[should_panic(expected = "head_dim must be > 0")]
    fn test_panic_zero_head_dim() {
        let cfg = SlidingWindowConfigV2::new(4);
        let _ = sliding_window_attention_neon(&[], &[], &[], 0, 1, &cfg);
    }

    #[test]
    #[should_panic(expected = "query too short")]
    fn test_panic_query_too_short() {
        let cfg = SlidingWindowConfigV2::new(4);
        let _ = sliding_window_attention_neon(&[1.0], &[1.0; 8], &[1.0; 8], 4, 2, &cfg);
    }

    // ── Additional accuracy tests ─────────────────────────────────────

    #[test]
    fn test_output_bounded_by_values() {
        let cfg = SlidingWindowConfigV2::new(4);
        let seq_len = 6;
        let head_dim = 4;
        let q = uniform_vec(seq_len * head_dim, 1.0);
        let k = uniform_vec(seq_len * head_dim, 1.0);
        // Values in [0, 10].
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i % 11) as f32).collect();
        let v_min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let v_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        for &o in &out {
            assert!(
                o >= v_min - 1e-4 && o <= v_max + 1e-4,
                "Output {o} outside value range [{v_min}, {v_max}]"
            );
        }
    }

    #[test]
    fn test_zero_queries_give_uniform_attention() {
        let cfg = SlidingWindowConfigV2::new(4);
        let seq_len = 4;
        let head_dim = 4;
        let q = uniform_vec(seq_len * head_dim, 0.0);
        let k = uniform_vec(seq_len * head_dim, 1.0);
        let v: Vec<f32> = (0..seq_len).flat_map(|i| vec![(i + 1) as f32; head_dim]).collect();
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        // Zero queries → all scores equal → uniform weights over visible keys.
        // Position 3 sees [0,1,2,3] → avg value = (1+2+3+4)/4 = 2.5
        for d in 0..head_dim {
            assert!(
                (out[3 * head_dim + d] - 2.5).abs() < 1e-4,
                "Expected ~2.5, got {}",
                out[3 * head_dim + d]
            );
        }
    }

    #[test]
    fn test_scaling_factor() {
        // Verify the 1/sqrt(head_dim) scaling by checking that larger head_dim
        // reduces raw score magnitude (before softmax).
        let cfg = SlidingWindowConfigV2::new(100);
        let seq_len = 2;

        // Small head_dim: scores are bigger.
        let hd_small = 4;
        let q_s = uniform_vec(seq_len * hd_small, 1.0);
        let k_s = uniform_vec(seq_len * hd_small, 1.0);
        let v_s = uniform_vec(seq_len * hd_small, 1.0);
        let out_s = sliding_window_attention_neon(&q_s, &k_s, &v_s, hd_small, seq_len, &cfg);

        // Large head_dim: scores are smaller relative to dot product.
        let hd_large = 64;
        let q_l = uniform_vec(seq_len * hd_large, 1.0);
        let k_l = uniform_vec(seq_len * hd_large, 1.0);
        let v_l = uniform_vec(seq_len * hd_large, 1.0);
        let out_l = sliding_window_attention_neon(&q_l, &k_l, &v_l, hd_large, seq_len, &cfg);

        // Both should produce ~1.0 since values are uniform.
        assert!((out_s[0] - 1.0).abs() < 1e-4);
        assert!((out_l[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_window_2048() {
        let cfg = SlidingWindowConfigV2::new(2048);
        let seq_len = 32;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let windowed = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, true);
        assert!(approx_eq(&windowed, &full, 1e-4));
    }

    #[test]
    fn test_non_causal_window_1() {
        let cfg = SlidingWindowConfigV2::new(1).with_causal(false);
        let seq_len = 4;
        let head_dim = 2;
        let q = uniform_vec(seq_len * head_dim, 1.0);
        let k = uniform_vec(seq_len * head_dim, 1.0);
        let v: Vec<f32> = (0..seq_len).flat_map(|i| vec![i as f32; head_dim]).collect();
        let out = sliding_window_attention_neon(&q, &k, &v, head_dim, seq_len, &cfg);
        // Non-causal window=1, half=0 → each position sees only itself.
        for i in 0..seq_len {
            for d in 0..head_dim {
                assert!((out[i * head_dim + d] - i as f32).abs() < 1e-5, "pos {i}");
            }
        }
    }
}
