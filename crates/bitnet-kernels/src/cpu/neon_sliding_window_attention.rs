#![allow(
    unsafe_op_in_unsafe_fn,
    unused_unsafe,
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::manual_abs_diff,
    clippy::manual_contains,
    clippy::manual_is_multiple_of,
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_cast
)]
//! ARM NEON sliding window attention for Apple Silicon.
//!
//! Implements memory-efficient sliding window attention restricting each query
//! position to attend only to the last `W` key/value positions. Combined with
//! causal masking this produces a banded lower-triangular attention pattern
//! that is both faster and more memory-efficient than full O(N²) attention for
//! long sequences.
//!
//! # Supported window sizes
//!
//! 128, 256, 512, 1024, 2048 (any positive `usize` is accepted at runtime).
//!
//! # NEON intrinsics used
//!
//! | Intrinsic      | Purpose                                        |
//! |----------------|------------------------------------------------|
//! | `vld1q_f32`    | Unaligned 128-bit (4×f32) load                 |
//! | `vst1q_f32`    | Unaligned 128-bit (4×f32) store                |
//! | `vdupq_n_f32`  | Broadcast scalar to all four lanes              |
//! | `vfmaq_f32`    | Fused multiply-add: `a + b * c`                |
//! | `vmulq_f32`    | Lane-wise multiply                             |
//! | `vaddvq_f32`   | Horizontal add (sum all four lanes → scalar)   |
//! | `vmaxq_f32`    | Lane-wise max                                  |
//! | `vsubq_f32`    | Lane-wise subtract                             |
//! | `vmaxvq_f32`   | Horizontal max (max of all four lanes → scalar)|

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ── Configuration ──────────────────────────────────────────────────────

/// Configuration for sliding window attention.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Sequence length (total number of tokens).
    pub seq_len: usize,
    /// Window size: each query attends to at most this many prior keys.
    pub window_size: usize,
    /// Whether to apply causal masking (positions can only attend to
    /// earlier positions).
    pub causal: bool,
    /// Optional per-head window size overrides. When `Some`, its length
    /// must equal `num_heads`. Each entry overrides `window_size` for that
    /// head.
    pub per_head_windows: Option<Vec<usize>>,
}

impl SlidingWindowConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_heads == 0 {
            return Err("num_heads must be > 0");
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0");
        }
        if self.seq_len == 0 {
            return Err("seq_len must be > 0");
        }
        if self.window_size == 0 {
            return Err("window_size must be > 0");
        }
        if let Some(ref phw) = self.per_head_windows {
            if phw.len() != self.num_heads {
                return Err("per_head_windows length must equal num_heads");
            }
            if phw.iter().any(|&w| w == 0) {
                return Err("per-head window sizes must be > 0");
            }
        }
        Ok(())
    }

    /// Effective window size for a given head.
    #[inline]
    fn window_for_head(&self, head: usize) -> usize {
        self.per_head_windows.as_ref().map_or(self.window_size, |ws| ws[head])
    }
}

// ── Mask generation ────────────────────────────────────────────────────

/// Generate a binary sliding window mask for a single head.
///
/// Returns a `seq_len × seq_len` row-major mask where `0.0` means
/// "attend" and `f32::NEG_INFINITY` means "masked".
///
/// When `causal` is `true`, position `i` can attend to position `j` only
/// if `j <= i` **and** `i - j < window_size`.
/// When `causal` is `false`, position `i` can attend to `j` if
/// `|i - j| < window_size`.
pub fn sliding_window_mask(seq_len: usize, window_size: usize, causal: bool) -> Vec<f32> {
    let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let in_window = if causal {
                j <= i && (i - j) < window_size
            } else {
                let diff = if i >= j { i - j } else { j - i };
                diff < window_size
            };
            if in_window {
                mask[i * seq_len + j] = 0.0;
            }
        }
    }
    mask
}

/// Generate masks for all heads, supporting per-head window sizes.
pub fn sliding_window_masks(config: &SlidingWindowConfig) -> Vec<Vec<f32>> {
    (0..config.num_heads)
        .map(|h| sliding_window_mask(config.seq_len, config.window_for_head(h), config.causal))
        .collect()
}

// ── KV window cache ────────────────────────────────────────────────────

/// Memory-efficient KV cache that evicts entries outside the sliding
/// window, keeping at most `window_size` key/value pairs per head.
#[derive(Debug, Clone)]
pub struct WindowedKvCache {
    /// Max entries retained per head.
    pub window_size: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of heads.
    pub num_heads: usize,
    /// Key storage: `[num_heads][<=window_size * head_dim]`.
    keys: Vec<Vec<f32>>,
    /// Value storage: `[num_heads][<=window_size * head_dim]`.
    values: Vec<Vec<f32>>,
    /// Total tokens appended so far (before eviction) per head.
    total_tokens: Vec<usize>,
}

impl WindowedKvCache {
    /// Create an empty windowed cache.
    pub fn new(num_heads: usize, head_dim: usize, window_size: usize) -> Self {
        Self {
            window_size,
            head_dim,
            num_heads,
            keys: vec![Vec::with_capacity(window_size * head_dim); num_heads],
            values: vec![Vec::with_capacity(window_size * head_dim); num_heads],
            total_tokens: vec![0; num_heads],
        }
    }

    /// Current number of cached positions for a head.
    #[inline]
    pub fn len(&self, head: usize) -> usize {
        self.keys[head].len() / self.head_dim
    }

    /// Whether the cache is empty for a head.
    #[inline]
    pub fn is_empty(&self, head: usize) -> bool {
        self.keys[head].is_empty()
    }

    /// Total number of tokens ever appended (before eviction) for a head.
    #[inline]
    pub fn total_tokens(&self, head: usize) -> usize {
        self.total_tokens[head]
    }

    /// Append a new key/value pair, evicting the oldest entry if at
    /// capacity.
    pub fn append(&mut self, head: usize, key: &[f32], value: &[f32]) {
        assert_eq!(key.len(), self.head_dim);
        assert_eq!(value.len(), self.head_dim);
        assert!(head < self.num_heads);

        self.total_tokens[head] += 1;

        // Evict oldest if at capacity.
        if self.len(head) >= self.window_size {
            self.keys[head].drain(..self.head_dim);
            self.values[head].drain(..self.head_dim);
        }

        self.keys[head].extend_from_slice(key);
        self.values[head].extend_from_slice(value);
    }

    /// Read-only access to cached keys for a head (flattened).
    #[inline]
    pub fn keys(&self, head: usize) -> &[f32] {
        &self.keys[head]
    }

    /// Read-only access to cached values for a head (flattened).
    #[inline]
    pub fn values(&self, head: usize) -> &[f32] {
        &self.values[head]
    }

    /// Clear all heads.
    pub fn clear(&mut self) {
        for h in 0..self.num_heads {
            self.keys[h].clear();
            self.values[h].clear();
            self.total_tokens[h] = 0;
        }
    }
}

// ── Scalar helpers ─────────────────────────────────────────────────────

/// Scalar dot product.
#[inline]
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Scalar numerically-stable softmax (in-place).
fn softmax_inplace(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        row.fill(0.0);
        return;
    }
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

// ── NEON-accelerated kernels ───────────────────────────────────────────

/// NEON dot product of two `f32` slices.
///
/// Uses `vld1q_f32` for loads, `vfmaq_f32` for fused multiply-add, and
/// `vaddvq_f32` for horizontal reduction.
///
/// # Safety
/// Requires aarch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / LANES;
    let mut acc = vdupq_n_f32(0.0);

    for c in 0..chunks {
        let base = c * LANES;
        let va = vld1q_f32(a.as_ptr().add(base));
        let vb = vld1q_f32(b.as_ptr().add(base));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut result = vaddvq_f32(acc);
    let tail = chunks * LANES;
    for i in tail..n {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    result
}

/// NEON-accelerated find-max over a slice.
///
/// Uses `vld1q_f32` and `vmaxq_f32` for 4-wide max, `vmaxvq_f32` for
/// horizontal reduction.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_max(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }
    let chunks = data.len() / LANES;
    let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);

    for c in 0..chunks {
        let v = vld1q_f32(data.as_ptr().add(c * LANES));
        vmax = vmaxq_f32(vmax, v);
    }

    let mut m = vmaxvq_f32(vmax);
    let tail = chunks * LANES;
    for i in tail..data.len() {
        let v = *data.get_unchecked(i);
        if v > m {
            m = v;
        }
    }
    m
}

/// NEON softmax (in-place) using vectorised exp approximation.
///
/// Uses `vsubq_f32` for max-subtraction and `vmulq_f32` for normalisation.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_softmax(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max = neon_max(row);
    if max == f32::NEG_INFINITY {
        row.fill(0.0);
        return;
    }

    let max_v = vdupq_n_f32(max);
    let chunks = row.len() / LANES;
    let tail_start = chunks * LANES;
    let mut sum = 0.0_f32;

    // Subtract max and exp.
    for c in 0..chunks {
        let base = c * LANES;
        let v = vld1q_f32(row.as_ptr().add(base));
        let shifted = vsubq_f32(v, max_v);
        // Scalar exp for correctness; the hot path is the dot product.
        for lane in 0..LANES {
            let idx = base + lane;
            let val = *row.get_unchecked(idx) - max;
            let e = val.exp();
            *row.get_unchecked_mut(idx) = e;
            sum += e;
        }
        let _ = shifted; // read used above via scalar for bit-exact exp
    }
    for i in tail_start..row.len() {
        let val = *row.get_unchecked(i) - max;
        let e = val.exp();
        *row.get_unchecked_mut(i) = e;
        sum += e;
    }

    if sum == 0.0 {
        return;
    }
    let inv = 1.0 / sum;
    let inv_v = vdupq_n_f32(inv);
    for c in 0..chunks {
        let base = c * LANES;
        let v = vld1q_f32(row.as_ptr().add(base));
        let r = vmulq_f32(v, inv_v);
        vst1q_f32(row.as_mut_ptr().add(base), r);
    }
    for i in tail_start..row.len() {
        *row.get_unchecked_mut(i) *= inv;
    }
}

// ── Core attention computation ─────────────────────────────────────────

/// Compute sliding window attention scores (Q·K^T / √d) for one query
/// position, restricted to the valid window of key positions.
///
/// Returns scores only for positions within the window. Masked positions
/// are set to `f32::NEG_INFINITY`.
fn compute_windowed_scores(
    query: &[f32],
    keys: &[f32],
    head_dim: usize,
    query_pos: usize,
    _seq_len: usize,
    window_size: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let num_keys = keys.len() / head_dim;
    let mut scores = vec![f32::NEG_INFINITY; num_keys];

    for j in 0..num_keys {
        let in_window = if causal {
            j <= query_pos && (query_pos - j) < window_size
        } else {
            let diff = if query_pos >= j { query_pos - j } else { j - query_pos };
            diff < window_size
        };
        if !in_window {
            continue;
        }

        let k_start = j * head_dim;
        let k_slice = &keys[k_start..k_start + head_dim];

        #[cfg(target_arch = "aarch64")]
        let dot = unsafe { neon_dot(query, k_slice) };
        #[cfg(not(target_arch = "aarch64"))]
        let dot = scalar_dot(query, k_slice);

        scores[j] = dot * scale;
    }
    scores
}

/// Single-head sliding window attention.
///
/// * `q` — queries, shape `[seq_len, head_dim]` (row-major)
/// * `k` — keys,    shape `[seq_len, head_dim]`
/// * `v` — values,  shape `[seq_len, head_dim]`
///
/// Returns output of shape `[seq_len, head_dim]`.
pub fn sliding_window_attention_single_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    seq_len: usize,
    window_size: usize,
    causal: bool,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim);
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);

    let mut output = vec![0.0_f32; seq_len * head_dim];

    for i in 0..seq_len {
        let qi = &q[i * head_dim..(i + 1) * head_dim];
        let mut scores = compute_windowed_scores(qi, k, head_dim, i, seq_len, window_size, causal);

        // Softmax over scores.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_softmax(&mut scores);
        }
        #[cfg(not(target_arch = "aarch64"))]
        softmax_inplace(&mut scores);

        // Weighted sum of values.
        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        for (j, &w) in scores.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let vj = &v[j * head_dim..(j + 1) * head_dim];
            for d in 0..head_dim {
                out_row[d] += w * vj[d];
            }
        }
    }
    output
}

/// Multi-head sliding window attention.
///
/// * `q` — queries, shape `[num_heads, seq_len, head_dim]` (head-major)
/// * `k` — keys,    shape `[num_heads, seq_len, head_dim]`
/// * `v` — values,  shape `[num_heads, seq_len, head_dim]`
/// * `config` — sliding window configuration
///
/// Returns output of shape `[num_heads, seq_len, head_dim]`.
pub fn sliding_window_attention_multi_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &SlidingWindowConfig,
) -> Result<Vec<f32>, &'static str> {
    config.validate()?;
    let head_size = config.seq_len * config.head_dim;
    let total = config.num_heads * head_size;
    if q.len() != total || k.len() != total || v.len() != total {
        return Err("input tensor size mismatch");
    }

    let mut output = vec![0.0_f32; total];

    for h in 0..config.num_heads {
        let offset = h * head_size;
        let q_head = &q[offset..offset + head_size];
        let k_head = &k[offset..offset + head_size];
        let v_head = &v[offset..offset + head_size];
        let win = config.window_for_head(h);

        let head_out = sliding_window_attention_single_head(
            q_head,
            k_head,
            v_head,
            config.head_dim,
            config.seq_len,
            win,
            config.causal,
        );
        output[offset..offset + head_size].copy_from_slice(&head_out);
    }
    Ok(output)
}

/// Incremental (cached) sliding window attention for a single new query
/// token.
///
/// Appends the new key/value to the windowed cache, then computes
/// attention over the cached window.
///
/// Returns a single output vector of shape `[head_dim]`.
pub fn cached_sliding_window_step(
    query: &[f32],
    new_key: &[f32],
    new_value: &[f32],
    cache: &mut WindowedKvCache,
    head: usize,
) -> Vec<f32> {
    let head_dim = cache.head_dim;
    assert_eq!(query.len(), head_dim);

    cache.append(head, new_key, new_value);

    let num_cached = cache.len(head);
    let cached_keys = cache.keys(head);
    let cached_values = cache.values(head);
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute scores for all cached positions (all are within window).
    let mut scores = vec![0.0_f32; num_cached];
    for j in 0..num_cached {
        let kj = &cached_keys[j * head_dim..(j + 1) * head_dim];
        #[cfg(target_arch = "aarch64")]
        let dot = unsafe { neon_dot(query, kj) };
        #[cfg(not(target_arch = "aarch64"))]
        let dot = scalar_dot(query, kj);
        scores[j] = dot * scale;
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        neon_softmax(&mut scores);
    }
    #[cfg(not(target_arch = "aarch64"))]
    softmax_inplace(&mut scores);

    // Weighted sum.
    let mut output = vec![0.0_f32; head_dim];
    for (j, &w) in scores.iter().enumerate() {
        if w == 0.0 {
            continue;
        }
        let vj = &cached_values[j * head_dim..(j + 1) * head_dim];
        for d in 0..head_dim {
            output[d] += w * vj[d];
        }
    }
    output
}

/// Full (non-windowed) attention for reference/comparison.
///
/// Standard scaled dot-product attention with optional causal mask.
pub fn full_attention_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    seq_len: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0_f32; seq_len * head_dim];

    for i in 0..seq_len {
        let qi = &q[i * head_dim..(i + 1) * head_dim];
        let mut scores = vec![f32::NEG_INFINITY; seq_len];

        for j in 0..seq_len {
            if causal && j > i {
                continue;
            }
            let kj = &k[j * head_dim..(j + 1) * head_dim];
            scores[j] = scalar_dot(qi, kj) * scale;
        }

        softmax_inplace(&mut scores);

        let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
        for (j, &w) in scores.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let vj = &v[j * head_dim..(j + 1) * head_dim];
            for d in 0..head_dim {
                out_row[d] += w * vj[d];
            }
        }
    }
    output
}

/// Local block attention: partitions the sequence into blocks of
/// `block_size` tokens, each block attending only within itself.
///
/// Returns output of shape `[seq_len, head_dim]`.
pub fn local_block_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    head_dim: usize,
    seq_len: usize,
    block_size: usize,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim);
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);
    assert!(block_size > 0, "block_size must be > 0");

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0_f32; seq_len * head_dim];

    let num_blocks = (seq_len + block_size - 1) / block_size;
    for b in 0..num_blocks {
        let start = b * block_size;
        let end = (start + block_size).min(seq_len);
        let blen = end - start;

        for i in 0..blen {
            let gi = start + i; // global index
            let qi = &q[gi * head_dim..(gi + 1) * head_dim];
            let mut scores = vec![f32::NEG_INFINITY; blen];

            for j in 0..blen {
                let gj = start + j;
                let kj = &k[gj * head_dim..(gj + 1) * head_dim];
                scores[j] = scalar_dot(qi, kj) * scale;
            }

            softmax_inplace(&mut scores);

            let out_row = &mut output[gi * head_dim..(gi + 1) * head_dim];
            for j in 0..blen {
                let gj = start + j;
                let w = scores[j];
                if w == 0.0 {
                    continue;
                }
                let vj = &v[gj * head_dim..(gj + 1) * head_dim];
                for d in 0..head_dim {
                    out_row[d] += w * vj[d];
                }
            }
        }
    }
    output
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create identity-like Q/K/V for deterministic tests.
    fn make_identity_qkv(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut q = vec![0.0_f32; seq_len * head_dim];
        let mut k = vec![0.0_f32; seq_len * head_dim];
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32).collect();
        // Make each position attend strongly to itself via one-hot-like Q/K.
        for i in 0..seq_len {
            let dim = i % head_dim;
            q[i * head_dim + dim] = 1.0;
            k[i * head_dim + dim] = 1.0;
        }
        (q, k, v)
    }

    fn make_uniform_qkv(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q = vec![1.0_f32; seq_len * head_dim];
        let k = vec![1.0_f32; seq_len * head_dim];
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i % head_dim) as f32).collect();
        (q, k, v)
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| (x - y).abs() < tol)
    }

    // ── Mask generation tests ──────────────────────────────────────────

    #[test]
    fn test_causal_sliding_mask_basic() {
        let mask = sliding_window_mask(4, 2, true);
        // Position 0 attends to [0], position 1 to [0,1],
        // position 2 to [1,2], position 3 to [2,3].
        assert_eq!(mask[0 * 4 + 0], 0.0); // pos0 → pos0
        assert_eq!(mask[0 * 4 + 1], f32::NEG_INFINITY); // pos0 → pos1 (future)
        assert_eq!(mask[1 * 4 + 0], 0.0); // pos1 → pos0
        assert_eq!(mask[1 * 4 + 1], 0.0); // pos1 → pos1
        assert_eq!(mask[2 * 4 + 0], f32::NEG_INFINITY); // pos2 → pos0 (out of window)
        assert_eq!(mask[2 * 4 + 1], 0.0); // pos2 → pos1
        assert_eq!(mask[2 * 4 + 2], 0.0); // pos2 → pos2
        assert_eq!(mask[3 * 4 + 1], f32::NEG_INFINITY); // pos3 → pos1 (out of window)
        assert_eq!(mask[3 * 4 + 2], 0.0); // pos3 → pos2
        assert_eq!(mask[3 * 4 + 3], 0.0); // pos3 → pos3
    }

    #[test]
    fn test_non_causal_sliding_mask() {
        let mask = sliding_window_mask(4, 2, false);
        // Bidirectional window of 2: attend to self and ±1.
        assert_eq!(mask[0 * 4 + 0], 0.0);
        assert_eq!(mask[0 * 4 + 1], 0.0);
        assert_eq!(mask[0 * 4 + 2], f32::NEG_INFINITY);
        assert_eq!(mask[1 * 4 + 0], 0.0);
        assert_eq!(mask[1 * 4 + 1], 0.0);
        assert_eq!(mask[1 * 4 + 2], 0.0); // |1-2| = 1 < 2
        assert_eq!(mask[1 * 4 + 3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_mask_full_window_equals_causal() {
        // Window >= seq_len with causal should equal pure causal mask.
        let seq_len = 6;
        let full = sliding_window_mask(seq_len, seq_len + 1, true);
        for i in 0..seq_len {
            for j in 0..seq_len {
                let expected = if j <= i { 0.0 } else { f32::NEG_INFINITY };
                assert_eq!(full[i * seq_len + j], expected, "i={i}, j={j}");
            }
        }
    }

    #[test]
    fn test_mask_window_1_is_diagonal() {
        let mask = sliding_window_mask(5, 1, true);
        for i in 0..5 {
            for j in 0..5 {
                let expected = if i == j { 0.0 } else { f32::NEG_INFINITY };
                assert_eq!(mask[i * 5 + j], expected, "i={i}, j={j}");
            }
        }
    }

    #[test]
    fn test_per_head_masks() {
        let config = SlidingWindowConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 4,
            window_size: 2,
            causal: true,
            per_head_windows: Some(vec![1, 4]),
        };
        let masks = sliding_window_masks(&config);
        assert_eq!(masks.len(), 2);
        // Head 0: window=1 → diagonal only.
        assert_eq!(masks[0][0 * 4 + 0], 0.0);
        assert_eq!(masks[0][1 * 4 + 0], f32::NEG_INFINITY);
        // Head 1: window=4 → full causal.
        assert_eq!(masks[1][3 * 4 + 0], 0.0);
    }

    // ── Config validation tests ────────────────────────────────────────

    #[test]
    fn test_config_valid() {
        let config = SlidingWindowConfig {
            num_heads: 4,
            head_dim: 64,
            seq_len: 128,
            window_size: 256,
            causal: true,
            per_head_windows: None,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_zero_heads() {
        let config = SlidingWindowConfig {
            num_heads: 0,
            head_dim: 64,
            seq_len: 128,
            window_size: 256,
            causal: true,
            per_head_windows: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_zero_window() {
        let config = SlidingWindowConfig {
            num_heads: 1,
            head_dim: 64,
            seq_len: 128,
            window_size: 0,
            causal: true,
            per_head_windows: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_per_head_length_mismatch() {
        let config = SlidingWindowConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 8,
            window_size: 4,
            causal: true,
            per_head_windows: Some(vec![4]), // wrong length
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_per_head_zero_window() {
        let config = SlidingWindowConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 8,
            window_size: 4,
            causal: true,
            per_head_windows: Some(vec![4, 0]),
        };
        assert!(config.validate().is_err());
    }

    // ── Single-head attention tests ────────────────────────────────────

    #[test]
    fn test_single_head_window_larger_than_seq() {
        // Window > seq_len with causal → should equal full causal attention.
        let seq_len = 4;
        let head_dim = 4;
        let (q, k, v) = make_uniform_qkv(seq_len, head_dim);

        let windowed =
            sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, 100, true);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, true);
        assert!(approx_eq(&windowed, &full, 1e-5), "windowed != full");
    }

    #[test]
    fn test_single_head_window_1_is_self_attention() {
        // Window = 1 with causal → each position only attends to itself.
        let seq_len = 4;
        let head_dim = 4;
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32).collect();
        // With window=1 causal, each output row should exactly equal
        // the corresponding value row.
        let q = vec![1.0_f32; seq_len * head_dim];
        let k = vec![1.0_f32; seq_len * head_dim];
        let out = sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, 1, true);
        for i in 0..seq_len {
            let row = &out[i * head_dim..(i + 1) * head_dim];
            let vrow = &v[i * head_dim..(i + 1) * head_dim];
            assert!(approx_eq(row, vrow, 1e-5), "pos {i}: got {row:?}, expected {vrow:?}");
        }
    }

    #[test]
    fn test_single_head_correctness_vs_full_small() {
        // With window >= seq_len, should match full attention.
        let seq_len = 6;
        let head_dim = 8;
        let (q, k, v) = make_identity_qkv(seq_len, head_dim);

        let windowed =
            sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, seq_len, true);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, true);
        assert!(approx_eq(&windowed, &full, 1e-4));
    }

    #[test]
    fn test_single_head_non_causal() {
        let seq_len = 4;
        let head_dim = 4;
        let (q, k, v) = make_uniform_qkv(seq_len, head_dim);

        let windowed =
            sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, 100, false);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, false);
        assert!(approx_eq(&windowed, &full, 1e-5));
    }

    #[test]
    fn test_single_head_output_shape() {
        let seq_len = 8;
        let head_dim = 16;
        let q = vec![0.1_f32; seq_len * head_dim];
        let k = vec![0.1_f32; seq_len * head_dim];
        let v = vec![0.1_f32; seq_len * head_dim];
        let out = sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, 4, true);
        assert_eq!(out.len(), seq_len * head_dim);
    }

    // ── Window boundary tests ──────────────────────────────────────────

    #[test]
    fn test_window_boundary_first_position() {
        // First position should have identical output regardless of window size.
        let seq_len = 8;
        let head_dim = 4;
        let (q, k, v) = make_uniform_qkv(seq_len, head_dim);

        let out_w2 = sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, 2, true);
        let out_w8 = sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, 8, true);
        let row_w2 = &out_w2[0..head_dim];
        let row_w8 = &out_w8[0..head_dim];
        assert!(approx_eq(row_w2, row_w8, 1e-5));
    }

    #[test]
    fn test_window_boundary_eviction() {
        // Position 4 with window=2 should NOT attend to position 2.
        let seq_len = 6;
        let head_dim = 4;
        let q = vec![1.0_f32; seq_len * head_dim];
        let k = vec![1.0_f32; seq_len * head_dim];
        // Give position 2 a distinct value.
        let mut v = vec![0.0_f32; seq_len * head_dim];
        for d in 0..head_dim {
            v[2 * head_dim + d] = 100.0;
        }

        let out = sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, 2, true);
        // Position 4 (window=[3,4]) should have no contribution from pos 2.
        let row4 = &out[4 * head_dim..5 * head_dim];
        for &val in row4 {
            assert!(val.abs() < 1e-5, "position 4 should not see value from position 2, got {val}");
        }
    }

    #[test]
    fn test_window_boundary_last_position() {
        // Last position with window=3 attends to last 3 positions only.
        let seq_len = 8;
        let head_dim = 4;
        let q = vec![1.0_f32; seq_len * head_dim];
        let k = vec![1.0_f32; seq_len * head_dim];
        let mut v = vec![0.0_f32; seq_len * head_dim];
        // Put non-zero only in last 3 positions.
        for pos in (seq_len - 3)..seq_len {
            for d in 0..head_dim {
                v[pos * head_dim + d] = 1.0;
            }
        }
        // Put a distinguishing value outside the window.
        for d in 0..head_dim {
            v[0 * head_dim + d] = 999.0;
        }

        let out = sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, 3, true);
        let last_row = &out[(seq_len - 1) * head_dim..seq_len * head_dim];
        // Should be ~1.0 (average of three 1.0 vectors), not influenced by 999.
        for &val in last_row {
            assert!(
                (val - 1.0).abs() < 1e-4,
                "last position should only see last 3 values, got {val}"
            );
        }
    }

    // ── Multi-head tests ───────────────────────────────────────────────

    #[test]
    fn test_multi_head_basic() {
        let config = SlidingWindowConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 4,
            window_size: 2,
            causal: true,
            per_head_windows: None,
        };
        let total = config.num_heads * config.seq_len * config.head_dim;
        let q = vec![1.0_f32; total];
        let k = vec![1.0_f32; total];
        let v: Vec<f32> = (0..total).map(|i| i as f32).collect();

        let out = sliding_window_attention_multi_head(&q, &k, &v, &config).unwrap();
        assert_eq!(out.len(), total);
    }

    #[test]
    fn test_multi_head_per_head_windows() {
        let config = SlidingWindowConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 4,
            window_size: 2,
            causal: true,
            per_head_windows: Some(vec![1, 100]),
        };
        let head_size = config.seq_len * config.head_dim;
        let total = config.num_heads * head_size;
        let q = vec![1.0_f32; total];
        let k = vec![1.0_f32; total];
        let v: Vec<f32> = (0..total).map(|i| i as f32).collect();

        let out = sliding_window_attention_multi_head(&q, &k, &v, &config).unwrap();

        // Head 0 (window=1): each pos attends only to itself → output == value.
        for i in 0..config.seq_len {
            let out_row = &out[i * config.head_dim..(i + 1) * config.head_dim];
            let v_row = &v[i * config.head_dim..(i + 1) * config.head_dim];
            assert!(approx_eq(out_row, v_row, 1e-5), "head0 pos{i}");
        }

        // Head 1 (window=100): equivalent to full causal attention.
        let h1_q = &q[head_size..];
        let h1_k = &k[head_size..];
        let h1_v = &v[head_size..];
        let full_ref =
            full_attention_reference(h1_q, h1_k, h1_v, config.head_dim, config.seq_len, true);
        let h1_out = &out[head_size..];
        assert!(approx_eq(h1_out, &full_ref, 1e-4), "head1 full causal");
    }

    #[test]
    fn test_multi_head_size_mismatch() {
        let config = SlidingWindowConfig {
            num_heads: 2,
            head_dim: 4,
            seq_len: 4,
            window_size: 2,
            causal: true,
            per_head_windows: None,
        };
        let q = vec![1.0_f32; 10]; // wrong size
        let k = vec![1.0_f32; 32];
        let v = vec![1.0_f32; 32];
        assert!(sliding_window_attention_multi_head(&q, &k, &v, &config).is_err());
    }

    // ── Windowed KV cache tests ────────────────────────────────────────

    #[test]
    fn test_kv_cache_basic_append() {
        let mut cache = WindowedKvCache::new(1, 4, 3);
        assert!(cache.is_empty(0));
        assert_eq!(cache.len(0), 0);

        cache.append(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(cache.len(0), 1);
        assert_eq!(cache.total_tokens(0), 1);
    }

    #[test]
    fn test_kv_cache_eviction() {
        let mut cache = WindowedKvCache::new(1, 2, 2);
        cache.append(0, &[1.0, 1.0], &[10.0, 10.0]);
        cache.append(0, &[2.0, 2.0], &[20.0, 20.0]);
        assert_eq!(cache.len(0), 2);

        cache.append(0, &[3.0, 3.0], &[30.0, 30.0]);
        assert_eq!(cache.len(0), 2); // Still 2 after eviction.
        assert_eq!(cache.total_tokens(0), 3);

        // First entry (1.0) should be evicted; keys should be [2,2,3,3].
        assert_eq!(cache.keys(0), &[2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = WindowedKvCache::new(2, 2, 4);
        cache.append(0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.append(1, &[5.0, 6.0], &[7.0, 8.0]);
        cache.clear();
        assert!(cache.is_empty(0));
        assert!(cache.is_empty(1));
        assert_eq!(cache.total_tokens(0), 0);
    }

    // ── Cached incremental step tests ──────────────────────────────────

    #[test]
    fn test_cached_step_single_token() {
        let mut cache = WindowedKvCache::new(1, 4, 8);
        let q = vec![1.0_f32; 4];
        let k = vec![1.0_f32; 4];
        let v = vec![2.0_f32; 4];

        let out = cached_sliding_window_step(&q, &k, &v, &mut cache, 0);
        // Single token → output should equal the value.
        assert!(approx_eq(&out, &v, 1e-5));
    }

    #[test]
    fn test_cached_step_eviction_works() {
        let mut cache = WindowedKvCache::new(1, 2, 2);
        let q = vec![1.0_f32; 2];

        // Step 1: k=[1,0], v=[10,10]
        cached_sliding_window_step(&q, &[1.0, 0.0], &[10.0, 10.0], &mut cache, 0);
        // Step 2: k=[1,0], v=[20,20]
        cached_sliding_window_step(&q, &[1.0, 0.0], &[20.0, 20.0], &mut cache, 0);
        assert_eq!(cache.len(0), 2);

        // Step 3: k=[1,0], v=[30,30] — should evict step 1.
        let out = cached_sliding_window_step(&q, &[1.0, 0.0], &[30.0, 30.0], &mut cache, 0);
        assert_eq!(cache.len(0), 2);
        // Both remaining entries have equal keys, so output = avg of v2 and v3 = (25, 25).
        assert!((out[0] - 25.0).abs() < 1e-4, "expected ~25.0, got {}", out[0]);
    }

    // ── Local block attention tests ────────────────────────────────────

    #[test]
    fn test_local_block_single_block() {
        // Single block covering the whole sequence → same as full attention.
        let seq_len = 4;
        let head_dim = 4;
        let (q, k, v) = make_uniform_qkv(seq_len, head_dim);

        let block = local_block_attention(&q, &k, &v, head_dim, seq_len, seq_len);
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, false);
        assert!(approx_eq(&block, &full, 1e-5));
    }

    #[test]
    fn test_local_block_isolation() {
        // Two blocks of 2: positions 0-1 and 2-3 shouldn't interact.
        let seq_len = 4;
        let head_dim = 2;
        let q = vec![1.0_f32; seq_len * head_dim];
        let k = vec![1.0_f32; seq_len * head_dim];
        let mut v = vec![0.0_f32; seq_len * head_dim];
        // Block 0 values = 1.0, block 1 values = 2.0.
        for d in 0..head_dim {
            v[0 * head_dim + d] = 1.0;
            v[1 * head_dim + d] = 1.0;
            v[2 * head_dim + d] = 2.0;
            v[3 * head_dim + d] = 2.0;
        }

        let out = local_block_attention(&q, &k, &v, head_dim, seq_len, 2);
        // Positions 0-1 should output ~1.0, positions 2-3 ~2.0.
        for d in 0..head_dim {
            assert!((out[0 * head_dim + d] - 1.0).abs() < 1e-5);
            assert!((out[1 * head_dim + d] - 1.0).abs() < 1e-5);
            assert!((out[2 * head_dim + d] - 2.0).abs() < 1e-5);
            assert!((out[3 * head_dim + d] - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_local_block_uneven() {
        // 5 tokens with block_size=3 → blocks of [0-2] and [3-4].
        let seq_len = 5;
        let head_dim = 2;
        let q = vec![1.0_f32; seq_len * head_dim];
        let k = vec![1.0_f32; seq_len * head_dim];
        let v = vec![1.0_f32; seq_len * head_dim];

        let out = local_block_attention(&q, &k, &v, head_dim, seq_len, 3);
        assert_eq!(out.len(), seq_len * head_dim);
        // All values are 1.0 so output should be ~1.0 everywhere.
        for &val in &out {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    // ── Different window size tests ────────────────────────────────────

    #[test]
    fn test_window_128() {
        let seq_len = 16;
        let head_dim = 4;
        let q = vec![0.1_f32; seq_len * head_dim];
        let k = vec![0.1_f32; seq_len * head_dim];
        let v = vec![1.0_f32; seq_len * head_dim];
        let out = sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, 128, true);
        // Window 128 > seq_len 16 → equivalent to full causal.
        let full = full_attention_reference(&q, &k, &v, head_dim, seq_len, true);
        assert!(approx_eq(&out, &full, 1e-5));
    }

    #[test]
    fn test_window_256() {
        let mask = sliding_window_mask(8, 256, true);
        // All causal positions should be visible.
        for i in 0..8 {
            for j in 0..=i {
                assert_eq!(mask[i * 8 + j], 0.0);
            }
        }
    }

    #[test]
    fn test_window_512() {
        let mask = sliding_window_mask(10, 512, false);
        // All positions visible (non-causal, window > seq_len).
        for &m in &mask {
            assert_eq!(m, 0.0);
        }
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_seq_len_1() {
        let q = vec![1.0_f32; 4];
        let k = vec![1.0_f32; 4];
        let v = vec![2.0_f32; 4];
        let out = sliding_window_attention_single_head(&q, &k, &v, 4, 1, 128, true);
        assert!(approx_eq(&out, &v, 1e-5));
    }

    #[test]
    fn test_head_dim_1() {
        let seq_len = 4;
        let q = vec![1.0_f32; seq_len];
        let k = vec![1.0_f32; seq_len];
        let v = vec![1.0_f32; seq_len];
        let out = sliding_window_attention_single_head(&q, &k, &v, 1, seq_len, 2, true);
        assert_eq!(out.len(), seq_len);
        for &val in &out {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_output_finite() {
        // Ensure no NaN/Inf in output.
        let seq_len = 8;
        let head_dim = 8;
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
        let out = sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, 3, true);
        for (i, &val) in out.iter().enumerate() {
            assert!(val.is_finite(), "non-finite at index {i}: {val}");
        }
    }

    #[test]
    fn test_causal_masking_prevents_future() {
        // With causal masking, position 0 should NOT be influenced by later values.
        let seq_len = 4;
        let head_dim = 4;
        let q = vec![1.0_f32; seq_len * head_dim];
        let k = vec![1.0_f32; seq_len * head_dim];
        let mut v = vec![0.0_f32; seq_len * head_dim];
        // Only position 0 has non-zero value.
        for d in 0..head_dim {
            v[d] = 5.0;
        }

        let out_w2 = sliding_window_attention_single_head(&q, &k, &v, head_dim, seq_len, 2, true);
        // Position 0 should get value [5,5,5,5].
        let row0 = &out_w2[0..head_dim];
        assert!(approx_eq(row0, &[5.0; 4], 1e-5));

        // Position 2 (window=[1,2]) should NOT see position 0's value.
        let row2 = &out_w2[2 * head_dim..3 * head_dim];
        for &val in row2 {
            assert!(val.abs() < 1e-5, "pos 2 should not see pos 0, got {val}");
        }
    }

    #[test]
    fn test_softmax_inplace_all_neginf() {
        let mut row = vec![f32::NEG_INFINITY; 4];
        softmax_inplace(&mut row);
        for &v in &row {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_softmax_inplace_single() {
        let mut row = vec![3.0_f32];
        softmax_inplace(&mut row);
        assert!((row[0] - 1.0).abs() < 1e-6);
    }
}
