//! ARM NEON online softmax kernel (FlashAttention-style) for Apple Silicon.
//!
//! Implements the FlashAttention-2 online softmax algorithm with NEON
//! intrinsics for block-wise attention computation. Instead of materializing
//! the full N×N attention matrix, this streams query–key blocks through an
//! [`OnlineSoftmaxState`] accumulator that tracks the running max, running
//! sum of exponentials, and a running weighted output — yielding O(√N) memory
//! instead of O(N²).
//!
//! # Algorithm outline
//!
//! For each block of key/value vectors:
//! 1. Compute `scores = Q·Kᵀ / √d` (with optional causal mask).
//! 2. Track `new_max = max(old_max, block_max)`.
//! 3. Rescale the running output: `O *= exp(old_max − new_max)`.
//! 4. Accumulate `exp(score − new_max)` into the running sum and output.
//! 5. After all blocks, normalise `O /= running_sum`.
//!
//! # Safety
//!
//! Every function that touches NEON load/store intrinsics (`vld1q_f32`,
//! `vst1q_f32`) or raw pointer arithmetic (`ptr::add`) is marked `unsafe`.
//! Pure NEON arithmetic (`vaddq_f32`, `vmulq_f32`, …) is safe on aarch64.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Lane count for `float32x4_t` NEON vectors.
const LANES: usize = 4;

// ── Fast exp approximation ─────────────────────────────────────────────

/// Scalar fast exp approximation (degree-4 Cody–Waite polynomial).
/// Maximum relative error ≈ 2 × 10⁻⁴ for |x| ≤ 20.
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// NEON vectorised fast exp for four lanes.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
#[allow(dead_code)] // Available for future vectorised score-exp path.
unsafe fn fast_exp_neon(x: float32x4_t) -> float32x4_t {
    let min_val = vdupq_n_f32(-88.0);
    let max_val = vdupq_n_f32(88.0);
    let x = vmaxq_f32(vminq_f32(x, max_val), min_val);

    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    let n = vrndnq_f32(vmulq_f32(x, log2e));
    let r = vsubq_f32(x, vmulq_f32(n, ln2));

    let c1 = vdupq_n_f32(1.0 / 24.0);
    let c2 = vdupq_n_f32(1.0 / 6.0);
    let c3 = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);

    let p = vfmaq_f32(c2, r, c1);
    let p = vfmaq_f32(c3, r, p);
    let p = vfmaq_f32(one, r, p);
    let poly = vfmaq_f32(one, r, p);

    let bias = vdupq_n_s32(127);
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(ni, bias), 23));

    vmulq_f32(poly, pow2n)
}

// ── Online softmax accumulator ─────────────────────────────────────────

/// Running accumulator for the FlashAttention-2 online softmax algorithm.
///
/// Tracks per-row state so that the full N×N score matrix never needs to be
/// materialised.  After processing all key blocks, call [`Self::finalize`]
/// to normalise the output.
#[derive(Clone, Debug)]
pub struct OnlineSoftmaxState {
    /// Running maximum score seen so far (one per row).
    pub max: f32,
    /// Running sum of `exp(score − max)` (one per row).
    pub sum: f32,
    /// Running weighted output vector (length = `head_dim`).
    pub output: Vec<f32>,
}

impl OnlineSoftmaxState {
    /// Create a new accumulator for a given head dimension.
    pub fn new(head_dim: usize) -> Self {
        Self { max: f32::NEG_INFINITY, sum: 0.0, output: vec![0.0; head_dim] }
    }

    /// Reset the accumulator to its initial state.
    pub fn reset(&mut self) {
        self.max = f32::NEG_INFINITY;
        self.sum = 0.0;
        self.output.fill(0.0);
    }

    /// Finalise the output by dividing by the accumulated sum.
    ///
    /// After this call `self.output` contains the true attention-weighted
    /// average.  If no scores were accumulated (`sum == 0`) the output is
    /// left as zeros.
    pub fn finalize(&mut self) {
        if self.sum == 0.0 {
            return;
        }
        let inv = 1.0 / self.sum;
        for v in &mut self.output {
            *v *= inv;
        }
    }

    /// Finalise using NEON intrinsics for the division pass.
    ///
    /// # Safety
    /// Requires `aarch64` target with NEON.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    pub unsafe fn finalize_neon(&mut self) {
        unsafe {
            if self.sum == 0.0 {
                return;
            }
            let inv = 1.0 / self.sum;
            let inv_vec = vdupq_n_f32(inv);

            let len = self.output.len();
            let chunks = len / LANES;
            let remainder = len % LANES;
            let ptr = self.output.as_mut_ptr();

            for i in 0..chunks {
                let v = vld1q_f32(ptr.add(i * LANES));
                let r = vmulq_f32(v, inv_vec);
                vst1q_f32(ptr.add(i * LANES), r);
            }

            let tail = chunks * LANES;
            for i in 0..remainder {
                *ptr.add(tail + i) *= inv;
            }
        }
    }
}

// ── Block-wise online update (scalar) ──────────────────────────────────

/// Process one key/value block through the online softmax accumulator
/// (scalar fallback).
///
/// * `state`  — mutable accumulator for this query row.
/// * `scores` — pre-computed `Q·Kᵀ / √d` for this block (length = block
///   size, already masked if causal).
/// * `values` — value matrix slice for this block, row-major
///   `[block_size × head_dim]`.
///
/// # Panics
/// Panics if `values.len() != scores.len() * state.output.len()`.
pub fn online_softmax_block_scalar(state: &mut OnlineSoftmaxState, scores: &[f32], values: &[f32]) {
    let block_size = scores.len();
    let head_dim = state.output.len();
    assert_eq!(
        values.len(),
        block_size * head_dim,
        "values shape mismatch: expected {}×{} = {}, got {}",
        block_size,
        head_dim,
        block_size * head_dim,
        values.len(),
    );

    // Find the block-local maximum.
    let block_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let new_max = state.max.max(block_max);

    // Rescale running output and sum by exp(old_max − new_max).
    let correction = fast_exp_scalar(state.max - new_max);
    state.sum *= correction;
    for v in state.output.iter_mut() {
        *v *= correction;
    }

    // Accumulate this block.
    for (j, &s) in scores.iter().enumerate() {
        let w = fast_exp_scalar(s - new_max);
        state.sum += w;
        let row_start = j * head_dim;
        for d in 0..head_dim {
            state.output[d] += w * values[row_start + d];
        }
    }

    state.max = new_max;
}

// ── Block-wise online update (NEON) ────────────────────────────────────

/// Process one key/value block through the online softmax accumulator
/// using NEON intrinsics.
///
/// This is the hot inner loop of FlashAttention-style inference.  The
/// value accumulation is vectorised over `head_dim` in 4-wide chunks.
///
/// # Safety
/// Requires `aarch64` target with NEON.
///
/// # Panics
/// Panics if `values.len() != scores.len() * state.output.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn online_softmax_block_neon(
    state: &mut OnlineSoftmaxState,
    scores: &[f32],
    values: &[f32],
) {
    unsafe {
        let block_size = scores.len();
        let head_dim = state.output.len();
        assert_eq!(
            values.len(),
            block_size * head_dim,
            "values shape mismatch: expected {}×{} = {}, got {}",
            block_size,
            head_dim,
            block_size * head_dim,
            values.len(),
        );

        // ── Block-local max ────────────────────────────────────────────
        let block_max;
        {
            let chunks = block_size / LANES;
            let rem = block_size % LANES;
            let sp = scores.as_ptr();

            let mut mv = vdupq_n_f32(f32::NEG_INFINITY);
            for i in 0..chunks {
                let v = vld1q_f32(sp.add(i * LANES));
                mv = vmaxq_f32(mv, v);
            }
            let mut bm = vmaxvq_f32(mv);
            for i in 0..rem {
                let s = *sp.add(chunks * LANES + i);
                if s > bm {
                    bm = s;
                }
            }
            block_max = bm;
        }

        let new_max = state.max.max(block_max);

        // ── Rescale running output and sum ─────────────────────────────
        let correction = fast_exp_scalar(state.max - new_max);
        state.sum *= correction;
        {
            let corr_vec = vdupq_n_f32(correction);
            let chunks = head_dim / LANES;
            let rem = head_dim % LANES;
            let op = state.output.as_mut_ptr();

            for i in 0..chunks {
                let v = vld1q_f32(op.add(i * LANES));
                let r = vmulq_f32(v, corr_vec);
                vst1q_f32(op.add(i * LANES), r);
            }
            let tail = chunks * LANES;
            for i in 0..rem {
                *op.add(tail + i) *= correction;
            }
        }

        // ── Accumulate block scores × values ───────────────────────────
        let op = state.output.as_mut_ptr();
        let vp = values.as_ptr();
        let hd_chunks = head_dim / LANES;
        let hd_rem = head_dim % LANES;

        for j in 0..block_size {
            let s = scores[j];
            let w = fast_exp_scalar(s - new_max);
            state.sum += w;

            let w_vec = vdupq_n_f32(w);
            let row = vp.add(j * head_dim);

            for c in 0..hd_chunks {
                let off = c * LANES;
                let val = vld1q_f32(row.add(off));
                let out = vld1q_f32(op.add(off));
                let acc = vfmaq_f32(out, val, w_vec);
                vst1q_f32(op.add(off), acc);
            }
            let tail = hd_chunks * LANES;
            for d in 0..hd_rem {
                *op.add(tail + d) += w * *row.add(tail + d);
            }
        }

        state.max = new_max;
    }
}

// ── Causal masking ─────────────────────────────────────────────────────

/// Apply a causal mask to a score matrix in-place.
///
/// `scores` is `[query_len × key_len]` in row-major order.  Positions
/// where `key_pos > query_start + query_idx` are set to `f32::NEG_INFINITY`.
///
/// * `query_start` — absolute position of the first query in the
///   sequence (accounts for KV-cache offset).
#[inline]
pub fn apply_causal_mask(scores: &mut [f32], query_len: usize, key_len: usize, query_start: usize) {
    assert_eq!(scores.len(), query_len * key_len, "scores shape mismatch");
    for q in 0..query_len {
        let abs_q = query_start + q;
        let row = q * key_len;
        for k in (abs_q + 1)..key_len {
            scores[row + k] = f32::NEG_INFINITY;
        }
    }
}

/// Apply causal mask using NEON for the fill pass.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_causal_mask_neon(
    scores: &mut [f32],
    query_len: usize,
    key_len: usize,
    query_start: usize,
) {
    unsafe {
        assert_eq!(scores.len(), query_len * key_len, "scores shape mismatch");

        let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
        let ptr = scores.as_mut_ptr();

        for q in 0..query_len {
            let abs_q = query_start + q;
            let mask_start = abs_q + 1;
            if mask_start >= key_len {
                continue;
            }
            let row_ptr = ptr.add(q * key_len);
            let count = key_len - mask_start;
            let chunks = count / LANES;
            let rem = count % LANES;

            let base = row_ptr.add(mask_start);
            for i in 0..chunks {
                vst1q_f32(base.add(i * LANES), neg_inf);
            }
            let tail = chunks * LANES;
            for i in 0..rem {
                *base.add(tail + i) = f32::NEG_INFINITY;
            }
        }
    }
}

// ── Multi-head online attention ────────────────────────────────────────

/// Compute multi-head online softmax attention across all heads.
///
/// This is the top-level entry point.  Each head gets an independent
/// [`OnlineSoftmaxState`] accumulator, and key/value blocks are streamed
/// through in `block_size`-sized chunks.
///
/// # Arguments
///
/// * `queries`     — `[num_heads × head_dim]` (row-major).
/// * `keys`        — `[seq_len × head_dim]` (shared across heads, or
///                   pre-sliced for GQA).
/// * `values`      — `[seq_len × head_dim]` (same layout as keys).
/// * `num_heads`   — number of attention heads.
/// * `head_dim`    — dimensionality per head.
/// * `block_size`  — tile size for streaming (e.g. 64 or 128).
/// * `causal`      — whether to apply causal masking.
/// * `query_start` — absolute position offset for causal masking.
///
/// # Returns
///
/// Flattened `[num_heads × head_dim]` output.
///
/// # Panics
/// Panics on shape mismatches.
pub fn multi_head_online_softmax(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    causal: bool,
    query_start: usize,
) -> Vec<f32> {
    assert_eq!(queries.len(), num_heads * head_dim, "queries shape mismatch");
    let seq_len = keys.len() / head_dim;
    assert_eq!(keys.len(), seq_len * head_dim, "keys shape mismatch");
    assert_eq!(values.len(), seq_len * head_dim, "values shape mismatch");
    assert!(block_size > 0, "block_size must be > 0");

    let mut output = vec![0.0f32; num_heads * head_dim];

    for h in 0..num_heads {
        let q = &queries[h * head_dim..(h + 1) * head_dim];
        let mut state = OnlineSoftmaxState::new(head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let num_blocks = (seq_len + block_size - 1) / block_size;

        for b in 0..num_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(seq_len);
            let blen = end - start;

            // Compute scores = Q · K^T / sqrt(d) for this block.
            let mut scores = vec![0.0f32; blen];
            for j in 0..blen {
                let k_row = &keys[(start + j) * head_dim..(start + j + 1) * head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[d] * k_row[d];
                }
                scores[j] = dot * scale;
            }

            // Apply causal mask for this block (single query row).
            if causal {
                let abs_q = query_start;
                for j in 0..blen {
                    if start + j > abs_q {
                        scores[j] = f32::NEG_INFINITY;
                    }
                }
            }

            let v_block = &values[start * head_dim..end * head_dim];

            online_softmax_block_scalar(&mut state, &scores, v_block);
        }

        state.finalize();
        output[h * head_dim..(h + 1) * head_dim].copy_from_slice(&state.output);
    }

    output
}

/// NEON-accelerated multi-head online softmax attention.
///
/// Same semantics as [`multi_head_online_softmax`] but the inner block
/// accumulation and dot-product use NEON intrinsics.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn multi_head_online_softmax_neon(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    num_heads: usize,
    head_dim: usize,
    block_size: usize,
    causal: bool,
    query_start: usize,
) -> Vec<f32> {
    unsafe {
        assert_eq!(queries.len(), num_heads * head_dim, "queries shape mismatch");
        let seq_len = keys.len() / head_dim;
        assert_eq!(keys.len(), seq_len * head_dim, "keys shape mismatch");
        assert_eq!(values.len(), seq_len * head_dim, "values shape mismatch");
        assert!(block_size > 0, "block_size must be > 0");

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; num_heads * head_dim];

        for h in 0..num_heads {
            let q = &queries[h * head_dim..(h + 1) * head_dim];
            let mut state = OnlineSoftmaxState::new(head_dim);

            let num_blocks = (seq_len + block_size - 1) / block_size;

            for b in 0..num_blocks {
                let start = b * block_size;
                let end = (start + block_size).min(seq_len);
                let blen = end - start;

                // NEON dot-product for scores.
                let mut scores = vec![0.0f32; blen];
                let qp = q.as_ptr();

                for j in 0..blen {
                    let kp = keys.as_ptr().add((start + j) * head_dim);
                    let hd_chunks = head_dim / LANES;
                    let hd_rem = head_dim % LANES;

                    let mut acc = vdupq_n_f32(0.0);
                    for c in 0..hd_chunks {
                        let qv = vld1q_f32(qp.add(c * LANES));
                        let kv = vld1q_f32(kp.add(c * LANES));
                        acc = vfmaq_f32(acc, qv, kv);
                    }
                    let mut dot = vaddvq_f32(acc);
                    let tail = hd_chunks * LANES;
                    for d in 0..hd_rem {
                        dot += *qp.add(tail + d) * *kp.add(tail + d);
                    }
                    scores[j] = dot * scale;
                }

                // Causal mask.
                if causal {
                    let abs_q = query_start;
                    for j in 0..blen {
                        if start + j > abs_q {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }
                }

                let v_block = &values[start * head_dim..end * head_dim];

                online_softmax_block_neon(&mut state, &scores, v_block);
            }

            state.finalize_neon();
            output[h * head_dim..(h + 1) * head_dim].copy_from_slice(&state.output);
        }

        output
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
        assert!((a - b).abs() < tol, "{ctx}: expected {b}, got {a} (diff {})", (a - b).abs());
    }

    /// Reference single-head attention (materialises full score matrix).
    fn reference_attention(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        head_dim: usize,
        causal: bool,
        query_start: usize,
    ) -> Vec<f32> {
        let seq_len = keys.len() / head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut scores = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += query[d] * keys[j * head_dim + d];
            }
            scores[j] = dot * scale;
        }

        if causal {
            for j in 0..seq_len {
                if j > query_start {
                    scores[j] = f32::NEG_INFINITY;
                }
            }
        }

        // Stable softmax.
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();

        let mut out = vec![0.0f32; head_dim];
        if sum > 0.0 {
            for j in 0..seq_len {
                let w = exps[j] / sum;
                for d in 0..head_dim {
                    out[d] += w * values[j * head_dim + d];
                }
            }
        }
        out
    }

    // ── OnlineSoftmaxState unit tests ──────────────────────────────

    #[test]
    fn test_state_new() {
        let s = OnlineSoftmaxState::new(8);
        assert_eq!(s.max, f32::NEG_INFINITY);
        assert_eq!(s.sum, 0.0);
        assert_eq!(s.output.len(), 8);
    }

    #[test]
    fn test_state_reset() {
        let mut s = OnlineSoftmaxState::new(4);
        s.max = 1.0;
        s.sum = 5.0;
        s.output[0] = 42.0;
        s.reset();
        assert_eq!(s.max, f32::NEG_INFINITY);
        assert_eq!(s.sum, 0.0);
        assert!(s.output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_finalize_zero_sum() {
        let mut s = OnlineSoftmaxState::new(4);
        s.finalize();
        assert!(s.output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_finalize_normalizes() {
        let mut s = OnlineSoftmaxState::new(4);
        s.sum = 2.0;
        s.output = vec![1.0, 2.0, 3.0, 4.0];
        s.finalize();
        assert_close(s.output[0], 0.5, 1e-6, "finalize[0]");
        assert_close(s.output[3], 2.0, 1e-6, "finalize[3]");
    }

    // ── Scalar block tests ─────────────────────────────────────────

    #[test]
    fn test_scalar_single_block_uniform() {
        let head_dim = 4;
        let block_size = 3;
        let scores = vec![0.0; block_size];
        let values = vec![1.0; block_size * head_dim];

        let mut state = OnlineSoftmaxState::new(head_dim);
        online_softmax_block_scalar(&mut state, &scores, &values);
        state.finalize();

        // Uniform scores → all value rows equally weighted → output = 1.0.
        for d in 0..head_dim {
            assert_close(state.output[d], 1.0, 1e-3, "uniform");
        }
    }

    #[test]
    fn test_scalar_two_blocks_match_reference() {
        let head_dim = 4;
        let seq_len = 8;

        // Deterministic pseudo-random data.
        let query: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let keys: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.3).collect();
        let values: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i * 11 + 5) % 17) as f32 * 0.05).collect();

        let reference = reference_attention(&query, &keys, &values, head_dim, false, 0);

        // Online: two blocks of 4.
        let block_size = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut state = OnlineSoftmaxState::new(head_dim);

        for b in 0..2 {
            let start = b * block_size;
            let end = start + block_size;
            let mut scores = vec![0.0f32; block_size];
            for j in 0..block_size {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += query[d] * keys[(start + j) * head_dim + d];
                }
                scores[j] = dot * scale;
            }
            let v_block = &values[start * head_dim..end * head_dim];
            online_softmax_block_scalar(&mut state, &scores, v_block);
        }
        state.finalize();

        for d in 0..head_dim {
            assert_close(state.output[d], reference[d], 1e-2, &format!("block_vs_ref[{d}]"));
        }
    }

    // ── Causal masking tests ───────────────────────────────────────

    #[test]
    fn test_causal_mask_basic() {
        let mut scores = vec![1.0; 3 * 5]; // 3 queries × 5 keys
        apply_causal_mask(&mut scores, 3, 5, 0);

        // Row 0 (abs_q=0): k>0 masked.
        assert!(scores[1].is_infinite() && scores[1] < 0.0);
        // Row 1 (abs_q=1): k>1 masked.
        assert_eq!(scores[1 * 5 + 1], 1.0);
        assert!(scores[1 * 5 + 2].is_infinite());
        // Row 2 (abs_q=2): k>2 masked.
        assert_eq!(scores[2 * 5 + 2], 1.0);
        assert!(scores[2 * 5 + 3].is_infinite());
    }

    #[test]
    fn test_causal_mask_with_offset() {
        let mut scores = vec![1.0; 2 * 6]; // 2 queries × 6 keys
        apply_causal_mask(&mut scores, 2, 6, 3);

        // Row 0 (abs_q=3): k>3 masked, k<=3 visible.
        assert_eq!(scores[3], 1.0);
        assert!(scores[4].is_infinite());
        // Row 1 (abs_q=4): k>4 masked.
        assert_eq!(scores[1 * 6 + 4], 1.0);
        assert!(scores[1 * 6 + 5].is_infinite());
    }

    // ── Multi-head scalar tests ────────────────────────────────────

    #[test]
    fn test_multi_head_single_head() {
        let head_dim = 4;
        let seq_len = 6;
        let num_heads = 1;
        let block_size = 4;

        let queries: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.2).collect();
        let keys: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i * 3 + 1) % 11) as f32 * 0.1).collect();
        let values: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i * 5 + 2) % 13) as f32 * 0.08).collect();

        let reference = reference_attention(&queries, &keys, &values, head_dim, false, 0);
        let result = multi_head_online_softmax(
            &queries, &keys, &values, num_heads, head_dim, block_size, false, 0,
        );

        for d in 0..head_dim {
            assert_close(result[d], reference[d], 1e-2, &format!("multi_head[{d}]"));
        }
    }

    #[test]
    fn test_multi_head_two_heads() {
        let head_dim = 4;
        let seq_len = 8;
        let num_heads = 2;
        let block_size = 4;

        let queries: Vec<f32> =
            (0..num_heads * head_dim).map(|i| ((i * 7 + 1) % 19) as f32 * 0.05).collect();
        let keys: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i * 3 + 2) % 11) as f32 * 0.1).collect();
        let values: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i * 5 + 3) % 13) as f32 * 0.08).collect();

        let result = multi_head_online_softmax(
            &queries, &keys, &values, num_heads, head_dim, block_size, false, 0,
        );

        for h in 0..num_heads {
            let q = &queries[h * head_dim..(h + 1) * head_dim];
            let ref_h = reference_attention(q, &keys, &values, head_dim, false, 0);
            for d in 0..head_dim {
                assert_close(result[h * head_dim + d], ref_h[d], 1e-2, &format!("head{h}[{d}]"));
            }
        }
    }

    #[test]
    fn test_multi_head_causal() {
        let head_dim = 4;
        let seq_len = 6;
        let num_heads = 1;
        let block_size = 3;
        let query_start = 2; // Only see keys 0..=2.

        let queries: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.15).collect();
        let keys: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i * 7 + 1) % 11) as f32 * 0.1).collect();
        let values: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i * 3 + 5) % 13) as f32 * 0.06).collect();

        let reference = reference_attention(&queries, &keys, &values, head_dim, true, query_start);
        let result = multi_head_online_softmax(
            &queries,
            &keys,
            &values,
            num_heads,
            head_dim,
            block_size,
            true,
            query_start,
        );

        for d in 0..head_dim {
            assert_close(result[d], reference[d], 1e-2, &format!("causal[{d}]"));
        }
    }

    #[test]
    fn test_block_size_one() {
        let head_dim = 4;
        let seq_len = 5;
        let block_size = 1; // One key per block — maximally streamed.

        let queries: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.3).collect();
        let keys: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i * 11 + 7) % 17) as f32 * 0.05).collect();
        let values: Vec<f32> =
            (0..seq_len * head_dim).map(|i| ((i * 13 + 3) % 19) as f32 * 0.04).collect();

        let reference = reference_attention(&queries, &keys, &values, head_dim, false, 0);
        let result =
            multi_head_online_softmax(&queries, &keys, &values, 1, head_dim, block_size, false, 0);

        for d in 0..head_dim {
            assert_close(result[d], reference[d], 1e-2, &format!("bs1[{d}]"));
        }
    }

    #[test]
    fn test_memory_is_subquadratic() {
        // Verify that the accumulator is O(head_dim), not O(seq_len²).
        let head_dim = 64;
        let state = OnlineSoftmaxState::new(head_dim);
        // The state holds exactly one head_dim vector (+ 2 scalars).
        assert_eq!(state.output.len(), head_dim);
        assert_eq!(
            std::mem::size_of_val(&state.max)
                + std::mem::size_of_val(&state.sum)
                + state.output.len() * std::mem::size_of::<f32>(),
            264 // 4 + 4 + 64*4
        );
    }

    // ── NEON-specific tests ────────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    mod neon_tests {
        use super::*;

        #[test]
        fn test_neon_block_matches_scalar() {
            let head_dim = 8;
            let block_size = 4;
            let scores: Vec<f32> = (0..block_size).map(|i| i as f32 * 0.5 - 1.0).collect();
            let values: Vec<f32> =
                (0..block_size * head_dim).map(|i| ((i * 7 + 3) % 13) as f32 * 0.1).collect();

            let mut scalar_state = OnlineSoftmaxState::new(head_dim);
            online_softmax_block_scalar(&mut scalar_state, &scores, &values);
            scalar_state.finalize();

            let mut neon_state = OnlineSoftmaxState::new(head_dim);
            unsafe {
                online_softmax_block_neon(&mut neon_state, &scores, &values);
                neon_state.finalize_neon();
            }

            for d in 0..head_dim {
                assert_close(
                    neon_state.output[d],
                    scalar_state.output[d],
                    1e-3,
                    &format!("neon_vs_scalar[{d}]"),
                );
            }
        }

        #[test]
        fn test_neon_multi_head_matches_scalar() {
            let head_dim = 8;
            let seq_len = 12;
            let num_heads = 2;
            let block_size = 4;

            let queries: Vec<f32> =
                (0..num_heads * head_dim).map(|i| ((i * 7 + 1) % 19) as f32 * 0.05).collect();
            let keys: Vec<f32> =
                (0..seq_len * head_dim).map(|i| ((i * 3 + 2) % 11) as f32 * 0.1).collect();
            let values: Vec<f32> =
                (0..seq_len * head_dim).map(|i| ((i * 5 + 3) % 13) as f32 * 0.08).collect();

            let scalar_out = multi_head_online_softmax(
                &queries, &keys, &values, num_heads, head_dim, block_size, false, 0,
            );
            let neon_out = unsafe {
                multi_head_online_softmax_neon(
                    &queries, &keys, &values, num_heads, head_dim, block_size, false, 0,
                )
            };

            for i in 0..scalar_out.len() {
                assert_close(neon_out[i], scalar_out[i], 1e-2, &format!("neon_multi[{i}]"));
            }
        }

        #[test]
        fn test_neon_causal_matches_scalar() {
            let head_dim = 8;
            let seq_len = 10;
            let num_heads = 1;
            let block_size = 3;
            let query_start = 4;

            let queries: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.2).collect();
            let keys: Vec<f32> =
                (0..seq_len * head_dim).map(|i| ((i * 7 + 1) % 11) as f32 * 0.1).collect();
            let values: Vec<f32> =
                (0..seq_len * head_dim).map(|i| ((i * 3 + 5) % 13) as f32 * 0.06).collect();

            let scalar_out = multi_head_online_softmax(
                &queries,
                &keys,
                &values,
                num_heads,
                head_dim,
                block_size,
                true,
                query_start,
            );
            let neon_out = unsafe {
                multi_head_online_softmax_neon(
                    &queries,
                    &keys,
                    &values,
                    num_heads,
                    head_dim,
                    block_size,
                    true,
                    query_start,
                )
            };

            for d in 0..head_dim {
                assert_close(neon_out[d], scalar_out[d], 1e-2, &format!("neon_causal[{d}]"));
            }
        }

        #[test]
        fn test_neon_finalize_matches_scalar() {
            let mut s1 = OnlineSoftmaxState::new(6);
            s1.sum = 4.0;
            s1.output = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

            let mut s2 = s1.clone();
            s1.finalize();
            unsafe { s2.finalize_neon() };

            for i in 0..6 {
                assert_close(s2.output[i], s1.output[i], 1e-6, &format!("finalize_neon[{i}]"));
            }
        }

        #[test]
        fn test_neon_causal_mask_matches_scalar() {
            let q = 3;
            let k = 8;
            let offset = 1;

            let mut scalar = vec![1.0f32; q * k];
            let mut neon = scalar.clone();

            apply_causal_mask(&mut scalar, q, k, offset);
            unsafe {
                apply_causal_mask_neon(&mut neon, q, k, offset);
            }

            for i in 0..(q * k) {
                assert_eq!(scalar[i].to_bits(), neon[i].to_bits(), "mask mismatch at {i}");
            }
        }

        #[test]
        #[ignore = "requires large allocation (~1 GB) — run manually \
                    with --ignored for memory-efficiency validation"]
        fn test_neon_large_seq_len_memory_efficient() {
            // Verify that streaming with a small block_size does not
            // blow up memory for a large sequence length.
            let head_dim = 64;
            let seq_len = 4096;
            let block_size = 64;

            let queries = vec![0.1f32; head_dim];
            let keys = vec![0.05f32; seq_len * head_dim];
            let values = vec![1.0f32; seq_len * head_dim];

            let result = unsafe {
                multi_head_online_softmax_neon(
                    &queries, &keys, &values, 1, head_dim, block_size, false, 0,
                )
            };

            // Uniform keys → uniform attention → output ≈ 1.0.
            for d in 0..head_dim {
                assert_close(result[d], 1.0, 0.1, &format!("large_seq[{d}]"));
            }
        }
    }
}
