#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::let_and_return)]
//! ARM NEON flash attention kernels for Apple Silicon.
//!
//! Implements tiled flash attention with NEON SIMD intrinsics, following the
//! FlashAttention-2 algorithm for O(√N) memory usage. Instead of materializing
//! the full N×N attention matrix, queries and keys are processed in tiles
//! with an online softmax accumulator.
//!
//! # Variants
//!
//! | Function                          | Description                        |
//! |-----------------------------------|------------------------------------|
//! | `flash_attention_neon`            | Full (non-causal) flash attention  |
//! | `flash_attention_causal_neon`     | Causal-masked flash attention      |
//! | `flash_attention_multihead_neon`  | Multi-head flash attention         |
//!
//! Each has a `_scalar` counterpart for non-NEON fallback.
//!
//! # Safety
//!
//! NEON load/store intrinsics (`vld1q_f32`, `vst1q_f32`) and raw pointer
//! arithmetic (`ptr::add`) live inside `unsafe` blocks. Pure NEON arithmetic
//! (`vaddq_f32`, `vmulq_f32`, …) is safe on aarch64.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

/// Default tile size for query/key blocking.
const TILE_SIZE: usize = 32;

// ── Fast exp approximation ─────────────────────────────────────────────

/// Scalar fast exp (degree-4 Cody–Waite polynomial).
/// Max relative error ≈ 2 × 10⁻⁴ for |x| ≤ 88.
#[inline(always)]
fn fast_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    let poly = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0))));
    poly * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

// ── Online softmax accumulator ─────────────────────────────────────────

/// Per-row accumulator for the online softmax algorithm.
struct RowAccumulator {
    max: f32,
    sum: f32,
    output: Vec<f32>,
}

impl RowAccumulator {
    fn new(head_dim: usize) -> Self {
        Self { max: f32::NEG_INFINITY, sum: 0.0, output: vec![0.0; head_dim] }
    }

    /// Process one score against this accumulator (scalar path).
    #[inline]
    fn accumulate_scalar(&mut self, score: f32, value_row: &[f32]) {
        let new_max = self.max.max(score);
        let correction = fast_exp_scalar(self.max - new_max);
        self.sum *= correction;
        for v in self.output.iter_mut() {
            *v *= correction;
        }
        self.max = new_max;
        let w = fast_exp_scalar(score - new_max);
        self.sum += w;
        for (o, &v) in self.output.iter_mut().zip(value_row.iter()) {
            *o += w * v;
        }
    }

    /// Finalize by dividing accumulated output by sum.
    fn finalize(&mut self) {
        if self.sum > 0.0 {
            let inv = 1.0 / self.sum;
            for v in &mut self.output {
                *v *= inv;
            }
        }
    }
}

// ── NEON dot product helper ────────────────────────────────────────────

/// Compute dot product of two slices using NEON intrinsics.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / LANES;
    let mut acc = vdupq_n_f32(0.0);

    unsafe {
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        for i in 0..chunks {
            let va = vld1q_f32(pa.add(i * LANES));
            let vb = vld1q_f32(pb.add(i * LANES));
            acc = vfmaq_f32(acc, va, vb);
        }
    }

    let mut result = vaddvq_f32(acc);
    let tail = chunks * LANES;
    for i in tail..len {
        result += a[i] * b[i];
    }
    result
}

/// Scalar dot product fallback.
#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── NEON accumulate helper ─────────────────────────────────────────────

/// Accumulate `weight * value_row` into `output` using NEON.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accumulate_weighted_neon(output: &mut [f32], value_row: &[f32], weight: f32) {
    let len = output.len();
    let chunks = len / LANES;
    let w_vec = vdupq_n_f32(weight);

    unsafe {
        let po = output.as_mut_ptr();
        let pv = value_row.as_ptr();
        for i in 0..chunks {
            let o = vld1q_f32(po.add(i * LANES));
            let v = vld1q_f32(pv.add(i * LANES));
            let r = vfmaq_f32(o, v, w_vec);
            vst1q_f32(po.add(i * LANES), r);
        }
    }

    let tail = chunks * LANES;
    for i in tail..len {
        output[i] += weight * value_row[i];
    }
}

/// Scale output slice by `factor` using NEON.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_output_neon(output: &mut [f32], factor: f32) {
    let len = output.len();
    let chunks = len / LANES;
    let f_vec = vdupq_n_f32(factor);

    unsafe {
        let po = output.as_mut_ptr();
        for i in 0..chunks {
            let v = vld1q_f32(po.add(i * LANES));
            let r = vmulq_f32(v, f_vec);
            vst1q_f32(po.add(i * LANES), r);
        }
    }

    let tail = chunks * LANES;
    for i in tail..len {
        output[i] *= factor;
    }
}

// ── Tiled flash attention core (scalar) ────────────────────────────────

/// Compute flash attention for a single head (scalar fallback).
///
/// # Layout
/// - `q`: `[seq_len × head_dim]` row-major
/// - `k`: `[seq_len × head_dim]` row-major
/// - `v`: `[seq_len × head_dim]` row-major
/// - `output`: `[seq_len × head_dim]` row-major (pre-allocated)
///
/// If `causal` is true, position `i` only attends to positions `j ≤ i`.
fn flash_attention_core_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) {
    if seq_len == 0 || head_dim == 0 {
        return;
    }

    let expected = seq_len * head_dim;
    assert!(q.len() >= expected, "q too short");
    assert!(k.len() >= expected, "k too short");
    assert!(v.len() >= expected, "v too short");
    assert!(output.len() >= expected, "output too short");

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let mut acc = RowAccumulator::new(head_dim);

        let j_end = if causal { i + 1 } else { seq_len };

        // Process keys in tiles for cache efficiency.
        for tile_start in (0..j_end).step_by(TILE_SIZE) {
            let tile_end = (tile_start + TILE_SIZE).min(j_end);
            for j in tile_start..tile_end {
                let k_row = &k[j * head_dim..(j + 1) * head_dim];
                let score = dot_product_scalar(q_row, k_row) * scale;
                let v_row = &v[j * head_dim..(j + 1) * head_dim];
                acc.accumulate_scalar(score, v_row);
            }
        }

        acc.finalize();
        output[i * head_dim..(i + 1) * head_dim].copy_from_slice(&acc.output);
    }
}

// ── Tiled flash attention core (NEON) ──────────────────────────────────

/// Compute flash attention for a single head using NEON intrinsics.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn flash_attention_core_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) {
    if seq_len == 0 || head_dim == 0 {
        return;
    }

    let expected = seq_len * head_dim;
    assert!(q.len() >= expected, "q too short");
    assert!(k.len() >= expected, "k too short");
    assert!(v.len() >= expected, "v too short");
    assert!(output.len() >= expected, "output too short");

    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let mut row_max = f32::NEG_INFINITY;
        let mut row_sum: f32 = 0.0;
        let mut row_out = vec![0.0f32; head_dim];

        let j_end = if causal { i + 1 } else { seq_len };

        for tile_start in (0..j_end).step_by(TILE_SIZE) {
            let tile_end = (tile_start + TILE_SIZE).min(j_end);
            for j in tile_start..tile_end {
                let k_row = &k[j * head_dim..(j + 1) * head_dim];
                let score = unsafe { dot_product_neon(q_row, k_row) } * scale;

                let new_max = row_max.max(score);
                let correction = fast_exp_scalar(row_max - new_max);

                row_sum *= correction;
                unsafe {
                    scale_output_neon(&mut row_out, correction);
                }

                row_max = new_max;
                let w = fast_exp_scalar(score - new_max);
                row_sum += w;

                let v_row = &v[j * head_dim..(j + 1) * head_dim];
                unsafe {
                    accumulate_weighted_neon(&mut row_out, v_row, w);
                }
            }
        }

        // Finalize: divide by sum.
        if row_sum > 0.0 {
            let inv = 1.0 / row_sum;
            unsafe {
                scale_output_neon(&mut row_out, inv);
            }
        }

        output[i * head_dim..(i + 1) * head_dim].copy_from_slice(&row_out);
    }
}

// ── Public API: basic flash attention ──────────────────────────────────

/// Flash attention with NEON SIMD (non-causal).
///
/// # Layout
/// - `q`, `k`, `v`: `[seq_len × head_dim]` row-major
/// - `output`: `[seq_len × head_dim]` row-major (pre-allocated)
/// - `scale`: typically `1 / √head_dim`
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn flash_attention_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    unsafe {
        flash_attention_core_neon(q, k, v, output, seq_len, head_dim, scale, false);
    }
}

/// Flash attention scalar fallback (non-causal).
pub fn flash_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    flash_attention_core_scalar(q, k, v, output, seq_len, head_dim, scale, false);
}

// ── Public API: causal flash attention ─────────────────────────────────

/// Causal flash attention with NEON SIMD.
///
/// Position `i` can only attend to positions `j ≤ i`.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn flash_attention_causal_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    unsafe {
        flash_attention_core_neon(q, k, v, output, seq_len, head_dim, scale, true);
    }
}

/// Causal flash attention scalar fallback.
pub fn flash_attention_causal_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    flash_attention_core_scalar(q, k, v, output, seq_len, head_dim, scale, true);
}

// ── Public API: multi-head flash attention ─────────────────────────────

/// Multi-head flash attention with NEON SIMD (non-causal).
///
/// # Layout
/// - `q`, `k`, `v`: `[num_heads × seq_len × head_dim]` row-major
/// - `output`: `[num_heads × seq_len × head_dim]` row-major
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn flash_attention_multihead_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let head_size = seq_len * head_dim;
    let total = num_heads * head_size;
    assert!(q.len() >= total, "q too short for multihead");
    assert!(k.len() >= total, "k too short for multihead");
    assert!(v.len() >= total, "v too short for multihead");
    assert!(output.len() >= total, "output too short for multihead");

    for h in 0..num_heads {
        let offset = h * head_size;
        unsafe {
            flash_attention_core_neon(
                &q[offset..offset + head_size],
                &k[offset..offset + head_size],
                &v[offset..offset + head_size],
                &mut output[offset..offset + head_size],
                seq_len,
                head_dim,
                scale,
                false,
            );
        }
    }
}

/// Multi-head flash attention scalar fallback (non-causal).
pub fn flash_attention_multihead_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let head_size = seq_len * head_dim;
    let total = num_heads * head_size;
    assert!(q.len() >= total, "q too short for multihead");
    assert!(k.len() >= total, "k too short for multihead");
    assert!(v.len() >= total, "v too short for multihead");
    assert!(output.len() >= total, "output too short for multihead");

    for h in 0..num_heads {
        let offset = h * head_size;
        flash_attention_core_scalar(
            &q[offset..offset + head_size],
            &k[offset..offset + head_size],
            &v[offset..offset + head_size],
            &mut output[offset..offset + head_size],
            seq_len,
            head_dim,
            scale,
            false,
        );
    }
}

/// Multi-head causal flash attention with NEON SIMD.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn flash_attention_multihead_causal_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let head_size = seq_len * head_dim;
    let total = num_heads * head_size;
    assert!(q.len() >= total, "q too short for multihead");
    assert!(k.len() >= total, "k too short for multihead");
    assert!(v.len() >= total, "v too short for multihead");
    assert!(output.len() >= total, "output too short for multihead");

    for h in 0..num_heads {
        let offset = h * head_size;
        unsafe {
            flash_attention_core_neon(
                &q[offset..offset + head_size],
                &k[offset..offset + head_size],
                &v[offset..offset + head_size],
                &mut output[offset..offset + head_size],
                seq_len,
                head_dim,
                scale,
                true,
            );
        }
    }
}

/// Multi-head causal flash attention scalar fallback.
pub fn flash_attention_multihead_causal_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let head_size = seq_len * head_dim;
    let total = num_heads * head_size;
    assert!(q.len() >= total, "q too short for multihead");
    assert!(k.len() >= total, "k too short for multihead");
    assert!(v.len() >= total, "v too short for multihead");
    assert!(output.len() >= total, "output too short for multihead");

    for h in 0..num_heads {
        let offset = h * head_size;
        flash_attention_core_scalar(
            &q[offset..offset + head_size],
            &k[offset..offset + head_size],
            &v[offset..offset + head_size],
            &mut output[offset..offset + head_size],
            seq_len,
            head_dim,
            scale,
            true,
        );
    }
}

// ── Reference naive attention (for tests) ──────────────────────────────

/// Naive O(N²) attention for test reference. Not tiled, not numerically
/// stable for large scores — only used to validate the tiled kernels on
/// small inputs.
#[cfg(test)]
fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) {
    for i in 0..seq_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        let j_end = if causal { i + 1 } else { seq_len };

        // Compute scores.
        let mut scores: Vec<f32> = (0..j_end)
            .map(|j| {
                let k_row = &k[j * head_dim..(j + 1) * head_dim];
                dot_product_scalar(q_row, k_row) * scale
            })
            .collect();

        // Softmax with numerical stability.
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for s in &mut scores {
            *s = (*s - max_score).exp();
        }
        let sum: f32 = scores.iter().sum();
        if sum > 0.0 {
            for s in &mut scores {
                *s /= sum;
            }
        }

        // Weighted sum of values.
        let out = &mut output[i * head_dim..(i + 1) * head_dim];
        out.fill(0.0);
        for (j, &w) in scores.iter().enumerate() {
            let v_row = &v[j * head_dim..(j + 1) * head_dim];
            for (o, &val) in out.iter_mut().zip(v_row.iter()) {
                *o += w * val;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for comparing flash attention vs naive reference.
    /// The fast-exp approximation introduces small errors.
    const ATOL: f32 = 1e-2;

    /// Stricter tolerance for scalar-vs-scalar comparisons where both
    /// use the same fast-exp.
    const STRICT_ATOL: f32 = 1e-5;

    // Helper: deterministic pseudo-random f32 in [-1, 1].
    fn pseudo_random(seed: u64, idx: usize) -> f32 {
        let h = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(idx as u64)
            .wrapping_mul(1442695040888963407);
        let bits = ((h >> 33) ^ h) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn fill_random(buf: &mut [f32], seed: u64) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = pseudo_random(seed, i);
        }
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(diff < tol, "{msg}: index {i} differs: {x} vs {y} (diff={diff}, tol={tol})");
        }
    }

    // ── Empty / zero-length inputs ─────────────────────────────────────

    #[test]
    fn test_scalar_empty_seq() {
        let mut output = vec![0.0f32; 0];
        flash_attention_scalar(&[], &[], &[], &mut output, 0, 4, 1.0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_scalar_zero_head_dim() {
        let mut output = vec![0.0f32; 0];
        flash_attention_scalar(&[], &[], &[], &mut output, 4, 0, 1.0);
    }

    #[test]
    fn test_causal_scalar_empty_seq() {
        let mut output = vec![0.0f32; 0];
        flash_attention_causal_scalar(&[], &[], &[], &mut output, 0, 4, 1.0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_multihead_scalar_empty() {
        let mut output = vec![0.0f32; 0];
        flash_attention_multihead_scalar(&[], &[], &[], &mut output, 0, 0, 4, 1.0);
    }

    // ── Single element ─────────────────────────────────────────────────

    #[test]
    fn test_scalar_single_element() {
        let q = vec![1.0f32; 4];
        let k = vec![1.0f32; 4];
        let v = vec![2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 4];
        flash_attention_scalar(&q, &k, &v, &mut output, 1, 4, 0.5);
        // Single element: softmax of one score = 1.0, so output = v.
        assert_close(&output, &v, STRICT_ATOL, "single element");
    }

    #[test]
    fn test_causal_scalar_single_element() {
        let q = vec![1.0f32; 4];
        let k = vec![1.0f32; 4];
        let v = vec![2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 4];
        flash_attention_causal_scalar(&q, &k, &v, &mut output, 1, 4, 0.5);
        assert_close(&output, &v, STRICT_ATOL, "causal single element");
    }

    // ── Scalar basic correctness ───────────────────────────────────────

    #[test]
    fn test_scalar_identity_values() {
        // All-same Q, K → uniform attention → output = mean(V).
        let seq_len = 4;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q = vec![1.0f32; n];
        let k = vec![1.0f32; n];
        let mut v = vec![0.0f32; n];
        fill_random(&mut v, 42);
        let mut output = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut output, seq_len, head_dim, 0.5);

        // With identical Q and K rows, all scores are equal → uniform
        // attention → each output row ≈ mean of all V rows.
        let mean_v: Vec<f32> = (0..head_dim)
            .map(|d| (0..seq_len).map(|s| v[s * head_dim + d]).sum::<f32>() / seq_len as f32)
            .collect();
        for i in 0..seq_len {
            assert_close(
                &output[i * head_dim..(i + 1) * head_dim],
                &mean_v,
                ATOL,
                &format!("uniform attn row {i}"),
            );
        }
    }

    #[test]
    fn test_scalar_vs_naive_small() {
        let seq_len = 4;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 1);
        fill_random(&mut k, 2);
        fill_random(&mut v, 3);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
        assert_close(&out_flash, &out_naive, ATOL, "scalar vs naive");
    }

    #[test]
    fn test_causal_scalar_vs_naive() {
        let seq_len = 6;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 10);
        fill_random(&mut k, 20);
        fill_random(&mut v, 30);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_causal_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, true);
        assert_close(&out_flash, &out_naive, ATOL, "causal scalar vs naive");
    }

    #[test]
    fn test_causal_first_row_equals_v0() {
        // Causal: first query only attends to first key → output = V[0].
        let seq_len = 4;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 100);
        fill_random(&mut k, 200);
        fill_random(&mut v, 300);
        let mut output = vec![0.0f32; n];
        flash_attention_causal_scalar(&q, &k, &v, &mut output, seq_len, head_dim, 0.5);
        assert_close(&output[0..head_dim], &v[0..head_dim], STRICT_ATOL, "causal first row");
    }

    // ── Multi-head scalar tests ────────────────────────────────────────

    #[test]
    fn test_multihead_scalar_single_head() {
        let seq_len = 4;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 7);
        fill_random(&mut k, 8);
        fill_random(&mut v, 9);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out_single = vec![0.0f32; n];
        let mut out_multi = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_single, seq_len, head_dim, scale);
        flash_attention_multihead_scalar(&q, &k, &v, &mut out_multi, 1, seq_len, head_dim, scale);
        assert_close(&out_single, &out_multi, STRICT_ATOL, "multihead 1 == single");
    }

    #[test]
    fn test_multihead_scalar_two_heads() {
        let num_heads = 2;
        let seq_len = 4;
        let head_dim = 4;
        let n = num_heads * seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 11);
        fill_random(&mut k, 12);
        fill_random(&mut v, 13);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out_multi = vec![0.0f32; n];
        flash_attention_multihead_scalar(
            &q,
            &k,
            &v,
            &mut out_multi,
            num_heads,
            seq_len,
            head_dim,
            scale,
        );

        // Each head independently.
        let hs = seq_len * head_dim;
        for h in 0..num_heads {
            let off = h * hs;
            let mut out_single = vec![0.0f32; hs];
            flash_attention_scalar(
                &q[off..off + hs],
                &k[off..off + hs],
                &v[off..off + hs],
                &mut out_single,
                seq_len,
                head_dim,
                scale,
            );
            assert_close(
                &out_multi[off..off + hs],
                &out_single,
                STRICT_ATOL,
                &format!("multihead head {h}"),
            );
        }
    }

    #[test]
    fn test_multihead_causal_scalar() {
        let num_heads = 2;
        let seq_len = 4;
        let head_dim = 4;
        let n = num_heads * seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 14);
        fill_random(&mut k, 15);
        fill_random(&mut v, 16);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out_multi = vec![0.0f32; n];
        flash_attention_multihead_causal_scalar(
            &q,
            &k,
            &v,
            &mut out_multi,
            num_heads,
            seq_len,
            head_dim,
            scale,
        );

        let hs = seq_len * head_dim;
        for h in 0..num_heads {
            let off = h * hs;
            let mut out_single = vec![0.0f32; hs];
            flash_attention_causal_scalar(
                &q[off..off + hs],
                &k[off..off + hs],
                &v[off..off + hs],
                &mut out_single,
                seq_len,
                head_dim,
                scale,
            );
            assert_close(
                &out_multi[off..off + hs],
                &out_single,
                STRICT_ATOL,
                &format!("multihead causal head {h}"),
            );
        }
    }

    // ── Various seq_len / head_dim combinations ────────────────────────

    #[test]
    fn test_scalar_seq1_dim1() {
        let q = vec![0.5];
        let k = vec![0.5];
        let v = vec![3.0];
        let mut output = vec![0.0f32; 1];
        flash_attention_scalar(&q, &k, &v, &mut output, 1, 1, 1.0);
        assert_close(&output, &[3.0], STRICT_ATOL, "1x1");
    }

    #[test]
    fn test_scalar_seq2_dim1() {
        let q = vec![1.0, 1.0];
        let k = vec![1.0, 1.0];
        let v = vec![2.0, 4.0];
        let mut output = vec![0.0f32; 2];
        flash_attention_scalar(&q, &k, &v, &mut output, 2, 1, 1.0);
        // Uniform attention → mean of v = 3.0.
        assert_close(&output, &[3.0, 3.0], ATOL, "2x1 uniform");
    }

    #[test]
    fn test_scalar_seq3_dim4() {
        let seq_len = 3;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 50);
        fill_random(&mut k, 51);
        fill_random(&mut v, 52);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
        assert_close(&out_flash, &out_naive, ATOL, "3x4");
    }

    #[test]
    fn test_scalar_seq8_dim16() {
        let seq_len = 8;
        let head_dim = 16;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 60);
        fill_random(&mut k, 61);
        fill_random(&mut v, 62);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
        assert_close(&out_flash, &out_naive, ATOL, "8x16");
    }

    #[test]
    fn test_scalar_seq16_dim32() {
        let seq_len = 16;
        let head_dim = 32;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 70);
        fill_random(&mut k, 71);
        fill_random(&mut v, 72);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
        assert_close(&out_flash, &out_naive, ATOL, "16x32");
    }

    #[test]
    fn test_scalar_seq32_dim64() {
        let seq_len = 32;
        let head_dim = 64;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 80);
        fill_random(&mut k, 81);
        fill_random(&mut v, 82);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
        assert_close(&out_flash, &out_naive, ATOL, "32x64");
    }

    #[test]
    fn test_scalar_seq64_dim128() {
        let seq_len = 64;
        let head_dim = 128;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 90);
        fill_random(&mut k, 91);
        fill_random(&mut v, 92);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
        assert_close(&out_flash, &out_naive, ATOL, "64x128");
    }

    // Non-power-of-2 sizes.
    #[test]
    fn test_scalar_seq5_dim7() {
        let seq_len = 5;
        let head_dim = 7;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 101);
        fill_random(&mut k, 102);
        fill_random(&mut v, 103);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
        assert_close(&out_flash, &out_naive, ATOL, "5x7");
    }

    #[test]
    fn test_scalar_seq13_dim3() {
        let seq_len = 13;
        let head_dim = 3;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 111);
        fill_random(&mut k, 112);
        fill_random(&mut v, 113);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
        assert_close(&out_flash, &out_naive, ATOL, "13x3");
    }

    // ── Causal with various sizes ──────────────────────────────────────

    #[test]
    fn test_causal_scalar_seq8_dim16() {
        let seq_len = 8;
        let head_dim = 16;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 120);
        fill_random(&mut k, 121);
        fill_random(&mut v, 122);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_causal_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, true);
        assert_close(&out_flash, &out_naive, ATOL, "causal 8x16");
    }

    #[test]
    fn test_causal_scalar_seq16_dim64() {
        let seq_len = 16;
        let head_dim = 64;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 130);
        fill_random(&mut k, 131);
        fill_random(&mut v, 132);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_causal_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, true);
        assert_close(&out_flash, &out_naive, ATOL, "causal 16x64");
    }

    #[test]
    fn test_causal_scalar_seq33_dim5() {
        let seq_len = 33;
        let head_dim = 5;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 140);
        fill_random(&mut k, 141);
        fill_random(&mut v, 142);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_causal_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, true);
        assert_close(&out_flash, &out_naive, ATOL, "causal 33x5");
    }

    // ── Scale factor tests ─────────────────────────────────────────────

    #[test]
    fn test_scalar_scale_zero() {
        // Scale 0 → all scores are 0 → uniform attention.
        let seq_len = 4;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 200);
        fill_random(&mut k, 201);
        fill_random(&mut v, 202);
        let mut output = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut output, seq_len, head_dim, 0.0);

        let mean_v: Vec<f32> = (0..head_dim)
            .map(|d| (0..seq_len).map(|s| v[s * head_dim + d]).sum::<f32>() / seq_len as f32)
            .collect();
        for i in 0..seq_len {
            assert_close(
                &output[i * head_dim..(i + 1) * head_dim],
                &mean_v,
                ATOL,
                &format!("scale=0 row {i}"),
            );
        }
    }

    #[test]
    fn test_scalar_small_scale() {
        let seq_len = 4;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 210);
        fill_random(&mut k, 211);
        fill_random(&mut v, 212);
        let scale = 0.001;
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
        assert_close(&out_flash, &out_naive, ATOL, "small scale");
    }

    #[test]
    fn test_scalar_large_scale() {
        let seq_len = 4;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 220);
        fill_random(&mut k, 221);
        fill_random(&mut v, 222);
        let scale = 10.0;
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
        // Larger tolerance for large scale (sharper softmax).
        assert_close(&out_flash, &out_naive, 5e-2, "large scale");
    }

    // ── Numerical stability ────────────────────────────────────────────

    #[test]
    fn test_scalar_no_nan_with_large_values() {
        let seq_len = 4;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q = vec![100.0f32; n];
        let k = vec![100.0f32; n];
        let mut v = vec![0.0f32; n];
        fill_random(&mut v, 300);
        let mut output = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut output, seq_len, head_dim, 1.0);
        for &val in &output {
            assert!(!val.is_nan(), "NaN detected");
            assert!(!val.is_infinite(), "Inf detected");
        }
    }

    #[test]
    fn test_scalar_no_nan_with_negative_values() {
        let seq_len = 4;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q = vec![-50.0f32; n];
        let k = vec![-50.0f32; n];
        let mut v = vec![0.0f32; n];
        fill_random(&mut v, 310);
        let mut output = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut output, seq_len, head_dim, 1.0);
        for &val in &output {
            assert!(!val.is_nan(), "NaN detected");
            assert!(!val.is_infinite(), "Inf detected");
        }
    }

    #[test]
    fn test_scalar_output_bounded() {
        // Output should be in the convex hull of V rows.
        let seq_len = 8;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 320);
        fill_random(&mut k, 321);
        fill_random(&mut v, 322);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut output, seq_len, head_dim, scale);

        for d in 0..head_dim {
            let v_min = (0..seq_len).map(|s| v[s * head_dim + d]).fold(f32::INFINITY, f32::min);
            let v_max = (0..seq_len).map(|s| v[s * head_dim + d]).fold(f32::NEG_INFINITY, f32::max);
            for s in 0..seq_len {
                let o = output[s * head_dim + d];
                assert!(
                    o >= v_min - 1e-3 && o <= v_max + 1e-3,
                    "output[{s},{d}]={o} outside [{v_min},{v_max}]",
                );
            }
        }
    }

    // ── Large sequence (crosses tile boundary) ─────────────────────────

    #[test]
    fn test_scalar_large_sequence() {
        let seq_len = 128;
        let head_dim = 16;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 400);
        fill_random(&mut k, 401);
        fill_random(&mut v, 402);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
        assert_close(&out_flash, &out_naive, ATOL, "large seq 128x16");
    }

    #[test]
    fn test_causal_scalar_large_sequence() {
        let seq_len = 128;
        let head_dim = 16;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 410);
        fill_random(&mut k, 411);
        fill_random(&mut v, 412);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_flash = vec![0.0f32; n];
        let mut out_naive = vec![0.0f32; n];
        flash_attention_causal_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
        naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, true);
        assert_close(&out_flash, &out_naive, ATOL, "causal large seq 128x16");
    }

    // ── Determinism ────────────────────────────────────────────────────

    #[test]
    fn test_scalar_deterministic() {
        let seq_len = 8;
        let head_dim = 16;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 500);
        fill_random(&mut k, 501);
        fill_random(&mut v, 502);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out1 = vec![0.0f32; n];
        let mut out2 = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut out1, seq_len, head_dim, scale);
        flash_attention_scalar(&q, &k, &v, &mut out2, seq_len, head_dim, scale);
        assert_eq!(out1, out2, "scalar not deterministic");
    }

    // ── Multihead with many heads ──────────────────────────────────────

    #[test]
    fn test_multihead_scalar_four_heads() {
        let num_heads = 4;
        let seq_len = 8;
        let head_dim = 8;
        let n = num_heads * seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 600);
        fill_random(&mut k, 601);
        fill_random(&mut v, 602);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out_multi = vec![0.0f32; n];
        flash_attention_multihead_scalar(
            &q,
            &k,
            &v,
            &mut out_multi,
            num_heads,
            seq_len,
            head_dim,
            scale,
        );

        let hs = seq_len * head_dim;
        for h in 0..num_heads {
            let off = h * hs;
            let mut out_single = vec![0.0f32; hs];
            flash_attention_scalar(
                &q[off..off + hs],
                &k[off..off + hs],
                &v[off..off + hs],
                &mut out_single,
                seq_len,
                head_dim,
                scale,
            );
            assert_close(
                &out_multi[off..off + hs],
                &out_single,
                STRICT_ATOL,
                &format!("4-head head {h}"),
            );
        }
    }

    #[test]
    fn test_multihead_scalar_eight_heads() {
        let num_heads = 8;
        let seq_len = 4;
        let head_dim = 16;
        let n = num_heads * seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 610);
        fill_random(&mut k, 611);
        fill_random(&mut v, 612);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; n];
        flash_attention_multihead_scalar(
            &q,
            &k,
            &v,
            &mut output,
            num_heads,
            seq_len,
            head_dim,
            scale,
        );
        for &val in &output {
            assert!(!val.is_nan(), "NaN in 8-head output");
        }
    }

    // ── All-zeros input ────────────────────────────────────────────────

    #[test]
    fn test_scalar_all_zeros_q() {
        let seq_len = 4;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q = vec![0.0f32; n];
        let k = vec![1.0f32; n];
        let mut v = vec![0.0f32; n];
        fill_random(&mut v, 700);
        let mut output = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut output, seq_len, head_dim, 1.0);
        for &val in &output {
            assert!(!val.is_nan(), "NaN with zero Q");
        }
    }

    #[test]
    fn test_scalar_all_zeros_kv() {
        let seq_len = 4;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let mut q = vec![0.0f32; n];
        fill_random(&mut q, 710);
        let k = vec![0.0f32; n];
        let v = vec![0.0f32; n];
        let mut output = vec![0.0f32; n];
        flash_attention_scalar(&q, &k, &v, &mut output, seq_len, head_dim, 1.0);
        // All V are zero → output should be zero.
        for &val in &output {
            assert!(val.abs() < 1e-6, "expected ~0 with zero V, got {val}");
        }
    }

    // ── Dot product helpers ────────────────────────────────────────────

    #[test]
    fn test_dot_product_scalar_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let result = dot_product_scalar(&a, &b);
        assert!((result - 70.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_scalar_non_aligned() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0];
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let result = dot_product_scalar(&a, &b);
        assert!((result - expected).abs() < 1e-5);
    }

    // ── Fast exp tests ─────────────────────────────────────────────────

    #[test]
    fn test_fast_exp_zero() {
        let result = fast_exp_scalar(0.0);
        assert!((result - 1.0).abs() < 1e-3, "exp(0) = {result}");
    }

    #[test]
    fn test_fast_exp_one() {
        let result = fast_exp_scalar(1.0);
        assert!((result - std::f32::consts::E).abs() < 0.01, "exp(1) = {result}");
    }

    #[test]
    fn test_fast_exp_negative() {
        let result = fast_exp_scalar(-1.0);
        let expected = (-1.0f32).exp();
        assert!((result - expected).abs() < 0.01, "exp(-1) = {result}");
    }

    #[test]
    fn test_fast_exp_clamp_large() {
        let result = fast_exp_scalar(200.0);
        assert!(!result.is_nan(), "exp(200) should not be NaN");
        assert!(!result.is_infinite(), "exp(200) clamped, should be finite");
    }

    #[test]
    fn test_fast_exp_clamp_neg_large() {
        let result = fast_exp_scalar(-200.0);
        assert!(!result.is_nan(), "exp(-200) should not be NaN");
        assert!(result >= 0.0, "exp(-200) should be non-negative");
    }

    // ── Causal masking property ────────────────────────────────────────

    #[test]
    fn test_causal_last_row_equals_noncausal() {
        // Last row of causal attention = last row of non-causal
        // (attends to all positions in both cases).
        let seq_len = 8;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 800);
        fill_random(&mut k, 801);
        fill_random(&mut v, 802);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_causal = vec![0.0f32; n];
        let mut out_full = vec![0.0f32; n];
        flash_attention_causal_scalar(&q, &k, &v, &mut out_causal, seq_len, head_dim, scale);
        flash_attention_scalar(&q, &k, &v, &mut out_full, seq_len, head_dim, scale);
        let last = (seq_len - 1) * head_dim;
        assert_close(
            &out_causal[last..last + head_dim],
            &out_full[last..last + head_dim],
            STRICT_ATOL,
            "causal last row == full last row",
        );
    }

    // ── NEON tests (aarch64 only) ──────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    mod neon_tests {
        use super::*;

        #[test]
        fn test_neon_dot_product() {
            let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..16).map(|i| (i * 2) as f32).collect();
            let expected = dot_product_scalar(&a, &b);
            let result = unsafe { dot_product_neon(&a, &b) };
            assert!((result - expected).abs() < 1e-4, "NEON dot: {result} vs {expected}");
        }

        #[test]
        fn test_neon_dot_product_non_aligned() {
            let a: Vec<f32> = (0..7).map(|i| i as f32 + 0.5).collect();
            let b: Vec<f32> = (0..7).map(|i| i as f32 * 0.3).collect();
            let expected = dot_product_scalar(&a, &b);
            let result = unsafe { dot_product_neon(&a, &b) };
            assert!(
                (result - expected).abs() < 1e-4,
                "NEON dot non-aligned: {result} vs {expected}"
            );
        }

        #[test]
        fn test_neon_vs_scalar_basic() {
            let seq_len = 4;
            let head_dim = 8;
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, 1000);
            fill_random(&mut k, 1001);
            fill_random(&mut v, 1002);
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut out_scalar = vec![0.0f32; n];
            let mut out_neon = vec![0.0f32; n];
            flash_attention_scalar(&q, &k, &v, &mut out_scalar, seq_len, head_dim, scale);
            unsafe {
                flash_attention_neon(&q, &k, &v, &mut out_neon, seq_len, head_dim, scale);
            }
            assert_close(&out_neon, &out_scalar, STRICT_ATOL, "NEON vs scalar basic");
        }

        #[test]
        fn test_neon_vs_scalar_causal() {
            let seq_len = 8;
            let head_dim = 16;
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, 1010);
            fill_random(&mut k, 1011);
            fill_random(&mut v, 1012);
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut out_scalar = vec![0.0f32; n];
            let mut out_neon = vec![0.0f32; n];
            flash_attention_causal_scalar(&q, &k, &v, &mut out_scalar, seq_len, head_dim, scale);
            unsafe {
                flash_attention_causal_neon(&q, &k, &v, &mut out_neon, seq_len, head_dim, scale);
            }
            assert_close(&out_neon, &out_scalar, STRICT_ATOL, "NEON vs scalar causal");
        }

        #[test]
        fn test_neon_vs_scalar_multihead() {
            let num_heads = 2;
            let seq_len = 4;
            let head_dim = 8;
            let n = num_heads * seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, 1020);
            fill_random(&mut k, 1021);
            fill_random(&mut v, 1022);
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut out_scalar = vec![0.0f32; n];
            let mut out_neon = vec![0.0f32; n];
            flash_attention_multihead_scalar(
                &q,
                &k,
                &v,
                &mut out_scalar,
                num_heads,
                seq_len,
                head_dim,
                scale,
            );
            unsafe {
                flash_attention_multihead_neon(
                    &q,
                    &k,
                    &v,
                    &mut out_neon,
                    num_heads,
                    seq_len,
                    head_dim,
                    scale,
                );
            }
            assert_close(&out_neon, &out_scalar, STRICT_ATOL, "NEON vs scalar multihead");
        }

        #[test]
        fn test_neon_vs_scalar_multihead_causal() {
            let num_heads = 2;
            let seq_len = 8;
            let head_dim = 16;
            let n = num_heads * seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, 1030);
            fill_random(&mut k, 1031);
            fill_random(&mut v, 1032);
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut out_scalar = vec![0.0f32; n];
            let mut out_neon = vec![0.0f32; n];
            flash_attention_multihead_causal_scalar(
                &q,
                &k,
                &v,
                &mut out_scalar,
                num_heads,
                seq_len,
                head_dim,
                scale,
            );
            unsafe {
                flash_attention_multihead_causal_neon(
                    &q,
                    &k,
                    &v,
                    &mut out_neon,
                    num_heads,
                    seq_len,
                    head_dim,
                    scale,
                );
            }
            assert_close(&out_neon, &out_scalar, STRICT_ATOL, "NEON vs scalar multihead causal");
        }

        #[test]
        fn test_neon_empty_seq() {
            let mut output = vec![0.0f32; 0];
            unsafe {
                flash_attention_neon(&[], &[], &[], &mut output, 0, 4, 1.0);
            }
            assert!(output.is_empty());
        }

        #[test]
        fn test_neon_single_element() {
            let q = vec![1.0f32; 4];
            let k = vec![1.0f32; 4];
            let v = vec![2.0, 3.0, 4.0, 5.0];
            let mut output = vec![0.0f32; 4];
            unsafe {
                flash_attention_neon(&q, &k, &v, &mut output, 1, 4, 0.5);
            }
            assert_close(&output, &v, STRICT_ATOL, "NEON single element");
        }

        #[test]
        fn test_neon_large_sequence() {
            let seq_len = 128;
            let head_dim = 32;
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, 1100);
            fill_random(&mut k, 1101);
            fill_random(&mut v, 1102);
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut out_scalar = vec![0.0f32; n];
            let mut out_neon = vec![0.0f32; n];
            flash_attention_scalar(&q, &k, &v, &mut out_scalar, seq_len, head_dim, scale);
            unsafe {
                flash_attention_neon(&q, &k, &v, &mut out_neon, seq_len, head_dim, scale);
            }
            assert_close(&out_neon, &out_scalar, STRICT_ATOL, "NEON vs scalar large");
        }

        #[test]
        fn test_neon_non_aligned_dim() {
            let seq_len = 4;
            let head_dim = 7;
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, 1200);
            fill_random(&mut k, 1201);
            fill_random(&mut v, 1202);
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut out_scalar = vec![0.0f32; n];
            let mut out_neon = vec![0.0f32; n];
            flash_attention_scalar(&q, &k, &v, &mut out_scalar, seq_len, head_dim, scale);
            unsafe {
                flash_attention_neon(&q, &k, &v, &mut out_neon, seq_len, head_dim, scale);
            }
            assert_close(&out_neon, &out_scalar, STRICT_ATOL, "NEON non-aligned dim");
        }

        #[test]
        fn test_neon_vs_naive() {
            let seq_len = 8;
            let head_dim = 16;
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, 1300);
            fill_random(&mut k, 1301);
            fill_random(&mut v, 1302);
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut out_neon = vec![0.0f32; n];
            let mut out_naive = vec![0.0f32; n];
            unsafe {
                flash_attention_neon(&q, &k, &v, &mut out_neon, seq_len, head_dim, scale);
            }
            naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
            assert_close(&out_neon, &out_naive, ATOL, "NEON vs naive");
        }

        #[test]
        fn test_neon_causal_vs_naive() {
            let seq_len = 8;
            let head_dim = 16;
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, 1310);
            fill_random(&mut k, 1311);
            fill_random(&mut v, 1312);
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut out_neon = vec![0.0f32; n];
            let mut out_naive = vec![0.0f32; n];
            unsafe {
                flash_attention_causal_neon(&q, &k, &v, &mut out_neon, seq_len, head_dim, scale);
            }
            naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, true);
            assert_close(&out_neon, &out_naive, ATOL, "NEON causal vs naive");
        }

        #[test]
        fn test_neon_deterministic() {
            let seq_len = 8;
            let head_dim = 16;
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, 1400);
            fill_random(&mut k, 1401);
            fill_random(&mut v, 1402);
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut out1 = vec![0.0f32; n];
            let mut out2 = vec![0.0f32; n];
            unsafe {
                flash_attention_neon(&q, &k, &v, &mut out1, seq_len, head_dim, scale);
                flash_attention_neon(&q, &k, &v, &mut out2, seq_len, head_dim, scale);
            }
            assert_eq!(out1, out2, "NEON not deterministic");
        }

        #[test]
        fn test_neon_no_nan_large_values() {
            let seq_len = 4;
            let head_dim = 8;
            let n = seq_len * head_dim;
            let q = vec![100.0f32; n];
            let k = vec![100.0f32; n];
            let mut v = vec![0.0f32; n];
            fill_random(&mut v, 1500);
            let mut output = vec![0.0f32; n];
            unsafe {
                flash_attention_neon(&q, &k, &v, &mut output, seq_len, head_dim, 1.0);
            }
            for &val in &output {
                assert!(!val.is_nan(), "NaN in NEON with large values");
                assert!(!val.is_infinite(), "Inf in NEON with large values");
            }
        }

        #[test]
        fn test_neon_scale_zero() {
            let seq_len = 4;
            let head_dim = 8;
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, 1600);
            fill_random(&mut k, 1601);
            fill_random(&mut v, 1602);

            let mut out_scalar = vec![0.0f32; n];
            let mut out_neon = vec![0.0f32; n];
            flash_attention_scalar(&q, &k, &v, &mut out_scalar, seq_len, head_dim, 0.0);
            unsafe {
                flash_attention_neon(&q, &k, &v, &mut out_neon, seq_len, head_dim, 0.0);
            }
            assert_close(&out_neon, &out_scalar, STRICT_ATOL, "NEON scale=0 vs scalar");
        }
    }

    // ── Property-like parametric tests ─────────────────────────────────

    #[test]
    fn test_property_output_rows_sum_to_convex_hull() {
        // For multiple random seeds, verify output is in convex hull of V.
        for seed in 0..5u64 {
            let seq_len = 8;
            let head_dim = 4;
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, seed * 3);
            fill_random(&mut k, seed * 3 + 1);
            fill_random(&mut v, seed * 3 + 2);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut output = vec![0.0f32; n];
            flash_attention_scalar(&q, &k, &v, &mut output, seq_len, head_dim, scale);

            for d in 0..head_dim {
                let v_min = (0..seq_len).map(|s| v[s * head_dim + d]).fold(f32::INFINITY, f32::min);
                let v_max =
                    (0..seq_len).map(|s| v[s * head_dim + d]).fold(f32::NEG_INFINITY, f32::max);
                for s in 0..seq_len {
                    let o = output[s * head_dim + d];
                    assert!(
                        o >= v_min - 1e-3 && o <= v_max + 1e-3,
                        "seed={seed} [{s},{d}] {o} not in [{v_min},{v_max}]",
                    );
                }
            }
        }
    }

    #[test]
    fn test_property_causal_monotone_context() {
        // Causal: later rows attend to strictly more context.
        // Verify row 0 output = V[0] for all seeds.
        for seed in 0..5u64 {
            let seq_len = 6;
            let head_dim = 4;
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, 900 + seed);
            fill_random(&mut k, 910 + seed);
            fill_random(&mut v, 920 + seed);
            let mut output = vec![0.0f32; n];
            flash_attention_causal_scalar(&q, &k, &v, &mut output, seq_len, head_dim, 0.5);
            assert_close(
                &output[0..head_dim],
                &v[0..head_dim],
                STRICT_ATOL,
                &format!("causal row0 seed={seed}"),
            );
        }
    }

    #[test]
    fn test_property_multihead_independence() {
        // Changing data in head 0 should not affect head 1.
        let num_heads = 2;
        let seq_len = 4;
        let head_dim = 4;
        let hs = seq_len * head_dim;
        let n = num_heads * hs;
        let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        fill_random(&mut q, 950);
        fill_random(&mut k, 951);
        fill_random(&mut v, 952);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out1 = vec![0.0f32; n];
        flash_attention_multihead_scalar(
            &q, &k, &v, &mut out1, num_heads, seq_len, head_dim, scale,
        );

        // Mutate head 0 data.
        let mut q2 = q.clone();
        for i in 0..hs {
            q2[i] *= 2.0;
        }
        let mut out2 = vec![0.0f32; n];
        flash_attention_multihead_scalar(
            &q2, &k, &v, &mut out2, num_heads, seq_len, head_dim, scale,
        );

        // Head 1 should be unchanged.
        assert_close(&out1[hs..], &out2[hs..], STRICT_ATOL, "head independence");
    }

    #[test]
    fn test_property_no_nan_various_seeds() {
        for seed in 0..10u64 {
            let seq_len = 8;
            let head_dim = 8;
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, seed * 7);
            fill_random(&mut k, seed * 7 + 1);
            fill_random(&mut v, seed * 7 + 2);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut output = vec![0.0f32; n];
            flash_attention_scalar(&q, &k, &v, &mut output, seq_len, head_dim, scale);
            for (i, &val) in output.iter().enumerate() {
                assert!(!val.is_nan(), "NaN at {i} with seed {seed}");
            }
        }
    }

    #[test]
    fn test_property_scalar_vs_naive_sweep() {
        for (seq_len, head_dim) in [(2, 4), (4, 8), (8, 16), (16, 4), (5, 3), (7, 11)] {
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, (seq_len + head_dim) as u64);
            fill_random(&mut k, (seq_len + head_dim + 1) as u64);
            fill_random(&mut v, (seq_len + head_dim + 2) as u64);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut out_flash = vec![0.0f32; n];
            let mut out_naive = vec![0.0f32; n];
            flash_attention_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
            naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, false);
            assert_close(&out_flash, &out_naive, ATOL, &format!("sweep {seq_len}x{head_dim}"));
        }
    }

    #[test]
    fn test_property_causal_vs_naive_sweep() {
        for (seq_len, head_dim) in [(2, 4), (4, 8), (8, 16), (16, 4), (5, 3)] {
            let n = seq_len * head_dim;
            let (mut q, mut k, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            fill_random(&mut q, (seq_len + head_dim + 100) as u64);
            fill_random(&mut k, (seq_len + head_dim + 101) as u64);
            fill_random(&mut v, (seq_len + head_dim + 102) as u64);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut out_flash = vec![0.0f32; n];
            let mut out_naive = vec![0.0f32; n];
            flash_attention_causal_scalar(&q, &k, &v, &mut out_flash, seq_len, head_dim, scale);
            naive_attention(&q, &k, &v, &mut out_naive, seq_len, head_dim, scale, true);
            assert_close(
                &out_flash,
                &out_naive,
                ATOL,
                &format!("causal sweep {seq_len}x{head_dim}"),
            );
        }
    }
}
