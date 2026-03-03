//! NEON-optimized attention score computation for Apple Silicon.
//!
//! Provides six attention-score kernels with NEON-accelerated inner loops
//! and scalar fallback implementations for testing on non-aarch64 hosts:
//!
//! 1. `neon_attention_score` — single-head Q·Kᵀ / √d_k
//! 2. `neon_attention_score_multi_head` — batched multi-head scores
//! 3. `neon_attention_score_masked` — causal (lower-triangular) masking
//! 4. `neon_attention_score_gqa` — grouped-query attention (fewer KV heads)
//! 5. `neon_attention_weighted_sum` — attention_weights · V
//! 6. `neon_attention_score_with_alibi` — ALiBi position-bias addition
//!
//! All NEON paths require `target_arch = "aarch64"` and are gated behind
//! `#[target_feature(enable = "neon")]`.  Each public entry point performs
//! runtime feature detection and falls back to the scalar reference
//! implementation when NEON is unavailable.

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(
    clippy::missing_safety_doc,
    clippy::float_cmp,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::let_and_return,
    clippy::excessive_precision
)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ════════════════════════════════════════════════════════════════════════
// Scalar reference implementations
// ════════════════════════════════════════════════════════════════════════

/// Scalar softmax in-place (max-subtract-exp-normalize).
fn scalar_softmax_inplace(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in data.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in data.iter_mut() {
            *v /= sum;
        }
    }
}

/// Scalar dot product of two slices.
fn scalar_dot(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar: single-head Q·Kᵀ / √d_k.
///
/// * `q`      – query matrix, shape `[q_len, head_dim]` (row-major)
/// * `k`      – key matrix,   shape `[kv_len, head_dim]` (row-major)
/// * `output` – scores,       shape `[q_len, kv_len]`   (row-major)
/// * `scale`  – typically `1.0 / sqrt(head_dim)`
pub fn scalar_attention_score(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) {
    for i in 0..q_len {
        for j in 0..kv_len {
            let dot = scalar_dot(&q[i * head_dim..], &k[j * head_dim..], head_dim);
            output[i * kv_len + j] = dot * scale;
        }
    }
}

/// Scalar: multi-head attention scores.
///
/// `q`, `k` are `[num_heads, seq_len, head_dim]`.
/// `output` is `[num_heads, q_len, kv_len]`.
pub fn scalar_attention_score_multi_head(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    num_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let q_head_stride = q_len * head_dim;
    let k_head_stride = kv_len * head_dim;
    let o_head_stride = q_len * kv_len;
    for h in 0..num_heads {
        let q_off = h * q_head_stride;
        let k_off = h * k_head_stride;
        let o_off = h * o_head_stride;
        scalar_attention_score(
            &q[q_off..q_off + q_head_stride],
            &k[k_off..k_off + k_head_stride],
            &mut output[o_off..o_off + o_head_stride],
            q_len,
            kv_len,
            head_dim,
            scale,
        );
    }
}

/// Scalar: causal-masked attention scores.
///
/// Positions where `j > i + kv_len - q_len` are set to `mask_value`
/// (typically `f32::NEG_INFINITY`).
pub fn scalar_attention_score_masked(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    mask_value: f32,
) {
    scalar_attention_score(q, k, output, q_len, kv_len, head_dim, scale);
    let offset = kv_len as isize - q_len as isize;
    for i in 0..q_len {
        for j in 0..kv_len {
            if j as isize > i as isize + offset {
                output[i * kv_len + j] = mask_value;
            }
        }
    }
}

/// Scalar: grouped-query attention scores.
///
/// `q` is `[num_q_heads, q_len, head_dim]`.
/// `k` is `[num_kv_heads, kv_len, head_dim]` where `num_q_heads` is a
/// multiple of `num_kv_heads`.
/// `output` is `[num_q_heads, q_len, kv_len]`.
pub fn scalar_attention_score_gqa(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let group_size = num_q_heads / num_kv_heads;
    let q_head_stride = q_len * head_dim;
    let k_head_stride = kv_len * head_dim;
    let o_head_stride = q_len * kv_len;
    for qh in 0..num_q_heads {
        let kh = qh / group_size;
        let q_off = qh * q_head_stride;
        let k_off = kh * k_head_stride;
        let o_off = qh * o_head_stride;
        scalar_attention_score(
            &q[q_off..q_off + q_head_stride],
            &k[k_off..k_off + k_head_stride],
            &mut output[o_off..o_off + o_head_stride],
            q_len,
            kv_len,
            head_dim,
            scale,
        );
    }
}

/// Scalar: attention-weighted sum (attention_weights · V).
///
/// * `weights` – `[q_len, kv_len]` (already softmax'd)
/// * `v`       – `[kv_len, head_dim]`
/// * `output`  – `[q_len, head_dim]`
pub fn scalar_attention_weighted_sum(
    weights: &[f32],
    v: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
) {
    for i in 0..q_len {
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for j in 0..kv_len {
                sum += weights[i * kv_len + j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = sum;
        }
    }
}

/// Scalar: ALiBi position-biased attention scores.
///
/// Adds `slope * (j - i - offset)` to each raw score, where
/// `offset = kv_len - q_len`.  The slope is head-specific and derived
/// from the ALiBi schedule.
pub fn scalar_attention_score_with_alibi(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    alibi_slope: f32,
) {
    scalar_attention_score(q, k, output, q_len, kv_len, head_dim, scale);
    let offset = kv_len as isize - q_len as isize;
    for i in 0..q_len {
        for j in 0..kv_len {
            let distance = j as f32 - i as f32 - offset as f32;
            output[i * kv_len + j] += alibi_slope * distance;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// NEON-accelerated implementations
// ════════════════════════════════════════════════════════════════════════

/// NEON dot product of two f32 slices of length `len`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut i = 0usize;
    let mut acc = vdupq_n_f32(0.0);
    while i + LANES <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        acc = vfmaq_f32(acc, va, vb);
        i += LANES;
    }
    let mut sum = vaddvq_f32(acc);
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// NEON-accelerated single-head Q·Kᵀ / √d_k.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_attention_score_inner(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) {
    for i in 0..q_len {
        let q_row = &q[i * head_dim..(i + 1) * head_dim];
        for j in 0..kv_len {
            let k_row = &k[j * head_dim..(j + 1) * head_dim];
            let dot = neon_dot_f32(q_row, k_row, head_dim);
            output[i * kv_len + j] = dot * scale;
        }
    }
}

/// NEON-accelerated multi-head attention scores.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_attention_score_multi_head_inner(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    num_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let q_head_stride = q_len * head_dim;
    let k_head_stride = kv_len * head_dim;
    let o_head_stride = q_len * kv_len;
    for h in 0..num_heads {
        let q_off = h * q_head_stride;
        let k_off = h * k_head_stride;
        let o_off = h * o_head_stride;
        neon_attention_score_inner(
            &q[q_off..q_off + q_head_stride],
            &k[k_off..k_off + k_head_stride],
            &mut output[o_off..o_off + o_head_stride],
            q_len,
            kv_len,
            head_dim,
            scale,
        );
    }
}

/// NEON-accelerated causal-masked attention scores.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_attention_score_masked_inner(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    mask_value: f32,
) {
    neon_attention_score_inner(q, k, output, q_len, kv_len, head_dim, scale);
    let offset = kv_len as isize - q_len as isize;
    for i in 0..q_len {
        let row_start = i * kv_len;
        let causal_end = (i as isize + offset + 1) as usize;
        let mask_start = causal_end.min(kv_len);
        // Vectorised fill for the masked region.
        let mask_vec = vdupq_n_f32(mask_value);
        let mut j = mask_start;
        while j + LANES <= kv_len {
            vst1q_f32(output.as_mut_ptr().add(row_start + j), mask_vec);
            j += LANES;
        }
        while j < kv_len {
            output[row_start + j] = mask_value;
            j += 1;
        }
    }
}

/// NEON-accelerated grouped-query attention scores.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_attention_score_gqa_inner(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let group_size = num_q_heads / num_kv_heads;
    let q_head_stride = q_len * head_dim;
    let k_head_stride = kv_len * head_dim;
    let o_head_stride = q_len * kv_len;
    for qh in 0..num_q_heads {
        let kh = qh / group_size;
        let q_off = qh * q_head_stride;
        let k_off = kh * k_head_stride;
        let o_off = qh * o_head_stride;
        neon_attention_score_inner(
            &q[q_off..q_off + q_head_stride],
            &k[k_off..k_off + k_head_stride],
            &mut output[o_off..o_off + o_head_stride],
            q_len,
            kv_len,
            head_dim,
            scale,
        );
    }
}

/// NEON-accelerated attention-weighted sum (weights · V).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_attention_weighted_sum_inner(
    weights: &[f32],
    v: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
) {
    for i in 0..q_len {
        // Zero the output row.
        let mut d = 0usize;
        while d + LANES <= head_dim {
            vst1q_f32(output.as_mut_ptr().add(i * head_dim + d), vdupq_n_f32(0.0));
            d += LANES;
        }
        while d < head_dim {
            output[i * head_dim + d] = 0.0;
            d += 1;
        }
        // Accumulate weighted V rows.
        for j in 0..kv_len {
            let w = weights[i * kv_len + j];
            if w == 0.0 {
                continue;
            }
            let w_vec = vdupq_n_f32(w);
            let mut d2 = 0usize;
            while d2 + LANES <= head_dim {
                let cur = vld1q_f32(output.as_ptr().add(i * head_dim + d2));
                let vv = vld1q_f32(v.as_ptr().add(j * head_dim + d2));
                let res = vfmaq_f32(cur, w_vec, vv);
                vst1q_f32(output.as_mut_ptr().add(i * head_dim + d2), res);
                d2 += LANES;
            }
            while d2 < head_dim {
                output[i * head_dim + d2] += w * v[j * head_dim + d2];
                d2 += 1;
            }
        }
    }
}

/// NEON-accelerated ALiBi-biased attention scores.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_attention_score_with_alibi_inner(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    alibi_slope: f32,
) {
    neon_attention_score_inner(q, k, output, q_len, kv_len, head_dim, scale);
    let offset = kv_len as isize - q_len as isize;
    let slope_vec = vdupq_n_f32(alibi_slope);
    for i in 0..q_len {
        let base = -(i as f32) - offset as f32;
        let mut j = 0usize;
        // NEON: process 4 positions at a time.
        while j + LANES <= kv_len {
            let idx = vaddq_f32(
                vdupq_n_f32(base + j as f32),
                vcvtq_f32_s32(vld1q_s32([0i32, 1, 2, 3].as_ptr())),
            );
            let bias = vmulq_f32(slope_vec, idx);
            let cur = vld1q_f32(output.as_ptr().add(i * kv_len + j));
            let res = vaddq_f32(cur, bias);
            vst1q_f32(output.as_mut_ptr().add(i * kv_len + j), res);
            j += LANES;
        }
        while j < kv_len {
            let distance = j as f32 - i as f32 - offset as f32;
            output[i * kv_len + j] += alibi_slope * distance;
            j += 1;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Public dispatch entry points
// ════════════════════════════════════════════════════════════════════════

/// Compute Q·Kᵀ / √d_k for a single attention head.
///
/// `q` is `[q_len, head_dim]`, `k` is `[kv_len, head_dim]`,
/// `output` is `[q_len, kv_len]`.
pub fn neon_attention_score(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_attention_score_inner(q, k, output, q_len, kv_len, head_dim, scale);
            }
            return;
        }
    }
    scalar_attention_score(q, k, output, q_len, kv_len, head_dim, scale);
}

/// Compute multi-head attention scores.
///
/// `q`, `k` are `[num_heads, seq_len, head_dim]`.
/// `output` is `[num_heads, q_len, kv_len]`.
pub fn neon_attention_score_multi_head(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    num_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_attention_score_multi_head_inner(
                    q, k, output, num_heads, q_len, kv_len, head_dim, scale,
                );
            }
            return;
        }
    }
    scalar_attention_score_multi_head(q, k, output, num_heads, q_len, kv_len, head_dim, scale);
}

/// Compute causal-masked attention scores.
///
/// Future positions (j > i + kv_len - q_len) are set to `mask_value`.
pub fn neon_attention_score_masked(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    mask_value: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_attention_score_masked_inner(
                    q, k, output, q_len, kv_len, head_dim, scale, mask_value,
                );
            }
            return;
        }
    }
    scalar_attention_score_masked(q, k, output, q_len, kv_len, head_dim, scale, mask_value);
}

/// Compute grouped-query attention scores.
///
/// `num_q_heads` must be a multiple of `num_kv_heads`.
pub fn neon_attention_score_gqa(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_attention_score_gqa_inner(
                    q,
                    k,
                    output,
                    num_q_heads,
                    num_kv_heads,
                    q_len,
                    kv_len,
                    head_dim,
                    scale,
                );
            }
            return;
        }
    }
    scalar_attention_score_gqa(
        q,
        k,
        output,
        num_q_heads,
        num_kv_heads,
        q_len,
        kv_len,
        head_dim,
        scale,
    );
}

/// Compute attention_weights · V.
///
/// `weights` is `[q_len, kv_len]`, `v` is `[kv_len, head_dim]`,
/// `output` is `[q_len, head_dim]`.
pub fn neon_attention_weighted_sum(
    weights: &[f32],
    v: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_attention_weighted_sum_inner(weights, v, output, q_len, kv_len, head_dim);
            }
            return;
        }
    }
    scalar_attention_weighted_sum(weights, v, output, q_len, kv_len, head_dim);
}

/// Compute ALiBi-biased attention scores.
///
/// Adds `alibi_slope * (j - i - (kv_len - q_len))` to each raw score.
pub fn neon_attention_score_with_alibi(
    q: &[f32],
    k: &[f32],
    output: &mut [f32],
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    scale: f32,
    alibi_slope: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_attention_score_with_alibi_inner(
                    q,
                    k,
                    output,
                    q_len,
                    kv_len,
                    head_dim,
                    scale,
                    alibi_slope,
                );
            }
            return;
        }
    }
    scalar_attention_score_with_alibi(q, k, output, q_len, kv_len, head_dim, scale, alibi_slope);
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────

    /// Deterministic pseudo-random f32 in `[-1, 1]` seeded by index.
    fn pseudo_rand(seed: u32, idx: u32) -> f32 {
        let mut x = seed.wrapping_mul(1664525).wrapping_add(idx.wrapping_mul(1013904223));
        x ^= x >> 16;
        x = x.wrapping_mul(0x45d9f3b);
        x ^= x >> 16;
        ((x & 0xFFFF) as f32 / 32768.0) - 1.0
    }

    fn make_data(len: usize, seed: u32) -> Vec<f32> {
        (0..len).map(|i| pseudo_rand(seed, i as u32)).collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch {} vs {}", a.len(), b.len());
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            if va == vb {
                continue; // handles ±inf and exact matches
            }
            let diff = (va - vb).abs();
            assert!(diff <= tol, "{label}[{i}]: {va} vs {vb} (diff={diff}, tol={tol})");
        }
    }

    /// Compute reference f64-precision single-head scores for higher accuracy.
    fn ref_attention_score_f64(
        q: &[f32],
        k: &[f32],
        q_len: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; q_len * kv_len];
        let scale64 = scale as f64;
        for i in 0..q_len {
            for j in 0..kv_len {
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] as f64 * k[j * head_dim + d] as f64;
                }
                out[i * kv_len + j] = (dot * scale64) as f32;
            }
        }
        out
    }

    // ── 1. Basic Q·Kᵀ correctness (15 tests) ─────────────────────────

    #[test]
    fn test_score_identity_q_k() {
        let head_dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let mut out = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, head_dim, 1.0);
        assert!((out[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_orthogonal_vectors() {
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![0.0, 1.0, 0.0, 0.0];
        let mut out = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, 4, 1.0);
        assert!(out[0].abs() < 1e-6);
    }

    #[test]
    fn test_score_negative_dot() {
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![-1.0, 0.0, 0.0, 0.0];
        let mut out = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, 4, 1.0);
        assert!((out[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_2x2_matrix() {
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; 4];
        neon_attention_score(&q, &k, &mut out, 2, 2, 2, 1.0);
        assert!((out[0] - 1.0).abs() < 1e-6); // q0·k0
        assert!((out[1] - 0.0).abs() < 1e-6); // q0·k1
        assert!((out[2] - 0.0).abs() < 1e-6); // q1·k0
        assert!((out[3] - 1.0).abs() < 1e-6); // q1·k1
    }

    #[test]
    fn test_score_random_4x4_d8() {
        let q_len = 4;
        let kv_len = 4;
        let head_dim = 8;
        let q = make_data(q_len * head_dim, 42);
        let k = make_data(kv_len * head_dim, 43);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; q_len * kv_len];
        neon_attention_score(&q, &k, &mut out, q_len, kv_len, head_dim, scale);
        let expected = ref_attention_score_f64(&q, &k, q_len, kv_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "random_4x4_d8");
    }

    #[test]
    fn test_score_random_8x16_d64() {
        let q_len = 8;
        let kv_len = 16;
        let head_dim = 64;
        let q = make_data(q_len * head_dim, 100);
        let k = make_data(kv_len * head_dim, 101);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; q_len * kv_len];
        neon_attention_score(&q, &k, &mut out, q_len, kv_len, head_dim, scale);
        let expected = ref_attention_score_f64(&q, &k, q_len, kv_len, head_dim, scale);
        assert_close(&out, &expected, 1e-4, "random_8x16_d64");
    }

    #[test]
    fn test_score_random_16x16_d128() {
        let q_len = 16;
        let kv_len = 16;
        let head_dim = 128;
        let q = make_data(q_len * head_dim, 200);
        let k = make_data(kv_len * head_dim, 201);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; q_len * kv_len];
        neon_attention_score(&q, &k, &mut out, q_len, kv_len, head_dim, scale);
        let expected = ref_attention_score_f64(&q, &k, q_len, kv_len, head_dim, scale);
        assert_close(&out, &expected, 1e-4, "random_16x16_d128");
    }

    #[test]
    fn test_score_asymmetric_q4_kv8_d16() {
        let q_len = 4;
        let kv_len = 8;
        let head_dim = 16;
        let q = make_data(q_len * head_dim, 300);
        let k = make_data(kv_len * head_dim, 301);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; q_len * kv_len];
        neon_attention_score(&q, &k, &mut out, q_len, kv_len, head_dim, scale);
        let expected = ref_attention_score_f64(&q, &k, q_len, kv_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "asymmetric_q4_kv8_d16");
    }

    #[test]
    fn test_score_all_zeros() {
        let q = vec![0.0; 16];
        let k = vec![0.0; 16];
        let mut out = vec![999.0; 4];
        neon_attention_score(&q, &k, &mut out, 2, 2, 4, 0.5);
        for &v in &out {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_score_all_ones() {
        let head_dim = 4;
        let q = vec![1.0; 8]; // 2x4
        let k = vec![1.0; 12]; // 3x4
        let mut out = vec![0.0; 6]; // 2x3
        let scale = 0.5;
        neon_attention_score(&q, &k, &mut out, 2, 3, head_dim, scale);
        // dot = 4.0 for all, * 0.5 = 2.0
        for &v in &out {
            assert!((v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_score_matches_scalar_random_large() {
        let q_len = 32;
        let kv_len = 32;
        let head_dim = 64;
        let q = make_data(q_len * head_dim, 500);
        let k = make_data(kv_len * head_dim, 501);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_dispatch = vec![0.0; q_len * kv_len];
        let mut out_scalar = vec![0.0; q_len * kv_len];
        neon_attention_score(&q, &k, &mut out_dispatch, q_len, kv_len, head_dim, scale);
        scalar_attention_score(&q, &k, &mut out_scalar, q_len, kv_len, head_dim, scale);
        assert_close(&out_dispatch, &out_scalar, 1e-5, "dispatch_vs_scalar");
    }

    #[test]
    fn test_score_head_dim_5_non_multiple_of_4() {
        let q_len = 3;
        let kv_len = 2;
        let head_dim = 5;
        let q = make_data(q_len * head_dim, 600);
        let k = make_data(kv_len * head_dim, 601);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; q_len * kv_len];
        neon_attention_score(&q, &k, &mut out, q_len, kv_len, head_dim, scale);
        let expected = ref_attention_score_f64(&q, &k, q_len, kv_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "head_dim_5");
    }

    #[test]
    fn test_score_head_dim_7() {
        let q_len = 2;
        let kv_len = 3;
        let head_dim = 7;
        let q = make_data(q_len * head_dim, 700);
        let k = make_data(kv_len * head_dim, 701);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; q_len * kv_len];
        neon_attention_score(&q, &k, &mut out, q_len, kv_len, head_dim, scale);
        let expected = ref_attention_score_f64(&q, &k, q_len, kv_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "head_dim_7");
    }

    #[test]
    fn test_score_large_values() {
        let q = vec![1000.0, -1000.0, 500.0, -500.0];
        let k = vec![1000.0, -1000.0, 500.0, -500.0];
        let mut out = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, 4, 1.0);
        let expected = 1000.0 * 1000.0 + 1000.0 * 1000.0 + 500.0 * 500.0 + 500.0 * 500.0;
        assert!((out[0] - expected).abs() / expected.abs() < 1e-5);
    }

    #[test]
    fn test_score_negative_scale() {
        let q = vec![1.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, 4, -0.5);
        assert!((out[0] + 2.0).abs() < 1e-6);
    }

    // ── 2. Scale factor validation (10 tests) ────────────────────────

    #[test]
    fn test_scale_sqrt_d64() {
        let head_dim = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();
        assert!((scale - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_scale_sqrt_d128() {
        let head_dim = 128;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let expected = 1.0 / (128.0f32).sqrt();
        assert!((scale - expected).abs() < 1e-7);
    }

    #[test]
    fn test_scale_factor_applied_correctly() {
        let q = vec![2.0, 0.0, 0.0, 0.0];
        let k = vec![3.0, 0.0, 0.0, 0.0];
        let mut out = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, 4, 0.25);
        assert!((out[0] - 1.5).abs() < 1e-6); // 6.0 * 0.25
    }

    #[test]
    fn test_scale_factor_zero() {
        let q = vec![1.0; 4];
        let k = vec![1.0; 4];
        let mut out = vec![999.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, 4, 0.0);
        assert!((out[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_factor_one() {
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![4.0, 3.0, 2.0, 1.0];
        let mut out = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, 4, 1.0);
        let expected = 1.0 * 4.0 + 2.0 * 3.0 + 3.0 * 2.0 + 4.0 * 1.0;
        assert!((out[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_scale_factor_large() {
        let q = vec![1.0; 4];
        let k = vec![1.0; 4];
        let mut out = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, 4, 100.0);
        assert!((out[0] - 400.0).abs() < 1e-4);
    }

    #[test]
    fn test_scale_factor_tiny() {
        let q = vec![1.0; 4];
        let k = vec![1.0; 4];
        let mut out = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, 4, 1e-6);
        assert!((out[0] - 4e-6).abs() < 1e-10);
    }

    #[test]
    fn test_scale_preserves_relative_ordering() {
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let mut out_a = vec![0.0; 2];
        let mut out_b = vec![0.0; 2];
        neon_attention_score(&q, &k, &mut out_a, 1, 2, 4, 0.5);
        neon_attention_score(&q, &k, &mut out_b, 1, 2, 4, 2.0);
        // Relative ordering preserved regardless of scale.
        assert!(out_a[0] > out_a[1]);
        assert!(out_b[0] > out_b[1]);
    }

    #[test]
    fn test_scale_sqrt_d1() {
        let scale = 1.0 / (1.0f32).sqrt();
        assert!((scale - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_scale_sqrt_d256() {
        let scale = 1.0 / (256.0f32).sqrt();
        assert!((scale - 0.0625).abs() < 1e-7);
    }

    // ── 3. Multi-head attention (15 tests) ────────────────────────────

    #[test]
    fn test_mha_single_head() {
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 4;
        let q = make_data(q_len * head_dim, 10);
        let k = make_data(kv_len * head_dim, 11);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut mha_out = vec![0.0; q_len * kv_len];
        let mut single_out = vec![0.0; q_len * kv_len];
        neon_attention_score_multi_head(&q, &k, &mut mha_out, 1, q_len, kv_len, head_dim, scale);
        neon_attention_score(&q, &k, &mut single_out, q_len, kv_len, head_dim, scale);
        assert_close(&mha_out, &single_out, 1e-6, "mha_single_head");
    }

    #[test]
    fn test_mha_two_heads() {
        let num_heads = 2;
        let q_len = 2;
        let kv_len = 3;
        let head_dim = 4;
        let q = make_data(num_heads * q_len * head_dim, 20);
        let k = make_data(num_heads * kv_len * head_dim, 21);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(
            &q, &k, &mut out, num_heads, q_len, kv_len, head_dim, scale,
        );
        let mut expected = vec![0.0; num_heads * q_len * kv_len];
        scalar_attention_score_multi_head(
            &q,
            &k,
            &mut expected,
            num_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-5, "mha_two_heads");
    }

    #[test]
    fn test_mha_four_heads_d64() {
        let num_heads = 4;
        let q_len = 4;
        let kv_len = 4;
        let head_dim = 64;
        let q = make_data(num_heads * q_len * head_dim, 30);
        let k = make_data(num_heads * kv_len * head_dim, 31);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(
            &q, &k, &mut out, num_heads, q_len, kv_len, head_dim, scale,
        );
        let mut expected = vec![0.0; num_heads * q_len * kv_len];
        scalar_attention_score_multi_head(
            &q,
            &k,
            &mut expected,
            num_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-4, "mha_four_heads_d64");
    }

    #[test]
    fn test_mha_eight_heads_d128() {
        let num_heads = 8;
        let q_len = 8;
        let kv_len = 8;
        let head_dim = 128;
        let q = make_data(num_heads * q_len * head_dim, 40);
        let k = make_data(num_heads * kv_len * head_dim, 41);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(
            &q, &k, &mut out, num_heads, q_len, kv_len, head_dim, scale,
        );
        let mut expected = vec![0.0; num_heads * q_len * kv_len];
        scalar_attention_score_multi_head(
            &q,
            &k,
            &mut expected,
            num_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-3, "mha_eight_heads_d128");
    }

    #[test]
    fn test_mha_heads_independent() {
        let num_heads = 2;
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 4;
        // Head 0: identity, Head 1: anti-identity
        let mut q = vec![0.0; num_heads * q_len * head_dim];
        let mut k = vec![0.0; num_heads * kv_len * head_dim];
        q[0] = 1.0;
        q[head_dim] = 1.0;
        k[0] = 1.0;
        k[head_dim] = 1.0;
        // Head 1
        let h1_off = q_len * head_dim;
        q[h1_off] = -1.0;
        q[h1_off + head_dim] = -1.0;
        k[h1_off] = 1.0;
        k[h1_off + head_dim] = 1.0;
        let mut out = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(&q, &k, &mut out, num_heads, q_len, kv_len, head_dim, 1.0);
        // Head 0: q0·k0=1, q0·k1=0, q1·k0=0, q1·k1=1
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
        // Head 1: q0·k0=-1, q0·k1=-1
        let h1_out = q_len * kv_len;
        assert!((out[h1_out] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mha_different_q_kv_len() {
        let num_heads = 2;
        let q_len = 3;
        let kv_len = 5;
        let head_dim = 8;
        let q = make_data(num_heads * q_len * head_dim, 50);
        let k = make_data(num_heads * kv_len * head_dim, 51);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; num_heads * q_len * kv_len];
        let mut expected = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(
            &q, &k, &mut out, num_heads, q_len, kv_len, head_dim, scale,
        );
        scalar_attention_score_multi_head(
            &q,
            &k,
            &mut expected,
            num_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-5, "mha_diff_q_kv_len");
    }

    #[test]
    fn test_mha_zero_input() {
        let num_heads = 2;
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 4;
        let q = vec![0.0; num_heads * q_len * head_dim];
        let k = vec![0.0; num_heads * kv_len * head_dim];
        let mut out = vec![999.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(&q, &k, &mut out, num_heads, q_len, kv_len, head_dim, 1.0);
        for &v in &out {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mha_16_heads_d32() {
        let num_heads = 16;
        let q_len = 4;
        let kv_len = 4;
        let head_dim = 32;
        let q = make_data(num_heads * q_len * head_dim, 60);
        let k = make_data(num_heads * kv_len * head_dim, 61);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; num_heads * q_len * kv_len];
        let mut expected = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(
            &q, &k, &mut out, num_heads, q_len, kv_len, head_dim, scale,
        );
        scalar_attention_score_multi_head(
            &q,
            &k,
            &mut expected,
            num_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-4, "mha_16_heads_d32");
    }

    #[test]
    fn test_mha_head_dim_3() {
        let num_heads = 3;
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 3;
        let q = make_data(num_heads * q_len * head_dim, 70);
        let k = make_data(num_heads * kv_len * head_dim, 71);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; num_heads * q_len * kv_len];
        let mut expected = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(
            &q, &k, &mut out, num_heads, q_len, kv_len, head_dim, scale,
        );
        scalar_attention_score_multi_head(
            &q,
            &k,
            &mut expected,
            num_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-5, "mha_head_dim_3");
    }

    #[test]
    fn test_mha_1x1_per_head() {
        let num_heads = 4;
        let q = vec![1.0, 2.0, 3.0, 4.0]; // each head: 1 query, dim=1
        let k = vec![1.0, 1.0, 1.0, 1.0]; // each head: 1 key, dim=1
        let mut out = vec![0.0; 4]; // each head: 1x1
        neon_attention_score_multi_head(&q, &k, &mut out, 4, 1, 1, 1, 1.0);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_mha_32_heads_d16() {
        let num_heads = 32;
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 16;
        let q = make_data(num_heads * q_len * head_dim, 80);
        let k = make_data(num_heads * kv_len * head_dim, 81);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; num_heads * q_len * kv_len];
        let mut expected = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(
            &q, &k, &mut out, num_heads, q_len, kv_len, head_dim, scale,
        );
        scalar_attention_score_multi_head(
            &q,
            &k,
            &mut expected,
            num_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-4, "mha_32_heads_d16");
    }

    #[test]
    fn test_mha_consistency_across_scales() {
        let num_heads = 2;
        let q_len = 3;
        let kv_len = 3;
        let head_dim = 8;
        let q = make_data(num_heads * q_len * head_dim, 90);
        let k = make_data(num_heads * kv_len * head_dim, 91);
        let mut out_a = vec![0.0; num_heads * q_len * kv_len];
        let mut out_b = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(
            &q, &k, &mut out_a, num_heads, q_len, kv_len, head_dim, 1.0,
        );
        neon_attention_score_multi_head(
            &q, &k, &mut out_b, num_heads, q_len, kv_len, head_dim, 2.0,
        );
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!((b - a * 2.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_mha_q_longer_than_kv() {
        let num_heads = 2;
        let q_len = 6;
        let kv_len = 2;
        let head_dim = 4;
        let q = make_data(num_heads * q_len * head_dim, 95);
        let k = make_data(num_heads * kv_len * head_dim, 96);
        let scale = 0.5;
        let mut out = vec![0.0; num_heads * q_len * kv_len];
        let mut expected = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(
            &q, &k, &mut out, num_heads, q_len, kv_len, head_dim, scale,
        );
        scalar_attention_score_multi_head(
            &q,
            &k,
            &mut expected,
            num_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-5, "mha_q_longer");
    }

    #[test]
    fn test_mha_single_token_query() {
        let num_heads = 4;
        let q_len = 1;
        let kv_len = 8;
        let head_dim = 16;
        let q = make_data(num_heads * q_len * head_dim, 97);
        let k = make_data(num_heads * kv_len * head_dim, 98);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; num_heads * q_len * kv_len];
        let mut expected = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_multi_head(
            &q, &k, &mut out, num_heads, q_len, kv_len, head_dim, scale,
        );
        scalar_attention_score_multi_head(
            &q,
            &k,
            &mut expected,
            num_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-5, "mha_single_token_q");
    }

    // ── 4. Causal masking (10 tests) ──────────────────────────────────

    #[test]
    fn test_mask_2x2_causal() {
        let q = vec![1.0; 8]; // 2x4
        let k = vec![1.0; 8]; // 2x4
        let mut out = vec![0.0; 4]; // 2x2
        neon_attention_score_masked(&q, &k, &mut out, 2, 2, 4, 1.0, f32::NEG_INFINITY);
        // i=0: j=0 allowed, j=1 masked
        assert!(out[0].is_finite());
        assert!(out[1] == f32::NEG_INFINITY);
        // i=1: j=0 allowed, j=1 allowed
        assert!(out[2].is_finite());
        assert!(out[3].is_finite());
    }

    #[test]
    fn test_mask_3x3_lower_triangular() {
        let q = vec![1.0; 12];
        let k = vec![1.0; 12];
        let mut out = vec![0.0; 9];
        neon_attention_score_masked(&q, &k, &mut out, 3, 3, 4, 1.0, f32::NEG_INFINITY);
        // Row 0: only [0] visible
        assert!(out[0].is_finite());
        assert!(out[1] == f32::NEG_INFINITY);
        assert!(out[2] == f32::NEG_INFINITY);
        // Row 1: [0,1] visible
        assert!(out[3].is_finite());
        assert!(out[4].is_finite());
        assert!(out[5] == f32::NEG_INFINITY);
        // Row 2: all visible
        assert!(out[6].is_finite());
        assert!(out[7].is_finite());
        assert!(out[8].is_finite());
    }

    #[test]
    fn test_mask_1x1_no_mask() {
        let q = vec![2.0; 4];
        let k = vec![3.0; 4];
        let mut out = vec![0.0; 1];
        neon_attention_score_masked(&q, &k, &mut out, 1, 1, 4, 0.5, f32::NEG_INFINITY);
        // Single element: always visible.
        assert!(out[0].is_finite());
        let expected = 2.0 * 3.0 * 4.0 * 0.5; // dot=24, *0.5=12
        assert!((out[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_mask_values_matches_scalar() {
        let q_len = 5;
        let kv_len = 5;
        let head_dim = 8;
        let q = make_data(q_len * head_dim, 400);
        let k = make_data(kv_len * head_dim, 401);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; q_len * kv_len];
        let mut expected = vec![0.0; q_len * kv_len];
        neon_attention_score_masked(
            &q,
            &k,
            &mut out,
            q_len,
            kv_len,
            head_dim,
            scale,
            f32::NEG_INFINITY,
        );
        scalar_attention_score_masked(
            &q,
            &k,
            &mut expected,
            q_len,
            kv_len,
            head_dim,
            scale,
            f32::NEG_INFINITY,
        );
        assert_close(&out, &expected, 1e-5, "mask_vs_scalar");
    }

    #[test]
    fn test_mask_asymmetric_q2_kv5() {
        let q_len = 2;
        let kv_len = 5;
        let head_dim = 4;
        let q = vec![1.0; q_len * head_dim];
        let k = vec![1.0; kv_len * head_dim];
        let mut out = vec![0.0; q_len * kv_len];
        neon_attention_score_masked(
            &q,
            &k,
            &mut out,
            q_len,
            kv_len,
            head_dim,
            1.0,
            f32::NEG_INFINITY,
        );
        // offset = 5 - 2 = 3
        // Row 0: j <= 0+3=3 → visible [0..3], masked [4]
        assert!(out[0].is_finite());
        assert!(out[3].is_finite());
        assert!(out[4] == f32::NEG_INFINITY);
        // Row 1: j <= 1+3=4 → all visible
        for j in 0..kv_len {
            assert!(out[kv_len + j].is_finite());
        }
    }

    #[test]
    fn test_mask_custom_mask_value() {
        let q = vec![1.0; 8];
        let k = vec![1.0; 8];
        let mut out = vec![0.0; 4];
        neon_attention_score_masked(&q, &k, &mut out, 2, 2, 4, 1.0, -1e9);
        assert!(out[1] == -1e9);
    }

    #[test]
    fn test_mask_large_random() {
        let q_len = 16;
        let kv_len = 16;
        let head_dim = 32;
        let q = make_data(q_len * head_dim, 410);
        let k = make_data(kv_len * head_dim, 411);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; q_len * kv_len];
        let mut expected = vec![0.0; q_len * kv_len];
        neon_attention_score_masked(
            &q,
            &k,
            &mut out,
            q_len,
            kv_len,
            head_dim,
            scale,
            f32::NEG_INFINITY,
        );
        scalar_attention_score_masked(
            &q,
            &k,
            &mut expected,
            q_len,
            kv_len,
            head_dim,
            scale,
            f32::NEG_INFINITY,
        );
        assert_close(&out, &expected, 1e-4, "mask_large_random");
    }

    #[test]
    fn test_mask_last_row_all_visible() {
        let q_len = 4;
        let kv_len = 4;
        let q = vec![1.0; q_len * 4];
        let k = vec![1.0; kv_len * 4];
        let mut out = vec![0.0; q_len * kv_len];
        neon_attention_score_masked(&q, &k, &mut out, q_len, kv_len, 4, 1.0, f32::NEG_INFINITY);
        // Last row (i=3): all positions visible for square causal mask.
        for j in 0..kv_len {
            assert!(out[3 * kv_len + j].is_finite());
        }
    }

    #[test]
    fn test_mask_first_row_only_first_visible() {
        let q_len = 4;
        let kv_len = 4;
        let q = vec![1.0; q_len * 4];
        let k = vec![1.0; kv_len * 4];
        let mut out = vec![0.0; q_len * kv_len];
        neon_attention_score_masked(&q, &k, &mut out, q_len, kv_len, 4, 1.0, f32::NEG_INFINITY);
        assert!(out[0].is_finite());
        for j in 1..kv_len {
            assert!(out[j] == f32::NEG_INFINITY);
        }
    }

    #[test]
    fn test_mask_preserves_unmasked_scores() {
        let q_len = 3;
        let kv_len = 3;
        let head_dim = 4;
        let q = make_data(q_len * head_dim, 420);
        let k = make_data(kv_len * head_dim, 421);
        let scale = 0.5;
        let mut masked = vec![0.0; q_len * kv_len];
        let mut unmasked = vec![0.0; q_len * kv_len];
        neon_attention_score_masked(
            &q,
            &k,
            &mut masked,
            q_len,
            kv_len,
            head_dim,
            scale,
            f32::NEG_INFINITY,
        );
        neon_attention_score(&q, &k, &mut unmasked, q_len, kv_len, head_dim, scale);
        // Visible positions must match unmasked.
        for i in 0..q_len {
            for j in 0..=i {
                let idx = i * kv_len + j;
                assert!((masked[idx] - unmasked[idx]).abs() < 1e-6, "pos ({i},{j})");
            }
        }
    }

    // ── 5. Group query attention (10 tests) ───────────────────────────

    #[test]
    fn test_gqa_ratio_1_equals_mha() {
        let num_heads = 4;
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 4;
        let q = make_data(num_heads * q_len * head_dim, 800);
        let k = make_data(num_heads * kv_len * head_dim, 801);
        let scale = 0.5;
        let mut gqa_out = vec![0.0; num_heads * q_len * kv_len];
        let mut mha_out = vec![0.0; num_heads * q_len * kv_len];
        neon_attention_score_gqa(
            &q,
            &k,
            &mut gqa_out,
            num_heads,
            num_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        neon_attention_score_multi_head(
            &q,
            &k,
            &mut mha_out,
            num_heads,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&gqa_out, &mha_out, 1e-6, "gqa_ratio1");
    }

    #[test]
    fn test_gqa_ratio_2() {
        let num_q = 4;
        let num_kv = 2;
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 4;
        let q = make_data(num_q * q_len * head_dim, 810);
        let k = make_data(num_kv * kv_len * head_dim, 811);
        let scale = 0.5;
        let mut out = vec![0.0; num_q * q_len * kv_len];
        let mut expected = vec![0.0; num_q * q_len * kv_len];
        neon_attention_score_gqa(&q, &k, &mut out, num_q, num_kv, q_len, kv_len, head_dim, scale);
        scalar_attention_score_gqa(
            &q,
            &k,
            &mut expected,
            num_q,
            num_kv,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-5, "gqa_ratio2");
    }

    #[test]
    fn test_gqa_ratio_4() {
        let num_q = 8;
        let num_kv = 2;
        let q_len = 3;
        let kv_len = 3;
        let head_dim = 8;
        let q = make_data(num_q * q_len * head_dim, 820);
        let k = make_data(num_kv * kv_len * head_dim, 821);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; num_q * q_len * kv_len];
        let mut expected = vec![0.0; num_q * q_len * kv_len];
        neon_attention_score_gqa(&q, &k, &mut out, num_q, num_kv, q_len, kv_len, head_dim, scale);
        scalar_attention_score_gqa(
            &q,
            &k,
            &mut expected,
            num_q,
            num_kv,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-5, "gqa_ratio4");
    }

    #[test]
    fn test_gqa_shared_kv_heads_identical_output() {
        // Two Q heads sharing the same KV head produce identical K contributions.
        let num_q = 2;
        let num_kv = 1;
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 4;
        let q = make_data(num_q * q_len * head_dim, 830);
        let k = make_data(num_kv * kv_len * head_dim, 831);
        let scale = 0.5;
        let mut out = vec![0.0; num_q * q_len * kv_len];
        neon_attention_score_gqa(&q, &k, &mut out, num_q, num_kv, q_len, kv_len, head_dim, scale);
        // Both heads use same K but different Q, so outputs differ (unless Q identical).
        // Verify output shape is correct.
        assert_eq!(out.len(), num_q * q_len * kv_len);
    }

    #[test]
    fn test_gqa_single_kv_head() {
        let num_q = 8;
        let num_kv = 1;
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 4;
        let q = make_data(num_q * q_len * head_dim, 840);
        let k = make_data(num_kv * kv_len * head_dim, 841);
        let scale = 0.5;
        let mut out = vec![0.0; num_q * q_len * kv_len];
        let mut expected = vec![0.0; num_q * q_len * kv_len];
        neon_attention_score_gqa(&q, &k, &mut out, num_q, num_kv, q_len, kv_len, head_dim, scale);
        scalar_attention_score_gqa(
            &q,
            &k,
            &mut expected,
            num_q,
            num_kv,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-5, "gqa_single_kv_head");
    }

    #[test]
    fn test_gqa_d64_ratio_4() {
        let num_q = 16;
        let num_kv = 4;
        let q_len = 4;
        let kv_len = 4;
        let head_dim = 64;
        let q = make_data(num_q * q_len * head_dim, 850);
        let k = make_data(num_kv * kv_len * head_dim, 851);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; num_q * q_len * kv_len];
        let mut expected = vec![0.0; num_q * q_len * kv_len];
        neon_attention_score_gqa(&q, &k, &mut out, num_q, num_kv, q_len, kv_len, head_dim, scale);
        scalar_attention_score_gqa(
            &q,
            &k,
            &mut expected,
            num_q,
            num_kv,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-4, "gqa_d64_ratio4");
    }

    #[test]
    fn test_gqa_asymmetric_seq_lens() {
        let num_q = 4;
        let num_kv = 2;
        let q_len = 1;
        let kv_len = 8;
        let head_dim = 16;
        let q = make_data(num_q * q_len * head_dim, 860);
        let k = make_data(num_kv * kv_len * head_dim, 861);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; num_q * q_len * kv_len];
        let mut expected = vec![0.0; num_q * q_len * kv_len];
        neon_attention_score_gqa(&q, &k, &mut out, num_q, num_kv, q_len, kv_len, head_dim, scale);
        scalar_attention_score_gqa(
            &q,
            &k,
            &mut expected,
            num_q,
            num_kv,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-5, "gqa_asymmetric_seq");
    }

    #[test]
    fn test_gqa_ratio_8() {
        let num_q = 32;
        let num_kv = 4;
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 8;
        let q = make_data(num_q * q_len * head_dim, 870);
        let k = make_data(num_kv * kv_len * head_dim, 871);
        let scale = 0.25;
        let mut out = vec![0.0; num_q * q_len * kv_len];
        let mut expected = vec![0.0; num_q * q_len * kv_len];
        neon_attention_score_gqa(&q, &k, &mut out, num_q, num_kv, q_len, kv_len, head_dim, scale);
        scalar_attention_score_gqa(
            &q,
            &k,
            &mut expected,
            num_q,
            num_kv,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-5, "gqa_ratio8");
    }

    #[test]
    fn test_gqa_head_dim_1() {
        let num_q = 4;
        let num_kv = 2;
        let q_len = 3;
        let kv_len = 3;
        let head_dim = 1;
        let q = make_data(num_q * q_len * head_dim, 880);
        let k = make_data(num_kv * kv_len * head_dim, 881);
        let scale = 1.0;
        let mut out = vec![0.0; num_q * q_len * kv_len];
        let mut expected = vec![0.0; num_q * q_len * kv_len];
        neon_attention_score_gqa(&q, &k, &mut out, num_q, num_kv, q_len, kv_len, head_dim, scale);
        scalar_attention_score_gqa(
            &q,
            &k,
            &mut expected,
            num_q,
            num_kv,
            q_len,
            kv_len,
            head_dim,
            scale,
        );
        assert_close(&out, &expected, 1e-5, "gqa_head_dim_1");
    }

    #[test]
    fn test_gqa_zero_keys() {
        let num_q = 2;
        let num_kv = 1;
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 4;
        let q = make_data(num_q * q_len * head_dim, 890);
        let k = vec![0.0; num_kv * kv_len * head_dim];
        let mut out = vec![999.0; num_q * q_len * kv_len];
        neon_attention_score_gqa(&q, &k, &mut out, num_q, num_kv, q_len, kv_len, head_dim, 0.5);
        for &v in &out {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    // ── 6. Weighted sum (V application) (10 tests) ────────────────────

    #[test]
    fn test_weighted_sum_identity_weights() {
        // weights = [[1,0],[0,1]], V = [[a,b],[c,d]] → output = V
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        neon_attention_weighted_sum(&weights, &v, &mut out, 2, 2, 2);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_sum_uniform_weights() {
        let weights = vec![0.5, 0.5];
        let v = vec![2.0, 4.0, 6.0, 8.0];
        let mut out = vec![0.0; 2];
        neon_attention_weighted_sum(&weights, &v, &mut out, 1, 2, 2);
        assert!((out[0] - 4.0).abs() < 1e-6); // 0.5*2 + 0.5*6
        assert!((out[1] - 6.0).abs() < 1e-6); // 0.5*4 + 0.5*8
    }

    #[test]
    fn test_weighted_sum_single_kv() {
        let weights = vec![1.0];
        let v = vec![5.0, 10.0, 15.0, 20.0];
        let mut out = vec![0.0; 4];
        neon_attention_weighted_sum(&weights, &v, &mut out, 1, 1, 4);
        assert_close(&out, &v, 1e-6, "ws_single_kv");
    }

    #[test]
    fn test_weighted_sum_zero_weights() {
        let weights = vec![0.0, 0.0, 0.0, 0.0];
        let v = make_data(8, 900);
        let mut out = vec![999.0; 4];
        neon_attention_weighted_sum(&weights, &v, &mut out, 2, 2, 2);
        for &val in &out {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_weighted_sum_matches_scalar() {
        let q_len = 4;
        let kv_len = 6;
        let head_dim = 16;
        let weights = make_data(q_len * kv_len, 910);
        // Make weights non-negative for realistic scenario.
        let weights: Vec<f32> = weights.iter().map(|w| w.abs()).collect();
        let v = make_data(kv_len * head_dim, 911);
        let mut out = vec![0.0; q_len * head_dim];
        let mut expected = vec![0.0; q_len * head_dim];
        neon_attention_weighted_sum(&weights, &v, &mut out, q_len, kv_len, head_dim);
        scalar_attention_weighted_sum(&weights, &v, &mut expected, q_len, kv_len, head_dim);
        assert_close(&out, &expected, 1e-4, "ws_vs_scalar");
    }

    #[test]
    fn test_weighted_sum_d64() {
        let q_len = 3;
        let kv_len = 4;
        let head_dim = 64;
        let weights = make_data(q_len * kv_len, 920).iter().map(|w| w.abs()).collect::<Vec<_>>();
        let v = make_data(kv_len * head_dim, 921);
        let mut out = vec![0.0; q_len * head_dim];
        let mut expected = vec![0.0; q_len * head_dim];
        neon_attention_weighted_sum(&weights, &v, &mut out, q_len, kv_len, head_dim);
        scalar_attention_weighted_sum(&weights, &v, &mut expected, q_len, kv_len, head_dim);
        assert_close(&out, &expected, 1e-3, "ws_d64");
    }

    #[test]
    fn test_weighted_sum_d128() {
        let q_len = 2;
        let kv_len = 4;
        let head_dim = 128;
        let weights = make_data(q_len * kv_len, 930).iter().map(|w| w.abs()).collect::<Vec<_>>();
        let v = make_data(kv_len * head_dim, 931);
        let mut out = vec![0.0; q_len * head_dim];
        let mut expected = vec![0.0; q_len * head_dim];
        neon_attention_weighted_sum(&weights, &v, &mut out, q_len, kv_len, head_dim);
        scalar_attention_weighted_sum(&weights, &v, &mut expected, q_len, kv_len, head_dim);
        assert_close(&out, &expected, 1e-3, "ws_d128");
    }

    #[test]
    fn test_weighted_sum_concentrate_on_last() {
        let weights = vec![0.0, 0.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut out = vec![0.0; 2];
        neon_attention_weighted_sum(&weights, &v, &mut out, 1, 3, 2);
        assert!((out[0] - 50.0).abs() < 1e-6);
        assert!((out[1] - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_sum_softmax_weights() {
        // Simulate post-softmax weights (sum to 1.0).
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4x2
        let mut out = vec![0.0; 2];
        neon_attention_weighted_sum(&weights, &v, &mut out, 1, 4, 2);
        let mut expected = vec![0.0; 2];
        scalar_attention_weighted_sum(&weights, &v, &mut expected, 1, 4, 2);
        assert_close(&out, &expected, 1e-5, "ws_softmax_weights");
    }

    #[test]
    fn test_weighted_sum_head_dim_1() {
        let weights = vec![0.25, 0.75];
        let v = vec![10.0, 20.0];
        let mut out = vec![0.0; 1];
        neon_attention_weighted_sum(&weights, &v, &mut out, 1, 2, 1);
        assert!((out[0] - 17.5).abs() < 1e-5); // 0.25*10 + 0.75*20
    }

    // ── 7. ALiBi position encoding (10 tests) ────────────────────────

    #[test]
    fn test_alibi_zero_slope_no_bias() {
        let q_len = 2;
        let kv_len = 2;
        let head_dim = 4;
        let q = make_data(q_len * head_dim, 1000);
        let k = make_data(kv_len * head_dim, 1001);
        let scale = 0.5;
        let mut alibi_out = vec![0.0; q_len * kv_len];
        let mut plain_out = vec![0.0; q_len * kv_len];
        neon_attention_score_with_alibi(
            &q,
            &k,
            &mut alibi_out,
            q_len,
            kv_len,
            head_dim,
            scale,
            0.0,
        );
        neon_attention_score(&q, &k, &mut plain_out, q_len, kv_len, head_dim, scale);
        assert_close(&alibi_out, &plain_out, 1e-6, "alibi_zero_slope");
    }

    #[test]
    fn test_alibi_positive_slope() {
        let q = vec![0.0; 4]; // zero Q → dot=0 → raw score=0
        let k = vec![0.0; 8]; // 2 keys
        let mut out = vec![0.0; 2];
        let slope = 0.5;
        neon_attention_score_with_alibi(&q, &k, &mut out, 1, 2, 4, 1.0, slope);
        // offset = 2-1 = 1
        // j=0: distance = 0 - 0 - 1 = -1, bias = 0.5*(-1) = -0.5
        // j=1: distance = 1 - 0 - 1 = 0,  bias = 0.5*(0)  =  0.0
        assert!((out[0] + 0.5).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_negative_slope() {
        let q = vec![0.0; 4];
        let k = vec![0.0; 12]; // 3 keys
        let mut out = vec![0.0; 3];
        let slope = -1.0;
        neon_attention_score_with_alibi(&q, &k, &mut out, 1, 3, 4, 1.0, slope);
        // offset = 3-1 = 2
        // j=0: dist = 0-0-2 = -2, bias = -1*(-2) = 2
        // j=1: dist = 1-0-2 = -1, bias = -1*(-1) = 1
        // j=2: dist = 2-0-2 =  0, bias = -1*(0)  = 0
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_matches_scalar() {
        let q_len = 4;
        let kv_len = 6;
        let head_dim = 16;
        let q = make_data(q_len * head_dim, 1010);
        let k = make_data(kv_len * head_dim, 1011);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let slope = 0.125;
        let mut out = vec![0.0; q_len * kv_len];
        let mut expected = vec![0.0; q_len * kv_len];
        neon_attention_score_with_alibi(&q, &k, &mut out, q_len, kv_len, head_dim, scale, slope);
        scalar_attention_score_with_alibi(
            &q,
            &k,
            &mut expected,
            q_len,
            kv_len,
            head_dim,
            scale,
            slope,
        );
        assert_close(&out, &expected, 1e-4, "alibi_vs_scalar");
    }

    #[test]
    fn test_alibi_square_matrix() {
        let q_len = 3;
        let kv_len = 3;
        let q = vec![0.0; q_len * 4];
        let k = vec![0.0; kv_len * 4];
        let mut out = vec![0.0; q_len * kv_len];
        let slope = 1.0;
        neon_attention_score_with_alibi(&q, &k, &mut out, q_len, kv_len, 4, 1.0, slope);
        // offset=0, distance = j-i
        // (0,0)=0, (0,1)=1, (0,2)=2
        // (1,0)=-1, (1,1)=0, (1,2)=1
        // (2,0)=-2, (2,1)=-1, (2,2)=0
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 2.0).abs() < 1e-6);
        assert!((out[3] + 1.0).abs() < 1e-6);
        assert!((out[4] - 0.0).abs() < 1e-6);
        assert!((out[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_large_slope() {
        let q = vec![0.0; 4];
        let k = vec![0.0; 8]; // 2 keys
        let mut out = vec![0.0; 2];
        neon_attention_score_with_alibi(&q, &k, &mut out, 1, 2, 4, 1.0, 100.0);
        // Bias magnitudes scale linearly with slope.
        assert!((out[0] + 100.0).abs() < 1e-4); // dist=-1, slope*(-1)=-100
        assert!((out[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_typical_slopes_8_heads() {
        // ALiBi schedule for 8 heads: 2^(-8/8), 2^(-7/8), ..., 2^(-1/8)
        let slopes: Vec<f32> = (1..=8).map(|i| 2.0f32.powf(-(i as f32) / 8.0)).collect();
        assert!(slopes[0] > slopes[7]);
        assert!((slopes[7] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_alibi_d64_random() {
        let q_len = 8;
        let kv_len = 8;
        let head_dim = 64;
        let q = make_data(q_len * head_dim, 1020);
        let k = make_data(kv_len * head_dim, 1021);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let slope = 0.0625;
        let mut out = vec![0.0; q_len * kv_len];
        let mut expected = vec![0.0; q_len * kv_len];
        neon_attention_score_with_alibi(&q, &k, &mut out, q_len, kv_len, head_dim, scale, slope);
        scalar_attention_score_with_alibi(
            &q,
            &k,
            &mut expected,
            q_len,
            kv_len,
            head_dim,
            scale,
            slope,
        );
        assert_close(&out, &expected, 1e-4, "alibi_d64_random");
    }

    #[test]
    fn test_alibi_bias_is_additive() {
        let q_len = 2;
        let kv_len = 3;
        let head_dim = 4;
        let q = make_data(q_len * head_dim, 1030);
        let k = make_data(kv_len * head_dim, 1031);
        let scale = 0.5;
        let slope = 0.25;
        let mut alibi_out = vec![0.0; q_len * kv_len];
        let mut plain_out = vec![0.0; q_len * kv_len];
        neon_attention_score_with_alibi(
            &q,
            &k,
            &mut alibi_out,
            q_len,
            kv_len,
            head_dim,
            scale,
            slope,
        );
        neon_attention_score(&q, &k, &mut plain_out, q_len, kv_len, head_dim, scale);
        // Check that ALiBi = plain + bias
        let offset = kv_len as f32 - q_len as f32;
        for i in 0..q_len {
            for j in 0..kv_len {
                let distance = j as f32 - i as f32 - offset;
                let expected = plain_out[i * kv_len + j] + slope * distance;
                let actual = alibi_out[i * kv_len + j];
                assert!((actual - expected).abs() < 1e-5, "alibi additive ({i},{j})");
            }
        }
    }

    #[test]
    fn test_alibi_causal_decreasing_future() {
        // With negative slope, future tokens get increasingly penalised.
        let q = vec![0.0; 4];
        let k = vec![0.0; 20]; // 5 keys
        let mut out = vec![0.0; 5];
        let slope = -0.5; // negative: penalises distance
        neon_attention_score_with_alibi(&q, &k, &mut out, 1, 5, 4, 1.0, slope);
        // Verify monotonically increasing (negative slope + positive distance → more negative).
        // Wait: slope < 0 and dist increases → bias decreases.
        // offset=4, dist(j)=j-0-4=j-4 → [-4,-3,-2,-1,0]
        // bias = -0.5 * dist → [2, 1.5, 1, 0.5, 0]
        assert!(out[0] > out[1]);
        assert!(out[1] > out[2]);
        assert!(out[2] > out[3]);
        assert!(out[3] > out[4]);
    }

    // ── 8. Edge cases (12 tests) ──────────────────────────────────────

    #[test]
    fn test_edge_head_dim_1() {
        let q = vec![3.0];
        let k = vec![4.0];
        let mut out = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, 1, 1.0);
        assert!((out[0] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_seq_len_1() {
        let head_dim = 8;
        let q = make_data(head_dim, 1100);
        let k = make_data(head_dim, 1101);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; 1];
        let mut expected = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, head_dim, scale);
        scalar_attention_score(&q, &k, &mut expected, 1, 1, head_dim, scale);
        assert_close(&out, &expected, 1e-6, "edge_seq1");
    }

    #[test]
    fn test_edge_single_head_single_token_mha() {
        let q = vec![1.0, 2.0];
        let k = vec![3.0, 4.0];
        let mut out = vec![0.0; 1];
        neon_attention_score_multi_head(&q, &k, &mut out, 1, 1, 1, 2, 1.0);
        assert!((out[0] - 11.0).abs() < 1e-6); // 1*3+2*4=11
    }

    #[test]
    fn test_edge_mask_seq_len_1() {
        let q = vec![1.0; 4];
        let k = vec![1.0; 4];
        let mut out = vec![0.0; 1];
        neon_attention_score_masked(&q, &k, &mut out, 1, 1, 4, 1.0, f32::NEG_INFINITY);
        assert!(out[0].is_finite());
    }

    #[test]
    fn test_edge_gqa_single_head_each() {
        let q = make_data(8, 1200);
        let k = make_data(8, 1201);
        let mut gqa_out = vec![0.0; 4];
        let mut mha_out = vec![0.0; 4];
        neon_attention_score_gqa(&q, &k, &mut gqa_out, 1, 1, 2, 2, 4, 0.5);
        neon_attention_score_multi_head(&q, &k, &mut mha_out, 1, 2, 2, 4, 0.5);
        assert_close(&gqa_out, &mha_out, 1e-6, "edge_gqa_1_1");
    }

    #[test]
    fn test_edge_weighted_sum_single_element() {
        let weights = vec![1.0];
        let v = vec![42.0];
        let mut out = vec![0.0];
        neon_attention_weighted_sum(&weights, &v, &mut out, 1, 1, 1);
        assert!((out[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_alibi_single_position() {
        let q = vec![0.0; 4];
        let k = vec![0.0; 4];
        let mut out = vec![0.0; 1];
        neon_attention_score_with_alibi(&q, &k, &mut out, 1, 1, 4, 1.0, 0.5);
        // offset=0, dist=0-0-0=0, bias=0
        assert!((out[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_head_dim_2() {
        let q = vec![1.0, 2.0];
        let k = vec![3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0; 2];
        neon_attention_score(&q, &k, &mut out, 1, 2, 2, 1.0);
        assert!((out[0] - 11.0).abs() < 1e-6); // 1*3+2*4
        assert!((out[1] - 17.0).abs() < 1e-6); // 1*5+2*6
    }

    #[test]
    fn test_edge_head_dim_exactly_4() {
        let head_dim = 4;
        let q = make_data(4, 1300);
        let k = make_data(4, 1301);
        let mut out = vec![0.0; 1];
        let mut expected = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, head_dim, 0.5);
        scalar_attention_score(&q, &k, &mut expected, 1, 1, head_dim, 0.5);
        assert_close(&out, &expected, 1e-6, "edge_dim_4");
    }

    #[test]
    fn test_edge_head_dim_exactly_8() {
        let head_dim = 8;
        let q = make_data(head_dim, 1400);
        let k = make_data(head_dim, 1401);
        let mut out = vec![0.0; 1];
        let mut expected = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, head_dim, 0.5);
        scalar_attention_score(&q, &k, &mut expected, 1, 1, head_dim, 0.5);
        assert_close(&out, &expected, 1e-6, "edge_dim_8");
    }

    #[test]
    fn test_edge_q_len_larger_than_kv_len() {
        let q_len = 5;
        let kv_len = 2;
        let head_dim = 4;
        let q = make_data(q_len * head_dim, 1500);
        let k = make_data(kv_len * head_dim, 1501);
        let scale = 0.5;
        let mut out = vec![0.0; q_len * kv_len];
        let mut expected = vec![0.0; q_len * kv_len];
        neon_attention_score(&q, &k, &mut out, q_len, kv_len, head_dim, scale);
        scalar_attention_score(&q, &k, &mut expected, q_len, kv_len, head_dim, scale);
        assert_close(&out, &expected, 1e-5, "edge_q_gt_kv");
    }

    #[test]
    fn test_edge_large_head_dim_256() {
        let head_dim = 256;
        let q = make_data(head_dim, 1600);
        let k = make_data(head_dim, 1601);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; 1];
        let mut expected = vec![0.0; 1];
        neon_attention_score(&q, &k, &mut out, 1, 1, head_dim, scale);
        scalar_attention_score(&q, &k, &mut expected, 1, 1, head_dim, scale);
        assert_close(&out, &expected, 1e-3, "edge_dim_256");
    }

    // ── NEON-specific tests ────────────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    mod neon_specific {
        use super::*;

        #[test]
        fn test_neon_dot_basic() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = vec![1.0; 8];
            let result = unsafe { neon_dot_f32(&a, &b, 8) };
            assert!((result - 36.0).abs() < 1e-5);
        }

        #[test]
        fn test_neon_dot_non_aligned_length() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let b = vec![2.0; 5];
            let result = unsafe { neon_dot_f32(&a, &b, 5) };
            assert!((result - 30.0).abs() < 1e-5);
        }

        #[test]
        fn test_neon_inner_matches_scalar_d16() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let q_len = 4;
            let kv_len = 4;
            let head_dim = 16;
            let q = make_data(q_len * head_dim, 2000);
            let k = make_data(kv_len * head_dim, 2001);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut neon_out = vec![0.0; q_len * kv_len];
            let mut scalar_out = vec![0.0; q_len * kv_len];
            unsafe {
                neon_attention_score_inner(&q, &k, &mut neon_out, q_len, kv_len, head_dim, scale);
            }
            scalar_attention_score(&q, &k, &mut scalar_out, q_len, kv_len, head_dim, scale);
            assert_close(&neon_out, &scalar_out, 1e-5, "neon_inner_d16");
        }

        #[test]
        fn test_neon_masked_inner_matches() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let q_len = 6;
            let kv_len = 6;
            let head_dim = 8;
            let q = make_data(q_len * head_dim, 2010);
            let k = make_data(kv_len * head_dim, 2011);
            let scale = 0.5;
            let mut neon_out = vec![0.0; q_len * kv_len];
            let mut scalar_out = vec![0.0; q_len * kv_len];
            unsafe {
                neon_attention_score_masked_inner(
                    &q,
                    &k,
                    &mut neon_out,
                    q_len,
                    kv_len,
                    head_dim,
                    scale,
                    f32::NEG_INFINITY,
                );
            }
            scalar_attention_score_masked(
                &q,
                &k,
                &mut scalar_out,
                q_len,
                kv_len,
                head_dim,
                scale,
                f32::NEG_INFINITY,
            );
            assert_close(&neon_out, &scalar_out, 1e-5, "neon_masked_inner");
        }

        #[test]
        fn test_neon_weighted_sum_inner_matches() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let q_len = 3;
            let kv_len = 5;
            let head_dim = 32;
            let weights: Vec<f32> =
                make_data(q_len * kv_len, 2020).iter().map(|w| w.abs()).collect();
            let v = make_data(kv_len * head_dim, 2021);
            let mut neon_out = vec![0.0; q_len * head_dim];
            let mut scalar_out = vec![0.0; q_len * head_dim];
            unsafe {
                neon_attention_weighted_sum_inner(
                    &weights,
                    &v,
                    &mut neon_out,
                    q_len,
                    kv_len,
                    head_dim,
                );
            }
            scalar_attention_weighted_sum(&weights, &v, &mut scalar_out, q_len, kv_len, head_dim);
            assert_close(&neon_out, &scalar_out, 1e-4, "neon_ws_inner");
        }

        #[test]
        fn test_neon_alibi_inner_matches() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let q_len = 4;
            let kv_len = 6;
            let head_dim = 16;
            let q = make_data(q_len * head_dim, 2030);
            let k = make_data(kv_len * head_dim, 2031);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let slope = 0.25;
            let mut neon_out = vec![0.0; q_len * kv_len];
            let mut scalar_out = vec![0.0; q_len * kv_len];
            unsafe {
                neon_attention_score_with_alibi_inner(
                    &q,
                    &k,
                    &mut neon_out,
                    q_len,
                    kv_len,
                    head_dim,
                    scale,
                    slope,
                );
            }
            scalar_attention_score_with_alibi(
                &q,
                &k,
                &mut scalar_out,
                q_len,
                kv_len,
                head_dim,
                scale,
                slope,
            );
            assert_close(&neon_out, &scalar_out, 1e-4, "neon_alibi_inner");
        }

        #[test]
        fn test_neon_gqa_inner_matches() {
            if !std::arch::is_aarch64_feature_detected!("neon") {
                return;
            }
            let num_q = 8;
            let num_kv = 2;
            let q_len = 3;
            let kv_len = 3;
            let head_dim = 16;
            let q = make_data(num_q * q_len * head_dim, 2040);
            let k = make_data(num_kv * kv_len * head_dim, 2041);
            let scale = 0.5;
            let mut neon_out = vec![0.0; num_q * q_len * kv_len];
            let mut scalar_out = vec![0.0; num_q * q_len * kv_len];
            unsafe {
                neon_attention_score_gqa_inner(
                    &q,
                    &k,
                    &mut neon_out,
                    num_q,
                    num_kv,
                    q_len,
                    kv_len,
                    head_dim,
                    scale,
                );
            }
            scalar_attention_score_gqa(
                &q,
                &k,
                &mut scalar_out,
                num_q,
                num_kv,
                q_len,
                kv_len,
                head_dim,
                scale,
            );
            assert_close(&neon_out, &scalar_out, 1e-4, "neon_gqa_inner");
        }
    }
}
