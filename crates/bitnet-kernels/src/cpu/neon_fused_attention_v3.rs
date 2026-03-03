//! ARM NEON fused attention v3 for transformer inference on Apple Silicon.
//!
//! Provides NEON-accelerated scaled dot-product attention, causal attention,
//! multi-head attention, score computation, softmax, and weighted value sum.
//! Each operation has a NEON path, a scalar fallback, and a public dispatcher.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const LANES: usize = 4;

// ── Helpers ─────────────────────────────────────────────────────────────

/// Horizontal sum of a NEON float32x4 vector.
///
/// # Safety
/// Requires aarch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    unsafe {
        let pair = vpaddq_f32(v, v);
        vgetq_lane_f32(vpaddq_f32(pair, pair), 0)
    }
}

/// Horizontal max of a NEON float32x4 vector.
///
/// # Safety
/// Requires aarch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn hmax_f32x4(v: float32x4_t) -> f32 {
    unsafe {
        let pair = vpmaxq_f32(v, v);
        vgetq_lane_f32(vpmaxq_f32(pair, pair), 0)
    }
}

// ── 1. Scaled dot-product attention ─────────────────────────────────────

/// NEON-accelerated scaled dot-product attention (single head).
///
/// # Safety
/// Requires aarch64 with NEON. Caller must ensure slices have correct sizes:
///   q, k: [seq_len * head_dim], v: [seq_len * head_dim], output: [seq_len * head_dim]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_scaled_dot_product_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    unsafe {
        let mut scores = vec![0.0f32; seq_len * seq_len];

        // Q * K^T
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut acc = vdupq_n_f32(0.0);
                let qi = &q[i * head_dim..];
                let kj = &k[j * head_dim..];
                let mut d = 0;
                while d + LANES <= head_dim {
                    let qv = vld1q_f32(qi.as_ptr().add(d));
                    let kv = vld1q_f32(kj.as_ptr().add(d));
                    acc = vfmaq_f32(acc, qv, kv);
                    d += LANES;
                }
                let mut dot = hsum_f32x4(acc);
                while d < head_dim {
                    dot += qi[d] * kj[d];
                    d += 1;
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }

        // Softmax per row
        neon_softmax_rows(&mut scores, seq_len);

        // Scores * V
        neon_weighted_value_sum_inner(&scores, v, output, seq_len, head_dim);
    }
}

fn scalar_scaled_dot_product_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }
    scalar_softmax_rows(&mut scores, seq_len);
    scalar_weighted_value_sum_inner(&scores, v, output, seq_len, head_dim);
}

pub fn scaled_dot_product_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: checked at compile time
        unsafe {
            neon_scaled_dot_product_attention_f32(q, k, v, output, seq_len, head_dim, scale);
        }
        return;
    }
    #[allow(unreachable_code)]
    {
        scalar_scaled_dot_product_attention_f32(q, k, v, output, seq_len, head_dim, scale);
    }
}

// ── 2. Causal attention ─────────────────────────────────────────────────

/// NEON-accelerated causal (masked) attention.
///
/// # Safety
/// Requires aarch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_causal_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    unsafe {
        let mut scores = vec![0.0f32; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    scores[i * seq_len + j] = f32::NEG_INFINITY;
                } else {
                    let mut acc = vdupq_n_f32(0.0);
                    let qi = &q[i * head_dim..];
                    let kj = &k[j * head_dim..];
                    let mut d = 0;
                    while d + LANES <= head_dim {
                        let qv = vld1q_f32(qi.as_ptr().add(d));
                        let kv = vld1q_f32(kj.as_ptr().add(d));
                        acc = vfmaq_f32(acc, qv, kv);
                        d += LANES;
                    }
                    let mut dot = hsum_f32x4(acc);
                    while d < head_dim {
                        dot += qi[d] * kj[d];
                        d += 1;
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }
        }

        neon_softmax_rows(&mut scores, seq_len);
        neon_weighted_value_sum_inner(&scores, v, output, seq_len, head_dim);
    }
}

fn scalar_causal_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }
    }
    scalar_softmax_rows(&mut scores, seq_len);
    scalar_weighted_value_sum_inner(&scores, v, output, seq_len, head_dim);
}

pub fn causal_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_causal_attention_f32(q, k, v, output, seq_len, head_dim, scale);
        }
        return;
    }
    #[allow(unreachable_code)]
    {
        scalar_causal_attention_f32(q, k, v, output, seq_len, head_dim, scale);
    }
}

// ── 3. Multi-head attention ─────────────────────────────────────────────

/// NEON multi-head attention: iterates over batch × heads, dispatching each
/// head to single-head scaled dot-product attention.
///
/// # Safety
/// Requires aarch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_multi_head_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    unsafe {
        let head_size = seq_len * head_dim;
        for b in 0..batch {
            for h in 0..heads {
                let offset = (b * heads + h) * head_size;
                neon_scaled_dot_product_attention_f32(
                    &q[offset..offset + head_size],
                    &k[offset..offset + head_size],
                    &v[offset..offset + head_size],
                    &mut output[offset..offset + head_size],
                    seq_len,
                    head_dim,
                    scale,
                );
            }
        }
    }
}

fn scalar_multi_head_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    let head_size = seq_len * head_dim;
    for b in 0..batch {
        for h in 0..heads {
            let offset = (b * heads + h) * head_size;
            scalar_scaled_dot_product_attention_f32(
                &q[offset..offset + head_size],
                &k[offset..offset + head_size],
                &v[offset..offset + head_size],
                &mut output[offset..offset + head_size],
                seq_len,
                head_dim,
                scale,
            );
        }
    }
}

pub fn multi_head_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_multi_head_attention_f32(q, k, v, output, batch, heads, seq_len, head_dim, scale);
        }
        return;
    }
    #[allow(unreachable_code)]
    {
        scalar_multi_head_attention_f32(q, k, v, output, batch, heads, seq_len, head_dim, scale);
    }
}

// ── 4. Attention scores (Q * K^T) ──────────────────────────────────────

/// NEON Q*K^T score computation.
///
/// # Safety
/// Requires aarch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_attention_scores_f32(
    q: &[f32],
    k: &[f32],
    scores: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    unsafe {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut acc = vdupq_n_f32(0.0);
                let qi = &q[i * head_dim..];
                let kj = &k[j * head_dim..];
                let mut d = 0;
                while d + LANES <= head_dim {
                    let qv = vld1q_f32(qi.as_ptr().add(d));
                    let kv = vld1q_f32(kj.as_ptr().add(d));
                    acc = vfmaq_f32(acc, qv, kv);
                    d += LANES;
                }
                let mut dot = hsum_f32x4(acc);
                while d < head_dim {
                    dot += qi[d] * kj[d];
                    d += 1;
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }
    }
}

fn scalar_attention_scores_f32(
    q: &[f32],
    k: &[f32],
    scores: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }
}

pub fn attention_scores_f32(
    q: &[f32],
    k: &[f32],
    scores: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_attention_scores_f32(q, k, scores, seq_len, head_dim, scale);
        }
        return;
    }
    #[allow(unreachable_code)]
    {
        scalar_attention_scores_f32(q, k, scores, seq_len, head_dim, scale);
    }
}

// ── 5. Softmax (in-place, numerically stable) ──────────────────────────

/// NEON in-place softmax over a single row.
///
/// # Safety
/// Requires aarch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_softmax_row(row: &mut [f32]) {
    unsafe {
        let len = row.len();
        if len == 0 {
            return;
        }

        // Find max
        let mut max_v = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0;
        while i + LANES <= len {
            let v = vld1q_f32(row.as_ptr().add(i));
            max_v = vmaxq_f32(max_v, v);
            i += LANES;
        }
        let mut max_val = hmax_f32x4(max_v);
        while i < len {
            max_val = max_val.max(row[i]);
            i += 1;
        }

        // exp(x - max) and sum
        let max_splat = vdupq_n_f32(max_val);
        let mut sum_v = vdupq_n_f32(0.0);
        i = 0;
        while i + LANES <= len {
            let v = vld1q_f32(row.as_ptr().add(i));
            let shifted = vsubq_f32(v, max_splat);
            // Use scalar exp for correctness
            let mut tmp = [0.0f32; LANES];
            vst1q_f32(tmp.as_mut_ptr(), shifted);
            for t in &mut tmp {
                *t = (*t).exp();
            }
            let exp_v = vld1q_f32(tmp.as_ptr());
            vst1q_f32(row.as_mut_ptr().add(i), exp_v);
            sum_v = vaddq_f32(sum_v, exp_v);
            i += LANES;
        }
        let mut sum_val = hsum_f32x4(sum_v);
        while i < len {
            let e = (row[i] - max_val).exp();
            row[i] = e;
            sum_val += e;
            i += 1;
        }

        // Normalize
        if sum_val > 0.0 {
            let inv_sum = 1.0 / sum_val;
            let inv_splat = vdupq_n_f32(inv_sum);
            i = 0;
            while i + LANES <= len {
                let v = vld1q_f32(row.as_ptr().add(i));
                let normed = vmulq_f32(v, inv_splat);
                vst1q_f32(row.as_mut_ptr().add(i), normed);
                i += LANES;
            }
            while i < len {
                row[i] *= inv_sum;
                i += 1;
            }
        }
    }
}

/// Apply softmax to each row of a seq_len × seq_len score matrix (NEON).
///
/// # Safety
/// Requires aarch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_softmax_rows(scores: &mut [f32], seq_len: usize) {
    unsafe {
        for i in 0..seq_len {
            let start = i * seq_len;
            neon_softmax_row(&mut scores[start..start + seq_len]);
        }
    }
}

fn scalar_softmax_row(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

fn scalar_softmax_rows(scores: &mut [f32], seq_len: usize) {
    for i in 0..seq_len {
        let start = i * seq_len;
        scalar_softmax_row(&mut scores[start..start + seq_len]);
    }
}

pub fn attention_softmax_f32(scores: &mut [f32], seq_len: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_softmax_rows(scores, seq_len);
        }
        return;
    }
    #[allow(unreachable_code)]
    {
        scalar_softmax_rows(scores, seq_len);
    }
}

// ── 6. Weighted value sum (weights × V) ────────────────────────────────

/// NEON weights × V matrix multiply for a single head.
///
/// # Safety
/// Requires aarch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_weighted_value_sum_inner(
    weights: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    unsafe {
        for i in 0..seq_len {
            let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
            // Zero output
            let mut d = 0;
            let zero = vdupq_n_f32(0.0);
            while d + LANES <= head_dim {
                vst1q_f32(out_row.as_mut_ptr().add(d), zero);
                d += LANES;
            }
            while d < head_dim {
                out_row[d] = 0.0;
                d += 1;
            }

            for j in 0..seq_len {
                let w = weights[i * seq_len + j];
                if w == 0.0 {
                    continue;
                }
                let w_splat = vdupq_n_f32(w);
                let vj = &v[j * head_dim..];
                d = 0;
                while d + LANES <= head_dim {
                    let ov = vld1q_f32(out_row.as_ptr().add(d));
                    let vv = vld1q_f32(vj.as_ptr().add(d));
                    let res = vfmaq_f32(ov, w_splat, vv);
                    vst1q_f32(out_row.as_mut_ptr().add(d), res);
                    d += LANES;
                }
                while d < head_dim {
                    out_row[d] += w * vj[d];
                    d += 1;
                }
            }
        }
    }
}

fn scalar_weighted_value_sum_inner(
    weights: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    for i in 0..seq_len {
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for j in 0..seq_len {
                acc += weights[i * seq_len + j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = acc;
        }
    }
}

/// NEON-accelerated weighted value sum.
///
/// # Safety
/// Requires aarch64 with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_weighted_value_sum_f32(
    weights: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    unsafe {
        neon_weighted_value_sum_inner(weights, v, output, seq_len, head_dim);
    }
}

fn scalar_weighted_value_sum_f32(
    weights: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    scalar_weighted_value_sum_inner(weights, v, output, seq_len, head_dim);
}

pub fn weighted_value_sum_f32(
    weights: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon_weighted_value_sum_f32(weights, v, output, seq_len, head_dim);
        }
        return;
    }
    #[allow(unreachable_code)]
    {
        scalar_weighted_value_sum_f32(weights, v, output, seq_len, head_dim);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn slice_approx_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx_eq(*x, *y, eps))
    }

    /// Reference scalar attention for cross-checking.
    fn reference_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
    ) -> Vec<f32> {
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if causal && j > i {
                    scores[i * seq_len + j] = f32::NEG_INFINITY;
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }
        }
        // Softmax per row
        for i in 0..seq_len {
            let row = &mut scores[i * seq_len..(i + 1) * seq_len];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
        let mut output = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq_len {
                    acc += scores[i * seq_len + j] * v[j * head_dim + d];
                }
                output[i * head_dim + d] = acc;
            }
        }
        output
    }

    // ── 1. Scaled dot-product attention tests ───────────────────────────

    #[test]
    fn test_sdpa_single_token() {
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.5, 0.5, 0.5, 0.5];
        let mut out = vec![0.0; 4];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, 1, 4, 0.5);
        assert!(slice_approx_eq(&out, &v, EPS), "single token: out={out:?}");
    }

    #[test]
    fn test_sdpa_identity_qk() {
        // Q == K, seq_len=2, head_dim=4 — both keys are identical so
        // scores are uniform → output = average of V rows
        let q = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let k = q.clone();
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; 8];
        let scale = 1.0 / (4.0f32).sqrt();
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, 2, 4, scale);
        let expected = reference_attention(&q, &k, &v, 2, 4, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS), "identity: out={out:?} exp={expected:?}");
    }

    #[test]
    fn test_sdpa_two_tokens_distinct() {
        let q = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut out = vec![0.0; 6];
        let scale = 1.0 / (3.0f32).sqrt();
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, 2, 3, scale);
        let expected = reference_attention(&q, &k, &v, 2, 3, scale, false);
        assert!(
            slice_approx_eq(&out, &expected, EPS),
            "two_distinct: out={out:?} exp={expected:?}"
        );
    }

    #[test]
    fn test_sdpa_matches_reference_seq4_dim8() {
        let seq_len = 4;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05 + 1.0)).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS), "ref4x8: out={out:?} exp={expected:?}");
    }

    #[test]
    fn test_sdpa_scale_effect() {
        // Larger scale → sharper softmax
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut out_small = vec![0.0; 8];
        let mut out_large = vec![0.0; 8];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_small, 2, 4, 0.1);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_large, 2, 4, 10.0);
        // With large scale, first query should attend almost entirely to first key
        assert!(out_large[0] > out_small[0]);
    }

    #[test]
    fn test_sdpa_output_length() {
        let n = 3 * 4;
        let q = vec![0.0; n];
        let k = vec![0.0; n];
        let v = vec![1.0; n];
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, 3, 4, 1.0);
        assert_eq!(out.len(), n);
    }

    // ── 2. Causal attention tests ───────────────────────────────────────

    #[test]
    fn test_causal_single_token() {
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.5, 0.5, 0.5, 0.5];
        let mut out = vec![0.0; 4];
        causal_attention_f32(&q, &k, &v, &mut out, 1, 4, 0.5);
        assert!(slice_approx_eq(&out, &v, EPS));
    }

    #[test]
    fn test_causal_future_mask() {
        // seq_len=2: first position can only see itself
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; 8];
        let scale = 1.0 / 2.0;
        causal_attention_f32(&q, &k, &v, &mut out, 2, 4, scale);
        // First row can only attend to position 0 → output[0..4] == v[0..4]
        assert!(slice_approx_eq(&out[0..4], &v[0..4], EPS), "causal first row: {:?}", &out[0..4]);
    }

    #[test]
    fn test_causal_matches_reference() {
        let seq_len = 4;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05 + 1.0)).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, true);
        assert!(slice_approx_eq(&out, &expected, EPS), "causal_ref: out={out:?} exp={expected:?}");
    }

    #[test]
    fn test_causal_first_row_equals_non_causal() {
        let q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let k = q.clone();
        let v = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let scale = 1.0 / 2.0;

        let mut out_causal = vec![0.0; 8];
        let mut out_full = vec![0.0; 8];
        // seq_len=1 → causal and non-causal are identical (only one token)
        causal_attention_f32(&q[0..4], &k[0..4], &v[0..4], &mut out_causal[0..4], 1, 4, scale);
        scaled_dot_product_attention_f32(
            &q[0..4],
            &k[0..4],
            &v[0..4],
            &mut out_full[0..4],
            1,
            4,
            scale,
        );
        assert!(slice_approx_eq(&out_causal[0..4], &out_full[0..4], EPS));
    }

    #[test]
    fn test_causal_lower_triangular_weights() {
        // After softmax, future positions should have weight 0
        let seq_len = 4;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q = vec![1.0; n];
        let k = vec![1.0; n];
        let v = vec![1.0; n];
        let mut scores = vec![0.0; seq_len * seq_len];
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Compute scores manually with causal mask
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    scores[i * seq_len + j] = f32::NEG_INFINITY;
                } else {
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }
        }
        attention_softmax_f32(&mut scores, seq_len);
        // Verify upper triangle is 0
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                assert!(
                    scores[i * seq_len + j].abs() < EPS,
                    "future pos [{i}][{j}] = {}",
                    scores[i * seq_len + j]
                );
            }
        }
    }

    #[test]
    fn test_causal_seq8() {
        let seq_len = 8;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.3).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, true);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    // ── 3. Multi-head attention tests ───────────────────────────────────

    #[test]
    fn test_mha_single_head() {
        let seq_len = 2;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let k = q.clone();
        let v = vec![1.0; n];
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_mha = vec![0.0; n];
        let mut out_sdpa = vec![0.0; n];
        multi_head_attention_f32(&q, &k, &v, &mut out_mha, 1, 1, seq_len, head_dim, scale);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_sdpa, seq_len, head_dim, scale);
        assert!(slice_approx_eq(&out_mha, &out_sdpa, EPS));
    }

    #[test]
    fn test_mha_two_heads() {
        let batch = 1;
        let heads = 2;
        let seq_len = 2;
        let head_dim = 4;
        let head_size = seq_len * head_dim;
        let total = batch * heads * head_size;

        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..total).map(|i| i as f32 * 0.05).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out = vec![0.0; total];
        multi_head_attention_f32(&q, &k, &v, &mut out, batch, heads, seq_len, head_dim, scale);

        // Verify each head independently
        for h in 0..heads {
            let off = h * head_size;
            let mut expected = vec![0.0; head_size];
            let ref_out = reference_attention(
                &q[off..off + head_size],
                &k[off..off + head_size],
                &v[off..off + head_size],
                seq_len,
                head_dim,
                scale,
                false,
            );
            expected.copy_from_slice(&ref_out);
            assert!(slice_approx_eq(&out[off..off + head_size], &expected, EPS), "head {h}");
        }
    }

    #[test]
    fn test_mha_batch2_heads2() {
        let batch = 2;
        let heads = 2;
        let seq_len = 2;
        let head_dim = 4;
        let head_size = seq_len * head_dim;
        let total = batch * heads * head_size;

        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.07).sin()).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.13).cos()).collect();
        let v: Vec<f32> = (0..total).map(|i| i as f32 * 0.02).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out = vec![0.0; total];
        multi_head_attention_f32(&q, &k, &v, &mut out, batch, heads, seq_len, head_dim, scale);

        for b in 0..batch {
            for h in 0..heads {
                let off = (b * heads + h) * head_size;
                let expected = reference_attention(
                    &q[off..off + head_size],
                    &k[off..off + head_size],
                    &v[off..off + head_size],
                    seq_len,
                    head_dim,
                    scale,
                    false,
                );
                assert!(slice_approx_eq(&out[off..off + head_size], &expected, EPS), "b={b} h={h}");
            }
        }
    }

    #[test]
    fn test_mha_output_length() {
        let batch = 2;
        let heads = 3;
        let seq_len = 4;
        let head_dim = 8;
        let total = batch * heads * seq_len * head_dim;
        let q = vec![0.0; total];
        let k = vec![0.0; total];
        let v = vec![1.0; total];
        let mut out = vec![0.0; total];
        multi_head_attention_f32(&q, &k, &v, &mut out, batch, heads, seq_len, head_dim, 1.0);
        assert_eq!(out.len(), total);
    }

    // ── 4. Attention scores tests ───────────────────────────────────────

    #[test]
    fn test_scores_single() {
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let mut scores = vec![0.0; 1];
        attention_scores_f32(&q, &k, &mut scores, 1, 4, 0.5);
        assert!(approx_eq(scores[0], 0.5, EPS));
    }

    #[test]
    fn test_scores_orthogonal() {
        let q = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut scores = vec![0.0; 4];
        attention_scores_f32(&q, &k, &mut scores, 2, 3, 1.0);
        assert!(approx_eq(scores[0], 0.0, EPS), "q0·k0 orthogonal");
        assert!(approx_eq(scores[3], 0.0, EPS), "q1·k1 orthogonal");
    }

    #[test]
    fn test_scores_manual_dot_product() {
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![2.0, 0.0, 1.0, 0.0];
        let mut scores = vec![0.0; 1];
        attention_scores_f32(&q, &k, &mut scores, 1, 4, 1.0);
        // dot = 1*2 + 2*0 + 3*1 + 4*0 = 5
        assert!(approx_eq(scores[0], 5.0, EPS), "dot={}", scores[0]);
    }

    #[test]
    fn test_scores_with_scale() {
        let q = vec![1.0, 1.0, 1.0, 1.0];
        let k = vec![1.0, 1.0, 1.0, 1.0];
        let mut scores = vec![0.0; 1];
        let scale = 0.25;
        attention_scores_f32(&q, &k, &mut scores, 1, 4, scale);
        // dot=4, scaled=1.0
        assert!(approx_eq(scores[0], 1.0, EPS));
    }

    #[test]
    fn test_scores_shape_2x2() {
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let mut scores = vec![0.0; 4];
        attention_scores_f32(&q, &k, &mut scores, 2, 2, 1.0);
        assert!(approx_eq(scores[0], 1.0, EPS)); // q0·k0
        assert!(approx_eq(scores[1], 0.0, EPS)); // q0·k1
        assert!(approx_eq(scores[2], 0.0, EPS)); // q1·k0
        assert!(approx_eq(scores[3], 1.0, EPS)); // q1·k1
    }

    #[test]
    fn test_scores_negative_values() {
        let q = vec![-1.0, 2.0, -3.0, 4.0];
        let k = vec![1.0, -2.0, 3.0, -4.0];
        let mut scores = vec![0.0; 1];
        attention_scores_f32(&q, &k, &mut scores, 1, 4, 1.0);
        // dot = -1 + -4 + -9 + -16 = -30
        assert!(approx_eq(scores[0], -30.0, EPS), "neg={}", scores[0]);
    }

    // ── 5. Softmax tests ────────────────────────────────────────────────

    #[test]
    fn test_softmax_single_row() {
        let mut scores = vec![1.0, 2.0, 3.0];
        scalar_softmax_row(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!(approx_eq(sum, 1.0, EPS), "sum={sum}");
    }

    #[test]
    fn test_softmax_uniform() {
        let mut scores = vec![0.0; 4];
        attention_softmax_f32(&mut scores, 2);
        // 2 rows of 2: each should be [0.5, 0.5]
        assert!(approx_eq(scores[0], 0.5, EPS));
        assert!(approx_eq(scores[1], 0.5, EPS));
        assert!(approx_eq(scores[2], 0.5, EPS));
        assert!(approx_eq(scores[3], 0.5, EPS));
    }

    #[test]
    fn test_softmax_row_sums_to_one() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        attention_softmax_f32(&mut scores, 3);
        for i in 0..3 {
            let sum: f32 = scores[i * 3..(i + 1) * 3].iter().sum();
            assert!(approx_eq(sum, 1.0, EPS), "row {i} sum={sum}");
        }
    }

    #[test]
    fn test_softmax_preserves_order() {
        let mut scores = vec![1.0, 3.0, 2.0, 0.5];
        attention_softmax_f32(&mut scores, 2);
        // Row 0: [1, 3] → s[0] < s[1]
        assert!(scores[0] < scores[1]);
        // Row 1: [2, 0.5] → s[2] > s[3]
        assert!(scores[2] > scores[3]);
    }

    #[test]
    fn test_softmax_numerical_stability_large() {
        let mut scores = vec![1000.0, 1001.0, 1002.0, 1003.0];
        attention_softmax_f32(&mut scores, 2);
        for s in &scores {
            assert!(s.is_finite(), "overflow: {s}");
        }
        let sum0: f32 = scores[0..2].iter().sum();
        let sum1: f32 = scores[2..4].iter().sum();
        assert!(approx_eq(sum0, 1.0, EPS));
        assert!(approx_eq(sum1, 1.0, EPS));
    }

    #[test]
    fn test_softmax_numerical_stability_negative() {
        let mut scores = vec![-1000.0, -999.0, -998.0, -997.0];
        attention_softmax_f32(&mut scores, 2);
        for s in &scores {
            assert!(s.is_finite(), "underflow: {s}");
        }
        let sum: f32 = scores[0..2].iter().sum();
        assert!(approx_eq(sum, 1.0, EPS));
    }

    #[test]
    fn test_softmax_with_neg_inf() {
        let mut scores = vec![1.0, f32::NEG_INFINITY, f32::NEG_INFINITY, 2.0];
        attention_softmax_f32(&mut scores, 2);
        // Row 0: only first element should be 1.0
        assert!(approx_eq(scores[0], 1.0, EPS));
        assert!(approx_eq(scores[1], 0.0, EPS));
        // Row 1: only second element should be 1.0
        assert!(approx_eq(scores[2], 0.0, EPS));
        assert!(approx_eq(scores[3], 1.0, EPS));
    }

    #[test]
    fn test_softmax_single_element() {
        let mut scores = vec![42.0];
        attention_softmax_f32(&mut scores, 1);
        assert!(approx_eq(scores[0], 1.0, EPS));
    }

    #[test]
    fn test_softmax_all_same() {
        let mut scores = vec![5.0; 16];
        attention_softmax_f32(&mut scores, 4);
        for i in 0..4 {
            for j in 0..4 {
                assert!(approx_eq(scores[i * 4 + j], 0.25, EPS));
            }
        }
    }

    // ── 6. Weighted value sum tests ─────────────────────────────────────

    #[test]
    fn test_wvs_identity_weights() {
        // weights = identity → output = v
        let weights = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; 8];
        weighted_value_sum_f32(&weights, &v, &mut out, 2, 4);
        assert!(slice_approx_eq(&out[0..4], &[1.0, 2.0, 3.0, 4.0], EPS));
        assert!(slice_approx_eq(&out[4..8], &[5.0, 6.0, 7.0, 8.0], EPS));
    }

    #[test]
    fn test_wvs_uniform_weights() {
        let weights = vec![0.5, 0.5, 0.5, 0.5];
        let v = vec![0.0, 0.0, 10.0, 10.0];
        let mut out = vec![0.0; 4];
        weighted_value_sum_f32(&weights, &v, &mut out, 2, 2);
        // Each row: 0.5 * [0,0] + 0.5 * [10,10] = [5, 5]
        assert!(slice_approx_eq(&out, &[5.0, 5.0, 5.0, 5.0], EPS));
    }

    #[test]
    fn test_wvs_single_token() {
        let weights = vec![1.0];
        let v = vec![3.14, 2.72, 1.41, 1.73];
        let mut out = vec![0.0; 4];
        weighted_value_sum_f32(&weights, &v, &mut out, 1, 4);
        assert!(slice_approx_eq(&out, &v, EPS));
    }

    #[test]
    fn test_wvs_zero_weight() {
        let weights = vec![0.0, 0.0, 0.0, 0.0];
        let v = vec![999.0, 999.0, 999.0, 999.0];
        let mut out = vec![0.0; 4];
        weighted_value_sum_f32(&weights, &v, &mut out, 2, 2);
        assert!(slice_approx_eq(&out, &[0.0, 0.0, 0.0, 0.0], EPS));
    }

    // ── NEON vs scalar consistency ──────────────────────────────────────

    #[test]
    fn test_scalar_sdpa_matches_reference() {
        let seq_len = 3;
        let head_dim = 5;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scalar_scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_scalar_causal_matches_reference() {
        let seq_len = 3;
        let head_dim = 5;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scalar_causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, true);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_scalar_scores_manual() {
        let q = vec![2.0, 3.0];
        let k = vec![4.0, 5.0];
        let mut scores = vec![0.0; 1];
        scalar_attention_scores_f32(&q, &k, &mut scores, 1, 2, 1.0);
        assert!(approx_eq(scores[0], 23.0, EPS)); // 2*4+3*5
    }

    #[test]
    fn test_scalar_softmax_sums_to_one() {
        let mut row = vec![2.0, 4.0, 6.0, 8.0];
        scalar_softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!(approx_eq(sum, 1.0, EPS));
    }

    #[test]
    fn test_scalar_wvs_matches_manual() {
        // seq_len=2, head_dim=2: weights is 2×2, v is 2×2
        let weights = vec![0.3, 0.7, 0.5, 0.5];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 4];
        scalar_weighted_value_sum_f32(&weights, &v, &mut out, 2, 2);
        // row0: 0.3*[10,20] + 0.7*[30,40] = [24, 34]
        assert!(approx_eq(out[0], 24.0, EPS), "out[0]={}", out[0]);
        assert!(approx_eq(out[1], 34.0, EPS), "out[1]={}", out[1]);
        // row1: 0.5*[10,20] + 0.5*[30,40] = [20, 30]
        assert!(approx_eq(out[2], 20.0, EPS), "out[2]={}", out[2]);
        assert!(approx_eq(out[3], 30.0, EPS), "out[3]={}", out[3]);
    }

    // ── Dimension edge cases ────────────────────────────────────────────

    #[test]
    fn test_head_dim_1() {
        let q = vec![2.0, 3.0];
        let k = vec![4.0, 5.0];
        let v = vec![1.0, 0.0];
        let mut out = vec![0.0; 2];
        let scale = 1.0;
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, 2, 1, scale);
        let expected = reference_attention(&q, &k, &v, 2, 1, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_head_dim_4() {
        let seq_len = 3;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_head_dim_8() {
        let seq_len = 2;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_head_dim_16() {
        let seq_len = 2;
        let head_dim = 16;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_head_dim_32() {
        let seq_len = 2;
        let head_dim = 32;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_head_dim_64() {
        let seq_len = 2;
        let head_dim = 64;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.02).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_seq_len_1() {
        let head_dim = 4;
        let q = vec![1.0; head_dim];
        let k = vec![1.0; head_dim];
        let v = vec![7.0; head_dim];
        let mut out = vec![0.0; head_dim];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, 1, head_dim, 1.0);
        assert!(slice_approx_eq(&out, &v, EPS));
    }

    #[test]
    fn test_seq_len_2() {
        let seq_len = 2;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_seq_len_4() {
        let seq_len = 4;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_seq_len_8() {
        let seq_len = 8;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_seq_len_16() {
        let seq_len = 16;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.005).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    // ── Causal dimension edge cases ─────────────────────────────────────

    #[test]
    fn test_causal_head_dim_1() {
        let q = vec![1.0, 2.0];
        let k = vec![3.0, 4.0];
        let v = vec![10.0, 20.0];
        let mut out = vec![0.0; 2];
        causal_attention_f32(&q, &k, &v, &mut out, 2, 1, 1.0);
        let expected = reference_attention(&q, &k, &v, 2, 1, 1.0, true);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_causal_head_dim_16() {
        let seq_len = 4;
        let head_dim = 16;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, true);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_causal_seq_len_16() {
        let seq_len = 16;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.04).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.04).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * 0.002).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, true);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    // ── Additional cross-checks ─────────────────────────────────────────

    #[test]
    fn test_sdpa_all_zeros_q() {
        let seq_len = 2;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q = vec![0.0; n];
        let k: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let v = vec![1.0; n];
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, 1.0);
        // All q are zero → all scores equal → uniform attention → avg of v rows
        // Since all v = 1.0, output should be 1.0
        for o in &out {
            assert!(approx_eq(*o, 1.0, EPS));
        }
    }

    #[test]
    fn test_sdpa_deterministic() {
        let seq_len = 4;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out1 = vec![0.0; n];
        let mut out2 = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out1, seq_len, head_dim, scale);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out2, seq_len, head_dim, scale);
        assert_eq!(out1, out2, "deterministic");
    }

    #[test]
    fn test_causal_deterministic() {
        let seq_len = 4;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out1 = vec![0.0; n];
        let mut out2 = vec![0.0; n];
        causal_attention_f32(&q, &k, &v, &mut out1, seq_len, head_dim, scale);
        causal_attention_f32(&q, &k, &v, &mut out2, seq_len, head_dim, scale);
        assert_eq!(out1, out2, "causal deterministic");
    }

    #[test]
    fn test_mha_deterministic() {
        let batch = 1;
        let heads = 2;
        let seq_len = 3;
        let head_dim = 4;
        let total = batch * heads * seq_len * head_dim;
        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.11).sin()).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.13).cos()).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.17).sin()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out1 = vec![0.0; total];
        let mut out2 = vec![0.0; total];
        multi_head_attention_f32(&q, &k, &v, &mut out1, batch, heads, seq_len, head_dim, scale);
        multi_head_attention_f32(&q, &k, &v, &mut out2, batch, heads, seq_len, head_dim, scale);
        assert_eq!(out1, out2, "mha deterministic");
    }

    #[test]
    fn test_scores_seq4_dim8() {
        let seq_len = 4;
        let head_dim = 8;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0; seq_len * seq_len];
        let mut scores_ref = vec![0.0; seq_len * seq_len];
        attention_scores_f32(&q, &k, &mut scores, seq_len, head_dim, scale);
        scalar_attention_scores_f32(&q, &k, &mut scores_ref, seq_len, head_dim, scale);
        assert!(slice_approx_eq(&scores, &scores_ref, EPS));
    }

    #[test]
    fn test_wvs_seq4_dim8() {
        let seq_len = 4;
        let head_dim = 8;
        // Uniform weights (each row sums to 1)
        let mut weights = vec![0.0; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                weights[i * seq_len + j] = 1.0 / seq_len as f32;
            }
        }
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.0; seq_len * head_dim];
        let mut out_ref = vec![0.0; seq_len * head_dim];
        weighted_value_sum_f32(&weights, &v, &mut out, seq_len, head_dim);
        scalar_weighted_value_sum_f32(&weights, &v, &mut out_ref, seq_len, head_dim);
        assert!(slice_approx_eq(&out, &out_ref, EPS));
    }

    #[test]
    fn test_softmax_large_seq() {
        let seq_len = 8;
        let mut scores: Vec<f32> = (0..seq_len * seq_len).map(|i| (i as f32 * 0.3).sin()).collect();
        attention_softmax_f32(&mut scores, seq_len);
        for i in 0..seq_len {
            let sum: f32 = scores[i * seq_len..(i + 1) * seq_len].iter().sum();
            assert!(approx_eq(sum, 1.0, EPS), "row {i} sum={sum}");
        }
    }

    #[test]
    fn test_causal_last_row_attends_all() {
        // Last row of causal attention can see all positions
        let seq_len = 4;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let q = vec![1.0; n];
        let k = vec![1.0; n];
        let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out_causal = vec![0.0; n];
        let mut out_full = vec![0.0; n];
        causal_attention_f32(&q, &k, &v, &mut out_causal, seq_len, head_dim, scale);
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out_full, seq_len, head_dim, scale);
        // Last row should be identical between causal and full
        let last = (seq_len - 1) * head_dim;
        assert!(slice_approx_eq(
            &out_causal[last..last + head_dim],
            &out_full[last..last + head_dim],
            EPS
        ));
    }

    #[test]
    fn test_sdpa_non_multiple_of_4_dim() {
        // head_dim=5 (not a multiple of LANES)
        let seq_len = 3;
        let head_dim = 5;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.1).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        scaled_dot_product_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, false);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_causal_non_multiple_of_4_dim() {
        let seq_len = 3;
        let head_dim = 7;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.1).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; n];
        causal_attention_f32(&q, &k, &v, &mut out, seq_len, head_dim, scale);
        let expected = reference_attention(&q, &k, &v, seq_len, head_dim, scale, true);
        assert!(slice_approx_eq(&out, &expected, EPS));
    }

    #[test]
    fn test_scores_non_multiple_of_4_dim() {
        let seq_len = 2;
        let head_dim = 3;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let k: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let mut scores = vec![0.0; seq_len * seq_len];
        let mut scores_ref = vec![0.0; seq_len * seq_len];
        attention_scores_f32(&q, &k, &mut scores, seq_len, head_dim, 1.0);
        scalar_attention_scores_f32(&q, &k, &mut scores_ref, seq_len, head_dim, 1.0);
        assert!(slice_approx_eq(&scores, &scores_ref, EPS));
    }

    #[test]
    fn test_wvs_non_multiple_of_4_dim() {
        let seq_len = 2;
        let head_dim = 3;
        let weights = vec![0.6, 0.4, 0.3, 0.7];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0; seq_len * head_dim];
        let mut out_ref = vec![0.0; seq_len * head_dim];
        weighted_value_sum_f32(&weights, &v, &mut out, seq_len, head_dim);
        scalar_weighted_value_sum_f32(&weights, &v, &mut out_ref, seq_len, head_dim);
        assert!(slice_approx_eq(&out, &out_ref, EPS));
    }

    #[test]
    fn test_mha_heads4_seq4_dim16() {
        let batch = 1;
        let heads = 4;
        let seq_len = 4;
        let head_dim = 16;
        let head_size = seq_len * head_dim;
        let total = batch * heads * head_size;
        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.03).sin()).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.03).cos()).collect();
        let v: Vec<f32> = (0..total).map(|i| i as f32 * 0.001).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0; total];
        multi_head_attention_f32(&q, &k, &v, &mut out, batch, heads, seq_len, head_dim, scale);
        for h in 0..heads {
            let off = h * head_size;
            let expected = reference_attention(
                &q[off..off + head_size],
                &k[off..off + head_size],
                &v[off..off + head_size],
                seq_len,
                head_dim,
                scale,
                false,
            );
            assert!(slice_approx_eq(&out[off..off + head_size], &expected, EPS), "h={h}");
        }
    }

    #[test]
    fn test_softmax_monotone_row() {
        // Input is strictly increasing → output should be strictly increasing
        let mut scores = vec![0.0, 1.0, 2.0, 3.0];
        scalar_softmax_row(&mut scores);
        for i in 1..4 {
            assert!(scores[i] > scores[i - 1]);
        }
    }

    #[test]
    fn test_softmax_symmetric_input() {
        let mut scores = vec![-1.0, 1.0, 1.0, -1.0];
        attention_softmax_f32(&mut scores, 2);
        assert!(approx_eq(scores[0] + scores[1], 1.0, EPS));
        assert!(scores[0] < scores[1]); // -1 < 1
        assert!(scores[2] > scores[3]); // 1 > -1
    }

    #[test]
    fn test_scores_symmetric() {
        // If Q == K, the score matrix should be symmetric
        let seq_len = 3;
        let head_dim = 4;
        let n = seq_len * head_dim;
        let qk: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();
        let mut scores = vec![0.0; seq_len * seq_len];
        attention_scores_f32(&qk, &qk, &mut scores, seq_len, head_dim, 1.0);
        for i in 0..seq_len {
            for j in 0..seq_len {
                assert!(
                    approx_eq(scores[i * seq_len + j], scores[j * seq_len + i], EPS),
                    "sym [{i}][{j}]"
                );
            }
        }
    }
}
