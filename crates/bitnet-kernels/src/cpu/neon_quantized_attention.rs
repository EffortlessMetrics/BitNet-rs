//! ARM NEON accelerated quantized attention for ternary weight matrices.
//!
//! Computes multi-head attention scores and output using NEON SIMD when
//! weight matrices are ternary-encoded (`-1`, `0`, `+1` as `i8`).
//!
//! The ternary dot-product avoids multiplies: weight `+1` adds the input
//! element, weight `-1` subtracts it, and weight `0` skips it. NEON
//! intrinsics vectorise the comparison/blend loop over 4 `f32` lanes.

use std::arch::aarch64::*;

// ── Helpers ─────────────────────────────────────────────────────────────

/// Scalar ternary dot-product (reference / tail fallback).
#[inline]
#[allow(dead_code)]
fn ternary_dot_scalar(a: &[f32], weights: &[i8]) -> f32 {
    a.iter().zip(weights.iter()).fold(0.0f32, |acc, (&x, &w)| match w {
        1 => acc + x,
        -1 => acc - x,
        _ => acc,
    })
}

/// NEON-accelerated ternary dot-product.
///
/// For each lane the weight is one of {-1, 0, +1}. We convert weights to
/// `f32`, then use `vmlaq_f32` (fused multiply-accumulate) so the product
/// is exactly `{-x, 0, +x}`.
///
/// # Safety
///
/// Caller must ensure the `neon` target feature is available at runtime
/// and that `a` and `weights` have the same length.
#[allow(dead_code)]
#[target_feature(enable = "neon")]
unsafe fn ternary_dot_neon(a: &[f32], weights: &[i8]) -> f32 {
    debug_assert_eq!(a.len(), weights.len());

    let len = a.len();
    let chunks = len / 4;
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let va = unsafe { vld1q_f32(a.as_ptr().add(offset)) };
        // Convert i8 weights to f32 for FMA
        let w = [
            weights[offset] as f32,
            weights[offset + 1] as f32,
            weights[offset + 2] as f32,
            weights[offset + 3] as f32,
        ];
        let vw = unsafe { vld1q_f32(w.as_ptr()) };
        acc = vmlaq_f32(acc, va, vw);
    }

    // Horizontal sum of the 4-lane accumulator
    let sum = vaddvq_f32(acc);

    // Scalar tail
    let tail_start = chunks * 4;
    let tail = ternary_dot_scalar(&a[tail_start..], &weights[tail_start..]);

    sum + tail
}

// ── Public API ──────────────────────────────────────────────────────────

/// Compute quantized attention scores for all heads using NEON SIMD.
///
/// `query` and `key` are flat buffers of shape `[num_heads * head_dim]`.
/// `weights` is a flat ternary weight matrix of shape `[num_heads * head_dim]`
/// applied element-wise to the key before the dot-product with the query.
///
/// Returns a `Vec<f32>` of length `num_heads`, one score per head, each
/// scaled by `1 / sqrt(head_dim)`.
///
/// # Panics
///
/// Panics if the input slice lengths are not `num_heads * head_dim`.
pub fn quantized_attention_scores_neon(
    query: &[f32],
    key: &[f32],
    weights: &[i8],
    head_dim: usize,
    num_heads: usize,
) -> Vec<f32> {
    let total = num_heads * head_dim;
    assert_eq!(query.len(), total, "query length mismatch");
    assert_eq!(key.len(), total, "key length mismatch");
    assert_eq!(weights.len(), total, "weights length mismatch");
    assert!(head_dim > 0, "head_dim must be positive");

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        let start = h * head_dim;
        let end = start + head_dim;

        let q_head = &query[start..end];
        let k_head = &key[start..end];
        let w_head = &weights[start..end];

        // Apply ternary weights to key, then dot with query.
        // Ternary weight * key gives a weighted key; dot with query gives the score.
        // Equivalent to: sum_i q[i] * (w[i] * k[i]) = sum_i q[i]*k[i]*w[i]
        //
        // We compute this as two ternary dot-products composed, but it is
        // more efficient to fuse: accumulate q[i]*k[i] gated by the weight.
        let score = fused_ternary_qk_dot(q_head, k_head, w_head);

        scores.push(score * scale);
    }

    scores
}

/// Fused ternary gated Q·K dot-product.
///
/// Computes `sum_i q[i] * k[i] * w[i]` where w[i] ∈ {-1, 0, +1}.
#[inline]
fn fused_ternary_qk_dot(q: &[f32], k: &[f32], w: &[i8]) -> f32 {
    // Build element-wise product q*k, then ternary-dot with weights.
    // For better cache behaviour we do it in NEON-width chunks.
    // Safety: we are on aarch64 with neon always available.
    unsafe { fused_ternary_qk_dot_neon(q, k, w) }
}

/// NEON implementation of the fused Q·K·W dot-product.
#[target_feature(enable = "neon")]
unsafe fn fused_ternary_qk_dot_neon(q: &[f32], k: &[f32], w: &[i8]) -> f32 {
    let len = q.len();
    let chunks = len / 4;
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let off = i * 4;
        let vq = unsafe { vld1q_f32(q.as_ptr().add(off)) };
        let vk = unsafe { vld1q_f32(k.as_ptr().add(off)) };
        let qk = vmulq_f32(vq, vk);

        let wf = [w[off] as f32, w[off + 1] as f32, w[off + 2] as f32, w[off + 3] as f32];
        let vw = unsafe { vld1q_f32(wf.as_ptr()) };
        acc = vmlaq_f32(acc, qk, vw);
    }

    let mut sum = vaddvq_f32(acc);

    // Scalar tail
    let tail = chunks * 4;
    for i in tail..len {
        let wf = w[i] as f32;
        sum += q[i] * k[i] * wf;
    }

    sum
}

/// Compute multi-head attention output with ternary-weighted keys.
///
/// `q`, `k`, `v` are flat buffers of shape `[num_heads * head_dim]`.
/// `weights` is a flat ternary weight matrix of shape `[num_heads * head_dim]`
/// applied element-wise to the key in the score computation.
///
/// For each head:
///   1. score = (Q · (W ⊙ K)) / sqrt(head_dim)
///   2. attn_weight = softmax(score)  (single-key → trivially 1.0)
///   3. output = attn_weight * V
///
/// With a single key position the softmax is always 1.0, so the output
/// per head equals the value vector scaled by 1.0. This matches the
/// common "self-attention on a single token" pattern used during
/// autoregressive decoding.
///
/// Returns a `Vec<f32>` of length `num_heads * head_dim`.
///
/// # Panics
///
/// Panics if any input slice length is not `num_heads * head_dim`.
pub fn quantized_multi_head_attention_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    weights: &[i8],
    head_dim: usize,
    num_heads: usize,
) -> Vec<f32> {
    let total = num_heads * head_dim;
    assert_eq!(q.len(), total, "q length mismatch");
    assert_eq!(k.len(), total, "k length mismatch");
    assert_eq!(v.len(), total, "v length mismatch");
    assert_eq!(weights.len(), total, "weights length mismatch");
    assert!(head_dim > 0, "head_dim must be positive");

    // With a single key position, softmax over one element is 1.0.
    // The output for each head is simply V[h] (the attention weight is 1.0).
    //
    // We still compute the scores so the kernel exercises the full
    // NEON path and callers can extend to multi-key sequences.
    let _scores = quantized_attention_scores_neon(q, k, weights, head_dim, num_heads);

    // output = attn_weight * V = 1.0 * V  (single-key softmax)
    let mut output = Vec::with_capacity(total);

    for h in 0..num_heads {
        let start = h * head_dim;
        let end = start + head_dim;
        let v_head = &v[start..end];

        // Scale V by the attention weight (1.0 in single-key case)
        unsafe {
            scale_vector_neon(v_head, 1.0, &mut output);
        }
    }

    output
}

/// NEON-accelerated vector scaling: pushes `vec[i] * scale` onto `out`.
#[target_feature(enable = "neon")]
unsafe fn scale_vector_neon(vec: &[f32], scale: f32, out: &mut Vec<f32>) {
    let len = vec.len();
    let chunks = len / 4;
    let vs = vdupq_n_f32(scale);

    out.reserve(len);
    let base = out.len();

    // Pre-extend to avoid repeated bounds checks
    for _ in 0..len {
        out.push(0.0);
    }

    for i in 0..chunks {
        let off = i * 4;
        let v = unsafe { vld1q_f32(vec.as_ptr().add(off)) };
        let r = vmulq_f32(v, vs);
        unsafe { vst1q_f32(out.as_mut_ptr().add(base + off), r) };
    }

    // Scalar tail
    let tail = chunks * 4;
    for i in tail..len {
        out[base + i] = vec[i] * scale;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparisons
    const EPS: f32 = 1e-5;

    fn assert_approx_eq(a: f32, b: f32, label: &str) {
        assert!((a - b).abs() < EPS, "{label}: expected {b}, got {a} (diff {})", (a - b).abs());
    }

    // ── ternary_dot_scalar tests ────────────────────────────────────────

    #[test]
    fn test_scalar_dot_all_ones() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1, 1, 1, 1];
        assert_approx_eq(ternary_dot_scalar(&a, &w), 10.0, "all +1");
    }

    #[test]
    fn test_scalar_dot_all_neg() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![-1, -1, -1, -1];
        assert_approx_eq(ternary_dot_scalar(&a, &w), -10.0, "all -1");
    }

    #[test]
    fn test_scalar_dot_all_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![0, 0, 0, 0];
        assert_approx_eq(ternary_dot_scalar(&a, &w), 0.0, "all 0");
    }

    #[test]
    fn test_scalar_dot_mixed() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1, -1, 0, 1]; // 1 - 2 + 0 + 4 = 3
        assert_approx_eq(ternary_dot_scalar(&a, &w), 3.0, "mixed");
    }

    #[test]
    fn test_scalar_dot_empty() {
        assert_approx_eq(ternary_dot_scalar(&[], &[]), 0.0, "empty");
    }

    // ── NEON ternary dot tests ──────────────────────────────────────────

    #[test]
    fn test_neon_dot_matches_scalar() {
        let a: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        let w: Vec<i8> = (0..16)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();

        let expected = ternary_dot_scalar(&a, &w);
        let got = unsafe { ternary_dot_neon(&a, &w) };
        assert_approx_eq(got, expected, "neon vs scalar 16-elem");
    }

    #[test]
    fn test_neon_dot_non_aligned_length() {
        // Length not a multiple of 4 to exercise the scalar tail
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let w = vec![1, -1, 1, 0, -1, 1, -1]; // 1-2+3+0-5+6-7 = -4
        let expected = ternary_dot_scalar(&a, &w);
        let got = unsafe { ternary_dot_neon(&a, &w) };
        assert_approx_eq(got, expected, "neon 7-elem tail");
    }

    #[test]
    fn test_neon_dot_single_element() {
        let a = vec![42.0];
        let w = vec![-1];
        let got = unsafe { ternary_dot_neon(&a, &w) };
        assert_approx_eq(got, -42.0, "neon single");
    }

    // ── quantized_attention_scores_neon tests ───────────────────────────

    #[test]
    fn test_attention_scores_identity_weights() {
        // With all weights = +1, score = q · k / sqrt(d)
        let head_dim = 4;
        let num_heads = 2;
        let q = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let k = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let w = vec![1i8; 8];

        let scores = quantized_attention_scores_neon(&q, &k, &w, head_dim, num_heads);
        let scale = 1.0 / (head_dim as f32).sqrt(); // 0.5

        assert_eq!(scores.len(), num_heads);
        // Head 0: q·k = 1+0+0+1 = 2  →  2 * 0.5 = 1.0
        assert_approx_eq(scores[0], 2.0 * scale, "head0 identity");
        // Head 1: q·k = 0+1+1+0 = 2  →  2 * 0.5 = 1.0
        assert_approx_eq(scores[1], 2.0 * scale, "head1 identity");
    }

    #[test]
    fn test_attention_scores_negating_weights() {
        let head_dim = 4;
        let num_heads = 1;
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![1.0, 1.0, 1.0, 1.0];
        let w = vec![-1i8; 4];

        let scores = quantized_attention_scores_neon(&q, &k, &w, head_dim, num_heads);
        let scale = 1.0 / (head_dim as f32).sqrt();
        // q · (w ⊙ k) = 1*(-1) + 2*(-1) + 3*(-1) + 4*(-1) = -10
        assert_approx_eq(scores[0], -10.0 * scale, "negating");
    }

    #[test]
    fn test_attention_scores_zero_weights() {
        let head_dim = 4;
        let num_heads = 1;
        let q = vec![5.0, 6.0, 7.0, 8.0];
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![0i8; 4];

        let scores = quantized_attention_scores_neon(&q, &k, &w, head_dim, num_heads);
        assert_approx_eq(scores[0], 0.0, "zero weights");
    }

    #[test]
    fn test_attention_scores_mixed_weights() {
        let head_dim = 4;
        let num_heads = 1;
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![4.0, 3.0, 2.0, 1.0];
        let w: Vec<i8> = vec![1, 0, -1, 1];

        let scores = quantized_attention_scores_neon(&q, &k, &w, head_dim, num_heads);
        let scale = 1.0 / (head_dim as f32).sqrt();
        // q · (w ⊙ k) = 1*4 + 2*0 + 3*(-2) + 4*1 = 4 + 0 - 6 + 4 = 2
        assert_approx_eq(scores[0], 2.0 * scale, "mixed weights");
    }

    #[test]
    fn test_attention_scores_large_head_dim() {
        let head_dim = 64;
        let num_heads = 2;
        let total = head_dim * num_heads;
        let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        let k: Vec<f32> = (0..total).map(|i| ((total - i) as f32) * 0.01).collect();
        let w: Vec<i8> = (0..total)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();

        let scores = quantized_attention_scores_neon(&q, &k, &w, head_dim, num_heads);
        assert_eq!(scores.len(), num_heads);

        // Verify against scalar reference
        let scale = 1.0 / (head_dim as f32).sqrt();
        for h in 0..num_heads {
            let s = h * head_dim;
            let e = s + head_dim;
            let expected: f32 = q[s..e]
                .iter()
                .zip(k[s..e].iter())
                .zip(w[s..e].iter())
                .map(|((&qi, &ki), &wi)| qi * ki * (wi as f32))
                .sum::<f32>()
                * scale;
            assert_approx_eq(scores[h], expected, &format!("large head {h}"));
        }
    }

    // ── quantized_multi_head_attention_neon tests ───────────────────────

    #[test]
    fn test_mha_output_shape() {
        let head_dim = 4;
        let num_heads = 3;
        let total = head_dim * num_heads;
        let q = vec![1.0; total];
        let k = vec![1.0; total];
        let v = vec![1.0; total];
        let w = vec![1i8; total];

        let out = quantized_multi_head_attention_neon(&q, &k, &v, &w, head_dim, num_heads);
        assert_eq!(out.len(), total, "output length");
    }

    #[test]
    fn test_mha_output_equals_value() {
        // Single-key softmax is 1.0, so output == V
        let head_dim = 4;
        let num_heads = 2;
        let total = head_dim * num_heads;
        let q = vec![1.0; total];
        let k = vec![1.0; total];
        let v: Vec<f32> = (0..total).map(|i| (i + 1) as f32).collect();
        let w = vec![1i8; total];

        let out = quantized_multi_head_attention_neon(&q, &k, &v, &w, head_dim, num_heads);
        for (i, (&got, &expected)) in out.iter().zip(v.iter()).enumerate() {
            assert_approx_eq(got, expected, &format!("mha output[{i}]"));
        }
    }

    #[test]
    fn test_mha_large_head_dim() {
        let head_dim = 128;
        let num_heads = 4;
        let total = head_dim * num_heads;
        let q = vec![0.5; total];
        let k = vec![0.5; total];
        let v: Vec<f32> = (0..total).map(|i| (i as f32) * 0.001).collect();
        let w: Vec<i8> = (0..total).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();

        let out = quantized_multi_head_attention_neon(&q, &k, &v, &w, head_dim, num_heads);
        assert_eq!(out.len(), total);
        // Verify output == V (single-key softmax = 1.0)
        for i in 0..total {
            assert_approx_eq(out[i], v[i], &format!("mha large[{i}]"));
        }
    }

    // ── Panic / edge-case tests ─────────────────────────────────────────

    #[test]
    #[should_panic(expected = "query length mismatch")]
    fn test_scores_panics_on_query_mismatch() {
        quantized_attention_scores_neon(&[1.0], &[1.0, 2.0], &[1, 1], 2, 1);
    }

    #[test]
    #[should_panic(expected = "key length mismatch")]
    fn test_scores_panics_on_key_mismatch() {
        quantized_attention_scores_neon(&[1.0, 2.0], &[1.0], &[1, 1], 2, 1);
    }

    #[test]
    #[should_panic(expected = "weights length mismatch")]
    fn test_scores_panics_on_weights_mismatch() {
        quantized_attention_scores_neon(&[1.0, 2.0], &[1.0, 2.0], &[1], 2, 1);
    }

    #[test]
    #[should_panic(expected = "v length mismatch")]
    fn test_mha_panics_on_v_mismatch() {
        quantized_multi_head_attention_neon(&[1.0, 2.0], &[1.0, 2.0], &[1.0], &[1, 1], 2, 1);
    }

    #[test]
    fn test_scores_single_head_single_dim() {
        // Minimal case: 1 head, dim=1
        // Cannot use head_dim=1 with NEON chunks (pure scalar tail)
        let q = vec![3.0];
        let k = vec![4.0];
        let w = vec![1i8];
        let scores = quantized_attention_scores_neon(&q, &k, &w, 1, 1);
        // 3 * 4 * 1 / sqrt(1) = 12
        assert_approx_eq(scores[0], 12.0, "1x1 score");
    }

    #[test]
    fn test_neon_and_scalar_consistency_sweep() {
        // Sweep various lengths to ensure NEON and scalar agree
        for len in 1..=33 {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.7 - 5.0).collect();
            let w: Vec<i8> = (0..len)
                .map(|i| match i % 5 {
                    0 | 1 => 1,
                    2 | 3 => -1,
                    _ => 0,
                })
                .collect();
            let expected = ternary_dot_scalar(&a, &w);
            let got = unsafe { ternary_dot_neon(&a, &w) };
            assert!((got - expected).abs() < EPS, "length {len}: expected {expected}, got {got}");
        }
    }
}
