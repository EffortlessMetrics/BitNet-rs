#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]
//! AVX2 vs scalar parity tests for CPU attention kernels.
//!
//! Tests exercise the runtime AVX2 dispatch in `AttentionKernel::*` and
//! `quantized_dot_product_attention` against pure-scalar references to
//! verify numerical equivalence within defined tolerances.

use bitnet_kernels::cpu::attention::{AttentionConfig, AttentionKernel, GqaConfig, causal_mask};
use bitnet_kernels::cpu::quantized_attention::{
    QuantBits, QuantizedAttentionConfig, quantized_dot_product_attention,
};

#[path = "common/avx2_i8_parity.rs"]
mod avx2_i8_parity;
#[path = "common/avx2_parity.rs"]
mod avx2_parity;
use avx2_i8_parity::pseudo_rand_i8;
use avx2_parity::{assert_vec_parity, pseudo_rand};

// ── Tolerance constants ────────────────────────────────────────────────

/// Absolute tolerance for f32 attention kernels.
const ATTN_ABS_TOL: f32 = 1e-5;
/// Relative tolerance for f32 attention kernels.
const ATTN_REL_TOL: f32 = 1e-4;

/// Absolute tolerance for quantized (i8) attention kernels.
const QUANT_ATTN_ABS_TOL: f32 = 1e-4;
/// Relative tolerance for quantized (i8) attention kernels.
const QUANT_ATTN_REL_TOL: f32 = 1e-3;

// ── Scalar reference implementations ───────────────────────────────────

/// Pure scalar dot product — no SIMD dispatch.
fn reference_scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Numerically-stable softmax over a mutable row.
fn reference_softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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

/// Pure-scalar scaled dot-product attention for a single head.
///
/// Q: `[seq_q, head_dim]`, K: `[seq_k, head_dim]`, V: `[seq_k, head_dim]`
/// mask: optional `[seq_q, seq_k]` additive mask.
/// Returns `[seq_q, head_dim]`.
fn reference_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    scale: f32,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
) -> Vec<f32> {
    // Q · K^T → [seq_q, seq_k]
    let mut scores = vec![0.0_f32; seq_q * seq_k];
    for i in 0..seq_q {
        for j in 0..seq_k {
            scores[i * seq_k + j] = reference_scalar_dot(
                &q[i * head_dim..(i + 1) * head_dim],
                &k[j * head_dim..(j + 1) * head_dim],
            );
        }
    }

    // Scale
    for s in &mut scores {
        *s *= scale;
    }

    // Additive mask
    if let Some(m) = mask {
        for (s, &mv) in scores.iter_mut().zip(m.iter()) {
            *s += mv;
        }
    }

    // Row-wise softmax
    for i in 0..seq_q {
        reference_softmax_row(&mut scores[i * seq_k..(i + 1) * seq_k]);
    }

    // scores · V → [seq_q, head_dim]
    let mut out = vec![0.0_f32; seq_q * head_dim];
    for i in 0..seq_q {
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for j in 0..seq_k {
                acc += scores[i * seq_k + j] * v[j * head_dim + d];
            }
            out[i * head_dim + d] = acc;
        }
    }
    out
}

/// Extract head `h` from interleaved `[seq_len, num_heads * head_dim]`
/// layout into contiguous `[seq_len, head_dim]`.
fn ref_extract_head(
    data: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    h: usize,
) -> Vec<f32> {
    let model_dim = num_heads * head_dim;
    let mut out = vec![0.0_f32; seq_len * head_dim];
    for t in 0..seq_len {
        for d in 0..head_dim {
            out[t * head_dim + d] = data[t * model_dim + h * head_dim + d];
        }
    }
    out
}

/// Scatter head `h` from contiguous `[seq_len, head_dim]` back into
/// interleaved `[seq_len, num_heads * head_dim]`.
fn ref_scatter_head(
    output: &mut [f32],
    head_out: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    h: usize,
) {
    let model_dim = num_heads * head_dim;
    for t in 0..seq_len {
        for d in 0..head_dim {
            output[t * model_dim + h * head_dim + d] = head_out[t * head_dim + d];
        }
    }
}

/// Pure-scalar multi-head attention reference.
fn reference_mha(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mask_vec = if causal { Some(causal_mask(seq_len)) } else { None };
    let mask_ref = mask_vec.as_deref();

    let model_dim = num_heads * head_dim;
    let mut output = vec![0.0_f32; seq_len * model_dim];

    for h in 0..num_heads {
        let q_head = ref_extract_head(q, seq_len, num_heads, head_dim, h);
        let k_head = ref_extract_head(k, seq_len, num_heads, head_dim, h);
        let v_head = ref_extract_head(v, seq_len, num_heads, head_dim, h);

        let head_out =
            reference_sdpa(&q_head, &k_head, &v_head, mask_ref, scale, seq_len, seq_len, head_dim);

        ref_scatter_head(&mut output, &head_out, seq_len, num_heads, head_dim, h);
    }
    output
}

/// Pure-scalar GQA reference.
fn reference_gqa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mask_vec = if causal { Some(causal_mask(seq_len)) } else { None };
    let mask_ref = mask_vec.as_deref();
    let group_size = num_q_heads / num_kv_heads;

    let q_dim = num_q_heads * head_dim;
    let mut output = vec![0.0_f32; seq_len * q_dim];

    for kv_h in 0..num_kv_heads {
        let k_head = ref_extract_head(k, seq_len, num_kv_heads, head_dim, kv_h);
        let v_head = ref_extract_head(v, seq_len, num_kv_heads, head_dim, kv_h);

        for g in 0..group_size {
            let q_idx = kv_h * group_size + g;
            let q_head = ref_extract_head(q, seq_len, num_q_heads, head_dim, q_idx);

            let head_out = reference_sdpa(
                &q_head, &k_head, &v_head, mask_ref, scale, seq_len, seq_len, head_dim,
            );

            ref_scatter_head(&mut output, &head_out, seq_len, num_q_heads, head_dim, q_idx);
        }
    }
    output
}

/// Pure-scalar quantized scaled dot-product attention reference.
///
/// Q, K, V are `i8` tensors of shape `[seq_len, head_dim]` with per-tensor
/// scales.  Returns f32 output `[seq_len, head_dim]`.
#[allow(clippy::too_many_arguments)]
fn reference_quant_sdpa(
    q_i8: &[i8],
    k_i8: &[i8],
    v_i8: &[i8],
    q_scale: f32,
    k_scale: f32,
    v_scale: f32,
    scale_factor: f32,
    seq_len: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let combined_scale = scale_factor * q_scale * k_scale;

    // Compute scores: scores[i][j] = combined_scale * dot_i8(Q[i], K[j])
    let mut scores = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if causal && j > i {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0i32;
                for d in 0..head_dim {
                    dot += q_i8[i * head_dim + d] as i32 * k_i8[j * head_dim + d] as i32;
                }
                scores[i * seq_len + j] = dot as f32 * combined_scale;
            }
        }
        reference_softmax_row(&mut scores[i * seq_len..(i + 1) * seq_len]);
    }

    // Weighted sum over V (dequantized): output[i][d] = sum_j scores[i][j] * v[j][d] * v_scale
    let mut output = vec![0.0_f32; seq_len * head_dim];
    for i in 0..seq_len {
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for j in 0..seq_len {
                acc += scores[i * seq_len + j] * v_i8[j * head_dim + d] as f32 * v_scale;
            }
            output[i * head_dim + d] = acc;
        }
    }
    output
}

// ════════════════════════════════════════════════════════════════════════
// 1. SCALED DOT-PRODUCT ATTENTION (f32)
// ════════════════════════════════════════════════════════════════════════

#[test]
fn sdpa_parity_small() {
    let seq_len = 4;
    let head_dim = 8;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = pseudo_rand(seq_len * head_dim, 100);
    let k = pseudo_rand(seq_len * head_dim, 200);
    let v = pseudo_rand(seq_len * head_dim, 300);

    let expected = reference_sdpa(&q, &k, &v, None, scale, seq_len, seq_len, head_dim);
    let actual =
        AttentionKernel::scaled_dot_product(&q, &k, &v, None, scale, seq_len, seq_len, head_dim)
            .expect("sdpa should not fail");

    assert_vec_parity(&actual, &expected, ATTN_ABS_TOL, ATTN_REL_TOL, "sdpa_small");
}

#[test]
fn sdpa_parity_medium() {
    let seq_len = 8;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = pseudo_rand(seq_len * head_dim, 400);
    let k = pseudo_rand(seq_len * head_dim, 500);
    let v = pseudo_rand(seq_len * head_dim, 600);

    let expected = reference_sdpa(&q, &k, &v, None, scale, seq_len, seq_len, head_dim);
    let actual =
        AttentionKernel::scaled_dot_product(&q, &k, &v, None, scale, seq_len, seq_len, head_dim)
            .expect("sdpa should not fail");

    assert_vec_parity(&actual, &expected, ATTN_ABS_TOL, ATTN_REL_TOL, "sdpa_medium");
}

#[test]
fn sdpa_parity_non_aligned() {
    let seq_len = 5;
    let head_dim = 13;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = pseudo_rand(seq_len * head_dim, 700);
    let k = pseudo_rand(seq_len * head_dim, 800);
    let v = pseudo_rand(seq_len * head_dim, 900);

    let expected = reference_sdpa(&q, &k, &v, None, scale, seq_len, seq_len, head_dim);
    let actual =
        AttentionKernel::scaled_dot_product(&q, &k, &v, None, scale, seq_len, seq_len, head_dim)
            .expect("sdpa should not fail");

    assert_vec_parity(&actual, &expected, ATTN_ABS_TOL, ATTN_REL_TOL, "sdpa_non_aligned");
}

#[test]
fn sdpa_parity_with_causal_mask() {
    let seq_len = 8;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = pseudo_rand(seq_len * head_dim, 1000);
    let k = pseudo_rand(seq_len * head_dim, 1100);
    let v = pseudo_rand(seq_len * head_dim, 1200);
    let mask = causal_mask(seq_len);

    let expected = reference_sdpa(&q, &k, &v, Some(&mask), scale, seq_len, seq_len, head_dim);
    let actual = AttentionKernel::scaled_dot_product(
        &q,
        &k,
        &v,
        Some(&mask),
        scale,
        seq_len,
        seq_len,
        head_dim,
    )
    .expect("sdpa with causal mask should not fail");

    assert_vec_parity(&actual, &expected, ATTN_ABS_TOL, ATTN_REL_TOL, "sdpa_causal");
}

#[test]
fn sdpa_parity_single_token() {
    let seq_q = 1;
    let seq_k = 8;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = pseudo_rand(seq_q * head_dim, 1300);
    let k = pseudo_rand(seq_k * head_dim, 1400);
    let v = pseudo_rand(seq_k * head_dim, 1500);

    let expected = reference_sdpa(&q, &k, &v, None, scale, seq_q, seq_k, head_dim);
    let actual =
        AttentionKernel::scaled_dot_product(&q, &k, &v, None, scale, seq_q, seq_k, head_dim)
            .expect("sdpa single-token should not fail");

    assert_vec_parity(&actual, &expected, ATTN_ABS_TOL, ATTN_REL_TOL, "sdpa_single_token");
}

// ════════════════════════════════════════════════════════════════════════
// 2. MULTI-HEAD ATTENTION (f32)
// ════════════════════════════════════════════════════════════════════════

#[test]
fn mha_parity_basic() {
    let num_heads = 4;
    let seq_len = 8;
    let head_dim = 16;
    let model_dim = num_heads * head_dim;

    let q = pseudo_rand(seq_len * model_dim, 2000);
    let k = pseudo_rand(seq_len * model_dim, 2100);
    let v = pseudo_rand(seq_len * model_dim, 2200);

    let expected = reference_mha(&q, &k, &v, num_heads, head_dim, seq_len, false);

    let cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    let actual =
        AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).expect("mha should not fail");

    assert_vec_parity(&actual, &expected, ATTN_ABS_TOL, ATTN_REL_TOL, "mha_basic");
}

#[test]
fn mha_parity_causal() {
    let num_heads = 2;
    let seq_len = 6;
    let head_dim = 32;
    let model_dim = num_heads * head_dim;

    let q = pseudo_rand(seq_len * model_dim, 2300);
    let k = pseudo_rand(seq_len * model_dim, 2400);
    let v = pseudo_rand(seq_len * model_dim, 2500);

    let expected = reference_mha(&q, &k, &v, num_heads, head_dim, seq_len, true);

    let cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    let actual = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg)
        .expect("mha causal should not fail");

    assert_vec_parity(&actual, &expected, ATTN_ABS_TOL, ATTN_REL_TOL, "mha_causal");
}

#[test]
fn mha_parity_non_aligned() {
    let num_heads = 3;
    let seq_len = 5;
    let head_dim = 13;
    let model_dim = num_heads * head_dim;

    let q = pseudo_rand(seq_len * model_dim, 2600);
    let k = pseudo_rand(seq_len * model_dim, 2700);
    let v = pseudo_rand(seq_len * model_dim, 2800);

    let expected = reference_mha(&q, &k, &v, num_heads, head_dim, seq_len, false);

    let cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    let actual = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg)
        .expect("mha non-aligned should not fail");

    assert_vec_parity(&actual, &expected, ATTN_ABS_TOL, ATTN_REL_TOL, "mha_non_aligned");
}

// ════════════════════════════════════════════════════════════════════════
// 3. GROUPED-QUERY ATTENTION (f32)
// ════════════════════════════════════════════════════════════════════════

#[test]
fn gqa_parity_basic() {
    let num_q_heads = 4;
    let num_kv_heads = 2;
    let seq_len = 8;
    let head_dim = 16;

    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q = pseudo_rand(seq_len * q_dim, 3000);
    let k = pseudo_rand(seq_len * kv_dim, 3100);
    let v = pseudo_rand(seq_len * kv_dim, 3200);

    let expected = reference_gqa(&q, &k, &v, num_q_heads, num_kv_heads, head_dim, seq_len, false);

    let cfg =
        GqaConfig { num_q_heads, num_kv_heads, head_dim, seq_len, causal: false, scale: None };
    let actual =
        AttentionKernel::grouped_query_attention(&q, &k, &v, &cfg).expect("gqa should not fail");

    assert_vec_parity(&actual, &expected, ATTN_ABS_TOL, ATTN_REL_TOL, "gqa_basic");
}

#[test]
fn gqa_parity_causal() {
    let num_q_heads = 4;
    let num_kv_heads = 2;
    let seq_len = 8;
    let head_dim = 16;

    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q = pseudo_rand(seq_len * q_dim, 3300);
    let k = pseudo_rand(seq_len * kv_dim, 3400);
    let v = pseudo_rand(seq_len * kv_dim, 3500);

    let expected = reference_gqa(&q, &k, &v, num_q_heads, num_kv_heads, head_dim, seq_len, true);

    let cfg = GqaConfig { num_q_heads, num_kv_heads, head_dim, seq_len, causal: true, scale: None };
    let actual = AttentionKernel::grouped_query_attention(&q, &k, &v, &cfg)
        .expect("gqa causal should not fail");

    assert_vec_parity(&actual, &expected, ATTN_ABS_TOL, ATTN_REL_TOL, "gqa_causal");
}

// ════════════════════════════════════════════════════════════════════════
// 4. QUANTIZED SCALED DOT-PRODUCT ATTENTION (i8)
// ════════════════════════════════════════════════════════════════════════

#[test]
fn quant_sdpa_parity_small() {
    let seq_len = 4;
    let head_dim = 32;
    let q_scale = 0.05_f32;
    let k_scale = 0.05_f32;
    let v_scale = 0.05_f32;
    let scale_factor = 1.0 / (head_dim as f32).sqrt();

    let q = pseudo_rand_i8(seq_len * head_dim, 4000);
    let k = pseudo_rand_i8(seq_len * head_dim, 4100);
    let v = pseudo_rand_i8(seq_len * head_dim, 4200);

    let expected = reference_quant_sdpa(
        &q,
        &k,
        &v,
        q_scale,
        k_scale,
        v_scale,
        scale_factor,
        seq_len,
        head_dim,
        false,
    );

    let cfg = QuantizedAttentionConfig {
        num_heads: 1,
        num_kv_heads: 1,
        head_dim,
        seq_len,
        causal: false,
        quant_bits: QuantBits::Int8,
        scale: None,
    };
    let mut actual = vec![0.0_f32; seq_len * head_dim];
    quantized_dot_product_attention(&cfg, &q, &k, &v, q_scale, k_scale, v_scale, &mut actual)
        .expect("quant_sdpa should not fail");

    assert_vec_parity(
        &actual,
        &expected,
        QUANT_ATTN_ABS_TOL,
        QUANT_ATTN_REL_TOL,
        "quant_sdpa_small",
    );
}

#[test]
fn quant_sdpa_parity_medium() {
    let seq_len = 8;
    let head_dim = 64;
    let q_scale = 0.04_f32;
    let k_scale = 0.04_f32;
    let v_scale = 0.04_f32;
    let scale_factor = 1.0 / (head_dim as f32).sqrt();

    let q = pseudo_rand_i8(seq_len * head_dim, 4300);
    let k = pseudo_rand_i8(seq_len * head_dim, 4400);
    let v = pseudo_rand_i8(seq_len * head_dim, 4500);

    let expected = reference_quant_sdpa(
        &q,
        &k,
        &v,
        q_scale,
        k_scale,
        v_scale,
        scale_factor,
        seq_len,
        head_dim,
        false,
    );

    let cfg = QuantizedAttentionConfig {
        num_heads: 1,
        num_kv_heads: 1,
        head_dim,
        seq_len,
        causal: false,
        quant_bits: QuantBits::Int8,
        scale: None,
    };
    let mut actual = vec![0.0_f32; seq_len * head_dim];
    quantized_dot_product_attention(&cfg, &q, &k, &v, q_scale, k_scale, v_scale, &mut actual)
        .expect("quant_sdpa should not fail");

    assert_vec_parity(
        &actual,
        &expected,
        QUANT_ATTN_ABS_TOL,
        QUANT_ATTN_REL_TOL,
        "quant_sdpa_medium",
    );
}

#[test]
fn quant_sdpa_parity_non_aligned() {
    let seq_len = 5;
    let head_dim = 33;
    let q_scale = 0.06_f32;
    let k_scale = 0.06_f32;
    let v_scale = 0.06_f32;
    let scale_factor = 1.0 / (head_dim as f32).sqrt();

    let q = pseudo_rand_i8(seq_len * head_dim, 4600);
    let k = pseudo_rand_i8(seq_len * head_dim, 4700);
    let v = pseudo_rand_i8(seq_len * head_dim, 4800);

    let expected = reference_quant_sdpa(
        &q,
        &k,
        &v,
        q_scale,
        k_scale,
        v_scale,
        scale_factor,
        seq_len,
        head_dim,
        false,
    );

    let cfg = QuantizedAttentionConfig {
        num_heads: 1,
        num_kv_heads: 1,
        head_dim,
        seq_len,
        causal: false,
        quant_bits: QuantBits::Int8,
        scale: None,
    };
    let mut actual = vec![0.0_f32; seq_len * head_dim];
    quantized_dot_product_attention(&cfg, &q, &k, &v, q_scale, k_scale, v_scale, &mut actual)
        .expect("quant_sdpa non-aligned should not fail");

    assert_vec_parity(
        &actual,
        &expected,
        QUANT_ATTN_ABS_TOL,
        QUANT_ATTN_REL_TOL,
        "quant_sdpa_non_aligned",
    );
}
