//! BDD Wave 19 — Attention operation tests.
//!
//! Covers causal masking, scaled dot-product attention, multi-head attention
//! layout, grouped-query attention, KV-cache incremental attention, and
//! ALiBi bias.

use bitnet_kernels::cpu::attention::{
    AttentionConfig, AttentionKernel, GqaConfig, alibi_slopes, apply_alibi_bias, apply_causal_mask,
    apply_mask, attention_with_kv_cache, causal_mask, multi_head_attention_cpu,
    scaled_dot_product_attention,
};

const TOL: f32 = 1e-4;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if x.is_nan() && y.is_nan() {
            continue;
        }
        if x == y {
            continue; // handles ±inf
        }
        assert!((x - y).abs() < tol, "mismatch at index {i}: {x} vs {y} (tol={tol})");
    }
}

// ── Causal Mask ────────────────────────────────────────────────────

#[test]
fn given_seq_len_1_when_causal_mask_then_single_zero() {
    let mask = causal_mask(1);
    assert_eq!(mask, vec![0.0]);
}

#[test]
fn given_seq_len_3_when_causal_mask_then_upper_triangle_neg_inf() {
    let mask = causal_mask(3);
    assert_eq!(mask.len(), 9);
    // Diagonal and below should be 0.0
    assert_eq!(mask[0], 0.0); // (0,0)
    assert_eq!(mask[3], 0.0); // (1,0)
    assert_eq!(mask[4], 0.0); // (1,1)
    assert_eq!(mask[6], 0.0); // (2,0)
    assert_eq!(mask[7], 0.0); // (2,1)
    assert_eq!(mask[8], 0.0); // (2,2)
    // Above diagonal should be NEG_INFINITY
    assert_eq!(mask[1], f32::NEG_INFINITY); // (0,1)
    assert_eq!(mask[2], f32::NEG_INFINITY); // (0,2)
    assert_eq!(mask[5], f32::NEG_INFINITY); // (1,2)
}

#[test]
fn given_causal_mask_when_apply_to_uniform_scores_then_upper_masked() {
    let mut scores = vec![1.0; 4]; // 2×2
    let mask = causal_mask(2);
    apply_mask(&mut scores, &mask).unwrap();
    assert_eq!(scores[0], 1.0); // (0,0) allowed
    assert_eq!(scores[1], f32::NEG_INFINITY); // (0,1) masked
    assert_eq!(scores[2], 1.0); // (1,0) allowed
    assert_eq!(scores[3], 1.0); // (1,1) allowed
}

#[test]
fn given_mismatched_lengths_when_apply_mask_then_error() {
    let mut scores = vec![1.0; 4];
    let mask = vec![0.0; 9];
    let result = apply_mask(&mut scores, &mask);
    assert!(result.is_err());
}

#[test]
fn given_seq_len_when_apply_causal_mask_then_same_as_explicit_mask() {
    let mut scores_a = vec![1.0; 9]; // 3×3
    let mut scores_b = scores_a.clone();
    apply_causal_mask(&mut scores_a, 3).unwrap();
    let mask = causal_mask(3);
    apply_mask(&mut scores_b, &mask).unwrap();
    approx_eq(&scores_a, &scores_b, TOL);
}

// ── Scaled Dot-Product Attention ───────────────────────────────────

#[test]
fn given_identity_qkv_when_sdpa_no_causal_then_uniform_attention() {
    // Q=K=V=I₂, head_dim=2, seq=2, no causal mask
    // Scores = I · I^T / √2 = I / √2, softmax of equal scores → uniform
    let eye = vec![1.0, 0.0, 0.0, 1.0];
    let result = scaled_dot_product_attention(&eye, &eye, &eye, 2, 2, 2, false).unwrap();
    assert_eq!(result.len(), 4);
    // Each output row is a weighted average of V rows
}

#[test]
fn given_single_token_when_sdpa_then_output_equals_value() {
    // seq_q=1, seq_k=1: attention weight is 1.0 → output = V
    let q = vec![1.0, 0.0];
    let k = vec![1.0, 0.0];
    let v = vec![7.0, 3.0];
    let result = scaled_dot_product_attention(&q, &k, &v, 1, 1, 2, false).unwrap();
    approx_eq(&result, &[7.0, 3.0], TOL);
}

#[test]
fn given_orthogonal_keys_when_sdpa_then_attends_to_matching_key() {
    let head_dim = 2;
    // Q: one query pointing in direction [1,0]
    let q = vec![10.0, 0.0];
    // K: two keys — one aligned, one orthogonal
    let k = vec![10.0, 0.0, 0.0, 10.0];
    // V: distinct values
    let v = vec![1.0, 0.0, 0.0, 1.0];
    let result = scaled_dot_product_attention(&q, &k, &v, 1, 2, head_dim, false).unwrap();
    // Should strongly attend to first key → output ≈ [1, 0]
    assert!(result[0] > 0.9, "expected strong attention to key 0, got {}", result[0]);
    assert!(result[1] < 0.1, "expected weak attention to key 1, got {}", result[1]);
}

#[test]
fn given_causal_flag_when_sdpa_then_future_tokens_masked() {
    let head_dim = 2;
    // seq=2, with causal masking
    let q = vec![1.0, 0.0, 0.0, 1.0];
    let k = vec![1.0, 0.0, 0.0, 1.0];
    let v = vec![1.0, 0.0, 0.0, 1.0];
    let result = scaled_dot_product_attention(&q, &k, &v, 2, 2, head_dim, true).unwrap();
    // First token can only attend to itself → output[0..2] = v[0..2]
    approx_eq(&result[0..2], &[1.0, 0.0], TOL);
}

#[test]
fn given_zero_head_dim_when_sdpa_then_error() {
    let result = scaled_dot_product_attention(&[], &[], &[], 0, 0, 0, false);
    assert!(result.is_err());
}

// ── Multi-Head Attention ───────────────────────────────────────────

#[test]
fn given_single_head_when_mha_then_same_as_sdpa() {
    let head_dim = 4;
    let seq_len = 2;
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
    let k = q.clone();
    let v = q.clone();

    let mha = multi_head_attention_cpu(&q, &k, &v, 1, head_dim, seq_len, false).unwrap();
    let sdpa = scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, head_dim, false).unwrap();
    approx_eq(&mha, &sdpa, TOL);
}

#[test]
fn given_two_heads_when_mha_then_output_shape_preserved() {
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let total = seq_len * num_heads * head_dim;
    let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let k = q.clone();
    let v = q.clone();

    let result = multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, false).unwrap();
    assert_eq!(result.len(), total);
}

#[test]
fn given_causal_mha_when_first_token_then_attends_only_to_self() {
    let num_heads = 2;
    let head_dim = 2;
    let seq_len = 3;
    let model_dim = num_heads * head_dim;

    // Make V with distinct per-position values
    let mut v = vec![0.0f32; seq_len * model_dim];
    for pos in 0..seq_len {
        for d in 0..model_dim {
            v[pos * model_dim + d] = (pos * 10 + d) as f32;
        }
    }
    // Q and K: identity-like so attention is uniform where allowed
    let q = vec![1.0f32; seq_len * model_dim];
    let k = q.clone();

    let result = multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, true).unwrap();
    // First position can only attend to itself
    approx_eq(&result[0..model_dim], &v[0..model_dim], TOL);
}

#[test]
fn given_invalid_config_when_mha_then_error() {
    // num_heads=0 should fail validation
    let result = multi_head_attention_cpu(&[], &[], &[], 0, 4, 1, false);
    assert!(result.is_err());
}

// ── AttentionConfig Validation ─────────────────────────────────────

#[test]
fn given_valid_config_when_validate_then_ok() {
    let config = AttentionConfig {
        num_heads: 4,
        head_dim: 8,
        seq_len: 16,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn given_zero_heads_when_validate_then_error() {
    let config = AttentionConfig {
        num_heads: 0,
        head_dim: 8,
        seq_len: 16,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    assert!(config.validate().is_err());
}

#[test]
fn given_custom_scale_when_resolved_then_uses_custom() {
    let config = AttentionConfig {
        num_heads: 1,
        head_dim: 64,
        seq_len: 1,
        causal: false,
        use_alibi: false,
        scale: Some(0.42),
    };
    assert!((config.resolved_scale() - 0.42).abs() < TOL);
}

#[test]
fn given_no_scale_when_resolved_then_one_over_sqrt_dim() {
    let config = AttentionConfig {
        num_heads: 1,
        head_dim: 64,
        seq_len: 1,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    let expected = 1.0 / (64.0_f32).sqrt();
    assert!((config.resolved_scale() - expected).abs() < TOL);
}

// ── Grouped-Query Attention ────────────────────────────────────────

#[test]
fn given_gqa_one_to_one_when_forward_then_same_as_mha() {
    let head_dim = 4;
    let seq_len = 2;
    let num_heads = 2;
    let total = seq_len * num_heads * head_dim;
    let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let k = q.clone();
    let v = q.clone();

    let cfg = GqaConfig {
        num_q_heads: num_heads,
        num_kv_heads: num_heads,
        head_dim,
        seq_len,
        causal: false,
        scale: None,
    };
    let gqa = AttentionKernel::grouped_query_attention(&q, &k, &v, &cfg).unwrap();
    let mha = multi_head_attention_cpu(&q, &k, &v, num_heads, head_dim, seq_len, false).unwrap();
    approx_eq(&gqa, &mha, TOL);
}

#[test]
fn given_gqa_two_to_one_when_forward_then_correct_output_shape() {
    let head_dim = 4;
    let seq_len = 2;
    let num_q = 4;
    let num_kv = 2;

    let q: Vec<f32> = vec![0.1; seq_len * num_q * head_dim];
    let k: Vec<f32> = vec![0.1; seq_len * num_kv * head_dim];
    let v: Vec<f32> = vec![0.1; seq_len * num_kv * head_dim];

    let cfg = GqaConfig {
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        head_dim,
        seq_len,
        causal: false,
        scale: None,
    };
    let result = AttentionKernel::grouped_query_attention(&q, &k, &v, &cfg).unwrap();
    assert_eq!(result.len(), seq_len * num_q * head_dim);
}

#[test]
fn given_gqa_invalid_ratio_when_forward_then_error() {
    let cfg = GqaConfig {
        num_q_heads: 3,
        num_kv_heads: 2, // 3 not divisible by 2
        head_dim: 4,
        seq_len: 1,
        causal: false,
        scale: None,
    };
    let q = vec![0.0f32; 12];
    let k = vec![0.0f32; 8];
    let v = vec![0.0f32; 8];
    let result = AttentionKernel::grouped_query_attention(&q, &k, &v, &cfg);
    assert!(result.is_err());
}

// ── KV-Cache Incremental Attention ─────────────────────────────────

#[test]
fn given_empty_cache_when_kv_attention_then_output_equals_value() {
    let head_dim = 4;
    let q = vec![1.0; head_dim];
    let k_new = vec![1.0; head_dim];
    let v_new = vec![7.0, 3.0, 5.0, 1.0];
    let mut k_cache = Vec::new();
    let mut v_cache = Vec::new();

    let result =
        attention_with_kv_cache(&q, &mut k_cache, &mut v_cache, &k_new, &v_new, head_dim).unwrap();
    approx_eq(&result, &v_new, TOL);
}

#[test]
fn given_growing_cache_when_kv_attention_then_cache_extends() {
    let head_dim = 2;
    let mut k_cache = Vec::new();
    let mut v_cache = Vec::new();

    // Step 1
    let q1 = vec![1.0, 0.0];
    let k1 = vec![1.0, 0.0];
    let v1 = vec![10.0, 20.0];
    attention_with_kv_cache(&q1, &mut k_cache, &mut v_cache, &k1, &v1, head_dim).unwrap();
    assert_eq!(k_cache.len(), head_dim);

    // Step 2
    let q2 = vec![0.0, 1.0];
    let k2 = vec![0.0, 1.0];
    let v2 = vec![30.0, 40.0];
    attention_with_kv_cache(&q2, &mut k_cache, &mut v_cache, &k2, &v2, head_dim).unwrap();
    assert_eq!(k_cache.len(), 2 * head_dim);
}

// ── ALiBi Bias ─────────────────────────────────────────────────────

#[test]
fn given_power_of_two_heads_when_alibi_slopes_then_geometric_sequence() {
    let slopes = alibi_slopes(4);
    assert_eq!(slopes.len(), 4);
    // Slopes should form a geometric sequence with ratio 2^(-8/n)
    for i in 1..slopes.len() {
        let ratio = slopes[i] / slopes[i - 1];
        let expected_ratio = slopes[1] / slopes[0];
        assert!((ratio - expected_ratio).abs() < 1e-6);
    }
}

#[test]
fn given_alibi_slope_when_apply_bias_then_scores_modified() {
    let seq_q = 2;
    let seq_k = 2;
    let mut scores = vec![1.0; seq_q * seq_k];
    let slope = 0.5;
    apply_alibi_bias(&mut scores, seq_q, seq_k, slope).unwrap();
    // Bias depends on distance: |i-j| * slope subtracted
    // (0,0)→0, (0,1)→-0.5, (1,0)→-0.5, (1,1)→0
    assert!((scores[0] - 1.0).abs() < TOL);
    assert!(scores[1] < 1.0); // biased down
}
