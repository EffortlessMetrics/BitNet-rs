//! Edge-case tests for CPU attention operations.
//!
//! Tests cover scaled dot-product attention, causal masking,
//! multi-head attention, GQA, and KV cache attention.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::attention::{
    AttentionConfig, AttentionKernel, GqaConfig, apply_causal_mask, apply_mask, causal_mask,
    masked_attention, scaled_dot_product_attention,
};

// ── Causal mask ──────────────────────────────────────────────────────

#[test]
fn causal_mask_size_1() {
    let mask = causal_mask(1);
    assert_eq!(mask.len(), 1);
    assert!((mask[0] - 0.0).abs() < 1e-6); // no masking for single position
}

#[test]
fn causal_mask_size_3() {
    let mask = causal_mask(3);
    assert_eq!(mask.len(), 9); // 3x3
    // Upper triangle should be -inf (or large negative), lower+diag should be 0
    assert!((mask[0] - 0.0).abs() < 1e-6); // (0,0) attend
    assert!(mask[1] < -1e6); // (0,1) masked
    assert!((mask[3] - 0.0).abs() < 1e-6); // (1,0) attend
    assert!((mask[4] - 0.0).abs() < 1e-6); // (1,1) attend
}

// ── apply_mask ───────────────────────────────────────────────────────

#[test]
fn apply_mask_basic() {
    let mut scores = vec![1.0, 2.0, 3.0, 4.0];
    let mask = vec![0.0, 0.0, -1e9, -1e9];
    apply_mask(&mut scores, &mask).unwrap();
    assert!((scores[0] - 1.0).abs() < 1e-6);
    assert!(scores[2] < -1e6);
}

// ── apply_causal_mask ────────────────────────────────────────────────

#[test]
fn apply_causal_mask_seq2() {
    let mut scores = vec![1.0, 1.0, 1.0, 1.0]; // 2x2
    apply_causal_mask(&mut scores, 2).unwrap();
    assert!((scores[0] - 1.0).abs() < 1e-6); // (0,0) not masked
    assert!(scores[1] < -1e6); // (0,1) masked (future)
    assert!((scores[2] - 1.0).abs() < 1e-6); // (1,0) not masked
    assert!((scores[3] - 1.0).abs() < 1e-6); // (1,1) not masked
}

// ── Scaled dot-product attention ─────────────────────────────────────

#[test]
fn sdpa_single_token() {
    let head_dim = 4;
    let q = vec![1.0, 0.0, 0.0, 0.0]; // seq_q=1
    let k = vec![1.0, 0.0, 0.0, 0.0]; // seq_k=1
    let v = vec![1.0, 2.0, 3.0, 4.0]; // seq_k=1
    let result = scaled_dot_product_attention(&q, &k, &v, 1, 1, head_dim, false).unwrap();
    assert_eq!(result.len(), head_dim);
    // Single KV pair → output should be exactly v (after softmax of single score)
    for (r, expected) in result.iter().zip(v.iter()) {
        assert!((r - expected).abs() < 1e-4, "Expected {expected}, got {r}");
    }
}

#[test]
fn sdpa_causal_flag() {
    let head_dim = 2;
    // 2 query tokens, 2 key tokens
    let q = vec![1.0, 0.0, 0.0, 1.0]; // 2 queries
    let k = vec![1.0, 0.0, 0.0, 1.0]; // 2 keys
    let v = vec![1.0, 0.0, 0.0, 1.0]; // 2 values
    let result = scaled_dot_product_attention(&q, &k, &v, 2, 2, head_dim, true).unwrap();
    assert_eq!(result.len(), 4); // 2 queries * head_dim
    for val in &result {
        assert!(val.is_finite());
    }
}

#[test]
fn sdpa_uniform_keys() {
    let head_dim = 2;
    // All keys same → attention should be uniform
    let q = vec![1.0, 0.0];
    let k = vec![1.0, 1.0, 1.0, 1.0]; // 2 identical keys
    let v = vec![1.0, 0.0, 0.0, 1.0]; // 2 different values
    let result = scaled_dot_product_attention(&q, &k, &v, 1, 2, head_dim, false).unwrap();
    assert_eq!(result.len(), 2);
    // Uniform attention → average of v: [(1+0)/2, (0+1)/2] = [0.5, 0.5]
    assert!((result[0] - 0.5).abs() < 0.1);
    assert!((result[1] - 0.5).abs() < 0.1);
}

// ── AttentionKernel ──────────────────────────────────────────────────

#[test]
fn attention_kernel_sdpa() {
    let head_dim = 4;
    let q = vec![1.0; head_dim];
    let k = vec![1.0; head_dim];
    let v = vec![2.0; head_dim];
    let result =
        AttentionKernel::scaled_dot_product(&q, &k, &v, None, 0.5, 1, 1, head_dim).unwrap();
    assert_eq!(result.len(), head_dim);
    for val in &result {
        assert!((val - 2.0).abs() < 0.01);
    }
}

// ── masked_attention ─────────────────────────────────────────────────

#[test]
fn masked_attention_single() {
    let head_dim = 4;
    let q = vec![1.0; head_dim];
    let k = vec![1.0; head_dim];
    let v = vec![3.0; head_dim];
    let result = masked_attention(&q, &k, &v, 1, head_dim).unwrap();
    assert_eq!(result.len(), head_dim);
}

// ── AttentionConfig ──────────────────────────────────────────────────

#[test]
fn attention_config_scale() {
    let config =
        AttentionConfig { num_heads: 8, head_dim: 64, seq_len: 128, scale: None, causal: true };
    let scale = config.resolved_scale();
    let expected = 1.0 / (64.0f32).sqrt();
    assert!((scale - expected).abs() < 1e-6);
}

#[test]
fn attention_config_custom_scale() {
    let config = AttentionConfig {
        num_heads: 4,
        head_dim: 32,
        seq_len: 64,
        scale: Some(0.1),
        causal: false,
    };
    assert!((config.resolved_scale() - 0.1).abs() < 1e-6);
}

#[test]
fn attention_config_validate() {
    let config =
        AttentionConfig { num_heads: 4, head_dim: 32, seq_len: 64, scale: None, causal: true };
    assert!(config.validate().is_ok());
}

// ── GqaConfig ────────────────────────────────────────────────────────

#[test]
fn gqa_config_4to1() {
    let config = GqaConfig {
        num_q_heads: 8,
        num_kv_heads: 2,
        head_dim: 32,
        seq_len: 16,
        scale: None,
        causal: true,
    };
    // 4:1 group ratio
    assert_eq!(config.num_q_heads / config.num_kv_heads, 4);
}
