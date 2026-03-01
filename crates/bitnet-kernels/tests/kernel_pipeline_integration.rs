//! Integration tests for the BitNet kernel pipeline.
//!
//! Verifies that kernels chain correctly (embedding → attention →
//! layernorm → projection), produce deterministic output, propagate
//! shapes correctly, and remain numerically stable across edge cases.

#![cfg(feature = "cpu")]

use bitnet_kernels::cpu::activations::{ActivationType, activate};
use bitnet_kernels::cpu::attention::{
    causal_mask, masked_attention, multi_head_attention_cpu, scaled_dot_product_attention,
};
use bitnet_kernels::cpu::embedding::embedding_lookup;
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm};
use bitnet_kernels::cpu::quantize::quantize_symmetric_i8;
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};

// ── Helpers ────────────────────────────────────────────────────────

fn no_nan_inf(data: &[f32]) -> bool {
    data.iter().all(|v| v.is_finite())
}

/// Simple deterministic pseudo-random f32 in [-1, 1].
fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

// ── Pipeline integration ───────────────────────────────────────────

#[test]
fn pipeline_embedding_to_layernorm() {
    let vocab_size = 64;
    let embed_dim = 16;
    let seq_len = 4;
    let table = pseudo_random(42, vocab_size * embed_dim);
    let indices: Vec<u32> = vec![1, 5, 10, 3];

    let embeddings = embedding_lookup(&table, &indices, embed_dim).unwrap();
    assert_eq!(embeddings.len(), seq_len * embed_dim);

    let gamma = vec![1.0f32; embed_dim];
    let beta = vec![0.0f32; embed_dim];
    let config = LayerNormConfig::new(vec![embed_dim]);
    let normed = layer_norm(&embeddings, &gamma, Some(&beta), &config).unwrap();

    assert_eq!(normed.len(), seq_len * embed_dim);
    assert!(no_nan_inf(&normed));
}

#[test]
fn pipeline_layernorm_to_attention() {
    let seq_len = 4;
    let head_dim = 8;
    let num_heads = 2;
    let total = seq_len * num_heads * head_dim;

    let input = pseudo_random(99, total);
    let gamma = vec![1.0f32; num_heads * head_dim];
    let beta = vec![0.0f32; num_heads * head_dim];
    let config = LayerNormConfig::new(vec![num_heads * head_dim]);
    let normed = layer_norm(&input, &gamma, Some(&beta), &config).unwrap();

    let output =
        multi_head_attention_cpu(&normed, &normed, &normed, num_heads, head_dim, seq_len, true)
            .unwrap();
    assert_eq!(output.len(), total);
    assert!(no_nan_inf(&output));
}

#[test]
fn pipeline_attention_to_projection() {
    let seq_len = 4;
    let dim = 8;
    let proj_dim = 16;

    let qkv = pseudo_random(7, seq_len * dim);
    let attn_out =
        scaled_dot_product_attention(&qkv, &qkv, &qkv, seq_len, seq_len, dim, true).unwrap();
    assert_eq!(attn_out.len(), seq_len * dim);

    let weight = pseudo_random(13, dim * proj_dim);
    let mut proj_out = vec![0.0f32; seq_len * proj_dim];
    let cfg = SimdMatmulConfig::new(seq_len, proj_dim, dim);
    simd_matmul_f32(&attn_out, &weight, &mut proj_out, &cfg).unwrap();

    assert_eq!(proj_out.len(), seq_len * proj_dim);
    assert!(no_nan_inf(&proj_out));
}

#[test]
fn pipeline_full_embedding_attention_layernorm_projection() {
    let vocab_size = 128;
    let embed_dim = 16;
    let num_heads = 2;
    let head_dim = embed_dim / num_heads;
    let seq_len = 4;
    let ffn_dim = 32;

    // Step 1: Embedding
    let table = pseudo_random(1, vocab_size * embed_dim);
    let indices: Vec<u32> = vec![10, 20, 30, 40];
    let embeddings = embedding_lookup(&table, &indices, embed_dim).unwrap();
    assert_eq!(embeddings.len(), seq_len * embed_dim);

    // Step 2: Attention
    let attn_out = multi_head_attention_cpu(
        &embeddings,
        &embeddings,
        &embeddings,
        num_heads,
        head_dim,
        seq_len,
        true,
    )
    .unwrap();
    assert_eq!(attn_out.len(), seq_len * embed_dim);

    // Step 3: Residual + LayerNorm
    let residual: Vec<f32> = embeddings.iter().zip(&attn_out).map(|(e, a)| e + a).collect();
    let gamma = vec![1.0f32; embed_dim];
    let config = LayerNormConfig::new(vec![embed_dim]);
    let normed = layer_norm(&residual, &gamma, None, &config).unwrap();
    assert_eq!(normed.len(), seq_len * embed_dim);

    // Step 4: FFN projection
    let w_up = pseudo_random(77, embed_dim * ffn_dim);
    let mut ffn_out = vec![0.0f32; seq_len * ffn_dim];
    let cfg = SimdMatmulConfig::new(seq_len, ffn_dim, embed_dim);
    simd_matmul_f32(&normed, &w_up, &mut ffn_out, &cfg).unwrap();

    // Step 5: Activation
    let activated = activate(&ffn_out, ActivationType::SiLU);
    assert_eq!(activated.len(), seq_len * ffn_dim);
    assert!(no_nan_inf(&activated));
}

#[test]
fn pipeline_embedding_rope_attention() {
    let vocab_size = 64;
    let embed_dim = 8;
    let seq_len = 4;
    let head_dim = embed_dim;

    let table = pseudo_random(55, vocab_size * embed_dim);
    let indices: Vec<u32> = vec![2, 4, 6, 8];
    let mut embeddings = embedding_lookup(&table, &indices, embed_dim).unwrap();

    // Apply RoPE
    let rope_cfg = RopeConfig::new(head_dim, 128);
    let freqs = compute_frequencies(&rope_cfg);
    apply_rope_batch(&mut embeddings, 0, seq_len, 1, head_dim, &freqs);
    assert!(no_nan_inf(&embeddings));

    // Self-attention
    let output = scaled_dot_product_attention(
        &embeddings,
        &embeddings,
        &embeddings,
        seq_len,
        seq_len,
        head_dim,
        true,
    )
    .unwrap();
    assert_eq!(output.len(), seq_len * head_dim);
    assert!(no_nan_inf(&output));
}

// ── Determinism / consistency ──────────────────────────────────────

#[test]
fn determinism_matmul_same_input_same_output() {
    let m = 8;
    let n = 8;
    let k = 16;
    let a = pseudo_random(100, m * k);
    let b = pseudo_random(200, k * n);
    let cfg = SimdMatmulConfig::new(m, n, k);

    let mut c1 = vec![0.0f32; m * n];
    let mut c2 = vec![0.0f32; m * n];
    simd_matmul_f32(&a, &b, &mut c1, &cfg).unwrap();
    simd_matmul_f32(&a, &b, &mut c2, &cfg).unwrap();

    assert_eq!(c1, c2, "matmul must be deterministic");
}

#[test]
fn determinism_attention_same_input_same_output() {
    let seq_len = 4;
    let head_dim = 8;
    let qkv = pseudo_random(300, seq_len * head_dim);

    let out1 = masked_attention(&qkv, &qkv, &qkv, seq_len, head_dim).unwrap();
    let out2 = masked_attention(&qkv, &qkv, &qkv, seq_len, head_dim).unwrap();

    assert_eq!(out1, out2, "attention must be deterministic");
}

#[test]
fn determinism_layernorm_same_input_same_output() {
    let dim = 32;
    let batch = 4;
    let input = pseudo_random(400, batch * dim);
    let gamma = vec![1.0f32; dim];
    let config = LayerNormConfig::new(vec![dim]);

    let out1 = layer_norm(&input, &gamma, None, &config).unwrap();
    let out2 = layer_norm(&input, &gamma, None, &config).unwrap();

    assert_eq!(out1, out2, "layer_norm must be deterministic");
}

#[test]
fn determinism_quantize_roundtrip_same_input_same_output() {
    let input = pseudo_random(500, 128);
    let (q1, s1) = quantize_symmetric_i8(&input, 8);
    let (q2, s2) = quantize_symmetric_i8(&input, 8);

    assert_eq!(q1, q2);
    assert_eq!(s1, s2);
}

// ── Shape propagation ──────────────────────────────────────────────

#[test]
fn shape_matmul_output_dimensions() {
    for (m, n, k) in [(1, 1, 1), (4, 8, 16), (16, 32, 64)] {
        let a = vec![0.1f32; m * k];
        let b = vec![0.1f32; k * n];
        let mut c = vec![0.0f32; m * n];
        let cfg = SimdMatmulConfig::new(m, n, k);
        simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();
        assert_eq!(c.len(), m * n, "matmul output shape mismatch for ({m},{n},{k})");
    }
}

#[test]
fn shape_attention_preserves_dimensions() {
    for seq_len in [1, 2, 8, 16] {
        let head_dim = 8;
        let data = vec![0.1f32; seq_len * head_dim];
        let out =
            scaled_dot_product_attention(&data, &data, &data, seq_len, seq_len, head_dim, false)
                .unwrap();
        assert_eq!(out.len(), seq_len * head_dim, "attention shape mismatch for seq_len={seq_len}");
    }
}

#[test]
fn shape_multi_head_attention_output() {
    let seq_len = 4;
    let num_heads = 2;
    let head_dim = 8;
    let total = seq_len * num_heads * head_dim;

    let data = pseudo_random(600, total);
    let output =
        multi_head_attention_cpu(&data, &data, &data, num_heads, head_dim, seq_len, true).unwrap();
    assert_eq!(output.len(), total);
}

#[test]
fn shape_embedding_lookup_output() {
    for (num_tokens, embed_dim) in [(1, 4), (8, 16), (32, 64)] {
        let vocab = 128;
        let table = vec![0.1f32; vocab * embed_dim];
        let indices: Vec<u32> = (0..num_tokens as u32).collect();
        let output = embedding_lookup(&table, &indices, embed_dim).unwrap();
        assert_eq!(
            output.len(),
            num_tokens * embed_dim,
            "embedding shape mismatch for tokens={num_tokens}, dim={embed_dim}"
        );
    }
}

// ── Numerical stability ────────────────────────────────────────────

#[test]
fn stability_layernorm_large_values() {
    let dim = 16;
    let input: Vec<f32> = (0..dim).map(|i| (i as f32) * 1000.0).collect();
    let gamma = vec![1.0f32; dim];
    let config = LayerNormConfig::new(vec![dim]);
    let output = layer_norm(&input, &gamma, None, &config).unwrap();
    assert!(no_nan_inf(&output), "layernorm produced NaN/Inf for large inputs");
}

#[test]
fn stability_attention_no_nan_inf() {
    let seq_len = 8;
    let head_dim = 16;
    let data = pseudo_random(700, seq_len * head_dim);
    let output = masked_attention(&data, &data, &data, seq_len, head_dim).unwrap();
    assert!(no_nan_inf(&output), "attention produced NaN/Inf");
}

#[test]
fn stability_activation_functions_no_nan() {
    let input = pseudo_random(800, 64);
    for act in [
        ActivationType::ReLU,
        ActivationType::GELU,
        ActivationType::SiLU,
        ActivationType::Sigmoid,
        ActivationType::Tanh,
    ] {
        let output = activate(&input, act);
        assert!(no_nan_inf(&output), "activation {act:?} produced NaN/Inf");
    }
}

// ── Edge cases ─────────────────────────────────────────────────────

#[test]
fn edge_single_element_sequence_attention() {
    let head_dim = 8;
    let data = pseudo_random(900, head_dim);
    let output = scaled_dot_product_attention(&data, &data, &data, 1, 1, head_dim, true).unwrap();
    assert_eq!(output.len(), head_dim);
    assert!(no_nan_inf(&output));
}

#[test]
fn edge_single_element_layernorm() {
    let config = LayerNormConfig::new(vec![1]);
    let input = vec![42.0f32];
    let gamma = vec![1.0f32];
    let output = layer_norm(&input, &gamma, None, &config).unwrap();
    assert_eq!(output.len(), 1);
    assert!(no_nan_inf(&output));
}

#[test]
fn edge_large_sequence_attention() {
    let seq_len = 128;
    let head_dim = 8;
    let data = pseudo_random(1000, seq_len * head_dim);
    let output =
        scaled_dot_product_attention(&data, &data, &data, seq_len, seq_len, head_dim, true)
            .unwrap();
    assert_eq!(output.len(), seq_len * head_dim);
    assert!(no_nan_inf(&output));
}

#[test]
fn edge_causal_mask_first_row_sees_only_first_token() {
    let mask = causal_mask(4);
    // Row 0: only position 0 is visible (0.0), rest are -inf
    assert_eq!(mask[0], 0.0);
    assert!(mask[1].is_infinite() && mask[1] < 0.0);
    assert!(mask[2].is_infinite() && mask[2] < 0.0);
    assert!(mask[3].is_infinite() && mask[3] < 0.0);
}

#[test]
fn edge_rope_preserves_vector_norm() {
    let head_dim = 8;
    let rope_cfg = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&rope_cfg);

    let mut data = vec![1.0f32; head_dim];
    let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    apply_rope(&mut data, 0, head_dim, &freqs);
    let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!(
        (norm_before - norm_after).abs() < 1e-5,
        "RoPE should preserve vector norm: before={norm_before}, after={norm_after}"
    );
}

#[test]
fn edge_zero_input_layernorm() {
    let dim = 16;
    let input = vec![0.0f32; dim];
    let gamma = vec![1.0f32; dim];
    let config = LayerNormConfig::new(vec![dim]);
    let output = layer_norm(&input, &gamma, None, &config).unwrap();
    assert_eq!(output.len(), dim);
    // With zero input, mean=0 and variance=0, output should be 0/sqrt(eps) ≈ 0
    assert!(no_nan_inf(&output));
}
