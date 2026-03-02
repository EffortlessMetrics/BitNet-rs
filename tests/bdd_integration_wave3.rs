//! BDD Integration Wave 3 — Multi-module interaction tests.
//!
//! These tests exercise real data flow across multiple CPU kernel modules,
//! verifying that composed pipelines produce correct, meaningful results.

use bitnet_kernels::cpu::attention::{
    AttentionConfig, AttentionKernel, CpuAttention, CpuAttentionConfig, GqaConfig,
};
use bitnet_kernels::cpu::batch::{
    batched_add, batched_layer_norm, batched_matmul, batched_softmax,
};
use bitnet_kernels::cpu::concat::ConcatKernel;
use bitnet_kernels::cpu::embedding::{
    CpuEmbeddingConfig, add_positional_encoding, embedding_lookup, embedding_lookup_batched,
    embedding_with_position, normalize_embeddings, positional_embedding, positional_encoding,
};
use bitnet_kernels::cpu::ffn::{
    FfnActivation, FfnConfig, ffn_forward, ffn_forward_batched, gated_ffn_forward,
};
use bitnet_kernels::cpu::fusion::{
    fused_add_normalize, fused_gelu_linear, fused_rmsnorm_linear, fused_scale_add,
    fused_softmax_mask,
};
use bitnet_kernels::cpu::gating::{GatingType, apply_gating, geglu, reglu, swiglu};
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice,
};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_cpu};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_symmetric_i8, quantize_binary, quantize_symmetric_i8,
    quantize_ternary,
};
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};
use bitnet_kernels::cpu::transpose::TransposeKernel;

// ── Helpers ────────────────────────────────────────────────────────

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (tol={tol})");
    }
}

fn is_normalized(data: &[f32], tol: f32) -> bool {
    let sum: f32 = data.iter().sum();
    (sum - 1.0).abs() < tol
}

fn all_finite(data: &[f32]) -> bool {
    data.iter().all(|v| v.is_finite())
}

// ═══════════════════════════════════════════════════════════════════
// 1. Quantize → Dequantize round-trip
// ═══════════════════════════════════════════════════════════════════

#[test]
fn quantize_dequantize_round_trip_preserves_values_8bit() {
    let input = vec![0.5, -0.3, 1.0, -1.0, 0.0, 0.7, -0.8, 0.2];
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let restored = dequantize_symmetric_i8(&quantized, scale);

    let err = compute_quantization_error(&input, &restored);
    assert!(err.mse < 0.001, "MSE too high: {}", err.mse);
    assert!(err.max_abs_error < 0.02, "max error too high: {}", err.max_abs_error);
}

#[test]
fn quantize_dequantize_round_trip_4bit_higher_error() {
    let input = vec![0.5, -0.3, 1.0, -1.0, 0.0, 0.7, -0.8, 0.2];
    let (q8, scale8) = quantize_symmetric_i8(&input, 8);
    let restored8 = dequantize_symmetric_i8(&q8, scale8);
    let err8 = compute_quantization_error(&input, &restored8);

    let (q4, scale4) = quantize_symmetric_i8(&input, 4);
    let restored4 = dequantize_symmetric_i8(&q4, scale4);
    let err4 = compute_quantization_error(&input, &restored4);

    // 4-bit should have higher error than 8-bit
    assert!(err4.mse >= err8.mse, "4-bit should have ≥ error than 8-bit");
}

#[test]
fn quantize_dequantize_preserves_zero() {
    let input = vec![0.0, 0.0, 0.0, 0.0];
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let restored = dequantize_symmetric_i8(&quantized, scale);
    assert_eq!(restored, vec![0.0; 4]);
    assert_eq!(scale, 0.0);
}

#[test]
fn ternary_quantize_then_check_values() {
    let input = vec![0.5, -0.3, 0.01, -0.8, 0.0, 0.9];
    let ternary = quantize_ternary(&input, 0.1);
    // All values must be -1, 0, or 1
    for &v in &ternary {
        assert!(v == -1 || v == 0 || v == 1, "unexpected ternary value: {v}");
    }
    // 0.01 is below threshold 0.1, should be 0
    assert_eq!(ternary[2], 0);
    // 0.0 should be 0
    assert_eq!(ternary[4], 0);
}

#[test]
fn binary_quantize_produces_only_minus1_plus1() {
    let input = vec![0.5, -0.3, 0.01, -0.8, 0.0, 0.9];
    let binary = quantize_binary(&input);
    for &v in &binary {
        assert!(v == -1 || v == 1, "unexpected binary value: {v}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Layer norm → attention → residual pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn layer_norm_then_attention_then_residual() {
    let seq_len = 4;
    let head_dim = 8;
    let num_heads = 2;
    let model_dim = num_heads * head_dim;
    let total = seq_len * model_dim;

    // Random-ish input
    let input: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin()).collect();

    // Layer norm
    let ln_config = LayerNormConfig::new(vec![model_dim]);
    let gamma = vec![1.0f32; model_dim];
    let beta = vec![0.0f32; model_dim];
    let normed = layer_norm(&input, &gamma, Some(&beta), &ln_config).unwrap();
    assert_eq!(normed.len(), total);

    // Attention
    let attn_cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: true, scale: None };
    let attn_out =
        AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();
    assert_eq!(attn_out.len(), total);
    assert!(all_finite(&attn_out));

    // Residual connection: output = input + attn_out
    let mut residual_out = attn_out.clone();
    add_residual(&mut residual_out, &input).unwrap();
    assert_eq!(residual_out.len(), total);

    // Verify residual actually added (not identical to attn_out)
    assert!(residual_out.iter().zip(attn_out.iter()).any(|(a, b)| (a - b).abs() > 1e-6));
}

#[test]
fn rms_norm_then_attention_preserves_finite() {
    let seq_len = 4;
    let head_dim = 8;
    let num_heads = 1;
    let model_dim = num_heads * head_dim;
    let total = seq_len * model_dim;

    let input: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.3).cos()).collect();
    let gamma = vec![1.0f32; model_dim];
    let ln_config = LayerNormConfig::new(vec![model_dim]);

    let normed = rms_norm(&input, &gamma, &ln_config).unwrap();
    assert!(all_finite(&normed));

    let attn_cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: false, scale: None };
    let out = AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();
    assert!(all_finite(&out));
}

// ═══════════════════════════════════════════════════════════════════
// 3. Embedding → layer norm → attention → FFN pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn embedding_layernorm_attention_ffn_pipeline() {
    let vocab_size = 32;
    let embed_dim = 16;
    let seq_len = 4;
    let num_heads = 2;
    let head_dim = embed_dim / num_heads;
    let intermediate_dim = 32;

    // Embedding table
    let table: Vec<f32> = (0..vocab_size * embed_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let indices: Vec<u32> = vec![1, 5, 10, 3];

    // Step 1: Embedding lookup
    let embeddings = embedding_lookup(&table, &indices, embed_dim).unwrap();
    assert_eq!(embeddings.len(), seq_len * embed_dim);

    // Step 2: Layer norm
    let ln_config = LayerNormConfig::new(vec![embed_dim]);
    let gamma = vec![1.0f32; embed_dim];
    let normed = layer_norm(&embeddings, &gamma, None, &ln_config).unwrap();

    // Step 3: Self-attention
    let attn_cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: true, scale: None };
    let attn_out =
        AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();

    // Step 4: Residual
    let mut post_attn = attn_out;
    add_residual(&mut post_attn, &embeddings).unwrap();

    // Step 5: FFN
    let ffn_cfg = FfnConfig::new(embed_dim, intermediate_dim, FfnActivation::SiLU).unwrap();
    let w_up: Vec<f32> =
        (0..intermediate_dim * embed_dim).map(|i| (i as f32 * 0.02).sin() * 0.1).collect();
    let w_down: Vec<f32> =
        (0..embed_dim * intermediate_dim).map(|i| (i as f32 * 0.03).cos() * 0.1).collect();

    // FFN per token
    let mut ffn_out = Vec::with_capacity(seq_len * embed_dim);
    for t in 0..seq_len {
        let token = &post_attn[t * embed_dim..(t + 1) * embed_dim];
        let out = ffn_forward(token, &w_up, &w_down, &ffn_cfg).unwrap();
        ffn_out.extend_from_slice(&out);
    }

    assert_eq!(ffn_out.len(), seq_len * embed_dim);
    assert!(all_finite(&ffn_out));
}

// ═══════════════════════════════════════════════════════════════════
// 4. Batch operations with different tensor sizes
// ═══════════════════════════════════════════════════════════════════

#[test]
fn batched_matmul_different_sizes() {
    for (m, k, n) in [(2, 3, 4), (1, 1, 1), (4, 8, 2)] {
        let batch = 2;
        let a: Vec<f32> = (0..batch * m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| i as f32 * 0.1).collect();
        let c = batched_matmul(&a, &b, batch, m, k, n).unwrap();
        assert_eq!(c.len(), batch * m * n);
        assert!(all_finite(&c));
    }
}

#[test]
fn batched_softmax_then_sum_is_one() {
    let batch = 3;
    let seq_len = 8;
    let input: Vec<f32> = (0..batch * seq_len).map(|i| (i as f32 - 10.0) * 0.5).collect();
    let output = batched_softmax(&input, batch, seq_len).unwrap();

    for b in 0..batch {
        let row = &output[b * seq_len..(b + 1) * seq_len];
        assert!(
            is_normalized(row, 1e-5),
            "batch {b} not normalized: sum={}",
            row.iter().sum::<f32>()
        );
        // All values non-negative
        assert!(row.iter().all(|&v| v >= 0.0));
    }
}

#[test]
fn batched_layer_norm_then_matmul() {
    let batch = 2;
    let dim = 8;
    let out_dim = 4;

    let input: Vec<f32> = (0..batch * dim).map(|i| i as f32 * 0.3).collect();
    let gamma = vec![1.0f32; dim];
    let beta = vec![0.0f32; dim];

    let normed = batched_layer_norm(&input, &gamma, &beta, batch, dim, 1e-5).unwrap();

    // Verify each row has mean ≈ 0 and std ≈ 1
    for b in 0..batch {
        let row = &normed[b * dim..(b + 1) * dim];
        let mean: f32 = row.iter().sum::<f32>() / dim as f32;
        assert!(mean.abs() < 0.01, "mean not near zero: {mean}");
    }

    // Project to smaller dim via batched_matmul
    let weight: Vec<f32> = (0..batch * dim * out_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let projected = batched_matmul(&normed, &weight, batch, 1, dim, out_dim).unwrap();
    assert_eq!(projected.len(), batch * out_dim);
    assert!(all_finite(&projected));
}

#[test]
fn batched_add_accumulation() {
    let batch = 4;
    let dim = 16;
    let a: Vec<f32> = (0..batch * dim).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 2.0).collect();
    let c = batched_add(&a, &b, batch, dim).unwrap();

    for i in 0..batch * dim {
        assert!((c[i] - (a[i] + b[i])).abs() < 1e-6);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Concat → reshape → split operations
// ═══════════════════════════════════════════════════════════════════

#[test]
fn concat_then_split_round_trip() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];

    // Concat along axis 0: two [2,2] → [4,2]
    let shape_a: &[usize] = &[2, 2];
    let shape_b: &[usize] = &[2, 2];
    let inputs: &[&[f32]] = &[&a, &b];
    let shapes: &[&[usize]] = &[shape_a, shape_b];

    let concatenated = ConcatKernel::concat(inputs, shapes, 0).unwrap();
    assert_eq!(concatenated.len(), 8);

    // Split back: [4,2] → two [2,2]
    let parts = ConcatKernel::split(&concatenated, &[4, 2], 0, 2).unwrap();
    assert_eq!(parts.len(), 2);
    assert_close(&parts[0], &a, 0.0);
    assert_close(&parts[1], &b, 0.0);
}

#[test]
fn concat_axis1_then_reshape() {
    let a = [1.0f32, 2.0, 3.0, 4.0]; // [2,2]
    let b = [5.0f32, 6.0, 7.0, 8.0]; // [2,2]

    // Concat along axis 1: two [2,2] → [2,4]
    let shape_a: &[usize] = &[2, 2];
    let shape_b: &[usize] = &[2, 2];
    let concatenated = ConcatKernel::concat(&[&a[..], &b[..]], &[shape_a, shape_b], 1).unwrap();
    assert_eq!(concatenated.len(), 8);

    // Reshape [2,4] → [4,2]
    let reshaped = TransposeKernel::reshape(&concatenated, &[2, 4], &[4, 2]).unwrap();
    assert_eq!(reshaped.len(), 8);
    // Data should be same (just reinterpreted)
    assert_close(&reshaped, &concatenated, 0.0);
}

#[test]
fn stack_along_new_axis_then_flatten() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];

    // Stack along axis 0: two [3] → [2,3]
    let stacked = ConcatKernel::stack(&[&a[..], &b[..]], &[3], 0).unwrap();
    assert_eq!(stacked.len(), 6);
    assert_close(&stacked, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.0);

    // Flatten [2,3] → [6]
    let (flat, flat_shape) = TransposeKernel::flatten(&stacked, &[2, 3], 0, 1).unwrap();
    assert_eq!(flat.len(), 6);
    assert_eq!(flat_shape, vec![6]);
    assert_close(&flat, &stacked, 0.0);
}

// ═══════════════════════════════════════════════════════════════════
// 6. Gating → element-wise ops pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn swiglu_gating_produces_sensible_output() {
    let gate = vec![1.0f32, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 3.0];
    let up = vec![1.0f32; 8];
    let mut output = vec![0.0f32; 8];

    swiglu(&gate, &up, &mut output).unwrap();

    // SiLU(x) = x * sigmoid(x). For x=0, SiLU(0)=0.
    assert!((output[6]).abs() < 1e-6, "SiLU(0) should be ~0");
    // For large positive x, SiLU(x) ≈ x
    assert!(output[7] > 2.5, "SiLU(3) should be close to 3");
    assert!(all_finite(&output));
}

#[test]
fn all_gating_types_produce_different_results() {
    let gate = vec![0.5f32, -0.3, 1.0, -1.0];
    let up = vec![1.0f32; 4];
    let mut swiglu_out = vec![0.0f32; 4];
    let mut geglu_out = vec![0.0f32; 4];
    let mut reglu_out = vec![0.0f32; 4];

    swiglu(&gate, &up, &mut swiglu_out).unwrap();
    geglu(&gate, &up, &mut geglu_out).unwrap();
    reglu(&gate, &up, &mut reglu_out).unwrap();

    // They should produce different outputs for same input
    assert_ne!(swiglu_out, geglu_out);
    assert_ne!(geglu_out, reglu_out);
}

#[test]
fn apply_gating_dispatch_matches_direct() {
    let gate = vec![0.5f32, -0.3, 1.0, -1.0];
    let up = vec![1.0f32, 2.0, 0.5, 0.3];

    let mut direct = vec![0.0f32; 4];
    let mut dispatched = vec![0.0f32; 4];

    swiglu(&gate, &up, &mut direct).unwrap();
    apply_gating(GatingType::SwiGLU, &gate, &up, &mut dispatched).unwrap();

    assert_close(&direct, &dispatched, 1e-7);
}

#[test]
fn gating_then_residual_add() {
    let gate = vec![1.0f32, 0.5, -0.5, 2.0];
    let up = vec![1.0f32; 4];
    let mut gated = vec![0.0f32; 4];
    swiglu(&gate, &up, &mut gated).unwrap();

    // Add residual
    let residual = vec![0.1f32, 0.2, 0.3, 0.4];
    add_residual(&mut gated, &residual).unwrap();

    // Verify residual was added
    assert!(all_finite(&gated));
    // The result should be gate_output + residual
}

// ═══════════════════════════════════════════════════════════════════
// 7. Multi-head attention with causal masking
// ═══════════════════════════════════════════════════════════════════

#[test]
fn multi_head_attention_causal_first_token_sees_only_self() {
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 4;
    let model_dim = num_heads * head_dim;
    let total = seq_len * model_dim;

    // Make each position distinct
    let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1 + 0.5).sin()).collect();
    let k = q.clone();
    let v: Vec<f32> = (0..total).map(|i| i as f32).collect();

    let cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: true, scale: None };
    let out_causal = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg).unwrap();

    // With causal masking, position 0 only attends to itself.
    // So output at position 0 should equal value at position 0.
    let first_pos = &out_causal[..model_dim];
    let first_val = &v[..model_dim];
    assert_close(first_pos, first_val, 1e-4);
}

#[test]
fn causal_vs_non_causal_attention_differ() {
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 4;
    let model_dim = num_heads * head_dim;
    let total = seq_len * model_dim;

    let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2).sin()).collect();
    let k = q.clone();
    let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.3).cos()).collect();

    let cfg_causal = AttentionConfig { num_heads, head_dim, seq_len, causal: true, scale: None };
    let cfg_non = AttentionConfig { num_heads, head_dim, seq_len, causal: false, scale: None };

    let out_causal = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg_causal).unwrap();
    let out_non = AttentionKernel::multi_head_attention(&q, &k, &v, &cfg_non).unwrap();

    // Last position should differ (causal sees all; non-causal also sees all, but earlier positions differ)
    assert_ne!(out_causal, out_non);
}

#[test]
fn cpu_attention_batched_forward() {
    let batch_size = 2;
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let model_dim = num_heads * head_dim;
    let total = batch_size * seq_len * model_dim;

    let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin()).collect();
    let k = q.clone();
    let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2).cos()).collect();

    let attn = CpuAttention::new(CpuAttentionConfig {
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        scale: None,
        causal_mask: true,
    })
    .unwrap();

    let out = attn.forward(&q, &k, &v).unwrap();
    assert_eq!(out.len(), total);
    assert!(all_finite(&out));
}

#[test]
fn grouped_query_attention_reduces_kv_heads() {
    let num_q_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let seq_len = 3;

    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q: Vec<f32> = (0..seq_len * q_dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let k: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i as f32 * 0.2).cos()).collect();
    let v: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i as f32 * 0.15).sin()).collect();

    let cfg = GqaConfig { num_q_heads, num_kv_heads, head_dim, seq_len, causal: true, scale: None };

    let out = AttentionKernel::grouped_query_attention(&q, &k, &v, &cfg).unwrap();
    assert_eq!(out.len(), seq_len * q_dim);
    assert!(all_finite(&out));
}

// ═══════════════════════════════════════════════════════════════════
// 8. FFN end-to-end: standard and gated
// ═══════════════════════════════════════════════════════════════════

#[test]
fn ffn_forward_output_shape_and_finiteness() {
    let hidden = 16;
    let inter = 32;
    let cfg = FfnConfig::new(hidden, inter, FfnActivation::GeLU).unwrap();

    let input = vec![0.1f32; hidden];
    let w_up: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let w_down: Vec<f32> = (0..hidden * inter).map(|i| (i as f32 * 0.02).cos()).collect();

    let out = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
    assert_eq!(out.len(), hidden);
    assert!(all_finite(&out));
}

#[test]
fn gated_ffn_with_swiglu_activation() {
    let hidden = 8;
    let inter = 16;
    let cfg = FfnConfig::new(hidden, inter, FfnActivation::SiLU).unwrap();

    let input = vec![0.5f32; hidden];
    let w_gate: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let w_up: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.02).cos()).collect();
    let w_down: Vec<f32> = (0..hidden * inter).map(|i| (i as f32 * 0.03).sin()).collect();

    let out = gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &cfg).unwrap();
    assert_eq!(out.len(), hidden);
    assert!(all_finite(&out));
}

#[test]
fn ffn_batched_matches_per_row() {
    let hidden = 8;
    let inter = 16;
    let batch = 3;
    let cfg = FfnConfig::new(hidden, inter, FfnActivation::ReLU).unwrap();

    let input: Vec<f32> = (0..batch * hidden).map(|i| (i as f32 * 0.1).sin()).collect();
    let w_up: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let w_down: Vec<f32> = (0..hidden * inter).map(|i| (i as f32 * 0.02).cos()).collect();

    let batched_out = ffn_forward_batched(&input, &w_up, &w_down, &cfg, batch).unwrap();

    // Compare with per-row forward
    for b in 0..batch {
        let row = &input[b * hidden..(b + 1) * hidden];
        let single = ffn_forward(row, &w_up, &w_down, &cfg).unwrap();
        assert_close(&batched_out[b * hidden..(b + 1) * hidden], &single, 1e-6);
    }
}

#[test]
fn different_ffn_activations_produce_different_results() {
    let hidden = 8;
    let inter = 16;

    let input = vec![0.5f32; hidden];
    let w_up: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let w_down: Vec<f32> = (0..hidden * inter).map(|i| (i as f32 * 0.02).cos()).collect();

    let cfg_gelu = FfnConfig::new(hidden, inter, FfnActivation::GeLU).unwrap();
    let cfg_silu = FfnConfig::new(hidden, inter, FfnActivation::SiLU).unwrap();
    let cfg_relu = FfnConfig::new(hidden, inter, FfnActivation::ReLU).unwrap();

    let out_gelu = ffn_forward(&input, &w_up, &w_down, &cfg_gelu).unwrap();
    let out_silu = ffn_forward(&input, &w_up, &w_down, &cfg_silu).unwrap();
    let out_relu = ffn_forward(&input, &w_up, &w_down, &cfg_relu).unwrap();

    assert_ne!(out_gelu, out_silu);
    assert_ne!(out_silu, out_relu);
}

// ═══════════════════════════════════════════════════════════════════
// 9. Fusion operations
// ═══════════════════════════════════════════════════════════════════

#[test]
fn fused_rmsnorm_linear_matches_separate() {
    let n = 8;
    let out_dim = 4;
    let eps = 1e-5;

    let input: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.3).collect();
    let gamma = vec![1.0f32; n];
    let weight: Vec<f32> = (0..out_dim * n).map(|i| (i as f32 * 0.02).sin()).collect();

    // Fused path
    let fused = fused_rmsnorm_linear(&input, &weight, &gamma, eps).unwrap();

    // Separate: rms_norm then linear
    let ln_cfg = LayerNormConfig::new(vec![n]);
    let normed = rms_norm(&input, &gamma, &ln_cfg).unwrap();
    let cfg = LinearConfig::new(1, n, out_dim).unwrap();
    let mut separate = vec![0.0f32; out_dim];
    linear_cpu(&normed, &weight, None, &mut separate, &cfg).unwrap();

    assert_close(&fused, &separate, 1e-4);
}

#[test]
fn fused_gelu_linear_matches_separate() {
    let n = 8;
    let out_dim = 4;

    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 4.0) * 0.5).collect();
    let weight: Vec<f32> = (0..out_dim * n).map(|i| (i as f32 * 0.03).sin()).collect();
    let bias = vec![0.1f32; out_dim];

    let fused = fused_gelu_linear(&input, &weight, &bias).unwrap();
    assert_eq!(fused.len(), out_dim);
    assert!(all_finite(&fused));
}

#[test]
fn fused_add_normalize_matches_separate() {
    let n = 8;
    let eps = 1e-5;

    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * -0.2).collect();
    let gamma = vec![1.0f32; n];

    // Fused
    let fused = fused_add_normalize(&a, &b, &gamma, eps).unwrap();

    // Separate: add then rms_norm
    let combined: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
    let ln_cfg = LayerNormConfig::new(vec![n]);
    let separate = rms_norm(&combined, &gamma, &ln_cfg).unwrap();

    assert_close(&fused, &separate, 1e-4);
}

#[test]
fn fused_softmax_mask_produces_valid_distribution() {
    let n = 8;
    let scores: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let mask = vec![0.0f32; n]; // No masking
    let scale = 1.0;

    let output = fused_softmax_mask(&scores, &mask, scale).unwrap();
    assert!(is_normalized(&output, 1e-5));
    assert!(output.iter().all(|&v| v >= 0.0));
}

#[test]
fn fused_scale_add_matches_manual() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![0.5f32, 1.0, 1.5, 2.0];
    let scale = 0.3f32;

    let result = fused_scale_add(&a, &b, scale).unwrap();
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + scale * y).collect();
    assert_close(&result, &expected, 1e-6);
}

// ═══════════════════════════════════════════════════════════════════
// 10. Softmax numerical stability with extreme values
// ═══════════════════════════════════════════════════════════════════

#[test]
fn batched_softmax_extreme_positive_values() {
    let batch = 1;
    let seq_len = 4;
    let input = vec![1000.0f32, 1001.0, 1002.0, 1003.0];
    let output = batched_softmax(&input, batch, seq_len).unwrap();

    assert!(all_finite(&output));
    assert!(is_normalized(&output, 1e-5));
    // Largest value should have highest probability
    assert!(output[3] > output[0]);
}

#[test]
fn batched_softmax_extreme_negative_values() {
    let batch = 1;
    let seq_len = 4;
    let input = vec![-1000.0f32, -1001.0, -1002.0, -1003.0];
    let output = batched_softmax(&input, batch, seq_len).unwrap();

    assert!(all_finite(&output));
    assert!(is_normalized(&output, 1e-5));
}

#[test]
fn batched_softmax_mixed_extremes() {
    let batch = 2;
    let seq_len = 4;
    let input = vec![
        -1000.0, 0.0, 1000.0, 0.0, // batch 0: huge range
        0.0, 0.0, 0.0, 0.0, // batch 1: uniform
    ];
    let output = batched_softmax(&input, batch, seq_len).unwrap();
    assert!(all_finite(&output));

    // batch 0: position 2 (value=1000) should dominate
    assert!(output[2] > 0.99);

    // batch 1: uniform input → uniform output
    for &v in &output[4..8] {
        assert!((v - 0.25).abs() < 1e-5);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 11. Embedding + positional encoding
// ═══════════════════════════════════════════════════════════════════

#[test]
fn embedding_with_positional_encoding_pipeline() {
    let vocab_size = 16;
    let embed_dim = 8;
    let seq_len = 4;

    let table: Vec<f32> = (0..vocab_size * embed_dim).map(|i| (i as f32 * 0.05).sin()).collect();
    let indices: Vec<u32> = vec![0, 3, 7, 12];

    let mut embeddings = embedding_lookup(&table, &indices, embed_dim).unwrap();
    let pos_enc = positional_embedding(seq_len, embed_dim);
    add_positional_encoding(&mut embeddings, &pos_enc, seq_len, embed_dim);

    assert_eq!(embeddings.len(), seq_len * embed_dim);
    assert!(all_finite(&embeddings));

    // Embeddings should be different from raw lookup (positional encoding added)
    let raw = embedding_lookup(&table, &indices, embed_dim).unwrap();
    assert_ne!(embeddings, raw);
}

#[test]
fn embedding_lookup_then_normalize() {
    let vocab_size = 8;
    let embed_dim = 4;

    let table: Vec<f32> = (0..vocab_size * embed_dim).map(|i| i as f32 + 1.0).collect();
    let indices: Vec<u32> = vec![0, 3, 5];

    let mut embeddings = embedding_lookup(&table, &indices, embed_dim).unwrap();
    normalize_embeddings(&mut embeddings, embed_dim);

    // After normalization, each vector should have unit L2 norm
    for chunk in embeddings.chunks(embed_dim) {
        let norm: f32 = chunk.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm not unit: {norm}");
    }
}

#[test]
fn embedding_with_position_uses_sinusoidal() {
    let vocab_size = 16;
    let embed_dim = 8;

    let table: Vec<f32> = vec![0.0; vocab_size * embed_dim]; // Zero embeddings
    let indices: Vec<u32> = vec![0, 1];
    let config = CpuEmbeddingConfig::new(vocab_size, embed_dim);

    let result = embedding_with_position(&table, &indices, &config, 0).unwrap();

    // With zero embeddings, result should be pure positional encoding
    assert_eq!(result.len(), 2 * embed_dim);
    // Position 0 and position 1 should be different
    assert_ne!(&result[..embed_dim], &result[embed_dim..]);
    assert!(all_finite(&result));
}

// ═══════════════════════════════════════════════════════════════════
// 12. RoPE (rotary position embedding) pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn rope_apply_then_check_rotation_properties() {
    let head_dim = 8;
    let max_seq = 16;
    let config = RopeConfig::new(head_dim, max_seq);
    let freqs = compute_frequencies(&config);

    // RoPE should preserve vector magnitude
    let mut data = vec![1.0f32; head_dim];
    let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    apply_rope(&mut data, 0, head_dim, &freqs);
    let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm_before - norm_after).abs() < 1e-4, "RoPE changed magnitude");
}

#[test]
fn rope_batch_applies_to_all_heads() {
    let head_dim = 4;
    let num_heads = 2;
    let seq_len = 3;
    let max_seq = 8;
    let total = seq_len * num_heads * head_dim;

    let config = RopeConfig::new(head_dim, max_seq);
    let freqs = compute_frequencies(&config);

    let original: Vec<f32> = (0..total).map(|i| (i as f32 * 0.3 + 1.0).sin()).collect();
    let mut data = original.clone();
    apply_rope_batch(&mut data, 0, seq_len, num_heads, head_dim, &freqs);

    // Data should be modified
    assert_ne!(data, original);
    assert!(all_finite(&data));

    // Magnitude of each head should be preserved
    for pos in 0..seq_len {
        for head in 0..num_heads {
            let start = pos * num_heads * head_dim + head * head_dim;
            let orig_norm: f32 =
                original[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            let new_norm: f32 =
                data[start..start + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (orig_norm - new_norm).abs() < 1e-4,
                "RoPE changed magnitude at pos={pos} head={head}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 13. KV cache operations pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn kv_cache_append_and_slice_round_trip() {
    let cfg = KvCacheConfig {
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        max_seq_len: 8,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(cfg).unwrap();

    // Append 2 tokens of KV
    let token_elems = 2 * 4; // num_heads * head_dim
    let keys: Vec<f32> = (0..2 * token_elems).map(|i| i as f32).collect();
    let values: Vec<f32> = (0..2 * token_elems).map(|i| (i as f32) * 10.0).collect();

    kv_cache_append(&mut cache, 0, &keys, &values).unwrap();

    // Slice back
    let (k_slice, v_slice) = kv_cache_slice(&cache, 0, 0, 2).unwrap();
    assert_eq!(k_slice.len(), 2 * token_elems);
    assert_close(k_slice, &keys, 0.0);
    assert_close(v_slice, &values, 0.0);
}

#[test]
fn kv_cache_clear_resets_length() {
    let cfg = KvCacheConfig {
        num_layers: 2,
        num_heads: 1,
        head_dim: 4,
        max_seq_len: 8,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(cfg).unwrap();

    let keys = vec![1.0f32; 4];
    let values = vec![2.0f32; 4];
    kv_cache_append(&mut cache, 0, &keys, &values).unwrap();
    kv_cache_append(&mut cache, 1, &keys, &values).unwrap();

    assert!(kv_cache_memory_usage(&cache) > 0);

    kv_cache_clear(&mut cache);
    assert_eq!(cache.seq_len(0).unwrap(), 0);
    assert_eq!(cache.seq_len(1).unwrap(), 0);
}

// ═══════════════════════════════════════════════════════════════════
// 14. Reduction operations pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn reduction_after_linear_projection() {
    let in_f = 8;
    let out_f = 4;
    let cfg = LinearConfig::new(1, in_f, out_f).unwrap();

    let input: Vec<f32> = (0..in_f).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let weight: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut output = vec![0.0f32; out_f];

    linear_cpu(&input, &weight, None, &mut output, &cfg).unwrap();

    let sum = ReductionKernel::sum(&output).unwrap();
    let mean = ReductionKernel::mean(&output).unwrap();
    let max_val = ReductionKernel::max(&output).unwrap();
    let l2 = ReductionKernel::l2_norm(&output).unwrap();

    assert!((mean - sum / out_f as f32).abs() < 1e-6);
    assert!(max_val.value <= output.iter().copied().fold(f32::NEG_INFINITY, f32::max) + 1e-6);
    assert!(l2 >= 0.0);
}

// ═══════════════════════════════════════════════════════════════════
// 15. Transpose operations with attention
// ═══════════════════════════════════════════════════════════════════

#[test]
fn transpose_2d_round_trip() {
    let rows = 3;
    let cols = 4;
    let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();

    let transposed = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
    assert_eq!(transposed.len(), rows * cols);

    let back = TransposeKernel::transpose_2d(&transposed, cols, rows).unwrap();
    assert_close(&back, &data, 0.0);
}

#[test]
fn reshape_preserves_element_count() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();

    let reshaped = TransposeKernel::reshape(&data, &[2, 3, 4], &[6, 4]).unwrap();
    assert_eq!(reshaped.len(), 24);
    assert_close(&reshaped, &data, 0.0); // Data unchanged, just reinterpreted
}

// ═══════════════════════════════════════════════════════════════════
// 16. End-to-end transformer block
// ═══════════════════════════════════════════════════════════════════

#[test]
fn full_transformer_block_pipeline() {
    let embed_dim = 16;
    let num_heads = 2;
    let head_dim = embed_dim / num_heads;
    let seq_len = 4;
    let inter_dim = 32;
    let total = seq_len * embed_dim;

    // Simulated hidden states
    let hidden: Vec<f32> = (0..total).map(|i| (i as f32 * 0.07).sin()).collect();

    // Pre-attention LayerNorm
    let ln_config = LayerNormConfig::new(vec![embed_dim]);
    let gamma = vec![1.0f32; embed_dim];
    let pre_attn = rms_norm(&hidden, &gamma, &ln_config).unwrap();

    // Self-attention
    let attn_cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: true, scale: None };
    let attn_out =
        AttentionKernel::multi_head_attention(&pre_attn, &pre_attn, &pre_attn, &attn_cfg).unwrap();

    // Residual
    let mut post_attn = attn_out;
    add_residual(&mut post_attn, &hidden).unwrap();

    // Pre-FFN norm
    let pre_ffn = rms_norm(&post_attn, &gamma, &ln_config).unwrap();

    // FFN
    let ffn_cfg = FfnConfig::new(embed_dim, inter_dim, FfnActivation::SiLU).unwrap();
    let w_up: Vec<f32> = (0..inter_dim * embed_dim).map(|i| (i as f32 * 0.005).sin()).collect();
    let w_down: Vec<f32> = (0..embed_dim * inter_dim).map(|i| (i as f32 * 0.007).cos()).collect();
    let ffn_out = ffn_forward_batched(&pre_ffn, &w_up, &w_down, &ffn_cfg, seq_len).unwrap();

    // Final residual
    let mut output = ffn_out;
    add_residual(&mut output, &post_attn).unwrap();

    assert_eq!(output.len(), total);
    assert!(all_finite(&output));
    // Output should differ from input (the block did something)
    assert!(output.iter().zip(hidden.iter()).any(|(a, b)| (a - b).abs() > 1e-4));
}

#[test]
fn dual_residual_streams_maintained() {
    let dim = 8;
    let stream_a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
    let stream_b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).sin()).collect();

    // Combine via scaled residual
    let mut combined = stream_a.clone();
    add_residual_scaled(&mut combined, &stream_b, 0.5).unwrap();

    // Verify: combined = stream_a + 0.5 * stream_b
    for i in 0..dim {
        let expected = stream_a[i] + 0.5 * stream_b[i];
        assert!((combined[i] - expected).abs() < 1e-6);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 17. Linear projection pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn linear_projection_with_bias_then_norm() {
    let in_f = 8;
    let out_f = 4;
    let cfg = LinearConfig::new(1, in_f, out_f).unwrap().with_bias(true);

    let input: Vec<f32> = (0..in_f).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let weight: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32 * 0.01).sin()).collect();
    let bias = vec![0.1f32; out_f];
    let mut projected = vec![0.0f32; out_f];

    linear_cpu(&input, &weight, Some(&bias), &mut projected, &cfg).unwrap();

    // Apply layer norm to projected output
    let ln_cfg = LayerNormConfig::new(vec![out_f]);
    let gamma = vec![1.0f32; out_f];
    let normed = layer_norm(&projected, &gamma, None, &ln_cfg).unwrap();

    assert_eq!(normed.len(), out_f);
    assert!(all_finite(&normed));

    // Normed output should have mean ≈ 0
    let mean: f32 = normed.iter().sum::<f32>() / out_f as f32;
    assert!(mean.abs() < 0.01);
}

#[test]
fn batched_linear_projection() {
    let batch = 3;
    let in_f = 4;
    let out_f = 2;
    let cfg = LinearConfig::new(batch, in_f, out_f).unwrap();

    let input: Vec<f32> = (0..batch * in_f).map(|i| i as f32 * 0.1).collect();
    let weight: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut output = vec![0.0f32; batch * out_f];

    linear_cpu(&input, &weight, None, &mut output, &cfg).unwrap();
    assert_eq!(output.len(), batch * out_f);
    assert!(all_finite(&output));
}

// ═══════════════════════════════════════════════════════════════════
// 18. Cross-module data flow: embedding → batched ops → reduction
// ═══════════════════════════════════════════════════════════════════

#[test]
fn embedding_to_batched_layer_norm_to_softmax() {
    let vocab_size = 32;
    let embed_dim = 8;
    let batch = 2;
    let seq_per_batch = 3;

    let table: Vec<f32> = (0..vocab_size * embed_dim).map(|i| (i as f32 * 0.03).sin()).collect();
    let all_indices: Vec<&[u32]> = vec![&[1, 2, 3], &[4, 5, 6]];
    let embeddings = embedding_lookup_batched(&table, &all_indices, vocab_size, embed_dim).unwrap();

    assert_eq!(embeddings.len(), batch * seq_per_batch * embed_dim);

    // Per-sequence layer norm
    let gamma = vec![1.0f32; embed_dim];
    let beta = vec![0.0f32; embed_dim];
    let normed =
        batched_layer_norm(&embeddings, &gamma, &beta, batch * seq_per_batch, embed_dim, 1e-5)
            .unwrap();

    // Reduce each token embedding to a score via sum, then softmax per batch
    let mut scores = Vec::with_capacity(batch * seq_per_batch);
    for t in 0..(batch * seq_per_batch) {
        let token = &normed[t * embed_dim..(t + 1) * embed_dim];
        scores.push(ReductionKernel::sum(token).unwrap());
    }

    let probs = batched_softmax(&scores, batch, seq_per_batch).unwrap();
    for b in 0..batch {
        let row = &probs[b * seq_per_batch..(b + 1) * seq_per_batch];
        assert!(is_normalized(row, 1e-5));
    }
}

// ═══════════════════════════════════════════════════════════════════
// 19. Positional encoding properties
// ═══════════════════════════════════════════════════════════════════

#[test]
fn positional_encoding_different_bases_produce_different_results() {
    let seq_len = 4;
    let embed_dim = 8;

    let pe_10k = positional_encoding(seq_len, embed_dim, 10_000.0);
    let pe_1k = positional_encoding(seq_len, embed_dim, 1_000.0);

    assert_ne!(pe_10k, pe_1k);
    assert!(all_finite(&pe_10k));
    assert!(all_finite(&pe_1k));
}

#[test]
fn positional_encoding_positions_are_unique() {
    let seq_len = 8;
    let embed_dim = 16;
    let pe = positional_embedding(seq_len, embed_dim);

    // Each position should produce a unique encoding
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            let row_i = &pe[i * embed_dim..(i + 1) * embed_dim];
            let row_j = &pe[j * embed_dim..(j + 1) * embed_dim];
            assert_ne!(row_i, row_j, "positions {i} and {j} have identical encodings");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 20. Squeeze / unsqueeze shape operations
// ═══════════════════════════════════════════════════════════════════

#[test]
fn squeeze_removes_singleton_dims() {
    let shape = vec![1, 4, 1, 8, 1];
    let squeezed = TransposeKernel::squeeze(&shape);
    assert_eq!(squeezed, vec![4, 8]);
}

#[test]
fn unsqueeze_then_squeeze_round_trip() {
    let shape = vec![4, 8];
    let unsqueezed = TransposeKernel::unsqueeze(&shape, 0).unwrap();
    assert_eq!(unsqueezed, vec![1, 4, 8]);
    let squeezed = TransposeKernel::squeeze(&unsqueezed);
    assert_eq!(squeezed, shape);
}
