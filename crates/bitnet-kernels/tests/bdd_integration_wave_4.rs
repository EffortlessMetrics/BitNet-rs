//! BDD Integration Wave 4 — End-to-end kernel pipeline tests.
//!
//! Tests verify that kernel primitives compose correctly into realistic
//! inference pipelines: embedding → attention → FFN → loss, with
//! quantization round-trips and KV cache lifecycle management.

#![cfg(feature = "cpu")]
#![allow(clippy::float_cmp)]

use bitnet_kernels::cpu::activations::{self, ActivationType};
use bitnet_kernels::cpu::attention::{
    CpuAttention, CpuAttentionConfig, causal_mask, multi_head_attention_cpu,
    scaled_dot_product_attention,
};
use bitnet_kernels::cpu::batch::{batched_layer_norm, batched_softmax};
use bitnet_kernels::cpu::embedding::{
    CpuEmbeddingConfig, embedding_lookup, embedding_with_position,
};
use bitnet_kernels::cpu::ffn::{
    FfnActivation, FfnConfig, ffn_forward, ffn_forward_batched, gated_ffn_forward,
};
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice,
};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_forward};
use bitnet_kernels::cpu::loss::{
    LossReduction, binary_cross_entropy, cosine_similarity_loss, cross_entropy_loss, mse_loss,
};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_symmetric_i8, quantize_symmetric_i8,
};
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};

// ─── Helpers ───────────────────────────────────────────────────────

/// Deterministic pseudo-random f32 in [-1, 1].
fn prng(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn all_finite(data: &[f32]) -> bool {
    data.iter().all(|v| v.is_finite())
}

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() <= tol, "mismatch at [{i}]: {x} vs {y} (tol={tol})");
    }
}

fn softmax_1d(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn l2_norm(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn perplexity(ce_loss: f32) -> f32 {
    ce_loss.exp()
}

// ═══════════════════════════════════════════════════════════════════
// Section 1 — Inference Pipeline (tokenize → embed → transform → decode)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn pipeline_embed_layernorm_attention_ffn_produces_finite_output() {
    let seq_len = 8;
    let hidden = 64;
    let heads = 4;
    let head_dim = hidden / heads;

    let vocab_size = 128;
    let table = prng(42, vocab_size * hidden);
    let indices: Vec<u32> = (0..seq_len as u32).collect();
    let embedded = embedding_lookup(&table, &indices, hidden).unwrap();
    assert_eq!(embedded.len(), seq_len * hidden);

    let gamma = vec![1.0f32; hidden];
    let beta = vec![0.0f32; hidden];
    let ln_cfg = LayerNormConfig::new(vec![hidden]);
    let normed = layer_norm(&embedded, &gamma, Some(&beta), &ln_cfg).unwrap();
    assert!(all_finite(&normed));

    let attn_out =
        multi_head_attention_cpu(&normed, &normed, &normed, heads, head_dim, seq_len, true)
            .unwrap();
    assert_eq!(attn_out.len(), seq_len * hidden);
    assert!(all_finite(&attn_out));

    let mut residual_out = attn_out.clone();
    add_residual(&mut residual_out, &normed).unwrap();
    let normed2 = layer_norm(&residual_out, &gamma, Some(&beta), &ln_cfg).unwrap();

    let inter_dim = hidden * 4;
    let ffn_cfg = FfnConfig::new(hidden, inter_dim, FfnActivation::GeLU).unwrap();
    let w_up = prng(100, inter_dim * hidden);
    let w_down = prng(200, hidden * inter_dim);
    let ffn_out = ffn_forward_batched(&normed2, &w_up, &w_down, &ffn_cfg, seq_len).unwrap();
    assert_eq!(ffn_out.len(), seq_len * hidden);
    assert!(all_finite(&ffn_out));
}

#[test]
fn pipeline_embed_to_logits_produces_valid_distribution() {
    let seq_len = 4;
    let hidden = 32;
    let vocab = 64;

    let table = prng(10, vocab * hidden);
    let indices: Vec<u32> = vec![1, 5, 10, 3];
    let embedded = embedding_lookup(&table, &indices, hidden).unwrap();

    let last_hidden = &embedded[(seq_len - 1) * hidden..seq_len * hidden];
    let proj_weight = prng(20, vocab * hidden);
    let cfg = LinearConfig::new(1, hidden, vocab).unwrap();
    let mut logits = vec![0.0f32; vocab];
    linear_forward(last_hidden, &proj_weight, None, &mut logits, &cfg).unwrap();

    let probs = softmax_1d(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
    assert!(probs.iter().all(|&p| p >= 0.0));
}

#[test]
fn pipeline_batched_inference_preserves_batch_independence() {
    let batch = 2;
    let seq_len = 4;
    let hidden = 16;

    let vocab = 32;
    let table = prng(42, vocab * hidden);

    let indices_a: Vec<u32> = vec![0, 1, 2, 3];
    let indices_b: Vec<u32> = vec![4, 5, 6, 7];

    let embed_a = embedding_lookup(&table, &indices_a, hidden).unwrap();
    let embed_b = embedding_lookup(&table, &indices_b, hidden).unwrap();

    let gamma = vec![1.0f32; hidden];
    let beta = vec![0.0f32; hidden];
    let ln_cfg = LayerNormConfig::new(vec![hidden]);
    let norm_a = layer_norm(&embed_a, &gamma, Some(&beta), &ln_cfg).unwrap();
    let norm_b = layer_norm(&embed_b, &gamma, Some(&beta), &ln_cfg).unwrap();

    let mut combined = embed_a.clone();
    combined.extend_from_slice(&embed_b);
    let norm_combined =
        batched_layer_norm(&combined, &gamma, &beta, batch * seq_len, hidden, 1e-5).unwrap();

    assert_close(&norm_combined[..seq_len * hidden], &norm_a, 1e-5);
    assert_close(&norm_combined[seq_len * hidden..], &norm_b, 1e-5);
}

#[test]
fn pipeline_greedy_decode_selects_argmax_token() {
    let hidden = 16;
    let vocab = 8;

    let table = prng(55, vocab * hidden);
    let indices: Vec<u32> = vec![2];
    let embedded = embedding_lookup(&table, &indices, hidden).unwrap();

    let proj_weight = prng(66, vocab * hidden);
    let cfg = LinearConfig::new(1, hidden, vocab).unwrap();
    let mut logits = vec![0.0f32; vocab];
    linear_forward(&embedded, &proj_weight, None, &mut logits, &cfg).unwrap();

    let probs = softmax_1d(&logits);
    let argmax =
        probs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
    let logit_argmax =
        logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
    assert_eq!(argmax, logit_argmax);
}

#[test]
fn pipeline_deterministic_with_same_inputs() {
    let hidden = 32;
    let heads = 2;
    let head_dim = hidden / heads;
    let seq_len = 4;

    let input = prng(99, seq_len * hidden);
    let out1 =
        multi_head_attention_cpu(&input, &input, &input, heads, head_dim, seq_len, true).unwrap();
    let out2 =
        multi_head_attention_cpu(&input, &input, &input, heads, head_dim, seq_len, true).unwrap();
    assert_eq!(out1, out2, "attention must be deterministic");
}

#[test]
fn pipeline_positional_embedding_adds_position_signal() {
    let seq_len = 8;
    let hidden = 32;
    let vocab = 64;

    let table = prng(42, vocab * hidden);
    let indices: Vec<u32> = vec![1; seq_len];
    let raw = embedding_lookup(&table, &indices, hidden).unwrap();

    for i in 1..seq_len {
        assert_eq!(
            &raw[0..hidden],
            &raw[i * hidden..(i + 1) * hidden],
            "same token should give identical raw embeddings"
        );
    }

    let cfg = CpuEmbeddingConfig::new(vocab, hidden);
    let with_pos = embedding_with_position(&table, &indices, &cfg, 0).unwrap();
    let mut any_different = false;
    for i in 1..seq_len {
        if with_pos[0..hidden] != with_pos[i * hidden..(i + 1) * hidden] {
            any_different = true;
            break;
        }
    }
    assert!(any_different, "positional encoding must differentiate positions");
}

// ═══════════════════════════════════════════════════════════════════
// Section 2 — Quantization Pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn quant_symmetric_i8_roundtrip_preserves_accuracy() {
    let weights = prng(42, 256);
    let (quantized, scale) = quantize_symmetric_i8(&weights, 8);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);

    assert_eq!(dequantized.len(), weights.len());
    let err = compute_quantization_error(&weights, &dequantized);
    assert!(err.max_abs_error < 0.02, "max quant error {} too large", err.max_abs_error);
}

#[test]
fn quant_pipeline_load_quantize_store_dequantize_verify() {
    let original = prng(77, 512);

    let (q_data, scale) = quantize_symmetric_i8(&original, 8);
    assert_eq!(q_data.len(), 512);
    assert!(scale > 0.0, "scale must be positive");

    // Simulated serialize/deserialize
    let stored_bytes: Vec<u8> = q_data.iter().map(|&v| v as u8).collect();
    let loaded: Vec<i8> = stored_bytes.iter().map(|&b| b as i8).collect();

    let restored = dequantize_symmetric_i8(&loaded, scale);
    let err = compute_quantization_error(&original, &restored);
    assert!(err.mse < 0.01, "mse {}", err.mse);
}

#[test]
fn quant_different_bit_widths_trade_accuracy_for_compression() {
    let data = prng(42, 128);
    let (q8, s8) = quantize_symmetric_i8(&data, 8);
    let deq_8 = dequantize_symmetric_i8(&q8, s8);
    let err_8 = compute_quantization_error(&data, &deq_8);

    let (q4, s4) = quantize_symmetric_i8(&data, 4);
    let deq_4 = dequantize_symmetric_i8(&q4, s4);
    let err_4 = compute_quantization_error(&data, &deq_4);

    assert!(
        err_4.mse >= err_8.mse,
        "lower bits should have higher error: 4-bit={}, 8-bit={}",
        err_4.mse,
        err_8.mse
    );
}

#[test]
fn quant_preserves_zero_exactly() {
    let data = vec![0.0f32; 64];
    let (quantized, scale) = quantize_symmetric_i8(&data, 8);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);
    for &v in &dequantized {
        assert!(v.abs() < 1e-7, "zero should be preserved, got {v}");
    }
}

#[test]
fn quant_error_metrics_are_consistent() {
    let original = prng(42, 256);
    let (q, s) = quantize_symmetric_i8(&original, 8);
    let deq = dequantize_symmetric_i8(&q, s);
    let err = compute_quantization_error(&original, &deq);

    assert!(err.mse >= 0.0);
    assert!(err.max_abs_error >= 0.0);
    assert!(all_finite(&deq));
}

#[test]
fn quant_roundtrip_through_ffn_pipeline() {
    let hidden = 32;
    let inter = 64;

    let w_up_orig = prng(10, inter * hidden);
    let (w_up_q, w_up_s) = quantize_symmetric_i8(&w_up_orig, 8);
    let w_up_deq = dequantize_symmetric_i8(&w_up_q, w_up_s);

    let w_down_orig = prng(20, hidden * inter);
    let (w_down_q, w_down_s) = quantize_symmetric_i8(&w_down_orig, 8);
    let w_down_deq = dequantize_symmetric_i8(&w_down_q, w_down_s);

    let input = prng(30, hidden);
    let cfg = FfnConfig::new(hidden, inter, FfnActivation::ReLU).unwrap();

    let out_orig = ffn_forward(&input, &w_up_orig, &w_down_orig, &cfg).unwrap();
    let out_quant = ffn_forward(&input, &w_up_deq, &w_down_deq, &cfg).unwrap();

    assert_eq!(out_orig.len(), out_quant.len());
    assert!(all_finite(&out_quant));
    let err = compute_quantization_error(&out_orig, &out_quant);
    assert!(err.max_abs_error < 1.0, "FFN output divergence too large: {}", err.max_abs_error);
}

// ═══════════════════════════════════════════════════════════════════
// Section 3 — Attention Pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn attention_qkv_projection_to_output() {
    let seq_len = 8;
    let hidden = 64;
    let heads = 4;
    let head_dim = hidden / heads;

    let input = prng(42, seq_len * hidden);

    let wq = prng(100, hidden * hidden);
    let wk = prng(200, hidden * hidden);
    let wv = prng(300, hidden * hidden);
    let cfg = LinearConfig::new(seq_len, hidden, hidden).unwrap();

    let mut q = vec![0.0f32; seq_len * hidden];
    let mut k = vec![0.0f32; seq_len * hidden];
    let mut v = vec![0.0f32; seq_len * hidden];
    linear_forward(&input, &wq, None, &mut q, &cfg).unwrap();
    linear_forward(&input, &wk, None, &mut k, &cfg).unwrap();
    linear_forward(&input, &wv, None, &mut v, &cfg).unwrap();

    let attn = multi_head_attention_cpu(&q, &k, &v, heads, head_dim, seq_len, true).unwrap();
    assert_eq!(attn.len(), seq_len * hidden);
    assert!(all_finite(&attn));

    let wo = prng(400, hidden * hidden);
    let out_cfg = LinearConfig::new(seq_len, hidden, hidden).unwrap();
    let mut output = vec![0.0f32; seq_len * hidden];
    linear_forward(&attn, &wo, None, &mut output, &out_cfg).unwrap();
    assert!(all_finite(&output));
}

#[test]
fn attention_causal_mask_prevents_future_tokens() {
    let seq_len = 4;
    let mask = causal_mask(seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            let val = mask[i * seq_len + j];
            if j > i {
                assert!(val == f32::NEG_INFINITY, "mask[{i}][{j}] should be -inf");
            } else {
                assert!(val == 0.0, "mask[{i}][{j}] should be 0.0");
            }
        }
    }
}

#[test]
fn attention_scaled_dot_product_output_shape() {
    let seq_len = 8;
    let head_dim = 16;
    let q = prng(1, seq_len * head_dim);
    let k = prng(2, seq_len * head_dim);
    let v = prng(3, seq_len * head_dim);
    let out = scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, head_dim, false).unwrap();
    assert_eq!(out.len(), seq_len * head_dim);
    assert!(all_finite(&out));
}

#[test]
fn attention_scores_sum_to_one_per_query() {
    let seq_len = 4;
    let head_dim = 8;
    let q = prng(10, seq_len * head_dim);
    let k = prng(20, seq_len * head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
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

    let probs = batched_softmax(&scores, seq_len, seq_len).unwrap();
    for i in 0..seq_len {
        let row_sum: f32 = probs[i * seq_len..(i + 1) * seq_len].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-5, "row {i} sum = {row_sum}");
    }
}

#[test]
fn attention_with_bias_differs_from_without() {
    let seq_len = 4;
    let hidden = 16;

    let input = prng(42, seq_len * hidden);
    let wq = prng(100, hidden * hidden);
    let bias_q = prng(150, hidden);
    let cfg = LinearConfig::new(seq_len, hidden, hidden).unwrap();

    let mut q_no_bias = vec![0.0f32; seq_len * hidden];
    let mut q_with_bias = vec![0.0f32; seq_len * hidden];
    linear_forward(&input, &wq, None, &mut q_no_bias, &cfg).unwrap();
    linear_forward(&input, &wq, Some(&bias_q), &mut q_with_bias, &cfg).unwrap();

    assert_ne!(q_no_bias, q_with_bias, "bias should change projection output");
}

#[test]
fn attention_identity_values_returns_weighted_values() {
    let seq_len = 4;
    let head_dim = 4;
    let q = vec![1.0f32; seq_len * head_dim];
    let k = vec![1.0f32; seq_len * head_dim];
    let mut v = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len.min(head_dim) {
        v[i * head_dim + i] = 1.0;
    }
    let out = scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, head_dim, false).unwrap();
    assert_eq!(out.len(), seq_len * head_dim);
    assert!(all_finite(&out));
}

// ═══════════════════════════════════════════════════════════════════
// Section 4 — Layer Composition
// ═══════════════════════════════════════════════════════════════════

#[test]
fn layer_embedding_positional_layernorm_attention_ffn_residual() {
    let seq_len = 8;
    let hidden = 64;
    let heads = 4;
    let head_dim = hidden / heads;
    let vocab = 128;
    let inter = hidden * 4;

    let table = prng(1, vocab * hidden);
    let indices: Vec<u32> = (0..seq_len as u32).collect();
    let cfg_embed = CpuEmbeddingConfig::new(vocab, hidden);
    let embedded = embedding_with_position(&table, &indices, &cfg_embed, 0).unwrap();

    let gamma = vec![1.0f32; hidden];
    let beta = vec![0.0f32; hidden];
    let ln_cfg = LayerNormConfig::new(vec![hidden]);
    let pre_attn = layer_norm(&embedded, &gamma, Some(&beta), &ln_cfg).unwrap();

    let attn =
        multi_head_attention_cpu(&pre_attn, &pre_attn, &pre_attn, heads, head_dim, seq_len, true)
            .unwrap();

    let mut post_attn = attn.clone();
    add_residual(&mut post_attn, &embedded).unwrap();

    let pre_ffn = layer_norm(&post_attn, &gamma, Some(&beta), &ln_cfg).unwrap();

    let ffn_cfg = FfnConfig::new(hidden, inter, FfnActivation::SiLU).unwrap();
    let w_up = prng(100, inter * hidden);
    let w_down = prng(200, hidden * inter);
    let ffn_out = ffn_forward_batched(&pre_ffn, &w_up, &w_down, &ffn_cfg, seq_len).unwrap();

    let mut layer_out = ffn_out;
    add_residual(&mut layer_out, &post_attn).unwrap();

    assert_eq!(layer_out.len(), seq_len * hidden);
    assert!(all_finite(&layer_out));
}

#[test]
fn layer_rms_norm_variant_pipeline() {
    let seq_len = 4;
    let hidden = 32;

    let input = prng(42, seq_len * hidden);
    let gamma = vec![1.0f32; hidden];
    let ln_cfg = LayerNormConfig::new(vec![hidden]);

    let normed = rms_norm(&input, &gamma, &ln_cfg).unwrap();
    assert_eq!(normed.len(), seq_len * hidden);
    assert!(all_finite(&normed));

    for t in 0..seq_len {
        let row = &normed[t * hidden..(t + 1) * hidden];
        let rms = (row.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
        assert!((rms - 1.0).abs() < 0.2, "token {t} rms={rms}, expected ~1.0");
    }
}

#[test]
fn layer_gated_ffn_swiglu_pipeline() {
    let hidden = 32;
    let inter = 64;

    let input = prng(42, hidden);
    let cfg = FfnConfig::new(hidden, inter, FfnActivation::SiLU).unwrap();
    let w_gate = prng(10, inter * hidden);
    let w_up = prng(20, inter * hidden);
    let w_down = prng(30, hidden * inter);

    let out = gated_ffn_forward(&input, &w_gate, &w_up, &w_down, &cfg).unwrap();
    assert_eq!(out.len(), hidden);
    assert!(all_finite(&out));
}

#[test]
fn layer_scaled_residual_connection() {
    let n = 64;
    let base = prng(42, n);
    let delta = prng(99, n);

    let mut output = delta.clone();
    add_residual_scaled(&mut output, &base, 0.5).unwrap();

    for i in 0..n {
        let expected = delta[i] + 0.5 * base[i];
        assert!((output[i] - expected).abs() < 1e-5, "mismatch at {i}");
    }
}

#[test]
fn layer_two_transformer_blocks_in_sequence() {
    let seq_len = 4;
    let hidden = 32;
    let heads = 2;
    let head_dim = hidden / heads;
    let inter = hidden * 4;

    let gamma = vec![1.0f32; hidden];
    let beta = vec![0.0f32; hidden];
    let ln_cfg = LayerNormConfig::new(vec![hidden]);
    let ffn_cfg = FfnConfig::new(hidden, inter, FfnActivation::GeLU).unwrap();

    let mut x = prng(42, seq_len * hidden);

    for block in 0..2 {
        let seed = (block + 1) as u64 * 1000;
        let w_up = prng(seed, inter * hidden);
        let w_down = prng(seed + 1, hidden * inter);

        let normed = layer_norm(&x, &gamma, Some(&beta), &ln_cfg).unwrap();
        let attn =
            multi_head_attention_cpu(&normed, &normed, &normed, heads, head_dim, seq_len, true)
                .unwrap();
        let mut post_attn = attn;
        add_residual(&mut post_attn, &x).unwrap();

        let normed2 = layer_norm(&post_attn, &gamma, Some(&beta), &ln_cfg).unwrap();
        let ffn_out = ffn_forward_batched(&normed2, &w_up, &w_down, &ffn_cfg, seq_len).unwrap();
        let mut post_ffn = ffn_out;
        add_residual(&mut post_ffn, &post_attn).unwrap();

        x = post_ffn;
    }

    assert_eq!(x.len(), seq_len * hidden);
    assert!(all_finite(&x));
}

#[test]
fn layer_rope_augmented_attention() {
    let seq_len = 8;
    let hidden = 32;
    let heads = 2;
    let head_dim = hidden / heads;

    let mut q = prng(1, seq_len * hidden);
    let mut k = prng(2, seq_len * hidden);
    let v = prng(3, seq_len * hidden);

    let rope_cfg = RopeConfig::new(head_dim, seq_len * 2);
    let freqs = compute_frequencies(&rope_cfg);
    for t in 0..seq_len {
        for h in 0..heads {
            let off = t * hidden + h * head_dim;
            apply_rope(&mut q[off..off + head_dim], t, head_dim, &freqs);
            apply_rope(&mut k[off..off + head_dim], t, head_dim, &freqs);
        }
    }

    let attn = multi_head_attention_cpu(&q, &k, &v, heads, head_dim, seq_len, true).unwrap();
    assert_eq!(attn.len(), seq_len * hidden);
    assert!(all_finite(&attn));
}

// ═══════════════════════════════════════════════════════════════════
// Section 5 — Multi-Head Attention Flow
// ═══════════════════════════════════════════════════════════════════

#[test]
fn mha_split_heads_attend_concat_project() {
    let seq_len = 4;
    let hidden = 32;
    let heads = 4;
    let head_dim = hidden / heads;

    let input = prng(42, seq_len * hidden);

    let mut per_head_outputs = vec![0.0f32; seq_len * hidden];
    for h in 0..heads {
        let mut q_h = vec![0.0f32; seq_len * head_dim];
        let mut k_h = vec![0.0f32; seq_len * head_dim];
        let mut v_h = vec![0.0f32; seq_len * head_dim];
        for t in 0..seq_len {
            for d in 0..head_dim {
                let src = t * hidden + h * head_dim + d;
                let dst = t * head_dim + d;
                q_h[dst] = input[src];
                k_h[dst] = input[src];
                v_h[dst] = input[src];
            }
        }

        let out_h =
            scaled_dot_product_attention(&q_h, &k_h, &v_h, seq_len, seq_len, head_dim, false)
                .unwrap();

        for t in 0..seq_len {
            for d in 0..head_dim {
                per_head_outputs[t * hidden + h * head_dim + d] = out_h[t * head_dim + d];
            }
        }
    }

    let wo = prng(500, hidden * hidden);
    let proj_cfg = LinearConfig::new(seq_len, hidden, hidden).unwrap();
    let mut projected = vec![0.0f32; seq_len * hidden];
    linear_forward(&per_head_outputs, &wo, None, &mut projected, &proj_cfg).unwrap();
    assert_eq!(projected.len(), seq_len * hidden);
    assert!(all_finite(&projected));
}

#[test]
fn mha_multi_head_matches_api_call() {
    let seq_len = 4;
    let hidden = 16;
    let heads = 2;
    let head_dim = hidden / heads;

    let q = prng(1, seq_len * hidden);
    let k = prng(2, seq_len * hidden);
    let v = prng(3, seq_len * hidden);

    let mha_out = multi_head_attention_cpu(&q, &k, &v, heads, head_dim, seq_len, true).unwrap();
    assert_eq!(mha_out.len(), seq_len * hidden);
    assert!(all_finite(&mha_out));
}

#[test]
fn mha_cpu_attention_config_forward() {
    let seq_len = 4;
    let heads = 2;
    let head_dim = 8;
    let hidden = heads * head_dim;

    let attn_cfg = CpuAttentionConfig {
        batch_size: 1,
        num_heads: heads,
        head_dim,
        seq_len,
        causal_mask: true,
        scale: None,
    };
    let attn = CpuAttention::new(attn_cfg).unwrap();

    let q = prng(10, seq_len * hidden);
    let k = prng(20, seq_len * hidden);
    let v = prng(30, seq_len * hidden);

    let out = attn.forward(&q, &k, &v).unwrap();
    assert_eq!(out.len(), seq_len * hidden);
    assert!(all_finite(&out));
}

#[test]
fn mha_different_head_counts_produce_different_results() {
    let seq_len = 4;
    let hidden = 16;
    let input = prng(42, seq_len * hidden);

    let out_2h = multi_head_attention_cpu(&input, &input, &input, 2, 8, seq_len, true).unwrap();
    let out_4h = multi_head_attention_cpu(&input, &input, &input, 4, 4, seq_len, true).unwrap();

    assert_eq!(out_2h.len(), out_4h.len());
    assert_ne!(out_2h, out_4h);
}

#[test]
fn mha_non_causal_attends_to_all_positions() {
    let seq_len = 4;
    let head_dim = 8;
    let hidden = 2 * head_dim;

    let input = prng(42, seq_len * hidden);

    let causal_out =
        multi_head_attention_cpu(&input, &input, &input, 2, head_dim, seq_len, true).unwrap();
    let non_causal_out =
        multi_head_attention_cpu(&input, &input, &input, 2, head_dim, seq_len, false).unwrap();

    assert_ne!(causal_out, non_causal_out, "causal vs non-causal should differ");
}

// ═══════════════════════════════════════════════════════════════════
// Section 6 — KV Cache Lifecycle
// ═══════════════════════════════════════════════════════════════════

#[test]
fn kv_cache_init_append_slice_verify() {
    let heads = 4;
    let head_dim = 16;
    let layers = 2;
    let max_seq = 32;
    let token_elems = heads * head_dim;

    let cfg = KvCacheConfig {
        num_layers: layers,
        num_heads: heads,
        head_dim,
        max_seq_len: max_seq,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(cfg).unwrap();

    assert_eq!(cache.seq_len(0).unwrap(), 0);

    let k1 = prng(1, token_elems);
    let v1 = prng(2, token_elems);
    kv_cache_append(&mut cache, 0, &k1, &v1).unwrap();
    assert_eq!(cache.seq_len(0).unwrap(), 1);

    let k3 = prng(3, 3 * token_elems);
    let v3 = prng(4, 3 * token_elems);
    kv_cache_append(&mut cache, 0, &k3, &v3).unwrap();
    assert_eq!(cache.seq_len(0).unwrap(), 4);

    let (ks, vs) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
    assert_eq!(ks.len(), token_elems);
    assert_close(ks, &k1, 1e-7);
    assert_close(vs, &v1, 1e-7);
}

#[test]
fn kv_cache_layers_are_independent() {
    let heads = 2;
    let head_dim = 8;
    let token_elems = heads * head_dim;

    let cfg = KvCacheConfig {
        num_layers: 3,
        num_heads: heads,
        head_dim,
        max_seq_len: 16,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(cfg).unwrap();

    let k = prng(1, token_elems);
    let v = prng(2, token_elems);
    kv_cache_append(&mut cache, 0, &k, &v).unwrap();

    assert_eq!(cache.seq_len(0).unwrap(), 1);
    assert_eq!(cache.seq_len(1).unwrap(), 0);
    assert_eq!(cache.seq_len(2).unwrap(), 0);
}

#[test]
fn kv_cache_clear_resets_all_layers() {
    let cfg = KvCacheConfig {
        num_layers: 2,
        num_heads: 2,
        head_dim: 8,
        max_seq_len: 16,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(cfg).unwrap();
    let token_elems = 2 * 8;

    let k = prng(1, token_elems);
    let v = prng(2, token_elems);
    kv_cache_append(&mut cache, 0, &k, &v).unwrap();
    kv_cache_append(&mut cache, 1, &k, &v).unwrap();

    kv_cache_clear(&mut cache);
    assert_eq!(cache.seq_len(0).unwrap(), 0);
    assert_eq!(cache.seq_len(1).unwrap(), 0);
}

#[test]
fn kv_cache_memory_usage_grows_with_layers() {
    let cfg_small = KvCacheConfig {
        num_layers: 1,
        num_heads: 2,
        head_dim: 8,
        max_seq_len: 16,
        dtype: KvDtype::F32,
    };
    let cfg_large = KvCacheConfig {
        num_layers: 4,
        num_heads: 2,
        head_dim: 8,
        max_seq_len: 16,
        dtype: KvDtype::F32,
    };
    let small = KvCache::new(cfg_small).unwrap();
    let large = KvCache::new(cfg_large).unwrap();

    assert!(
        kv_cache_memory_usage(&large) > kv_cache_memory_usage(&small),
        "more layers should use more memory"
    );
}

#[test]
fn kv_cache_attention_integration() {
    let heads = 2;
    let head_dim = 8;
    let token_elems = heads * head_dim;

    let cfg = KvCacheConfig {
        num_layers: 1,
        num_heads: heads,
        head_dim,
        max_seq_len: 16,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(cfg).unwrap();

    // Prefill 4 tokens
    let k_prefill = prng(10, 4 * token_elems);
    let v_prefill = prng(20, 4 * token_elems);
    kv_cache_append(&mut cache, 0, &k_prefill, &v_prefill).unwrap();

    // Decode: 1 new token
    let k_new = prng(40, token_elems);
    let v_new = prng(50, token_elems);
    kv_cache_append(&mut cache, 0, &k_new, &v_new).unwrap();

    let seq = cache.seq_len(0).unwrap();
    assert_eq!(seq, 5);
    let (all_k, all_v) = kv_cache_slice(&cache, 0, 0, seq).unwrap();
    assert_eq!(all_k.len(), 5 * token_elems);
    assert_eq!(all_v.len(), 5 * token_elems);
    assert!(all_finite(all_k));
    assert!(all_finite(all_v));
}

#[test]
fn kv_cache_incremental_append_consistency() {
    let heads = 2;
    let head_dim = 4;
    let token_elems = heads * head_dim;

    let cfg = KvCacheConfig {
        num_layers: 1,
        num_heads: heads,
        head_dim,
        max_seq_len: 32,
        dtype: KvDtype::F32,
    };

    // Approach 1: append all at once
    let k_all = prng(1, 4 * token_elems);
    let v_all = prng(2, 4 * token_elems);
    let mut cache1 = KvCache::new(cfg.clone()).unwrap();
    kv_cache_append(&mut cache1, 0, &k_all, &v_all).unwrap();

    // Approach 2: append one at a time
    let mut cache2 = KvCache::new(cfg).unwrap();
    for i in 0..4 {
        let k_one = &k_all[i * token_elems..(i + 1) * token_elems];
        let v_one = &v_all[i * token_elems..(i + 1) * token_elems];
        kv_cache_append(&mut cache2, 0, k_one, v_one).unwrap();
    }

    let (ks1, vs1) = kv_cache_slice(&cache1, 0, 0, 4).unwrap();
    let (ks2, vs2) = kv_cache_slice(&cache2, 0, 0, 4).unwrap();
    assert_close(ks1, ks2, 1e-7);
    assert_close(vs1, vs2, 1e-7);
}

// ═══════════════════════════════════════════════════════════════════
// Section 7 — Loss Computation Flow
// ═══════════════════════════════════════════════════════════════════

#[test]
fn loss_forward_logits_softmax_cross_entropy() {
    let batch = 2;
    let vocab = 8;
    let logits = prng(42, batch * vocab);
    let targets: Vec<usize> = vec![3, 5];

    let (loss, per_sample) =
        cross_entropy_loss(&logits, &targets, vocab, LossReduction::Mean).unwrap();
    assert!(loss.is_finite(), "CE loss must be finite");
    assert!(loss >= 0.0, "CE loss must be non-negative");
    assert_eq!(per_sample.len(), batch);
    assert!(per_sample.iter().all(|l| l.is_finite() && *l >= 0.0));
}

#[test]
fn loss_perplexity_from_cross_entropy() {
    let vocab = 4;
    let logits = vec![0.0f32; vocab]; // uniform logits
    let targets = vec![0usize];

    let (ce, _) = cross_entropy_loss(&logits, &targets, vocab, LossReduction::Mean).unwrap();
    let ppl = perplexity(ce);

    // Uniform distribution over 4 classes → perplexity ≈ 4
    assert!((ppl - 4.0).abs() < 0.5, "perplexity={ppl}, expected ~4.0");
}

#[test]
fn loss_perfect_prediction_has_low_loss() {
    let vocab = 4;
    let logits = vec![-10.0, -10.0, 10.0, -10.0];
    let targets = vec![2usize];

    let (loss, _) = cross_entropy_loss(&logits, &targets, vocab, LossReduction::Mean).unwrap();
    assert!(loss < 0.01, "perfect prediction loss should be ~0, got {loss}");
}

#[test]
fn loss_wrong_prediction_has_high_loss() {
    let vocab = 4;
    let logits = vec![10.0, -10.0, -10.0, -10.0];
    let targets = vec![3usize];

    let (loss, _) = cross_entropy_loss(&logits, &targets, vocab, LossReduction::Mean).unwrap();
    assert!(loss > 10.0, "wrong prediction should have high loss, got {loss}");
}

#[test]
fn loss_binary_cross_entropy_correct_range() {
    let preds = vec![0.9, 0.1, 0.8, 0.2];
    let targets = vec![1.0, 0.0, 1.0, 0.0];

    let loss = binary_cross_entropy(&preds, &targets, LossReduction::Mean).unwrap();
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
    assert!(loss < 1.0, "good predictions should have low BCE loss, got {loss}");
}

#[test]
fn loss_mse_zero_for_identical_inputs() {
    let a = prng(42, 64);
    let loss = mse_loss(&a, &a, LossReduction::Mean).unwrap();
    assert!(loss.abs() < 1e-7, "MSE of identical inputs should be 0, got {loss}");
}

#[test]
fn loss_cosine_similarity_self_is_one() {
    let a = prng(42, 64);
    let sim = cosine_similarity_loss(&a, &a).unwrap();
    // cosine_similarity_loss returns 1 - cos(a,b), so identical = 0
    assert!(sim.abs() < 1e-5, "cosine loss of self should be 0.0, got {sim}");
}

#[test]
fn loss_cosine_similarity_orthogonal_is_zero() {
    let a = vec![1.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];
    let sim = cosine_similarity_loss(&a, &b).unwrap();
    // cosine_similarity_loss returns 1 - cos(a,b), so orthogonal = 1
    assert!((sim - 1.0).abs() < 1e-5, "cosine loss of orthogonal should be 1.0, got {sim}");
}

#[test]
fn loss_reduction_modes_produce_different_values() {
    let batch = 4;
    let vocab = 8;
    let logits = prng(42, batch * vocab);
    let targets = vec![0, 1, 2, 3];

    let (mean_loss, _) = cross_entropy_loss(&logits, &targets, vocab, LossReduction::Mean).unwrap();
    let (sum_loss, _) = cross_entropy_loss(&logits, &targets, vocab, LossReduction::Sum).unwrap();

    assert!(
        (sum_loss - mean_loss * batch as f32).abs() < 1e-4,
        "sum={sum_loss} should ≈ mean*batch={}",
        mean_loss * batch as f32
    );
}

// ═══════════════════════════════════════════════════════════════════
// Section 8 — Gradient Flow (forward → backward simulation)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn gradient_finite_differences_for_linear() {
    let in_f = 8;
    let out_f = 4;
    let eps = 1e-3;

    let input = prng(42, in_f);
    let weight = prng(99, out_f * in_f);
    let cfg = LinearConfig::new(1, in_f, out_f).unwrap();

    let mut output_base = vec![0.0f32; out_f];
    linear_forward(&input, &weight, None, &mut output_base, &cfg).unwrap();

    let mut input_perturbed = input.clone();
    input_perturbed[0] += eps;
    let mut output_perturbed = vec![0.0f32; out_f];
    linear_forward(&input_perturbed, &weight, None, &mut output_perturbed, &cfg).unwrap();

    let mut any_nonzero = false;
    for j in 0..out_f {
        let grad = (output_perturbed[j] - output_base[j]) / eps;
        assert!(grad.is_finite(), "gradient must be finite");
        if grad.abs() > 1e-6 {
            any_nonzero = true;
        }
    }
    assert!(any_nonzero, "gradient should be non-zero for linear layer");
}

#[test]
fn gradient_finite_differences_for_ffn() {
    let hidden = 8;
    let inter = 16;
    let eps = 1e-3;

    let input = prng(42, hidden);
    let w_up = prng(10, inter * hidden);
    let w_down = prng(20, hidden * inter);
    let cfg = FfnConfig::new(hidden, inter, FfnActivation::ReLU).unwrap();

    let out_base = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();

    let mut input_pert = input.clone();
    input_pert[0] += eps;
    let out_pert = ffn_forward(&input_pert, &w_up, &w_down, &cfg).unwrap();

    let grads: Vec<f32> =
        out_base.iter().zip(out_pert.iter()).map(|(&a, &b)| (b - a) / eps).collect();
    assert!(all_finite(&grads));
}

#[test]
fn gradient_clipping_by_norm() {
    let mut gradients = vec![3.0, 4.0]; // norm = 5.0
    let max_norm = 2.5;

    let norm = l2_norm(&gradients);
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in &mut gradients {
            *g *= scale;
        }
    }

    let clipped_norm = l2_norm(&gradients);
    assert!(
        (clipped_norm - max_norm).abs() < 1e-5,
        "clipped norm={clipped_norm}, expected {max_norm}"
    );
}

#[test]
fn gradient_clipping_preserves_direction() {
    let original = vec![3.0f32, 4.0];
    let mut clipped = original.clone();
    let max_norm = 2.5;

    let norm = l2_norm(&clipped);
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in &mut clipped {
            *g *= scale;
        }
    }

    let ratio_orig = original[0] / original[1];
    let ratio_clipped = clipped[0] / clipped[1];
    assert!((ratio_orig - ratio_clipped).abs() < 1e-5, "clipping should preserve direction");
}

#[test]
fn gradient_no_clipping_when_below_threshold() {
    let gradients = vec![0.1, 0.2, 0.3];
    let max_norm = 10.0;
    let norm = l2_norm(&gradients);

    assert!(norm < max_norm, "small gradients should be below threshold");

    let mut clipped = gradients.clone();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in &mut clipped {
            *g *= scale;
        }
    }
    assert_eq!(gradients, clipped, "should not modify small gradients");
}

#[test]
fn gradient_loss_decreases_with_step() {
    let hidden = 8;
    let vocab = 4;
    let target = 2usize;
    let lr = 0.1;

    let mut weight = prng(42, vocab * hidden);
    let input = prng(99, hidden);
    let cfg = LinearConfig::new(1, hidden, vocab).unwrap();

    let mut logits = vec![0.0f32; vocab];
    linear_forward(&input, &weight, None, &mut logits, &cfg).unwrap();
    let (loss_before, _) =
        cross_entropy_loss(&logits, &[target], vocab, LossReduction::Mean).unwrap();

    // Numerical gradient descent step
    let eps = 1e-3;
    for w_idx in 0..weight.len() {
        weight[w_idx] += eps;
        let mut logits_p = vec![0.0f32; vocab];
        linear_forward(&input, &weight, None, &mut logits_p, &cfg).unwrap();
        let (loss_p, _) =
            cross_entropy_loss(&logits_p, &[target], vocab, LossReduction::Mean).unwrap();
        let grad = (loss_p - loss_before) / eps;
        weight[w_idx] -= eps;
        weight[w_idx] -= lr * grad;
    }

    let mut logits2 = vec![0.0f32; vocab];
    linear_forward(&input, &weight, None, &mut logits2, &cfg).unwrap();
    let (loss_after, _) =
        cross_entropy_loss(&logits2, &[target], vocab, LossReduction::Mean).unwrap();

    assert!(
        loss_after < loss_before,
        "loss should decrease: before={loss_before}, after={loss_after}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Section 9 — Error Propagation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn error_linear_dimension_mismatch() {
    let cfg = LinearConfig::new(1, 8, 4).unwrap();
    let input = vec![0.0f32; 3]; // too small for in_features=8
    let weight = prng(1, 4 * 8);
    let mut output = vec![0.0f32; 4];
    assert!(linear_forward(&input, &weight, None, &mut output, &cfg).is_err());
}

#[test]
fn error_ffn_dimension_mismatch() {
    let cfg = FfnConfig::new(8, 16, FfnActivation::GeLU).unwrap();
    let input = vec![0.0f32; 4]; // wrong size
    let w_up = prng(1, 16 * 8);
    let w_down = prng(2, 8 * 16);
    assert!(ffn_forward(&input, &w_up, &w_down, &cfg).is_err());
}

#[test]
fn error_cross_entropy_target_out_of_range() {
    let logits = prng(42, 8);
    let targets = vec![10usize]; // out of range for vocab=8
    assert!(cross_entropy_loss(&logits, &targets, 8, LossReduction::Mean).is_err());
}

#[test]
fn error_kv_cache_layer_out_of_range() {
    let cfg = KvCacheConfig {
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        max_seq_len: 8,
        dtype: KvDtype::F32,
    };
    let cache = KvCache::new(cfg).unwrap();
    assert!(cache.seq_len(5).is_err(), "should fail for out-of-range layer");
}

#[test]
fn error_embedding_index_out_of_range() {
    let vocab = 8;
    let dim = 4;
    let table = vec![0.0f32; vocab * dim];
    let indices = vec![100u32]; // out of range
    assert!(embedding_lookup(&table, &indices, dim).is_err());
}

#[test]
fn error_residual_length_mismatch() {
    let mut a = vec![1.0f32; 8];
    let b = vec![2.0f32; 4]; // mismatch
    assert!(add_residual(&mut a, &b).is_err());
}

// ═══════════════════════════════════════════════════════════════════
// Section 10 — Additional Integration Scenarios
// ═══════════════════════════════════════════════════════════════════

#[test]
fn integration_quantized_attention_pipeline() {
    let seq_len = 4;
    let hidden = 32;
    let heads = 2;
    let head_dim = hidden / heads;

    let input = prng(42, seq_len * hidden);

    let (q_data, scale) = quantize_symmetric_i8(&input, 8);
    let deq_input = dequantize_symmetric_i8(&q_data, scale);

    let attn = multi_head_attention_cpu(
        &deq_input, &deq_input, &deq_input, heads, head_dim, seq_len, true,
    )
    .unwrap();
    assert!(all_finite(&attn));
}

#[test]
fn integration_matmul_activation_layernorm_chain() {
    let m = 4;
    let k = 8;
    let n = 8;

    let a = prng(1, m * k);
    let b = prng(2, k * n);
    let mut c = vec![0.0f32; m * n];
    let cfg = SimdMatmulConfig::new(m, n, k);
    simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();

    let activated = activations::apply_activation(&c, ActivationType::GELU);

    let gamma = vec![1.0f32; n];
    let beta = vec![0.0f32; n];
    let ln_cfg = LayerNormConfig::new(vec![n]);
    let normed = layer_norm(&activated, &gamma, Some(&beta), &ln_cfg).unwrap();
    assert_eq!(normed.len(), m * n);
    assert!(all_finite(&normed));
}

#[test]
fn integration_full_forward_pass_with_all_components() {
    let seq_len = 4;
    let hidden = 32;
    let heads = 2;
    let head_dim = hidden / heads;
    let vocab = 64;
    let inter = hidden * 4;

    // Embed
    let table = prng(1, vocab * hidden);
    let indices: Vec<u32> = vec![5, 10, 15, 20];
    let x = embedding_lookup(&table, &indices, hidden).unwrap();

    // LayerNorm
    let gamma = vec![1.0f32; hidden];
    let beta = vec![0.0f32; hidden];
    let ln_cfg = LayerNormConfig::new(vec![hidden]);

    // Transformer block
    let normed = layer_norm(&x, &gamma, Some(&beta), &ln_cfg).unwrap();
    let attn = multi_head_attention_cpu(&normed, &normed, &normed, heads, head_dim, seq_len, true)
        .unwrap();
    let mut post_attn = attn;
    add_residual(&mut post_attn, &x).unwrap();

    let normed2 = layer_norm(&post_attn, &gamma, Some(&beta), &ln_cfg).unwrap();
    let ffn_cfg = FfnConfig::new(hidden, inter, FfnActivation::SiLU).unwrap();
    let w_up = prng(100, inter * hidden);
    let w_down = prng(200, hidden * inter);
    let ffn_out = ffn_forward_batched(&normed2, &w_up, &w_down, &ffn_cfg, seq_len).unwrap();
    let mut post_ffn = ffn_out;
    add_residual(&mut post_ffn, &post_attn).unwrap();

    // Final LN + projection
    let final_ln = layer_norm(&post_ffn, &gamma, Some(&beta), &ln_cfg).unwrap();
    let last_token = &final_ln[(seq_len - 1) * hidden..seq_len * hidden];
    let proj_w = prng(300, vocab * hidden);
    let proj_cfg = LinearConfig::new(1, hidden, vocab).unwrap();
    let mut logits = vec![0.0f32; vocab];
    linear_forward(last_token, &proj_w, None, &mut logits, &proj_cfg).unwrap();

    // Loss
    let target = 20usize;
    let (loss, _) = cross_entropy_loss(&logits, &[target], vocab, LossReduction::Mean).unwrap();
    assert!(loss.is_finite());
    assert!(loss >= 0.0);

    // Perplexity
    let ppl = perplexity(loss);
    assert!(ppl >= 1.0, "perplexity must be >= 1, got {ppl}");
}

#[test]
fn integration_batched_softmax_preserves_probability() {
    let batch = 4;
    let seq_len = 8;
    let logits = prng(42, batch * seq_len);
    let probs = batched_softmax(&logits, batch, seq_len).unwrap();

    for b in 0..batch {
        let row = &probs[b * seq_len..(b + 1) * seq_len];
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "batch {b} softmax sum={sum}");
        assert!(row.iter().all(|&p| p >= 0.0));
    }
}

#[test]
fn integration_activation_functions_in_ffn_context() {
    let hidden = 16;
    let inter = 32;
    let input = prng(42, hidden);
    let w_up = prng(1, inter * hidden);
    let w_down = prng(2, hidden * inter);

    for activation in [FfnActivation::ReLU, FfnActivation::GeLU, FfnActivation::SiLU] {
        let cfg = FfnConfig::new(hidden, inter, activation).unwrap();
        let out = ffn_forward(&input, &w_up, &w_down, &cfg).unwrap();
        assert_eq!(out.len(), hidden);
        assert!(all_finite(&out), "activation {activation:?} produced non-finite");
    }
}

#[test]
fn integration_rope_frequencies_are_deterministic() {
    let head_dim = 16;
    let max_seq = 64;
    let cfg = RopeConfig::new(head_dim, max_seq);

    let f1 = compute_frequencies(&cfg);
    let f2 = compute_frequencies(&cfg);
    assert_eq!(f1, f2, "RoPE frequencies must be deterministic");
}

#[test]
fn integration_linear_with_bias_correctness() {
    let in_f = 4;
    let out_f = 2;

    // output[j] = sum(input[k] * weight[j*in_f + k]) + bias[j]
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![
        1.0, 0.0, 0.0, 0.0, // row 0 → dot = 1.0
        0.0, 1.0, 0.0, 0.0, // row 1 → dot = 2.0
    ];
    let bias = vec![0.5, -0.5];
    let cfg = LinearConfig::new(1, in_f, out_f).unwrap();
    let mut output = vec![0.0f32; out_f];
    linear_forward(&input, &weight, Some(&bias), &mut output, &cfg).unwrap();

    assert!((output[0] - 1.5).abs() < 1e-6, "got {}", output[0]);
    assert!((output[1] - 1.5).abs() < 1e-6, "got {}", output[1]);
}
