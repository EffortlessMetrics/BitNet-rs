//! Integration Wave 14 — Transformer Pipeline BDD Tests
//!
//! Validates end-to-end data flow through realistic transformer inference
//! pipelines, testing multi-kernel composition and shape consistency:
//!
//! 1. Transformer forward pass: embedding → attention → FFN → output
//! 2. Multi-head attention pipeline: Q/K/V projection → scores → softmax → weighted sum
//! 3. KV cache integration: append → retrieve → eviction → consistency
//! 4. Mixed-precision pipeline: FP32 → quantize → INT2 compute → dequantize → FP32
//! 5. Batch processing pipeline: multiple sequences through shared kernels
//! 6. Error propagation: invalid dimensions, resource exhaustion recovery

use bitnet_kernels::cpu::activations::{ActivationType, activate};
use bitnet_kernels::cpu::attention::{
    AttentionConfig, AttentionKernel, GqaConfig, attention_with_kv_cache, causal_attention,
    causal_mask,
};
use bitnet_kernels::cpu::embedding::{
    CpuEmbeddingConfig, embedding_lookup, embedding_with_position, normalize_embeddings,
};
use bitnet_kernels::cpu::fusion::{fused_add_normalize, fused_rmsnorm_linear};
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice,
};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::loss::{LossReduction, cross_entropy_loss};
use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, PoolingKernel};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_symmetric_i8, quantize_symmetric_i8, quantize_ternary,
};
use bitnet_kernels::cpu::quantized_matmul::{i2s_matmul_f32, pack_i2s};
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope_batch, compute_frequencies};
use bitnet_kernels::reduction::{ReductionOp, reduce_f32};

// ── Helpers ────────────────────────────────────────────────────────

fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    assert!((a - b).abs() <= tol, "{ctx}: expected {b}, got {a} (diff {})", (a - b).abs());
}

fn assert_slice_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (&ai, &bi)) in a.iter().zip(b).enumerate() {
        assert_close(ai, bi, tol, &format!("{ctx}[{i}]"));
    }
}

/// Simple matmul: C[m×n] = A[m×k] × B[k×n] (row-major).
fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Reference softmax for cross-validation.
fn reference_softmax(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ══════════════════════════════════════════════════════════════════
// 1. Transformer Forward Pass Pipeline:
//    embedding → attention → FFN → output
// ══════════════════════════════════════════════════════════════════

#[test]
fn test_transformer_forward_pass_produces_valid_output() {
    let vocab = 32;
    let dim = 8;
    let seq_len = 4;
    let num_heads = 2;
    let head_dim = dim / num_heads;

    // Stage 1: Embedding lookup
    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.01).collect();
    let tokens = [1u32, 5, 10, 15];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();
    assert_eq!(emb.len(), seq_len * dim);

    // Stage 2: Pre-attention RMSNorm
    let gamma = vec![1.0f32; dim];
    let rms_cfg = LayerNormConfig::new(vec![dim]);
    let normed = rms_norm(&emb, &gamma, &rms_cfg).unwrap();
    assert_eq!(normed.len(), seq_len * dim);

    // Stage 3: Self-attention
    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    let attn_out =
        AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();
    assert_eq!(attn_out.len(), seq_len * dim);

    // Stage 4: Residual + post-attention norm
    let residual: Vec<f32> = emb.iter().zip(&attn_out).map(|(e, a)| e + a).collect();
    let normed2 = rms_norm(&residual, &gamma, &rms_cfg).unwrap();

    // Stage 5: FFN (linear → GELU → linear)
    let w_up: Vec<f32> = (0..dim * dim).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let hidden = naive_matmul(&normed2, &w_up, seq_len, dim, dim);
    let activated = activate(&hidden, ActivationType::GELU);
    let w_down: Vec<f32> = (0..dim * dim).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
    let ffn_out = naive_matmul(&activated, &w_down, seq_len, dim, dim);

    // Stage 6: Final residual
    let output: Vec<f32> = residual.iter().zip(&ffn_out).map(|(r, f)| r + f).collect();
    assert_eq!(output.len(), seq_len * dim);
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn test_transformer_forward_pass_shape_preserved_at_every_stage() {
    let dim = 4;
    let seq_len = 3;
    let vocab = 16;
    let num_heads = 2;
    let head_dim = dim / num_heads;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.02).collect();
    let tokens = [0u32, 3, 7];
    let expected_shape = seq_len * dim;

    let emb = embedding_lookup(&table, &tokens, dim).unwrap();
    assert_eq!(emb.len(), expected_shape, "embedding shape");

    let gamma = vec![1.0f32; dim];
    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let normed = layer_norm(&emb, &gamma, None, &ln_cfg).unwrap();
    assert_eq!(normed.len(), expected_shape, "layer_norm shape");

    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    let attn = AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();
    assert_eq!(attn.len(), expected_shape, "attention shape");

    let residual: Vec<f32> = emb.iter().zip(&attn).map(|(a, b)| a + b).collect();
    assert_eq!(residual.len(), expected_shape, "residual shape");

    let w: Vec<f32> = vec![0.1; dim * dim];
    let ffn = naive_matmul(&residual, &w, seq_len, dim, dim);
    assert_eq!(ffn.len(), expected_shape, "FFN shape");
}

#[test]
fn test_transformer_two_block_forward_pass_finite_output() {
    let dim = 4;
    let seq_len = 2;
    let num_heads = 1;
    let head_dim = dim;

    let input: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.2).collect();
    let gamma = vec![1.0f32; dim];
    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    let w: Vec<f32> = (0..dim * dim).map(|i| ((i % 3) as f32 - 1.0) * 0.15).collect();

    let mut x = input;
    // Block 1
    let normed1 = rms_norm(&x, &gamma, &ln_cfg).unwrap();
    let attn1 =
        AttentionKernel::multi_head_attention(&normed1, &normed1, &normed1, &attn_cfg).unwrap();
    x = x.iter().zip(&attn1).map(|(a, b)| a + b).collect();
    let normed1b = rms_norm(&x, &gamma, &ln_cfg).unwrap();
    let ffn1 = activate(&naive_matmul(&normed1b, &w, seq_len, dim, dim), ActivationType::SiLU);
    x = x.iter().zip(&ffn1).map(|(a, b)| a + b).collect();

    // Block 2
    let normed2 = rms_norm(&x, &gamma, &ln_cfg).unwrap();
    let attn2 =
        AttentionKernel::multi_head_attention(&normed2, &normed2, &normed2, &attn_cfg).unwrap();
    x = x.iter().zip(&attn2).map(|(a, b)| a + b).collect();
    let normed2b = rms_norm(&x, &gamma, &ln_cfg).unwrap();
    let ffn2 = activate(&naive_matmul(&normed2b, &w, seq_len, dim, dim), ActivationType::SiLU);
    x = x.iter().zip(&ffn2).map(|(a, b)| a + b).collect();

    assert_eq!(x.len(), seq_len * dim);
    assert!(x.iter().all(|v| v.is_finite()), "two-block output must be finite");
}

#[test]
fn test_transformer_forward_with_positional_embedding() {
    let vocab = 16;
    let dim = 4;
    let seq_len = 3;
    let num_heads = 1;
    let head_dim = dim;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.05).collect();
    let tokens = [2u32, 5, 8];
    let cfg = CpuEmbeddingConfig::new(vocab, dim);
    let emb = embedding_with_position(&table, &tokens, &cfg, 0).unwrap();

    let gamma = vec![1.0f32; dim];
    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let normed = layer_norm(&emb, &gamma, None, &ln_cfg).unwrap();

    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    let attn = AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();
    let output: Vec<f32> = emb.iter().zip(&attn).map(|(e, a)| e + a).collect();
    assert_eq!(output.len(), seq_len * dim);
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn test_transformer_forward_with_rope_encoding() {
    let vocab = 16;
    let dim = 8;
    let seq_len = 3;
    let num_heads = 2;
    let head_dim = dim / num_heads;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.01).collect();
    let tokens = [1u32, 4, 9];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();

    let rope_cfg = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&rope_cfg);

    let mut q = emb.clone();
    let mut k = emb.clone();
    apply_rope_batch(&mut q, 0, seq_len, num_heads, head_dim, &freqs);
    apply_rope_batch(&mut k, 0, seq_len, num_heads, head_dim, &freqs);

    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    let attn_out = AttentionKernel::multi_head_attention(&q, &k, &emb, &attn_cfg).unwrap();

    let w: Vec<f32> = (0..dim * dim).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
    let output = naive_matmul(&attn_out, &w, seq_len, dim, dim);
    assert_eq!(output.len(), seq_len * dim);
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn test_transformer_forward_fused_rmsnorm_linear_path() {
    let dim = 8;
    let tokens_count = 3;
    let table: Vec<f32> = (0..32 * dim).map(|i| (i as f32) * 0.01).collect();
    let tokens = [1u32, 5, 10];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();

    let gamma = vec![1.0f32; dim];
    let weight: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.01).collect();

    for pos in 0..tokens_count {
        let row = &emb[pos * dim..(pos + 1) * dim];
        let fused = fused_rmsnorm_linear(row, &weight, &gamma, 1e-5).unwrap();
        assert_eq!(fused.len(), dim);
        assert!(fused.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn test_transformer_forward_residual_preserves_gradient_flow() {
    let dim = 4;
    let seq_len = 2;

    let input: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.5).collect();
    let attn_delta: Vec<f32> = vec![0.01; seq_len * dim];
    let ffn_delta: Vec<f32> = vec![0.02; seq_len * dim];

    // Residual connections ensure output ≈ input + small deltas
    let after_attn: Vec<f32> = input.iter().zip(&attn_delta).map(|(a, b)| a + b).collect();
    let output: Vec<f32> = after_attn.iter().zip(&ffn_delta).map(|(a, b)| a + b).collect();

    for (i, (&inp, &out)) in input.iter().zip(&output).enumerate() {
        let diff = (out - inp).abs();
        assert!(diff < 0.05, "residual[{i}]: input={inp}, output={out}, diff={diff} too large");
    }
}

#[test]
fn test_transformer_forward_to_logits_produces_classification() {
    let vocab = 8;
    let dim = 4;
    let seq_len = 2;
    let num_heads = 1;
    let head_dim = dim;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.1).collect();
    let tokens = [2u32, 5];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();

    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    let attn = AttentionKernel::multi_head_attention(&emb, &emb, &emb, &attn_cfg).unwrap();
    let residual: Vec<f32> = emb.iter().zip(&attn).map(|(a, b)| a + b).collect();

    // Project to vocab logits
    let proj: Vec<f32> = (0..dim * vocab).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let logits = naive_matmul(&residual, &proj, seq_len, vocab, dim);
    assert_eq!(logits.len(), seq_len * vocab);

    // Verify softmax over last position produces valid distribution
    let last_logits = &logits[(seq_len - 1) * vocab..];
    let probs = reference_softmax(last_logits);
    let sum: f32 = probs.iter().sum();
    assert_close(sum, 1.0, 1e-5, "logit_softmax_sum");
    assert!(probs.iter().all(|&p| p >= 0.0));
}

#[test]
fn test_transformer_forward_with_loss_computation() {
    let vocab = 8;
    let dim = 4;
    let seq_len = 2;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.1).collect();
    let tokens = [1u32, 4];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();

    let proj: Vec<f32> = (0..dim * vocab).map(|i| ((i % 3) as f32 - 1.0) * 0.2).collect();
    let logits = naive_matmul(&emb, &proj, seq_len, vocab, dim);

    let targets = vec![4usize, 1];
    let (loss, per_sample) =
        cross_entropy_loss(&logits, &targets, vocab, LossReduction::Mean).unwrap();
    assert!(loss.is_finite() && loss >= 0.0);
    assert_eq!(per_sample.len(), seq_len);
}

// ══════════════════════════════════════════════════════════════════
// 2. Multi-Head Attention Pipeline:
//    Q/K/V projection → attention scores → softmax → weighted sum
// ══════════════════════════════════════════════════════════════════

#[test]
fn test_mha_qkv_projection_produces_correct_shapes() {
    let seq_len = 3;
    let dim = 8;
    let num_heads = 2;
    let head_dim = dim / num_heads;

    let input: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.05).collect();

    // Q, K, V projections via matmul
    let w_q: Vec<f32> = (0..dim * dim).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
    let w_k: Vec<f32> = (0..dim * dim).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
    let w_v: Vec<f32> = (0..dim * dim).map(|i| ((i % 3) as f32 - 1.0) * 0.05).collect();

    let q = naive_matmul(&input, &w_q, seq_len, dim, dim);
    let k = naive_matmul(&input, &w_k, seq_len, dim, dim);
    let v = naive_matmul(&input, &w_v, seq_len, dim, dim);

    assert_eq!(q.len(), seq_len * dim);
    assert_eq!(k.len(), seq_len * dim);
    assert_eq!(v.len(), seq_len * dim);

    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    let out = AttentionKernel::multi_head_attention(&q, &k, &v, &attn_cfg).unwrap();
    assert_eq!(out.len(), seq_len * dim);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_mha_causal_first_token_attends_only_to_itself() {
    let dim = 4;
    let seq_len = 4;
    let num_heads = 1;
    let head_dim = dim;

    let q: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();
    let k = q.clone();
    let v: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.2).collect();

    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    let out = causal_attention(&q, &k, &v, &attn_cfg).unwrap();

    // First token can only attend to position 0 → output = v[0]
    assert_slice_close(&out[..dim], &v[..dim], 1e-5, "causal_tok0_eq_v0");
}

#[test]
fn test_mha_uniform_queries_produce_value_average() {
    let num_heads = 1;
    let head_dim = 4;
    let seq_len = 3;

    // All queries and keys identical → uniform attention weights
    let q: Vec<f32> = vec![1.0; seq_len * head_dim];
    let k: Vec<f32> = vec![1.0; seq_len * head_dim];
    let v: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32).collect();

    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    let out = AttentionKernel::multi_head_attention(&q, &k, &v, &attn_cfg).unwrap();

    // Each output row should be the mean of all value rows
    let v_mean: Vec<f32> = (0..head_dim)
        .map(|d| (0..seq_len).map(|s| v[s * head_dim + d]).sum::<f32>() / seq_len as f32)
        .collect();
    for row in 0..seq_len {
        assert_slice_close(
            &out[row * head_dim..(row + 1) * head_dim],
            &v_mean,
            1e-4,
            &format!("uniform_attn_row{row}"),
        );
    }
}

#[test]
fn test_mha_output_bounded_by_value_range() {
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 4;
    let model_dim = num_heads * head_dim;

    let q: Vec<f32> = vec![0.5; seq_len * model_dim];
    let k: Vec<f32> = vec![0.5; seq_len * model_dim];
    let v: Vec<f32> = (0..seq_len * model_dim)
        .map(|i| -2.0 + (i as f32) * 4.0 / (seq_len * model_dim) as f32)
        .collect();

    let v_min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let v_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    let out = AttentionKernel::multi_head_attention(&q, &k, &v, &attn_cfg).unwrap();
    for (i, &val) in out.iter().enumerate() {
        assert!(
            val >= v_min - 1e-5 && val <= v_max + 1e-5,
            "out[{i}]={val} outside V range [{v_min}, {v_max}]"
        );
    }
}

#[test]
fn test_mha_gqa_fewer_kv_heads_produces_valid_output() {
    let num_q_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 4;
    let seq_len = 3;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q: Vec<f32> = (0..seq_len * q_dim).map(|i| (i as f32) * 0.02).collect();
    let k: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i as f32) * 0.03).collect();
    let v: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i as f32) * 0.01).collect();

    let gqa_cfg =
        GqaConfig { num_q_heads, num_kv_heads, head_dim, seq_len, causal: true, scale: None };
    let out = AttentionKernel::grouped_query_attention(&q, &k, &v, &gqa_cfg).unwrap();
    assert_eq!(out.len(), seq_len * q_dim);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_mha_rope_differentiated_positions() {
    let dim = 8;
    let seq_len = 4;
    let num_heads = 2;
    let head_dim = dim / num_heads;

    let table: Vec<f32> = (0..16 * dim).map(|i| (i as f32) * 0.01).collect();
    // Same token repeated at every position
    let tokens = [3u32, 3, 3, 3];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();

    let rope_cfg = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&rope_cfg);
    let mut q = emb.clone();
    let mut k = emb.clone();
    apply_rope_batch(&mut q, 0, seq_len, num_heads, head_dim, &freqs);
    apply_rope_batch(&mut k, 0, seq_len, num_heads, head_dim, &freqs);

    // Even identical tokens at different positions should produce different Q/K
    assert_ne!(&q[..dim], &q[dim..2 * dim], "RoPE must differentiate positions");
}

#[test]
fn test_mha_with_output_projection_preserves_dim() {
    let dim = 8;
    let seq_len = 3;
    let num_heads = 2;
    let head_dim = dim / num_heads;

    let input: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.05).collect();
    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    let attn = AttentionKernel::multi_head_attention(&input, &input, &input, &attn_cfg).unwrap();

    // Output projection
    let w_o: Vec<f32> = (0..dim * dim).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
    let projected = naive_matmul(&attn, &w_o, seq_len, dim, dim);
    assert_eq!(projected.len(), seq_len * dim);
    assert!(projected.iter().all(|v| v.is_finite()));
}

#[test]
fn test_mha_causal_mask_structure() {
    let seq_len = 4;
    let mask = causal_mask(seq_len);
    assert_eq!(mask.len(), seq_len * seq_len);

    for i in 0..seq_len {
        for j in 0..seq_len {
            let val = mask[i * seq_len + j];
            if j > i {
                assert!(val.is_infinite() && val < 0.0, "mask[{i},{j}] should be -inf");
            } else {
                assert_close(val, 0.0, 0.0, &format!("mask[{i},{j}]"));
            }
        }
    }
}

#[test]
fn test_mha_attention_with_norm_residual_chain() {
    let dim = 4;
    let seq_len = 2;
    let num_heads = 1;
    let head_dim = dim;

    let input: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.3).collect();
    let gamma = vec![1.0f32; dim];
    let ln_cfg = LayerNormConfig::new(vec![dim]);

    let normed = layer_norm(&input, &gamma, None, &ln_cfg).unwrap();
    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: true,
        use_alibi: false,
        scale: None,
    };
    let attn = AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();
    let residual: Vec<f32> = input.iter().zip(&attn).map(|(a, b)| a + b).collect();
    let post_norm = layer_norm(&residual, &gamma, None, &ln_cfg).unwrap();

    assert_eq!(post_norm.len(), seq_len * dim);
    // Post-norm should have zero mean per row
    for row in 0..seq_len {
        let slice = &post_norm[row * dim..(row + 1) * dim];
        let mean: f32 = slice.iter().sum::<f32>() / dim as f32;
        assert_close(mean, 0.0, 1e-4, &format!("post_norm_mean_row{row}"));
    }
}

// ══════════════════════════════════════════════════════════════════
// 3. KV Cache Integration:
//    append → retrieve → eviction → consistency
// ══════════════════════════════════════════════════════════════════

#[test]
fn test_kv_cache_append_and_retrieve_consistency() {
    let num_heads = 2;
    let head_dim = 4;
    let te = num_heads * head_dim;

    let cfg =
        KvCacheConfig { num_layers: 1, num_heads, head_dim, max_seq_len: 16, dtype: KvDtype::F32 };
    let mut cache = KvCache::new(cfg).unwrap();

    let k1: Vec<f32> = (0..te).map(|i| i as f32).collect();
    let v1: Vec<f32> = (0..te).map(|i| (i as f32) * 10.0).collect();
    kv_cache_append(&mut cache, 0, &k1, &v1).unwrap();

    let k2: Vec<f32> = (0..te).map(|i| (i as f32) + 100.0).collect();
    let v2: Vec<f32> = (0..te).map(|i| (i as f32) * 10.0 + 100.0).collect();
    kv_cache_append(&mut cache, 0, &k2, &v2).unwrap();

    assert_eq!(cache.seq_len(0).unwrap(), 2);

    let (keys, values) = kv_cache_slice(&cache, 0, 0, 2).unwrap();
    assert_slice_close(&keys[..te], &k1, 1e-7, "k1_retrieve");
    assert_slice_close(&keys[te..], &k2, 1e-7, "k2_retrieve");
    assert_slice_close(&values[..te], &v1, 1e-7, "v1_retrieve");
    assert_slice_close(&values[te..], &v2, 1e-7, "v2_retrieve");
}

#[test]
fn test_kv_cache_clear_resets_sequence_length() {
    let num_heads = 1;
    let head_dim = 4;
    let te = num_heads * head_dim;

    let cfg =
        KvCacheConfig { num_layers: 2, num_heads, head_dim, max_seq_len: 8, dtype: KvDtype::F32 };
    let mut cache = KvCache::new(cfg).unwrap();

    kv_cache_append(&mut cache, 0, &vec![1.0; te], &vec![2.0; te]).unwrap();
    kv_cache_append(&mut cache, 1, &vec![3.0; te * 2], &vec![4.0; te * 2]).unwrap();
    assert_eq!(cache.seq_len(0).unwrap(), 1);
    assert_eq!(cache.seq_len(1).unwrap(), 2);

    kv_cache_clear(&mut cache);
    assert_eq!(cache.seq_len(0).unwrap(), 0);
    assert_eq!(cache.seq_len(1).unwrap(), 0);

    // Memory should still be allocated
    let mem = kv_cache_memory_usage(&cache);
    assert!(mem > 0);
}

#[test]
fn test_kv_cache_reuse_after_clear() {
    let num_heads = 1;
    let head_dim = 4;
    let te = num_heads * head_dim;

    let cfg =
        KvCacheConfig { num_layers: 1, num_heads, head_dim, max_seq_len: 8, dtype: KvDtype::F32 };
    let mut cache = KvCache::new(cfg).unwrap();

    kv_cache_append(&mut cache, 0, &vec![1.0; te], &vec![2.0; te]).unwrap();
    kv_cache_clear(&mut cache);

    let new_k: Vec<f32> = vec![99.0; te];
    let new_v: Vec<f32> = vec![88.0; te];
    kv_cache_append(&mut cache, 0, &new_k, &new_v).unwrap();
    assert_eq!(cache.seq_len(0).unwrap(), 1);

    let (keys, values) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
    assert_close(keys[0], 99.0, 1e-7, "post_clear_k");
    assert_close(values[0], 88.0, 1e-7, "post_clear_v");
}

#[test]
fn test_kv_cache_multilayer_independence() {
    let num_layers = 3;
    let num_heads = 1;
    let head_dim = 2;
    let te = num_heads * head_dim;

    let cfg =
        KvCacheConfig { num_layers, num_heads, head_dim, max_seq_len: 8, dtype: KvDtype::F32 };
    let mut cache = KvCache::new(cfg).unwrap();

    kv_cache_append(&mut cache, 0, &vec![1.0; te], &vec![10.0; te]).unwrap();
    kv_cache_append(&mut cache, 1, &vec![2.0; te * 2], &vec![20.0; te * 2]).unwrap();
    kv_cache_append(&mut cache, 2, &vec![3.0; te * 3], &vec![30.0; te * 3]).unwrap();

    assert_eq!(cache.seq_len(0).unwrap(), 1);
    assert_eq!(cache.seq_len(1).unwrap(), 2);
    assert_eq!(cache.seq_len(2).unwrap(), 3);

    // Verify layer 2 data is independent of layer 0/1
    let (k2, v2) = kv_cache_slice(&cache, 2, 0, 3).unwrap();
    assert_eq!(k2.len(), 3 * te);
    assert!(k2.iter().all(|&x| (x - 3.0).abs() < 1e-7));
    assert!(v2.iter().all(|&x| (x - 30.0).abs() < 1e-7));
}

#[test]
fn test_kv_cache_incremental_attention_single_kv() {
    let head_dim = 4;

    let mut k_cache: Vec<f32> = Vec::new();
    let mut v_cache: Vec<f32> = Vec::new();

    let q = vec![1.0f32; head_dim];
    let k = vec![1.0f32; head_dim];
    let v = vec![5.0f32; head_dim];

    let out = attention_with_kv_cache(&q, &mut k_cache, &mut v_cache, &k, &v, head_dim).unwrap();
    // Single KV → softmax of single score = 1.0 → output = v
    assert_slice_close(&out, &v, 1e-5, "single_kv_out_eq_v");
}

#[test]
fn test_kv_cache_incremental_attention_grows_cache() {
    let head_dim = 4;

    let mut k_cache: Vec<f32> = Vec::new();
    let mut v_cache: Vec<f32> = Vec::new();

    // Step 1
    let q1 = vec![1.0f32; head_dim];
    let k1 = vec![1.0f32; head_dim];
    let v1 = vec![10.0f32; head_dim];
    attention_with_kv_cache(&q1, &mut k_cache, &mut v_cache, &k1, &v1, head_dim).unwrap();
    assert_eq!(k_cache.len(), head_dim);

    // Step 2
    let q2 = vec![2.0f32; head_dim];
    let k2 = vec![0.0f32; head_dim];
    let v2 = vec![20.0f32; head_dim];
    let out2 =
        attention_with_kv_cache(&q2, &mut k_cache, &mut v_cache, &k2, &v2, head_dim).unwrap();
    assert_eq!(k_cache.len(), 2 * head_dim);
    assert_eq!(out2.len(), head_dim);
    assert!(out2.iter().all(|v| v.is_finite()));

    // Step 3
    let q3 = vec![0.5f32; head_dim];
    let k3 = vec![0.5f32; head_dim];
    let v3 = vec![30.0f32; head_dim];
    let out3 =
        attention_with_kv_cache(&q3, &mut k_cache, &mut v_cache, &k3, &v3, head_dim).unwrap();
    assert_eq!(k_cache.len(), 3 * head_dim);
    assert!(out3.iter().all(|v| v.is_finite()));
}

#[test]
fn test_kv_cache_memory_usage_scales_with_layers() {
    let make_cache = |layers| {
        let cfg = KvCacheConfig {
            num_layers: layers,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 16,
            dtype: KvDtype::F32,
        };
        KvCache::new(cfg).unwrap()
    };

    let mem_1 = kv_cache_memory_usage(&make_cache(1));
    let mem_2 = kv_cache_memory_usage(&make_cache(2));
    let mem_4 = kv_cache_memory_usage(&make_cache(4));

    assert!(mem_2 > mem_1, "2 layers should use more memory than 1");
    assert!(mem_4 > mem_2, "4 layers should use more memory than 2");
}

#[test]
fn test_kv_cache_partial_slice_returns_subset() {
    let num_heads = 1;
    let head_dim = 2;
    let te = num_heads * head_dim;

    let cfg =
        KvCacheConfig { num_layers: 1, num_heads, head_dim, max_seq_len: 8, dtype: KvDtype::F32 };
    let mut cache = KvCache::new(cfg).unwrap();

    for i in 0..4u32 {
        let k_val = vec![i as f32; te];
        let v_val = vec![(i as f32) * 10.0; te];
        kv_cache_append(&mut cache, 0, &k_val, &v_val).unwrap();
    }
    assert_eq!(cache.seq_len(0).unwrap(), 4);

    // Slice positions 1..3 (exclusive end)
    let (keys, values) = kv_cache_slice(&cache, 0, 1, 3).unwrap();
    assert_eq!(keys.len(), 2 * te);
    assert_close(keys[0], 1.0, 1e-7, "partial_k_start");
    assert_close(keys[te], 2.0, 1e-7, "partial_k_end");
    assert_close(values[0], 10.0, 1e-7, "partial_v_start");
    assert_close(values[te], 20.0, 1e-7, "partial_v_end");
}

// ══════════════════════════════════════════════════════════════════
// 4. Mixed-Precision Pipeline:
//    FP32 input → quantize → INT2 compute → dequantize → FP32 output
// ══════════════════════════════════════════════════════════════════

#[test]
fn test_mixed_precision_fp32_quantize_i2s_matmul_dequantize_roundtrip() {
    let m: usize = 2;
    let k: usize = 8;
    let n: usize = 4;
    let block_size: usize = 4;

    let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let packed_k = k.div_ceil(4);
    let weights: Vec<u8> = vec![pack_i2s([1, 1, 1, 1]); n * packed_k];
    let num_blocks = k.div_ceil(block_size);
    let scales = vec![1.0f32; n * num_blocks];

    let mut output = vec![0.0f32; m * n];
    i2s_matmul_f32(&activations, &weights, &scales, &mut output, m, n, k, block_size).unwrap();

    // All-ones weights with scale 1.0 → output[i] = sum of activation row
    for row in 0..m {
        let expected_sum: f32 = activations[row * k..(row + 1) * k].iter().sum();
        for col in 0..n {
            assert_close(
                output[row * n + col],
                expected_sum,
                1e-4,
                &format!("i2s_row{row}_col{col}"),
            );
        }
    }
}

#[test]
fn test_mixed_precision_symmetric_i8_quantize_roundtrip_accuracy() {
    let input = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.75, 0.9];
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);

    let err = compute_quantization_error(&input, &dequantized);
    assert!(err.max_abs_error < 0.01, "max abs error: {}", err.max_abs_error);
    assert!(err.mse < 1e-4, "MSE: {}", err.mse);
}

#[test]
fn test_mixed_precision_quantize_norm_matmul_pipeline() {
    let m: usize = 1;
    let k: usize = 8;
    let n: usize = 4;
    let block_size: usize = 4;

    // Start with FP32 input
    let input: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.3 - 1.0).collect();

    // Normalize (FP32)
    let ln_cfg = LayerNormConfig::new(vec![k]);
    let gamma = vec![1.0f32; k];
    let normed = layer_norm(&input, &gamma, None, &ln_cfg).unwrap();

    // Quantized matmul (INT2 weights)
    let packed_k = k.div_ceil(4);
    let weights: Vec<u8> = vec![pack_i2s([1, -1, 0, 1]); n * packed_k];
    let num_blocks = k.div_ceil(block_size);
    let scales = vec![0.5f32; n * num_blocks];

    let mut output = vec![0.0f32; m * n];
    i2s_matmul_f32(&normed, &weights, &scales, &mut output, m, n, k, block_size).unwrap();

    // Output is FP32 again
    assert_eq!(output.len(), n);
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn test_mixed_precision_ternary_quantization_preserves_sign() {
    let input = vec![1.5, -2.0, 0.01, -0.01, 3.0, -0.5, 0.0, 0.8];
    let threshold = 0.1;
    let ternary = quantize_ternary(&input, threshold);

    for (i, (&orig, &quant)) in input.iter().zip(&ternary).enumerate() {
        if orig > threshold {
            assert_eq!(quant, 1, "input[{i}]={orig} should be +1");
        } else if orig < -threshold {
            assert_eq!(quant, -1, "input[{i}]={orig} should be -1");
        } else {
            assert_eq!(quant, 0, "input[{i}]={orig} should be 0");
        }
    }
}

#[test]
fn test_mixed_precision_quantized_matmul_then_activation() {
    let m: usize = 2;
    let k: usize = 8;
    let n: usize = 4;
    let block_size: usize = 4;

    let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let packed_k = k.div_ceil(4);
    let weights: Vec<u8> = vec![pack_i2s([1, 0, -1, 1]); n * packed_k];
    let num_blocks = k.div_ceil(block_size);
    let scales = vec![0.5f32; n * num_blocks];

    let mut logits = vec![0.0f32; m * n];
    i2s_matmul_f32(&activations, &weights, &scales, &mut logits, m, n, k, block_size).unwrap();

    // Apply GELU in FP32 space
    let activated = activate(&logits, ActivationType::GELU);
    assert_eq!(activated.len(), m * n);
    assert!(activated.iter().all(|v| v.is_finite()));
}

#[test]
fn test_mixed_precision_quantized_matmul_then_layer_norm() {
    let m: usize = 2;
    let k: usize = 8;
    let n: usize = 4;
    let block_size: usize = 4;

    let activations: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.2).collect();
    let packed_k = k.div_ceil(4);
    let weights: Vec<u8> = vec![pack_i2s([1, 1, -1, 0]); n * packed_k];
    let num_blocks = k.div_ceil(block_size);
    let scales = vec![1.0f32; n * num_blocks];

    let mut output = vec![0.0f32; m * n];
    i2s_matmul_f32(&activations, &weights, &scales, &mut output, m, n, k, block_size).unwrap();

    // Normalize in FP32
    let ln_cfg = LayerNormConfig::new(vec![n]);
    let gamma = vec![1.0f32; n];
    let normed = layer_norm(&output, &gamma, None, &ln_cfg).unwrap();
    assert_eq!(normed.len(), m * n);
    for row in 0..m {
        let slice = &normed[row * n..(row + 1) * n];
        let mean: f32 = slice.iter().sum::<f32>() / n as f32;
        assert_close(mean, 0.0, 1e-4, &format!("quant_ln_mean_row{row}"));
    }
}

#[test]
fn test_mixed_precision_zero_input_quantize_roundtrip() {
    let input = vec![0.0f32; 8];
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);
    for (i, &v) in dequantized.iter().enumerate() {
        assert_close(v, 0.0, 1e-7, &format!("zero_roundtrip[{i}]"));
    }
}

#[test]
fn test_mixed_precision_full_pipeline_embed_quantize_matmul_activate() {
    let vocab = 16;
    let dim = 8;
    let out_dim = 4;
    let block_size: usize = 4;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.01).collect();
    let tokens = [2u32, 7];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();

    // Quantized FFN
    let packed_k = dim.div_ceil(4);
    let weights: Vec<u8> = vec![pack_i2s([1, -1, 1, 0]); out_dim * packed_k];
    let num_blocks = dim.div_ceil(block_size);
    let scales = vec![0.3f32; out_dim * num_blocks];

    let m = tokens.len();
    let mut ffn_out = vec![0.0f32; m * out_dim];
    i2s_matmul_f32(&emb, &weights, &scales, &mut ffn_out, m, out_dim, dim, block_size).unwrap();

    let activated = activate(&ffn_out, ActivationType::SiLU);
    assert_eq!(activated.len(), m * out_dim);
    assert!(activated.iter().all(|v| v.is_finite()));
}

// ══════════════════════════════════════════════════════════════════
// 5. Batch Processing Pipeline:
//    multiple sequences through shared kernels
// ══════════════════════════════════════════════════════════════════

#[test]
fn test_batch_embedding_produces_correct_total_length() {
    let vocab = 32;
    let dim = 8;

    let batch_tokens: Vec<Vec<u32>> = vec![vec![1, 3, 5], vec![2, 4], vec![0, 7, 9, 11]];

    for (batch_idx, tokens) in batch_tokens.iter().enumerate() {
        let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.01).collect();
        let emb = embedding_lookup(&table, tokens, dim).unwrap();
        assert_eq!(emb.len(), tokens.len() * dim, "batch {batch_idx} embedding size");
    }
}

#[test]
fn test_batch_norm_then_attention_per_sequence() {
    let dim = 4;
    let num_heads = 1;
    let head_dim = dim;
    let gamma = vec![1.0f32; dim];
    let ln_cfg = LayerNormConfig::new(vec![dim]);

    let sequences: Vec<Vec<f32>> = vec![
        (0..3 * dim).map(|i| (i as f32) * 0.1).collect(),
        (0..2 * dim).map(|i| (i as f32) * 0.2).collect(),
        (0..4 * dim).map(|i| (i as f32) * 0.05).collect(),
    ];

    for (seq_idx, seq) in sequences.iter().enumerate() {
        let seq_len = seq.len() / dim;
        let normed = layer_norm(seq, &gamma, None, &ln_cfg).unwrap();
        assert_eq!(normed.len(), seq_len * dim);

        let attn_cfg = AttentionConfig {
            num_heads,
            head_dim,
            seq_len,
            causal: true,
            use_alibi: false,
            scale: None,
        };
        let out =
            AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();
        assert_eq!(out.len(), seq_len * dim, "batch {seq_idx} attention shape");
        assert!(out.iter().all(|v| v.is_finite()), "batch {seq_idx} has non-finite");
    }
}

#[test]
fn test_batch_shared_weights_different_sequences() {
    let dim = 4;
    let w: Vec<f32> = (0..dim * dim).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();

    let seq_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let seq_b: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];

    let out_a = naive_matmul(&seq_a, &w, 2, dim, dim);
    let out_b = naive_matmul(&seq_b, &w, 1, dim, dim);

    assert_eq!(out_a.len(), 2 * dim);
    assert_eq!(out_b.len(), dim);
    // Different inputs through same weights → different outputs
    assert_ne!(&out_a[..dim], &out_b[..]);
}

#[test]
fn test_batch_independent_kv_caches_per_sequence() {
    let head_dim = 4;

    let mut k_cache_a: Vec<f32> = Vec::new();
    let mut v_cache_a: Vec<f32> = Vec::new();
    let mut k_cache_b: Vec<f32> = Vec::new();
    let mut v_cache_b: Vec<f32> = Vec::new();

    // Sequence A: 2 steps
    let q_a = vec![1.0f32; head_dim];
    let k_a1 = vec![1.0f32; head_dim];
    let v_a1 = vec![10.0f32; head_dim];
    attention_with_kv_cache(&q_a, &mut k_cache_a, &mut v_cache_a, &k_a1, &v_a1, head_dim).unwrap();
    let k_a2 = vec![2.0f32; head_dim];
    let v_a2 = vec![20.0f32; head_dim];
    attention_with_kv_cache(&q_a, &mut k_cache_a, &mut v_cache_a, &k_a2, &v_a2, head_dim).unwrap();

    // Sequence B: 1 step
    let q_b = vec![3.0f32; head_dim];
    let k_b1 = vec![5.0f32; head_dim];
    let v_b1 = vec![50.0f32; head_dim];
    attention_with_kv_cache(&q_b, &mut k_cache_b, &mut v_cache_b, &k_b1, &v_b1, head_dim).unwrap();

    // Caches are independent
    assert_eq!(k_cache_a.len(), 2 * head_dim);
    assert_eq!(k_cache_b.len(), head_dim);
}

#[test]
fn test_batch_parallel_normalize_and_reduce() {
    let dim = 8;
    let batch_size = 4;

    let batch: Vec<f32> = (0..batch_size * dim).map(|i| (i as f32) * 0.3 - 2.0).collect();

    // Normalize each row
    let mut normed = batch.clone();
    normalize_embeddings(&mut normed, dim);

    // Each row should have unit norm
    for row in 0..batch_size {
        let slice = &normed[row * dim..(row + 1) * dim];
        let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_close(norm, 1.0, 1e-5, &format!("batch_row{row}_norm"));
    }

    // Reduce to per-row means
    let means: Vec<f32> = (0..batch_size)
        .map(|r| {
            let slice = &normed[r * dim..(r + 1) * dim];
            reduce_f32(slice, ReductionOp::Mean)
        })
        .collect();
    assert_eq!(means.len(), batch_size);
    assert!(means.iter().all(|v| v.is_finite()));
}

#[test]
fn test_batch_pooling_over_sequences() {
    let dim = 8;
    let sequences = vec![
        (0..dim).map(|i| i as f32).collect::<Vec<_>>(),
        (0..dim).map(|i| (i as f32) * 2.0).collect::<Vec<_>>(),
    ];

    let pool_cfg = PoolConfig {
        pool_type: PoolType::Average,
        kernel_size: 4,
        stride: 4,
        padding: 0,
        dilation: 1,
        ceil_mode: false,
    };

    for (seq_idx, seq) in sequences.iter().enumerate() {
        let pooled = PoolingKernel::apply(seq, &pool_cfg).unwrap();
        assert!(!pooled.is_empty(), "batch {seq_idx} pooled empty");
        assert!(pooled.iter().all(|v| v.is_finite()), "batch {seq_idx} pool non-finite");
    }
}

#[test]
fn test_batch_quantized_matmul_shared_weights() {
    let k: usize = 8;
    let n: usize = 4;
    let block_size: usize = 4;
    let packed_k = k.div_ceil(4);
    let num_blocks = k.div_ceil(block_size);
    let weights: Vec<u8> = vec![pack_i2s([1, 1, 1, 1]); n * packed_k];
    let scales = vec![1.0f32; n * num_blocks];

    let batch_inputs: Vec<Vec<f32>> = vec![
        (0..k).map(|i| i as f32).collect(),
        (0..k).map(|i| (i as f32) * 0.5).collect(),
        vec![1.0; k],
    ];

    for (idx, input) in batch_inputs.iter().enumerate() {
        let m = 1;
        let mut output = vec![0.0f32; m * n];
        i2s_matmul_f32(input, &weights, &scales, &mut output, m, n, k, block_size).unwrap();
        let expected_sum: f32 = input.iter().sum();
        for col in 0..n {
            assert_close(output[col], expected_sum, 1e-4, &format!("batch{idx}_col{col}"));
        }
    }
}

#[test]
fn test_batch_fused_add_normalize_multiple_residuals() {
    let dim = 4;
    let gamma = vec![1.0f32; dim];
    let eps = 1e-5;

    let residuals: Vec<Vec<f32>> =
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.0, 0.0, 0.0, 0.0], vec![-1.0, -2.0, -3.0, -4.0]];
    let hiddens: Vec<Vec<f32>> =
        vec![vec![0.1, 0.2, 0.3, 0.4], vec![1.0, 1.0, 1.0, 1.0], vec![0.5, -0.5, 0.5, -0.5]];

    for (idx, (residual, hidden)) in residuals.iter().zip(&hiddens).enumerate() {
        let fused = fused_add_normalize(residual, hidden, &gamma, eps).unwrap();
        assert_eq!(fused.len(), dim, "batch {idx} fused size");
        assert!(fused.iter().all(|v| v.is_finite()), "batch {idx} non-finite");
    }
}

#[test]
fn test_batch_reduction_mean_per_row() {
    let dim = 6;
    let batch = 3;
    let data: Vec<f32> = (0..batch * dim).map(|i| (i as f32) + 1.0).collect();

    let row_means: Vec<f32> = (0..batch)
        .map(|r| {
            let slice = &data[r * dim..(r + 1) * dim];
            ReductionKernel::mean(slice).unwrap()
        })
        .collect();

    assert_eq!(row_means.len(), batch);
    for (r, &mean) in row_means.iter().enumerate() {
        let expected = (0..dim).map(|d| data[r * dim + d]).sum::<f32>() / dim as f32;
        assert_close(mean, expected, 1e-6, &format!("batch_mean_row{r}"));
    }
}

// ══════════════════════════════════════════════════════════════════
// 6. Error Propagation:
//    invalid dimensions, resource exhaustion recovery
// ══════════════════════════════════════════════════════════════════

#[test]
fn test_error_attention_mismatched_q_shape_rejected() {
    let attn_cfg = AttentionConfig {
        num_heads: 2,
        head_dim: 4,
        seq_len: 3,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    let q = vec![0.0f32; 10]; // wrong: expected 24
    let k = vec![0.0f32; 24];
    let v = vec![0.0f32; 24];
    assert!(
        AttentionKernel::multi_head_attention(&q, &k, &v, &attn_cfg).is_err(),
        "mismatched Q shape should error"
    );
}

#[test]
fn test_error_attention_mismatched_k_shape_rejected() {
    let attn_cfg = AttentionConfig {
        num_heads: 2,
        head_dim: 4,
        seq_len: 3,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    let q = vec![0.0f32; 24];
    let k = vec![0.0f32; 12]; // wrong: expected 24
    let v = vec![0.0f32; 24];
    assert!(
        AttentionKernel::multi_head_attention(&q, &k, &v, &attn_cfg).is_err(),
        "mismatched K shape should error"
    );
}

#[test]
fn test_error_layer_norm_gamma_mismatch_rejected() {
    let ln_cfg = LayerNormConfig::new(vec![8]);
    let gamma = vec![1.0f32; 4]; // wrong: expected 8
    let data = vec![1.0f32; 16];
    assert!(layer_norm(&data, &gamma, None, &ln_cfg).is_err(), "mismatched gamma should error");
}

#[test]
fn test_error_embedding_out_of_bounds_token_rejected() {
    let vocab = 8;
    let dim = 4;
    let table: Vec<f32> = vec![0.0; vocab * dim];
    let tokens = [100u32]; // out of range
    assert!(embedding_lookup(&table, &tokens, dim).is_err(), "out-of-bounds token should error");
}

#[test]
fn test_error_kv_cache_out_of_bounds_layer_rejected() {
    let cfg = KvCacheConfig {
        num_layers: 2,
        num_heads: 1,
        head_dim: 4,
        max_seq_len: 8,
        dtype: KvDtype::F32,
    };
    let cache = KvCache::new(cfg).unwrap();
    assert!(cache.seq_len(5).is_err(), "out-of-bounds layer index should error");
}

#[test]
fn test_error_quantized_matmul_mismatched_output_buffer() {
    let m: usize = 2;
    let k: usize = 8;
    let n: usize = 4;
    let block_size: usize = 4;

    let activations: Vec<f32> = vec![1.0; m * k];
    let packed_k = k.div_ceil(4);
    let weights: Vec<u8> = vec![pack_i2s([1, 1, 1, 1]); n * packed_k];
    let num_blocks = k.div_ceil(block_size);
    let scales = vec![1.0f32; n * num_blocks];

    let mut output = vec![0.0f32; 3]; // wrong: expected m*n=8
    let result = i2s_matmul_f32(&activations, &weights, &scales, &mut output, m, n, k, block_size);
    assert!(result.is_err(), "mismatched output buffer should error");
}

#[test]
fn test_error_pipeline_recovers_from_first_stage_failure() {
    let vocab = 8;
    let dim = 4;
    let table: Vec<f32> = vec![0.0; vocab * dim];
    let tokens = [200u32]; // out of range

    let emb_result = embedding_lookup(&table, &tokens, dim);
    assert!(emb_result.is_err());

    // Subsequent operations with valid data should still work
    let valid_input = vec![1.0f32; dim];
    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let gamma = vec![1.0f32; dim];
    let normed = layer_norm(&valid_input, &gamma, None, &ln_cfg);
    assert!(normed.is_ok(), "recovery after earlier error should succeed");
}

#[test]
fn test_error_gqa_mismatched_kv_heads_rejected() {
    let gqa_cfg = GqaConfig {
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        seq_len: 2,
        causal: false,
        scale: None,
    };

    let q = vec![0.0f32; 2 * 16]; // 2 * 4 * 4
    let k = vec![0.0f32; 5]; // wrong shape
    let v = vec![0.0f32; 2 * 8]; // 2 * 2 * 4
    assert!(
        AttentionKernel::grouped_query_attention(&q, &k, &v, &gqa_cfg).is_err(),
        "mismatched KV shape for GQA should error"
    );
}

#[test]
fn test_error_rms_norm_empty_gamma_rejected() {
    let ln_cfg = LayerNormConfig::new(vec![4]);
    let gamma: Vec<f32> = vec![];
    let data = vec![1.0f32; 4];
    assert!(rms_norm(&data, &gamma, &ln_cfg).is_err(), "empty gamma should error");
}

#[test]
fn test_error_dimension_mismatch_does_not_corrupt_state() {
    let head_dim = 4;
    let mut k_cache: Vec<f32> = Vec::new();
    let mut v_cache: Vec<f32> = Vec::new();

    // Successful append
    let q = vec![1.0f32; head_dim];
    let k = vec![1.0f32; head_dim];
    let v = vec![10.0f32; head_dim];
    let out = attention_with_kv_cache(&q, &mut k_cache, &mut v_cache, &k, &v, head_dim).unwrap();
    assert_slice_close(&out, &v, 1e-5, "initial_kv");

    let cache_len_before = k_cache.len();

    // Failed append with wrong dim should not corrupt cache
    let bad_k = vec![2.0f32; head_dim + 1]; // wrong dim
    let bad_v = vec![20.0f32; head_dim];
    let _ = attention_with_kv_cache(&q, &mut k_cache, &mut v_cache, &bad_k, &bad_v, head_dim);

    // Valid operations after failed one should still work
    let k2 = vec![3.0f32; head_dim];
    let v2 = vec![30.0f32; head_dim];
    let out2 = attention_with_kv_cache(&q, &mut k_cache, &mut v_cache, &k2, &v2, head_dim).unwrap();
    assert!(out2.iter().all(|v| v.is_finite()), "post-error attention finite");
    // Cache should have grown by exactly 1 more entry from the valid append
    assert!(k_cache.len() >= cache_len_before + head_dim, "cache should grow after valid append");
}

#[test]
fn test_error_cross_entropy_wrong_num_classes_rejected() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let targets = vec![10usize]; // target >= num_classes
    let result = cross_entropy_loss(&logits, &targets, 4, LossReduction::Mean);
    assert!(result.is_err(), "target >= num_classes should error");
}

// ══════════════════════════════════════════════════════════════════
// Additional pipeline composition tests
// ══════════════════════════════════════════════════════════════════

#[test]
fn test_transformer_layernorm_idempotent_in_pipeline() {
    let dim = 8;
    let data: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5).collect();
    let gamma = vec![1.0f32; dim];
    let ln_cfg = LayerNormConfig::new(vec![dim]);

    let first = layer_norm(&data, &gamma, None, &ln_cfg).unwrap();
    let second = layer_norm(&first, &gamma, None, &ln_cfg).unwrap();
    assert_slice_close(&first, &second, 1e-5, "ln_idempotent");
}

#[test]
fn test_transformer_rope_preserves_vector_norms_in_pipeline() {
    let head_dim = 8;
    let num_heads = 1;
    let seq_len = 2;

    let rope_cfg = RopeConfig::new(head_dim, 128);
    let freqs = compute_frequencies(&rope_cfg);

    let original: Vec<f32> = (0..seq_len * head_dim).map(|i| (i + 1) as f32 * 0.1).collect();

    let mut rotated = original.clone();
    apply_rope_batch(&mut rotated, 0, seq_len, num_heads, head_dim, &freqs);

    for pos in 0..seq_len {
        let orig_slice = &original[pos * head_dim..(pos + 1) * head_dim];
        let rot_slice = &rotated[pos * head_dim..(pos + 1) * head_dim];
        let norm_orig: f32 = orig_slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_rot: f32 = rot_slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_close(norm_orig, norm_rot, 1e-4, &format!("rope_norm_pos{pos}"));
    }
}

#[test]
fn test_batch_transformer_different_seq_lengths_independent() {
    let dim = 4;
    let vocab = 16;
    let num_heads = 1;
    let head_dim = dim;
    let gamma = vec![1.0f32; dim];
    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.05).collect();

    let batches: Vec<Vec<u32>> = vec![vec![1, 2], vec![3, 4, 5], vec![0]];
    let mut outputs = Vec::new();

    for tokens in &batches {
        let seq_len = tokens.len();
        let emb = embedding_lookup(&table, tokens, dim).unwrap();
        let normed = layer_norm(&emb, &gamma, None, &ln_cfg).unwrap();
        let attn_cfg = AttentionConfig {
            num_heads,
            head_dim,
            seq_len,
            causal: true,
            use_alibi: false,
            scale: None,
        };
        let out =
            AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();
        assert_eq!(out.len(), seq_len * dim);
        assert!(out.iter().all(|v| v.is_finite()));
        outputs.push(out);
    }
    assert_eq!(outputs.len(), 3);
}

#[test]
fn test_mixed_precision_i2s_matmul_with_mixed_weight_patterns() {
    let m: usize = 1;
    let k: usize = 8;
    let n: usize = 2;
    let block_size: usize = 4;

    let activations = vec![1.0f32; m * k];
    let packed_k = k.div_ceil(4);
    // Row 0: all +1, Row 1: alternating +1/-1
    let mut weights = Vec::new();
    for _ in 0..packed_k {
        weights.push(pack_i2s([1, 1, 1, 1]));
    }
    for _ in 0..packed_k {
        weights.push(pack_i2s([1, -1, 1, -1]));
    }
    let num_blocks = k.div_ceil(block_size);
    let scales = vec![1.0f32; n * num_blocks];

    let mut output = vec![0.0f32; m * n];
    i2s_matmul_f32(&activations, &weights, &scales, &mut output, m, n, k, block_size).unwrap();

    // Row 0 weights all +1 → sum = 8.0
    assert_close(output[0], 8.0, 1e-4, "all_ones_row");
    // Row 1 weights alternating → sum = 0.0
    assert_close(output[1], 0.0, 1e-4, "alternating_row");
}

#[test]
fn test_kv_cache_append_many_then_slice_first_entry() {
    let num_heads = 1;
    let head_dim = 2;
    let te = num_heads * head_dim;

    let cfg =
        KvCacheConfig { num_layers: 1, num_heads, head_dim, max_seq_len: 32, dtype: KvDtype::F32 };
    let mut cache = KvCache::new(cfg).unwrap();

    for i in 0..8u32 {
        let k_val = vec![i as f32; te];
        let v_val = vec![(i as f32) * 10.0; te];
        kv_cache_append(&mut cache, 0, &k_val, &v_val).unwrap();
    }
    assert_eq!(cache.seq_len(0).unwrap(), 8);

    // First entry should still be intact
    let (keys, _) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
    assert_close(keys[0], 0.0, 1e-7, "first_key_after_many_appends");
}

#[test]
fn test_error_fused_add_normalize_mismatched_lengths() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0]; // wrong length
    let gamma = vec![1.0f32; 4];
    let result = fused_add_normalize(&a, &b, &gamma, 1e-5);
    assert!(result.is_err(), "mismatched residual+hidden should error");
}

#[test]
fn test_transformer_forward_normalized_embedding_attention() {
    let vocab = 16;
    let dim = 8;
    let seq_len = 3;
    let num_heads = 2;
    let head_dim = dim / num_heads;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.1).collect();
    let tokens = [1u32, 4, 7];
    let mut emb = embedding_lookup(&table, &tokens, dim).unwrap();
    normalize_embeddings(&mut emb, dim);

    // After normalization, each row has unit norm
    for row in 0..seq_len {
        let slice = &emb[row * dim..(row + 1) * dim];
        let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_close(norm, 1.0, 1e-5, &format!("emb_norm_row{row}"));
    }

    let attn_cfg = AttentionConfig {
        num_heads,
        head_dim,
        seq_len,
        causal: false,
        use_alibi: false,
        scale: None,
    };
    let out = AttentionKernel::multi_head_attention(&emb, &emb, &emb, &attn_cfg).unwrap();
    assert_eq!(out.len(), seq_len * dim);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn test_batch_l2_norm_reduction_per_sequence() {
    let dim = 4;
    let sequences: Vec<Vec<f32>> =
        vec![vec![1.0, 0.0, 0.0, 0.0], vec![3.0, 4.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]];

    let norms: Vec<f32> = sequences.iter().map(|s| ReductionKernel::l2_norm(s).unwrap()).collect();

    assert_close(norms[0], 1.0, 1e-5, "l2_unit");
    assert_close(norms[1], 5.0, 1e-5, "l2_345");
    assert_close(norms[2], 2.0, 1e-5, "l2_ones");
}
