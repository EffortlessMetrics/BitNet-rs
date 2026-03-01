//! Integration Wave 10 — End-to-End Kernel Pipeline Tests
//!
//! Validates realistic inference sub-graphs that chain multiple kernel
//! primitives and verify mathematical correctness at each stage:
//!
//! - Embedding → RoPE → attention → output
//! - Norm → linear → activation
//! - Loss computation pipeline
//! - Batch processing pipeline
//! - KV cache pipeline
//! - Quantization → dequantization roundtrip accuracy

use bitnet_kernels::convolution::{Conv2DParams, conv2d};
use bitnet_kernels::cpu::activations::{gelu_exact_activate, relu_activate, silu_activate};
use bitnet_kernels::cpu::attention::{
    AttentionConfig, AttentionKernel, attention_with_kv_cache, causal_attention,
};
use bitnet_kernels::cpu::batch_norm::{BatchNormConfig, batch_norm_forward, batch_norm_inference};
use bitnet_kernels::cpu::embedding::{
    CpuEmbeddingConfig, embedding_lookup, embedding_with_position, normalize_embeddings,
};
use bitnet_kernels::cpu::fusion::{fused_add_normalize, fused_rmsnorm_linear};
use bitnet_kernels::cpu::kv_cache::{
    KvCache, KvCacheConfig, KvDtype, kv_cache_append, kv_cache_clear, kv_cache_memory_usage,
    kv_cache_slice,
};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::loss::{
    LossReduction, cosine_similarity_loss, cross_entropy_loss, l1_loss, mse_loss,
};
use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, PoolingKernel};
use bitnet_kernels::cpu::quantize::{
    compute_quantization_error, dequantize_asymmetric_u8, dequantize_symmetric_i8,
    quantize_asymmetric_u8, quantize_binary, quantize_symmetric_i8, quantize_ternary,
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

// ══════════════════════════════════════════════════════════════════
// 1. End-to-End Attention Pipeline:
//    embedding → RoPE → attention → output
// ══════════════════════════════════════════════════════════════════

#[test]
fn e2e_embedding_rope_attention_output() {
    let vocab = 16;
    let dim = 8;
    let seq_len = 3;
    let num_heads = 2;
    let head_dim = dim / num_heads;

    // Embedding lookup
    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.01).collect();
    let tokens = [1u32, 5, 10];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();
    assert_eq!(emb.len(), seq_len * dim);

    // Apply RoPE
    let rope_cfg = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&rope_cfg);
    let mut q = emb.clone();
    let mut k = emb.clone();
    apply_rope_batch(&mut q, 0, seq_len, num_heads, head_dim, &freqs);
    apply_rope_batch(&mut k, 0, seq_len, num_heads, head_dim, &freqs);

    // Multi-head attention
    let attn_cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: true, scale: None };
    let out = AttentionKernel::multi_head_attention(&q, &k, &emb, &attn_cfg).unwrap();
    assert_eq!(out.len(), seq_len * dim);
    assert!(out.iter().all(|v| v.is_finite()), "attention output must be finite");

    // Output projection (naive matmul as linear layer)
    let proj: Vec<f32> = (0..dim * dim).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let projected = naive_matmul(&out, &proj, seq_len, dim, dim);
    assert_eq!(projected.len(), seq_len * dim);
    assert!(projected.iter().all(|v| v.is_finite()));
}

#[test]
fn e2e_embedding_rope_causal_attention_preserves_causality() {
    let dim = 4;
    let seq_len = 4;
    let num_heads = 1;
    let head_dim = dim;

    let table: Vec<f32> = (0..32 * dim).map(|i| ((i % 5) as f32) * 0.2).collect();
    let tokens = [0u32, 1, 2, 3];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();

    let rope_cfg = RopeConfig::new(head_dim, 32);
    let freqs = compute_frequencies(&rope_cfg);
    let mut q = emb.clone();
    let mut k = emb.clone();
    apply_rope_batch(&mut q, 0, seq_len, num_heads, head_dim, &freqs);
    apply_rope_batch(&mut k, 0, seq_len, num_heads, head_dim, &freqs);

    // Causal attention
    let attn_cfg = AttentionConfig { num_heads, head_dim, seq_len, causal: true, scale: None };
    let out = causal_attention(&q, &k, &emb, &attn_cfg).unwrap();
    assert_eq!(out.len(), seq_len * dim);

    // First token output depends only on first token
    // (with causal masking, position 0 can only attend to itself)
    let out_tok0 = &out[..dim];
    // Should be a convex combination of value at position 0 only → equals v[0]
    assert_slice_close(out_tok0, &emb[..dim], 1e-5, "causal_tok0");
}

#[test]
fn e2e_positional_embedding_rope_attention_position_sensitivity() {
    let vocab = 8;
    let dim = 4;
    let max_seq = 16;
    let num_heads = 1;
    let head_dim = dim;

    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.05).collect();
    let tokens = [3u32, 3]; // Same token at different positions

    let config = CpuEmbeddingConfig::new(vocab, dim);
    let emb = embedding_with_position(&table, &tokens, &config, 0).unwrap();
    // Positional embedding adds position info, so same token at pos 0 and pos 1 differ
    let tok0 = &emb[..dim];
    let tok1 = &emb[dim..2 * dim];
    assert_ne!(tok0, tok1, "positional embeddings should differ for different positions");

    let rope_cfg = RopeConfig::new(head_dim, max_seq);
    let freqs = compute_frequencies(&rope_cfg);
    let mut q = emb.clone();
    apply_rope_batch(&mut q, 0, 2, num_heads, head_dim, &freqs);

    // After RoPE, same base embeddings at different positions are further differentiated
    let rq0 = &q[..dim];
    let rq1 = &q[dim..2 * dim];
    assert_ne!(rq0, rq1, "RoPE should differentiate same-token at different positions");
}

// ══════════════════════════════════════════════════════════════════
// 2. Norm → Linear → Activation Pipeline:
//    layer_norm → matmul → activation
// ══════════════════════════════════════════════════════════════════

#[test]
fn norm_linear_gelu_pipeline() {
    let seq = 2;
    let dim = 8;
    let hidden = 4;

    let input: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.3 - 1.0).collect();
    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let gamma = vec![1.0f32; dim];
    let beta = vec![0.0f32; dim];

    let normed = layer_norm(&input, &gamma, Some(&beta), &ln_cfg).unwrap();
    // Layer norm output should have ~zero mean and ~unit variance per row
    for row in 0..seq {
        let slice = &normed[row * dim..(row + 1) * dim];
        let mean: f32 = slice.iter().sum::<f32>() / dim as f32;
        assert_close(mean, 0.0, 1e-4, &format!("ln_mean_row{row}"));
    }

    // Linear projection
    let weight: Vec<f32> = (0..dim * hidden).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let projected = naive_matmul(&normed, &weight, seq, hidden, dim);

    // GELU activation
    let mut activated = vec![0.0f32; projected.len()];
    gelu_exact_activate(&projected, &mut activated);
    // GELU(x) >= 0 for x > 0, approximately zero for large negative x
    assert!(activated.iter().all(|v| v.is_finite()));
}

#[test]
fn rmsnorm_linear_silu_pipeline() {
    let seq = 3;
    let dim = 4;
    let ff_dim = 8;

    let input: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.5).collect();
    let rms_cfg = LayerNormConfig::new(vec![dim]);
    let gamma = vec![1.0f32; dim];
    let normed = rms_norm(&input, &gamma, &rms_cfg).unwrap();

    // RMSNorm: output should have unit RMS per token
    for row in 0..seq {
        let slice = &normed[row * dim..(row + 1) * dim];
        let rms = (slice.iter().map(|x| x * x).sum::<f32>() / dim as f32).sqrt();
        assert_close(rms, 1.0, 0.05, &format!("rms_row{row}"));
    }

    // Linear projection
    let weight: Vec<f32> = (0..dim * ff_dim).map(|i| ((i % 3) as f32 - 1.0) * 0.2).collect();
    let projected = naive_matmul(&normed, &weight, seq, ff_dim, dim);

    // SiLU activation
    let mut activated = vec![0.0f32; projected.len()];
    silu_activate(&projected, &mut activated);
    assert!(activated.iter().all(|v| v.is_finite()));
}

#[test]
fn fused_rmsnorm_linear_matches_separate_ops() {
    let dim = 4;
    let out_dim = 4;
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0f32; dim];
    // weight is [out_dim × dim] row-major
    let weight: Vec<f32> = (0..out_dim * dim).map(|i| (i as f32) * 0.1).collect();

    // Separate: rms_norm then matmul (weight × normed)
    let rms_cfg = LayerNormConfig::new(vec![dim]);
    let normed = rms_norm(&input, &gamma, &rms_cfg).unwrap();
    // fused does weight[out_dim × dim] . (input * gamma / rms), equivalent to:
    let mut separate_result = vec![0.0f32; out_dim];
    for (o, row) in separate_result.iter_mut().zip(weight.chunks_exact(dim)) {
        *o = row.iter().zip(normed.iter()).map(|(w, n)| w * n).sum();
    }

    // Fused version: (input, weight, gamma, eps)
    let fused_result = fused_rmsnorm_linear(&input, &weight, &gamma, 1e-5).unwrap();

    assert_slice_close(&fused_result, &separate_result, 1e-4, "fused_vs_separate_rmsnorm_linear");
}

#[test]
fn layernorm_quantized_matmul_relu_pipeline() {
    let m = 2;
    let dim = 8;
    let out_dim = 4;
    let block_size = 4;

    let input: Vec<f32> = (0..m * dim).map(|i| (i as f32) * 0.2 - 0.8).collect();
    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let gamma = vec![1.0f32; dim];
    let normed = layer_norm(&input, &gamma, None, &ln_cfg).unwrap();

    // Quantized matmul with I2S weights
    let packed_k = dim.div_ceil(4);
    let weights: Vec<u8> = vec![pack_i2s([1, 0, -1, 1]); out_dim * packed_k];
    let num_blocks = dim.div_ceil(block_size);
    let scales = vec![0.5f32; out_dim * num_blocks];
    let mut logits = vec![0.0f32; m * out_dim];
    i2s_matmul_f32(&normed, &weights, &scales, &mut logits, m, out_dim, dim, block_size).unwrap();

    // ReLU activation
    let mut activated = vec![0.0f32; logits.len()];
    relu_activate(&logits, &mut activated);
    assert!(activated.iter().all(|&v| v >= 0.0), "ReLU output must be non-negative");
}

// ══════════════════════════════════════════════════════════════════
// 3. Loss Computation Pipeline:
//    forward pass → loss → gradient direction
// ══════════════════════════════════════════════════════════════════

#[test]
fn forward_cross_entropy_loss_gradient_direction() {
    let num_classes = 4;
    let batch = 2;
    // Logits: batch of 2 samples, 4 classes each
    let logits = vec![
        2.0, 1.0, 0.5, 0.1, // sample 0: class 0 is highest
        0.1, 0.5, 3.0, 0.2, // sample 1: class 2 is highest
    ];
    let targets = vec![0, 2]; // correct predictions

    let (loss, per_sample) =
        cross_entropy_loss(&logits, &targets, num_classes, LossReduction::Mean).unwrap();
    assert!(loss >= 0.0, "cross-entropy loss must be non-negative");
    assert!(loss < 2.0, "loss should be small when predictions are correct");
    assert_eq!(per_sample.len(), batch);

    // Perturbed logits: move correct class down → loss should increase
    let bad_logits = vec![
        0.1, 1.0, 0.5, 2.0, // sample 0: class 3 is highest (wrong)
        0.1, 3.0, 0.5, 0.2, // sample 1: class 1 is highest (wrong)
    ];
    let (bad_loss, _) =
        cross_entropy_loss(&bad_logits, &targets, num_classes, LossReduction::Mean).unwrap();
    assert!(bad_loss > loss, "loss should increase when predictions are wrong");
}

#[test]
fn forward_mse_loss_decreases_toward_target() {
    let predictions = vec![1.0, 2.0, 3.0, 4.0];
    let targets = vec![1.1, 2.1, 3.1, 4.1];
    let loss_near = mse_loss(&predictions, &targets, LossReduction::Mean).unwrap();

    let far_predictions = vec![5.0, 6.0, 7.0, 8.0];
    let loss_far = mse_loss(&far_predictions, &targets, LossReduction::Mean).unwrap();

    assert!(loss_near < loss_far, "MSE should be smaller when predictions are closer to targets");
    assert!(loss_near >= 0.0);
}

#[test]
fn forward_cosine_similarity_loss_identical_vectors() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![2.0, 4.0, 6.0]; // same direction, different magnitude
    let loss = cosine_similarity_loss(&a, &b).unwrap();
    // cosine_similarity_loss returns 1 - cos_sim, so parallel vectors → ~0
    assert_close(loss, 0.0, 1e-5, "cosine_loss_parallel_vectors");

    // Orthogonal vectors should have similarity 0 → loss ≈ 1
    let c = vec![1.0, 0.0, 0.0];
    let d = vec![0.0, 1.0, 0.0];
    let loss_orth = cosine_similarity_loss(&c, &d).unwrap();
    assert_close(loss_orth, 1.0, 1e-5, "cosine_loss_orthogonal");
}

#[test]
fn forward_l1_mse_cross_entropy_all_agree_on_perfect_prediction() {
    let num_classes = 3;
    // Logits strongly favour the correct class
    let logits = vec![10.0, -10.0, -10.0];
    let targets_ce = vec![0usize];
    let (ce_loss, _) =
        cross_entropy_loss(&logits, &targets_ce, num_classes, LossReduction::Mean).unwrap();

    // For MSE/L1: predictions == targets → loss == 0
    let preds = vec![1.0, 2.0, 3.0];
    let tgts = vec![1.0, 2.0, 3.0];
    let mse = mse_loss(&preds, &tgts, LossReduction::Mean).unwrap();
    let l1 = l1_loss(&preds, &tgts, LossReduction::Mean).unwrap();

    assert_close(mse, 0.0, 1e-7, "mse_perfect");
    assert_close(l1, 0.0, 1e-7, "l1_perfect");
    assert!(ce_loss < 0.01, "ce should be near zero for strong prediction, got {ce_loss}");
}

// ══════════════════════════════════════════════════════════════════
// 4. Batch Processing Pipeline:
//    batch_norm → conv-like → pooling → reduction
// ══════════════════════════════════════════════════════════════════

#[test]
fn batch_norm_conv2d_pool_reduction_pipeline() {
    let batch = 2;
    let channels = 2;
    let n = batch * channels;

    // Batch norm
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0 - 3.0).collect();
    let gamma = vec![1.0f32; channels];
    let beta = vec![0.0f32; channels];
    let running_mean = vec![0.0f32; channels];
    let running_var = vec![1.0f32; channels];
    let bn_cfg =
        BatchNormConfig { num_features: channels, eps: 1e-5, momentum: 0.1, training: true };
    let (normed, _, _) =
        batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &bn_cfg).unwrap();
    assert_eq!(normed.len(), n);

    // Conv2D: treat normed as [1, 1, 2, 2] (batch=1, channels=1, h=2, w=2)
    let in_ch = 1;
    let out_ch = 1;
    let h = 2;
    let w = 2;
    let kh = 1;
    let kw = 1;
    let kernel = vec![1.0f32; out_ch * in_ch * kh * kw];
    let params = Conv2DParams::default(); // stride=1, padding=0, dilation=1
    let mut conv_out = vec![0.0f32; out_ch * h * w];
    conv2d(
        &normed,
        &kernel,
        None,
        &mut conv_out,
        params,
        (1, in_ch, h, w),
        (out_ch, in_ch, kh, kw),
    )
    .unwrap();
    assert_eq!(conv_out.len(), out_ch * h * w);

    // Average pooling
    let pool_cfg =
        PoolConfig { pool_type: PoolType::Average, kernel_size: 2, stride: 2, padding: 0 };
    let pooled = PoolingKernel::apply(&conv_out, &pool_cfg).unwrap();
    assert!(pooled.len() < conv_out.len());

    // Reduction to scalar
    let sum = reduce_f32(&pooled, ReductionOp::Sum);
    assert!(sum.is_finite());
}

#[test]
fn batch_norm_inference_silu_pool_pipeline() {
    let batch = 4;
    let features = 3;
    let input: Vec<f32> = (0..batch * features).map(|i| (i as f32) * 0.5 - 2.0).collect();
    let gamma = vec![1.0f32; features];
    let beta = vec![0.0f32; features];
    let running_mean = vec![0.0f32; features];
    let running_var = vec![1.0f32; features];

    let normed =
        batch_norm_inference(&input, &gamma, &beta, &running_mean, &running_var, 1e-5).unwrap();
    assert_eq!(normed.len(), batch * features);

    // SiLU activation
    let mut activated = vec![0.0f32; normed.len()];
    silu_activate(&normed, &mut activated);

    // Max pooling over features per sample
    let pool_cfg = PoolConfig {
        pool_type: PoolType::Max,
        kernel_size: features,
        stride: features,
        padding: 0,
    };
    let pooled = PoolingKernel::apply(&activated, &pool_cfg).unwrap();
    assert!(pooled.iter().all(|v: &f32| v.is_finite()));
}

#[test]
fn batch_norm_training_updates_running_stats() {
    let features = 2;
    let input: Vec<f32> = vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0];
    let gamma = vec![1.0; features];
    let beta = vec![0.0; features];
    let running_mean = vec![0.0; features];
    let running_var = vec![1.0; features];
    let cfg = BatchNormConfig { num_features: features, eps: 1e-5, momentum: 0.1, training: true };

    let (output, new_mean, new_var) =
        batch_norm_forward(&input, &gamma, &beta, &running_mean, &running_var, &cfg).unwrap();

    // Running stats should have been updated (momentum = 0.1)
    assert!(new_mean[0] > 0.0, "running mean should shift toward batch mean");
    assert!(new_mean[1] > 0.0);
    // Output should be normalized
    assert!(output.iter().all(|v| v.is_finite()));

    // Second pass with updated stats produces different running stats
    let (_, newer_mean, _) =
        batch_norm_forward(&input, &gamma, &beta, &new_mean, &new_var, &cfg).unwrap();
    // After two updates, running mean should be closer to batch mean
    let batch_mean_ch0 = (1.0 + 2.0 + 3.0 + 4.0) / 4.0;
    assert!(
        (newer_mean[0] - batch_mean_ch0).abs() < (new_mean[0] - batch_mean_ch0).abs(),
        "running mean should converge toward batch mean"
    );
}

// ══════════════════════════════════════════════════════════════════
// 5. KV Cache Pipeline:
//    cache init → append → attention with cached keys
// ══════════════════════════════════════════════════════════════════

#[test]
fn kv_cache_init_append_attention_pipeline() {
    let num_layers = 1;
    let num_heads = 1;
    let head_dim = 4;
    let max_seq = 16;

    let cfg = KvCacheConfig {
        num_layers,
        num_heads,
        head_dim,
        max_seq_len: max_seq,
        dtype: KvDtype::F32,
    };
    let mut cache = KvCache::new(cfg).unwrap();

    // Append first token KV
    let te = num_heads * head_dim;
    let k1 = vec![1.0f32; te];
    let v1 = vec![10.0f32; te];
    kv_cache_append(&mut cache, 0, &k1, &v1).unwrap();
    assert_eq!(cache.seq_len(0).unwrap(), 1);

    // Append second token KV
    let k2 = vec![2.0f32; te];
    let v2 = vec![20.0f32; te];
    kv_cache_append(&mut cache, 0, &k2, &v2).unwrap();
    assert_eq!(cache.seq_len(0).unwrap(), 2);

    // Slice and verify
    let (keys, values) = kv_cache_slice(&cache, 0, 0, 2).unwrap();
    assert_eq!(keys.len(), 2 * te);
    assert_eq!(values.len(), 2 * te);
    assert_slice_close(&keys[..te], &k1, 1e-7, "cached_k1");
    assert_slice_close(&keys[te..], &k2, 1e-7, "cached_k2");
}

#[test]
fn kv_cache_attention_with_kv_cache_incremental() {
    let head_dim = 4;

    let mut k_cache: Vec<f32> = Vec::new();
    let mut v_cache: Vec<f32> = Vec::new();

    // Step 1: first query with first KV pair
    let q1 = vec![1.0f32; head_dim];
    let k1 = vec![1.0f32; head_dim];
    let v1 = vec![10.0f32; head_dim];
    let out1 =
        attention_with_kv_cache(&q1, &mut k_cache, &mut v_cache, &k1, &v1, head_dim).unwrap();
    assert_eq!(out1.len(), head_dim);
    // With only one KV pair, output should be exactly v1 (softmax of single score = 1.0)
    assert_slice_close(&out1, &v1, 1e-5, "single_kv_attention");

    // Step 2: second query with second KV pair
    let q2 = vec![2.0f32; head_dim];
    let k2 = vec![0.0f32; head_dim];
    let v2 = vec![20.0f32; head_dim];
    let out2 =
        attention_with_kv_cache(&q2, &mut k_cache, &mut v_cache, &k2, &v2, head_dim).unwrap();
    assert_eq!(out2.len(), head_dim);
    // Output should be a weighted average of v1 and v2
    assert!(out2.iter().all(|v| v.is_finite()));
    // Cache should now have 2 entries
    assert_eq!(k_cache.len(), 2 * head_dim);
}

#[test]
fn kv_cache_clear_resets_for_new_sequence() {
    let cfg = KvCacheConfig {
        num_layers: 2,
        num_heads: 2,
        head_dim: 4,
        max_seq_len: 8,
        dtype: KvDtype::F32,
    };
    let te = cfg.num_heads * cfg.head_dim;
    let mut cache = KvCache::new(cfg).unwrap();

    // Fill some data
    kv_cache_append(&mut cache, 0, &vec![1.0; te], &vec![2.0; te]).unwrap();
    kv_cache_append(&mut cache, 1, &vec![3.0; te * 2], &vec![4.0; te * 2]).unwrap();
    assert_eq!(cache.seq_len(0).unwrap(), 1);
    assert_eq!(cache.seq_len(1).unwrap(), 2);

    // Clear and verify
    kv_cache_clear(&mut cache);
    assert_eq!(cache.seq_len(0).unwrap(), 0);
    assert_eq!(cache.seq_len(1).unwrap(), 0);

    // Memory is still allocated (just reset)
    let mem = kv_cache_memory_usage(&cache);
    assert!(mem > 0, "memory should still be allocated after clear");

    // Can append again after clear
    kv_cache_append(&mut cache, 0, &vec![5.0; te], &vec![6.0; te]).unwrap();
    assert_eq!(cache.seq_len(0).unwrap(), 1);
    let (keys, _) = kv_cache_slice(&cache, 0, 0, 1).unwrap();
    assert_close(keys[0], 5.0, 1e-7, "post_clear_append");
}

#[test]
fn kv_cache_multilayer_independent_sequences() {
    let num_layers = 3;
    let num_heads = 1;
    let head_dim = 2;
    let cfg =
        KvCacheConfig { num_layers, num_heads, head_dim, max_seq_len: 8, dtype: KvDtype::F32 };
    let te = num_heads * head_dim;
    let mut cache = KvCache::new(cfg).unwrap();

    // Append different amounts per layer
    kv_cache_append(&mut cache, 0, &vec![1.0; te], &vec![10.0; te]).unwrap();
    kv_cache_append(&mut cache, 1, &vec![2.0; te * 3], &vec![20.0; te * 3]).unwrap();
    kv_cache_append(&mut cache, 2, &vec![3.0; te * 2], &vec![30.0; te * 2]).unwrap();

    assert_eq!(cache.seq_len(0).unwrap(), 1);
    assert_eq!(cache.seq_len(1).unwrap(), 3);
    assert_eq!(cache.seq_len(2).unwrap(), 2);

    // Verify independence: layer 1 data is correct
    let (k1, v1) = kv_cache_slice(&cache, 1, 0, 3).unwrap();
    assert_eq!(k1.len(), 3 * te);
    assert!(k1.iter().all(|&x| (x - 2.0).abs() < 1e-7));
    assert!(v1.iter().all(|&x| (x - 20.0).abs() < 1e-7));
}

// ══════════════════════════════════════════════════════════════════
// 6. Quantization → Dequantization Roundtrip Accuracy
// ══════════════════════════════════════════════════════════════════

#[test]
fn symmetric_i8_roundtrip_accuracy() {
    let input = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.75, 0.9];
    let bits = 8;
    let (quantized, scale) = quantize_symmetric_i8(&input, bits);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);

    let err = compute_quantization_error(&input, &dequantized);
    // 8-bit symmetric quantization should have low error
    assert!(err.max_abs_error < 0.01, "max_abs_error too high: {}", err.max_abs_error);
    assert!(err.mse < 1e-4, "MSE too high: {}", err.mse);
}

#[test]
fn asymmetric_u8_roundtrip_accuracy() {
    let input = vec![0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5];
    let (quantized, scale, zero_point) = quantize_asymmetric_u8(&input);
    let dequantized = dequantize_asymmetric_u8(&quantized, scale, zero_point);

    let err = compute_quantization_error(&input, &dequantized);
    assert!(err.max_abs_error < 0.02, "asymmetric max_abs_error too high: {}", err.max_abs_error);
}

#[test]
fn ternary_quantization_preserves_sign() {
    let input = vec![1.0, -1.0, 0.01, -0.01, 5.0, -3.0, 0.0, 0.1];
    let threshold = 0.5;
    let ternary = quantize_ternary(&input, threshold);

    // Values above threshold → +1, below -threshold → -1, between → 0
    assert_eq!(ternary[0], 1); // 1.0 > 0.5
    assert_eq!(ternary[1], -1); // -1.0 < -0.5
    assert_eq!(ternary[2], 0); // 0.01 within threshold
    assert_eq!(ternary[3], 0); // -0.01 within threshold
    assert_eq!(ternary[4], 1); // 5.0 > 0.5
    assert_eq!(ternary[5], -1); // -3.0 < -0.5
}

#[test]
fn binary_quantization_preserves_sign() {
    let input = vec![0.5, -0.3, 0.0, 1.0, -2.0];
    let binary = quantize_binary(&input);

    for (i, (&orig, &quant)) in input.iter().zip(binary.iter()).enumerate() {
        if orig >= 0.0 {
            assert_eq!(quant, 1, "input[{i}]={orig} should quantize to 1");
        } else {
            assert_eq!(quant, -1, "input[{i}]={orig} should quantize to -1");
        }
    }
}

#[test]
fn i2s_quantized_matmul_roundtrip_correctness() {
    // Test that quantized matmul produces results consistent with
    // dequantized weights multiplied naively
    let m: usize = 1;
    let k: usize = 8;
    let n: usize = 2;
    let block_size: usize = 4;

    let activations: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    // Pack weights: all +1 → pack_i2s([1,1,1,1])
    let packed_k = k.div_ceil(4);
    let weights: Vec<u8> = vec![pack_i2s([1, 1, 1, 1]); n * packed_k];
    let num_blocks = k.div_ceil(block_size);
    let scales = vec![1.0f32; n * num_blocks];

    let mut output = vec![0.0f32; m * n];
    i2s_matmul_f32(&activations, &weights, &scales, &mut output, m, n, k, block_size).unwrap();

    // Each output element should be sum of activations * 1 * scale
    // = 1.0 * 8 * 1.0 = 8.0 (but computed per-block: 4*1.0 + 4*1.0 = 8.0)
    for &val in &output {
        assert_close(val, 8.0, 1e-5, "i2s_matmul_all_ones");
    }
}

#[test]
fn quantize_dequantize_preserves_zero() {
    let input = vec![0.0f32; 8];
    let (quantized, scale) = quantize_symmetric_i8(&input, 8);
    let dequantized = dequantize_symmetric_i8(&quantized, scale);
    for (i, &v) in dequantized.iter().enumerate() {
        assert_close(v, 0.0, 1e-7, &format!("zero_roundtrip[{i}]"));
    }
}

// ══════════════════════════════════════════════════════════════════
// 7. Cross-Pipeline Compositions
// ══════════════════════════════════════════════════════════════════

#[test]
fn full_transformer_block_pipeline() {
    // Simulate a single transformer block:
    // input → layer_norm → attention → residual → layer_norm → FFN → residual
    let seq = 2;
    let dim = 4;
    let num_heads = 1;
    let head_dim = dim;

    let input: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.3).collect();

    // Pre-attention layer norm
    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let gamma = vec![1.0f32; dim];
    let normed = layer_norm(&input, &gamma, None, &ln_cfg).unwrap();

    // Self-attention
    let attn_cfg = AttentionConfig { num_heads, head_dim, seq_len: seq, causal: true, scale: None };
    let attn_out =
        AttentionKernel::multi_head_attention(&normed, &normed, &normed, &attn_cfg).unwrap();

    // Residual connection
    let residual1: Vec<f32> = input.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

    // Post-attention layer norm
    let normed2 = layer_norm(&residual1, &gamma, None, &ln_cfg).unwrap();

    // FFN: linear → GELU → linear
    let w1: Vec<f32> = (0..dim * dim).map(|i| ((i % 3) as f32 - 1.0) * 0.2).collect();
    let hidden = naive_matmul(&normed2, &w1, seq, dim, dim);
    let mut activated = vec![0.0f32; hidden.len()];
    gelu_exact_activate(&hidden, &mut activated);
    let w2: Vec<f32> = (0..dim * dim).map(|i| ((i % 4) as f32 - 1.5) * 0.1).collect();
    let ffn_out = naive_matmul(&activated, &w2, seq, dim, dim);

    // Residual connection
    let output: Vec<f32> = residual1.iter().zip(ffn_out.iter()).map(|(a, b)| a + b).collect();
    assert_eq!(output.len(), seq * dim);
    assert!(output.iter().all(|v| v.is_finite()), "transformer block output must be finite");
}

#[test]
fn embedding_norm_loss_training_loop_simulation() {
    let vocab = 8;
    let dim = 4;
    let num_classes = vocab;

    // Forward pass: embedding → normalize → project to logits → loss
    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.1).collect();
    let tokens = [2u32, 5];
    let emb = embedding_lookup(&table, &tokens, dim).unwrap();

    // Normalize embeddings
    let mut normed = emb.clone();
    normalize_embeddings(&mut normed, dim);

    // Project to vocab-size logits
    let proj: Vec<f32> = (0..dim * num_classes).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let logits = naive_matmul(&normed, &proj, 2, num_classes, dim);

    // Cross-entropy loss (predict next token)
    let targets = vec![5usize, 2]; // next-token targets
    let (loss, _) =
        cross_entropy_loss(&logits, &targets, num_classes, LossReduction::Mean).unwrap();
    assert!(loss.is_finite() && loss >= 0.0, "training loss must be non-negative finite");

    // Verify loss is in reasonable range for random logits
    let max_expected = (num_classes as f32).ln() + 1.0; // slightly above uniform CE
    assert!(loss < max_expected, "loss {loss} exceeds expected max {max_expected}");
}

#[test]
fn reduction_after_attention_computes_sequence_summary() {
    let dim = 4;
    let seq = 3;
    let num_heads = 1;

    let input: Vec<f32> = (0..seq * dim).map(|i| (i as f32) * 0.2).collect();
    let attn_cfg =
        AttentionConfig { num_heads, head_dim: dim, seq_len: seq, causal: false, scale: None };
    let attn_out =
        AttentionKernel::multi_head_attention(&input, &input, &input, &attn_cfg).unwrap();

    // Mean-pool across sequence dimension
    let mean = ReductionKernel::mean(&attn_out).unwrap();
    assert!(mean.is_finite());

    // L2 norm of the output
    let l2 = ReductionKernel::l2_norm(&attn_out).unwrap();
    assert!(l2 >= 0.0);
    assert!(l2.is_finite());
}

#[test]
fn fused_add_normalize_matches_manual() {
    let dim = 4;
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.5, -0.5, 1.0, -1.0];
    let gamma = vec![1.0; dim];
    let eps = 1e-5;

    let fused = fused_add_normalize(&a, &b, &gamma, eps).unwrap();

    // Manual: add then rms_norm (fused_add_normalize uses RMS, not layer norm)
    let added: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let rms_cfg = LayerNormConfig::new(vec![dim]);
    let manual = rms_norm(&added, &gamma, &rms_cfg).unwrap();

    assert_slice_close(&fused, &manual, 1e-4, "fused_add_norm_vs_manual");
}
