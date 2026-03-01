//! Integration Wave 7 — Kernel Pipeline Cross-Validation Tests
//!
//! Validates that CPU kernel outputs match expected values when
//! chained in realistic inference pipelines:
//!
//! - softmax → top-k / argmax
//! - embedding → attention (RoPE + scaled dot-product)
//! - quantized matmul → softmax → sampling
//! - conv1d → pooling → layer norm
//! - RoPE → attention
//! - error propagation through pipelines

use bitnet_kernels::cpu::conv1d::{Conv1dConfig, PaddingMode, conv1d_forward};
use bitnet_kernels::cpu::embedding::{
    EmbeddingConfig, embedding_lookup, embedding_lookup_simd, normalize_embeddings,
};
use bitnet_kernels::cpu::fusion::{
    FusionError, fused_add_normalize, fused_rmsnorm_linear, fused_softmax_mask,
};
use bitnet_kernels::cpu::pooling::{PoolConfig, PoolType, PoolingKernel};
use bitnet_kernels::cpu::quantized_matmul::{i2s_matmul_f32, pack_i2s};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, apply_rope_batch, compute_frequencies};
use bitnet_kernels::cpu::softmax::{softmax, softmax_batch};
use bitnet_kernels::reduction::{ReductionOp, reduce_f32, reduce_rows_f32};

// ── Helpers ────────────────────────────────────────────────────────

const EPS: f32 = 1e-5;

fn assert_close(a: f32, b: f32, tol: f32, ctx: &str) {
    assert!((a - b).abs() <= tol, "{ctx}: expected {b}, got {a} (diff {})", (a - b).abs());
}

fn assert_slice_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
    for (i, (&ai, &bi)) in a.iter().zip(b).enumerate() {
        assert_close(ai, bi, tol, &format!("{ctx}[{i}]"));
    }
}

/// Argmax over a slice.
fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap()
}

/// Top-k indices (descending by value).
fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}

/// Manual softmax reference for cross-validation.
fn reference_softmax(input: &[f32], temperature: f32) -> Vec<f32> {
    let inv_t = 1.0 / temperature;
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| ((x - max_val) * inv_t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ═══════════════════════════════════════════════════════════════════
// 1. Softmax → Top-K Pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn softmax_then_argmax_picks_highest_logit() {
    let logits = vec![1.0, 3.0, 0.5, 2.0, -1.0];
    let probs = softmax(&logits, 1.0).unwrap();

    // Softmax preserves ordering → argmax must be index 1.
    assert_eq!(argmax(&probs), 1);
    // Probabilities sum to 1.
    assert_close(probs.iter().sum::<f32>(), 1.0, 1e-6, "prob_sum");
}

#[test]
fn softmax_then_topk_returns_correct_order() {
    let logits = vec![0.1, 5.0, 3.0, 4.0, 2.0, 1.0];
    let probs = softmax(&logits, 1.0).unwrap();
    let top3 = top_k(&probs, 3);

    assert_eq!(top3, vec![1, 3, 2], "top-3 indices by descending prob");
}

#[test]
fn softmax_temperature_affects_topk_spread() {
    let logits = vec![2.0, 2.1, 0.0, 0.0];

    // Low temperature → sharper distribution.
    let sharp = softmax(&logits, 0.1).unwrap();
    // High temperature → flatter distribution.
    let flat = softmax(&logits, 10.0).unwrap();

    let sharp_gap = sharp[1] - sharp[0];
    let flat_gap = flat[1] - flat[0];
    assert!(sharp_gap > flat_gap, "sharp gap ({sharp_gap}) should exceed flat gap ({flat_gap})");
}

#[test]
fn softmax_cross_validates_with_manual_reference() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let kernel_probs = softmax(&logits, 1.0).unwrap();
    let ref_probs = reference_softmax(&logits, 1.0);

    assert_slice_close(&kernel_probs, &ref_probs, 1e-5, "softmax_xval");
}

// ═══════════════════════════════════════════════════════════════════
// 2. Embedding → Attention Pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn embedding_lookup_then_rope_applies_rotation() {
    let vocab_size = 8;
    let embed_dim = 4; // must be even for RoPE
    // Simple embedding table: row i = [i, i, i, i] as f32.
    let table: Vec<f32> = (0..vocab_size).flat_map(|i| vec![i as f32; embed_dim]).collect();
    let indices: Vec<u32> = vec![2, 5]; // 2 tokens

    let embeddings = embedding_lookup(&table, &indices, embed_dim).unwrap();
    assert_eq!(embeddings.len(), 2 * embed_dim);

    // Apply RoPE to each token position.
    let rope_cfg = RopeConfig::new(embed_dim, 16);
    let freqs = compute_frequencies(&rope_cfg);

    let mut q = embeddings.clone();
    for pos in 0..2 {
        let start = pos * embed_dim;
        apply_rope(&mut q[start..start + embed_dim], pos, embed_dim, &freqs);
    }

    // RoPE should modify the vectors (rotation ≠ identity except pos 0
    // dim 0 pair where cos=1, sin=0 can leave values unchanged).
    // At position 1 the rotation angle is non-zero for all dim pairs,
    // so the output must differ from the input.
    let pos1_orig = &embeddings[embed_dim..2 * embed_dim];
    let pos1_rope = &q[embed_dim..2 * embed_dim];
    assert_ne!(pos1_orig, pos1_rope, "RoPE must rotate pos-1 vector");
}

#[test]
fn embedding_simd_matches_scalar_lookup() {
    let vocab = 16;
    let dim = 8;
    let table: Vec<f32> = (0..vocab * dim).map(|i| (i as f32) * 0.1).collect();
    let indices: Vec<u32> = vec![0, 3, 7, 15];

    let scalar = embedding_lookup(&table, &indices, dim).unwrap();
    let cfg = EmbeddingConfig { vocab_size: vocab, embedding_dim: dim, padding_idx: None };
    let simd = embedding_lookup_simd(&table, &indices, &cfg).unwrap();

    assert_slice_close(&scalar, &simd, 0.0, "scalar_vs_simd_embed");
}

#[test]
fn embedding_then_dot_product_attention_scores() {
    let dim = 4;
    // Q and K from embedding lookup (2 tokens × dim).
    let table: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // token 0
        0.0, 1.0, 0.0, 0.0, // token 1
        0.0, 0.0, 1.0, 0.0, // token 2
        0.0, 0.0, 0.0, 1.0, // token 3
    ];
    let q_ids: Vec<u32> = vec![0, 1];
    let k_ids: Vec<u32> = vec![0, 1];

    let q = embedding_lookup(&table, &q_ids, dim).unwrap();
    let k = embedding_lookup(&table, &k_ids, dim).unwrap();

    // Compute attention scores: score[i][j] = Q[i] · K[j] / sqrt(dim).
    let scale = 1.0 / (dim as f32).sqrt();
    let seq_len = 2;
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let dot: f32 = (0..dim).map(|d| q[i * dim + d] * k[j * dim + d]).sum();
            scores[i * seq_len + j] = dot * scale;
        }
    }

    // Apply softmax row-wise.
    let attn_weights = softmax_batch(&scores, seq_len, 1.0).unwrap();

    // Orthogonal embeddings → diagonal-dominant attention.
    assert!(attn_weights[0] > attn_weights[1], "self-attention on token 0 should be strongest");
    // Each row sums to 1.
    for r in 0..seq_len {
        let row_sum: f32 = attn_weights[r * seq_len..(r + 1) * seq_len].iter().sum();
        assert_close(row_sum, 1.0, 1e-6, &format!("attn_row_{r}"));
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Quantized MatMul → Softmax → Sampling Pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn quantized_matmul_then_softmax_produces_valid_distribution() {
    // m=1 (single input row), n=4 (4 output logits), k=8.
    let m = 1;
    let n = 4;
    let k = 8;
    let block_size = 8;

    let activations = vec![1.0f32; m * k];

    // Weights: all +1 packed → each output = sum(activations * scale).
    let packed_k = k.div_ceil(4);
    let weights_packed: Vec<u8> = vec![pack_i2s([1, 1, 1, 1]); packed_k * n];
    let num_blocks = k.div_ceil(block_size);
    let scales = vec![1.0f32; n * num_blocks];

    let mut logits = vec![0.0f32; m * n];
    i2s_matmul_f32(&activations, &weights_packed, &scales, &mut logits, m, n, k, block_size)
        .unwrap();

    // All outputs should be equal (uniform weights + activations).
    for &l in &logits {
        assert_close(l, logits[0], EPS, "uniform_logits");
    }

    // Softmax of uniform logits → uniform distribution.
    let probs = softmax(&logits, 1.0).unwrap();
    for &p in &probs {
        assert_close(p, 0.25, 1e-5, "uniform_prob");
    }
}

#[test]
fn quantized_matmul_with_mixed_weights_affects_sampling() {
    let m = 1;
    let n = 4;
    let k = 4;
    let block_size = 4;

    let activations = vec![1.0, 1.0, 1.0, 1.0];

    // Column 0: all +1, Column 1: all -1, Column 2: mixed, Col 3: 0.
    let col0 = pack_i2s([1, 1, 1, 1]);
    let col1 = pack_i2s([-1, -1, -1, -1]);
    let col2 = pack_i2s([1, -1, 1, -1]);
    let col3 = pack_i2s([0, 0, 0, 0]);
    let weights_packed = vec![col0, col1, col2, col3];
    let scales = vec![1.0f32; n]; // one block per column

    let mut logits = vec![0.0f32; n];
    i2s_matmul_f32(&activations, &weights_packed, &scales, &mut logits, m, n, k, block_size)
        .unwrap();

    // Expected logits: [4, -4, 0, 0].
    assert_close(logits[0], 4.0, EPS, "col0_logit");
    assert_close(logits[1], -4.0, EPS, "col1_logit");
    assert_close(logits[2], 0.0, EPS, "col2_logit");
    assert_close(logits[3], 0.0, EPS, "col3_logit");

    let probs = softmax(&logits, 1.0).unwrap();
    // Column 0 should dominate after softmax.
    assert_eq!(argmax(&probs), 0);
    assert!(probs[0] > 0.95, "dominant prob: {}", probs[0]);
}

// ═══════════════════════════════════════════════════════════════════
// 4. Conv1D → Pooling → Layer Norm Pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn conv1d_then_max_pool_then_rmsnorm() {
    let in_ch = 1;
    let out_ch = 1;
    let input_width = 8;
    // Input: [1, 2, 3, 4, 5, 6, 7, 8].
    let input: Vec<f32> = (1..=input_width).map(|x| x as f32).collect();

    // Identity-ish kernel: single weight = 1.0, kernel_size = 1.
    let weight = vec![1.0f32];
    let conv_cfg = Conv1dConfig {
        in_channels: in_ch,
        out_channels: out_ch,
        kernel_size: 1,
        stride: 1,
        padding: PaddingMode::Zero(0),
        dilation: 1,
        groups: 1,
        bias: false,
    };

    let conv_out = conv1d_forward(&input, &weight, None, &conv_cfg).unwrap();
    assert_eq!(conv_out.len(), input_width);

    // Max pool with kernel=2, stride=2 → 4 outputs.
    let pool_cfg = PoolConfig { pool_type: PoolType::Max, kernel_size: 2, stride: 2, padding: 0 };
    let pooled = PoolingKernel::apply(&conv_out, &pool_cfg).unwrap();
    assert_eq!(pooled.len(), 4);
    // Max of consecutive pairs: [2, 4, 6, 8].
    assert_slice_close(&pooled, &[2.0, 4.0, 6.0, 8.0], EPS, "pool");

    // Fused residual-add + RMSNorm (add zeros as residual).
    let residual = vec![0.0f32; pooled.len()];
    let gamma = vec![1.0f32; pooled.len()];
    let normed = fused_add_normalize(&pooled, &residual, &gamma, 1e-6).unwrap();

    // Verify RMSNorm: output = x / rms(x).
    let sum_sq: f32 = pooled.iter().map(|x| x * x).sum();
    let rms = (sum_sq / pooled.len() as f32 + 1e-6).sqrt();
    let expected: Vec<f32> = pooled.iter().map(|&x| x / rms).collect();
    assert_slice_close(&normed, &expected, 1e-5, "rmsnorm");
}

#[test]
fn conv1d_with_bias_then_avg_pool() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 1 channel, width 4
    let weight = vec![1.0, 1.0]; // kernel_size = 2
    let bias = vec![0.5]; // single output channel
    let cfg = Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 2,
        stride: 1,
        padding: PaddingMode::Zero(0),
        dilation: 1,
        groups: 1,
        bias: true,
    };

    let conv_out = conv1d_forward(&input, &weight, Some(&bias), &cfg).unwrap();
    // output_width = (4 - 2)/1 + 1 = 3.
    // Values: [1+2+0.5, 2+3+0.5, 3+4+0.5] = [3.5, 5.5, 7.5].
    assert_slice_close(&conv_out, &[3.5, 5.5, 7.5], EPS, "conv1d_bias");

    // Global average pool.
    let pool_cfg =
        PoolConfig { pool_type: PoolType::GlobalAverage, kernel_size: 0, stride: 0, padding: 0 };
    let pooled = PoolingKernel::apply(&conv_out, &pool_cfg).unwrap();
    assert_eq!(pooled.len(), 1);
    let expected_avg = (3.5 + 5.5 + 7.5) / 3.0;
    assert_close(pooled[0], expected_avg, EPS, "global_avg");
}

// ═══════════════════════════════════════════════════════════════════
// 5. RoPE → Attention Pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn rope_preserves_vector_norm() {
    let head_dim = 8;
    let rope_cfg = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&rope_cfg);

    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let orig_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();

    let mut rotated = original.clone();
    apply_rope(&mut rotated, 3, head_dim, &freqs);

    let rot_norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Rotation is norm-preserving.
    assert_close(rot_norm, orig_norm, 1e-4, "rope_norm_preserve");
}

#[test]
fn rope_batch_then_attention_scores() {
    let head_dim = 4;
    let seq_len = 3;
    let num_heads = 1;
    let rope_cfg = RopeConfig::new(head_dim, 16);
    let freqs = compute_frequencies(&rope_cfg);

    // Q and K: each is [seq_len * num_heads * head_dim].
    let mut q = vec![1.0f32; seq_len * num_heads * head_dim];
    let mut k = vec![1.0f32; seq_len * num_heads * head_dim];

    apply_rope_batch(&mut q, 0, seq_len, num_heads, head_dim, &freqs);
    apply_rope_batch(&mut k, 0, seq_len, num_heads, head_dim, &freqs);

    // Compute attention scores: Q·K^T / sqrt(d).
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let dot: f32 = (0..head_dim).map(|d| q[i * head_dim + d] * k[j * head_dim + d]).sum();
            scores[i * seq_len + j] = dot * scale;
        }
    }

    let attn = softmax_batch(&scores, seq_len, 1.0).unwrap();

    // Diagonal should be strongest (same-position alignment).
    for i in 0..seq_len {
        let self_score = attn[i * seq_len + i];
        for j in 0..seq_len {
            if j != i {
                assert!(
                    self_score >= attn[i * seq_len + j] - 1e-6,
                    "pos {i}: self ({self_score}) < cross ({j}: {})",
                    attn[i * seq_len + j]
                );
            }
        }
    }
}

#[test]
fn rope_different_positions_yield_different_rotations() {
    let head_dim = 4;
    let rope_cfg = RopeConfig::new(head_dim, 16);
    let freqs = compute_frequencies(&rope_cfg);
    let base = vec![1.0, 2.0, 3.0, 4.0];

    let mut r0 = base.clone();
    apply_rope(&mut r0, 0, head_dim, &freqs);

    let mut r1 = base.clone();
    apply_rope(&mut r1, 1, head_dim, &freqs);

    let mut r5 = base.clone();
    apply_rope(&mut r5, 5, head_dim, &freqs);

    // All three must differ.
    assert_ne!(r0, r1, "pos 0 vs 1");
    assert_ne!(r1, r5, "pos 1 vs 5");
    assert_ne!(r0, r5, "pos 0 vs 5");
}

// ═══════════════════════════════════════════════════════════════════
// 6. Cross-Validation: Reduction + Softmax Consistency
// ═══════════════════════════════════════════════════════════════════

#[test]
fn reduction_max_agrees_with_softmax_argmax() {
    let data = vec![0.5, 3.2, 1.1, 2.8, 0.1];
    let max_val = reduce_f32(&data, ReductionOp::Max);
    let probs = softmax(&data, 1.0).unwrap();
    let max_idx = argmax(&probs);

    assert_close(max_val, data[max_idx], 0.0, "max_idx_match");
}

#[test]
fn row_reduction_sum_matches_softmax_row_sums() {
    let rows = 3;
    let cols = 4;
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.5).collect();

    let attn = softmax_batch(&data, cols, 1.0).unwrap();
    let row_sums = reduce_rows_f32(&attn, rows, cols, ReductionOp::Sum).unwrap();

    for (r, &s) in row_sums.iter().enumerate() {
        assert_close(s, 1.0, 1e-5, &format!("row_sum_{r}"));
    }
}

// ═══════════════════════════════════════════════════════════════════
// 7. Fused Softmax-Mask → Sampling Pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn fused_softmax_mask_then_argmax_respects_mask() {
    let scores = vec![1.0, 5.0, 3.0, 2.0];
    // Mask out index 1 (the highest raw score).
    let mask = vec![0.0, -1e9, 0.0, 0.0];
    let scale = 1.0;

    let probs = fused_softmax_mask(&scores, &mask, scale).unwrap();

    // Index 1 should have near-zero probability.
    assert!(probs[1] < 1e-6, "masked position prob: {}", probs[1]);
    // Index 2 (next highest) should win.
    assert_eq!(argmax(&probs), 2);
    // Still sums to ~1.
    assert_close(probs.iter().sum::<f32>(), 1.0, 1e-5, "masked_sum");
}

// ═══════════════════════════════════════════════════════════════════
// 8. Error Propagation Through Pipelines
// ═══════════════════════════════════════════════════════════════════

#[test]
fn empty_input_propagates_through_softmax_pipeline() {
    let empty: Vec<f32> = vec![];
    let res = softmax(&empty, 1.0);
    assert!(res.is_err(), "softmax on empty should fail");
}

#[test]
fn embedding_oob_index_propagates_error() {
    let table = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, dim=2
    let indices = vec![99u32]; // out of bounds
    let res = embedding_lookup(&table, &indices, 2);
    assert!(res.is_err(), "OOB embedding index should fail");
}

#[test]
fn fusion_dimension_mismatch_propagates() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0]; // wrong length
    let gamma = vec![1.0, 1.0, 1.0];

    let res = fused_add_normalize(&a, &b, &gamma, 1e-6);
    assert!(
        matches!(res, Err(FusionError::DimensionMismatch { .. })),
        "expected DimensionMismatch, got {res:?}"
    );
}

#[test]
fn conv1d_invalid_groups_propagates_error() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0];
    let cfg = Conv1dConfig {
        in_channels: 1,
        out_channels: 1,
        kernel_size: 1,
        stride: 1,
        padding: PaddingMode::Zero(0),
        dilation: 1,
        groups: 0, // invalid
        bias: false,
    };
    let res = conv1d_forward(&input, &weight, None, &cfg);
    assert!(res.is_err(), "groups=0 should fail");
}

// ═══════════════════════════════════════════════════════════════════
// 9. End-to-End: Embedding → Normalize → MatMul → Softmax
// ═══════════════════════════════════════════════════════════════════

#[test]
fn full_embedding_to_sampling_pipeline() {
    let vocab = 8;
    let dim = 4;
    let n_classes = 2;

    // Embedding table.
    let table: Vec<f32> = (0..vocab * dim).map(|i| ((i % 7) as f32) * 0.3 - 0.5).collect();

    // Look up token 3.
    let emb = embedding_lookup(&table, &[3], dim).unwrap();
    assert_eq!(emb.len(), dim);

    // Normalize embeddings.
    let mut normed = emb.clone();
    normalize_embeddings(&mut normed, dim);
    let norm: f32 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_close(norm, 1.0, 1e-5, "emb_norm");

    // Project through fused RMSNorm + linear → logits.
    let gamma = vec![1.0f32; dim];
    let weight = vec![0.5f32; n_classes * dim]; // [n_classes × dim]
    let logits = fused_rmsnorm_linear(&normed, &weight, &gamma, 1e-6).unwrap();
    assert_eq!(logits.len(), n_classes);

    // Softmax → probabilities.
    let probs = softmax(&logits, 1.0).unwrap();
    assert_close(probs.iter().sum::<f32>(), 1.0, 1e-5, "final_prob_sum");
    // With uniform weights both classes should be ~0.5.
    for &p in &probs {
        assert_close(p, 0.5, 0.05, "uniform_class_prob");
    }
}
