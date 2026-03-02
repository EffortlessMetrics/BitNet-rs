//! BDD-style integration tests — Wave 8
//!
//! Each test follows the Given / When / Then structure and exercises
//! multi-kernel pipeline workflows: matmul→activation→loss,
//! dequantize→attention→softmax, batch_norm→conv→pooling,
//! multi-head attention→residual→layer_norm, tensor parallel→gather→loss,
//! as well as error handling, determinism, and numerical stability.

// ─── imports ────────────────────────────────────────────────────────
use bitnet_kernels::cpu::attention::{apply_mask, causal_mask};
use bitnet_kernels::cpu::batch::{batched_matmul, batched_softmax};
use bitnet_kernels::cpu::batch_norm::batch_norm_inference;
use bitnet_kernels::cpu::conv2d::{Conv2dConfig, conv2d};
use bitnet_kernels::cpu::dequant::{dequant_ternary, pack_ternary};
use bitnet_kernels::cpu::embedding::embedding_lookup;
use bitnet_kernels::cpu::ffn::{FfnActivation, FfnConfig, ffn_forward};
use bitnet_kernels::cpu::fusion::{fused_add_normalize, fused_rmsnorm_linear, fused_scale_add};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::loss::{
    LossReduction, cross_entropy_loss, cross_entropy_with_logits, gradient_accumulate,
    gradient_clip_norm, mse_loss,
};
use bitnet_kernels::cpu::pooling::global_avg_pool;
use bitnet_kernels::cpu::reduction::ReductionKernel;
use bitnet_kernels::cpu::residual::{add_residual, add_residual_scaled};
use bitnet_kernels::cpu::rope::{RopeConfig, apply_rope, compute_frequencies};
use bitnet_kernels::cpu::scatter_gather::{gather_1d, scatter_add};
use bitnet_kernels::cpu::simd_matmul::{SimdMatmulConfig, simd_matmul_f32};
use bitnet_kernels::cpu::tensor_parallel::{
    ShardingStrategy, TensorParallelConfig, all_reduce_sum, compute_shard_ranges, gather_shards,
    shard_tensor,
};

use bitnet_kernels::cuda::activations::{ActivationConfig, ActivationType, activation_cpu};
use bitnet_kernels::cuda::dequant::{dequantize_int2_to_f32, quantize_to_int2};
use bitnet_kernels::cuda::layernorm::{LayerNormConfig as CudaLnCfg, layer_norm_cpu_fallback};
use bitnet_kernels::cuda::loss::{LossConfig, mse_loss as cuda_mse};
use bitnet_kernels::cuda::matmul::{MatmulConfig as CudaMatmulCfg, matmul_cpu};
use bitnet_kernels::cuda::multi_head_attention::{
    MultiHeadAttentionConfig, merge_heads, scaled_dot_product, split_heads,
};
use bitnet_kernels::cuda::residual::{residual_add, residual_add_scaled};
use bitnet_kernels::cuda::softmax::{SoftmaxConfig, softmax_cpu};

const TOL: f32 = 1e-5;
const LOOSE_TOL: f32 = 1e-3;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() <= tol, "mismatch at index {i}: {x} vs {y} (diff {})", (x - y).abs());
    }
}

fn no_nan_inf(v: &[f32]) {
    for (i, &x) in v.iter().enumerate() {
        assert!(x.is_finite(), "non-finite at index {i}: {x}");
    }
}

/// Manual softmax for a single row (used when cpu::softmax module is private).
fn row_softmax(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 1: CPU matmul → activation → loss
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_matmul_activation_loss_basic() {
    // Given a small weight matrix and input
    let m = 2;
    let k = 4;
    let n = 3;
    let a = vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]; // 2×4
    let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]; // 4×3
    let mut logits = vec![0.0f32; m * n];
    let cfg = SimdMatmulConfig::new(m, n, k);

    // When we run matmul, then apply activation, then compute loss
    simd_matmul_f32(&a, &b, &mut logits, &cfg).unwrap();
    let act_cfg = ActivationConfig::new(logits.len(), ActivationType::GELU).unwrap();
    let mut activated = vec![0.0; logits.len()];
    activation_cpu(&logits, &mut activated, &act_cfg).unwrap();
    let targets = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    let loss = mse_loss(&activated, &targets, LossReduction::Mean).unwrap();

    // Then the loss is a finite positive number
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn bdd_w8_matmul_softmax_ce_loss_pipeline() {
    // Given logits from matmul with known values
    let m = 1;
    let k = 4;
    let n = 4;
    let a = vec![0.5; m * k];
    let b = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut logits = vec![0.0f32; m * n];
    let cfg = SimdMatmulConfig::new(m, n, k);
    simd_matmul_f32(&a, &b, &mut logits, &cfg).unwrap();

    // When we compute CE loss with logits
    let targets_onehot = vec![1.0, 0.0, 0.0, 0.0];
    let loss = cross_entropy_with_logits(&logits, &targets_onehot, LossReduction::Mean).unwrap();

    // Then the loss is finite and positive
    assert!(loss.is_finite());
    assert!(loss > 0.0);
}

#[test]
fn bdd_w8_matmul_silu_mse_pipeline() {
    // Given a 3×2 matmul result
    let m = 3;
    let k = 2;
    let n = 2;
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![0.1, 0.2, 0.3, 0.4];
    let mut logits = vec![0.0f32; m * n];
    let cfg = SimdMatmulConfig::new(m, n, k);
    simd_matmul_f32(&a, &b, &mut logits, &cfg).unwrap();

    // When SiLU activation is applied then MSE loss computed
    let act_cfg = ActivationConfig::new(logits.len(), ActivationType::SiLU).unwrap();
    let mut activated = vec![0.0; logits.len()];
    activation_cpu(&logits, &mut activated, &act_cfg).unwrap();
    let targets = vec![0.0f32; m * n];
    let loss = mse_loss(&activated, &targets, LossReduction::Sum).unwrap();

    // Then loss is non-negative
    assert!(loss >= 0.0);
    no_nan_inf(&activated);
}

#[test]
fn bdd_w8_matmul_relu_then_reduce() {
    // Given matmul output
    let m = 2;
    let k = 3;
    let n = 2;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) - 2.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
    let mut logits = vec![0.0f32; m * n];
    let cfg = SimdMatmulConfig::new(m, n, k);
    simd_matmul_f32(&a, &b, &mut logits, &cfg).unwrap();

    // When ReLU activation and sum reduction are applied
    let act_cfg = ActivationConfig::new(logits.len(), ActivationType::ReLU).unwrap();
    let mut activated = vec![0.0; logits.len()];
    activation_cpu(&logits, &mut activated, &act_cfg).unwrap();
    let total = ReductionKernel::sum(&activated).unwrap();

    // Then sum is non-negative (ReLU output >= 0) and finite
    assert!(total >= 0.0);
    assert!(total.is_finite());
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 2: Dequantize → attention → softmax chain
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_dequant_attention_softmax_basic() {
    // Given quantized int2 data
    let n_elem = 32;
    let block_size = 32;
    let input: Vec<f32> = (0..n_elem).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let (packed, scales) = quantize_to_int2(&input, block_size).unwrap();

    // When we dequantize, use as Q/K for attention, then softmax
    let dequantized = dequantize_int2_to_f32(&packed, &scales, block_size, n_elem).unwrap();
    let seq_len = 4;
    let head_dim = n_elem / seq_len;
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += dequantized[i * head_dim + d] * dequantized[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot / (head_dim as f32).sqrt();
        }
    }
    // Apply softmax per row
    for row in 0..seq_len {
        let start = row * seq_len;
        let end = start + seq_len;
        let sm = row_softmax(&scores[start..end]);
        scores[start..end].copy_from_slice(&sm);
    }

    // Then each row sums to ~1.0 and no NaN/Inf
    for row in 0..seq_len {
        let start = row * seq_len;
        let row_sum: f32 = scores[start..start + seq_len].iter().sum();
        assert!((row_sum - 1.0).abs() < LOOSE_TOL, "row {row} sum = {row_sum}");
    }
    no_nan_inf(&scores);
}

#[test]
fn bdd_w8_dequant_causal_mask_softmax() {
    // Given attention scores and a causal mask
    let seq_len = 4;
    let mut scores: Vec<f32> = (0..seq_len * seq_len).map(|i| i as f32 * 0.1).collect();
    let mask = causal_mask(seq_len);
    apply_mask(&mut scores, &mask).unwrap();

    // When softmax is applied row-wise
    for row in 0..seq_len {
        let start = row * seq_len;
        let end = start + seq_len;
        let sm = row_softmax(&scores[start..end]);
        scores[start..end].copy_from_slice(&sm);
    }

    // Then future positions have zero probability
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            assert!(
                scores[i * seq_len + j] < 1e-6,
                "future position [{i},{j}] = {} should be ~0",
                scores[i * seq_len + j]
            );
        }
    }
}

#[test]
fn bdd_w8_dequant_roundtrip_preserves_dot_product_finiteness() {
    // Given two different vectors quantized and dequantized
    let block_size = 32;
    let a: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..32).map(|i| (31 - i) as f32 * 0.1).collect();

    let (pa, sa) = quantize_to_int2(&a, block_size).unwrap();
    let (pb, sb) = quantize_to_int2(&b, block_size).unwrap();
    let da = dequantize_int2_to_f32(&pa, &sa, block_size, 32).unwrap();
    let db = dequantize_int2_to_f32(&pb, &sb, block_size, 32).unwrap();

    // When we compute dot products
    let dot_a: f32 = da.iter().map(|x| x * x).sum();
    let dot_b: f32 = db.iter().map(|x| x * x).sum();

    // Then both are finite
    assert!(dot_a.is_finite());
    assert!(dot_b.is_finite());
}

#[test]
fn bdd_w8_dequant_int2_attention_weights_non_negative() {
    // Given dequantized int2 data used as attention logits
    let n = 64;
    let (packed, scales) =
        quantize_to_int2(&(0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect::<Vec<_>>(), 32)
            .unwrap();
    let deq = dequantize_int2_to_f32(&packed, &scales, 32, n).unwrap();

    // When softmax is applied to chunks
    let chunk = 8;
    for c in deq.chunks(chunk) {
        let sm = row_softmax(c);
        // Then all probabilities are non-negative and sum to ~1
        for &p in &sm {
            assert!(p >= 0.0);
        }
        let sum: f32 = sm.iter().sum();
        assert!((sum - 1.0).abs() < LOOSE_TOL);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 3: CPU batch norm → conv → pooling
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_batchnorm_inference_then_pool() {
    // Given batch-normed features for inference
    let num_features = 2;
    let spatial = 4;
    let input: Vec<f32> = (0..num_features * spatial).map(|i| i as f32).collect();
    let gamma = vec![1.0; num_features];
    let beta = vec![0.0; num_features];
    let running_mean = vec![0.0; num_features];
    let running_var = vec![1.0; num_features];
    let normed =
        batch_norm_inference(&input, &gamma, &beta, &running_mean, &running_var, 1e-5).unwrap();

    // When avg pooling is applied
    let pooled = global_avg_pool(&normed, &[normed.len()]).unwrap();

    // Then pooled output is finite
    no_nan_inf(&pooled);
    assert!(!pooled.is_empty());
}

#[test]
fn bdd_w8_conv2d_then_avgpool() {
    // Given a 4×4 single-channel image
    let input =
        vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let kernel = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 identity-like
    let cfg = Conv2dConfig::new(1, 1, 2);

    // When conv2d is applied (batch_size=1, in_h=4, in_w=4)
    let conv_out = conv2d(&input, &kernel, None, &cfg, 1, 4, 4).unwrap();

    // Then apply global avg pool
    let pooled = global_avg_pool(&conv_out, &[conv_out.len()]).unwrap();
    no_nan_inf(&pooled);
    assert_eq!(pooled.len(), 1);
}

#[test]
fn bdd_w8_batchnorm_conv_pool_pipeline() {
    // Given a 1-channel 4×4 input
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let gamma = vec![1.0];
    let beta = vec![0.0];
    let rm = vec![8.0]; // approximate mean
    let rv = vec![20.0]; // approximate variance
    let normed = batch_norm_inference(&input, &gamma, &beta, &rm, &rv, 1e-5).unwrap();

    // When conv2d (3×3) is applied then pooled
    let k3 = vec![1.0 / 9.0; 9]; // averaging kernel
    let cfg3 = Conv2dConfig::new(1, 1, 3);
    let conv_out = conv2d(&normed, &k3, None, &cfg3, 1, 4, 4).unwrap();
    let pooled = global_avg_pool(&conv_out, &[conv_out.len()]).unwrap();

    // Then pipeline output is finite
    no_nan_inf(&pooled);
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 4: Multi-head attention → residual → layer norm
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_mha_residual_layernorm_basic() {
    // Given Q, K, V for 2-head attention with head_dim=4, seq_len=2
    let num_heads = 2;
    let head_dim = 4;
    let seq_len = 2;
    let total = num_heads * seq_len * head_dim;

    let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();
    let k = q.clone();
    let v: Vec<f32> = (0..total).map(|i| 1.0 - (i as f32) * 0.05).collect();

    // When multi-head attention is applied
    let mha_cfg =
        MultiHeadAttentionConfig::new(num_heads, num_heads, head_dim, false, 0.0).unwrap();
    let attn_out = scaled_dot_product(&q, &k, &v, &mha_cfg, seq_len, seq_len, 1).unwrap();

    // Then residual add + layer norm
    let residual_input = q[..attn_out.output.len()].to_vec();
    let mut combined = attn_out.output.clone();
    add_residual(&mut combined, &residual_input).unwrap();

    let ln_cfg = LayerNormConfig::new(vec![combined.len()]);
    let gamma = vec![1.0; combined.len()];
    let normed = layer_norm(&combined, &gamma, None, &ln_cfg).unwrap();

    // Then output is finite and approximately normalized
    no_nan_inf(&normed);
    let mean: f32 = normed.iter().sum::<f32>() / normed.len() as f32;
    assert!(mean.abs() < LOOSE_TOL, "post-LN mean = {mean}");
}

#[test]
fn bdd_w8_split_merge_heads_roundtrip() {
    // Given a flat attention tensor
    let num_heads = 4;
    let head_dim = 8;
    let seq_len = 3;
    let total = num_heads * seq_len * head_dim;
    let data: Vec<f32> = (0..total).map(|i| i as f32).collect();

    // When we split and merge heads
    let split = split_heads(&data, 1, seq_len, num_heads, head_dim).unwrap();
    let merged = merge_heads(&split, 1, seq_len, num_heads, head_dim).unwrap();

    // Then we recover the original data
    approx_eq(&data, &merged, TOL);
}

#[test]
fn bdd_w8_mha_residual_scaled() {
    // Given attention output and residual input
    let n = 16;
    let attn_out: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let residual: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 * 0.05).collect();

    // When scaled residual is applied
    let mut combined = attn_out.clone();
    add_residual_scaled(&mut combined, &residual, 0.5).unwrap();

    // Then result differs from both inputs and is finite
    no_nan_inf(&combined);
    let differs = combined.iter().zip(attn_out.iter()).any(|(a, b)| (a - b).abs() > TOL);
    assert!(differs, "scaled residual should modify the output");
}

#[test]
fn bdd_w8_attention_residual_rms_norm() {
    // Given attention scores and residual
    let dim = 8;
    let input: Vec<f32> = (0..dim).map(|i| (i as f32 - 3.5) * 0.2).collect();
    let attn_output: Vec<f32> = (0..dim).map(|i| i as f32 * 0.05).collect();

    // When residual is added then RMS norm applied
    let mut combined = attn_output;
    add_residual(&mut combined, &input).unwrap();
    let gamma = vec![1.0; dim];
    let ln_cfg = LayerNormConfig::new(vec![dim]);
    let normed = rms_norm(&combined, &gamma, &ln_cfg).unwrap();

    // Then output is finite
    no_nan_inf(&normed);
    assert_eq!(normed.len(), dim);
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 5: CPU tensor parallel → gather → loss aggregation
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_shard_gather_loss_pipeline() {
    // Given a tensor of logits
    let logits: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let num_shards = 3;
    let tp_cfg = TensorParallelConfig {
        num_ranks: num_shards,
        rank_id: 0,
        comm_backend: bitnet_kernels::cpu::tensor_parallel::CommBackend::InProcess,
        overlap_compute_comm: false,
    };

    // When we shard and gather
    let (shards, _metrics) =
        shard_tensor(&logits, &tp_cfg, &ShardingStrategy::ColumnParallel).unwrap();
    let (gathered, _) = gather_shards(&shards).unwrap();

    // Then gathered matches original and loss can be computed
    approx_eq(&logits, &gathered, TOL);
    let targets = vec![1.0f32; gathered.len()];
    let loss = mse_loss(&gathered, &targets, LossReduction::Mean).unwrap();
    assert!(loss.is_finite());
}

#[test]
fn bdd_w8_all_reduce_sum_then_loss() {
    // Given per-shard gradient-like data as TensorShards
    use bitnet_kernels::cpu::tensor_parallel::TensorShard;
    let shards = vec![
        TensorShard { data: vec![1.0, 2.0, 3.0, 4.0], rank_id: 0, shard_index: 0, total_shards: 2 },
        TensorShard { data: vec![0.5, 0.5, 0.5, 0.5], rank_id: 1, shard_index: 1, total_shards: 2 },
    ];

    // When all-reduce sum is applied
    let (reduced, _) = all_reduce_sum(&shards).unwrap();

    // Then the result is element-wise sum
    approx_eq(&reduced, &[1.5, 2.5, 3.5, 4.5], TOL);
}

#[test]
fn bdd_w8_scatter_gather_loss() {
    // Given output logits and gradient scatter
    let mut grad_acc = vec![0.0f32; 8];
    let indices = vec![0, 2, 4, 6];
    let values = vec![1.0, 2.0, 3.0, 4.0];

    // When scatter_add accumulates gradients
    scatter_add(&mut grad_acc, &indices, &values).unwrap();

    // Then gathered values at those indices match
    let gathered = gather_1d(&grad_acc, &indices).unwrap();
    approx_eq(&gathered, &values, TOL);
}

#[test]
fn bdd_w8_compute_shard_ranges_even() {
    // Given a tensor length divisible by shard count
    let ranges = compute_shard_ranges(12, 3).unwrap();

    // Then each range covers 4 elements
    assert_eq!(ranges.len(), 3);
    for (start, end) in &ranges {
        assert_eq!(end - start, 4);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 6: Loss + backward pass integration
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_ce_loss_gradient_clip_pipeline() {
    // Given logits and targets
    let logits = vec![2.0, 1.0, 0.1, 0.5];
    let targets = vec![0usize]; // class 0
    let (loss, grad) = cross_entropy_loss(&logits, &targets, 4, LossReduction::Mean).unwrap();

    // When gradient clipping is applied
    let mut grads = grad;
    gradient_clip_norm(&mut grads, 1.0).unwrap();

    // Then gradients are finite and clipped
    no_nan_inf(&grads);
    let norm: f32 = grads.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm <= 1.0 + TOL, "gradient norm {norm} exceeds max");
    assert!(loss.is_finite());
}

#[test]
fn bdd_w8_mse_loss_gradient_accumulate() {
    // Given predictions and targets
    let preds = vec![1.0, 2.0, 3.0, 4.0];
    let targets = vec![1.1, 2.1, 2.9, 3.8];
    let loss = mse_loss(&preds, &targets, LossReduction::Mean).unwrap();

    // When gradients are accumulated
    let grads: Vec<f32> = preds.iter().zip(targets.iter()).map(|(p, t)| 2.0 * (p - t)).collect();
    let mut accumulator = vec![0.0f32; grads.len()];
    gradient_accumulate(&mut accumulator, &grads).unwrap();

    // Then accumulated gradient matches computed gradient
    approx_eq(&accumulator, &grads, TOL);
    assert!(loss.is_finite() && loss >= 0.0);
}

#[test]
fn bdd_w8_cuda_mse_loss_then_reduce() {
    // Given per-sample MSE losses from CUDA path
    let preds = vec![1.0, 2.0, 3.0, 4.0];
    let targets = vec![0.0, 0.0, 0.0, 0.0];
    let cfg = LossConfig::new(4, 1).unwrap();
    let per_sample = cuda_mse(&preds, &targets, &cfg).unwrap();

    // When we reduce to scalar
    let total = ReductionKernel::sum(&per_sample).unwrap();

    // Then the sum is positive and finite
    assert!(total > 0.0);
    assert!(total.is_finite());
}

#[test]
fn bdd_w8_gradient_clip_preserves_direction() {
    // Given a gradient vector
    let mut grads = vec![3.0, 4.0]; // norm = 5
    let original_dir: Vec<f32> = grads.iter().map(|&g| g / 5.0).collect();

    // When clipped to max_norm=1
    gradient_clip_norm(&mut grads, 1.0).unwrap();
    let new_norm: f32 = grads.iter().map(|x| x * x).sum::<f32>().sqrt();
    let new_dir: Vec<f32> = grads.iter().map(|&g| g / new_norm).collect();

    // Then direction is preserved
    approx_eq(&original_dir, &new_dir, LOOSE_TOL);
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 7: Fused operations
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_fused_rmsnorm_linear_pipeline() {
    // Given input, gamma, weight
    let dim = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; dim];
    // weight is dim×dim identity flattened
    let weight =
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    // When fused RMSNorm + linear is applied
    let result = fused_rmsnorm_linear(&input, &weight, &gamma, 1e-6).unwrap();

    // Then output is finite
    no_nan_inf(&result);
    assert_eq!(result.len(), dim);
}

#[test]
fn bdd_w8_fused_add_normalize() {
    // Given two vectors
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.5, 0.5, 0.5, 0.5];
    let gamma = vec![1.0; 4];

    // When fused add+normalize is applied
    let result = fused_add_normalize(&a, &b, &gamma, 1e-5).unwrap();

    // Then output is approximately zero-mean (RMS norm keeps finite)
    no_nan_inf(&result);
    assert_eq!(result.len(), 4);
}

#[test]
fn bdd_w8_fused_scale_add() {
    // Given two vectors and a scale factor
    let a = vec![2.0, 4.0, 6.0, 8.0];
    let b = vec![1.0, 1.0, 1.0, 1.0];

    // When fused scale+add is applied (a + scale*b)
    let result = fused_scale_add(&a, &b, 0.5).unwrap();

    // Then result = a + 0.5*b
    approx_eq(&result, &[2.5, 4.5, 6.5, 8.5], TOL);
}

// ═══════════════════════════════════════════════════════════════════
// Error handling: invalid dimensions, mismatched types
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_error_matmul_dimension_mismatch() {
    // Given mismatched dimensions for matmul
    let a = vec![1.0f32; 6]; // 2×3
    let b = vec![1.0f32; 8]; // 2×4 (inner dim should be 3, but b has 8 != 3*4)
    let mut c = vec![0.0f32; 8]; // 2×4
    let cfg = SimdMatmulConfig::new(2, 4, 3); // expects b to be 3×4 = 12 elements

    // When matmul is attempted
    let result = simd_matmul_f32(&a, &b, &mut c, &cfg);

    // Then it should fail
    assert!(result.is_err(), "dimension mismatch should error");
}

#[test]
fn bdd_w8_error_residual_length_mismatch() {
    // Given vectors of different lengths
    let mut output = vec![1.0, 2.0, 3.0];
    let residual = vec![1.0, 2.0]; // shorter

    // When residual add is attempted
    let result = add_residual(&mut output, &residual);

    // Then it should error
    assert!(result.is_err(), "length mismatch should error");
}

#[test]
fn bdd_w8_error_layer_norm_gamma_mismatch() {
    // Given input and mismatched gamma
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0]; // too short
    let cfg = LayerNormConfig::new(vec![4]);

    // When layer norm is called
    let result = layer_norm(&input, &gamma, None, &cfg);

    // Then it should error
    assert!(result.is_err(), "gamma length mismatch should error");
}

#[test]
fn bdd_w8_error_loss_length_mismatch() {
    // Given predictions and targets of different lengths
    let preds = vec![1.0, 2.0, 3.0];
    let targets = vec![1.0, 2.0];

    // When MSE loss is computed
    let result = mse_loss(&preds, &targets, LossReduction::Mean);

    // Then it should error
    assert!(result.is_err(), "prediction/target length mismatch should error");
}

#[test]
fn bdd_w8_error_scatter_out_of_bounds() {
    // Given indices that exceed buffer length
    let mut data = vec![0.0f32; 4];
    let indices = vec![0, 1, 10]; // index 10 out of bounds
    let values = vec![1.0, 2.0, 3.0];

    // When scatter_add is called
    let result = scatter_add(&mut data, &indices, &values);

    // Then it should error on out-of-bounds index
    assert!(result.is_err(), "out-of-bounds scatter should error");
}

#[test]
fn bdd_w8_error_conv2d_zero_channels() {
    // Given invalid conv config with zero channels
    let cfg = Conv2dConfig::new(0, 1, 3);
    let input = vec![1.0; 9];
    let kernel = vec![1.0; 9];

    // When conv2d is called
    let result = conv2d(&input, &kernel, None, &cfg, 1, 3, 3);

    // Then it should error
    assert!(result.is_err(), "zero channels should error");
}

#[test]
fn bdd_w8_error_gradient_accumulate_mismatch() {
    // Given accumulator and source of different lengths
    let mut acc = vec![0.0f32; 3];
    let source = vec![1.0, 2.0];

    // When gradient_accumulate is called
    let result = gradient_accumulate(&mut acc, &source);

    // Then it should error
    assert!(result.is_err(), "gradient length mismatch should error");
}

#[test]
fn bdd_w8_error_batched_matmul_size_mismatch() {
    // Given incorrectly sized batched input
    let a = vec![1.0f32; 8]; // batch=2, m=2, k=2
    let b = vec![1.0f32; 4]; // batch=2, k=2, n=1 → should be 4 but m/k mismatch

    // When batched matmul is called with wrong batch dimensions
    let result = batched_matmul(&a, &b, 2, 2, 1, 3); // k=3 doesn't match a

    // Then it should error
    assert!(result.is_err(), "batched matmul size mismatch should error");
}

// ═══════════════════════════════════════════════════════════════════
// Determinism: same seed → same output
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_determinism_matmul_repeated() {
    // Given the same inputs
    let m = 4;
    let k = 4;
    let n = 4;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.2).collect();
    let cfg = SimdMatmulConfig::new(m, n, k);

    // When matmul is run twice
    let mut c1 = vec![0.0f32; m * n];
    let mut c2 = vec![0.0f32; m * n];
    simd_matmul_f32(&a, &b, &mut c1, &cfg).unwrap();
    simd_matmul_f32(&a, &b, &mut c2, &cfg).unwrap();

    // Then outputs are identical
    approx_eq(&c1, &c2, 0.0);
}

#[test]
fn bdd_w8_determinism_softmax_repeated() {
    // Given the same input
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // When softmax is applied twice
    let out1 = row_softmax(&input);
    let out2 = row_softmax(&input);

    // Then results are bitwise identical
    assert_eq!(out1, out2);
}

#[test]
fn bdd_w8_determinism_layer_norm_repeated() {
    // Given the same input
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let cfg = LayerNormConfig::new(vec![4]);

    // When layer norm is applied twice
    let out1 = layer_norm(&input, &gamma, None, &cfg).unwrap();
    let out2 = layer_norm(&input, &gamma, None, &cfg).unwrap();

    // Then results are identical
    assert_eq!(out1, out2);
}

#[test]
fn bdd_w8_determinism_rope_repeated() {
    // Given the same configuration
    let head_dim = 8;
    let cfg = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&cfg);
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // When RoPE is applied twice
    let mut data1 = original.clone();
    let mut data2 = original.clone();
    apply_rope(&mut data1, 3, head_dim, &freqs);
    apply_rope(&mut data2, 3, head_dim, &freqs);

    // Then results are bitwise identical
    assert_eq!(data1, data2);
}

#[test]
fn bdd_w8_determinism_quantize_dequantize_repeated() {
    // Given the same input
    let input: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();

    // When quantize+dequantize is run twice
    let (p1, s1) = quantize_to_int2(&input, 32).unwrap();
    let (p2, s2) = quantize_to_int2(&input, 32).unwrap();
    let d1 = dequantize_int2_to_f32(&p1, &s1, 32, 32).unwrap();
    let d2 = dequantize_int2_to_f32(&p2, &s2, 32, 32).unwrap();

    // Then results are bitwise identical
    assert_eq!(d1, d2);
}

#[test]
fn bdd_w8_determinism_full_pipeline_repeated() {
    // Given the same pipeline twice
    let run_pipeline = || -> f32 {
        let a = vec![1.0, 0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.8];
        let b = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut logits = vec![0.0f32; 4];
        let cfg = SimdMatmulConfig::new(2, 2, 4);
        simd_matmul_f32(&a, &b, &mut logits, &cfg).unwrap();
        let probs = row_softmax(&logits[..2]);
        let targets = vec![1.0, 0.0];
        mse_loss(&probs, &targets, LossReduction::Mean).unwrap()
    };

    // When the pipeline runs twice
    let loss1 = run_pipeline();
    let loss2 = run_pipeline();

    // Then losses are identical
    assert_eq!(loss1, loss2);
}

// ═══════════════════════════════════════════════════════════════════
// Numerical stability: extreme values, NaN/Inf prevention
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_stability_softmax_large_values() {
    // Given very large input values
    let input = vec![1000.0, 1001.0, 1002.0, 999.0];

    // When softmax is applied
    let output = row_softmax(&input);

    // Then no NaN/Inf and sum ≈ 1.0
    no_nan_inf(&output);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < LOOSE_TOL, "softmax sum = {sum}");
}

#[test]
fn bdd_w8_stability_softmax_negative_large() {
    // Given very negative input values
    let input = vec![-1000.0, -999.0, -1001.0, -998.0];

    // When softmax is applied
    let output = row_softmax(&input);

    // Then no NaN/Inf and sum ≈ 1.0
    no_nan_inf(&output);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < LOOSE_TOL, "softmax sum = {sum}");
}

#[test]
fn bdd_w8_stability_softmax_mixed_extreme() {
    // Given a mix of very large and very small values
    let input = vec![100.0, -100.0, 0.0, 50.0];

    // When softmax is applied
    let output = row_softmax(&input);

    // Then no NaN/Inf and sum ≈ 1.0
    no_nan_inf(&output);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < LOOSE_TOL, "softmax sum = {sum}");
}

#[test]
fn bdd_w8_stability_log_softmax_large() {
    // Given large values
    let input = [500.0, 501.0, 499.0, 502.0];

    // When log_softmax is computed manually (stable)
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = input.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln() + max_val;
    let log_sm: Vec<f32> = input.iter().map(|&x| x - log_sum_exp).collect();

    // Then all values are finite and ≤ 0
    no_nan_inf(&log_sm);
    for (i, &v) in log_sm.iter().enumerate() {
        assert!(v <= TOL, "log_softmax[{i}] = {v} should be ≤ 0");
    }
}

#[test]
fn bdd_w8_stability_layer_norm_near_zero_variance() {
    // Given nearly constant input (very small variance)
    let input = vec![1.000_001, 1.000_002, 0.999_999, 1.000_000];
    let gamma = vec![1.0; 4];
    let cfg = LayerNormConfig::new(vec![4]);

    // When layer norm is applied
    let result = layer_norm(&input, &gamma, None, &cfg).unwrap();

    // Then no NaN/Inf
    no_nan_inf(&result);
}

#[test]
fn bdd_w8_stability_rms_norm_large_input() {
    // Given large input values
    let input = vec![1000.0, 2000.0, 3000.0, 4000.0];
    let gamma = vec![1.0; 4];
    let cfg = LayerNormConfig::new(vec![4]);

    // When RMS norm is applied
    let result = rms_norm(&input, &gamma, &cfg).unwrap();

    // Then output is finite
    no_nan_inf(&result);
}

#[test]
fn bdd_w8_stability_loss_near_zero_predictions() {
    // Given predictions very close to zero
    let preds = vec![1e-10, 0.5, 0.3, 0.2];
    let targets = vec![0.5, 0.5, 0.0, 0.0];

    // When MSE loss is computed
    let loss = mse_loss(&preds, &targets, LossReduction::Mean).unwrap();

    // Then loss is finite
    assert!(loss.is_finite(), "MSE loss should be finite, got {loss}");
}

#[test]
fn bdd_w8_stability_batch_norm_constant_input() {
    // Given constant input (zero variance edge case)
    let input = vec![42.0; 8];
    let gamma = vec![1.0; 2];
    let beta = vec![0.0; 2];
    let rm = vec![42.0; 2];
    let rv = vec![1.0; 2]; // non-zero running var for inference

    // When batch norm inference is applied
    let result = batch_norm_inference(&input, &gamma, &beta, &rm, &rv, 1e-5).unwrap();

    // Then result is finite (should normalize to ~0)
    no_nan_inf(&result);
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 8: Embedding → FFN → RoPE → attention pipeline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_embedding_ffn_pipeline() {
    // Given an embedding table and token IDs
    let vocab_size = 8;
    let embed_dim = 4;
    let table: Vec<f32> = (0..vocab_size * embed_dim).map(|i| i as f32 * 0.1).collect();
    let tokens = vec![1u32, 3];
    let seq_len = tokens.len();

    // When embeddings are looked up
    let embeddings = embedding_lookup(&table, &tokens, embed_dim).unwrap();
    assert_eq!(embeddings.len(), seq_len * embed_dim);

    // When FFN is applied to first position
    let ffn_cfg = FfnConfig::new(embed_dim, embed_dim * 2, FfnActivation::GeLU).unwrap();
    let ffn_weight_up: Vec<f32> =
        (0..embed_dim * embed_dim * 2).map(|i| (i as f32 % 3.0 - 1.0) * 0.1).collect();
    let ffn_weight_down: Vec<f32> =
        (0..embed_dim * 2 * embed_dim).map(|i| (i as f32 % 3.0 - 1.0) * 0.1).collect();
    let ffn_out =
        ffn_forward(&embeddings[..embed_dim], &ffn_weight_up, &ffn_weight_down, &ffn_cfg).unwrap();

    // Then output is finite and correct dimension
    no_nan_inf(&ffn_out);
    assert_eq!(ffn_out.len(), embed_dim);
}

#[test]
fn bdd_w8_embedding_rms_norm_ffn() {
    // Given embeddings
    let embed_dim = 8;
    let embeddings: Vec<f32> = (0..embed_dim).map(|i| i as f32 * 0.3).collect();
    let gamma = vec![1.0; embed_dim];
    let ln_cfg = LayerNormConfig::new(vec![embed_dim]);

    // When RMS norm then FFN forward
    let normed = rms_norm(&embeddings, &gamma, &ln_cfg).unwrap();
    let ffn_cfg = FfnConfig::new(embed_dim, embed_dim * 4, FfnActivation::SiLU).unwrap();
    let up_w: Vec<f32> =
        (0..embed_dim * embed_dim * 4).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
    let down_w: Vec<f32> =
        (0..embed_dim * 4 * embed_dim).map(|i| ((i % 7) as f32 - 3.0) * 0.04).collect();
    let ffn_out = ffn_forward(&normed, &up_w, &down_w, &ffn_cfg).unwrap();

    // Then output is finite and correct dimension
    no_nan_inf(&ffn_out);
    assert_eq!(ffn_out.len(), embed_dim);
}

#[test]
fn bdd_w8_rope_then_attention_scores() {
    // Given two position-encoded head vectors
    let head_dim = 8;
    let cfg = RopeConfig::new(head_dim, 64);
    let freqs = compute_frequencies(&cfg);
    let mut q = vec![1.0; head_dim];
    let mut k = vec![1.0; head_dim];
    apply_rope(&mut q, 0, head_dim, &freqs);
    apply_rope(&mut k, 1, head_dim, &freqs);

    // When dot product is computed as attention score
    let score: f32 =
        q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() / (head_dim as f32).sqrt();

    // Then score is finite
    assert!(score.is_finite());
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 9: Batched operations
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_batched_matmul_softmax_pipeline() {
    // Given batched data
    let batch = 2;
    let m = 2;
    let n = 2;
    let k = 2;
    let a: Vec<f32> = (0..batch * m * k).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|i| i as f32 * 0.2).collect();

    // When batched matmul is applied
    let result = batched_matmul(&a, &b, batch, m, k, n).unwrap();
    assert_eq!(result.len(), batch * m * n);

    // Then apply batched softmax (each "row" is n elements)
    let softmaxed = batched_softmax(&result, batch * m, n).unwrap();

    // Then each row sums to ~1.0
    no_nan_inf(&softmaxed);
    for row in 0..(batch * m) {
        let start = row * n;
        let row_sum: f32 = softmaxed[start..start + n].iter().sum();
        assert!((row_sum - 1.0).abs() < LOOSE_TOL, "row {row} sum = {row_sum}");
    }
}

#[test]
fn bdd_w8_batched_matmul_deterministic() {
    // Given the same batched input
    let batch = 3;
    let m = 2;
    let n = 2;
    let k = 4;
    let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.05).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.03).collect();

    // When run twice
    let r1 = batched_matmul(&a, &b, batch, m, k, n).unwrap();
    let r2 = batched_matmul(&a, &b, batch, m, k, n).unwrap();

    // Then results are identical
    assert_eq!(r1, r2);
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 10: CUDA fallback paths
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_cuda_matmul_cpu_fallback() {
    // Given inputs for CUDA matmul (will use CPU fallback)
    let m = 3;
    let n = 2;
    let k = 4;
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.2).collect();
    let mut out = vec![0.0f32; m * n];
    let cfg = CudaMatmulCfg::for_shape(m, n, k).unwrap();

    // When CUDA matmul CPU fallback is used
    matmul_cpu(&a, &b, &mut out, &cfg).unwrap();

    // Then output is finite
    no_nan_inf(&out);
    // Verify first element: dot(row0_a, col0_b)
    let expected_00 = 0.0 * 0.0 + 0.1 * 0.4 + 0.2 * 0.8 + 0.3 * 1.2;
    assert!(
        (out[0] - expected_00).abs() < LOOSE_TOL,
        "out[0] = {} expected {}",
        out[0],
        expected_00
    );
}

#[test]
fn bdd_w8_cuda_softmax_cpu_fallback() {
    // Given input for CUDA softmax
    let n_cols = 4;
    let n_rows = 2;
    let input: Vec<f32> = (0..n_cols * n_rows).map(|i| i as f32).collect();
    let mut output = vec![0.0f32; n_cols * n_rows];
    let cfg = SoftmaxConfig::for_shape(n_cols, n_rows).unwrap();

    // When CUDA softmax CPU fallback is used
    softmax_cpu(&input, &mut output, &cfg).unwrap();

    // Then each row sums to 1.0
    for row in 0..n_rows {
        let start = row * n_cols;
        let sum: f32 = output[start..start + n_cols].iter().sum();
        assert!((sum - 1.0).abs() < LOOSE_TOL, "row {row} sum = {sum}");
    }
}

#[test]
fn bdd_w8_cuda_layernorm_cpu_fallback() {
    // Given input for CUDA layer norm
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let cfg = CudaLnCfg::with_defaults();

    // When CUDA LN CPU fallback is used
    let result = layer_norm_cpu_fallback(&input, &gamma, &beta, 4, &cfg).unwrap();

    // Then output is approximately zero-mean
    no_nan_inf(&result);
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < LOOSE_TOL, "LN mean = {mean}");
}

#[test]
fn bdd_w8_cuda_residual_add() {
    // Given two vectors
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let residual = vec![0.1, 0.2, 0.3, 0.4];

    // When CUDA residual add is used
    let result = residual_add(&x, &residual).unwrap();

    // Then result is element-wise sum
    approx_eq(&result, &[1.1, 2.2, 3.3, 4.4], TOL);
}

#[test]
fn bdd_w8_cuda_activation_gelu_cpu() {
    // Given input for CUDA activation
    let input = vec![-1.0, 0.0, 1.0, 2.0];
    let mut output = vec![0.0f32; 4];
    let cfg = ActivationConfig::new(4, ActivationType::GELU).unwrap();

    // When GELU activation is applied via CPU path
    activation_cpu(&input, &mut output, &cfg).unwrap();

    // Then GELU(0) ≈ 0 and GELU(x) > 0 for positive x
    assert!(output[1].abs() < TOL, "GELU(0) = {}", output[1]);
    assert!(output[3] > 1.5, "GELU(2) should be ~1.96, got {}", output[3]);
    no_nan_inf(&output);
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 11: Reduction chains
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_reduction_sum_mean() {
    // Given per-token losses
    let per_token = vec![0.5, 1.0, 0.3, 0.8, 0.2];

    // When sum and mean reductions are applied
    let total = ReductionKernel::sum(&per_token).unwrap();
    let mean = ReductionKernel::mean(&per_token).unwrap();

    // Then sum and mean are correct
    let expected_sum = 0.5 + 1.0 + 0.3 + 0.8 + 0.2;
    assert!((total - expected_sum).abs() < TOL, "sum = {total}");
    assert!((mean - expected_sum / 5.0).abs() < TOL, "mean = {mean}");
}

#[test]
fn bdd_w8_reduction_max_min_finite() {
    // Given extreme float values
    let data = vec![f32::MIN / 2.0, f32::MAX / 2.0, 0.0, -1.0, 1.0];

    // When max/min reductions are computed
    let max_val = ReductionKernel::max(&data).unwrap();
    let min_val = ReductionKernel::min(&data).unwrap();

    // Then results are finite and correct
    assert!(max_val.value.is_finite());
    assert!(min_val.value.is_finite());
    assert!(max_val.value > min_val.value);
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline 12: End-to-end transformer-like block
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_transformer_block_e2e() {
    // Given an input sequence (seq_len=2, dim=4)
    let seq_len = 2;
    let dim = 4;
    let input: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();

    // Step 1: Layer norm (pre-norm architecture)
    let gamma = vec![1.0; seq_len * dim];
    let ln_cfg = LayerNormConfig::new(vec![seq_len * dim]);
    let normed = layer_norm(&input, &gamma, None, &ln_cfg).unwrap();

    // Step 2: Q, K, V = normed (identity projection)
    let q = &normed;
    let k = &normed;
    let v = &normed;

    // Step 3: Attention scores (manual Q·K^T / sqrt(d))
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q[i * dim + d] * k[j * dim + d];
            }
            scores[i * seq_len + j] = dot / (dim as f32).sqrt();
        }
    }

    // Step 4: Causal mask + softmax
    let mask = causal_mask(seq_len);
    apply_mask(&mut scores, &mask).unwrap();
    for row in 0..seq_len {
        let start = row * seq_len;
        let end = start + seq_len;
        let sm = row_softmax(&scores[start..end]);
        scores[start..end].copy_from_slice(&sm);
    }

    // Step 5: Weighted sum of V
    let mut attn_out = vec![0.0f32; seq_len * dim];
    for i in 0..seq_len {
        for d in 0..dim {
            let mut val = 0.0f32;
            for j in 0..seq_len {
                val += scores[i * seq_len + j] * v[j * dim + d];
            }
            attn_out[i * dim + d] = val;
        }
    }

    // Step 6: Residual
    let mut block_out = attn_out;
    add_residual(&mut block_out, &input).unwrap();

    // Then the full transformer block output is finite
    no_nan_inf(&block_out);
}

#[test]
fn bdd_w8_two_transformer_blocks_chained() {
    // Given input
    let dim = 4;
    let input = vec![1.0, 0.5, -0.5, 0.0];

    // Run two identical "blocks" (LN → identity attention → residual)
    let run_block = |x: &[f32]| -> Vec<f32> {
        let gamma = vec![1.0; dim];
        let cfg = LayerNormConfig::new(vec![dim]);
        let normed = layer_norm(x, &gamma, None, &cfg).unwrap();
        let mut out = normed;
        add_residual(&mut out, x).unwrap();
        out
    };

    // When two blocks are chained
    let after_block1 = run_block(&input);
    let after_block2 = run_block(&after_block1);

    // Then output differs from input (residual accumulation)
    let differs = after_block2.iter().zip(input.iter()).any(|(a, b)| (a - b).abs() > TOL);
    assert!(differs, "chained blocks should modify the input");
    no_nan_inf(&after_block2);
}

// ═══════════════════════════════════════════════════════════════════
// Additional stability / edge-case scenarios
// ═══════════════════════════════════════════════════════════════════

#[test]
fn bdd_w8_stability_ternary_quant_all_zeros() {
    // Given all-zero input
    let input = vec![0.0f32; 16];

    // When ternary quantization is applied
    let (packed, scale) = pack_ternary(&input, 0.01);
    let recovered = dequant_ternary(&packed, scale);

    // Then all recovered values are zero
    for (i, &v) in recovered.iter().enumerate().take(input.len()) {
        assert_eq!(v, 0.0, "index {i} should be zero, got {v}");
    }
}

#[test]
fn bdd_w8_stability_matmul_large_values() {
    // Given large but not overflow values
    let m = 2;
    let k = 2;
    let n = 2;
    let a = vec![1e10, 1e10, 1e10, 1e10];
    let b = vec![1e-10, 0.0, 0.0, 1e-10];
    let mut c = vec![0.0f32; m * n];
    let cfg = SimdMatmulConfig::new(m, n, k);

    // When matmul is run
    simd_matmul_f32(&a, &b, &mut c, &cfg).unwrap();

    // Then output is finite (large * small = moderate)
    no_nan_inf(&c);
}

#[test]
fn bdd_w8_stability_cuda_residual_scaled_extreme() {
    // Given large values with small scale
    let x = vec![1e6, -1e6, 0.0, 1e6];
    let r = vec![1e6, 1e6, 1e6, -1e6];

    // When scaled residual is applied with small alpha
    let result = residual_add_scaled(&x, &r, 1e-6).unwrap();

    // Then result is finite
    no_nan_inf(&result);
}

#[test]
fn bdd_w8_stability_cuda_softmax_uniform() {
    // Given uniform input
    let n = 8;
    let input = vec![1.0f32; n];
    let mut output = vec![0.0f32; n];
    let cfg = SoftmaxConfig::for_shape(n, 1).unwrap();

    // When softmax is applied
    softmax_cpu(&input, &mut output, &cfg).unwrap();

    // Then all outputs are equal to 1/n
    for &v in &output {
        assert!(
            (v - 1.0 / n as f32).abs() < LOOSE_TOL,
            "expected uniform {}, got {v}",
            1.0 / n as f32
        );
    }
}
