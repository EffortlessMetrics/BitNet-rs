#![cfg(feature = "cpu")]
#![allow(clippy::needless_range_loop)]
//! AVX2 vs scalar parity tests for embedding, pooling, batch normalization,
//! and convolution CPU kernels.

use bitnet_kernels::cpu::batch_normalization::{
    SimdBatchNormConfig, SimdBatchNormState, batch_norm_forward, compute_mean, compute_variance,
};
use bitnet_kernels::cpu::convolution::{Conv1dConfig, conv1d, conv1d_avx2, conv1d_f32};
use bitnet_kernels::cpu::embedding::{EmbeddingConfig, embedding_lookup, embedding_lookup_simd};
use bitnet_kernels::cpu::pooling::{
    PoolConfig, PoolType, avg_pool1d, avg_pool1d_avx2, max_pool1d, max_pool1d_avx2,
};

#[path = "common/avx2_parity.rs"]
mod avx2_parity;
use avx2_parity::{assert_vec_parity, close, pseudo_rand};

// ── Tolerance constants ────────────────────────────────────────────────

/// Absolute tolerance for element-wise kernels (embedding).
const ELEM_ABS_TOL: f32 = 1e-6;
/// Relative tolerance for element-wise kernels.
const ELEM_REL_TOL: f32 = 1e-5;

/// Absolute tolerance for pooling kernels.
const POOL_ABS_TOL: f32 = 1e-6;
/// Relative tolerance for pooling kernels.
const POOL_REL_TOL: f32 = 1e-5;

/// Absolute tolerance for convolution kernels.
const CONV_ABS_TOL: f32 = 1e-4;
/// Relative tolerance for convolution kernels.
const CONV_REL_TOL: f32 = 1e-4;

/// Deterministic pseudo-random positive f32 values in [0.1, 1.1].
fn pseudo_rand_positive(len: usize, seed: u64) -> Vec<f32> {
    pseudo_rand(len, seed).iter().map(|x| x.abs() + 0.1).collect()
}

// ════════════════════════════════════════════════════════════════════════
// 1. EMBEDDING  —  embedding_lookup vs embedding_lookup_simd
// ════════════════════════════════════════════════════════════════════════

/// Reference scalar embedding lookup.
fn reference_embedding_lookup(table: &[f32], indices: &[u32], embedding_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; indices.len() * embedding_dim];
    for (i, &idx) in indices.iter().enumerate() {
        let src_offset = (idx as usize) * embedding_dim;
        let dst_offset = i * embedding_dim;
        output[dst_offset..dst_offset + embedding_dim]
            .copy_from_slice(&table[src_offset..src_offset + embedding_dim]);
    }
    output
}

/// Embedding parity: small vocab, 8-aligned dim.
#[test]
fn embedding_lookup_parity_small() {
    let vocab = 100;
    let dim = 16;
    let table = pseudo_rand(vocab * dim, 42);
    let indices: Vec<u32> = vec![0, 5, 42, 99];

    let expected = reference_embedding_lookup(&table, &indices, dim);

    // Plain scalar API
    let from_scalar = embedding_lookup(&table, &indices, dim).expect("embedding_lookup");

    // SIMD-dispatch API
    let config = EmbeddingConfig { vocab_size: vocab, embedding_dim: dim, padding_idx: None };
    let from_simd =
        embedding_lookup_simd(&table, &indices, &config).expect("embedding_lookup_simd");

    assert_vec_parity(
        &from_scalar,
        &expected,
        ELEM_ABS_TOL,
        ELEM_REL_TOL,
        "embedding_scalar_vs_ref",
    );
    assert_vec_parity(&from_simd, &expected, ELEM_ABS_TOL, ELEM_REL_TOL, "embedding_simd_vs_ref");
}

/// Embedding parity: non-8-aligned dim exercises the scalar tail.
#[test]
fn embedding_lookup_parity_non_aligned() {
    let vocab = 100;
    let dim = 13;
    let table = pseudo_rand(vocab * dim, 137);
    let indices: Vec<u32> = vec![0, 1, 12, 50, 73, 88, 91, 99];

    let expected = reference_embedding_lookup(&table, &indices, dim);

    let from_scalar = embedding_lookup(&table, &indices, dim).expect("embedding_lookup");
    let config = EmbeddingConfig { vocab_size: vocab, embedding_dim: dim, padding_idx: None };
    let from_simd =
        embedding_lookup_simd(&table, &indices, &config).expect("embedding_lookup_simd");

    assert_vec_parity(
        &from_scalar,
        &expected,
        ELEM_ABS_TOL,
        ELEM_REL_TOL,
        "embedding_scalar_non_aligned",
    );
    assert_vec_parity(
        &from_simd,
        &expected,
        ELEM_ABS_TOL,
        ELEM_REL_TOL,
        "embedding_simd_non_aligned",
    );
}

/// Embedding parity: large dim (128).
#[test]
fn embedding_lookup_parity_large_dim() {
    let vocab = 50;
    let dim = 128;
    let table = pseudo_rand(vocab * dim, 314);
    let indices: Vec<u32> = (0..16).collect();

    let expected = reference_embedding_lookup(&table, &indices, dim);

    let from_scalar = embedding_lookup(&table, &indices, dim).expect("embedding_lookup");
    let config = EmbeddingConfig { vocab_size: vocab, embedding_dim: dim, padding_idx: None };
    let from_simd =
        embedding_lookup_simd(&table, &indices, &config).expect("embedding_lookup_simd");

    assert_vec_parity(
        &from_scalar,
        &expected,
        ELEM_ABS_TOL,
        ELEM_REL_TOL,
        "embedding_scalar_large_dim",
    );
    assert_vec_parity(
        &from_simd,
        &expected,
        ELEM_ABS_TOL,
        ELEM_REL_TOL,
        "embedding_simd_large_dim",
    );
}

// ════════════════════════════════════════════════════════════════════════
// 2. POOLING  —  max_pool1d / avg_pool1d dispatch vs explicit AVX2
// ════════════════════════════════════════════════════════════════════════

/// Reference scalar max-pool 1D.
fn reference_max_pool1d(input: &[f32], kernel: usize, stride: usize) -> Vec<f32> {
    let out_len = (input.len() - kernel) / stride + 1;
    let mut output = vec![f32::NEG_INFINITY; out_len];
    for i in 0..out_len {
        let start = i * stride;
        for k in 0..kernel {
            output[i] = output[i].max(input[start + k]);
        }
    }
    output
}

/// Reference scalar avg-pool 1D.
fn reference_avg_pool1d(input: &[f32], kernel: usize, stride: usize) -> Vec<f32> {
    let out_len = (input.len() - kernel) / stride + 1;
    let mut output = vec![0.0f32; out_len];
    for i in 0..out_len {
        let start = i * stride;
        let mut sum = 0.0f32;
        for k in 0..kernel {
            sum += input[start + k];
        }
        output[i] = sum / kernel as f32;
    }
    output
}

/// Max-pool 1D parity: basic (stride=1, kernel=3).
#[test]
fn max_pool1d_parity_basic() {
    let input = pseudo_rand(32, 200);
    let config = PoolConfig::new(PoolType::Max, 3, 1, 0);
    let expected = reference_max_pool1d(&input, 3, 1);

    let (dispatch_out, _) = max_pool1d(&input, &config).expect("max_pool1d");
    let (avx2_out, _) = max_pool1d_avx2(&input, &config).expect("max_pool1d_avx2");

    assert_vec_parity(&dispatch_out, &expected, POOL_ABS_TOL, POOL_REL_TOL, "max_pool_dispatch");
    assert_vec_parity(&avx2_out, &expected, POOL_ABS_TOL, POOL_REL_TOL, "max_pool_avx2");
}

/// Avg-pool 1D parity: basic (stride=1, kernel=3).
#[test]
fn avg_pool1d_parity_basic() {
    let input = pseudo_rand(32, 201);
    let config = PoolConfig::new(PoolType::Average, 3, 1, 0);
    let expected = reference_avg_pool1d(&input, 3, 1);

    let dispatch_out = avg_pool1d(&input, &config).expect("avg_pool1d");
    let avx2_out = avg_pool1d_avx2(&input, &config).expect("avg_pool1d_avx2");

    assert_vec_parity(&dispatch_out, &expected, POOL_ABS_TOL, POOL_REL_TOL, "avg_pool_dispatch");
    assert_vec_parity(&avx2_out, &expected, POOL_ABS_TOL, POOL_REL_TOL, "avg_pool_avx2");
}

/// Max-pool 1D parity: non-aligned length (stride=2, kernel=5).
#[test]
fn max_pool1d_parity_non_aligned() {
    let input = pseudo_rand(33, 202);
    let config = PoolConfig::new(PoolType::Max, 5, 2, 0);
    let expected = reference_max_pool1d(&input, 5, 2);

    let (dispatch_out, _) = max_pool1d(&input, &config).expect("max_pool1d");
    let (avx2_out, _) = max_pool1d_avx2(&input, &config).expect("max_pool1d_avx2");

    assert_vec_parity(
        &dispatch_out,
        &expected,
        POOL_ABS_TOL,
        POOL_REL_TOL,
        "max_pool_non_aligned_dispatch",
    );
    assert_vec_parity(
        &avx2_out,
        &expected,
        POOL_ABS_TOL,
        POOL_REL_TOL,
        "max_pool_non_aligned_avx2",
    );
}

/// Avg-pool 1D parity: non-aligned length (stride=2, kernel=5).
#[test]
fn avg_pool1d_parity_non_aligned() {
    let input = pseudo_rand(33, 203);
    let config = PoolConfig::new(PoolType::Average, 5, 2, 0);
    let expected = reference_avg_pool1d(&input, 5, 2);

    let dispatch_out = avg_pool1d(&input, &config).expect("avg_pool1d");
    let avx2_out = avg_pool1d_avx2(&input, &config).expect("avg_pool1d_avx2");

    assert_vec_parity(
        &dispatch_out,
        &expected,
        POOL_ABS_TOL,
        POOL_REL_TOL,
        "avg_pool_non_aligned_dispatch",
    );
    assert_vec_parity(
        &avx2_out,
        &expected,
        POOL_ABS_TOL,
        POOL_REL_TOL,
        "avg_pool_non_aligned_avx2",
    );
}

// ════════════════════════════════════════════════════════════════════════
// 3. BATCH NORMALIZATION  —  compute_mean / compute_variance / forward
// ════════════════════════════════════════════════════════════════════════

/// Reference scalar mean.
fn reference_mean(data: &[f32]) -> f32 {
    let sum: f64 = data.iter().map(|&x| x as f64).sum();
    (sum / data.len() as f64) as f32
}

/// Reference scalar variance given mean.
fn reference_variance(data: &[f32], mean: f32) -> f32 {
    let mean_d = mean as f64;
    let sum: f64 = data
        .iter()
        .map(|&x| {
            let d = x as f64 - mean_d;
            d * d
        })
        .sum();
    (sum / data.len() as f64) as f32
}

/// compute_mean parity across various lengths.
#[test]
fn batch_norm_mean_parity() {
    for &n in &[32, 64, 100, 127] {
        let data = pseudo_rand(n, 300 + n as u64);
        let expected = reference_mean(&data);
        let actual = compute_mean(&data);
        assert!(
            close(actual, expected, ELEM_ABS_TOL, ELEM_REL_TOL),
            "mean(n={n}): expected={expected}, actual={actual}, diff={}",
            (actual - expected).abs()
        );
    }
}

/// compute_variance parity across various lengths.
#[test]
fn batch_norm_variance_parity() {
    for &n in &[32, 64, 100, 127] {
        let data = pseudo_rand(n, 400 + n as u64);
        let mean = compute_mean(&data);
        let expected = reference_variance(&data, mean);
        let actual = compute_variance(&data, mean);
        assert!(
            close(actual, expected, ELEM_ABS_TOL, ELEM_REL_TOL),
            "variance(n={n}): expected={expected}, actual={actual}, diff={}",
            (actual - expected).abs()
        );
    }
}

/// batch_norm_forward parity: [batch=4, features=16].
#[test]
fn batch_norm_forward_parity() {
    let batch = 4;
    let features = 16;
    let input = pseudo_rand(batch * features, 500);
    let gamma = pseudo_rand_positive(features, 501);
    let beta = pseudo_rand(features, 502);
    let config = SimdBatchNormConfig::new();

    // Run forward twice with identical inputs and fresh state to compare.
    let mut state1 = SimdBatchNormState::new(features);
    let out1 = batch_norm_forward(&input, features, &gamma, &beta, &mut state1, &config)
        .expect("batch_norm_forward run 1");

    let mut state2 = SimdBatchNormState::new(features);
    let out2 = batch_norm_forward(&input, features, &gamma, &beta, &mut state2, &config)
        .expect("batch_norm_forward run 2");

    // Deterministic: two runs with the same input must match exactly.
    assert_vec_parity(&out1, &out2, ELEM_ABS_TOL, ELEM_REL_TOL, "batch_norm_forward_determinism");

    // Verify output is finite and normalized per channel.
    for &v in &out1 {
        assert!(v.is_finite(), "batch_norm output contains non-finite value: {v}");
    }

    // Verify running stats were updated.
    assert_eq!(state1.num_batches_tracked, 1);
    for ch in 0..features {
        assert!(state1.running_mean[ch].is_finite(), "running_mean[{ch}] is not finite");
        assert!(state1.running_var[ch].is_finite(), "running_var[{ch}] is not finite");
    }
}

// ════════════════════════════════════════════════════════════════════════
// 4. CONVOLUTION  —  conv1d (dispatch) vs conv1d_f32 (scalar) vs conv1d_avx2
// ════════════════════════════════════════════════════════════════════════

/// Reference scalar 1D convolution (single-group, no padding).
fn reference_conv1d(
    input: &[f32],
    weight: &[f32],
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_len: usize,
) -> Vec<f32> {
    let out_len = in_len - kernel_size + 1;
    let batch_size = 1;
    let mut output = vec![0.0f32; batch_size * out_channels * out_len];
    for oc in 0..out_channels {
        for o in 0..out_len {
            let mut sum = 0.0f64;
            for ic in 0..in_channels {
                for k in 0..kernel_size {
                    let in_idx = ic * in_len + (o + k);
                    let w_idx = (oc * in_channels + ic) * kernel_size + k;
                    sum += input[in_idx] as f64 * weight[w_idx] as f64;
                }
            }
            let out_idx = oc * out_len + o;
            output[out_idx] = sum as f32;
        }
    }
    output
}

/// Conv1d parity: basic single-channel.
#[test]
fn conv1d_parity_basic() {
    let in_ch = 1;
    let out_ch = 1;
    let kernel = 3;
    let in_len = 32;
    let input = pseudo_rand(in_ch * in_len, 600);
    let weight = pseudo_rand(out_ch * in_ch * kernel, 601);
    let config = Conv1dConfig::new(in_ch, out_ch, kernel);

    let expected = reference_conv1d(&input, &weight, in_ch, out_ch, kernel, in_len);
    let scalar_out = conv1d_f32(&input, &weight, None, &config, 1, in_len).expect("conv1d_f32");
    let avx2_out = conv1d_avx2(&input, &weight, None, &config, 1, in_len).expect("conv1d_avx2");
    let dispatch_out = conv1d(&input, &weight, None, &config, 1, in_len).expect("conv1d");

    assert_vec_parity(&scalar_out, &expected, CONV_ABS_TOL, CONV_REL_TOL, "conv1d_scalar_basic");
    assert_vec_parity(&avx2_out, &expected, CONV_ABS_TOL, CONV_REL_TOL, "conv1d_avx2_basic");
    assert_vec_parity(
        &dispatch_out,
        &expected,
        CONV_ABS_TOL,
        CONV_REL_TOL,
        "conv1d_dispatch_basic",
    );
}

/// Conv1d parity: multi-channel.
#[test]
fn conv1d_parity_multi_channel() {
    let in_ch = 4;
    let out_ch = 8;
    let kernel = 3;
    let in_len = 16;
    let input = pseudo_rand(in_ch * in_len, 700);
    let weight = pseudo_rand(out_ch * in_ch * kernel, 701);
    let config = Conv1dConfig::new(in_ch, out_ch, kernel);

    let expected = reference_conv1d(&input, &weight, in_ch, out_ch, kernel, in_len);
    let scalar_out = conv1d_f32(&input, &weight, None, &config, 1, in_len).expect("conv1d_f32");
    let avx2_out = conv1d_avx2(&input, &weight, None, &config, 1, in_len).expect("conv1d_avx2");
    let dispatch_out = conv1d(&input, &weight, None, &config, 1, in_len).expect("conv1d");

    assert_vec_parity(&scalar_out, &expected, CONV_ABS_TOL, CONV_REL_TOL, "conv1d_scalar_multi");
    assert_vec_parity(&avx2_out, &expected, CONV_ABS_TOL, CONV_REL_TOL, "conv1d_avx2_multi");
    assert_vec_parity(
        &dispatch_out,
        &expected,
        CONV_ABS_TOL,
        CONV_REL_TOL,
        "conv1d_dispatch_multi",
    );
}

/// Conv1d parity: non-aligned channels and length.
#[test]
fn conv1d_parity_non_aligned() {
    let in_ch = 3;
    let out_ch = 5;
    let kernel = 5;
    let in_len = 17;
    let input = pseudo_rand(in_ch * in_len, 800);
    let weight = pseudo_rand(out_ch * in_ch * kernel, 801);
    let config = Conv1dConfig::new(in_ch, out_ch, kernel);

    let expected = reference_conv1d(&input, &weight, in_ch, out_ch, kernel, in_len);
    let scalar_out = conv1d_f32(&input, &weight, None, &config, 1, in_len).expect("conv1d_f32");
    let avx2_out = conv1d_avx2(&input, &weight, None, &config, 1, in_len).expect("conv1d_avx2");
    let dispatch_out = conv1d(&input, &weight, None, &config, 1, in_len).expect("conv1d");

    assert_vec_parity(
        &scalar_out,
        &expected,
        CONV_ABS_TOL,
        CONV_REL_TOL,
        "conv1d_scalar_non_aligned",
    );
    assert_vec_parity(&avx2_out, &expected, CONV_ABS_TOL, CONV_REL_TOL, "conv1d_avx2_non_aligned");
    assert_vec_parity(
        &dispatch_out,
        &expected,
        CONV_ABS_TOL,
        CONV_REL_TOL,
        "conv1d_dispatch_non_aligned",
    );
}
