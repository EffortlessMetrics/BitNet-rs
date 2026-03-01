//! Tests for OpenCL RMSNorm and LayerNorm normalization kernels.
//!
//! Uses CPU reference implementations to verify kernel source correctness,
//! numerical stability, edge cases, and statistical properties.

use bitnet_kernels::kernels;

// === CPU Reference Implementations ===

/// CPU reference RMSNorm: output[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]
fn cpu_rmsnorm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    assert_eq!(n, weight.len());
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    input.iter().zip(weight.iter()).map(|(x, w)| x * rms * w).collect()
}

/// CPU reference RMSNorm for batched input: [batch, hidden_dim]
fn cpu_rmsnorm_batch(input: &[f32], weight: &[f32], hidden_dim: usize, eps: f32) -> Vec<f32> {
    let batch_size = input.len() / hidden_dim;
    let mut output = vec![0.0f32; input.len()];
    for b in 0..batch_size {
        let row = &input[b * hidden_dim..(b + 1) * hidden_dim];
        let result = cpu_rmsnorm(row, weight, eps);
        output[b * hidden_dim..(b + 1) * hidden_dim].copy_from_slice(&result);
    }
    output
}

/// CPU reference LayerNorm: output[i] = (input[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
fn cpu_layernorm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    assert_eq!(n, gamma.len());
    assert_eq!(n, beta.len());
    let mean: f32 = input.iter().sum::<f32>() / n as f32;
    let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    input
        .iter()
        .zip(gamma.iter())
        .zip(beta.iter())
        .map(|((x, g), b)| (x - mean) * inv_std * g + b)
        .collect()
}

/// CPU reference LayerNorm for batched input: [batch, hidden_dim]
fn cpu_layernorm_batch(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    hidden_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let batch_size = input.len() / hidden_dim;
    let mut output = vec![0.0f32; input.len()];
    for b in 0..batch_size {
        let row = &input[b * hidden_dim..(b + 1) * hidden_dim];
        let result = cpu_layernorm(row, gamma, beta, eps);
        output[b * hidden_dim..(b + 1) * hidden_dim].copy_from_slice(&result);
    }
    output
}

// === Kernel Source Validation ===

#[test]
fn normalization_source_contains_rmsnorm_kernel() {
    let src = kernels::NORMALIZATION_SRC;
    assert!(src.contains("__kernel void rmsnorm"), "missing rmsnorm kernel definition");
}

#[test]
fn normalization_source_contains_layernorm_kernel() {
    let src = kernels::NORMALIZATION_SRC;
    assert!(src.contains("__kernel void layernorm"), "missing layernorm kernel definition");
}

#[test]
fn rmsnorm_kernel_uses_local_memory_reduction() {
    let src = kernels::NORMALIZATION_SRC;
    assert!(src.contains("__local float* local_buf"), "rmsnorm should use local memory");
    assert!(src.contains("barrier(CLK_LOCAL_MEM_FENCE)"), "rmsnorm should use barriers");
}

#[test]
fn layernorm_kernel_uses_local_memory_reduction() {
    let src = kernels::NORMALIZATION_SRC;
    // LayerNorm also uses local memory and barriers
    let layernorm_section =
        &src[src.find("__kernel void layernorm").expect("layernorm kernel not found")..];
    assert!(
        layernorm_section.contains("__local float* local_buf"),
        "layernorm should use local memory"
    );
    assert!(
        layernorm_section.contains("barrier(CLK_LOCAL_MEM_FENCE)"),
        "layernorm should use barriers"
    );
}

#[test]
fn rmsnorm_kernel_uses_rsqrt() {
    let src = kernels::NORMALIZATION_SRC;
    let rmsnorm_section = &src[src.find("__kernel void rmsnorm").expect("rmsnorm not found")..];
    let end = rmsnorm_section.find("__kernel void layernorm").unwrap_or(rmsnorm_section.len());
    let rmsnorm_only = &rmsnorm_section[..end];
    assert!(rmsnorm_only.contains("rsqrt("), "rmsnorm should use rsqrt for efficiency");
}

#[test]
fn layernorm_kernel_uses_rsqrt() {
    let src = kernels::NORMALIZATION_SRC;
    let layernorm_section =
        &src[src.find("__kernel void layernorm").expect("layernorm not found")..];
    assert!(layernorm_section.contains("rsqrt("), "layernorm should use rsqrt for inv_std");
}

#[test]
fn rmsnorm_kernel_has_eps_parameter() {
    let src = kernels::NORMALIZATION_SRC;
    assert!(src.contains("const float eps"), "rmsnorm should accept eps parameter");
}

#[test]
fn layernorm_kernel_has_gamma_beta_parameters() {
    let src = kernels::NORMALIZATION_SRC;
    let layernorm_section =
        &src[src.find("__kernel void layernorm").expect("layernorm not found")..];
    assert!(layernorm_section.contains("gamma"), "layernorm should accept gamma parameter");
    assert!(layernorm_section.contains("beta"), "layernorm should accept beta parameter");
}

#[test]
fn kernels_use_tree_reduction_pattern() {
    let src = kernels::NORMALIZATION_SRC;
    // Tree reduction pattern: for (s = local_size/2; s > 0; s >>= 1)
    assert!(
        src.contains("s >>= 1") || src.contains("s /= 2"),
        "should use tree reduction with halving stride"
    );
    assert!(
        src.contains("local_buf[lid] += local_buf[lid + s]"),
        "should accumulate in tree reduction"
    );
}

#[test]
fn no_normalization_kernel_uses_printf() {
    let src = kernels::NORMALIZATION_SRC;
    assert!(!src.contains("printf"), "normalization kernels should not use printf");
}

// === RMSNorm Correctness Tests ===

#[test]
fn rmsnorm_basic_correctness() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0; 4];
    let eps = 1e-5;

    let result = cpu_rmsnorm(&input, &weight, eps);

    // With unit weights, RMSNorm just normalizes by RMS
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / 4.0 + eps).sqrt();
    for (i, &val) in result.iter().enumerate() {
        let expected = input[i] / rms;
        assert!((val - expected).abs() < 1e-5, "rmsnorm[{i}]: got {val}, expected {expected}");
    }
}

#[test]
fn rmsnorm_known_values() {
    // All ones: mean(x^2) = 1.0, rms = rsqrt(1 + eps) ≈ 1.0
    let input = vec![1.0; 4];
    let weight = vec![1.0; 4];
    let eps = 1e-5;

    let result = cpu_rmsnorm(&input, &weight, eps);
    for &val in &result {
        assert!(
            (val - 1.0).abs() < 1e-4,
            "rmsnorm of ones with unit weight should be ~1.0, got {val}"
        );
    }
}

#[test]
fn rmsnorm_batch_dimension() {
    let hidden_dim = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // batch=2
    let weight = vec![1.0; hidden_dim];
    let eps = 1e-5;

    let result = cpu_rmsnorm_batch(&input, &weight, hidden_dim, eps);

    // Each row should be independently normalized
    let row0 = cpu_rmsnorm(&input[0..4], &weight, eps);
    let row1 = cpu_rmsnorm(&input[4..8], &weight, eps);
    for i in 0..hidden_dim {
        assert!((result[i] - row0[i]).abs() < 1e-6, "batch row 0 mismatch at {i}");
        assert!(
            (result[hidden_dim + i] - row1[i]).abs() < 1e-6,
            "batch row 1 mismatch at {i}"
        );
    }
}

#[test]
fn rmsnorm_with_non_unit_weights() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![0.5, 1.0, 1.5, 2.0];
    let eps = 1e-5;

    let result = cpu_rmsnorm(&input, &weight, eps);

    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = 1.0 / (sum_sq / 4.0 + eps).sqrt();
    for (i, &val) in result.iter().enumerate() {
        let expected = input[i] * rms * weight[i];
        assert!((val - expected).abs() < 1e-5, "weighted rmsnorm[{i}] mismatch");
    }
}

// === LayerNorm Correctness Tests ===

#[test]
fn layernorm_basic_correctness() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let eps = 1e-5;

    let result = cpu_layernorm(&input, &gamma, &beta, eps);

    // With unit gamma and zero beta, output should have zero mean
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 1e-5, "layernorm output mean should be ~0, got {mean}");
}

#[test]
fn layernorm_known_values() {
    // Two values: [-1, 1] → mean=0, var=1, so output should be [-1, 1] * gamma + beta
    let input = vec![-1.0, 1.0];
    let gamma = vec![1.0; 2];
    let beta = vec![0.0; 2];
    let eps = 1e-5;

    let result = cpu_layernorm(&input, &gamma, &beta, eps);
    assert!(
        (result[0] - (-1.0)).abs() < 1e-3,
        "layernorm[-1,1][0] should be ~-1.0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 1.0).abs() < 1e-3,
        "layernorm[-1,1][1] should be ~1.0, got {}",
        result[1]
    );
}

#[test]
fn layernorm_batch_dimension() {
    let hidden_dim = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]; // batch=2
    let gamma = vec![1.0; hidden_dim];
    let beta = vec![0.0; hidden_dim];
    let eps = 1e-5;

    let result = cpu_layernorm_batch(&input, &gamma, &beta, hidden_dim, eps);

    let row0 = cpu_layernorm(&input[0..4], &gamma, &beta, eps);
    let row1 = cpu_layernorm(&input[4..8], &gamma, &beta, eps);
    for i in 0..hidden_dim {
        assert!((result[i] - row0[i]).abs() < 1e-6, "batch row 0 mismatch at {i}");
        assert!(
            (result[hidden_dim + i] - row1[i]).abs() < 1e-6,
            "batch row 1 mismatch at {i}"
        );
    }
}

#[test]
fn layernorm_with_gamma_beta() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![2.0; 4];
    let beta = vec![1.0; 4];
    let eps = 1e-5;

    let result = cpu_layernorm(&input, &gamma, &beta, eps);

    // With gamma=2, beta=1: output = 2*(normalized) + 1
    // Mean of result should be ~1.0 (mean of 2*0 + 1)
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(
        (mean - 1.0).abs() < 1e-4,
        "layernorm with beta=1 should have mean ~1.0, got {mean}"
    );
}

// === Numerical Stability Tests ===

#[test]
fn rmsnorm_very_small_inputs() {
    let input = vec![1e-20, 2e-20, 3e-20, 4e-20];
    let weight = vec![1.0; 4];
    let eps = 1e-5;

    let result = cpu_rmsnorm(&input, &weight, eps);
    for &val in &result {
        assert!(val.is_finite(), "rmsnorm should produce finite output for tiny inputs");
    }
}

#[test]
fn rmsnorm_large_inputs() {
    let input = vec![1e10, 2e10, 3e10, 4e10];
    let weight = vec![1.0; 4];
    let eps = 1e-5;

    let result = cpu_rmsnorm(&input, &weight, eps);
    for &val in &result {
        assert!(val.is_finite(), "rmsnorm should produce finite output for large inputs");
    }
}

#[test]
fn layernorm_very_small_inputs() {
    let input = vec![1e-20, 2e-20, 3e-20, 4e-20];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let eps = 1e-5;

    let result = cpu_layernorm(&input, &gamma, &beta, eps);
    for &val in &result {
        assert!(val.is_finite(), "layernorm should produce finite output for tiny inputs");
    }
}

#[test]
fn layernorm_large_inputs() {
    let input = vec![1e10, 2e10, 3e10, 4e10];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let eps = 1e-5;

    let result = cpu_layernorm(&input, &gamma, &beta, eps);
    for &val in &result {
        assert!(val.is_finite(), "layernorm should produce finite output for large inputs");
    }
}

// === Eps Parameter Tests ===

#[test]
fn rmsnorm_eps_prevents_division_by_zero() {
    // All zeros → mean(x^2) = 0, needs eps to avoid div by zero
    let input = vec![0.0; 4];
    let weight = vec![1.0; 4];
    let eps = 1e-5;

    let result = cpu_rmsnorm(&input, &weight, eps);
    for &val in &result {
        assert!(val.is_finite(), "rmsnorm of zeros should be finite with eps");
        assert!(val.abs() < 1e-10, "rmsnorm of zeros should be ~0");
    }
}

#[test]
fn layernorm_eps_prevents_division_by_zero() {
    // Constant input → var = 0, needs eps
    let input = vec![5.0; 4];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let eps = 1e-5;

    let result = cpu_layernorm(&input, &gamma, &beta, eps);
    for &val in &result {
        assert!(val.is_finite(), "layernorm of constant input should be finite with eps");
    }
}

// === Edge Case Tests ===

#[test]
fn rmsnorm_hidden_dim_1() {
    let input = vec![3.0];
    let weight = vec![2.0];
    let eps = 1e-5;

    let result = cpu_rmsnorm(&input, &weight, eps);
    assert_eq!(result.len(), 1);
    // mean(x^2) = 9.0, rms = rsqrt(9 + eps) ≈ 1/3
    let expected = 3.0 * (1.0 / (9.0 + eps).sqrt()) * 2.0;
    assert!(
        (result[0] - expected).abs() < 1e-5,
        "rmsnorm dim=1: got {}, expected {expected}",
        result[0]
    );
}

#[test]
fn layernorm_hidden_dim_1() {
    let input = vec![3.0];
    let gamma = vec![1.0];
    let beta = vec![0.0];
    let eps = 1e-5;

    let result = cpu_layernorm(&input, &gamma, &beta, eps);
    assert_eq!(result.len(), 1);
    // mean=3, var=0, (3-3)/sqrt(eps) = 0
    assert!(result[0].abs() < 1e-2, "layernorm dim=1 should be ~0, got {}", result[0]);
}

#[test]
fn rmsnorm_large_hidden_dim() {
    let hidden_dim = 4096;
    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.001).collect();
    let weight = vec![1.0; hidden_dim];
    let eps = 1e-5;

    let result = cpu_rmsnorm(&input, &weight, eps);
    assert_eq!(result.len(), hidden_dim);
    for &val in &result {
        assert!(val.is_finite(), "rmsnorm dim=4096 should produce finite values");
    }
}

#[test]
fn layernorm_large_hidden_dim() {
    let hidden_dim = 4096;
    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.001).collect();
    let gamma = vec![1.0; hidden_dim];
    let beta = vec![0.0; hidden_dim];
    let eps = 1e-5;

    let result = cpu_layernorm(&input, &gamma, &beta, eps);
    assert_eq!(result.len(), hidden_dim);
    let mean: f32 = result.iter().sum::<f32>() / hidden_dim as f32;
    assert!(
        mean.abs() < 1e-4,
        "layernorm dim=4096 output mean should be ~0, got {mean}"
    );
}

// === Property Tests ===

#[test]
fn property_rmsnorm_output_scale_similar_to_input() {
    // RMSNorm preserves relative scale (output RMS ≈ 1 with unit weights)
    let input = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
    let weight = vec![1.0; 8];
    let eps = 1e-5;

    let result = cpu_rmsnorm(&input, &weight, eps);
    let output_rms: f32 =
        (result.iter().map(|x| x * x).sum::<f32>() / result.len() as f32).sqrt();
    assert!(
        (output_rms - 1.0).abs() < 0.1,
        "rmsnorm with unit weights should have output RMS ~1.0, got {output_rms}"
    );
}

#[test]
fn property_layernorm_output_zero_mean_unit_variance() {
    let input = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
    let gamma = vec![1.0; 8];
    let beta = vec![0.0; 8];
    let eps = 1e-5;

    let result = cpu_layernorm(&input, &gamma, &beta, eps);
    let n = result.len() as f32;

    let mean: f32 = result.iter().sum::<f32>() / n;
    assert!(mean.abs() < 1e-5, "layernorm output mean should be ~0, got {mean}");

    let var: f32 = result.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    assert!(
        (var - 1.0).abs() < 0.01,
        "layernorm output variance should be ~1.0, got {var}"
    );
}

#[test]
fn property_both_produce_finite_outputs() {
    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0; 8],
        vec![1.0; 8],
        vec![-1.0; 8],
        vec![1e-10, -1e-10, 1e-10, -1e-10, 1e-10, -1e-10, 1e-10, -1e-10],
        vec![1e5, -1e5, 1e5, -1e5, 1e5, -1e5, 1e5, -1e5],
        (0..8).map(|i| if i % 2 == 0 { 100.0 } else { -100.0 }).collect(),
    ];
    let weight = vec![1.0; 8];
    let gamma = vec![1.0; 8];
    let beta = vec![0.0; 8];
    let eps = 1e-5;

    for (idx, input) in inputs.iter().enumerate() {
        let rms_result = cpu_rmsnorm(input, &weight, eps);
        for &val in &rms_result {
            assert!(val.is_finite(), "rmsnorm case {idx} produced non-finite value");
        }

        let ln_result = cpu_layernorm(input, &gamma, &beta, eps);
        for &val in &ln_result {
            assert!(val.is_finite(), "layernorm case {idx} produced non-finite value");
        }
    }
}

// === Kernel source balanced braces ===

#[test]
fn normalization_kernel_source_has_balanced_braces() {
    let src = kernels::NORMALIZATION_SRC;
    let opens = src.chars().filter(|&c| c == '{').count();
    let closes = src.chars().filter(|&c| c == '}').count();
    assert_eq!(
        opens, closes,
        "normalization.cl has unbalanced braces: {opens} {{ vs {closes} }}"
    );
}

#[test]
fn normalization_kernel_source_has_balanced_parens() {
    let src = kernels::NORMALIZATION_SRC;
    let opens = src.chars().filter(|&c| c == '(').count();
    let closes = src.chars().filter(|&c| c == ')').count();
    assert_eq!(
        opens, closes,
        "normalization.cl has unbalanced parens: {opens} ( vs {closes} )"
    );
}
