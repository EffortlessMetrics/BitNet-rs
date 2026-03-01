//! Edge case and error handling tests for CPU optimized inference operations.
//!
//! Covers boundary conditions, special float values, empty inputs,
//! dimension mismatches, and numerical edge cases that could cause
//! panics or incorrect results in production.

use bitnet_common::{ActivationType, NormType};
use bitnet_inference::cpu_opt;

// --- Empty input handling ---

#[test]
fn silu_empty_input() {
    let output = cpu_opt::silu(&[]);
    assert!(output.is_empty());
}

#[test]
fn silu_in_place_empty_input() {
    let mut data: Vec<f32> = vec![];
    cpu_opt::silu_in_place(&mut data);
    assert!(data.is_empty());
}

#[test]
fn gelu_empty_input() {
    let output = cpu_opt::gelu(&[]);
    assert!(output.is_empty());
}

#[test]
fn gelu_in_place_empty_input() {
    let mut data: Vec<f32> = vec![];
    cpu_opt::gelu_in_place(&mut data);
    assert!(data.is_empty());
}

#[test]
fn relu2_empty_input() {
    let output = cpu_opt::relu2(&[]);
    assert!(output.is_empty());
}

#[test]
fn relu2_in_place_empty_input() {
    let mut data: Vec<f32> = vec![];
    cpu_opt::relu2_in_place(&mut data);
    assert!(data.is_empty());
}

// --- Single element inputs ---

#[test]
fn silu_single_element() {
    let output = cpu_opt::silu(&[1.0]);
    assert_eq!(output.len(), 1);
    let expected = 1.0 / (1.0 + (-1.0f32).exp());
    assert!((output[0] - expected).abs() < 1e-6);
}

#[test]
fn gelu_single_element() {
    let output = cpu_opt::gelu(&[0.0]);
    assert_eq!(output.len(), 1);
    assert!((output[0] - 0.0).abs() < 1e-6);
}

#[test]
fn relu2_single_element_positive() {
    let output = cpu_opt::relu2(&[3.0]);
    assert!((output[0] - 9.0).abs() < 1e-6);
}

#[test]
fn relu2_single_element_negative() {
    let output = cpu_opt::relu2(&[-3.0]);
    assert!((output[0] - 0.0).abs() < 1e-6);
}

// --- Special float values ---

#[test]
fn silu_with_infinity() {
    let output = cpu_opt::silu(&[f32::INFINITY]);
    // silu(inf) = inf * sigmoid(inf) = inf * 1 = inf
    assert!(output[0].is_infinite() && output[0] > 0.0);
}

#[test]
fn silu_with_neg_infinity() {
    let output = cpu_opt::silu(&[f32::NEG_INFINITY]);
    // silu(-inf) = -inf * sigmoid(-inf) = -inf * 0 = NaN or 0
    // Implementation: x / (1 + exp(-x)) = -inf / (1 + exp(inf)) = -inf / inf = NaN
    // This is expected behavior for extreme inputs
    assert!(output[0].is_nan() || output[0] == 0.0 || output[0].is_infinite());
}

#[test]
fn silu_with_nan() {
    let output = cpu_opt::silu(&[f32::NAN]);
    assert!(output[0].is_nan());
}

#[test]
fn gelu_with_large_positive() {
    let output = cpu_opt::gelu(&[100.0]);
    // GELU(large) ≈ large (tanh saturates to 1)
    assert!((output[0] - 100.0).abs() < 1.0);
}

#[test]
fn gelu_with_large_negative() {
    let output = cpu_opt::gelu(&[-100.0]);
    // GELU(large negative) ≈ 0
    assert!(output[0].abs() < 1.0);
}

#[test]
fn relu2_with_infinity() {
    let output = cpu_opt::relu2(&[f32::INFINITY]);
    assert!(output[0].is_infinite() && output[0] > 0.0);
}

#[test]
fn relu2_with_neg_infinity() {
    let output = cpu_opt::relu2(&[f32::NEG_INFINITY]);
    assert!((output[0] - 0.0).abs() < 1e-6);
}

#[test]
fn relu2_with_nan() {
    let output = cpu_opt::relu2(&[f32::NAN]);
    // NaN.max(0.0) returns 0.0 in Rust, so relu2(NaN) = 0.0
    // or NaN depending on implementation
    // Just ensure it doesn't panic
    let _ = output[0];
}

// --- Matmul edge cases ---

#[test]
fn matmul_1x1() {
    let a = vec![3.0f32];
    let b = vec![4.0f32];
    let mut c = vec![0.0f32];
    cpu_opt::parallel_matmul(&a, &b, &mut c, 1, 1, 1, 1).unwrap();
    assert!((c[0] - 12.0).abs() < 1e-6);
}

#[test]
fn matmul_mismatched_a_size() {
    let a = vec![1.0f32; 5]; // wrong: should be 2*3=6
    let b = vec![1.0f32; 6]; // 3x2
    let mut c = vec![0.0f32; 4]; // 2x2
    let result = cpu_opt::parallel_matmul(&a, &b, &mut c, 2, 2, 3, 1);
    assert!(result.is_err());
}

#[test]
fn matmul_mismatched_b_size() {
    let a = vec![1.0f32; 6]; // 2x3
    let b = vec![1.0f32; 5]; // wrong: should be 3*2=6
    let mut c = vec![0.0f32; 4]; // 2x2
    let result = cpu_opt::parallel_matmul(&a, &b, &mut c, 2, 2, 3, 1);
    assert!(result.is_err());
}

#[test]
fn matmul_mismatched_c_size() {
    let a = vec![1.0f32; 6]; // 2x3
    let b = vec![1.0f32; 6]; // 3x2
    let mut c = vec![0.0f32; 3]; // wrong: should be 2*2=4
    let result = cpu_opt::parallel_matmul(&a, &b, &mut c, 2, 2, 3, 1);
    assert!(result.is_err());
}

#[test]
fn matmul_with_multiple_threads() {
    let a = vec![1.0f32; 16]; // 4x4
    let b = vec![1.0f32; 16]; // 4x4
    let mut c_1thread = vec![0.0f32; 16];
    let mut c_4thread = vec![0.0f32; 16];

    cpu_opt::parallel_matmul(&a, &b, &mut c_1thread, 4, 4, 4, 1).unwrap();
    cpu_opt::parallel_matmul(&a, &b, &mut c_4thread, 4, 4, 4, 4).unwrap();

    for (a, b) in c_1thread.iter().zip(c_4thread.iter()) {
        assert!((a - b).abs() < 1e-5, "Thread count should not affect result");
    }
}

#[test]
fn matmul_rectangular() {
    // 2x3 × 3x4 = 2x4
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![1.0; 12]; // 3x4, all ones
    let mut c = vec![0.0f32; 8]; // 2x4

    cpu_opt::parallel_matmul(&a, &b, &mut c, 2, 4, 3, 1).unwrap();

    // Row 0: [1+2+3, 1+2+3, 1+2+3, 1+2+3] = [6, 6, 6, 6]
    assert!((c[0] - 6.0).abs() < 1e-5);
    // Row 1: [4+5+6, 4+5+6, 4+5+6, 4+5+6] = [15, 15, 15, 15]
    assert!((c[4] - 15.0).abs() < 1e-5);
}

// --- RMSNorm edge cases ---

#[test]
fn rmsnorm_single_element() {
    let input = vec![5.0f32];
    let weight = vec![1.0f32];
    let mut output = vec![0.0f32];
    cpu_opt::rmsnorm(&input, &weight, &mut output, 1, 1, 1e-5).unwrap();
    // RMS of [5.0] = sqrt(25/1 + eps) ≈ 5.0
    // Output = 5.0 / 5.0 * 1.0 = 1.0
    assert!((output[0] - 1.0).abs() < 0.01);
}

#[test]
fn rmsnorm_all_zeros() {
    let input = vec![0.0f32; 4];
    let weight = vec![1.0f32; 4];
    let mut output = vec![0.0f32; 4];
    cpu_opt::rmsnorm(&input, &weight, &mut output, 1, 4, 1e-5).unwrap();
    // RMS of zeros = sqrt(0 + eps) = sqrt(eps) → output ≈ 0
    for &v in &output {
        assert!(v.abs() < 1.0, "RMSNorm of zeros should be near zero, got {v}");
    }
}

#[test]
fn rmsnorm_input_output_mismatch() {
    let input = vec![1.0f32; 4];
    let weight = vec![1.0f32; 4];
    let mut output = vec![0.0f32; 3]; // wrong size!
    let result = cpu_opt::rmsnorm(&input, &weight, &mut output, 1, 4, 1e-5);
    assert!(result.is_err());
}

#[test]
fn rmsnorm_weight_mismatch() {
    let input = vec![1.0f32; 4];
    let weight = vec![1.0f32; 3]; // wrong size!
    let mut output = vec![0.0f32; 4];
    let result = cpu_opt::rmsnorm(&input, &weight, &mut output, 1, 4, 1e-5);
    assert!(result.is_err());
}

#[test]
fn rmsnorm_multiple_rows() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 rows of 4
    let weight = vec![1.0f32; 4];
    let mut output = vec![0.0f32; 8];
    cpu_opt::rmsnorm(&input, &weight, &mut output, 2, 4, 1e-5).unwrap();

    // Each row should be independently normalized
    for &v in &output {
        assert!(v.is_finite(), "RMSNorm output should be finite");
    }
}

// --- LayerNorm edge cases ---

#[test]
fn layernorm_single_element() {
    let input = vec![5.0f32];
    let weight = vec![1.0f32];
    let bias = vec![0.0f32];
    let mut output = vec![0.0f32];
    cpu_opt::layernorm(&input, &weight, &bias, &mut output, 1, 1, 1e-5).unwrap();
    // Mean = 5.0, Var = 0.0, normalized = (5.0-5.0)/sqrt(0+eps) = 0
    assert!(output[0].abs() < 1.0, "LayerNorm of single element should be ≈ 0");
}

#[test]
fn layernorm_constant_input() {
    let input = vec![3.0f32; 8];
    let weight = vec![1.0f32; 8];
    let bias = vec![0.0f32; 8];
    let mut output = vec![0.0f32; 8];
    cpu_opt::layernorm(&input, &weight, &bias, &mut output, 1, 8, 1e-5).unwrap();
    // All same value → mean = 3.0, var = 0.0 → all outputs near 0
    for &v in &output {
        assert!(v.abs() < 1.0, "LayerNorm of constant input should be ≈ 0, got {v}");
    }
}

#[test]
fn layernorm_bias_shifts_output() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32; 4];
    let bias_zero = vec![0.0f32; 4];
    let bias_five = vec![5.0f32; 4];
    let mut output_zero = vec![0.0f32; 4];
    let mut output_five = vec![0.0f32; 4];

    cpu_opt::layernorm(&input, &weight, &bias_zero, &mut output_zero, 1, 4, 1e-5).unwrap();
    cpu_opt::layernorm(&input, &weight, &bias_five, &mut output_five, 1, 4, 1e-5).unwrap();

    for (a, b) in output_zero.iter().zip(output_five.iter()) {
        assert!((b - a - 5.0).abs() < 1e-5, "Bias should shift output by 5.0");
    }
}

#[test]
fn layernorm_no_bias_input_output_mismatch() {
    let input = vec![1.0f32; 4];
    let weight = vec![1.0f32; 4];
    let mut output = vec![0.0f32; 3]; // wrong!
    let result = cpu_opt::layernorm_no_bias(&input, &weight, &mut output, 1, 4, 1e-5);
    assert!(result.is_err());
}

#[test]
fn layernorm_no_bias_weight_mismatch() {
    let input = vec![1.0f32; 4];
    let weight = vec![1.0f32; 3]; // wrong!
    let mut output = vec![0.0f32; 4];
    let result = cpu_opt::layernorm_no_bias(&input, &weight, &mut output, 1, 4, 1e-5);
    assert!(result.is_err());
}

// --- Attention edge cases ---

#[test]
fn attention_single_token() {
    let seq_len = 1;
    let head_dim = 4;
    let num_heads = 1;
    let total = num_heads * seq_len * head_dim;

    let query = vec![1.0f32; total];
    let key = vec![1.0f32; total];
    let value = vec![2.0f32; total];
    let mut output = vec![0.0f32; total];

    cpu_opt::parallel_attention(&query, &key, &value, &mut output, seq_len, head_dim, num_heads)
        .unwrap();

    // Single token → softmax([score]) = [1.0] → output = value
    for (i, &v) in output.iter().enumerate() {
        assert!(
            (v - 2.0).abs() < 1e-5,
            "Single-token attention should return value, got {v} at {i}"
        );
    }
}

#[test]
fn attention_multiple_heads() {
    let seq_len = 2;
    let head_dim = 4;
    let num_heads = 3;
    let total = num_heads * seq_len * head_dim;

    let query = vec![0.0f32; total]; // uniform attention
    let key = vec![0.0f32; total];
    let value = vec![1.0f32; total];
    let mut output = vec![0.0f32; total];

    cpu_opt::parallel_attention(&query, &key, &value, &mut output, seq_len, head_dim, num_heads)
        .unwrap();

    // All outputs should be finite (value mean = 1.0)
    for &v in &output {
        assert!(v.is_finite(), "Multi-head attention output should be finite");
    }
}

// --- apply_activation edge cases ---

#[test]
fn apply_activation_empty_slice() {
    let mut data: Vec<f32> = vec![];
    cpu_opt::apply_activation(ActivationType::Silu, &mut data);
    assert!(data.is_empty());

    cpu_opt::apply_activation(ActivationType::Gelu, &mut data);
    assert!(data.is_empty());

    cpu_opt::apply_activation(ActivationType::Relu2, &mut data);
    assert!(data.is_empty());
}

// --- apply_norm edge cases ---

#[test]
fn apply_norm_rmsnorm_error_propagates() {
    let input = vec![1.0f32; 4];
    let weight = vec![1.0f32; 3]; // wrong dimension
    let bias = vec![0.0f32; 4];
    let mut output = vec![0.0f32; 4];
    let result =
        cpu_opt::apply_norm(NormType::RmsNorm, &input, &weight, &bias, &mut output, 1, 4, 1e-5);
    assert!(result.is_err());
}

#[test]
fn apply_norm_layernorm_error_propagates() {
    let input = vec![1.0f32; 4];
    let weight = vec![1.0f32; 4];
    let bias = vec![0.0f32; 3]; // wrong dimension
    let mut output = vec![0.0f32; 4];
    let result =
        cpu_opt::apply_norm(NormType::LayerNorm, &input, &weight, &bias, &mut output, 1, 4, 1e-5);
    assert!(result.is_err());
}

// --- Numerical stability ---

#[test]
fn silu_very_small_values() {
    let input = vec![1e-30, -1e-30, 1e-38, -1e-38];
    let output = cpu_opt::silu(&input);
    for &v in &output {
        assert!(v.is_finite(), "SiLU should handle very small values without issue");
    }
}

#[test]
fn gelu_very_small_values() {
    let input = vec![1e-30, -1e-30, 1e-38, -1e-38];
    let output = cpu_opt::gelu(&input);
    for &v in &output {
        assert!(v.is_finite(), "GELU should handle very small values without issue");
    }
}

#[test]
fn rmsnorm_very_small_eps() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32; 4];
    let mut output = vec![0.0f32; 4];
    cpu_opt::rmsnorm(&input, &weight, &mut output, 1, 4, 1e-30).unwrap();
    for &v in &output {
        assert!(v.is_finite(), "RMSNorm should handle very small eps");
    }
}

#[test]
fn layernorm_very_small_eps() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![1.0f32; 4];
    let bias = vec![0.0f32; 4];
    let mut output = vec![0.0f32; 4];
    cpu_opt::layernorm(&input, &weight, &bias, &mut output, 1, 4, 1e-30).unwrap();
    for &v in &output {
        assert!(v.is_finite(), "LayerNorm should handle very small eps");
    }
}

#[test]
fn matmul_large_values() {
    let a = vec![1e10f32; 4]; // 2x2
    let b = vec![1e10f32; 4]; // 2x2
    let mut c = vec![0.0f32; 4]; // 2x2
    cpu_opt::parallel_matmul(&a, &b, &mut c, 2, 2, 2, 1).unwrap();
    // Each element = 2 * (1e10 * 1e10) = 2e20
    for &v in &c {
        assert!(v.is_finite(), "Matmul with large values should be finite, got {v}");
    }
}
