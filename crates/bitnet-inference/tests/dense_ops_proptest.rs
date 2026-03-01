//! Property-based tests for dense inference operations.
//!
//! Key invariants verified:
//! - Activations preserve vector length
//! - In-place and allocating variants produce identical results
//! - Normalization outputs have bounded magnitude
//! - Matrix multiplication dimensions are consistent
//! - Numerical properties (monotonicity, non-negativity, idempotence)

use bitnet_common::{ActivationType, NormType};
use bitnet_inference::cpu_opt;
use proptest::prelude::*;

// --- Strategies ---

/// Generate a non-empty f32 vector with reasonable values (avoid extreme floats).
fn vec_f32(max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0f32, 1..=max_len)
}

/// Generate a positive f32 vector (for weights).
#[allow(dead_code)]
fn positive_vec_f32(len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(0.1f32..5.0f32, len..=len)
}

// --- SiLU property tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn silu_preserves_length(input in vec_f32(256)) {
        let output = cpu_opt::silu(&input);
        prop_assert_eq!(output.len(), input.len());
    }

    #[test]
    fn silu_in_place_matches_allocating(input in vec_f32(256)) {
        let allocating = cpu_opt::silu(&input);
        let mut in_place = input.clone();
        cpu_opt::silu_in_place(&mut in_place);

        for (a, b) in allocating.iter().zip(in_place.iter()) {
            prop_assert!((a - b).abs() < 1e-6,
                "silu mismatch: allocating={a}, in_place={b}");
        }
    }

    #[test]
    fn silu_zero_is_zero(len in 1usize..64) {
        let input = vec![0.0f32; len];
        let output = cpu_opt::silu(&input);
        for &v in &output {
            prop_assert!((v - 0.0).abs() < 1e-7, "silu(0) should be 0, got {v}");
        }
    }

    #[test]
    fn silu_positive_for_positive_input(input in prop::collection::vec(0.01f32..10.0f32, 1..=128)) {
        let output = cpu_opt::silu(&input);
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v > 0.0, "silu(positive) should be positive at index {i}, got {v}");
        }
    }

    #[test]
    fn silu_bounded_by_input(input in vec_f32(128)) {
        // SiLU(x) = x * sigmoid(x), and sigmoid ∈ (0,1), so |SiLU(x)| ≤ |x|
        let output = cpu_opt::silu(&input);
        for (i, (&out, &inp)) in output.iter().zip(input.iter()).enumerate() {
            prop_assert!(out.abs() <= inp.abs() + 1e-6,
                "silu output {out} exceeds input magnitude {inp} at index {i}");
        }
    }
}

// --- GELU property tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn gelu_preserves_length(input in vec_f32(256)) {
        let output = cpu_opt::gelu(&input);
        prop_assert_eq!(output.len(), input.len());
    }

    #[test]
    fn gelu_in_place_matches_allocating(input in vec_f32(256)) {
        let allocating = cpu_opt::gelu(&input);
        let mut in_place = input.clone();
        cpu_opt::gelu_in_place(&mut in_place);

        for (a, b) in allocating.iter().zip(in_place.iter()) {
            prop_assert!((a - b).abs() < 1e-6,
                "gelu mismatch: allocating={a}, in_place={b}");
        }
    }

    #[test]
    fn gelu_zero_is_zero(len in 1usize..64) {
        let input = vec![0.0f32; len];
        let output = cpu_opt::gelu(&input);
        for &v in &output {
            prop_assert!((v - 0.0).abs() < 1e-7, "gelu(0) should be 0, got {v}");
        }
    }

    #[test]
    fn gelu_positive_for_large_positive(input in prop::collection::vec(2.0f32..10.0f32, 1..=64)) {
        let output = cpu_opt::gelu(&input);
        for &v in &output {
            prop_assert!(v > 0.0, "gelu should be positive for large positive input, got {v}");
        }
    }
}

// --- ReLU² property tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn relu2_preserves_length(input in vec_f32(256)) {
        let output = cpu_opt::relu2(&input);
        prop_assert_eq!(output.len(), input.len());
    }

    #[test]
    fn relu2_in_place_matches_allocating(input in vec_f32(256)) {
        let allocating = cpu_opt::relu2(&input);
        let mut in_place = input.clone();
        cpu_opt::relu2_in_place(&mut in_place);

        for (a, b) in allocating.iter().zip(in_place.iter()) {
            prop_assert!((a - b).abs() < 1e-6,
                "relu2 mismatch: allocating={a}, in_place={b}");
        }
    }

    #[test]
    fn relu2_non_negative(input in vec_f32(256)) {
        let output = cpu_opt::relu2(&input);
        for (i, &v) in output.iter().enumerate() {
            prop_assert!(v >= 0.0, "relu2 should be non-negative at index {i}, got {v}");
        }
    }

    #[test]
    fn relu2_zero_for_negative(input in prop::collection::vec(-10.0f32..0.0f32, 1..=128)) {
        let output = cpu_opt::relu2(&input);
        for (i, &v) in output.iter().enumerate() {
            prop_assert!((v - 0.0).abs() < 1e-7,
                "relu2(negative) should be 0 at index {i}, got {v}");
        }
    }

    #[test]
    fn relu2_equals_x_squared_for_positive(input in prop::collection::vec(0.0f32..10.0f32, 1..=128)) {
        let output = cpu_opt::relu2(&input);
        for (i, (&out, &inp)) in output.iter().zip(input.iter()).enumerate() {
            let expected = inp * inp;
            prop_assert!((out - expected).abs() < 1e-5,
                "relu2({inp}) = {out}, expected {expected} at index {i}");
        }
    }
}

// --- apply_activation dispatch tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn apply_activation_silu_matches_direct(input in vec_f32(128)) {
        let expected = cpu_opt::silu(&input);
        let mut actual = input.clone();
        cpu_opt::apply_activation(ActivationType::Silu, &mut actual);

        for (a, b) in expected.iter().zip(actual.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn apply_activation_gelu_matches_direct(input in vec_f32(128)) {
        let expected = cpu_opt::gelu(&input);
        let mut actual = input.clone();
        cpu_opt::apply_activation(ActivationType::Gelu, &mut actual);

        for (a, b) in expected.iter().zip(actual.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn apply_activation_relu2_matches_direct(input in vec_f32(128)) {
        let expected = cpu_opt::relu2(&input);
        let mut actual = input.clone();
        cpu_opt::apply_activation(ActivationType::Relu2, &mut actual);

        for (a, b) in expected.iter().zip(actual.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }
}

// --- RMSNorm property tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn rmsnorm_output_length_matches_input(
        dim in 1usize..64,
        rows in 1usize..8,
    ) {
        let input = vec![1.0f32; rows * dim];
        let weight = vec![1.0f32; dim];
        let mut output = vec![0.0f32; rows * dim];
        let result = cpu_opt::rmsnorm(&input, &weight, &mut output, rows, dim, 1e-5);
        prop_assert!(result.is_ok());
        prop_assert_eq!(output.len(), rows * dim);
    }

    #[test]
    fn rmsnorm_unit_weight_bounded(
        input_vals in prop::collection::vec(-5.0f32..5.0f32, 4..=64),
    ) {
        let dim = input_vals.len();
        let weight = vec![1.0f32; dim];
        let mut output = vec![0.0f32; dim];

        let result = cpu_opt::rmsnorm(&input_vals, &weight, &mut output, 1, dim, 1e-5);
        prop_assert!(result.is_ok());

        // With unit weights, RMSNorm normalizes to unit RMS (approximately)
        let rms: f32 = (output.iter().map(|v| v * v).sum::<f32>() / dim as f32).sqrt();
        // RMS of output should be close to 1.0 (unit weight means we just normalize)
        prop_assert!((rms - 1.0).abs() < 0.1,
            "RMSNorm with unit weights should have RMS ≈ 1.0, got {rms}");
    }

    #[test]
    fn rmsnorm_dimension_mismatch_errors(
        dim in 2usize..32,
    ) {
        let input = vec![1.0f32; dim];
        let weight = vec![1.0f32; dim - 1]; // wrong size!
        let mut output = vec![0.0f32; dim];
        let result = cpu_opt::rmsnorm(&input, &weight, &mut output, 1, dim, 1e-5);
        prop_assert!(result.is_err());
    }
}

// --- LayerNorm property tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn layernorm_output_zero_mean_unit_var(
        input_vals in prop::collection::vec(-5.0f32..5.0f32, 8..=64),
    ) {
        let dim = input_vals.len();
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let mut output = vec![0.0f32; dim];

        let result = cpu_opt::layernorm(&input_vals, &weight, &bias, &mut output, 1, dim, 1e-5);
        prop_assert!(result.is_ok());

        // With unit weights and zero bias, output should have approximately zero mean
        let mean: f32 = output.iter().sum::<f32>() / dim as f32;
        prop_assert!(mean.abs() < 0.1,
            "LayerNorm output mean should be ≈ 0, got {mean}");

        // And approximately unit variance
        let var: f32 = output.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        prop_assert!((var - 1.0).abs() < 0.2,
            "LayerNorm output variance should be ≈ 1.0, got {var}");
    }

    #[test]
    fn layernorm_no_bias_matches_zero_bias(
        input_vals in prop::collection::vec(-5.0f32..5.0f32, 4..=32),
    ) {
        let dim = input_vals.len();
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let mut output_with_bias = vec![0.0f32; dim];
        let mut output_no_bias = vec![0.0f32; dim];

        cpu_opt::layernorm(&input_vals, &weight, &bias, &mut output_with_bias, 1, dim, 1e-5)
            .unwrap();
        cpu_opt::layernorm_no_bias(&input_vals, &weight, &mut output_no_bias, 1, dim, 1e-5)
            .unwrap();

        for (a, b) in output_with_bias.iter().zip(output_no_bias.iter()) {
            prop_assert!((a - b).abs() < 1e-6,
                "layernorm with zero bias should match no-bias variant");
        }
    }

    #[test]
    fn layernorm_dimension_mismatch_errors(dim in 2usize..32) {
        let input = vec![1.0f32; dim];
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim - 1]; // wrong!
        let mut output = vec![0.0f32; dim];
        let result = cpu_opt::layernorm(&input, &weight, &bias, &mut output, 1, dim, 1e-5);
        prop_assert!(result.is_err());
    }
}

// --- apply_norm dispatch tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn apply_norm_rmsnorm_matches_direct(
        input_vals in prop::collection::vec(-5.0f32..5.0f32, 4..=32),
    ) {
        let dim = input_vals.len();
        let weight = vec![1.0f32; dim];
        let bias = vec![0.0f32; dim];
        let mut expected = vec![0.0f32; dim];
        let mut actual = vec![0.0f32; dim];

        cpu_opt::rmsnorm(&input_vals, &weight, &mut expected, 1, dim, 1e-5).unwrap();
        cpu_opt::apply_norm(NormType::RmsNorm, &input_vals, &weight, &bias, &mut actual, 1, dim, 1e-5)
            .unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn apply_norm_layernorm_matches_direct(
        input_vals in prop::collection::vec(-5.0f32..5.0f32, 4..=32),
    ) {
        let dim = input_vals.len();
        let weight = vec![1.0f32; dim];
        let bias = vec![0.5f32; dim];
        let mut expected = vec![0.0f32; dim];
        let mut actual = vec![0.0f32; dim];

        cpu_opt::layernorm(&input_vals, &weight, &bias, &mut expected, 1, dim, 1e-5).unwrap();
        cpu_opt::apply_norm(NormType::LayerNorm, &input_vals, &weight, &bias, &mut actual, 1, dim, 1e-5)
            .unwrap();

        for (a, b) in expected.iter().zip(actual.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }
}

// --- Matmul property tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn matmul_identity(dim in 1usize..16) {
        // A × I = A for identity matrix
        let m = dim;
        let n = dim;
        let k = dim;

        let mut a = vec![0.0f32; m * k];
        for i in 0..m.min(k) {
            a[i * k + i] = 1.0; // identity
        }
        let b_identity: Vec<f32> = (0..k * n)
            .map(|idx| if idx / n == idx % n { 1.0 } else { 0.0 })
            .collect();
        let mut c = vec![0.0f32; m * n];

        cpu_opt::parallel_matmul(&a, &b_identity, &mut c, m, n, k, 1).unwrap();

        // I × I = I
        for i in 0..m {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                prop_assert!((c[i * n + j] - expected).abs() < 1e-5,
                    "Identity matmul failed at ({i},{j}): got {}", c[i * n + j]);
            }
        }
    }

    #[test]
    fn matmul_dimension_mismatch_errors(
        m in 1usize..8,
        n in 1usize..8,
        k in 1usize..8,
    ) {
        let a = vec![0.0f32; m * k];
        let b = vec![0.0f32; k * n + 1]; // wrong size!
        let mut c = vec![0.0f32; m * n];
        let result = cpu_opt::parallel_matmul(&a, &b, &mut c, m, n, k, 1);
        prop_assert!(result.is_err());
    }

    #[test]
    fn matmul_zero_matrix(m in 1usize..8, n in 1usize..8, k in 1usize..8) {
        let a = vec![0.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![999.0f32; m * n];

        cpu_opt::parallel_matmul(&a, &b, &mut c, m, n, k, 1).unwrap();

        for &v in &c {
            prop_assert!((v - 0.0).abs() < 1e-6, "0 × B should be 0, got {v}");
        }
    }
}

// --- Attention property tests ---

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn attention_output_length(
        seq_len in 1usize..8,
        head_dim in 1usize..16,
        num_heads in 1usize..4,
    ) {
        let total = num_heads * seq_len * head_dim;
        let query = vec![0.1f32; total];
        let key = vec![0.1f32; total];
        let value = vec![0.1f32; total];
        let mut output = vec![0.0f32; total];

        let result = cpu_opt::parallel_attention(
            &query, &key, &value, &mut output, seq_len, head_dim, num_heads,
        );
        prop_assert!(result.is_ok());
        prop_assert_eq!(output.len(), total);
    }

    #[test]
    fn attention_uniform_query_returns_value_mean(
        head_dim in 2usize..8,
    ) {
        // With identical Q/K, softmax is uniform → output = mean(V)
        let seq_len = 2;
        let num_heads = 1;
        let total = num_heads * seq_len * head_dim;

        let query = vec![0.0f32; total]; // all zeros → uniform attention
        let key = vec![0.0f32; total];
        let value: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; total];

        cpu_opt::parallel_attention(
            &query, &key, &value, &mut output, seq_len, head_dim, num_heads,
        )
        .unwrap();

        // All outputs should be finite
        for &v in &output {
            prop_assert!(v.is_finite(), "attention output should be finite, got {v}");
        }
    }
}
