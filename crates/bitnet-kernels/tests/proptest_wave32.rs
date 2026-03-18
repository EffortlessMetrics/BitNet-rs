#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
//! Property-based tests for bitnet-kernels (wave 32).

use bitnet_kernels::cpu::batch::{batched_matmul, batched_softmax};
use bitnet_kernels::cpu::layer_norm::{LayerNormConfig, layer_norm, rms_norm};
use bitnet_kernels::cpu::linear::{LinearConfig, linear_cpu};
use bitnet_kernels::cpu::quantize::{dequantize_symmetric_i8, quantize_symmetric_i8};
use bitnet_kernels::cpu::transpose::TransposeKernel;
use proptest::prelude::*;

// ── Helpers ────────────────────────────────────────────────────────────

fn finite_vec(min_len: usize, max_len: usize) -> BoxedStrategy<Vec<f32>> {
    prop::collection::vec(-10.0f32..10.0, min_len..=max_len).boxed()
}

// ── Tests ──────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    // 1. Softmax: output sums to 1.0
    #[test]
    fn proptest_wave32_softmax_sums_to_one(
        len in 1usize..=64,
        seed in 0u64..10000,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let input: Vec<f32> = (0..len).map(|_| rng.gen_range(-10.0f32..10.0)).collect();
        let output = batched_softmax(&input, 1, len).unwrap();
        let sum: f32 = output.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "sum = {}", sum);
    }

    // 2. Softmax: all outputs non-negative
    #[test]
    fn proptest_wave32_softmax_all_nonnegative(
        input in finite_vec(1, 64),
    ) {
        let len = input.len();
        let output = batched_softmax(&input, 1, len).unwrap();
        for &v in &output {
            prop_assert!(v >= 0.0, "negative softmax output: {}", v);
        }
    }

    // 3. Softmax: monotonicity preserved
    #[test]
    fn proptest_wave32_softmax_monotonicity(
        input in finite_vec(2, 32),
    ) {
        let len = input.len();
        let output = batched_softmax(&input, 1, len).unwrap();
        for i in 0..len {
            for j in 0..len {
                if input[i] > input[j] {
                    prop_assert!(output[i] >= output[j] - 1e-7,
                        "monotonicity violated: input[{}]={} > input[{}]={} but output[{}]={} < output[{}]={}",
                        i, input[i], j, input[j], i, output[i], j, output[j]);
                }
            }
        }
    }

    // 4. Linear: zero input → zero result
    #[test]
    fn proptest_wave32_linear_zero_input(
        in_f in 1usize..=16,
        out_f in 1usize..=16,
    ) {
        let weight: Vec<f32> = (0..out_f * in_f).map(|i| (i as f32) * 0.1).collect();
        let x = vec![0.0f32; in_f];
        let mut y = vec![0.0f32; out_f];
        let config = LinearConfig {
            in_features: in_f,
            out_features: out_f,
            batch_size: 1,
            has_bias: false,
            ..Default::default()
        };
        linear_cpu(&x, &weight, None, &mut y, &config).unwrap();
        for &v in &y {
            prop_assert!(v.abs() < 1e-6, "expected zero, got {}", v);
        }
    }

    // 5. Linear: identity matrix → input unchanged
    #[test]
    fn proptest_wave32_linear_identity(
        n in 1usize..=16,
    ) {
        // Identity weight matrix (out_f × in_f, row-major)
        let mut weight = vec![0.0f32; n * n];
        for i in 0..n {
            weight[i * n + i] = 1.0;
        }
        let x: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let mut y = vec![0.0f32; n];
        let config = LinearConfig {
            in_features: n,
            out_features: n,
            batch_size: 1,
            has_bias: false,
            ..Default::default()
        };
        linear_cpu(&x, &weight, None, &mut y, &config).unwrap();
        for i in 0..n {
            prop_assert!((y[i] - x[i]).abs() < 1e-5,
                "identity mismatch at {}: expected {} got {}", i, x[i], y[i]);
        }
    }

    // 6. Layer norm: output has approximately zero mean (no affine)
    #[test]
    fn proptest_wave32_layer_norm_zero_mean(
        input in finite_vec(4, 64),
    ) {
        let dim = input.len();
        let config = LayerNormConfig {
            normalized_shape: vec![dim],
            eps: 1e-5,
            elementwise_affine: false,
        };
        let gamma = vec![1.0f32; dim];
        let output = layer_norm(&input, &gamma, None, &config).unwrap();
        let mean: f32 = output.iter().sum::<f32>() / dim as f32;
        prop_assert!(mean.abs() < 1e-4, "mean = {}", mean);
    }

    // 7. Layer norm: output has unit variance (no affine)
    #[test]
    fn proptest_wave32_layer_norm_unit_variance(
        input in prop::collection::vec(-10.0f32..10.0, 8..=64),
    ) {
        let dim = input.len();
        // Skip near-constant inputs where variance is essentially zero
        let input_var = {
            let m = input.iter().sum::<f32>() / dim as f32;
            input.iter().map(|&x| (x - m) * (x - m)).sum::<f32>() / dim as f32
        };
        prop_assume!(input_var > 1e-4);

        let config = LayerNormConfig {
            normalized_shape: vec![dim],
            eps: 1e-5,
            elementwise_affine: false,
        };
        let gamma = vec![1.0f32; dim];
        let output = layer_norm(&input, &gamma, None, &config).unwrap();
        let mean = output.iter().sum::<f32>() / dim as f32;
        let var = output.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
        prop_assert!((var - 1.0).abs() < 1e-3, "variance = {}", var);
    }

    // 8. RMS norm: output has expected RMS (gamma=1 → RMS ≈ 1)
    #[test]
    fn proptest_wave32_rms_norm_expected_rms(
        input in prop::collection::vec(-10.0f32..10.0, 4..=64),
    ) {
        let dim = input.len();
        let input_rms = (input.iter().map(|x| x * x).sum::<f32>() / dim as f32).sqrt();
        prop_assume!(input_rms > 1e-4);

        let config = LayerNormConfig {
            normalized_shape: vec![dim],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let gamma = vec![1.0f32; dim];
        let output = rms_norm(&input, &gamma, &config).unwrap();
        // After RMS norm with gamma=1, the RMS of output should be close to 1
        let output_rms = (output.iter().map(|x| x * x).sum::<f32>() / dim as f32).sqrt();
        prop_assert!((output_rms - 1.0).abs() < 0.1, "output RMS = {}", output_rms);
    }

    // 9. Quantize/dequantize: bounded error
    #[test]
    fn proptest_wave32_quantize_dequantize_bounded_error(
        input in prop::collection::vec(-5.0f32..5.0, 4..=64),
    ) {
        let (quantized, scale) = quantize_symmetric_i8(&input, 8);
        let recovered = dequantize_symmetric_i8(&quantized, scale);
        for (orig, rec) in input.iter().zip(recovered.iter()) {
            let err = (orig - rec).abs();
            // 8-bit quantization error should be bounded by step_size ≈ max/127
            let max_abs = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let bound = max_abs / 127.0 + 1e-5;
            prop_assert!(err <= bound + 1e-5,
                "quant error {} exceeds bound {} for input {}", err, bound, orig);
        }
    }

    // 10. Transpose: double transpose = identity
    #[test]
    fn proptest_wave32_transpose_2d_roundtrip(
        rows in 1usize..=16,
        cols in 1usize..=16,
    ) {
        let data: Vec<f32> = (0..(rows * cols) as u32).map(|i| i as f32).collect();
        let transposed = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        let roundtrip = TransposeKernel::transpose_2d(&transposed, cols, rows).unwrap();
        prop_assert_eq!(data, roundtrip);
    }

    // 11. Softmax batch: each row sums to 1
    #[test]
    fn proptest_wave32_batched_softmax_each_row_sums_to_one(
        batch in 1usize..=4,
        seq_len in 2usize..=16,
        seed in 0u64..10000,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let input: Vec<f32> = (0..batch * seq_len).map(|_| rng.gen_range(-10.0f32..10.0)).collect();
        let output = batched_softmax(&input, batch, seq_len).unwrap();
        for b in 0..batch {
            let row = &output[b * seq_len..(b + 1) * seq_len];
            let sum: f32 = row.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-5, "batch {} sum = {}", b, sum);
        }
    }

    // 12. Batched matmul: zero matrix → zero result
    #[test]
    fn proptest_wave32_batched_matmul_zero(
        m in 1usize..=4,
        k in 1usize..=4,
        n in 1usize..=4,
    ) {
        let a = vec![0.0f32; m * k];
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.1).collect();
        let result = batched_matmul(&a, &b, 1, m, k, n).unwrap();
        for &v in &result {
            prop_assert!(v.abs() < 1e-6, "expected zero, got {}", v);
        }
    }

    // 13. Layer norm with affine: gamma scales output
    #[test]
    fn proptest_wave32_layer_norm_gamma_scaling(
        input in prop::collection::vec(-10.0f32..10.0, 8..=32),
        scale in 0.1f32..5.0,
    ) {
        let dim = input.len();
        let config_base = LayerNormConfig {
            normalized_shape: vec![dim],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let gamma_one = vec![1.0f32; dim];
        let gamma_scaled = vec![scale; dim];
        let out_one = layer_norm(&input, &gamma_one, None, &config_base).unwrap();
        let out_scaled = layer_norm(&input, &gamma_scaled, None, &config_base).unwrap();
        for (a, b) in out_one.iter().zip(out_scaled.iter()) {
            prop_assert!((b - a * scale).abs() < 1e-4,
                "gamma scaling: {} vs {} * {}", b, a, scale);
        }
    }

    // 14. Transpose dimensions: output shape is (cols, rows)
    #[test]
    fn proptest_wave32_transpose_output_length(
        rows in 1usize..=16,
        cols in 1usize..=16,
    ) {
        let data: Vec<f32> = (0..(rows * cols) as u32).map(|i| i as f32).collect();
        let transposed = TransposeKernel::transpose_2d(&data, rows, cols).unwrap();
        prop_assert_eq!(transposed.len(), rows * cols);
        // Verify element at (0,j) in original → (j,0) in transposed
        for j in 0..cols {
            prop_assert!((data[j] - transposed[j * rows]).abs() < 1e-6);
        }
    }

    // 15. Quantize ternary: round-trip preserves sign
    #[test]
    fn proptest_wave32_quantize_ternary_sign_preservation(
        input in prop::collection::vec(-5.0f32..5.0, 4..=32),
    ) {
        use bitnet_kernels::cpu::quantize::quantize_ternary;
        let ternary = quantize_ternary(&input, 0.5);
        for (orig, &q) in input.iter().zip(ternary.iter()) {
            if orig.abs() <= 0.5 {
                prop_assert_eq!(q, 0, "expected zero for input {}", orig);
            } else if *orig > 0.5 {
                prop_assert_eq!(q, 1, "expected +1 for input {}", orig);
            } else {
                prop_assert_eq!(q, -1, "expected -1 for input {}", orig);
            }
        }
    }

    // 16. Linear: scalar multiplication (1×1 weight)
    #[test]
    fn proptest_wave32_linear_scalar(
        a_val in -10.0f32..10.0,
        x_val in -10.0f32..10.0,
    ) {
        let weight = vec![a_val];
        let x = vec![x_val];
        let mut y = vec![0.0f32];
        let config = LinearConfig {
            in_features: 1,
            out_features: 1,
            batch_size: 1,
            has_bias: false,
            ..Default::default()
        };
        linear_cpu(&x, &weight, None, &mut y, &config).unwrap();
        prop_assert!((y[0] - a_val * x_val).abs() < 1e-5);
    }

    // 17. Softmax with uniform input → uniform output
    #[test]
    fn proptest_wave32_softmax_uniform_input(
        val in -10.0f32..10.0,
        len in 2usize..=32,
    ) {
        let input = vec![val; len];
        let output = batched_softmax(&input, 1, len).unwrap();
        let expected = 1.0 / len as f32;
        for &v in &output {
            prop_assert!((v - expected).abs() < 1e-5,
                "expected uniform {} got {}", expected, v);
        }
    }

    // 18. Quantize/dequantize: zero input → zero output
    #[test]
    fn proptest_wave32_quantize_zero_input(
        len in 1usize..=32,
    ) {
        let input = vec![0.0f32; len];
        let (quantized, _scale) = quantize_symmetric_i8(&input, 8);
        let recovered = dequantize_symmetric_i8(&quantized, _scale);
        for &v in &recovered {
            prop_assert!(v.abs() < 1e-6, "expected zero, got {}", v);
        }
    }

    // 19. RMS norm: scaling input scales output equally
    #[test]
    fn proptest_wave32_rms_norm_scale_invariance(
        input in prop::collection::vec(0.1f32..10.0, 4..=32),
        scale_factor in 0.5f32..3.0,
    ) {
        let dim = input.len();
        let config = LayerNormConfig {
            normalized_shape: vec![dim],
            eps: 1e-5,
            elementwise_affine: true,
        };
        let gamma = vec![1.0f32; dim];
        let out_a = rms_norm(&input, &gamma, &config).unwrap();
        let scaled_input: Vec<f32> = input.iter().map(|&x| x * scale_factor).collect();
        let out_b = rms_norm(&scaled_input, &gamma, &config).unwrap();
        // RMS norm is approximately scale-invariant (except for eps)
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            prop_assert!((a - b).abs() < 0.1,
                "scale invariance violated: {} vs {} (scale={})", a, b, scale_factor);
        }
    }

    // 20. Batched matmul: identity matrix → input unchanged
    #[test]
    fn proptest_wave32_batched_matmul_identity(
        n in 1usize..=8,
    ) {
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        let b: Vec<f32> = (0..n * n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let result = batched_matmul(&identity, &b, 1, n, n, n).unwrap();
        for i in 0..result.len() {
            prop_assert!((result[i] - b[i]).abs() < 1e-5,
                "identity matmul mismatch at {}: {} vs {}", i, result[i], b[i]);
        }
    }
}
