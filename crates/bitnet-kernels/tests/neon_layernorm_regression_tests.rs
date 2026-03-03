#![cfg(all(feature = "cpu", target_arch = "aarch64"))]
#![allow(
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    unused_imports,
    dead_code
)]

// ============================================================================
// Helper module with pure-Rust reference implementations
// ============================================================================

mod reference_impl {
    /// Standard LayerNorm reference implementation
    /// Computes: output = gamma * (input - mean) / sqrt(variance + eps) + beta
    pub fn reference_layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();

        // Compute mean
        let mean = input.iter().sum::<f32>() / n as f32;

        // Compute variance
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

        // Normalize and apply gamma/beta
        let std_dev = (variance + eps).sqrt();
        input
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let normalized = (x - mean) / std_dev;
                gamma[i] * normalized + beta[i]
            })
            .collect()
    }

    /// RMSNorm reference implementation (used in LLaMA/BitNet)
    /// Computes: output = gamma * input / sqrt(mean_of_squares + eps)
    pub fn reference_rms_norm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len();

        // Compute mean of squares (RMS)
        let mean_of_squares = input.iter().map(|x| x.powi(2)).sum::<f32>() / n as f32;
        let rms = (mean_of_squares + eps).sqrt();

        // Apply gamma scaling
        input
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let normalized = x / rms;
                gamma[i] * normalized
            })
            .collect()
    }
}

// ============================================================================
// Module: Basic LayerNorm tests
// ============================================================================

mod basic_layernorm {
    use crate::reference_impl::{reference_layer_norm, reference_rms_norm};

    /// Test that gamma=1, beta=0 produces zero-mean unit-variance output
    #[test]
    fn test_layernorm_unit_params() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        let eps = 1e-6;

        let output = reference_layer_norm(&input, &gamma, &beta, eps);

        // Output should have mean ≈ 0
        let mean = output.iter().sum::<f32>() / output.len() as f32;
        assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);

        // Output should have variance ≈ 1
        let variance = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output.len() as f32;
        assert!((variance - 1.0).abs() < 1e-5, "Variance should be ~1, got {}", variance);
    }

    /// Test that if input already has mean=0 and variance=1, output ≈ input
    #[test]
    fn test_layernorm_identity_preserves_mean_zero() {
        // Create input with mean=0 and variance=1
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mean = input.iter().sum::<f32>() / input.len() as f32;
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;

        // Normalize to mean=0, var=1
        let input_normalized: Vec<f32> =
            input.iter().map(|x| (x - mean) / variance.sqrt()).collect();

        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        let eps = 1e-6;

        let output = reference_layer_norm(&input_normalized, &gamma, &beta, eps);

        // Output should be very close to input
        for i in 0..input_normalized.len() {
            assert!(
                (output[i] - input_normalized[i]).abs() < 1e-5,
                "Output at index {} should match input",
                i
            );
        }
    }

    /// Test that gamma scales output correctly
    #[test]
    fn test_layernorm_scaling_with_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let beta = vec![0.0; 5];
        let eps = 1e-6;

        let gamma_1x = vec![1.0; 5];
        let output_1x = reference_layer_norm(&input, &gamma_1x, &beta, eps);

        let gamma_2x = vec![2.0; 5];
        let output_2x = reference_layer_norm(&input, &gamma_2x, &beta, eps);

        // output_2x should be approximately 2x output_1x
        for i in 0..output_1x.len() {
            assert!(
                (output_2x[i] - 2.0 * output_1x[i]).abs() < 1e-5,
                "Scaling with gamma=2 should double output at index {}",
                i
            );
        }
    }

    /// Test that beta shifts output correctly
    #[test]
    fn test_layernorm_shift_with_beta() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![1.0; 5];
        let eps = 1e-6;

        let beta_0 = vec![0.0; 5];
        let output_no_shift = reference_layer_norm(&input, &gamma, &beta_0, eps);

        let beta_shift = vec![10.0; 5];
        let output_shifted = reference_layer_norm(&input, &gamma, &beta_shift, eps);

        // output_shifted should be output_no_shift + 10
        for i in 0..output_no_shift.len() {
            assert!(
                (output_shifted[i] - (output_no_shift[i] + 10.0)).abs() < 1e-5,
                "Beta shift should add 10 to output at index {}",
                i
            );
        }
    }

    /// Test that each row normalizes independently
    #[test]
    fn test_layernorm_batch_independence() {
        // Simulate two sequences
        let row1 = vec![1.0, 2.0, 3.0];
        let row2 = vec![100.0, 200.0, 300.0];

        let gamma_1 = vec![1.0; 3];
        let beta_1 = vec![0.0; 3];
        let eps = 1e-6;

        let out1 = reference_layer_norm(&row1, &gamma_1, &beta_1, eps);
        let out2 = reference_layer_norm(&row2, &gamma_1, &beta_1, eps);

        // Both outputs should have mean ≈ 0 and variance ≈ 1 independently
        let mean1 = out1.iter().sum::<f32>() / out1.len() as f32;
        let var1 = out1.iter().map(|x| (x - mean1).powi(2)).sum::<f32>() / out1.len() as f32;

        let mean2 = out2.iter().sum::<f32>() / out2.len() as f32;
        let var2 = out2.iter().map(|x| (x - mean2).powi(2)).sum::<f32>() / out2.len() as f32;

        assert!(mean1.abs() < 1e-5, "Row1 mean should be ~0");
        assert!((var1 - 1.0).abs() < 1e-5, "Row1 variance should be ~1");
        assert!(mean2.abs() < 1e-5, "Row2 mean should be ~0");
        assert!((var2 - 1.0).abs() < 1e-5, "Row2 variance should be ~1");
    }
}

// ============================================================================
// Module: RMSNorm tests
// ============================================================================

mod rms_norm {
    use crate::reference_impl::reference_rms_norm;

    /// Test that with gamma=1, output magnitude is ~1.0
    #[test]
    fn test_rms_norm_unit_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![1.0; 5];
        let eps = 1e-6;

        let output = reference_rms_norm(&input, &gamma, eps);

        // RMS of output should be ~1.0
        let rms = (output.iter().map(|x| x.powi(2)).sum::<f32>() / output.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-5, "RMS of output should be ~1.0, got {}", rms);
    }

    /// Test that gamma=2.0 doubles output magnitude
    #[test]
    fn test_rms_norm_scaling_with_gamma() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let eps = 1e-6;

        let gamma_1x = vec![1.0; 5];
        let output_1x = reference_rms_norm(&input, &gamma_1x, eps);

        let gamma_2x = vec![2.0; 5];
        let output_2x = reference_rms_norm(&input, &gamma_2x, eps);

        // RMS of output_2x should be 2x the RMS of output_1x
        let rms_1x =
            (output_1x.iter().map(|x| x.powi(2)).sum::<f32>() / output_1x.len() as f32).sqrt();
        let rms_2x =
            (output_2x.iter().map(|x| x.powi(2)).sum::<f32>() / output_2x.len() as f32).sqrt();

        assert!((rms_2x - 2.0 * rms_1x).abs() < 1e-5, "RMS should scale by gamma factor");
    }

    /// Test that all zeros produce all zeros (with eps)
    #[test]
    fn test_rms_norm_zero_input() {
        let input = vec![0.0, 0.0, 0.0, 0.0];
        let gamma = vec![1.0; 4];
        let eps = 1e-6;

        let output = reference_rms_norm(&input, &gamma, eps);

        // All outputs should be 0 (since input is 0, normalized by eps)
        for val in output {
            assert!(val.abs() < 1e-10, "Zero input should produce near-zero output");
        }
    }

    /// Test that constant input produces constant*gamma output
    #[test]
    fn test_rms_norm_constant_input() {
        let input = vec![5.0, 5.0, 5.0, 5.0];
        let gamma = vec![2.0; 4];
        let eps = 1e-6;

        let output = reference_rms_norm(&input, &gamma, eps);

        // All outputs should be gamma * (constant / constant) = gamma
        for val in output {
            assert!(
                (val - 2.0).abs() < 1e-5,
                "Constant input should produce constant*gamma output"
            );
        }
    }
}

// ============================================================================
// Module: Numerical Stability tests
// ============================================================================

mod numerical_stability {
    use crate::reference_impl::{reference_layer_norm, reference_rms_norm};

    /// Test that large value inputs normalize correctly
    #[test]
    fn test_layernorm_large_values() {
        let input = vec![1e6, 2e6, 3e6, 4e6, 5e6];
        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        let eps = 1e-6;

        let output = reference_layer_norm(&input, &gamma, &beta, eps);

        // Output should still have mean ≈ 0 and variance ≈ 1
        let mean = output.iter().sum::<f32>() / output.len() as f32;
        let variance = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output.len() as f32;

        assert!(mean.abs() < 1e-3, "Mean should be ~0 for large values");
        assert!((variance - 1.0).abs() < 1e-3, "Variance should be ~1 for large values");
    }

    /// Test that very small eps works without overflow/underflow
    #[test]
    fn test_layernorm_small_eps() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        let eps = 1e-12;

        let output = reference_layer_norm(&input, &gamma, &beta, eps);

        // Should not panic or produce NaN/Inf
        for val in output {
            assert!(val.is_finite(), "Output should be finite with small eps");
        }
    }

    /// Test near-constant input uses eps properly to avoid division by zero
    #[test]
    fn test_layernorm_near_zero_variance() {
        let input = vec![1.0, 1.0000001, 1.0000002, 1.0, 1.0000001];
        let gamma = vec![1.0; 5];
        let beta = vec![0.0; 5];
        let eps = 1e-6;

        let output = reference_layer_norm(&input, &gamma, &beta, eps);

        // Should not produce NaN or Inf
        for val in output {
            assert!(val.is_finite(), "Output should be finite for near-constant input");
        }
    }

    /// Test RMSNorm with large values
    #[test]
    fn test_rms_norm_large_values() {
        let input = vec![1e6, 2e6, 3e6, 4e6, 5e6];
        let gamma = vec![1.0; 5];
        let eps = 1e-6;

        let output = reference_rms_norm(&input, &gamma, eps);

        // RMS should still be ~1.0
        let rms = (output.iter().map(|x| x.powi(2)).sum::<f32>() / output.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-3, "RMS should be ~1.0 even for large values");

        // All outputs should be finite
        for val in output {
            assert!(val.is_finite(), "Output should be finite for large input");
        }
    }
}

// ============================================================================
// Module: BitNet-specific tests
// ============================================================================

mod bitnet_specific {
    use crate::reference_impl::{reference_layer_norm, reference_rms_norm};

    /// Test with typical BitNet model hidden dimensions
    #[test]
    fn test_layernorm_typical_hidden_dims() {
        // Test with dim=2048
        let input_2048: Vec<f32> = (0..2048).map(|i| (i as f32).sin()).collect();
        let gamma_2048 = vec![1.0; 2048];
        let beta_2048 = vec![0.0; 2048];
        let eps = 1e-6;

        let output_2048 = reference_layer_norm(&input_2048, &gamma_2048, &beta_2048, eps);

        let mean_2048 = output_2048.iter().sum::<f32>() / output_2048.len() as f32;
        let var_2048 = output_2048.iter().map(|x| (x - mean_2048).powi(2)).sum::<f32>()
            / output_2048.len() as f32;

        assert!(mean_2048.abs() < 1e-5, "2048-dim LayerNorm mean should be ~0");
        assert!((var_2048 - 1.0).abs() < 1e-5, "2048-dim LayerNorm variance should be ~1");

        // Test with dim=4096
        let input_4096: Vec<f32> = (0..4096).map(|i| (i as f32).sin()).collect();
        let gamma_4096 = vec![1.0; 4096];
        let beta_4096 = vec![0.0; 4096];

        let output_4096 = reference_layer_norm(&input_4096, &gamma_4096, &beta_4096, eps);

        let mean_4096 = output_4096.iter().sum::<f32>() / output_4096.len() as f32;
        let var_4096 = output_4096.iter().map(|x| (x - mean_4096).powi(2)).sum::<f32>()
            / output_4096.len() as f32;

        assert!(mean_4096.abs() < 1e-5, "4096-dim LayerNorm mean should be ~0");
        assert!((var_4096 - 1.0).abs() < 1e-5, "4096-dim LayerNorm variance should be ~1");
    }

    /// Verify RMSNorm output before ternary quantization
    #[test]
    fn test_rms_norm_pre_quantization() {
        let input = vec![0.5, -0.3, 0.8, -0.1, 0.6];
        let gamma = vec![0.5; 5];
        let eps = 1e-6;

        let output = reference_rms_norm(&input, &gamma, eps);

        // After RMSNorm, values should be in a reasonable range for quantization
        // Typically ternary quantization operates on [-1, 0, 1]
        for val in &output {
            // Most values should be < 2 in magnitude (reasonable pre-quant range)
            assert!(
                val.abs() < 3.0,
                "RMS-normalized value should be in reasonable range for quantization: {}",
                val
            );
        }

        // Output should still be finite
        for val in output {
            assert!(val.is_finite(), "Output should be finite");
        }
    }

    /// Simulate F16 precision by quantizing gamma to lower precision
    #[test]
    fn test_layernorm_f16_gamma_simulation() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let beta = vec![0.0; 5];
        let eps = 1e-6;

        // Full precision gamma
        let gamma_f32 = vec![1.5; 5];
        let output_f32 = reference_layer_norm(&input, &gamma_f32, &beta, eps);

        // Simulate F16 by quantizing gamma
        let gamma_f16_simulated: Vec<f32> = gamma_f32
            .iter()
            .map(|&x| {
                // Simple F16 simulation: round to 2 decimal places
                (x * 100.0).round() / 100.0
            })
            .collect();
        let output_f16_simulated = reference_layer_norm(&input, &gamma_f16_simulated, &beta, eps);

        // Outputs should be close (within F16 precision tolerance)
        let tolerance = 1e-3;
        for i in 0..output_f32.len() {
            assert!(
                (output_f32[i] - output_f16_simulated[i]).abs() < tolerance,
                "F16 gamma simulation should produce similar output at index {}",
                i
            );
        }
    }
}
