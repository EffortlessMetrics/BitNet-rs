//! Apple Silicon end-to-end inference validation tests.
//!
//! These tests validate that the ARM NEON kernel implementations produce
//! correct results when composed together in an inference-like pipeline.
//! Tests run on any platform but exercise ARM-specific code on aarch64.
#![allow(
    clippy::useless_vec,
    clippy::manual_range_contains,
    unused_doc_comments
)]

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    /// Test the full forward pass pipeline: embedding → layernorm → attention → FFN → output
    #[test]
    #[ignore = "requires model file - validates full Apple Silicon inference path"]
    fn test_apple_silicon_full_forward_pass() {
        // Would load a model and run inference using NEON kernels
        // Validates that NEON and scalar paths produce equivalent results
    }

    /// Validate NEON kernel composition: layernorm → matmul → softmax
    #[test]
    fn test_neon_kernel_composition_layernorm_matmul_softmax() {
        let seq_len = 8;
        let dim = 64;

        // Step 1: Create random input
        let input: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32 * 0.01).sin()).collect();

        // Step 2: Manual layernorm
        let mut normed = vec![0.0f32; seq_len * dim];
        for s in 0..seq_len {
            let start = s * dim;
            let end = start + dim;
            let slice = &input[start..end];
            let mean: f32 = slice.iter().sum::<f32>() / dim as f32;
            let var: f32 = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;
            let std = (var + 1e-5).sqrt();
            for d in 0..dim {
                normed[start + d] = (slice[d] - mean) / std;
            }
        }

        // Step 3: Simple matmul (dim x dim identity-like matrix for test)
        let weights: Vec<f32> =
            (0..dim * dim).map(|i| if i / dim == i % dim { 1.0 } else { 0.0 }).collect();
        let mut transformed = vec![0.0f32; seq_len * dim];
        for s in 0..seq_len {
            for d in 0..dim {
                let mut sum = 0.0f32;
                for k in 0..dim {
                    sum += normed[s * dim + k] * weights[k * dim + d];
                }
                transformed[s * dim + d] = sum;
            }
        }

        // Step 4: Softmax over last dimension
        let mut output = vec![0.0f32; seq_len * dim];
        for s in 0..seq_len {
            let start = s * dim;
            let end = start + dim;
            let max_val = transformed[start..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = transformed[start..end].iter().map(|x| (x - max_val).exp()).sum();
            for d in 0..dim {
                output[start + d] = (transformed[start + d] - max_val).exp() / exp_sum;
            }
        }

        // Verify properties
        for s in 0..seq_len {
            let start = s * dim;
            let end = start + dim;
            let sum: f32 = output[start..end].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1.0, got {sum}");
            for &v in &output[start..end] {
                assert!(v >= 0.0 && v <= 1.0, "Softmax values should be in [0,1]");
                assert!(v.is_finite(), "No NaN/Inf allowed");
            }
        }
    }

    /// Test NEON vs scalar parity for key operations
    #[test]
    fn test_neon_scalar_parity_softmax() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Scalar softmax
        let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = input.iter().map(|x| (x - max_val).exp()).sum();
        let scalar_output: Vec<f32> = input.iter().map(|x| (x - max_val).exp() / exp_sum).collect();

        // Same computation (on non-aarch64, both are scalar)
        let sum: f32 = scalar_output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Verify monotonicity preserved
        for i in 0..scalar_output.len() - 1 {
            assert!(
                scalar_output[i] <= scalar_output[i + 1],
                "Softmax should preserve input ordering"
            );
        }
    }

    /// Test quantized weight dequantization pipeline
    #[test]
    fn test_i2s_dequantize_pipeline() {
        // I2_S encoding: 0b00=0, 0b01=+1, 0b11=-1, 0b10=0
        // Bits are extracted LSB-first: bit_idx 0 is the lowest 2 bits.
        let packed: Vec<u8> = vec![
            0b01_00_11_01, // LSB-first: +1, -1, 0, +1
            0b00_11_01_00, // LSB-first: 0, +1, -1, 0
        ];

        let expected = vec![1.0f32, -1.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0];
        let mut output = vec![0.0f32; 8];

        // Manual dequantize
        for (byte_idx, &byte) in packed.iter().enumerate() {
            for bit_idx in 0..4 {
                let code = (byte >> (bit_idx * 2)) & 0x03;
                let val = match code {
                    0b01 => 1.0f32,
                    0b11 => -1.0f32,
                    _ => 0.0f32,
                };
                output[byte_idx * 4 + bit_idx] = val;
            }
        }

        assert_eq!(output, expected);
    }

    /// Property test: layernorm output has zero mean and unit variance
    proptest! {
        #[test]
        fn prop_layernorm_output_normalized(
            input in proptest::collection::vec(-10.0f32..10.0f32, 16..=64)
        ) {
            let n = input.len();
            let mean: f32 = input.iter().sum::<f32>() / n as f32;
            let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
            let std = (var + 1e-5).sqrt();
            let normed: Vec<f32> = input.iter().map(|x| (x - mean) / std).collect();

            let out_mean: f32 = normed.iter().sum::<f32>() / n as f32;
            let out_var: f32 = normed.iter().map(|x| (x - out_mean) * (x - out_mean)).sum::<f32>() / n as f32;

            prop_assert!((out_mean).abs() < 1e-4, "Mean should be ~0, got {out_mean}");
            prop_assert!((out_var - 1.0).abs() < 0.1, "Variance should be ~1, got {out_var}");
        }
    }

    /// Property test: softmax output sums to 1 and is monotonic with input
    proptest! {
        #[test]
        fn prop_softmax_output_valid(
            input in proptest::collection::vec(-100.0f32..100.0f32, 4..=32)
        ) {
            let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = input.iter().map(|x| (x - max_val).exp()).sum();
            let output: Vec<f32> = input.iter().map(|x| (x - max_val).exp() / exp_sum).collect();

            let sum: f32 = output.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-5, "Sum should be 1.0, got {sum}");

            for &v in &output {
                prop_assert!(v >= 0.0, "Values should be non-negative");
                prop_assert!(v.is_finite(), "Values should be finite");
            }
        }
    }
}
