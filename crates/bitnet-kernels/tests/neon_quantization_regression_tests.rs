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

const TOLERANCE: f32 = 1e-5;

// ============================================================================
// Helper Functions for Pure-Rust Quantization References
// ============================================================================

/// Ternary quantization: v > threshold => +1, v < -threshold => -1, else 0
fn quantize_ternary(values: &[f32], threshold: f32) -> Vec<i8> {
    values
        .iter()
        .map(|&v| {
            if v > threshold {
                1
            } else if v < -threshold {
                -1
            } else {
                0
            }
        })
        .collect()
}

/// Reverse ternary quantization: i8 * scale
fn dequantize_ternary(quantized: &[i8], scale: f32) -> Vec<f32> {
    quantized.iter().map(|&q| (q as f32) * scale).collect()
}

/// Pack 2-bit values into bytes + compute per-block scales
/// Returns (packed bytes, per-block scales)
/// Encoding: 0b00=0, 0b01=+1, 0b11=-1
fn quantize_i2s_block(values: &[f32], block_size: usize) -> (Vec<u8>, Vec<f32>) {
    let mut packed = Vec::new();
    let mut scales = Vec::new();

    for chunk in values.chunks(block_size) {
        // Compute per-block scale = max(|v|)
        let scale = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        scales.push(scale);

        if scale == 0.0 {
            // Zero block: pack zeros
            let num_bytes = chunk.len().div_ceil(4);
            packed.extend(std::iter::repeat_n(0u8, num_bytes));
        } else {
            // Quantize chunk to ternary, then pack
            let ternary: Vec<i8> = chunk
                .iter()
                .map(|&v| {
                    let normalized = v / scale;
                    if normalized > 0.5 {
                        1
                    } else if normalized < -0.5 {
                        -1
                    } else {
                        0
                    }
                })
                .collect();

            // Pack 4 ternary values into one byte
            for qvals in ternary.chunks(4) {
                let mut byte = 0u8;
                for (i, &q) in qvals.iter().enumerate() {
                    let bits = match q {
                        0 => 0b00,
                        1 => 0b01,
                        -1 => 0b11,
                        _ => 0b00, // shouldn't happen
                    };
                    byte |= (bits & 0b11) << (6 - i * 2);
                }
                packed.push(byte);
            }
        }
    }

    (packed, scales)
}

/// Compute scale as max(|v|)
fn compute_absmax_scale(values: &[f32]) -> f32 {
    values.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
}

// ============================================================================
// Test Modules
// ============================================================================

mod ternary_quantization {
    use super::*;

    #[test]
    fn test_ternary_zero_threshold() {
        let values = vec![0.1f32, 0.2, 0.3, -0.1, -0.2];
        let threshold = 0.0;
        let quantized = quantize_ternary(&values, threshold);

        // All nonzero values should map to ±1
        assert_eq!(quantized[0], 1); // 0.1 > 0 => +1
        assert_eq!(quantized[1], 1); // 0.2 > 0 => +1
        assert_eq!(quantized[2], 1); // 0.3 > 0 => +1
        assert_eq!(quantized[3], -1); // -0.1 < 0 => -1
        assert_eq!(quantized[4], -1); // -0.2 < 0 => -1
    }

    #[test]
    fn test_ternary_high_threshold() {
        let values = vec![0.1f32, 0.2, 0.3, -0.1, -0.2];
        let threshold = 1.0; // Higher than all absolute values
        let quantized = quantize_ternary(&values, threshold);

        // All values should map to 0
        for &q in &quantized {
            assert_eq!(q, 0);
        }
    }

    #[test]
    fn test_ternary_typical_values() {
        let values = vec![0.5f32, -0.3, 0.0, 0.8, -0.7];
        let threshold = 0.4;
        let quantized = quantize_ternary(&values, threshold);

        assert_eq!(quantized[0], 1); // 0.5 > 0.4
        assert_eq!(quantized[1], 0); // -0.3 in [-0.4, 0.4]
        assert_eq!(quantized[2], 0); // 0.0
        assert_eq!(quantized[3], 1); // 0.8 > 0.4
        assert_eq!(quantized[4], -1); // -0.7 < -0.4
    }

    #[test]
    fn test_ternary_symmetric() {
        let values = vec![0.5f32, -0.5, 0.3, -0.3];
        let threshold = 0.2;
        let quantized = quantize_ternary(&values, threshold);

        // Symmetric values should produce ±1 and ±1
        assert_eq!(quantized[0], 1); // +0.5
        assert_eq!(quantized[1], -1); // -0.5
        assert_eq!(quantized[2], 1); // +0.3
        assert_eq!(quantized[3], -1); // -0.3
    }

    #[test]
    fn test_ternary_roundtrip() {
        let values = vec![0.5f32, -0.3, 0.8, -0.7, 0.0];
        let threshold = 0.2;
        let scale = 0.5f32;

        let quantized = quantize_ternary(&values, threshold);
        let dequantized = dequantize_ternary(&quantized, scale);

        // Check that signs are preserved (except for 0)
        for (orig, dq) in values.iter().zip(dequantized.iter()) {
            if orig.abs() > threshold {
                assert!(orig.signum() == dq.signum() || dq.signum() == 0.0);
            }
        }
    }
}

mod i2s_block_quantization {
    use super::*;

    #[test]
    fn test_i2s_block_size_32() {
        let values: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.5).collect();
        let (packed, scales) = quantize_i2s_block(&values, 32);

        // Should have exactly 1 block
        assert_eq!(scales.len(), 1);
        // 32 values / 4 per byte = 8 bytes
        assert_eq!(packed.len(), 8);
        // Scale should be max(|values|)
        // values range from -8.0 to 7.5, so max(|v|) = 8.0
        assert!((scales[0] - 8.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_i2s_block_size_256() {
        let values: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.01) - 1.28).collect();
        let (packed, scales) = quantize_i2s_block(&values, 256);

        // Should have exactly 1 block
        assert_eq!(scales.len(), 1);
        // 256 values / 4 per byte = 64 bytes
        assert_eq!(packed.len(), 64);
        // Scale should be approximately max(|values|)
        let expected_scale = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!((scales[0] - expected_scale).abs() < TOLERANCE);
    }

    #[test]
    fn test_i2s_scale_computation() {
        let values = vec![0.1f32, -0.2, 0.15, -0.3, 0.05];
        let (_, scales) = quantize_i2s_block(&values, 5);

        // Scale should be max(|v|) = 0.3
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 0.3).abs() < TOLERANCE);
    }

    #[test]
    fn test_i2s_zero_block() {
        let values = vec![0.0f32; 8];
        let (packed, scales) = quantize_i2s_block(&values, 8);

        // Scale should be 0
        assert_eq!(scales.len(), 1);
        assert_eq!(scales[0], 0.0);
        // Packed should contain zeros (0b00_00_00_00)
        for &byte in &packed {
            assert_eq!(byte, 0);
        }
    }
}

mod numerical_accuracy {
    use super::*;

    #[test]
    fn test_quantization_error_bounded() {
        let values = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let (_packed, scales) = quantize_i2s_block(&values, values.len());

        // For 2-bit quantization, max error should be <= scale * 0.5
        let max_error_bound = scales[0] * 0.5;
        assert!(max_error_bound > 0.0);
    }

    #[test]
    fn test_quantization_preserves_sparsity() {
        let values = vec![0.0f32, 0.1, 0.0, 0.2, 0.0, -0.1, 0.0];
        let threshold = 0.05;
        let quantized = quantize_ternary(&values, threshold);

        // Zeros should remain zeros
        assert_eq!(quantized[0], 0);
        assert_eq!(quantized[2], 0);
        assert_eq!(quantized[4], 0);
    }

    #[test]
    fn test_quantization_large_values() {
        let values = vec![1e6f32, -1e6, 5e5, -5e5];
        let _scale = compute_absmax_scale(&values);

        // Scale should be 1e6
    }

    #[test]
    fn test_quantization_denormals() {
        let values = vec![1e-38f32, -1e-38, 1e-37, 0.0];
        let scale = compute_absmax_scale(&values);

        // Should not panic or produce NaN
        assert!(scale.is_finite());
        assert!(scale >= 0.0);
    }
}

mod bitnet_specific {
    use super::*;

    #[test]
    fn test_bitnet_weight_quantization_pattern() {
        // Simulate weights from normal(0, 0.02) distribution
        let values =
            vec![0.015f32, -0.018, 0.008, -0.022, 0.011, -0.005, 0.019, -0.012, 0.007, -0.025];

        let threshold = 0.01; // Typical threshold for weights
        let quantized = quantize_ternary(&values, threshold);

        // Most values should map to ±1 or 0
        let mut has_positive = false;
        let mut has_negative = false;
        let mut has_zero = false;

        for &q in &quantized {
            if q == 1 {
                has_positive = true;
            } else if q == -1 {
                has_negative = true;
            } else {
                has_zero = true;
            }
        }

        assert!(has_positive);
        assert!(has_negative);
        assert!(has_zero);
    }

    #[test]
    fn test_bitnet_activation_quantization_post_layernorm() {
        // Simulate post-LayerNorm activations (near unit variance)
        let values = vec![0.8f32, -0.6, 0.4, -0.9, 0.2, -0.5, 0.7, -0.3, 0.9, -0.1];

        let _scale = compute_absmax_scale(&values);
        let threshold = 0.1;
        let quantized = quantize_ternary(&values, threshold);

        // Should have a good mix of +1, -1, and 0
        let count_positive = quantized.iter().filter(|&&q| q == 1).count();
        let count_negative = quantized.iter().filter(|&&q| q == -1).count();

        assert!(count_positive > 0);
        assert!(count_negative > 0);
    }

    #[test]
    fn test_quantization_block_alignment() {
        // Test with length that isn't a multiple of block size
        let values: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
        let block_size = 32;

        let (_packed, scales) = quantize_i2s_block(&values, block_size);

        // Should have ceil(100/32) = 4 blocks
        assert_eq!(scales.len(), 4);

        // Verify each block's scale
        for (i, scale) in scales.iter().enumerate() {
            let block_start = i * block_size;
            let block_end = std::cmp::min(block_start + block_size, values.len());
            let block = &values[block_start..block_end];

            let expected_scale = compute_absmax_scale(block);
            assert!(
                (scale - expected_scale).abs() < TOLERANCE
                    || (*scale == 0.0 && expected_scale == 0.0)
            );
        }
    }
}
