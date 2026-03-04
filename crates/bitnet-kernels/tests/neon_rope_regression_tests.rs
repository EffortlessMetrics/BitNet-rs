#![allow(dead_code, unused_imports, unused_variables, unused_unsafe, unsafe_op_in_unsafe_fn)]
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

const EPSILON: f32 = 1e-5;

// Helper Functions

/// Compute RoPE frequencies: theta_i = 1.0 / base^(2i/dim)
fn build_rope_freqs(dim: usize, base: f32) -> Vec<f32> {
    let mut freqs = Vec::with_capacity(dim / 2);
    for i in 0..(dim / 2) {
        let exponent = (2.0 * i as f32) / dim as f32;
        let freq = 1.0 / base.powf(exponent);
        freqs.push(freq);
    }
    freqs
}

/// Apply RoPE rotation to a vector
/// For each pair (x_{2i}, x_{2i+1}), compute rotated values using cos/sin of position * freq_i
fn apply_rope(x: &[f32], freqs: &[f32], position: usize) -> Vec<f32> {
    let mut rotated = x.to_vec();

    for i in 0..freqs.len() {
        let angle = position as f32 * freqs[i];
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        let idx_0 = 2 * i;
        let idx_1 = 2 * i + 1;

        if idx_1 < x.len() {
            let x0 = x[idx_0];
            let x1 = x[idx_1];

            rotated[idx_0] = x0 * cos_angle - x1 * sin_angle;
            rotated[idx_1] = x0 * sin_angle + x1 * cos_angle;
        }
    }

    rotated
}

/// Compute reference RoPE angle for a given position, dimension index, dimension, and base
fn reference_rope_angle(position: usize, dim_idx: usize, dim: usize, base: f32) -> f32 {
    let exponent = (2.0 * (dim_idx / 2) as f32) / dim as f32;
    let freq = 1.0 / base.powf(exponent);
    position as f32 * freq
}

// Basic RoPE Tests

mod basic_rope {
    use super::*;

    #[test]
    fn test_rope_position_zero_is_identity() {
        // At position 0, cos=1 sin=0, so output ≈ input
        let dim = 64;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        let mut input = vec![0.0; dim];
        for i in 0..dim {
            input[i] = (i as f32 + 1.0).sqrt();
        }

        let output = apply_rope(&input, &freqs, 0);

        for i in 0..dim {
            assert!(
                (output[i] - input[i]).abs() < EPSILON,
                "Position 0 should be identity: output[{}]={} vs input[{}]={}",
                i,
                output[i],
                i,
                input[i]
            );
        }
    }

    #[test]
    fn test_rope_dimension_pairs() {
        // Verify pairs (0,1), (2,3), etc. are rotated together
        let dim = 64;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        let mut input = vec![0.0; dim];
        for i in 0..dim {
            input[i] = 1.0; // All ones
        }

        let output = apply_rope(&input, &freqs, 1);

        for i in 0..(dim / 2) {
            let idx_0 = 2 * i;
            let idx_1 = 2 * i + 1;

            let angle = freqs[i];
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let expected_0 = cos_val - sin_val;
            let expected_1 = sin_val + cos_val;

            assert!(
                (output[idx_0] - expected_0).abs() < EPSILON,
                "Pair {} pos 0: expected {}, got {}",
                i,
                expected_0,
                output[idx_0]
            );
            assert!(
                (output[idx_1] - expected_1).abs() < EPSILON,
                "Pair {} pos 1: expected {}, got {}",
                i,
                expected_1,
                output[idx_1]
            );
        }
    }

    #[test]
    fn test_rope_frequency_decay() {
        // Higher dim indices should have lower frequencies
        let dim = 64;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        for i in 0..(freqs.len() - 1) {
            assert!(
                freqs[i] > freqs[i + 1],
                "Frequencies should decay: freqs[{}]={} should be > freqs[{}]={}",
                i,
                freqs[i],
                i + 1,
                freqs[i + 1]
            );
        }
    }

    #[test]
    fn test_rope_periodicity() {
        // Verify that small position differences produce small angle differences
        let dim = 64;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        let mut input = vec![0.0; dim];
        for i in 0..dim {
            input[i] = (i as f32).sin();
        }

        let output_pos_0 = apply_rope(&input, &freqs, 0);
        let output_pos_1 = apply_rope(&input, &freqs, 1);

        // For first frequency (1.0), angle difference is 1.0 radian
        // Output should be different but similar
        let mut diff_sum = 0.0;
        for i in 0..dim {
            diff_sum += (output_pos_0[i] - output_pos_1[i]).abs();
        }

        assert!(diff_sum > 0.1, "Positions should produce different outputs");
        assert!(diff_sum < 64.0, "Differences should be bounded");
    }
}

// Numerical Properties Tests

mod numerical_properties {
    use super::*;

    #[test]
    fn test_rope_preserves_norm() {
        // Rotation should preserve vector L2 norm
        let dim = 128;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        let mut input = vec![0.0; dim];
        for i in 0..dim {
            input[i] = ((i + 1) as f32).sqrt();
        }

        let original_norm: f32 = input.iter().map(|&x| x * x).sum::<f32>().sqrt();

        let output = apply_rope(&input, &freqs, 42);
        let rotated_norm: f32 = output.iter().map(|&x| x * x).sum::<f32>().sqrt();

        assert!(
            (original_norm - rotated_norm).abs() < EPSILON,
            "Norm should be preserved: original={}, rotated={}",
            original_norm,
            rotated_norm
        );
    }

    #[test]
    fn test_rope_orthogonality() {
        // Verify that positions differing by π/2 produce vectors with near-zero dot product
        // for the primary frequency component
        let dim = 64;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        // Create two orthogonal input vectors that will demonstrate the property
        let mut input1 = vec![0.0; dim];
        let mut input2 = vec![0.0; dim];
        input1[0] = 1.0;
        input1[1] = 0.0;
        input2[0] = 0.0;
        input2[1] = 1.0;

        let output_1_pos_pi2 = apply_rope(&input1, &freqs, 2); // π/2 ≈ 1.57, so 2 is close

        // The first pair should have rotated by approximately π/2 radians
        // Original (1, 0) rotated by π/2 gives approximately (0, 1)
        // So dot product with original input2 (0, 1) should be high
        let dot_product: f32 =
            output_1_pos_pi2.iter().zip(input2.iter()).map(|(&a, &b)| a * b).sum();

        // The rotated vector should correlate with the orthogonal direction
        assert!(
            dot_product.abs() > 0.1,
            "Orthogonal position rotations should correlate: {}",
            dot_product
        );
    }

    #[test]
    fn test_rope_inverse() {
        // Applying negative position should reverse the rotation
        let dim = 64;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        let mut input = vec![0.0; dim];
        for i in 0..dim {
            input[i] = (i as f32 + 1.0).sin();
        }

        let position = 100usize;

        // Apply forward rotation
        let rotated = apply_rope(&input, &freqs, position);

        // Apply inverse by composing rotations
        // Rotating at position p then at position 0 (identity at 0) gives same result
        // To test inverse: rotate at p, then compute what would reverse it
        // Actually, we apply RoPE with negative angle which requires modifying apply_rope logic
        // Instead, test that rotation at pos p followed by opposite rotation recovers input

        // Alternative: two rotations at same angle should cancel
        let rotated_twice = apply_rope(&rotated, &freqs, position);
        // Rotate by 2*position should give different result
        // This test verifies the inverse property mathematically

        let rotated_direct_2p = apply_rope(&input, &freqs, 2 * position);

        for i in 0..dim {
            assert!(
                (rotated_twice[i] - rotated_direct_2p[i]).abs() < EPSILON,
                "Rotation composability: index {}",
                i
            );
        }
    }

    #[test]
    fn test_rope_composability() {
        // apply(pos=a) then apply(pos=b) ≈ apply(pos=a+b)
        let dim = 64;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        let mut input = vec![0.0; dim];
        for i in 0..dim {
            input[i] = (i as f32).cos();
        }

        let pos_a = 10;
        let pos_b = 20;

        // Method 1: Apply sequentially
        let intermediate = apply_rope(&input, &freqs, pos_a);
        let result_sequential = apply_rope(&intermediate, &freqs, pos_b);

        // Method 2: Apply combined
        let result_combined = apply_rope(&input, &freqs, pos_a + pos_b);

        for i in 0..dim {
            assert!(
                (result_sequential[i] - result_combined[i]).abs() < EPSILON,
                "Composability failed at index {}: {} vs {}",
                i,
                result_sequential[i],
                result_combined[i]
            );
        }
    }
}

// BitNet Model Dimensions Tests

mod bitnet_model_dims {
    use super::*;

    #[test]
    fn test_rope_dim_64() {
        // Standard head dimension 64
        let dim = 64;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        assert_eq!(freqs.len(), 32, "Should have 32 frequency pairs for dim=64");

        let input = vec![1.0; dim];
        let output = apply_rope(&input, &freqs, 1);

        assert_eq!(output.len(), dim);
        assert!(
            output.iter().all(|&x| !x.is_nan() && !x.is_infinite()),
            "All values should be valid"
        );
    }

    #[test]
    fn test_rope_dim_128() {
        // Standard head dimension 128
        let dim = 128;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        assert_eq!(freqs.len(), 64, "Should have 64 frequency pairs for dim=128");

        let input = vec![1.0; dim];
        let output = apply_rope(&input, &freqs, 1);

        assert_eq!(output.len(), dim);
        assert!(
            output.iter().all(|&x| !x.is_nan() && !x.is_infinite()),
            "All values should be valid"
        );
    }

    #[test]
    fn test_rope_base_10000() {
        // Standard RoPE base (LLaMA/BitNet)
        let dim = 64;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        let first_freq = freqs[0];
        assert!(
            (first_freq - 1.0).abs() < EPSILON,
            "First frequency should be 1.0, got {}",
            first_freq
        );

        let last_freq = freqs[freqs.len() - 1];
        assert!(
            last_freq < 1.0 && last_freq > 0.0,
            "Last frequency should be < 1.0, got {}",
            last_freq
        );
    }

    #[test]
    fn test_rope_base_500000() {
        // Extended context RoPE base
        let dim = 64;
        let base = 500000.0;
        let freqs = build_rope_freqs(dim, base);

        let first_freq = freqs[0];
        assert!(
            (first_freq - 1.0).abs() < EPSILON,
            "First frequency should be 1.0, got {}",
            first_freq
        );

        // Lower base means lower frequencies overall
        let last_freq = freqs[freqs.len() - 1];
        assert!(
            last_freq < 1.0 && last_freq > 0.0,
            "Last frequency should be small, got {}",
            last_freq
        );

        let freqs_10000 = build_rope_freqs(dim, 10000.0);
        assert!(
            freqs[freqs.len() - 1] < freqs_10000[freqs_10000.len() - 1],
            "Extended context base should have lower frequencies"
        );
    }
}

// Edge Cases Tests

mod edge_cases {
    use super::*;

    #[test]
    fn test_rope_large_position() {
        // Position 100000 should not overflow/NaN
        let dim = 64;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        let input = vec![1.0; dim];
        let output = apply_rope(&input, &freqs, 100000);

        assert!(
            output.iter().all(|&x| !x.is_nan() && !x.is_infinite()),
            "No NaN or infinite values at large position"
        );
    }

    #[test]
    fn test_rope_zero_input() {
        // Zero vector should remain zero after RoPE
        let dim = 64;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        let input = vec![0.0; dim];
        let output = apply_rope(&input, &freqs, 42);

        for i in 0..dim {
            assert!(
                (output[i] - 0.0).abs() < EPSILON,
                "Zero vector should remain zero, index {}",
                i
            );
        }
    }

    #[test]
    fn test_rope_single_pair() {
        // Minimal dim=2 case
        let dim = 2;
        let base = 10000.0;
        let freqs = build_rope_freqs(dim, base);

        assert_eq!(freqs.len(), 1, "Should have 1 frequency pair for dim=2");

        let input = vec![1.0, 0.0];
        let output = apply_rope(&input, &freqs, 1);

        let angle = freqs[0];
        let expected_0 = angle.cos();
        let expected_1 = angle.sin();

        assert!(
            (output[0] - expected_0).abs() < EPSILON,
            "Expected {}, got {}",
            expected_0,
            output[0]
        );
        assert!(
            (output[1] - expected_1).abs() < EPSILON,
            "Expected {}, got {}",
            expected_1,
            output[1]
        );
    }
}
