//! CPU reference tests for the RMSNorm OpenCL kernel logic.
//!
//! These tests verify the mathematical correctness of the RMSNorm algorithm
//! using a CPU reference implementation that mirrors the OpenCL kernel logic.

/// CPU reference implementation of RMS normalization (mirrors rmsnorm.cl).
fn rms_norm_reference(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) {
    for row in 0..rows {
        let offset = row * hidden_dim;

        // Sum of squares
        let sum_sq: f32 = (0..hidden_dim)
            .map(|i| {
                let v = input[offset + i];
                v * v
            })
            .sum();

        let rms = 1.0 / (sum_sq / hidden_dim as f32 + eps).sqrt();

        for i in 0..hidden_dim {
            output[offset + i] = input[offset + i] * rms * weight[i];
        }
    }
}

/// CPU reference for fused RMSNorm + residual.
fn rms_norm_residual_reference(
    input: &[f32],
    residual: &[f32],
    weight: &[f32],
    output: &mut [f32],
    rows: usize,
    hidden_dim: usize,
    eps: f32,
) {
    for row in 0..rows {
        let offset = row * hidden_dim;

        let sum_sq: f32 = (0..hidden_dim)
            .map(|i| {
                let v = input[offset + i] + residual[offset + i];
                v * v
            })
            .sum();

        let rms = 1.0 / (sum_sq / hidden_dim as f32 + eps).sqrt();

        for i in 0..hidden_dim {
            let v = input[offset + i] + residual[offset + i];
            output[offset + i] = v * rms * weight[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_single_row_unit_weights() {
        let hidden_dim = 4;
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32; hidden_dim];
        let mut output = vec![0.0f32; hidden_dim];

        rms_norm_reference(&input, &weight, &mut output, 1, hidden_dim, 1e-5);

        // RMS = sqrt(mean([1,4,9,16])) = sqrt(30/4) = sqrt(7.5) ≈ 2.7386
        // Each element: x / rms
        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + 1e-5).sqrt();
        for i in 0..hidden_dim {
            let expected = input[i] / rms;
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "mismatch at {}: got {} expected {}",
                i,
                output[i],
                expected,
            );
        }
    }

    #[test]
    fn test_rmsnorm_multi_row() {
        let hidden_dim = 3;
        let rows = 2;
        let input = vec![1.0f32, 0.0, -1.0, 2.0, 2.0, 2.0];
        let weight = vec![1.0f32; hidden_dim];
        let mut output = vec![0.0f32; rows * hidden_dim];

        rms_norm_reference(&input, &weight, &mut output, rows, hidden_dim, 1e-5);

        // Row 0: RMS = sqrt((1+0+1)/3) = sqrt(2/3)
        // Row 1: RMS = sqrt((4+4+4)/3) = sqrt(4) = 2
        let rms0 = (2.0f32 / 3.0 + 1e-5).sqrt();
        let rms1 = (4.0f32 + 1e-5).sqrt();

        assert!((output[0] - 1.0 / rms0).abs() < 1e-5);
        assert!((output[1]).abs() < 1e-5); // 0 / anything = 0
        assert!((output[2] - (-1.0 / rms0)).abs() < 1e-5);
        assert!((output[3] - 2.0 / rms1).abs() < 1e-5);
    }

    #[test]
    fn test_rmsnorm_with_nontrivial_weights() {
        let hidden_dim = 4;
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![0.5f32, 1.0, 1.5, 2.0];
        let mut output = vec![0.0f32; hidden_dim];

        rms_norm_reference(&input, &weight, &mut output, 1, hidden_dim, 1e-5);

        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + 1e-5).sqrt();
        for i in 0..hidden_dim {
            let expected = input[i] / rms * weight[i];
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "mismatch at {}: got {} expected {}",
                i,
                output[i],
                expected,
            );
        }
    }

    #[test]
    fn test_rmsnorm_residual_basic() {
        let hidden_dim = 4;
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let residual = vec![0.5f32, -0.5, 0.5, -0.5];
        let weight = vec![1.0f32; hidden_dim];
        let mut output = vec![0.0f32; hidden_dim];

        rms_norm_residual_reference(
            &input,
            &residual,
            &weight,
            &mut output,
            1,
            hidden_dim,
            1e-5,
        );

        // After add: [1.5, 1.5, 3.5, 3.5]
        let combined: Vec<f32> = input
            .iter()
            .zip(residual.iter())
            .map(|(a, b)| a + b)
            .collect();
        let sum_sq: f32 = combined.iter().map(|x| x * x).sum();
        let rms = (sum_sq / hidden_dim as f32 + 1e-5).sqrt();
        for i in 0..hidden_dim {
            let expected = combined[i] / rms;
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "mismatch at {}: got {} expected {}",
                i,
                output[i],
                expected,
            );
        }
    }

    #[test]
    fn test_rmsnorm_eps_prevents_division_by_zero() {
        let hidden_dim = 4;
        let input = vec![0.0f32; hidden_dim];
        let weight = vec![1.0f32; hidden_dim];
        let mut output = vec![0.0f32; hidden_dim];

        rms_norm_reference(&input, &weight, &mut output, 1, hidden_dim, 1e-5);

        // All zeros in, all zeros out (0 * anything = 0)
        for &val in &output {
            assert!(val.is_finite(), "output should be finite, got {}", val);
            assert!((val).abs() < 1e-3, "expected ~0, got {}", val);
        }
    }

    #[test]
    fn test_rmsnorm_output_norm_is_approximately_one() {
        // After RMSNorm with unit weights, the RMS of the output should be ~1
        let hidden_dim = 128;
        let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.1) - 6.4).collect();
        let weight = vec![1.0f32; hidden_dim];
        let mut output = vec![0.0f32; hidden_dim];

        rms_norm_reference(&input, &weight, &mut output, 1, hidden_dim, 1e-5);

        let output_rms: f32 = output.iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32;
        let output_rms = output_rms.sqrt();

        assert!(
            (output_rms - 1.0).abs() < 0.01,
            "output RMS should be ~1.0, got {}",
            output_rms,
        );
    }
}
