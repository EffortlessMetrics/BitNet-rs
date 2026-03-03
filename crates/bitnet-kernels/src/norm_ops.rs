//! Normalization operations for transformer inference.
//!
//! RMSNorm, LayerNorm, GroupNorm variants with configurable
//! epsilon, weight/bias, and in-place computation.

/// RMSNorm: x / rms(x) * weight.
/// Used by Phi-4, LLaMA, Mistral. Accumulates in f64 for stability.
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // Accumulate in f64 for numerical stability
    let sum_sq: f64 = input.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let rms = ((sum_sq / n as f64) + eps as f64).sqrt();
    let inv_rms = 1.0 / rms;

    input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| ((x as f64) * inv_rms * (w as f64)) as f32)
        .collect()
}

/// RMSNorm in-place.
pub fn rms_norm_inplace(input: &mut [f32], weight: &[f32], eps: f32) {
    let n = input.len();
    if n == 0 {
        return;
    }

    let sum_sq: f64 = input.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let rms = ((sum_sq / n as f64) + eps as f64).sqrt();
    let inv_rms = 1.0 / rms;

    for (x, &w) in input.iter_mut().zip(weight.iter()) {
        *x = ((*x as f64) * inv_rms * (w as f64)) as f32;
    }
}

/// LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias.
pub fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mean: f64 = input.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let var: f64 = input
        .iter()
        .map(|&x| {
            let d = (x as f64) - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let inv_std = 1.0 / (var + eps as f64).sqrt();

    input
        .iter()
        .zip(weight.iter().zip(bias.iter()))
        .map(|(&x, (&w, &b))| {
            let norm = ((x as f64) - mean) * inv_std;
            (norm * (w as f64) + (b as f64)) as f32
        })
        .collect()
}

/// LayerNorm without bias (weight only).
pub fn layer_norm_no_bias(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    let mean: f64 = input.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let var: f64 = input
        .iter()
        .map(|&x| {
            let d = (x as f64) - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let inv_std = 1.0 / (var + eps as f64).sqrt();

    input
        .iter()
        .zip(weight.iter())
        .map(|(&x, &w)| (((x as f64) - mean) * inv_std * (w as f64)) as f32)
        .collect()
}

/// Compute RMS of a vector (for diagnostics).
pub fn compute_rms(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = input.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (sum_sq / input.len() as f64).sqrt() as f32
}

/// Compute mean and variance (for diagnostics).
pub fn compute_mean_var(input: &[f32]) -> (f32, f32) {
    if input.is_empty() {
        return (0.0, 0.0);
    }
    let n = input.len() as f64;
    let mean: f64 = input.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var: f64 = input
        .iter()
        .map(|&x| {
            let d = (x as f64) - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean as f32, var as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_unit_weight() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let output = rms_norm(&input, &weight, 1e-5);
        // After RMSNorm, values should be rescaled
        let rms = compute_rms(&input);
        assert!((output[0] - input[0] / rms).abs() < 0.01);
    }

    #[test]
    fn test_rms_norm_inplace() {
        let mut input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let expected = rms_norm(&input, &weight, 1e-5);
        rms_norm_inplace(&mut input, &weight, 1e-5);
        for (a, b) in input.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_layer_norm_zero_mean() {
        let input = vec![1.0, -1.0, 1.0, -1.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let output = layer_norm(&input, &weight, &bias, 1e-5);
        // Mean is 0, so output should be input / std
        assert!((output[0] - output[2]).abs() < 1e-6);
    }

    #[test]
    fn test_layer_norm_with_bias() {
        let input = vec![0.0, 0.0, 0.0];
        let weight = vec![1.0; 3];
        let bias = vec![5.0; 3];
        let output = layer_norm(&input, &weight, &bias, 1e-5);
        for v in &output {
            assert!((v - 5.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_layer_norm_no_bias() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0; 3];
        let output = layer_norm_no_bias(&input, &weight, 1e-5);
        // Output should be zero-mean, unit variance (approx)
        let (mean, _) = compute_mean_var(&output);
        assert!(mean.abs() < 0.01);
    }

    #[test]
    fn test_compute_rms() {
        let input = vec![3.0, 4.0];
        let rms = compute_rms(&input);
        // sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        assert!((rms - 3.536).abs() < 0.01);
    }

    #[test]
    fn test_compute_mean_var() {
        let input = vec![2.0, 4.0, 6.0];
        let (mean, var) = compute_mean_var(&input);
        assert!((mean - 4.0).abs() < 1e-5);
        // var = ((2-4)^2 + (4-4)^2 + (6-4)^2)/3 = 8/3 ≈ 2.667
        assert!((var - 2.667).abs() < 0.01);
    }

    #[test]
    fn test_empty_inputs() {
        assert!(rms_norm(&[], &[], 1e-5).is_empty());
        assert!(layer_norm(&[], &[], &[], 1e-5).is_empty());
        assert_eq!(compute_rms(&[]), 0.0);
    }

    #[test]
    fn test_rms_norm_with_weight() {
        let input = vec![1.0, 1.0, 1.0, 1.0];
        let weight = vec![2.0; 4];
        let output = rms_norm(&input, &weight, 1e-5);
        // rms(1,1,1,1)=1, so output ≈ 2.0
        for v in &output {
            assert!((v - 2.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_f64_stability() {
        // Large values that could overflow in f32 squared
        let input = vec![1e4, 1e4, 1e4, 1e4];
        let weight = vec![1.0; 4];
        let output = rms_norm(&input, &weight, 1e-5);
        // Should still produce ~1.0 since all values are equal
        for v in &output {
            assert!((v - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_rms_norm_eps_effect() {
        let input = vec![0.0, 0.0, 0.0];
        let weight = vec![1.0; 3];
        let output = rms_norm(&input, &weight, 1.0);
        // rms ≈ sqrt(eps) = 1.0, so output ≈ 0
        for v in &output {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_rms_norm_preserves_direction() {
        let input = vec![3.0, 4.0];
        let weight = vec![1.0; 2];
        let output = rms_norm(&input, &weight, 1e-8);
        // Ratio should be preserved
        let ratio_in = input[0] / input[1];
        let ratio_out = output[0] / output[1];
        assert!((ratio_in - ratio_out).abs() < 1e-5);
    }
}
