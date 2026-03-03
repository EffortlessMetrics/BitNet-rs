//! Softmax utility functions.
//!
//! Numerically stable softmax implementations for different
//! precision modes and batch processing.

/// Numerically stable softmax in f32.
pub fn softmax_f32(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    // Find max for numerical stability
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f64;
    for val in logits.iter_mut() {
        *val = (*val - max).exp();
        sum += *val as f64;
    }

    if sum > 0.0 {
        let inv_sum = 1.0 / sum as f32;
        for val in logits.iter_mut() {
            *val *= inv_sum;
        }
    }
}

/// Softmax with f64 accumulation for higher precision.
pub fn softmax_f64_accum(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut exp_vals: Vec<f64> = logits.iter().map(|&v| ((v - max) as f64).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();

    if sum > 0.0 {
        for (i, val) in exp_vals.iter_mut().enumerate() {
            *val /= sum;
            logits[i] = *val as f32;
        }
    }
}

/// Log-softmax: log(softmax(x)).
pub fn log_softmax_f32(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let sum: f64 = logits.iter().map(|&v| ((v - max) as f64).exp()).sum();
    let log_sum = sum.ln() as f32;

    for val in logits.iter_mut() {
        *val = *val - max - log_sum;
    }
}

/// Softmax with temperature scaling.
pub fn softmax_with_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 || logits.is_empty() {
        return;
    }
    let inv_temp = 1.0 / temperature;
    for val in logits.iter_mut() {
        *val *= inv_temp;
    }
    softmax_f32(logits);
}

/// Batched softmax: apply softmax to each row of a 2D matrix.
pub fn softmax_batch(data: &mut [f32], batch_size: usize, seq_len: usize) {
    assert_eq!(data.len(), batch_size * seq_len);
    for b in 0..batch_size {
        let start = b * seq_len;
        let end = start + seq_len;
        softmax_f32(&mut data[start..end]);
    }
}

/// Check if a slice is a valid probability distribution.
pub fn is_valid_distribution(probs: &[f32], tolerance: f32) -> bool {
    if probs.is_empty() {
        return true;
    }
    // All non-negative
    if probs.iter().any(|&v| v < 0.0) {
        return false;
    }
    // Sums to ~1.0
    let sum: f64 = probs.iter().map(|&v| v as f64).sum();
    (sum - 1.0).abs() < tolerance as f64
}

/// Compute entropy of a probability distribution.
pub fn entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0f64;
    for &p in probs {
        if p > 0.0 {
            h -= (p as f64) * (p as f64).ln();
        }
    }
    h as f32
}

/// Find top-k indices and values.
pub fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax_f32(&mut logits);
        assert!(is_valid_distribution(&logits, 1e-5));
        assert!(logits[2] > logits[1] && logits[1] > logits[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let mut logits = vec![0.0; 4];
        softmax_f32(&mut logits);
        for &v in &logits {
            assert!((v - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut logits = vec![1000.0, 1001.0, 1002.0];
        softmax_f32(&mut logits);
        assert!(is_valid_distribution(&logits, 1e-5));
    }

    #[test]
    fn test_softmax_f64_accum() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax_f64_accum(&mut logits);
        assert!(is_valid_distribution(&logits, 1e-6));
    }

    #[test]
    fn test_log_softmax() {
        let mut logits = vec![1.0, 2.0, 3.0];
        log_softmax_f32(&mut logits);
        // All values should be <= 0
        for &v in &logits {
            assert!(v <= 0.0);
        }
    }

    #[test]
    fn test_temperature_scaling() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax_with_temperature(&mut logits, 0.5);
        assert!(is_valid_distribution(&logits, 1e-5));
        // Low temperature → more peaked distribution
        assert!(logits[2] > 0.8);
    }

    #[test]
    fn test_temperature_high() {
        let mut logits = vec![1.0, 2.0, 3.0];
        softmax_with_temperature(&mut logits, 10.0);
        assert!(is_valid_distribution(&logits, 1e-5));
        // High temperature → more uniform
        assert!(logits[0] > 0.2);
    }

    #[test]
    fn test_batch_softmax() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        softmax_batch(&mut data, 2, 3);
        assert!(is_valid_distribution(&data[0..3], 1e-5));
        assert!(is_valid_distribution(&data[3..6], 1e-5));
    }

    #[test]
    fn test_entropy() {
        let uniform = vec![0.25f32; 4];
        let peaked = vec![0.97f32, 0.01, 0.01, 0.01];
        assert!(entropy(&uniform) > entropy(&peaked));
    }

    #[test]
    fn test_top_k() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let top = top_k(&logits, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 3); // index of 0.9
        assert_eq!(top[1].0, 1); // index of 0.5
    }

    #[test]
    fn test_empty_softmax() {
        let mut empty: Vec<f32> = vec![];
        softmax_f32(&mut empty);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_valid_distribution() {
        assert!(is_valid_distribution(&[0.5, 0.3, 0.2], 0.01));
        assert!(!is_valid_distribution(&[0.5, 0.3, 0.3], 0.01)); // sums to 1.1
        assert!(!is_valid_distribution(&[-0.1, 1.1], 0.01)); // negative
    }
}
