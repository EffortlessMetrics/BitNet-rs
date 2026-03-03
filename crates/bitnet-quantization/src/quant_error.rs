//! Quantization error estimation.
//!
//! Measure and analyze the error introduced by quantization:
//! MSE, RMSE, SNR, max absolute error, and per-channel stats.

/// Error metrics for a single tensor.
#[derive(Debug, Clone)]
pub struct QuantError {
    pub mse: f64,
    pub rmse: f64,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub signal_noise_ratio_db: f64,
    pub num_elements: usize,
}

/// Compute quantization error between original and quantized values.
pub fn compute_error(original: &[f32], quantized: &[f32]) -> Option<QuantError> {
    if original.len() != quantized.len() || original.is_empty() {
        return None;
    }

    let n = original.len();
    let mut sum_sq_error = 0.0f64;
    let mut sum_abs_error = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut sum_signal_sq = 0.0f64;

    for (o, q) in original.iter().zip(quantized.iter()) {
        let diff = (*o as f64) - (*q as f64);
        sum_sq_error += diff * diff;
        sum_abs_error += diff.abs();
        max_abs = max_abs.max(diff.abs());
        sum_signal_sq += (*o as f64) * (*o as f64);
    }

    let mse = sum_sq_error / n as f64;
    let rmse = mse.sqrt();
    let mean_abs = sum_abs_error / n as f64;
    let snr_db = if sum_sq_error > 0.0 {
        10.0 * (sum_signal_sq / sum_sq_error).log10()
    } else {
        f64::INFINITY
    };

    Some(QuantError {
        mse,
        rmse,
        max_abs_error: max_abs,
        mean_abs_error: mean_abs,
        signal_noise_ratio_db: snr_db,
        num_elements: n,
    })
}

/// Per-channel error analysis.
#[derive(Debug, Clone)]
pub struct ChannelError {
    pub channel_idx: usize,
    pub error: QuantError,
}

/// Analyze error per channel (assuming channels are contiguous rows).
pub fn per_channel_error(
    original: &[f32],
    quantized: &[f32],
    num_channels: usize,
) -> Option<Vec<ChannelError>> {
    if original.len() != quantized.len() || original.is_empty() {
        return None;
    }
    let channel_size = original.len() / num_channels;
    if channel_size == 0 {
        return None;
    }

    let mut results = Vec::with_capacity(num_channels);
    for ch in 0..num_channels {
        let start = ch * channel_size;
        let end = start + channel_size;
        if end > original.len() {
            break;
        }
        if let Some(err) = compute_error(&original[start..end], &quantized[start..end]) {
            results.push(ChannelError { channel_idx: ch, error: err });
        }
    }
    Some(results)
}

/// Error budget check.
#[derive(Debug, Clone, Copy)]
pub struct ErrorBudget {
    pub max_mse: f64,
    pub max_rmse: f64,
    pub max_abs_error: f64,
    pub min_snr_db: f64,
}

impl ErrorBudget {
    pub fn strict() -> Self {
        Self { max_mse: 1e-6, max_rmse: 1e-3, max_abs_error: 1e-2, min_snr_db: 60.0 }
    }

    pub fn relaxed() -> Self {
        Self { max_mse: 1e-2, max_rmse: 0.1, max_abs_error: 1.0, min_snr_db: 20.0 }
    }

    pub fn check(&self, error: &QuantError) -> bool {
        error.mse <= self.max_mse
            && error.rmse <= self.max_rmse
            && error.max_abs_error <= self.max_abs_error
            && error.signal_noise_ratio_db >= self.min_snr_db
    }
}

/// Find the worst channel.
pub fn worst_channel(channels: &[ChannelError]) -> Option<&ChannelError> {
    channels
        .iter()
        .max_by(|a, b| a.error.mse.partial_cmp(&b.error.mse).unwrap_or(std::cmp::Ordering::Equal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_error() {
        let data = vec![1.0, 2.0, 3.0];
        let err = compute_error(&data, &data).unwrap();
        assert_eq!(err.mse, 0.0);
        assert_eq!(err.rmse, 0.0);
        assert_eq!(err.max_abs_error, 0.0);
        assert!(err.signal_noise_ratio_db.is_infinite());
    }

    #[test]
    fn test_known_error() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let quant = vec![1.1, 2.1, 3.1, 4.1];
        let err = compute_error(&orig, &quant).unwrap();
        assert!((err.mse - 0.01).abs() < 1e-6);
        assert!((err.rmse - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_max_abs_error() {
        let orig = vec![0.0, 0.0, 0.0];
        let quant = vec![0.5, 0.1, 0.3];
        let err = compute_error(&orig, &quant).unwrap();
        assert!((err.max_abs_error - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mismatched_lengths() {
        assert!(compute_error(&[1.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_empty() {
        assert!(compute_error(&[], &[]).is_none());
    }

    #[test]
    fn test_per_channel() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let quant = vec![1.0, 2.0, 3.5, 4.5];
        let ch = per_channel_error(&orig, &quant, 2).unwrap();
        assert_eq!(ch.len(), 2);
        assert_eq!(ch[0].error.mse, 0.0); // first channel exact
        assert!(ch[1].error.mse > 0.0); // second channel has error
    }

    #[test]
    fn test_budget_strict() {
        let err = compute_error(&[1.0], &[1.0]).unwrap();
        assert!(ErrorBudget::strict().check(&err));
    }

    #[test]
    fn test_budget_fail() {
        let err = compute_error(&[1.0], &[2.0]).unwrap();
        assert!(!ErrorBudget::strict().check(&err));
    }

    #[test]
    fn test_budget_relaxed() {
        let err = compute_error(&[1.0, 2.0], &[1.05, 2.05]).unwrap();
        assert!(ErrorBudget::relaxed().check(&err));
    }

    #[test]
    fn test_snr_positive() {
        let orig = vec![1.0, 2.0, 3.0];
        let quant = vec![1.01, 2.01, 3.01];
        let err = compute_error(&orig, &quant).unwrap();
        assert!(err.signal_noise_ratio_db > 30.0);
    }

    #[test]
    fn test_worst_channel() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let quant = vec![1.0, 2.0, 5.0, 6.0];
        let ch = per_channel_error(&orig, &quant, 2).unwrap();
        let worst = worst_channel(&ch).unwrap();
        assert_eq!(worst.channel_idx, 1);
    }

    #[test]
    fn test_mean_abs_error() {
        let orig = vec![0.0, 0.0];
        let quant = vec![1.0, 3.0];
        let err = compute_error(&orig, &quant).unwrap();
        assert!((err.mean_abs_error - 2.0).abs() < 1e-6);
    }
}
