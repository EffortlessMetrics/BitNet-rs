//! Per-layer quantization quality metrics.
//!
//! Measure quantization error (MSE, SNR, max error),
//! distribution analysis, and quality reports.

/// Quantization error metrics between original and quantized values.
#[derive(Debug, Clone)]
pub struct QuantError {
    pub mse: f64,
    pub rmse: f64,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub snr_db: f64,
    pub num_elements: usize,
}

/// Compute quantization error between original and reconstructed values.
pub fn compute_error(original: &[f32], reconstructed: &[f32]) -> QuantError {
    let n = original.len().min(reconstructed.len());
    if n == 0 {
        return QuantError {
            mse: 0.0,
            rmse: 0.0,
            max_abs_error: 0.0,
            mean_abs_error: 0.0,
            snr_db: 0.0,
            num_elements: 0,
        };
    }

    let mut sum_sq_err = 0.0f64;
    let mut sum_abs_err = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut signal_power = 0.0f64;

    for i in 0..n {
        let err = (original[i] - reconstructed[i]) as f64;
        let abs_err = err.abs();
        sum_sq_err += err * err;
        sum_abs_err += abs_err;
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        signal_power += (original[i] as f64) * (original[i] as f64);
    }

    let mse = sum_sq_err / n as f64;
    let snr_db =
        if sum_sq_err > 0.0 { 10.0 * (signal_power / sum_sq_err).log10() } else { f64::INFINITY };

    QuantError {
        mse,
        rmse: mse.sqrt(),
        max_abs_error: max_abs,
        mean_abs_error: sum_abs_err / n as f64,
        snr_db,
        num_elements: n,
    }
}

/// Distribution bucket for value analysis.
#[derive(Debug, Clone)]
pub struct DistributionBucket {
    pub lower: f32,
    pub upper: f32,
    pub count: usize,
}

/// Analyze value distribution with fixed buckets.
pub fn value_distribution(values: &[f32], num_buckets: usize) -> Vec<DistributionBucket> {
    if values.is_empty() || num_buckets == 0 {
        return Vec::new();
    }

    let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if (max_val - min_val).abs() < f32::EPSILON {
        return vec![DistributionBucket { lower: min_val, upper: max_val, count: values.len() }];
    }

    let width = (max_val - min_val) / num_buckets as f32;
    let mut buckets: Vec<DistributionBucket> = (0..num_buckets)
        .map(|i| DistributionBucket {
            lower: min_val + i as f32 * width,
            upper: min_val + (i + 1) as f32 * width,
            count: 0,
        })
        .collect();

    for &v in values {
        let idx = ((v - min_val) / width) as usize;
        let idx = idx.min(num_buckets - 1);
        buckets[idx].count += 1;
    }

    buckets
}

/// Sparsity: fraction of values that are exactly zero.
pub fn sparsity(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let zeros = values.iter().filter(|&&v| v == 0.0).count();
    zeros as f64 / values.len() as f64
}

/// Per-layer quality report.
#[derive(Debug)]
pub struct LayerQualityReport {
    pub layer_name: String,
    pub error: QuantError,
    pub sparsity: f64,
    pub original_range: (f32, f32),
    pub reconstructed_range: (f32, f32),
}

/// Generate a quality report for a layer.
pub fn layer_report(name: &str, original: &[f32], reconstructed: &[f32]) -> LayerQualityReport {
    let error = compute_error(original, reconstructed);
    let sp = sparsity(reconstructed);
    let orig_min = original.iter().copied().fold(f32::INFINITY, f32::min);
    let orig_max = original.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let recon_min = reconstructed.iter().copied().fold(f32::INFINITY, f32::min);
    let recon_max = reconstructed.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    LayerQualityReport {
        layer_name: name.to_string(),
        error,
        sparsity: sp,
        original_range: (orig_min, orig_max),
        reconstructed_range: (recon_min, recon_max),
    }
}

impl LayerQualityReport {
    pub fn summary(&self) -> String {
        format!(
            "{}: MSE={:.6}, SNR={:.1}dB, sparsity={:.1}%",
            self.layer_name,
            self.error.mse,
            self.error.snr_db,
            self.sparsity * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_reconstruction() {
        let orig = vec![1.0, 2.0, 3.0];
        let err = compute_error(&orig, &orig);
        assert!(err.mse < 1e-10);
        assert!(err.snr_db.is_infinite());
    }

    #[test]
    fn test_quantization_error() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let recon = vec![1.1, 1.9, 3.1, 3.9];
        let err = compute_error(&orig, &recon);
        assert!(err.mse > 0.0);
        assert!(err.mse < 0.1);
        assert!(err.max_abs_error > 0.0);
    }

    #[test]
    fn test_empty_error() {
        let err = compute_error(&[], &[]);
        assert_eq!(err.num_elements, 0);
        assert_eq!(err.mse, 0.0);
    }

    #[test]
    fn test_snr_positive() {
        let orig = vec![1.0, 2.0, 3.0];
        let recon = vec![1.01, 2.01, 3.01];
        let err = compute_error(&orig, &recon);
        assert!(err.snr_db > 0.0);
    }

    #[test]
    fn test_distribution() {
        let vals = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let dist = value_distribution(&vals, 4);
        assert_eq!(dist.len(), 4);
        let total: usize = dist.iter().map(|b| b.count).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_distribution_empty() {
        let dist = value_distribution(&[], 4);
        assert!(dist.is_empty());
    }

    #[test]
    fn test_distribution_single_value() {
        let dist = value_distribution(&[5.0, 5.0, 5.0], 4);
        assert_eq!(dist.len(), 1);
        assert_eq!(dist[0].count, 3);
    }

    #[test]
    fn test_sparsity() {
        let vals = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        assert!((sparsity(&vals) - 0.6).abs() < 1e-6);
        assert_eq!(sparsity(&[]), 0.0);
    }

    #[test]
    fn test_layer_report() {
        let orig = vec![1.0, 2.0, 3.0];
        let recon = vec![1.0, 2.0, 3.0];
        let report = layer_report("test_layer", &orig, &recon);
        assert_eq!(report.layer_name, "test_layer");
        assert!(report.error.mse < 1e-10);
    }

    #[test]
    fn test_report_summary() {
        let orig = vec![1.0, 2.0];
        let recon = vec![1.1, 1.9];
        let report = layer_report("fc1", &orig, &recon);
        let s = report.summary();
        assert!(s.contains("fc1"));
        assert!(s.contains("MSE"));
    }

    #[test]
    fn test_rmse_and_mae() {
        let orig = vec![0.0, 0.0, 0.0];
        let recon = vec![1.0, 1.0, 1.0];
        let err = compute_error(&orig, &recon);
        assert!((err.rmse - 1.0).abs() < 1e-6);
        assert!((err.mean_abs_error - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ranges_in_report() {
        let orig = vec![-1.0, 0.0, 1.0];
        let recon = vec![-0.5, 0.0, 0.5];
        let report = layer_report("layer0", &orig, &recon);
        assert_eq!(report.original_range, (-1.0, 1.0));
        assert_eq!(report.reconstructed_range, (-0.5, 0.5));
    }
}
