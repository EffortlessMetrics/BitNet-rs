//! Quantization error metrics for evaluating quantization quality.
//!
//! Provides MSE, SNR, and cosine similarity measurements between original
//! and quantized tensors. These metrics help evaluate the fidelity of
//! different quantization strategies and block sizes.

use bitnet_common::{QuantizationError, Result};

/// Aggregated quantization quality metrics.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationMetrics {
    /// Mean Squared Error between original and dequantized values.
    pub mse: f32,
    /// Signal-to-Noise Ratio in decibels.
    pub snr_db: f32,
    /// Cosine similarity in `[−1, 1]`.
    pub cosine_similarity: f32,
    /// Peak Signal-to-Noise Ratio in decibels, relative to the data range.
    pub psnr_db: f32,
    /// Maximum absolute element-wise error.
    pub max_abs_error: f32,
    /// Number of elements compared.
    pub num_elements: usize,
}

impl QuantizationMetrics {
    /// Compute all metrics for a pair of equal-length f32 slices.
    ///
    /// # Errors
    ///
    /// Returns an error if the slices have different lengths or are empty.
    pub fn compute(original: &[f32], dequantized: &[f32]) -> Result<Self> {
        if original.len() != dequantized.len() {
            return Err(QuantizationError::QuantizationFailed {
                reason: format!(
                    "Length mismatch in metrics computation: original={}, dequantized={}",
                    original.len(),
                    dequantized.len()
                ),
            }
            .into());
        }
        if original.is_empty() {
            return Err(QuantizationError::QuantizationFailed {
                reason: "Cannot compute metrics on empty data".to_string(),
            }
            .into());
        }

        let n = original.len() as f64;
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_orig = 0.0f64;
        let mut dot = 0.0f64;
        let mut sum_sq_deq = 0.0f64;
        let mut max_abs = 0.0f32;
        let mut data_min = f32::INFINITY;
        let mut data_max = f32::NEG_INFINITY;

        for (&o, &d) in original.iter().zip(dequantized.iter()) {
            let o64 = o as f64;
            let d64 = d as f64;
            let err = o64 - d64;
            sum_sq_err += err * err;
            sum_sq_orig += o64 * o64;
            dot += o64 * d64;
            sum_sq_deq += d64 * d64;
            let ae = (o - d).abs();
            if ae > max_abs {
                max_abs = ae;
            }
            if o.is_finite() {
                if o < data_min {
                    data_min = o;
                }
                if o > data_max {
                    data_max = o;
                }
            }
        }

        let mse = (sum_sq_err / n) as f32;

        let snr_db = if sum_sq_err == 0.0 {
            f32::INFINITY
        } else {
            (10.0 * (sum_sq_orig / sum_sq_err).log10()) as f32
        };

        let denom = (sum_sq_orig.sqrt()) * (sum_sq_deq.sqrt());
        let cosine_similarity = if denom == 0.0 { 0.0 } else { (dot / denom) as f32 };

        let data_range = (data_max - data_min).max(0.0) as f64;
        let psnr_db = if mse == 0.0 {
            f32::INFINITY
        } else if data_range == 0.0 {
            0.0
        } else {
            (10.0 * (data_range * data_range / sum_sq_err * n).log10()) as f32
        };

        Ok(Self {
            mse,
            snr_db,
            cosine_similarity,
            psnr_db,
            max_abs_error: max_abs,
            num_elements: original.len(),
        })
    }

    /// Returns `true` when all quality thresholds are met.
    pub fn meets_thresholds(&self, max_mse: f32, min_snr_db: f32, min_cosine: f32) -> bool {
        self.mse <= max_mse && self.snr_db >= min_snr_db && self.cosine_similarity >= min_cosine
    }
}

impl std::fmt::Display for QuantizationMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MSE={:.6} SNR={:.1}dB cos={:.6} PSNR={:.1}dB max_err={:.6} (n={})",
            self.mse,
            self.snr_db,
            self.cosine_similarity,
            self.psnr_db,
            self.max_abs_error,
            self.num_elements,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_data_yields_perfect_metrics() {
        let data = vec![1.0, -2.0, 3.0, 0.5, -0.25];
        let m = QuantizationMetrics::compute(&data, &data).unwrap();
        assert_eq!(m.mse, 0.0);
        assert!(m.snr_db.is_infinite() && m.snr_db > 0.0);
        assert!((m.cosine_similarity - 1.0).abs() < 1e-6);
        assert_eq!(m.max_abs_error, 0.0);
    }

    #[test]
    fn known_error_metrics() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        // Shift every element by +0.1
        let deq: Vec<f32> = orig.iter().map(|x| x + 0.1).collect();
        let m = QuantizationMetrics::compute(&orig, &deq).unwrap();
        assert!((m.mse - 0.01).abs() < 1e-5, "mse={}", m.mse);
        assert!(m.max_abs_error > 0.09 && m.max_abs_error < 0.11);
        assert!(m.cosine_similarity > 0.99);
    }

    #[test]
    fn length_mismatch_returns_error() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(QuantizationMetrics::compute(&a, &b).is_err());
    }

    #[test]
    fn empty_data_returns_error() {
        assert!(QuantizationMetrics::compute(&[], &[]).is_err());
    }

    #[test]
    fn meets_thresholds_works() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let deq: Vec<f32> = orig.iter().map(|x| x + 0.01).collect();
        let m = QuantizationMetrics::compute(&orig, &deq).unwrap();
        assert!(m.meets_thresholds(0.001, 30.0, 0.999));
        assert!(!m.meets_thresholds(0.00001, 30.0, 0.999));
    }

    #[test]
    fn display_impl_contains_key_fields() {
        let m = QuantizationMetrics {
            mse: 0.01,
            snr_db: 30.0,
            cosine_similarity: 0.999,
            psnr_db: 40.0,
            max_abs_error: 0.1,
            num_elements: 100,
        };
        let s = m.to_string();
        assert!(s.contains("MSE="));
        assert!(s.contains("SNR="));
        assert!(s.contains("cos="));
    }

    #[test]
    fn orthogonal_vectors_have_zero_cosine() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        let m = QuantizationMetrics::compute(&a, &b).unwrap();
        assert!(m.cosine_similarity.abs() < 1e-6);
    }
}
