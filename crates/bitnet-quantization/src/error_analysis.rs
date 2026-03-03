//! Quantization error analysis.
//!
//! Measure and analyze errors introduced by quantization.

/// Error metrics for a quantized tensor.
#[derive(Debug, Clone)]
pub struct QuantError {
    pub mse: f64,
    pub mae: f64,
    pub max_error: f64,
    pub snr_db: f64,
    pub element_count: usize,
}

/// Compute quantization error between original and quantized values.
pub fn compute_error(original: &[f32], quantized: &[f32]) -> QuantError {
    assert_eq!(original.len(), quantized.len(), "length mismatch");
    let n = original.len();
    if n == 0 {
        return QuantError { mse: 0.0, mae: 0.0, max_error: 0.0, snr_db: 0.0, element_count: 0 };
    }

    let mut sum_sq_error = 0.0f64;
    let mut sum_abs_error = 0.0f64;
    let mut max_err = 0.0f64;
    let mut signal_power = 0.0f64;

    for (o, q) in original.iter().zip(quantized.iter()) {
        let diff = (*o as f64) - (*q as f64);
        sum_sq_error += diff * diff;
        sum_abs_error += diff.abs();
        max_err = max_err.max(diff.abs());
        signal_power += (*o as f64) * (*o as f64);
    }

    let mse = sum_sq_error / n as f64;
    let mae = sum_abs_error / n as f64;
    let snr_db = if sum_sq_error > 0.0 {
        10.0 * (signal_power / sum_sq_error).log10()
    } else {
        f64::INFINITY
    };

    QuantError { mse, mae, max_error: max_err, snr_db, element_count: n }
}

impl QuantError {
    pub fn rmse(&self) -> f64 {
        self.mse.sqrt()
    }

    pub fn is_acceptable(&self, max_mse: f64, min_snr: f64) -> bool {
        self.mse <= max_mse && self.snr_db >= min_snr
    }
}

/// Per-layer error report.
#[derive(Debug, Clone)]
pub struct LayerErrorReport {
    pub layer_name: String,
    pub error: QuantError,
}

/// Model-level error summary.
#[derive(Debug, Clone)]
pub struct ModelErrorSummary {
    pub layers: Vec<LayerErrorReport>,
    pub overall_mse: f64,
    pub overall_snr_db: f64,
    pub worst_layer: Option<String>,
}

impl ModelErrorSummary {
    pub fn from_layers(layers: Vec<LayerErrorReport>) -> Self {
        if layers.is_empty() {
            return Self { layers, overall_mse: 0.0, overall_snr_db: 0.0, worst_layer: None };
        }

        let total_elements: usize = layers.iter().map(|l| l.error.element_count).sum();
        let weighted_mse: f64 =
            layers.iter().map(|l| l.error.mse * l.error.element_count as f64).sum::<f64>()
                / total_elements.max(1) as f64;

        let worst = layers
            .iter()
            .max_by(|a, b| {
                a.error.mse.partial_cmp(&b.error.mse).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|l| l.layer_name.clone());

        let avg_snr = layers.iter().map(|l| l.error.snr_db).sum::<f64>() / layers.len() as f64;

        Self { layers, overall_mse: weighted_mse, overall_snr_db: avg_snr, worst_layer: worst }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn all_acceptable(&self, max_mse: f64, min_snr: f64) -> bool {
        self.layers.iter().all(|l| l.error.is_acceptable(max_mse, min_snr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_error() {
        let data = vec![1.0, 2.0, 3.0];
        let e = compute_error(&data, &data);
        assert_eq!(e.mse, 0.0);
        assert_eq!(e.mae, 0.0);
        assert_eq!(e.snr_db, f64::INFINITY);
    }

    #[test]
    fn test_known_error() {
        let orig = vec![1.0, 2.0, 3.0, 4.0];
        let quant = vec![1.1, 2.1, 3.1, 4.1];
        let e = compute_error(&orig, &quant);
        assert!((e.mse - 0.01).abs() < 0.001);
        assert!((e.mae - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_max_error() {
        let orig = vec![0.0, 0.0, 0.0, 10.0];
        let quant = vec![0.0, 0.0, 0.0, 8.0];
        let e = compute_error(&orig, &quant);
        assert!((e.max_error - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_rmse() {
        let orig = vec![1.0, 2.0, 3.0];
        let quant = vec![1.1, 1.9, 3.2];
        let e = compute_error(&orig, &quant);
        assert!(e.rmse() > 0.0);
        assert!(e.rmse() < 1.0);
    }

    #[test]
    fn test_snr() {
        let orig = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let quant = vec![1.01, 2.01, 3.01, 4.01, 5.01];
        let e = compute_error(&orig, &quant);
        assert!(e.snr_db > 30.0); // High SNR for small error
    }

    #[test]
    fn test_acceptable() {
        let orig = vec![1.0, 2.0, 3.0];
        let quant = vec![1.01, 2.01, 3.01];
        let e = compute_error(&orig, &quant);
        assert!(e.is_acceptable(0.01, 20.0));
    }

    #[test]
    fn test_empty() {
        let e = compute_error(&[], &[]);
        assert_eq!(e.element_count, 0);
    }

    #[test]
    fn test_model_summary() {
        let layers = vec![
            LayerErrorReport {
                layer_name: "layer.0".into(),
                error: compute_error(&[1.0, 2.0], &[1.1, 2.1]),
            },
            LayerErrorReport {
                layer_name: "layer.1".into(),
                error: compute_error(&[3.0, 4.0], &[3.0, 4.0]),
            },
        ];
        let summary = ModelErrorSummary::from_layers(layers);
        assert_eq!(summary.layer_count(), 2);
        assert!(summary.worst_layer.is_some());
    }

    #[test]
    fn test_all_acceptable() {
        let layers = vec![
            LayerErrorReport { layer_name: "l0".into(), error: compute_error(&[1.0], &[1.001]) },
            LayerErrorReport { layer_name: "l1".into(), error: compute_error(&[2.0], &[2.001]) },
        ];
        let summary = ModelErrorSummary::from_layers(layers);
        assert!(summary.all_acceptable(0.01, 10.0));
    }

    #[test]
    fn test_empty_summary() {
        let summary = ModelErrorSummary::from_layers(vec![]);
        assert_eq!(summary.layer_count(), 0);
        assert!(summary.worst_layer.is_none());
    }

    #[test]
    fn test_element_count() {
        let e = compute_error(&[1.0, 2.0, 3.0], &[1.1, 2.1, 3.1]);
        assert_eq!(e.element_count, 3);
    }
}
