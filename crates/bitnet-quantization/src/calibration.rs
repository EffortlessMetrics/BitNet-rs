//! Quantization calibration.
//!
//! Calibrate quantization parameters (scale, zero-point) from
//! weight distributions using various strategies.

/// Calibration strategy for determining scale and zero-point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationStrategy {
    /// Min-max calibration: use full range of observed values.
    MinMax,
    /// Percentile-based: clip outliers at given percentile.
    Percentile,
    /// Entropy-based: minimize KL divergence.
    Entropy,
    /// Moving average of min/max (for dynamic quantization).
    MovingAverage,
}

/// Quantization parameters derived from calibration.
#[derive(Debug, Clone)]
pub struct QuantParams {
    pub scale: f64,
    pub zero_point: f64,
    pub bits: u8,
    pub symmetric: bool,
}

impl QuantParams {
    /// Quantize a float value to integer representation.
    pub fn quantize(&self, value: f64) -> i64 {
        let qmax = (1i64 << (self.bits - 1)) - 1;
        let qmin = -(1i64 << (self.bits - 1));
        let q = ((value - self.zero_point) / self.scale).round() as i64;
        q.clamp(qmin, qmax)
    }

    /// Dequantize an integer value back to float.
    pub fn dequantize(&self, value: i64) -> f64 {
        value as f64 * self.scale + self.zero_point
    }

    /// Quantization range (min, max) in float space.
    pub fn range(&self) -> (f64, f64) {
        let qmax = (1i64 << (self.bits - 1)) - 1;
        let qmin = -(1i64 << (self.bits - 1));
        (self.dequantize(qmin), self.dequantize(qmax))
    }
}

/// Calibrate using min-max strategy.
pub fn calibrate_minmax(values: &[f32], bits: u8, symmetric: bool) -> QuantParams {
    if values.is_empty() {
        return QuantParams { scale: 1.0, zero_point: 0.0, bits, symmetric };
    }

    let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;

    compute_params(min_val, max_val, bits, symmetric)
}

/// Calibrate using percentile clipping.
pub fn calibrate_percentile(
    values: &[f32],
    bits: u8,
    symmetric: bool,
    percentile: f64,
) -> QuantParams {
    if values.is_empty() {
        return QuantParams { scale: 1.0, zero_point: 0.0, bits, symmetric };
    }

    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let low_idx = ((1.0 - percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    let high_idx = ((percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;

    let min_val = sorted[low_idx.min(sorted.len() - 1)] as f64;
    let max_val = sorted[high_idx.min(sorted.len() - 1)] as f64;

    compute_params(min_val, max_val, bits, symmetric)
}

fn compute_params(min_val: f64, max_val: f64, bits: u8, symmetric: bool) -> QuantParams {
    let qmax = (1i64 << (bits - 1)) - 1;
    let qmin = -(1i64 << (bits - 1));

    if symmetric {
        let abs_max = min_val.abs().max(max_val.abs());
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / qmax as f64 };
        QuantParams { scale, zero_point: 0.0, bits, symmetric: true }
    } else {
        let range = max_val - min_val;
        let scale = if range == 0.0 { 1.0 } else { range / (qmax - qmin) as f64 };
        let zero_point = min_val;
        QuantParams { scale, zero_point, bits, symmetric: false }
    }
}

/// Running calibration stats for dynamic quantization.
#[derive(Debug, Clone)]
pub struct RunningCalibration {
    pub min_val: f64,
    pub max_val: f64,
    pub count: u64,
    pub momentum: f64,
}

impl RunningCalibration {
    pub fn new(momentum: f64) -> Self {
        Self { min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, count: 0, momentum }
    }

    /// Update with a new batch of observations.
    pub fn update(&mut self, values: &[f32]) {
        if values.is_empty() {
            return;
        }
        let batch_min = values.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
        let batch_max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;

        if self.count == 0 {
            self.min_val = batch_min;
            self.max_val = batch_max;
        } else {
            self.min_val = self.momentum * self.min_val + (1.0 - self.momentum) * batch_min;
            self.max_val = self.momentum * self.max_val + (1.0 - self.momentum) * batch_max;
        }
        self.count += 1;
    }

    /// Get current quantization params.
    pub fn params(&self, bits: u8, symmetric: bool) -> QuantParams {
        compute_params(self.min_val, self.max_val, bits, symmetric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minmax_symmetric() {
        let values = vec![-1.0, 0.0, 1.0];
        let params = calibrate_minmax(&values, 8, true);
        assert!(params.symmetric);
        assert!((params.zero_point).abs() < 1e-10);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_minmax_asymmetric() {
        let values = vec![0.0, 1.0, 2.0, 3.0];
        let params = calibrate_minmax(&values, 8, false);
        assert!(!params.symmetric);
    }

    #[test]
    fn test_quantize_dequantize() {
        let values: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.1).collect();
        let params = calibrate_minmax(&values, 8, true);
        for &v in &values {
            let q = params.quantize(v as f64);
            let dq = params.dequantize(q);
            assert!((dq - v as f64).abs() < params.scale + 1e-6);
        }
    }

    #[test]
    fn test_percentile_calibration() {
        let mut values: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        values.push(100.0); // outlier
        let params = calibrate_percentile(&values, 8, true, 99.0);
        // Should clip the outlier
        let (_, hi) = params.range();
        assert!(hi < 150.0);
    }

    #[test]
    fn test_running_calibration() {
        let mut cal = RunningCalibration::new(0.9);
        cal.update(&[0.0, 1.0]);
        cal.update(&[-1.0, 2.0]);
        assert!(cal.count == 2);
        let params = cal.params(8, true);
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_quant_range() {
        let params = calibrate_minmax(&[-1.0, 1.0], 8, true);
        let (lo, hi) = params.range();
        assert!(lo < 0.0);
        assert!(hi > 0.0);
    }

    #[test]
    fn test_empty_values() {
        let params = calibrate_minmax(&[], 8, true);
        assert_eq!(params.scale, 1.0);
    }

    #[test]
    fn test_4bit_quantization() {
        let values = vec![-1.0, 0.0, 1.0];
        let params = calibrate_minmax(&values, 4, true);
        let q = params.quantize(1.0);
        assert!(q <= 7); // 4-bit signed max
        assert!(q >= -8);
    }

    #[test]
    fn test_2bit_quantization() {
        let values = vec![-1.0, 0.0, 1.0];
        let params = calibrate_minmax(&values, 2, true);
        let q = params.quantize(0.5);
        assert!(q <= 1);
        assert!(q >= -2);
    }

    #[test]
    fn test_running_momentum() {
        let mut cal = RunningCalibration::new(0.0); // no history
        cal.update(&[0.0, 10.0]);
        cal.update(&[0.0, 1.0]); // should fully replace
        assert!((cal.max_val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_symmetric_zero_centered() {
        let params = calibrate_minmax(&[-5.0, 5.0], 8, true);
        assert!((params.zero_point).abs() < 1e-10);
    }

    #[test]
    fn test_calibration_strategies() {
        // Just verify enum variants exist
        assert_ne!(CalibrationStrategy::MinMax, CalibrationStrategy::Percentile);
        assert_ne!(CalibrationStrategy::Entropy, CalibrationStrategy::MovingAverage);
    }
}
