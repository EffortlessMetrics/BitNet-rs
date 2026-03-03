//! Quantization calibrator.
//!
//! Collects activation statistics for calibration-aware quantization.

/// Calibration method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    MinMax,
    Percentile,
    Entropy,
    Mse,
}

impl CalibrationMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MinMax => "minmax",
            Self::Percentile => "percentile",
            Self::Entropy => "entropy",
            Self::Mse => "mse",
        }
    }
}

/// Running statistics for a tensor.
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub variance: f64,
    pub count: u64,
    sum: f64,
    sum_sq: f64,
}

impl TensorStats {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            mean: 0.0,
            variance: 0.0,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Update stats with new values.
    pub fn update(&mut self, values: &[f32]) {
        for &v in values {
            let v = v as f64;
            self.min = self.min.min(v);
            self.max = self.max.max(v);
            self.sum += v;
            self.sum_sq += v * v;
            self.count += 1;
        }
        if self.count > 0 {
            self.mean = self.sum / self.count as f64;
            self.variance = (self.sum_sq / self.count as f64) - self.mean * self.mean;
        }
    }

    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    pub fn absmax(&self) -> f64 {
        self.max.abs().max(self.min.abs())
    }
}

/// Calibration result for computing quantization parameters.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    pub scale: f64,
    pub zero_point: i64,
    pub bits: u32,
    pub symmetric: bool,
}

/// Compute quantization parameters.
pub fn compute_params(
    stats: &TensorStats,
    bits: u32,
    symmetric: bool,
    method: CalibrationMethod,
) -> CalibrationResult {
    let (min_val, max_val) = match method {
        CalibrationMethod::MinMax => (stats.min, stats.max),
        CalibrationMethod::Percentile => {
            // Approximate: use 99.9% range from mean ± 3.5*std
            let std_dev = stats.variance.sqrt();
            let lo = stats.mean - 3.5 * std_dev;
            let hi = stats.mean + 3.5 * std_dev;
            (lo.max(stats.min), hi.min(stats.max))
        }
        CalibrationMethod::Entropy | CalibrationMethod::Mse => {
            // Simplified: use absmax symmetric
            let absmax = stats.absmax();
            (-absmax, absmax)
        }
    };

    let qmax = (1i64 << (bits - 1)) - 1;
    let qmin = -(1i64 << (bits - 1));

    if symmetric {
        let absmax = max_val.abs().max(min_val.abs());
        let scale = if absmax == 0.0 { 1.0 } else { absmax / qmax as f64 };
        CalibrationResult { scale, zero_point: 0, bits, symmetric: true }
    } else {
        let range = max_val - min_val;
        let scale = if range == 0.0 { 1.0 } else { range / (qmax - qmin) as f64 };
        let zero_point = (qmin as f64 - min_val / scale).round() as i64;
        CalibrationResult { scale, zero_point, bits, symmetric: false }
    }
}

/// Calibrator collecting stats across batches.
#[derive(Debug)]
pub struct Calibrator {
    pub method: CalibrationMethod,
    pub bits: u32,
    pub symmetric: bool,
    stats: Vec<TensorStats>,
}

impl Calibrator {
    pub fn new(method: CalibrationMethod, bits: u32, symmetric: bool) -> Self {
        Self { method, bits, symmetric, stats: Vec::new() }
    }

    pub fn int8_symmetric() -> Self {
        Self::new(CalibrationMethod::MinMax, 8, true)
    }

    pub fn int4_symmetric() -> Self {
        Self::new(CalibrationMethod::MinMax, 4, true)
    }

    pub fn observe(&mut self, name: &str, values: &[f32]) {
        if let Some(s) = self.stats.iter_mut().find(|s| s.name == name) {
            s.update(values);
        } else {
            let mut s = TensorStats::new(name);
            s.update(values);
            self.stats.push(s);
        }
    }

    pub fn tensor_count(&self) -> usize {
        self.stats.len()
    }

    pub fn calibrate(&self) -> Vec<(String, CalibrationResult)> {
        self.stats
            .iter()
            .map(|s| (s.name.clone(), compute_params(s, self.bits, self.symmetric, self.method)))
            .collect()
    }

    pub fn get_stats(&self, name: &str) -> Option<&TensorStats> {
        self.stats.iter().find(|s| s.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_stats_new() {
        let s = TensorStats::new("test");
        assert_eq!(s.count, 0);
        assert_eq!(s.min, f64::INFINITY);
    }

    #[test]
    fn test_tensor_stats_update() {
        let mut s = TensorStats::new("test");
        s.update(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert_eq!(s.count, 5);
    }

    #[test]
    fn test_range() {
        let mut s = TensorStats::new("test");
        s.update(&[-2.0, 3.0]);
        assert!((s.range() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_absmax() {
        let mut s = TensorStats::new("test");
        s.update(&[-5.0, 3.0]);
        assert!((s.absmax() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_int8() {
        let mut s = TensorStats::new("test");
        s.update(&[-1.0, 0.5, 1.0]);
        let result = compute_params(&s, 8, true, CalibrationMethod::MinMax);
        assert!(result.symmetric);
        assert_eq!(result.zero_point, 0);
        assert_eq!(result.bits, 8);
        assert!(result.scale > 0.0);
    }

    #[test]
    fn test_asymmetric_int8() {
        let mut s = TensorStats::new("test");
        s.update(&[0.0, 1.0, 2.0]);
        let result = compute_params(&s, 8, false, CalibrationMethod::MinMax);
        assert!(!result.symmetric);
    }

    #[test]
    fn test_int4_params() {
        let mut s = TensorStats::new("test");
        s.update(&[-8.0, 7.0]);
        let result = compute_params(&s, 4, true, CalibrationMethod::MinMax);
        assert_eq!(result.bits, 4);
    }

    #[test]
    fn test_calibrator_observe() {
        let mut c = Calibrator::int8_symmetric();
        c.observe("layer.0.weight", &[1.0, 2.0, 3.0]);
        c.observe("layer.0.weight", &[4.0, 5.0]);
        assert_eq!(c.tensor_count(), 1);
        let stats = c.get_stats("layer.0.weight").unwrap();
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_calibrator_multiple_tensors() {
        let mut c = Calibrator::int4_symmetric();
        c.observe("w1", &[1.0]);
        c.observe("w2", &[2.0]);
        assert_eq!(c.tensor_count(), 2);
    }

    #[test]
    fn test_calibrate() {
        let mut c = Calibrator::int8_symmetric();
        c.observe("w1", &[-1.0, 1.0]);
        c.observe("w2", &[0.0, 0.5]);
        let results = c.calibrate();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_percentile_method() {
        let mut s = TensorStats::new("test");
        s.update(&[-10.0, -1.0, 0.0, 1.0, 10.0]);
        let result = compute_params(&s, 8, true, CalibrationMethod::Percentile);
        assert!(result.scale > 0.0);
    }

    #[test]
    fn test_method_str() {
        assert_eq!(CalibrationMethod::MinMax.as_str(), "minmax");
        assert_eq!(CalibrationMethod::Entropy.as_str(), "entropy");
    }

    #[test]
    fn test_zero_range() {
        let mut s = TensorStats::new("test");
        s.update(&[5.0, 5.0, 5.0]);
        let result = compute_params(&s, 8, true, CalibrationMethod::MinMax);
        assert!(result.scale > 0.0); // should not be zero
    }
}
