//! Weight statistics analysis for model diagnostics.
//!
//! Computes distribution statistics over weight tensors to detect
//! anomalies (NaN, Inf, extreme values, dead neurons).

use std::fmt;

/// Statistics for a single weight tensor.
#[derive(Debug, Clone)]
pub struct TensorStats {
    pub name: String,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub variance: f64,
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
}

impl TensorStats {
    /// Compute statistics from a slice of f32 values.
    pub fn from_f32(name: &str, shape: &[usize], data: &[f32]) -> Self {
        let n = data.len();
        if n == 0 {
            return Self {
                name: name.to_string(),
                shape: shape.to_vec(),
                element_count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                variance: 0.0,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
            };
        }

        let mut min = f64::MAX;
        let mut max = f64::MIN;
        let mut sum = 0.0_f64;
        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        let mut zero_count = 0usize;

        for &v in data {
            let v64 = v as f64;
            if v.is_nan() {
                nan_count += 1;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
                continue;
            }
            if v == 0.0 {
                zero_count += 1;
            }
            if v64 < min {
                min = v64;
            }
            if v64 > max {
                max = v64;
            }
            sum += v64;
        }

        let valid = n - nan_count - inf_count;
        let mean = if valid > 0 { sum / valid as f64 } else { 0.0 };

        // Second pass for variance
        let mut var_sum = 0.0_f64;
        for &v in data {
            if v.is_nan() || v.is_infinite() {
                continue;
            }
            let diff = v as f64 - mean;
            var_sum += diff * diff;
        }
        let variance = if valid > 1 { var_sum / (valid - 1) as f64 } else { 0.0 };

        if min == f64::MAX {
            min = 0.0;
        }
        if max == f64::MIN {
            max = 0.0;
        }

        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            element_count: n,
            min,
            max,
            mean,
            variance,
            nan_count,
            inf_count,
            zero_count,
        }
    }

    /// Standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Whether any NaN or Inf values exist.
    pub fn has_anomalies(&self) -> bool {
        self.nan_count > 0 || self.inf_count > 0
    }

    /// Fraction of zero-valued elements.
    pub fn sparsity(&self) -> f64 {
        if self.element_count == 0 {
            return 0.0;
        }
        self.zero_count as f64 / self.element_count as f64
    }

    /// Whether the tensor is highly sparse (>90% zeros).
    pub fn is_sparse(&self) -> bool {
        self.sparsity() > 0.9
    }
}

impl fmt::Display for TensorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: shape={:?}, mean={:.6}, std={:.6}, range=[{:.6}, {:.6}], zeros={}, nan={}, inf={}",
            self.name,
            self.shape,
            self.mean,
            self.std_dev(),
            self.min,
            self.max,
            self.zero_count,
            self.nan_count,
            self.inf_count
        )
    }
}

/// Anomaly detected in weight analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightAnomaly {
    /// Tensor contains NaN values.
    NanDetected { tensor: String, count: usize },
    /// Tensor contains Inf values.
    InfDetected { tensor: String, count: usize },
    /// Tensor is all zeros (dead).
    DeadTensor { tensor: String },
    /// Extremely high variance (possible uninitialized).
    HighVariance { tensor: String },
    /// Extremely sparse (>95% zeros).
    HighSparsity { tensor: String },
}

impl fmt::Display for WeightAnomaly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NanDetected { tensor, count } => {
                write!(f, "NaN detected in {tensor}: {count} values")
            }
            Self::InfDetected { tensor, count } => {
                write!(f, "Inf detected in {tensor}: {count} values")
            }
            Self::DeadTensor { tensor } => write!(f, "Dead tensor (all zeros): {tensor}"),
            Self::HighVariance { tensor } => {
                write!(f, "High variance (possible uninitialized): {tensor}")
            }
            Self::HighSparsity { tensor } => write!(f, "High sparsity (>95% zeros): {tensor}"),
        }
    }
}

/// Analyze a tensor for anomalies.
pub fn detect_anomalies(stats: &TensorStats) -> Vec<WeightAnomaly> {
    let mut anomalies = Vec::new();
    if stats.nan_count > 0 {
        anomalies.push(WeightAnomaly::NanDetected {
            tensor: stats.name.clone(),
            count: stats.nan_count,
        });
    }
    if stats.inf_count > 0 {
        anomalies.push(WeightAnomaly::InfDetected {
            tensor: stats.name.clone(),
            count: stats.inf_count,
        });
    }
    if stats.element_count > 0
        && stats.zero_count == stats.element_count
        && stats.nan_count == 0
        && stats.inf_count == 0
    {
        anomalies.push(WeightAnomaly::DeadTensor { tensor: stats.name.clone() });
    }
    if stats.variance > 1e6 {
        anomalies.push(WeightAnomaly::HighVariance { tensor: stats.name.clone() });
    }
    if stats.element_count > 0 && stats.sparsity() > 0.95 {
        anomalies.push(WeightAnomaly::HighSparsity { tensor: stats.name.clone() });
    }
    anomalies
}

/// Summary report across all tensors.
#[derive(Debug, Clone)]
pub struct WeightReport {
    pub tensor_count: usize,
    pub total_elements: usize,
    pub total_nan: usize,
    pub total_inf: usize,
    pub anomalies: Vec<WeightAnomaly>,
}

/// Generate a report from a collection of tensor stats.
pub fn generate_report(all_stats: &[TensorStats]) -> WeightReport {
    let mut total_elements = 0;
    let mut total_nan = 0;
    let mut total_inf = 0;
    let mut anomalies = Vec::new();

    for s in all_stats {
        total_elements += s.element_count;
        total_nan += s.nan_count;
        total_inf += s.inf_count;
        anomalies.extend(detect_anomalies(s));
    }

    WeightReport { tensor_count: all_stats.len(), total_elements, total_nan, total_inf, anomalies }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_normal() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::from_f32("test", &[5], &data);
        assert_eq!(stats.element_count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_empty() {
        let stats = TensorStats::from_f32("empty", &[], &[]);
        assert_eq!(stats.element_count, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_stats_with_nan() {
        let data = vec![1.0f32, f32::NAN, 3.0];
        let stats = TensorStats::from_f32("nan_test", &[3], &data);
        assert_eq!(stats.nan_count, 1);
        assert!(stats.has_anomalies());
    }

    #[test]
    fn test_stats_with_inf() {
        let data = vec![1.0f32, f32::INFINITY, -f32::INFINITY];
        let stats = TensorStats::from_f32("inf_test", &[3], &data);
        assert_eq!(stats.inf_count, 2);
        assert!(stats.has_anomalies());
    }

    #[test]
    fn test_stats_zeros() {
        let data = vec![0.0f32; 100];
        let stats = TensorStats::from_f32("zeros", &[100], &data);
        assert_eq!(stats.zero_count, 100);
        assert_eq!(stats.sparsity(), 1.0);
        assert!(stats.is_sparse());
    }

    #[test]
    fn test_stats_no_anomalies() {
        let data = vec![0.1f32, 0.2, 0.3];
        let stats = TensorStats::from_f32("clean", &[3], &data);
        assert!(!stats.has_anomalies());
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = TensorStats::from_f32("test", &[8], &data);
        assert!(stats.std_dev() > 0.0);
    }

    #[test]
    fn test_sparsity_dense() {
        let data = vec![1.0f32, 2.0, 3.0];
        let stats = TensorStats::from_f32("dense", &[3], &data);
        assert_eq!(stats.sparsity(), 0.0);
        assert!(!stats.is_sparse());
    }

    #[test]
    fn test_display() {
        let data = vec![1.0f32, 2.0, 3.0];
        let stats = TensorStats::from_f32("layer.0.weight", &[1, 3], &data);
        let s = format!("{stats}");
        assert!(s.contains("layer.0.weight"));
        assert!(s.contains("shape="));
    }

    #[test]
    fn test_detect_nan_anomaly() {
        let data = vec![f32::NAN; 5];
        let stats = TensorStats::from_f32("bad", &[5], &data);
        let anomalies = detect_anomalies(&stats);
        assert!(anomalies.iter().any(|a| matches!(a, WeightAnomaly::NanDetected { .. })));
    }

    #[test]
    fn test_detect_dead_tensor() {
        let data = vec![0.0f32; 10];
        let stats = TensorStats::from_f32("dead", &[10], &data);
        let anomalies = detect_anomalies(&stats);
        assert!(anomalies.iter().any(|a| matches!(a, WeightAnomaly::DeadTensor { .. })));
    }

    #[test]
    fn test_detect_high_sparsity() {
        let mut data = vec![0.0f32; 100];
        data[0] = 1.0; // 99% sparse
        let stats = TensorStats::from_f32("sparse", &[100], &data);
        let anomalies = detect_anomalies(&stats);
        assert!(anomalies.iter().any(|a| matches!(a, WeightAnomaly::HighSparsity { .. })));
    }

    #[test]
    fn test_detect_high_variance() {
        // Variance > 1e6
        let data = vec![0.0f32, 2000.0];
        let stats = TensorStats::from_f32("wild", &[2], &data);
        assert!(stats.variance > 1e6);
        let anomalies = detect_anomalies(&stats);
        assert!(anomalies.iter().any(|a| matches!(a, WeightAnomaly::HighVariance { .. })));
    }

    #[test]
    fn test_detect_no_anomalies() {
        let data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let stats = TensorStats::from_f32("clean", &[5], &data);
        assert!(detect_anomalies(&stats).is_empty());
    }

    #[test]
    fn test_generate_report_empty() {
        let report = generate_report(&[]);
        assert_eq!(report.tensor_count, 0);
        assert_eq!(report.total_elements, 0);
    }

    #[test]
    fn test_generate_report_mixed() {
        let clean = TensorStats::from_f32("clean", &[3], &[1.0, 2.0, 3.0]);
        let bad = TensorStats::from_f32("bad", &[2], &[f32::NAN, 1.0]);
        let report = generate_report(&[clean, bad]);
        assert_eq!(report.tensor_count, 2);
        assert_eq!(report.total_nan, 1);
        assert!(!report.anomalies.is_empty());
    }

    #[test]
    fn test_anomaly_display() {
        let a = WeightAnomaly::NanDetected { tensor: "test".into(), count: 5 };
        let s = format!("{a}");
        assert!(s.contains("NaN"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_sparsity_empty() {
        let stats = TensorStats::from_f32("e", &[], &[]);
        assert_eq!(stats.sparsity(), 0.0);
    }

    #[test]
    fn test_single_element() {
        let stats = TensorStats::from_f32("one", &[1], &[42.0]);
        assert!((stats.mean - 42.0).abs() < 1e-6);
        assert_eq!(stats.variance, 0.0); // only 1 element
    }
}
