//! Runtime numerical validation for GPU inference outputs.
//!
//! Detects NaN/Inf, computes distribution statistics, compares CPU
//! vs GPU outputs, and detects divergence over sequences.

use std::fmt;

// ── Distribution stats ──────────────────────────────────────────────

/// Summary statistics for a tensor's value distribution.
#[derive(Debug, Clone)]
pub struct DistributionStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f32,
    pub max: f32,
    pub nan_count: usize,
    pub inf_count: usize,
    pub element_count: usize,
}

impl fmt::Display for DistributionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "n={}, mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
            self.element_count, self.mean, self.std_dev, self.min, self.max,
        )?;
        if self.nan_count > 0 || self.inf_count > 0 {
            write!(f, " [NaN={}, Inf={}]", self.nan_count, self.inf_count,)?;
        }
        Ok(())
    }
}

// ── Comparison result ───────────────────────────────────────────────

/// Result of comparing two tensors element-wise.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Whether all elements are within tolerance.
    pub matching: bool,
    /// Maximum absolute difference.
    pub max_diff: f32,
    /// Mean absolute difference.
    pub mean_diff: f32,
    /// Number of elements exceeding tolerance.
    pub outlier_count: usize,
    /// Total element count.
    pub element_count: usize,
}

impl fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.matching { "MATCH" } else { "MISMATCH" };
        write!(
            f,
            "{status}: max_diff={:.6}, mean_diff={:.6}, \
             outliers={}/{}",
            self.max_diff, self.mean_diff, self.outlier_count, self.element_count,
        )
    }
}

// ── Divergence detection ────────────────────────────────────────────

/// A point at which sequential outputs begin diverging.
#[derive(Debug, Clone)]
pub struct DivergencePoint {
    /// Index in the history where divergence was first detected.
    pub step: usize,
    /// The metric value at the divergence point.
    pub metric: f64,
    /// Description of the divergence pattern.
    pub description: String,
}

impl fmt::Display for DivergencePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Divergence at step {}: metric={:.6} — {}",
            self.step, self.metric, self.description,
        )
    }
}

// ── NumericalValidator ──────────────────────────────────────────────

/// Runtime output validator for numerical correctness.
pub struct NumericalValidator {
    /// Default tolerance for CPU/GPU comparison.
    pub default_tolerance: f32,
    /// Threshold for detecting divergence (ratio of std-dev growth).
    pub divergence_threshold: f64,
}

impl NumericalValidator {
    #[must_use]
    pub fn new() -> Self {
        Self { default_tolerance: 1e-5, divergence_threshold: 10.0 }
    }

    /// Returns `true` if the tensor contains any NaN or Inf values.
    #[must_use]
    pub fn check_nan_inf(tensor: &[f32]) -> bool {
        tensor.iter().any(|v| v.is_nan() || v.is_infinite())
    }

    /// Compute distribution statistics for a tensor.
    #[must_use]
    pub fn check_distribution(tensor: &[f32]) -> DistributionStats {
        if tensor.is_empty() {
            return DistributionStats {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                nan_count: 0,
                inf_count: 0,
                element_count: 0,
            };
        }

        let mut sum = 0.0_f64;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut nan_count = 0_usize;
        let mut inf_count = 0_usize;
        let mut finite_count = 0_usize;

        for &v in tensor {
            if v.is_nan() {
                nan_count += 1;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
                continue;
            }
            sum += v as f64;
            finite_count += 1;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }

        let mean = if finite_count > 0 { sum / finite_count as f64 } else { 0.0 };

        let variance = if finite_count > 1 {
            tensor
                .iter()
                .filter(|v| v.is_finite())
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / finite_count as f64
        } else {
            0.0
        };

        // Handle edge case where all values are NaN/Inf
        if finite_count == 0 {
            min = 0.0;
            max = 0.0;
        }

        DistributionStats {
            mean,
            std_dev: variance.sqrt(),
            min,
            max,
            nan_count,
            inf_count,
            element_count: tensor.len(),
        }
    }

    /// Compare CPU and GPU outputs element-wise.
    #[must_use]
    pub fn compare_outputs(cpu: &[f32], gpu: &[f32], tolerance: f32) -> ComparisonResult {
        assert_eq!(cpu.len(), gpu.len(), "CPU and GPU output lengths must match");

        if cpu.is_empty() {
            return ComparisonResult {
                matching: true,
                max_diff: 0.0,
                mean_diff: 0.0,
                outlier_count: 0,
                element_count: 0,
            };
        }

        let mut max_diff: f32 = 0.0;
        let mut sum_diff = 0.0_f64;
        let mut outlier_count = 0_usize;

        for (&c, &g) in cpu.iter().zip(gpu.iter()) {
            let diff = (c - g).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            sum_diff += diff as f64;
            if diff > tolerance {
                outlier_count += 1;
            }
        }

        let mean_diff = (sum_diff / cpu.len() as f64) as f32;

        ComparisonResult {
            matching: outlier_count == 0,
            max_diff,
            mean_diff,
            outlier_count,
            element_count: cpu.len(),
        }
    }

    /// Detect divergence in a sequence of output snapshots.
    ///
    /// Computes the standard deviation of each snapshot and flags the
    /// first step where the std-dev grows by more than
    /// `divergence_threshold` relative to the initial std-dev.
    #[must_use]
    pub fn detect_divergence(&self, history: &[Vec<f32>]) -> Option<DivergencePoint> {
        if history.len() < 2 {
            return None;
        }

        let std_devs: Vec<f64> =
            history.iter().map(|snap| Self::check_distribution(snap).std_dev).collect();

        let baseline = std_devs[0];
        if baseline == 0.0 {
            // If baseline is zero, any non-zero std signals divergence
            for (i, &sd) in std_devs.iter().enumerate().skip(1) {
                if sd > 0.0 {
                    return Some(DivergencePoint {
                        step: i,
                        metric: sd,
                        description: format!("std-dev grew from 0.0 to {sd:.6}"),
                    });
                }
            }
            return None;
        }

        for (i, &sd) in std_devs.iter().enumerate().skip(1) {
            let ratio = sd / baseline;
            if ratio > self.divergence_threshold {
                return Some(DivergencePoint {
                    step: i,
                    metric: sd,
                    description: format!(
                        "std-dev ratio {ratio:.2}x exceeds \
                         threshold {:.1}x (baseline={baseline:.6}, \
                         current={sd:.6})",
                        self.divergence_threshold,
                    ),
                });
            }
        }

        None
    }
}

impl Default for NumericalValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_nan_inf_in_clean_tensor() {
        let data = vec![1.0, 2.0, 3.0, -1.0];
        assert!(!NumericalValidator::check_nan_inf(&data));
    }

    #[test]
    fn detect_nan() {
        let data = vec![1.0, f32::NAN, 3.0];
        assert!(NumericalValidator::check_nan_inf(&data));
    }

    #[test]
    fn detect_inf() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        assert!(NumericalValidator::check_nan_inf(&data));
    }

    #[test]
    fn distribution_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let stats = NumericalValidator::check_distribution(&data);
        assert!((stats.mean - 2.5).abs() < 1e-6);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 4.0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.element_count, 4);
    }

    #[test]
    fn compare_matching_outputs() {
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.0, 2.0, 3.0];
        let result = NumericalValidator::compare_outputs(&cpu, &gpu, 1e-5);
        assert!(result.matching);
        assert_eq!(result.outlier_count, 0);
    }

    #[test]
    fn compare_diverging_outputs() {
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.0, 2.5, 3.0];
        let result = NumericalValidator::compare_outputs(&cpu, &gpu, 0.1);
        assert!(!result.matching);
        assert_eq!(result.outlier_count, 1);
    }

    #[test]
    fn no_divergence_in_stable_sequence() {
        let v = NumericalValidator::new();
        let history = vec![vec![1.0, 2.0, 3.0], vec![1.1, 2.1, 3.1], vec![1.0, 1.9, 3.0]];
        assert!(v.detect_divergence(&history).is_none());
    }
}
