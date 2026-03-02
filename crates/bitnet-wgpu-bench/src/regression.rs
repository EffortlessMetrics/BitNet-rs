//! Regression detection by comparing baseline vs current benchmark receipts.

use crate::receipt::BenchReceipt;

/// The outcome of comparing a single kernel's performance against its baseline.
#[derive(Debug, Clone)]
pub struct RegressionResult {
    pub kernel: String,
    pub baseline_us: u64,
    pub current_us: u64,
    pub change_pct: f64,
    pub is_regression: bool,
}

/// Compares current benchmark receipts against a baseline set.
pub struct RegressionDetector;

impl RegressionDetector {
    /// Check for regressions by matching kernels by name.
    ///
    /// A regression is flagged when `current_us` exceeds `baseline_us` by more
    /// than `threshold_pct` percent. Kernels present in `current` but absent
    /// from `baseline` are reported with zero baseline and no regression flag.
    pub fn check(
        baseline: &[BenchReceipt],
        current: &[BenchReceipt],
        threshold_pct: f64,
    ) -> Vec<RegressionResult> {
        use std::collections::HashMap;

        let baseline_map: HashMap<&str, u64> =
            baseline.iter().map(|r| (r.kernel_name.as_str(), r.elapsed_us)).collect();

        current
            .iter()
            .map(|r| {
                let current_us = r.elapsed_us;
                match baseline_map.get(r.kernel_name.as_str()) {
                    Some(&baseline_us) if baseline_us > 0 => {
                        let change_pct =
                            ((current_us as f64 - baseline_us as f64) / baseline_us as f64) * 100.0;
                        RegressionResult {
                            kernel: r.kernel_name.clone(),
                            baseline_us,
                            current_us,
                            change_pct,
                            is_regression: change_pct > threshold_pct,
                        }
                    }
                    _ => RegressionResult {
                        kernel: r.kernel_name.clone(),
                        baseline_us: 0,
                        current_us,
                        change_pct: 0.0,
                        is_regression: false,
                    },
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipt::BenchReceipt;

    fn receipt(name: &str, elapsed_us: u64) -> BenchReceipt {
        BenchReceipt::new(name, [256, 1, 1], [1, 1, 1], elapsed_us, 0.0, 0, "", "")
    }

    #[test]
    fn test_no_regression_when_faster() {
        let baseline = vec![receipt("k", 1000)];
        let current = vec![receipt("k", 900)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_regression);
    }

    #[test]
    fn test_no_regression_within_threshold() {
        let baseline = vec![receipt("k", 1000)];
        let current = vec![receipt("k", 1050)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert!(!results[0].is_regression);
        assert!((results[0].change_pct - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_regression_above_threshold() {
        let baseline = vec![receipt("k", 1000)];
        let current = vec![receipt("k", 1200)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert!(results[0].is_regression);
        assert!((results[0].change_pct - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_exact_threshold_not_regression() {
        let baseline = vec![receipt("k", 1000)];
        let current = vec![receipt("k", 1100)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        // 10% change == threshold, not strictly greater
        assert!(!results[0].is_regression);
    }

    #[test]
    fn test_new_kernel_no_baseline() {
        let baseline = vec![];
        let current = vec![receipt("new_kernel", 500)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_regression);
        assert_eq!(results[0].baseline_us, 0);
    }

    #[test]
    fn test_multiple_kernels_mixed() {
        let baseline = vec![receipt("fast", 100), receipt("slow", 1000)];
        let current = vec![receipt("fast", 90), receipt("slow", 1500)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert!(!results[0].is_regression); // fast got faster
        assert!(results[1].is_regression); // slow regressed 50%
    }

    #[test]
    fn test_empty_inputs() {
        let results = RegressionDetector::check(&[], &[], 10.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_change_pct_negative_for_improvement() {
        let baseline = vec![receipt("k", 1000)];
        let current = vec![receipt("k", 800)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert!(results[0].change_pct < 0.0);
        assert!((results[0].change_pct - (-20.0)).abs() < 0.01);
    }

    #[test]
    fn test_zero_baseline_no_panic() {
        let baseline = vec![receipt("k", 0)];
        let current = vec![receipt("k", 100)];
        let results = RegressionDetector::check(&baseline, &current, 10.0);
        assert!(!results[0].is_regression);
        assert_eq!(results[0].baseline_us, 0);
    }
}
