//! Model weight pruning analysis.
//!
//! Analyze weight distributions to estimate pruning impact:
//! sparsity levels, magnitude thresholds, and sensitivity.

/// Statistics about a weight tensor for pruning decisions.
#[derive(Debug, Clone)]
pub struct WeightStats {
    pub name: String,
    pub num_elements: usize,
    pub num_zeros: usize,
    pub mean_magnitude: f64,
    pub max_magnitude: f64,
    pub percentiles: Percentiles,
}

impl WeightStats {
    pub fn sparsity(&self) -> f64 {
        if self.num_elements == 0 {
            return 0.0;
        }
        self.num_zeros as f64 / self.num_elements as f64
    }

    pub fn density(&self) -> f64 {
        1.0 - self.sparsity()
    }
}

/// Magnitude percentiles for threshold selection.
#[derive(Debug, Clone, Default)]
pub struct Percentiles {
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p99: f64,
}

/// Compute percentile from sorted values.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Analyze a weight tensor (as f32 values) for pruning characteristics.
pub fn analyze_weights(name: impl Into<String>, weights: &[f32]) -> WeightStats {
    let num_elements = weights.len();
    let num_zeros = weights.iter().filter(|&&w| w == 0.0).count();

    let mut magnitudes: Vec<f64> = weights.iter().map(|&w| (w as f64).abs()).collect();
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_magnitude = if magnitudes.is_empty() {
        0.0
    } else {
        magnitudes.iter().sum::<f64>() / magnitudes.len() as f64
    };
    let max_magnitude = magnitudes.last().copied().unwrap_or(0.0);

    let percentiles = Percentiles {
        p10: percentile(&magnitudes, 10.0),
        p25: percentile(&magnitudes, 25.0),
        p50: percentile(&magnitudes, 50.0),
        p75: percentile(&magnitudes, 75.0),
        p90: percentile(&magnitudes, 90.0),
        p99: percentile(&magnitudes, 99.0),
    };

    WeightStats {
        name: name.into(),
        num_elements,
        num_zeros,
        mean_magnitude,
        max_magnitude,
        percentiles,
    }
}

/// Estimate how many elements would be pruned at a given magnitude threshold.
pub fn pruning_impact(weights: &[f32], threshold: f32) -> PruningImpact {
    let total = weights.len();
    let pruned = weights.iter().filter(|&&w| w.abs() < threshold).count();
    PruningImpact {
        total_elements: total,
        pruned_elements: pruned,
        sparsity: if total == 0 { 0.0 } else { pruned as f64 / total as f64 },
        threshold,
    }
}

/// Result of pruning impact estimation.
#[derive(Debug, Clone)]
pub struct PruningImpact {
    pub total_elements: usize,
    pub pruned_elements: usize,
    pub sparsity: f64,
    pub threshold: f32,
}

impl PruningImpact {
    pub fn remaining(&self) -> usize {
        self.total_elements - self.pruned_elements
    }

    pub fn compression_ratio(&self) -> f64 {
        if self.remaining() == 0 {
            return f64::INFINITY;
        }
        self.total_elements as f64 / self.remaining() as f64
    }
}

/// Find the magnitude threshold that achieves a target sparsity level.
pub fn threshold_for_sparsity(weights: &[f32], target_sparsity: f64) -> f32 {
    if weights.is_empty() || target_sparsity <= 0.0 {
        return 0.0;
    }
    if target_sparsity >= 1.0 {
        return f32::INFINITY;
    }

    let mut magnitudes: Vec<f32> = weights.iter().map(|w| w.abs()).collect();
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((target_sparsity * magnitudes.len() as f64) as usize).min(magnitudes.len() - 1);
    magnitudes[idx]
}

/// Layer-level sensitivity estimate (higher = more sensitive to pruning).
pub fn layer_sensitivity(weights: &[f32]) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }
    let mean: f64 = weights.iter().map(|&w| w as f64).sum::<f64>() / weights.len() as f64;
    let variance: f64 =
        weights.iter().map(|&w| (w as f64 - mean).powi(2)).sum::<f64>() / weights.len() as f64;
    let std_dev = variance.sqrt();

    // Coefficient of variation — higher means weights are more spread out
    // (more sensitive to magnitude-based pruning)
    if mean.abs() < 1e-10 { std_dev } else { std_dev / mean.abs() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_weights() {
        let weights = vec![0.0, 1.0, -1.0, 0.5, -0.5];
        let stats = analyze_weights("test", &weights);
        assert_eq!(stats.num_elements, 5);
        assert_eq!(stats.num_zeros, 1);
        assert!((stats.sparsity() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_empty_weights() {
        let stats = analyze_weights("empty", &[]);
        assert_eq!(stats.num_elements, 0);
        assert_eq!(stats.sparsity(), 0.0);
    }

    #[test]
    fn test_pruning_impact() {
        let weights = vec![0.01, 0.1, 1.0, -0.01, -0.1, -1.0];
        let impact = pruning_impact(&weights, 0.05);
        assert_eq!(impact.pruned_elements, 2);
        assert_eq!(impact.remaining(), 4);
    }

    #[test]
    fn test_compression_ratio() {
        let impact = PruningImpact {
            total_elements: 100,
            pruned_elements: 75,
            sparsity: 0.75,
            threshold: 0.1,
        };
        assert!((impact.compression_ratio() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_for_sparsity() {
        let weights: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let threshold = threshold_for_sparsity(&weights, 0.5);
        // ~50% of values should be below this threshold
        let below = weights.iter().filter(|&&w| w.abs() < threshold).count();
        assert!((45..=55).contains(&below));
    }

    #[test]
    fn test_layer_sensitivity() {
        // Uniform weights — low sensitivity
        let uniform = vec![1.0; 100];
        let s1 = layer_sensitivity(&uniform);
        // Spread weights — higher sensitivity
        let spread: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let s2 = layer_sensitivity(&spread);
        assert!(s2 > s1);
    }

    #[test]
    fn test_percentiles() {
        let weights: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let stats = analyze_weights("test", &weights);
        assert!(stats.percentiles.p50 > 40.0 && stats.percentiles.p50 < 60.0);
        assert!(stats.percentiles.p90 > stats.percentiles.p50);
    }

    #[test]
    fn test_density() {
        let stats = analyze_weights("test", &[0.0, 0.0, 1.0, 2.0]);
        assert!((stats.density() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_all_zeros() {
        let weights = vec![0.0; 10];
        let stats = analyze_weights("zeros", &weights);
        assert!((stats.sparsity() - 1.0).abs() < 1e-6);
        assert_eq!(stats.mean_magnitude, 0.0);
    }

    #[test]
    fn test_pruning_impact_all_pruned() {
        let weights = vec![0.001, 0.002, 0.003];
        let impact = pruning_impact(&weights, 1.0);
        assert_eq!(impact.pruned_elements, 3);
    }

    #[test]
    fn test_threshold_edge_cases() {
        assert_eq!(threshold_for_sparsity(&[], 0.5), 0.0);
        assert_eq!(threshold_for_sparsity(&[1.0], 0.0), 0.0);
    }

    #[test]
    fn test_max_magnitude() {
        let stats = analyze_weights("test", &[-5.0, 3.0, 1.0]);
        assert!((stats.max_magnitude - 5.0).abs() < 1e-6);
    }
}
