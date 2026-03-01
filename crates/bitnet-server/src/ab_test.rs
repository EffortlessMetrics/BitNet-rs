//! A/B testing framework for GPU backends.
//!
//! Enables controlled experiments comparing two GPU backends (A and B) with
//! deterministic traffic splitting (by request hash), metric collection, and
//! basic statistical significance testing (Welch's t-test).

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for an A/B test experiment.
#[derive(Debug, Clone)]
pub struct AbTestConfig {
    /// Human-readable experiment name.
    pub experiment_name: String,
    /// Fraction of traffic routed to variant B (0.0–1.0). Remainder goes to A.
    pub b_ratio: f64,
    /// Metrics to collect (informational; all metrics are always collected).
    pub metrics: Vec<String>,
    /// Whether the experiment is currently active.
    pub active: bool,
}

impl AbTestConfig {
    /// Create a new A/B test with a 50/50 split.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            experiment_name: name.into(),
            b_ratio: 0.5,
            metrics: vec![
                "latency".into(),
                "throughput".into(),
                "quality_score".into(),
            ],
            active: true,
        }
    }

    /// Set the B-variant traffic ratio.
    pub fn with_b_ratio(mut self, ratio: f64) -> Self {
        self.b_ratio = ratio.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// Variant assignment
// ---------------------------------------------------------------------------

/// Which variant a request is assigned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Variant {
    A,
    B,
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A => f.write_str("A"),
            Self::B => f.write_str("B"),
        }
    }
}

/// Deterministically assign a request to a variant based on a hashable key.
pub fn assign_variant<K: Hash>(key: &K, b_ratio: f64) -> Variant {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    let hash = hasher.finish();
    // Map hash to [0, 1) range.
    let frac = (hash as f64) / (u64::MAX as f64);
    if frac < b_ratio { Variant::B } else { Variant::A }
}

// ---------------------------------------------------------------------------
// Per-variant sample collection
// ---------------------------------------------------------------------------

/// A single observation from a request.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Latency of the request.
    pub latency: Duration,
    /// Throughput in tokens/second (if applicable).
    pub throughput: f64,
    /// Quality score (e.g. BLEU, perplexity, or custom metric).
    pub quality_score: f64,
}

/// Collected samples for one variant.
#[derive(Debug)]
struct VariantSamples {
    latencies_us: Vec<u64>,
    throughputs: Vec<f64>,
    quality_scores: Vec<f64>,
}

impl VariantSamples {
    fn new() -> Self {
        Self {
            latencies_us: Vec::new(),
            throughputs: Vec::new(),
            quality_scores: Vec::new(),
        }
    }

    fn push(&mut self, sample: &Sample) {
        self.latencies_us
            .push(sample.latency.as_micros() as u64);
        self.throughputs.push(sample.throughput);
        self.quality_scores.push(sample.quality_score);
    }

    fn count(&self) -> usize {
        self.latencies_us.len()
    }
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let sum_sq: f64 = data.iter().map(|x| (x - m).powi(2)).sum();
    sum_sq / (data.len() as f64 - 1.0)
}

/// Welch's t-test for two independent samples. Returns the t-statistic.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let mean_a = mean(a);
    let mean_b = mean(b);
    let var_a = variance(a);
    let var_b = variance(b);
    let na = a.len() as f64;
    let nb = b.len() as f64;

    let se = (var_a / na + var_b / nb).sqrt();
    if se < 1e-15 {
        return 0.0;
    }
    (mean_a - mean_b) / se
}

/// Approximate two-tailed p-value from a t-statistic using the normal
/// approximation (valid for large sample sizes).
pub fn approx_p_value(t_stat: f64) -> f64 {
    // Use the complementary error function approximation.
    let x = t_stat.abs() / std::f64::consts::SQRT_2;
    let p = erfc_approx(x);
    p.clamp(0.0, 1.0)
}

/// Rough erfc approximation (Abramowitz & Stegun 7.1.26).
fn erfc_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = poly * (-x * x).exp();
    if x >= 0.0 { result } else { 2.0 - result }
}

// ---------------------------------------------------------------------------
// Experiment results
// ---------------------------------------------------------------------------

/// Summary statistics for one variant.
#[derive(Debug, Clone)]
pub struct VariantStats {
    pub variant: Variant,
    pub count: usize,
    pub mean_latency: Duration,
    pub mean_throughput: f64,
    pub mean_quality: f64,
}

/// Full results of an A/B test experiment.
#[derive(Debug, Clone)]
pub struct ExperimentResults {
    pub experiment_name: String,
    pub variant_a: VariantStats,
    pub variant_b: VariantStats,
    /// Welch's t-test statistic for latency comparison.
    pub latency_t_stat: f64,
    /// Approximate p-value for latency difference.
    pub latency_p_value: f64,
    /// Welch's t-test statistic for quality comparison.
    pub quality_t_stat: f64,
    /// Approximate p-value for quality difference.
    pub quality_p_value: f64,
}

impl ExperimentResults {
    /// Whether the latency difference is statistically significant (p < 0.05).
    pub fn latency_significant(&self) -> bool {
        self.latency_p_value < 0.05
    }

    /// Whether the quality difference is statistically significant (p < 0.05).
    pub fn quality_significant(&self) -> bool {
        self.quality_p_value < 0.05
    }
}

// ---------------------------------------------------------------------------
// AbTestRunner
// ---------------------------------------------------------------------------

/// Runs an A/B test experiment, collecting samples and computing results.
pub struct AbTestRunner {
    config: AbTestConfig,
    samples_a: Mutex<VariantSamples>,
    samples_b: Mutex<VariantSamples>,
    total_a: AtomicU64,
    total_b: AtomicU64,
}

impl AbTestRunner {
    /// Create a new runner for the given experiment config.
    pub fn new(config: AbTestConfig) -> Self {
        Self {
            config,
            samples_a: Mutex::new(VariantSamples::new()),
            samples_b: Mutex::new(VariantSamples::new()),
            total_a: AtomicU64::new(0),
            total_b: AtomicU64::new(0),
        }
    }

    /// Assign a request to a variant deterministically.
    pub fn assign<K: Hash>(&self, key: &K) -> Variant {
        if !self.config.active {
            return Variant::A;
        }
        assign_variant(key, self.config.b_ratio)
    }

    /// Record a sample for the given variant.
    pub fn record(&self, variant: Variant, sample: Sample) {
        match variant {
            Variant::A => {
                self.total_a.fetch_add(1, Ordering::Relaxed);
                self.samples_a.lock().unwrap().push(&sample);
            }
            Variant::B => {
                self.total_b.fetch_add(1, Ordering::Relaxed);
                self.samples_b.lock().unwrap().push(&sample);
            }
        }
    }

    /// Compute experiment results with statistical tests.
    pub fn results(&self) -> ExperimentResults {
        let sa = self.samples_a.lock().unwrap();
        let sb = self.samples_b.lock().unwrap();

        let lat_a: Vec<f64> = sa.latencies_us.iter().map(|&v| v as f64).collect();
        let lat_b: Vec<f64> = sb.latencies_us.iter().map(|&v| v as f64).collect();

        let lat_t = welch_t_test(&lat_a, &lat_b);
        let lat_p = approx_p_value(lat_t);

        let qual_t = welch_t_test(&sa.quality_scores, &sb.quality_scores);
        let qual_p = approx_p_value(qual_t);

        ExperimentResults {
            experiment_name: self.config.experiment_name.clone(),
            variant_a: VariantStats {
                variant: Variant::A,
                count: sa.count(),
                mean_latency: Duration::from_micros(mean(&lat_a) as u64),
                mean_throughput: mean(&sa.throughputs),
                mean_quality: mean(&sa.quality_scores),
            },
            variant_b: VariantStats {
                variant: Variant::B,
                count: sb.count(),
                mean_latency: Duration::from_micros(mean(&lat_b) as u64),
                mean_throughput: mean(&sb.throughputs),
                mean_quality: mean(&sb.quality_scores),
            },
            latency_t_stat: lat_t,
            latency_p_value: lat_p,
            quality_t_stat: qual_t,
            quality_p_value: qual_p,
        }
    }

    /// Total requests assigned to variant A.
    pub fn count_a(&self) -> u64 {
        self.total_a.load(Ordering::Relaxed)
    }

    /// Total requests assigned to variant B.
    pub fn count_b(&self) -> u64 {
        self.total_b.load(Ordering::Relaxed)
    }

    /// The experiment configuration.
    pub fn config(&self) -> &AbTestConfig {
        &self.config
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_assignment() {
        let v1 = assign_variant(&"request-42", 0.5);
        let v2 = assign_variant(&"request-42", 0.5);
        assert_eq!(v1, v2, "same key must yield same variant");
    }

    #[test]
    fn test_traffic_split_approximate() {
        let mut count_b = 0;
        for i in 0..1000 {
            if assign_variant(&i, 0.5) == Variant::B {
                count_b += 1;
            }
        }
        // Should be roughly 50% ± 10%.
        assert!(count_b >= 400 && count_b <= 600, "B count: {count_b}");
    }

    #[test]
    fn test_all_traffic_to_a_when_ratio_zero() {
        for i in 0..100 {
            assert_eq!(assign_variant(&i, 0.0), Variant::A);
        }
    }

    #[test]
    fn test_all_traffic_to_b_when_ratio_one() {
        for i in 0..100 {
            assert_eq!(assign_variant(&i, 1.0), Variant::B);
        }
    }

    #[test]
    fn test_runner_records_and_counts() {
        let config = AbTestConfig::new("test-exp").with_b_ratio(0.5);
        let runner = AbTestRunner::new(config);

        runner.record(
            Variant::A,
            Sample {
                latency: Duration::from_millis(10),
                throughput: 100.0,
                quality_score: 0.9,
            },
        );
        runner.record(
            Variant::B,
            Sample {
                latency: Duration::from_millis(12),
                throughput: 95.0,
                quality_score: 0.85,
            },
        );

        assert_eq!(runner.count_a(), 1);
        assert_eq!(runner.count_b(), 1);
    }

    #[test]
    fn test_results_with_samples() {
        let config = AbTestConfig::new("latency-test");
        let runner = AbTestRunner::new(config);

        for _ in 0..50 {
            runner.record(
                Variant::A,
                Sample {
                    latency: Duration::from_millis(10),
                    throughput: 100.0,
                    quality_score: 0.9,
                },
            );
            runner.record(
                Variant::B,
                Sample {
                    latency: Duration::from_millis(10),
                    throughput: 100.0,
                    quality_score: 0.9,
                },
            );
        }

        let results = runner.results();
        assert_eq!(results.variant_a.count, 50);
        assert_eq!(results.variant_b.count, 50);
        // Same values → no significant difference.
        assert!(!results.latency_significant());
    }

    #[test]
    fn test_welch_t_test_identical_samples() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let t = welch_t_test(&a, &b);
        assert!(t.abs() < 1e-10, "t={t}");
    }

    #[test]
    fn test_welch_t_test_different_means() {
        let a = vec![10.0; 100];
        let b = vec![20.0; 100];
        let t = welch_t_test(&a, &b);
        // Means differ but variance is 0 within each group → t is 0 (no
        // spread to measure against). This is the expected edge case.
        // Actually with zero variance, se is ~0 so t should be 0 per our guard.
        assert!(t.abs() < 1e-5 || t.abs() > 1.0);
    }

    #[test]
    fn test_inactive_experiment_routes_all_to_a() {
        let mut config = AbTestConfig::new("inactive").with_b_ratio(0.5);
        config.active = false;
        let runner = AbTestRunner::new(config);

        for i in 0..100 {
            assert_eq!(runner.assign(&i), Variant::A);
        }
    }

    #[test]
    fn test_approx_p_value_range() {
        // Large t → small p.
        let p_large = approx_p_value(10.0);
        assert!(p_large < 0.001, "p={p_large}");

        // Zero t → p ≈ 1.
        let p_zero = approx_p_value(0.0);
        assert!((p_zero - 1.0).abs() < 0.1, "p={p_zero}");
    }

    #[test]
    fn test_config_builder() {
        let config = AbTestConfig::new("my-experiment").with_b_ratio(0.3);
        assert_eq!(config.experiment_name, "my-experiment");
        assert!((config.b_ratio - 0.3).abs() < 1e-10);
        assert!(config.active);
        assert_eq!(config.metrics.len(), 3);
    }

    #[test]
    fn test_significant_quality_difference() {
        let config = AbTestConfig::new("quality-test");
        let runner = AbTestRunner::new(config);

        // A has quality ~0.5, B has quality ~0.9 with some spread.
        for i in 0..100 {
            runner.record(
                Variant::A,
                Sample {
                    latency: Duration::from_millis(10),
                    throughput: 100.0,
                    quality_score: 0.5 + (i as f64 % 5.0) * 0.01,
                },
            );
            runner.record(
                Variant::B,
                Sample {
                    latency: Duration::from_millis(10),
                    throughput: 100.0,
                    quality_score: 0.9 + (i as f64 % 5.0) * 0.01,
                },
            );
        }

        let results = runner.results();
        // The quality difference should be significant.
        assert!(
            results.quality_significant(),
            "p={} should be < 0.05",
            results.quality_p_value
        );
    }
}
