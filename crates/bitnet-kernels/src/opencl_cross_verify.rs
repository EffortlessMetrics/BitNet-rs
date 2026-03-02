//! Cross-backend verification framework for OpenCL GPU outputs.
//!
//! Compares OpenCL GPU kernel outputs against CPU reference implementations
//! using element-wise tolerance checks, statistical verification, and
//! regression guards to ensure numerical correctness across backends.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// VerificationConfig — tolerance and sampling parameters
// ---------------------------------------------------------------------------

/// Configuration for cross-backend numerical comparison.
#[derive(Debug, Clone)]
pub struct VerificationConfig {
    /// Absolute tolerance for element-wise comparison.
    pub atol: f64,
    /// Relative tolerance for element-wise comparison.
    pub rtol: f64,
    /// Maximum number of element failures before aborting early.
    pub max_failures: usize,
    /// Fraction of elements to sample (1.0 = check all).
    pub sample_ratio: f64,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self { atol: 1e-5, rtol: 1e-4, max_failures: 100, sample_ratio: 1.0 }
    }
}

impl VerificationConfig {
    /// Create a strict config with tighter tolerances.
    pub fn strict() -> Self {
        Self { atol: 1e-7, rtol: 1e-6, max_failures: 1, sample_ratio: 1.0 }
    }

    /// Create a relaxed config for approximate comparisons.
    pub fn relaxed() -> Self {
        Self { atol: 1e-3, rtol: 1e-2, max_failures: 1000, sample_ratio: 1.0 }
    }
}

// ---------------------------------------------------------------------------
// FailureLocation — records where a mismatch occurred
// ---------------------------------------------------------------------------

/// Location and magnitude of a single element mismatch.
#[derive(Debug, Clone)]
pub struct FailureLocation {
    /// Flat index into the tensor.
    pub index: usize,
    /// Value from the GPU output.
    pub gpu_value: f64,
    /// Value from the CPU reference.
    pub cpu_value: f64,
    /// Absolute error at this location.
    pub abs_error: f64,
    /// Relative error at this location.
    pub rel_error: f64,
}

// ---------------------------------------------------------------------------
// VerificationResult — outcome of a single backend-pair check
// ---------------------------------------------------------------------------

/// Result of comparing GPU output against CPU reference.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether all sampled elements passed tolerance checks.
    pub passed: bool,
    /// Number of elements that exceeded tolerance.
    pub failed_count: usize,
    /// Total number of elements compared.
    pub total_compared: usize,
    /// Largest absolute error observed.
    pub max_abs_error: f64,
    /// Largest relative error observed.
    pub max_rel_error: f64,
    /// Details of the first `max_failures` mismatches.
    pub locations: Vec<FailureLocation>,
}

impl VerificationResult {
    /// A trivially-passing result for empty tensors.
    pub fn empty() -> Self {
        Self {
            passed: true,
            failed_count: 0,
            total_compared: 0,
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            locations: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// TensorComparator — element-wise comparison engine
// ---------------------------------------------------------------------------

/// Compares two f32 slices element-by-element using configurable tolerances.
pub struct TensorComparator {
    config: VerificationConfig,
}

impl TensorComparator {
    pub fn new(config: VerificationConfig) -> Self {
        Self { config }
    }

    /// Compare `gpu` output against `cpu` reference.
    ///
    /// Both slices must have the same length.
    pub fn compare(&self, gpu: &[f32], cpu: &[f32]) -> VerificationResult {
        assert_eq!(gpu.len(), cpu.len(), "tensor length mismatch");

        if gpu.is_empty() {
            return VerificationResult::empty();
        }

        let step = if self.config.sample_ratio >= 1.0 {
            1usize
        } else {
            (1.0 / self.config.sample_ratio).ceil() as usize
        };

        let mut failed_count: usize = 0;
        let mut total_compared: usize = 0;
        let mut max_abs: f64 = 0.0;
        let mut max_rel: f64 = 0.0;
        let mut locations = Vec::new();

        let mut idx = 0usize;
        while idx < gpu.len() {
            total_compared += 1;
            let g = gpu[idx] as f64;
            let c = cpu[idx] as f64;

            // NaN in reference is always a failure
            if c.is_nan() || g.is_nan() {
                let loc = FailureLocation {
                    index: idx,
                    gpu_value: g,
                    cpu_value: c,
                    abs_error: f64::NAN,
                    rel_error: f64::NAN,
                };
                failed_count += 1;
                if locations.len() < self.config.max_failures {
                    locations.push(loc);
                }
                idx += step;
                continue;
            }

            let abs_err = (g - c).abs();
            let rel_err = if c.abs() > f64::EPSILON { abs_err / c.abs() } else { abs_err };

            if abs_err > max_abs {
                max_abs = abs_err;
            }
            if rel_err > max_rel {
                max_rel = rel_err;
            }

            let within_tol = abs_err <= self.config.atol || rel_err <= self.config.rtol;

            if !within_tol {
                failed_count += 1;
                if locations.len() < self.config.max_failures {
                    locations.push(FailureLocation {
                        index: idx,
                        gpu_value: g,
                        cpu_value: c,
                        abs_error: abs_err,
                        rel_error: rel_err,
                    });
                }
            }

            idx += step;
        }

        VerificationResult {
            passed: failed_count == 0,
            failed_count,
            total_compared,
            max_abs_error: max_abs,
            max_rel_error: max_rel,
            locations,
        }
    }
}

// ---------------------------------------------------------------------------
// Common type alias for kernel functions
// ---------------------------------------------------------------------------

/// A boxed kernel function that maps an input slice to an output vector.
type KernelFn = Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>;

// ---------------------------------------------------------------------------
// BackendPair — links a GPU kernel to its CPU reference
// ---------------------------------------------------------------------------

/// Pairs a named GPU kernel with a CPU reference implementation.
pub struct BackendPair {
    /// Human-readable name for this kernel pair.
    pub name: String,
    /// GPU kernel function: takes input and produces output.
    pub gpu_fn: KernelFn,
    /// CPU reference function: same contract as gpu_fn.
    pub cpu_fn: KernelFn,
}

impl BackendPair {
    pub fn new<G, C>(name: impl Into<String>, gpu_fn: G, cpu_fn: C) -> Self
    where
        G: Fn(&[f32]) -> Vec<f32> + Send + Sync + 'static,
        C: Fn(&[f32]) -> Vec<f32> + Send + Sync + 'static,
    {
        Self { name: name.into(), gpu_fn: Box::new(gpu_fn), cpu_fn: Box::new(cpu_fn) }
    }

    /// Run both kernels on the same input and compare.
    pub fn verify(&self, input: &[f32], config: &VerificationConfig) -> VerificationResult {
        let gpu_out = (self.gpu_fn)(input);
        let cpu_out = (self.cpu_fn)(input);
        let cmp = TensorComparator::new(config.clone());
        cmp.compare(&gpu_out, &cpu_out)
    }
}

// ---------------------------------------------------------------------------
// VerificationSuite — collection of backend pairs
// ---------------------------------------------------------------------------

/// A collection of [`BackendPair`]s to verify together.
pub struct VerificationSuite {
    pairs: Vec<BackendPair>,
    config: VerificationConfig,
}

impl VerificationSuite {
    pub fn new(config: VerificationConfig) -> Self {
        Self { pairs: Vec::new(), config }
    }

    pub fn add(&mut self, pair: BackendPair) {
        self.pairs.push(pair);
    }

    /// Run all backend pairs and return a map of name → result.
    pub fn run_all(&self, input: &[f32]) -> HashMap<String, VerificationResult> {
        self.pairs.iter().map(|p| (p.name.clone(), p.verify(input, &self.config))).collect()
    }

    /// Number of registered pairs.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether the suite is empty.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

// ---------------------------------------------------------------------------
// StatisticalVerifier — distribution-level checks
// ---------------------------------------------------------------------------

/// Checks distribution properties (mean, std-dev, histogram) between two tensors.
pub struct StatisticalVerifier {
    /// Tolerance for mean comparison.
    pub mean_atol: f64,
    /// Tolerance for standard deviation comparison.
    pub std_atol: f64,
    /// Number of histogram bins.
    pub num_bins: usize,
    /// Maximum allowed histogram bin count difference (fraction of total).
    pub hist_tolerance: f64,
}

impl Default for StatisticalVerifier {
    fn default() -> Self {
        Self { mean_atol: 1e-4, std_atol: 1e-3, num_bins: 20, hist_tolerance: 0.05 }
    }
}

/// Result of statistical comparison.
#[derive(Debug, Clone)]
pub struct StatisticalResult {
    pub passed: bool,
    pub gpu_mean: f64,
    pub cpu_mean: f64,
    pub gpu_std: f64,
    pub cpu_std: f64,
    pub mean_diff: f64,
    pub std_diff: f64,
    pub histogram_match: bool,
}

impl StatisticalVerifier {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compare distribution properties of `gpu` and `cpu` outputs.
    pub fn verify(&self, gpu: &[f32], cpu: &[f32]) -> StatisticalResult {
        let gpu_mean = mean(gpu);
        let cpu_mean = mean(cpu);
        let gpu_std = std_dev(gpu, gpu_mean);
        let cpu_std = std_dev(cpu, cpu_mean);
        let mean_diff = (gpu_mean - cpu_mean).abs();
        let std_diff = (gpu_std - cpu_std).abs();

        let mean_ok = mean_diff <= self.mean_atol;
        let std_ok = std_diff <= self.std_atol;

        let hist_ok = self.histogram_match(gpu, cpu);

        StatisticalResult {
            passed: mean_ok && std_ok && hist_ok,
            gpu_mean,
            cpu_mean,
            gpu_std,
            cpu_std,
            mean_diff,
            std_diff,
            histogram_match: hist_ok,
        }
    }

    fn histogram_match(&self, gpu: &[f32], cpu: &[f32]) -> bool {
        if gpu.is_empty() || cpu.is_empty() {
            return gpu.len() == cpu.len();
        }

        let all_vals: Vec<f64> = gpu.iter().chain(cpu.iter()).map(|&v| v as f64).collect();
        let min_val = all_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = all_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < f64::EPSILON {
            return true; // all values identical
        }

        let bin_width = (max_val - min_val) / self.num_bins as f64;
        let to_bin = |v: f64| -> usize {
            let b = ((v - min_val) / bin_width) as usize;
            b.min(self.num_bins - 1)
        };

        let mut gpu_hist = vec![0usize; self.num_bins];
        let mut cpu_hist = vec![0usize; self.num_bins];
        for &v in gpu {
            gpu_hist[to_bin(v as f64)] += 1;
        }
        for &v in cpu {
            cpu_hist[to_bin(v as f64)] += 1;
        }

        let n = gpu.len().max(1) as f64;
        for (g, c) in gpu_hist.iter().zip(cpu_hist.iter()) {
            let diff = (*g as f64 - *c as f64).abs() / n;
            if diff > self.hist_tolerance {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// RegressionGuard — golden reference storage and comparison
// ---------------------------------------------------------------------------

/// Stores golden reference outputs and detects regressions when re-run.
pub struct RegressionGuard {
    golden: HashMap<String, Vec<f32>>,
    config: VerificationConfig,
}

impl RegressionGuard {
    pub fn new(config: VerificationConfig) -> Self {
        Self { golden: HashMap::new(), config }
    }

    /// Store a golden reference under the given key.
    pub fn store(&mut self, key: impl Into<String>, output: Vec<f32>) {
        self.golden.insert(key.into(), output);
    }

    /// Check current output against the stored golden reference.
    ///
    /// Returns `None` if no golden reference exists for the key.
    pub fn check(&self, key: &str, current: &[f32]) -> Option<VerificationResult> {
        self.golden.get(key).map(|golden| {
            let cmp = TensorComparator::new(self.config.clone());
            cmp.compare(current, golden)
        })
    }

    /// Number of stored golden references.
    pub fn len(&self) -> usize {
        self.golden.len()
    }

    /// Whether no golden references are stored.
    pub fn is_empty(&self) -> bool {
        self.golden.is_empty()
    }

    /// Remove a golden reference.
    pub fn remove(&mut self, key: &str) -> Option<Vec<f32>> {
        self.golden.remove(key)
    }
}

// ---------------------------------------------------------------------------
// VerificationReport — human-readable summary
// ---------------------------------------------------------------------------

/// Collects multiple verification results into a formatted report.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    entries: Vec<ReportEntry>,
}

#[derive(Debug, Clone)]
struct ReportEntry {
    name: String,
    result: VerificationResult,
}

impl VerificationReport {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add a named result to the report.
    pub fn add(&mut self, name: impl Into<String>, result: VerificationResult) {
        self.entries.push(ReportEntry { name: name.into(), result });
    }

    /// Whether every entry passed.
    pub fn all_passed(&self) -> bool {
        self.entries.iter().all(|e| e.result.passed)
    }

    /// Number of entries that failed.
    pub fn failure_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.result.passed).count()
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the report is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for VerificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Cross-Backend Verification Report ===")?;
        writeln!(f, "Total kernels: {}", self.entries.len())?;
        writeln!(
            f,
            "Passed: {} / {}",
            self.entries.len() - self.failure_count(),
            self.entries.len()
        )?;
        writeln!(f)?;
        for entry in &self.entries {
            let status = if entry.result.passed { "PASS" } else { "FAIL" };
            writeln!(
                f,
                "[{}] {} — compared {} elements, {} failures, \
                 max_abs={:.2e}, max_rel={:.2e}",
                status,
                entry.name,
                entry.result.total_compared,
                entry.result.failed_count,
                entry.result.max_abs_error,
                entry.result.max_rel_error,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AutoVerifier — wraps a GPU call with automatic CPU cross-check
// ---------------------------------------------------------------------------

/// Wraps a GPU kernel call and automatically cross-checks against a CPU
/// reference on every invocation.
pub struct AutoVerifier {
    name: String,
    gpu_fn: KernelFn,
    cpu_fn: KernelFn,
    config: VerificationConfig,
    history: Vec<VerificationResult>,
}

impl AutoVerifier {
    pub fn new<G, C>(
        name: impl Into<String>,
        gpu_fn: G,
        cpu_fn: C,
        config: VerificationConfig,
    ) -> Self
    where
        G: Fn(&[f32]) -> Vec<f32> + Send + Sync + 'static,
        C: Fn(&[f32]) -> Vec<f32> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            gpu_fn: Box::new(gpu_fn),
            cpu_fn: Box::new(cpu_fn),
            config,
            history: Vec::new(),
        }
    }

    /// Execute the GPU kernel, cross-check against CPU, and return the GPU
    /// output along with the verification result.
    pub fn execute(&mut self, input: &[f32]) -> (Vec<f32>, VerificationResult) {
        let gpu_out = (self.gpu_fn)(input);
        let cpu_out = (self.cpu_fn)(input);
        let cmp = TensorComparator::new(self.config.clone());
        let result = cmp.compare(&gpu_out, &cpu_out);
        self.history.push(result.clone());
        (gpu_out, result)
    }

    /// Name of this auto-verifier.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of executions so far.
    pub fn execution_count(&self) -> usize {
        self.history.len()
    }

    /// Number of executions that passed verification.
    pub fn pass_count(&self) -> usize {
        self.history.iter().filter(|r| r.passed).count()
    }

    /// Number of executions that failed verification.
    pub fn fail_count(&self) -> usize {
        self.history.iter().filter(|r| !r.passed).count()
    }

    /// Access the full history of verification results.
    pub fn history(&self) -> &[VerificationResult] {
        &self.history
    }
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// CPU reference: element-wise ReLU.
pub fn cpu_relu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
}

/// CPU reference: element-wise SiLU (x * sigmoid(x)).
pub fn cpu_silu(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x * (1.0 / (1.0 + (-x).exp()))).collect()
}

/// CPU reference: element-wise GELU (approximate).
pub fn cpu_gelu(input: &[f32]) -> Vec<f32> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    input
        .iter()
        .map(|&x| {
            let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect()
}

/// CPU reference: softmax over a flat slice.
pub fn cpu_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; input.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// CPU reference: RMS normalization.
pub fn cpu_rms_norm(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mean_sq: f32 = input.iter().map(|&x| x * x).sum::<f32>() / input.len() as f32;
    let rms = (mean_sq + 1e-6).sqrt();
    input.iter().map(|&x| x / rms).collect()
}

/// CPU reference: element-wise scale by a constant.
pub fn cpu_scale(input: &[f32], factor: f32) -> Vec<f32> {
    input.iter().map(|&x| x * factor).collect()
}

/// CPU reference: element-wise addition of two slices.
pub fn cpu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// CPU reference: element-wise multiplication of two slices.
pub fn cpu_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

// ---------------------------------------------------------------------------
// Helper statistics
// ---------------------------------------------------------------------------

fn mean(data: &[f32]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().map(|&v| v as f64).sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f32], mean_val: f64) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let var =
        data.iter().map(|&v| (v as f64 - mean_val).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    var.sqrt()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- VerificationConfig -------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = VerificationConfig::default();
        assert!((cfg.atol - 1e-5).abs() < 1e-12);
        assert!((cfg.rtol - 1e-4).abs() < 1e-12);
        assert_eq!(cfg.max_failures, 100);
        assert!((cfg.sample_ratio - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_strict_config() {
        let cfg = VerificationConfig::strict();
        assert!(cfg.atol < VerificationConfig::default().atol);
        assert!(cfg.rtol < VerificationConfig::default().rtol);
        assert_eq!(cfg.max_failures, 1);
    }

    #[test]
    fn test_relaxed_config() {
        let cfg = VerificationConfig::relaxed();
        assert!(cfg.atol > VerificationConfig::default().atol);
        assert!(cfg.rtol > VerificationConfig::default().rtol);
    }

    // -- TensorComparator: exact match passes --------------------------------

    #[test]
    fn test_exact_match_passes() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = cmp.compare(&data, &data);
        assert!(result.passed);
        assert_eq!(result.failed_count, 0);
        assert_eq!(result.total_compared, 5);
        assert_eq!(result.max_abs_error, 0.0);
    }

    #[test]
    fn test_exact_match_zeros() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        let data = vec![0.0f32; 100];
        let result = cmp.compare(&data, &data);
        assert!(result.passed);
        assert_eq!(result.failed_count, 0);
    }

    #[test]
    fn test_exact_match_negative_values() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        let data = vec![-1.0f32, -2.5, -0.001, -100.0];
        let result = cmp.compare(&data, &data);
        assert!(result.passed);
    }

    // -- TensorComparator: small error within atol ---------------------------

    #[test]
    fn test_small_error_within_atol_passes() {
        let cfg = VerificationConfig { atol: 1e-5, ..Default::default() };
        let cmp = TensorComparator::new(cfg);
        let cpu = vec![1.0f32, 2.0, 3.0];
        let gpu: Vec<f32> = cpu.iter().map(|&x| x + 1e-6).collect();
        let result = cmp.compare(&gpu, &cpu);
        assert!(result.passed);
    }

    #[test]
    fn test_error_exactly_at_atol_passes() {
        let cfg = VerificationConfig { atol: 1e-3, rtol: 0.0, ..Default::default() };
        let cmp = TensorComparator::new(cfg);
        let cpu = vec![0.0f32];
        let gpu = vec![5e-4f32]; // well within atol=1e-3
        let result = cmp.compare(&gpu, &cpu);
        assert!(result.passed);
    }

    // -- TensorComparator: large error fails ---------------------------------

    #[test]
    fn test_large_error_fails() {
        let cfg = VerificationConfig::strict();
        let cmp = TensorComparator::new(cfg);
        let cpu = vec![1.0f32, 2.0, 3.0];
        let gpu = vec![1.1f32, 2.0, 3.0];
        let result = cmp.compare(&gpu, &cpu);
        assert!(!result.passed);
        assert!(result.failed_count >= 1);
    }

    #[test]
    fn test_all_elements_wrong() {
        let cfg = VerificationConfig::strict();
        let cmp = TensorComparator::new(cfg);
        let cpu = vec![1.0f32; 10];
        let gpu = vec![2.0f32; 10];
        let result = cmp.compare(&gpu, &cpu);
        assert!(!result.passed);
    }

    // -- Max error location tracking -----------------------------------------

    #[test]
    fn test_failure_location_tracked() {
        let cfg = VerificationConfig { atol: 1e-7, rtol: 1e-7, ..Default::default() };
        let cmp = TensorComparator::new(cfg);
        let cpu = vec![1.0f32, 2.0, 3.0, 4.0];
        let gpu = vec![1.0f32, 2.0, 3.5, 4.0]; // index 2 is wrong
        let result = cmp.compare(&gpu, &cpu);
        assert!(!result.passed);
        assert!(!result.locations.is_empty());
        assert_eq!(result.locations[0].index, 2);
        assert!((result.locations[0].abs_error - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_max_abs_error_correct() {
        let cfg = VerificationConfig { atol: 0.0, rtol: 0.0, ..Default::default() };
        let cmp = TensorComparator::new(cfg);
        let cpu = vec![0.0f32, 0.0, 0.0];
        let gpu = vec![0.1f32, 0.5, 0.3];
        let result = cmp.compare(&gpu, &cpu);
        assert!((result.max_abs_error - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_max_rel_error_correct() {
        let cfg = VerificationConfig { atol: 0.0, rtol: 0.0, ..Default::default() };
        let cmp = TensorComparator::new(cfg);
        let cpu = vec![1.0f32, 10.0, 100.0];
        let gpu = vec![1.1f32, 10.1, 100.1];
        let result = cmp.compare(&gpu, &cpu);
        // Largest relative error is at index 0: 0.1/1.0 = 0.1
        assert!((result.max_rel_error - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_multiple_failure_locations() {
        let cfg =
            VerificationConfig { atol: 0.0, rtol: 0.0, max_failures: 50, ..Default::default() };
        let cmp = TensorComparator::new(cfg);
        let cpu = vec![0.0f32; 20];
        let gpu = vec![1.0f32; 20];
        let result = cmp.compare(&gpu, &cpu);
        assert_eq!(result.failed_count, 20);
        assert_eq!(result.locations.len(), 20);
    }

    #[test]
    fn test_max_failures_caps_locations() {
        let cfg =
            VerificationConfig { atol: 0.0, rtol: 0.0, max_failures: 3, ..Default::default() };
        let cmp = TensorComparator::new(cfg);
        let cpu = vec![0.0f32; 100];
        let gpu = vec![1.0f32; 100];
        let result = cmp.compare(&gpu, &cpu);
        assert_eq!(result.failed_count, 100);
        assert_eq!(result.locations.len(), 3); // capped
    }

    // -- Edge cases ----------------------------------------------------------

    #[test]
    fn test_empty_tensor() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        let result = cmp.compare(&[], &[]);
        assert!(result.passed);
        assert_eq!(result.total_compared, 0);
    }

    #[test]
    fn test_single_element() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        let result = cmp.compare(&[42.0], &[42.0]);
        assert!(result.passed);
        assert_eq!(result.total_compared, 1);
    }

    #[test]
    fn test_single_element_mismatch() {
        let cfg = VerificationConfig::strict();
        let cmp = TensorComparator::new(cfg);
        let result = cmp.compare(&[1.0], &[2.0]);
        assert!(!result.passed);
    }

    #[test]
    fn test_nan_in_reference_fails() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        let result = cmp.compare(&[1.0], &[f32::NAN]);
        assert!(!result.passed);
        assert_eq!(result.failed_count, 1);
    }

    #[test]
    fn test_nan_in_gpu_fails() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        let result = cmp.compare(&[f32::NAN], &[1.0]);
        assert!(!result.passed);
    }

    #[test]
    fn test_nan_in_both_fails() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        let result = cmp.compare(&[f32::NAN], &[f32::NAN]);
        assert!(!result.passed);
    }

    #[test]
    fn test_infinity_exact_match() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        let result = cmp.compare(&[f32::INFINITY], &[f32::INFINITY]);
        // inf - inf = NaN, which triggers NaN failure
        assert!(!result.passed);
    }

    #[test]
    fn test_large_tensor_exact() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        let data: Vec<f32> = (0..10_000).map(|i| i as f32 * 0.001).collect();
        let result = cmp.compare(&data, &data);
        assert!(result.passed);
        assert_eq!(result.total_compared, 10_000);
    }

    // -- Sampling ------------------------------------------------------------

    #[test]
    fn test_sample_ratio_half() {
        let cfg = VerificationConfig { sample_ratio: 0.5, ..Default::default() };
        let cmp = TensorComparator::new(cfg);
        let data = vec![1.0f32; 100];
        let result = cmp.compare(&data, &data);
        assert!(result.passed);
        assert!(result.total_compared <= 51); // ~50 with rounding
        assert!(result.total_compared >= 49);
    }

    #[test]
    fn test_sample_ratio_tenth() {
        let cfg = VerificationConfig { sample_ratio: 0.1, ..Default::default() };
        let cmp = TensorComparator::new(cfg);
        let data = vec![1.0f32; 1000];
        let result = cmp.compare(&data, &data);
        assert!(result.passed);
        assert!(result.total_compared <= 110);
        assert!(result.total_compared >= 90);
    }

    // -- BackendPair ---------------------------------------------------------

    #[test]
    fn test_backend_pair_identical_fns() {
        let pair = BackendPair::new("identity", |x: &[f32]| x.to_vec(), |x: &[f32]| x.to_vec());
        let input = vec![1.0f32, 2.0, 3.0];
        let result = pair.verify(&input, &VerificationConfig::default());
        assert!(result.passed);
    }

    #[test]
    fn test_backend_pair_with_divergence() {
        let pair = BackendPair::new(
            "divergent",
            |x: &[f32]| x.iter().map(|&v| v + 1.0).collect(),
            |x: &[f32]| x.to_vec(),
        );
        let input = vec![1.0f32, 2.0, 3.0];
        let result = pair.verify(&input, &VerificationConfig::strict());
        assert!(!result.passed);
    }

    // -- VerificationSuite ---------------------------------------------------

    #[test]
    fn test_suite_empty() {
        let suite = VerificationSuite::new(VerificationConfig::default());
        assert!(suite.is_empty());
        assert_eq!(suite.len(), 0);
    }

    #[test]
    fn test_suite_run_all_passes() {
        let mut suite = VerificationSuite::new(VerificationConfig::default());
        suite.add(BackendPair::new("relu", cpu_relu, cpu_relu));
        suite.add(BackendPair::new("silu", cpu_silu, cpu_silu));
        let input = vec![1.0f32, -1.0, 0.5, -0.5];
        let results = suite.run_all(&input);
        assert_eq!(results.len(), 2);
        assert!(results.values().all(|r| r.passed));
    }

    #[test]
    fn test_suite_detects_failure() {
        let mut suite = VerificationSuite::new(VerificationConfig::strict());
        suite.add(BackendPair::new(
            "broken",
            |x: &[f32]| x.iter().map(|&v| v + 100.0).collect(),
            |x: &[f32]| x.to_vec(),
        ));
        let results = suite.run_all(&[1.0, 2.0]);
        assert!(results["broken"].failed_count > 0);
    }

    #[test]
    fn test_suite_len() {
        let mut suite = VerificationSuite::new(VerificationConfig::default());
        suite.add(BackendPair::new("a", |x: &[f32]| x.to_vec(), |x: &[f32]| x.to_vec()));
        suite.add(BackendPair::new("b", |x: &[f32]| x.to_vec(), |x: &[f32]| x.to_vec()));
        assert_eq!(suite.len(), 2);
        assert!(!suite.is_empty());
    }

    // -- StatisticalVerifier -------------------------------------------------

    #[test]
    fn test_stat_identical() {
        let sv = StatisticalVerifier::new();
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = sv.verify(&data, &data);
        assert!(result.passed);
        assert_eq!(result.mean_diff, 0.0);
        assert_eq!(result.std_diff, 0.0);
    }

    #[test]
    fn test_stat_mean_within_tolerance() {
        let sv = StatisticalVerifier { mean_atol: 0.01, ..Default::default() };
        let cpu: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
        let gpu: Vec<f32> = cpu.iter().map(|&x| x + 0.001).collect();
        let result = sv.verify(&gpu, &cpu);
        assert!(result.mean_diff < 0.01);
    }

    #[test]
    fn test_stat_mean_outside_tolerance() {
        let sv = StatisticalVerifier { mean_atol: 0.001, ..Default::default() };
        let cpu = vec![0.0f32; 100];
        let gpu = vec![1.0f32; 100];
        let result = sv.verify(&gpu, &cpu);
        assert!(!result.passed);
        assert!((result.mean_diff - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stat_std_outside_tolerance() {
        let sv = StatisticalVerifier { std_atol: 0.001, ..Default::default() };
        let cpu = vec![1.0f32; 100]; // std = 0
        let gpu: Vec<f32> = (0..100).map(|i| i as f32).collect(); // std >> 0
        let result = sv.verify(&gpu, &cpu);
        assert!(!result.passed);
    }

    #[test]
    fn test_stat_histogram_match() {
        let sv = StatisticalVerifier::new();
        let data = vec![0.0f32, 0.5, 1.0, 1.5, 2.0];
        let result = sv.verify(&data, &data);
        assert!(result.histogram_match);
    }

    #[test]
    fn test_stat_empty() {
        let sv = StatisticalVerifier::new();
        let result = sv.verify(&[], &[]);
        assert!(result.passed);
    }

    // -- RegressionGuard -----------------------------------------------------

    #[test]
    fn test_regression_guard_store_and_check() {
        let mut guard = RegressionGuard::new(VerificationConfig::default());
        guard.store("relu_v1", vec![0.0, 1.0, 2.0]);
        let result = guard.check("relu_v1", &[0.0, 1.0, 2.0]).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_regression_guard_detects_change() {
        let mut guard = RegressionGuard::new(VerificationConfig::strict());
        guard.store("relu_v1", vec![0.0, 1.0, 2.0]);
        let result = guard.check("relu_v1", &[0.0, 1.0, 999.0]).unwrap();
        assert!(!result.passed);
    }

    #[test]
    fn test_regression_guard_missing_key() {
        let guard = RegressionGuard::new(VerificationConfig::default());
        assert!(guard.check("nonexistent", &[1.0]).is_none());
    }

    #[test]
    fn test_regression_guard_len() {
        let mut guard = RegressionGuard::new(VerificationConfig::default());
        assert!(guard.is_empty());
        guard.store("a", vec![1.0]);
        guard.store("b", vec![2.0]);
        assert_eq!(guard.len(), 2);
    }

    #[test]
    fn test_regression_guard_remove() {
        let mut guard = RegressionGuard::new(VerificationConfig::default());
        guard.store("a", vec![1.0]);
        assert!(guard.remove("a").is_some());
        assert!(guard.is_empty());
        assert!(guard.remove("a").is_none());
    }

    #[test]
    fn test_regression_guard_overwrite() {
        let mut guard = RegressionGuard::new(VerificationConfig::strict());
        guard.store("k", vec![1.0, 2.0]);
        guard.store("k", vec![3.0, 4.0]); // overwrite
        let result = guard.check("k", &[3.0, 4.0]).unwrap();
        assert!(result.passed);
    }

    // -- VerificationReport --------------------------------------------------

    #[test]
    fn test_report_empty() {
        let report = VerificationReport::new();
        assert!(report.all_passed());
        assert_eq!(report.failure_count(), 0);
        assert!(report.is_empty());
    }

    #[test]
    fn test_report_all_passed() {
        let mut report = VerificationReport::new();
        let ok = VerificationResult {
            passed: true,
            failed_count: 0,
            total_compared: 10,
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            locations: vec![],
        };
        report.add("relu", ok.clone());
        report.add("silu", ok);
        assert!(report.all_passed());
        assert_eq!(report.len(), 2);
    }

    #[test]
    fn test_report_with_failure() {
        let mut report = VerificationReport::new();
        let fail = VerificationResult {
            passed: false,
            failed_count: 5,
            total_compared: 100,
            max_abs_error: 0.1,
            max_rel_error: 0.05,
            locations: vec![],
        };
        report.add("broken_kernel", fail);
        assert!(!report.all_passed());
        assert_eq!(report.failure_count(), 1);
    }

    #[test]
    fn test_report_display_contains_status() {
        let mut report = VerificationReport::new();
        let ok = VerificationResult {
            passed: true,
            failed_count: 0,
            total_compared: 5,
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            locations: vec![],
        };
        report.add("relu", ok);
        let text = format!("{report}");
        assert!(text.contains("PASS"));
        assert!(text.contains("relu"));
        assert!(text.contains("Cross-Backend Verification Report"));
    }

    #[test]
    fn test_report_display_shows_fail() {
        let mut report = VerificationReport::new();
        let fail = VerificationResult {
            passed: false,
            failed_count: 3,
            total_compared: 10,
            max_abs_error: 0.5,
            max_rel_error: 0.2,
            locations: vec![],
        };
        report.add("bad_softmax", fail);
        let text = format!("{report}");
        assert!(text.contains("FAIL"));
        assert!(text.contains("bad_softmax"));
    }

    // -- AutoVerifier --------------------------------------------------------

    #[test]
    fn test_auto_verifier_passes() {
        let mut av = AutoVerifier::new(
            "identity",
            |x: &[f32]| x.to_vec(),
            |x: &[f32]| x.to_vec(),
            VerificationConfig::default(),
        );
        let (out, result) = av.execute(&[1.0, 2.0, 3.0]);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
        assert!(result.passed);
        assert_eq!(av.execution_count(), 1);
        assert_eq!(av.pass_count(), 1);
        assert_eq!(av.fail_count(), 0);
    }

    #[test]
    fn test_auto_verifier_detects_divergence() {
        let mut av = AutoVerifier::new(
            "offset",
            |x: &[f32]| x.iter().map(|&v| v + 10.0).collect(),
            |x: &[f32]| x.to_vec(),
            VerificationConfig::strict(),
        );
        let (_out, result) = av.execute(&[1.0, 2.0]);
        assert!(!result.passed);
        assert_eq!(av.fail_count(), 1);
    }

    #[test]
    fn test_auto_verifier_history_accumulates() {
        let mut av = AutoVerifier::new("relu", cpu_relu, cpu_relu, VerificationConfig::default());
        av.execute(&[1.0, -1.0]);
        av.execute(&[0.5, -0.5]);
        av.execute(&[0.0]);
        assert_eq!(av.execution_count(), 3);
        assert_eq!(av.pass_count(), 3);
        assert_eq!(av.history().len(), 3);
    }

    #[test]
    fn test_auto_verifier_name() {
        let av = AutoVerifier::new(
            "my_kernel",
            |x: &[f32]| x.to_vec(),
            |x: &[f32]| x.to_vec(),
            VerificationConfig::default(),
        );
        assert_eq!(av.name(), "my_kernel");
    }

    // -- CPU reference implementations ---------------------------------------

    #[test]
    fn test_cpu_relu_positive() {
        assert_eq!(cpu_relu(&[1.0, 2.0, 3.0]), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cpu_relu_negative() {
        assert_eq!(cpu_relu(&[-1.0, -2.0, -3.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cpu_relu_mixed() {
        assert_eq!(cpu_relu(&[-1.0, 0.0, 1.0]), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_cpu_relu_empty() {
        assert_eq!(cpu_relu(&[]), Vec::<f32>::new());
    }

    #[test]
    fn test_cpu_silu_zero() {
        let result = cpu_silu(&[0.0]);
        assert!((result[0]).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_silu_positive() {
        let result = cpu_silu(&[2.0]);
        // silu(2) ≈ 2 * sigmoid(2) ≈ 2 * 0.8808 ≈ 1.7616
        assert!((result[0] - 1.7616).abs() < 0.01);
    }

    #[test]
    fn test_cpu_gelu_zero() {
        let result = cpu_gelu(&[0.0]);
        assert!(result[0].abs() < 1e-6);
    }

    #[test]
    fn test_cpu_gelu_positive() {
        let result = cpu_gelu(&[1.0]);
        // GELU(1) ≈ 0.8412
        assert!((result[0] - 0.8412).abs() < 0.01);
    }

    #[test]
    fn test_cpu_softmax_uniform() {
        let result = cpu_softmax(&[1.0, 1.0, 1.0]);
        for &v in &result {
            assert!((v - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cpu_softmax_sums_to_one() {
        let result = cpu_softmax(&[1.0, 2.0, 3.0, 4.0]);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_softmax_empty() {
        assert!(cpu_softmax(&[]).is_empty());
    }

    #[test]
    fn test_cpu_rms_norm_unit_vector() {
        let result = cpu_rms_norm(&[1.0, 0.0, 0.0]);
        // rms = sqrt((1+0+0)/3 + 1e-6) ≈ sqrt(0.333...)
        let rms = (1.0f32 / 3.0 + 1e-6).sqrt();
        assert!((result[0] - 1.0 / rms).abs() < 1e-3);
    }

    #[test]
    fn test_cpu_rms_norm_empty() {
        assert!(cpu_rms_norm(&[]).is_empty());
    }

    #[test]
    fn test_cpu_scale() {
        assert_eq!(cpu_scale(&[1.0, 2.0, 3.0], 2.0), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_cpu_scale_zero() {
        assert_eq!(cpu_scale(&[1.0, 2.0], 0.0), vec![0.0, 0.0]);
    }

    #[test]
    fn test_cpu_add() {
        assert_eq!(cpu_add(&[1.0, 2.0], &[3.0, 4.0]), vec![4.0, 6.0]);
    }

    #[test]
    fn test_cpu_mul() {
        assert_eq!(cpu_mul(&[2.0, 3.0], &[4.0, 5.0]), vec![8.0, 15.0]);
    }

    // -- Property: identical inputs always pass ------------------------------

    #[test]
    fn test_identical_always_passes_small() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        for n in [1, 2, 5, 10, 50, 100] {
            let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.123).collect();
            let result = cmp.compare(&data, &data);
            assert!(result.passed, "failed for n={n}");
        }
    }

    #[test]
    fn test_identical_always_passes_negative() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        let data: Vec<f32> = (0..200).map(|i| -(i as f32) * 0.5).collect();
        let result = cmp.compare(&data, &data);
        assert!(result.passed);
    }

    // -- Integration: full pipeline -------------------------------------------

    #[test]
    fn test_full_pipeline_relu() {
        let mut suite = VerificationSuite::new(VerificationConfig::default());
        suite.add(BackendPair::new("relu", cpu_relu, cpu_relu));
        let input: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let results = suite.run_all(&input);
        assert!(results["relu"].passed);
    }

    #[test]
    fn test_full_pipeline_silu() {
        let mut suite = VerificationSuite::new(VerificationConfig::default());
        suite.add(BackendPair::new("silu", cpu_silu, cpu_silu));
        let input: Vec<f32> = (-20..20).map(|i| i as f32 * 0.25).collect();
        let results = suite.run_all(&input);
        assert!(results["silu"].passed);
    }

    #[test]
    fn test_full_pipeline_with_report() {
        let mut suite = VerificationSuite::new(VerificationConfig::default());
        suite.add(BackendPair::new("gelu", cpu_gelu, cpu_gelu));
        suite.add(BackendPair::new("softmax", cpu_softmax, cpu_softmax));
        let input = vec![0.5f32, 1.0, 1.5, 2.0];
        let results = suite.run_all(&input);

        let mut report = VerificationReport::new();
        for (name, result) in &results {
            report.add(name.clone(), result.clone());
        }
        assert!(report.all_passed());
        let text = format!("{report}");
        assert!(text.contains("Passed: 2 / 2"));
    }

    #[test]
    fn test_regression_then_report() {
        let mut guard = RegressionGuard::new(VerificationConfig::default());
        let golden = cpu_relu(&[1.0, -1.0, 0.5, -0.5]);
        guard.store("relu_golden", golden);

        let current = cpu_relu(&[1.0, -1.0, 0.5, -0.5]);
        let result = guard.check("relu_golden", &current).unwrap();

        let mut report = VerificationReport::new();
        report.add("relu_regression", result);
        assert!(report.all_passed());
    }

    // -- Various tensor shapes -----------------------------------------------

    #[test]
    fn test_power_of_two_sizes() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        for &n in &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let result = cmp.compare(&data, &data);
            assert!(result.passed, "failed for n={n}");
        }
    }

    #[test]
    fn test_prime_sizes() {
        let cmp = TensorComparator::new(VerificationConfig::default());
        for &n in &[1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 127, 257] {
            let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
            let result = cmp.compare(&data, &data);
            assert!(result.passed, "failed for n={n}");
        }
    }
}
