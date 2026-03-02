//! Cross-validation framework for comparing OpenCL kernel outputs against
//! CPU golden reference vectors.
//!
//! Provides deterministic golden vector generation, multiple tolerance modes,
//! and detailed error metrics for systematic kernel validation.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Deterministic PRNG (no external deps)
// ---------------------------------------------------------------------------

/// Xorshift64 PRNG — fast, deterministic, seedable.
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Map PRNG output to f32 in [-1, 1].
fn deterministic_f32(state: &mut u64) -> f32 {
    (xorshift64(state) as f32) / (u64::MAX as f32) * 2.0 - 1.0
}

/// Fill a slice with deterministic f32 values from the given seed.
fn fill_deterministic(buf: &mut [f32], seed: u64) {
    let mut state = seed;
    for v in buf.iter_mut() {
        *v = deterministic_f32(&mut state);
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A golden reference vector with known-correct input/output pairs.
#[derive(Clone, Debug)]
pub struct GoldenVector {
    pub name: String,
    pub input: Vec<f32>,
    pub expected_output: Vec<f32>,
    pub tolerance: f32,
    pub metadata: HashMap<String, String>,
}

/// Detailed result of comparing an actual output against a golden vector.
#[derive(Clone, Debug)]
pub struct CrossValResult {
    pub test_name: String,
    pub passed: bool,
    pub max_error: f32,
    pub mean_error: f32,
    pub rms_error: f32,
    pub outlier_count: usize,
    pub outlier_indices: Vec<usize>,
}

/// Aggregated results for a suite of cross-validation tests.
#[derive(Clone, Debug)]
pub struct CrossValSuite {
    pub name: String,
    pub results: Vec<CrossValResult>,
    pub total_tests: usize,
    pub passed_tests: usize,
}

/// Multiple tolerance modes for numeric comparison.
#[derive(Clone, Debug)]
pub struct ToleranceSpec {
    pub absolute: f32,
    pub relative: f32,
    pub ulp: u32,
}

impl Default for ToleranceSpec {
    fn default() -> Self {
        Self { absolute: 1e-5, relative: 1e-4, ulp: 4 }
    }
}

/// How two floating-point vectors should be compared.
#[derive(Clone, Debug, PartialEq)]
pub enum ComparisonMode {
    Absolute,
    Relative,
    ULP,
    /// Combines all three — passes only when *all* thresholds are met.
    Mixed,
}

/// Errors specific to cross-validation.
#[derive(Clone, Debug, PartialEq)]
pub enum CrossValError {
    ShapeMismatch { expected: usize, actual: usize },
    ToleranceExceeded { max_error: f32, tolerance: f32 },
    NaNDetected { indices: Vec<usize> },
    InfDetected { indices: Vec<usize> },
    EmptyVector,
}

impl fmt::Display for CrossValError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected}, got {actual}")
            }
            Self::ToleranceExceeded { max_error, tolerance } => {
                write!(
                    f,
                    "tolerance exceeded: max_error={max_error}, \
                     tolerance={tolerance}"
                )
            }
            Self::NaNDetected { indices } => {
                write!(f, "NaN detected at indices: {indices:?}")
            }
            Self::InfDetected { indices } => {
                write!(f, "Inf detected at indices: {indices:?}")
            }
            Self::EmptyVector => write!(f, "empty vector"),
        }
    }
}

impl std::error::Error for CrossValError {}

// ---------------------------------------------------------------------------
// Error metric functions
// ---------------------------------------------------------------------------

/// Maximum absolute element-wise error between two equal-length slices.
pub fn compute_max_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual.iter().zip(expected.iter()).map(|(a, e)| (a - e).abs()).fold(0.0_f32, f32::max)
}

/// Mean absolute element-wise error.
pub fn compute_mean_error(actual: &[f32], expected: &[f32]) -> f32 {
    if actual.is_empty() {
        return 0.0;
    }
    let sum: f32 = actual.iter().zip(expected.iter()).map(|(a, e)| (a - e).abs()).sum();
    sum / actual.len() as f32
}

/// Root-mean-square element-wise error.
pub fn compute_rms_error(actual: &[f32], expected: &[f32]) -> f32 {
    if actual.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = actual.iter().zip(expected.iter()).map(|(a, e)| (a - e).powi(2)).sum();
    (sum_sq / actual.len() as f32).sqrt()
}

/// Compute the ULP (units in last place) distance between two f32 values.
pub fn compute_ulp_error(actual: f32, expected: f32) -> u32 {
    if actual.is_nan() || expected.is_nan() {
        return u32::MAX;
    }
    if actual.is_infinite() || expected.is_infinite() {
        if actual == expected {
            return 0;
        }
        return u32::MAX;
    }
    let a_bits = actual.to_bits() as i32;
    let b_bits = expected.to_bits() as i32;
    (a_bits.wrapping_sub(b_bits)).unsigned_abs()
}

/// Indices where the absolute error exceeds `tolerance`.
pub fn find_outliers(actual: &[f32], expected: &[f32], tolerance: f32) -> Vec<usize> {
    actual
        .iter()
        .zip(expected.iter())
        .enumerate()
        .filter(|(_, (a, e))| (*a - *e).abs() > tolerance)
        .map(|(i, _)| i)
        .collect()
}

/// Returns `(nan_indices, inf_indices)` found in `data`.
pub fn check_nan_inf(data: &[f32]) -> (Vec<usize>, Vec<usize>) {
    let mut nans = Vec::new();
    let mut infs = Vec::new();
    for (i, &v) in data.iter().enumerate() {
        if v.is_nan() {
            nans.push(i);
        } else if v.is_infinite() {
            infs.push(i);
        }
    }
    (nans, infs)
}

// ---------------------------------------------------------------------------
// Detailed comparison
// ---------------------------------------------------------------------------

/// Compare `actual` against `expected` using the given tolerance spec,
/// returning a detailed `CrossValResult`.
pub fn compare_vectors(
    actual: &[f32],
    expected: &[f32],
    tolerance: &ToleranceSpec,
) -> CrossValResult {
    let max_error = compute_max_error(actual, expected);
    let mean_error = compute_mean_error(actual, expected);
    let rms_error = compute_rms_error(actual, expected);
    let outlier_indices = find_outliers(actual, expected, tolerance.absolute);
    let outlier_count = outlier_indices.len();
    let passed = outlier_count == 0;

    CrossValResult {
        test_name: String::new(),
        passed,
        max_error,
        mean_error,
        rms_error,
        outlier_count,
        outlier_indices,
    }
}

// ---------------------------------------------------------------------------
// CPU reference golden-vector generators
// ---------------------------------------------------------------------------

/// Naive CPU matrix multiplication: C = A × B  (row-major).
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// CPU softmax: `exp(x_i) / Σ exp(x_j)`.
fn cpu_softmax(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// CPU RMS normalization: `x_i / rms(x)` where `rms = sqrt(mean(x²))`.
fn cpu_rmsnorm(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let mean_sq: f32 = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let rms = (mean_sq + 1e-6).sqrt();
    input.iter().map(|&v| v / rms).collect()
}

/// Simplified single-head attention: softmax(Q·Kᵀ / √d) · V.
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    // scores = Q · Kᵀ  (seq_len × seq_len)
    let mut scores = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }
    // row-wise softmax
    for i in 0..seq_len {
        let row = &mut scores[i * seq_len..(i + 1) * seq_len];
        let sm = cpu_softmax(row);
        row.copy_from_slice(&sm);
    }
    // output = scores · V  (seq_len × head_dim)
    cpu_matmul(&scores, v, seq_len, head_dim, seq_len)
}

/// Generate a deterministic golden vector for matrix multiplication.
pub fn generate_golden_matmul(m: usize, n: usize, k: usize, seed: u64) -> GoldenVector {
    let mut a = vec![0.0_f32; m * k];
    let mut b = vec![0.0_f32; k * n];
    fill_deterministic(&mut a, seed);
    fill_deterministic(&mut b, seed.wrapping_add(1));
    let c = cpu_matmul(&a, &b, m, n, k);

    let mut input = Vec::with_capacity(a.len() + b.len());
    input.extend_from_slice(&a);
    input.extend_from_slice(&b);

    let mut metadata = HashMap::new();
    metadata.insert("m".into(), m.to_string());
    metadata.insert("n".into(), n.to_string());
    metadata.insert("k".into(), k.to_string());
    metadata.insert("seed".into(), seed.to_string());

    GoldenVector {
        name: format!("matmul_{m}x{n}x{k}_seed{seed}"),
        input,
        expected_output: c,
        tolerance: 1e-4,
        metadata,
    }
}

/// Generate a deterministic golden vector for softmax.
pub fn generate_golden_softmax(len: usize, seed: u64) -> GoldenVector {
    let mut input = vec![0.0_f32; len];
    fill_deterministic(&mut input, seed);
    let output = cpu_softmax(&input);

    let mut metadata = HashMap::new();
    metadata.insert("len".into(), len.to_string());
    metadata.insert("seed".into(), seed.to_string());

    GoldenVector {
        name: format!("softmax_{len}_seed{seed}"),
        input,
        expected_output: output,
        tolerance: 1e-6,
        metadata,
    }
}

/// Generate a deterministic golden vector for RMS normalization.
pub fn generate_golden_rmsnorm(len: usize, seed: u64) -> GoldenVector {
    let mut input = vec![0.0_f32; len];
    fill_deterministic(&mut input, seed);
    let output = cpu_rmsnorm(&input);

    let mut metadata = HashMap::new();
    metadata.insert("len".into(), len.to_string());
    metadata.insert("seed".into(), seed.to_string());

    GoldenVector {
        name: format!("rmsnorm_{len}_seed{seed}"),
        input,
        expected_output: output,
        tolerance: 1e-5,
        metadata,
    }
}

/// Generate a deterministic golden vector for single-head attention.
pub fn generate_golden_attention(seq_len: usize, head_dim: usize, seed: u64) -> GoldenVector {
    let elem = seq_len * head_dim;
    let mut q = vec![0.0_f32; elem];
    let mut k = vec![0.0_f32; elem];
    let mut v = vec![0.0_f32; elem];
    fill_deterministic(&mut q, seed);
    fill_deterministic(&mut k, seed.wrapping_add(1));
    fill_deterministic(&mut v, seed.wrapping_add(2));

    let output = cpu_attention(&q, &k, &v, seq_len, head_dim);

    let mut input = Vec::with_capacity(3 * elem);
    input.extend_from_slice(&q);
    input.extend_from_slice(&k);
    input.extend_from_slice(&v);

    let mut metadata = HashMap::new();
    metadata.insert("seq_len".into(), seq_len.to_string());
    metadata.insert("head_dim".into(), head_dim.to_string());
    metadata.insert("seed".into(), seed.to_string());

    GoldenVector {
        name: format!("attention_s{seq_len}_d{head_dim}_seed{seed}"),
        input,
        expected_output: output,
        tolerance: 1e-4,
        metadata,
    }
}

// ---------------------------------------------------------------------------
// Suite runner
// ---------------------------------------------------------------------------

/// Run all golden-vector tests, comparing each golden's `expected_output`
/// against itself (CPU reference self-check).
pub fn run_cross_validation_suite(goldens: &[GoldenVector]) -> CrossValSuite {
    let mut results = Vec::with_capacity(goldens.len());
    let mut passed_tests = 0;

    for g in goldens {
        let tol = ToleranceSpec { absolute: g.tolerance, ..Default::default() };
        let mut r = compare_vectors(&g.expected_output, &g.expected_output, &tol);
        r.test_name.clone_from(&g.name);
        if r.passed {
            passed_tests += 1;
        }
        results.push(r);
    }

    CrossValSuite {
        name: "OpenCL Cross-Validation".into(),
        results,
        total_tests: goldens.len(),
        passed_tests,
    }
}

/// Format a human-readable report for the given suite.
pub fn format_crossval_report(suite: &CrossValSuite) -> String {
    let mut report = String::new();
    report.push_str(&format!(
        "=== {} ===\nTotal: {}  Passed: {}  Failed: {}\n\n",
        suite.name,
        suite.total_tests,
        suite.passed_tests,
        suite.total_tests - suite.passed_tests,
    ));

    for r in &suite.results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        report.push_str(&format!(
            "[{status}] {}\n  max_error={:.2e}  mean_error={:.2e}  \
             rms_error={:.2e}  outliers={}\n",
            r.test_name, r.max_error, r.mean_error, r.rms_error, r.outlier_count,
        ));
    }
    report
}

// ---------------------------------------------------------------------------
// Pre-computed golden vectors (const-like; computed once at test time)
// ---------------------------------------------------------------------------

/// Small 4×4 matmul golden vector (seed=42).
pub fn precomputed_matmul_4x4() -> GoldenVector {
    generate_golden_matmul(4, 4, 4, 42)
}

/// 8-element softmax golden vector (seed=42).
pub fn precomputed_softmax_8() -> GoldenVector {
    generate_golden_softmax(8, 42)
}

/// 16-element RMSNorm golden vector (seed=42).
pub fn precomputed_rmsnorm_16() -> GoldenVector {
    generate_golden_rmsnorm(16, 42)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- error metric correctness -----------------------------------------

    #[test]
    fn test_max_error_known() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.5, 3.0];
        assert!((compute_max_error(&a, &b) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_max_error_identical() {
        let v = [1.0, 2.0, 3.0];
        assert_eq!(compute_max_error(&v, &v), 0.0);
    }

    #[test]
    fn test_mean_error_known() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 4.0];
        let expected = 1.0 / 3.0;
        assert!((compute_mean_error(&a, &b) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_mean_error_identical() {
        let v = [5.0, 6.0];
        assert_eq!(compute_mean_error(&v, &v), 0.0);
    }

    #[test]
    fn test_rms_error_known() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        // rms = sqrt((9+16)/2) = sqrt(12.5)
        let expected = 12.5_f32.sqrt();
        assert!((compute_rms_error(&a, &b) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_rms_error_identical() {
        let v = [1.0, 2.0, 3.0];
        assert_eq!(compute_rms_error(&v, &v), 0.0);
    }

    // -- ULP error --------------------------------------------------------

    #[test]
    fn test_ulp_exact() {
        assert_eq!(compute_ulp_error(1.0, 1.0), 0);
    }

    #[test]
    fn test_ulp_near_exact() {
        let a = 1.0_f32;
        let b = f32::from_bits(a.to_bits() + 1);
        assert_eq!(compute_ulp_error(a, b), 1);
    }

    #[test]
    fn test_ulp_distant() {
        let a = 1.0_f32;
        let b = 2.0_f32;
        assert!(compute_ulp_error(a, b) > 1000);
    }

    #[test]
    fn test_ulp_nan() {
        assert_eq!(compute_ulp_error(f32::NAN, 1.0), u32::MAX);
        assert_eq!(compute_ulp_error(1.0, f32::NAN), u32::MAX);
    }

    // -- outlier detection ------------------------------------------------

    #[test]
    fn test_find_outliers_known() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 5.0, 3.0, 4.5];
        let out = find_outliers(&a, &b, 1.0);
        assert_eq!(out, vec![1]);
    }

    #[test]
    fn test_find_outliers_none() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        assert!(find_outliers(&a, &b, 0.0).is_empty());
    }

    // -- NaN / Inf detection ----------------------------------------------

    #[test]
    fn test_check_nan() {
        let data = [1.0, f32::NAN, 3.0];
        let (nans, infs) = check_nan_inf(&data);
        assert_eq!(nans, vec![1]);
        assert!(infs.is_empty());
    }

    #[test]
    fn test_check_inf() {
        let data = [f32::INFINITY, 2.0, f32::NEG_INFINITY];
        let (nans, infs) = check_nan_inf(&data);
        assert!(nans.is_empty());
        assert_eq!(infs, vec![0, 2]);
    }

    #[test]
    fn test_check_nan_inf_clean() {
        let data = [1.0, 2.0, 3.0];
        let (nans, infs) = check_nan_inf(&data);
        assert!(nans.is_empty());
        assert!(infs.is_empty());
    }

    // -- golden matmul ----------------------------------------------------

    #[test]
    fn test_golden_matmul_self_check() {
        let g = generate_golden_matmul(4, 4, 4, 42);
        assert_eq!(g.expected_output.len(), 16);
        // Re-compute and compare
        let g2 = generate_golden_matmul(4, 4, 4, 42);
        assert_eq!(g.expected_output, g2.expected_output);
    }

    #[test]
    fn test_golden_matmul_identity() {
        // 2×2 identity × vec should give same vec
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [3.0, 7.0, 5.0, 9.0];
        let c = cpu_matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, b);
    }

    #[test]
    fn test_golden_matmul_known_small() {
        // [1,2;3,4] × [5,6;7,8] = [19,22;43,50]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let c = cpu_matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
    }

    // -- golden softmax ---------------------------------------------------

    #[test]
    fn test_golden_softmax_sums_to_one() {
        let g = generate_golden_softmax(8, 42);
        let sum: f32 = g.expected_output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_golden_softmax_all_positive() {
        let g = generate_golden_softmax(16, 99);
        assert!(g.expected_output.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_golden_softmax_uniform() {
        // equal inputs → uniform output
        let input = [0.0; 4];
        let out = cpu_softmax(&input);
        for &v in &out {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    // -- golden RMSNorm ---------------------------------------------------

    #[test]
    fn test_golden_rmsnorm_rms_approx_one() {
        let g = generate_golden_rmsnorm(16, 42);
        let mean_sq: f32 =
            g.expected_output.iter().map(|v| v * v).sum::<f32>() / g.expected_output.len() as f32;
        // Should be close to 1.0 (epsilon causes tiny deviation)
        assert!((mean_sq - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_golden_rmsnorm_length() {
        let g = generate_golden_rmsnorm(32, 7);
        assert_eq!(g.expected_output.len(), 32);
    }

    // -- golden attention -------------------------------------------------

    #[test]
    fn test_golden_attention_output_shape() {
        let g = generate_golden_attention(4, 8, 42);
        assert_eq!(g.expected_output.len(), 4 * 8);
    }

    #[test]
    fn test_golden_attention_output_finite() {
        let g = generate_golden_attention(4, 8, 42);
        assert!(g.expected_output.iter().all(|v| v.is_finite()));
    }

    // -- cross-val suite --------------------------------------------------

    #[test]
    fn test_suite_all_pass() {
        let goldens = vec![
            generate_golden_matmul(4, 4, 4, 42),
            generate_golden_softmax(8, 42),
            generate_golden_rmsnorm(16, 42),
            generate_golden_attention(4, 8, 42),
        ];
        let suite = run_cross_validation_suite(&goldens);
        assert_eq!(suite.passed_tests, suite.total_tests);
    }

    // -- tolerance modes --------------------------------------------------

    #[test]
    fn test_tolerance_absolute_pass() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.001, 2.001, 3.001];
        let tol = ToleranceSpec { absolute: 0.01, ..Default::default() };
        let r = compare_vectors(&a, &b, &tol);
        assert!(r.passed);
    }

    #[test]
    fn test_tolerance_absolute_fail() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.1, 2.0, 3.0];
        let tol = ToleranceSpec { absolute: 0.01, ..Default::default() };
        let r = compare_vectors(&a, &b, &tol);
        assert!(!r.passed);
    }

    #[test]
    fn test_tolerance_relative_field() {
        let tol = ToleranceSpec { absolute: 1e-5, relative: 1e-3, ulp: 4 };
        assert!((tol.relative - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn test_tolerance_ulp_field() {
        let tol = ToleranceSpec { absolute: 1e-5, relative: 1e-4, ulp: 8 };
        assert_eq!(tol.ulp, 8);
    }

    #[test]
    fn test_comparison_mode_mixed_variant() {
        let mode = ComparisonMode::Mixed;
        assert_eq!(mode, ComparisonMode::Mixed);
    }

    // -- report formatting ------------------------------------------------

    #[test]
    fn test_report_contains_header() {
        let suite = CrossValSuite {
            name: "Test Suite".into(),
            results: vec![],
            total_tests: 0,
            passed_tests: 0,
        };
        let report = format_crossval_report(&suite);
        assert!(report.contains("=== Test Suite ==="));
    }

    #[test]
    fn test_report_contains_pass_fail() {
        let goldens = vec![generate_golden_softmax(8, 42)];
        let suite = run_cross_validation_suite(&goldens);
        let report = format_crossval_report(&suite);
        assert!(report.contains("PASS"));
        assert!(report.contains("Passed: 1"));
    }

    // -- edge cases -------------------------------------------------------

    #[test]
    fn test_empty_vectors() {
        let tol = ToleranceSpec::default();
        let r = compare_vectors(&[], &[], &tol);
        assert!(r.passed);
        assert_eq!(r.max_error, 0.0);
    }

    #[test]
    fn test_single_element() {
        let a = [1.0];
        let b = [1.0];
        let tol = ToleranceSpec::default();
        let r = compare_vectors(&a, &b, &tol);
        assert!(r.passed);
    }

    #[test]
    fn test_all_zeros() {
        let z = [0.0; 8];
        let tol = ToleranceSpec::default();
        let r = compare_vectors(&z, &z, &tol);
        assert!(r.passed);
        assert_eq!(r.rms_error, 0.0);
    }

    // -- properties -------------------------------------------------------

    #[test]
    fn test_rms_error_non_negative() {
        let a = [-1.0, 0.5, 3.0];
        let b = [2.0, -0.5, 1.0];
        assert!(compute_rms_error(&a, &b) >= 0.0);
    }

    #[test]
    fn test_max_ge_mean_ge_zero() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.1, 2.2, 2.9, 4.5];
        let max_e = compute_max_error(&a, &b);
        let mean_e = compute_mean_error(&a, &b);
        assert!(max_e >= mean_e);
        assert!(mean_e >= 0.0);
    }

    // -- determinism ------------------------------------------------------

    #[test]
    fn test_deterministic_matmul_same_seed() {
        let g1 = generate_golden_matmul(8, 8, 8, 123);
        let g2 = generate_golden_matmul(8, 8, 8, 123);
        assert_eq!(g1.expected_output, g2.expected_output);
        assert_eq!(g1.input, g2.input);
    }

    #[test]
    fn test_deterministic_softmax_same_seed() {
        let g1 = generate_golden_softmax(32, 77);
        let g2 = generate_golden_softmax(32, 77);
        assert_eq!(g1.expected_output, g2.expected_output);
    }

    #[test]
    fn test_deterministic_different_seed() {
        let g1 = generate_golden_matmul(4, 4, 4, 1);
        let g2 = generate_golden_matmul(4, 4, 4, 2);
        assert_ne!(g1.expected_output, g2.expected_output);
    }

    // -- pre-computed golden vectors --------------------------------------

    #[test]
    fn test_precomputed_matmul_4x4() {
        let g = precomputed_matmul_4x4();
        assert_eq!(g.expected_output.len(), 16);
        assert!(g.expected_output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_precomputed_softmax_8() {
        let g = precomputed_softmax_8();
        let sum: f32 = g.expected_output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_precomputed_rmsnorm_16() {
        let g = precomputed_rmsnorm_16();
        assert_eq!(g.expected_output.len(), 16);
        assert!(g.expected_output.iter().all(|v| v.is_finite()));
    }

    // -- CrossValError display -------------------------------------------

    #[test]
    fn test_error_display_shape_mismatch() {
        let e = CrossValError::ShapeMismatch { expected: 10, actual: 5 };
        let msg = format!("{e}");
        assert!(msg.contains("shape mismatch"));
    }

    #[test]
    fn test_error_display_empty_vector() {
        let e = CrossValError::EmptyVector;
        assert_eq!(format!("{e}"), "empty vector");
    }
}
