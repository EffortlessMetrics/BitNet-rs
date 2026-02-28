//! Cross-backend kernel test harness.
//!
//! Provides [`KernelTestCase`], [`TestSuite`], and [`TestRunner`] for
//! validating GPU kernel implementations across multiple backends with
//! configurable tolerance modes.

use std::collections::HashMap;
use std::time::Instant;

// ── Core types ───────────────────────────────────────────────────────────

/// A single tensor used as test input or expected output.
#[derive(Debug, Clone)]
pub struct TestTensor {
    /// Dimension sizes, e.g. `[2, 3]` for a 2×3 matrix.
    pub shape: Vec<usize>,
    /// Flat row-major data.
    pub data: Vec<f32>,
    /// Human-readable label for diagnostics.
    pub label: String,
}

/// Tolerance mode for comparing actual vs expected tensors.
#[derive(Debug, Clone)]
pub enum Tolerance {
    /// Bit-exact equality.
    Exact,
    /// Maximum allowed absolute difference per element.
    Absolute(f64),
    /// Maximum allowed relative difference per element.
    Relative(f64),
    /// Maximum allowed ULP (units in the last place) distance.
    UlpDistance(u32),
}

/// A typed parameter value passed to a kernel under test.
#[derive(Debug, Clone)]
pub enum TestParam {
    Int(i64),
    Float(f64),
    Bool(bool),
    IntVec(Vec<i64>),
}

/// A single kernel test case.
#[derive(Debug, Clone)]
pub struct KernelTestCase {
    pub name: String,
    pub kernel_name: String,
    pub inputs: Vec<TestTensor>,
    pub expected_output: TestTensor,
    pub tolerance: Tolerance,
    pub params: HashMap<String, TestParam>,
}

/// Result of running one test case on one backend.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub backend: String,
    pub passed: bool,
    pub actual_output: Option<TestTensor>,
    pub max_error: Option<f64>,
    pub mean_error: Option<f64>,
    pub execution_time_us: u64,
    pub error_message: Option<String>,
}

/// An ordered collection of test cases.
#[derive(Debug, Clone)]
pub struct TestSuite {
    pub name: String,
    pub cases: Vec<KernelTestCase>,
}

/// Runs suites across multiple named backends.
#[derive(Debug)]
pub struct TestRunner {
    pub backends: Vec<String>,
    pub suites: Vec<TestSuite>,
}

/// Cross-backend comparison for a single test case.
#[derive(Debug)]
pub struct ComparisonReport {
    pub test_name: String,
    pub results_by_backend: Vec<(String, TestResult)>,
    pub all_agree: bool,
    pub max_cross_backend_diff: f64,
}

// ── TestTensor ───────────────────────────────────────────────────────────

impl TestTensor {
    /// Create a tensor from existing data.
    ///
    /// # Panics
    /// Panics if `data.len()` does not equal the product of `shape`.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>, label: &str) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "data length {} != shape product {}",
            data.len(),
            expected_len,
        );
        Self { shape, data, label: label.to_string() }
    }

    /// Create a zero-filled tensor.
    pub fn zeros(shape: Vec<usize>, label: &str) -> Self {
        let len: usize = shape.iter().product();
        Self { shape, data: vec![0.0; len], label: label.to_string() }
    }

    /// Create a deterministic pseudo-random tensor using a simple LCG.
    #[allow(clippy::cast_precision_loss)]
    pub fn random(shape: Vec<usize>, seed: u64, label: &str) -> Self {
        let len: usize = shape.iter().product();
        let mut state = seed;
        let data: Vec<f32> = (0..len)
            .map(|_| {
                // LCG: Numerical Recipes constants
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Map to [-1, 1]
                ((state >> 33) as f32 / u32::MAX as f32).mul_add(2.0, -1.0)
            })
            .collect();
        Self { shape, data, label: label.to_string() }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

// ── Tolerance helpers ────────────────────────────────────────────────────

/// Returns the ULP distance between two `f32` values.
///
/// Special cases: if either value is NaN the distance is `u32::MAX`.
const fn ulp_distance(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    let ai = a.to_bits().cast_signed();
    let bi = b.to_bits().cast_signed();
    (ai.wrapping_sub(bi)).unsigned_abs()
}

/// Check whether every element of `actual` is within `tolerance` of
/// `expected`.  Returns `(passed, max_error, mean_error, error_msg)`.
pub fn within_tolerance(
    actual: &[f32],
    expected: &[f32],
    tolerance: &Tolerance,
) -> (bool, f64, f64, Option<String>) {
    if actual.len() != expected.len() {
        return (
            false,
            f64::NAN,
            f64::NAN,
            Some(format!(
                "length mismatch: actual {} vs expected {}",
                actual.len(),
                expected.len(),
            )),
        );
    }

    // Check for NaN / Inf in actual
    for (i, &v) in actual.iter().enumerate() {
        if v.is_nan() {
            return (
                false,
                f64::NAN,
                f64::NAN,
                Some(format!("NaN detected in actual output at index {i}")),
            );
        }
        if v.is_infinite() {
            return (
                false,
                f64::INFINITY,
                f64::INFINITY,
                Some(format!("Inf detected in actual output at index {i}")),
            );
        }
    }

    let mut max_err: f64 = 0.0;
    let mut sum_err: f64 = 0.0;
    let mut passed = true;

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = match tolerance {
            Tolerance::Exact => {
                #[allow(clippy::float_cmp)]
                if a != e {
                    return (
                        false,
                        (f64::from(a) - f64::from(e)).abs(),
                        0.0,
                        Some(format!("exact mismatch at index {i}: {a} != {e}")),
                    );
                }
                0.0
            }
            Tolerance::Absolute(tol) => {
                let diff = (f64::from(a) - f64::from(e)).abs();
                if diff > *tol {
                    passed = false;
                }
                diff
            }
            Tolerance::Relative(tol) => {
                let denom = f64::from(e).abs().max(1e-12);
                let rel = (f64::from(a) - f64::from(e)).abs() / denom;
                if rel > *tol {
                    passed = false;
                }
                rel
            }
            Tolerance::UlpDistance(max_ulp) => {
                let dist = ulp_distance(a, e);
                if dist > *max_ulp {
                    passed = false;
                }
                f64::from(dist)
            }
        };
        if err > max_err {
            max_err = err;
        }
        sum_err += err;
    }

    #[allow(clippy::cast_precision_loss)]
    let mean_err = if actual.is_empty() { 0.0 } else { sum_err / actual.len() as f64 };

    let msg = if passed {
        None
    } else {
        Some(format!(
            "tolerance exceeded: max_error={max_err:.6e}, \
             mean_error={mean_err:.6e}",
        ))
    };
    (passed, max_err, mean_err, msg)
}

// ── KernelTestCase ───────────────────────────────────────────────────────

impl KernelTestCase {
    pub fn new(
        name: &str,
        kernel_name: &str,
        inputs: Vec<TestTensor>,
        expected_output: TestTensor,
        tolerance: Tolerance,
    ) -> Self {
        Self {
            name: name.to_string(),
            kernel_name: kernel_name.to_string(),
            inputs,
            expected_output,
            tolerance,
            params: HashMap::new(),
        }
    }

    /// Add a typed parameter.
    #[must_use]
    pub fn with_param(mut self, key: &str, value: TestParam) -> Self {
        self.params.insert(key.to_string(), value);
        self
    }

    /// Execute this case against `actual_output` produced by a backend.
    pub fn evaluate(
        &self,
        actual_output: TestTensor,
        backend: &str,
        elapsed_us: u64,
    ) -> TestResult {
        if actual_output.shape != self.expected_output.shape {
            return TestResult {
                test_name: self.name.clone(),
                backend: backend.to_string(),
                passed: false,
                actual_output: Some(actual_output.clone()),
                max_error: None,
                mean_error: None,
                execution_time_us: elapsed_us,
                error_message: Some(format!(
                    "shape mismatch: actual {:?} vs expected {:?}",
                    actual_output.shape, self.expected_output.shape,
                )),
            };
        }

        let (passed, max_err, mean_err, msg) =
            within_tolerance(&actual_output.data, &self.expected_output.data, &self.tolerance);

        TestResult {
            test_name: self.name.clone(),
            backend: backend.to_string(),
            passed,
            actual_output: Some(actual_output),
            max_error: Some(max_err),
            mean_error: Some(mean_err),
            execution_time_us: elapsed_us,
            error_message: msg,
        }
    }
}

// ── TestSuite ────────────────────────────────────────────────────────────

impl TestSuite {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), cases: Vec::new() }
    }

    pub fn add_case(&mut self, case: KernelTestCase) {
        self.cases.push(case);
    }

    /// Run all cases by feeding the *expected* output back as actual (CPU
    /// reference).  Real backends would replace this with kernel dispatch.
    pub fn run(&self, backend: &str) -> Vec<TestResult> {
        self.cases
            .iter()
            .map(|tc| {
                let start = Instant::now();
                // Default: echo expected output (reference backend).
                let actual = tc.expected_output.clone();
                #[allow(clippy::cast_possible_truncation)]
                let elapsed = start.elapsed().as_micros() as u64;
                tc.evaluate(actual, backend, elapsed)
            })
            .collect()
    }
}

// ── TestRunner ───────────────────────────────────────────────────────────

impl TestRunner {
    pub const fn new() -> Self {
        Self { backends: Vec::new(), suites: Vec::new() }
    }

    pub fn add_backend(&mut self, name: &str) {
        self.backends.push(name.to_string());
    }

    pub fn add_suite(&mut self, suite: TestSuite) {
        self.suites.push(suite);
    }

    /// Run every suite on every backend and produce comparison reports.
    pub fn run_all(&self) -> Vec<ComparisonReport> {
        let mut reports = Vec::new();

        for suite in &self.suites {
            // Collect results per backend.
            let mut backend_results: Vec<(String, Vec<TestResult>)> = Vec::new();
            for backend in &self.backends {
                backend_results.push((backend.clone(), suite.run(backend)));
            }

            // Build per-test-case comparison reports.
            for (case_idx, case) in suite.cases.iter().enumerate() {
                let mut results_by_backend = Vec::new();
                for (backend, results) in &backend_results {
                    if let Some(r) = results.get(case_idx) {
                        results_by_backend.push((backend.clone(), r.clone()));
                    }
                }

                let (all_agree, max_diff) = cross_backend_diff(&results_by_backend);

                reports.push(ComparisonReport {
                    test_name: case.name.clone(),
                    results_by_backend,
                    all_agree,
                    max_cross_backend_diff: max_diff,
                });
            }
        }

        reports
    }
}

impl Default for TestRunner {
    fn default() -> Self {
        Self::new()
    }
}

// ── ComparisonReport ─────────────────────────────────────────────────────

impl ComparisonReport {
    pub fn summary(&self) -> String {
        let status = if self.all_agree { "PASS" } else { "FAIL" };
        let backends: Vec<&str> = self.results_by_backend.iter().map(|(b, _)| b.as_str()).collect();
        format!(
            "[{status}] {} — backends: [{}], \
             max cross-backend diff: {:.6e}",
            self.test_name,
            backends.join(", "),
            self.max_cross_backend_diff,
        )
    }
}

/// Compute the maximum element-wise difference between any two backends.
fn cross_backend_diff(results: &[(String, TestResult)]) -> (bool, f64) {
    if results.len() < 2 {
        let all_pass = results.iter().all(|(_, r)| r.passed);
        return (all_pass, 0.0);
    }

    let mut all_agree = true;
    let mut max_diff: f64 = 0.0;

    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let (_, ri) = &results[i];
            let (_, rj) = &results[j];

            if ri.passed != rj.passed {
                all_agree = false;
            }

            if let (Some(oi), Some(oj)) = (&ri.actual_output, &rj.actual_output) {
                if oi.data.len() == oj.data.len() {
                    for (&a, &b) in oi.data.iter().zip(&oj.data) {
                        let d = (f64::from(a) - f64::from(b)).abs();
                        if d > max_diff {
                            max_diff = d;
                        }
                    }
                } else {
                    all_agree = false;
                }
            }
        }
    }

    (all_agree, max_diff)
}

// ── Built-in suites ──────────────────────────────────────────────────────

/// Simple 2×2 matmul test suite.
pub fn matmul_suite() -> TestSuite {
    let mut s = TestSuite::new("matmul");

    // Identity matmul: A × I = A
    let a = TestTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], "A");
    let eye = TestTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], "I");
    let expected = TestTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], "A*I");
    s.add_case(KernelTestCase::new(
        "matmul_identity",
        "matmul",
        vec![a, eye],
        expected,
        Tolerance::Absolute(1e-6),
    ));

    // Zero matrix
    let z = TestTensor::zeros(vec![2, 2], "Z");
    let b = TestTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], "B");
    let zero_out = TestTensor::zeros(vec![2, 2], "Z*B");
    s.add_case(KernelTestCase::new(
        "matmul_zero",
        "matmul",
        vec![z, b],
        zero_out,
        Tolerance::Exact,
    ));

    // General 2x2
    let m1 = TestTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], "M1");
    let m2 = TestTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], "M2");
    // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19,22,43,50]
    let prod = TestTensor::from_vec(vec![19.0, 22.0, 43.0, 50.0], vec![2, 2], "M1*M2");
    s.add_case(KernelTestCase::new(
        "matmul_2x2",
        "matmul",
        vec![m1, m2],
        prod,
        Tolerance::Absolute(1e-5),
    ));

    s
}

/// Softmax test suite.
#[allow(clippy::cast_possible_truncation)]
pub fn softmax_suite() -> TestSuite {
    let mut s = TestSuite::new("softmax");

    // Uniform input → uniform output
    let uniform = TestTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4], "uniform");
    let uniform_out =
        TestTensor::from_vec(vec![0.25, 0.25, 0.25, 0.25], vec![4], "softmax(uniform)");
    s.add_case(KernelTestCase::new(
        "softmax_uniform",
        "softmax",
        vec![uniform],
        uniform_out,
        Tolerance::Absolute(1e-6),
    ));

    // One-hot dominant
    let hot = TestTensor::from_vec(vec![10.0, 0.0, 0.0], vec![3], "hot");
    // e^10 / (e^10 + 2) ≈ 0.99991
    let e10 = 10.0_f64.exp();
    let denom = e10 + 2.0;
    let h0 = (e10 / denom) as f32;
    let h1 = (1.0 / denom) as f32;
    let hot_out = TestTensor::from_vec(vec![h0, h1, h1], vec![3], "softmax(hot)");
    s.add_case(KernelTestCase::new(
        "softmax_one_hot",
        "softmax",
        vec![hot],
        hot_out,
        Tolerance::Absolute(1e-5),
    ));

    s
}

/// RMS-norm test suite.
#[allow(clippy::cast_possible_truncation)]
pub fn rmsnorm_suite() -> TestSuite {
    let mut s = TestSuite::new("rmsnorm");

    // All ones → all ones (rms = 1)
    let ones = TestTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4], "ones");
    let ones_out = TestTensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![4], "rmsnorm(ones)");
    s.add_case(KernelTestCase::new(
        "rmsnorm_ones",
        "rmsnorm",
        vec![ones],
        ones_out,
        Tolerance::Absolute(1e-6),
    ));

    // [3, 4] → rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
    let v = TestTensor::from_vec(vec![3.0, 4.0], vec![2], "v");
    let rms = (12.5_f64).sqrt();
    let v_out = TestTensor::from_vec(
        vec![(3.0_f64 / rms) as f32, (4.0_f64 / rms) as f32],
        vec![2],
        "rmsnorm(v)",
    );
    s.add_case(KernelTestCase::new(
        "rmsnorm_3_4",
        "rmsnorm",
        vec![v],
        v_out,
        Tolerance::Absolute(1e-5),
    ));

    s
}

/// `RoPE` (Rotary Position Embedding) test suite.
#[allow(clippy::cast_possible_truncation)]
pub fn rope_suite() -> TestSuite {
    let mut s = TestSuite::new("rope");

    // Position 0 → no rotation (cos=1, sin=0)
    let input = TestTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![4], "x");
    let expected = TestTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![4], "rope(x)");
    s.add_case(
        KernelTestCase::new("rope_pos0", "rope", vec![input], expected, Tolerance::Absolute(1e-6))
            .with_param("position", TestParam::Int(0)),
    );

    // Simple 2-dim rotation at position 1 with base 10000
    // theta = 1 / 10000^(0/2) = 1.0
    // cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
    let x = TestTensor::from_vec(vec![1.0, 0.0], vec![2], "x");
    let cos1 = 1.0_f64.cos() as f32;
    let sin1 = 1.0_f64.sin() as f32;
    // rotated: [x0*cos - x1*sin, x0*sin + x1*cos]
    let y = TestTensor::from_vec(vec![cos1, sin1], vec![2], "rope(x)");
    s.add_case(
        KernelTestCase::new("rope_pos1_2d", "rope", vec![x], y, Tolerance::Absolute(1e-5))
            .with_param("position", TestParam::Int(1))
            .with_param("base", TestParam::Float(10000.0)),
    );

    s
}

/// Attention mechanism test suite.
pub fn attention_suite() -> TestSuite {
    let mut s = TestSuite::new("attention");

    // Single-head, seq_len=1 → output = V (attention weight is trivially 1)
    let q = TestTensor::from_vec(vec![1.0, 0.0], vec![1, 2], "Q");
    let k = TestTensor::from_vec(vec![1.0, 0.0], vec![1, 2], "K");
    let v = TestTensor::from_vec(vec![0.5, 0.5], vec![1, 2], "V");
    let expected = TestTensor::from_vec(vec![0.5, 0.5], vec![1, 2], "attn_out");
    s.add_case(
        KernelTestCase::new(
            "attention_single_token",
            "attention",
            vec![q, k, v],
            expected,
            Tolerance::Absolute(1e-5),
        )
        .with_param("num_heads", TestParam::Int(1)),
    );

    // Two tokens, uniform K → uniform attention → mean of V rows
    let q2 = TestTensor::from_vec(vec![1.0, 0.0], vec![1, 2], "Q");
    let k2 = TestTensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2], "K");
    let v2 = TestTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], "V");
    // Uniform attention → mean of V = [0.5, 0.5]
    let expected2 = TestTensor::from_vec(vec![0.5, 0.5], vec![1, 2], "attn_out");
    s.add_case(
        KernelTestCase::new(
            "attention_uniform_keys",
            "attention",
            vec![q2, k2, v2],
            expected2,
            Tolerance::Absolute(1e-4),
        )
        .with_param("num_heads", TestParam::Int(1)),
    );

    s
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TestTensor construction ──────────────────────────────────────

    #[test]
    fn test_tensor_from_vec() {
        let t = TestTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], "t");
        assert_eq!(t.numel(), 6);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.label, "t");
    }

    #[test]
    #[should_panic(expected = "data length 3 != shape product 4")]
    fn test_tensor_from_vec_shape_mismatch() {
        TestTensor::from_vec(vec![1.0, 2.0, 3.0], vec![2, 2], "bad");
    }

    #[test]
    fn test_tensor_zeros() {
        let t = TestTensor::zeros(vec![3, 4], "z");
        assert_eq!(t.numel(), 12);
        assert!(t.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_tensor_random_deterministic() {
        let a = TestTensor::random(vec![10], 42, "a");
        let b = TestTensor::random(vec![10], 42, "b");
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn test_tensor_random_different_seeds() {
        let a = TestTensor::random(vec![10], 1, "a");
        let b = TestTensor::random(vec![10], 2, "b");
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn test_tensor_scalar() {
        let t = TestTensor::from_vec(vec![42.0], vec![1], "scalar");
        assert_eq!(t.numel(), 1);
    }

    #[test]
    fn test_tensor_empty_shape() {
        let t = TestTensor::zeros(vec![0], "empty");
        assert_eq!(t.numel(), 0);
        assert!(t.data.is_empty());
    }

    #[test]
    fn test_tensor_3d() {
        let t = TestTensor::zeros(vec![2, 3, 4], "3d");
        assert_eq!(t.numel(), 24);
    }

    // ── Exact tolerance ──────────────────────────────────────────────

    #[test]
    fn test_exact_tolerance_pass() {
        let data = vec![1.0, 2.0, 3.0];
        let (ok, _, _, msg) = within_tolerance(&data, &data, &Tolerance::Exact);
        assert!(ok);
        assert!(msg.is_none());
    }

    #[test]
    fn test_exact_tolerance_fail() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.001];
        let (ok, _, _, msg) = within_tolerance(&a, &b, &Tolerance::Exact);
        assert!(!ok);
        assert!(msg.unwrap().contains("exact mismatch"));
    }

    #[test]
    fn test_exact_tolerance_empty() {
        let (ok, _, _, _) = within_tolerance(&[], &[], &Tolerance::Exact);
        assert!(ok);
    }

    // ── Absolute tolerance ───────────────────────────────────────────

    #[test]
    fn test_absolute_tolerance_pass() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0001, 2.0001, 3.0001];
        let (ok, max_err, _, _) = within_tolerance(&a, &b, &Tolerance::Absolute(1e-3));
        assert!(ok);
        assert!(max_err < 1e-3);
    }

    #[test]
    fn test_absolute_tolerance_fail() {
        let a = vec![1.0];
        let b = vec![2.0];
        let (ok, _, _, _) = within_tolerance(&a, &b, &Tolerance::Absolute(0.5));
        assert!(!ok);
    }

    #[test]
    fn test_absolute_tolerance_boundary() {
        let a = vec![1.0];
        let b = vec![1.5];
        let (ok, _, _, _) = within_tolerance(&a, &b, &Tolerance::Absolute(0.5));
        assert!(ok); // 0.5 <= 0.5
    }

    #[test]
    fn test_absolute_tolerance_just_over() {
        let a = vec![0.0];
        let b = vec![0.500_01];
        let (ok, _, _, _) = within_tolerance(&a, &b, &Tolerance::Absolute(0.5));
        assert!(!ok);
    }

    #[test]
    fn test_absolute_tolerance_negative_values() {
        let a = vec![-1.0, -2.0];
        let b = vec![-1.001, -2.001];
        let (ok, _, _, _) = within_tolerance(&a, &b, &Tolerance::Absolute(0.01));
        assert!(ok);
    }

    // ── Relative tolerance ───────────────────────────────────────────

    #[test]
    fn test_relative_tolerance_pass() {
        let a = vec![100.0];
        let b = vec![100.5];
        // relative = 0.5/100.5 ≈ 0.005
        let (ok, _, _, _) = within_tolerance(&a, &b, &Tolerance::Relative(0.01));
        assert!(ok);
    }

    #[test]
    fn test_relative_tolerance_fail() {
        let a = vec![1.0];
        let b = vec![2.0];
        // relative = 1.0/2.0 = 0.5
        let (ok, _, _, _) = within_tolerance(&a, &b, &Tolerance::Relative(0.1));
        assert!(!ok);
    }

    #[test]
    fn test_relative_tolerance_near_zero() {
        // Near-zero expected: clamp denom to 1e-12
        let a = vec![1e-15];
        let b = vec![0.0];
        let (ok, _, _, _) = within_tolerance(&a, &b, &Tolerance::Relative(1.0));
        // 1e-15 / 1e-12 = 1e-3 < 1.0 → pass
        assert!(ok);
    }

    // ── ULP tolerance ────────────────────────────────────────────────

    #[test]
    fn test_ulp_distance_same() {
        assert_eq!(ulp_distance(1.0, 1.0), 0);
    }

    #[test]
    fn test_ulp_distance_adjacent() {
        let a: f32 = 1.0;
        let b = f32::from_bits(a.to_bits() + 1);
        assert_eq!(ulp_distance(a, b), 1);
    }

    #[test]
    fn test_ulp_distance_nan() {
        assert_eq!(ulp_distance(f32::NAN, 1.0), u32::MAX);
        assert_eq!(ulp_distance(1.0, f32::NAN), u32::MAX);
    }

    #[test]
    fn test_ulp_tolerance_pass() {
        let a: f32 = 1.0;
        let b = f32::from_bits(a.to_bits() + 2);
        let (ok, _, _, _) = within_tolerance(&[a], &[b], &Tolerance::UlpDistance(5));
        assert!(ok);
    }

    #[test]
    fn test_ulp_tolerance_fail() {
        let a: f32 = 1.0;
        let b = f32::from_bits(a.to_bits() + 10);
        let (ok, _, _, _) = within_tolerance(&[a], &[b], &Tolerance::UlpDistance(5));
        assert!(!ok);
    }

    // ── NaN / Inf detection ──────────────────────────────────────────

    #[test]
    fn test_nan_in_actual() {
        let a = vec![1.0, f32::NAN, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let (ok, _, _, msg) = within_tolerance(&a, &b, &Tolerance::Absolute(1.0));
        assert!(!ok);
        assert!(msg.unwrap().contains("NaN"));
    }

    #[test]
    fn test_inf_in_actual() {
        let a = vec![f32::INFINITY];
        let b = vec![1.0];
        let (ok, _, _, msg) = within_tolerance(&a, &b, &Tolerance::Absolute(1.0));
        assert!(!ok);
        assert!(msg.unwrap().contains("Inf"));
    }

    #[test]
    fn test_neg_inf_in_actual() {
        let a = vec![f32::NEG_INFINITY];
        let b = vec![1.0];
        let (ok, _, _, msg) = within_tolerance(&a, &b, &Tolerance::Absolute(1.0));
        assert!(!ok);
        assert!(msg.unwrap().contains("Inf"));
    }

    // ── Length mismatch ──────────────────────────────────────────────

    #[test]
    fn test_length_mismatch() {
        let (ok, _, _, msg) = within_tolerance(&[1.0, 2.0], &[1.0], &Tolerance::Exact);
        assert!(!ok);
        assert!(msg.unwrap().contains("length mismatch"));
    }

    // ── KernelTestCase ───────────────────────────────────────────────

    #[test]
    fn test_case_evaluate_pass() {
        let expected = TestTensor::from_vec(vec![1.0, 2.0], vec![2], "exp");
        let actual = TestTensor::from_vec(vec![1.0, 2.0], vec![2], "act");
        let tc = KernelTestCase::new("t", "k", vec![], expected, Tolerance::Exact);
        let r = tc.evaluate(actual, "cpu", 100);
        assert!(r.passed);
        assert_eq!(r.execution_time_us, 100);
        assert!(r.error_message.is_none());
    }

    #[test]
    fn test_case_evaluate_fail() {
        let expected = TestTensor::from_vec(vec![1.0], vec![1], "exp");
        let actual = TestTensor::from_vec(vec![2.0], vec![1], "act");
        let tc = KernelTestCase::new("t", "k", vec![], expected, Tolerance::Exact);
        let r = tc.evaluate(actual, "cpu", 50);
        assert!(!r.passed);
        assert!(r.error_message.is_some());
    }

    #[test]
    fn test_case_shape_mismatch() {
        let expected = TestTensor::from_vec(vec![1.0, 2.0], vec![2], "exp");
        let actual = TestTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4], "act");
        let tc = KernelTestCase::new("t", "k", vec![], expected, Tolerance::Exact);
        let r = tc.evaluate(actual, "cpu", 0);
        assert!(!r.passed);
        assert!(r.error_message.as_ref().unwrap().contains("shape mismatch"));
    }

    #[test]
    fn test_case_with_param() {
        let exp = TestTensor::zeros(vec![1], "e");
        let tc = KernelTestCase::new("p", "k", vec![], exp, Tolerance::Exact)
            .with_param("alpha", TestParam::Float(0.5))
            .with_param("flag", TestParam::Bool(true));

        assert_eq!(tc.params.len(), 2);
        assert!(matches!(
            tc.params.get("alpha"),
            Some(TestParam::Float(v)) if (*v - 0.5).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn test_case_with_int_param() {
        let exp = TestTensor::zeros(vec![1], "e");
        let tc = KernelTestCase::new("p", "k", vec![], exp, Tolerance::Exact)
            .with_param("n", TestParam::Int(42));

        assert!(matches!(tc.params.get("n"), Some(TestParam::Int(42))));
    }

    #[test]
    fn test_case_with_int_vec_param() {
        let exp = TestTensor::zeros(vec![1], "e");
        let tc = KernelTestCase::new("p", "k", vec![], exp, Tolerance::Exact)
            .with_param("dims", TestParam::IntVec(vec![1, 2, 3]));

        assert!(matches!(
            tc.params.get("dims"),
            Some(TestParam::IntVec(v)) if v == &[1, 2, 3]
        ));
    }

    // ── TestSuite ────────────────────────────────────────────────────

    #[test]
    fn test_suite_empty() {
        let s = TestSuite::new("empty");
        let results = s.run("cpu");
        assert!(results.is_empty());
    }

    #[test]
    fn test_suite_run_collects_all() {
        let mut s = TestSuite::new("s");
        for i in 0..5 {
            let t = TestTensor::zeros(vec![2], &format!("t{i}"));
            s.add_case(KernelTestCase::new(&format!("case_{i}"), "k", vec![], t, Tolerance::Exact));
        }
        let results = s.run("cpu");
        assert_eq!(results.len(), 5);
        assert!(results.iter().all(|r| r.passed));
    }

    #[test]
    fn test_suite_mixed_results() {
        let mut s = TestSuite::new("mixed");

        // This one will pass (echo expected back)
        let ok = TestTensor::from_vec(vec![1.0], vec![1], "ok");
        s.add_case(KernelTestCase::new("pass_case", "k", vec![], ok, Tolerance::Exact));

        // All cases pass with default runner (echo mode)
        let results = s.run("cpu");
        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
    }

    #[test]
    fn test_suite_name() {
        let s = TestSuite::new("my_suite");
        assert_eq!(s.name, "my_suite");
    }

    // ── TestRunner ───────────────────────────────────────────────────

    #[test]
    fn test_runner_no_backends() {
        let mut runner = TestRunner::new();
        let mut s = TestSuite::new("s");
        let t = TestTensor::zeros(vec![1], "t");
        s.add_case(KernelTestCase::new("c", "k", vec![], t, Tolerance::Exact));
        runner.add_suite(s);
        let reports = runner.run_all();
        // One report per case, but with no backend results.
        assert_eq!(reports.len(), 1);
        assert!(reports[0].results_by_backend.is_empty());
    }

    #[test]
    fn test_runner_single_backend() {
        let mut runner = TestRunner::new();
        runner.add_backend("cpu");
        let mut s = TestSuite::new("s");
        let t = TestTensor::zeros(vec![2], "t");
        s.add_case(KernelTestCase::new("c", "k", vec![], t, Tolerance::Exact));
        runner.add_suite(s);
        let reports = runner.run_all();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].all_agree);
    }

    #[test]
    fn test_runner_multiple_backends() {
        let mut runner = TestRunner::new();
        runner.add_backend("cpu");
        runner.add_backend("cuda");
        runner.add_backend("vulkan");

        let mut s = TestSuite::new("s");
        let t = TestTensor::zeros(vec![2], "t");
        s.add_case(KernelTestCase::new("c", "k", vec![], t, Tolerance::Exact));
        runner.add_suite(s);

        let reports = runner.run_all();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].results_by_backend.len(), 3);
        assert!(reports[0].all_agree);
        assert_eq!(reports[0].max_cross_backend_diff, 0.0);
    }

    #[test]
    fn test_runner_multiple_suites() {
        let mut runner = TestRunner::new();
        runner.add_backend("cpu");

        for i in 0..3 {
            let mut s = TestSuite::new(&format!("suite_{i}"));
            let t = TestTensor::zeros(vec![1], "t");
            s.add_case(KernelTestCase::new(&format!("case_{i}"), "k", vec![], t, Tolerance::Exact));
            runner.add_suite(s);
        }

        let reports = runner.run_all();
        assert_eq!(reports.len(), 3);
    }

    #[test]
    fn test_runner_default() {
        let runner = TestRunner::default();
        assert!(runner.backends.is_empty());
        assert!(runner.suites.is_empty());
    }

    // ── Cross-backend comparison ─────────────────────────────────────

    #[test]
    fn test_cross_backend_identical() {
        let r1 = TestResult {
            test_name: "t".into(),
            backend: "a".into(),
            passed: true,
            actual_output: Some(TestTensor::from_vec(vec![1.0, 2.0], vec![2], "o")),
            max_error: Some(0.0),
            mean_error: Some(0.0),
            execution_time_us: 10,
            error_message: None,
        };
        let r2 = TestResult {
            test_name: "t".into(),
            backend: "b".into(),
            passed: true,
            actual_output: Some(TestTensor::from_vec(vec![1.0, 2.0], vec![2], "o")),
            max_error: Some(0.0),
            mean_error: Some(0.0),
            execution_time_us: 20,
            error_message: None,
        };
        let results = vec![("a".to_string(), r1), ("b".to_string(), r2)];
        let (agree, diff) = cross_backend_diff(&results);
        assert!(agree);
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn test_cross_backend_different() {
        let r1 = TestResult {
            test_name: "t".into(),
            backend: "a".into(),
            passed: true,
            actual_output: Some(TestTensor::from_vec(vec![1.0], vec![1], "o")),
            max_error: Some(0.0),
            mean_error: Some(0.0),
            execution_time_us: 10,
            error_message: None,
        };
        let r2 = TestResult {
            test_name: "t".into(),
            backend: "b".into(),
            passed: true,
            actual_output: Some(TestTensor::from_vec(vec![1.5], vec![1], "o")),
            max_error: Some(0.0),
            mean_error: Some(0.0),
            execution_time_us: 10,
            error_message: None,
        };
        let results = vec![("a".to_string(), r1), ("b".to_string(), r2)];
        let (_, diff) = cross_backend_diff(&results);
        assert!((diff - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cross_backend_pass_fail_disagree() {
        let r1 = TestResult {
            test_name: "t".into(),
            backend: "a".into(),
            passed: true,
            actual_output: Some(TestTensor::from_vec(vec![1.0], vec![1], "o")),
            max_error: Some(0.0),
            mean_error: Some(0.0),
            execution_time_us: 10,
            error_message: None,
        };
        let r2 = TestResult {
            test_name: "t".into(),
            backend: "b".into(),
            passed: false,
            actual_output: Some(TestTensor::from_vec(vec![1.0], vec![1], "o")),
            max_error: Some(0.0),
            mean_error: Some(0.0),
            execution_time_us: 10,
            error_message: Some("failed".into()),
        };
        let results = vec![("a".to_string(), r1), ("b".to_string(), r2)];
        let (agree, _) = cross_backend_diff(&results);
        assert!(!agree);
    }

    #[test]
    fn test_cross_backend_single_backend() {
        let r = TestResult {
            test_name: "t".into(),
            backend: "a".into(),
            passed: true,
            actual_output: None,
            max_error: None,
            mean_error: None,
            execution_time_us: 10,
            error_message: None,
        };
        let (agree, diff) = cross_backend_diff(&[("a".to_string(), r)]);
        assert!(agree);
        assert_eq!(diff, 0.0);
    }

    #[test]
    fn test_cross_backend_three_backends() {
        let make = |name: &str, val: f32| -> (String, TestResult) {
            (
                name.to_string(),
                TestResult {
                    test_name: "t".into(),
                    backend: name.into(),
                    passed: true,
                    actual_output: Some(TestTensor::from_vec(vec![val], vec![1], "o")),
                    max_error: Some(0.0),
                    mean_error: Some(0.0),
                    execution_time_us: 0,
                    error_message: None,
                },
            )
        };
        let results = vec![make("a", 1.0), make("b", 1.1), make("c", 1.3)];
        let (agree, diff) = cross_backend_diff(&results);
        assert!(agree); // all passed
        // max diff = |1.0 - 1.3| = 0.3
        assert!((diff - 0.3).abs() < 1e-6);
    }

    // ── ComparisonReport ─────────────────────────────────────────────

    #[test]
    fn test_report_summary_pass() {
        let report = ComparisonReport {
            test_name: "matmul_2x2".into(),
            results_by_backend: vec![(
                "cpu".into(),
                TestResult {
                    test_name: "matmul_2x2".into(),
                    backend: "cpu".into(),
                    passed: true,
                    actual_output: None,
                    max_error: None,
                    mean_error: None,
                    execution_time_us: 0,
                    error_message: None,
                },
            )],
            all_agree: true,
            max_cross_backend_diff: 0.0,
        };
        let s = report.summary();
        assert!(s.contains("[PASS]"));
        assert!(s.contains("matmul_2x2"));
        assert!(s.contains("cpu"));
    }

    #[test]
    fn test_report_summary_fail() {
        let report = ComparisonReport {
            test_name: "bad_test".into(),
            results_by_backend: vec![],
            all_agree: false,
            max_cross_backend_diff: 1.5,
        };
        let s = report.summary();
        assert!(s.contains("[FAIL]"));
    }

    #[test]
    fn test_report_summary_multi_backend() {
        let report = ComparisonReport {
            test_name: "test".into(),
            results_by_backend: vec![
                (
                    "cpu".into(),
                    TestResult {
                        test_name: "test".into(),
                        backend: "cpu".into(),
                        passed: true,
                        actual_output: None,
                        max_error: None,
                        mean_error: None,
                        execution_time_us: 0,
                        error_message: None,
                    },
                ),
                (
                    "cuda".into(),
                    TestResult {
                        test_name: "test".into(),
                        backend: "cuda".into(),
                        passed: true,
                        actual_output: None,
                        max_error: None,
                        mean_error: None,
                        execution_time_us: 0,
                        error_message: None,
                    },
                ),
            ],
            all_agree: true,
            max_cross_backend_diff: 1e-7,
        };
        let s = report.summary();
        assert!(s.contains("cpu"));
        assert!(s.contains("cuda"));
    }

    // ── Built-in suites ──────────────────────────────────────────────

    #[test]
    fn test_matmul_suite_valid() {
        let s = matmul_suite();
        assert!(!s.cases.is_empty());
        for c in &s.cases {
            assert!(!c.name.is_empty());
            assert_eq!(c.kernel_name, "matmul");
        }
    }

    #[test]
    fn test_matmul_suite_runs() {
        let s = matmul_suite();
        let results = s.run("cpu");
        assert_eq!(results.len(), s.cases.len());
        for r in &results {
            assert!(r.passed, "case {} failed: {:?}", r.test_name, r.error_message,);
        }
    }

    #[test]
    fn test_softmax_suite_valid() {
        let s = softmax_suite();
        assert!(!s.cases.is_empty());
        for c in &s.cases {
            assert_eq!(c.kernel_name, "softmax");
        }
    }

    #[test]
    fn test_softmax_suite_runs() {
        let s = softmax_suite();
        let results = s.run("cpu");
        for r in &results {
            assert!(r.passed, "{} failed", r.test_name);
        }
    }

    #[test]
    fn test_rmsnorm_suite_valid() {
        let s = rmsnorm_suite();
        assert!(!s.cases.is_empty());
        for c in &s.cases {
            assert_eq!(c.kernel_name, "rmsnorm");
        }
    }

    #[test]
    fn test_rmsnorm_suite_runs() {
        let s = rmsnorm_suite();
        let results = s.run("cpu");
        for r in &results {
            assert!(r.passed, "{} failed", r.test_name);
        }
    }

    #[test]
    fn test_rope_suite_valid() {
        let s = rope_suite();
        assert!(!s.cases.is_empty());
        for c in &s.cases {
            assert_eq!(c.kernel_name, "rope");
        }
    }

    #[test]
    fn test_rope_suite_runs() {
        let s = rope_suite();
        let results = s.run("cpu");
        for r in &results {
            assert!(r.passed, "{} failed", r.test_name);
        }
    }

    #[test]
    fn test_attention_suite_valid() {
        let s = attention_suite();
        assert!(!s.cases.is_empty());
        for c in &s.cases {
            assert_eq!(c.kernel_name, "attention");
        }
    }

    #[test]
    fn test_attention_suite_runs() {
        let s = attention_suite();
        let results = s.run("cpu");
        for r in &results {
            assert!(r.passed, "{} failed", r.test_name);
        }
    }

    // ── Large tensor performance ─────────────────────────────────────

    #[test]
    fn test_large_tensor_comparison() {
        let n = 100_000;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = a.iter().map(|v| v + 1e-8).collect();
        let (ok, max_err, mean_err, _) = within_tolerance(&a, &b, &Tolerance::Absolute(1e-6));
        assert!(ok);
        assert!(max_err < 1e-6);
        assert!(mean_err < 1e-6);
    }

    #[test]
    fn test_large_tensor_exact_fail() {
        let n = 10_000;
        let a: Vec<f32> = vec![1.0; n];
        let mut b = a.clone();
        b[n - 1] = 1.001;
        let (ok, _, _, msg) = within_tolerance(&a, &b, &Tolerance::Exact);
        assert!(!ok);
        assert!(msg.is_some());
    }

    // ── Integration: runner with built-in suites ─────────────────────

    #[test]
    fn test_runner_with_builtin_suites() {
        let mut runner = TestRunner::new();
        runner.add_backend("reference");
        runner.add_suite(matmul_suite());
        runner.add_suite(softmax_suite());
        runner.add_suite(rmsnorm_suite());
        runner.add_suite(rope_suite());
        runner.add_suite(attention_suite());

        let reports = runner.run_all();
        assert!(!reports.is_empty());
        for r in &reports {
            assert!(r.all_agree, "report {} disagrees: {}", r.test_name, r.summary(),);
        }
    }

    #[test]
    fn test_runner_cross_backend_with_builtins() {
        let mut runner = TestRunner::new();
        runner.add_backend("cpu");
        runner.add_backend("reference");
        runner.add_suite(matmul_suite());

        let reports = runner.run_all();
        for r in &reports {
            assert!(r.all_agree);
            assert_eq!(r.max_cross_backend_diff, 0.0);
        }
    }

    // ── Mean error computation ───────────────────────────────────────

    #[test]
    fn test_mean_error_absolute() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.1, 2.1, 3.1];
        let (ok, max_err, mean_err, _) = within_tolerance(&a, &b, &Tolerance::Absolute(0.2));
        assert!(ok);
        assert!((max_err - 0.1).abs() < 1e-5);
        assert!((mean_err - 0.1).abs() < 1e-5);
    }

    // ── TestResult fields ────────────────────────────────────────────

    #[test]
    fn test_result_backend_name() {
        let exp = TestTensor::zeros(vec![1], "e");
        let act = TestTensor::zeros(vec![1], "a");
        let tc = KernelTestCase::new("t", "k", vec![], exp, Tolerance::Exact);
        let r = tc.evaluate(act, "my_backend", 42);
        assert_eq!(r.backend, "my_backend");
        assert_eq!(r.test_name, "t");
    }

    #[test]
    fn test_result_timing() {
        let exp = TestTensor::zeros(vec![1], "e");
        let act = TestTensor::zeros(vec![1], "a");
        let tc = KernelTestCase::new("t", "k", vec![], exp, Tolerance::Exact);
        let r = tc.evaluate(act, "cpu", 999);
        assert_eq!(r.execution_time_us, 999);
    }
}
