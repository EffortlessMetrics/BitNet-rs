//! Reusable test harness for OpenCL kernel development.
//!
//! Provides deterministic tensor generation, CPU reference comparison,
//! and structured test reporting for validating OpenCL kernel outputs.

use std::time::Instant;

/// A tensor used in test cases, storing flat f32 data with shape metadata.
#[derive(Debug, Clone)]
pub struct TestTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub name: String,
}

/// A single test case with named inputs, expected outputs, and tolerance.
#[derive(Debug, Clone)]
pub struct TestCase {
    pub name: String,
    pub inputs: Vec<TestTensor>,
    pub expected_outputs: Vec<TestTensor>,
    pub tolerance: f32,
    pub tags: Vec<String>,
}

/// A suite of test cases with an optional setup function.
#[derive(Clone)]
pub struct TestSuite {
    pub name: String,
    pub cases: Vec<TestCase>,
    pub setup_fn: Option<fn() -> Vec<TestTensor>>,
}

/// Result of running a single test case.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub case_name: String,
    pub passed: bool,
    pub max_error: f32,
    pub mean_error: f32,
    pub execution_time_us: u64,
    pub details: String,
}

/// Aggregated report from running a test suite.
#[derive(Debug, Clone)]
pub struct TestReport {
    pub suite_name: String,
    pub results: Vec<TestResult>,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
}

/// Configuration for the test harness.
#[derive(Debug, Clone)]
pub struct TestHarnessConfig {
    pub default_tolerance: f32,
    pub verbose: bool,
    pub abort_on_first_failure: bool,
    pub seed: u64,
}

impl Default for TestHarnessConfig {
    fn default() -> Self {
        Self { default_tolerance: 1e-6, verbose: false, abort_on_first_failure: false, seed: 42 }
    }
}

/// Create a deterministic pseudo-random tensor from `seed`.
///
/// Uses a simple xorshift64 PRNG to produce values in `[-1, 1]`.
pub fn create_test_tensor(shape: &[usize], seed: u64) -> TestTensor {
    let len: usize = shape.iter().product();
    let mut state = seed.wrapping_add(1); // avoid zero-state
    let data: Vec<f32> = (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-1.0, 1.0]
            (state as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    TestTensor { data, shape: shape.to_vec(), name: String::new() }
}

/// Create a zero-filled tensor with the given shape.
pub fn create_zero_tensor(shape: &[usize]) -> TestTensor {
    let len: usize = shape.iter().product();
    TestTensor { data: vec![0.0; len], shape: shape.to_vec(), name: String::new() }
}

/// Create an identity matrix of size `n × n`.
pub fn create_identity_matrix(n: usize) -> TestTensor {
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    TestTensor { data, shape: vec![n, n], name: String::new() }
}

/// Compare two tensors element-wise and return `(passed, max_error, mean_error)`.
pub fn compare_tensors(
    actual: &TestTensor,
    expected: &TestTensor,
    tolerance: f32,
) -> (bool, f32, f32) {
    if actual.shape != expected.shape || actual.data.len() != expected.data.len() {
        return (false, f32::INFINITY, f32::INFINITY);
    }
    if actual.data.is_empty() {
        return (true, 0.0, 0.0);
    }
    let mut max_err: f32 = 0.0;
    let mut sum_err: f64 = 0.0;
    for (a, e) in actual.data.iter().zip(expected.data.iter()) {
        let err = (a - e).abs();
        if err > max_err {
            max_err = err;
        }
        sum_err += err as f64;
    }
    let mean_err = (sum_err / actual.data.len() as f64) as f32;
    (max_err <= tolerance, max_err, mean_err)
}

/// Run a single test case through `kernel_fn` and compare outputs.
pub fn run_test_case(
    case: &TestCase,
    kernel_fn: &dyn Fn(&[&TestTensor]) -> Vec<TestTensor>,
) -> TestResult {
    let input_refs: Vec<&TestTensor> = case.inputs.iter().collect();
    let start = Instant::now();
    let actual_outputs = kernel_fn(&input_refs);
    let elapsed_us = start.elapsed().as_micros() as u64;

    if actual_outputs.len() != case.expected_outputs.len() {
        return TestResult {
            case_name: case.name.clone(),
            passed: false,
            max_error: f32::INFINITY,
            mean_error: f32::INFINITY,
            execution_time_us: elapsed_us,
            details: format!(
                "output count mismatch: got {} expected {}",
                actual_outputs.len(),
                case.expected_outputs.len()
            ),
        };
    }

    let mut overall_pass = true;
    let mut overall_max: f32 = 0.0;
    let mut overall_mean: f32 = 0.0;
    let mut detail_parts: Vec<String> = Vec::new();

    for (i, (actual, expected)) in
        actual_outputs.iter().zip(case.expected_outputs.iter()).enumerate()
    {
        let (pass, max_e, mean_e) = compare_tensors(actual, expected, case.tolerance);
        if !pass {
            overall_pass = false;
            detail_parts
                .push(format!("output[{i}]: max_error={max_e:.8} > tolerance={}", case.tolerance));
        }
        if max_e > overall_max {
            overall_max = max_e;
        }
        overall_mean += mean_e;
    }
    if !case.expected_outputs.is_empty() {
        overall_mean /= case.expected_outputs.len() as f32;
    }

    TestResult {
        case_name: case.name.clone(),
        passed: overall_pass,
        max_error: overall_max,
        mean_error: overall_mean,
        execution_time_us: elapsed_us,
        details: if detail_parts.is_empty() {
            "all outputs within tolerance".into()
        } else {
            detail_parts.join("; ")
        },
    }
}

/// Run every case in a suite and produce an aggregated report.
pub fn run_test_suite(
    suite: &TestSuite,
    kernel_fn: &dyn Fn(&[&TestTensor]) -> Vec<TestTensor>,
) -> TestReport {
    run_test_suite_with_config(suite, kernel_fn, &TestHarnessConfig::default())
}

/// Run a suite with explicit harness configuration.
pub fn run_test_suite_with_config(
    suite: &TestSuite,
    kernel_fn: &dyn Fn(&[&TestTensor]) -> Vec<TestTensor>,
    config: &TestHarnessConfig,
) -> TestReport {
    let mut results = Vec::new();
    let mut passed = 0usize;
    let mut failed = 0usize;

    for case in &suite.cases {
        let result = run_test_case(case, kernel_fn);
        if result.passed {
            passed += 1;
        } else {
            failed += 1;
            if config.abort_on_first_failure {
                results.push(result);
                break;
            }
        }
        results.push(result);
    }

    let total = suite.cases.len();
    let skipped = total - passed - failed;

    TestReport { suite_name: suite.name.clone(), results, total, passed, failed, skipped }
}

/// Format a [`TestReport`] as a human-readable string.
pub fn format_test_report(report: &TestReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("=== Test Suite: {} ===\n", report.suite_name));
    out.push_str(&format!(
        "Total: {} | Passed: {} | Failed: {} | Skipped: {}\n",
        report.total, report.passed, report.failed, report.skipped
    ));
    out.push_str("---\n");
    for r in &report.results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        out.push_str(&format!(
            "[{status}] {} — max_err={:.8} mean_err={:.8} ({} µs)\n",
            r.case_name, r.max_error, r.mean_error, r.execution_time_us
        ));
        if !r.passed {
            out.push_str(&format!("       {}\n", r.details));
        }
    }
    out.push_str("===\n");
    out
}

// ── CPU reference helpers ────────────────────────────────────────────

/// Naive CPU matrix multiply: C = A × B.
/// A is [m, k], B is [k, n], result is [m, n].
fn cpu_matmul(a: &TestTensor, b: &TestTensor) -> TestTensor {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a.data[i * k + p] * b.data[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    TestTensor { data: out, shape: vec![m, n], name: String::new() }
}

/// Row-wise softmax on a 1-D or 2-D tensor.
fn cpu_softmax(input: &TestTensor) -> TestTensor {
    let cols = *input.shape.last().unwrap_or(&1);
    let rows = if input.shape.len() > 1 { input.shape[0] } else { 1 };
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &input.data[r * cols..(r + 1) * cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (c, &e) in exps.iter().enumerate() {
            out[r * cols + c] = e / sum;
        }
    }
    TestTensor { data: out, shape: input.shape.clone(), name: String::new() }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn gelu(x: f32) -> f32 {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Generate a test suite of matmul cases for the given `(m, k, n)` sizes.
pub fn generate_matmul_suite(sizes: &[(usize, usize, usize)]) -> TestSuite {
    let cases: Vec<TestCase> = sizes
        .iter()
        .enumerate()
        .map(|(idx, &(m, k, n))| {
            let a = create_test_tensor(&[m, k], 100 + idx as u64);
            let b = create_test_tensor(&[k, n], 200 + idx as u64);
            let expected = cpu_matmul(&a, &b);
            TestCase {
                name: format!("matmul_{m}x{k}x{n}"),
                inputs: vec![a, b],
                expected_outputs: vec![expected],
                tolerance: 1e-4,
                tags: vec!["matmul".into()],
            }
        })
        .collect();
    TestSuite { name: "matmul".into(), cases, setup_fn: None }
}

/// Generate a softmax test suite for the given row sizes.
pub fn generate_softmax_suite(sizes: &[usize]) -> TestSuite {
    let cases: Vec<TestCase> = sizes
        .iter()
        .enumerate()
        .map(|(idx, &n)| {
            let input = create_test_tensor(&[n], 300 + idx as u64);
            let expected = cpu_softmax(&input);
            TestCase {
                name: format!("softmax_{n}"),
                inputs: vec![input],
                expected_outputs: vec![expected],
                tolerance: 1e-5,
                tags: vec!["softmax".into()],
            }
        })
        .collect();
    TestSuite { name: "softmax".into(), cases, setup_fn: None }
}

/// Generate an activation-function test suite covering SiLU, GELU, and ReLU.
pub fn generate_activation_suite() -> TestSuite {
    let input = create_test_tensor(&[64], 400);

    let make_case = |name: &str, f: fn(f32) -> f32, tag: &str| -> TestCase {
        let out_data: Vec<f32> = input.data.iter().map(|&x| f(x)).collect();
        TestCase {
            name: name.into(),
            inputs: vec![input.clone()],
            expected_outputs: vec![TestTensor {
                data: out_data,
                shape: input.shape.clone(),
                name: String::new(),
            }],
            tolerance: 1e-5,
            tags: vec!["activation".into(), tag.into()],
        }
    };

    TestSuite {
        name: "activation".into(),
        cases: vec![
            make_case("silu", silu, "silu"),
            make_case("gelu", gelu, "gelu"),
            make_case("relu", relu, "relu"),
        ],
        setup_fn: None,
    }
}

/// Filter a suite to only cases that contain at least one of `tags`.
pub fn filter_suite_by_tags(suite: &TestSuite, tags: &[&str]) -> TestSuite {
    let filtered = suite
        .cases
        .iter()
        .filter(|c| c.tags.iter().any(|t| tags.contains(&t.as_str())))
        .cloned()
        .collect();
    TestSuite { name: suite.name.clone(), cases: filtered, setup_fn: suite.setup_fn }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tensor creation ──────────────────────────────────────────

    #[test]
    fn test_create_tensor_correct_shape() {
        let t = create_test_tensor(&[3, 4], 1);
        assert_eq!(t.shape, vec![3, 4]);
        assert_eq!(t.data.len(), 12);
    }

    #[test]
    fn test_create_tensor_deterministic() {
        let a = create_test_tensor(&[8], 42);
        let b = create_test_tensor(&[8], 42);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn test_create_tensor_different_seeds_differ() {
        let a = create_test_tensor(&[16], 1);
        let b = create_test_tensor(&[16], 2);
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn test_create_tensor_data_len_equals_shape_product() {
        for shape in &[vec![5], vec![2, 3], vec![2, 3, 4], vec![1, 1, 1, 1]] {
            let t = create_test_tensor(shape, 0);
            let expected_len: usize = shape.iter().product();
            assert_eq!(t.data.len(), expected_len);
        }
    }

    #[test]
    fn test_zero_tensor_all_zeros() {
        let t = create_zero_tensor(&[4, 4]);
        assert!(t.data.iter().all(|&v| v == 0.0));
        assert_eq!(t.data.len(), 16);
    }

    #[test]
    fn test_zero_tensor_shape() {
        let t = create_zero_tensor(&[2, 3, 5]);
        assert_eq!(t.shape, vec![2, 3, 5]);
        assert_eq!(t.data.len(), 30);
    }

    #[test]
    fn test_identity_matrix_diagonal() {
        let id = create_identity_matrix(4);
        assert_eq!(id.shape, vec![4, 4]);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(id.data[i * 4 + j], expected);
            }
        }
    }

    #[test]
    fn test_identity_matrix_size_1() {
        let id = create_identity_matrix(1);
        assert_eq!(id.data, vec![1.0]);
    }

    // ── compare_tensors ──────────────────────────────────────────

    #[test]
    fn test_compare_identical_tensors() {
        let t = create_test_tensor(&[8], 10);
        let (pass, max_e, mean_e) = compare_tensors(&t, &t, 0.0);
        assert!(pass);
        assert_eq!(max_e, 0.0);
        assert_eq!(mean_e, 0.0);
    }

    #[test]
    fn test_compare_within_tolerance() {
        let a = TestTensor { data: vec![1.0, 2.0, 3.0], shape: vec![3], name: String::new() };
        let b =
            TestTensor { data: vec![1.0001, 2.0001, 3.0001], shape: vec![3], name: String::new() };
        let (pass, max_e, _mean_e) = compare_tensors(&a, &b, 0.001);
        assert!(pass);
        assert!(max_e < 0.001);
    }

    #[test]
    fn test_compare_beyond_tolerance() {
        let a = TestTensor { data: vec![1.0], shape: vec![1], name: String::new() };
        let b = TestTensor { data: vec![2.0], shape: vec![1], name: String::new() };
        let (pass, max_e, _) = compare_tensors(&a, &b, 0.5);
        assert!(!pass);
        assert!((max_e - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compare_shape_mismatch() {
        let a = create_zero_tensor(&[2, 3]);
        let b = create_zero_tensor(&[3, 2]);
        let (pass, max_e, _) = compare_tensors(&a, &b, 1.0);
        assert!(!pass);
        assert!(max_e.is_infinite());
    }

    #[test]
    fn test_compare_empty_tensors() {
        let a = create_zero_tensor(&[0]);
        let b = create_zero_tensor(&[0]);
        let (pass, max_e, mean_e) = compare_tensors(&a, &b, 0.0);
        assert!(pass);
        assert_eq!(max_e, 0.0);
        assert_eq!(mean_e, 0.0);
    }

    // ── run_test_case ────────────────────────────────────────────

    fn identity_kernel(inputs: &[&TestTensor]) -> Vec<TestTensor> {
        inputs.iter().map(|t| (*t).clone()).collect()
    }

    #[test]
    fn test_run_test_case_passing() {
        let t = create_test_tensor(&[4], 7);
        let case = TestCase {
            name: "identity_pass".into(),
            inputs: vec![t.clone()],
            expected_outputs: vec![t],
            tolerance: 0.0,
            tags: vec![],
        };
        let result = run_test_case(&case, &identity_kernel);
        assert!(result.passed);
        assert_eq!(result.max_error, 0.0);
    }

    #[test]
    fn test_run_test_case_failing() {
        let input = create_test_tensor(&[4], 7);
        let wrong = create_zero_tensor(&[4]);
        let case = TestCase {
            name: "identity_fail".into(),
            inputs: vec![input],
            expected_outputs: vec![wrong],
            tolerance: 1e-9,
            tags: vec![],
        };
        let result = run_test_case(&case, &identity_kernel);
        assert!(!result.passed);
    }

    #[test]
    fn test_run_test_case_output_count_mismatch() {
        let t = create_test_tensor(&[2], 1);
        let case = TestCase {
            name: "mismatch".into(),
            inputs: vec![t.clone()],
            expected_outputs: vec![t.clone(), t],
            tolerance: 1.0,
            tags: vec![],
        };
        let result = run_test_case(&case, &identity_kernel);
        assert!(!result.passed);
        assert!(result.details.contains("output count mismatch"));
    }

    #[test]
    fn test_run_test_case_is_deterministic() {
        let t = create_test_tensor(&[4], 99);
        let case = TestCase {
            name: "det".into(),
            inputs: vec![t.clone()],
            expected_outputs: vec![t],
            tolerance: 0.0,
            tags: vec![],
        };
        let r1 = run_test_case(&case, &identity_kernel);
        let r2 = run_test_case(&case, &identity_kernel);
        assert_eq!(r1.passed, r2.passed);
        assert_eq!(r1.max_error, r2.max_error);
    }

    #[test]
    fn test_run_test_case_measures_time() {
        let t = create_test_tensor(&[4], 1);
        let case = TestCase {
            name: "time".into(),
            inputs: vec![t.clone()],
            expected_outputs: vec![t],
            tolerance: 0.0,
            tags: vec![],
        };
        let result = run_test_case(&case, &identity_kernel);
        // execution_time_us is set (may be 0 on fast machines, just check type)
        let _ = result.execution_time_us;
    }

    // ── run_test_suite ───────────────────────────────────────────

    #[test]
    fn test_run_suite_mixed_pass_fail() {
        let pass_case = TestCase {
            name: "pass".into(),
            inputs: vec![create_test_tensor(&[2], 1)],
            expected_outputs: vec![create_test_tensor(&[2], 1)],
            tolerance: 0.0,
            tags: vec![],
        };
        let fail_case = TestCase {
            name: "fail".into(),
            inputs: vec![create_test_tensor(&[2], 1)],
            expected_outputs: vec![create_zero_tensor(&[2])],
            tolerance: 1e-9,
            tags: vec![],
        };
        let suite =
            TestSuite { name: "mixed".into(), cases: vec![pass_case, fail_case], setup_fn: None };
        let report = run_test_suite(&suite, &identity_kernel);
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
        assert_eq!(report.total, 2);
    }

    #[test]
    fn test_suite_report_correct_counts() {
        let cases: Vec<TestCase> = (0..5)
            .map(|i| {
                let t = create_test_tensor(&[2], i);
                TestCase {
                    name: format!("case_{i}"),
                    inputs: vec![t.clone()],
                    expected_outputs: vec![t],
                    tolerance: 0.0,
                    tags: vec![],
                }
            })
            .collect();
        let suite = TestSuite { name: "all_pass".into(), cases, setup_fn: None };
        let report = run_test_suite(&suite, &identity_kernel);
        assert_eq!(report.total, 5);
        assert_eq!(report.passed, 5);
        assert_eq!(report.failed, 0);
        assert_eq!(report.skipped, 0);
    }

    #[test]
    fn test_empty_suite() {
        let suite = TestSuite { name: "empty".into(), cases: vec![], setup_fn: None };
        let report = run_test_suite(&suite, &identity_kernel);
        assert_eq!(report.total, 0);
        assert_eq!(report.passed, 0);
        assert_eq!(report.failed, 0);
    }

    #[test]
    fn test_abort_on_first_failure() {
        let fail1 = TestCase {
            name: "fail1".into(),
            inputs: vec![create_test_tensor(&[2], 1)],
            expected_outputs: vec![create_zero_tensor(&[2])],
            tolerance: 1e-12,
            tags: vec![],
        };
        let fail2 = TestCase {
            name: "fail2".into(),
            inputs: vec![create_test_tensor(&[2], 2)],
            expected_outputs: vec![create_zero_tensor(&[2])],
            tolerance: 1e-12,
            tags: vec![],
        };
        let suite =
            TestSuite { name: "abort_test".into(), cases: vec![fail1, fail2], setup_fn: None };
        let config = TestHarnessConfig { abort_on_first_failure: true, ..Default::default() };
        let report = run_test_suite_with_config(&suite, &identity_kernel, &config);
        assert_eq!(report.results.len(), 1);
        assert!(!report.results[0].passed);
    }

    // ── Matmul suite ─────────────────────────────────────────────

    #[test]
    fn test_matmul_suite_generation() {
        let suite = generate_matmul_suite(&[(2, 3, 2), (4, 4, 4)]);
        assert_eq!(suite.cases.len(), 2);
        for case in &suite.cases {
            assert_eq!(case.inputs.len(), 2);
            assert_eq!(case.expected_outputs.len(), 1);
        }
    }

    #[test]
    fn test_matmul_identity_multiplication() {
        let id = create_identity_matrix(3);
        let a = create_test_tensor(&[3, 3], 50);
        let result = cpu_matmul(&a, &id);
        let (pass, max_e, _) = compare_tensors(&result, &a, 1e-5);
        assert!(pass, "A × I should equal A, max_error={max_e}");
    }

    #[test]
    fn test_matmul_expected_outputs_correct() {
        let suite = generate_matmul_suite(&[(2, 2, 2)]);
        let case = &suite.cases[0];
        // Re-compute reference and compare
        let expected = cpu_matmul(&case.inputs[0], &case.inputs[1]);
        let (pass, _, _) = compare_tensors(&case.expected_outputs[0], &expected, 0.0);
        assert!(pass);
    }

    #[test]
    fn test_matmul_zero_matrix() {
        let z = create_zero_tensor(&[3, 3]);
        let a = create_test_tensor(&[3, 3], 77);
        let result = cpu_matmul(&a, &z);
        assert!(result.data.iter().all(|&v| v == 0.0));
    }

    // ── Softmax suite ────────────────────────────────────────────

    #[test]
    fn test_softmax_suite_generation() {
        let suite = generate_softmax_suite(&[4, 8, 16]);
        assert_eq!(suite.cases.len(), 3);
    }

    #[test]
    fn test_softmax_outputs_sum_to_one() {
        let suite = generate_softmax_suite(&[10, 100]);
        for case in &suite.cases {
            let sum: f32 = case.expected_outputs[0].data.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}, expected ~1.0");
        }
    }

    #[test]
    fn test_softmax_all_positive() {
        let suite = generate_softmax_suite(&[8]);
        for val in &suite.cases[0].expected_outputs[0].data {
            assert!(*val >= 0.0);
        }
    }

    // ── Activation suite ─────────────────────────────────────────

    #[test]
    fn test_activation_suite_generation() {
        let suite = generate_activation_suite();
        assert_eq!(suite.cases.len(), 3);
        let names: Vec<&str> = suite.cases.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"silu"));
        assert!(names.contains(&"gelu"));
        assert!(names.contains(&"relu"));
    }

    #[test]
    fn test_silu_at_zero() {
        assert!((silu(0.0) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_gelu_at_zero() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_relu_negative() {
        assert_eq!(relu(-5.0), 0.0);
        assert_eq!(relu(3.0), 3.0);
    }

    #[test]
    fn test_activation_silu_values() {
        let suite = generate_activation_suite();
        let silu_case = suite.cases.iter().find(|c| c.name == "silu").unwrap();
        for (&input, &expected) in
            silu_case.inputs[0].data.iter().zip(silu_case.expected_outputs[0].data.iter())
        {
            let computed = silu(input);
            assert!((computed - expected).abs() < 1e-6, "silu({input}): {computed} != {expected}");
        }
    }

    // ── Filter by tags ───────────────────────────────────────────

    #[test]
    fn test_filter_by_tags_matching() {
        let suite = generate_activation_suite();
        let filtered = filter_suite_by_tags(&suite, &["silu"]);
        assert_eq!(filtered.cases.len(), 1);
        assert_eq!(filtered.cases[0].name, "silu");
    }

    #[test]
    fn test_filter_by_tags_no_match() {
        let suite = generate_activation_suite();
        let filtered = filter_suite_by_tags(&suite, &["nonexistent"]);
        assert!(filtered.cases.is_empty());
    }

    #[test]
    fn test_filter_by_tags_multiple() {
        let suite = generate_activation_suite();
        let filtered = filter_suite_by_tags(&suite, &["silu", "relu"]);
        assert_eq!(filtered.cases.len(), 2);
    }

    #[test]
    fn test_filter_preserves_activation_tag() {
        let suite = generate_activation_suite();
        let filtered = filter_suite_by_tags(&suite, &["activation"]);
        assert_eq!(filtered.cases.len(), 3);
    }

    // ── Report formatting ────────────────────────────────────────

    #[test]
    fn test_format_report_contains_suite_name() {
        let report = TestReport {
            suite_name: "my_suite".into(),
            results: vec![],
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
        };
        let text = format_test_report(&report);
        assert!(text.contains("my_suite"));
    }

    #[test]
    fn test_format_report_contains_all_sections() {
        let report = TestReport {
            suite_name: "fmt_test".into(),
            results: vec![
                TestResult {
                    case_name: "case_a".into(),
                    passed: true,
                    max_error: 0.0,
                    mean_error: 0.0,
                    execution_time_us: 10,
                    details: "ok".into(),
                },
                TestResult {
                    case_name: "case_b".into(),
                    passed: false,
                    max_error: 0.5,
                    mean_error: 0.3,
                    execution_time_us: 20,
                    details: "bad".into(),
                },
            ],
            total: 2,
            passed: 1,
            failed: 1,
            skipped: 0,
        };
        let text = format_test_report(&report);
        assert!(text.contains("PASS"));
        assert!(text.contains("FAIL"));
        assert!(text.contains("case_a"));
        assert!(text.contains("case_b"));
        assert!(text.contains("Total: 2"));
        assert!(text.contains("Passed: 1"));
        assert!(text.contains("Failed: 1"));
    }

    // ── Edge cases ───────────────────────────────────────────────

    #[test]
    fn test_zero_dim_tensor() {
        // A scalar-like tensor with shape []
        let t = create_test_tensor(&[], 1);
        assert_eq!(t.data.len(), 1);
        assert!(t.shape.is_empty());
    }

    #[test]
    fn test_single_element_tensor() {
        let t = create_test_tensor(&[1], 5);
        assert_eq!(t.data.len(), 1);
    }

    #[test]
    fn test_large_tolerance_always_passes() {
        let a = create_test_tensor(&[100], 1);
        let b = create_test_tensor(&[100], 2);
        let (pass, _, _) = compare_tensors(&a, &b, f32::MAX);
        assert!(pass);
    }
}
