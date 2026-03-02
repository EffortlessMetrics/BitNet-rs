//! OpenCL GPU health check module for validating GPU functionality.
//!
//! Provides diagnostic checks for device detection, memory allocation,
//! kernel compilation, compute correctness, numerical precision, bandwidth,
//! and thermal status. CPU reference implementations simulate GPU behavior
//! for validation and testing without requiring actual OpenCL hardware.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// CheckCategory
// ---------------------------------------------------------------------------

/// Category of a health check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckCategory {
    DeviceDetection,
    MemoryAllocation,
    KernelCompilation,
    ComputeCorrectness,
    NumericalPrecision,
    Bandwidth,
    Thermal,
}

impl fmt::Display for CheckCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceDetection => write!(f, "Device Detection"),
            Self::MemoryAllocation => write!(f, "Memory Allocation"),
            Self::KernelCompilation => write!(f, "Kernel Compilation"),
            Self::ComputeCorrectness => write!(f, "Compute Correctness"),
            Self::NumericalPrecision => write!(f, "Numerical Precision"),
            Self::Bandwidth => write!(f, "Bandwidth"),
            Self::Thermal => write!(f, "Thermal"),
        }
    }
}

// ---------------------------------------------------------------------------
// CheckResult
// ---------------------------------------------------------------------------

/// Result of a single health check.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckResult {
    Pass,
    Fail(String),
    Warning(String),
    Skip(String),
}

impl CheckResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass)
    }

    pub fn is_fail(&self) -> bool {
        matches!(self, Self::Fail(_))
    }

    pub fn is_warning(&self) -> bool {
        matches!(self, Self::Warning(_))
    }

    pub fn is_skip(&self) -> bool {
        matches!(self, Self::Skip(_))
    }
}

impl fmt::Display for CheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail(msg) => write!(f, "FAIL: {msg}"),
            Self::Warning(msg) => write!(f, "WARN: {msg}"),
            Self::Skip(msg) => write!(f, "SKIP: {msg}"),
        }
    }
}

// ---------------------------------------------------------------------------
// HealthCheck
// ---------------------------------------------------------------------------

/// A single health check result.
#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: String,
    pub category: CheckCategory,
    pub result: CheckResult,
    pub duration_ms: u64,
    pub details: String,
}

// ---------------------------------------------------------------------------
// HealthCheckSuite
// ---------------------------------------------------------------------------

/// Aggregated results from running a suite of health checks.
#[derive(Debug, Clone)]
pub struct HealthCheckSuite {
    pub checks: Vec<HealthCheck>,
    pub passed: usize,
    pub failed: usize,
    pub warnings: usize,
    pub total_time_ms: u64,
}

impl HealthCheckSuite {
    /// Create an empty suite.
    pub fn new() -> Self {
        Self { checks: Vec::new(), passed: 0, failed: 0, warnings: 0, total_time_ms: 0 }
    }

    /// Add a check and update tallies.
    pub fn add(&mut self, check: HealthCheck) {
        match &check.result {
            CheckResult::Pass => self.passed += 1,
            CheckResult::Fail(_) => self.failed += 1,
            CheckResult::Warning(_) => self.warnings += 1,
            CheckResult::Skip(_) => {} // skipped checks don't count
        }
        self.total_time_ms += check.duration_ms;
        self.checks.push(check);
    }
}

impl Default for HealthCheckSuite {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DeviceHealth
// ---------------------------------------------------------------------------

/// Summary of device health status.
#[derive(Debug, Clone)]
pub struct DeviceHealth {
    pub device_name: String,
    pub driver_version: String,
    pub memory_total_mb: u64,
    pub memory_available_mb: u64,
    pub temperature_c: Option<f32>,
    pub health_score: f32,
}

// ---------------------------------------------------------------------------
// DiagnosticReport
// ---------------------------------------------------------------------------

/// Full diagnostic report combining device health, suite results, and recommendations.
#[derive(Debug, Clone)]
pub struct DiagnosticReport {
    pub device_health: DeviceHealth,
    pub suite_results: HealthCheckSuite,
    pub recommendations: Vec<String>,
    pub timestamp_ns: u64,
}

// ---------------------------------------------------------------------------
// HealthChecker
// ---------------------------------------------------------------------------

/// Stateful health checker bound to a specific device.
#[derive(Debug, Clone)]
pub struct HealthChecker {
    pub device_name: String,
    pub checks_run: u64,
    pub issues_found: u64,
}

// ---------------------------------------------------------------------------
// HealthError
// ---------------------------------------------------------------------------

/// Errors that can occur during health checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthError {
    DeviceNotFound,
    CheckFailed(String),
    Timeout,
}

impl fmt::Display for HealthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound => write!(f, "device not found"),
            Self::CheckFailed(msg) => write!(f, "check failed: {msg}"),
            Self::Timeout => write!(f, "health check timed out"),
        }
    }
}

impl std::error::Error for HealthError {}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Create a new health checker for the given device.
pub fn create_health_checker(device_name: &str) -> HealthChecker {
    HealthChecker { device_name: device_name.to_string(), checks_run: 0, issues_found: 0 }
}

/// Verify that the device exists (CPU stub matches on name substring).
pub fn cpu_check_device_detection(checker: &HealthChecker) -> HealthCheck {
    let start = Instant::now();
    let known = ["Intel(R) Arc(TM) A770", "Intel(R) Arc(TM) A750", "Intel(R) Arc(TM) A580"];
    let result = if known.iter().any(|d| checker.device_name.contains(d)) {
        CheckResult::Pass
    } else if checker.device_name.to_lowercase().contains("intel") {
        CheckResult::Warning(format!(
            "Unrecognized Intel device '{}'; may work with reduced support",
            checker.device_name,
        ))
    } else {
        CheckResult::Fail(format!("Device '{}' not recognized", checker.device_name))
    };
    let details = format!("Checked device: {}", checker.device_name);
    HealthCheck {
        name: "device_detection".into(),
        category: CheckCategory::DeviceDetection,
        result,
        duration_ms: start.elapsed().as_millis() as u64,
        details,
    }
}

/// Simulate a memory allocation of `size_mb` megabytes.
pub fn cpu_check_memory_allocation(checker: &HealthChecker, size_mb: usize) -> HealthCheck {
    let _ = checker;
    let start = Instant::now();
    // Simulate A770 having 16 GB VRAM
    let total_mb: usize = 16_384;
    let available_mb = total_mb * 90 / 100; // 90% available

    let result = if size_mb == 0 {
        CheckResult::Fail("Requested allocation of 0 MB".into())
    } else if size_mb > total_mb {
        CheckResult::Fail(format!(
            "Requested {size_mb} MB exceeds total device memory ({total_mb} MB)"
        ))
    } else if size_mb > available_mb {
        CheckResult::Warning(format!(
            "Requested {size_mb} MB is near device memory limit ({available_mb} MB available)"
        ))
    } else {
        CheckResult::Pass
    };
    let details =
        format!("Requested {size_mb} MB; device total {total_mb} MB, available ~{available_mb} MB");
    HealthCheck {
        name: "memory_allocation".into(),
        category: CheckCategory::MemoryAllocation,
        result,
        duration_ms: start.elapsed().as_millis() as u64,
        details,
    }
}

/// Validate OpenCL kernel source syntax (CPU stub uses basic heuristics).
pub fn cpu_check_kernel_compilation(checker: &HealthChecker, source: &str) -> HealthCheck {
    let _ = checker;
    let start = Instant::now();

    let result = if source.trim().is_empty() {
        CheckResult::Fail("Empty kernel source".into())
    } else if !source.contains("__kernel") && !source.contains("kernel void") {
        CheckResult::Fail("Source missing kernel entry point (__kernel or kernel void)".into())
    } else if source.matches('{').count() != source.matches('}').count() {
        CheckResult::Fail("Mismatched braces in kernel source".into())
    } else {
        CheckResult::Pass
    };
    let details = format!("Validated kernel source ({} bytes)", source.len());
    HealthCheck {
        name: "kernel_compilation".into(),
        category: CheckCategory::KernelCompilation,
        result,
        duration_ms: start.elapsed().as_millis() as u64,
        details,
    }
}

/// Check compute correctness via a small matmul reference comparison.
pub fn cpu_check_compute_correctness(checker: &HealthChecker) -> HealthCheck {
    let _ = checker;
    let start = Instant::now();

    // 2×2 matmul: A * B = C
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];
    // expected: [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
    let expected = [19.0_f32, 22.0, 43.0, 50.0];

    let mut c = [0.0_f32; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                c[i * 2 + j] += a[i * 2 + k] * b[k * 2 + j];
            }
        }
    }

    let max_err = c.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);

    let result = if max_err < 1e-6 {
        CheckResult::Pass
    } else {
        CheckResult::Fail(format!("Matmul max error {max_err:.6e} exceeds tolerance 1e-6"))
    };
    let details = format!("2×2 matmul reference check; max error = {max_err:.6e}");
    HealthCheck {
        name: "compute_correctness".into(),
        category: CheckCategory::ComputeCorrectness,
        result,
        duration_ms: start.elapsed().as_millis() as u64,
        details,
    }
}

/// Check numerical precision of floating-point operations.
pub fn cpu_check_numerical_precision(checker: &HealthChecker) -> HealthCheck {
    let _ = checker;
    let start = Instant::now();

    // Kahan summation test: sum 1/i for i in 1..=10000
    let n = 10_000u32;
    let mut sum = 0.0_f32;
    let mut compensation = 0.0_f32;
    for i in 1..=n {
        let y = 1.0 / i as f32 - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    // Reference (f64 for ground truth)
    let reference: f64 = (1..=n as u64).map(|i| 1.0 / i as f64).sum();

    let rel_err = ((sum as f64 - reference) / reference).abs();

    let result = if rel_err < 1e-5 {
        CheckResult::Pass
    } else if rel_err < 1e-3 {
        CheckResult::Warning(format!("Reduced float precision: relative error {rel_err:.6e}"))
    } else {
        CheckResult::Fail(format!("Numerical precision failure: relative error {rel_err:.6e}"))
    };
    let details = format!(
        "Harmonic series sum (n={n}): f32={sum:.6}, f64_ref={reference:.6}, rel_err={rel_err:.2e}"
    );
    HealthCheck {
        name: "numerical_precision".into(),
        category: CheckCategory::NumericalPrecision,
        result,
        duration_ms: start.elapsed().as_millis() as u64,
        details,
    }
}

/// Simulate bandwidth measurement for the given transfer size.
pub fn cpu_check_bandwidth(checker: &HealthChecker, size_mb: usize) -> HealthCheck {
    let _ = checker;
    let start = Instant::now();

    if size_mb == 0 {
        return HealthCheck {
            name: "bandwidth".into(),
            category: CheckCategory::Bandwidth,
            result: CheckResult::Fail("Transfer size must be > 0".into()),
            duration_ms: start.elapsed().as_millis() as u64,
            details: "Cannot measure bandwidth with 0 MB transfer".into(),
        };
    }

    // Simulate: A770 theoretical peak ~560 GB/s, practical ~400 GB/s
    let simulated_gbps = 400.0_f64;
    let transfer_time_ms = (size_mb as f64 / 1024.0) / simulated_gbps * 1000.0;

    let result = if simulated_gbps > 200.0 {
        CheckResult::Pass
    } else if simulated_gbps > 50.0 {
        CheckResult::Warning(format!("Bandwidth {simulated_gbps:.1} GB/s below expected"))
    } else {
        CheckResult::Fail(format!("Bandwidth critically low: {simulated_gbps:.1} GB/s"))
    };
    let details = format!(
        "Simulated {size_mb} MB transfer at {simulated_gbps:.1} GB/s ({transfer_time_ms:.3} ms)"
    );
    HealthCheck {
        name: "bandwidth".into(),
        category: CheckCategory::Bandwidth,
        result,
        duration_ms: start.elapsed().as_millis() as u64,
        details,
    }
}

/// Run the full suite of health checks.
pub fn cpu_run_full_suite(checker: &mut HealthChecker) -> HealthCheckSuite {
    let mut suite = HealthCheckSuite::new();

    let valid_kernel = "__kernel void test(__global float* out) { out[get_global_id(0)] = 1.0f; }";

    suite.add(cpu_check_device_detection(checker));
    suite.add(cpu_check_memory_allocation(checker, 1024));
    suite.add(cpu_check_kernel_compilation(checker, valid_kernel));
    suite.add(cpu_check_compute_correctness(checker));
    suite.add(cpu_check_numerical_precision(checker));
    suite.add(cpu_check_bandwidth(checker, 256));

    checker.checks_run = suite.checks.len() as u64;
    checker.issues_found = suite.failed as u64 + suite.warnings as u64;

    suite
}

/// Compute a health score in [0.0, 1.0] from suite results.
pub fn cpu_compute_health_score(suite: &HealthCheckSuite) -> f32 {
    let total = suite.passed + suite.failed + suite.warnings;
    if total == 0 {
        return 1.0;
    }
    // Failures count as 0, warnings as 0.5, passes as 1.0
    let score = (suite.passed as f32 + suite.warnings as f32 * 0.5) / total as f32;
    score.clamp(0.0, 1.0)
}

/// Generate actionable recommendations based on check results.
pub fn cpu_generate_recommendations(suite: &HealthCheckSuite) -> Vec<String> {
    let mut recs = Vec::new();
    for check in &suite.checks {
        match &check.result {
            CheckResult::Fail(msg) => {
                let rec = match check.category {
                    CheckCategory::DeviceDetection => {
                        format!(
                            "Device detection failed ({msg}). Verify driver installation and device visibility."
                        )
                    }
                    CheckCategory::MemoryAllocation => {
                        format!(
                            "Memory allocation failed ({msg}). Reduce model size or close other GPU applications."
                        )
                    }
                    CheckCategory::KernelCompilation => {
                        format!(
                            "Kernel compilation failed ({msg}). Check OpenCL source for syntax errors."
                        )
                    }
                    CheckCategory::ComputeCorrectness => {
                        format!(
                            "Compute correctness failed ({msg}). Possible hardware fault or driver bug."
                        )
                    }
                    CheckCategory::NumericalPrecision => {
                        format!(
                            "Numerical precision failed ({msg}). Consider enabling FP32 accumulation."
                        )
                    }
                    CheckCategory::Bandwidth => {
                        format!("Bandwidth check failed ({msg}). Check PCIe link width and speed.")
                    }
                    CheckCategory::Thermal => {
                        format!("Thermal check failed ({msg}). Ensure adequate cooling.")
                    }
                };
                recs.push(rec);
            }
            CheckResult::Warning(msg) => {
                recs.push(format!(
                    "{} warning: {msg}. Monitor during extended workloads.",
                    check.category,
                ));
            }
            CheckResult::Pass | CheckResult::Skip(_) => {}
        }
    }
    recs
}

/// Generate a full diagnostic report.
pub fn cpu_generate_report(checker: &HealthChecker, suite: &HealthCheckSuite) -> DiagnosticReport {
    let score = cpu_compute_health_score(suite);
    let recommendations = cpu_generate_recommendations(suite);
    let timestamp_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    DiagnosticReport {
        device_health: DeviceHealth {
            device_name: checker.device_name.clone(),
            driver_version: "simulated-cpu-stub".into(),
            memory_total_mb: 16_384,
            memory_available_mb: 14_745,
            temperature_c: None,
            health_score: score,
        },
        suite_results: suite.clone(),
        recommendations,
        timestamp_ns,
    }
}

/// Check whether the device is usable (no critical failures).
pub fn cpu_is_device_usable(suite: &HealthCheckSuite) -> bool {
    suite.failed == 0
}

/// Format a diagnostic report as a human-readable string.
pub fn format_diagnostic_report(report: &DiagnosticReport) -> String {
    let mut out = String::new();
    out.push_str("=== GPU Health Check Report ===\n");
    out.push_str(&format!("Device: {}\n", report.device_health.device_name));
    out.push_str(&format!("Driver: {}\n", report.device_health.driver_version));
    out.push_str(&format!(
        "Memory: {} / {} MB\n",
        report.device_health.memory_available_mb, report.device_health.memory_total_mb,
    ));
    if let Some(temp) = report.device_health.temperature_c {
        out.push_str(&format!("Temperature: {temp:.1} °C\n"));
    }
    out.push_str(&format!("Health Score: {:.1}%\n\n", report.device_health.health_score * 100.0,));

    out.push_str("--- Check Results ---\n");
    for check in &report.suite_results.checks {
        out.push_str(&format!(
            "[{}] {} — {} ({}ms)\n",
            check.result, check.name, check.details, check.duration_ms,
        ));
    }
    out.push_str(&format!(
        "\nSummary: {} passed, {} failed, {} warnings ({}ms total)\n",
        report.suite_results.passed,
        report.suite_results.failed,
        report.suite_results.warnings,
        report.suite_results.total_time_ms,
    ));

    if !report.recommendations.is_empty() {
        out.push_str("\n--- Recommendations ---\n");
        for (i, rec) in report.recommendations.iter().enumerate() {
            out.push_str(&format!("{}. {rec}\n", i + 1));
        }
    }
    out
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- create_health_checker -----------------------------------------------

    #[test]
    fn test_create_checker_device_name() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        assert_eq!(checker.device_name, "Intel(R) Arc(TM) A770 Graphics");
    }

    #[test]
    fn test_create_checker_initial_counters() {
        let checker = create_health_checker("test-device");
        assert_eq!(checker.checks_run, 0);
        assert_eq!(checker.issues_found, 0);
    }

    // -- device detection ----------------------------------------------------

    #[test]
    fn test_device_detection_passes_a770() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let check = cpu_check_device_detection(&checker);
        assert!(check.result.is_pass());
        assert_eq!(check.category, CheckCategory::DeviceDetection);
    }

    #[test]
    fn test_device_detection_passes_a750() {
        let checker = create_health_checker("Intel(R) Arc(TM) A750 Graphics");
        let check = cpu_check_device_detection(&checker);
        assert!(check.result.is_pass());
    }

    #[test]
    fn test_device_detection_warns_unknown_intel() {
        let checker = create_health_checker("Intel UHD 770");
        let check = cpu_check_device_detection(&checker);
        assert!(check.result.is_warning());
    }

    #[test]
    fn test_device_detection_fails_unknown() {
        let checker = create_health_checker("NVIDIA RTX 4090");
        let check = cpu_check_device_detection(&checker);
        assert!(check.result.is_fail());
    }

    // -- memory allocation ---------------------------------------------------

    #[test]
    fn test_memory_allocation_passes_reasonable() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let check = cpu_check_memory_allocation(&checker, 4096);
        assert!(check.result.is_pass());
    }

    #[test]
    fn test_memory_allocation_warns_near_limit() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        // 15000 MB > 90% of 16384 = 14745
        let check = cpu_check_memory_allocation(&checker, 15_000);
        assert!(check.result.is_warning());
    }

    #[test]
    fn test_memory_allocation_fails_exceeds_total() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let check = cpu_check_memory_allocation(&checker, 20_000);
        assert!(check.result.is_fail());
    }

    #[test]
    fn test_memory_allocation_fails_zero() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let check = cpu_check_memory_allocation(&checker, 0);
        assert!(check.result.is_fail());
    }

    // -- kernel compilation --------------------------------------------------

    #[test]
    fn test_kernel_compilation_valid_source() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let src = "__kernel void add(__global float* a, __global float* b) { int i = get_global_id(0); a[i] += b[i]; }";
        let check = cpu_check_kernel_compilation(&checker, src);
        assert!(check.result.is_pass());
    }

    #[test]
    fn test_kernel_compilation_kernel_void_syntax() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let src = "kernel void test(__global float* o) { o[0] = 1.0f; }";
        let check = cpu_check_kernel_compilation(&checker, src);
        assert!(check.result.is_pass());
    }

    #[test]
    fn test_kernel_compilation_empty_source() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let check = cpu_check_kernel_compilation(&checker, "");
        assert!(check.result.is_fail());
    }

    #[test]
    fn test_kernel_compilation_no_entry_point() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let check = cpu_check_kernel_compilation(&checker, "void helper(int x) { }");
        assert!(check.result.is_fail());
    }

    #[test]
    fn test_kernel_compilation_mismatched_braces() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let src = "__kernel void test(__global float* o) { o[0] = 1.0f; ";
        let check = cpu_check_kernel_compilation(&checker, src);
        assert!(check.result.is_fail());
    }

    // -- compute correctness -------------------------------------------------

    #[test]
    fn test_compute_correctness_passes() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let check = cpu_check_compute_correctness(&checker);
        assert!(check.result.is_pass());
    }

    #[test]
    fn test_compute_correctness_category() {
        let checker = create_health_checker("test");
        let check = cpu_check_compute_correctness(&checker);
        assert_eq!(check.category, CheckCategory::ComputeCorrectness);
    }

    // -- numerical precision -------------------------------------------------

    #[test]
    fn test_numerical_precision_passes() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let check = cpu_check_numerical_precision(&checker);
        // Kahan summation in f32 should be within tolerance
        assert!(
            check.result.is_pass() || check.result.is_warning(),
            "Expected pass or warning, got {:?}",
            check.result,
        );
    }

    #[test]
    fn test_numerical_precision_has_details() {
        let checker = create_health_checker("test");
        let check = cpu_check_numerical_precision(&checker);
        assert!(check.details.contains("Harmonic"));
    }

    // -- bandwidth -----------------------------------------------------------

    #[test]
    fn test_bandwidth_reasonable_estimate() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let check = cpu_check_bandwidth(&checker, 256);
        assert!(check.result.is_pass());
    }

    #[test]
    fn test_bandwidth_fails_zero_size() {
        let checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let check = cpu_check_bandwidth(&checker, 0);
        assert!(check.result.is_fail());
    }

    #[test]
    fn test_bandwidth_has_details() {
        let checker = create_health_checker("test");
        let check = cpu_check_bandwidth(&checker, 128);
        assert!(check.details.contains("128 MB"));
    }

    // -- full suite ----------------------------------------------------------

    #[test]
    fn test_full_suite_all_checks_run() {
        let mut checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let suite = cpu_run_full_suite(&mut checker);
        assert_eq!(suite.checks.len(), 6);
        assert!(checker.checks_run == 6);
    }

    #[test]
    fn test_full_suite_a770_all_pass() {
        let mut checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let suite = cpu_run_full_suite(&mut checker);
        assert_eq!(suite.failed, 0);
        assert_eq!(suite.passed, 6);
    }

    #[test]
    fn test_full_suite_unknown_device_has_failure() {
        let mut checker = create_health_checker("Unknown GPU");
        let suite = cpu_run_full_suite(&mut checker);
        assert!(suite.failed > 0);
    }

    #[test]
    fn test_full_suite_updates_checker_issues() {
        let mut checker = create_health_checker("Unknown GPU");
        let suite = cpu_run_full_suite(&mut checker);
        assert_eq!(checker.issues_found, suite.failed as u64 + suite.warnings as u64);
    }

    // -- health score --------------------------------------------------------

    #[test]
    fn test_health_score_all_pass() {
        let mut checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let suite = cpu_run_full_suite(&mut checker);
        let score = cpu_compute_health_score(&suite);
        assert!((score - 1.0).abs() < f32::EPSILON, "Expected 1.0, got {score}");
    }

    #[test]
    fn test_health_score_with_failures() {
        let mut suite = HealthCheckSuite::new();
        suite.add(HealthCheck {
            name: "test".into(),
            category: CheckCategory::DeviceDetection,
            result: CheckResult::Fail("fail".into()),
            duration_ms: 0,
            details: String::new(),
        });
        suite.add(HealthCheck {
            name: "test2".into(),
            category: CheckCategory::Bandwidth,
            result: CheckResult::Pass,
            duration_ms: 0,
            details: String::new(),
        });
        let score = cpu_compute_health_score(&suite);
        assert!(score < 1.0, "Score should be < 1.0 with failures, got {score}");
        assert!(score > 0.0, "Score should be > 0.0 with one pass, got {score}");
    }

    #[test]
    fn test_health_score_empty_suite() {
        let suite = HealthCheckSuite::new();
        let score = cpu_compute_health_score(&suite);
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_health_score_all_warnings() {
        let mut suite = HealthCheckSuite::new();
        for i in 0..3 {
            suite.add(HealthCheck {
                name: format!("w{i}"),
                category: CheckCategory::Thermal,
                result: CheckResult::Warning("warm".into()),
                duration_ms: 0,
                details: String::new(),
            });
        }
        let score = cpu_compute_health_score(&suite);
        assert!((score - 0.5).abs() < f32::EPSILON, "Expected 0.5, got {score}");
    }

    // -- recommendations -----------------------------------------------------

    #[test]
    fn test_recommendations_for_failures() {
        let mut suite = HealthCheckSuite::new();
        suite.add(HealthCheck {
            name: "mem".into(),
            category: CheckCategory::MemoryAllocation,
            result: CheckResult::Fail("OOM".into()),
            duration_ms: 0,
            details: String::new(),
        });
        let recs = cpu_generate_recommendations(&suite);
        assert!(!recs.is_empty());
        assert!(recs[0].contains("Memory allocation failed"));
    }

    #[test]
    fn test_recommendations_empty_for_all_pass() {
        let mut checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let suite = cpu_run_full_suite(&mut checker);
        let recs = cpu_generate_recommendations(&suite);
        assert!(recs.is_empty());
    }

    #[test]
    fn test_recommendations_for_warnings() {
        let mut suite = HealthCheckSuite::new();
        suite.add(HealthCheck {
            name: "bw".into(),
            category: CheckCategory::Bandwidth,
            result: CheckResult::Warning("low".into()),
            duration_ms: 0,
            details: String::new(),
        });
        let recs = cpu_generate_recommendations(&suite);
        assert_eq!(recs.len(), 1);
        assert!(recs[0].contains("warning"));
    }

    // -- report --------------------------------------------------------------

    #[test]
    fn test_report_contains_all_sections() {
        let mut checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let suite = cpu_run_full_suite(&mut checker);
        let report = cpu_generate_report(&checker, &suite);
        assert_eq!(report.device_health.device_name, "Intel(R) Arc(TM) A770 Graphics");
        assert!(!report.suite_results.checks.is_empty());
        assert!(report.timestamp_ns > 0);
    }

    #[test]
    fn test_report_health_score_matches() {
        let mut checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let suite = cpu_run_full_suite(&mut checker);
        let report = cpu_generate_report(&checker, &suite);
        let expected = cpu_compute_health_score(&suite);
        assert!((report.device_health.health_score - expected).abs() < f32::EPSILON);
    }

    // -- device usable -------------------------------------------------------

    #[test]
    fn test_device_usable_true_no_failures() {
        let mut checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let suite = cpu_run_full_suite(&mut checker);
        assert!(cpu_is_device_usable(&suite));
    }

    #[test]
    fn test_device_usable_false_with_failure() {
        let mut suite = HealthCheckSuite::new();
        suite.add(HealthCheck {
            name: "bad".into(),
            category: CheckCategory::ComputeCorrectness,
            result: CheckResult::Fail("wrong answer".into()),
            duration_ms: 0,
            details: String::new(),
        });
        assert!(!cpu_is_device_usable(&suite));
    }

    // -- edge cases ----------------------------------------------------------

    #[test]
    fn test_empty_suite_is_usable() {
        let suite = HealthCheckSuite::new();
        assert!(cpu_is_device_usable(&suite));
    }

    #[test]
    fn test_all_checks_skip() {
        let mut suite = HealthCheckSuite::new();
        for i in 0..4 {
            suite.add(HealthCheck {
                name: format!("skip{i}"),
                category: CheckCategory::Thermal,
                result: CheckResult::Skip("no hardware".into()),
                duration_ms: 0,
                details: String::new(),
            });
        }
        assert_eq!(suite.passed, 0);
        assert_eq!(suite.failed, 0);
        assert_eq!(suite.warnings, 0);
        assert!(cpu_is_device_usable(&suite));
        assert!((cpu_compute_health_score(&suite) - 1.0).abs() < f32::EPSILON);
    }

    // -- property tests ------------------------------------------------------

    #[test]
    fn test_property_score_in_range() {
        for pass in 0..=5 {
            for fail in 0..=5 {
                for warn in 0..=5 {
                    let mut suite = HealthCheckSuite::new();
                    for i in 0..pass {
                        suite.add(HealthCheck {
                            name: format!("p{i}"),
                            category: CheckCategory::DeviceDetection,
                            result: CheckResult::Pass,
                            duration_ms: 0,
                            details: String::new(),
                        });
                    }
                    for i in 0..fail {
                        suite.add(HealthCheck {
                            name: format!("f{i}"),
                            category: CheckCategory::DeviceDetection,
                            result: CheckResult::Fail("x".into()),
                            duration_ms: 0,
                            details: String::new(),
                        });
                    }
                    for i in 0..warn {
                        suite.add(HealthCheck {
                            name: format!("w{i}"),
                            category: CheckCategory::DeviceDetection,
                            result: CheckResult::Warning("x".into()),
                            duration_ms: 0,
                            details: String::new(),
                        });
                    }
                    let score = cpu_compute_health_score(&suite);
                    assert!(
                        (0.0..=1.0).contains(&score),
                        "Score {score} out of range for p={pass} f={fail} w={warn}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_property_passed_failed_warnings_eq_total() {
        let mut checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let suite = cpu_run_full_suite(&mut checker);
        let counted = suite.passed + suite.failed + suite.warnings;
        // Skipped checks don't count toward pass/fail/warn but are in checks vec
        let non_skipped = suite.checks.iter().filter(|c| !c.result.is_skip()).count();
        assert_eq!(counted, non_skipped);
    }

    // -- format report -------------------------------------------------------

    #[test]
    fn test_format_report_contains_device() {
        let mut checker = create_health_checker("Intel(R) Arc(TM) A770 Graphics");
        let suite = cpu_run_full_suite(&mut checker);
        let report = cpu_generate_report(&checker, &suite);
        let formatted = format_diagnostic_report(&report);
        assert!(formatted.contains("Intel(R) Arc(TM) A770 Graphics"));
        assert!(formatted.contains("Health Score"));
        assert!(formatted.contains("PASS"));
    }

    #[test]
    fn test_format_report_with_recommendations() {
        let mut checker = create_health_checker("Unknown GPU");
        let suite = cpu_run_full_suite(&mut checker);
        let report = cpu_generate_report(&checker, &suite);
        let formatted = format_diagnostic_report(&report);
        assert!(formatted.contains("Recommendations"));
    }

    // -- HealthError ---------------------------------------------------------

    #[test]
    fn test_health_error_display() {
        assert_eq!(HealthError::DeviceNotFound.to_string(), "device not found");
        assert_eq!(HealthError::Timeout.to_string(), "health check timed out");
        assert!(HealthError::CheckFailed("x".into()).to_string().contains("x"));
    }

    #[test]
    fn test_health_error_equality() {
        assert_eq!(HealthError::DeviceNotFound, HealthError::DeviceNotFound);
        assert_eq!(HealthError::Timeout, HealthError::Timeout);
        assert_ne!(HealthError::DeviceNotFound, HealthError::Timeout);
    }

    // -- CheckResult helpers -------------------------------------------------

    #[test]
    fn test_check_result_display() {
        assert_eq!(CheckResult::Pass.to_string(), "PASS");
        assert!(CheckResult::Fail("err".into()).to_string().contains("FAIL"));
        assert!(CheckResult::Warning("w".into()).to_string().contains("WARN"));
        assert!(CheckResult::Skip("s".into()).to_string().contains("SKIP"));
    }

    #[test]
    fn test_check_result_predicates() {
        assert!(CheckResult::Pass.is_pass());
        assert!(!CheckResult::Pass.is_fail());
        assert!(CheckResult::Fail("x".into()).is_fail());
        assert!(CheckResult::Warning("x".into()).is_warning());
        assert!(CheckResult::Skip("x".into()).is_skip());
    }
}
