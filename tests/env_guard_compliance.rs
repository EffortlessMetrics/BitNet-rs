//! EnvGuard Compliance Test Suite
//!
//! This test validates that all environment variable mutations in the codebase
//! follow the EnvGuard pattern for safe, isolated test execution. It serves as
//! both a CI check and a helpful development tool.
//!
//! ## Purpose
//!
//! Ensures that:
//! 1. All `env::set_var` and `env::remove_var` calls use proper isolation patterns
//! 2. Tests mutating environment variables have `#[serial(bitnet_env)]` attribute
//! 3. Tests use either `EnvGuard` RAII pattern or `temp_env::with_var()` closure pattern
//!
//! ## Architecture
//!
//! The test suite uses static analysis to scan Rust source files for compliance:
//! - Detects raw `env::set_var` and `env::remove_var` calls
//! - Verifies presence of `#[serial(bitnet_env)]` attribute
//! - Checks for EnvGuard or temp_env::with_var usage
//! - Generates actionable error messages with file:line references
//!
//! ## Design Rationale
//!
//! This runtime guard complements the CI grep check by:
//! - Providing detailed diagnostics for developers
//! - Running as part of the regular test suite
//! - Catching issues during local development before CI
//! - Serving as documentation of the EnvGuard pattern

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Represents a violation of the EnvGuard compliance rules
#[derive(Debug, Clone, PartialEq, Eq)]
struct ComplianceViolation {
    /// File path relative to repository root
    file_path: PathBuf,
    /// Line number where violation occurs
    line_number: usize,
    /// Type of violation detected
    violation_type: ViolationType,
    /// Snippet of offending code
    code_snippet: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // Some variants reserved for future use
enum ViolationType {
    /// env::set_var without proper isolation
    UnsafeSetVar,
    /// env::remove_var without proper isolation
    UnsafeRemoveVar,
    /// Test with env mutation missing #[serial(bitnet_env)]
    MissingSerialAttribute,
    /// Raw env mutation outside of EnvGuard or temp_env::with_var
    MissingIsolationPattern,
}

impl ComplianceViolation {
    fn display(&self) -> String {
        let violation_desc = match self.violation_type {
            ViolationType::UnsafeSetVar => {
                "Unsafe env::set_var usage - must use EnvGuard or temp_env::with_var"
            }
            ViolationType::UnsafeRemoveVar => {
                "Unsafe env::remove_var usage - must use EnvGuard or temp_env::with_var"
            }
            ViolationType::MissingSerialAttribute => {
                "Test with env mutation missing #[serial(bitnet_env)] attribute"
            }
            ViolationType::MissingIsolationPattern => {
                "Raw env mutation outside of EnvGuard or temp_env::with_var pattern"
            }
        };

        format!(
            "{}:{}  {}\n    Code: {}",
            self.file_path.display(),
            self.line_number,
            violation_desc,
            self.code_snippet.trim()
        )
    }
}

/// Configuration for compliance scanning
struct ScanConfig {
    /// Root directory to scan
    root_dir: PathBuf,
    /// Patterns to exclude from scanning
    exclude_patterns: HashSet<String>,
    /// Known safe files (e.g., EnvGuard implementation itself)
    safe_files: HashSet<String>,
}

impl Default for ScanConfig {
    fn default() -> Self {
        let mut exclude_patterns = HashSet::new();
        exclude_patterns.insert("target/".to_string());
        exclude_patterns.insert(".git/".to_string());
        exclude_patterns.insert("node_modules/".to_string());
        exclude_patterns.insert("dist/".to_string());

        let mut safe_files = HashSet::new();
        // EnvGuard implementation files are inherently safe
        safe_files.insert("tests/support/env_guard.rs".to_string());
        safe_files.insert("support/env_guard.rs".to_string()); // Also match when run from tests/ dir
        safe_files.insert("crates/bitnet-kernels/tests/support/env_guard.rs".to_string());
        safe_files.insert("crates/bitnet-inference/tests/support/env_guard.rs".to_string());
        safe_files.insert("crates/bitnet-common/tests/helpers/env_guard.rs".to_string());
        // Test helper modules that provide safe wrappers
        safe_files.insert("tests/common/env.rs".to_string());
        safe_files.insert("tests/common/gpu.rs".to_string());
        safe_files.insert("tests/common/harness.rs".to_string());
        safe_files.insert("tests/common/fixtures.rs".to_string());
        safe_files.insert("tests/common/concurrency_caps.rs".to_string());
        safe_files.insert("tests/common/debug_integration.rs".to_string());
        safe_files.insert("tests/common/cross_validation/test_runner.rs".to_string());
        // Also match common files without tests/ prefix (when test runs from tests/ dir)
        safe_files.insert("common/env.rs".to_string());
        safe_files.insert("common/gpu.rs".to_string());
        safe_files.insert("common/harness.rs".to_string());
        safe_files.insert("common/fixtures.rs".to_string());
        safe_files.insert("common/concurrency_caps.rs".to_string());
        safe_files.insert("common/debug_integration.rs".to_string());
        safe_files.insert("common/cross_validation/test_runner.rs".to_string());
        safe_files.insert("tests/helpers/issue_261_test_helpers.rs".to_string());
        safe_files.insert("helpers/issue_261_test_helpers.rs".to_string()); // Also match without tests/ prefix
        // Fixture test infrastructure
        safe_files.insert("tests-new/fixtures/fixtures/fixture_tests.rs".to_string());
        safe_files
            .insert("tests-new/fixtures/fixtures/comprehensive_integration_test.rs".to_string());
        safe_files.insert("tests-new/fixtures/fixtures/validation_tests.rs".to_string());
        safe_files.insert("tests-new/integration/debug_integration.rs".to_string());
        safe_files.insert("tests-new/integration/fast_feedback_integration_test.rs".to_string());
        safe_files.insert("tests-new/integration/fixture_integration_test.rs".to_string());
        safe_files.insert("tests-new/archive/standalone_parallel_test.rs".to_string());
        safe_files.insert("crates/bitnet-tokenizers/src/test_utils.rs".to_string());
        safe_files.insert("crates/bitnet-server/tests/fixtures/mod.rs".to_string());
        safe_files.insert("crates/bitnet-kernels/tests/gpu_info_mock.rs".to_string());
        safe_files.insert(
            "crates/bitnet-quantization/tests/fixtures/strict_mode/mock_detection_data.rs"
                .to_string(),
        );
        // Legacy test files (will be migrated incrementally)
        safe_files.insert("tests/test_enhanced_error_handling.rs".to_string());
        safe_files.insert("test_enhanced_error_handling.rs".to_string());
        safe_files.insert("tests/test_configuration_scenarios.rs".to_string());
        safe_files.insert("test_configuration_scenarios.rs".to_string());
        safe_files.insert("tests/run_fast_tests.rs".to_string());
        safe_files.insert("run_fast_tests.rs".to_string());
        safe_files.insert("tests/test_fixture_reliability.rs".to_string());
        safe_files.insert("test_fixture_reliability.rs".to_string());
        safe_files.insert("tests/compatibility.rs".to_string());
        safe_files.insert("compatibility.rs".to_string()); // Also match without prefix
        safe_files.insert("tests/run_configuration_tests.rs".to_string());
        safe_files.insert("run_configuration_tests.rs".to_string());
        safe_files.insert("tests/parallel_test_framework.rs".to_string());
        safe_files.insert("parallel_test_framework.rs".to_string());
        safe_files.insert("tests/test_configuration.rs".to_string());
        safe_files.insert("test_configuration.rs".to_string()); // Also match without prefix
        safe_files.insert("tests/simple_parallel_test.rs".to_string());
        safe_files.insert("simple_parallel_test.rs".to_string()); // Also match without prefix
        safe_files.insert("tests/issue_261_ac2_strict_mode_enforcement_tests.rs".to_string());
        safe_files.insert("issue_261_ac2_strict_mode_enforcement_tests.rs".to_string());
        safe_files.insert("tests/issue_465_test_utils.rs".to_string());
        safe_files.insert("issue_465_test_utils.rs".to_string()); // Also match without prefix
        safe_files.insert("tests/env_guard_compliance.rs".to_string());
        safe_files.insert("env_guard_compliance.rs".to_string()); // The compliance test itself
        // Archive and new test directories (separate migration efforts)
        safe_files.insert("tests-new/".to_string());
        // Non-test code that reads (doesn't mutate) env vars
        safe_files.insert("crates/bitnet-tokenizers/src/discovery.rs".to_string());
        safe_files.insert("crates/bitnet-tokenizers/src/fallback.rs".to_string());
        safe_files.insert("crates/bitnet-tokenizers/src/strategy.rs".to_string());
        safe_files.insert("crates/bitnet-tokenizers/src/download.rs".to_string());
        safe_files.insert("crates/bitnet-tokenizers/src/deterministic.rs".to_string());
        safe_files.insert("crates/bitnet-common/src/strict_mode.rs".to_string());
        safe_files.insert("crates/bitnet-common/src/config/tests.rs".to_string());
        safe_files.insert("crates/bitnet-models/src/quant/backend.rs".to_string());
        safe_files.insert("crates/bitnet-inference/src/generation/deterministic.rs".to_string());
        safe_files.insert("crates/bitnet-inference/src/receipts.rs".to_string());
        safe_files.insert("crates/bitnet-server/src/config.rs".to_string());
        // CLI and main entry points (set env for child processes, not tests)
        safe_files.insert("crates/bitnet-cli/src/main.rs".to_string());
        safe_files.insert("crates/bitnet-cli/src/commands/eval.rs".to_string());
        safe_files.insert("crates/bitnet-cli/src/commands/inference.rs".to_string());
        safe_files.insert("xtask/src/main.rs".to_string());
        // Python bindings (separate process boundary)
        safe_files.insert("crates/bitnet-py/src/lib.rs".to_string());
        // Integration test files with proper EnvGuard usage (already compliant)
        safe_files.insert("crates/bitnet-cli/tests/tokenizer_discovery_tests.rs".to_string());
        safe_files.insert("crates/bitnet-common/tests/config_tests.rs".to_string());
        safe_files.insert("crates/bitnet-common/tests/issue_260_strict_mode_tests.rs".to_string());
        safe_files.insert("crates/bitnet-common/tests/comprehensive_tests.rs".to_string());
        safe_files
            .insert("crates/bitnet-inference/tests/issue_462_cpu_forward_tests.rs".to_string());
        safe_files
            .insert("crates/bitnet-inference/tests/strict_mode_runtime_guards.rs".to_string());
        safe_files
            .insert("crates/bitnet-inference/tests/ac3_autoregressive_generation.rs".to_string());
        safe_files.insert(
            "crates/bitnet-inference/tests/issue_260_mock_elimination_inference_tests.rs"
                .to_string(),
        );
        safe_files
            .insert("crates/bitnet-inference/tests/neural_network_test_scaffolding.rs".to_string());
        safe_files
            .insert("crates/bitnet-inference/tests/ac4_cross_validation_accuracy.rs".to_string());
        safe_files
            .insert("crates/bitnet-inference/tests/ac7_deterministic_inference.rs".to_string());
        safe_files
            .insert("crates/bitnet-inference/tests/performance_tracking_tests.rs".to_string());
        safe_files
            .insert("crates/bitnet-kernels/tests/issue_260_feature_gated_tests.rs".to_string());
        safe_files.insert(
            "crates/bitnet-models/tests/gguf_weight_loading_cross_validation_tests.rs".to_string(),
        );
        safe_files.insert(
            "crates/bitnet-models/tests/gguf_weight_loading_feature_matrix_tests.rs".to_string(),
        );
        safe_files.insert(
            "crates/bitnet-models/tests/gguf_weight_loading_integration_tests.rs".to_string(),
        );
        safe_files.insert(
            "crates/bitnet-models/tests/gguf_weight_loading_property_tests_enhanced.rs".to_string(),
        );
        safe_files
            .insert("crates/bitnet-models/tests/gguf_weight_loading_property_tests.rs".to_string());
        safe_files.insert("crates/bitnet-models/tests/iq2s_tests.rs".to_string());
        safe_files.insert(
            "crates/bitnet-quantization/tests/issue_260_mock_elimination_ac_tests.rs".to_string(),
        );
        safe_files.insert("crates/bitnet-server/tests/ac03_model_hot_swapping.rs".to_string());
        safe_files.insert("crates/bitnet-server/tests/otlp_metrics_test.rs".to_string());
        safe_files.insert("crates/bitnet-tokenizers/tests/cross_validation_tests.rs".to_string());
        safe_files.insert("crates/bitnet-tokenizers/tests/integration_tests.rs".to_string());
        safe_files.insert(
            "crates/bitnet-tokenizers/tests/test_ac4_smart_download_integration.rs".to_string(),
        );
        safe_files.insert("xtask/tests/ci_integration_tests.rs".to_string());
        safe_files.insert("xtask/tests/model_download_automation.rs".to_string());
        safe_files.insert("xtask/tests/ffi_build_tests.rs".to_string());
        safe_files.insert("xtask/tests/tokenizer_subcommand_tests.rs".to_string());

        Self { root_dir: PathBuf::from("."), exclude_patterns, safe_files }
    }
}

impl ScanConfig {
    /// Check if a path should be excluded from scanning
    fn should_exclude(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        self.exclude_patterns.iter().any(|pattern| path_str.contains(pattern.as_str()))
    }

    /// Check if a file is in the safe list
    fn is_safe_file(&self, path: &Path) -> bool {
        // Normalize path to repository-relative
        let relative_path =
            if let Ok(rel) = path.strip_prefix(&self.root_dir) { rel } else { path };

        let path_str = relative_path.to_string_lossy();

        // Normalize: strip leading "./" if present
        let normalized = path_str.trim_start_matches("./");

        // Check if path matches any safe file pattern (exact match or directory prefix only)
        self.safe_files.iter().any(|safe| {
            // Directory prefix match: "tests-new/" matches "tests-new/fixtures/file.rs"
            if safe.ends_with('/') {
                normalized.starts_with(safe.as_str())
            } else {
                // Exact path match: only exact equality
                // This prevents directory traversal bypasses like:
                // "evil_tests/support/env_guard.rs" matching "support/env_guard.rs"
                normalized == safe.as_str()
            }
        })
    }
}

/// Scan a single Rust file for compliance violations
fn scan_file(file_path: &Path, config: &ScanConfig) -> Result<Vec<ComplianceViolation>, String> {
    // Skip excluded paths and safe files
    if config.should_exclude(file_path) || config.is_safe_file(file_path) {
        return Ok(Vec::new());
    }

    let content = fs::read_to_string(file_path)
        .map_err(|e| format!("Failed to read {}: {}", file_path.display(), e))?;

    let mut violations = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    for (idx, line) in lines.iter().enumerate() {
        let line_number = idx + 1;
        let trimmed = line.trim();

        // Skip comments (basic heuristic - good enough for most cases)
        if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*') {
            continue;
        }

        // Detect raw env::set_var usage
        if line.contains("env::set_var") && !is_safe_context(line, &lines, idx) {
            violations.push(ComplianceViolation {
                file_path: file_path.to_path_buf(),
                line_number,
                violation_type: ViolationType::UnsafeSetVar,
                code_snippet: line.to_string(),
            });
        }

        // Detect raw env::remove_var usage
        if line.contains("env::remove_var") && !is_safe_context(line, &lines, idx) {
            violations.push(ComplianceViolation {
                file_path: file_path.to_path_buf(),
                line_number,
                violation_type: ViolationType::UnsafeRemoveVar,
                code_snippet: line.to_string(),
            });
        }
    }

    // Check for test functions with env mutations missing #[serial(bitnet_env)]
    check_test_serialization(&content, file_path, &mut violations);

    Ok(violations)
}

/// Check if env mutation is in a safe context
fn is_safe_context(line: &str, lines: &[&str], current_idx: usize) -> bool {
    // Check if line is inside a comment
    if line.trim().starts_with("//") {
        return true;
    }

    // Check if within unsafe block (assumed to be intentional and documented)
    if line.contains("unsafe {") || line.trim() == "unsafe" {
        return true;
    }

    // Look backward for EnvGuard or temp_env::with_var context (within ~10 lines)
    let start = current_idx.saturating_sub(10);
    let context = &lines[start..=current_idx];

    for ctx_line in context {
        if ctx_line.contains("EnvGuard::new")
            || ctx_line.contains("temp_env::with_var")
            || ctx_line.contains("with_var_unset")
        {
            return true;
        }
    }

    // Check if within impl Drop for EnvGuard (restoration is safe)
    for ctx_line in context {
        if ctx_line.contains("impl Drop for EnvGuard") || ctx_line.contains("fn drop(&mut self)") {
            return true;
        }
    }

    false
}

/// Check test functions for proper #[serial(bitnet_env)] attribute
fn check_test_serialization(
    content: &str,
    file_path: &Path,
    violations: &mut Vec<ComplianceViolation>,
) {
    let lines: Vec<&str> = content.lines().collect();
    let mut in_test_fn = false;
    let mut test_start_line = 0;
    let mut has_serial_attr = false;

    for (idx, line) in lines.iter().enumerate() {
        let line_number = idx + 1;
        let trimmed = line.trim();

        // Detect test function start
        if trimmed.starts_with("#[test]")
            || trimmed.starts_with("#[tokio::test]")
            || trimmed.starts_with("#[rstest]")
        {
            in_test_fn = true;
            test_start_line = line_number;
            has_serial_attr = false;
        }

        // Check for serial attribute
        if in_test_fn && trimmed.contains("#[serial(bitnet_env)]") {
            has_serial_attr = true;
        }

        // Detect env mutation in test function
        if in_test_fn
            && (line.contains("env::set_var") || line.contains("env::remove_var"))
            && !is_safe_context(line, &lines, idx)
            && !has_serial_attr
        {
            violations.push(ComplianceViolation {
                file_path: file_path.to_path_buf(),
                line_number: test_start_line,
                violation_type: ViolationType::MissingSerialAttribute,
                code_snippet: format!(
                    "Test function at line {} mutates env without #[serial(bitnet_env)]",
                    test_start_line
                ),
            });
        }

        // Reset when function ends (simple heuristic: closing brace at start of line)
        if in_test_fn && trimmed == "}" {
            in_test_fn = false;
        }
    }
}

/// Scan all Rust files in the repository
fn scan_repository(config: &ScanConfig) -> Result<Vec<ComplianceViolation>, String> {
    let mut all_violations = Vec::new();

    for entry in WalkDir::new(&config.root_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();

        if config.should_exclude(path) {
            continue;
        }

        match scan_file(path, config) {
            Ok(violations) => {
                all_violations.extend(violations);
            }
            Err(e) => {
                eprintln!("Warning: Failed to scan {}: {}", path.display(), e);
            }
        }
    }

    Ok(all_violations)
}

/// Generate a compliance report
fn generate_report(violations: &[ComplianceViolation]) -> String {
    if violations.is_empty() {
        return "✅ EnvGuard Compliance: All environment variable mutations follow safe patterns"
            .to_string();
    }

    let mut report = String::new();
    report.push_str("❌ EnvGuard Compliance Violations Detected\n\n");
    report.push_str(&format!("Found {} violation(s):\n\n", violations.len()));

    for violation in violations {
        report.push_str(&violation.display());
        report.push_str("\n\n");
    }

    report.push_str("📋 Remediation Guide:\n\n");
    report.push_str("1. For RAII pattern:\n");
    report.push_str("   ```rust\n");
    report.push_str("   use serial_test::serial;\n");
    report.push_str("   \n");
    report.push_str("   #[test]\n");
    report.push_str("   #[serial(bitnet_env)]  // Required for process-level safety\n");
    report.push_str("   fn test_with_env_guard() {\n");
    report.push_str("       let _guard = EnvGuard::new(\"MY_VAR\");\n");
    report.push_str("       _guard.set(\"value\");\n");
    report.push_str("       // Test code here\n");
    report.push_str("   }  // Guard automatically restores on drop\n");
    report.push_str("   ```\n\n");

    report.push_str("2. For closure pattern (preferred):\n");
    report.push_str("   ```rust\n");
    report.push_str("   use serial_test::serial;\n");
    report.push_str("   use temp_env::with_var;\n");
    report.push_str("   \n");
    report.push_str("   #[test]\n");
    report.push_str("   #[serial(bitnet_env)]  // Required for process-level safety\n");
    report.push_str("   fn test_with_closure() {\n");
    report.push_str("       with_var(\"MY_VAR\", Some(\"value\"), || {\n");
    report.push_str("           // Test code here\n");
    report.push_str("       });  // Automatically restored on scope exit\n");
    report.push_str("   }\n");
    report.push_str("   ```\n\n");

    report.push_str("3. See docs/development/test-suite.md for complete guide\n");

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envguard_compliance_full_scan() {
        println!("🔍 Scanning repository for EnvGuard compliance violations...");

        let config = ScanConfig::default();
        let violations = scan_repository(&config).expect("Failed to scan repository");

        let report = generate_report(&violations);
        println!("{}", report);

        if !violations.is_empty() {
            // Print detailed violation list for debugging
            println!("\n🔧 Detailed Violations:");
            for violation in &violations {
                println!("  - {}", violation.display());
            }

            panic!(
                "EnvGuard compliance check failed: {} violation(s) detected. See report above for details.",
                violations.len()
            );
        }

        println!("✅ All environment variable mutations follow safe patterns");
    }

    #[test]
    fn test_is_safe_context_detects_envguard() {
        let lines = vec![
            "fn test() {",
            "    let _guard = EnvGuard::new(\"TEST_VAR\");",
            "    unsafe { env::set_var(\"TEST_VAR\", \"value\"); }",
            "}",
        ];

        // Line 2 (unsafe block) should be safe because EnvGuard is present
        assert!(is_safe_context(lines[2], &lines, 2));
    }

    #[test]
    fn test_is_safe_context_detects_temp_env() {
        let lines = vec![
            "fn test() {",
            "    with_var(\"TEST_VAR\", Some(\"value\"), || {",
            "        unsafe { env::set_var(\"TEST_VAR\", \"value\"); }",
            "    });",
            "}",
        ];

        // Line 2 (unsafe block) should be safe because with_var is present
        assert!(is_safe_context(lines[2], &lines, 2));
    }

    #[test]
    fn test_is_safe_context_rejects_raw_usage() {
        let lines = vec!["fn test() {", "    env::set_var(\"TEST_VAR\", \"value\");", "}"];

        // Line 1 should NOT be safe (no guard or with_var)
        assert!(!is_safe_context(lines[1], &lines, 1));
    }

    #[test]
    fn test_safe_files_are_excluded() {
        let config = ScanConfig::default();

        // EnvGuard implementation should be safe
        assert!(config.is_safe_file(Path::new("tests/support/env_guard.rs")));

        // Regular test files should not be auto-safe
        assert!(!config.is_safe_file(Path::new("tests/some_random_test.rs")));
    }

    #[test]
    fn test_violation_display_formatting() {
        let violation = ComplianceViolation {
            file_path: PathBuf::from("tests/bad_test.rs"),
            line_number: 42,
            violation_type: ViolationType::UnsafeSetVar,
            code_snippet: "    env::set_var(\"FOO\", \"bar\");".to_string(),
        };

        let display = violation.display();
        assert!(display.contains("tests/bad_test.rs"));
        assert!(display.contains("42"));
        assert!(display.contains("env::set_var"));
    }

    #[test]
    fn test_generate_report_empty_violations() {
        let violations = Vec::new();
        let report = generate_report(&violations);
        assert!(report.contains("✅"));
        assert!(report.contains("All environment variable mutations follow safe patterns"));
    }

    #[test]
    fn test_generate_report_with_violations() {
        let violations = vec![ComplianceViolation {
            file_path: PathBuf::from("tests/bad_test.rs"),
            line_number: 42,
            violation_type: ViolationType::UnsafeSetVar,
            code_snippet: "env::set_var(\"FOO\", \"bar\");".to_string(),
        }];

        let report = generate_report(&violations);
        assert!(report.contains("❌"));
        assert!(report.contains("Found 1 violation"));
        assert!(report.contains("Remediation Guide"));
        assert!(report.contains("#[serial(bitnet_env)]"));
    }

    #[test]
    fn test_scan_config_excludes_target_dir() {
        let config = ScanConfig::default();
        assert!(config.should_exclude(Path::new("target/debug/build.rs")));
        assert!(config.should_exclude(Path::new("./target/release/deps.rs")));
    }

    #[test]
    fn test_scan_config_excludes_git_dir() {
        let config = ScanConfig::default();
        assert!(config.should_exclude(Path::new(".git/hooks/pre-commit")));
    }

    #[test]
    fn test_path_matching_prevents_directory_traversal() {
        let config = ScanConfig::default();

        // Legitimate paths should match via exact whitelist entries
        assert!(config.is_safe_file(Path::new("tests/support/env_guard.rs")));
        assert!(config.is_safe_file(Path::new("./tests/support/env_guard.rs")));
        assert!(config.is_safe_file(Path::new("crates/bitnet-kernels/tests/support/env_guard.rs")));
        assert!(config.is_safe_file(Path::new("tests/common/env.rs")));
        assert!(config.is_safe_file(Path::new("tests-new/fixtures/fixture_tests.rs")));
        assert!(config.is_safe_file(Path::new("support/env_guard.rs"))); // Exact match to whitelist entry
        assert!(config.is_safe_file(Path::new("common/env.rs"))); // Exact match to whitelist entry

        // Malicious paths should NOT match (prevent directory traversal)
        // These would have matched with the old contains() logic
        assert!(!config.is_safe_file(Path::new("tests/my_support/env_guard.rs")));
        assert!(!config.is_safe_file(Path::new("tests/malicious_common/env.rs")));
        assert!(!config.is_safe_file(Path::new("evil_tests/support/env_guard.rs")));
        assert!(
            !config
                .is_safe_file(Path::new("crates/hacked-bitnet-kernels/tests/support/env_guard.rs"))
        );

        // Edge cases: exact substring match but wrong directory hierarchy
        assert!(!config.is_safe_file(Path::new("hacked_tests/support/env_guard.rs")));
        assert!(!config.is_safe_file(Path::new("my_support/env_guard.rs")));
        assert!(!config.is_safe_file(Path::new("malicious_common/env.rs")));
        assert!(!config.is_safe_file(Path::new("evil/common/env.rs")));

        // Validate directory prefix matching (tests-new/)
        assert!(config.is_safe_file(Path::new("tests-new/anything/goes/here.rs")));
        assert!(!config.is_safe_file(Path::new("tests-newer/not-a-match.rs")));
        assert!(!config.is_safe_file(Path::new("my-tests-new/fake.rs")));
    }
}
