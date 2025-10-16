//! Test scaffolding for Issue #465: CPU Path Followup - CI Gate Tests
//!
//! Work Stream 3: CI Gate Enforcement (AC5, AC6)
//!
//! Tests feature spec: docs/explanation/issue-465-implementation-spec.md
//!
//! This test suite validates:
//! - AC5: Branch protection configuration with Model Gates (CPU)
//! - AC6: Smoke test validating CI blocks mocked receipts

mod issue_465_test_utils;

use anyhow::{Context, Result};
use issue_465_test_utils::{configure_deterministic_env, create_test_receipt, workspace_root};
use std::fs;
use std::path::Path;

/// Tests feature spec: issue-465-implementation-spec.md#ac5-branch-protection-configuration
///
/// Validates that branch protection is configured to enforce Model Gates (CPU):
/// - GitHub branch protection rules exist for main branch
/// - Required status check "Model Gates (CPU)" is present
/// - Model gates workflow is operational
///
/// Note: This test validates workflow configuration only. Branch protection settings
/// require GitHub API access and must be verified manually via GitHub settings.
#[test]
#[ignore = "Requires GitHub API access for branch protection verification"]
fn test_ac5_branch_protection_configured() -> Result<()> {
    // AC5: Branch protection validation
    let root = workspace_root();

    // Verify model-gates.yml workflow exists
    let workflow_path = root.join(".github/workflows/model-gates.yml");

    assert!(workflow_path.exists(), "Model Gates workflow not found: {:?}", workflow_path);

    let workflow_content =
        fs::read_to_string(&workflow_path).context("Failed to read model-gates.yml")?;

    // Verify workflow contains CPU job
    assert!(
        workflow_content.contains("cpu") || workflow_content.contains("CPU"),
        "Model Gates workflow missing CPU job"
    );

    // Verify workflow uses verify-receipt command
    assert!(
        workflow_content.contains("verify-receipt") || workflow_content.contains("verify_receipt"),
        "Model Gates workflow missing verify-receipt command"
    );

    // Verify deterministic environment configuration
    let required_env_vars = vec!["BITNET_DETERMINISTIC", "BITNET_SEED", "RAYON_NUM_THREADS"];

    for env_var in &required_env_vars {
        assert!(
            workflow_content.contains(env_var),
            "Model Gates workflow missing environment variable: {}",
            env_var
        );
    }

    // Evidence tag for validation
    println!("// AC5: Model Gates workflow validated");

    // FIXME: This test cannot verify GitHub branch protection settings via filesystem
    // Expected: Branch protection enforces "Model Gates (CPU)" status check
    // Actual: Requires GitHub API or admin dashboard verification
    //
    // To verify manually:
    // 1. Navigate to: https://github.com/EffortlessMetrics/BitNet-rs/settings/branches
    // 2. Check main branch protection rules
    // 3. Verify "Model Gates (CPU)" in required status checks
    //
    // Uncomment when GitHub CLI integration is ready:
    // let output = std::process::Command::new("gh")
    //     .args(&["api", "repos/:owner/:repo/branches/main/protection"])
    //     .output()
    //     .context("Failed to fetch branch protection settings")?;
    //
    // if output.status.success() {
    //     let protection_json: Value = serde_json::from_slice(&output.stdout)?;
    //     let required_checks = protection_json["required_status_checks"]["contexts"]
    //         .as_array()
    //         .context("Missing required_status_checks")?;
    //
    //     let has_model_gates_cpu = required_checks.iter().any(|check| {
    //         check.as_str().map_or(false, |s| s.contains("Model Gates") && s.contains("CPU"))
    //     });
    //
    //     assert!(
    //         has_model_gates_cpu,
    //         "Branch protection missing 'Model Gates (CPU)' required check"
    //     );
    // }

    panic!(
        "AC5 implementation incomplete: GitHub branch protection verification requires admin access or GitHub CLI integration"
    );
}

/// Tests feature spec: issue-465-implementation-spec.md#ac6-smoke-test-validation
///
/// Validates that CI blocks PRs with mocked receipts:
/// - Mocked receipt (compute_path: "mocked") fails verification
/// - CI rejects receipts without real kernel IDs
/// - Smoke test validates honest compute enforcement
#[test]
fn test_ac6_mocked_receipt_rejected() -> Result<()> {
    // AC6: Smoke test validation
    configure_deterministic_env();

    // Create temporary test directory for test fixtures
    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Test Case 1: Mocked receipt should fail
    let mocked_receipt = create_test_receipt("mocked", vec![]);
    let mocked_path = temp_dir.path().join("mocked-receipt.json");
    write_test_receipt(&mocked_path, &mocked_receipt)?;

    assert!(
        issue_465_test_utils::verify_receipt_schema(&mocked_path).is_err(),
        "Mocked receipt (compute_path='mocked') should fail verification but passed"
    );

    // Test Case 2: Empty kernels should fail
    let empty_kernels_receipt = create_test_receipt("real", vec![]);
    let empty_path = temp_dir.path().join("empty-kernels.json");
    write_test_receipt(&empty_path, &empty_kernels_receipt)?;

    assert!(
        issue_465_test_utils::verify_receipt_schema(&empty_path).is_err(),
        "Receipt with empty kernels array should fail verification but passed"
    );

    // Test Case 3: Invalid kernel IDs should fail
    let invalid_kernels = vec!["".to_string(), "a".repeat(129)];
    let invalid_receipt = create_test_receipt("real", invalid_kernels);
    let invalid_path = temp_dir.path().join("invalid-kernels.json");
    write_test_receipt(&invalid_path, &invalid_receipt)?;

    assert!(
        issue_465_test_utils::verify_receipt_schema(&invalid_path).is_err(),
        "Receipt with invalid kernel IDs (empty string or >128 chars) should fail verification but passed"
    );

    // Test Case 4: Valid receipt should pass
    let valid_kernels =
        vec!["i2s_cpu_quantized_matmul".to_string(), "tl1_lut_dequant_forward".to_string()];
    let valid_receipt = create_test_receipt("real", valid_kernels);
    let valid_path = temp_dir.path().join("valid-receipt.json");
    write_test_receipt(&valid_path, &valid_receipt)?;

    issue_465_test_utils::verify_receipt_schema(&valid_path)
        .context("Valid receipt with real compute_path and proper kernels should pass verification but failed")?;

    // Evidence tag for validation
    println!("// AC6: Smoke test validated - CI properly blocks invalid receipts");
    println!("//   ✓ Mocked receipt (compute_path='mocked'): REJECTED");
    println!("//   ✓ Empty kernels: REJECTED");
    println!("//   ✓ Invalid kernel IDs: REJECTED");
    println!("//   ✓ Valid receipt: ACCEPTED");

    Ok(())
}

/// Write test receipt to file
fn write_test_receipt(path: &Path, receipt: &serde_json::Value) -> Result<()> {
    fs::write(path, serde_json::to_string_pretty(receipt)?)
        .with_context(|| format!("Failed to write test receipt to {}", path.display()))
}

/// Comprehensive test: Kernel ID hygiene - type safety
///
/// This test validates that kernel IDs are always strings, never other types.
#[test]
fn test_comprehensive_kernel_id_type_safety() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Test kernel IDs with non-string types
    let test_cases = vec![
        (
            serde_json::json!({"version": "1.0.0", "compute_path": "real", "kernels": [123], "performance": {"tokens_per_sec": 10.0}}),
            "numeric-kernel-id.json",
        ),
        (
            serde_json::json!({"version": "1.0.0", "compute_path": "real", "kernels": [true], "performance": {"tokens_per_sec": 10.0}}),
            "boolean-kernel-id.json",
        ),
        (
            serde_json::json!({"version": "1.0.0", "compute_path": "real", "kernels": [null], "performance": {"tokens_per_sec": 10.0}}),
            "null-kernel-id.json",
        ),
        (
            serde_json::json!({"version": "1.0.0", "compute_path": "real", "kernels": [{"nested": "object"}], "performance": {"tokens_per_sec": 10.0}}),
            "object-kernel-id.json",
        ),
    ];

    for (receipt, filename) in test_cases {
        let receipt_path = temp_dir.path().join(filename);
        write_test_receipt(&receipt_path, &receipt)?;

        assert!(
            issue_465_test_utils::verify_receipt_schema(&receipt_path).is_err(),
            "Receipt {} with non-string kernel ID should be rejected",
            filename
        );
    }

    println!("// Comprehensive test passed: Kernel ID type safety enforced");
    Ok(())
}

/// Comprehensive test: Multiple invalid receipt patterns
///
/// This test validates comprehensive receipt validation across multiple failure modes.
#[test]
fn test_comprehensive_multiple_invalid_patterns() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    let invalid_patterns = vec![
        // Empty string kernel IDs
        (
            create_test_receipt("real", vec!["".to_string()]),
            "empty-string-kernel.json",
            "empty string kernel ID",
        ),
        // Mixed valid and invalid kernels
        (
            create_test_receipt("real", vec!["valid_kernel".to_string(), "".to_string()]),
            "mixed-kernels.json",
            "mixed valid/invalid kernels",
        ),
        // Whitespace-only kernel IDs
        (
            create_test_receipt("real", vec!["   ".to_string()]),
            "whitespace-kernel.json",
            "whitespace-only kernel ID",
        ),
        // Special characters in excessive length
        (
            create_test_receipt("real", vec!["k".repeat(129)]),
            "excessive-length.json",
            "excessive length kernel ID",
        ),
    ];

    for (receipt, filename, description) in invalid_patterns {
        let receipt_path = temp_dir.path().join(filename);
        write_test_receipt(&receipt_path, &receipt)?;

        assert!(
            issue_465_test_utils::verify_receipt_schema(&receipt_path).is_err(),
            "Receipt with {} should be rejected",
            description
        );
    }

    println!("// Comprehensive test passed: Multiple invalid patterns detected");
    Ok(())
}

/// Comprehensive test: CI workflow file structure validation
///
/// This test validates that CI workflow files have proper structure.
#[test]
fn test_comprehensive_ci_workflow_structure() -> Result<()> {
    let root = workspace_root();
    let workflows_dir = root.join(".github/workflows");

    if !workflows_dir.exists() {
        println!("// Note: No .github/workflows directory found");
        return Ok(());
    }

    // Check for model-gates workflow
    let model_gates_path = workflows_dir.join("model-gates.yml");
    if model_gates_path.exists() {
        let content =
            fs::read_to_string(&model_gates_path).context("Failed to read model-gates.yml")?;

        // Validate workflow structure
        let required_elements = vec![
            ("name:", "workflow name"),
            ("on:", "trigger configuration"),
            ("jobs:", "job definitions"),
            ("runs-on:", "runner specification"),
        ];

        for (pattern, description) in required_elements {
            assert!(
                content.contains(pattern),
                "Model Gates workflow missing {}: {}",
                description,
                pattern
            );
        }

        println!("// Comprehensive test passed: CI workflow structure validated");
    } else {
        println!("// Note: model-gates.yml not found (may be named differently)");
    }

    Ok(())
}

/// Comprehensive test: Kernel hygiene with realistic patterns
///
/// This test validates kernel ID hygiene with patterns from real CPU/GPU kernels.
#[test]
fn test_comprehensive_kernel_hygiene_realistic_patterns() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Test realistic CPU kernel patterns
    let cpu_kernels = vec![
        "i2s_cpu_quantized_matmul".to_string(),
        "tl1_lut_dequant_forward".to_string(),
        "tl2_lut_backward".to_string(),
        "cpu_attention_qkvo".to_string(),
        "quantized_matmul_impl".to_string(),
    ];

    let cpu_receipt = create_test_receipt("real", cpu_kernels);
    let cpu_path = temp_dir.path().join("cpu-kernels.json");
    write_test_receipt(&cpu_path, &cpu_receipt)?;

    issue_465_test_utils::verify_receipt_schema(&cpu_path)
        .context("Valid CPU kernel pattern should pass")?;

    // Test realistic GPU kernel patterns (should also be valid structure)
    let gpu_kernels = vec![
        "gemm_gpu_fp16".to_string(),
        "cuda_i2s_quantize".to_string(),
        "gpu_attention_flash".to_string(),
    ];

    let gpu_receipt = create_test_receipt("real", gpu_kernels);
    let gpu_path = temp_dir.path().join("gpu-kernels.json");
    write_test_receipt(&gpu_path, &gpu_receipt)?;

    issue_465_test_utils::verify_receipt_schema(&gpu_path)
        .context("Valid GPU kernel pattern should pass")?;

    println!("// Comprehensive test passed: Realistic kernel patterns validated");
    Ok(())
}

/// Comprehensive test: Receipt schema version compatibility
///
/// This test validates backward compatibility with different schema versions.
#[test]
fn test_comprehensive_schema_version_compatibility() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Test both accepted schema versions
    let valid_versions = vec!["1.0.0", "1.0"];

    for version in valid_versions {
        let mut receipt = create_test_receipt("real", vec!["test_kernel".to_string()]);
        receipt["version"] = serde_json::json!(version);

        let receipt_path =
            temp_dir.path().join(format!("version-{}.json", version.replace('.', "_")));
        write_test_receipt(&receipt_path, &receipt)?;

        issue_465_test_utils::verify_receipt_schema(&receipt_path)
            .with_context(|| format!("Schema version {} should be accepted", version))?;
    }

    println!("// Comprehensive test passed: Schema version compatibility validated");
    Ok(())
}

/// Comprehensive test: Compute path validation strictness
///
/// This test validates that only "real" compute paths are accepted.
#[test]
fn test_comprehensive_compute_path_strictness() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Test invalid compute paths
    let invalid_paths = vec![
        "mocked",
        "mock",
        "fake",
        "simulated",
        "test",
        "",
        "Real", // Case-sensitive
        "REAL",
    ];

    for compute_path in invalid_paths {
        let mut receipt = create_test_receipt("real", vec!["test_kernel".to_string()]);
        receipt["compute_path"] = serde_json::json!(compute_path);

        let receipt_path = temp_dir.path().join(format!("compute-path-{}.json", compute_path));
        write_test_receipt(&receipt_path, &receipt)?;

        assert!(
            issue_465_test_utils::verify_receipt_schema(&receipt_path).is_err(),
            "Compute path '{}' should be rejected",
            compute_path
        );
    }

    println!("// Comprehensive test passed: Compute path strictness enforced");
    Ok(())
}

/// Comprehensive test: Performance metrics edge cases
///
/// This test validates edge cases in performance metric validation.
#[test]
fn test_comprehensive_performance_edge_cases() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Test edge case performance values
    // Note: f64::NAN and f64::INFINITY serialize as JSON null via serde_json
    let test_cases = vec![
        (0.0, true, "zero-performance.json", "zero performance (valid for initialization)"),
        (0.1, true, "minimal-performance.json", "minimal viable performance"),
        (50.0, true, "high-performance.json", "high CPU performance"),
        (-0.1, false, "negative-performance.json", "negative performance"),
        (
            f64::INFINITY,
            false,
            "infinite-performance.json",
            "infinite performance (serializes as null)",
        ),
        (f64::NAN, false, "nan-performance.json", "NaN performance (serializes as null)"),
    ];

    for (perf_value, should_pass, filename, description) in test_cases {
        let mut receipt = create_test_receipt("real", vec!["test_kernel".to_string()]);
        receipt["performance"]["tokens_per_sec"] = serde_json::json!(perf_value);

        let receipt_path = temp_dir.path().join(filename);
        write_test_receipt(&receipt_path, &receipt)?;

        let result = issue_465_test_utils::verify_receipt_schema(&receipt_path);

        if should_pass {
            result.with_context(|| format!("Receipt with {} should pass", description))?;
        } else {
            assert!(result.is_err(), "Receipt with {} should be rejected", description);
        }
    }

    println!("// Comprehensive test passed: Performance edge cases validated");
    Ok(())
}
