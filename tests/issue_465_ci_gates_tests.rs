//! Test scaffolding for Issue #465: CPU Path Followup - CI Gate Tests
//!
//! Work Stream 3: CI Gate Enforcement (AC5, AC6)
//!
//! Tests feature spec: docs/explanation/issue-465-implementation-spec.md
//!
//! This test suite validates:
//! - AC5: Branch protection configuration with Model Gates (CPU)
//! - AC6: Smoke test validating CI blocks mocked receipts

use anyhow::{Context, Result};
use serde_json::Value;
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

    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();

    // Verify model-gates.yml workflow exists
    let workflow_path = workspace_root.join(".github/workflows/model-gates.yml");

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

    // Configure deterministic environment (unsafe required in Rust 1.90+)
    unsafe {
        std::env::set_var("BITNET_DETERMINISTIC", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
        std::env::set_var("BITNET_SEED", "42");
    }

    let _workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();

    // Create temporary test directory
    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Create mocked receipt (should fail verification)
    let mocked_receipt = serde_json::json!({
        "version": "1.0.0",
        "compute_path": "mocked",
        "kernels": [],
        "performance": {
            "tokens_per_sec": 15.3
        },
        "success": true
    });

    let mocked_receipt_path = temp_dir.path().join("mocked-receipt.json");
    fs::write(&mocked_receipt_path, serde_json::to_string_pretty(&mocked_receipt)?)
        .context("Failed to write mocked receipt")?;

    // Verify mocked receipt fails schema validation
    let validation_result = verify_receipt_honest_compute(&mocked_receipt_path);

    assert!(validation_result.is_err(), "Mocked receipt passed verification (should fail)");

    // Create receipt with empty kernels (should fail)
    let empty_kernels_receipt = serde_json::json!({
        "version": "1.0.0",
        "compute_path": "real",
        "kernels": [],
        "performance": {
            "tokens_per_sec": 15.3
        },
        "success": true
    });

    let empty_kernels_path = temp_dir.path().join("empty-kernels-receipt.json");
    fs::write(&empty_kernels_path, serde_json::to_string_pretty(&empty_kernels_receipt)?)
        .context("Failed to write empty kernels receipt")?;

    // Verify empty kernels receipt fails validation
    let validation_result = verify_receipt_honest_compute(&empty_kernels_path);

    assert!(validation_result.is_err(), "Empty kernels receipt passed verification (should fail)");

    // Create receipt with invalid kernel IDs (should fail)
    let invalid_kernels_receipt = serde_json::json!({
        "version": "1.0.0",
        "compute_path": "real",
        "kernels": ["", "a".repeat(129)],
        "performance": {
            "tokens_per_sec": 15.3
        },
        "success": true
    });

    let invalid_kernels_path = temp_dir.path().join("invalid-kernels-receipt.json");
    fs::write(&invalid_kernels_path, serde_json::to_string_pretty(&invalid_kernels_receipt)?)
        .context("Failed to write invalid kernels receipt")?;

    // Verify invalid kernels receipt fails validation
    let validation_result = verify_receipt_honest_compute(&invalid_kernels_path);

    assert!(
        validation_result.is_err(),
        "Invalid kernels receipt passed verification (should fail)"
    );

    // Create valid receipt (should pass)
    let valid_receipt = serde_json::json!({
        "version": "1.0.0",
        "compute_path": "real",
        "kernels": ["i2s_cpu_quantized_matmul", "tl1_lut_dequant_forward"],
        "performance": {
            "tokens_per_sec": 15.3
        },
        "success": true
    });

    let valid_receipt_path = temp_dir.path().join("valid-receipt.json");
    fs::write(&valid_receipt_path, serde_json::to_string_pretty(&valid_receipt)?)
        .context("Failed to write valid receipt")?;

    // Verify valid receipt passes validation
    let validation_result = verify_receipt_honest_compute(&valid_receipt_path);

    assert!(validation_result.is_ok(), "Valid receipt failed verification (should pass)");

    // Evidence tag for validation
    println!("// AC6: Smoke test validated");
    println!("// Mocked receipt: REJECTED ✓");
    println!("// Empty kernels: REJECTED ✓");
    println!("// Invalid kernels: REJECTED ✓");
    println!("// Valid receipt: ACCEPTED ✓");

    Ok(())
}

/// Helper function to verify receipt honest compute requirements
fn verify_receipt_honest_compute(path: &Path) -> Result<()> {
    let content = fs::read_to_string(path).context("Failed to read receipt file")?;

    let receipt: Value = serde_json::from_str(&content).context("Failed to parse receipt JSON")?;

    // Validate compute_path is "real"
    let compute_path =
        receipt["compute_path"].as_str().context("compute_path field missing or not a string")?;

    if compute_path != "real" {
        anyhow::bail!("Receipt has invalid compute_path: {} (expected 'real')", compute_path);
    }

    // Validate non-empty kernels array
    let kernels = receipt["kernels"].as_array().context("kernels field missing or not an array")?;

    if kernels.is_empty() {
        anyhow::bail!("Receipt has empty kernels array (honest compute requires kernel IDs)");
    }

    // Validate kernel ID hygiene
    for kernel in kernels {
        let kernel_id = kernel.as_str().context("kernel ID is not a string")?;

        if kernel_id.is_empty() {
            anyhow::bail!("Receipt contains empty kernel ID");
        }

        if kernel_id.len() > 128 {
            anyhow::bail!("Kernel ID exceeds 128 characters: {}", kernel_id);
        }
    }

    if kernels.len() > 10_000 {
        anyhow::bail!("Kernel count exceeds 10,000: {}", kernels.len());
    }

    Ok(())
}

#[cfg(test)]
mod test_helpers {
    use super::*;

    /// Test helper to create test receipt with specified compute path
    #[allow(dead_code)]
    pub fn create_test_receipt(compute_path: &str, kernels: Vec<String>) -> Value {
        serde_json::json!({
            "version": "1.0.0",
            "compute_path": compute_path,
            "kernels": kernels,
            "performance": {
                "tokens_per_sec": 15.3
            },
            "success": true
        })
    }

    /// Test helper to verify workflow configuration
    #[allow(dead_code)]
    pub fn verify_workflow_config(workflow_path: &Path) -> Result<()> {
        let content = fs::read_to_string(workflow_path)?;

        // Check for required sections
        let required_sections = vec!["jobs:", "verify-receipt", "BITNET_DETERMINISTIC"];

        for section in &required_sections {
            if !content.contains(section) {
                anyhow::bail!("Workflow missing required section: {}", section);
            }
        }

        Ok(())
    }
}
