//! Test scaffolding for Issue #465: CPU Path Followup - Baseline Tests
//!
//! Work Stream 2: Baseline Establishment (AC3, AC4)
//!
//! Tests feature spec: docs/explanation/issue-465-implementation-spec.md
//!
//! This test suite validates:
//! - AC3: CPU baseline generation with deterministic receipt
//! - AC4: Baseline verification against quality gates

use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Receipt schema structure for validation
#[derive(Debug, serde::Deserialize)]
struct Receipt {
    version: String,
    compute_path: String,
    kernels: Vec<String>,
    performance: Performance,
    success: bool,
}

#[derive(Debug, serde::Deserialize)]
struct Performance {
    tokens_per_sec: f64,
}

/// Tests feature spec: issue-465-implementation-spec.md#ac3-generate-pinned-cpu-baseline
///
/// Validates that CPU baseline receipt exists with:
/// - Deterministic generation (BITNET_DETERMINISTIC=1, BITNET_SEED=42)
/// - Real compute path (compute_path: "real")
/// - Non-empty kernel array with CPU kernel IDs
/// - Measured performance metrics
#[test]
fn test_ac3_cpu_baseline_generated() -> Result<()> {
    // AC3: CPU baseline generation validation

    // Configure deterministic environment (unsafe required in Rust 1.90+)
    unsafe {
        std::env::set_var("BITNET_DETERMINISTIC", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
        std::env::set_var("BITNET_SEED", "42");
    }

    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let baselines_dir = workspace_root.join("docs/baselines");

    assert!(baselines_dir.exists(), "Baselines directory not found: {:?}", baselines_dir);

    // Look for CPU baseline with date stamp pattern (YYYYMMDD-cpu.json)
    let mut cpu_baseline_found = false;
    let mut cpu_baseline_path = None;

    if let Ok(entries) = fs::read_dir(&baselines_dir) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();

            if name.ends_with("-cpu.json") && name.len() >= 13 {
                cpu_baseline_found = true;
                cpu_baseline_path = Some(entry.path());
                break;
            }
        }
    }

    if !cpu_baseline_found {
        // FIXME: This test fails because implementation is missing
        // Expected: CPU baseline at docs/baselines/YYYYMMDD-cpu.json
        // Actual: No CPU baseline found
        panic!("AC3 implementation missing: CPU baseline not found in docs/baselines/");
    }

    let baseline_path = cpu_baseline_path.unwrap();
    println!("Found CPU baseline: {:?}", baseline_path);

    // Validate receipt schema
    let receipt_content =
        fs::read_to_string(&baseline_path).context("Failed to read CPU baseline receipt")?;

    let receipt: Receipt =
        serde_json::from_str(&receipt_content).context("Failed to parse CPU baseline receipt")?;

    // Validate schema version
    assert!(
        receipt.version == "1.0.0" || receipt.version == "1.0",
        "Invalid receipt version: {}",
        receipt.version
    );

    // Validate compute path (must be "real" for honest compute)
    assert_eq!(
        receipt.compute_path, "real",
        "CPU baseline has invalid compute_path: {}",
        receipt.compute_path
    );

    // Validate non-empty kernels array
    assert!(!receipt.kernels.is_empty(), "CPU baseline has empty kernels array");

    // Validate CPU kernel IDs (should include i2s_, tl1_, tl2_ prefixes)
    let cpu_kernel_prefixes = vec!["i2s_", "tl1_", "tl2_", "cpu_", "quantized_matmul"];
    let mut has_cpu_kernels = false;

    for kernel_id in &receipt.kernels {
        for prefix in &cpu_kernel_prefixes {
            if kernel_id.contains(prefix) {
                has_cpu_kernels = true;
                break;
            }
        }
        if has_cpu_kernels {
            break;
        }
    }

    assert!(
        has_cpu_kernels,
        "CPU baseline missing CPU kernel IDs (expected i2s_*, tl1_*, tl2_* prefixes)"
    );

    // Validate kernel ID hygiene
    for kernel_id in &receipt.kernels {
        assert!(!kernel_id.is_empty(), "CPU baseline contains empty kernel ID");

        assert!(
            kernel_id.len() <= 128,
            "CPU baseline kernel ID exceeds 128 characters: {}",
            kernel_id
        );
    }

    assert!(
        receipt.kernels.len() <= 10_000,
        "CPU baseline kernel count exceeds 10,000: {}",
        receipt.kernels.len()
    );

    // Validate performance metrics
    assert!(
        receipt.performance.tokens_per_sec > 0.0,
        "CPU baseline has invalid tokens_per_sec: {}",
        receipt.performance.tokens_per_sec
    );

    // Validate success flag
    assert!(receipt.success, "CPU baseline has success=false");

    // Neural Network Context: Verify realistic CPU performance (10-20 tok/s for I2_S)
    assert!(
        receipt.performance.tokens_per_sec >= 5.0 && receipt.performance.tokens_per_sec <= 50.0,
        "CPU baseline performance outside realistic range (5-50 tok/s): {} tok/s",
        receipt.performance.tokens_per_sec
    );

    // Evidence tag for validation
    println!("// AC3: CPU baseline generated and validated");
    println!(
        "// Receipt: compute_path={}, kernels={}, tps={:.2}",
        receipt.compute_path,
        receipt.kernels.len(),
        receipt.performance.tokens_per_sec
    );

    Ok(())
}

/// Tests feature spec: issue-465-implementation-spec.md#ac4-baseline-verification
///
/// Validates that baseline verification passes:
/// - cargo run -p xtask -- verify-receipt succeeds
/// - Schema v1.0.0 compliance
/// - Kernel hygiene checks pass
/// - Honest compute validation passes
#[test]
fn test_ac4_baseline_verification_passes() -> Result<()> {
    // AC4: Baseline verification validation

    // Configure deterministic environment (unsafe required in Rust 1.90+)
    unsafe {
        std::env::set_var("BITNET_DETERMINISTIC", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
        std::env::set_var("BITNET_SEED", "42");
    }

    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let baselines_dir = workspace_root.join("docs/baselines");

    // Find CPU baseline
    let mut cpu_baseline_path = None;

    if let Ok(entries) = fs::read_dir(&baselines_dir) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();

            if name.ends_with("-cpu.json") && name.len() >= 13 {
                cpu_baseline_path = Some(entry.path());
                break;
            }
        }
    }

    if cpu_baseline_path.is_none() {
        // FIXME: This test fails because implementation is missing
        // Expected: CPU baseline exists for verification
        // Actual: No CPU baseline found
        panic!("AC4 implementation missing: CPU baseline not found for verification");
    }

    let baseline_path = cpu_baseline_path.unwrap();

    // Verify receipt schema (detailed validation)
    verify_receipt_schema(&baseline_path).context("CPU baseline failed schema validation")?;

    // FIXME: This test requires xtask verify-receipt command to be run
    // Expected: cargo run -p xtask -- verify-receipt --path <baseline> succeeds
    // Actual: xtask command integration needed
    //
    // Uncomment when xtask integration is ready:
    // let output = std::process::Command::new("cargo")
    //     .args(&["run", "-p", "xtask", "--", "verify-receipt", "--path"])
    //     .arg(&baseline_path)
    //     .output()
    //     .context("Failed to run xtask verify-receipt")?;
    //
    // assert!(
    //     output.status.success(),
    //     "xtask verify-receipt failed for CPU baseline"
    // );

    // Evidence tag for validation
    println!("// AC4: Baseline verification passed");

    // FIXME: This test fails because implementation is incomplete
    // Expected: Full xtask verify-receipt integration
    // Actual: Schema validation only (xtask integration pending)
    panic!("AC4 implementation incomplete: xtask verify-receipt integration needed");
}

/// Helper function to verify receipt schema
fn verify_receipt_schema(path: &Path) -> Result<()> {
    let content = fs::read_to_string(path).context("Failed to read receipt file")?;

    let receipt: Value = serde_json::from_str(&content).context("Failed to parse receipt JSON")?;

    // Validate required fields
    let required_fields = vec!["version", "compute_path", "kernels", "performance", "success"];

    for field in &required_fields {
        assert!(receipt.get(field).is_some(), "Receipt missing required field: {}", field);
    }

    // Validate version format
    let version = receipt["version"].as_str().context("version field is not a string")?;

    assert!(version == "1.0.0" || version == "1.0", "Invalid receipt version: {}", version);

    // Validate compute_path
    let compute_path =
        receipt["compute_path"].as_str().context("compute_path field is not a string")?;

    assert_eq!(compute_path, "real", "Invalid compute_path: {}", compute_path);

    // Validate kernels array
    let kernels = receipt["kernels"].as_array().context("kernels field is not an array")?;

    assert!(!kernels.is_empty(), "Receipt has empty kernels array");

    // Validate kernel hygiene
    for kernel in kernels {
        let kernel_id = kernel.as_str().context("kernel ID is not a string")?;

        assert!(!kernel_id.is_empty(), "Receipt contains empty kernel ID");

        assert!(kernel_id.len() <= 128, "Kernel ID exceeds 128 characters: {}", kernel_id);
    }

    assert!(kernels.len() <= 10_000, "Kernel count exceeds 10,000: {}", kernels.len());

    // Validate performance metrics
    let performance =
        receipt["performance"].as_object().context("performance field is not an object")?;

    let tokens_per_sec =
        performance["tokens_per_sec"].as_f64().context("tokens_per_sec is not a number")?;

    assert!(tokens_per_sec > 0.0, "Invalid tokens_per_sec: {}", tokens_per_sec);

    // Validate success flag
    let success = receipt["success"].as_bool().context("success field is not a boolean")?;

    assert!(success, "Receipt has success=false");

    Ok(())
}

/// Helper function to verify kernel IDs are CPU-specific
fn verify_kernel_ids(receipt: &Receipt) -> bool {
    let cpu_kernel_prefixes = vec!["i2s_", "tl1_", "tl2_", "cpu_", "quantized_matmul"];

    for kernel_id in &receipt.kernels {
        for prefix in &cpu_kernel_prefixes {
            if kernel_id.contains(prefix) {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod test_helpers {
    use super::*;

    /// Test helper to create mock receipt for testing
    pub fn create_mock_receipt(compute_path: &str, kernel_count: usize) -> Receipt {
        Receipt {
            version: "1.0.0".to_string(),
            compute_path: compute_path.to_string(),
            kernels: (0..kernel_count).map(|i| format!("i2s_cpu_kernel_{}", i)).collect(),
            performance: Performance { tokens_per_sec: 15.3 },
            success: true,
        }
    }

    /// Test helper to validate receipt structure
    pub fn validate_receipt_structure(receipt: &Receipt) -> Result<()> {
        assert!(receipt.version == "1.0.0" || receipt.version == "1.0", "Invalid version");

        assert_eq!(receipt.compute_path, "real", "Invalid compute_path");

        assert!(!receipt.kernels.is_empty(), "Empty kernels array");

        assert!(receipt.performance.tokens_per_sec > 0.0, "Invalid tokens_per_sec");

        assert!(receipt.success, "Success flag is false");

        Ok(())
    }
}
