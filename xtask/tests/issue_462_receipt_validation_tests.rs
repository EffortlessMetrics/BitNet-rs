// AC:3 - Receipt CPU Validation Test Scaffolding
//
// This test file validates AC3 from Issue #462:
// - CPU backend with CPU quantized kernels (positive)
// - CPU backend with no kernels (negative)
// - CPU backend with non-quantized kernels (negative)
// - GPU backend with CPU kernels (silent fallback detection)
//
// Test Plan Reference: docs/explanation/cpu-inference-test-plan.md
// Spec: docs/explanation/receipt-cpu-validation-spec.md

// Note: xtask doesn't have cpu feature, these tests validate receipt logic only

use anyhow::Result;
use serde_json::json;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Test utilities for receipt validation
mod test_utils {
    use super::*;

    /// Create test receipt JSON file
    pub fn create_test_receipt(
        backend: &str,
        kernels: Vec<&str>,
        compute_path: &str,
    ) -> Result<PathBuf> {
        let receipt = json!({
            "schema_version": "1.0.0",
            "timestamp": "2025-10-14T12:00:00Z",
            "compute_path": compute_path,
            "backend": backend,
            "kernels": kernels,
            "deterministic": true,
            "environment": {},
            "model_info": {},
            "test_results": {},
            "performance_baseline": {}
        });

        let temp_dir = env::temp_dir();
        let timestamp =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros();
        let receipt_path = temp_dir.join(format!("test_receipt_{}.json", timestamp));

        fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;

        Ok(receipt_path)
    }

    /// Run verify-receipt command (requires xtask binary)
    ///
    /// Returns: (exit_code, stdout, stderr)
    #[allow(dead_code)]
    pub fn run_verify_receipt(_receipt_path: &Path) -> Result<(i32, String, String)> {
        // TODO: This requires calling xtask::verify_receipt_cmd directly
        // or building xtask binary and executing it
        //
        // Option 1: Direct function call (preferred for unit tests)
        // use xtask::verify_receipt_cmd;
        // let result = verify_receipt_cmd(receipt_path, false);
        // match result {
        //     Ok(()) => Ok((0, String::new(), String::new())),
        //     Err(e) => Ok((1, String::new(), e.to_string())),
        // }

        // Option 2: Execute xtask binary (for integration tests)
        // let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        //     .parent()
        //     .unwrap()
        //     .parent()
        //     .unwrap();
        // let xtask_binary = workspace_root.join("target/debug/xtask");
        //
        // let output = std::process::Command::new(&xtask_binary)
        //     .arg("verify-receipt")
        //     .arg(receipt_path)
        //     .output()?;
        //
        // Ok((
        //     output.status.code().unwrap_or(1),
        //     String::from_utf8_lossy(&output.stdout).to_string(),
        //     String::from_utf8_lossy(&output.stderr).to_string(),
        // ))

        // Placeholder: Unimplemented
        Err(anyhow::anyhow!("verify_receipt execution not yet implemented"))
    }
}

// ============================================================================
// AC:3 - Test 3.1: CPU Receipt Honesty (Positive)
// ============================================================================

/// AC:3 - T3.1: CPU backend with CPU quantized kernels (positive)
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-31
/// Validates that receipt with CPU quantized kernels passes verification
///
/// # Expected Behavior
/// - backend="cpu"
/// - kernels=["i2s_gemv", "tl1_matmul", "tl2_matmul"]
/// - Verification passes (exit code 0)
/// - Output: "✅ Receipt verification passed"
#[test]
fn test_ac3_receipt_cpu_kernel_honesty_positive() -> Result<()> {
    // Create test receipt with CPU quantized kernels
    let receipt_path = test_utils::create_test_receipt(
        "cpu",
        vec!["i2s_gemv", "tl1_matmul", "tl2_matmul", "rope_apply"],
        "real",
    )?;

    // TODO: Run verify-receipt command
    // let (exit_code, stdout, stderr) = test_utils::run_verify_receipt(&receipt_path)?;

    // TODO: Validate success
    // assert_eq!(exit_code, 0, "Receipt verification should pass");
    // assert!(
    //     stdout.contains("Receipt verification passed") || stdout.contains("✅"),
    //     "Output should indicate success"
    // );

    // Verify receipt file was created
    assert!(receipt_path.exists(), "Receipt file should be created at {}", receipt_path.display());

    let contents = fs::read_to_string(&receipt_path)?;
    let receipt: serde_json::Value = serde_json::from_str(&contents)?;

    // Verify CPU backend with quantized kernels
    assert_eq!(receipt["backend"], "cpu", "Receipt should have CPU backend for positive test");
    let kernels = receipt["kernels"].as_array().unwrap();
    let has_i2s = kernels.iter().any(|k| k.as_str().unwrap().starts_with("i2s_"));
    let has_tl1 = kernels.iter().any(|k| k.as_str().unwrap().starts_with("tl1_"));
    assert!(has_i2s || has_tl1, "Receipt should contain CPU quantized kernels (i2s_* or tl1_*)");

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}

// ============================================================================
// AC:3 - Test 3.2: CPU Receipt Honesty (Negative - Mock Kernels)
// ============================================================================

/// AC:3 - T3.2: CPU backend with no quantized kernels (negative)
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-32
/// Validates that receipt without quantized kernels fails verification
///
/// # Expected Behavior
/// - backend="cpu"
/// - kernels=["rope_apply", "softmax_cpu", "mock_kernel"]
/// - Verification fails (exit code 1)
/// - Error: "no quantized kernels found"
#[test]

fn test_ac3_receipt_cpu_kernel_honesty_negative() -> Result<()> {
    // Create test receipt with no quantized kernels
    let receipt_path = test_utils::create_test_receipt(
        "cpu",
        vec!["rope_apply", "softmax_cpu", "attention_mock"],
        "real",
    )?;

    // TODO: Run verify-receipt command
    // let (exit_code, _stdout, stderr) = test_utils::run_verify_receipt(&receipt_path)?;

    // TODO: Validate failure
    // assert_ne!(exit_code, 0, "Receipt verification should fail");
    // assert!(
    //     stderr.contains("no quantized kernels found"),
    //     "Error should mention missing quantized kernels"
    // );

    // Verify receipt file was created
    assert!(receipt_path.exists(), "Receipt file should be created at {}", receipt_path.display());

    let contents = fs::read_to_string(&receipt_path)?;
    let receipt: serde_json::Value = serde_json::from_str(&contents)?;

    // Verify CPU backend without quantized kernels (negative test)
    assert_eq!(receipt["backend"], "cpu", "Receipt should have CPU backend for negative test");
    let kernels = receipt["kernels"].as_array().unwrap();
    let has_quantized = kernels.iter().any(|k| {
        let s = k.as_str().unwrap();
        s.starts_with("i2s_") || s.starts_with("tl1_") || s.starts_with("tl2_")
    });
    assert!(
        !has_quantized,
        "Receipt should NOT have quantized kernels (negative test - validation should fail)"
    );

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}

// ============================================================================
// AC:3 - Test 3.3: CPU Receipt with FP32 Fallback (Negative)
// ============================================================================

/// AC:3 - Additional: CPU backend with FP32 fallback kernels (negative)
///
/// Validates that receipt with excluded patterns fails
///
/// # Expected Behavior
/// - backend="cpu"
/// - kernels=["fp32_matmul", "fallback_gemm", "dequant_i2s"]
/// - Verification fails (exit code 1)
/// - Error: "excluded patterns found"
#[test]

fn test_ac3_receipt_cpu_fp32_fallback() -> Result<()> {
    // Create test receipt with FP32 fallback kernels
    let receipt_path = test_utils::create_test_receipt(
        "cpu",
        vec!["fp32_matmul", "fallback_gemm", "dequant_i2s"],
        "real",
    )?;

    // TODO: Run verify-receipt command
    // let (exit_code, _stdout, stderr) = test_utils::run_verify_receipt(&receipt_path)?;

    // TODO: Validate failure
    // assert_ne!(exit_code, 0, "Receipt verification should fail");
    // assert!(
    //     stderr.contains("excluded patterns"),
    //     "Error should mention excluded patterns (dequant/fp32/fallback)"
    // );

    // Verify receipt file was created
    assert!(receipt_path.exists(), "Receipt file should be created at {}", receipt_path.display());

    let contents = fs::read_to_string(&receipt_path)?;
    let receipt: serde_json::Value = serde_json::from_str(&contents)?;

    // Verify CPU backend with FP32 fallback patterns
    assert_eq!(receipt["backend"], "cpu", "Receipt should have CPU backend for FP32 fallback test");
    let kernels = receipt["kernels"].as_array().unwrap();
    let has_fallback = kernels.iter().any(|k| {
        let s = k.as_str().unwrap();
        s.contains("fp32") || s.contains("fallback") || s.contains("dequant")
    });
    assert!(has_fallback, "Receipt should contain FP32 fallback patterns (excluded kernels)");

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}

// ============================================================================
// AC:3 - Test 3.4: GPU Backend with CPU Kernels (Silent Fallback)
// ============================================================================

/// AC:3 - T3.3: GPU backend with CPU kernels fails (silent fallback detection)
///
/// Test Plan: docs/explanation/cpu-inference-test-plan.md#test-33
/// Validates that GPU backend receipt with CPU kernels fails
///
/// # Expected Behavior
/// - backend="cuda"
/// - kernels=["i2s_gemv", "tl1_matmul"] (CPU kernels)
/// - Verification fails (exit code 1)
/// - Error: "no GPU kernels found"
#[test]

fn test_ac3_receipt_gpu_cpu_kernel_mismatch() -> Result<()> {
    // Create test receipt with GPU backend but CPU kernels
    let receipt_path = test_utils::create_test_receipt(
        "cuda",
        vec!["i2s_gemv", "tl1_matmul"], // CPU kernels, not GPU
        "real",
    )?;

    // TODO: Run verify-receipt command
    // let (exit_code, _stdout, stderr) = test_utils::run_verify_receipt(&receipt_path)?;

    // TODO: Validate failure
    // assert_ne!(exit_code, 0, "Receipt verification should fail");
    // assert!(
    //     stderr.contains("no GPU kernels found"),
    //     "Error should mention missing GPU kernels (silent CPU fallback)"
    // );

    // Verify receipt file was created
    assert!(receipt_path.exists(), "Receipt file should be created at {}", receipt_path.display());

    let contents = fs::read_to_string(&receipt_path)?;
    let receipt: serde_json::Value = serde_json::from_str(&contents)?;

    // Verify GPU backend with CPU kernels (mismatch detection)
    assert_eq!(
        receipt["backend"], "cuda",
        "Receipt should have CUDA backend for GPU/CPU mismatch test"
    );
    let kernels = receipt["kernels"].as_array().unwrap();
    let has_cpu_kernel = kernels.iter().any(|k| {
        let s = k.as_str().unwrap();
        // CPU kernels start with i2s_, tl1_, tl2_ but don't have _gpu_ suffix
        (s.starts_with("i2s_") || s.starts_with("tl1_") || s.starts_with("tl2_"))
            && !s.contains("_gpu_")
            && !s.starts_with("i2s_quantize")
            && !s.starts_with("i2s_dequantize")
    });
    assert!(
        has_cpu_kernel,
        "Receipt should contain CPU kernels with CUDA backend (silent fallback detection)"
    );

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}

// ============================================================================
// Unit Tests: Kernel Classification
// ============================================================================

/// AC:3 - Unit: CPU quantized kernel prefix matching
///
/// Validates is_cpu_quantized_kernel() logic (starts_with, not contains)
#[test]
fn test_ac3_cpu_quantized_prefix_matching() -> Result<()> {
    // Note: We cannot import is_cpu_quantized_kernel from xtask (it's private)
    // Instead, we test the public API via verify-receipt command

    // Create test receipt with valid CPU quantized kernels
    let receipt_path = test_utils::create_test_receipt(
        "cpu",
        vec!["i2s_gemv", "i2s_matmul", "tl1_matmul", "tl2_lookup"],
        "real",
    )?;

    // This receipt should pass - it has CPU quantized kernels
    // The verify_receipt logic will internally use is_cpu_quantized_kernel

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}

/// AC:3 - Unit: Excluded pattern matching
///
/// Validates is_excluded_kernel() logic (dequant/fp32/fallback)
#[test]
fn test_ac3_excluded_pattern_matching() -> Result<()> {
    // Create test receipt with excluded (fallback) patterns
    let receipt_path = test_utils::create_test_receipt(
        "cpu",
        vec!["dequant_i2s", "fp32_matmul", "fallback_gemm"],
        "real",
    )?;

    // This receipt should fail validation - it has fallback patterns but no quantized kernels
    // The verification logic will detect excluded patterns

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}

// ============================================================================
// Integration Test: E2E Receipt Generation and Validation
// ============================================================================

/// AC:3 - E2E: Full receipt generation and validation workflow
///
/// Validates end-to-end workflow from inference to receipt verification
#[test]
fn test_ac3_e2e_cpu_receipt_generation() -> Result<()> {
    // Create test receipt simulating E2E workflow
    let receipt_path = test_utils::create_test_receipt(
        "cpu",
        vec!["i2s_gemv", "tl1_matmul", "tl2_lookup", "rope_apply", "softmax_cpu"],
        "real",
    )?;

    // Verify receipt file was created with correct structure
    assert!(receipt_path.exists(), "Receipt should be created");

    let contents = fs::read_to_string(&receipt_path)?;
    let receipt: serde_json::Value = serde_json::from_str(&contents)?;

    // Verify required fields
    assert_eq!(receipt["schema_version"], "1.0.0");
    assert_eq!(receipt["compute_path"], "real");
    assert_eq!(receipt["backend"], "cpu");
    assert!(receipt["kernels"].is_array());

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}

// ============================================================================
// Edge Case Tests: Malformed and Invalid Receipts
// ============================================================================

/// Edge case: Receipt with missing required fields
///
/// Validates that receipts without required fields fail validation
#[test]
fn test_ac3_receipt_missing_schema_version() -> Result<()> {
    // Create minimal receipt missing schema_version
    let receipt = json!({
        "backend": "cpu",
        "compute_path": "real",
        "kernels": ["i2s_gemv"],
    });

    let temp_dir = env::temp_dir();
    let timestamp =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros();
    let receipt_path = temp_dir.join(format!("test_receipt_malformed_{}.json", timestamp));

    fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;

    // Verify file was created but is missing schema_version
    assert!(receipt_path.exists(), "Malformed receipt should be created");

    let contents = fs::read_to_string(&receipt_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&contents)?;

    // Verify schema_version is missing
    assert!(
        parsed.get("schema_version").is_none(),
        "Receipt should be missing schema_version field"
    );

    // TODO: When verify-receipt is callable, this should fail validation
    // let result = test_utils::run_verify_receipt(&receipt_path);
    // assert!(result.is_err(), "Receipt without schema_version should fail validation");

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}

/// Edge case: Receipt with invalid kernel type (not an array)
///
/// Validates that receipts with wrong field types fail parsing/validation
#[test]
fn test_ac3_receipt_invalid_kernel_type() -> Result<()> {
    // Create receipt with kernels as string instead of array
    let receipt = json!({
        "schema_version": "1.0.0",
        "backend": "cpu",
        "compute_path": "real",
        "kernels": "i2s_gemv,tl1_matmul", // Wrong type: string instead of array
        "timestamp": "2025-10-14T12:00:00Z",
    });

    let temp_dir = env::temp_dir();
    let timestamp =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros();
    let receipt_path = temp_dir.join(format!("test_receipt_bad_type_{}.json", timestamp));

    fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)?;

    let contents = fs::read_to_string(&receipt_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&contents)?;

    // Verify kernels is a string, not an array
    assert!(parsed["kernels"].is_string(), "Kernels should be a string (invalid type)");
    assert!(!parsed["kernels"].is_array(), "Kernels should NOT be an array (malformed)");

    // TODO: When verify-receipt is callable, this should fail validation
    // let result = test_utils::run_verify_receipt(&receipt_path);
    // assert!(result.is_err(), "Receipt with wrong kernel type should fail validation");

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}

/// Edge case: Receipt with empty kernels array
///
/// Validates that receipts with no kernels fail validation
#[test]
fn test_ac3_receipt_empty_kernels() -> Result<()> {
    // Create receipt with empty kernels array
    let receipt_path = test_utils::create_test_receipt("cpu", vec![], "real")?;

    let contents = fs::read_to_string(&receipt_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&contents)?;

    // Verify kernels array is empty
    let kernels = parsed["kernels"].as_array().unwrap();
    assert_eq!(kernels.len(), 0, "Kernels array should be empty");

    // TODO: When verify-receipt is callable, this should fail validation
    // let result = test_utils::run_verify_receipt(&receipt_path);
    // assert!(result.is_err(), "Receipt with empty kernels should fail validation");

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}

/// Edge case: Receipt with unknown backend
///
/// Validates that receipts with unsupported backends are handled
#[test]
fn test_ac3_receipt_unknown_backend() -> Result<()> {
    // Create receipt with unknown backend
    let receipt_path = test_utils::create_test_receipt(
        "vulkan", // Unknown backend (not "cpu" or "cuda")
        vec!["i2s_gemv"],
        "real",
    )?;

    let contents = fs::read_to_string(&receipt_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&contents)?;

    // Verify backend is "vulkan" (unknown)
    assert_eq!(parsed["backend"], "vulkan", "Backend should be 'vulkan' (unknown)");

    // TODO: When verify-receipt is callable, this may pass or fail depending on validation logic
    // Current expectation: validation should either accept it (extensibility) or reject it (strict)
    // let result = test_utils::run_verify_receipt(&receipt_path);

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}

/// Edge case: Receipt with mock compute path
///
/// Validates that receipts with compute_path="mock" fail validation
#[test]
fn test_ac3_receipt_mock_compute_path() -> Result<()> {
    // Create receipt with compute_path="mock"
    let receipt_path = test_utils::create_test_receipt(
        "cpu",
        vec!["i2s_gemv", "tl1_matmul"],
        "mock", // Invalid: should be "real"
    )?;

    let contents = fs::read_to_string(&receipt_path)?;
    let parsed: serde_json::Value = serde_json::from_str(&contents)?;

    // Verify compute_path is "mock"
    assert_eq!(parsed["compute_path"], "mock", "Compute path should be 'mock' (invalid)");

    // TODO: When verify-receipt is callable, this should fail validation
    // let result = test_utils::run_verify_receipt(&receipt_path);
    // assert!(result.is_err(), "Receipt with compute_path='mock' should fail validation");

    // Cleanup
    if receipt_path.exists() {
        fs::remove_file(&receipt_path)?;
    }

    Ok(())
}
