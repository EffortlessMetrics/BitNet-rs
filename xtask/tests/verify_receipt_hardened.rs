//! Hardened receipt verification tests for Issue #462
//!
//! This test file provides comprehensive test coverage for receipt validation
//! to achieve ≥80% mutation testing score by testing all branches in the
//! verification logic.
//!
//! Test Coverage:
//! - CPU backend positive case (with quantized kernels)
//! - CPU backend negative cases:
//!   - Empty kernels
//!   - No quantized kernels (only non-quantized operations)
//!   - Fallback-only kernels (FP32 dequant paths)
//!   - Contains trap (kernel contains "i2s_" but doesn't start with it)
//! - GPU backend positive case (with GPU kernels)
//! - GPU backend negative cases:
//!   - CPU-only kernels (silent fallback detection)
//!   - Contains trap (GPU backend with CPU kernel patterns)
//! - Compute path validation:
//!   - Mock compute path rejection
//!
//! Mutation Testing Target: Kill 13+/16 mutants (≥80%)

use assert_cmd::Command;
use predicates::prelude::*;
use std::{env, fs, path::PathBuf};

/// Helper to get workspace root
fn workspace_root() -> PathBuf {
    if let Ok(dir) = env::var("CARGO_WORKSPACE_DIR") {
        return PathBuf::from(dir);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if path.join(".git").exists() {
            return path;
        }
        let cargo_toml = path.join("Cargo.toml");
        if cargo_toml.exists()
            && let Ok(contents) = fs::read_to_string(&cargo_toml)
            && contents.contains("[workspace]")
        {
            return path;
        }
        if !path.pop() {
            panic!("Could not find workspace root");
        }
    }
}

/// Helper to get fixture path
fn fixture_path(name: &str) -> PathBuf {
    workspace_root().join("xtask/tests/fixtures/receipts").join(format!("{}.json", name))
}

// ============================================================================
// CPU Backend Tests
// ============================================================================

/// Test 1: CPU backend with valid quantized kernels passes
///
/// This test validates the positive case where CPU backend receipts
/// contain proper quantized kernels (i2s_*, tl1_*, tl2_*).
///
/// Kills mutants:
/// - Mutations that disable CPU quantized kernel validation
/// - Mutations that change the validation logic to always fail
#[test]
fn test_cpu_with_quantized_kernels_passes() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", fixture_path("valid_receipt").to_str().unwrap()]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Receipt verification passed"))
        .stdout(predicate::str::contains("Schema: 1.0.0"))
        .stdout(predicate::str::contains("Compute path: real"));
}

/// Test 2: CPU backend with empty kernels fails
///
/// This test validates that CPU backend receipts must have non-empty kernel arrays.
///
/// Kills mutants:
/// - Mutations that disable empty kernel validation
/// - Mutations that change the error message
#[test]
fn test_cpu_with_empty_kernels_fails() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", fixture_path("missing_kernels").to_str().unwrap()]);

    cmd.assert().failure().stderr(predicate::str::contains("empty kernels[]"));
}

/// Test 3: CPU backend with no quantized kernels fails
///
/// This test validates that CPU backend receipts without any quantized kernels
/// (i2s_*, tl1_*, tl2_*) fail validation. This catches silent fallback to
/// non-quantized operations.
///
/// Fixture: cpu_no_quant_kernels.json
/// Kernels: ["rope_apply", "softmax_cpu", "attention_real"]
///
/// Kills mutants:
/// - Mutations in is_cpu_quantized_kernel() that disable validation
/// - Mutations that change cpu_quant_count == 0 to != 0
/// - Mutations that remove the CPU backend validation branch
#[test]
fn test_cpu_no_quant_kernels_fails() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", fixture_path("cpu_no_quant_kernels").to_str().unwrap()]);

    cmd.assert().failure().stderr(predicate::str::contains("no quantized kernels found"));
}

/// Test 4: CPU backend with fallback-only kernels fails
///
/// This test validates that CPU backend receipts containing only FP32 fallback
/// kernels (patterns: dequant, fp32_, fallback_) fail validation.
///
/// Fixture: cpu_fallback_only.json
/// Kernels: ["dequant_fp32_path", "fp32_matmul", "fallback_gemm"]
///
/// Kills mutants:
/// - Mutations in is_fallback_kernel_id() that disable detection
/// - Mutations in fallback_count calculation
/// - Mutations that change the fallback error message branch
#[test]
fn test_cpu_fallback_only_fails() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", fixture_path("cpu_fallback_only").to_str().unwrap()]);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("no quantized kernels found"))
        .stderr(predicate::str::contains("fallback patterns detected"));
}

/// Test 5: CPU backend with "contains" trap fails (prefix matching test)
///
/// This test validates that is_cpu_quantized_kernel() uses starts_with()
/// matching, not contains() matching. A kernel ID like "dequantize_i2s_helper"
/// contains "i2s_" but doesn't START with it, so it should NOT be classified
/// as a CPU quantized kernel.
///
/// Fixture: cpu_contains_trap.json
/// Kernels: ["dequantize_i2s_helper", "rope_apply"]
///
/// Kills mutants:
/// - Mutations that change starts_with() to contains() in is_cpu_quantized_kernel()
/// - Mutations that disable prefix validation
/// - Mutations in the CPU quantized kernel classification logic
#[test]
fn test_cpu_contains_trap_fails() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", fixture_path("cpu_contains_trap").to_str().unwrap()]);

    cmd.assert().failure().stderr(predicate::str::contains("no quantized kernels found"));
}

// ============================================================================
// GPU Backend Tests
// ============================================================================

/// Test 6: GPU backend with valid GPU kernels passes
///
/// This test validates the positive case where GPU backend receipts
/// contain proper GPU kernels (gemm_*, wmma_*, cuda_*, etc.).
///
/// Kills mutants:
/// - Mutations that disable GPU kernel validation
/// - Mutations that change the validation logic to always fail
#[test]
fn test_gpu_with_gpu_kernels_passes() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args([
        "verify-receipt",
        "--path",
        fixture_path("valid_gpu_receipt").to_str().unwrap(),
        "--require-gpu-kernels",
    ]);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Receipt verification passed"))
        .stdout(predicate::str::contains("Backend: cuda"));
}

/// Test 7: GPU backend with CPU-only kernels fails
///
/// This test validates that GPU backend receipts without any GPU kernels
/// fail validation, catching silent CPU fallback scenarios.
///
/// Fixture: gpu_cpu_kernels_only.json
/// Backend: "cuda"
/// Kernels: ["i2s_gemv", "tl1_matmul", "rope_apply"] (all CPU kernels)
///
/// Kills mutants:
/// - Mutations in is_gpu_kernel_id() that disable validation
/// - Mutations that change has_gpu_kernel check
/// - Mutations in GPU kernel prefix patterns
#[test]
fn test_gpu_cpu_kernels_only_fails() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", fixture_path("gpu_cpu_kernels_only").to_str().unwrap()]);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("no GPU kernels found"))
        .stderr(predicate::str::contains("silent CPU fallback"));
}

/// Test 8: GPU backend auto-enforcement (backend="cuda" requires GPU kernels)
///
/// This test validates that when backend="cuda", GPU kernels are automatically
/// required even without --require-gpu-kernels flag.
///
/// Fixture: invalid_gpu_receipt.json (already tests this scenario)
///
/// Kills mutants:
/// - Mutations in must_require_gpu logic
/// - Mutations that disable auto-enforcement for CUDA backend
/// - Mutations in the backend.eq_ignore_ascii_case("cuda") check
#[test]
fn test_gpu_backend_auto_enforcement() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    // Note: NO --require-gpu-kernels flag, but backend="cuda" should enforce it
    cmd.args(["verify-receipt", "--path", fixture_path("invalid_gpu_receipt").to_str().unwrap()]);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("no GPU kernels found"))
        .stderr(predicate::str::contains("backend is 'cuda'"));
}

// ============================================================================
// Compute Path Tests
// ============================================================================

/// Test 9: Mock compute path fails validation
///
/// This test validates that receipts with compute_path="mock" are rejected,
/// ensuring only real inference evidence is accepted.
///
/// Fixture: invalid_compute_path.json
/// Compute path: "mock"
///
/// Kills mutants:
/// - Mutations in compute_path validation
/// - Mutations that change compute_path != "real" to == "real"
/// - Mutations that remove the compute_path check
#[test]
fn test_mock_compute_path_fails() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", fixture_path("invalid_compute_path").to_str().unwrap()]);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("compute_path must be 'real'"))
        .stderr(predicate::str::contains("mock"));
}

// ============================================================================
// Edge Cases and Boundary Tests
// ============================================================================

/// Test 10: Receipt with missing file fails gracefully
///
/// This test validates that missing receipt files produce appropriate errors.
///
/// Kills mutants:
/// - Mutations that change error handling for missing files
#[test]
fn test_missing_receipt_file_fails() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", "nonexistent/receipt.json"]);

    cmd.assert().failure().stderr(predicate::str::contains("Failed to read receipt"));
}

/// Test 11: Default path behavior (ci/inference.json)
///
/// This test validates that when no path is provided, the command defaults
/// to checking ci/inference.json.
///
/// Kills mutants:
/// - Mutations in default path handling
#[test]
fn test_default_path_behavior() {
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.arg("verify-receipt");

    // Should fail either because file doesn't exist OR because it contains invalid content
    cmd.assert().failure().stderr(
        predicate::str::contains("ci/inference.json")
            .or(predicate::str::contains("no quantized kernels found")),
    );
}

// ============================================================================
// Kernel Classification Unit Tests
// ============================================================================

/// Test 12: CPU quantized kernel prefix validation
///
/// This test validates that the kernel classification logic correctly identifies
/// CPU quantized kernels using starts_with() prefix matching.
///
/// Tests the public API through the verify-receipt command to ensure
/// is_cpu_quantized_kernel() internal logic works correctly.
///
/// Kills mutants:
/// - Mutations in CPU_QUANT_PREFIXES constant
/// - Mutations that change starts_with() to contains()
/// - Mutations in the prefix matching loop
#[test]
fn test_cpu_quantized_prefix_validation() {
    // Create a temporary receipt with valid CPU quantized kernels
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "compute_path": "real",
        "backend": "cpu",
        "kernels": ["i2s_gemv", "i2s_matmul", "tl1_matmul", "tl2_lookup"],
        "timestamp": "2025-10-15T12:00:00Z",
        "deterministic": true,
        "environment": {}
    });

    let temp_dir = env::temp_dir();
    let receipt_path = temp_dir.join("test_cpu_quant_prefix.json");
    fs::write(&receipt_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();

    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", receipt_path.to_str().unwrap()]);

    cmd.assert().success();

    // Cleanup
    let _ = fs::remove_file(&receipt_path);
}

/// Test 13: GPU kernel prefix validation
///
/// This test validates that the kernel classification logic correctly identifies
/// GPU kernels using the documented prefix patterns.
///
/// Kills mutants:
/// - Mutations in GPU_KERNEL_PATTERNS regex patterns
/// - Mutations in is_gpu_kernel_id() matching logic
/// - Mutations that disable GPU kernel detection
#[test]
fn test_gpu_kernel_prefix_validation() {
    // Create a temporary receipt with various GPU kernel patterns
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "compute_path": "real",
        "backend": "cuda",
        "kernels": ["gemm_fp16", "wmma_m16n16k16", "cuda_sync", "tl1_gpu_pack"],
        "timestamp": "2025-10-15T12:00:00Z",
        "deterministic": true,
        "environment": {}
    });

    let temp_dir = env::temp_dir();
    let receipt_path = temp_dir.join("test_gpu_kernel_prefix.json");
    fs::write(&receipt_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();

    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", receipt_path.to_str().unwrap()]);

    cmd.assert().success();

    // Cleanup
    let _ = fs::remove_file(&receipt_path);
}

/// Test 14: Fallback kernel detection
///
/// This test validates that fallback patterns (dequant, fp32_, fallback_)
/// are correctly detected and reported.
///
/// Kills mutants:
/// - Mutations in FALLBACK_PATTERNS constant
/// - Mutations in is_fallback_kernel_id() matching logic
/// - Mutations in fallback_count calculation
#[test]
fn test_fallback_kernel_detection() {
    // Already tested via test_cpu_fallback_only_fails(), but this provides
    // additional coverage for the error message formatting with fallback details
    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", fixture_path("cpu_fallback_only").to_str().unwrap()]);

    cmd.assert()
        .failure()
        // Verify that the error message mentions the fallback patterns
        .stderr(predicate::str::contains("fallback"))
        .stderr(predicate::str::contains("dequant").or(predicate::str::contains("fp32")));
}

// ============================================================================
// Schema and Structure Tests
// ============================================================================

/// Test 15: Schema version validation
///
/// This test validates that receipts with missing or invalid schema versions
/// are rejected.
///
/// Kills mutants:
/// - Mutations in schema version validation
/// - Mutations that change supported schema versions
#[test]
fn test_schema_version_validation() {
    // Create a temporary receipt with missing schema_version
    let receipt = serde_json::json!({
        "compute_path": "real",
        "backend": "cpu",
        "kernels": ["i2s_gemv"],
        "timestamp": "2025-10-15T12:00:00Z"
    });

    let temp_dir = env::temp_dir();
    let receipt_path = temp_dir.join("test_missing_schema.json");
    fs::write(&receipt_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();

    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", receipt_path.to_str().unwrap()]);

    cmd.assert().failure().stderr(predicate::str::contains("schema_version"));

    // Cleanup
    let _ = fs::remove_file(&receipt_path);
}

/// Test 16: Kernel hygiene - empty kernel IDs
///
/// This test validates that receipts cannot contain empty string kernel IDs.
///
/// Kills mutants:
/// - Mutations in kernel hygiene validation
/// - Mutations that disable empty kernel ID checks
#[test]
fn test_empty_kernel_ids_rejected() {
    // Create a temporary receipt with empty kernel ID
    let receipt = serde_json::json!({
        "schema_version": "1.0.0",
        "compute_path": "real",
        "backend": "cpu",
        "kernels": ["i2s_gemv", "", "tl1_matmul"],
        "timestamp": "2025-10-15T12:00:00Z",
        "deterministic": true,
        "environment": {}
    });

    let temp_dir = env::temp_dir();
    let receipt_path = temp_dir.join("test_empty_kernel_id.json");
    fs::write(&receipt_path, serde_json::to_string_pretty(&receipt).unwrap()).unwrap();

    let mut cmd = Command::cargo_bin("xtask").unwrap();
    cmd.args(["verify-receipt", "--path", receipt_path.to_str().unwrap()]);

    cmd.assert().failure().stderr(predicate::str::contains("empty kernel ID"));

    // Cleanup
    let _ = fs::remove_file(&receipt_path);
}
