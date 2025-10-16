//! Test scaffolding for Issue #465: CPU Path Followup - Baseline Tests
//!
//! Work Stream 2: Baseline Establishment (AC3, AC4)
//!
//! Tests feature spec: docs/explanation/issue-465-implementation-spec.md
//!
//! This test suite validates:
//! - AC3: CPU baseline generation with deterministic receipt
//! - AC4: Baseline verification against quality gates

mod issue_465_test_utils;

use anyhow::{Context, Result};
use issue_465_test_utils::{
    configure_deterministic_env, create_test_receipt, find_cpu_baseline, has_cpu_kernel_ids,
    verify_receipt_schema,
};
use std::fs;

/// Performance bounds for CPU baseline validation
const MIN_VIABLE_TPS: f64 = 0.1;
/// Maximum realistic CPU TPS for 2B I2S model (receipts govern exact values)
const MAX_REALISTIC_CPU_TPS: f64 = 50.0;

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
    configure_deterministic_env();

    // Find CPU baseline using shared utility
    let baseline_path = find_cpu_baseline().context(
        "AC3 implementation missing: CPU baseline not found in docs/baselines/. \
        Run `cargo run -p xtask -- benchmark --model models/*.gguf --tokens 128` to generate.",
    )?;

    println!("Found CPU baseline: {:?}", baseline_path);

    // Validate receipt schema using shared utility
    verify_receipt_schema(&baseline_path).context("CPU baseline failed schema validation")?;

    // Parse receipt for detailed validation
    let receipt_content =
        fs::read_to_string(&baseline_path).context("Failed to read CPU baseline receipt")?;

    let receipt: issue_465_test_utils::Receipt =
        serde_json::from_str(&receipt_content).context("Failed to parse CPU baseline receipt")?;

    // Validate CPU-specific kernel IDs
    assert!(
        has_cpu_kernel_ids(&receipt.kernels),
        "CPU baseline missing CPU kernel IDs (expected i2s_*, tl1_*, tl2_*, cpu_*, quantized_matmul prefixes). \
        Found kernels: {:?}",
        receipt.kernels
    );

    // Neural Network Context: Verify realistic CPU performance (10-20 tok/s for I2_S quantization)
    // Allow MIN_VIABLE_TPS-MAX_REALISTIC_CPU_TPS range to accommodate short benchmarks and warm-up effects
    if receipt.tokens_per_sec > 0.0 {
        assert!(
            (MIN_VIABLE_TPS..=MAX_REALISTIC_CPU_TPS).contains(&receipt.tokens_per_sec),
            "CPU baseline performance outside realistic range (got {}, expected {}..={}). \
            This may indicate timing issues or incorrect kernel selection.",
            receipt.tokens_per_sec,
            MIN_VIABLE_TPS,
            MAX_REALISTIC_CPU_TPS
        );
    }

    // Evidence tag for validation
    println!("// AC3: CPU baseline generated and validated");
    println!(
        "// Receipt: compute_path={}, kernels={}, tps={:.2}",
        receipt.compute_path,
        receipt.kernels.len(),
        receipt.tokens_per_sec
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
    configure_deterministic_env();

    // Find CPU baseline using shared utility
    let baseline_path = find_cpu_baseline().context(
        "AC4 implementation missing: CPU baseline not found for verification. \
        Run `cargo run -p xtask -- benchmark --model models/*.gguf --tokens 128` to generate.",
    )?;

    // Verify receipt schema using shared utility
    verify_receipt_schema(&baseline_path).context("CPU baseline failed schema validation")?;

    // Evidence tag for validation
    println!("// AC4: Baseline verification passed");
    println!("// Schema validation: compute_path=real, kernels present, version valid");

    Ok(())
}

/// Edge case: Test that empty kernel arrays are properly rejected
///
/// This test validates kernel hygiene enforcement - receipts with empty kernel
/// arrays should fail validation as they indicate no real computation occurred.
#[test]
fn test_edge_case_empty_kernels_rejected() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;
    let receipt = create_test_receipt("real", vec![]);
    let receipt_path = temp_dir.path().join("empty-kernels.json");
    fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)
        .context("Failed to write test receipt")?;

    // Verify that empty kernels are rejected
    assert!(
        verify_receipt_schema(&receipt_path).is_err(),
        "Receipt with empty kernels should be rejected"
    );

    println!("// Edge case validated: Empty kernels properly rejected");
    Ok(())
}

/// Edge case: Test that invalid schema versions are rejected
///
/// This test validates strict schema version checking - only v1.0.0 and v1.0
/// should be accepted.
#[test]
fn test_edge_case_invalid_schema_versions() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Test various invalid versions
    let invalid_versions = vec!["2.0.0", "0.9.0", "1.1.0", "", "invalid"];

    for version in invalid_versions {
        let mut receipt = create_test_receipt("real", vec!["test_kernel".to_string()]);
        receipt["schema_version"] = serde_json::json!(version);

        let receipt_path = temp_dir.path().join(format!("version-{}.json", version));
        fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)
            .context("Failed to write test receipt")?;

        assert!(
            verify_receipt_schema(&receipt_path).is_err(),
            "Invalid schema version '{}' should be rejected",
            version
        );
    }

    println!("// Edge case validated: Invalid schema versions properly rejected");
    Ok(())
}

/// Edge case: Test that malformed JSON receipts are handled gracefully
///
/// This test validates robust error handling for corrupted receipt files.
#[test]
fn test_edge_case_malformed_json() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Test various malformed JSON scenarios
    let malformed_cases = vec![
        ("", "empty-file.json"),
        ("{", "incomplete-object.json"),
        ("{\"version\": }", "missing-value.json"),
        ("not json at all", "invalid-syntax.json"),
        (
            "{\"version\": \"1.0.0\", \"compute_path\": \"real\", \"kernels\": [}",
            "incomplete-array.json",
        ),
    ];

    for (content, filename) in malformed_cases {
        let receipt_path = temp_dir.path().join(filename);
        fs::write(&receipt_path, content).context("Failed to write malformed receipt")?;

        assert!(
            verify_receipt_schema(&receipt_path).is_err(),
            "Malformed JSON in {} should be rejected",
            filename
        );
    }

    println!("// Edge case validated: Malformed JSON properly rejected");
    Ok(())
}

/// Edge case: Test that missing required fields are detected
///
/// This test validates that all required receipt fields are enforced.
#[test]
fn test_edge_case_missing_required_fields() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Test missing each required field
    let test_cases = vec![
        (
            serde_json::json!({
                "compute_path": "real",
                "kernels": ["test_kernel"],
                "performance": {"tokens_per_sec": 10.0}
            }),
            "missing-version.json",
        ),
        (
            serde_json::json!({
                "version": "1.0.0",
                "kernels": ["test_kernel"],
                "performance": {"tokens_per_sec": 10.0}
            }),
            "missing-compute-path.json",
        ),
        (
            serde_json::json!({
                "version": "1.0.0",
                "compute_path": "real",
                "performance": {"tokens_per_sec": 10.0}
            }),
            "missing-kernels.json",
        ),
        (
            serde_json::json!({
                "version": "1.0.0",
                "compute_path": "real",
                "kernels": ["test_kernel"]
            }),
            "missing-performance.json",
        ),
    ];

    for (receipt, filename) in test_cases {
        let receipt_path = temp_dir.path().join(filename);
        fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)
            .context("Failed to write test receipt")?;

        assert!(
            verify_receipt_schema(&receipt_path).is_err(),
            "Receipt {} with missing required field should be rejected",
            filename
        );
    }

    println!("// Edge case validated: Missing required fields properly detected");
    Ok(())
}

/// Edge case: Test performance bounds validation
///
/// This test validates that performance metrics are within realistic bounds
/// for CPU inference with I2_S quantization.
#[test]
fn test_edge_case_performance_bounds() -> Result<()> {
    configure_deterministic_env();

    let baseline_path = find_cpu_baseline().context("CPU baseline not found")?;
    let receipt_content = fs::read_to_string(&baseline_path)?;
    let receipt: issue_465_test_utils::Receipt = serde_json::from_str(&receipt_content)?;

    // Validate performance is within realistic bounds (0.1-50 tok/s for CPU I2_S)
    // Allow zero for initialization benchmarks
    assert!(
        receipt.tokens_per_sec >= 0.0 && receipt.tokens_per_sec <= 50.0,
        "Performance {} tok/s outside realistic CPU bounds (0-50)",
        receipt.tokens_per_sec
    );

    // If non-zero, should be at least minimal viable performance
    if receipt.tokens_per_sec > 0.0 {
        assert!(
            receipt.tokens_per_sec >= 0.1,
            "Non-zero performance {} tok/s below minimal viable threshold (0.1)",
            receipt.tokens_per_sec
        );
    }

    println!(
        "// Edge case validated: Performance bounds checked ({:.2} tok/s)",
        receipt.tokens_per_sec
    );
    Ok(())
}

/// Edge case: Test kernel ID hygiene - length constraints
///
/// This test validates that kernel IDs respect the 128 character limit.
#[test]
fn test_edge_case_kernel_id_length_constraints() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Test kernel ID that's too long (>128 chars)
    let long_kernel_id = "k".repeat(129);
    let receipt = create_test_receipt("real", vec![long_kernel_id]);
    let receipt_path = temp_dir.path().join("long-kernel-id.json");
    fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)
        .context("Failed to write test receipt")?;

    assert!(
        verify_receipt_schema(&receipt_path).is_err(),
        "Kernel ID exceeding 128 characters should be rejected"
    );

    // Test kernel ID at exactly 128 chars (should pass)
    let max_length_kernel = "k".repeat(128);
    let valid_receipt = create_test_receipt("real", vec![max_length_kernel]);
    let valid_path = temp_dir.path().join("max-length-kernel.json");
    fs::write(&valid_path, serde_json::to_string_pretty(&valid_receipt)?)
        .context("Failed to write test receipt")?;

    verify_receipt_schema(&valid_path)
        .context("Valid kernel ID at 128 characters should be accepted")?;

    println!("// Edge case validated: Kernel ID length constraints enforced");
    Ok(())
}

/// Edge case: Test kernel count limits
///
/// This test validates that kernel counts respect the 10,000 limit.
#[test]
fn test_edge_case_kernel_count_limits() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Test kernel count exceeding limit (10,000)
    // Testing with 10,001 (limit+1) to validate enforcement
    let excessive_kernels: Vec<String> = (0..10_001).map(|i| format!("kernel_{}", i)).collect();
    let receipt = create_test_receipt("real", excessive_kernels);
    let receipt_path = temp_dir.path().join("excessive-kernels.json");
    fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)
        .context("Failed to write test receipt")?;

    assert!(
        verify_receipt_schema(&receipt_path).is_err(),
        "Kernel count exceeding 10,000 should be rejected"
    );

    println!("// Edge case validated: Kernel count limits enforced");
    Ok(())
}

/// Edge case: Test deterministic configuration validation
///
/// This test validates that deterministic environment variables are properly configured.
#[test]
fn test_edge_case_deterministic_configuration() -> Result<()> {
    configure_deterministic_env();

    // Verify deterministic environment is configured
    assert_eq!(
        std::env::var("BITNET_DETERMINISTIC").unwrap_or_default(),
        "1",
        "BITNET_DETERMINISTIC should be set to 1"
    );
    assert_eq!(
        std::env::var("BITNET_SEED").unwrap_or_default(),
        "42",
        "BITNET_SEED should be set to 42"
    );
    assert_eq!(
        std::env::var("RAYON_NUM_THREADS").unwrap_or_default(),
        "1",
        "RAYON_NUM_THREADS should be set to 1"
    );

    println!("// Edge case validated: Deterministic configuration properly set");
    Ok(())
}

/// Edge case: Test negative performance values are rejected
///
/// This test validates that negative performance metrics are rejected.
#[test]
fn test_edge_case_negative_performance() -> Result<()> {
    configure_deterministic_env();

    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;

    // Create receipt with negative performance
    let mut receipt = create_test_receipt("real", vec!["test_kernel".to_string()]);
    receipt["performance"]["tokens_per_sec"] = serde_json::json!(-1.0);

    let receipt_path = temp_dir.path().join("negative-performance.json");
    fs::write(&receipt_path, serde_json::to_string_pretty(&receipt)?)
        .context("Failed to write test receipt")?;

    assert!(
        verify_receipt_schema(&receipt_path).is_err(),
        "Receipt with negative performance should be rejected"
    );

    println!("// Edge case validated: Negative performance properly rejected");
    Ok(())
}

/// Edge case: Test boundary token counts
///
/// This test validates that various token count configurations are handled correctly.
#[test]
fn test_edge_case_boundary_token_counts() -> Result<()> {
    configure_deterministic_env();

    let baseline_path = find_cpu_baseline().context("CPU baseline not found")?;
    let receipt_content = fs::read_to_string(&baseline_path)?;
    let receipt: serde_json::Value = serde_json::from_str(&receipt_content)?;

    // Check if token_count field exists and validate it
    if let Some(token_count) = receipt.get("token_count").and_then(|v| v.as_u64()) {
        assert!(token_count > 0, "Token count should be positive if present");
        assert!(
            token_count <= 10_000,
            "Token count {} exceeds reasonable limit for baseline",
            token_count
        );
        println!("// Edge case validated: Token count {} within bounds", token_count);
    } else {
        println!("// Edge case note: Token count field not present in receipt");
    }

    Ok(())
}
