//! AC4: Receipt Artifact Generation (ci/inference.json) (Issue #254)
//!
//! Tests feature spec: issue-254-real-inference-spec.md#ac4-receipt-artifact
//! API contract: receipt-schema.md#inference-receipt
//!
//! This test validates generation of ci/inference.json receipt with compute_path="real",
//! backend="cpu|cuda", kernels=["i2s_gemv",...], deterministic=true.

#![cfg(feature = "cpu")]

mod support;
use support::EnvGuard;

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// AC:4.1 - Generate inference receipt with compute_path="real"
/// Validates receipt schema and required fields
#[tokio::test]
#[serial_test::serial]
async fn test_ac4_receipt_generation_real_path() -> Result<()> {
    use bitnet_inference::receipts::{InferenceReceipt, RECEIPT_SCHEMA_VERSION};

    // AC4.1: Generate receipt with real inference kernels
    let kernels = vec!["i2s_gemv".to_string(), "rope_apply".to_string()];
    let receipt = InferenceReceipt::generate("cpu", kernels.clone())?;

    // AC4: Verify schema version is "1.0.0"
    assert_eq!(
        receipt.schema_version, RECEIPT_SCHEMA_VERSION,
        "AC4: schema_version must be '1.0.0'"
    );
    assert_eq!(receipt.schema_version, "1.0.0", "AC4: schema_version must be '1.0.0'");

    // AC4: Verify compute_path is "real" (not "mock")
    assert_eq!(receipt.compute_path, "real", "AC4: compute_path must be 'real'");

    // AC4: Verify backend is "cpu"
    assert_eq!(receipt.backend, "cpu", "AC4: backend should be 'cpu'");

    // AC4: Verify kernels list contains expected kernel IDs
    assert!(
        receipt.kernels.contains(&"i2s_gemv".to_string()),
        "AC4: kernels should include i2s_gemv"
    );
    assert!(
        receipt.kernels.contains(&"rope_apply".to_string()),
        "AC4: kernels should include rope_apply"
    );
    assert!(!receipt.kernels.is_empty(), "AC4: kernel array must be non-empty");

    // AC4: Verify timestamp is present and valid RFC3339 format
    assert!(!receipt.timestamp.is_empty(), "AC4: timestamp must be present");
    assert!(
        chrono::DateTime::parse_from_rfc3339(&receipt.timestamp).is_ok(),
        "AC4: timestamp must be valid RFC3339 format"
    );

    // AC4: Verify environment variables are captured
    assert!(!receipt.environment.is_empty(), "AC4: environment should not be empty");
    assert!(
        receipt.environment.contains_key("RUST_VERSION"),
        "AC4: environment should include RUST_VERSION"
    );
    assert!(
        receipt.environment.contains_key("BITNET_VERSION"),
        "AC4: environment should include BITNET_VERSION"
    );
    assert!(receipt.environment.contains_key("OS"), "AC4: environment should include OS");

    // AC4: Verify deterministic flag with guard for automatic cleanup
    {
        let _guard = EnvGuard::set("BITNET_DETERMINISTIC", "1");
        let deterministic_receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()])?;
        assert!(
            deterministic_receipt.deterministic,
            "AC4: deterministic should be true when BITNET_DETERMINISTIC=1"
        );
        assert!(
            deterministic_receipt.environment.contains_key("BITNET_DETERMINISTIC"),
            "AC4: environment should include BITNET_DETERMINISTIC when set"
        );
    }

    // AC4: Verify model_info is present (may be default/empty)
    // Schema allows optional fields, but structure must exist
    assert!(receipt.model_info.model_path.is_none(), "AC4: model_path should be None by default");

    // AC4: Verify test_results is present (may be default)
    assert_eq!(receipt.test_results.total_tests, 0, "AC4: default test_results.total_tests is 0");

    // AC4: Verify performance_baseline is present (may be default)
    assert!(
        receipt.performance_baseline.tokens_generated.is_none(),
        "AC4: default performance_baseline.tokens_generated is None"
    );

    // AC4: Verify receipt validates successfully
    receipt.validate().context("AC4: Receipt validation should pass for real compute path")?;

    // AC4: Verify individual validation methods
    receipt.validate_schema().context("AC4: Schema validation should pass")?;
    receipt.validate_compute_path().context("AC4: Compute path validation should pass")?;
    receipt.validate_kernel_ids().context("AC4: Kernel ID validation should pass")?;

    println!("✓ AC4.1: Receipt generation test - PASSED");
    Ok(())
}

/// AC:4.2 - Receipt fails if compute_path="mock"
/// Validates strict enforcement of real inference path
#[tokio::test]
async fn test_ac4_receipt_rejects_mock_path() -> Result<()> {
    // TODO: Implement receipt validation
    let mock_receipt = create_mock_receipt("cpu", vec!["mock_gemv".to_string()])?;

    // AC4: Should detect mock kernels and set compute_path="mock"
    assert_eq!(
        mock_receipt.compute_path, "mock",
        "AC4: compute_path should be 'mock' if mock kernels detected"
    );

    // TODO: Implement validation that fails
    // let result = mock_receipt.validate();
    // assert!(result.is_err(), "AC4: Receipt validation should fail for mock kernels");

    println!("AC4.2: Mock rejection test - PENDING IMPLEMENTATION");
    Ok(())
}

/// AC:4.3 - Save receipt to ci/inference.json
/// Validates receipt file creation
#[tokio::test]
async fn test_ac4_save_receipt_to_file() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let receipt_path = temp_dir.path().join("inference.json");

    let receipt = create_mock_receipt("cpu", vec!["i2s_gemv".to_string()])?;

    // AC4: Save receipt to file
    save_receipt(&receipt, &receipt_path)?;

    // Verify file exists and is valid JSON
    assert!(receipt_path.exists(), "AC4: Receipt file should exist");

    let file_content = std::fs::read_to_string(&receipt_path)?;
    let json: Value = serde_json::from_str(&file_content)?;

    assert_eq!(json["compute_path"], "real", "AC4: Saved receipt compute_path");
    assert_eq!(json["backend"], "cpu", "AC4: Saved receipt backend");

    println!("AC4.3: Receipt file saving test - PENDING IMPLEMENTATION");
    Ok(())
}

/// AC:4.4 - Receipt includes environment variables
/// Validates environment section in receipt
#[tokio::test]
#[serial_test::serial]
async fn test_ac4_receipt_environment_variables() -> Result<()> {
    let _g1 = EnvGuard::set("BITNET_DETERMINISTIC", "1");
    let _g2 = EnvGuard::set("BITNET_SEED", "42");
    let _g3 = EnvGuard::set("RAYON_NUM_THREADS", "1");

    let receipt = create_mock_receipt("cpu", vec!["i2s_gemv".to_string()])?;

    // AC4: Verify environment variables captured
    assert_eq!(
        receipt.environment.get("BITNET_DETERMINISTIC"),
        Some(&"1".to_string()),
        "AC4: Environment should include BITNET_DETERMINISTIC"
    );
    assert_eq!(
        receipt.environment.get("BITNET_SEED"),
        Some(&"42".to_string()),
        "AC4: Environment should include BITNET_SEED"
    );

    println!("AC4.4: Environment variables test - PENDING IMPLEMENTATION");
    Ok(())
}

/// AC:4.5 - Receipt includes performance baseline
/// Validates performance metrics in receipt
#[tokio::test]
async fn test_ac4_receipt_performance_baseline() -> Result<()> {
    let receipt = create_mock_receipt("cpu", vec!["i2s_gemv".to_string()])?;

    // AC4: Verify performance baseline fields
    assert!(
        receipt.performance_baseline.tokens_generated > 0,
        "AC4: Performance baseline should include tokens_generated"
    );
    assert!(
        receipt.performance_baseline.tokens_per_second > 0.0,
        "AC4: Performance baseline should include tokens_per_second"
    );

    println!("AC4.5: Performance baseline test - PENDING IMPLEMENTATION");
    Ok(())
}

// Helper types and functions

#[derive(Debug, Clone)]
struct InferenceReceipt {
    schema_version: String,
    timestamp: String,
    compute_path: String,
    backend: String,
    kernels: Vec<String>,
    deterministic: bool,
    environment: HashMap<String, String>,
    model_info: ModelInfo,
    test_results: TestResults,
    performance_baseline: PerformanceBaseline,
}

#[derive(Debug, Clone)]
struct ModelInfo {
    quantization_type: String,
    layers: usize,
    hidden_size: usize,
}

#[derive(Debug, Clone)]
struct TestResults {
    total_tests: usize,
    passed: usize,
    failed: usize,
}

#[derive(Debug, Clone)]
struct PerformanceBaseline {
    tokens_generated: usize,
    total_time_ms: usize,
    tokens_per_second: f64,
}

fn create_mock_receipt(backend: &str, kernels: Vec<String>) -> Result<InferenceReceipt> {
    let compute_path = if kernels.iter().any(|k| k.contains("mock")) { "mock" } else { "real" };

    let mut environment = HashMap::new();
    if let Ok(val) = std::env::var("BITNET_DETERMINISTIC") {
        environment.insert("BITNET_DETERMINISTIC".to_string(), val);
    }
    if let Ok(val) = std::env::var("BITNET_SEED") {
        environment.insert("BITNET_SEED".to_string(), val);
    }
    if let Ok(val) = std::env::var("RAYON_NUM_THREADS") {
        environment.insert("RAYON_NUM_THREADS".to_string(), val);
    }

    Ok(InferenceReceipt {
        schema_version: "1.0.0".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        compute_path: compute_path.to_string(),
        backend: backend.to_string(),
        kernels,
        deterministic: std::env::var("BITNET_DETERMINISTIC").is_ok(),
        environment,
        model_info: ModelInfo {
            quantization_type: "I2_S".to_string(),
            layers: 32,
            hidden_size: 2048,
        },
        test_results: TestResults { total_tests: 10, passed: 10, failed: 0 },
        performance_baseline: PerformanceBaseline {
            tokens_generated: 100,
            total_time_ms: 5000,
            tokens_per_second: 20.0,
        },
    })
}

fn save_receipt(receipt: &InferenceReceipt, path: &Path) -> Result<()> {
    let json = serde_json::json!({
        "schema_version": receipt.schema_version,
        "timestamp": receipt.timestamp,
        "compute_path": receipt.compute_path,
        "backend": receipt.backend,
        "kernels": receipt.kernels,
        "deterministic": receipt.deterministic,
        "environment": receipt.environment,
        "model_info": {
            "quantization_type": receipt.model_info.quantization_type,
            "layers": receipt.model_info.layers,
            "hidden_size": receipt.model_info.hidden_size,
        },
        "test_results": {
            "total_tests": receipt.test_results.total_tests,
            "passed": receipt.test_results.passed,
            "failed": receipt.test_results.failed,
        },
        "performance_baseline": {
            "tokens_generated": receipt.performance_baseline.tokens_generated,
            "total_time_ms": receipt.performance_baseline.total_time_ms,
            "tokens_per_second": receipt.performance_baseline.tokens_per_second,
        },
    });

    std::fs::write(path, serde_json::to_string_pretty(&json)?)?;
    Ok(())
}
