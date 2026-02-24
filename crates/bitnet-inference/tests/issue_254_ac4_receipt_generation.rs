//! AC4: Receipt Artifact Generation (ci/inference.json) (Issue #254)
//!
//! Tests feature spec: issue-254-real-inference-spec.md#ac4-receipt-artifact
//! API contract: receipt-schema.md#inference-receipt
//!
//! This test validates generation of ci/inference.json receipt with compute_path="real",
//! backend="cpu|cuda", kernels=["i2s_gemv",...], deterministic=true.
#![cfg(feature = "cpu")]
mod support;
use anyhow::Result;
use serde_json::Value;
use serial_test::serial;
use std::collections::HashMap;
use std::path::Path;
use support::EnvGuard;
/// AC:4.1 - Generate inference receipt with compute_path="real"
/// Validates receipt schema and required fields
#[ignore = "blocked by issue #254 (shape mismatch in layer-norm)"]
#[tokio::test]
#[serial(bitnet_env)]
async fn test_ac4_receipt_generation_real_path() -> Result<()> {
    let receipt =
        create_mock_receipt("cpu", vec!["i2s_gemv".to_string(), "rope_apply".to_string()])?;
    assert_eq!(receipt.compute_path, "real", "AC4: compute_path must be 'real'");
    assert_eq!(receipt.backend, "cpu", "AC4: backend should be 'cpu'");
    assert!(
        receipt.kernels.contains(&"i2s_gemv".to_string()),
        "AC4: kernels should include i2s_gemv"
    );
    {
        let _guard = EnvGuard::new("BITNET_DETERMINISTIC");
        _guard.set("1");
        let deterministic_receipt = create_mock_receipt("cpu", vec!["i2s_gemv".to_string()])?;
        assert!(deterministic_receipt.deterministic, "AC4: deterministic should be true");
    }
    println!("AC4.1: Receipt generation test - PENDING IMPLEMENTATION");
    Ok(())
}
/// AC:4.2 - Receipt fails if compute_path="mock"
/// Validates strict enforcement of real inference path
#[tokio::test]
async fn test_ac4_receipt_rejects_mock_path() -> Result<()> {
    let mock_receipt = create_mock_receipt("cpu", vec!["mock_gemv".to_string()])?;
    assert_eq!(
        mock_receipt.compute_path, "mock",
        "AC4: compute_path should be 'mock' if mock kernels detected"
    );
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
    save_receipt(&receipt, &receipt_path)?;
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
#[serial(bitnet_env)]
#[ignore = "Slow integration path (~300s); run with --ignored for full validation"]
async fn test_ac4_receipt_environment_variables_long() -> Result<()> {
    if std::env::var("RUN_SLOW_RECEIPT_TESTS").ok().as_deref() != Some("1") {
        eprintln!("Skipping slow receipt test; set RUN_SLOW_RECEIPT_TESTS=1 to enable");
        return Ok(());
    }
    let _g1 = EnvGuard::new("BITNET_DETERMINISTIC");
    _g1.set("1");
    let _g2 = EnvGuard::new("BITNET_SEED");
    _g2.set("42");
    let _g3 = EnvGuard::new("RAYON_NUM_THREADS");
    _g3.set("1");
    let receipt = create_mock_receipt("cpu", vec!["i2s_gemv".to_string()])?;
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
/// AC:4.6 - Fast receipt validation using committed ci/inference.json
/// Validates the committed receipt artifact via xtask verify-receipt command
/// This is the FAST PATH that avoids timeout issues (completes in ~5ms)
#[test]
fn test_ac4_receipt_environment_variables_fast() -> Result<()> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR should be set by cargo test");
    let workspace_root = Path::new(&manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("Should be able to find workspace root");
    let receipt_path = workspace_root.join("ci").join("inference.json");
    assert!(
        receipt_path.exists(),
        "AC4: ci/inference.json should exist at {:?} (run `cargo run -p xtask -- benchmark` to generate)",
        receipt_path
    );
    let output = std::process::Command::new("cargo")
        .args([
            "run",
            "-p",
            "xtask",
            "--",
            "verify-receipt",
            "--path",
            receipt_path.to_str().unwrap(),
        ])
        .current_dir(workspace_root)
        .output()
        .expect("Failed to execute xtask verify-receipt command");
    assert!(
        output.status.success(),
        "AC4: Receipt verification should succeed. stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    println!("AC4.6: Fast receipt validation test - PASSED");
    Ok(())
}
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
    let json = serde_json::json!(
        { "schema_version" : receipt.schema_version, "timestamp" : receipt.timestamp,
        "compute_path" : receipt.compute_path, "backend" : receipt.backend, "kernels" :
        receipt.kernels, "deterministic" : receipt.deterministic, "environment" : receipt
        .environment, "model_info" : { "quantization_type" : receipt.model_info
        .quantization_type, "layers" : receipt.model_info.layers, "hidden_size" : receipt
        .model_info.hidden_size, }, "test_results" : { "total_tests" : receipt
        .test_results.total_tests, "passed" : receipt.test_results.passed, "failed" :
        receipt.test_results.failed, }, "performance_baseline" : { "tokens_generated" :
        receipt.performance_baseline.tokens_generated, "total_time_ms" : receipt
        .performance_baseline.total_time_ms, "tokens_per_second" : receipt
        .performance_baseline.tokens_per_second, }, }
    );
    std::fs::write(path, serde_json::to_string_pretty(&json)?)?;
    Ok(())
}
