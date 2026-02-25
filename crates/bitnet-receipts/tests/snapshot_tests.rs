//! Snapshot tests for bitnet-receipts.
//!
//! Pins the serialized structure of InferenceReceipt variants:
//! - CPU-only receipt with real kernels
//! - Receipt with model info populated
//! - Receipt with performance baseline
//! - Receipt with test results
//! - Mock receipt (compute_path = "mock")
//! - Backend summary field in receipt

use bitnet_receipts::{InferenceReceipt, ModelInfo, PerformanceBaseline, TestResults};

/// Normalize non-deterministic fields for stable snapshots.
fn normalize(mut r: InferenceReceipt) -> InferenceReceipt {
    r.timestamp = "2024-01-01T00:00:00+00:00".to_string();
    r.environment.clear();
    r
}

#[test]
fn snapshot_cpu_real_receipt() {
    let receipt = normalize(
        InferenceReceipt::generate(
            "cpu",
            vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
            None,
        )
        .unwrap(),
    );
    insta::assert_json_snapshot!("cpu_real_receipt", receipt, {
        ".timestamp" => "[timestamp]",
    });
}

#[test]
fn snapshot_mock_receipt() {
    let receipt =
        normalize(InferenceReceipt::generate("cpu", vec!["mock_gemv".to_string()], None).unwrap());
    insta::assert_json_snapshot!("mock_receipt", receipt, {
        ".timestamp" => "[timestamp]",
    });
}

#[test]
fn snapshot_receipt_with_backend_summary() {
    let receipt = normalize(
        InferenceReceipt::generate(
            "cpu",
            vec!["i2s_gemv".to_string()],
            Some("requested=cpu detected=[cpu] selected=cpu".to_string()),
        )
        .unwrap(),
    );
    insta::assert_json_snapshot!("receipt_with_backend_summary", receipt, {
        ".timestamp" => "[timestamp]",
    });
}

#[test]
fn snapshot_receipt_with_model_info() {
    let model_info = ModelInfo {
        model_path: Some("models/test.gguf".to_string()),
        quantization_type: Some("i2_s".to_string()),
        layers: Some(32),
        hidden_size: Some(2048),
        num_attention_heads: Some(32),
        num_key_value_heads: Some(8),
        vocab_size: Some(32000),
        ..ModelInfo::default()
    };
    let receipt = normalize(
        InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
            .unwrap()
            .with_model_info(model_info),
    );
    insta::assert_json_snapshot!("receipt_with_model_info", receipt, {
        ".timestamp" => "[timestamp]",
    });
}

#[test]
fn snapshot_receipt_with_performance() {
    let perf = PerformanceBaseline {
        tokens_generated: Some(128),
        total_time_ms: Some(1024),
        tokens_per_second: Some(125.0),
        first_token_latency_ms: Some(32),
        average_token_latency_ms: Some(8),
        memory_usage_mb: Some(512),
        cache_efficiency: None,
    };
    let receipt = normalize(
        InferenceReceipt::generate(
            "cpu",
            vec!["i2s_gemv".to_string(), "attention_real".to_string()],
            None,
        )
        .unwrap()
        .with_performance_baseline(perf),
    );
    insta::assert_json_snapshot!("receipt_with_performance", receipt, {
        ".timestamp" => "[timestamp]",
    });
}

#[test]
fn snapshot_receipt_with_test_results() {
    let test_results = TestResults {
        total_tests: 100,
        passed: 98,
        failed: 2,
        skipped: Some(5),
        accuracy_tests: None,
        determinism_tests: None,
        kv_cache_tests: None,
    };
    let receipt = normalize(
        InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None)
            .unwrap()
            .with_test_results(test_results),
    );
    insta::assert_json_snapshot!("receipt_with_test_results", receipt, {
        ".timestamp" => "[timestamp]",
    });
}
