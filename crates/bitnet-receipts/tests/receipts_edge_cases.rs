//! Edge-case tests for bitnet-receipts: InferenceReceipt generation,
//! validation, serialization, schema version, compute path rules.

use bitnet_receipts::{
    AccuracyMetric, AccuracyTestResults, CacheEfficiency, CrossValidation, DeterminismTestResults,
    InferenceReceipt, ModelInfo, ParityMetadata, PerformanceBaseline, RECEIPT_SCHEMA,
    RECEIPT_SCHEMA_VERSION, TestResults,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn schema_version_is_1_0_0() {
    assert_eq!(RECEIPT_SCHEMA_VERSION, "1.0.0");
    assert_eq!(RECEIPT_SCHEMA, "1.0.0");
}

// ---------------------------------------------------------------------------
// InferenceReceipt::generate
// ---------------------------------------------------------------------------

#[test]
fn generate_receipt_real_kernels() {
    let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    assert_eq!(receipt.schema_version, "1.0.0");
    assert_eq!(receipt.compute_path, "real");
    assert_eq!(receipt.backend, "cpu");
    assert_eq!(receipt.kernels, vec!["i2s_gemv"]);
}

#[test]
fn generate_receipt_mock_kernels() {
    let receipt = InferenceReceipt::generate("cpu", vec!["mock_gemv".to_string()], None).unwrap();
    assert_eq!(receipt.compute_path, "mock");
}

#[test]
fn generate_receipt_with_backend_summary() {
    let receipt = InferenceReceipt::generate(
        "cuda",
        vec!["cuda_matmul".to_string()],
        Some("requested=cuda detected=[cuda] selected=cuda".to_string()),
    )
    .unwrap();
    assert_eq!(receipt.backend, "cuda");
    assert!(receipt.backend_summary.contains("cuda"));
}

#[test]
fn generate_receipt_no_backend_summary_default_empty() {
    let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    assert!(receipt.backend_summary.is_empty());
}

#[test]
fn generate_receipt_multiple_kernels() {
    let kernels = vec!["i2s_gemv".to_string(), "rope_apply".to_string(), "softmax_cpu".to_string()];
    let receipt = InferenceReceipt::generate("cpu", kernels.clone(), None).unwrap();
    assert_eq!(receipt.kernels, kernels);
    assert_eq!(receipt.compute_path, "real");
}

#[test]
fn generate_receipt_has_timestamp() {
    let receipt = InferenceReceipt::generate("cpu", vec!["k".to_string()], None).unwrap();
    assert!(!receipt.timestamp.is_empty());
}

#[test]
fn generate_receipt_has_environment() {
    let receipt = InferenceReceipt::generate("cpu", vec!["k".to_string()], None).unwrap();
    assert!(receipt.environment.contains_key("RUST_VERSION"));
    assert!(receipt.environment.contains_key("BITNET_VERSION"));
    assert!(receipt.environment.contains_key("OS"));
}

// ---------------------------------------------------------------------------
// InferenceReceipt::validate
// ---------------------------------------------------------------------------

#[test]
fn validate_real_receipt_ok() {
    let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    assert!(receipt.validate().is_ok());
}

#[test]
fn validate_mock_receipt_fails() {
    let receipt = InferenceReceipt::generate("cpu", vec!["mock_gemv".to_string()], None).unwrap();
    assert!(receipt.validate().is_err());
}

#[test]
fn validate_receipt_with_failed_tests() {
    let mut receipt =
        InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    receipt.test_results.failed = 1;
    let err = receipt.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("Failed tests"));
}

// ---------------------------------------------------------------------------
// InferenceReceipt — serialization
// ---------------------------------------------------------------------------

#[test]
fn to_json_string_valid() {
    let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    let json = receipt.to_json_string().unwrap();
    assert!(json.contains("\"schema_version\""));
    assert!(json.contains("\"compute_path\""));
    assert!(json.contains("\"real\""));
}

#[test]
fn json_roundtrip() {
    let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    let json = receipt.to_json_string().unwrap();
    let parsed: InferenceReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.schema_version, receipt.schema_version);
    assert_eq!(parsed.compute_path, receipt.compute_path);
    assert_eq!(parsed.backend, receipt.backend);
    assert_eq!(parsed.kernels, receipt.kernels);
}

// ---------------------------------------------------------------------------
// ModelInfo
// ---------------------------------------------------------------------------

#[test]
fn model_info_default() {
    let info = ModelInfo::default();
    assert!(info.model_path.is_none());
    assert!(info.quantization_type.is_none());
    assert!(info.layers.is_none());
    assert!(info.hidden_size.is_none());
}

#[test]
fn model_info_custom() {
    let info = ModelInfo {
        model_path: Some("model.gguf".into()),
        quantization_type: Some("I2_S".into()),
        layers: Some(30),
        hidden_size: Some(2560),
        num_attention_heads: Some(20),
        num_key_value_heads: Some(5),
        vocab_size: Some(32000),
        sha256: Some("abc123".into()),
        effective_correction_digest: None,
    };
    assert_eq!(info.layers, Some(30));
}

#[test]
fn model_info_serde_skips_none() {
    let info = ModelInfo::default();
    let json = serde_json::to_string(&info).unwrap();
    // None fields should be skipped
    assert!(!json.contains("model_path"));
}

// ---------------------------------------------------------------------------
// TestResults
// ---------------------------------------------------------------------------

#[test]
fn test_results_default() {
    let tr = TestResults::default();
    assert_eq!(tr.total_tests, 0);
    assert_eq!(tr.passed, 0);
    assert_eq!(tr.failed, 0);
    assert!(tr.skipped.is_none());
}

#[test]
fn test_results_with_accuracy() {
    let tr = TestResults {
        total_tests: 10,
        passed: 9,
        failed: 1,
        skipped: Some(0),
        accuracy_tests: Some(AccuracyTestResults {
            i2s_accuracy: Some(AccuracyMetric { mse: 0.001, tolerance: 0.01, passed: true }),
            tl1_accuracy: None,
            tl2_accuracy: None,
        }),
        determinism_tests: None,
        kv_cache_tests: None,
    };
    assert_eq!(tr.total_tests, 10);
    assert!(tr.accuracy_tests.is_some());
}

// ---------------------------------------------------------------------------
// PerformanceBaseline
// ---------------------------------------------------------------------------

#[test]
fn performance_baseline_default() {
    let pb = PerformanceBaseline::default();
    assert!(pb.tokens_generated.is_none());
    assert!(pb.tokens_per_second.is_none());
}

#[test]
fn performance_baseline_custom() {
    let pb = PerformanceBaseline {
        tokens_generated: Some(100),
        total_time_ms: Some(5000),
        tokens_per_second: Some(20.0),
        first_token_latency_ms: Some(50),
        average_token_latency_ms: Some(10),
        memory_usage_mb: Some(4096),
        cache_efficiency: Some(CacheEfficiency {
            kv_cache_hit_rate: 0.95,
            tensor_cache_hits: 100,
            tensor_cache_misses: 5,
        }),
    };
    assert_eq!(pb.tokens_generated, Some(100));
}

// ---------------------------------------------------------------------------
// ParityMetadata
// ---------------------------------------------------------------------------

#[test]
fn parity_metadata_rust_only() {
    let parity = ParityMetadata {
        cpp_available: false,
        cosine_similarity: None,
        exact_match_rate: None,
        status: "rust_only".into(),
    };
    assert_eq!(parity.status, "rust_only");
    assert!(!parity.cpp_available);
}

#[test]
fn parity_metadata_ok() {
    let parity = ParityMetadata {
        cpp_available: true,
        cosine_similarity: Some(0.999),
        exact_match_rate: Some(1.0),
        status: "ok".into(),
    };
    assert_eq!(parity.status, "ok");
}

// ---------------------------------------------------------------------------
// DeterminismTestResults
// ---------------------------------------------------------------------------

#[test]
fn determinism_results() {
    let dr = DeterminismTestResults { identical_sequences: true, runs: 5, tokens_per_run: 32 };
    assert!(dr.identical_sequences);
    assert_eq!(dr.runs, 5);
}

// ---------------------------------------------------------------------------
// CrossValidation
// ---------------------------------------------------------------------------

#[test]
fn cross_validation_default() {
    let cv = CrossValidation::default();
    assert!(!cv.cpp_reference_available);
    assert!(cv.tolerance.is_none());
}
