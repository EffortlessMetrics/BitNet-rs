//! Edge-case tests for receipt generation, validation, and serde.
//!
//! Covers InferenceReceipt::generate, validate (AC4/AC9 gates), schema
//! version, compute path detection, kernel ID hygiene, and serde roundtrips.

use bitnet_receipts::*;

// ---------------------------------------------------------------------------
// Helper: minimal valid receipt
// ---------------------------------------------------------------------------

fn real_receipt() -> InferenceReceipt {
    InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap()
}

fn real_receipt_with_summary() -> InferenceReceipt {
    InferenceReceipt::generate(
        "cpu",
        vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
        Some("requested=cpu detected=[cpu] selected=cpu".to_string()),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// generate()
// ---------------------------------------------------------------------------

#[test]
fn generate_real_kernels() {
    let r = real_receipt();
    assert_eq!(r.schema_version, "1.0.0");
    assert_eq!(r.compute_path, "real");
    assert_eq!(r.backend, "cpu");
    assert!(!r.timestamp.is_empty());
}

#[test]
fn generate_with_backend_summary() {
    let r = real_receipt_with_summary();
    assert!(r.backend_summary.contains("selected="));
}

#[test]
fn generate_cuda_backend() {
    let r = InferenceReceipt::generate(
        "cuda",
        vec!["cuda_gemv".to_string()],
        Some("requested=cuda detected=[cuda] selected=cuda".to_string()),
    )
    .unwrap();
    assert_eq!(r.backend, "cuda");
    assert_eq!(r.compute_path, "real");
}

#[test]
fn generate_mock_kernel_detected() {
    let r = InferenceReceipt::generate("cpu", vec!["mock_matmul".to_string()], None).unwrap();
    assert_eq!(r.compute_path, "mock");
}

#[test]
fn generate_empty_kernels() {
    let r = InferenceReceipt::generate("cpu", vec![], None).unwrap();
    assert!(r.kernels.is_empty());
}

// ---------------------------------------------------------------------------
// validate_schema()
// ---------------------------------------------------------------------------

#[test]
fn validate_schema_ok() {
    let r = real_receipt();
    assert!(r.validate_schema().is_ok());
}

#[test]
fn validate_schema_wrong_version() {
    let mut r = real_receipt();
    r.schema_version = "2.0.0".to_string();
    let err = r.validate_schema().unwrap_err();
    assert!(err.to_string().contains("2.0.0"));
}

// ---------------------------------------------------------------------------
// validate_compute_path()
// ---------------------------------------------------------------------------

#[test]
fn validate_compute_path_real_ok() {
    let r = real_receipt();
    assert!(r.validate_compute_path().is_ok());
}

#[test]
fn validate_compute_path_mock_fails() {
    let mut r = real_receipt();
    r.compute_path = "mock".to_string();
    assert!(r.validate_compute_path().is_err());
}

// ---------------------------------------------------------------------------
// validate_kernel_ids()
// ---------------------------------------------------------------------------

#[test]
fn validate_kernel_ids_real_ok() {
    let r = real_receipt();
    assert!(r.validate_kernel_ids().is_ok());
}

#[test]
fn validate_kernel_ids_empty_kernel_fails() {
    let mut r = real_receipt();
    r.kernels = vec!["".to_string()];
    assert!(r.validate_kernel_ids().is_err());
}

#[test]
fn validate_kernel_ids_mock_kernel_fails() {
    let mut r = real_receipt();
    r.kernels = vec!["mock_gemv".to_string()];
    assert!(r.validate_kernel_ids().is_err());
}

#[test]
fn validate_kernel_ids_whitespace_only_fails() {
    let mut r = real_receipt();
    r.kernels = vec!["   ".to_string()];
    assert!(r.validate_kernel_ids().is_err());
}

#[test]
fn validate_kernel_ids_too_long_fails() {
    let mut r = real_receipt();
    r.kernels = vec!["x".repeat(129)];
    assert!(r.validate_kernel_ids().is_err());
}

#[test]
fn validate_kernel_ids_max_length_ok() {
    let mut r = real_receipt();
    r.kernels = vec!["x".repeat(128)];
    assert!(r.validate_kernel_ids().is_ok());
}

// ---------------------------------------------------------------------------
// validate() — full AC9 gate
// ---------------------------------------------------------------------------

#[test]
fn validate_full_real_receipt_ok() {
    let r = real_receipt();
    assert!(r.validate().is_ok());
}

#[test]
fn validate_fails_on_failed_tests() {
    let mut r = real_receipt();
    r.test_results.failed = 1;
    let err = r.validate().unwrap_err();
    assert!(err.to_string().contains("Failed tests"));
}

#[test]
fn validate_fails_on_bad_schema() {
    let mut r = real_receipt();
    r.schema_version = "0.1.0".to_string();
    assert!(r.validate().is_err());
}

#[test]
fn validate_fails_on_mock_compute_path() {
    let mut r = real_receipt();
    r.compute_path = "mock".to_string();
    assert!(r.validate().is_err());
}

#[test]
fn validate_fails_on_mock_kernel_in_real_receipt() {
    let mut r = real_receipt();
    r.kernels.push("MOCK_attention".to_string());
    assert!(r.validate().is_err());
}

#[test]
fn validate_backend_summary_invalid_format() {
    let mut r = real_receipt();
    r.backend_summary = "bad format".to_string();
    assert!(r.validate().is_err());
}

#[test]
fn validate_backend_summary_empty_is_ok() {
    let mut r = real_receipt();
    r.backend_summary = String::new();
    assert!(r.validate().is_ok());
}

#[test]
fn validate_backend_summary_valid_format() {
    let r = real_receipt_with_summary();
    assert!(r.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Accuracy test validation
// ---------------------------------------------------------------------------

#[test]
fn validate_accuracy_i2s_failed() {
    let mut r = real_receipt();
    r.test_results.accuracy_tests = Some(AccuracyTestResults {
        i2s_accuracy: Some(AccuracyMetric { mse: 0.1, tolerance: 0.01, passed: false }),
        tl1_accuracy: None,
        tl2_accuracy: None,
    });
    let err = r.validate().unwrap_err();
    assert!(err.to_string().contains("I2S accuracy"));
}

#[test]
fn validate_accuracy_tl1_failed() {
    let mut r = real_receipt();
    r.test_results.accuracy_tests = Some(AccuracyTestResults {
        i2s_accuracy: None,
        tl1_accuracy: Some(AccuracyMetric { mse: 0.5, tolerance: 0.01, passed: false }),
        tl2_accuracy: None,
    });
    let err = r.validate().unwrap_err();
    assert!(err.to_string().contains("TL1 accuracy"));
}

#[test]
fn validate_accuracy_all_pass() {
    let mut r = real_receipt();
    r.test_results.accuracy_tests = Some(AccuracyTestResults {
        i2s_accuracy: Some(AccuracyMetric { mse: 0.001, tolerance: 0.01, passed: true }),
        tl1_accuracy: Some(AccuracyMetric { mse: 0.002, tolerance: 0.01, passed: true }),
        tl2_accuracy: Some(AccuracyMetric { mse: 0.003, tolerance: 0.01, passed: true }),
    });
    assert!(r.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Determinism test validation
// ---------------------------------------------------------------------------

#[test]
fn validate_determinism_fails_when_enabled() {
    let mut r = real_receipt();
    r.deterministic = true;
    r.test_results.determinism_tests =
        Some(DeterminismTestResults { identical_sequences: false, runs: 3, tokens_per_run: 16 });
    let err = r.validate().unwrap_err();
    assert!(err.to_string().contains("Determinism"));
}

#[test]
fn validate_determinism_ok_when_identical() {
    let mut r = real_receipt();
    r.deterministic = true;
    r.test_results.determinism_tests =
        Some(DeterminismTestResults { identical_sequences: true, runs: 3, tokens_per_run: 16 });
    assert!(r.validate().is_ok());
}

#[test]
fn validate_determinism_ignored_when_not_deterministic() {
    let mut r = real_receipt();
    r.deterministic = false;
    r.test_results.determinism_tests =
        Some(DeterminismTestResults { identical_sequences: false, runs: 3, tokens_per_run: 16 });
    assert!(r.validate().is_ok());
}

// ---------------------------------------------------------------------------
// Serde roundtrips
// ---------------------------------------------------------------------------

#[test]
fn receipt_serde_roundtrip() {
    let r = real_receipt_with_summary();
    let json = r.to_json_string().unwrap();
    let r2: InferenceReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(r.schema_version, r2.schema_version);
    assert_eq!(r.compute_path, r2.compute_path);
    assert_eq!(r.backend, r2.backend);
    assert_eq!(r.kernels, r2.kernels);
    assert_eq!(r.backend_summary, r2.backend_summary);
}

#[test]
fn model_info_serde_roundtrip() {
    let info = ModelInfo {
        model_path: Some("model.gguf".to_string()),
        quantization_type: Some("I2_S".to_string()),
        layers: Some(30),
        hidden_size: Some(2560),
        num_attention_heads: Some(20),
        num_key_value_heads: Some(5),
        vocab_size: Some(32000),
        sha256: Some("abc123".to_string()),
        effective_correction_digest: None,
    };
    let json = serde_json::to_string(&info).unwrap();
    let info2: ModelInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(info.layers, info2.layers);
    assert_eq!(info.hidden_size, info2.hidden_size);
}

#[test]
fn test_results_serde_roundtrip() {
    let tr = TestResults {
        total_tests: 100,
        passed: 98,
        failed: 2,
        skipped: Some(5),
        accuracy_tests: None,
        determinism_tests: None,
        kv_cache_tests: None,
    };
    let json = serde_json::to_string(&tr).unwrap();
    let tr2: TestResults = serde_json::from_str(&json).unwrap();
    assert_eq!(tr.total_tests, tr2.total_tests);
    assert_eq!(tr.failed, tr2.failed);
}

#[test]
fn performance_baseline_serde_roundtrip() {
    let pb = PerformanceBaseline {
        tokens_generated: Some(100),
        total_time_ms: Some(5000),
        tokens_per_second: Some(20.0),
        first_token_latency_ms: Some(100),
        average_token_latency_ms: Some(50),
        memory_usage_mb: Some(2048),
        cache_efficiency: Some(CacheEfficiency {
            kv_cache_hit_rate: 0.95,
            tensor_cache_hits: 500,
            tensor_cache_misses: 25,
        }),
    };
    let json = serde_json::to_string(&pb).unwrap();
    let pb2: PerformanceBaseline = serde_json::from_str(&json).unwrap();
    assert_eq!(pb.tokens_generated, pb2.tokens_generated);
}

#[test]
fn parity_metadata_serde_roundtrip() {
    let pm = ParityMetadata {
        cpp_available: true,
        cosine_similarity: Some(0.999),
        exact_match_rate: Some(1.0),
        status: "ok".to_string(),
    };
    let json = serde_json::to_string(&pm).unwrap();
    let pm2: ParityMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(pm.status, pm2.status);
    assert_eq!(pm.cpp_available, pm2.cpp_available);
}

// ---------------------------------------------------------------------------
// Default values
// ---------------------------------------------------------------------------

#[test]
fn model_info_default() {
    let info = ModelInfo::default();
    assert!(info.model_path.is_none());
    assert!(info.layers.is_none());
    assert!(info.vocab_size.is_none());
}

#[test]
fn test_results_default() {
    let tr = TestResults::default();
    assert_eq!(tr.total_tests, 0);
    assert_eq!(tr.passed, 0);
    assert_eq!(tr.failed, 0);
}

#[test]
fn performance_baseline_default() {
    let pb = PerformanceBaseline::default();
    assert!(pb.tokens_generated.is_none());
    assert!(pb.tokens_per_second.is_none());
}

// ---------------------------------------------------------------------------
// Schema constants
// ---------------------------------------------------------------------------

#[test]
fn schema_version_constant() {
    assert_eq!(RECEIPT_SCHEMA_VERSION, "1.0.0");
}

#[test]
fn schema_alias_constant() {
    assert_eq!(RECEIPT_SCHEMA, RECEIPT_SCHEMA_VERSION);
}

// ---------------------------------------------------------------------------
// JSON string output
// ---------------------------------------------------------------------------

#[test]
fn to_json_string_pretty_prints() {
    let r = real_receipt();
    let json = r.to_json_string().unwrap();
    assert!(json.contains("\"schema_version\""));
    assert!(json.contains("\"compute_path\""));
    // Pretty-printed should have newlines
    assert!(json.contains('\n'));
}

// ---------------------------------------------------------------------------
// Multi-SLM model info configs
// ---------------------------------------------------------------------------

#[test]
fn phi4_model_info() {
    let info = ModelInfo {
        model_path: Some("phi-4.gguf".to_string()),
        layers: Some(40),
        hidden_size: Some(5120),
        num_attention_heads: Some(40),
        num_key_value_heads: Some(10),
        vocab_size: Some(100352),
        ..Default::default()
    };
    assert_eq!(info.layers, Some(40));
    assert_eq!(info.vocab_size, Some(100352));
}

#[test]
fn llama3_model_info() {
    let info = ModelInfo {
        model_path: Some("llama-3-8b.gguf".to_string()),
        layers: Some(32),
        hidden_size: Some(4096),
        num_attention_heads: Some(32),
        num_key_value_heads: Some(8),
        vocab_size: Some(128256),
        ..Default::default()
    };
    assert_eq!(info.layers, Some(32));
    assert_eq!(info.vocab_size, Some(128256));
}
