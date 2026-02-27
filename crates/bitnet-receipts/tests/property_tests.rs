//! Property-based tests for bitnet-receipts.
//!
//! Key invariants:
//! - Receipts with only real kernels have compute_path == "real"
//! - Receipts with any mock kernel have compute_path == "mock"
//! - JSON round-trip is lossless
//! - Schema version is always "1.0.0"
//! - validate() passes for correctly generated receipts
//! - Kernel count ≤ 10,000 is required; exceeding it produces a specific error
//! - tokens_per_second positive values survive JSON round-trip
//! - cuda backend with GPU kernels passes all validation gates
//! - Invalid schema_version is rejected with a specific error message

use bitnet_receipts::{
    InferenceReceipt, ModelInfo, ParityMetadata, PerformanceBaseline, RECEIPT_SCHEMA_VERSION,
    TestResults,
};
use proptest::prelude::*;

fn real_kernel() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-z][a-z0-9_]{0,30}").unwrap()
}

fn mock_kernel() -> impl Strategy<Value = String> {
    prop::string::string_regex("mock_[a-z0-9_]{1,20}").unwrap()
}

proptest! {
    /// Receipts with only real kernels must have compute_path == "real".
    #[test]
    fn real_kernels_produce_real_path(
        kernels in prop::collection::vec(real_kernel(), 1..10)
    ) {
        // Filter out any accidental "mock" matches from the regex
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }

        let receipt = InferenceReceipt::generate("cpu", kernels, None).unwrap();
        prop_assert_eq!(&receipt.compute_path, "real");
    }

    /// Any receipt containing a mock kernel must have compute_path == "mock".
    #[test]
    fn mock_kernel_produces_mock_path(
        real_kernels in prop::collection::vec(real_kernel(), 0..5),
        mock_kern in mock_kernel()
    ) {
        let mut kernels = real_kernels;
        kernels.push(mock_kern);
        let receipt = InferenceReceipt::generate("cpu", kernels, None).unwrap();
        prop_assert_eq!(&receipt.compute_path, "mock");
    }

    /// Schema version is always the canonical value.
    #[test]
    fn schema_version_is_constant(
        backend in "cpu|cuda|metal",
        kernels in prop::collection::vec(real_kernel(), 1..5)
    ) {
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }
        let receipt = InferenceReceipt::generate(&backend, kernels, None).unwrap();
        prop_assert_eq!(&receipt.schema_version, RECEIPT_SCHEMA_VERSION);
    }

    /// JSON round-trip is lossless for generated receipts.
    #[test]
    fn json_round_trip(
        kernels in prop::collection::vec(real_kernel(), 1..5)
    ) {
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }

        let receipt = InferenceReceipt::generate("cpu", kernels, None).unwrap();
        let json = serde_json::to_string(&receipt).unwrap();
        let restored: InferenceReceipt = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(&receipt.compute_path, &restored.compute_path);
        prop_assert_eq!(&receipt.schema_version, &restored.schema_version);
        prop_assert_eq!(&receipt.backend, &restored.backend);
        prop_assert_eq!(&receipt.kernels, &restored.kernels);
    }

    // ── New property tests ───────────────────────────────────────────────────

    /// RECEIPT_SCHEMA_VERSION constant is always the literal string "1.0.0".
    ///
    /// The constant must never be changed; this property encodes that contract.
    #[test]
    fn schema_version_const_is_one_point_zero_point_zero(
        // Use an arbitrary u8 to force proptest to run the body multiple times.
        _noise in any::<u8>()
    ) {
        prop_assert_eq!(RECEIPT_SCHEMA_VERSION, "1.0.0");
    }

    /// The backend passed to `generate()` is faithfully preserved in the receipt.
    #[test]
    fn builder_preserves_backend(
        backend in "cpu|cuda|metal",
        kernels in prop::collection::vec(real_kernel(), 1..5)
    ) {
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }
        let receipt = InferenceReceipt::generate(&backend, kernels, None).unwrap();
        prop_assert_eq!(&receipt.backend, &backend);
    }

    /// The kernel list passed to `generate()` is faithfully preserved in the receipt.
    #[test]
    fn builder_preserves_kernels(
        kernels in prop::collection::vec(real_kernel(), 1..8)
    ) {
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }
        let receipt = InferenceReceipt::generate("cpu", kernels.clone(), None).unwrap();
        prop_assert_eq!(&receipt.kernels, &kernels);
    }

    /// Valid kernel IDs (non-empty, ≤128 chars, no "mock") pass `validate_kernel_ids()`.
    #[test]
    fn valid_kernel_ids_pass_validation(
        kernels in prop::collection::vec(real_kernel(), 1..5)
    ) {
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.is_empty() && k.len() <= 128 && !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }
        let receipt = InferenceReceipt::generate("cpu", kernels, None).unwrap();
        prop_assert!(receipt.validate_kernel_ids().is_ok());
    }

    /// A kernel ID longer than 128 characters causes `validate_kernel_ids()` to fail.
    #[test]
    fn oversized_kernel_id_fails_validation(
        extra in 1_usize..50
    ) {
        let oversized = "k".repeat(128 + extra);
        let receipt = InferenceReceipt::generate("cpu", vec![oversized], None).unwrap();
        prop_assert!(receipt.validate_kernel_ids().is_err());
    }

    /// Receipts with `compute_path = "real"` pass the honest-compute gate.
    #[test]
    fn real_compute_path_passes_honest_compute(
        kernels in prop::collection::vec(real_kernel(), 1..5)
    ) {
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }
        let receipt = InferenceReceipt::generate("cpu", kernels, None).unwrap();
        // generate() sets compute_path = "real" for non-mock kernels
        prop_assert_eq!(&receipt.compute_path, "real");
        prop_assert!(receipt.validate_compute_path().is_ok());
    }

    /// Receipts with `compute_path = "mock"` fail the honest-compute gate.
    #[test]
    fn mock_compute_path_fails_honest_compute(
        mock_kern in mock_kernel()
    ) {
        let receipt = InferenceReceipt::generate("cpu", vec![mock_kern], None).unwrap();
        prop_assert_eq!(&receipt.compute_path, "mock");
        prop_assert!(receipt.validate_compute_path().is_err());
    }

    /// Receipts built with `with_test_results()` round-trip test counts through JSON.
    #[test]
    fn builder_with_test_results_round_trips(
        total in 0_usize..200,
        failed in 0_usize..50,
        kernels in prop::collection::vec(real_kernel(), 1..3)
    ) {
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }

        let passed = total.saturating_sub(failed);
        let tr = TestResults { total_tests: total, passed, failed, ..Default::default() };
        let receipt = InferenceReceipt::generate("cpu", kernels, None)
            .unwrap()
            .with_test_results(tr);

        let json = serde_json::to_string(&receipt).unwrap();
        let restored: InferenceReceipt = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(restored.test_results.total_tests, total);
        prop_assert_eq!(restored.test_results.passed, passed);
        prop_assert_eq!(restored.test_results.failed, failed);
    }

    /// Receipts built with `with_model_info()` round-trip model metadata through JSON.
    #[test]
    fn builder_with_model_info_round_trips(
        layers in prop::option::of(1_usize..128),
        hidden in prop::option::of(64_usize..4096),
        kernels in prop::collection::vec(real_kernel(), 1..3)
    ) {
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }

        let mi = ModelInfo { layers, hidden_size: hidden, ..Default::default() };
        let receipt = InferenceReceipt::generate("cpu", kernels, None)
            .unwrap()
            .with_model_info(mi);

        let json = serde_json::to_string(&receipt).unwrap();
        let restored: InferenceReceipt = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(restored.model_info.layers, layers);
        prop_assert_eq!(restored.model_info.hidden_size, hidden);
    }

    /// ParityMetadata fields survive a JSON round-trip inside a receipt.
    #[test]
    fn parity_metadata_round_trips(
        cpp_available in any::<bool>(),
        cosine in prop::option::of(0.0_f32..=1.0_f32),
        kernels in prop::collection::vec(real_kernel(), 1..3)
    ) {
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }

        let parity = ParityMetadata {
            cpp_available,
            cosine_similarity: if cpp_available { cosine } else { None },
            exact_match_rate: if cpp_available { Some(1.0) } else { None },
            status: if cpp_available { "ok".to_string() } else { "rust_only".to_string() },
        };
        let receipt = InferenceReceipt::generate("cpu", kernels, None)
            .unwrap()
            .with_parity(parity.clone());

        let json = serde_json::to_string(&receipt).unwrap();
        let restored: InferenceReceipt = serde_json::from_str(&json).unwrap();
        let rp = restored.parity.expect("parity should survive round-trip");
        prop_assert_eq!(rp.cpp_available, cpp_available);
        prop_assert_eq!(rp.status, parity.status);
        // cosine_similarity is an f32; compare at reduced precision to avoid float noise
        match (rp.cosine_similarity, parity.cosine_similarity) {
            (Some(a), Some(b)) => prop_assert!((a - b).abs() < 1e-5),
            (None, None) => {}
            _ => return Err(TestCaseError::fail("cosine_similarity mismatch after round-trip")),
        }
    }

    /// PerformanceBaseline fields survive a JSON round-trip inside a receipt.
    #[test]
    fn performance_baseline_round_trips(
        tokens_generated in prop::option::of(1_usize..10_000),
        total_time_ms in prop::option::of(1_u64..3_600_000_u64),
        kernels in prop::collection::vec(real_kernel(), 1..3)
    ) {
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }

        let pb = PerformanceBaseline {
            tokens_generated,
            total_time_ms,
            ..Default::default()
        };
        let receipt = InferenceReceipt::generate("cpu", kernels, None)
            .unwrap()
            .with_performance_baseline(pb);

        let json = serde_json::to_string(&receipt).unwrap();
        let restored: InferenceReceipt = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(restored.performance_baseline.tokens_generated, tokens_generated);
        prop_assert_eq!(restored.performance_baseline.total_time_ms, total_time_ms);
    }

    /// A positive `tokens_per_second` value survives JSON round-trip faithfully.
    ///
    /// `tokens_per_second > 0.0` is the invariant for a meaningful performance
    /// baseline; this property encodes that such values are never lost in transit.
    #[test]
    fn tokens_per_second_positive_round_trips(
        tps in (1e-3_f64..1e6_f64),
        kernels in prop::collection::vec(real_kernel(), 1..3)
    ) {
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }

        let pb = PerformanceBaseline {
            tokens_per_second: Some(tps),
            ..Default::default()
        };
        let receipt = InferenceReceipt::generate("cpu", kernels, None)
            .unwrap()
            .with_performance_baseline(pb);

        let json = serde_json::to_string(&receipt).unwrap();
        let restored: InferenceReceipt = serde_json::from_str(&json).unwrap();
        let restored_tps = restored.performance_baseline.tokens_per_second
            .expect("tokens_per_second must survive round-trip");
        prop_assert!(restored_tps > 0.0, "tokens_per_second must be positive after round-trip");
        prop_assert!((restored_tps - tps).abs() < tps * 1e-9,
            "tokens_per_second changed after round-trip: {} -> {}", tps, restored_tps);
    }

    /// Kernel counts within [1, 10 000] always pass `validate_kernel_ids()`.
    #[test]
    fn kernel_count_within_10k_passes_validation(
        count in 1_usize..=10_000_usize
    ) {
        let kernels = vec!["i2s_gemv".to_string(); count];
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = kernels;
        prop_assert!(
            receipt.validate_kernel_ids().is_ok(),
            "count={} should be within the 10,000 limit",
            count
        );
    }

    /// Kernel counts exceeding 10 000 always fail `validate_kernel_ids()` with
    /// an error message that mentions "10,000".
    #[test]
    fn kernel_count_exceeding_10k_fails_validation(
        extra in 1_usize..=1_000_usize
    ) {
        let count = 10_000 + extra;
        let kernels = vec!["i2s_gemv".to_string(); count];
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = kernels;
        let err = receipt.validate_kernel_ids();
        prop_assert!(err.is_err(), "count={} should exceed the 10,000 limit", count);
        prop_assert!(
            err.unwrap_err().to_string().contains("10,000"),
            "error for count={} must mention '10,000'",
            count
        );
    }

    /// A receipt with `backend = "cuda"` and GPU-style kernel IDs passes all
    /// validation gates.  This encodes the contract that GPU receipts are valid
    /// as long as the kernel IDs satisfy the hygiene rules.
    #[test]
    fn cuda_backend_with_gpu_kernels_passes_validation(
        suffix in "[a-z0-9]{1,12}"
    ) {
        let kernels = vec![
            format!("gemm_gpu_{suffix}"),
            format!("cuda_i2s_{suffix}"),
        ];
        let receipt = InferenceReceipt::generate("cuda", kernels, None).unwrap();
        prop_assert_eq!(&receipt.backend, "cuda");
        prop_assert!(
            receipt.validate().is_ok(),
            "cuda receipt with GPU-style kernels must pass all validation gates"
        );
    }

    /// An `InferenceReceipt` whose `schema_version` has been mutated away from
    /// "1.0.0" fails `validate_schema()` with an error mentioning the bad value.
    #[test]
    fn invalid_schema_version_rejected_with_specific_error(
        bad_version in "[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}",
        kernels in prop::collection::vec(real_kernel(), 1..3)
    ) {
        // Only test versions that actually differ from the canonical one.
        prop_assume!(bad_version != "1.0.0");
        let kernels: Vec<String> = kernels
            .into_iter()
            .filter(|k| !k.to_lowercase().contains("mock"))
            .collect();
        if kernels.is_empty() { return Ok(()); }

        let mut receipt = InferenceReceipt::generate("cpu", kernels, None).unwrap();
        receipt.schema_version = bad_version.clone();
        let err = receipt.validate_schema();
        prop_assert!(err.is_err(), "schema_version '{}' should fail validate_schema()", bad_version);
        let msg = err.unwrap_err().to_string();
        prop_assert!(
            msg.contains(&bad_version) || msg.contains("1.0.0"),
            "error must reference the invalid version or the expected version, got: {}", msg
        );
    }
}

#[test]
fn generated_receipt_validates() {
    let receipt = InferenceReceipt::generate(
        "cpu",
        vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
        None,
    )
    .unwrap();
    receipt.validate().expect("Valid receipt should pass validation");
}

#[test]
fn mock_receipt_fails_validation() {
    let receipt = InferenceReceipt::generate("cpu", vec!["mock_gemv".to_string()], None).unwrap();
    assert!(receipt.validate().is_err(), "Receipt with mock kernels should fail validation");
}

#[test]
fn snapshot_receipt_structure() {
    let mut receipt = InferenceReceipt::generate(
        "cpu",
        vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
        None,
    )
    .unwrap();
    // Normalize non-deterministic fields for snapshot
    receipt.timestamp = "2024-01-01T00:00:00+00:00".to_string();
    receipt.environment.clear();

    insta::assert_json_snapshot!("receipt_structure", receipt, {
        ".timestamp" => "[timestamp]",
    });
}
