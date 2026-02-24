//! Property-based tests for bitnet-receipts.
//!
//! Key invariants:
//! - Receipts with only real kernels have compute_path == "real"
//! - Receipts with any mock kernel have compute_path == "mock"
//! - JSON round-trip is lossless
//! - Schema version is always "1.0.0"
//! - validate() passes for correctly generated receipts

use bitnet_receipts::{InferenceReceipt, RECEIPT_SCHEMA_VERSION};
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
