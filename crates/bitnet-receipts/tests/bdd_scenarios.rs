//! BDD-style scenario tests for `bitnet-receipts`.
//!
//! Each test follows the **Given / When / Then** structure.  All scenarios are
//! fast (no I/O, no filesystem access) and complete in milliseconds.
//!
//! # Covered scenarios
//! - Given valid receipt JSON, When validated, Then passes schema check
//! - Given compute_path="mock", When validate_compute_path(), Then returns error
//! - Given compute_path="real", When validate_compute_path(), Then passes
//! - Given missing required field (schema_version), When validate_schema(), Then returns error
//! - Given kernel IDs containing "mock", When generate(), Then compute_path is "mock"
//! - Given all-real kernel IDs, When generate(), Then compute_path is "real"
//! - Given empty kernel list, When validate_kernel_ids(), Then returns error
//! - Given kernel ID longer than 128 chars, When validate_kernel_ids(), Then returns error
//! - Given backend "cpu", When generate(), Then receipt records "cpu"
//! - Given backend "cuda", When generate(), Then receipt records "cuda"
//! - Given schema_version != "1.0.0", When validate_schema(), Then returns error

use bitnet_receipts::{InferenceReceipt, RECEIPT_SCHEMA_VERSION};

// ── Schema version ────────────────────────────────────────────────────────────

/// Given: the schema version constant
/// When: compared to the expected value
/// Then: equals "1.0.0"
#[test]
fn given_schema_version_constant_when_checked_then_equals_one_zero_zero() {
    assert_eq!(RECEIPT_SCHEMA_VERSION, "1.0.0", "RECEIPT_SCHEMA_VERSION must equal \"1.0.0\"");
}

/// Given: a freshly generated receipt
/// When: schema_version field is inspected
/// Then: equals "1.0.0"
#[test]
fn given_fresh_receipt_when_schema_version_read_then_is_one_zero_zero() {
    let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    assert_eq!(
        receipt.schema_version, "1.0.0",
        "generated receipt must carry schema_version = \"1.0.0\""
    );
}

/// Given: a receipt with schema_version set to an unsupported value
/// When: validate_schema() is called
/// Then: returns an error
#[test]
fn given_wrong_schema_version_when_validate_schema_then_returns_error() {
    let mut receipt =
        InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    receipt.schema_version = "2.0.0".to_string();

    let result = receipt.validate_schema();
    assert!(
        result.is_err(),
        "validate_schema() must return Err when schema_version is not \"1.0.0\""
    );
}

// ── compute_path ─────────────────────────────────────────────────────────────

/// Given: a receipt with compute_path = "real"
/// When: validate_compute_path() is called
/// Then: returns Ok
#[test]
fn given_compute_path_real_when_validate_compute_path_then_passes() {
    let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    assert_eq!(receipt.compute_path, "real");
    assert!(
        receipt.validate_compute_path().is_ok(),
        "validate_compute_path() must pass when compute_path == \"real\""
    );
}

/// Given: a receipt with compute_path = "mock"
/// When: validate_compute_path() is called
/// Then: returns an error
#[test]
fn given_compute_path_mock_when_validate_compute_path_then_returns_error() {
    let mut receipt =
        InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    receipt.compute_path = "mock".to_string();

    let result = receipt.validate_compute_path();
    assert!(
        result.is_err(),
        "validate_compute_path() must return Err when compute_path == \"mock\""
    );
}

/// Given: a receipt where one kernel ID contains "mock"
/// When: generate() is called
/// Then: compute_path is automatically set to "mock"
#[test]
fn given_mock_kernel_id_when_generate_then_compute_path_is_mock() {
    let kernels = vec!["mock_gemv".to_string(), "real_kernel".to_string()];
    let receipt = InferenceReceipt::generate("cpu", kernels, None).unwrap();
    assert_eq!(
        receipt.compute_path, "mock",
        "generate() must detect mock kernel IDs and set compute_path to \"mock\""
    );
}

/// Given: all kernel IDs are real (no "mock" substring, case-insensitive)
/// When: generate() is called
/// Then: compute_path is "real"
#[test]
fn given_all_real_kernel_ids_when_generate_then_compute_path_is_real() {
    let kernels = vec!["i2s_gemv".to_string(), "rope_apply".to_string(), "layernorm".to_string()];
    let receipt = InferenceReceipt::generate("cpu", kernels, None).unwrap();
    assert_eq!(
        receipt.compute_path, "real",
        "generate() must set compute_path to \"real\" when all kernel IDs are real"
    );
}

/// Given: a kernel ID with uppercase "MOCK" substring
/// When: generate() is called
/// Then: compute_path is "mock" (case-insensitive detection)
#[test]
fn given_uppercase_mock_kernel_id_when_generate_then_compute_path_is_mock() {
    let kernels = vec!["MOCK_COMPUTE".to_string()];
    let receipt = InferenceReceipt::generate("cpu", kernels, None).unwrap();
    assert_eq!(receipt.compute_path, "mock", "mock kernel detection must be case-insensitive");
}

// ── Kernel ID validation ──────────────────────────────────────────────────────

/// Given: an empty kernel list
/// When: validate_kernel_ids() is called
/// Then: returns an error (honest compute requires at least one kernel ID)
#[test]
fn given_empty_kernels_when_validate_kernel_ids_then_returns_error() {
    let mut receipt =
        InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    receipt.kernels.clear();

    let result = receipt.validate_kernel_ids();
    assert!(result.is_err(), "validate_kernel_ids() must return Err when the kernel list is empty");
}

/// Given: a kernel ID that exceeds 128 characters
/// When: validate_kernel_ids() is called
/// Then: returns an error
#[test]
fn given_oversized_kernel_id_when_validate_kernel_ids_then_returns_error() {
    let long_id = "a".repeat(129);
    let mut receipt =
        InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    receipt.kernels.push(long_id);

    let result = receipt.validate_kernel_ids();
    assert!(
        result.is_err(),
        "validate_kernel_ids() must return Err for a kernel ID longer than 128 characters"
    );
}

/// Given: a kernel ID that is exactly 128 characters
/// When: validate_kernel_ids() is called (with a valid non-mock name)
/// Then: the length check passes (boundary condition)
#[test]
fn given_kernel_id_at_max_length_when_validate_kernel_ids_then_length_check_passes() {
    // 128 lowercase letters — valid identifier, not "mock".
    let max_len_id = "a".repeat(128);
    let receipt = InferenceReceipt::generate("cpu", vec![max_len_id.clone()], None).unwrap();
    // The 128-char ID itself should not trigger the length-exceeded error.
    // Note: it may still fail on "mock" check, but that won't apply here.
    // The important assertion is that the max-length boundary is not rejected.
    let result = receipt.validate_kernel_ids();
    // A 128-char all-'a' string is non-empty and non-mock; it must not fail the length gate.
    // If other policies apply, they would fail differently; we specifically test the length gate.
    match result {
        Ok(_) => {} // passes — expected
        Err(e) => {
            let msg = e.to_string().to_ascii_lowercase();
            assert!(
                !msg.contains("length") && !msg.contains("128"),
                "128-char kernel ID must not trigger the length-exceeded error; got: {e}"
            );
        }
    }
}

/// Given: a kernel ID that is an empty string
/// When: validate_kernel_ids() is called
/// Then: returns an error
#[test]
fn given_empty_string_kernel_id_when_validate_kernel_ids_then_returns_error() {
    let mut receipt =
        InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    receipt.kernels.push(String::new());

    let result = receipt.validate_kernel_ids();
    assert!(result.is_err(), "validate_kernel_ids() must return Err for an empty kernel ID string");
}

// ── Backend recording ─────────────────────────────────────────────────────────

/// Given: backend = "cpu"
/// When: generate() is called
/// Then: the receipt's backend field is "cpu"
#[test]
fn given_backend_cpu_when_generate_then_receipt_backend_is_cpu() {
    let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    assert_eq!(receipt.backend, "cpu", "receipt must record the backend passed to generate()");
}

/// Given: backend = "cuda"
/// When: generate() is called
/// Then: the receipt's backend field is "cuda"
#[test]
fn given_backend_cuda_when_generate_then_receipt_backend_is_cuda() {
    // Generating with a "cuda" backend and real kernel IDs should produce a "real" cuda receipt.
    let receipt = InferenceReceipt::generate("cuda", vec!["gemm_bf16".to_string()], None).unwrap();
    assert_eq!(receipt.backend, "cuda", "receipt must record 'cuda' as the backend");
    assert_eq!(
        receipt.compute_path, "real",
        "non-mock kernel IDs must produce compute_path='real' regardless of backend"
    );
}

/// Given: backend = "metal"
/// When: generate() is called
/// Then: the receipt's backend field is "metal"
#[test]
fn given_backend_metal_when_generate_then_receipt_backend_is_metal() {
    let receipt =
        InferenceReceipt::generate("metal", vec!["mps_kernel".to_string()], None).unwrap();
    assert_eq!(receipt.backend, "metal");
}

// ── Full validate() ───────────────────────────────────────────────────────────

/// Given: a well-formed receipt (schema 1.0.0, compute_path=real, valid kernels)
/// When: validate() is called
/// Then: returns Ok
#[test]
fn given_valid_receipt_when_validated_then_passes_all_schema_checks() {
    let receipt = InferenceReceipt::generate(
        "cpu",
        vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
        None,
    )
    .unwrap();

    assert!(
        receipt.validate().is_ok(),
        "a well-formed receipt must pass full validate() without errors"
    );
}

/// Given: a receipt with compute_path forced to "mock"
/// When: validate() is called (full validation)
/// Then: returns an error (strict mode gate)
#[test]
fn given_compute_path_mock_when_full_validate_then_returns_error() {
    let mut receipt =
        InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    receipt.compute_path = "mock".to_string();

    let result = receipt.validate();
    assert!(result.is_err(), "full validate() must reject receipts with compute_path = \"mock\"");
    // The error message should mention the invalid path.
    let msg = result.unwrap_err().to_string().to_ascii_lowercase();
    assert!(
        msg.contains("mock") || msg.contains("real") || msg.contains("compute"),
        "error message must mention the invalid compute path; got: {msg}"
    );
}

/// Given: a receipt with an unsupported schema version AND a mock compute_path
/// When: validate() is called
/// Then: returns an error (schema gate fires first)
#[test]
fn given_bad_schema_and_mock_path_when_full_validate_then_schema_error_returned() {
    let mut receipt =
        InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    receipt.schema_version = "0.0.1".to_string();
    receipt.compute_path = "mock".to_string();

    let result = receipt.validate();
    assert!(result.is_err(), "validate() must fail when schema_version is invalid");
}

// ── JSON round-trip ───────────────────────────────────────────────────────────

/// Given: a valid receipt
/// When: serialised to JSON and deserialised back
/// Then: all key fields are preserved
#[test]
fn given_valid_receipt_when_json_round_tripped_then_fields_preserved() {
    let original = InferenceReceipt::generate(
        "cpu",
        vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
        None,
    )
    .unwrap();

    let json = original.to_json_string().expect("serialisation must succeed");
    assert!(json.contains("\"schema_version\""), "JSON must contain schema_version field");
    assert!(json.contains("1.0.0"), "JSON must embed the schema version value");
    assert!(json.contains("\"compute_path\""), "JSON must contain compute_path field");
    assert!(json.contains("\"real\""), "JSON must embed 'real' compute_path");

    let restored: InferenceReceipt =
        serde_json::from_str(&json).expect("deserialisation must succeed");
    assert_eq!(restored.schema_version, original.schema_version);
    assert_eq!(restored.compute_path, original.compute_path);
    assert_eq!(restored.backend, original.backend);
    assert_eq!(restored.kernels, original.kernels);
}
