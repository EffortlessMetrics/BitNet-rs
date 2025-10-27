//! Comprehensive test scaffolding for TokenizerAuthority integration in parity-both
//!
//! **Specification Reference**: `docs/specs/tokenizer-authority-integration-parity-both.md` (AC1-AC6)
//!
//! ## Test Coverage (6 integration tests)
//!
//! This test suite validates TokenizerAuthority SHA256 integration in the dual-lane
//! cross-validation workflow, ensuring both Rust and C++ lanes use identical tokenizer
//! configurations:
//!
//! - **AC1**: Hash Computation Once Before Dual-Lane Execution (1 test)
//!   - compute_tokenizer_file_hash() called once in shared setup
//!   - Tokenizer authority computed before lane A and lane B execution
//!   - Same TokenizerAuthority instance passed to both lanes
//!
//! - **AC2**: Both Receipts Have Matching TokenizerAuthority.sha256_hash (1 test)
//!   - set_tokenizer_authority() called in run_single_lane() before finalize()
//!   - Lane A and Lane B receipts have tokenizer_authority populated
//!   - Both receipts have identical file_hash and config_hash values
//!
//! - **AC3**: Validate Tokenizer Consistency After Both Lanes Complete (1 test)
//!   - validate_tokenizer_consistency() called after loading both receipts
//!   - Validation checks config_hash match between lanes
//!   - Validation checks token_count match between lanes
//!
//! - **AC4**: Exit Code 2 if Tokenizer Hashes Differ (1 test)
//!   - validate_tokenizer_consistency() failure triggers exit code 2
//!   - Exit code 2 is distinct from parity failure exit code 1
//!   - Error message displays both lane config hashes
//!
//! - **AC5**: Summary Output Includes Tokenizer Hash (1 test)
//!   - Text format summary has "Tokenizer Consistency" section
//!   - JSON format has tokenizer.config_hash field
//!   - Summary shows abbreviated and full hash values
//!
//! - **AC6**: Receipts Serialize TokenizerAuthority Properly (1 test)
//!   - Receipt serialization includes tokenizer_authority field
//!   - Deserialization preserves tokenizer_authority data
//!   - infer_version() returns "2.0.0" when populated
//!   - Schema evolution is backward-compatible
//!
//! ## Test Helpers
//!
//! - `mock_tokenizer_authority()` - Create mock TokenizerAuthority for testing
//! - `mock_parity_receipt_with_authority()` - Create mock receipt with tokenizer authority
//! - `create_temp_tokenizer_file()` - Create temporary tokenizer.json for hash computation
//! - `assert_tokenizer_consistency()` - Verify tokenizer authority across lanes
//!
//! ## Environment Isolation
//!
//! All tests use `#[serial(bitnet_env)]` to prevent race conditions during parallel
//! test execution when accessing file system resources and environment state.

#![cfg(feature = "crossval-all")]

use serial_test::serial;
use std::fs;
use std::path::{Path, PathBuf};

// ============================================================================
// Mock Data Structures (mirrors crossval crate types)
// ============================================================================

/// Mock TokenizerSource enum for testing
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum MockTokenizerSource {
    /// Tokenizer embedded in GGUF file
    GgufEmbedded,
    /// External tokenizer.json file
    External,
    /// Auto-discovered via path resolution
    AutoDiscovered,
}

/// Mock TokenizerAuthority for testing receipt schema
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
struct MockTokenizerAuthority {
    /// Tokenizer source: GgufEmbedded, External, or AutoDiscovered
    source: MockTokenizerSource,

    /// Path to tokenizer (GGUF path or tokenizer.json path)
    path: String,

    /// SHA256 hash of tokenizer.json file (if external)
    #[serde(skip_serializing_if = "Option::is_none")]
    file_hash: Option<String>,

    /// SHA256 hash of effective tokenizer config (canonical JSON)
    config_hash: String,

    /// Token count (for quick validation)
    token_count: usize,
}

/// Mock ParityReceipt v2 with tokenizer authority
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct MockParityReceipt {
    version: u32,
    timestamp: String,
    model: String,
    backend: String,
    prompt: String,
    positions: usize,

    // v2 fields (optional for backward compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenizer_authority: Option<MockTokenizerAuthority>,

    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_template: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    model_sha256: Option<String>,
}

// ============================================================================
// Test Helpers
// ============================================================================

/// Helper to create mock TokenizerAuthority
fn mock_tokenizer_authority(source: MockTokenizerSource) -> MockTokenizerAuthority {
    MockTokenizerAuthority {
        source,
        path: "tests/fixtures/tokenizer.json".to_string(),
        file_hash: match source {
            MockTokenizerSource::External => {
                Some("a3f7b8c9d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8".to_string())
            }
            _ => None,
        },
        config_hash: "e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8".to_string(),
        token_count: 8,
    }
}

/// Helper to create mock ParityReceipt with tokenizer authority
fn mock_parity_receipt_with_authority(backend: &str) -> MockParityReceipt {
    MockParityReceipt {
        version: 1,
        timestamp: "2025-10-27T10:30:00Z".to_string(),
        model: "models/model.gguf".to_string(),
        backend: backend.to_string(),
        prompt: "What is 2+2?".to_string(),
        positions: 4,
        tokenizer_authority: Some(mock_tokenizer_authority(MockTokenizerSource::External)),
        prompt_template: Some("instruct".to_string()),
        model_sha256: Some(
            "fedcba98765432109876543210fedcba98765432109876543210fedcba9876".to_string(),
        ),
    }
}

/// Helper to create temporary tokenizer.json file for hash computation tests
fn create_temp_tokenizer_file(dir: &Path, content: &str) -> PathBuf {
    let path = dir.join("tokenizer.json");
    fs::write(&path, content).expect("Failed to write temp tokenizer file");
    path
}

/// Helper to assert tokenizer authority consistency across lanes
fn assert_tokenizer_consistency(lane_a: &MockTokenizerAuthority, lane_b: &MockTokenizerAuthority) {
    assert_eq!(
        lane_a.config_hash, lane_b.config_hash,
        "Tokenizer config hash must match across lanes"
    );
    assert_eq!(lane_a.token_count, lane_b.token_count, "Token count must match across lanes");

    // If both have file_hash, they should match
    if let (Some(hash_a), Some(hash_b)) = (&lane_a.file_hash, &lane_b.file_hash) {
        assert_eq!(hash_a, hash_b, "File hash must match across lanes");
    }
}

// ============================================================================
// AC1: Hash Computation Once Before Dual-Lane Execution (1 test)
// ============================================================================

/// AC1: Test hash computation called once before dual-lane execution
/// Tests feature spec: tokenizer-authority-integration-parity-both.md#AC1
///
/// **Requirements**:
/// - compute_tokenizer_file_hash() called in shared setup phase (after line 471)
/// - Hash computation happens **before** lane A and lane B execution
/// - Same TokenizerAuthority instance passed to both lanes
/// - Verbose output shows "Computing tokenizer authority" message
///
/// **Implementation Status**: Red (TDD)
/// - Integration point: xtask/src/crossval/parity_both.rs::run_dual_lanes_and_summarize()
/// - Need to add: TokenizerAuthority computation after Rust tokenization (line 471)
/// - Need to pass: tokenizer_authority to both run_single_lane() calls
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "crossval-all")]
fn test_hash_computed_once_before_dual_lanes() {
    // AC1: compute_tokenizer_file_hash() called once before dual-lane execution
    //
    // This test validates that:
    // 1. TokenizerAuthority is computed in shared setup (not per-lane)
    // 2. Hash computation happens before any lane execution
    // 3. Same authority instance is reused for both lanes
    //
    // Expected behavior (after implementation):
    // - run_dual_lanes_and_summarize() computes authority once after Rust tokenization
    // - Authority contains file_hash, config_hash, token_count
    // - Both lanes receive identical authority via run_single_lane() parameter
    // - Verbose mode prints "Computing tokenizer authority" message

    todo!(
        "Implement after integration points are wired in parity_both.rs:\n\
         1. Add TokenizerAuthority computation in run_dual_lanes_and_summarize() after line 471\n\
         2. Add tokenizer_authority parameter to run_single_lane() signature\n\
         3. Pass authority to both lane A and lane B calls\n\
         \n\
         Test Strategy:\n\
         - Create temp tokenizer.json file\n\
         - Mock run_dual_lanes_and_summarize() to capture authority computation\n\
         - Verify compute_tokenizer_file_hash() called exactly once\n\
         - Verify authority passed to both lanes with same reference"
    );
}

// ============================================================================
// AC2: Both Receipts Have Matching TokenizerAuthority.sha256_hash (1 test)
// ============================================================================

/// AC2: Test receipts have matching TokenizerAuthority
/// Tests feature spec: tokenizer-authority-integration-parity-both.md#AC2
///
/// **Requirements**:
/// - set_tokenizer_authority() called in run_single_lane() before finalize()
/// - Lane A receipt has tokenizer_authority.file_hash populated
/// - Lane B receipt has tokenizer_authority.file_hash populated
/// - Both receipts have identical file_hash and config_hash values
///
/// **Implementation Status**: Red (TDD)
/// - Integration point: xtask/src/crossval/parity_both.rs::run_single_lane()
/// - Need to add: receipt.set_tokenizer_authority(tokenizer_authority.clone()) before line 682
/// - Need to update: run_single_lane() function signature to accept tokenizer_authority param
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "crossval-all")]
fn test_receipts_have_matching_authority() {
    // AC2: Both receipts have matching TokenizerAuthority.sha256_hash
    //
    // This test validates that:
    // 1. set_tokenizer_authority() is called before receipt finalization
    // 2. Lane A receipt contains populated tokenizer_authority
    // 3. Lane B receipt contains populated tokenizer_authority
    // 4. Both receipts have matching file_hash and config_hash
    //
    // Expected behavior (after implementation):
    // - run_single_lane() receives tokenizer_authority parameter
    // - Calls receipt.set_tokenizer_authority() before receipt.finalize()
    // - Both lanes produce receipts with identical authority data
    // - Receipt JSON includes tokenizer_authority field with all hashes

    todo!(
        "Implement after set_tokenizer_authority() is wired:\n\
         1. Update run_single_lane() signature to accept &TokenizerAuthority\n\
         2. Call receipt.set_tokenizer_authority(tokenizer_authority.clone()) before finalize()\n\
         3. Update lane A and lane B call sites with authority argument\n\
         \n\
         Test Strategy:\n\
         - Run parity-both with test model and tokenizer\n\
         - Load both receipt_bitnet.json and receipt_llama.json\n\
         - Parse receipts and extract tokenizer_authority\n\
         - Assert both have Some(authority) with matching hashes\n\
         - Verify file_hash is 64-char SHA256 hex string"
    );
}

// ============================================================================
// AC3: Validate Tokenizer Consistency After Both Lanes Complete (1 test)
// ============================================================================

/// AC3: Test validate_tokenizer_consistency() called after lanes complete
/// Tests feature spec: tokenizer-authority-integration-parity-both.md#AC3
///
/// **Requirements**:
/// - validate_tokenizer_consistency() called after loading both receipts (after line 533)
/// - Validation checks config_hash match between lanes
/// - Validation checks token_count match between lanes
/// - Verbose output shows "Validating tokenizer consistency" message
///
/// **Implementation Status**: Red (TDD)
/// - Integration point: xtask/src/crossval/parity_both.rs::run_dual_lanes_and_summarize()
/// - Need to add: Cross-lane validation after line 533 (after loading receipts)
/// - Need to call: bitnet_crossval::receipt::validate_tokenizer_consistency()
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "crossval-all")]
fn test_validate_tokenizer_consistency_called() {
    // AC3: validate_tokenizer_consistency() called after both lanes complete
    //
    // This test validates that:
    // 1. validate_tokenizer_consistency() is called after dual-lane execution
    // 2. Validation checks config_hash equality
    // 3. Validation checks token_count equality
    // 4. Verbose mode prints "Validating tokenizer consistency" message
    //
    // Expected behavior (after implementation):
    // - run_dual_lanes_and_summarize() loads both receipts after lanes complete
    // - Extracts tokenizer_authority from both receipts
    // - Calls validate_tokenizer_consistency(auth_a, auth_b)
    // - Continues to summary if validation passes
    // - Fails with exit code 2 if validation fails

    todo!(
        "Implement after cross-lane validation is wired:\n\
         1. Add receipt loading after line 533 in run_dual_lanes_and_summarize()\n\
         2. Extract tokenizer_authority from both receipts\n\
         3. Call validate_tokenizer_consistency(auth_a, auth_b)\n\
         4. Add verbose output before validation\n\
         \n\
         Test Strategy:\n\
         - Run parity-both with same tokenizer for both lanes\n\
         - Capture verbose output\n\
         - Verify 'Validating tokenizer consistency' appears in output\n\
         - Verify exit code 0 when consistency passes\n\
         - Mock receipts with matching authority to test validation logic"
    );
}

// ============================================================================
// AC4: Exit Code 2 if Tokenizer Hashes Differ (1 test)
// ============================================================================

/// AC4: Test exit code 2 on tokenizer hash mismatch
/// Tests feature spec: tokenizer-authority-integration-parity-both.md#AC4
///
/// **Requirements**:
/// - validate_tokenizer_consistency() failure triggers std::process::exit(2)
/// - Exit code is **2** (distinct from parity failure exit code 1)
/// - Error message displays both lane config hashes
/// - Error message clearly identifies tokenizer mismatch
///
/// **Implementation Status**: Red (TDD)
/// - Integration point: xtask/src/crossval/parity_both.rs::run_dual_lanes_and_summarize()
/// - Need to add: Error handling for validate_tokenizer_consistency() failure
/// - Need to emit: Clear error message with both hashes before exit(2)
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "crossval-all")]
fn test_exit_code_2_on_hash_mismatch() {
    // AC4: Exit code 2 if tokenizer hashes differ
    //
    // This test validates that:
    // 1. validate_tokenizer_consistency() failure causes exit code 2
    // 2. Exit code 2 is distinct from parity failure (exit code 1)
    // 3. Error message shows both lane config_hash values
    // 4. Error message identifies the mismatch clearly
    //
    // Expected behavior (after implementation):
    // - validate_tokenizer_consistency() returns Err on mismatch
    // - Error handler prints diagnostic message with both hashes
    // - Process exits with std::process::exit(2)
    // - Error format: "Tokenizer consistency validation failed"

    todo!(
        "Implement after error handling is wired:\n\
         1. Add error handling for validate_tokenizer_consistency() in run_dual_lanes_and_summarize()\n\
         2. On Err, print diagnostic message with auth_a and auth_b hashes\n\
         3. Call std::process::exit(2) for tokenizer mismatch\n\
         4. Ensure exit code 2 is distinct from parity failure (exit code 1)\n\
         \n\
         Test Strategy:\n\
         - Create two receipts with different tokenizer config_hash\n\
         - Mock validate_tokenizer_consistency() to return Err\n\
         - Run parity-both and capture exit code\n\
         - Assert exit code == 2 (not 0 or 1)\n\
         - Capture stderr and verify error message shows both hashes"
    );
}

// ============================================================================
// AC5: Summary Output Includes Tokenizer Hash (1 test)
// ============================================================================

/// AC5: Test summary output includes tokenizer hash
/// Tests feature spec: tokenizer-authority-integration-parity-both.md#AC5
///
/// **Requirements**:
/// - Text format summary has "Tokenizer Consistency" section
/// - Text format shows first 32 chars of config hash
/// - Text format shows full hash (64 chars)
/// - JSON format has tokenizer.config_hash field
/// - JSON format has tokenizer.status: "consistent" field
///
/// **Implementation Status**: Red (TDD)
/// - Integration point: xtask/src/crossval/parity_both.rs::print_unified_summary()
/// - Need to update: Function signature to accept tokenizer_hash: Option<&str>
/// - Need to add: "Tokenizer Consistency" section in text format (after line 272)
/// - Need to add: tokenizer.config_hash field in JSON format (in print_json_summary)
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "crossval-all")]
fn test_summary_includes_tokenizer_hash() {
    // AC5: Summary output includes tokenizer hash
    //
    // This test validates that:
    // 1. Text format has "Tokenizer Consistency" section
    // 2. Text format shows abbreviated (32 chars) and full (64 chars) hash
    // 3. JSON format has tokenizer.config_hash field
    // 4. JSON format indicates consistency status
    //
    // Expected behavior (after implementation):
    // - print_unified_summary() receives tokenizer_hash parameter
    // - Text output includes:
    //   "Tokenizer Consistency"
    //   "Config hash:      <first 32 chars>"
    //   "Full hash:        <64 chars>"
    // - JSON output includes:
    //   "tokenizer": {
    //     "config_hash": "<64 chars>",
    //     "status": "consistent"
    //   }

    todo!(
        "Implement after summary enhancements are wired:\n\
         1. Update print_unified_summary() signature with tokenizer_hash: Option<&str>\n\
         2. Add 'Tokenizer Consistency' section in text format (after line 272)\n\
         3. Add tokenizer object in JSON format (in print_json_summary)\n\
         4. Update call site at line 535 to extract and pass tokenizer_hash\n\
         \n\
         Test Strategy:\n\
         - Run parity-both with --format text\n\
         - Capture stdout and verify 'Tokenizer Consistency' section exists\n\
         - Verify config_hash displayed (both abbreviated and full)\n\
         - Run parity-both with --format json\n\
         - Parse JSON output and verify 'tokenizer.config_hash' field\n\
         - Verify 'tokenizer.status' == 'consistent'"
    );
}

// ============================================================================
// AC6: Receipts Serialize TokenizerAuthority Properly (1 test)
// ============================================================================

/// AC6: Test receipt serialization with TokenizerAuthority
/// Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
///
/// **Requirements**:
/// - Receipt serialization includes tokenizer_authority field
/// - Deserialized receipt preserves tokenizer_authority data
/// - infer_version() returns "2.0.0" when tokenizer_authority is Some(...)
/// - Schema evolution is backward-compatible (v1 receipts still valid)
///
/// **Implementation Status**: Red (TDD)
/// - Schema already exists in crossval/src/receipt.rs (lines 88-137)
/// - tokenizer_authority field is Option<TokenizerAuthority> (backward-compatible)
/// - Need to verify: Serialization/deserialization round-trip preserves data
/// - Need to verify: infer_version() logic correctly detects v2.0.0
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "crossval-all")]
fn test_receipt_serialization_with_tokenizer_authority() {
    // AC6: Receipts serialize TokenizerAuthority properly
    //
    // This test validates that:
    // 1. Receipt with tokenizer_authority serializes to JSON correctly
    // 2. Deserialization from JSON preserves all authority fields
    // 3. infer_version() returns "2.0.0" for receipts with authority
    // 4. Schema is backward-compatible (v1 receipts without authority still work)
    //
    // Expected behavior:
    // - Receipt with Some(TokenizerAuthority) serializes with all fields
    // - JSON includes: source, path, file_hash, config_hash, token_count
    // - Deserialize + re-serialize produces identical JSON
    // - infer_version() detects v2.0.0 when tokenizer_authority is present

    // Test serialization/deserialization round-trip
    let receipt = mock_parity_receipt_with_authority("bitnet");

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&receipt).expect("Failed to serialize receipt to JSON");

    // Verify JSON contains tokenizer_authority fields
    assert!(json.contains("tokenizer_authority"), "JSON should contain tokenizer_authority field");
    assert!(json.contains("file_hash"), "JSON should contain file_hash field");
    assert!(json.contains("config_hash"), "JSON should contain config_hash field");
    assert!(json.contains("token_count"), "JSON should contain token_count field");

    // Deserialize back
    let deserialized: MockParityReceipt =
        serde_json::from_str(&json).expect("Failed to deserialize receipt from JSON");

    // Verify tokenizer_authority preserved
    assert!(
        deserialized.tokenizer_authority.is_some(),
        "Deserialized receipt should have tokenizer_authority"
    );

    let authority = deserialized.tokenizer_authority.unwrap();
    assert_eq!(authority.config_hash.len(), 64, "Config hash should be 64-char SHA256");
    assert_eq!(authority.token_count, 8, "Token count should be preserved");

    // Test backward compatibility (v1 receipt without authority)
    let v1_receipt = MockParityReceipt {
        version: 1,
        timestamp: "2025-10-27T10:30:00Z".to_string(),
        model: "models/model.gguf".to_string(),
        backend: "bitnet".to_string(),
        prompt: "What is 2+2?".to_string(),
        positions: 4,
        tokenizer_authority: None, // v1: no authority
        prompt_template: None,
        model_sha256: None,
    };

    let v1_json =
        serde_json::to_string_pretty(&v1_receipt).expect("Failed to serialize v1 receipt");

    // v1 JSON should NOT contain tokenizer_authority (skip_serializing_if)
    assert!(
        !v1_json.contains("tokenizer_authority"),
        "v1 receipt JSON should omit tokenizer_authority field"
    );

    // Deserialize v1 receipt should work
    let v1_deserialized: MockParityReceipt =
        serde_json::from_str(&v1_json).expect("Failed to deserialize v1 receipt");

    assert!(
        v1_deserialized.tokenizer_authority.is_none(),
        "v1 receipt should have None tokenizer_authority"
    );

    todo!(
        "Extend test after integration:\n\
         1. Verify infer_version() returns '2.0.0' for receipts with authority\n\
         2. Verify infer_version() returns '1.0.0' for receipts without authority\n\
         3. Test with real ParityReceipt from crossval crate (not mock)\n\
         4. Verify round-trip with actual parity-both generated receipts\n\
         \n\
         Current Status:\n\
         - Mock receipt serialization: ✓ PASSING\n\
         - Backward compatibility: ✓ PASSING\n\
         - Need integration: infer_version() test (requires real ParityReceipt)"
    );
}

// ============================================================================
// Additional Helper Tests (for comprehensive coverage)
// ============================================================================

/// Helper test: Verify TokenizerAuthority hash format validation
/// This test ensures hashes are 64-char lowercase hex strings (SHA256)
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "crossval-all")]
fn test_tokenizer_authority_hash_format() {
    let authority = mock_tokenizer_authority(MockTokenizerSource::External);

    // Verify file_hash format (if present)
    if let Some(file_hash) = &authority.file_hash {
        assert_eq!(file_hash.len(), 64, "File hash should be 64-char SHA256");
        assert!(file_hash.chars().all(|c| c.is_ascii_hexdigit()), "File hash should be hex string");
    }

    // Verify config_hash format
    assert_eq!(authority.config_hash.len(), 64, "Config hash should be 64-char SHA256");
    assert!(
        authority.config_hash.chars().all(|c| c.is_ascii_hexdigit()),
        "Config hash should be hex string"
    );
}

/// Helper test: Verify mock consistency helper works correctly
#[test]
#[serial(bitnet_env)]
#[cfg(feature = "crossval-all")]
fn test_assert_tokenizer_consistency_helper() {
    let auth_a = mock_tokenizer_authority(MockTokenizerSource::External);
    let auth_b = mock_tokenizer_authority(MockTokenizerSource::External);

    // Should not panic - identical authorities
    assert_tokenizer_consistency(&auth_a, &auth_b);

    // Test mismatch detection (should panic)
    let mut auth_c = mock_tokenizer_authority(MockTokenizerSource::External);
    auth_c.config_hash =
        "different_hash_0000000000000000000000000000000000000000000000000000".to_string();

    let result = std::panic::catch_unwind(|| {
        assert_tokenizer_consistency(&auth_a, &auth_c);
    });

    assert!(result.is_err(), "assert_tokenizer_consistency should panic on mismatch");
}
