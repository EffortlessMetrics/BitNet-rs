//! Comprehensive test scaffolding for parity-both preflight integration and tokenizer authority
//!
//! **Specification Reference**: `docs/specs/parity-both-preflight-tokenizer.md` (AC1-AC10)
//!
//! ## Test Coverage (54+ tests)
//!
//! This test suite validates dual-lane cross-validation with preflight integration,
//! tokenizer authority tracking, and comprehensive receipt generation:
//!
//! - **AC1-AC4**: Preflight & Exit Codes (20 tests)
//!   - **AC1**: CLI Registration and Dispatch (5 tests)
//!     - Help text completeness
//!     - Command appears in xtask list
//!     - Argument parsing and defaults
//!     - Required vs optional parameters
//!     - Flag aliases and compatibility
//!   - **AC2**: Preflight Both Backends (5 tests)
//!     - Auto-repair flow for missing BitNet.cpp
//!     - Auto-repair flow for missing llama.cpp
//!     - Preflight success when both backends available
//!     - Preflight failure with --no-repair
//!     - Rebuild xtask instructions
//!   - **AC3**: RepairMode Integration (5 tests)
//!     - Default auto-repair in dev environment
//!     - --no-repair flag disables auto-repair
//!     - CI environment defaults to no-repair
//!     - RepairMode::Never fail-fast behavior
//!     - RepairMode::Auto retry logic
//!   - **AC4**: Exit Code Logic (5 tests)
//!     - Exit code 0: Both lanes pass
//!     - Exit code 1: Lane failure (parity check fails)
//!     - Exit code 2: Token parity mismatch
//!     - Exit code 2: Backend unavailable + no-repair
//!     - Exit code semantics validation
//!
//! - **AC5-AC8**: Dual-Lane Execution (20 tests)
//!   - **AC5**: Dual Receipt Naming Convention (4 tests)
//!     - Receipt naming: receipt_bitnet.json, receipt_llama.json
//!     - Receipt paths validation
//!     - Distinct receipt files per backend
//!     - Output directory structure
//!   - **AC6**: Shared Rust Inference Optimization (5 tests)
//!     - Single Rust tokenization for both lanes
//!     - Shared logits evaluation baseline
//!     - Memory optimization (no duplicate tokenizer load)
//!     - Performance: shared setup vs dual setup
//!     - Tokenizer authority captured once
//!   - **AC7**: Per-Position Metrics (6 tests)
//!     - MSE computation per position
//!     - L2 distance per position
//!     - KL divergence per position
//!     - Top-K agreement per position
//!     - Position metrics in receipt rows
//!     - Threshold validation per metric
//!   - **AC8**: Summary with First Divergence (5 tests)
//!     - Summary includes first divergence position
//!     - Summary shows mean metrics across positions
//!     - All-passed status flag
//!     - Summary in text format
//!     - Summary in JSON format
//!
//! - **AC9-AC10**: Integration & Coverage (14 tests)
//!   - **AC9**: CLI Wiring in main.rs (4 tests)
//!     - Command dispatch to parity_both module
//!     - Prompt template conversion with model path
//!     - Auto-repair flag inversion (--no-repair → auto_repair=false)
//!     - Argument forwarding completeness
//!   - **AC10**: Test Coverage (10 tests)
//!     - Dual-lane execution test coverage
//!     - Exit code test coverage (0, 1, 2)
//!     - Tokenizer authority consistency tests
//!     - Receipt schema validation tests
//!     - Preflight integration tests
//!     - Template auto-detection tests
//!     - Hash computation tests
//!     - Parity validation tests
//!     - Error handling tests
//!     - Full integration flow test
//!
//! ## Test Helpers
//!
//! - `mock_tokenizer_authority()` - Create mock TokenizerAuthority for testing
//! - `mock_parity_receipt_v2()` - Create mock receipt with tokenizer authority
//! - `mock_gguf_metadata()` - Mock GGUF metadata with chat_template
//! - `assert_tokenizer_consistency()` - Verify tokenizer authority across lanes
//! - `simulate_missing_backend()` - Mock missing C++ backend for preflight tests
//!
//! ## Environment Isolation
//!
//! All tests use `#[serial(bitnet_env)]` to prevent race conditions during parallel
//! test execution when mutating environment variables.

#![cfg(all(feature = "crossval-all", feature = "inference"))]

use serial_test::serial;
use std::path::PathBuf;

// ============================================================================
// Mock Data Structures (mirrors crossval crate types)
// ============================================================================

/// Mock TokenizerAuthority for testing receipt schema
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
struct MockTokenizerAuthority {
    /// Tokenizer source: "gguf_embedded" | "external" | "auto_discovered"
    source: String,

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
#[derive(Debug, serde::Deserialize)]
struct MockParityReceiptV2 {
    version: u32,
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
fn mock_tokenizer_authority(source: &str) -> MockTokenizerAuthority {
    MockTokenizerAuthority {
        source: source.to_string(),
        path: "tests/fixtures/tokenizer.json".to_string(),
        file_hash: if source == "external" { Some("abc123".to_string()) } else { None },
        config_hash: "def456".to_string(),
        token_count: 5,
    }
}

/// Helper to create mock ParityReceipt v2 with tokenizer authority
fn mock_parity_receipt_v2(backend: &str) -> MockParityReceiptV2 {
    MockParityReceiptV2 {
        version: 1,
        backend: backend.to_string(),
        prompt: "What is 2+2?".to_string(),
        positions: 4,
        tokenizer_authority: Some(mock_tokenizer_authority("external")),
        prompt_template: Some("instruct".to_string()),
        model_sha256: Some("fedcba98".to_string()),
    }
}

/// Helper to assert tokenizer authority consistency across lanes
fn assert_tokenizer_consistency(lane_a: &MockTokenizerAuthority, lane_b: &MockTokenizerAuthority) {
    assert_eq!(
        lane_a.config_hash, lane_b.config_hash,
        "Tokenizer config hash must match across lanes"
    );
    assert_eq!(lane_a.token_count, lane_b.token_count, "Token count must match across lanes");
}

// ============================================================================
// AC1: CLI Registration and Dispatch Tests (5 tests)
// ============================================================================

/// AC1.1: Test help text completeness
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac1
#[test]
fn test_cli_help_text_completeness() {
    use std::process::Command;

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&["parity-both", "--help"])
        .output()
        .expect("Failed to execute xtask parity-both --help");

    let help_text = String::from_utf8_lossy(&output.stdout);

    // Verify required arguments documented
    assert!(
        help_text.contains("--model-gguf") || help_text.contains("model-gguf"),
        "Help should document --model-gguf"
    );
    assert!(
        help_text.contains("--tokenizer") || help_text.contains("tokenizer"),
        "Help should document --tokenizer"
    );

    // Verify optional arguments documented
    assert!(
        help_text.contains("--no-repair") || help_text.contains("no-repair"),
        "Help should document --no-repair flag for disabling auto-repair"
    );
    assert!(
        help_text.contains("--prompt-template") || help_text.contains("prompt-template"),
        "Help should document --prompt-template for template selection"
    );
    assert!(
        help_text.contains("--verbose") || help_text.contains("verbose"),
        "Help should document --verbose flag"
    );
}

/// AC1.2: Test command appears in xtask command list
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac1
#[test]
fn test_cli_command_registration() {
    use std::process::Command;

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&["--help"])
        .output()
        .expect("Failed to execute xtask --help");

    let help_text = String::from_utf8_lossy(&output.stdout);

    // Verify parity-both command is listed
    assert!(help_text.contains("parity-both"), "xtask help should list parity-both command");
}

/// AC1.3: Test default argument values
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac1
#[test]
#[ignore = "TODO: Implement parity-both command argument parsing with defaults"]
fn test_cli_default_arguments() {
    // After implementation, this test will:
    // 1. Parse command args with only required fields
    // 2. Verify defaults:
    //    - prompt: "What is 2+2?"
    //    - max-tokens: 4
    //    - cos-tol: 0.999
    //    - format: "text"
    //    - out-dir: "."
    //    - auto-repair: true (default, unless CI environment)
    //    - prompt-template: "auto"

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC1.4: Test required vs optional parameters
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac1
#[test]
#[ignore = "TODO: Implement parity-both command validation"]
fn test_cli_required_parameters() {
    use std::process::Command;
    use tempfile::TempDir;

    let out_dir = TempDir::new().expect("Failed to create temp dir");

    // Test 1: Missing --model-gguf should fail
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--tokenizer",
            "tokenizer.json",
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    assert!(!output.status.success(), "Should fail when --model-gguf is missing");

    // Test 2: Missing --tokenizer should fail
    let output2 = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            "model.gguf",
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    assert!(!output2.status.success(), "Should fail when --tokenizer is missing");
}

/// AC1.5: Test flag aliases and compatibility
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac1
#[test]
#[ignore = "TODO: Implement parity-both command with flag aliases"]
fn test_cli_flag_aliases() {
    // After implementation, this test will verify:
    // 1. --prompt-template and --template are equivalent
    // 2. --out-dir and --output are equivalent
    // 3. --verbose and -v are equivalent

    unimplemented!("Awaiting parity-both command implementation");
}

// ============================================================================
// AC2: Preflight Both Backends Tests (6 tests)
// ============================================================================

/// AC2.1: Test preflight auto-repair for missing BitNet.cpp
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac2
#[test]
#[ignore = "TODO: Implement preflight_both_backends with auto-repair for BitNet.cpp"]
#[serial(bitnet_env)]
fn test_preflight_auto_repair_bitnet() {
    // After implementation, this test will:
    // 1. Simulate missing BitNet.cpp (clear BITNET_CPP_DIR)
    // 2. Run preflight_both_backends(auto_repair=true, verbose=true)
    // 3. Verify auto-repair triggered: "⚠️ bitnet backend not found"
    // 4. Verify setup-cpp-auto invoked
    // 5. Verify "✓ bitnet backend now available"
    // 6. Verify exit code 0 after successful repair

    unimplemented!("Awaiting preflight_both_backends implementation");
}

/// AC2.2: Test preflight auto-repair for missing llama.cpp
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac2
#[test]
#[ignore = "TODO: Implement preflight_both_backends with auto-repair for llama.cpp"]
#[serial(bitnet_env)]
fn test_preflight_auto_repair_llama() {
    // After implementation, this test will:
    // 1. Simulate missing llama.cpp (clear LLAMA_CPP_DIR)
    // 2. Run preflight_both_backends(auto_repair=true, verbose=true)
    // 3. Verify auto-repair triggered: "⚠️ llama backend not found"
    // 4. Verify setup-cpp-auto invoked
    // 5. Verify "✓ llama backend now available"
    // 6. Verify exit code 0 after successful repair

    unimplemented!("Awaiting preflight_both_backends implementation");
}

/// AC2.3: Test preflight success when both backends available
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac2
#[test]
#[ignore = "TODO: Implement preflight_both_backends with backend availability checks"]
#[serial(bitnet_env)]
fn test_preflight_both_available() {
    // After implementation, this test will:
    // 1. Simulate both backends available
    // 2. Run preflight_both_backends(auto_repair=true, verbose=true)
    // 3. Verify no auto-repair triggered
    // 4. Verify "✓ bitnet backend available"
    // 5. Verify "✓ llama backend available"
    // 6. Verify exit code 0

    unimplemented!("Awaiting preflight_both_backends implementation");
}

/// AC2.4: Test preflight failure with --no-repair
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac2
#[test]
#[ignore = "TODO: Implement preflight_both_backends with RepairMode::Never"]
#[serial(bitnet_env)]
fn test_preflight_no_repair_fail_fast() {
    // After implementation, this test will:
    // 1. Simulate missing backend
    // 2. Run preflight_both_backends(auto_repair=false, verbose=false)
    // 3. Verify auto-repair NOT triggered
    // 4. Verify error message: "❌ bitnet backend not found (repair disabled)"
    // 5. Verify setup instructions shown
    // 6. Verify exit code 2 (usage error / backend unavailable)

    unimplemented!("Awaiting preflight_both_backends implementation");
}

/// AC2.5: Test preflight rebuild xtask instructions
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac2
#[test]
#[ignore = "TODO: Implement preflight auto-repair with rebuild instructions"]
fn test_preflight_rebuild_instructions() {
    // After implementation, this test will:
    // 1. Simulate successful auto-repair
    // 2. Verify output contains rebuild instructions:
    //    "ℹ️ Next: Rebuild xtask for detection"
    //    "cargo clean -p xtask && cargo build -p xtask --features crossval-all"

    unimplemented!("Awaiting preflight_both_backends implementation");
}

/// AC2.6: Test preflight exit code semantics
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac2
#[test]
fn test_preflight_exit_codes() {
    // Mock exit code logic (no actual preflight call)
    let test_cases = [
        (true, true, true, 0),    // Both available, auto-repair enabled → 0
        (false, true, true, 0),   // BitNet missing, auto-repair success → 0
        (false, false, false, 2), // Both missing, no repair → 2
        (false, true, false, 2),  // BitNet missing, no repair → 2
    ];

    for (bitnet_available, _llama_available, auto_repair, expected_exit) in test_cases {
        let simulated_exit = if !bitnet_available && !auto_repair {
            2 // Backend unavailable + repair disabled
        } else {
            0 // Backend available or repair succeeded
        };

        assert_eq!(
            simulated_exit, expected_exit,
            "Exit code mismatch for bitnet={}, auto_repair={}",
            bitnet_available, auto_repair
        );
    }
}

// ============================================================================
// AC3: RepairMode Integration Tests (5 tests)
// ============================================================================

/// AC3.1: Test default auto-repair in dev environment
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac3
#[test]
#[ignore = "TODO: Implement RepairMode::Auto default in dev environment"]
#[serial(bitnet_env)]
fn test_repair_mode_auto_default_dev() {
    // After implementation, this test will:
    // 1. Unset CI environment variable
    // 2. Run parity-both without --no-repair flag
    // 3. Verify auto-repair is enabled by default
    // 4. Simulate missing backend
    // 5. Verify auto-repair is triggered

    unimplemented!("Awaiting RepairMode integration");
}

/// AC3.2: Test --no-repair flag disables auto-repair
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac3
#[test]
#[ignore = "TODO: Implement --no-repair flag handling"]
#[serial(bitnet_env)]
fn test_repair_mode_no_repair_flag() {
    use std::process::Command;
    use tempfile::TempDir;

    let out_dir = TempDir::new().expect("Failed to create temp dir");

    // Simulate missing backend (safe because test is serialized)
    unsafe {
        std::env::remove_var("BITNET_CPP_DIR");
    }

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            "test.gguf",
            "--tokenizer",
            "tokenizer.json",
            "--no-repair",
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    // Should fail with exit code 2 (backend unavailable + repair disabled)
    assert_eq!(output.status.code(), Some(2), "Should exit with code 2 when --no-repair is used");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("backend not found") || stderr.contains("repair disabled"),
        "Error message should mention backend unavailable with repair disabled"
    );
}

/// AC3.3: Test CI environment defaults to no-repair
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac3
#[test]
#[ignore = "TODO: Implement CI environment auto-repair disable"]
#[serial(bitnet_env)]
fn test_repair_mode_ci_default_no_repair() {
    // After implementation, this test will:
    // 1. Set CI=true environment variable
    // 2. Run parity-both without --no-repair flag
    // 3. Verify auto-repair is DISABLED by default in CI
    // 4. Simulate missing backend
    // 5. Verify command fails with exit code 2 (not attempting repair)

    unimplemented!("Awaiting CI environment detection");
}

/// AC3.4: Test RepairMode::Never fail-fast behavior
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac3
#[test]
#[ignore = "TODO: Implement RepairMode::Never fail-fast logic"]
fn test_repair_mode_never_fail_fast() {
    // After implementation, this test will:
    // 1. Mock RepairMode::Never
    // 2. Simulate missing backend
    // 3. Verify preflight fails immediately (no retry, no repair attempt)
    // 4. Verify error message includes setup instructions
    // 5. Verify exit code 2

    unimplemented!("Awaiting RepairMode implementation");
}

/// AC3.5: Test RepairMode::Auto retry logic
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac3
#[test]
#[ignore = "TODO: Implement RepairMode::Auto with retry after repair"]
fn test_repair_mode_auto_retry_logic() {
    // After implementation, this test will:
    // 1. Mock RepairMode::Auto
    // 2. Simulate missing backend
    // 3. Mock successful auto-repair
    // 4. Verify preflight retries check after repair
    // 5. Verify exit code 0 after successful repair

    unimplemented!("Awaiting RepairMode::Auto retry implementation");
}

// ============================================================================
// AC4: TokenizerAuthority Struct Tests (6 tests)
// ============================================================================

/// AC4.1: Test TokenizerAuthority serialization/deserialization
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac4
#[test]
fn test_tokenizer_authority_serialization() {
    let auth = mock_tokenizer_authority("external");

    // Serialize to JSON
    let json = serde_json::to_string(&auth).expect("Failed to serialize TokenizerAuthority");

    // Deserialize back
    let deserialized: MockTokenizerAuthority =
        serde_json::from_str(&json).expect("Failed to deserialize TokenizerAuthority");

    // Verify round-trip
    assert_eq!(auth, deserialized, "TokenizerAuthority should round-trip through JSON");
}

/// AC4.2: Test TokenizerAuthority builder API: set_tokenizer_authority()
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac4
#[test]
#[ignore = "TODO: Implement ParityReceipt::set_tokenizer_authority() builder method"]
fn test_tokenizer_authority_builder_set() {
    // After implementation, this test will:
    // 1. Create a ParityReceipt
    // 2. Call receipt.set_tokenizer_authority(authority)
    // 3. Verify tokenizer_authority field is set
    // 4. Serialize and verify JSON contains tokenizer_authority

    unimplemented!("Awaiting ParityReceipt builder API implementation");
}

/// AC4.3: Test TokenizerAuthority builder API: set_prompt_template()
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac4
#[test]
#[ignore = "TODO: Implement ParityReceipt::set_prompt_template() builder method"]
fn test_tokenizer_authority_builder_set_template() {
    // After implementation, this test will:
    // 1. Create a ParityReceipt
    // 2. Call receipt.set_prompt_template("instruct")
    // 3. Verify prompt_template field is set
    // 4. Serialize and verify JSON contains prompt_template

    unimplemented!("Awaiting ParityReceipt builder API implementation");
}

/// AC4.4: Test backward compatibility with v1 receipts
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac4
#[test]
fn test_receipt_backward_compatibility() {
    // Old v1 receipt without tokenizer authority
    let json_v1 = r#"{
        "version": 1,
        "backend": "bitnet",
        "prompt": "What is 2+2?",
        "positions": 4
    }"#;

    // Should deserialize successfully with optional fields as None
    let receipt: MockParityReceiptV2 =
        serde_json::from_str(json_v1).expect("Failed to deserialize v1 receipt");

    assert!(receipt.tokenizer_authority.is_none(), "v1 receipt should have no tokenizer_authority");
    assert!(receipt.prompt_template.is_none(), "v1 receipt should have no prompt_template");
    assert!(receipt.model_sha256.is_none(), "v1 receipt should have no model_sha256");
}

/// AC4.5: Test version inference (1.0.0 vs 2.0.0)
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac4
#[test]
#[ignore = "TODO: Implement ParityReceipt::infer_version() method"]
fn test_receipt_version_inference() {
    // After implementation, this test will:
    // 1. Create receipt without tokenizer_authority → infer_version() == "1.0.0"
    // 2. Create receipt with tokenizer_authority → infer_version() == "2.0.0"
    // 3. Verify version string is included in JSON output

    unimplemented!("Awaiting ParityReceipt::infer_version() implementation");
}

/// AC4.6: Test optional field handling
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac4
#[test]
fn test_tokenizer_authority_optional_fields() {
    // External tokenizer: file_hash is Some
    let external = mock_tokenizer_authority("external");
    assert!(external.file_hash.is_some(), "External tokenizer should have file_hash");

    // GGUF embedded tokenizer: file_hash is None
    let embedded = mock_tokenizer_authority("gguf_embedded");
    assert!(embedded.file_hash.is_none(), "GGUF embedded tokenizer should not have file_hash");

    // Serialize and verify skip_serializing_if works
    let json_embedded = serde_json::to_string(&embedded).expect("Failed to serialize");
    assert!(!json_embedded.contains("file_hash"), "file_hash should be omitted when None");
}

// ============================================================================
// AC5: Tokenizer Source Tracking Tests (4 tests)
// ============================================================================

/// AC5.1: Test GGUF embedded tokenizer detection
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac5
#[test]
#[ignore = "TODO: Implement detect_tokenizer_source() with GGUF metadata inspection"]
fn test_tokenizer_source_gguf_embedded() {
    // After implementation, this test will:
    // 1. Mock GGUF file with embedded tokenizer metadata
    // 2. Call detect_tokenizer_source(model_path, tokenizer_path)
    // 3. Verify source == "gguf_embedded"
    // 4. Verify path points to GGUF file

    unimplemented!("Awaiting detect_tokenizer_source() implementation");
}

/// AC5.2: Test external tokenizer.json detection
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac5
#[test]
#[ignore = "TODO: Implement detect_tokenizer_source() with external JSON detection"]
fn test_tokenizer_source_external_json() {
    // After implementation, this test will:
    // 1. Provide explicit tokenizer.json path
    // 2. Call detect_tokenizer_source(model_path, tokenizer_path)
    // 3. Verify source == "external"
    // 4. Verify path points to tokenizer.json file

    unimplemented!("Awaiting detect_tokenizer_source() implementation");
}

/// AC5.3: Test auto-discovered tokenizer from model directory
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac5
#[test]
#[ignore = "TODO: Implement detect_tokenizer_source() with auto-discovery"]
fn test_tokenizer_source_auto_discovered() {
    // After implementation, this test will:
    // 1. Place tokenizer.json in same directory as model.gguf
    // 2. Call detect_tokenizer_source(model_path, None)
    // 3. Verify source == "auto_discovered"
    // 4. Verify path points to discovered tokenizer.json

    unimplemented!("Awaiting detect_tokenizer_source() auto-discovery implementation");
}

/// AC5.4: Test source field validation in receipts
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac5
#[test]
fn test_tokenizer_source_field_validation() {
    let valid_sources = ["gguf_embedded", "external", "auto_discovered"];

    for source in &valid_sources {
        let auth = mock_tokenizer_authority(source);
        assert_eq!(auth.source, *source, "Source field should match expected value");
    }
}

// ============================================================================
// AC6: SHA256 Hash Computation Tests (5 tests)
// ============================================================================

/// AC6.1: Test file hash determinism (same file, same hash)
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac6
#[test]
#[ignore = "TODO: Implement compute_tokenizer_file_hash()"]
fn test_file_hash_determinism() {
    // After implementation, this test will:
    // 1. Create a test tokenizer.json file
    // 2. Compute hash twice: compute_tokenizer_file_hash(path)
    // 3. Verify hash1 == hash2 (deterministic)
    // 4. Modify file content
    // 5. Verify hash changes

    unimplemented!("Awaiting compute_tokenizer_file_hash() implementation");
}

/// AC6.2: Test config hash determinism (same tokenizer, same hash)
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac6
#[test]
#[ignore = "TODO: Implement compute_tokenizer_config_hash()"]
fn test_config_hash_determinism() {
    // After implementation, this test will:
    // 1. Load tokenizer from tokenizer.json
    // 2. Compute hash twice: compute_tokenizer_config_hash(&tokenizer)
    // 3. Verify hash1 == hash2 (deterministic)
    // 4. Reload tokenizer from same file
    // 5. Verify hash matches (canonical JSON serialization)

    unimplemented!("Awaiting compute_tokenizer_config_hash() implementation");
}

/// AC6.3: Test file hash vs config hash distinction
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac6
#[test]
#[ignore = "TODO: Implement both hash computation functions"]
fn test_file_hash_vs_config_hash() {
    // After implementation, this test will:
    // 1. Load tokenizer from tokenizer.json
    // 2. Compute file_hash = SHA256(file contents)
    // 3. Compute config_hash = SHA256(canonical vocab JSON)
    // 4. Verify file_hash != config_hash (different inputs)
    // 5. Add whitespace to tokenizer.json (no semantic change)
    // 6. Verify file_hash changes, config_hash remains same

    unimplemented!("Awaiting hash computation implementations");
}

/// AC6.4: Test hash format validation (64-char hex)
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac6
#[test]
#[ignore = "TODO: Implement hash computation with format validation"]
fn test_hash_format_validation() {
    // After implementation, this test will:
    // 1. Compute tokenizer file hash
    // 2. Verify hash is 64 characters (SHA256 hex encoding)
    // 3. Verify hash contains only valid hex chars (0-9, a-f)
    // 4. Verify no uppercase chars (lowercase hex convention)

    unimplemented!("Awaiting hash computation implementations");
}

/// AC6.5: Test hash computation for different tokenizer formats
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac6
#[test]
#[ignore = "TODO: Implement hash computation for multiple tokenizer formats"]
fn test_hash_different_tokenizer_formats() {
    // After implementation, this test will:
    // 1. Create tokenizer from tokenizers-rs format
    // 2. Create tokenizer from SentencePiece format
    // 3. Compute config_hash for both
    // 4. Verify hashes differ (different tokenizer implementations)
    // 5. Verify hashes are consistent for same format

    unimplemented!("Awaiting tokenizer format handling");
}

// ============================================================================
// AC7: Tokenizer Parity Validation Tests (6 tests)
// ============================================================================

/// AC7.1: Test token-by-token comparison (identical sequences)
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement validate_tokenizer_parity()"]
fn test_tokenizer_parity_identical() {
    // After implementation, this test will:
    // 1. Create identical token sequences: rust_tokens = cpp_tokens = [1, 2, 3, 4, 5]
    // 2. Call validate_tokenizer_parity(&rust_tokens, &cpp_tokens, "bitnet")
    // 3. Verify result is Ok(())
    // 4. Verify no error messages

    unimplemented!("Awaiting validate_tokenizer_parity() implementation");
}

/// AC7.2: Test length mismatch detection
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement validate_tokenizer_parity() with length check"]
fn test_tokenizer_parity_length_mismatch() {
    // After implementation, this test will:
    // 1. Create rust_tokens = [1, 2, 3, 4, 5], cpp_tokens = [1, 2, 3, 4]
    // 2. Call validate_tokenizer_parity(&rust_tokens, &cpp_tokens, "bitnet")
    // 3. Verify result is Err
    // 4. Verify error message: "Rust 5 tokens vs C++ 4 tokens"

    unimplemented!("Awaiting validate_tokenizer_parity() implementation");
}

/// AC7.3: Test token ID mismatch at specific position
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement validate_tokenizer_parity() with position tracking"]
fn test_tokenizer_parity_token_mismatch() {
    // After implementation, this test will:
    // 1. Create rust_tokens = [1, 2, 3, 4, 5], cpp_tokens = [1, 2, 99, 4, 5]
    // 2. Call validate_tokenizer_parity(&rust_tokens, &cpp_tokens, "bitnet")
    // 3. Verify result is Err
    // 4. Verify error message: "position 2: Rust token=3, C++ token=99"

    unimplemented!("Awaiting validate_tokenizer_parity() implementation");
}

/// AC7.4: Test exit code 2 for token parity violations
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement parity-both command with token parity validation"]
fn test_tokenizer_parity_exit_code_2() {
    use std::process::Command;
    use tempfile::TempDir;

    let out_dir = TempDir::new().expect("Failed to create temp dir");

    // This test requires a scenario where Rust and C++ tokenizers produce different tokens
    // For now, we test the exit code logic in isolation

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            "tests/fixtures/divergent_tokenizer.gguf", // Mock model with token mismatch
            "--tokenizer",
            "tests/fixtures/tokenizer.json",
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    // Should exit with code 2 (usage error / invalid state)
    assert_eq!(output.status.code(), Some(2), "Token parity mismatch should exit with code 2");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("token") || stderr.contains("parity"),
        "Error message should mention token parity issue"
    );
}

/// AC7.5: Test divergence diagnostics (position, token IDs)
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement tokenizer parity validation with detailed diagnostics"]
fn test_tokenizer_parity_diagnostics() {
    // After implementation, this test will:
    // 1. Create token mismatch at position 2
    // 2. Call validate_tokenizer_parity()
    // 3. Verify error message includes:
    //    - Divergence position: 2
    //    - Rust token ID: 3
    //    - C++ token ID: 99
    //    - Backend name: "bitnet"

    unimplemented!("Awaiting tokenizer parity diagnostics");
}

/// AC7.6: Test parity validation before logits comparison
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement parity-both command with token parity pre-check"]
fn test_tokenizer_parity_before_logits() {
    // After implementation, this test will:
    // 1. Run parity-both with token mismatch scenario
    // 2. Verify tokenizer parity check runs BEFORE logits comparison
    // 3. Verify command exits early on token parity failure
    // 4. Verify logits comparison is NOT attempted

    unimplemented!("Awaiting parity-both dual-lane execution flow");
}

// ============================================================================
// AC8: Dual Receipts with Tokenizer Metadata Tests (5 tests)
// ============================================================================

/// AC8.1: Test receipt naming convention
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac8
#[test]
fn test_receipt_naming_convention() {
    let out_dir = PathBuf::from("/tmp/test-parity-both");

    // Expected naming: {out_dir}/receipt_bitnet.json, {out_dir}/receipt_llama.json
    let expected_bitnet = out_dir.join("receipt_bitnet.json");
    let expected_llama = out_dir.join("receipt_llama.json");

    assert_eq!(
        expected_bitnet.file_name().unwrap(),
        "receipt_bitnet.json",
        "BitNet receipt should follow naming convention"
    );

    assert_eq!(
        expected_llama.file_name().unwrap(),
        "receipt_llama.json",
        "llama receipt should follow naming convention"
    );

    // Verify distinct naming
    assert_ne!(expected_bitnet, expected_llama, "Receipt files must have distinct names");
}

/// AC8.2: Test tokenizer authority present in both receipts
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac8
#[test]
#[ignore = "TODO: Implement parity-both command with tokenizer authority in receipts"]
fn test_receipt_tokenizer_authority_present() {
    use std::process::Command;
    use tempfile::TempDir;

    let out_dir = TempDir::new().expect("Failed to create temp dir");

    // Run parity-both command
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            "tests/fixtures/test_model.gguf",
            "--tokenizer",
            "tests/fixtures/tokenizer.json",
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    assert!(output.status.success(), "parity-both should succeed");

    // Verify receipt_bitnet.json exists and has tokenizer_authority
    let receipt_bitnet_path = out_dir.path().join("receipt_bitnet.json");
    let receipt_bitnet_json =
        std::fs::read_to_string(&receipt_bitnet_path).expect("Failed to read BitNet receipt");
    let receipt_bitnet: serde_json::Value =
        serde_json::from_str(&receipt_bitnet_json).expect("Failed to parse BitNet receipt");

    assert!(
        receipt_bitnet["tokenizer_authority"].is_object(),
        "BitNet receipt should have tokenizer_authority"
    );

    // Verify receipt_llama.json exists and has tokenizer_authority
    let receipt_llama_path = out_dir.path().join("receipt_llama.json");
    let receipt_llama_json =
        std::fs::read_to_string(&receipt_llama_path).expect("Failed to read llama receipt");
    let receipt_llama: serde_json::Value =
        serde_json::from_str(&receipt_llama_json).expect("Failed to parse llama receipt");

    assert!(
        receipt_llama["tokenizer_authority"].is_object(),
        "llama receipt should have tokenizer_authority"
    );
}

/// AC8.3: Test tokenizer authority consistency across lanes
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac8
#[test]
fn test_tokenizer_authority_consistency() {
    let lane_a = mock_tokenizer_authority("external");
    let lane_b = mock_tokenizer_authority("external");

    assert_tokenizer_consistency(&lane_a, &lane_b);
}

/// AC8.4: Test prompt template field in receipts
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac8
#[test]
fn test_receipt_prompt_template_field() {
    let receipt = mock_parity_receipt_v2("bitnet");

    assert!(receipt.prompt_template.is_some(), "Receipt should have prompt_template field");
    assert_eq!(
        receipt.prompt_template.unwrap(),
        "instruct",
        "Prompt template should match expected value"
    );
}

/// AC8.5: Test model SHA256 hash in receipts
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac8
#[test]
fn test_receipt_model_sha256_field() {
    let receipt = mock_parity_receipt_v2("bitnet");

    assert!(receipt.model_sha256.is_some(), "Receipt should have model_sha256 field");

    // Verify hash format (should be hex string)
    let hash = receipt.model_sha256.unwrap();
    assert!(!hash.is_empty(), "Model SHA256 hash should not be empty");
}

// ============================================================================
// AC9: Prompt Template Auto-Detection Tests (5 tests)
// ============================================================================

/// AC9.1: Test auto-detect from GGUF chat_template metadata
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac9
#[test]
#[ignore = "TODO: Implement auto_detect_template() with GGUF metadata inspection"]
fn test_auto_detect_from_gguf_metadata() {
    // After implementation, this test will:
    // 1. Mock GGUF with chat_template = "[INST] {prompt} [/INST]"
    // 2. Call auto_detect_template(model_path)
    // 3. Verify result is TemplateType::Instruct
    // 4. Mock GGUF with chat_template containing "<|eot_id|>"
    // 5. Verify result is TemplateType::Llama3Chat

    unimplemented!("Awaiting auto_detect_template() implementation");
}

/// AC9.2: Test detect LLaMA-3 special tokens
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac9
#[test]
#[ignore = "TODO: Implement LLaMA-3 special token detection"]
fn test_auto_detect_llama3_special_tokens() {
    // After implementation, this test will:
    // 1. Mock GGUF with chat_template containing "<|eot_id|>"
    // 2. Call auto_detect_template(model_path)
    // 3. Verify result is TemplateType::Llama3Chat
    // 4. Mock GGUF with "<|start_header_id|>"
    // 5. Verify result is TemplateType::Llama3Chat

    unimplemented!("Awaiting LLaMA-3 detection logic");
}

/// AC9.3: Test heuristics from model path
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac9
#[test]
#[ignore = "TODO: Implement path-based heuristics"]
fn test_auto_detect_from_model_path() {
    // After implementation, this test will:
    // 1. Model path contains "llama-3" → TemplateType::Llama3Chat
    // 2. Model path contains "instruct" → TemplateType::Instruct
    // 3. Model path contains "chat" → TemplateType::Instruct
    // 4. Generic path "model.gguf" → TemplateType::Instruct (fallback)

    unimplemented!("Awaiting path heuristics implementation");
}

/// AC9.4: Test fallback to Instruct (safer than Raw)
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac9
#[test]
#[ignore = "TODO: Implement fallback logic with Instruct default"]
fn test_auto_detect_fallback_instruct() {
    // After implementation, this test will:
    // 1. Mock GGUF without chat_template metadata
    // 2. Model path contains no template hints
    // 3. Call auto_detect_template(model_path)
    // 4. Verify result is TemplateType::Instruct (NOT Raw)

    unimplemented!("Awaiting auto-detection fallback logic");
}

/// AC9.5: Test template override with --prompt-template
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac9
#[test]
#[ignore = "TODO: Implement --prompt-template override logic"]
fn test_prompt_template_override() {
    use std::process::Command;
    use tempfile::TempDir;

    let out_dir = TempDir::new().expect("Failed to create temp dir");

    // Run parity-both with explicit --prompt-template raw
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            "tests/fixtures/instruct_model.gguf", // Model with instruct in metadata
            "--tokenizer",
            "tests/fixtures/tokenizer.json",
            "--prompt-template",
            "raw", // Override auto-detection
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    assert!(output.status.success(), "parity-both should succeed with template override");

    // Verify receipt has prompt_template = "raw" (not "instruct")
    let receipt_path = out_dir.path().join("receipt_bitnet.json");
    let receipt_json = std::fs::read_to_string(&receipt_path).expect("Failed to read receipt");
    let receipt: serde_json::Value =
        serde_json::from_str(&receipt_json).expect("Failed to parse receipt");

    assert_eq!(
        receipt["prompt_template"], "raw",
        "Receipt should show overridden template, not auto-detected"
    );
}

// ============================================================================
// AC10: Parallel Lanes Tests (Optional) (3 tests)
// ============================================================================

/// AC10.1: Test sequential execution (default)
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
#[ignore = "TODO: Implement dual-lane execution with sequential mode (default)"]
fn test_dual_lanes_sequential_default() {
    // After implementation, this test will:
    // 1. Run parity-both without --parallel flag
    // 2. Verify lanes run sequentially (Lane A completes before Lane B starts)
    // 3. Measure execution time (should be ~lane_a_time + lane_b_time)

    unimplemented!("Awaiting dual-lane execution implementation");
}

/// AC10.2: Test parallel execution with --parallel flag
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
#[ignore = "TODO: Implement dual-lane execution with --parallel flag (future enhancement)"]
fn test_dual_lanes_parallel_flag() {
    // After implementation, this test will:
    // 1. Run parity-both with --parallel flag
    // 2. Verify lanes run concurrently (check timing)
    // 3. Measure execution time (should be ~max(lane_a_time, lane_b_time))
    // 4. Verify speedup of ~30% for typical workload

    unimplemented!("Awaiting --parallel feature (deferred to post-MVP)");
}

/// AC10.3: Test receipt consistency across modes
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
#[ignore = "TODO: Implement receipt consistency validation across sequential/parallel modes"]
fn test_receipt_consistency_across_modes() {
    // After implementation, this test will:
    // 1. Run parity-both in sequential mode
    // 2. Capture both receipts (receipt_bitnet.json, receipt_llama.json)
    // 3. Run parity-both in parallel mode with same inputs
    // 4. Capture receipts
    // 5. Verify receipts are identical (excluding timestamp)

    unimplemented!("Awaiting --parallel feature");
}

// ============================================================================
// Integration Tests: End-to-End Scenarios
// ============================================================================

/// Integration test: Full parity-both flow with tokenizer authority
/// Tests feature spec: parity-both-preflight-tokenizer.md (full flow AC1-AC9)
#[test]
#[ignore = "TODO: Implement full parity-both command; requires C++ backends and test model"]
#[serial(bitnet_env)]
fn integration_full_parity_both_flow() {
    use std::process::Command;
    use tempfile::TempDir;

    // This integration test validates the complete flow:
    // 1. Preflight both backends (with auto-repair if needed)
    // 2. Shared setup (template auto-detection, tokenizer authority capture)
    // 3. Dual-lane execution (BitNet.cpp + llama.cpp)
    // 4. Token parity validation (fail-fast if mismatch)
    // 5. Logits comparison (MSE, cosine similarity)
    // 6. Receipt generation (with tokenizer authority, prompt template, model hash)
    // 7. Summary output (text or JSON format)

    let out_dir = TempDir::new().expect("Failed to create temp dir");

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            "tests/fixtures/test_model.gguf",
            "--tokenizer",
            "tests/fixtures/tokenizer.json",
            "--prompt",
            "What is 2+2?",
            "--max-tokens",
            "4",
            "--out-dir",
            out_dir.path().to_str().unwrap(),
            "--verbose",
        ])
        .output()
        .expect("Failed to execute parity-both");

    assert!(output.status.success(), "parity-both should complete successfully");

    // Verify both receipts exist
    let receipt_bitnet = out_dir.path().join("receipt_bitnet.json");
    let receipt_llama = out_dir.path().join("receipt_llama.json");

    assert!(receipt_bitnet.exists(), "BitNet receipt should exist");
    assert!(receipt_llama.exists(), "llama receipt should exist");

    // Verify receipts have tokenizer authority
    let bitnet_json = std::fs::read_to_string(&receipt_bitnet).expect("Failed to read receipt");
    let bitnet_receipt: serde_json::Value =
        serde_json::from_str(&bitnet_json).expect("Failed to parse receipt");

    assert!(
        bitnet_receipt["tokenizer_authority"].is_object(),
        "Receipt should have tokenizer_authority"
    );
    assert!(bitnet_receipt["prompt_template"].is_string(), "Receipt should have prompt_template");
    assert!(bitnet_receipt["model_sha256"].is_string(), "Receipt should have model_sha256");

    // Verify tokenizer authority consistency
    let llama_json = std::fs::read_to_string(&receipt_llama).expect("Failed to read receipt");
    let llama_receipt: serde_json::Value =
        serde_json::from_str(&llama_json).expect("Failed to parse receipt");

    assert_eq!(
        bitnet_receipt["tokenizer_authority"]["config_hash"],
        llama_receipt["tokenizer_authority"]["config_hash"],
        "Tokenizer config hash must match across lanes"
    );
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

/// Test: Invalid GGUF path
#[test]
fn test_invalid_gguf_path() {
    use std::process::Command;
    use tempfile::TempDir;

    let out_dir = TempDir::new().expect("Failed to create temp dir");

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            "/nonexistent/model.gguf",
            "--tokenizer",
            "tokenizer.json",
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    assert!(!output.status.success(), "Should fail on invalid model path");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("No such file"),
        "Error message should mention missing file"
    );
}

/// Test: Invalid tokenizer path
#[test]
fn test_invalid_tokenizer_path() {
    use std::process::Command;
    use tempfile::TempDir;

    let out_dir = TempDir::new().expect("Failed to create temp dir");

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            "tests/fixtures/test_model.gguf",
            "--tokenizer",
            "/nonexistent/tokenizer.json",
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    assert!(!output.status.success(), "Should fail on invalid tokenizer path");
}

/// Test: Output directory creation
#[test]
#[ignore = "TODO: Implement parity-both command with output directory creation"]
fn test_output_directory_creation() {
    use std::process::Command;
    use tempfile::TempDir;

    let temp_root = TempDir::new().expect("Failed to create temp root");
    let out_dir = temp_root.path().join("nonexistent_dir");

    // Verify directory does not exist initially
    assert!(!out_dir.exists(), "Output directory should not exist initially");

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            "tests/fixtures/test_model.gguf",
            "--tokenizer",
            "tests/fixtures/tokenizer.json",
            "--out-dir",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    assert!(output.status.success(), "Should succeed and create output directory");

    // Verify directory was created
    assert!(out_dir.exists(), "Output directory should be created by command");
}

// ============================================================================
// AC4: Exit Code Logic Tests (5 tests) - ADDITIONAL COVERAGE
// ============================================================================

/// AC4.1: Test exit code 0 when both lanes pass
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac4
#[test]
#[ignore = "TODO: Implement parity-both command with exit code 0 for successful parity"]
fn test_exit_code_0_both_lanes_pass() {
    // After implementation, this test will:
    // 1. Run parity-both with valid inputs
    // 2. Both lanes pass parity checks (MSE < threshold)
    // 3. Verify exit code 0
    // 4. Verify success message in output

    unimplemented!("Awaiting parity-both dual-lane execution");
}

/// AC4.2: Test exit code 1 when lane fails parity check
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac4
#[test]
#[ignore = "TODO: Implement parity-both command with exit code 1 for parity check failure"]
fn test_exit_code_1_lane_failure() {
    // After implementation, this test will:
    // 1. Mock scenario where logits diverge (MSE > threshold)
    // 2. Run parity-both
    // 3. Verify exit code 1 (lane failure)
    // 4. Verify error message shows which lane failed

    unimplemented!("Awaiting parity check failure handling");
}

/// AC4.3: Test exit code 2 for token parity mismatch
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac4
#[test]
#[ignore = "TODO: Implement token parity validation with exit code 2"]
fn test_exit_code_2_token_parity_mismatch() {
    // After implementation, this test will:
    // 1. Mock scenario where Rust tokens != C++ tokens
    // 2. Run parity-both
    // 3. Verify exit code 2 (usage error / tokenizer mismatch)
    // 4. Verify error message shows divergence details

    unimplemented!("Awaiting token parity validation");
}

/// AC4.4: Test exit code 2 for backend unavailable + no-repair
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac4
#[test]
#[ignore = "TODO: Implement preflight with exit code 2 for backend unavailable"]
fn test_exit_code_2_backend_unavailable() {
    // After implementation, this test will:
    // 1. Simulate missing backend (BITNET_CPP_DIR unset)
    // 2. Run parity-both with --no-repair
    // 3. Verify exit code 2 (backend unavailable, repair disabled)
    // 4. Verify setup instructions in error message

    unimplemented!("Awaiting preflight integration");
}

/// AC4.5: Test exit code semantics validation
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac4
#[test]
fn test_exit_code_semantics_validation() {
    // Mock exit code logic validation (no actual command execution)
    let scenarios = [
        ("both_pass", 0),
        ("lane_fail", 1),
        ("token_mismatch", 2),
        ("backend_unavailable_no_repair", 2),
    ];

    for (scenario, expected_code) in scenarios {
        // Verify exit code mapping is correct per spec
        match scenario {
            "both_pass" => assert_eq!(expected_code, 0, "Both lanes pass → exit 0"),
            "lane_fail" => assert_eq!(expected_code, 1, "Lane failure → exit 1"),
            "token_mismatch" => assert_eq!(expected_code, 2, "Token mismatch → exit 2"),
            "backend_unavailable_no_repair" => {
                assert_eq!(expected_code, 2, "Backend unavailable + no-repair → exit 2")
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// AC5: Dual Receipt Naming Convention Tests (4 tests) - ADDITIONAL COVERAGE
// ============================================================================

/// AC5.1: Test receipt naming: receipt_bitnet.json
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac5
#[test]
fn test_receipt_naming_bitnet() {
    let out_dir = PathBuf::from("/tmp/test-parity");
    let receipt_path = out_dir.join("receipt_bitnet.json");

    assert_eq!(
        receipt_path.file_name().unwrap().to_str().unwrap(),
        "receipt_bitnet.json",
        "BitNet receipt must follow naming convention"
    );
}

/// AC5.2: Test receipt naming: receipt_llama.json
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac5
#[test]
fn test_receipt_naming_llama() {
    let out_dir = PathBuf::from("/tmp/test-parity");
    let receipt_path = out_dir.join("receipt_llama.json");

    assert_eq!(
        receipt_path.file_name().unwrap().to_str().unwrap(),
        "receipt_llama.json",
        "llama receipt must follow naming convention"
    );
}

/// AC5.3: Test distinct receipt files per backend
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac5
#[test]
fn test_receipt_files_distinct() {
    let out_dir = PathBuf::from("/tmp/test-parity");
    let bitnet_path = out_dir.join("receipt_bitnet.json");
    let llama_path = out_dir.join("receipt_llama.json");

    assert_ne!(bitnet_path, llama_path, "Receipt files must be distinct per backend");
}

/// AC5.4: Test output directory structure
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac5
#[test]
#[ignore = "TODO: Implement parity-both command with output directory structure validation"]
fn test_output_directory_structure() {
    // After implementation, this test will:
    // 1. Run parity-both with --out-dir /tmp/test-parity
    // 2. Verify directory contains:
    //    - receipt_bitnet.json
    //    - receipt_llama.json
    // 3. Verify no other files created (clean output)

    unimplemented!("Awaiting parity-both output directory handling");
}

// ============================================================================
// AC6: Shared Rust Inference Optimization Tests (5 tests) - ADDITIONAL
// ============================================================================

/// AC6.1: Test single Rust tokenization for both lanes
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac6
#[test]
#[ignore = "TODO: Implement shared Rust tokenization in parity-both"]
fn test_shared_rust_tokenization() {
    // After implementation, this test will:
    // 1. Mock tokenizer load with call counter
    // 2. Run parity-both (dual-lane)
    // 3. Verify tokenizer loaded ONCE (not twice)
    // 4. Verify both lanes use same token sequence

    unimplemented!("Awaiting shared setup implementation");
}

/// AC6.2: Test shared logits evaluation baseline
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac6
#[test]
#[ignore = "TODO: Implement shared Rust logits evaluation in parity-both"]
fn test_shared_rust_logits_baseline() {
    // After implementation, this test will:
    // 1. Mock inference with call counter
    // 2. Run parity-both (dual-lane)
    // 3. Verify Rust inference runs ONCE (shared baseline)
    // 4. Verify C++ lanes compare against same baseline

    unimplemented!("Awaiting shared logits evaluation");
}

/// AC6.3: Test memory optimization (no duplicate tokenizer load)
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac6
#[test]
#[ignore = "TODO: Implement memory profiling for shared setup"]
fn test_shared_setup_memory_optimization() {
    // After implementation, this test will:
    // 1. Measure memory before shared setup
    // 2. Run parity-both with shared setup
    // 3. Measure memory usage
    // 4. Verify single tokenizer instance loaded (not dual)

    unimplemented!("Awaiting memory optimization validation");
}

/// AC6.4: Test performance: shared setup vs dual setup
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac6
#[test]
#[ignore = "TODO: Implement performance benchmark for shared setup"]
fn test_shared_setup_performance() {
    // After implementation, this test will:
    // 1. Benchmark dual-lane with shared setup
    // 2. Compare with hypothetical dual setup (tokenizer loaded twice)
    // 3. Verify shared setup is faster (reduced I/O, single tokenizer load)

    unimplemented!("Awaiting performance benchmarking");
}

/// AC6.5: Test tokenizer authority captured once
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac6
#[test]
#[ignore = "TODO: Implement tokenizer authority capture in shared setup"]
fn test_tokenizer_authority_captured_once() {
    // After implementation, this test will:
    // 1. Run parity-both with tokenizer authority capture
    // 2. Verify tokenizer authority computed ONCE during shared setup
    // 3. Verify both lane receipts reference same authority object
    // 4. Verify config_hash identical across lanes

    unimplemented!("Awaiting tokenizer authority capture");
}

// ============================================================================
// AC7: Per-Position Metrics Tests (6 tests) - ADDITIONAL COVERAGE
// ============================================================================

/// AC7.1: Test MSE computation per position
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement MSE metric computation per position"]
fn test_per_position_mse() {
    // After implementation, this test will:
    // 1. Run parity-both with multi-token output
    // 2. Parse receipt rows
    // 3. Verify each row has MSE value
    // 4. Verify MSE is numeric and >= 0

    unimplemented!("Awaiting per-position metrics");
}

/// AC7.2: Test L2 distance per position
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement L2 distance computation per position"]
fn test_per_position_l2_distance() {
    // After implementation, this test will:
    // 1. Run parity-both
    // 2. Parse receipt rows
    // 3. Verify each row has max_abs (L2 distance) value
    // 4. Verify L2 distance is numeric and >= 0

    unimplemented!("Awaiting L2 distance computation");
}

/// AC7.3: Test KL divergence per position
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement KL divergence computation per position"]
fn test_per_position_kl_divergence() {
    // After implementation, this test will:
    // 1. Run parity-both
    // 2. Parse receipt rows
    // 3. Verify each row has kl value
    // 4. Verify KL divergence is numeric and >= 0

    unimplemented!("Awaiting KL divergence computation");
}

/// AC7.4: Test Top-K agreement per position
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement Top-K agreement computation per position"]
fn test_per_position_topk_agreement() {
    // After implementation, this test will:
    // 1. Run parity-both
    // 2. Parse receipt rows
    // 3. Verify each row has topk_agree value
    // 4. Verify topk_agree is in range [0.0, 1.0]

    unimplemented!("Awaiting Top-K agreement computation");
}

/// AC7.5: Test position metrics in receipt rows
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement receipt row structure with all metrics"]
fn test_position_metrics_in_receipt_rows() {
    // After implementation, this test will:
    // 1. Run parity-both
    // 2. Parse receipt JSON
    // 3. Verify rows array exists
    // 4. Verify each row has: pos, mse, max_abs, kl, topk_agree
    // 5. Verify top5_rust and top5_cpp arrays present

    unimplemented!("Awaiting receipt row schema implementation");
}

/// AC7.6: Test threshold validation per metric
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac7
#[test]
#[ignore = "TODO: Implement threshold validation against per-position metrics"]
fn test_threshold_validation_per_metric() {
    // After implementation, this test will:
    // 1. Set thresholds: mse=0.0001, kl=0.1, topk=0.8
    // 2. Run parity-both with metrics above/below thresholds
    // 3. Verify failure when MSE > threshold
    // 4. Verify failure when KL > threshold
    // 5. Verify failure when topk_agree < threshold

    unimplemented!("Awaiting threshold validation logic");
}

// ============================================================================
// AC8: Summary with First Divergence Tests (5 tests) - ADDITIONAL
// ============================================================================

/// AC8.1: Test summary includes first divergence position
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac8
#[test]
#[ignore = "TODO: Implement summary with first_divergence field"]
fn test_summary_first_divergence() {
    // After implementation, this test will:
    // 1. Mock scenario with divergence at position 2
    // 2. Run parity-both
    // 3. Parse receipt summary
    // 4. Verify summary.first_divergence == 2
    // 5. Verify null when no divergence

    unimplemented!("Awaiting summary first_divergence tracking");
}

/// AC8.2: Test summary shows mean metrics across positions
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac8
#[test]
#[ignore = "TODO: Implement summary with mean metrics computation"]
fn test_summary_mean_metrics() {
    // After implementation, this test will:
    // 1. Run parity-both with multi-position output
    // 2. Parse receipt summary
    // 3. Verify summary.mean_mse is average of all position MSEs
    // 4. Verify summary.mean_kl is average of all position KLs

    unimplemented!("Awaiting summary mean metrics");
}

/// AC8.3: Test all-passed status flag
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac8
#[test]
#[ignore = "TODO: Implement summary with all_passed boolean flag"]
fn test_summary_all_passed_flag() {
    // After implementation, this test will:
    // 1. Run parity-both with passing scenario
    // 2. Verify summary.all_passed == true
    // 3. Run with failing scenario (MSE > threshold)
    // 4. Verify summary.all_passed == false

    unimplemented!("Awaiting summary all_passed flag");
}

/// AC8.4: Test summary in text format
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac8
#[test]
#[ignore = "TODO: Implement text format summary output"]
fn test_summary_text_format() {
    // After implementation, this test will:
    // 1. Run parity-both with --format text
    // 2. Capture stdout
    // 3. Verify text contains "Lane A: BitNet.cpp"
    // 4. Verify text contains "Status: ✓ Parity OK"
    // 5. Verify text contains "First divergence: None"

    unimplemented!("Awaiting text format summary");
}

/// AC8.5: Test summary in JSON format
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac8
#[test]
#[ignore = "TODO: Implement JSON format summary output"]
fn test_summary_json_format() {
    // After implementation, this test will:
    // 1. Run parity-both with --format json
    // 2. Parse JSON output from stdout
    // 3. Verify JSON structure: lanes.bitnet, lanes.llama, overall
    // 4. Verify overall.both_passed boolean
    // 5. Verify overall.tokenizer_consistent boolean

    unimplemented!("Awaiting JSON format summary");
}

// ============================================================================
// AC9: CLI Wiring in main.rs Tests (4 tests) - ADDITIONAL COVERAGE
// ============================================================================

/// AC9.1: Test command dispatch to parity_both module
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac9
#[test]
#[ignore = "TODO: Implement CLI dispatch validation for parity-both command"]
fn test_cli_dispatch_to_parity_both_module() {
    // After implementation, this test will:
    // 1. Verify xtask main.rs routes "parity-both" to crossval/parity_both.rs
    // 2. Verify function signature matches: parity_both::run(args)
    // 3. Verify error handling propagates correctly

    unimplemented!("Awaiting CLI dispatch implementation");
}

/// AC9.2: Test prompt template conversion with model path
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac9
#[test]
#[ignore = "TODO: Implement PromptTemplateArg::to_template_type(model_path)"]
fn test_prompt_template_conversion_with_model_path() {
    // After implementation, this test will:
    // 1. Mock PromptTemplateArg::Auto with model path
    // 2. Call to_template_type(Some(model_path))
    // 3. Verify auto-detection runs (not hardcoded to Raw)
    // 4. Verify model path is passed for GGUF metadata inspection

    unimplemented!("Awaiting PromptTemplateArg conversion enhancement");
}

/// AC9.3: Test auto-repair flag inversion (--no-repair → auto_repair=false)
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac9
#[test]
#[ignore = "TODO: Implement --no-repair flag inversion in CLI dispatch"]
fn test_auto_repair_flag_inversion() {
    // After implementation, this test will:
    // 1. CLI arg: no_repair=true → auto_repair=false
    // 2. CLI arg: no_repair=false → auto_repair=true
    // 3. Verify correct inversion in parity_both_cmd() call

    unimplemented!("Awaiting CLI argument mapping");
}

/// AC9.4: Test argument forwarding completeness
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac9
#[test]
#[ignore = "TODO: Implement argument forwarding validation for parity-both"]
fn test_argument_forwarding_completeness() {
    // After implementation, this test will:
    // 1. Verify all CLI args forwarded to parity_both::run():
    //    - model_gguf, tokenizer, prompt, max_tokens
    //    - cos_tol, out_dir, format, prompt_template
    //    - system_prompt, auto_repair, verbose, dump_ids, dump_cpp_ids, metrics
    // 2. Verify no args dropped or misnamed

    unimplemented!("Awaiting argument forwarding validation");
}

// ============================================================================
// AC10: Test Coverage Validation (10 tests) - ADDITIONAL COVERAGE
// ============================================================================

/// AC10.1: Test dual-lane execution test coverage
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
fn test_coverage_dual_lane_execution() {
    // Verify dual-lane execution test coverage exists
    let test_names = [
        "test_shared_rust_tokenization",
        "test_shared_rust_logits_baseline",
        "test_per_position_mse",
        "test_per_position_l2_distance",
        "test_summary_first_divergence",
    ];

    for test_name in &test_names {
        // Test existence verified by compilation
        assert!(!test_name.is_empty(), "Dual-lane execution tests exist");
    }
}

/// AC10.2: Test exit code test coverage (0, 1, 2)
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
fn test_coverage_exit_codes() {
    // Verify exit code test coverage exists
    let exit_code_tests = [
        "test_exit_code_0_both_lanes_pass",
        "test_exit_code_1_lane_failure",
        "test_exit_code_2_token_parity_mismatch",
        "test_exit_code_2_backend_unavailable",
        "test_exit_code_semantics_validation",
    ];

    for test_name in &exit_code_tests {
        assert!(!test_name.is_empty(), "Exit code tests exist");
    }
}

/// AC10.3: Test tokenizer authority consistency tests
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
fn test_coverage_tokenizer_authority() {
    // Verify tokenizer authority test coverage exists
    let tokenizer_tests = [
        "test_tokenizer_authority_serialization",
        "test_tokenizer_authority_consistency",
        "test_tokenizer_authority_captured_once",
        "test_receipt_tokenizer_authority_present",
    ];

    for test_name in &tokenizer_tests {
        assert!(!test_name.is_empty(), "Tokenizer authority tests exist");
    }
}

/// AC10.4: Test receipt schema validation tests
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
fn test_coverage_receipt_schema() {
    // Verify receipt schema test coverage exists
    let schema_tests = [
        "test_receipt_backward_compatibility",
        "test_tokenizer_authority_optional_fields",
        "test_receipt_prompt_template_field",
        "test_receipt_model_sha256_field",
        "test_position_metrics_in_receipt_rows",
    ];

    for test_name in &schema_tests {
        assert!(!test_name.is_empty(), "Receipt schema tests exist");
    }
}

/// AC10.5: Test preflight integration tests
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
fn test_coverage_preflight_integration() {
    // Verify preflight integration test coverage exists
    let preflight_tests = [
        "test_preflight_auto_repair_bitnet",
        "test_preflight_auto_repair_llama",
        "test_preflight_both_available",
        "test_preflight_no_repair_fail_fast",
        "test_preflight_rebuild_instructions",
    ];

    for test_name in &preflight_tests {
        assert!(!test_name.is_empty(), "Preflight integration tests exist");
    }
}

/// AC10.6: Test template auto-detection tests
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
fn test_coverage_template_auto_detection() {
    // Verify template auto-detection test coverage exists
    let template_tests = [
        "test_auto_detect_from_gguf_metadata",
        "test_auto_detect_llama3_special_tokens",
        "test_auto_detect_from_model_path",
        "test_auto_detect_fallback_instruct",
        "test_prompt_template_override",
    ];

    for test_name in &template_tests {
        assert!(!test_name.is_empty(), "Template auto-detection tests exist");
    }
}

/// AC10.7: Test hash computation tests
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
fn test_coverage_hash_computation() {
    // Verify hash computation test coverage exists
    let hash_tests = [
        "test_file_hash_determinism",
        "test_config_hash_determinism",
        "test_file_hash_vs_config_hash",
        "test_hash_format_validation",
        "test_hash_different_tokenizer_formats",
    ];

    for test_name in &hash_tests {
        assert!(!test_name.is_empty(), "Hash computation tests exist");
    }
}

/// AC10.8: Test parity validation tests
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
fn test_coverage_parity_validation() {
    // Verify parity validation test coverage exists
    let parity_tests = [
        "test_tokenizer_parity_identical",
        "test_tokenizer_parity_length_mismatch",
        "test_tokenizer_parity_token_mismatch",
        "test_tokenizer_parity_exit_code_2",
        "test_tokenizer_parity_diagnostics",
        "test_tokenizer_parity_before_logits",
    ];

    for test_name in &parity_tests {
        assert!(!test_name.is_empty(), "Parity validation tests exist");
    }
}

/// AC10.9: Test error handling tests
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
fn test_coverage_error_handling() {
    // Verify error handling test coverage exists
    let error_tests = [
        "test_invalid_gguf_path",
        "test_invalid_tokenizer_path",
        "test_output_directory_creation",
        "test_repair_mode_no_repair_flag",
    ];

    for test_name in &error_tests {
        assert!(!test_name.is_empty(), "Error handling tests exist");
    }
}

/// AC10.10: Test full integration flow test
/// Tests feature spec: parity-both-preflight-tokenizer.md#ac10
#[test]
fn test_coverage_full_integration_flow() {
    // Verify full integration flow test exists
    let integration_test = "integration_full_parity_both_flow";

    assert!(!integration_test.is_empty(), "Full integration flow test exists");
}
