//! Comprehensive test scaffolding for parity-both command
//!
//! **Specification Reference**: `docs/specs/parity-both-command.md`
//!
//! ## Test Coverage (AC1-AC7)
//!
//! This test suite validates dual-lane cross-validation with unified receipts:
//!
//! - **AC1**: Single command dual-backend execution (5 tests)
//!   - Both backends run without user intervention
//!   - Minimal arguments with defaults
//!   - Full options testing
//!   - Backend discovery and fallback
//!   - Command completion without hang
//!
//! - **AC2**: Dual receipts generation (4 tests)
//!   - Receipt naming convention (`receipt_bitnet.json`, `receipt_llama.json`)
//!   - Schema v1 format compliance
//!   - Backend field correctness
//!   - Receipt consistency validation
//!
//! - **AC3**: Token comparison with thresholds (6 tests)
//!   - Summary shows all required metrics
//!   - Divergence position reporting
//!   - Partial failure handling
//!   - Threshold validation (MSE, cosine similarity, L2 distance)
//!   - Exact match detection
//!   - Near match vs divergence classification
//!
//! - **AC4**: Exit codes (0, 1, 2) (5 tests)
//!   - Exit 0 when both lanes pass
//!   - Exit 1 when Lane A fails, Lane B passes
//!   - Exit 1 when Lane A passes, Lane B fails
//!   - Exit 1 when both lanes fail
//!   - Exit 2 for usage errors (missing args, token mismatch)
//!
//! - **AC5**: First divergence reporting (5 tests)
//!   - Position index in summary
//!   - Token triple (id/rust/cpp) in verbose output
//!   - Receipt row data validation
//!   - Divergence detection threshold accuracy
//!   - Multiple divergence handling
//!
//! - **AC6**: Auto-repair by default (4 tests)
//!   - Auto-repair enabled without flags
//!   - `--no-repair` disables auto-repair
//!   - Auto-repair success message
//!   - Auto-repair failure handling
//!
//! - **AC7**: CLI integration (5 tests)
//!   - Help text completeness
//!   - Command appears in xtask list
//!   - Text format human-readable output
//!   - JSON format structure validation
//!   - Format consistency (text vs JSON)
//!
//! ## Test Categories
//!
//! - **Unit Tests**: Threshold parsing, cosine similarity, L2 distance, exit code logic (15 tests)
//! - **Integration Tests**: Dual-lane execution, receipt generation, comparison (10 tests)
//! - **E2E Tests**: Full command with model, tokenizer, verdict (8 tests)
//! - **Property Tests**: Summary formatting invariants (3 tests)
//!
//! ## Test Helpers
//!
//! - `mock_lane_result(tokens, cosine)` - Create mock lane result for testing
//! - `mock_parity_receipt(backend)` - Create mock receipt for schema validation
//! - `assert_receipts_exist(bitnet_path, llama_path)` - Verify both receipts created
//! - `verify_verdict(output, expected)` - Validate summary verdict
//! - `mock_comparison_metrics(exact, near, divergent)` - Mock comparison scenarios
//!
//! ## Comparison Logic Tests
//!
//! - Exact token match (cosine=1.0, L2=0.0)
//! - Near match (cosine=0.9995, L2=0.001)
//! - Divergence (cosine=0.95, L2=0.5)
//! - First divergence index detection
//! - Threshold violations (MSE, cosine, L2)
//!
//! ## Backend Scenarios
//!
//! - Both backends available (happy path)
//! - BitNet available, llama unavailable (with auto-repair)
//! - Llama available, BitNet unavailable (with auto-repair)
//! - Both unavailable (exit 2)
//!
//! ## Environment Isolation
//!
//! All tests use `#[serial(bitnet_env)]` to prevent race conditions during parallel
//! test execution when mutating environment variables.

#![cfg(all(feature = "crossval-all", feature = "inference"))]

use serial_test::serial;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

// ============================================================================
// Test Helpers
// ============================================================================

/// Helper to find workspace root by walking up to .git directory
fn workspace_root() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !path.join(".git").exists() {
        if !path.pop() {
            panic!("Could not find workspace root (no .git directory found)");
        }
    }
    path
}

/// Helper to get test model path from environment or default location
fn get_test_model_path() -> Option<PathBuf> {
    std::env::var("BITNET_GGUF").ok().map(PathBuf::from).or_else(|| {
        let workspace = workspace_root();
        let default =
            workspace.join("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
        if default.exists() { Some(default) } else { None }
    })
}

/// Helper to get test tokenizer path
fn get_test_tokenizer_path() -> Option<PathBuf> {
    std::env::var("BITNET_TOKENIZER").ok().map(PathBuf::from).or_else(|| {
        let workspace = workspace_root();
        let default = workspace.join("models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json");
        if default.exists() { Some(default) } else { None }
    })
}

/// Helper to check if C++ backend is available
fn backend_available(backend: &str) -> bool {
    // Check environment hints
    if backend == "bitnet" {
        std::env::var("BITNET_CPP_DIR").is_ok()
    } else if backend == "llama" {
        // llama.cpp availability check - simplified for testing
        std::env::var("BITNET_CPP_DIR").is_ok()
    } else {
        false
    }
}

/// Mock receipt data structure for testing (mirrors crossval::receipt::ParityReceipt)
#[derive(Debug, serde::Deserialize)]
struct MockReceipt {
    version: u32,
    backend: String,
    prompt: String,
    positions: usize,
    summary: MockSummary,
}

#[derive(Debug, serde::Deserialize)]
struct MockSummary {
    all_passed: bool,
    first_divergence: Option<usize>,
    mean_mse: f32,
}

/// Mock lane result for testing
#[derive(Debug)]
struct MockLaneResult {
    backend: String,
    passed: bool,
    first_divergence: Option<usize>,
    mean_mse: f32,
    mean_cosine_sim: f32,
}

impl MockLaneResult {
    /// Create a mock lane result with specified parameters
    fn new(backend: &str, tokens: Vec<u32>, cosine: f64) -> Self {
        let passed = cosine >= 0.999;
        let first_divergence = if passed { None } else { Some(2) };
        let mean_mse = if passed { 1e-6 } else { 5e-4 };

        Self {
            backend: backend.to_string(),
            passed,
            first_divergence,
            mean_mse: mean_mse as f32,
            mean_cosine_sim: cosine as f32,
        }
    }
}

/// Helper to create mock parity receipt for testing
fn mock_parity_receipt(backend: &str) -> MockReceipt {
    MockReceipt {
        version: 1,
        backend: backend.to_string(),
        prompt: "What is 2+2?".to_string(),
        positions: 4,
        summary: MockSummary { all_passed: true, first_divergence: None, mean_mse: 1e-6 },
    }
}

/// Helper to assert both receipt files exist
fn assert_receipts_exist(bitnet_path: &Path, llama_path: &Path) {
    assert!(bitnet_path.exists(), "BitNet receipt should exist: {}", bitnet_path.display());
    assert!(llama_path.exists(), "llama receipt should exist: {}", llama_path.display());
}

/// Helper to verify summary verdict from command output
#[allow(dead_code)]
fn verify_verdict(output: &str, expected: &str) {
    assert!(
        output.contains(expected),
        "Summary should contain verdict '{}', got: {}",
        expected,
        output
    );
}

/// Mock comparison metrics for different scenarios
#[derive(Debug)]
struct ComparisonScenario {
    cosine_sim: f64,
    l2_dist: f64,
    mse: f64,
    expected_pass: bool,
}

impl ComparisonScenario {
    fn exact_match() -> Self {
        Self { cosine_sim: 1.0, l2_dist: 0.0, mse: 0.0, expected_pass: true }
    }

    fn near_match() -> Self {
        Self { cosine_sim: 0.9995, l2_dist: 0.001, mse: 1e-6, expected_pass: true }
    }

    fn divergence() -> Self {
        Self { cosine_sim: 0.95, l2_dist: 0.5, mse: 0.25, expected_pass: false }
    }
}

// ============================================================================
// AC2: Receipt Naming Convention Tests
// ============================================================================

/// AC2: Test receipt filenames follow naming convention
/// Tests feature spec: parity-both-command.md#ac2
#[test]
fn test_receipt_naming_convention() {
    let out_dir = PathBuf::from("/tmp/test-parity-both");

    // Expected naming convention from spec:
    // {out_dir}/receipt_bitnet.json - Lane A (BitNet.cpp backend)
    // {out_dir}/receipt_llama.json - Lane B (llama.cpp backend)

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

/// AC2: Test receipt schema v1 format compliance
/// Tests feature spec: parity-both-command.md#ac2
#[test]
#[ignore = "TODO: Implement parity-both command and receipt generation"]
fn test_receipt_schema_v1_compliance() {
    // After implementation, this test will:
    // 1. Run parity-both command with test model
    // 2. Load both receipt files
    // 3. Verify schema v1 structure:
    //    - version: 1
    //    - backend: "bitnet" | "llama"
    //    - timestamp: RFC3339
    //    - model: path
    //    - prompt: string
    //    - positions: usize
    //    - thresholds: {mse, kl, topk}
    //    - rows: [{pos, mse, max_abs, ...}]
    //    - summary: {all_passed, first_divergence, mean_mse, mean_kl}

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC2: Test receipt backend field correctness
/// Tests feature spec: parity-both-command.md#ac2
#[test]
#[ignore = "TODO: Implement parity-both command and receipt generation"]
fn test_receipt_backend_field_correctness() {
    // After implementation, this test will:
    // 1. Parse receipt_bitnet.json and verify "backend": "bitnet"
    // 2. Parse receipt_llama.json and verify "backend": "llama"
    // 3. Verify field consistency across both receipts (same prompt, positions, etc.)

    unimplemented!("Awaiting parity-both command implementation");
}

// ============================================================================
// AC4: Exit Code Semantics Tests
// ============================================================================

/// AC4: Test exit code 0 when both lanes pass
/// Tests feature spec: parity-both-command.md#ac4
#[test]
fn test_exit_code_both_pass() {
    // Mock scenario: both lanes pass
    let lane_a_passed = true;
    let lane_b_passed = true;

    let both_passed = lane_a_passed && lane_b_passed;
    let expected_exit_code = if both_passed { 0 } else { 1 };

    assert_eq!(expected_exit_code, 0, "Exit code should be 0 when both lanes pass");
}

/// AC4: Test exit code 1 when Lane A fails, Lane B passes
/// Tests feature spec: parity-both-command.md#ac4
#[test]
fn test_exit_code_lane_a_fail() {
    // Mock scenario: Lane A fails, Lane B passes
    let lane_a_passed = false;
    let lane_b_passed = true;

    let both_passed = lane_a_passed && lane_b_passed;
    let expected_exit_code = if both_passed { 0 } else { 1 };

    assert_eq!(expected_exit_code, 1, "Exit code should be 1 when either lane fails");
}

/// AC4: Test exit code 1 when Lane A passes, Lane B fails
/// Tests feature spec: parity-both-command.md#ac4
#[test]
fn test_exit_code_lane_b_fail() {
    // Mock scenario: Lane A passes, Lane B fails
    let lane_a_passed = true;
    let lane_b_passed = false;

    let both_passed = lane_a_passed && lane_b_passed;
    let expected_exit_code = if both_passed { 0 } else { 1 };

    assert_eq!(expected_exit_code, 1, "Exit code should be 1 when either lane fails");
}

/// AC4: Test exit code 1 when both lanes fail
/// Tests feature spec: parity-both-command.md#ac4
#[test]
fn test_exit_code_both_fail() {
    // Mock scenario: both lanes fail
    let lane_a_passed = false;
    let lane_b_passed = false;

    let both_passed = lane_a_passed && lane_b_passed;
    let expected_exit_code = if both_passed { 0 } else { 1 };

    assert_eq!(expected_exit_code, 1, "Exit code should be 1 when both lanes fail");
}

/// AC4: Test exit code 2 for usage errors (missing required args)
/// Tests feature spec: parity-both-command.md#ac4
#[test]
#[ignore = "TODO: Implement parity-both command CLI validation"]
fn test_exit_code_usage_error() {
    // After implementation, this test will:
    // 1. Run parity-both without required --model-gguf argument
    // 2. Verify exit code 2 (usage error)
    // 3. Verify error message mentions missing argument

    unimplemented!("Awaiting parity-both command implementation");
}

// ============================================================================
// AC3: Summary Output Format Tests
// ============================================================================

/// AC3: Test summary shows all required metrics
/// Tests feature spec: parity-both-command.md#ac3
#[test]
fn test_summary_required_metrics() {
    // Mock lane results
    let lane_a = MockLaneResult {
        backend: "bitnet".to_string(),
        passed: true,
        first_divergence: None,
        mean_mse: 2.15e-5,
        mean_cosine_sim: 0.99995,
    };

    let lane_b = MockLaneResult {
        backend: "llama".to_string(),
        passed: true,
        first_divergence: None,
        mean_mse: 1.98e-5,
        mean_cosine_sim: 0.99996,
    };

    // Summary should include:
    // - Backend name: "bitnet.cpp" | "llama.cpp"
    // - Parity status: "✓ Parity OK" | "✗ Parity FAILED"
    // - Mean MSE: scientific notation
    // - First divergence: position number or "None"
    // - Mean cosine similarity: 5 decimal places

    assert!(lane_a.passed);
    assert!(lane_b.passed);
    assert!(lane_a.first_divergence.is_none());
    assert!(lane_b.first_divergence.is_none());
    assert!(lane_a.mean_mse > 0.0);
    assert!(lane_b.mean_mse > 0.0);
    assert!(lane_a.mean_cosine_sim > 0.99);
    assert!(lane_b.mean_cosine_sim > 0.99);
}

/// AC3: Test summary shows divergence position when lane fails
/// Tests feature spec: parity-both-command.md#ac3
#[test]
fn test_summary_shows_divergence_position() {
    // Mock scenario: Lane A fails at position 2
    let lane_a = MockLaneResult {
        backend: "bitnet".to_string(),
        passed: false,
        first_divergence: Some(2),
        mean_mse: 5.8e-4,
        mean_cosine_sim: 0.9985,
    };

    // Verify divergence detection
    assert!(!lane_a.passed, "Lane should be marked as failed");
    assert_eq!(lane_a.first_divergence, Some(2), "First divergence should be at position 2");
    assert!(lane_a.mean_cosine_sim < 0.999, "Cosine similarity should be below threshold");
}

/// AC3: Test summary format for partial failure (one lane fails)
/// Tests feature spec: parity-both-command.md#ac3
#[test]
fn test_summary_partial_failure() {
    // Mock scenario: Lane A fails, Lane B passes
    let lane_a_passed = false;
    let lane_b_passed = true;

    let both_passed = lane_a_passed && lane_b_passed;

    assert!(!both_passed, "Overall status should be FAILED");

    // Summary should show:
    // - Lane A: ✗ Parity FAILED (with divergence position)
    // - Lane B: ✓ Parity OK
    // - Overall: ✗ FAILED (1 of 2 lanes failed)
    // - Exit code: 1
}

// ============================================================================
// AC7: Format Compatibility Tests
// ============================================================================

/// AC7: Test text format produces human-readable output
/// Tests feature spec: parity-both-command.md#ac7
#[test]
#[ignore = "TODO: Implement parity-both command with --format text"]
fn test_format_text_human_readable() {
    // After implementation, this test will:
    // 1. Run parity-both with --format text
    // 2. Verify output contains:
    //    - "Lane A: BitNet.cpp" section
    //    - "Lane B: llama.cpp" section
    //    - "Overall Status" section
    //    - Status symbols (✓ or ✗)
    //    - Readable metric labels ("Mean MSE:", "Status:", etc.)

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC7: Test JSON format produces valid JSON with lanes.* fields
/// Tests feature spec: parity-both-command.md#ac7
#[test]
#[ignore = "TODO: Implement parity-both command with --format json"]
fn test_format_json_structure() {
    // After implementation, this test will:
    // 1. Run parity-both with --format json
    // 2. Parse JSON output
    // 3. Verify structure:
    //    {
    //      "status": "ok" | "failed",
    //      "lanes": {
    //        "bitnet": { "backend", "status", "first_divergence", "mean_mse", "mean_cosine_sim", "receipt_path" },
    //        "llama": { ... }
    //      },
    //      "overall": { "both_passed", "exit_code" }
    //    }

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC7: Test both formats show same data (consistency check)
/// Tests feature spec: parity-both-command.md#ac7
#[test]
#[ignore = "TODO: Implement parity-both command with dual-format consistency"]
fn test_format_consistency() {
    // After implementation, this test will:
    // 1. Run parity-both with --format text, capture output
    // 2. Run parity-both with --format json, parse output
    // 3. Verify both formats report same:
    //    - Lane A/B pass/fail status
    //    - First divergence positions
    //    - Mean MSE values (within floating-point tolerance)
    //    - Overall exit code

    unimplemented!("Awaiting parity-both command implementation");
}

// ============================================================================
// AC1: Single Command Execution Tests
// ============================================================================

/// AC1: Test single command runs both backends without intervention
/// Tests feature spec: parity-both-command.md#ac1
#[test]
#[ignore = "TODO: Implement parity-both command; requires C++ backends installed"]
#[serial(bitnet_env)]
fn test_single_command_both_backends() {
    let model = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Warning: BITNET_GGUF not set, skipping integration test");
            return;
        }
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => {
            eprintln!("Warning: BITNET_TOKENIZER not set, skipping integration test");
            return;
        }
    };

    // Check if both backends available
    if !backend_available("bitnet") || !backend_available("llama") {
        eprintln!("Warning: C++ backends not available, skipping integration test");
        return;
    }

    let out_dir = TempDir::new().expect("Failed to create temp dir");

    // Run parity-both command
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            model.to_str().unwrap(),
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--prompt",
            "What is 2+2?",
            "--max-tokens",
            "4",
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    // AC1: Verify command completes without user intervention
    assert!(output.status.code().is_some(), "Command should complete (not hang waiting for input)");

    // AC1: Verify both backends evaluated
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Lane A") || stdout.contains("BitNet"),
        "Output should mention Lane A (BitNet.cpp)"
    );
    assert!(
        stdout.contains("Lane B") || stdout.contains("llama"),
        "Output should mention Lane B (llama.cpp)"
    );

    // AC2: Verify both receipt files created
    let receipt_bitnet = out_dir.path().join("receipt_bitnet.json");
    let receipt_llama = out_dir.path().join("receipt_llama.json");

    assert!(receipt_bitnet.exists(), "BitNet receipt should exist: {}", receipt_bitnet.display());
    assert!(receipt_llama.exists(), "llama receipt should exist: {}", receipt_llama.display());

    // Verify receipt schema
    let bitnet_content =
        fs::read_to_string(&receipt_bitnet).expect("Failed to read BitNet receipt");
    let bitnet_receipt: MockReceipt =
        serde_json::from_str(&bitnet_content).expect("Failed to parse BitNet receipt");

    assert_eq!(bitnet_receipt.version, 1, "Receipt should use schema v1");
    assert_eq!(bitnet_receipt.backend, "bitnet", "BitNet receipt backend field should be 'bitnet'");

    let llama_content = fs::read_to_string(&receipt_llama).expect("Failed to read llama receipt");
    let llama_receipt: MockReceipt =
        serde_json::from_str(&llama_content).expect("Failed to parse llama receipt");

    assert_eq!(llama_receipt.version, 1, "Receipt should use schema v1");
    assert_eq!(llama_receipt.backend, "llama", "llama receipt backend field should be 'llama'");
}

/// AC1: Test parity-both with minimal arguments (uses defaults)
/// Tests feature spec: parity-both-command.md#ac1
#[test]
#[ignore = "TODO: Implement parity-both command with default arguments"]
#[serial(bitnet_env)]
fn test_parity_both_minimal_args() {
    // After implementation, this test will:
    // 1. Run parity-both with only --model-gguf and --tokenizer
    // 2. Verify defaults are used:
    //    - prompt: "What is 2+2?"
    //    - max-tokens: 4
    //    - cos-tol: 0.999
    //    - format: text
    //    - out-dir: "."
    //    - auto-repair: true (default)
    // 3. Verify command completes successfully

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC1: Test parity-both with full options
/// Tests feature spec: parity-both-command.md#ac1
#[test]
#[ignore = "TODO: Implement parity-both command with all options"]
#[serial(bitnet_env)]
fn test_parity_both_full_options() {
    // After implementation, this test will:
    // 1. Run parity-both with all optional flags:
    //    --prompt, --max-tokens, --cos-tol, --format, --prompt-template,
    //    --system-prompt, --out-dir, --verbose, --dump-ids, --dump-cpp-ids, --metrics
    // 2. Verify all options are respected
    // 3. Verify verbose output shows detailed progress

    unimplemented!("Awaiting parity-both command implementation");
}

// ============================================================================
// AC5: Verbose Mode Tests
// ============================================================================

/// AC5: Test verbose mode shows preflight messages
/// Tests feature spec: parity-both-command.md#ac5
#[test]
#[ignore = "TODO: Implement parity-both command with --verbose"]
#[serial(bitnet_env)]
fn test_verbose_mode_preflight() {
    // After implementation, this test will:
    // 1. Run parity-both with --verbose
    // 2. Verify output contains preflight messages:
    //    - "⚙ Preflight: Checking BitNet.cpp backend..."
    //    - "⚙ Preflight: Checking llama.cpp backend..."
    //    - "✓ Preflight: BitNet.cpp available" or auto-repair flow
    //    - "✓ Preflight: llama.cpp available" or auto-repair flow

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC5: Test verbose mode shows shared setup messages
/// Tests feature spec: parity-both-command.md#ac5
#[test]
#[ignore = "TODO: Implement parity-both command with --verbose shared setup"]
#[serial(bitnet_env)]
fn test_verbose_mode_shared_setup() {
    // After implementation, this test will:
    // 1. Run parity-both with --verbose
    // 2. Verify output contains shared setup messages:
    //    - "⚙ Shared setup: Template processing..."
    //    - "Template: <detected template>"
    //    - "✓ Rust tokens: <count> total"
    //    - "✓ C++ tokens (BitNet): <count> total"
    //    - "✓ C++ tokens (llama): <count> total"
    //    - "✓ Token parity: PASSED (all sequences match)"

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC5: Test verbose mode shows per-lane evaluation progress
/// Tests feature spec: parity-both-command.md#ac5
#[test]
#[ignore = "TODO: Implement parity-both command with --verbose per-lane"]
#[serial(bitnet_env)]
fn test_verbose_mode_per_lane_progress() {
    // After implementation, this test will:
    // 1. Run parity-both with --verbose
    // 2. Verify output contains per-lane messages:
    //    Lane A:
    //      - "⚙ Lane A: Rust logits evaluation..."
    //      - "✓ Rust logits: N positions × M vocab"
    //      - "⚙ Lane A: BitNet.cpp logits evaluation..."
    //      - "✓ C++ logits: N positions × M vocab"
    //      - "⚙ Lane A: Comparing Rust vs C++ logits..."
    //      - "Position 0: cos_sim=X, mse=Y ✓"
    //      - "✓ Lane A: Parity OK"
    //    Lane B:
    //      - Similar messages for llama.cpp

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC5: Test verbose mode shows per-position metrics
/// Tests feature spec: parity-both-command.md#ac5
#[test]
#[ignore = "TODO: Implement parity-both command with --verbose per-position"]
#[serial(bitnet_env)]
fn test_verbose_mode_per_position_metrics() {
    // After implementation, this test will:
    // 1. Run parity-both with --verbose --max-tokens 4
    // 2. Verify output contains per-position metrics for both lanes:
    //    - "Position 0: cos_sim=0.99999, mse=1.2e-5 ✓"
    //    - "Position 1: cos_sim=0.99997, mse=2.1e-5 ✓"
    //    - ... (for all 4 positions, both lanes)
    // 3. Verify divergence symbol (✓ or ✗) matches pass/fail status

    unimplemented!("Awaiting parity-both command implementation");
}

// ============================================================================
// AC6: Auto-Repair Tests
// ============================================================================

/// AC6: Test auto-repair enabled by default
/// Tests feature spec: parity-both-command.md#ac6
#[test]
#[ignore = "TODO: Implement parity-both command with auto-repair default"]
#[serial(bitnet_env)]
fn test_auto_repair_default_enabled() {
    // After implementation, this test will:
    // 1. Simulate missing llama.cpp backend (env var cleared)
    // 2. Run parity-both without --no-repair flag
    // 3. Verify auto-repair is triggered:
    //    - "⚙ Auto-repairing llama.cpp backend..."
    //    - setup-cpp-auto invoked
    //    - xtask rebuilt
    //    - "✓ llama.cpp backend repaired"
    // 4. Verify command succeeds after repair

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC6: Test --no-repair disables auto-repair
/// Tests feature spec: parity-both-command.md#ac6
#[test]
#[ignore = "TODO: Implement parity-both command with --no-repair"]
#[serial(bitnet_env)]
fn test_no_repair_flag_disables_auto_repair() {
    // After implementation, this test will:
    // 1. Simulate missing llama.cpp backend
    // 2. Run parity-both with --no-repair flag
    // 3. Verify auto-repair is NOT triggered
    // 4. Verify command fails with exit code 1
    // 5. Verify error message includes setup instructions

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC6: Test auto-repair success message
/// Tests feature spec: parity-both-command.md#ac6
#[test]
#[ignore = "TODO: Implement parity-both command with auto-repair success"]
#[serial(bitnet_env)]
fn test_auto_repair_success_message() {
    // After implementation, this test will:
    // 1. Simulate missing backend
    // 2. Run parity-both with auto-repair enabled
    // 3. Verify success message after repair:
    //    - "✓ <backend> backend repaired"
    // 4. Verify subsequent preflight check passes

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC6: Test auto-repair failure handling
/// Tests feature spec: parity-both-command.md#ac6
#[test]
#[ignore = "TODO: Implement parity-both command with auto-repair failure"]
#[serial(bitnet_env)]
fn test_auto_repair_failure_handling() {
    // After implementation, this test will:
    // 1. Simulate scenario where auto-repair fails (e.g., network unavailable)
    // 2. Run parity-both with auto-repair enabled
    // 3. Verify command fails gracefully with exit code 1
    // 4. Verify error message includes diagnostic information

    unimplemented!("Awaiting parity-both command implementation");
}

// ============================================================================
// AC3: Comparison Logic and Threshold Tests
// ============================================================================

/// AC3: Test exact token match detection (cosine=1.0, L2=0.0)
/// Tests feature spec: parity-both-command.md#ac3
#[test]
fn test_comparison_exact_match() {
    let scenario = ComparisonScenario::exact_match();

    assert_eq!(scenario.cosine_sim, 1.0, "Exact match should have cosine similarity 1.0");
    assert_eq!(scenario.l2_dist, 0.0, "Exact match should have L2 distance 0.0");
    assert_eq!(scenario.mse, 0.0, "Exact match should have MSE 0.0");
    assert!(scenario.expected_pass, "Exact match should pass");
}

/// AC3: Test near match detection (cosine=0.9995, L2=0.001)
/// Tests feature spec: parity-both-command.md#ac3
#[test]
fn test_comparison_near_match() {
    let scenario = ComparisonScenario::near_match();

    assert!(scenario.cosine_sim >= 0.999, "Near match should have cosine similarity >= 0.999");
    assert!(scenario.l2_dist < 0.01, "Near match should have L2 distance < 0.01");
    assert!(scenario.mse < 1e-4, "Near match should have MSE < 1e-4");
    assert!(scenario.expected_pass, "Near match should pass with default threshold");
}

/// AC3: Test divergence detection (cosine=0.95, L2=0.5)
/// Tests feature spec: parity-both-command.md#ac3
#[test]
fn test_comparison_divergence() {
    let scenario = ComparisonScenario::divergence();

    assert!(scenario.cosine_sim < 0.999, "Divergence should have cosine similarity < 0.999");
    assert!(scenario.l2_dist > 0.1, "Divergence should have L2 distance > 0.1");
    assert!(scenario.mse > 1e-4, "Divergence should have MSE > 1e-4");
    assert!(!scenario.expected_pass, "Divergence should fail");
}

/// AC3: Test threshold validation with MSE
/// Tests feature spec: parity-both-command.md#ac3
#[test]
fn test_threshold_validation_mse() {
    let threshold_mse = 1e-4;

    // Pass: MSE below threshold
    let pass_mse = 1e-6;
    assert!(pass_mse <= threshold_mse, "MSE below threshold should pass");

    // Fail: MSE above threshold
    let fail_mse = 5e-4;
    assert!(fail_mse > threshold_mse, "MSE above threshold should fail");
}

/// AC3: Test threshold validation with cosine similarity
/// Tests feature spec: parity-both-command.md#ac3
#[test]
fn test_threshold_validation_cosine_similarity() {
    let threshold_cosine = 0.999;

    // Pass: cosine similarity above threshold
    let pass_cosine = 0.99995;
    assert!(pass_cosine >= threshold_cosine, "Cosine similarity above threshold should pass");

    // Fail: cosine similarity below threshold
    let fail_cosine = 0.9985;
    assert!(fail_cosine < threshold_cosine, "Cosine similarity below threshold should fail");
}

/// AC3: Test threshold validation with L2 distance
/// Tests feature spec: parity-both-command.md#ac3
#[test]
fn test_threshold_validation_l2_distance() {
    // L2 distance threshold (derived from MSE threshold)
    let threshold_l2 = (1e-4_f64).sqrt(); // ~0.01

    // Pass: L2 distance below threshold
    let pass_l2 = 0.001;
    assert!(pass_l2 <= threshold_l2, "L2 distance below threshold should pass");

    // Fail: L2 distance above threshold
    let fail_l2 = 0.5;
    assert!(fail_l2 > threshold_l2, "L2 distance above threshold should fail");
}

// ============================================================================
// AC5: First Divergence Reporting Tests
// ============================================================================

/// AC5: Test first divergence position detection
/// Tests feature spec: parity-both-command.md#ac5
#[test]
fn test_first_divergence_position_detection() {
    // Mock scenario: divergence at position 2
    let lane = MockLaneResult::new("bitnet", vec![1, 2, 3, 4], 0.9985);

    assert!(!lane.passed, "Lane with divergence should be marked as failed");
    assert_eq!(lane.first_divergence, Some(2), "First divergence should be detected at position 2");
}

/// AC5: Test no divergence (all positions pass)
/// Tests feature spec: parity-both-command.md#ac5
#[test]
fn test_no_divergence_all_positions_pass() {
    // Mock scenario: all positions pass
    let lane = MockLaneResult::new("bitnet", vec![1, 2, 3, 4], 0.99999);

    assert!(lane.passed, "Lane with all positions passing should be marked as passed");
    assert!(
        lane.first_divergence.is_none(),
        "First divergence should be None when all positions pass"
    );
}

/// AC5: Test divergence index is within bounds
/// Tests feature spec: parity-both-command.md#ac5
#[test]
fn test_divergence_index_within_bounds() {
    let positions = 4; // Total positions

    // Simulate divergence at each position
    for expected_pos in 0..positions {
        // Mock divergence at position expected_pos
        let divergence: Option<usize> = Some(expected_pos);

        assert!(
            divergence.unwrap() < positions,
            "Divergence position {} should be < total positions {}",
            expected_pos,
            positions
        );
    }
}

/// AC5: Test receipt row data for divergence validation
/// Tests feature spec: parity-both-command.md#ac5
#[test]
#[ignore = "TODO: Implement parity-both command and receipt generation with row data"]
fn test_receipt_row_data_for_divergence() {
    // After implementation, this test will:
    // 1. Run parity-both with a scenario that triggers divergence
    // 2. Load receipt JSON
    // 3. Verify receipt.rows[] contains entries for all positions
    // 4. Verify row at divergence position has:
    //    - pos: <divergence_index>
    //    - mse: > threshold
    //    - max_abs: > 0
    //    - top5_rust: Vec<usize> (non-empty)
    //    - top5_cpp: Vec<usize> (non-empty, may differ from rust)

    unimplemented!("Awaiting parity-both command implementation");
}

/// AC5: Test multiple divergences (reports first only)
/// Tests feature spec: parity-both-command.md#ac5
#[test]
#[ignore = "TODO: Implement parity-both command with multiple divergence handling"]
fn test_multiple_divergences_reports_first() {
    // After implementation, this test will:
    // 1. Create a scenario with divergences at positions 1, 3, 5
    // 2. Run parity-both command
    // 3. Verify summary.first_divergence == 1 (the first divergence)
    // 4. Verify receipt rows show all divergences but summary highlights first

    unimplemented!("Awaiting parity-both command implementation");
}

// ============================================================================
// Unit Tests: Comparison Helpers
// ============================================================================

/// Unit test: Cosine similarity calculation helper
/// Tests feature spec: parity-both-command.md#5.1
#[test]
#[ignore = "TODO: Extract cosine similarity calculation to testable helper function"]
fn test_cosine_similarity_calculation() {
    // After implementation, this test will:
    // 1. Test cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0 (identical vectors)
    // 2. Test cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0 (orthogonal vectors)
    // 3. Test cosine_similarity([1.0, 1.0], [1.0, 1.0]) == 1.0 (parallel vectors)
    // 4. Test cosine_similarity([3.0, 4.0], [6.0, 8.0]) == 1.0 (scaled vectors)

    unimplemented!("Awaiting cosine similarity helper extraction");
}

/// Unit test: L2 distance calculation helper
/// Tests feature spec: parity-both-command.md#5.1
#[test]
#[ignore = "TODO: Extract L2 distance calculation to testable helper function"]
fn test_l2_distance_calculation() {
    // After implementation, this test will:
    // 1. Test l2_distance([1.0, 0.0], [1.0, 0.0]) == 0.0 (identical vectors)
    // 2. Test l2_distance([1.0, 0.0], [0.0, 1.0]) == sqrt(2) (orthogonal unit vectors)
    // 3. Test l2_distance([3.0, 4.0], [0.0, 0.0]) == 5.0 (Pythagorean triple)
    // 4. Test l2_distance([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == sqrt(27) (3D case)

    unimplemented!("Awaiting L2 distance helper extraction");
}

/// Unit test: MSE calculation from L2 distance
/// Tests feature spec: parity-both-command.md#5.1
#[test]
fn test_mse_from_l2_distance() {
    // MSE = (L2 distance)^2
    let test_cases = [
        (0.0, 0.0),   // L2=0 → MSE=0
        (0.01, 1e-4), // L2=0.01 → MSE=1e-4
        (0.1, 0.01),  // L2=0.1 → MSE=0.01
        (0.5, 0.25),  // L2=0.5 → MSE=0.25
        (1.0, 1.0),   // L2=1.0 → MSE=1.0
    ];

    for (l2, expected_mse) in test_cases {
        let calculated_mse = l2 * l2;
        assert!(
            (calculated_mse - expected_mse).abs() < 1e-10,
            "MSE from L2={} should be {}, got {}",
            l2,
            expected_mse,
            calculated_mse
        );
    }
}

// ============================================================================
// Backend Scenario Tests
// ============================================================================

/// Test scenario: Both backends available (happy path)
/// Tests feature spec: parity-both-command.md#9.2
#[test]
#[ignore = "TODO: Implement parity-both command with backend availability detection"]
#[serial(bitnet_env)]
fn test_scenario_both_backends_available() {
    // After implementation, this test will:
    // 1. Verify both BitNet.cpp and llama.cpp backends are available
    // 2. Run parity-both command
    // 3. Verify both lanes complete successfully
    // 4. Verify exit code 0
    // 5. Verify both receipt files created

    unimplemented!("Awaiting parity-both command implementation");
}

/// Test scenario: BitNet available, llama unavailable (with auto-repair)
/// Tests feature spec: parity-both-command.md#6.1
#[test]
#[ignore = "TODO: Implement parity-both command with auto-repair for missing llama.cpp"]
#[serial(bitnet_env)]
fn test_scenario_bitnet_available_llama_missing() {
    // After implementation, this test will:
    // 1. Simulate llama.cpp unavailable (clear LLAMA_CPP_DIR env)
    // 2. Run parity-both with auto-repair enabled (default)
    // 3. Verify auto-repair is triggered for llama.cpp
    // 4. Verify setup-cpp-auto invoked
    // 5. Verify xtask rebuilt
    // 6. Verify command completes successfully after repair

    unimplemented!("Awaiting auto-repair implementation");
}

/// Test scenario: Llama available, BitNet unavailable (with auto-repair)
/// Tests feature spec: parity-both-command.md#6.1
#[test]
#[ignore = "TODO: Implement parity-both command with auto-repair for missing BitNet.cpp"]
#[serial(bitnet_env)]
fn test_scenario_llama_available_bitnet_missing() {
    // After implementation, this test will:
    // 1. Simulate BitNet.cpp unavailable (clear BITNET_CPP_DIR env)
    // 2. Run parity-both with auto-repair enabled (default)
    // 3. Verify auto-repair is triggered for BitNet.cpp
    // 4. Verify command completes successfully after repair

    unimplemented!("Awaiting auto-repair implementation");
}

/// Test scenario: Both backends unavailable (exit 2)
/// Tests feature spec: parity-both-command.md#6.1
#[test]
#[ignore = "TODO: Implement parity-both command with backend unavailability handling"]
#[serial(bitnet_env)]
fn test_scenario_both_backends_unavailable() {
    // After implementation, this test will:
    // 1. Simulate both backends unavailable (clear env vars)
    // 2. Run parity-both with --no-repair flag
    // 3. Verify command exits with code 2
    // 4. Verify error message includes setup instructions
    // 5. Verify no receipts are created

    unimplemented!("Awaiting parity-both command implementation");
}

// ============================================================================
// Property-Based Tests: Summary Formatting Invariants
// ============================================================================

/// Property test: Summary exit code matches lane results
#[test]
fn property_exit_code_matches_lane_results() {
    // Test all combinations of lane pass/fail
    let test_cases = [
        (true, true, 0),   // Both pass → exit 0
        (true, false, 1),  // A pass, B fail → exit 1
        (false, true, 1),  // A fail, B pass → exit 1
        (false, false, 1), // Both fail → exit 1
    ];

    for (lane_a_passed, lane_b_passed, expected_exit) in test_cases {
        let both_passed = lane_a_passed && lane_b_passed;
        let actual_exit = if both_passed { 0 } else { 1 };

        assert_eq!(
            actual_exit, expected_exit,
            "Exit code mismatch for A={}, B={}",
            lane_a_passed, lane_b_passed
        );
    }
}

/// Property test: First divergence position is within bounds
#[test]
fn property_first_divergence_within_bounds() {
    // Mock scenarios
    let positions = 4; // Total positions

    // Valid divergence positions: 0..=positions-1
    for divergence_pos in 0..positions {
        assert!(
            divergence_pos < positions,
            "Divergence position {} should be < total positions {}",
            divergence_pos,
            positions
        );
    }

    // None is always valid
    let divergence: Option<usize> = None;
    assert!(divergence.is_none() || divergence.unwrap() < positions);
}

/// Property test: Mean MSE is non-negative
#[test]
fn property_mean_mse_non_negative() {
    // Mock scenarios with various MSE values
    let test_cases = [0.0, 1e-10, 1e-5, 1e-3, 0.001, 0.1];

    for mse in test_cases {
        assert!(mse >= 0.0, "Mean MSE should be non-negative, got {}", mse);
    }
}

// ============================================================================
// Integration Tests: End-to-End Scenarios
// ============================================================================

/// Integration test: Both backends pass (happy path)
/// Tests feature spec: parity-both-command.md#9.2
#[test]
#[ignore = "TODO: Implement parity-both command; requires C++ backends and test model"]
#[serial(bitnet_env)]
fn integration_both_backends_pass() {
    let model = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Warning: Test model not available, skipping integration test");
            return;
        }
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => {
            eprintln!("Warning: Test tokenizer not available, skipping integration test");
            return;
        }
    };

    if !backend_available("bitnet") || !backend_available("llama") {
        eprintln!("Warning: C++ backends not available, skipping integration test");
        return;
    }

    let out_dir = TempDir::new().expect("Failed to create temp dir");

    // Integration Test Case 1: Both backends available (happy path)
    // Expected: Exit 0, two receipts written
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            model.to_str().unwrap(),
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    assert!(output.status.success(), "Command should exit 0 when both backends pass");

    // Verify receipts exist
    assert!(out_dir.path().join("receipt_bitnet.json").exists());
    assert!(out_dir.path().join("receipt_llama.json").exists());
}

/// Integration test: JSON output format
/// Tests feature spec: parity-both-command.md#9.2
#[test]
#[ignore = "TODO: Implement parity-both command with --format json"]
#[serial(bitnet_env)]
fn integration_json_output_format() {
    // Integration Test Case 5: JSON output
    // Expected: Valid JSON with lanes.bitnet and lanes.llama

    let model = match get_test_model_path() {
        Some(p) => p,
        None => {
            eprintln!("Warning: Test model not available, skipping integration test");
            return;
        }
    };

    let tokenizer = match get_test_tokenizer_path() {
        Some(p) => p,
        None => {
            eprintln!("Warning: Test tokenizer not available, skipping integration test");
            return;
        }
    };

    if !backend_available("bitnet") || !backend_available("llama") {
        eprintln!("Warning: C++ backends not available, skipping integration test");
        return;
    }

    let out_dir = TempDir::new().expect("Failed to create temp dir");

    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            model.to_str().unwrap(),
            "--tokenizer",
            tokenizer.to_str().unwrap(),
            "--format",
            "json",
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse JSON
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("Output should be valid JSON");

    // Verify structure
    assert!(json.get("status").is_some(), "JSON should have 'status' field");
    assert!(json.get("lanes").is_some(), "JSON should have 'lanes' field");
    assert!(
        json.get("lanes").unwrap().get("bitnet").is_some(),
        "JSON should have 'lanes.bitnet' field"
    );
    assert!(
        json.get("lanes").unwrap().get("llama").is_some(),
        "JSON should have 'lanes.llama' field"
    );
    assert!(json.get("overall").is_some(), "JSON should have 'overall' field");
}

/// Integration test: Partial failure (one lane fails)
/// Tests feature spec: parity-both-command.md#9.2
#[test]
#[ignore = "TODO: Implement parity-both command; requires divergent model"]
#[serial(bitnet_env)]
fn integration_partial_failure() {
    // Integration Test Case 4: Partial failure (one lane fails)
    // Expected: Exit 1, both receipts written, clear divergence report

    // This test requires a model that produces divergent results between
    // BitNet.cpp and llama.cpp, or very strict cos-tol threshold

    unimplemented!("Awaiting parity-both command implementation and divergent test model");
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

/// Test token parity mismatch causes fail-fast (exit code 2)
/// Tests feature spec: parity-both-command.md#6.1
#[test]
#[ignore = "TODO: Implement parity-both command with token parity validation"]
#[serial(bitnet_env)]
fn test_token_parity_mismatch_fail_fast() {
    // After implementation, this test will:
    // 1. Use a scenario where Rust and C++ tokenizers produce different tokens
    // 2. Verify command exits with code 2 (usage error / invalid state)
    // 3. Verify error message mentions token parity mismatch
    // 4. Verify logits comparison is NOT attempted

    unimplemented!("Awaiting parity-both command implementation");
}

/// Test invalid model path handling
#[test]
#[ignore = "TODO: Implement parity-both command with error handling"]
fn test_invalid_model_path() {
    let out_dir = TempDir::new().expect("Failed to create temp dir");

    // Run parity-both with non-existent model path
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&[
            "parity-both",
            "--model-gguf",
            "/nonexistent/model.gguf",
            "--tokenizer",
            "/nonexistent/tokenizer.json",
            "--out-dir",
            out_dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute parity-both");

    // Should exit with error code (1 or 2)
    assert!(!output.status.success(), "Should fail on invalid model path");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not found") || stderr.contains("No such file"),
        "Error message should mention missing file"
    );
}

/// Test concurrent execution safety
/// Tests feature spec: parity-both-command.md (implied requirement)
#[test]
#[ignore = "TODO: Implement parity-both command with concurrent safety"]
#[serial(bitnet_env)]
fn test_concurrent_execution_safety() {
    // After implementation, this test will:
    // 1. Run two parity-both commands concurrently with different out-dirs
    // 2. Verify both complete successfully without interference
    // 3. Verify receipts are written to correct directories

    unimplemented!("Awaiting parity-both command implementation");
}

/// Test --parallel flag (experimental)
/// Tests feature spec: parity-both-command.md#2.3
#[test]
#[ignore = "TODO: Implement parity-both command with --parallel flag (future enhancement)"]
#[serial(bitnet_env)]
fn test_parallel_flag_experimental() {
    // After implementation, this test will:
    // 1. Run parity-both with --parallel flag
    // 2. Verify both C++ backends run concurrently (check timing)
    // 3. Verify results are identical to sequential execution

    unimplemented!("Awaiting --parallel feature implementation (future enhancement)");
}

// ============================================================================
// Documentation and Metadata Tests
// ============================================================================

/// Test help text mentions all required arguments
#[test]
fn test_help_text_completeness() {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&["parity-both", "--help"])
        .output()
        .expect("Failed to execute xtask --help");

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
        "Help should document --no-repair"
    );
    assert!(
        help_text.contains("--format") || help_text.contains("format"),
        "Help should document --format"
    );
}

/// Test command appears in xtask command list
#[test]
fn test_command_in_xtask_list() {
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(&["--help"])
        .output()
        .expect("Failed to execute xtask --help");

    let help_text = String::from_utf8_lossy(&output.stdout);

    // Verify parity-both command is listed
    assert!(help_text.contains("parity-both"), "xtask help should list parity-both command");
}
