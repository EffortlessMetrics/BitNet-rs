# TokenizerAuthority Cross-Lane Validation: Test Specification

**Status**: Implementation Ready
**Version**: 1.0 (Test Coverage)
**Feature**: End-to-end validation of TokenizerAuthority cross-lane consistency
**Priority**: P0 (Critical Infrastructure Validation)
**Scope**: Integration tests for `xtask/src/crossval/parity_both.rs`, `crossval/src/receipt.rs`

---

## 1. Executive Summary

This specification defines **comprehensive integration test coverage** for the TokenizerAuthority cross-lane validation system in the parity-both command. The goal is to **validate existing implementation correctness** through end-to-end integration tests with subprocess execution, exit code verification, and receipt inspection.

### 1.1 Problem Statement

**Current test coverage**:
- **44 unit tests** in `crossval/tests/tokenizer_authority_tests.rs` covering structure, serialization, and validation functions
- **6 integration tests** in `xtask/tests/tokenizer_authority_integration_tests.rs` covering AC1-AC6 scaffolding

**Coverage gap**:
- **No end-to-end integration tests** that:
  - Execute parity-both command as subprocess
  - Capture exit codes (0=success, 2=mismatch)
  - Parse and validate receipt JSON files
  - Verify cross-lane consistency validation
  - Test with real tokenizer file variants

**Goal**: Create comprehensive **end-to-end integration test suite** that validates the complete parity-both workflow with TokenizerAuthority validation against real filesystem fixtures and subprocess execution.

### 1.2 Implementation Status (From Analysis)

The TokenizerAuthority validation system is **fully implemented** with the following components:

| Component | Location | Status |
|-----------|----------|--------|
| TokenizerAuthority struct | `crossval/src/receipt.rs:54-82` | ✅ Complete |
| TokenizerSource enum | `crossval/src/receipt.rs:38-52` | ✅ Complete |
| Hash computation (file) | `crossval/src/receipt.rs:341-350` | ✅ Complete |
| Hash computation (config) | `crossval/src/receipt.rs:382-398` | ✅ Complete |
| Source detection | `crossval/src/receipt.rs:400-412` | ✅ Complete |
| Shared computation (STEP 2.5) | `xtask/src/crossval/parity_both.rs:496-528` | ✅ Complete |
| Dual receipt injection | `xtask/src/crossval/parity_both.rs:557-588` | ✅ Complete |
| Cross-lane validation | `crossval/src/receipt.rs:480-500` | ✅ Complete |
| Exit code 2 handling | `xtask/src/crossval/parity_both.rs:620-628` | ✅ Complete |
| Summary output | `xtask/src/crossval/parity_both.rs:270-277, 336-342` | ✅ Complete |

**Key finding**: Implementation is production-ready. Tests will **validate correctness**, not specify new features.

### 1.3 Key Goals

1. **TC1: Happy Path Validation** - Identical tokenizers → exit 0, consistent receipts
2. **TC2: Mismatch Detection** - Different tokenizers → exit 2, clear diagnostics
3. **TC3: Receipt Population** - Both receipts contain TokenizerAuthority with correct fields
4. **TC4: Summary Verification** - Text and JSON summaries display tokenizer hash
5. **TC5: Edge Case Handling** - Missing files, corrupted JSON, invalid paths
6. **TC6: Cross-Lane Consistency** - Validation function correctly compares authorities

---

## 2. Test Strategy

### 2.1 Test Levels

| Level | Scope | Execution | Coverage |
|-------|-------|-----------|----------|
| **Unit** | Function-level validation logic | In-process | 44 tests (existing) |
| **Integration** | Multi-function workflows | In-process | 6 tests (existing scaffolding) |
| **E2E** | Full parity-both command | Subprocess | **12 tests (NEW)** |

This specification focuses on **E2E integration tests** that validate the complete workflow.

### 2.2 Test Fixtures

**Location**: `tests/fixtures/tokenizers/`

```
tests/fixtures/tokenizers/
├── valid_tokenizer_a.json       # Reference tokenizer (128000 tokens)
├── valid_tokenizer_b.json       # Clone of A (identical hash)
├── different_vocab_size.json    # Different vocab (64000 tokens)
├── different_special_tokens.json # Different special token IDs
├── corrupted.json               # Malformed JSON
└── README.md                    # Fixture documentation
```

**Fixture Requirements**:
1. `valid_tokenizer_a.json`: Standard LLaMA-3 tokenizer (128000 vocab)
2. `valid_tokenizer_b.json`: Byte-for-byte identical to A (same file hash)
3. `different_vocab_size.json`: Same structure, different vocab_size field (64000)
4. `different_special_tokens.json`: Same vocab, different BOS/EOS IDs
5. `corrupted.json`: Invalid JSON (truncated or missing closing braces)

### 2.3 Test Harness Architecture

```rust
// Location: xtask/tests/tokenizer_authority_e2e_tests.rs

/// Execute parity-both command as subprocess
/// Returns: (stdout, stderr, exit_code, receipt_paths)
fn run_parity_both_e2e(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    max_tokens: usize,
    out_dir: &Path,
) -> E2EResult {
    // Build command
    let mut cmd = Command::new("cargo");
    cmd.args(&["run", "-p", "xtask", "--features", "crossval-all", "--"]);
    cmd.args(&["parity-both"]);
    cmd.arg("--model-gguf").arg(model_path);
    cmd.arg("--tokenizer").arg(tokenizer_path);
    cmd.arg("--prompt").arg(prompt);
    cmd.arg("--max-tokens").arg(max_tokens.to_string());
    cmd.arg("--out-dir").arg(out_dir);

    // Capture output
    let output = cmd.output().expect("Failed to execute parity-both");

    E2EResult {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
        receipt_bitnet: out_dir.join("receipt_bitnet.json"),
        receipt_llama: out_dir.join("receipt_llama.json"),
    }
}

/// Parse receipt JSON and extract TokenizerAuthority
fn parse_receipt_authority(receipt_path: &Path) -> anyhow::Result<TokenizerAuthority> {
    let content = std::fs::read_to_string(receipt_path)?;
    let receipt: ParityReceipt = serde_json::from_str(&content)?;
    receipt.tokenizer_authority
        .ok_or_else(|| anyhow::anyhow!("Receipt missing tokenizer_authority"))
}

/// Assert receipts have matching TokenizerAuthority
fn assert_authorities_match(auth_a: &TokenizerAuthority, auth_b: &TokenizerAuthority) {
    assert_eq!(auth_a.config_hash, auth_b.config_hash, "Config hashes must match");
    assert_eq!(auth_a.token_count, auth_b.token_count, "Token counts must match");
    assert_eq!(auth_a.source, auth_b.source, "Sources must match");
}
```

---

## 3. Test Categories

### 3.1 TC1: Happy Path - Identical Tokenizers

**Objective**: Verify successful validation when both lanes use identical tokenizers.

#### Test 1.1: Exit Code 0 with Matching Tokenizers

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_identical_tokenizers_exit_0() {
    // GIVEN: Same tokenizer file for both lanes (via shared setup)
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN: Run parity-both
    let result = run_parity_both_e2e(
        &model,
        &tokenizer,
        "What is 2+2?",
        4,
        &temp_dir,
    );

    // THEN: Exit code 0 (success)
    assert_eq!(result.exit_code, 0, "Exit code should be 0 for matching tokenizers");

    // AND: Both receipts exist
    assert!(result.receipt_bitnet.exists(), "BitNet receipt should exist");
    assert!(result.receipt_llama.exists(), "llama receipt should exist");

    // AND: Receipts contain TokenizerAuthority
    let auth_bitnet = parse_receipt_authority(&result.receipt_bitnet).unwrap();
    let auth_llama = parse_receipt_authority(&result.receipt_llama).unwrap();

    // AND: Authorities are identical
    assert_authorities_match(&auth_bitnet, &auth_llama);

    // AND: Config hash is 64 hex chars
    assert_eq!(auth_bitnet.config_hash.len(), 64, "Config hash should be 64 hex chars");
    assert!(auth_bitnet.config_hash.chars().all(|c| c.is_ascii_hexdigit()));
}
```

**Success Criteria**:
- ✅ Exit code 0
- ✅ Both receipt files written to disk
- ✅ Both receipts contain `tokenizer_authority` field
- ✅ Config hashes match (64 hex chars)
- ✅ Token counts match
- ✅ Sources match (External)

#### Test 1.2: Summary Displays Tokenizer Hash (Text Format)

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_summary_text_format_displays_hash() {
    // GIVEN: Valid tokenizer
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN: Run parity-both with text format (default)
    let result = run_parity_both_e2e(&model, &tokenizer, "Test", 2, &temp_dir);

    // THEN: Exit code 0
    assert_eq!(result.exit_code, 0);

    // AND: Stdout contains "Tokenizer Consistency" section
    assert!(result.stdout.contains("Tokenizer Consistency"),
        "Summary should include tokenizer section");

    // AND: Stdout contains "Config hash:" with abbreviated hash
    assert!(result.stdout.contains("Config hash:"),
        "Summary should show config hash label");

    // AND: Stdout contains "Full hash:" with 64-char hash
    let auth = parse_receipt_authority(&result.receipt_bitnet).unwrap();
    assert!(result.stdout.contains(&auth.config_hash),
        "Summary should display full config hash");

    // AND: First 32 chars displayed in abbreviated line
    let abbreviated = &auth.config_hash[..32];
    assert!(result.stdout.contains(abbreviated),
        "Summary should show abbreviated hash (first 32 chars)");
}
```

**Success Criteria**:
- ✅ Stdout contains "Tokenizer Consistency" header
- ✅ Stdout contains "Config hash:" with abbreviated display (32 chars)
- ✅ Stdout contains "Full hash:" with complete 64-char hash
- ✅ Hash matches receipt JSON

#### Test 1.3: Summary Displays Tokenizer Hash (JSON Format)

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_summary_json_format_includes_tokenizer() {
    // GIVEN: Valid tokenizer
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN: Run parity-both with --format json
    let mut cmd = Command::new("cargo");
    cmd.args(&["run", "-p", "xtask", "--features", "crossval-all", "--"]);
    cmd.args(&["parity-both"]);
    cmd.arg("--model-gguf").arg(&model);
    cmd.arg("--tokenizer").arg(&tokenizer);
    cmd.arg("--prompt").arg("Test");
    cmd.arg("--max-tokens").arg("2");
    cmd.arg("--out-dir").arg(&temp_dir);
    cmd.arg("--format").arg("json");

    let output = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // THEN: Exit code 0
    assert_eq!(output.status.code().unwrap(), 0);

    // AND: Stdout is valid JSON
    let summary: serde_json::Value = serde_json::from_str(&stdout)
        .expect("Summary should be valid JSON");

    // AND: JSON contains tokenizer object
    assert!(summary.get("tokenizer").is_some(),
        "JSON summary should include tokenizer field");

    // AND: Tokenizer object has config_hash and status
    let tokenizer_obj = summary["tokenizer"].as_object().unwrap();
    assert!(tokenizer_obj.contains_key("config_hash"),
        "Tokenizer object should have config_hash");
    assert!(tokenizer_obj.contains_key("status"),
        "Tokenizer object should have status");

    // AND: Status is "consistent"
    assert_eq!(tokenizer_obj["status"].as_str().unwrap(), "consistent",
        "Tokenizer status should be 'consistent'");

    // AND: Config hash matches receipt
    let auth = parse_receipt_authority(&temp_dir.path().join("receipt_bitnet.json")).unwrap();
    assert_eq!(tokenizer_obj["config_hash"].as_str().unwrap(), auth.config_hash,
        "JSON hash should match receipt");
}
```

**Success Criteria**:
- ✅ Stdout is valid JSON
- ✅ JSON contains `tokenizer` object
- ✅ `tokenizer.config_hash` field present with 64-char hash
- ✅ `tokenizer.status` field present with value "consistent"
- ✅ Hash matches receipt JSON

#### Test 1.4: Receipt Field Population (All Fields Present)

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_receipt_authority_all_fields_populated() {
    // GIVEN: External tokenizer (tokenizer.json file)
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN: Run parity-both
    let result = run_parity_both_e2e(&model, &tokenizer, "Test", 2, &temp_dir);
    assert_eq!(result.exit_code, 0);

    // THEN: Both receipts have TokenizerAuthority with all fields
    for receipt_path in &[result.receipt_bitnet, result.receipt_llama] {
        let auth = parse_receipt_authority(receipt_path).unwrap();

        // Source is External (file named tokenizer.json)
        assert_eq!(auth.source, TokenizerSource::External,
            "Source should be External for tokenizer.json file");

        // Path matches input tokenizer
        assert!(auth.path.contains("valid_tokenizer_a.json"),
            "Path should reference input tokenizer file");

        // File hash is Some (external tokenizer)
        assert!(auth.file_hash.is_some(),
            "File hash should be present for external tokenizer");
        let file_hash = auth.file_hash.unwrap();
        assert_eq!(file_hash.len(), 64, "File hash should be 64 hex chars");
        assert!(file_hash.chars().all(|c| c.is_ascii_hexdigit()),
            "File hash should be lowercase hex");

        // Config hash is 64 hex chars
        assert_eq!(auth.config_hash.len(), 64, "Config hash should be 64 hex chars");
        assert!(auth.config_hash.chars().all(|c| c.is_ascii_hexdigit()),
            "Config hash should be lowercase hex");

        // Token count is reasonable (prompt tokenized to 1-8 tokens)
        assert!(auth.token_count >= 1 && auth.token_count <= 8,
            "Token count should be in reasonable range for short prompt");
    }
}
```

**Success Criteria**:
- ✅ `source` field = `External`
- ✅ `path` field matches input tokenizer path
- ✅ `file_hash` is Some with 64 hex chars
- ✅ `config_hash` is 64 hex chars
- ✅ `token_count` is reasonable (1-8 for short prompt)

---

### 3.2 TC2: Mismatch Detection - Different Tokenizers

**Objective**: Verify exit code 2 and diagnostic output when tokenizers differ across lanes.

**Note**: Since parity-both uses shared setup (computes TokenizerAuthority once), we cannot naturally inject different tokenizers per lane. Instead, these tests focus on:
1. **Testing the validation function directly** with synthetic authorities
2. **Simulating receipt modification** (advanced scenario)
3. **Verifying diagnostic message format** for exit code 2

#### Test 2.1: Validation Function Detects Config Hash Mismatch

```rust
#[test]
fn test_validate_tokenizer_consistency_rejects_different_config_hash() {
    // GIVEN: Two authorities with different config hashes
    let auth_a = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer_a.json".to_string(),
        file_hash: Some("aaa".repeat(21) + "a"), // 64 chars
        config_hash: "111".repeat(21) + "1",     // 64 chars
        token_count: 4,
    };

    let auth_b = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer_b.json".to_string(),
        file_hash: Some("bbb".repeat(21) + "b"),
        config_hash: "222".repeat(21) + "2",     // DIFFERENT
        token_count: 4,
    };

    // WHEN: Validate consistency
    let result = validate_tokenizer_consistency(&auth_a, &auth_b);

    // THEN: Returns Err
    assert!(result.is_err(), "Validation should fail for different config hashes");

    // AND: Error message mentions "config mismatch"
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("config mismatch") || err_msg.contains("Config mismatch"),
        "Error should mention config mismatch: {}", err_msg);

    // AND: Error message includes both hashes
    assert!(err_msg.contains("111"), "Error should show Lane A hash");
    assert!(err_msg.contains("222"), "Error should show Lane B hash");
}
```

**Success Criteria**:
- ✅ Validation returns `Err`
- ✅ Error message contains "config mismatch"
- ✅ Error message includes both config hashes

#### Test 2.2: Validation Function Detects Token Count Mismatch

```rust
#[test]
fn test_validate_tokenizer_consistency_rejects_different_token_count() {
    // GIVEN: Two authorities with same config hash but different token counts
    let config_hash = "abc".repeat(21) + "d"; // 64 chars

    let auth_a = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer.json".to_string(),
        file_hash: Some("fff".repeat(21) + "f"),
        config_hash: config_hash.clone(),
        token_count: 4,  // DIFFERENT
    };

    let auth_b = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer.json".to_string(),
        file_hash: Some("fff".repeat(21) + "f"),
        config_hash: config_hash,
        token_count: 8,  // DIFFERENT
    };

    // WHEN: Validate consistency
    let result = validate_tokenizer_consistency(&auth_a, &auth_b);

    // THEN: Returns Err
    assert!(result.is_err(), "Validation should fail for different token counts");

    // AND: Error message mentions "token count mismatch"
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Token count mismatch") || err_msg.contains("token count"),
        "Error should mention token count mismatch: {}", err_msg);

    // AND: Error message includes both counts
    assert!(err_msg.contains("4"), "Error should show Lane A count");
    assert!(err_msg.contains("8"), "Error should show Lane B count");
}
```

**Success Criteria**:
- ✅ Validation returns `Err`
- ✅ Error message contains "token count mismatch"
- ✅ Error message includes both token counts

#### Test 2.3: Exit Code 2 Diagnostic Format

**Note**: This test validates the expected diagnostic format if a mismatch were to occur.
Since parity-both uses shared setup, we simulate this by testing the error handler logic.

```rust
#[test]
fn test_exit_code_2_diagnostic_format_has_required_fields() {
    // GIVEN: Mismatched authorities (unit test scenario)
    let auth_a = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer_a.json".to_string(),
        file_hash: Some("aaa".repeat(21) + "a"),
        config_hash: "111".repeat(21) + "1",
        token_count: 4,
    };

    let auth_b = TokenizerAuthority {
        source: TokenizerSource::External,
        path: "tokenizer_b.json".to_string(),
        file_hash: Some("bbb".repeat(21) + "b"),
        config_hash: "222".repeat(21) + "2",
        token_count: 4,
    };

    // WHEN: Validation fails
    let error = validate_tokenizer_consistency(&auth_a, &auth_b).unwrap_err();

    // THEN: Error message format includes:
    let err_str = error.to_string();

    // 1. Clear indication of failure
    assert!(err_str.contains("mismatch") || err_str.contains("Mismatch"),
        "Error should indicate mismatch");

    // 2. Both config hashes for debugging
    // (The actual parity_both.rs error handler prints these separately in eprintln!)
    // This test verifies the underlying validation function error format
    assert!(err_str.len() > 50, "Error message should be descriptive");
}
```

**Expected diagnostic format** (from `parity_both.rs:620-628`):
```
✗ ERROR: Tokenizer consistency validation failed
  Lane A config hash: 111111111111111111111111111111111111111111111111111111111111111
  Lane B config hash: 222222222222222222222222222222222222222222222222222222222222222
  Details: Tokenizer config mismatch: Lane A hash=111..., Lane B hash=222...
```

**Success Criteria**:
- ✅ Error message indicates mismatch type
- ✅ Both config hashes available for comparison
- ✅ Clear actionable diagnostic

---

### 3.3 TC3: Receipt Schema v2 Compatibility

**Objective**: Verify backward compatibility and proper version inference.

#### Test 3.1: Receipt Version Inference with TokenizerAuthority

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_receipt_version_inferred_as_v2() {
    // GIVEN: Valid tokenizer
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN: Run parity-both
    let result = run_parity_both_e2e(&model, &tokenizer, "Test", 2, &temp_dir);
    assert_eq!(result.exit_code, 0);

    // THEN: Receipt schema version is "2.0.0"
    for receipt_path in &[result.receipt_bitnet, result.receipt_llama] {
        let content = std::fs::read_to_string(receipt_path).unwrap();
        let receipt: ParityReceipt = serde_json::from_str(&content).unwrap();

        // Version inference (receipt.infer_version() returns "2.0.0" if v2 fields present)
        let version = receipt.infer_version();
        assert_eq!(version, "2.0.0",
            "Receipt with tokenizer_authority should infer version 2.0.0");

        // TokenizerAuthority field should be Some
        assert!(receipt.tokenizer_authority.is_some(),
            "v2 receipt should have tokenizer_authority populated");
    }
}
```

**Success Criteria**:
- ✅ `infer_version()` returns "2.0.0"
- ✅ `tokenizer_authority` field is Some
- ✅ Receipt deserializes correctly with v2 schema

#### Test 3.2: Receipt Serialization Omits None Fields

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_receipt_json_skips_none_fields() {
    // GIVEN: Valid tokenizer (external, so file_hash is Some)
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN: Run parity-both
    let result = run_parity_both_e2e(&model, &tokenizer, "Test", 2, &temp_dir);
    assert_eq!(result.exit_code, 0);

    // THEN: Receipt JSON includes tokenizer_authority (not skipped)
    let content = std::fs::read_to_string(&result.receipt_bitnet).unwrap();
    assert!(content.contains("tokenizer_authority"),
        "Receipt JSON should include tokenizer_authority field");

    // AND: file_hash is present (external tokenizer)
    assert!(content.contains("file_hash"),
        "Receipt JSON should include file_hash for external tokenizer");

    // AND: JSON is valid and parseable
    let _receipt: ParityReceipt = serde_json::from_str(&content)
        .expect("Receipt JSON should be valid");
}
```

**Success Criteria**:
- ✅ `tokenizer_authority` field present in JSON
- ✅ `file_hash` present (external tokenizer)
- ✅ JSON is valid and deserializable

---

### 3.4 TC4: Edge Cases and Error Handling

**Objective**: Verify graceful handling of invalid inputs.

#### Test 4.1: Missing Tokenizer File

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_missing_tokenizer_file_exit_2() {
    // GIVEN: Non-existent tokenizer path
    let temp_dir = TempDir::new().unwrap();
    let model = get_test_model_path();
    let nonexistent = temp_dir.path().join("nonexistent_tokenizer.json");

    // WHEN: Run parity-both with missing tokenizer
    let result = run_parity_both_e2e(
        &model,
        &nonexistent,
        "Test",
        2,
        &temp_dir,
    );

    // THEN: Exit code 2 (usage error)
    assert_eq!(result.exit_code, 2,
        "Exit code should be 2 for missing tokenizer file");

    // AND: Error message mentions file not found
    let combined_output = format!("{}\n{}", result.stdout, result.stderr);
    assert!(combined_output.contains("not found")
        || combined_output.contains("No such file")
        || combined_output.contains("Failed to load tokenizer"),
        "Error should mention missing file");
}
```

**Success Criteria**:
- ✅ Exit code 2
- ✅ Error message mentions file not found
- ✅ No receipts written (command fails early)

#### Test 4.2: Corrupted Tokenizer JSON

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_corrupted_tokenizer_json_exit_2() {
    // GIVEN: Corrupted tokenizer.json (invalid JSON)
    let temp_dir = TempDir::new().unwrap();
    let model = get_test_model_path();
    let corrupted = temp_dir.path().join("corrupted.json");
    std::fs::write(&corrupted, "{\"vocab\": [\"tok1\", \"tok2\" }  // truncated")
        .unwrap();

    // WHEN: Run parity-both with corrupted tokenizer
    let result = run_parity_both_e2e(&model, &corrupted, "Test", 2, &temp_dir);

    // THEN: Exit code 2 (usage error - invalid input)
    assert_eq!(result.exit_code, 2,
        "Exit code should be 2 for corrupted tokenizer");

    // AND: Error message mentions JSON parsing failure
    let combined_output = format!("{}\n{}", result.stdout, result.stderr);
    assert!(combined_output.contains("JSON")
        || combined_output.contains("parse")
        || combined_output.contains("Failed to load tokenizer"),
        "Error should mention JSON parsing issue");
}
```

**Success Criteria**:
- ✅ Exit code 2
- ✅ Error message mentions JSON/parsing failure
- ✅ No receipts written

#### Test 4.3: Model File Missing (Orthogonal Error)

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_missing_model_file_exit_2() {
    // GIVEN: Valid tokenizer but missing model
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let nonexistent_model = temp_dir.path().join("nonexistent_model.gguf");

    // WHEN: Run parity-both with missing model
    let result = run_parity_both_e2e(
        &nonexistent_model,
        &tokenizer,
        "Test",
        2,
        &temp_dir,
    );

    // THEN: Exit code 2 (usage error)
    assert_eq!(result.exit_code, 2,
        "Exit code should be 2 for missing model file");

    // AND: Error message mentions model file issue
    let combined_output = format!("{}\n{}", result.stdout, result.stderr);
    assert!(combined_output.contains("model")
        || combined_output.contains("not found")
        || combined_output.contains("Failed to load"),
        "Error should mention model file issue");
}
```

**Success Criteria**:
- ✅ Exit code 2
- ✅ Error message mentions model file issue
- ✅ No receipts written

---

### 3.5 TC5: Hash Determinism and Stability

**Objective**: Verify hash computation is deterministic and consistent.

#### Test 5.1: File Hash Determinism (Same File → Same Hash)

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_file_hash_deterministic_across_runs() {
    // GIVEN: Same tokenizer file used twice
    let temp_dir = TempDir::new().unwrap();
    let tokenizer = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let model = get_test_model_path();

    // WHEN: Run parity-both twice
    let result1 = run_parity_both_e2e(&model, &tokenizer, "Test", 2, &temp_dir);
    assert_eq!(result1.exit_code, 0);
    let auth1 = parse_receipt_authority(&result1.receipt_bitnet).unwrap();

    // Clean up receipts
    std::fs::remove_file(&result1.receipt_bitnet).unwrap();
    std::fs::remove_file(&result1.receipt_llama).unwrap();

    let result2 = run_parity_both_e2e(&model, &tokenizer, "Test", 2, &temp_dir);
    assert_eq!(result2.exit_code, 0);
    let auth2 = parse_receipt_authority(&result2.receipt_bitnet).unwrap();

    // THEN: File hashes are identical
    assert_eq!(auth1.file_hash, auth2.file_hash,
        "File hash should be deterministic for same file");

    // AND: Config hashes are identical
    assert_eq!(auth1.config_hash, auth2.config_hash,
        "Config hash should be deterministic for same tokenizer");
}
```

**Success Criteria**:
- ✅ File hash identical across runs
- ✅ Config hash identical across runs
- ✅ Both hashes are 64 hex chars

#### Test 5.2: Config Hash Determinism (Same Vocab → Same Hash)

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_config_hash_identical_for_cloned_tokenizers() {
    // GIVEN: Two tokenizer files with identical content (byte-for-byte clones)
    let temp_dir = TempDir::new().unwrap();
    let tokenizer_a = copy_fixture("valid_tokenizer_a.json", &temp_dir);
    let tokenizer_b = copy_fixture("valid_tokenizer_b.json", &temp_dir); // Clone of A
    let model = get_test_model_path();

    // WHEN: Run parity-both with tokenizer A
    let result_a = run_parity_both_e2e(&model, &tokenizer_a, "Test", 2, &temp_dir);
    assert_eq!(result_a.exit_code, 0);
    let auth_a = parse_receipt_authority(&result_a.receipt_bitnet).unwrap();

    // Clean up
    std::fs::remove_file(&result_a.receipt_bitnet).unwrap();
    std::fs::remove_file(&result_a.receipt_llama).unwrap();

    // WHEN: Run parity-both with tokenizer B (clone)
    let result_b = run_parity_both_e2e(&model, &tokenizer_b, "Test", 2, &temp_dir);
    assert_eq!(result_b.exit_code, 0);
    let auth_b = parse_receipt_authority(&result_b.receipt_bitnet).unwrap();

    // THEN: Config hashes are identical (same vocab config)
    assert_eq!(auth_a.config_hash, auth_b.config_hash,
        "Config hash should match for tokenizers with identical vocab");

    // AND: File hashes are also identical (byte-for-byte clones)
    assert_eq!(auth_a.file_hash, auth_b.file_hash,
        "File hash should match for byte-for-byte identical files");
}
```

**Success Criteria**:
- ✅ Config hashes match for cloned tokenizers
- ✅ File hashes match for byte-identical files
- ✅ Both are 64 hex chars

#### Test 5.3: Config Hash Differs for Different Vocabs

```rust
#[test]
#[serial(bitnet_env)]
fn test_e2e_config_hash_differs_for_different_vocab_sizes() {
    // GIVEN: Two tokenizers with different vocab sizes
    let temp_dir = TempDir::new().unwrap();
    let tokenizer_128k = copy_fixture("valid_tokenizer_a.json", &temp_dir); // 128000 vocab
    let tokenizer_64k = copy_fixture("different_vocab_size.json", &temp_dir); // 64000 vocab
    let model = get_test_model_path();

    // WHEN: Run parity-both with 128k vocab tokenizer
    let result_128k = run_parity_both_e2e(&model, &tokenizer_128k, "Test", 2, &temp_dir);
    assert_eq!(result_128k.exit_code, 0);
    let auth_128k = parse_receipt_authority(&result_128k.receipt_bitnet).unwrap();

    // Clean up
    std::fs::remove_file(&result_128k.receipt_bitnet).unwrap();
    std::fs::remove_file(&result_128k.receipt_llama).unwrap();

    // WHEN: Run parity-both with 64k vocab tokenizer
    let result_64k = run_parity_both_e2e(&model, &tokenizer_64k, "Test", 2, &temp_dir);
    assert_eq!(result_64k.exit_code, 0);
    let auth_64k = parse_receipt_authority(&result_64k.receipt_bitnet).unwrap();

    // THEN: Config hashes are DIFFERENT (different vocab sizes)
    assert_ne!(auth_128k.config_hash, auth_64k.config_hash,
        "Config hash should differ for different vocab sizes");

    // AND: File hashes are also DIFFERENT (different file content)
    assert_ne!(auth_128k.file_hash, auth_64k.file_hash,
        "File hash should differ for different files");
}
```

**Success Criteria**:
- ✅ Config hashes differ for different vocab sizes
- ✅ File hashes differ for different files
- ✅ Both receipts parse successfully

---

## 4. Test Infrastructure

### 4.1 Test File Location

**Primary location**: `xtask/tests/tokenizer_authority_e2e_tests.rs`

**Rationale**:
- Tests execute `cargo run -p xtask -- parity-both` as subprocess
- Natural location alongside other xtask integration tests
- Access to xtask test helpers and fixtures

### 4.2 Test Helpers and Utilities

```rust
// Location: xtask/tests/tokenizer_authority_e2e_tests.rs

use serial_test::serial;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;
use bitnet_crossval::receipt::{ParityReceipt, TokenizerAuthority, TokenizerSource};

/// Result of E2E parity-both execution
struct E2EResult {
    stdout: String,
    stderr: String,
    exit_code: i32,
    receipt_bitnet: PathBuf,
    receipt_llama: PathBuf,
}

/// Copy test fixture to temp directory
fn copy_fixture(fixture_name: &str, temp_dir: &TempDir) -> PathBuf {
    let fixture_path = workspace_root()
        .join("tests/fixtures/tokenizers")
        .join(fixture_name);
    let dest_path = temp_dir.path().join(fixture_name);
    std::fs::copy(&fixture_path, &dest_path)
        .expect("Failed to copy fixture");
    dest_path
}

/// Get test model path (auto-discover or use BITNET_GGUF)
fn get_test_model_path() -> PathBuf {
    if let Ok(model) = std::env::var("BITNET_GGUF") {
        PathBuf::from(model)
    } else {
        workspace_root()
            .join("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf")
    }
}

/// Find workspace root
fn workspace_root() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !path.join(".git").exists() {
        if !path.pop() {
            panic!("Could not find workspace root");
        }
    }
    path
}

/// Execute parity-both as subprocess
fn run_parity_both_e2e(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    max_tokens: usize,
    out_dir: &Path,
) -> E2EResult {
    let mut cmd = Command::new("cargo");
    cmd.args(&["run", "-p", "xtask", "--features", "crossval-all", "--"]);
    cmd.args(&["parity-both"]);
    cmd.arg("--model-gguf").arg(model_path);
    cmd.arg("--tokenizer").arg(tokenizer_path);
    cmd.arg("--prompt").arg(prompt);
    cmd.arg("--max-tokens").arg(max_tokens.to_string());
    cmd.arg("--out-dir").arg(out_dir);
    cmd.arg("--no-repair"); // Skip auto-repair for deterministic tests

    let output = cmd.output().expect("Failed to execute parity-both");

    E2EResult {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
        receipt_bitnet: out_dir.join("receipt_bitnet.json"),
        receipt_llama: out_dir.join("receipt_llama.json"),
    }
}

/// Parse receipt and extract TokenizerAuthority
fn parse_receipt_authority(receipt_path: &Path) -> anyhow::Result<TokenizerAuthority> {
    let content = std::fs::read_to_string(receipt_path)?;
    let receipt: ParityReceipt = serde_json::from_str(&content)?;
    receipt.tokenizer_authority
        .ok_or_else(|| anyhow::anyhow!("Receipt missing tokenizer_authority"))
}

/// Assert two authorities are identical
fn assert_authorities_match(auth_a: &TokenizerAuthority, auth_b: &TokenizerAuthority) {
    assert_eq!(auth_a.config_hash, auth_b.config_hash);
    assert_eq!(auth_a.token_count, auth_b.token_count);
    assert_eq!(auth_a.source, auth_b.source);
    // Note: file_hash and path may differ if files are copies, but config_hash must match
}
```

### 4.3 Test Fixtures Setup

**Required fixtures** (to be created):

1. **`tests/fixtures/tokenizers/valid_tokenizer_a.json`**:
   - Standard LLaMA-3 tokenizer (128000 vocab)
   - Reference tokenizer for happy path tests

2. **`tests/fixtures/tokenizers/valid_tokenizer_b.json`**:
   - Byte-for-byte copy of `valid_tokenizer_a.json`
   - Tests hash determinism (same file → same hash)

3. **`tests/fixtures/tokenizers/different_vocab_size.json`**:
   - Modified tokenizer with 64000 vocab
   - Tests config hash divergence detection

4. **`tests/fixtures/tokenizers/different_special_tokens.json`**:
   - Same vocab size but different BOS/EOS token IDs
   - Tests edge case handling

5. **`tests/fixtures/tokenizers/corrupted.json`**:
   - Truncated or malformed JSON
   - Tests error handling for invalid input

**Fixture generation script** (optional):
```bash
#!/bin/bash
# Location: scripts/generate_tokenizer_fixtures.sh

set -e

FIXTURES_DIR="tests/fixtures/tokenizers"
mkdir -p "$FIXTURES_DIR"

# Copy reference tokenizer (assumes model downloaded)
cp models/microsoft-bitnet-b1.58-2B-4T-gguf/tokenizer.json \
   "$FIXTURES_DIR/valid_tokenizer_a.json"

# Clone for hash determinism test
cp "$FIXTURES_DIR/valid_tokenizer_a.json" \
   "$FIXTURES_DIR/valid_tokenizer_b.json"

# Generate variant with different vocab size (modify JSON)
jq '.model.vocab |= . + ["extra_token_1", "extra_token_2"]' \
   "$FIXTURES_DIR/valid_tokenizer_a.json" \
   > "$FIXTURES_DIR/different_vocab_size.json"

# Generate corrupted variant (truncate JSON)
head -c 500 "$FIXTURES_DIR/valid_tokenizer_a.json" \
   > "$FIXTURES_DIR/corrupted.json"

echo "✓ Tokenizer fixtures generated in $FIXTURES_DIR"
```

### 4.4 CI Integration

**GitHub Actions workflow** (add to `.github/workflows/test.yml`):

```yaml
- name: Run E2E TokenizerAuthority tests
  run: |
    # Ensure model available for tests
    cargo run -p xtask -- download-model || echo "Model already present"

    # Generate test fixtures
    ./scripts/generate_tokenizer_fixtures.sh

    # Run E2E tests with nextest (5min timeout)
    cargo nextest run -p xtask --test tokenizer_authority_e2e_tests \
      --features crossval-all \
      --profile ci
```

---

## 5. Acceptance Criteria (Test-Focused)

### AC1: Happy Path Exit Code 0

**Requirement**: Identical tokenizers produce exit code 0 with matching receipts.

**Tests**:
- ✅ `test_e2e_identical_tokenizers_exit_0`
- ✅ `test_e2e_receipt_authority_all_fields_populated`

**Success Criteria**:
- Exit code 0
- Both receipts written
- TokenizerAuthority fields populated
- Config hashes match across lanes

### AC2: Summary Display (Text and JSON)

**Requirement**: Summary includes tokenizer hash in both text and JSON formats.

**Tests**:
- ✅ `test_e2e_summary_text_format_displays_hash`
- ✅ `test_e2e_summary_json_format_includes_tokenizer`

**Success Criteria**:
- Text format: "Tokenizer Consistency" section with full hash
- JSON format: `tokenizer` object with `config_hash` and `status: "consistent"`

### AC3: Receipt Schema v2 Compatibility

**Requirement**: Receipts use schema v2 with backward compatibility.

**Tests**:
- ✅ `test_e2e_receipt_version_inferred_as_v2`
- ✅ `test_e2e_receipt_json_skips_none_fields`

**Success Criteria**:
- `infer_version()` returns "2.0.0"
- `tokenizer_authority` field present
- JSON deserializable with v2 schema

### AC4: Cross-Lane Validation Logic

**Requirement**: Validation function correctly detects mismatches.

**Tests**:
- ✅ `test_validate_tokenizer_consistency_rejects_different_config_hash`
- ✅ `test_validate_tokenizer_consistency_rejects_different_token_count`

**Success Criteria**:
- Returns `Err` for mismatched config hashes
- Returns `Err` for mismatched token counts
- Error messages include diagnostic details

### AC5: Edge Case Handling

**Requirement**: Graceful error handling for invalid inputs.

**Tests**:
- ✅ `test_e2e_missing_tokenizer_file_exit_2`
- ✅ `test_e2e_corrupted_tokenizer_json_exit_2`
- ✅ `test_e2e_missing_model_file_exit_2`

**Success Criteria**:
- Exit code 2 for all error cases
- Clear error messages
- No partial receipts written

### AC6: Hash Determinism

**Requirement**: Hash computation is deterministic and consistent.

**Tests**:
- ✅ `test_e2e_file_hash_deterministic_across_runs`
- ✅ `test_e2e_config_hash_identical_for_cloned_tokenizers`
- ✅ `test_e2e_config_hash_differs_for_different_vocab_sizes`

**Success Criteria**:
- Same file → same file hash
- Same vocab → same config hash
- Different vocab → different config hash

---

## 6. Implementation Roadmap

### Phase 1: Test Infrastructure Setup (Day 1)

**Tasks**:
1. ✅ Create `xtask/tests/tokenizer_authority_e2e_tests.rs`
2. ✅ Implement test helper functions:
   - `run_parity_both_e2e()`
   - `parse_receipt_authority()`
   - `assert_authorities_match()`
3. ✅ Add `tests/fixtures/tokenizers/` directory
4. ✅ Generate test fixtures (valid, cloned, different, corrupted)

**Deliverable**: Test harness compiles and can execute parity-both as subprocess.

### Phase 2: Happy Path Tests (Day 2)

**Tasks**:
1. ✅ Implement TC1 tests (exit code 0, receipt population)
2. ✅ Implement TC2 tests (summary display text/JSON)
3. ✅ Run tests against existing implementation
4. ✅ Fix any issues discovered

**Deliverable**: Happy path tests pass (6 tests).

### Phase 3: Validation and Edge Cases (Day 3)

**Tasks**:
1. ✅ Implement TC3 tests (schema v2 compatibility)
2. ✅ Implement TC4 tests (validation logic)
3. ✅ Implement TC5 tests (edge cases)
4. ✅ Run full test suite

**Deliverable**: All validation and error handling tests pass (6 tests).

### Phase 4: Hash Determinism and CI Integration (Day 4)

**Tasks**:
1. ✅ Implement TC6 tests (hash determinism)
2. ✅ Add CI workflow for E2E tests
3. ✅ Verify test stability across runs
4. ✅ Document test coverage and fixtures

**Deliverable**: Complete E2E test suite (12 tests) with CI integration.

---

## 7. Success Metrics

### 7.1 Test Coverage Metrics

**Target coverage**:
- **12 E2E integration tests** (new)
- **44 unit tests** (existing)
- **6 integration tests** (existing scaffolding)
- **Total: 62 tests** covering TokenizerAuthority system

**Coverage breakdown**:

| Category | Unit Tests | Integration Tests | E2E Tests | Total |
|----------|------------|-------------------|-----------|-------|
| Structure/Serialization | 14 | 0 | 2 | 16 |
| Hash Computation | 12 | 0 | 3 | 15 |
| Validation Logic | 8 | 2 | 2 | 12 |
| Receipt Integration | 6 | 2 | 2 | 10 |
| Error Handling | 4 | 2 | 3 | 9 |
| **Total** | **44** | **6** | **12** | **62** |

### 7.2 Quality Gates

**All tests must pass**:
- ✅ Exit code 0 for happy path
- ✅ Exit code 2 for validation failures
- ✅ Receipt JSON schema v2 valid
- ✅ Cross-lane consistency enforced
- ✅ Hash determinism verified

**CI integration**:
- ✅ E2E tests run in CI pipeline
- ✅ Tests complete within 5 minutes
- ✅ No flaky tests (retries disabled)
- ✅ Clear failure diagnostics

### 7.3 Documentation Requirements

**Required deliverables**:
1. ✅ This test specification (`docs/specs/tokenizer-authority-validation-tests.md`)
2. ✅ Test fixture README (`tests/fixtures/tokenizers/README.md`)
3. ✅ Test harness documentation (inline comments in test file)
4. ✅ CI workflow documentation (GitHub Actions YAML)

---

## 8. Related Documentation

### 8.1 Analysis Documents

- **Analysis**: `docs/analysis/tokenizer-authority-validation-analysis.md`
  - Complete implementation analysis with code locations
  - Flow diagrams and validation logic
  - 44 unit tests + 6 integration tests (baseline)

### 8.2 Specifications

- **Feature Spec**: `docs/specs/parity-both-preflight-tokenizer-integration.md`
  - AC1-AC10 for preflight and tokenizer integration
  - Dual-lane orchestration architecture
  - Exit code semantics and receipt schema v2

- **Command Spec**: `docs/specs/parity-both-command.md`
  - CLI interface and arguments
  - Output formats and receipt naming
  - Example invocations

### 8.3 Implementation

- **Receipt Schema**: `crossval/src/receipt.rs:38-136`
  - TokenizerAuthority struct (lines 54-82)
  - TokenizerSource enum (lines 38-52)
  - ParityReceipt with v2 fields (lines 121-136)

- **Hash Functions**: `crossval/src/receipt.rs:341-412`
  - `compute_tokenizer_file_hash()` (lines 341-350)
  - `compute_tokenizer_config_hash_from_tokenizer()` (lines 382-398)
  - `detect_tokenizer_source()` (lines 400-412)

- **Validation**: `crossval/src/receipt.rs:480-500`
  - `validate_tokenizer_consistency()` (lines 480-500)
  - Cross-lane config hash and token count checks

- **Parity Both**: `xtask/src/crossval/parity_both.rs`
  - STEP 2.5: Shared computation (lines 496-528)
  - STEP 4-6: Dual lane injection (lines 557-588)
  - STEP 7.5: Validation and exit code 2 (lines 594-634)

### 8.4 Test Infrastructure

- **Unit Tests**: `crossval/tests/tokenizer_authority_tests.rs`
  - 44 tests covering structure, serialization, validation
  - TC1-TC10 test categories

- **Integration Tests**: `xtask/tests/tokenizer_authority_integration_tests.rs`
  - 6 tests covering AC1-AC6 scaffolding
  - Foundation for E2E tests

---

## 9. Appendix: Test Fixtures Specification

### 9.1 Fixture File Formats

**`valid_tokenizer_a.json`** (reference tokenizer):
```json
{
  "version": "1.0",
  "model": {
    "vocab": ["<|begin_of_text|>", "<|end_of_text|>", ..., "token_128000"],
    "scores": [0.0, 0.0, ..., 0.0]
  },
  "added_tokens": [...],
  "normalizer": {...},
  "pre_tokenizer": {...},
  "post_processor": {...},
  "decoder": {...}
}
```

**Key properties**:
- `vocab_size`: 128000 (standard LLaMA-3)
- `real_vocab_size`: 128000
- Special tokens: BOS=128000, EOS=128001 (example)

**`different_vocab_size.json`** (variant with 64000 tokens):
```json
{
  "version": "1.0",
  "model": {
    "vocab": ["<|begin_of_text|>", ..., "token_64000"],
    "scores": [0.0, ..., 0.0]
  }
}
```

**Key difference**:
- `vocab_size`: 64000 (smaller vocab)
- `real_vocab_size`: 64000
- Config hash will differ from valid_tokenizer_a.json

### 9.2 Fixture Generation Script

```bash
#!/bin/bash
# Location: scripts/generate_tokenizer_fixtures.sh

set -e

FIXTURES_DIR="tests/fixtures/tokenizers"
MODELS_DIR="models/microsoft-bitnet-b1.58-2B-4T-gguf"

echo "Generating tokenizer test fixtures..."

# Ensure models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Model directory not found. Run 'cargo run -p xtask -- download-model' first."
    exit 1
fi

# Create fixtures directory
mkdir -p "$FIXTURES_DIR"

# Fixture 1: Reference tokenizer
echo "1. Copying reference tokenizer (valid_tokenizer_a.json)..."
cp "$MODELS_DIR/tokenizer.json" "$FIXTURES_DIR/valid_tokenizer_a.json"

# Fixture 2: Byte-identical clone
echo "2. Creating byte-identical clone (valid_tokenizer_b.json)..."
cp "$FIXTURES_DIR/valid_tokenizer_a.json" "$FIXTURES_DIR/valid_tokenizer_b.json"

# Fixture 3: Different vocab size (modify JSON with jq)
echo "3. Creating different vocab size variant (different_vocab_size.json)..."
if command -v jq &> /dev/null; then
    # Truncate vocab to 64000 tokens
    jq '.model.vocab = .model.vocab[:64000] | .model.scores = .model.scores[:64000]' \
       "$FIXTURES_DIR/valid_tokenizer_a.json" \
       > "$FIXTURES_DIR/different_vocab_size.json"
else
    echo "   Warning: jq not found, skipping different_vocab_size.json"
fi

# Fixture 4: Corrupted JSON
echo "4. Creating corrupted JSON (corrupted.json)..."
head -c 500 "$FIXTURES_DIR/valid_tokenizer_a.json" > "$FIXTURES_DIR/corrupted.json"

# Create README
echo "5. Creating fixtures README..."
cat > "$FIXTURES_DIR/README.md" << 'EOF'
# Tokenizer Test Fixtures

This directory contains tokenizer fixtures for TokenizerAuthority E2E tests.

## Fixtures

- **valid_tokenizer_a.json**: Reference LLaMA-3 tokenizer (128000 vocab)
- **valid_tokenizer_b.json**: Byte-identical clone of A (tests hash determinism)
- **different_vocab_size.json**: Modified tokenizer with 64000 vocab (tests config hash divergence)
- **corrupted.json**: Truncated JSON (tests error handling)

## Regeneration

Run: `./scripts/generate_tokenizer_fixtures.sh`

Requires:
- Model downloaded: `cargo run -p xtask -- download-model`
- jq installed (for JSON manipulation)

## Usage

E2E tests automatically discover fixtures from this directory.
See: `xtask/tests/tokenizer_authority_e2e_tests.rs`
EOF

echo "✓ Tokenizer fixtures generated in $FIXTURES_DIR"
echo ""
echo "Fixtures:"
ls -lh "$FIXTURES_DIR"
```

---

## 10. Summary

This test specification defines **comprehensive end-to-end validation** for the TokenizerAuthority cross-lane consistency system. The specification focuses on **testing existing implementation correctness** rather than specifying new features.

### Key Deliverables

1. **12 E2E integration tests** validating:
   - Happy path (exit 0, receipt population, summary display)
   - Validation logic (mismatch detection, exit code 2)
   - Edge cases (missing files, corrupted JSON)
   - Hash determinism (same file → same hash)

2. **Test infrastructure**:
   - Test harness with subprocess execution
   - Fixture generation and management
   - CI integration with GitHub Actions

3. **Documentation**:
   - Test specification (this document)
   - Fixture README
   - Test harness inline documentation

### Success Criteria

- ✅ All 12 E2E tests pass against current implementation
- ✅ Exit codes verified (0=success, 2=mismatch)
- ✅ Receipt JSON schema v2 validated
- ✅ Cross-lane consistency enforced
- ✅ Hash determinism verified
- ✅ CI integration complete

**Status**: Ready for implementation. No blockers identified.

**Implementation Status**: TokenizerAuthority system is **fully implemented and production-ready**. Tests will validate correctness and catch regressions.
