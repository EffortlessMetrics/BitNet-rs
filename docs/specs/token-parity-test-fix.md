# Technical Specification: Token Parity Test Fix

**Document Version**: 1.0.0
**Date**: 2025-10-25
**Status**: Draft
**Author**: BitNet.rs Spec Analyzer

---

## Executive Summary

The `test_error_message_format` test in `xtask/tests/crossval_token_parity.rs` is failing to compile due to missing `backend` field in `TokenParityError` struct initialization. This is a simple test scaffolding update to align with the production code that was enhanced to support backend-specific error diagnostics.

**Scope**: Test-only changes (no production code modifications)
**Effort**: ~5 minutes (single line change)
**Risk**: Minimal (test-only change, no functional impact)

---

## Problem Statement

### Issue Description

Three tests in `xtask/tests/crossval_token_parity.rs` are affected:

1. **test_error_message_format** (line 356): **COMPILATION ERROR**
   - Missing `backend: CppBackend` field in TokenParityError struct initialization
   - Error: `error[E0063]: missing field 'backend' in initializer of 'TokenParityError'`
   - Location: Line 359

2. **test_cli_flag_parsing** (line 390): **IGNORED** (placeholder test)
   - Marked with `#[ignore]` - not yet implemented
   - No action required

3. **Other ignored tests**: All properly marked with `#[ignore]` and not blocking

### Root Cause

The `TokenParityError` struct was updated in `crossval/src/token_parity.rs` (commit 58c881e5) to include a `backend: CppBackend` field for backend-specific error diagnostics. The integration test in `xtask/tests/crossval_token_parity.rs` was not updated to match the new struct signature.

**Production code** (crossval/src/token_parity.rs:30-43):

```rust
#[derive(Debug, Clone)]
pub struct TokenParityError {
    pub rust_tokens: Vec<u32>,
    pub cpp_tokens: Vec<u32>,
    pub first_diff_index: usize,
    pub prompt: String,
    pub backend: CppBackend,  // <- ADDED for backend-specific diagnostics
}
```

**Test code** (xtask/tests/crossval_token_parity.rs:359-364) - **OUTDATED**:

```rust
let error = TokenParityError {
    rust_tokens: vec![128000, 128000, 1229, 374],
    cpp_tokens: vec![128000, 1229, 374],
    first_diff_index: 1,
    prompt: "What is 2+2?".to_string(),
    // MISSING: backend field
};
```

### Impact Analysis

**Compilation**: Test file fails to compile, blocking test execution
**Runtime**: N/A (compilation error)
**CI/CD**: Blocks CI pipeline if tests run with `--features inference`
**Dependencies**: No cross-crate impact (test-only change)

---

## Requirements Analysis

### Functional Requirements

**FR1**: Test must compile without errors
**FR2**: Test must validate error message format with 4 assertions
**FR3**: Test must use BitNet backend for error generation
**FR4**: Error message must maintain 8-section diagnostic format

### Non-Functional Requirements

**NFR1**: No production code changes required
**NFR2**: Maintain backward compatibility with Display trait contract
**NFR3**: Test execution time < 100ms (existing acceptance criteria)
**NFR4**: Zero impact on other test suites

### Acceptance Criteria

**AC1**: Test compiles successfully with `--features inference`
**AC2**: All 4 test assertions pass
**AC3**: Error message contains "Token Sequence Mismatch"
**AC4**: Error message contains "Rust" and "C++" sections
**AC5**: Error message contains token value "128000"
**AC6**: Error message contains "Common fixes" section
**AC7**: CppBackend::BitNet appears in formatted output
**AC8**: No warnings or clippy errors introduced

---

## Technical Approach

### Architecture Alignment

This fix aligns with BitNet.rs cross-validation architecture:

- **TDD Practices**: Test scaffolding follows production code updates
- **Backend-Aware Diagnostics**: Tests validate backend-specific error messages
- **Cross-Validation Integration**: Ensures token parity pre-gate tests match production behavior

### Component Analysis

**Affected Files**:
- `xtask/tests/crossval_token_parity.rs` (line 359) - **PRIMARY**
- No changes to `crossval/src/token_parity.rs` (production code)
- No changes to `crossval/src/backend.rs` (enum definition)

**Import Analysis**:

The test already imports `CppBackend` at line 272:

```rust
#[test]
fn test_mock_ffi_session_token_comparison() {
    use bitnet_crossval::backend::CppBackend;  // <- EXISTING IMPORT
    // ...
}
```

However, `test_error_message_format` needs its own import since each test is a separate scope.

**Solution**: Add scoped import to `test_error_message_format` test function.

---

## Implementation Plan

### Change 1: Add CppBackend Import to Test Function

**File**: `xtask/tests/crossval_token_parity.rs`
**Location**: Line 356 (inside test function)
**Type**: Add import statement

**Before** (lines 354-365):

```rust
// Test: Verify error message formatting
#[test]
fn test_error_message_format() {
    use bitnet_crossval::token_parity::{TokenParityError, format_token_mismatch_error};

    let error = TokenParityError {
        rust_tokens: vec![128000, 128000, 1229, 374],
        cpp_tokens: vec![128000, 1229, 374],
        first_diff_index: 1,
        prompt: "What is 2+2?".to_string(),
    };

    let formatted = format_token_mismatch_error(&error);
```

**After**:

```rust
// Test: Verify error message formatting
#[test]
fn test_error_message_format() {
    use bitnet_crossval::backend::CppBackend;
    use bitnet_crossval::token_parity::{TokenParityError, format_token_mismatch_error};

    let error = TokenParityError {
        rust_tokens: vec![128000, 128000, 1229, 374],
        cpp_tokens: vec![128000, 1229, 374],
        first_diff_index: 1,
        prompt: "What is 2+2?".to_string(),
        backend: CppBackend::BitNet,
    };

    let formatted = format_token_mismatch_error(&error);
```

**Rationale**:
- Uses `CppBackend::BitNet` for consistency with BitNet model testing
- Matches the backend used in `test_mock_ffi_session_token_comparison` (line 291)
- Aligns with test scenario (microsoft-bitnet model)

### Change 2: Verify Assertion Coverage

**File**: `xtask/tests/crossval_token_parity.rs`
**Location**: Lines 368-384
**Type**: Validation (no changes needed)

**Existing Assertions**:

```rust
// Assertion 1: Error title check
assert!(
    formatted.contains("Token Sequence Mismatch") || formatted.contains("mismatch"),
    "Should include error title"
);

// Assertion 2: Both Rust and C++ sections
assert!(
    formatted.contains("Rust") && formatted.contains("C++"),
    "Should show both Rust and C++ sections"
);

// Assertion 3: Token values displayed
assert!(
    formatted.contains("128000") || formatted.contains("tokens"),
    "Should display token values"
);

// Assertion 4: Suggestions section
assert!(
    formatted.contains("Suggested fixes") || formatted.contains("fix"),
    "Should include suggestions section"
);
```

**Validation Against Production Code**:

| Assertion | Production Output | Match? | Notes |
|-----------|------------------|--------|-------|
| #1 | `"❌ Token Sequence Mismatch with C++ Backend: BitNet"` (line 167) | ✅ YES | Contains "Token Sequence Mismatch" |
| #2 | `"Rust tokens (4):"` (line 175)<br>`"C++ tokens (3):"` (line 183) | ✅ YES | Both "Rust" and "C++" present |
| #3 | `[128000, 128000, 1229, 374]` (line 177) | ✅ YES | Token value "128000" displayed |
| #4 | `"Common fixes:"` (line 242) | ✅ YES | Contains "fix" substring |

**Note on Assertion #4**: The test uses OR condition `"Suggested fixes" || "fix"`. Production code outputs `"Common fixes:"`, which matches the second condition (`"fix"`). This is acceptable behavior.

**Recommendation**: No changes to assertions needed. All 4 assertions will pass with current production code.

---

## Code Changes

### File: xtask/tests/crossval_token_parity.rs

**Line Numbers**: 356-364

**Diff**:

```diff
 // Test: Verify error message formatting
 #[test]
 fn test_error_message_format() {
+    use bitnet_crossval::backend::CppBackend;
     use bitnet_crossval::token_parity::{TokenParityError, format_token_mismatch_error};

     let error = TokenParityError {
         rust_tokens: vec![128000, 128000, 1229, 374],
         cpp_tokens: vec![128000, 1229, 374],
         first_diff_index: 1,
         prompt: "What is 2+2?".to_string(),
+        backend: CppBackend::BitNet,
     };

     let formatted = format_token_mismatch_error(&error);
```

**Total Changes**: 2 lines added (1 import, 1 struct field)

---

## Validation Strategy

### Unit Test Validation

**Test Command**:

```bash
cargo test --package xtask --test crossval_token_parity test_error_message_format \
  --no-default-features --features inference
```

**Expected Output**:

```
running 1 test
test test_error_message_format ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 13 filtered out
```

**Success Criteria**:
- ✅ Test compiles without errors
- ✅ Test executes without panics
- ✅ All 4 assertions pass
- ✅ No clippy warnings

### Integration Test Validation

**Run All Token Parity Tests**:

```bash
cargo test --package xtask --test crossval_token_parity \
  --no-default-features --features inference
```

**Expected Results**:

```
running 14 tests
test test_cli_flag_parsing ... ignored (TODO: Add CLI flag parsing tests)
test test_cli_integration_full_flow ... ignored (TODO: Full integration)
test test_error_message_format ... ok
test test_instruct_template_tokenization_parity ... ignored (TODO: Requires model)
test test_llama3_chat_template_tokenization_parity ... ignored (TODO: Requires model)
test test_mock_ffi_session_token_comparison ... ok
test test_no_bos_flag_prevents_duplicate ... ignored (TODO: Requires tokenizer integration)
test test_proceeds_to_logits_on_token_match ... ignored (TODO: Requires model)
test test_raw_template_tokenization_parity ... ignored (TODO: Requires model)
test test_template_auto_detection ... ignored (TODO: Requires template logic)
test test_token_parity_performance_overhead ... ok

test result: ok. 3 passed; 0 failed; 11 ignored; 0 measured; 0 filtered out
```

**Success Criteria**:
- ✅ `test_error_message_format` passes
- ✅ `test_mock_ffi_session_token_comparison` still passes (no regression)
- ✅ `test_token_parity_performance_overhead` still passes (no regression)
- ✅ 11 tests remain ignored (expected TDD scaffolding)

### CI/CD Validation

**Full Workspace Test**:

```bash
cargo test --workspace --no-default-features --features cpu,inference
```

**Expected**:
- ✅ All previously passing tests still pass
- ✅ No new compilation errors
- ✅ No new clippy warnings

---

## Error Message Format Specification

### Production Error Format (8 Sections)

When `format_token_mismatch_error()` is called with `CppBackend::BitNet`, the output follows this structure:

**Section 1: Header** (line 164-172)

```
❌ Token Sequence Mismatch with C++ Backend: BitNet
Fix BOS/template before comparing logits
```

**Section 2: Rust Tokens** (line 175-180)

```
Rust tokens (4):
  [128000, 128000, 1229, 374]
```

**Section 3: C++ Tokens** (line 183-188)

```
C++ tokens (3):
  [128000, 1229, 374]
```

**Section 4: First Diff Position** (line 191-192)

```
First diff at index: 1
```

**Section 5: Mismatch Detail** (line 198-205)

```
Mismatch: Rust token=128000, C++ token=1229
```

**Section 6: Backend-Specific Troubleshooting** (line 209-239)

```
Troubleshooting for BitNet backend:
  • Verify your model is BitNet-compatible (microsoft-bitnet-b1.58-2B-4T-gguf)
  • Check tokenizer path matches model format
  • Try --cpp-backend llama if this is a LLaMA model
  • Verify --prompt-template setting (current: auto)
  • Check BOS token handling with --dump-ids
```

**Section 7: Common Fixes** (line 242-245)

```
Common fixes:
  • Use --prompt-template raw (disable template formatting)
  • Add --no-bos flag (if BOS is duplicated)
  • Check GGUF chat_template metadata
```

**Section 8: Example Command** (line 248-255)

```
Example command:
  cargo run -p xtask -- crossval-per-token \
    --model <model.gguf> \
    --tokenizer <tokenizer.json> \
    --prompt "What is 2+2?" \
    --prompt-template raw \
    --cpp-backend bitnet \
    --max-tokens 4
```

### Test Assertion Mapping

| Test Assertion | Section | Line | Content Match |
|----------------|---------|------|---------------|
| Assertion #1 | Section 1 | 167 | "Token Sequence Mismatch" |
| Assertion #2 | Sections 2, 3 | 175, 183 | "Rust" and "C++" |
| Assertion #3 | Section 2 | 177 | "128000" |
| Assertion #4 | Section 7 | 242 | "fix" (in "Common fixes") |

---

## Risk Assessment

### Technical Risks

**Risk 1: Backend Choice Mismatch**
**Likelihood**: Low
**Impact**: Low
**Mitigation**: Using `CppBackend::BitNet` is consistent with test scenario (BitNet model tokenization). The `test_mock_ffi_session_token_comparison` already uses `CppBackend::BitNet` at line 291, confirming this is the expected backend for test scenarios.

**Risk 2: Assertion Failures**
**Likelihood**: Very Low
**Impact**: Low
**Mitigation**: All 4 assertions validated against production code output (see table above). Production code has 15 passing tests in `crossval/src/token_parity.rs`, confirming stable behavior.

**Risk 3: Import Scope Issues**
**Likelihood**: Very Low
**Impact**: Low
**Mitigation**: Using scoped imports (inside test function) is standard Rust practice and avoids namespace pollution. The `test_mock_ffi_session_token_comparison` already uses this pattern successfully.

### Quality Risks

**Risk 1: Test Semantics Drift**
**Likelihood**: Low
**Impact**: Low
**Mitigation**: Test assertions are intentionally flexible (e.g., `"Suggested fixes" || "fix"`). This design choice allows production code to evolve section headers without breaking tests. If stricter validation is needed, update assertions separately.

**Risk 2: Incomplete Test Coverage**
**Likelihood**: N/A
**Impact**: N/A
**Note**: 11 ignored tests are intentional TDD scaffolding (see `docs/explanation/token-parity-pregate.md`). These tests are blocked by issues #254, #260, #469 and will be enabled post-MVP.

---

## Success Criteria

### Compilation Success

```bash
✅ cargo test --package xtask --test crossval_token_parity --no-default-features --features inference
```

**Expected**: Zero compilation errors

### Test Execution Success

```bash
✅ cargo test --package xtask --test crossval_token_parity test_error_message_format \
     --no-default-features --features inference -- --nocapture
```

**Expected**:
- Test passes without panics
- All 4 assertions pass
- Output shows formatted error message (with `--nocapture`)

### Regression Prevention

```bash
✅ cargo test --package xtask --test crossval_token_parity \
     --no-default-features --features inference
```

**Expected**:
- 3 tests pass: `test_error_message_format`, `test_mock_ffi_session_token_comparison`, `test_token_parity_performance_overhead`
- 11 tests ignored (expected TDD scaffolding)
- 0 tests fail

### Code Quality

```bash
✅ cargo clippy --package xtask --tests --no-default-features --features inference -- -D warnings
```

**Expected**: Zero clippy warnings for test file

---

## Backward Compatibility

### Production Code Impact

**None** - This is a test-only change. No modifications to:
- `crossval/src/token_parity.rs` (production error handling)
- `crossval/src/backend.rs` (CppBackend enum)
- `crossval/src/lib.rs` (public API)

### Error Display Contract

The `TokenParityError::Display` trait implementation remains unchanged:

```rust
impl fmt::Display for TokenParityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Token sequence mismatch at index {}: rust={:?} cpp={:?}",
            self.first_diff_index, self.rust_tokens, self.cpp_tokens
        )
    }
}
```

**Contract**: Single-line summary for error chain display (does not include backend).

**Detailed formatting**: Provided by `format_token_mismatch_error()` function (includes backend-specific hints).

---

## Migration Path

### Step 1: Apply Code Changes

```bash
# Navigate to repository root
cd /home/steven/code/Rust/BitNet-rs

# Edit test file
# Apply changes from "Code Changes" section above
```

### Step 2: Verify Compilation

```bash
cargo test --package xtask --test crossval_token_parity \
  --no-default-features --features inference --no-run
```

**Expected**: Compilation succeeds without errors

### Step 3: Run Test

```bash
cargo test --package xtask --test crossval_token_parity test_error_message_format \
  --no-default-features --features inference -- --nocapture
```

**Expected**: Test passes with all 4 assertions succeeding

### Step 4: Regression Check

```bash
cargo test --package xtask --test crossval_token_parity \
  --no-default-features --features inference
```

**Expected**: 3 passing tests, 11 ignored tests, 0 failures

### Step 5: Workspace Validation

```bash
cargo test --workspace --no-default-features --features cpu,inference
```

**Expected**: No regressions in other test suites

---

## References

### Related Documentation

- `docs/explanation/token-parity-pregate.md`: Token parity pre-gate specification
- `docs/explanation/dual-backend-crossval.md`: Dual-backend cross-validation architecture
- `/tmp/token-parity-exploration.md`: Deep dive analysis (provided context)

### Related Source Files

- `crossval/src/token_parity.rs`: Production token parity validation (15 passing tests)
- `crossval/src/backend.rs`: CppBackend enum definition
- `xtask/tests/crossval_token_parity.rs`: Integration tests (3 passing, 11 ignored)

### Issue Tracker

- Issue #469: Tokenizer parity and FFI build hygiene (blocks some ignored tests)
- Issue #254: Shape mismatch in layer-norm (blocks real inference tests)
- Issue #260: Mock elimination not complete (blocks end-to-end tests)

---

## Implementation Checklist

**Pre-Implementation**:
- [x] Analyze production code structure
- [x] Validate test assertions against production output
- [x] Confirm no production code changes needed
- [x] Document error message format (8 sections)

**Implementation**:
- [ ] Add `use bitnet_crossval::backend::CppBackend;` import to test function
- [ ] Add `backend: CppBackend::BitNet,` field to TokenParityError initialization
- [ ] Verify formatting (rustfmt)

**Validation**:
- [ ] Compile test file successfully
- [ ] Run `test_error_message_format` - expect pass
- [ ] Run all token parity tests - expect 3 pass, 11 ignored
- [ ] Check clippy for warnings - expect zero
- [ ] Run workspace tests - expect no regressions

**Documentation**:
- [ ] Update this spec with implementation results
- [ ] Add to commit message: "test: fix TokenParityError initialization in test_error_message_format"

---

## Appendix A: Assertion Validation Details

### Assertion #1: Error Title Check

**Test Code** (line 369-372):

```rust
assert!(
    formatted.contains("Token Sequence Mismatch") || formatted.contains("mismatch"),
    "Should include error title"
);
```

**Production Output** (line 167):

```rust
writeln!(
    output,
    "\n{}",
    style(format!("❌ Token Sequence Mismatch with C++ Backend: {}", error.backend.name()))
        .red()
        .bold()
)
```

**Match**: ✅ `"Token Sequence Mismatch with C++ Backend: BitNet"` contains `"Token Sequence Mismatch"`

---

### Assertion #2: Both Rust and C++ Sections

**Test Code** (line 373-376):

```rust
assert!(
    formatted.contains("Rust") && formatted.contains("C++"),
    "Should show both Rust and C++ sections"
);
```

**Production Output**:
- Line 175: `writeln!(output, "\n{} ({}):", style("Rust tokens").cyan(), error.rust_tokens.len())`
- Line 183: `writeln!(output, "\n{} ({}):", style("C++ tokens").cyan(), error.cpp_tokens.len())`

**Match**: ✅ Both `"Rust tokens"` and `"C++ tokens"` present

---

### Assertion #3: Token Values Displayed

**Test Code** (line 377-380):

```rust
assert!(
    formatted.contains("128000") || formatted.contains("tokens"),
    "Should display token values"
);
```

**Production Output** (line 177):

```rust
writeln!(output, "  {:?}", error.rust_tokens)  // [128000, 128000, 1229, 374]
```

**Match**: ✅ Both `"128000"` and `"tokens"` present

---

### Assertion #4: Suggestions Section

**Test Code** (line 381-384):

```rust
assert!(
    formatted.contains("Suggested fixes") || formatted.contains("fix"),
    "Should include suggestions section"
);
```

**Production Output** (line 242):

```rust
writeln!(output, "\n{}:", style("Common fixes").green().bold())
```

**Match**: ✅ `"Common fixes"` contains substring `"fix"`

**Note**: Test uses flexible OR condition. If stricter validation desired (exact match on "Suggested fixes"), update assertion separately. Current behavior is acceptable.

---

## Appendix B: Backend Selection Rationale

### Why CppBackend::BitNet?

**Reason 1: Test Scenario Consistency**

The test scenario uses BitNet model token sequences:

```rust
rust_tokens: vec![128000, 128000, 1229, 374],  // Duplicate BOS (BitNet model behavior)
cpp_tokens: vec![128000, 1229, 374],            // Single BOS (C++ reference)
```

Token ID `128000` is the BOS token for LLaMA-3 tokenizer, commonly used with BitNet models. The duplicate BOS scenario is specific to BitNet model tokenization quirks.

**Reason 2: Consistency with Other Tests**

The `test_mock_ffi_session_token_comparison` test (line 291) already uses `CppBackend::BitNet`:

```rust
let result_match = bitnet_crossval::token_parity::validate_token_parity(
    &rust_tokens,
    &cpp_tokens_match,
    "What is 2+2?",
    CppBackend::BitNet,  // <- Same backend
);
```

**Reason 3: Error Message Validation**

Using `CppBackend::BitNet` produces backend-specific error messages that test assertions validate:

- Section 6 outputs: "Troubleshooting for BitNet backend"
- Section 8 example command includes: `--cpp-backend bitnet`

**Alternative**: Could use `CppBackend::Llama`, but this would be inconsistent with test scenario semantics (BitNet model token sequences).

---

## Appendix C: Test Status Summary

### Passing Tests (3)

1. **test_error_message_format** (line 356) - **WILL PASS** after fix
2. **test_mock_ffi_session_token_comparison** (line 271) - PASSING
3. **test_token_parity_performance_overhead** (line 326) - PASSING

### Ignored Tests (11) - Intentional TDD Scaffolding

1. **test_proceeds_to_logits_on_token_match** (line 46) - Requires model and C++ FFI
2. **test_raw_template_tokenization_parity** (line 106) - Requires model and C++ FFI
3. **test_instruct_template_tokenization_parity** (line 144) - Requires model and C++ FFI
4. **test_llama3_chat_template_tokenization_parity** (line 182) - Requires model and C++ FFI
5. **test_no_bos_flag_prevents_duplicate** (line 222) - Requires tokenizer integration
6. **test_cli_integration_full_flow** (line 258) - Requires full xtask integration
7. **test_cli_flag_parsing** (line 390) - Placeholder for CLI flag validation
8. **test_template_auto_detection** (line 406) - Requires template auto-detection logic

**Note**: These tests are blocked by active issues (#254, #260, #469) and will be enabled post-MVP. This is **expected** and documented in `CLAUDE.md` section "Test Status (MVP Phase)".

---

## Appendix D: Production Code Stability

### Library Tests (crossval/src/token_parity.rs)

**15 passing tests** validate production code behavior:

1. test_detect_token_mismatch
2. test_first_diff_position
3. test_first_diff_length_mismatch
4. test_error_message_includes_suggestions
5. test_error_message_actionable
6. test_error_message_shows_examples
7. test_error_detects_duplicate_bos
8. test_silent_success_on_match
9. test_performance_under_100ms
10. test_backend_specific_error_messages
11. test_backend_in_example_command
12. test_scenario_tokens_match
13. test_scenario_length_mismatch
14. test_empty_sequences
15. test_single_token

**Status**: All passing as of commit 58c881e5
**Confidence**: High - Production code is stable and well-tested

---

## Document Metadata

**File**: `/home/steven/code/Rust/BitNet-rs/docs/specs/token-parity-test-fix.md`
**Created**: 2025-10-25
**Last Updated**: 2025-10-25
**Spec Version**: 1.0.0
**Implementation Status**: Draft (awaiting implementation)
**Estimated Effort**: 5 minutes
**Risk Level**: Minimal (test-only change)
**Dependencies**: None (standalone test fix)
**Related Issues**: None (test scaffolding update)
**Review Status**: Ready for implementation

---

## Quick Reference

**Problem**: Missing `backend` field in TokenParityError test initialization
**Solution**: Add `backend: CppBackend::BitNet,` to struct initialization
**Files Changed**: `xtask/tests/crossval_token_parity.rs` (line 359)
**Lines Changed**: 2 (1 import, 1 field)
**Test Command**: `cargo test --package xtask --test crossval_token_parity test_error_message_format --no-default-features --features inference`
**Expected Result**: Test compiles and passes with all 4 assertions succeeding
**Risk**: Minimal (test-only change, no production code impact)
**Effort**: ~5 minutes

---

**END OF SPECIFICATION**
