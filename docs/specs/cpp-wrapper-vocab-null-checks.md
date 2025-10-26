# Technical Specification: C++ Wrapper Vocab NULL Check Safety

**Document Type**: Explanation (Technical Specification)
**Status**: Draft
**Priority**: HIGH (Runtime Crash Prevention)
**Target File**: `crossval/src/bitnet_cpp_wrapper.cc`
**Created**: 2025-10-25
**Feature ID**: FEAT-CROSSVAL-NULL-CHECKS-001

---

## Executive Summary

This specification addresses a **critical safety issue** in the BitNet.rs C++ FFI wrapper: missing NULL checks after `llama_model_get_vocab()` calls. The vulnerability exists at 4 locations (lines 102, 261, 558, 710) where the code unconditionally uses the vocab pointer without validation, creating **high-severity crash risk** when model loading fails or incompatible GGUF formats are provided.

**Risk Level**: ⚠️ **HIGH** - Unhandled NULL dereference leads to immediate segfault
**Impact**: Cross-validation tests, FFI integration, production C++ inference
**Affected Sockets**: Socket 0 (stateless), Socket 2 (tokenization), Socket 3 (inference)

---

## 1. Problem Statement

### 1.1 Root Cause

The llama.cpp API function `llama_model_get_vocab()` can return `NULL` in several scenarios:

1. **Model loading failures**: Corrupted GGUF files, incomplete downloads
2. **Incompatible formats**: GGUF version mismatches, unsupported quantization formats
3. **Memory allocation failures**: Insufficient RAM during large model loads
4. **Invalid model structures**: Missing vocab tensors, malformed metadata

**Current Code Pattern (Unsafe)**:
```cpp
const llama_vocab* vocab = llama_model_get_vocab(model);
int32_t n_tokens = llama_tokenize(vocab, ...);  // ⚠️ SEGFAULT if vocab == NULL
```

This pattern is repeated at 4 critical locations without NULL validation.

### 1.2 Impact Analysis

**Immediate Consequences**:
- **Segmentation Fault**: Instant process crash on NULL dereference
- **Undefined Behavior**: `llama_tokenize()` and `llama_vocab_n_tokens()` receive invalid pointer
- **No Error Propagation**: Crash occurs before error strings can be written

**Affected Workflows**:
- Cross-validation against C++ reference (`cargo run -p xtask -- crossval`)
- Per-token parity testing (`cargo run -p xtask -- crossval-per-token`)
- FFI integration tests (`cargo test -p crossval --features ffi`)
- Production inference when using C++ backend

**Severity Justification**:
- **High Probability**: Model loading failures are common during development (invalid paths, format changes)
- **High Impact**: Immediate crash with no diagnostic information
- **Production Risk**: Silent failures in deployed systems

### 1.3 Current Affected Locations

| Line | Function | Socket | Context |
|------|----------|--------|---------|
| 102 | `crossval_bitnet_tokenize()` | Socket 0 | Stateless tokenization |
| 261 | `crossval_bitnet_eval_with_tokens()` | Socket 0 | Stateless inference |
| 558 | `bitnet_cpp_tokenize_with_context()` | Socket 2 | Stateful tokenization |
| 710 | `bitnet_cpp_eval_with_context()` | Socket 3 | Stateful inference |

---

## 2. Technical Design

### 2.1 NULL Check Strategy

**Design Principle**: Fail-fast with actionable error messages

**Pattern**:
```cpp
const llama_vocab* vocab = llama_model_get_vocab(model);
if (!vocab) {
    snprintf(err, err_len, "<context>: Failed to get vocab from model (check model format/compatibility)");
    err[err_len - 1] = '\0';
    // Resource cleanup (model, context if allocated)
    return -1;
}
```

**Key Design Decisions**:
1. **Immediate validation**: Check vocab immediately after retrieval
2. **Contextual error messages**: Include function name and operation context
3. **Guaranteed cleanup**: Free resources before error return
4. **Consistent return code**: Use `-1` for all FFI errors
5. **Actionable diagnostics**: Suggest root cause in error message

### 2.2 Error Handling Strategy

**Error Message Format**:
```
<function_name>: Failed to get vocab from model (check model format/compatibility)
```

**Components**:
- **Function context**: Clear identification of failure location
- **Root cause hint**: "check model format/compatibility"
- **User action**: Implies GGUF validation needed

**Why This Format**:
- Consistent with existing error patterns in wrapper
- Provides enough context for debugging
- Fits within typical 256-byte error buffers
- Suggests next troubleshooting step

### 2.3 Resource Cleanup Patterns

**Socket 0 (Stateless) - Lines 102, 261**:
```cpp
// At line 102 (tokenize):
const llama_vocab* vocab = llama_model_get_vocab(model);
if (!vocab) {
    snprintf(err, err_len, "crossval_bitnet_tokenize: Failed to get vocab from model (check model format/compatibility)");
    err[err_len - 1] = '\0';
    llama_model_free(model);  // ← Must free model before return
    return -1;
}

// At line 261 (eval):
const llama_vocab* vocab = llama_model_get_vocab(model);
if (!vocab) {
    snprintf(err, err_len, "crossval_bitnet_eval_with_tokens: Failed to get vocab from model (check model format/compatibility)");
    err[err_len - 1] = '\0';
    llama_free(ctx);           // ← Free context first (depends on model)
    llama_model_free(model);   // ← Then free model
    return -1;
}
```

**Socket 2/3 (Stateful) - Lines 558, 710**:
```cpp
// At line 558 (tokenize_with_context):
const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
if (!vocab) {
    snprintf(err, err_len, "bitnet_cpp_tokenize_with_context: Failed to get vocab from model (check model format/compatibility)");
    err[err_len - 1] = '\0';
    // ⚠️ DO NOT free ctx->model here (persistent context owns it)
    return -1;
}

// At line 710 (eval_with_context):
const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
if (!vocab) {
    snprintf(err, err_len, "bitnet_cpp_eval_with_context: Failed to get vocab from model (check model format/compatibility)");
    err[err_len - 1] = '\0';
    // ⚠️ DO NOT free ctx->model here (persistent context owns it)
    return -1;
}
```

**Cleanup Ownership Rules**:
- **Socket 0 (stateless)**: Function owns model/context → must free on error
- **Socket 2/3 (stateful)**: `bitnet_context_t` owns model/context → must NOT free on error (caller will free via `bitnet_cpp_free_context()`)

---

## 3. API Contract

### 3.1 Function Signatures (No Changes Required)

All 4 affected functions maintain their existing signatures:

```cpp
// Socket 0: Stateless tokenization
int crossval_bitnet_tokenize(
    const char* model_path,
    const char* prompt,
    int32_t add_bos,
    int32_t parse_special,
    int32_t* out_tokens,
    int32_t out_capacity,
    int32_t* out_len,
    char* err,
    int32_t err_len
);

// Socket 0: Stateless inference
int crossval_bitnet_eval_with_tokens(
    const char* model_path,
    const int32_t* tokens,
    int32_t n_tokens,
    int32_t n_ctx,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
);

// Socket 2: Stateful tokenization
int bitnet_cpp_tokenize_with_context(
    bitnet_context_t* ctx,
    const char* prompt,
    int32_t add_bos,
    int32_t parse_special,
    int32_t* out_tokens,
    int32_t out_capacity,
    int32_t* out_len,
    char* err,
    int32_t err_len
);

// Socket 3: Stateful inference
int bitnet_cpp_eval_with_context(
    bitnet_context_t* ctx,
    const int32_t* tokens,
    int32_t n_tokens,
    float* out_logits,
    int32_t logits_capacity,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
);
```

**API Compatibility**: ✅ **No breaking changes** - return value semantics remain identical.

### 3.2 Return Value Conventions

**Existing Convention** (unchanged):
- `0`: Success
- `-1`: Error (check `err` buffer for details)
- Positive values: Context-specific (e.g., required token count)

**New Error Code** (NULL vocab):
- Return `-1` with error message in `err` buffer
- Consistent with all other FFI error patterns

### 3.3 Error String Format

**Template**:
```
<function_name>: Failed to get vocab from model (check model format/compatibility)
```

**Examples**:
```cpp
// Line 102:
"crossval_bitnet_tokenize: Failed to get vocab from model (check model format/compatibility)"

// Line 261:
"crossval_bitnet_eval_with_tokens: Failed to get vocab from model (check model format/compatibility)"

// Line 558:
"bitnet_cpp_tokenize_with_context: Failed to get vocab from model (check model format/compatibility)"

// Line 710:
"bitnet_cpp_eval_with_context: Failed to get vocab from model (check model format/compatibility)"
```

**Length Guarantee**: Max 108 bytes (including NUL terminator) - fits within typical 256-byte error buffers.

---

## 4. Implementation Details

### 4.1 Location 1: Line 102 (Socket 0 Tokenize)

**Context**: `crossval_bitnet_tokenize()` - Stateless tokenization

**Before** (lines 102-112):
```cpp
    // Step 2: Two-pass tokenization pattern
    // Pass 1: Get token count (tokens=NULL, n_tokens_max=0)
    // New API: Get vocab from model first
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int32_t text_len = static_cast<int32_t>(std::strlen(prompt));
    int32_t n_tokens = llama_tokenize(
        vocab,             // Use vocab instead of model
        prompt,
        text_len,
        nullptr,           // tokens=NULL for size query
        0,                 // n_tokens_max=0 for size query
        add_bos != 0,      // Convert C int to bool
        parse_special != 0 // Convert C int to bool
    );
```

**After** (insert NULL check at line 103):
```cpp
    // Step 2: Two-pass tokenization pattern
    // Pass 1: Get token count (tokens=NULL, n_tokens_max=0)
    // New API: Get vocab from model first
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        snprintf(err, err_len, "crossval_bitnet_tokenize: Failed to get vocab from model (check model format/compatibility)");
        err[err_len - 1] = '\0';
        llama_model_free(model);  // Must free model before return
        return -1;
    }
    int32_t text_len = static_cast<int32_t>(std::strlen(prompt));
    int32_t n_tokens = llama_tokenize(
        vocab,             // Use vocab instead of model
        prompt,
        text_len,
        nullptr,           // tokens=NULL for size query
        0,                 // n_tokens_max=0 for size query
        add_bos != 0,      // Convert C int to bool
        parse_special != 0 // Convert C int to bool
    );
```

**Cleanup Requirements**:
- ✅ Free `model` (allocated at line ~92)
- ⚠️ Model was successfully loaded, must not leak

### 4.2 Location 2: Line 261 (Socket 0 Eval)

**Context**: `crossval_bitnet_eval_with_tokens()` - Stateless inference

**Before** (lines 261-266):
```cpp
    // Step 3: Get vocab size for logits shape
    // New API: llama_model_get_vocab + llama_vocab_n_tokens
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Step 4: Set output shape
    *out_rows = n_tokens;
    *out_cols = n_vocab;
```

**After** (insert NULL check at line 262):
```cpp
    // Step 3: Get vocab size for logits shape
    // New API: llama_model_get_vocab + llama_vocab_n_tokens
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        snprintf(err, err_len, "crossval_bitnet_eval_with_tokens: Failed to get vocab from model (check model format/compatibility)");
        err[err_len - 1] = '\0';
        llama_free(ctx);           // Free context first (depends on model)
        llama_model_free(model);   // Then free model
        return -1;
    }
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Step 4: Set output shape
    *out_rows = n_tokens;
    *out_cols = n_vocab;
```

**Cleanup Requirements**:
- ✅ Free `ctx` (allocated at line ~248, depends on model)
- ✅ Free `model` (allocated at line ~238)
- ⚠️ Order matters: context must be freed before model

### 4.3 Location 3: Line 558 (Socket 2 Tokenize)

**Context**: `bitnet_cpp_tokenize_with_context()` - Stateful tokenization

**Before** (lines 558-570):
```cpp
    // Get vocab from model
    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    int32_t text_len = static_cast<int32_t>(std::strlen(prompt));

    // Pass 1: Get token count
    int32_t n_tokens = llama_tokenize(
        vocab,
        prompt,
        text_len,
        nullptr,           // tokens=NULL for size query
        0,                 // n_tokens_max=0 for size query
        add_bos != 0,
        parse_special != 0
    );
```

**After** (insert NULL check at line 559):
```cpp
    // Get vocab from model
    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    if (!vocab) {
        snprintf(err, err_len, "bitnet_cpp_tokenize_with_context: Failed to get vocab from model (check model format/compatibility)");
        err[err_len - 1] = '\0';
        // DO NOT free ctx->model here (persistent context owns it)
        return -1;
    }
    int32_t text_len = static_cast<int32_t>(std::strlen(prompt));

    // Pass 1: Get token count
    int32_t n_tokens = llama_tokenize(
        vocab,
        prompt,
        text_len,
        nullptr,           // tokens=NULL for size query
        0,                 // n_tokens_max=0 for size query
        add_bos != 0,
        parse_special != 0
    );
```

**Cleanup Requirements**:
- ❌ DO NOT free `ctx->model` (owned by persistent context)
- ❌ DO NOT free `ctx` (caller will free via `bitnet_cpp_free_context()`)
- ✅ Return error code immediately

### 4.4 Location 4: Line 710 (Socket 3 Eval)

**Context**: `bitnet_cpp_eval_with_context()` - Stateful inference

**Before** (lines 710-715):
```cpp
    // Get vocab size
    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Set output shape
    *out_rows = n_tokens;
    *out_cols = n_vocab;
```

**After** (insert NULL check at line 711):
```cpp
    // Get vocab size
    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    if (!vocab) {
        snprintf(err, err_len, "bitnet_cpp_eval_with_context: Failed to get vocab from model (check model format/compatibility)");
        err[err_len - 1] = '\0';
        // DO NOT free ctx->model here (persistent context owns it)
        return -1;
    }
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Set output shape
    *out_rows = n_tokens;
    *out_cols = n_vocab;
```

**Cleanup Requirements**:
- ❌ DO NOT free `ctx->model` (owned by persistent context)
- ❌ DO NOT free `ctx->ctx` (owned by persistent context)
- ✅ Return error code immediately

### 4.5 Code Change Summary

**Total Changes**:
- **4 NULL checks added** (one per location)
- **2 cleanup paths modified** (Socket 0 locations require resource freeing)
- **0 API changes** (fully backward compatible)
- **0 new dependencies** (uses existing error handling patterns)

**Lines of Code**:
- **+20 lines** total (5 lines per location: if + snprintf + NUL term + cleanup + return)
- **0 lines removed**

---

## 5. Testing Requirements

### 5.1 Unit Test: NULL Vocab Scenario

**Test Case**: `test_cpp_wrapper_vocab_null_handling`

**Test File**: `crossval/tests/cpp_wrapper_null_vocab_tests.rs`

**Test Strategy**:
```rust
#[cfg(all(feature = "ffi", feature = "cpu"))]
#[test]
fn test_crossval_tokenize_null_vocab_graceful_failure() {
    // Scenario: Invalid model path → model load fails OR valid model with incompatible format
    let invalid_model = "/tmp/bitnet_test_invalid_model_12345.gguf";

    // Create minimal invalid GGUF (header only, no vocab)
    std::fs::write(invalid_model, b"GGUF").unwrap();

    let prompt = "Test";
    let mut tokens = vec![0i32; 128];
    let mut out_len = 0;
    let mut err = vec![0u8; 256];

    // Call FFI function (should fail gracefully with error message)
    let result = unsafe {
        crossval::crossval_bitnet_tokenize(
            invalid_model.as_ptr() as *const i8,
            prompt.as_ptr() as *const i8,
            1, // add_bos
            0, // parse_special
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            &mut out_len,
            err.as_mut_ptr() as *mut i8,
            err.len() as i32,
        )
    };

    // Assert: Should return -1 (error)
    assert_eq!(result, -1, "Expected tokenize to fail with invalid model");

    // Assert: Error message should mention vocab failure
    let err_str = std::ffi::CStr::from_bytes_until_nul(&err)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        err_str.contains("Failed to get vocab from model"),
        "Expected vocab error message, got: {}", err_str
    );

    // Cleanup
    std::fs::remove_file(invalid_model).ok();
}

#[cfg(all(feature = "ffi", feature = "cpu"))]
#[test]
fn test_crossval_eval_null_vocab_graceful_failure() {
    // Similar test for crossval_bitnet_eval_with_tokens
    // ... (pattern matches above)
}

#[cfg(all(feature = "ffi", feature = "cpu"))]
#[test]
fn test_stateful_tokenize_null_vocab_graceful_failure() {
    // Test bitnet_cpp_tokenize_with_context with invalid context
    // ... (requires initializing context with invalid model)
}

#[cfg(all(feature = "ffi", feature = "cpu"))]
#[test]
fn test_stateful_eval_null_vocab_graceful_failure() {
    // Test bitnet_cpp_eval_with_context with invalid context
    // ... (requires initializing context with invalid model)
}
```

**Test Coverage**:
- ✅ All 4 locations tested (Socket 0 tokenize/eval, Socket 2 tokenize, Socket 3 eval)
- ✅ Error return code validation (`-1`)
- ✅ Error message content validation (contains "Failed to get vocab")
- ✅ No crashes (graceful degradation)
- ✅ No memory leaks (valgrind-clean on Linux)

### 5.2 Integration Test: Invalid Model Format

**Test Case**: `test_crossval_invalid_gguf_format`

**Test Strategy**:
```bash
# Integration test script: scripts/test_vocab_null_safety.sh

#!/bin/bash
set -euo pipefail

# Create test artifacts
INVALID_GGUF=/tmp/bitnet_invalid_test.gguf
echo "GGUF" > "$INVALID_GGUF"  # Header only, no vocab

# Test 1: Tokenize with invalid model (should fail gracefully)
echo "Testing tokenize with invalid model..."
if cargo run -p xtask --features crossval-all -- crossval-per-token \
    --model "$INVALID_GGUF" \
    --tokenizer /dev/null \
    --prompt "Test" \
    --max-tokens 1 2>&1 | grep -q "Failed to get vocab"; then
    echo "✅ Tokenize error handling: PASS"
else
    echo "❌ Tokenize error handling: FAIL (no vocab error message)"
    exit 1
fi

# Test 2: Eval with invalid model (should fail gracefully)
echo "Testing eval with invalid model..."
# ... (similar pattern)

# Cleanup
rm -f "$INVALID_GGUF"
echo "All vocab NULL safety tests passed"
```

**Integration Coverage**:
- ✅ End-to-end workflow testing (xtask → FFI → C++ wrapper)
- ✅ Invalid GGUF format handling
- ✅ Error propagation from C++ to Rust
- ✅ User-facing error messages

### 5.3 Memory Leak Validation

**Test Case**: `test_vocab_null_no_memory_leaks`

**Validation Method**:
```bash
# Run unit tests under valgrind
cargo test -p crossval --features ffi,cpu \
    --test cpp_wrapper_null_vocab_tests \
    -- --test-threads=1 \
    | valgrind --leak-check=full --error-exitcode=1

# Expected output:
# ==12345== LEAK SUMMARY:
# ==12345==    definitely lost: 0 bytes in 0 blocks
# ==12345==    indirectly lost: 0 bytes in 0 blocks
# ==12345==      possibly lost: 0 bytes in 0 blocks
# ==12345== All heap blocks were freed -- no leaks are possible
```

**Leak Scenarios to Validate**:
- ✅ Socket 0 tokenize: Model freed after vocab NULL
- ✅ Socket 0 eval: Both context and model freed after vocab NULL
- ✅ Socket 2/3: No leaks when returning error (context ownership respected)

---

## 6. Acceptance Criteria

### 6.1 Functional Requirements

| ID | Requirement | Validation Method | Status |
|----|-------------|-------------------|--------|
| **AC1** | All 4 locations have NULL checks after `llama_model_get_vocab()` | Code inspection | ⏳ Pending |
| **AC2** | Error messages follow consistent format (`<function>: Failed to get vocab...`) | Unit tests | ⏳ Pending |
| **AC3** | Socket 0 functions free model/context on vocab NULL | Unit tests + valgrind | ⏳ Pending |
| **AC4** | Socket 2/3 functions do NOT free context on vocab NULL | Unit tests + valgrind | ⏳ Pending |
| **AC5** | All functions return `-1` on vocab NULL | Unit tests | ⏳ Pending |
| **AC6** | No segfaults when testing with invalid models | Integration tests | ⏳ Pending |

### 6.2 Non-Functional Requirements

| ID | Requirement | Validation Method | Status |
|----|-------------|-------------------|--------|
| **AC7** | No memory leaks on error paths | Valgrind full suite | ⏳ Pending |
| **AC8** | Error messages fit within 256-byte buffers | Static analysis | ⏳ Pending |
| **AC9** | No API breaking changes | Signature diff | ⏳ Pending |
| **AC10** | No performance regression (NULL check is O(1)) | Benchmark suite | ⏳ Pending |

### 6.3 Testing Requirements

| ID | Requirement | Validation Method | Status |
|----|-------------|-------------------|--------|
| **AC11** | Unit tests for all 4 locations | `cargo test -p crossval` | ⏳ Pending |
| **AC12** | Integration test with invalid GGUF | `scripts/test_vocab_null_safety.sh` | ⏳ Pending |
| **AC13** | Valgrind clean on error paths | CI pipeline | ⏳ Pending |
| **AC14** | Error message validation in tests | Unit test assertions | ⏳ Pending |

### 6.4 Documentation Requirements

| ID | Requirement | Validation Method | Status |
|----|-------------|-------------------|--------|
| **AC15** | Code comments explain vocab NULL check rationale | Code review | ⏳ Pending |
| **AC16** | CHANGELOG.md entry for safety fix | Manual inspection | ⏳ Pending |
| **AC17** | Updated C++ wrapper API analysis report | Doc review | ⏳ Pending |

---

## 7. Risk Analysis

### 7.1 Implementation Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Incorrect cleanup ordering** (Socket 0 eval) | Medium | High | Code review + valgrind validation |
| **Double-free in stateful functions** (Socket 2/3) | Low | Critical | Explicit ownership comments + tests |
| **Error message buffer overflow** | Very Low | Medium | Static length validation (108 bytes < 256) |
| **Performance regression** | Very Low | Low | NULL check is single pointer comparison (O(1)) |

### 7.2 Testing Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Difficulty creating NULL vocab scenario** | Medium | Medium | Use minimal invalid GGUF (header-only) |
| **Flaky tests on invalid model cleanup** | Low | Medium | Use `/tmp` with unique names + explicit cleanup |
| **Valgrind false positives** | Low | Low | Baseline valgrind run before changes |

### 7.3 Deployment Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Silent behavior change** (errors where crashes occurred) | Low | Medium | Document error message format + test coverage |
| **Downstream FFI consumers unaware of new error** | Low | Low | Maintain existing error code convention (`-1`) |

---

## 8. Implementation Plan

### 8.1 Phase 1: Code Changes (2-4 hours)

**Tasks**:
1. Add NULL check at line 102 (Socket 0 tokenize) + cleanup
2. Add NULL check at line 261 (Socket 0 eval) + cleanup
3. Add NULL check at line 558 (Socket 2 tokenize) - no cleanup
4. Add NULL check at line 710 (Socket 3 eval) - no cleanup
5. Code review: Verify cleanup ownership rules
6. Build verification: `cargo build --features cpu,ffi`

**Artifacts**:
- Modified `crossval/src/bitnet_cpp_wrapper.cc`
- Commit: `fix(crossval): add NULL checks for llama_model_get_vocab() calls`

### 8.2 Phase 2: Unit Tests (4-6 hours)

**Tasks**:
1. Create test file: `crossval/tests/cpp_wrapper_null_vocab_tests.rs`
2. Implement `test_crossval_tokenize_null_vocab_graceful_failure`
3. Implement `test_crossval_eval_null_vocab_graceful_failure`
4. Implement `test_stateful_tokenize_null_vocab_graceful_failure`
5. Implement `test_stateful_eval_null_vocab_graceful_failure`
6. Run tests: `cargo test -p crossval --features ffi,cpu`

**Artifacts**:
- New test file with 4+ test cases
- Commit: `test(crossval): add vocab NULL safety tests for C++ wrapper`

### 8.3 Phase 3: Integration Tests (2-3 hours)

**Tasks**:
1. Create script: `scripts/test_vocab_null_safety.sh`
2. Test invalid GGUF handling end-to-end
3. Validate error messages in CLI output
4. Run valgrind on full test suite
5. Verify no memory leaks with invalid models

**Artifacts**:
- Integration test script
- Commit: `test(crossval): add integration tests for vocab NULL safety`

### 8.4 Phase 4: Documentation (1-2 hours)

**Tasks**:
1. Update API analysis report (`/tmp/cpp_wrapper_current_api_analysis.md`)
2. Add CHANGELOG.md entry
3. Update `docs/howto/cpp-setup.md` with error message examples
4. Code review: Verify comment clarity

**Artifacts**:
- Updated documentation
- Commit: `docs(crossval): document vocab NULL check safety improvements`

### 8.5 Total Estimate

**Implementation Time**: 9-15 hours
**Complexity**: Low-Medium (straightforward pattern, careful cleanup logic)
**Dependencies**: None (isolated change to C++ wrapper)

---

## 9. Alternatives Considered

### 9.1 Alternative 1: Assert on NULL Vocab

**Approach**: Use `assert(vocab != nullptr)` instead of graceful error handling

**Pros**:
- Simpler code (no error message formatting)
- Fails fast in debug builds

**Cons**:
- ❌ No-op in release builds (UB still occurs)
- ❌ No error message for debugging
- ❌ Violates FFI error propagation pattern

**Decision**: **Rejected** - Graceful error handling is mandatory for FFI boundary.

### 9.2 Alternative 2: Return Different Error Codes per Location

**Approach**: Use location-specific error codes (-2, -3, -4, -5) for each vocab NULL

**Pros**:
- Easier automated diagnosis

**Cons**:
- ❌ Breaks existing error code convention
- ❌ Requires documenting new error codes
- ❌ Error message already provides context

**Decision**: **Rejected** - Error messages provide sufficient context, maintain `-1` convention.

### 9.3 Alternative 3: Retry Logic on NULL Vocab

**Approach**: Retry `llama_model_get_vocab()` with backoff

**Pros**:
- Could handle transient failures

**Cons**:
- ❌ NULL vocab is non-transient (invalid model format)
- ❌ Adds complexity and latency
- ❌ Retry belongs in caller code, not FFI wrapper

**Decision**: **Rejected** - Fail-fast is appropriate for invalid models.

---

## 10. Future Enhancements

### 10.1 Enhanced Diagnostics (v0.3)

**Proposal**: Log llama.cpp internal error state before returning

**Rationale**: llama.cpp may provide additional context via internal logging

**Implementation**:
```cpp
if (!vocab) {
    // Check if llama.cpp provides error details
    const char* llama_error = llama_get_last_error();  // Hypothetical API
    if (llama_error) {
        snprintf(err, err_len, "%s: Failed to get vocab - %s", __func__, llama_error);
    } else {
        snprintf(err, err_len, "%s: Failed to get vocab from model", __func__);
    }
    // ... cleanup
}
```

**Dependency**: Requires llama.cpp to expose error state API (currently unavailable).

### 10.2 Vocab Validation (v0.4)

**Proposal**: Validate vocab structure after successful retrieval

**Rationale**: Catch corrupted vocab (non-NULL but invalid data)

**Implementation**:
```cpp
const llama_vocab* vocab = llama_model_get_vocab(model);
if (!vocab) {
    // ... existing NULL check
}

// Additional validation
int32_t n_vocab = llama_vocab_n_tokens(vocab);
if (n_vocab <= 0 || n_vocab > 1000000) {  // Sanity bounds
    snprintf(err, err_len, "%s: Invalid vocab size %d", __func__, n_vocab);
    // ... cleanup
    return -1;
}
```

**Trade-off**: Adds overhead to happy path (1 extra function call per vocab retrieval).

---

## 11. References

### 11.1 Related Documents

- **API Analysis Report**: `/tmp/cpp_wrapper_current_api_analysis.md` (Section 3, Problem 1)
- **C++ Wrapper Architecture**: `docs/explanation/dual-backend-crossval.md`
- **FFI Error Handling**: `crossval/src/bitnet_cpp_wrapper.cc` (existing patterns)
- **Socket Architecture**: `docs/architecture-overview.md` (Socket 0-3 descriptions)

### 11.2 External References

- **llama.cpp API**: `llama.h` - `llama_model_get_vocab()` documentation
- **GGUF Specification**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Memory Safety Best Practices**: MISRA C++ guidelines for pointer validation

---

## Appendix A: Code Snippets

### A.1 Complete NULL Check Pattern (Reference Implementation)

```cpp
// Location 1: Socket 0 tokenize (line 102)
const llama_vocab* vocab = llama_model_get_vocab(model);
if (!vocab) {
    snprintf(err, err_len, "crossval_bitnet_tokenize: Failed to get vocab from model (check model format/compatibility)");
    err[err_len - 1] = '\0';
    llama_model_free(model);  // Must free model (stateless function owns it)
    return -1;
}

// Location 2: Socket 0 eval (line 261)
const llama_vocab* vocab = llama_model_get_vocab(model);
if (!vocab) {
    snprintf(err, err_len, "crossval_bitnet_eval_with_tokens: Failed to get vocab from model (check model format/compatibility)");
    err[err_len - 1] = '\0';
    llama_free(ctx);           // Free context first (depends on model)
    llama_model_free(model);   // Then free model (order matters)
    return -1;
}

// Location 3: Socket 2 tokenize (line 558)
const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
if (!vocab) {
    snprintf(err, err_len, "bitnet_cpp_tokenize_with_context: Failed to get vocab from model (check model format/compatibility)");
    err[err_len - 1] = '\0';
    // DO NOT free ctx->model (persistent context owns it)
    return -1;
}

// Location 4: Socket 3 eval (line 710)
const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
if (!vocab) {
    snprintf(err, err_len, "bitnet_cpp_eval_with_context: Failed to get vocab from model (check model format/compatibility)");
    err[err_len - 1] = '\0';
    // DO NOT free ctx->model (persistent context owns it)
    return -1;
}
```

### A.2 Test Invalid GGUF Generator (Helper Function)

```rust
/// Creates a minimal invalid GGUF file for testing vocab NULL scenarios
fn create_invalid_gguf(path: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;

    // Write minimal GGUF header (will fail llama.cpp parsing)
    file.write_all(b"GGUF")?;           // Magic
    file.write_all(&[3u8, 0, 0, 0])?;   // Version (invalid)
    file.write_all(&[0u8; 12])?;        // Truncated header

    Ok(())
}
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-25 | BitNet.rs Spec Generator | Initial specification |

---

**Status**: ✅ **READY FOR IMPLEMENTATION**

**Reviewer Sign-Off**:
- [ ] Technical Lead - Approve design and cleanup patterns
- [ ] FFI Maintainer - Approve error handling consistency
- [ ] QA - Approve test coverage plan
