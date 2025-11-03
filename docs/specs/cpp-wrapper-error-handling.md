# C++ Wrapper Error Handling Enhancement Specification

**Component**: C++ FFI Error Handling (cpp_bridge.cpp, cpp_bindings.rs, crossval)
**Location**: `crates/bitnet-kernels/src/ffi/cpp_bridge.cpp`, `crossval/src/cpp_bindings.rs`, `crossval/src/lib.rs`
**Dependencies**: Thread-local storage, anyhow, thiserror, dlopen, signal handling
**Version**: 1.0.0
**Date**: 2025-10-25
**Status**: Draft

---

## Executive Summary

This specification defines comprehensive error handling enhancements across the C++ wrapper, FFI boundary, and Rust integration layers. The current implementation has well-structured error types but critical gaps in error coverage, logging, timeout protection, and cleanup validation that impact debugging, reliability, and user experience.

### Problem Statement

**Analysis Source**: `/tmp/error_handling_analysis.md` identifies 9 critical gaps:

1. **Silent failures** in optional symbol resolution (no distinction between compile-time vs runtime unavailability)
2. **Missing logging** in C++ bridge (errors stored in thread-local but never logged)
3. **No timeout mechanism** for C++ operations (hangs block Rust threads indefinitely)
4. **Incomplete cleanup validation** (resource leaks on error paths go undetected)
5. **Limited error context** in FFI boundary (Pass 1 vs Pass 2 errors indistinguishable)
6. **Incomplete error enums** (missing 8+ variants: LibraryNotFound, SymbolNotFound, OutOfMemory, etc.)
7. **Silent fallback** in quantization (no indication whether real kernel or fallback used)
8. **No diagnostics infrastructure** (no CLI flag for FFI debugging)
9. **52 ignored error tests** representing unimplemented error path coverage

### Key Goals

- **Priority 1 (4-6 hours)**: Timeout mechanism, cleanup validation, C++ error logging
- **Priority 2 (5-7 hours)**: Expand error enum, pass-phase distinction, error context
- **Priority 3 (13-17 hours)**: Diagnostics flag, implement 52 ignored error tests

### Expected Impact

- **Debugging**: Clear error messages with context (operation, phase, C++ error codes)
- **Reliability**: No hangs, validated cleanup, memory leak detection
- **User Experience**: Actionable error messages guiding users to fix root causes
- **Test Coverage**: 52 error path tests validate all failure scenarios

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Error Type Taxonomy](#2-error-type-taxonomy)
3. [Technical Design](#3-technical-design)
4. [Logging Infrastructure](#4-logging-infrastructure)
5. [Timeout Mechanism](#5-timeout-mechanism)
6. [Cleanup Validation](#6-cleanup-validation)
7. [Implementation Strategy](#7-implementation-strategy)
8. [Testing Requirements](#8-testing-requirements)
9. [Acceptance Criteria](#9-acceptance-criteria)

---

## 1. Problem Statement

### 1.1 Current Error Flow

**Layer 1: C++ Bridge** (`bitnet-kernels/src/ffi/cpp_bridge.cpp`)
- **Mechanism**: Exception → thread-local `std::string last_error` → `bitnet_cpp_get_last_error()`
- **Strengths**: Captures exception messages, handles unknown exceptions
- **Gaps**: No logging, fallback silent, error codes unused, no input validation

**Layer 2: FFI Bindings** (`crossval/src/cpp_bindings.rs`)
- **Mechanism**: Two-pass buffer negotiation, null pointer validation, RAII cleanup
- **Strengths**: Position-aware errors, sanity checks, UTF-8 graceful fallback
- **Gaps**: No timeout, cleanup not validated, buffer truncation, Pass 1/2 indistinguishable

**Layer 3: Rust Integration** (`xtask/src/main.rs`, `crossval/src/validation.rs`)
- **Mechanism**: `anyhow::Result` with structured exit codes
- **Strengths**: Flexible error context, scripting-friendly exit codes
- **Gaps**: Limited context, no fallback diagnostics, "error in Result struct" pattern

### 1.2 Critical Gaps (Detailed Analysis)

#### Gap 1: Silent Symbol Resolution Fallback

**Location**: `crossval/src/cpp_bindings.rs` (lines 304-309)

**Problem**: No distinction between:
1. Compile-time feature missing (reported via `CppNotAvailable`)
2. Runtime symbol missing (crashes or undefined behavior)
3. Runtime library missing (crashes or undefined behavior)

**Impact**: Test files document **40+ ignored tests** for missing error path coverage:

```rust
#[test]
#[ignore] // TODO: Implement CppNotAvailable error handling
fn test_error_cpp_not_available() {
    todo!("Implement CppNotAvailable error when FFI not compiled");
}

#[test]
#[ignore] // TODO: Implement LibraryNotFound error handling
fn test_error_library_not_found() {
    todo!("Implement LibraryNotFound error when libbitnet.so missing at runtime");
}

#[test]
#[ignore] // TODO: Implement SymbolNotFound error handling
fn test_error_symbol_not_found_required() {
    todo!("Implement SymbolNotFound error for required symbols");
}
```

**Evidence**: From `/tmp/error_handling_analysis.md` lines 408-423

#### Gap 2: Missing Error Logging in C++ Bridge

**Location**: `crates/bitnet-kernels/src/ffi/cpp_bridge.cpp` (lines 114-120)

**Current Behavior**:
```cpp
} catch (const std::exception& e) {
    set_last_error(e.what());  // Stored in thread-local, NEVER LOGGED
    return -1;
}
```

**Problem**:
- C++ errors disappear if Rust doesn't call `bitnet_cpp_get_last_error()`
- No audit trail of what went wrong
- Difficult to debug production issues

**Evidence**: From `/tmp/error_handling_analysis.md` lines 452-470

#### Gap 3: Buffer Size Assumptions

**Location**: `crossval/src/cpp_bindings.rs` (lines 328, 468)

**Current Pattern**:
```rust
let mut err_buf = vec![0u8; 512];  // Hardcoded 512 bytes

let error_msg = std::str::from_utf8(&err_buf)
    .unwrap_or("unknown error")
    .trim_end_matches('\0');  // If C++ wrote >512 bytes, truncated
```

**Problem**:
- Detailed error messages are truncated
- No way to know if error was truncated
- Difficult to debug issues with detailed error context

**Evidence**: From `/tmp/error_handling_analysis.md` lines 472-489

#### Gap 4: Incomplete Resource Cleanup Validation

**Location**: `crossval/src/cpp_bindings.rs` (lines 861-872)

**Current Pattern**:
```rust
impl Drop for BitnetSession {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                let _ = bitnet_cpp_free_context(self.ctx);  // Ignores cleanup errors
            }
            self.ctx = std::ptr::null_mut();
        }
    }
}
```

**Problem**: No validation that cleanup succeeded. If C++ `free_context` fails:
- Memory might leak
- Subsequent operations on freed memory are possible
- No way to know cleanup failed

**Test Coverage**: `crossval/tests/ffi_error_tests.rs` lines 413-482:
```rust
#[test]
#[ignore] // TODO: Implement cleanup on error validation
fn test_error_cleanup_on_session_creation_failure() {
    todo!("Implement cleanup on error validation (run with valgrind)");
}
```

**Evidence**: From `/tmp/error_handling_analysis.md` lines 491-521

#### Gap 5: No Timeout for C++ Operations

**Problem**: C++ calls (tokenize, evaluate) could hang indefinitely.

**Impact**:
- Long-running operations block Rust thread
- No way to interrupt hung inference
- Test suites can timeout

**Example Need**: `crossval/tests/ffi_socket_tests.rs` requires:
```rust
// Not implemented:
let result = with_timeout(
    Duration::from_secs(30),
    || session.evaluate(&tokens)
)?;
```

**Evidence**: From `/tmp/error_handling_analysis.md` lines 523-539

#### Gap 6: Null Pointer Handling Inconsistency

**Location**: `crossval/src/cpp_bindings.rs` (lines 174-176, 232-234)

**Current Pattern**:
```rust
pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<Vec<u32>> {
    if self.handle.is_null() {
        return Err(CrossvalError::InferenceError("Model handle is null".to_string()));
    }
    // ... proceeds with inference ...
}
```

**Problem**: Doesn't explain why handle is null:
- Was load attempted?
- Did load fail?
- Was model dropped?

**Better Error**:
```rust
if self.handle.is_null() {
    return Err(CrossvalError::InferenceError(
        "Model not loaded. Did CppModel::load() fail? Check earlier errors.".to_string()
    ));
}
```

**Evidence**: From `/tmp/error_handling_analysis.md` lines 540-565

#### Gap 7: Incomplete Error Enums

**Location**: `crossval/src/lib.rs` (lines 87-105)

**Current State**: 6 variants
```rust
#[derive(thiserror::Error, Debug)]
pub enum CrossvalError {
    #[error("C++ implementation not available")]
    CppNotAvailable,                    // ✓ Covered

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),             // ✓ Covered

    #[error("Inference failed: {0}")]
    InferenceError(String),             // ✓ Covered

    #[error("Numerical comparison failed: {0}")]
    ComparisonError(String),            // ✓ Covered

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),    // ✓ Covered

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),  // ✓ Covered
}
```

**Missing Variants** (8+):
- `LibraryNotFound` (dlopen failed)
- `SymbolNotFound` (dlsym failed for required symbol)
- `OptionalSymbolMissing` (dlsym failed for optional symbol, fallback available)
- `OutOfMemory` (C++ malloc failed)
- `ContextOverflow` (too many tokens for context size)
- `ThreadSafetyError` (race condition in C++)
- `CleanupFailed` (C++ free_context failed)
- `OperationTimeout` (C++ operation exceeded timeout)

**Evidence**: From `/tmp/error_handling_analysis.md` lines 567-597

#### Gap 8: C++ Error Codes Not Mapped to Rust Errors

**Location**: `bitnet.h` defines error codes, but `cpp_bridge.cpp` doesn't return them

**Declared but Unused**:
```c
// Declared in bitnet.h but never used in practice:
#define BITNET_ERROR_OUT_OF_MEMORY -5
#define BITNET_ERROR_THREAD_SAFETY -6
#define BITNET_ERROR_CONTEXT_LENGTH_EXCEEDED -8
```

**Impact**: Can't distinguish between:
- Memory exhausted vs general failure
- Thread safety issue vs other error
- Context too large vs other error

**Evidence**: From `/tmp/error_handling_analysis.md` lines 598-615

#### Gap 9: Silent Failures in Quantization

**Location**: `cpp_bridge.cpp` (lines 152-218)

**Current Pattern**:
```cpp
#ifdef HAVE_GGML_BITNET_H
    switch (qtype) {
        case 0: return bitnet_cpp_quantize_i2s(...);  // Real kernel
        // ...
    }
#else
    // Fallback: implement simple quantization (lines 166-217)
    // NO ERROR, just succeeds with potentially incorrect results
#endif

return 0;  // Success (but maybe not real success!)
```

**Impact**:
- Caller doesn't know if real kernel or fallback was used
- Numerical results silently differ
- Hard to debug why quantization differs between runs

**Evidence**: From `/tmp/error_handling_analysis.md` lines 616-640

### 1.3 Test Status

**Total Ignored Tests**: 52 tests (documented in test files)

**Error Categories** (from `crossval/tests/ffi_error_tests.rs`, `ffi_fallback_tests.rs`):

1. Library availability (3 tests) - Lines 36-87
2. Symbol resolution (3 tests) - Lines 127-150
3. Model loading (3 tests) - TODO markers
4. Inference operations (3 tests) - TODO markers
5. Buffer negotiation (2 tests) - TODO markers
6. Cleanup on error (3 tests) - Lines 413-482
7. Error message quality (2 tests) - TODO markers
8. Fallback chain (6 tests) - Lines 36-100
9. Symbol resolution fallback (2 tests) - Lines 36-58
10. Fallback performance (1 test) - TODO marker
11. Fallback consistency (2 tests) - Lines 65-87
12. Fallback diagnostics (2 tests) - Lines 95-100

**Evidence**: From `/tmp/error_handling_analysis.md` lines 1009-1028

---

## 2. Error Type Taxonomy

### 2.1 Error Classification

**By Layer**:
- **C++ Layer**: Exception handling, input validation, resource management
- **FFI Layer**: Symbol resolution, buffer negotiation, null pointer handling
- **Rust Layer**: Context chaining, structured errors, exit codes

**By Severity**:
- **Critical**: Hangs, crashes, memory corruption, resource leaks
- **Recoverable**: Invalid input, missing resources, API misuse
- **Informational**: Fallback used, optional feature unavailable, warnings

**By Phase**:
- **Compile-Time**: Feature not enabled, header not found
- **Link-Time**: Symbol not found, library not found
- **Runtime**: Invalid input, inference failure, timeout

### 2.2 New Error Variants (Priority 2)

#### Expanded CrossvalError Enum

```rust
// crossval/src/lib.rs

#[derive(thiserror::Error, Debug)]
pub enum CrossvalError {
    // ===================================================================
    // Existing Variants (Keep)
    // ===================================================================

    #[error("C++ implementation not available (compile with --features crossval)")]
    CppNotAvailable,

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference failed: {0}")]
    InferenceError(String),

    #[error("Numerical comparison failed: {0}")]
    ComparisonError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    // ===================================================================
    // NEW: Library and Symbol Resolution Errors
    // ===================================================================

    /// C++ library not found at runtime
    ///
    /// Actionable error message guides user to set environment variables:
    /// - BITNET_CPP_DIR: Path to BitNet.cpp installation
    /// - LD_LIBRARY_PATH: Dynamic loader search path (Linux)
    /// - DYLD_LIBRARY_PATH: Dynamic loader search path (macOS)
    ///
    /// Example:
    /// ```
    /// C++ library not found: libbitnet.so
    ///
    /// Set BITNET_CPP_DIR or run:
    ///   eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"
    ///
    /// Or manually set library path:
    ///   export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH
    /// ```
    #[error("C++ library not found: {0}\n\nSet BITNET_CPP_DIR or run:\n  eval \"$(cargo run -p xtask -- setup-cpp-auto --emit=sh)\"\n\nOr manually set library path:\n  export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH")]
    LibraryNotFound(String),

    /// Required C++ symbol not found (version mismatch)
    ///
    /// This indicates Rust bindings and C++ library are out of sync.
    ///
    /// Actionable error message guides user to:
    /// 1. Verify BitNet.cpp version matches expected version
    /// 2. Rebuild BitNet.cpp with same compiler/flags as Rust FFI
    /// 3. Check for symbol name mangling issues
    ///
    /// Example:
    /// ```
    /// Required C++ symbol not found: bitnet_cpp_init_context
    ///
    /// Mismatch between Rust bindings and C++ library version.
    ///
    /// Expected C++ API version: 0.2.x
    /// Found C++ library: /path/to/libbitnet.so
    ///
    /// Rebuild BitNet.cpp or downgrade Rust bindings.
    /// ```
    #[error("Required C++ symbol not found: {0}\n\nMismatch between Rust bindings and C++ library version.\n\nRebuild BitNet.cpp or check version compatibility.")]
    SymbolNotFound(String),

    /// Optional C++ symbol not found, using fallback
    ///
    /// This is an informational warning, not a hard error.
    /// The operation will succeed using a fallback implementation (e.g., llama.cpp).
    ///
    /// Example:
    /// ```
    /// Optional C++ symbol not found, falling back: bitnet_cpp_tokenize_with_context
    ///
    /// Using llama.cpp fallback for tokenization.
    /// This is expected for embedded BitNet.cpp builds.
    ///
    /// Performance impact: minimal (< 5% overhead)
    /// ```
    #[error("Optional C++ symbol not found, falling back: {0}\n\nThis is expected behavior when using llama.cpp fallback.")]
    OptionalSymbolMissing(String),

    // ===================================================================
    // NEW: Resource Management Errors
    // ===================================================================

    /// Out of memory (C++ malloc/new failed)
    ///
    /// Actionable error message guides user to:
    /// 1. Reduce context size (--n-ctx parameter)
    /// 2. Reduce batch size
    /// 3. Enable GPU offloading to free CPU memory
    ///
    /// Example:
    /// ```
    /// Out of memory: Failed to allocate 2048 MB for model weights
    ///
    /// Try reducing context size:
    ///   --n-ctx 512  (current: 2048)
    ///
    /// Or enable GPU offloading:
    ///   --n-gpu-layers 32
    /// ```
    #[error("Out of memory: {0}\n\nTry reducing context size or enabling GPU offloading.")]
    OutOfMemory(String),

    /// Context size exceeded (too many tokens)
    ///
    /// Actionable error message guides user to:
    /// 1. Increase context size (--n-ctx parameter)
    /// 2. Reduce input prompt length
    /// 3. Use sliding window context
    ///
    /// Example:
    /// ```
    /// Context size exceeded: 1024 tokens > 512 context size
    ///
    /// Increase context size:
    ///   --n-ctx 2048  (current: 512)
    ///
    /// Or reduce prompt length.
    /// ```
    #[error("Context size exceeded: {0}\n\nIncrease --n-ctx or reduce prompt length.")]
    ContextOverflow(String),

    /// Thread safety violation (race condition detected)
    ///
    /// This indicates improper concurrent access to C++ objects.
    ///
    /// Actionable error message guides user to:
    /// 1. Ensure BitnetSession is not shared across threads without synchronization
    /// 2. Use separate sessions per thread
    /// 3. Enable thread-safe mode (if available)
    ///
    /// Example:
    /// ```
    /// Thread safety violation: Concurrent access to BitnetContext detected
    ///
    /// BitnetSession is not thread-safe. Use one of:
    /// 1. Separate sessions per thread
    /// 2. Mutex/RwLock around session access
    /// 3. Arc<Mutex<BitnetSession>> for shared access
    /// ```
    #[error("Thread safety violation: {0}\n\nBitnetSession is not thread-safe. Use separate sessions per thread.")]
    ThreadSafetyError(String),

    // ===================================================================
    // NEW: Cleanup Errors
    // ===================================================================

    /// Resource cleanup failed (memory leak possible)
    ///
    /// This indicates C++ destructor or cleanup function failed.
    /// While not immediately fatal, repeated failures can cause memory leaks.
    ///
    /// Actionable error message guides user to:
    /// 1. Run with valgrind to detect leaks
    /// 2. Check for corrupted context handles
    /// 3. Report issue if reproducible
    ///
    /// Example:
    /// ```
    /// Resource cleanup failed: bitnet_cpp_free_context returned -1
    ///
    /// This may indicate a memory leak.
    ///
    /// Run with leak detection:
    ///   valgrind --leak-check=full cargo test <test_name>
    ///
    /// If reproducible, report issue with backtrace.
    /// ```
    #[error("Resource cleanup failed: {0}\n\nThis may indicate a memory leak. Run with valgrind.")]
    CleanupFailed(String),

    // ===================================================================
    // NEW: Timeout Errors
    // ===================================================================

    /// Operation timed out (C++ call exceeded timeout)
    ///
    /// Actionable error message guides user to:
    /// 1. Increase timeout (--timeout parameter)
    /// 2. Reduce input size
    /// 3. Check for infinite loops in model
    ///
    /// Example:
    /// ```
    /// Operation timed out after 30s
    ///
    /// C++ operation exceeded timeout. Try:
    /// 1. Increase timeout: --timeout 60
    /// 2. Reduce input size: --max-tokens 32
    /// 3. Check model for issues
    ///
    /// If model hangs consistently, report issue.
    /// ```
    #[error("Operation timed out after {0:?}\n\nIncrease --timeout or reduce input size.")]
    OperationTimeout(std::time::Duration),
}
```

### 2.3 Error Message Guidelines

**Actionable Error Format**:
1. **What happened**: Clear description of error
2. **Why it happened**: Root cause (if known)
3. **How to fix it**: Concrete steps to resolve
4. **Where to look**: Relevant logs, files, environment variables

**Example Template**:
```
<Error Type>: <Specific Error>

<Root Cause>

<Actionable Steps>:
1. <Step 1>
2. <Step 2>
3. <Step 3>

<Additional Context>
```

---

## 3. Technical Design

### 3.1 Priority 1: Critical Safety (4-6 hours)

#### 3.1.1 Timeout Mechanism

**Objective**: Prevent C++ hangs from blocking Rust threads indefinitely.

**Design**: Thread-based timeout with channel communication

**Location**: `crossval/src/cpp_bindings.rs`

**Implementation**:

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Wrapper for timeout-protected C++ calls
fn with_timeout<F, T>(
    timeout: Duration,
    operation: F,
) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = mpsc::channel();

    let handle = thread::spawn(move || {
        let result = operation();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            // Thread still running - we leak it (no safe way to kill)
            // Log warning for debugging
            eprintln!("WARNING: C++ operation timed out after {:?}, thread leaked", timeout);
            Err(CrossvalError::OperationTimeout(timeout))
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            // Thread panicked
            Err(CrossvalError::InferenceError(
                "C++ operation thread panicked".to_string()
            ))
        }
    }
}

impl BitnetSession {
    /// Evaluate tokens with timeout protection
    pub fn evaluate_with_timeout(
        &self,
        tokens: &[i32],
        timeout: Duration,
    ) -> Result<Vec<Vec<f32>>> {
        let ctx = self.ctx;
        let tokens_copy = tokens.to_vec();

        with_timeout(timeout, move || {
            // Standard unsafe evaluation inside timeout-protected thread
            let mut err_buf = vec![0u8; 512];
            let mut out_rows = 0i32;
            let mut out_cols = 0i32;

            // Pass 1: Query logits size
            let result = unsafe {
                bitnet_cpp_eval_with_context(
                    ctx,
                    tokens_copy.as_ptr(),
                    tokens_copy.len() as i32,
                    0, // seq_id
                    std::ptr::null_mut(),
                    0,
                    &mut out_rows,
                    &mut out_cols,
                    err_buf.as_mut_ptr() as *mut c_char,
                    err_buf.len() as i32,
                )
            };

            if result != 0 {
                let err_msg = extract_error_message(&err_buf);
                return Err(CrossvalError::InferenceError(format!(
                    "Evaluation (query phase) failed: {}",
                    err_msg
                )));
            }

            // Allocate and fill (Pass 2)
            // ... rest of evaluation logic ...

            Ok(logits)
        })
    }
}
```

**Configuration**:
- Default timeout: 30 seconds (configurable via `--timeout` CLI flag)
- Timeout applies to: tokenize, evaluate, model loading operations
- Thread leak on timeout is acceptable (no safe alternative without C++ cooperation)

**Testing**:
```rust
#[test]
fn test_timeout_mechanism() {
    let session = BitnetSession::create(test_model_path(), 512, 0).unwrap();

    // Simulate long-running operation (mock with sleep)
    let result = session.evaluate_with_timeout(
        &[1, 2, 3],
        Duration::from_millis(100),
    );

    match result {
        Err(CrossvalError::OperationTimeout(d)) => {
            assert_eq!(d, Duration::from_millis(100));
        }
        _ => panic!("Expected timeout error"),
    }
}
```

#### 3.1.2 Cleanup Validation

**Objective**: Validate resource cleanup succeeds, detect memory leaks early.

**Design**: Log cleanup failures in debug builds, count active contexts

**Location**: `crossval/src/cpp_bindings.rs`

**Implementation**:

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

// Global counter for active contexts (debug only)
#[cfg(debug_assertions)]
static ACTIVE_CONTEXTS: AtomicUsize = AtomicUsize::new(0);

impl BitnetSession {
    pub fn create(
        model_path: &Path,
        n_ctx: i32,
        n_gpu_layers: i32,
    ) -> Result<Self> {
        // ... existing initialization ...

        // Track active contexts (debug only)
        #[cfg(debug_assertions)]
        {
            let count = ACTIVE_CONTEXTS.fetch_add(1, Ordering::SeqCst);
            eprintln!("DEBUG: BitnetSession created (active contexts: {})", count + 1);
        }

        Ok(Self {
            ctx: ctx_ptr,
            model_path: model_path.to_path_buf(),
            n_ctx,
        })
    }
}

impl Drop for BitnetSession {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                let result = bitnet_cpp_free_context(self.ctx);

                if result != 0 {
                    // Cleanup failed - log in debug, warn in release
                    #[cfg(debug_assertions)]
                    {
                        eprintln!(
                            "ERROR: bitnet_cpp_free_context failed with code {} for {:?}",
                            result, self.model_path
                        );
                        eprintln!("This may indicate a memory leak. Run with valgrind.");
                    }

                    #[cfg(not(debug_assertions))]
                    {
                        // In release, just log to stderr (can't panic in Drop)
                        eprintln!(
                            "WARNING: Resource cleanup failed (code {}), possible memory leak",
                            result
                        );
                    }
                }

                // Track active contexts (debug only)
                #[cfg(debug_assertions)]
                {
                    let count = ACTIVE_CONTEXTS.fetch_sub(1, Ordering::SeqCst);
                    eprintln!("DEBUG: BitnetSession dropped (active contexts: {})", count - 1);
                }
            }
            self.ctx = std::ptr::null_mut();
        }
    }
}

/// Get count of active contexts (debug builds only, for testing)
#[cfg(debug_assertions)]
pub fn active_context_count() -> usize {
    ACTIVE_CONTEXTS.load(Ordering::SeqCst)
}
```

**Testing**:
```rust
#[test]
#[cfg(debug_assertions)]
fn test_cleanup_validation_detects_leaks() {
    let initial_count = active_context_count();

    {
        let session = BitnetSession::create(test_model_path(), 512, 0).unwrap();
        assert_eq!(active_context_count(), initial_count + 1);

        // Session dropped here
    }

    // Validate cleanup happened
    assert_eq!(active_context_count(), initial_count);
}

#[test]
fn test_cleanup_on_error_paths() {
    // Try to create session with invalid model
    let result = BitnetSession::create(
        Path::new("/nonexistent/model.gguf"),
        512,
        0,
    );

    assert!(result.is_err());

    // Validate no leaked contexts
    // Run with valgrind:
    // valgrind --leak-check=full cargo test test_cleanup_on_error_paths
}
```

#### 3.1.3 C++ Error Logging

**Objective**: Log C++ errors immediately for debugging, maintain audit trail.

**Design**: stderr logging at C++ layer, structured log macros

**Location**: `crates/bitnet-kernels/src/ffi/cpp_bridge.cpp`

**Implementation**:

```cpp
#include <iostream>
#include <cstring>
#include <ctime>
#include <iomanip>

// Logging levels
enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

// Current log level (configurable via environment variable)
static LogLevel g_log_level = LogLevel::INFO;

// Initialize log level from environment variable
static void init_log_level() {
    static bool initialized = false;
    if (initialized) return;

    const char* env = std::getenv("BITNET_CPP_LOG_LEVEL");
    if (env) {
        std::string level(env);
        if (level == "DEBUG") g_log_level = LogLevel::DEBUG;
        else if (level == "INFO") g_log_level = LogLevel::INFO;
        else if (level == "WARN") g_log_level = LogLevel::WARN;
        else if (level == "ERROR") g_log_level = LogLevel::ERROR;
    }

    initialized = true;
}

// Log message with timestamp and level
static void log_message(LogLevel level, const char* component, const char* message) {
    init_log_level();

    if (level < g_log_level) return;

    // Get current timestamp
    auto now = std::time(nullptr);
    auto tm = std::localtime(&now);

    // Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] [Component] Message
    std::cerr << "[" << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "] ";

    switch (level) {
        case LogLevel::DEBUG: std::cerr << "[DEBUG] "; break;
        case LogLevel::INFO:  std::cerr << "[INFO] ";  break;
        case LogLevel::WARN:  std::cerr << "[WARN] ";  break;
        case LogLevel::ERROR: std::cerr << "[ERROR] "; break;
    }

    std::cerr << "[" << component << "] " << message << std::endl;
}

// Convenience macros
#define LOG_DEBUG(component, message) log_message(LogLevel::DEBUG, component, message)
#define LOG_INFO(component, message)  log_message(LogLevel::INFO, component, message)
#define LOG_WARN(component, message)  log_message(LogLevel::WARN, component, message)
#define LOG_ERROR(component, message) log_message(LogLevel::ERROR, component, message)

// Enhanced error setter with logging
static void set_last_error_with_logging(const char* message, const char* component) {
    // Log to stderr immediately
    LOG_ERROR(component, message);

    // Also store in thread_local for API access
    last_error = message;
}

// Updated C++ functions with logging
int bitnet_cpp_init_context(
    bitnet_context_t** out_ctx,
    const char* model_path,
    int32_t n_ctx,
    int32_t n_gpu_layers,
    char* err,
    int32_t err_len
) {
    try {
        LOG_INFO("init_context", "Initializing BitNet context");

        if (!model_path || !out_ctx) {
            set_last_error_with_logging("Invalid arguments", "init_context");
            return -1;
        }

        LOG_DEBUG("init_context",
            fmt::format("model_path={}, n_ctx={}, n_gpu_layers={}",
                model_path, n_ctx, n_gpu_layers).c_str());

        // ... actual initialization ...

        LOG_INFO("init_context", "Context initialized successfully");
        return 0;

    } catch (const std::exception& e) {
        set_last_error_with_logging(e.what(), "init_context");
        return -1;
    } catch (...) {
        set_last_error_with_logging("Unknown error", "init_context");
        return -1;
    }
}

int bitnet_cpp_matmul_i2s(
    const int8_t* a,
    const uint8_t* b,
    float* c,
    int m, int n, int k
) {
    try {
        if (!a || !b || !c) {
            set_last_error_with_logging("Null pointer in matmul_i2s", "matmul");
            return -1;
        }

        if (m <= 0 || n <= 0 || k <= 0) {
            std::string msg = fmt::format(
                "Invalid dimensions: m={}, n={}, k={}", m, n, k);
            set_last_error_with_logging(msg.c_str(), "matmul");
            return -1;
        }

        LOG_DEBUG("matmul", fmt::format("matmul_i2s: {}x{}x{}", m, n, k).c_str());

        // ... actual matmul ...

        return 0;

    } catch (const std::exception& e) {
        set_last_error_with_logging(e.what(), "matmul");
        return -1;
    } catch (...) {
        set_last_error_with_logging("Unknown matmul error", "matmul");
        return -1;
    }
}
```

**Configuration**:
- `BITNET_CPP_LOG_LEVEL=DEBUG|INFO|WARN|ERROR` (default: INFO)
- Logs to stderr (not buffered, immediately visible)
- Structured format for easy parsing

**Testing**:
```rust
#[test]
fn test_cpp_error_logging() {
    use std::process::Command;

    // Run test with ERROR log level
    let output = Command::new("cargo")
        .args(&["test", "test_invalid_model_path", "--", "--nocapture"])
        .env("BITNET_CPP_LOG_LEVEL", "ERROR")
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Validate error was logged
    assert!(stderr.contains("[ERROR] [init_context]"));
    assert!(stderr.contains("Model loading failed"));
}
```

### 3.2 Priority 2: Error Completeness (5-7 hours)

#### 3.2.1 Expand Error Enum

**Objective**: Add 8+ missing error variants for complete error coverage.

**Implementation**: See Section 2.2 for complete `CrossvalError` enum expansion.

**Key Changes**:
- `LibraryNotFound(String)` with LD_LIBRARY_PATH guidance
- `SymbolNotFound(String)` with version mismatch guidance
- `OptionalSymbolMissing(String)` for fallback scenarios
- `OutOfMemory(String)` with context size reduction guidance
- `ContextOverflow(String)` with context size increase guidance
- `ThreadSafetyError(String)` with concurrency guidance
- `CleanupFailed(String)` with valgrind guidance
- `OperationTimeout(Duration)` with timeout increase guidance

**Migration Strategy**:
1. Add new variants to `CrossvalError` enum
2. Update `From` implementations for new error types
3. Update error construction sites to use specific variants
4. Update tests to validate new error types
5. Update documentation with error handling examples

#### 3.2.2 Pass-Phase Distinction

**Objective**: Distinguish between Pass 1 (query) and Pass 2 (fill) errors in two-pass pattern.

**Location**: `crossval/src/cpp_bindings.rs`

**Implementation**:

```rust
pub fn tokenize_bitnet(
    model_path: &Path,
    prompt: &str,
    add_bos: bool,
    parse_special: bool,
) -> Result<Vec<i32>> {
    // ... setup ...

    // Pass 1: Query size
    let result = unsafe {
        crossval_bitnet_tokenize(
            model_path_c.as_ptr(),
            prompt_c.as_ptr(),
            if add_bos { 1 } else { 0 },
            if parse_special { 1 } else { 0 },
            std::ptr::null_mut(),  // NULL buffer signals query mode
            0,
            &mut out_len,
            err_buf.as_mut_ptr() as *mut c_char,
            err_buf.len() as i32,
        )
    };

    if result != 0 {
        let err_msg = extract_error_message(&err_buf);
        return Err(CrossvalError::InferenceError(format!(
            "BitNet tokenization (QUERY PHASE) failed: {} (result code: {})\n\
             Model: {:?}\n\
             Prompt length: {} chars",
            err_msg, result, model_path, prompt.len()
        )));
    }

    // Sanity checks
    if out_len <= 0 {
        return Ok(Vec::new());
    }

    if out_len > 100_000 {
        return Err(CrossvalError::InferenceError(format!(
            "Unreasonable token count from BitNet.cpp (QUERY PHASE): {}\n\
             This may indicate a bug in C++ tokenizer or corrupted model.",
            out_len
        )));
    }

    // Pass 2: Allocate and fill
    let mut tokens = vec![0i32; out_len as usize];
    let result = unsafe {
        crossval_bitnet_tokenize(
            model_path_c.as_ptr(),
            prompt_c.as_ptr(),
            if add_bos { 1 } else { 0 },
            if parse_special { 1 } else { 0 },
            tokens.as_mut_ptr(),
            out_len,
            &mut out_len,
            err_buf.as_mut_ptr() as *mut c_char,
            err_buf.len() as i32,
        )
    };

    if result != 0 {
        let err_msg = extract_error_message(&err_buf);
        return Err(CrossvalError::InferenceError(format!(
            "BitNet tokenization (FILL PHASE) failed: {} (result code: {})\n\
             Model: {:?}\n\
             Expected tokens: {}\n\
             This may indicate buffer corruption or C++ state change.",
            err_msg, result, model_path, out_len
        )));
    }

    // Verify return value consistency
    if out_len < 0 {
        return Err(CrossvalError::InferenceError(
            "BitNet.cpp returned negative token count (FILL PHASE)\n\
             This indicates a serious bug in C++ implementation.".to_string(),
        ));
    }

    tokens.truncate(out_len as usize);
    Ok(tokens)
}
```

**Key Improvements**:
- **Phase labels**: "(QUERY PHASE)" vs "(FILL PHASE)" in error messages
- **Context enrichment**: Include model path, prompt length, expected token count
- **Debugging hints**: Suggest possible root causes (corruption, state change, C++ bug)

### 3.3 Priority 3: Diagnostics and Testing (13-17 hours)

#### 3.3.1 Error Context struct

**Objective**: Structured error context for debugging.

**Location**: New file `crossval/src/error_context.rs`

**Implementation**:

```rust
use std::fmt;
use std::time::SystemTime;

/// Structured error context for debugging FFI errors
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed (e.g., "tokenize", "evaluate", "load_model")
    pub operation: String,

    /// Phase within operation (e.g., "query", "fill", "setup")
    pub phase: Option<String>,

    /// Input parameters (e.g., "model=/path/to/model.gguf, n_ctx=512")
    pub input_params: String,

    /// C++ error code (if available)
    pub c_error_code: Option<i32>,

    /// C++ error message (if available)
    pub c_error_message: Option<String>,

    /// Timestamp of error
    pub timestamp: SystemTime,

    /// Thread ID (for diagnosing race conditions)
    pub thread_id: Option<std::thread::ThreadId>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            phase: None,
            input_params: String::new(),
            c_error_code: None,
            c_error_message: None,
            timestamp: SystemTime::now(),
            thread_id: Some(std::thread::current().id()),
        }
    }

    /// Set phase
    pub fn with_phase(mut self, phase: impl Into<String>) -> Self {
        self.phase = Some(phase.into());
        self
    }

    /// Set input parameters
    pub fn with_params(mut self, params: impl Into<String>) -> Self {
        self.input_params = params.into();
        self
    }

    /// Set C++ error code
    pub fn with_c_error_code(mut self, code: i32) -> Self {
        self.c_error_code = Some(code);
        self
    }

    /// Set C++ error message
    pub fn with_c_error_message(mut self, message: impl Into<String>) -> Self {
        self.c_error_message = Some(message.into());
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}): {}",
            self.operation,
            self.phase.as_deref().unwrap_or("N/A"),
            self.input_params
        )?;

        if let Some(code) = self.c_error_code {
            write!(f, " [C++ error code: {}]", code)?;
        }

        if let Some(msg) = &self.c_error_message {
            write!(f, " [C++ message: {}]", msg)?;
        }

        if let Some(thread_id) = self.thread_id {
            write!(f, " [thread: {:?}]", thread_id)?;
        }

        Ok(())
    }
}

/// Extension trait for adding context to errors
pub trait WithContext<T> {
    fn with_context(self, ctx: ErrorContext) -> Result<T, CrossvalError>;
}

impl<T, E> WithContext<T> for Result<T, E>
where
    E: Into<CrossvalError>,
{
    fn with_context(self, ctx: ErrorContext) -> Result<T, CrossvalError> {
        self.map_err(|e| {
            let mut base_error = e.into();
            // Enrich error with context
            match &mut base_error {
                CrossvalError::InferenceError(msg) => {
                    *msg = format!("{}\n\nContext: {}", msg, ctx);
                }
                CrossvalError::ModelLoadError(msg) => {
                    *msg = format!("{}\n\nContext: {}", msg, ctx);
                }
                _ => {}
            }
            base_error
        })
    }
}
```

**Usage Example**:

```rust
pub fn tokenize_bitnet(
    model_path: &Path,
    prompt: &str,
    add_bos: bool,
    parse_special: bool,
) -> Result<Vec<i32>> {
    let ctx = ErrorContext::new("tokenize")
        .with_params(format!(
            "model={:?}, prompt_len={}, add_bos={}, parse_special={}",
            model_path, prompt.len(), add_bos, parse_special
        ));

    // Pass 1: Query size
    let result = /* ... */;

    if result != 0 {
        let err_msg = extract_error_message(&err_buf);
        return Err(CrossvalError::InferenceError(format!(
            "BitNet tokenization (QUERY PHASE) failed: {}",
            err_msg
        )))
        .with_context(
            ctx.clone()
                .with_phase("query")
                .with_c_error_code(result)
                .with_c_error_message(err_msg)
        );
    }

    // ... rest of implementation ...
}
```

#### 3.3.2 Diagnostic CLI Flag

**Objective**: Add `--dlopen-diagnostics` flag for FFI debugging.

**Location**: `xtask/src/crossval/preflight.rs`

**Implementation**:

```rust
// xtask/src/crossval/preflight.rs

use anyhow::Result;
use crossval::backend::CppBackend;

/// Enhanced preflight command with diagnostics
pub fn preflight(backend: Option<CppBackend>, verbose: bool, diagnostics: bool) -> Result<()> {
    if diagnostics {
        println!("\n=== FFI Diagnostics ===\n");

        // Environment variables
        println!("Environment Variables:");
        println!("  BITNET_CPP_DIR: {:?}", std::env::var("BITNET_CPP_DIR"));
        println!("  LD_LIBRARY_PATH: {:?}", std::env::var("LD_LIBRARY_PATH"));
        println!("  DYLD_LIBRARY_PATH: {:?}", std::env::var("DYLD_LIBRARY_PATH"));
        println!("  BITNET_CPP_LOG_LEVEL: {:?}", std::env::var("BITNET_CPP_LOG_LEVEL"));
        println!();

        // Compile-time detection
        println!("Compile-Time Detection:");
        println!("  CROSSVAL_HAS_BITNET: {}", crossval::HAS_BITNET);
        println!("  CROSSVAL_HAS_LLAMA: {}", crossval::HAS_LLAMA);
        println!("  BACKEND_STATE: {}", crossval::BACKEND_STATE);
        println!();

        // Runtime library discovery
        println!("Runtime Library Discovery:");
        match discover_bitnet_libraries() {
            Ok(paths) => {
                println!("  ✓ BitNet.cpp libraries found:");
                for path in paths {
                    println!("    - {:?}", path);
                }
            }
            Err(e) => {
                println!("  ✗ BitNet.cpp libraries not found: {}", e);
            }
        }

        match discover_llama_libraries() {
            Ok(paths) => {
                println!("  ✓ llama.cpp libraries found:");
                for path in paths {
                    println!("    - {:?}", path);
                }
            }
            Err(e) => {
                println!("  ✗ llama.cpp libraries not found: {}", e);
            }
        }
        println!();

        // Symbol resolution (if libraries found)
        if crossval::HAS_BITNET {
            println!("Symbol Resolution (BitNet.cpp):");
            check_symbol_availability("bitnet_cpp_init_context", true);
            check_symbol_availability("bitnet_cpp_free_context", true);
            check_symbol_availability("bitnet_cpp_tokenize_with_context", false);
            check_symbol_availability("bitnet_cpp_eval_with_context", false);
            println!();
        }

        if crossval::HAS_LLAMA {
            println!("Symbol Resolution (llama.cpp):");
            check_symbol_availability("crossval_bitnet_tokenize", true);
            check_symbol_availability("crossval_bitnet_eval_with_tokens", true);
            println!();
        }
    }

    // Standard preflight checks
    // ... rest of implementation ...

    Ok(())
}

/// Check if symbol is available at runtime
fn check_symbol_availability(symbol_name: &str, required: bool) {
    // Attempt dynamic symbol resolution (pseudo-code, requires dlopen wrapper)
    match try_resolve_symbol(symbol_name) {
        Ok(_) => {
            let status = if required { "✓ REQUIRED" } else { "✓ OPTIONAL" };
            println!("  {} {}", status, symbol_name);
        }
        Err(_) => {
            let status = if required { "✗ REQUIRED (MISSING)" } else { "⚠ OPTIONAL (missing, fallback available)" };
            println!("  {} {}", status, symbol_name);
        }
    }
}

/// Discover BitNet.cpp library paths
fn discover_bitnet_libraries() -> Result<Vec<std::path::PathBuf>> {
    // Search standard locations for libbitnet.so/dylib
    // Return list of found libraries
    todo!("Implement library discovery")
}

/// Discover llama.cpp library paths
fn discover_llama_libraries() -> Result<Vec<std::path::PathBuf>> {
    // Search standard locations for libllama.so/dylib, libggml.so/dylib
    // Return list of found libraries
    todo!("Implement library discovery")
}
```

**CLI Integration**:

```rust
// xtask/src/main.rs

#[derive(Parser)]
#[command(name = "xtask")]
struct Args {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Preflight {
        /// Specific backend to check (bitnet or llama)
        #[arg(long)]
        backend: Option<String>,

        /// Enable verbose output
        #[arg(long)]
        verbose: bool,

        /// Enable FFI diagnostics (dlopen, symbol resolution)
        #[arg(long)]
        diagnostics: bool,
    },
    // ... other commands ...
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.cmd {
        Commands::Preflight { backend, verbose, diagnostics } => {
            let backend_enum = backend.map(|s| match s.as_str() {
                "bitnet" => CppBackend::BitNet,
                "llama" => CppBackend::Llama,
                _ => panic!("Invalid backend: {}", s),
            });

            crossval::preflight::preflight(backend_enum, verbose, diagnostics)
        }
        // ... other commands ...
    }
}
```

**Usage Examples**:

```bash
# Standard preflight (compact output)
cargo run -p xtask --features crossval-all -- preflight

# Verbose preflight (backend detection details)
cargo run -p xtask --features crossval-all -- preflight --verbose

# Full diagnostics (environment, libraries, symbols)
cargo run -p xtask --features crossval-all -- preflight --diagnostics

# Check specific backend with diagnostics
cargo run -p xtask --features crossval-all -- preflight --backend bitnet --diagnostics
```

---

## 4. Logging Infrastructure

### 4.1 C++ Logging (Priority 1)

**Implementation**: See Section 3.1.3 for complete C++ logging implementation.

**Key Features**:
- **Structured format**: `[YYYY-MM-DD HH:MM:SS] [LEVEL] [Component] Message`
- **Log levels**: DEBUG, INFO, WARN, ERROR
- **Configuration**: `BITNET_CPP_LOG_LEVEL` environment variable
- **Output**: stderr (unbuffered, immediately visible)
- **Thread-safe**: Per-call logging (no global state mutations)

### 4.2 FFI Error Trace Capture

**Objective**: Capture and log C++ errors from Rust FFI layer.

**Location**: `crossval/src/cpp_bindings.rs`

**Implementation**:

```rust
/// Extract and log C++ error message from buffer
fn extract_and_log_error_message(err_buf: &[u8], operation: &str, phase: &str) -> String {
    let err_msg = std::str::from_utf8(err_buf)
        .unwrap_or("unknown error (invalid UTF-8)")
        .trim_end_matches('\0');

    // Log to Rust logger (if enabled)
    if err_msg != "unknown error (invalid UTF-8)" && !err_msg.is_empty() {
        log::error!(
            "C++ error in {}({}): {}",
            operation,
            phase,
            err_msg
        );
    }

    err_msg.to_string()
}

/// Replace existing `extract_error_message` calls with `extract_and_log_error_message`
pub fn tokenize_bitnet(
    model_path: &Path,
    prompt: &str,
    add_bos: bool,
    parse_special: bool,
) -> Result<Vec<i32>> {
    // ... setup ...

    // Pass 1: Query size
    let result = /* ... */;

    if result != 0 {
        let err_msg = extract_and_log_error_message(&err_buf, "tokenize", "query");
        return Err(CrossvalError::InferenceError(format!(
            "BitNet tokenization (QUERY PHASE) failed: {}",
            err_msg
        )));
    }

    // ... rest ...
}
```

**Configuration**:
- Use `log` crate with `env_logger` or similar
- Configure via `RUST_LOG=crossval=debug` or `RUST_LOG=error`
- Separate from C++ logging (allows independent filtering)

### 4.3 Rust Error Context Chaining

**Objective**: Chain error contexts using `anyhow` for full error traces.

**Location**: `xtask/src/crossval/mod.rs`

**Implementation**:

```rust
use anyhow::{Context, Result};

pub fn run_crossval(
    model_path: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    max_tokens: usize,
) -> Result<()> {
    let session = BitnetSession::create(model_path, 512, 0)
        .context("Failed to create BitNet session")
        .context(format!("Model path: {:?}", model_path))?;

    let tokens = session.tokenize(prompt)
        .context("Failed to tokenize prompt")
        .context(format!("Prompt: {:?}", prompt))?;

    let logits = session.evaluate(&tokens)
        .context("Failed to evaluate tokens")
        .context(format!("Token count: {}", tokens.len()))?;

    // ... rest of cross-validation ...

    Ok(())
}
```

**Error Output Example**:

```
Error: Failed to evaluate tokens

Caused by:
    0: Token count: 42
    1: BitNet inference (FILL PHASE) failed: Invalid context size
    2: Context: evaluate(fill): model=/path/to/model.gguf, tokens=42 [C++ error code: -1] [C++ message: Context size 512 exceeded by token count 1024]
```

---

## 5. Timeout Mechanism

**Complete implementation in Section 3.1.1**.

### 5.1 Design Summary

- **Thread-based timeout**: Spawn operation in separate thread, use `mpsc::channel` for communication
- **Configurable timeout**: Default 30 seconds, configurable via `--timeout` CLI flag
- **Applies to**: tokenize, evaluate, model loading operations
- **Thread leak acceptable**: No safe way to kill thread without C++ cooperation

### 5.2 Configuration

**CLI Flag**:
```bash
cargo run -p xtask --features crossval-all -- crossval-per-token \
  --model model.gguf \
  --tokenizer tokenizer.json \
  --prompt "test" \
  --timeout 60  # 60 second timeout
```

**Environment Variable** (alternative):
```bash
export BITNET_CPP_TIMEOUT=60
cargo run -p xtask --features crossval-all -- crossval-per-token ...
```

### 5.3 Testing

**Test 1: Timeout Triggers**:
```rust
#[test]
fn test_timeout_mechanism_triggers() {
    let session = BitnetSession::create(test_model_path(), 512, 0).unwrap();

    // Simulate long-running operation
    let result = session.evaluate_with_timeout(
        &[1, 2, 3],
        Duration::from_millis(100),
    );

    match result {
        Err(CrossvalError::OperationTimeout(d)) => {
            assert_eq!(d, Duration::from_millis(100));
        }
        _ => panic!("Expected timeout error"),
    }
}
```

**Test 2: Normal Operations Complete**:
```rust
#[test]
fn test_timeout_mechanism_allows_completion() {
    let session = BitnetSession::create(test_model_path(), 512, 0).unwrap();

    // Normal operation should complete within timeout
    let result = session.evaluate_with_timeout(
        &[1, 2, 3],
        Duration::from_secs(30),
    );

    assert!(result.is_ok(), "Normal operation should complete within timeout");
}
```

---

## 6. Cleanup Validation

**Complete implementation in Section 3.1.2**.

### 6.1 Design Summary

- **Debug context counter**: Track active contexts using `AtomicUsize` (debug builds only)
- **Cleanup failure logging**: Log cleanup errors to stderr in debug builds, warn in release
- **Drop validation**: Assert cleanup succeeds in tests
- **Valgrind integration**: Run memory leak detection tests with valgrind

### 6.2 RAII Enforcement

**Pattern**: Use `Drop` trait for automatic cleanup:

```rust
impl Drop for BitnetSession {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                let result = bitnet_cpp_free_context(self.ctx);

                if result != 0 {
                    // Log cleanup failure
                    #[cfg(debug_assertions)]
                    eprintln!("ERROR: bitnet_cpp_free_context failed");
                }
            }
            self.ctx = std::ptr::null_mut();
        }
    }
}
```

**Benefits**:
- Automatic cleanup on panic
- Automatic cleanup on early return
- Automatic cleanup on scope exit

### 6.3 Valgrind Integration

**Test Execution**:
```bash
# Run memory leak detection
valgrind --leak-check=full --show-leak-kinds=all \
  cargo test -p crossval --features ffi test_cleanup_on_error_paths

# Expected output (no leaks):
# ==12345== HEAP SUMMARY:
# ==12345==     in use at exit: 0 bytes in 0 blocks
# ==12345==   total heap usage: ... allocs, ... frees, ... bytes allocated
# ==12345==
# ==12345== All heap blocks were freed -- no leaks are possible
```

**CI Integration**:
```yaml
# .github/workflows/ci.yml

- name: Memory leak detection (Valgrind)
  if: runner.os == 'Linux'
  run: |
    sudo apt-get install -y valgrind
    valgrind --leak-check=full --error-exitcode=1 \
      cargo test -p crossval --features ffi --test cleanup_validation
```

---

## 7. Implementation Strategy

### 7.1 Phased Rollout

**Phase 1: Priority 1 (Week 1, 4-6 hours)**
1. Implement timeout mechanism (`with_timeout` wrapper)
2. Implement cleanup validation (context counter, Drop logging)
3. Implement C++ error logging (log macros, stderr output)
4. Test timeout mechanism (2 tests)
5. Test cleanup validation (2 tests)
6. Update documentation

**Phase 2: Priority 2 (Week 2, 5-7 hours)**
1. Expand `CrossvalError` enum (8+ new variants)
2. Implement pass-phase distinction (query vs fill)
3. Implement `ErrorContext` struct
4. Update error construction sites (10+ locations)
5. Test new error variants (8 tests)
6. Update error handling documentation

**Phase 3: Priority 3 (Week 3-4, 13-17 hours)**
1. Implement diagnostic CLI flag (`--diagnostics`)
2. Implement library discovery helpers
3. Implement symbol resolution helpers
4. Implement 52 ignored error tests (12 categories)
5. Run memory leak detection with valgrind
6. Update troubleshooting guide

### 7.2 C++ Error Reporting Strategy

**Step 1: Add logging to cpp_bridge.cpp**
- Implement log macros (DEBUG, INFO, WARN, ERROR)
- Add `set_last_error_with_logging` wrapper
- Update all error sites to use logging

**Step 2: Update error code conventions**
- Map C++ error codes to Rust error variants
- Return specific error codes (-1, -2, -3, ...) instead of generic -1
- Document error code meanings in header

**Step 3: Add error context in FFI layer**
- Capture C++ error codes and messages
- Enrich with Rust-side context (operation, phase, parameters)
- Chain error contexts using `anyhow`

### 7.3 FFI Error Propagation Strategy

**Step 1: Expand error enum**
- Add 8+ missing variants to `CrossvalError`
- Update `From` implementations
- Update error construction sites

**Step 2: Improve error messages**
- Add pass-phase distinction (query vs fill)
- Add debugging hints (likely root causes)
- Add actionable steps (how to fix)

**Step 3: Add error context**
- Implement `ErrorContext` struct
- Implement `WithContext` extension trait
- Update error construction to include context

### 7.4 User-Facing Error Strategy

**Step 1: Structured error messages**
- What happened (error description)
- Why it happened (root cause)
- How to fix it (actionable steps)
- Where to look (relevant logs, files, env vars)

**Step 2: Error message templates**
- Create template for each error type
- Include examples in documentation
- Test error message clarity

**Step 3: Error message validation**
- Add tests for error message content
- Validate actionable steps are present
- Validate error messages are clear

---

## 8. Testing Requirements

### 8.1 Error Path Tests (52 Tests Total)

**From `/tmp/error_handling_analysis.md` lines 1009-1028**

#### Category 1: Library Availability (3 tests)

**Test 1.1: CppNotAvailable Error**
```rust
#[test]
fn test_error_cpp_not_available() {
    // Simulate FFI not compiled
    // Expected: CrossvalError::CppNotAvailable
    todo!("Implement test");
}
```

**Test 1.2: Actionable Error Message**
```rust
#[test]
fn test_error_cpp_not_available_actionable_message() {
    // Validate error message contains:
    // - BITNET_CPP_DIR mention
    // - --features ffi mention
    // - setup-cpp-auto command
    todo!("Implement test");
}
```

**Test 1.3: LibraryNotFound Error**
```rust
#[test]
fn test_error_library_not_found() {
    // Simulate libbitnet.so missing at runtime
    // Expected: CrossvalError::LibraryNotFound
    todo!("Implement test");
}
```

#### Category 2: Symbol Resolution (3 tests)

**Test 2.1: SymbolNotFound Error (Required)**
```rust
#[test]
fn test_error_symbol_not_found_required() {
    // Simulate missing bitnet_cpp_init_context symbol
    // Expected: CrossvalError::SymbolNotFound
    todo!("Implement test");
}
```

**Test 2.2: OptionalSymbolMissing Warning**
```rust
#[test]
fn test_error_optional_symbol_missing_fallback() {
    // Simulate missing bitnet_cpp_tokenize_with_context symbol
    // Expected: Warning logged, fallback to llama.cpp
    todo!("Implement test");
}
```

**Test 2.3: Symbol Resolution Diagnostics**
```rust
#[test]
fn test_error_symbol_resolution_diagnostics() {
    // Run with --diagnostics flag
    // Validate output shows symbol availability
    todo!("Implement test");
}
```

#### Category 3: Model Loading (3 tests)

**Test 3.1: Invalid Model Path**
```rust
#[test]
fn test_error_invalid_model_path() {
    let result = BitnetSession::create(
        Path::new("/nonexistent/model.gguf"),
        512,
        0,
    );

    assert!(matches!(result, Err(CrossvalError::ModelLoadError(_))));

    // Validate error message is actionable
    if let Err(CrossvalError::ModelLoadError(msg)) = result {
        assert!(msg.contains("nonexistent.gguf"));
        assert!(msg.contains("not found") || msg.contains("No such file"));
    }
}
```

**Test 3.2: Corrupted Model File**
```rust
#[test]
fn test_error_corrupted_model_file() {
    // Create corrupted GGUF file
    let temp_path = create_corrupted_gguf();

    let result = BitnetSession::create(&temp_path, 512, 0);

    assert!(matches!(result, Err(CrossvalError::ModelLoadError(_))));
}
```

**Test 3.3: Model Loading Timeout**
```rust
#[test]
fn test_error_model_loading_timeout() {
    // Simulate very large model (slow loading)
    let result = BitnetSession::create_with_timeout(
        Path::new("models/large-model.gguf"),
        512,
        0,
        Duration::from_millis(100),
    );

    assert!(matches!(result, Err(CrossvalError::OperationTimeout(_))));
}
```

#### Category 4: Inference Operations (3 tests)

**Test 4.1: Null Pointer Validation**
```rust
#[test]
fn test_error_null_pointer_validation() {
    // Simulate null handle
    let model = unsafe {
        CppModel { handle: std::ptr::null_mut() }
    };

    let result = model.generate("test", 10);

    assert!(matches!(result, Err(CrossvalError::InferenceError(_))));

    // Validate error message mentions null handle
    if let Err(CrossvalError::InferenceError(msg)) = result {
        assert!(msg.contains("null"));
        assert!(msg.contains("load"));
    }
}
```

**Test 4.2: Invalid Token Count**
```rust
#[test]
fn test_error_invalid_token_count() {
    let session = BitnetSession::create(test_model_path(), 512, 0).unwrap();

    // Empty tokens
    let result = session.evaluate(&[]);
    assert!(matches!(result, Err(CrossvalError::InferenceError(_))));

    // Negative token count (shouldn't happen, but test C++ validation)
    // ... mock C++ to return negative count ...
}
```

**Test 4.3: Context Size Exceeded**
```rust
#[test]
fn test_error_context_size_exceeded() {
    let session = BitnetSession::create(test_model_path(), 512, 0).unwrap();

    // Generate 1024 tokens (exceeds 512 context)
    let tokens = vec![1i32; 1024];
    let result = session.evaluate(&tokens);

    assert!(matches!(result, Err(CrossvalError::ContextOverflow(_))));
}
```

#### Category 5: Buffer Negotiation (2 tests)

**Test 5.1: Pass 1 vs Pass 2 Error Distinction**
```rust
#[test]
fn test_error_pass_phase_distinction() {
    // Mock C++ to fail in Pass 1 (query)
    let result = tokenize_bitnet_mock_fail_pass1();
    if let Err(CrossvalError::InferenceError(msg)) = result {
        assert!(msg.contains("QUERY PHASE"));
    }

    // Mock C++ to fail in Pass 2 (fill)
    let result = tokenize_bitnet_mock_fail_pass2();
    if let Err(CrossvalError::InferenceError(msg)) = result {
        assert!(msg.contains("FILL PHASE"));
    }
}
```

**Test 5.2: Buffer Overflow Protection**
```rust
#[test]
fn test_error_buffer_overflow_protection() {
    let mut small_buf = vec![0u8; 10];

    // Attempt to write more than buffer size
    let result = unsafe {
        crossval_bitnet_eval_with_tokens(
            model_ptr,
            tokens.as_ptr(),
            tokens.len() as i32,
            512,
            small_buf.as_mut_ptr() as *mut f32,
            10,  // Buffer too small
            &mut rows,
            &mut cols,
            err_buf.as_mut_ptr() as *mut c_char,
            err_buf.len() as i32,
        )
    };

    // Should error or truncate safely
    assert!(result != 0 || rows * cols <= 10);
}
```

#### Category 6: Cleanup on Error (3 tests)

**Test 6.1: Cleanup on Session Creation Failure**
```rust
#[test]
#[cfg(debug_assertions)]
fn test_error_cleanup_on_session_creation_failure() {
    let initial_count = active_context_count();

    // Attempt to create session with invalid model
    let _ = BitnetSession::create(
        Path::new("/nonexistent/model.gguf"),
        512,
        0,
    );

    // Validate no leaked contexts
    assert_eq!(active_context_count(), initial_count);
}
```

**Test 6.2: Cleanup on Tokenization Failure**
```rust
#[test]
#[cfg(debug_assertions)]
fn test_error_cleanup_on_tokenization_failure() {
    let session = BitnetSession::create(test_model_path(), 512, 0).unwrap();

    // Attempt tokenization with invalid input
    let _ = session.tokenize("");  // Empty prompt

    // Session should still be usable
    let result = session.tokenize("test");
    assert!(result.is_ok());
}
```

**Test 6.3: No Memory Leaks on Repeated Errors**
```rust
#[test]
fn test_error_no_memory_leaks_on_repeated_errors() {
    for _ in 0..100 {
        // Try to create session with invalid model
        let _ = BitnetSession::create(
            Path::new("/nonexistent/model.gguf"),
            512,
            0,
        );
    }

    // Run with valgrind:
    // valgrind --leak-check=full cargo test test_error_no_memory_leaks_on_repeated_errors
}
```

#### Category 7: Error Message Quality (2 tests)

**Test 7.1: Actionable Error Messages**
```rust
#[test]
fn test_error_messages_contain_actionable_steps() {
    let result = BitnetSession::create(
        Path::new("nonexistent.gguf"),
        512,
        0,
    );

    if let Err(CrossvalError::ModelLoadError(msg)) = result {
        // Should tell user what to do
        assert!(
            msg.contains("nonexistent.gguf") ||      // Show the problem
            msg.contains("not found") ||              // Explain what happened
            msg.contains("compat-check"),             // Suggest next step
            "Error message should be actionable: {}",
            msg
        );
    }
}
```

**Test 7.2: Error Message Clarity**
```rust
#[test]
fn test_error_message_clarity() {
    // Collect all error variants
    let errors = vec![
        CrossvalError::CppNotAvailable,
        CrossvalError::ModelLoadError("test".into()),
        CrossvalError::InferenceError("test".into()),
        CrossvalError::LibraryNotFound("libbitnet.so".into()),
        CrossvalError::SymbolNotFound("bitnet_cpp_init_context".into()),
        // ... all variants ...
    ];

    for error in errors {
        let msg = format!("{}", error);

        // Validate error message:
        // 1. Not empty
        assert!(!msg.is_empty());

        // 2. Contains error-specific details
        assert!(msg.len() > 50, "Error message too short: {}", msg);

        // 3. Contains actionable guidance (one of these keywords)
        let has_guidance = msg.contains("Set") ||
                           msg.contains("Try") ||
                           msg.contains("Run") ||
                           msg.contains("Check") ||
                           msg.contains("Increase") ||
                           msg.contains("Reduce");
        assert!(has_guidance, "Error message lacks actionable guidance: {}", msg);
    }
}
```

#### Category 8-12: Fallback Tests (22 tests)

**Complete test definitions in `crossval/tests/ffi_fallback_tests.rs`**

- Tokenization fallback (6 tests)
- Inference fallback (6 tests)
- Symbol resolution fallback (3 tests)
- Fallback performance (2 tests)
- Fallback consistency (3 tests)
- Fallback diagnostics (2 tests)

### 8.2 Test Execution Strategy

**Test Grouping**:
1. **Unit tests**: Error construction, message formatting, context building
2. **Integration tests**: FFI error paths, cleanup validation, timeout mechanism
3. **End-to-end tests**: Cross-validation with error injection, diagnostic CLI

**Test Environment**:
- Mock C++ failures using test-only FFI functions
- Use feature flags to enable/disable error injection
- Run memory leak tests with valgrind in CI

**CI Integration**:
```yaml
# .github/workflows/ci.yml

- name: Error handling tests
  run: |
    cargo test -p crossval --features ffi --test ffi_error_tests
    cargo test -p crossval --features ffi --test ffi_fallback_tests

- name: Memory leak detection
  if: runner.os == 'Linux'
  run: |
    sudo apt-get install -y valgrind
    valgrind --leak-check=full --error-exitcode=1 \
      cargo test -p crossval --features ffi --test cleanup_validation
```

---

## 9. Acceptance Criteria

### 9.1 Priority 1 Acceptance Criteria (Must Have)

**AC1: Timeout Mechanism Works**
- [ ] Timeout wrapper implemented (`with_timeout` function)
- [ ] Timeout applies to tokenize, evaluate, model loading
- [ ] Timeout configurable via `--timeout` CLI flag
- [ ] Timeout tests pass (2 tests: timeout triggers, normal completion)
- [ ] Thread leak on timeout is documented and acceptable

**AC2: Cleanup Validation Works**
- [ ] Context counter implemented (debug builds only)
- [ ] Cleanup failure logged in debug builds
- [ ] Drop validation works (context count decrements on drop)
- [ ] Cleanup tests pass (3 tests: session creation, tokenization, repeated errors)
- [ ] Valgrind detects no memory leaks

**AC3: C++ Error Logging Works**
- [ ] Log macros implemented (DEBUG, INFO, WARN, ERROR)
- [ ] `set_last_error_with_logging` replaces `set_last_error`
- [ ] Logging configurable via `BITNET_CPP_LOG_LEVEL`
- [ ] Logs output to stderr (unbuffered)
- [ ] Logging tests pass (1 test: error logged to stderr)

### 9.2 Priority 2 Acceptance Criteria (Should Have)

**AC4: Error Enum Expanded**
- [ ] 8+ new error variants added to `CrossvalError`
- [ ] All new variants have actionable error messages
- [ ] Error construction sites updated (10+ locations)
- [ ] Error variant tests pass (8 tests, one per new variant)
- [ ] Documentation updated with error handling examples

**AC5: Pass-Phase Distinction Implemented**
- [ ] Query vs fill errors distinguishable in error messages
- [ ] Error messages include context (model path, prompt length, token count)
- [ ] Debugging hints included (likely root causes)
- [ ] Pass-phase tests pass (1 test: Pass 1 vs Pass 2 distinction)

**AC6: Error Context Implemented**
- [ ] `ErrorContext` struct implemented
- [ ] `WithContext` extension trait implemented
- [ ] Error construction sites use `with_context`
- [ ] Error context tests pass (2 tests: context building, context chaining)

### 9.3 Priority 3 Acceptance Criteria (Nice to Have)

**AC7: Diagnostics Flag Implemented**
- [ ] `--diagnostics` CLI flag implemented
- [ ] Environment variables displayed
- [ ] Compile-time detection displayed
- [ ] Library discovery implemented
- [ ] Symbol resolution checking implemented
- [ ] Diagnostics tests pass (2 tests: standard output, detailed output)

**AC8: Error Tests Implemented**
- [ ] 52 ignored tests implemented (or reasonably scoped subset)
- [ ] All error categories covered (12 categories)
- [ ] Memory leak tests pass with valgrind
- [ ] CI runs error tests on every PR
- [ ] Error test coverage ≥ 80% (measured with cargo-tarpaulin)

**AC9: Documentation Updated**
- [ ] Error handling guide created (`docs/howto/error-handling.md`)
- [ ] Troubleshooting guide updated with new error types
- [ ] API documentation includes error examples
- [ ] CLAUDE.md updated with error handling section

### 9.4 Success Metrics

**Quantitative**:
- **Test Coverage**: ≥ 80% error path coverage (measured with cargo-tarpaulin)
- **Error Tests**: 52 tests passing (or reasonably scoped subset)
- **Memory Leaks**: 0 memory leaks detected by valgrind
- **Timeout Coverage**: 100% of C++ FFI calls protected by timeout

**Qualitative**:
- **Error Messages**: All error messages are actionable (contain "how to fix")
- **Debugging**: Error logs provide sufficient context for debugging
- **Reliability**: No hangs, no crashes, no resource leaks
- **User Experience**: Users can diagnose and fix issues without deep FFI knowledge

### 9.5 Validation Checklist

**Before Merging PR**:
- [ ] All Priority 1 acceptance criteria met
- [ ] Priority 2 acceptance criteria met (or documented as future work)
- [ ] All tests passing (unit, integration, end-to-end)
- [ ] Memory leak tests passing (valgrind)
- [ ] Documentation updated (error handling guide, troubleshooting)
- [ ] CI passing (all platforms: Linux, macOS, Windows)
- [ ] Code review completed (2 approvals)
- [ ] Backward compatibility maintained (no breaking changes)

**Post-Merge Validation**:
- [ ] Monitoring shows no new error patterns
- [ ] User reports validate error messages are actionable
- [ ] Performance impact acceptable (< 5% overhead from logging/timeout)
- [ ] Integration with existing error handling seamless

---

## 10. References

### 10.1 Related Documentation

- **Analysis Report**: `/tmp/error_handling_analysis.md` (source of 9 critical gaps)
- **FFI Sockets Spec**: `docs/specs/bitnet-cpp-ffi-sockets.md` (FFI architecture)
- **Build Detection Spec**: `docs/specs/bitnet-buildrs-detection-enhancement.md` (library detection)
- **CLAUDE.md**: Project guidelines and error handling conventions

### 10.2 Test Files

- **Error Path Tests**: `crossval/tests/ffi_error_tests.rs` (40+ ignored tests)
- **Fallback Tests**: `crossval/tests/ffi_fallback_tests.rs` (12+ ignored tests)
- **Socket Tests**: `crossval/tests/ffi_socket_tests.rs` (integration tests)

### 10.3 Implementation Files

- **C++ Bridge**: `crates/bitnet-kernels/src/ffi/cpp_bridge.cpp` (exception handling)
- **FFI Bindings**: `crossval/src/cpp_bindings.rs` (safe Rust wrappers)
- **Error Types**: `crossval/src/lib.rs` (CrossvalError enum)
- **Validation**: `crossval/src/validation.rs` (error in Result struct pattern)

### 10.4 External Dependencies

- **thiserror**: Error derive macro (existing dependency)
- **anyhow**: Flexible error context (existing dependency)
- **log**: Logging facade (new dependency for Rust logging)
- **env_logger**: Log configuration (new dependency, dev-only)
- **valgrind**: Memory leak detection (CI tool, not Rust dependency)

---

## Appendix A: Error Message Examples

### Good Error Example 1: LibraryNotFound

```
Error: C++ library not found: libbitnet.so

Root Cause: Dynamic loader could not find BitNet.cpp libraries.

How to Fix:
1. Set BITNET_CPP_DIR environment variable:
   export BITNET_CPP_DIR=/path/to/bitnet.cpp

2. Or run auto-setup (one command):
   eval "$(cargo run -p xtask -- setup-cpp-auto --emit=sh)"

3. Or manually set library path:
   export LD_LIBRARY_PATH=$BITNET_CPP_DIR/build/bin:$LD_LIBRARY_PATH

Debugging:
- Check library exists: ls /path/to/bitnet.cpp/build/bin/libbitnet.so
- Check LD_LIBRARY_PATH: echo $LD_LIBRARY_PATH
- Run diagnostics: cargo run -p xtask -- preflight --diagnostics
```

### Good Error Example 2: OperationTimeout

```
Error: Operation timed out after 30s

Root Cause: C++ evaluation exceeded timeout limit.

How to Fix:
1. Increase timeout:
   cargo run -p xtask -- crossval-per-token --timeout 60 ...

2. Reduce input size:
   cargo run -p xtask -- crossval-per-token --max-tokens 32 ...

3. Check model for issues:
   cargo run -p bitnet-cli -- inspect model.gguf

Debugging:
- Enable C++ logging: export BITNET_CPP_LOG_LEVEL=DEBUG
- Check model is not corrupted: cargo run -p bitnet-cli -- compat-check model.gguf
- If model hangs consistently, report issue at: https://github.com/codekansas/bitnet-rs/issues
```

### Good Error Example 3: SymbolNotFound

```
Error: Required C++ symbol not found: bitnet_cpp_init_context

Root Cause: Mismatch between Rust bindings and C++ library version.

Expected C++ API version: 0.2.x
Found C++ library: /usr/local/lib/libbitnet.so

How to Fix:
1. Rebuild BitNet.cpp with correct version:
   cd $BITNET_CPP_DIR && git checkout v0.2.0 && ./build.sh

2. Or downgrade Rust bindings to match C++ version:
   git checkout <compatible-commit>

3. Or check symbol manually:
   nm -D /usr/local/lib/libbitnet.so | grep bitnet_cpp_init_context

Debugging:
- Check C++ version: cat $BITNET_CPP_DIR/VERSION
- Check Rust bindings version: grep "version" crossval/Cargo.toml
- Run symbol diagnostics: cargo run -p xtask -- preflight --backend bitnet --diagnostics
```

---

## Appendix B: Implementation Timeline

### Week 1: Priority 1 (4-6 hours)

**Day 1-2: Timeout Mechanism**
- [ ] Implement `with_timeout` wrapper (2 hours)
- [ ] Add timeout to tokenize, evaluate, load (1 hour)
- [ ] Add `--timeout` CLI flag (30 minutes)
- [ ] Write timeout tests (1 hour)

**Day 3-4: Cleanup Validation**
- [ ] Implement context counter (1 hour)
- [ ] Update Drop implementation with logging (1 hour)
- [ ] Write cleanup tests (1 hour)
- [ ] Run valgrind tests (1 hour)

**Day 5: C++ Error Logging**
- [ ] Implement log macros (1 hour)
- [ ] Update error sites (1 hour)
- [ ] Write logging tests (30 minutes)
- [ ] Update documentation (30 minutes)

### Week 2: Priority 2 (5-7 hours)

**Day 1-2: Error Enum Expansion**
- [ ] Add 8+ new variants (1 hour)
- [ ] Update error construction sites (2 hours)
- [ ] Write error variant tests (2 hours)

**Day 3: Pass-Phase Distinction**
- [ ] Update tokenize_bitnet (1 hour)
- [ ] Update eval_with_tokens (1 hour)
- [ ] Write pass-phase tests (1 hour)

**Day 4-5: Error Context**
- [ ] Implement ErrorContext struct (1 hour)
- [ ] Implement WithContext trait (1 hour)
- [ ] Update error construction sites (1 hour)

### Week 3-4: Priority 3 (13-17 hours)

**Day 1-3: Diagnostics CLI**
- [ ] Implement library discovery (2 hours)
- [ ] Implement symbol resolution (2 hours)
- [ ] Implement `--diagnostics` flag (2 hours)
- [ ] Write diagnostics tests (2 hours)

**Day 4-10: Error Tests**
- [ ] Implement library availability tests (2 hours)
- [ ] Implement symbol resolution tests (2 hours)
- [ ] Implement model loading tests (2 hours)
- [ ] Implement inference tests (2 hours)
- [ ] Implement cleanup tests (2 hours)
- [ ] Implement fallback tests (3 hours)

**Day 11-12: Documentation**
- [ ] Write error handling guide (2 hours)
- [ ] Update troubleshooting guide (1 hour)
- [ ] Update API documentation (1 hour)
- [ ] Update CLAUDE.md (1 hour)

---

**End of Specification**
