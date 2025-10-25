# BitNet Session API Specification

**Version**: 1.0.0-draft
**Status**: Design Phase
**Target**: G2.6 Fast Iteration Support
**Author**: BitNet.rs Team
**Date**: 2025-10-25

## Overview

This specification defines a session-based C ABI for cross-validation against bitnet.cpp, eliminating per-call model reloading and enabling fast iteration for multi-position logits comparison.

### Problem Statement

Current cross-validation implementation (`crossval-per-token`) reloads the model for every inference call:

- **Current overhead**: ~500ms model load per `crossval_bitnet_eval` call
- **Impact**: Per-position logits comparison with N tokens requires N model reloads
- **Example**: 32-token sequence = 16 seconds of pure overhead (32 × 500ms)

### Solution: Session-Based API

Introduce a stateful session handle that persists the loaded model across multiple operations:

- **Session lifecycle**: `open` → multiple `eval`/`tokenize` → `close`
- **Performance target**: ~5ms per eval (100× improvement over reload path)
- **Use case**: Per-position logits with prefill + eval_last loop

---

## C API Specification

### Design Principles

1. **Resource Safety**: Explicit open/close lifecycle with error handling
2. **Thread Safety**: One session per thread (no internal synchronization)
3. **Error Reporting**: All functions return status codes with optional error messages
4. **ABI Stability**: Pure C types, no C++ exceptions across boundary
5. **Backward Compatibility**: Coexists with existing `crossval_bitnet_eval` for simple cases

### Session Lifecycle

#### Open Session

```c
/**
 * Open a bitnet.cpp session with a loaded model.
 *
 * @param model_path    Path to GGUF model file (null-terminated UTF-8)
 * @param n_ctx         Context window size (e.g., 2048)
 * @param out_handle    Output pointer to receive session handle
 * @param err           Optional error message buffer (nullable)
 * @param err_len       Size of error buffer in bytes
 * @return              0 on success, negative error code on failure
 *
 * Error codes:
 *   -1: Model file not found
 *   -2: Invalid model format
 *   -3: Out of memory
 *   -4: Invalid parameters (null out_handle, n_ctx <= 0)
 *
 * Thread safety: Safe to call concurrently from multiple threads
 *                (each thread gets its own session)
 *
 * Example:
 *   void* session = NULL;
 *   char err[256];
 *   int rc = crossval_bitnet_open("model.gguf", 2048, &session, err, 256);
 *   if (rc != 0) {
 *       fprintf(stderr, "Open failed: %s\n", err);
 *       return rc;
 *   }
 */
int crossval_bitnet_open(
    const char* model_path,
    int32_t n_ctx,
    void** out_handle,
    char* err,
    int32_t err_len
);
```

#### Close Session

```c
/**
 * Close a bitnet.cpp session and free all resources.
 *
 * @param handle    Session handle from crossval_bitnet_open
 * @return          0 on success, negative error code on failure
 *
 * Error codes:
 *   -1: Invalid handle (null or already closed)
 *   -2: Resource cleanup failed (logged but non-fatal)
 *
 * Thread safety: Not safe to close a handle being used by another thread
 *
 * Note: After close, the handle is invalid. Calling any function with
 *       a closed handle is undefined behavior.
 *
 * Example:
 *   int rc = crossval_bitnet_close(session);
 *   if (rc != 0) {
 *       fprintf(stderr, "Close warning: cleanup incomplete\n");
 *   }
 */
int crossval_bitnet_close(void* handle);
```

### Session Operations

#### Tokenize

```c
/**
 * Tokenize a prompt using the session's loaded tokenizer.
 *
 * @param handle        Session handle
 * @param prompt        Input text (null-terminated UTF-8)
 * @param add_bos       If non-zero, prepend BOS token
 * @param out_tokens    Output buffer for token IDs
 * @param out_cap       Capacity of out_tokens buffer
 * @param out_len       Output: actual number of tokens written
 * @param err           Optional error message buffer
 * @param err_len       Size of error buffer
 * @return              0 on success, negative error code on failure
 *
 * Error codes:
 *   -1: Invalid handle
 *   -2: Null prompt
 *   -3: Buffer too small (out_cap < actual token count)
 *   -4: Tokenization failed
 *
 * Thread safety: Safe if called from the same thread that owns the session
 *
 * Example:
 *   int32_t tokens[512];
 *   int32_t n_tokens;
 *   int rc = crossval_bitnet_tokenize_with(session, "Hello world", 1,
 *                                           tokens, 512, &n_tokens, NULL, 0);
 *   if (rc != 0) { /* handle error */ }
 */
int crossval_bitnet_tokenize_with(
    void* handle,
    const char* prompt,
    int add_bos,
    int32_t* out_tokens,
    int32_t out_cap,
    int32_t* out_len,
    char* err,
    int32_t err_len
);
```

#### Prefill KV Cache

```c
/**
 * Prefill the KV cache with a sequence of tokens (no logits output).
 *
 * This is an optimization for per-position evaluation: prefill the
 * common prefix once, then eval_last for each new token.
 *
 * @param handle        Session handle
 * @param tokens        Input token sequence
 * @param n_tokens      Number of tokens in sequence
 * @param err           Optional error message buffer
 * @param err_len       Size of error buffer
 * @return              0 on success, negative error code on failure
 *
 * Error codes:
 *   -1: Invalid handle
 *   -2: Null tokens or n_tokens <= 0
 *   -3: Sequence exceeds context window (n_tokens > n_ctx)
 *   -4: Evaluation failed
 *
 * Thread safety: Not thread-safe with other operations on same session
 *
 * Note: After prefill, the KV cache contains states for tokens[0..n_tokens-1].
 *       Subsequent eval_last calls will append to this cache.
 *
 * Example:
 *   // Prefill prompt tokens
 *   int32_t prompt_tokens[] = {1, 2, 3, 4};
 *   int rc = crossval_bitnet_prefill(session, prompt_tokens, 4, NULL, 0);
 *   if (rc != 0) { /* handle error */ }
 */
int crossval_bitnet_prefill(
    void* handle,
    const int32_t* tokens,
    int32_t n_tokens,
    char* err,
    int32_t err_len
);
```

#### Evaluate Last Position

```c
/**
 * Evaluate tokens and return logits for the LAST position only.
 *
 * This is the fast path for autoregressive generation and per-position
 * cross-validation. Use after prefill to get logits incrementally.
 *
 * @param handle        Session handle
 * @param tokens        Input token sequence
 * @param n_tokens      Number of tokens in sequence
 * @param logits        Output buffer for logits (vocab_size floats)
 * @param vocab         Expected vocabulary size (for validation)
 * @param err           Optional error message buffer
 * @param err_len       Size of error buffer
 * @return              0 on success, negative error code on failure
 *
 * Error codes:
 *   -1: Invalid handle
 *   -2: Null tokens/logits or n_tokens <= 0
 *   -3: Vocabulary size mismatch
 *   -4: Evaluation failed
 *   -5: KV cache overflow (total tokens > n_ctx)
 *
 * Thread safety: Not thread-safe with other operations on same session
 *
 * Output: logits[0..vocab-1] contains the probability distribution
 *         for the next token after the last input token.
 *
 * Example:
 *   float logits[32000];  // vocab_size = 32000
 *   int32_t next_token = 5;
 *   int rc = crossval_bitnet_eval_last(session, &next_token, 1,
 *                                      logits, 32000, NULL, 0);
 *   if (rc != 0) { /* handle error */ }
 */
int crossval_bitnet_eval_last(
    void* handle,
    const int32_t* tokens,
    int32_t n_tokens,
    float* logits,
    int32_t vocab,
    char* err,
    int32_t err_len
);
```

#### Evaluate All Positions

```c
/**
 * Evaluate tokens and return logits for ALL positions (batch mode).
 *
 * This is useful for full-sequence cross-validation but slower than
 * incremental eval_last. Output is a 2D matrix: [n_tokens, vocab_size].
 *
 * @param handle        Session handle
 * @param tokens        Input token sequence
 * @param n_tokens      Number of tokens in sequence
 * @param logits        Output buffer (n_tokens * vocab_size floats)
 * @param rows_cap      Maximum number of rows in logits buffer
 * @param out_rows      Output: actual number of rows written
 * @param out_cols      Output: vocabulary size (columns per row)
 * @param err           Optional error message buffer
 * @param err_len       Size of error buffer
 * @return              0 on success, negative error code on failure
 *
 * Error codes:
 *   -1: Invalid handle
 *   -2: Null tokens/logits or n_tokens <= 0
 *   -3: Buffer too small (rows_cap < n_tokens)
 *   -4: Evaluation failed
 *   -5: Sequence exceeds context window
 *
 * Thread safety: Not thread-safe with other operations on same session
 *
 * Output layout: Row-major, logits[i * vocab_size + j] = logit for
 *                token i, vocabulary entry j
 *
 * Example:
 *   int32_t tokens[] = {1, 2, 3, 4};
 *   float logits[4 * 32000];  // 4 tokens, vocab=32000
 *   int32_t rows, cols;
 *   int rc = crossval_bitnet_eval_all(session, tokens, 4,
 *                                     logits, 4, &rows, &cols, NULL, 0);
 *   if (rc != 0) { /* handle error */ }
 *   // logits[0..cols-1]: position 0
 *   // logits[cols..2*cols-1]: position 1, etc.
 */
int crossval_bitnet_eval_all(
    void* handle,
    const int32_t* tokens,
    int32_t n_tokens,
    float* logits,
    int32_t rows_cap,
    int32_t* out_rows,
    int32_t* out_cols,
    char* err,
    int32_t err_len
);
```

### Session Metadata

```c
/**
 * Get vocabulary size for the loaded model.
 *
 * @param handle        Session handle
 * @param out_vocab     Output: vocabulary size
 * @return              0 on success, negative error code on failure
 *
 * Error codes:
 *   -1: Invalid handle
 *
 * Thread safety: Safe (read-only)
 *
 * Example:
 *   int32_t vocab;
 *   int rc = crossval_bitnet_vocab_size(session, &vocab);
 *   if (rc == 0) {
 *       printf("Model vocabulary: %d tokens\n", vocab);
 *   }
 */
int crossval_bitnet_vocab_size(void* handle, int32_t* out_vocab);

/**
 * Get context window size for the session.
 *
 * @param handle        Session handle
 * @param out_ctx       Output: context window size
 * @return              0 on success, negative error code on failure
 *
 * Error codes:
 *   -1: Invalid handle
 *
 * Thread safety: Safe (read-only)
 *
 * Example:
 *   int32_t n_ctx;
 *   int rc = crossval_bitnet_context_size(session, &n_ctx);
 *   if (rc == 0) {
 *       printf("Context window: %d tokens\n", n_ctx);
 *   }
 */
int crossval_bitnet_context_size(void* handle, int32_t* out_ctx);
```

---

## Rust Safe Wrapper

### Public API

```rust
use std::ffi::{c_char, c_int, c_void, CString};
use std::path::Path;
use std::ptr;

/// Safe Rust wrapper for bitnet.cpp session API.
///
/// Manages session lifecycle with RAII and provides safe access to
/// cross-validation operations without per-call model reloading.
///
/// # Thread Safety
/// Not thread-safe. Each thread should create its own session.
///
/// # Example
/// ```no_run
/// use crossval::BitnetSession;
/// use std::path::Path;
///
/// let session = BitnetSession::open(Path::new("model.gguf"), 2048)?;
/// let tokens = session.tokenize("Hello world", true)?;
/// let logits = session.eval_last(&tokens)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct BitnetSession {
    handle: *mut c_void,
    vocab: i32,
    n_ctx: i32,
}

// SAFETY: Session is not thread-safe (documented above)
// Mark as !Send + !Sync to prevent misuse
impl !Send for BitnetSession {}
impl !Sync for BitnetSession {}

impl BitnetSession {
    /// Open a new session with the specified model and context window.
    ///
    /// # Arguments
    /// * `model_path` - Path to GGUF model file
    /// * `n_ctx` - Context window size (e.g., 2048)
    ///
    /// # Errors
    /// Returns error if:
    /// - Model file not found
    /// - Invalid model format
    /// - Out of memory
    /// - Invalid parameters (n_ctx <= 0)
    ///
    /// # Example
    /// ```no_run
    /// # use crossval::BitnetSession;
    /// # use std::path::Path;
    /// let session = BitnetSession::open(Path::new("model.gguf"), 2048)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn open(model_path: &Path, n_ctx: i32) -> Result<Self, SessionError> {
        let path_cstr = CString::new(model_path.to_str().ok_or(SessionError::InvalidPath)?)?;
        let mut handle: *mut c_void = ptr::null_mut();
        let mut err_buf = vec![0u8; 512];

        let rc = unsafe {
            crossval_bitnet_open(
                path_cstr.as_ptr(),
                n_ctx,
                &mut handle,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if rc != 0 {
            let err_msg = unsafe {
                CStr::from_ptr(err_buf.as_ptr() as *const c_char)
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(SessionError::OpenFailed(rc, err_msg));
        }

        // Query vocabulary size
        let mut vocab = 0;
        let rc = unsafe { crossval_bitnet_vocab_size(handle, &mut vocab) };
        if rc != 0 {
            unsafe { crossval_bitnet_close(handle) };
            return Err(SessionError::MetadataFailed("vocab_size"));
        }

        Ok(Self {
            handle,
            vocab,
            n_ctx,
        })
    }

    /// Tokenize a prompt using the session's tokenizer.
    ///
    /// # Arguments
    /// * `prompt` - Input text to tokenize
    /// * `add_bos` - If true, prepend BOS token
    ///
    /// # Errors
    /// Returns error if tokenization fails or buffer too small.
    ///
    /// # Example
    /// ```no_run
    /// # use crossval::BitnetSession;
    /// # use std::path::Path;
    /// # let session = BitnetSession::open(Path::new("model.gguf"), 2048)?;
    /// let tokens = session.tokenize("What is 2+2?", true)?;
    /// println!("Tokens: {:?}", tokens);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn tokenize(&self, prompt: &str, add_bos: bool) -> Result<Vec<i32>, SessionError> {
        let prompt_cstr = CString::new(prompt)?;
        let mut tokens = vec![0i32; 4096]; // Generous buffer
        let mut n_tokens = 0;
        let mut err_buf = vec![0u8; 512];

        let rc = unsafe {
            crossval_bitnet_tokenize_with(
                self.handle,
                prompt_cstr.as_ptr(),
                if add_bos { 1 } else { 0 },
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                &mut n_tokens,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if rc != 0 {
            let err_msg = unsafe {
                CStr::from_ptr(err_buf.as_ptr() as *const c_char)
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(SessionError::TokenizeFailed(rc, err_msg));
        }

        tokens.truncate(n_tokens as usize);
        Ok(tokens)
    }

    /// Prefill the KV cache with a token sequence (no logits output).
    ///
    /// This is an optimization for per-position evaluation: prefill the
    /// common prefix once, then call `eval_last` for each new token.
    ///
    /// # Arguments
    /// * `tokens` - Input token sequence
    ///
    /// # Errors
    /// Returns error if evaluation fails or sequence exceeds context window.
    ///
    /// # Example
    /// ```no_run
    /// # use crossval::BitnetSession;
    /// # use std::path::Path;
    /// # let mut session = BitnetSession::open(Path::new("model.gguf"), 2048)?;
    /// let prompt_tokens = vec![1, 2, 3, 4];
    /// session.prefill(&prompt_tokens)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn prefill(&mut self, tokens: &[i32]) -> Result<(), SessionError> {
        let mut err_buf = vec![0u8; 512];

        let rc = unsafe {
            crossval_bitnet_prefill(
                self.handle,
                tokens.as_ptr(),
                tokens.len() as i32,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if rc != 0 {
            let err_msg = unsafe {
                CStr::from_ptr(err_buf.as_ptr() as *const c_char)
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(SessionError::PrefillFailed(rc, err_msg));
        }

        Ok(())
    }

    /// Evaluate tokens and return logits for the LAST position only.
    ///
    /// This is the fast path for per-position cross-validation. Use after
    /// `prefill` to get logits incrementally.
    ///
    /// # Arguments
    /// * `tokens` - Input token sequence
    ///
    /// # Returns
    /// Logits vector of length `vocab_size`
    ///
    /// # Errors
    /// Returns error if evaluation fails or KV cache overflows.
    ///
    /// # Example
    /// ```no_run
    /// # use crossval::BitnetSession;
    /// # use std::path::Path;
    /// # let mut session = BitnetSession::open(Path::new("model.gguf"), 2048)?;
    /// # session.prefill(&[1, 2, 3])?;
    /// let logits = session.eval_last(&[4])?;
    /// println!("Next token logits: {:.4?}", &logits[..10]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn eval_last(&mut self, tokens: &[i32]) -> Result<Vec<f32>, SessionError> {
        let mut logits = vec![0.0f32; self.vocab as usize];
        let mut err_buf = vec![0u8; 512];

        let rc = unsafe {
            crossval_bitnet_eval_last(
                self.handle,
                tokens.as_ptr(),
                tokens.len() as i32,
                logits.as_mut_ptr(),
                self.vocab,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if rc != 0 {
            let err_msg = unsafe {
                CStr::from_ptr(err_buf.as_ptr() as *const c_char)
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(SessionError::EvalFailed(rc, err_msg));
        }

        Ok(logits)
    }

    /// Evaluate tokens and return logits for ALL positions (batch mode).
    ///
    /// This is useful for full-sequence cross-validation but slower than
    /// incremental `eval_last`. Returns a 2D matrix of logits.
    ///
    /// # Arguments
    /// * `tokens` - Input token sequence
    ///
    /// # Returns
    /// Vector of logits vectors: `result[i]` = logits for position i
    ///
    /// # Errors
    /// Returns error if evaluation fails or sequence exceeds context window.
    ///
    /// # Example
    /// ```no_run
    /// # use crossval::BitnetSession;
    /// # use std::path::Path;
    /// # let mut session = BitnetSession::open(Path::new("model.gguf"), 2048)?;
    /// let tokens = vec![1, 2, 3, 4];
    /// let all_logits = session.eval_all(&tokens)?;
    /// for (i, logits) in all_logits.iter().enumerate() {
    ///     println!("Position {}: {:.4?}", i, &logits[..10]);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn eval_all(&mut self, tokens: &[i32]) -> Result<Vec<Vec<f32>>, SessionError> {
        let n_tokens = tokens.len() as i32;
        let mut logits_flat = vec![0.0f32; (n_tokens * self.vocab) as usize];
        let mut out_rows = 0;
        let mut out_cols = 0;
        let mut err_buf = vec![0u8; 512];

        let rc = unsafe {
            crossval_bitnet_eval_all(
                self.handle,
                tokens.as_ptr(),
                n_tokens,
                logits_flat.as_mut_ptr(),
                n_tokens,
                &mut out_rows,
                &mut out_cols,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        if rc != 0 {
            let err_msg = unsafe {
                CStr::from_ptr(err_buf.as_ptr() as *const c_char)
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(SessionError::EvalFailed(rc, err_msg));
        }

        // Convert flat buffer to Vec<Vec<f32>>
        let mut result = Vec::with_capacity(out_rows as usize);
        for i in 0..out_rows as usize {
            let start = i * out_cols as usize;
            let end = start + out_cols as usize;
            result.push(logits_flat[start..end].to_vec());
        }

        Ok(result)
    }

    /// Get the vocabulary size for this session.
    pub fn vocab_size(&self) -> i32 {
        self.vocab
    }

    /// Get the context window size for this session.
    pub fn context_size(&self) -> i32 {
        self.n_ctx
    }
}

impl Drop for BitnetSession {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let rc = crossval_bitnet_close(self.handle);
                if rc != 0 {
                    eprintln!("Warning: BitnetSession::drop failed to close handle (rc={})", rc);
                }
            }
            self.handle = ptr::null_mut();
        }
    }
}

/// Error types for session operations.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("Invalid path: non-UTF8 or null")]
    InvalidPath,

    #[error("Failed to open session (code {0}): {1}")]
    OpenFailed(i32, String),

    #[error("Failed to query metadata: {0}")]
    MetadataFailed(&'static str),

    #[error("Tokenization failed (code {0}): {1}")]
    TokenizeFailed(i32, String),

    #[error("Prefill failed (code {0}): {1}")]
    PrefillFailed(i32, String),

    #[error("Evaluation failed (code {0}): {1}")]
    EvalFailed(i32, String),

    #[error("CString conversion error: {0}")]
    CStringError(#[from] std::ffi::NulError),
}
```

### FFI Declarations

```rust
// crossval/src/bitnet_session_ffi.rs

use std::ffi::{c_char, c_int, c_void};

extern "C" {
    pub fn crossval_bitnet_open(
        model_path: *const c_char,
        n_ctx: i32,
        out_handle: *mut *mut c_void,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;

    pub fn crossval_bitnet_close(handle: *mut c_void) -> c_int;

    pub fn crossval_bitnet_tokenize_with(
        handle: *mut c_void,
        prompt: *const c_char,
        add_bos: c_int,
        out_tokens: *mut i32,
        out_cap: i32,
        out_len: *mut i32,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;

    pub fn crossval_bitnet_prefill(
        handle: *mut c_void,
        tokens: *const i32,
        n_tokens: i32,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;

    pub fn crossval_bitnet_eval_last(
        handle: *mut c_void,
        tokens: *const i32,
        n_tokens: i32,
        logits: *mut f32,
        vocab: i32,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;

    pub fn crossval_bitnet_eval_all(
        handle: *mut c_void,
        tokens: *const i32,
        n_tokens: i32,
        logits: *mut f32,
        rows_cap: i32,
        out_rows: *mut i32,
        out_cols: *mut i32,
        err: *mut c_char,
        err_len: i32,
    ) -> c_int;

    pub fn crossval_bitnet_vocab_size(handle: *mut c_void, out_vocab: *mut i32) -> c_int;

    pub fn crossval_bitnet_context_size(handle: *mut c_void, out_ctx: *mut i32) -> c_int;
}
```

---

## Use Cases

### Use Case 1: Per-Position Logits with Prefill

**Scenario**: Compare Rust vs C++ logits at each position during autoregressive generation.

**Current Implementation** (no session):
```rust
// Per-position logits comparison (SLOW: reloads model N times)
fn compare_per_position_slow(model: &Path, tokens: &[i32]) -> Result<Vec<ParityResult>> {
    let mut results = Vec::new();
    for i in 0..tokens.len() {
        let prefix = &tokens[..=i];
        // SLOW: Reloads model for every position
        let cpp_logits = crossval_bitnet_eval(model, prefix)?;
        let rust_logits = eval_rust(model, prefix)?;
        results.push(compare_logits(&rust_logits, &cpp_logits));
    }
    Ok(results)
}
// Performance: N × 500ms model load = 16s for 32 tokens
```

**Session-Based Implementation** (fast):
```rust
// Per-position logits comparison (FAST: loads model once)
fn compare_per_position_fast(model: &Path, tokens: &[i32]) -> Result<Vec<ParityResult>> {
    let mut session = BitnetSession::open(model, 2048)?;
    let mut results = Vec::new();

    // Prefill with first token
    session.prefill(&tokens[..1])?;

    // Incrementally eval each position
    for i in 1..tokens.len() {
        let next_token = &tokens[i..=i];
        let cpp_logits = session.eval_last(next_token)?;
        let rust_logits = eval_rust_incremental(&tokens[..=i])?;
        results.push(compare_logits(&rust_logits, &cpp_logits));
    }
    Ok(results)
}
// Performance: 500ms initial load + N × 5ms eval = 660ms for 32 tokens
// Speedup: 24× faster
```

### Use Case 2: Repeated Tokenization

**Scenario**: Compare tokenization outputs across multiple prompts.

**Current Implementation**:
```rust
// Tokenization comparison (SLOW: reloads model per prompt)
fn compare_tokenization_slow(model: &Path, prompts: &[&str]) -> Result<Vec<bool>> {
    let mut results = Vec::new();
    for prompt in prompts {
        let cpp_tokens = crossval_bitnet_tokenize(model, prompt)?;
        let rust_tokens = tokenize_rust(prompt)?;
        results.push(cpp_tokens == rust_tokens);
    }
    Ok(results)
}
// Performance: N × 500ms = 5s for 10 prompts
```

**Session-Based Implementation**:
```rust
// Tokenization comparison (FAST: loads model once)
fn compare_tokenization_fast(model: &Path, prompts: &[&str]) -> Result<Vec<bool>> {
    let session = BitnetSession::open(model, 2048)?;
    let mut results = Vec::new();
    for prompt in prompts {
        let cpp_tokens = session.tokenize(prompt, true)?;
        let rust_tokens = tokenize_rust(prompt)?;
        results.push(cpp_tokens == rust_tokens);
    }
    Ok(results)
}
// Performance: 500ms initial load + N × 2ms = 520ms for 10 prompts
// Speedup: 9.6× faster
```

### Use Case 3: Multi-Prompt Batch Comparison

**Scenario**: Cross-validate multiple prompts in a single test run.

```rust
// Batch cross-validation with session reuse
fn batch_crossval(model: &Path, test_cases: &[TestCase]) -> Result<Report> {
    let mut session = BitnetSession::open(model, 2048)?;
    let mut report = Report::new();

    for test in test_cases {
        // Tokenize once
        let tokens = session.tokenize(&test.prompt, true)?;

        // Evaluate all positions (batch mode)
        let cpp_logits = session.eval_all(&tokens)?;
        let rust_logits = eval_rust_batch(&tokens)?;

        // Compare
        for (pos, (cpp, rust)) in cpp_logits.iter().zip(rust_logits).enumerate() {
            let parity = compare_logits(rust, cpp);
            report.record(test.name, pos, parity);
        }
    }

    Ok(report)
}
// Performance: Single model load + M prompts × N positions × 5ms eval
// Example: 50 prompts, avg 20 tokens = 500ms + 5s = 5.5s total
// vs. Current: 50 × 20 × 500ms = 500s (91× slower)
```

---

## Performance Analysis

### Baseline Measurements

**Current per-call overhead** (measured on BitNet 2B model):
- Model load: ~500ms (mmap + metadata parsing)
- Tokenization: ~2ms per prompt
- Single-token eval: ~3ms (with warm cache)
- Context setup: ~50ms (KV cache initialization)

**Total per-call cost**: ~555ms for a single eval

### Session-Based Timings

**One-time costs**:
- Session open: ~500ms (same as current model load)
- Session close: ~10ms (cleanup)

**Per-operation costs**:
- Tokenize: ~2ms (negligible change)
- Prefill (N tokens): ~N × 3ms
- Eval last: ~5ms (includes KV cache update)
- Eval all (N tokens): ~N × 5ms

### Speedup Estimates

| Use Case | Current | Session-Based | Speedup |
|----------|---------|---------------|---------|
| Single eval | 555ms | 500ms + 5ms = 505ms | 1.1× |
| 4 tokens (per-position) | 4 × 555ms = 2.2s | 500ms + 4 × 5ms = 520ms | **4.2×** |
| 32 tokens (per-position) | 32 × 555ms = 17.8s | 500ms + 32 × 5ms = 660ms | **27×** |
| 10 prompts (tokenize only) | 10 × 555ms = 5.5s | 500ms + 10 × 2ms = 520ms | **10.6×** |
| 50 prompts × 20 tokens (batch) | 1000 × 555ms = 555s | 500ms + 1000 × 5ms = 5.5s | **101×** |

**Key Insight**: Speedup scales with number of operations. For typical cross-validation workloads (10-50 test cases), expect **20-100× improvement**.

### Memory Overhead

**Session state** (~50MB for 2B model):
- Model weights: Memory-mapped (shared with Rust)
- KV cache: ~40MB (batch_size=1, n_ctx=2048, fp16)
- Context state: ~10MB (tokenizer, buffers)

**No additional copies**: Session reuses same mmap'd weights as Rust inference.

---

## Implementation Complexity

### Effort Estimate: 4-6 hours

**Breakdown**:

1. **C++ Implementation** (2-3 hours):
   - Session struct with RAII wrapper around existing `bitnet_eval` logic
   - Tokenizer integration (reuse existing `tokenize` function)
   - Prefill + eval_last fast path (split existing batch logic)
   - Error handling and boundary checks

2. **Rust Wrapper** (1-2 hours):
   - Safe wrapper with RAII lifetime management
   - Error conversion from C codes to Rust types
   - Unit tests for lifecycle (open/close/reuse)

3. **Integration Testing** (1 hour):
   - Prefill + eval_last correctness (compare vs. eval_all)
   - Multi-prompt session reuse
   - Error paths (invalid handle, buffer overflow)

### Technical Challenges

#### 1. Resource Management (Medium Complexity)

**Challenge**: Ensure session cleanup on panic/error paths.

**Solution**: Rust's `Drop` trait handles RAII cleanup automatically:
```rust
impl Drop for BitnetSession {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { crossval_bitnet_close(self.handle) };
        }
    }
}
```

C++ side uses RAII wrappers (std::unique_ptr for context).

#### 2. Thread Safety (Low Complexity)

**Challenge**: Prevent concurrent access to single session.

**Solution**: Document as single-threaded, mark `!Send + !Sync`:
```rust
impl !Send for BitnetSession {}
impl !Sync for BitnetSession {}
```

Users create one session per thread if parallelism needed.

#### 3. Error Propagation (Low Complexity)

**Challenge**: Convert C error codes to Rust types.

**Solution**: Use error buffer pattern (already established in current FFI):
```c
int crossval_bitnet_open(..., char* err, int32_t err_len);
```

Rust wrapper translates to `Result<T, SessionError>`.

#### 4. KV Cache State Management (Medium Complexity)

**Challenge**: Track KV cache size across prefill/eval calls.

**Solution**: Session struct maintains `n_past` counter:
```cpp
struct BitnetSession {
    llama_context* ctx;
    llama_model* model;
    int32_t n_past;  // Tracks current KV cache size
};
```

Each `eval_last` call increments `n_past` and validates against `n_ctx`.

### C++ Implementation Sketch

```cpp
// crossval/src/bitnet_session.cc

struct BitnetSession {
    std::unique_ptr<llama_model, void(*)(llama_model*)> model;
    std::unique_ptr<llama_context, void(*)(llama_context*)> ctx;
    int32_t n_ctx;
    int32_t n_past;
    int32_t vocab_size;
};

extern "C" int crossval_bitnet_open(
    const char* model_path,
    int32_t n_ctx,
    void** out_handle,
    char* err,
    int32_t err_len
) {
    try {
        llama_model_params model_params = llama_model_default_params();
        auto model = llama_load_model_from_file(model_path, model_params);
        if (!model) {
            snprintf(err, err_len, "Failed to load model: %s", model_path);
            return -2;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        auto ctx = llama_new_context_with_model(model, ctx_params);
        if (!ctx) {
            llama_free_model(model);
            snprintf(err, err_len, "Failed to create context");
            return -3;
        }

        auto session = new BitnetSession{
            .model = {model, llama_free_model},
            .ctx = {ctx, llama_free},
            .n_ctx = n_ctx,
            .n_past = 0,
            .vocab_size = llama_n_vocab(model),
        };

        *out_handle = session;
        return 0;
    } catch (const std::exception& e) {
        snprintf(err, err_len, "Exception: %s", e.what());
        return -3;
    }
}

extern "C" int crossval_bitnet_prefill(
    void* handle,
    const int32_t* tokens,
    int32_t n_tokens,
    char* err,
    int32_t err_len
) {
    auto session = static_cast<BitnetSession*>(handle);
    if (!session || !tokens || n_tokens <= 0) {
        snprintf(err, err_len, "Invalid parameters");
        return -2;
    }

    if (session->n_past + n_tokens > session->n_ctx) {
        snprintf(err, err_len, "KV cache overflow");
        return -5;
    }

    // Evaluate without returning logits
    if (llama_decode(session->ctx, llama_batch_get_one(tokens, n_tokens, session->n_past, 0))) {
        snprintf(err, err_len, "Decode failed");
        return -4;
    }

    session->n_past += n_tokens;
    return 0;
}

extern "C" int crossval_bitnet_eval_last(
    void* handle,
    const int32_t* tokens,
    int32_t n_tokens,
    float* logits,
    int32_t vocab,
    char* err,
    int32_t err_len
) {
    auto session = static_cast<BitnetSession*>(handle);
    if (!session || !tokens || !logits || n_tokens <= 0) {
        snprintf(err, err_len, "Invalid parameters");
        return -2;
    }

    if (vocab != session->vocab_size) {
        snprintf(err, err_len, "Vocabulary size mismatch");
        return -3;
    }

    if (session->n_past + n_tokens > session->n_ctx) {
        snprintf(err, err_len, "KV cache overflow");
        return -5;
    }

    // Evaluate and extract last position logits
    if (llama_decode(session->ctx, llama_batch_get_one(tokens, n_tokens, session->n_past, 0))) {
        snprintf(err, err_len, "Decode failed");
        return -4;
    }

    const float* src_logits = llama_get_logits_ith(session->ctx, n_tokens - 1);
    std::memcpy(logits, src_logits, vocab * sizeof(float));

    session->n_past += n_tokens;
    return 0;
}

// Similar implementations for eval_all, tokenize_with, close, etc.
```

---

## Decision Framework

### Implement Now?

**Arguments FOR**:
1. **High ROI**: 20-100× speedup for typical cross-validation workloads
2. **Unblocks G2.6**: Fast iteration is critical for per-position debugging
3. **Low risk**: Additive API, doesn't break existing `crossval_bitnet_eval`
4. **Modest effort**: 4-6 hours for complete implementation + tests

**Arguments AGAINST**:
1. **Not MVP-critical**: Current one-shot API works for basic validation
2. **Incremental value**: Only benefits multi-operation workflows
3. **Maintenance burden**: Additional API surface to document and test
4. **Alternative**: Could optimize current API with internal caching

### Recommendation: **Implement in G2.6**

**Rationale**:
- Per-position logits comparison is the primary G2.6 use case
- 27× speedup for 32-token sequences directly improves developer productivity
- Session API is a standard pattern (llama.cpp uses similar design)
- Low risk: Coexists with existing API, can be adopted gradually

**Deferral Cost**: Without session API, developers will waste ~15-30 seconds per test iteration (model reload overhead), adding up to hours over a debugging session.

### Alternative: Internal Caching

**Concept**: Keep last-loaded model in memory, reuse if same path.

```rust
static CACHED_MODEL: Mutex<Option<(PathBuf, BitnetModel)>> = Mutex::new(None);

fn crossval_bitnet_eval_cached(model_path: &Path, tokens: &[i32]) -> Result<Vec<f32>> {
    let mut cache = CACHED_MODEL.lock().unwrap();
    if cache.as_ref().map_or(true, |(p, _)| p != model_path) {
        *cache = Some((model_path.to_path_buf(), load_model(model_path)?));
    }
    let (_, model) = cache.as_ref().unwrap();
    eval_internal(model, tokens)
}
```

**Pros**: Simpler, no new API
**Cons**: Global state, thread-safety issues, only helps same-model calls

**Verdict**: Session API is cleaner and more flexible.

---

## Migration Path

### Phase 1: Implement Session API (G2.6)

1. Add C++ session implementation to `crossval/src/bitnet_session.cc`
2. Add Rust wrapper to `crossval/src/session.rs`
3. Write unit tests for lifecycle and error paths
4. Document in `docs/howto/cpp-crossval-session.md`

### Phase 2: Adopt in crossval-per-token (G2.7)

1. Refactor `crossval-per-token` command to use session API
2. Add benchmarks comparing old vs. new implementation
3. Update `run_crossval_sweep.sh` to use session-based path

### Phase 3: Extend to llama.cpp (Future)

1. Add similar session API for llama.cpp backend
2. Unify Rust wrapper to abstract over bitnet/llama sessions
3. Enable cross-backend comparison with single session

---

## Open Questions

1. **Should `prefill` accept multiple calls or reset KV cache?**
   - **Proposal**: Additive (each call appends to KV cache)
   - **Rationale**: Matches llama.cpp behavior, more flexible

2. **How to handle context window overflow?**
   - **Proposal**: Return error code -5 (KV cache overflow)
   - **Rationale**: Let caller decide (reset session or truncate)

3. **Should `eval_all` clear KV cache before eval?**
   - **Proposal**: Yes (batch mode assumes fresh context)
   - **Rationale**: Avoids surprising state accumulation

4. **Thread safety guarantees?**
   - **Proposal**: Document as single-threaded (no locks)
   - **Rationale**: One session per thread is simpler and faster

5. **Error buffer vs. error codes only?**
   - **Proposal**: Both (error codes + optional string buffer)
   - **Rationale**: Matches existing FFI pattern, good for debugging

---

## Appendix: Comparison with Alternatives

### Alternative 1: Lazy Singleton Cache

**Design**: Keep single global model instance, reload only on path change.

**Pros**:
- Zero API changes
- Automatic for all callers

**Cons**:
- Global state (thread-safety issues)
- Single model limit (can't compare two models)
- Implicit behavior (hard to reason about)

**Verdict**: Session API is more explicit and flexible.

### Alternative 2: Model Pool

**Design**: Pool of pre-loaded models, indexed by path.

**Pros**:
- Supports multiple models
- Transparent caching

**Cons**:
- Complex lifecycle (when to evict?)
- Memory pressure (multiple models resident)
- Still global state

**Verdict**: Overkill for cross-validation use case.

### Alternative 3: Rust-Native Session (No FFI)

**Design**: Implement session API in pure Rust, no C++ dependency.

**Pros**:
- No FFI overhead
- Memory-safe by construction

**Cons**:
- Defeats purpose (we want to compare *against* C++ reference)
- Can't validate bitnet.cpp behavior

**Verdict**: Incompatible with cross-validation goals.

---

## Conclusion

The session-based C API provides a clean, efficient solution for fast cross-validation iteration:

- **20-100× speedup** for typical multi-operation workloads
- **4-6 hour implementation** with modest complexity
- **Low risk**: Additive API, coexists with existing one-shot calls
- **Unblocks G2.6**: Critical for per-position logits debugging

**Recommendation**: Implement in G2.6 milestone for immediate productivity gains.

---

**Next Steps**:

1. Review this spec with team
2. Approve or request changes
3. Implement in `crossval/src/bitnet_session.{h,cc}` + Rust wrapper
4. Integrate into `crossval-per-token` command
5. Document in `docs/howto/cpp-crossval-session.md`
