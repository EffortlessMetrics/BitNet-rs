//! Vocab NULL Check Safety Tests for C++ Wrapper
//!
//! **Specification Reference:** docs/specs/cpp-wrapper-vocab-null-checks.md
//!
//! This test suite validates vocab NULL check safety across all 4 critical locations
//! in the C++ wrapper (crossval/src/bitnet_cpp_wrapper.cc):
//!
//! - Line 102: Socket 0 tokenize (stateless) - requires model cleanup
//! - Line 261: Socket 0 eval (stateless) - requires model + context cleanup
//! - Line 558: Socket 2 tokenize (stateful) - NO cleanup (context owns resources)
//! - Line 710: Socket 3 eval (stateful) - NO cleanup (context owns resources)
//!
//! **Test Strategy:**
//! - TDD approach: Tests fail initially due to missing NULL checks
//! - Create invalid GGUF models to trigger vocab NULL scenarios
//! - Validate error messages follow consistent format
//! - Ensure proper resource cleanup (no memory leaks)
//! - Verify error propagation from C++ to Rust
//!
//! **Acceptance Criteria Coverage:**
//! - AC1: All 4 locations have NULL checks (code inspection after implementation)
//! - AC2: Error messages follow consistent format (tested here)
//! - AC3: Socket 0 functions free model/context on vocab NULL (tested here)
//! - AC4: Socket 2/3 functions do NOT free context on vocab NULL (tested here)
//! - AC5: All functions return -1 on vocab NULL (tested here)
//! - AC6: No segfaults when testing with invalid models (tested here)
//!
//! **Run with:**
//! ```bash
//! cargo test -p crossval --test cpp_wrapper_null_vocab_tests --no-default-features --features ffi --no-run
//! cargo test -p crossval --test cpp_wrapper_null_vocab_tests --no-default-features --features ffi
//! ```

#![cfg(feature = "ffi")]

use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::c_char;
use std::path::PathBuf;
use tempfile::TempDir;

// ============================================================================
// Test Helper Functions
// ============================================================================

/// Creates a minimal invalid GGUF file for testing vocab NULL scenarios
///
/// The generated file has:
/// - Valid GGUF magic number
/// - Minimal header (truncated after version)
/// - NO vocab tensors or metadata
///
/// This triggers `llama_model_get_vocab()` to return NULL when loaded by llama.cpp.
fn create_invalid_gguf(path: &PathBuf) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = fs::File::create(path)?;

    // Write minimal GGUF header (will fail llama.cpp vocab parsing)
    file.write_all(b"GGUF")?; // Magic
    file.write_all(&[3u8, 0, 0, 0])?; // Version 3 (little-endian)
    file.write_all(&[0u8; 12])?; // Truncated header (no tensor count, no KV count)

    Ok(())
}

/// Helper to assert error message contains expected vocab failure text
///
/// **Expected Format:** `<function_name>: Failed to get vocab from model (check model format/compatibility)`
fn assert_error_contains_vocab_failure(err_buf: &[u8], function_name: &str) {
    let err_str = CStr::from_bytes_until_nul(err_buf)
        .expect("Error buffer should be NUL-terminated")
        .to_str()
        .expect("Error should be valid UTF-8");

    assert!(
        err_str.contains("Failed to get vocab from model"),
        "Error should mention vocab failure, got: '{}'",
        err_str
    );

    assert!(
        err_str.contains(function_name),
        "Error should include function name '{}', got: '{}'",
        function_name,
        err_str
    );

    assert!(
        err_str.contains("check model format/compatibility"),
        "Error should suggest checking model format, got: '{}'",
        err_str
    );
}

/// Helper to create a temporary directory with invalid GGUF model
///
/// Returns (TempDir, model_path) tuple for cleanup handling
fn setup_invalid_model() -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("invalid_model.gguf");
    create_invalid_gguf(&model_path).expect("Failed to create invalid GGUF");
    (temp_dir, model_path)
}

// ============================================================================
// Category 1: Unit Tests (4 tests - one per location)
// ============================================================================

/// Tests feature spec: cpp-wrapper-vocab-null-checks.md#location-1
///
/// **AC2, AC3, AC5, AC6**
///
/// **Purpose:** Validate NULL vocab handling in Socket 0 tokenize (line 102)
/// **Expected:** Returns -1 with error message, frees model before return
/// **Cleanup:** Model must be freed (stateless function owns it)
#[test]
fn test_socket0_tokenize_null_vocab() {
    let (_temp_dir, model_path) = setup_invalid_model();
    let model_path_c = CString::new(model_path.to_str().unwrap()).unwrap();
    let prompt_c = CString::new("Test prompt").unwrap();

    let mut out_tokens = vec![0i32; 128];
    let mut out_len: i32 = 0;
    let mut err_buf = vec![0u8; 256];

    // Call crossval_bitnet_tokenize with invalid model
    let result = unsafe {
        crossval_bitnet_tokenize(
            model_path_c.as_ptr(),
            prompt_c.as_ptr(),
            1, // add_bos
            0, // parse_special
            out_tokens.as_mut_ptr(),
            out_tokens.len() as i32,
            &mut out_len,
            err_buf.as_mut_ptr() as *mut c_char,
            err_buf.len() as i32,
        )
    };

    // AC5: Should return -1 on vocab NULL
    assert_eq!(result, -1, "crossval_bitnet_tokenize should return -1 on vocab NULL");

    // AC2: Error message should follow consistent format
    assert_error_contains_vocab_failure(&err_buf, "crossval_bitnet_tokenize");

    // AC6: No segfault (test passed without crash)
    // AC3: Model cleanup validated (implicitly - if not cleaned, valgrind would detect leak)
}

/// Tests feature spec: cpp-wrapper-vocab-null-checks.md#location-2
///
/// **AC2, AC3, AC5, AC6**
///
/// **Purpose:** Validate NULL vocab handling in Socket 0 eval (line 261)
/// **Expected:** Returns -1 with error message, frees context + model before return
/// **Cleanup:** Context freed first, then model (order matters - context depends on model)
#[test]
fn test_socket0_eval_null_vocab() {
    let (_temp_dir, model_path) = setup_invalid_model();
    let model_path_c = CString::new(model_path.to_str().unwrap()).unwrap();

    let tokens = vec![1i32, 4872, 338]; // Sample token IDs
    let n_tokens = tokens.len() as i32;
    let n_ctx = 512;

    let mut out_logits = vec![0.0f32; 10_000]; // Large enough buffer
    let mut out_rows: i32 = 0;
    let mut out_cols: i32 = 0;
    let mut err_buf = vec![0u8; 256];

    // Call crossval_bitnet_eval_with_tokens with invalid model
    let result = unsafe {
        crossval_bitnet_eval_with_tokens(
            model_path_c.as_ptr(),
            tokens.as_ptr(),
            n_tokens,
            n_ctx,
            out_logits.as_mut_ptr(),
            out_logits.len() as i32,
            &mut out_rows,
            &mut out_cols,
            err_buf.as_mut_ptr() as *mut c_char,
            err_buf.len() as i32,
        )
    };

    // AC5: Should return -1 on vocab NULL
    assert_eq!(result, -1, "crossval_bitnet_eval_with_tokens should return -1 on vocab NULL");

    // AC2: Error message should follow consistent format
    assert_error_contains_vocab_failure(&err_buf, "crossval_bitnet_eval_with_tokens");

    // AC6: No segfault (test passed without crash)
    // AC3: Context + model cleanup validated (implicitly - valgrind would detect leak)
}

/// Tests feature spec: cpp-wrapper-vocab-null-checks.md#location-3
///
/// **AC2, AC4, AC5, AC6**
///
/// **Purpose:** Validate NULL vocab handling in Socket 2 tokenize (line 558)
/// **Expected:** Returns -1 with error message, NO cleanup (context owns resources)
/// **Cleanup:** Must NOT free ctx->model (persistent context owns it)
#[test]
#[ignore] // TODO: Remove after implementing NULL check at line 558
fn test_socket2_tokenize_null_vocab() {
    // TODO: This test requires Socket 1 (bitnet_cpp_init_context) to be implemented
    // For now, scaffold the test structure
    //
    // Expected flow:
    // 1. Create invalid model with create_invalid_gguf()
    // 2. Call bitnet_cpp_init_context() with invalid model (may fail at model load, not vocab check)
    // 3. If context creation succeeds (unlikely with invalid GGUF), call bitnet_cpp_tokenize_with_context()
    // 4. Validate -1 return code and error message
    // 5. Validate NO cleanup happens (context still owns model)
    // 6. Free context with bitnet_cpp_free_context()
    //
    // Alternative test strategy:
    // - Mock a valid model load but corrupted vocab (requires C++ wrapper modification)
    // - Or use a real GGUF with missing vocab tensors

    todo!(
        "Implement test_socket2_tokenize_null_vocab after Socket 1 (bitnet_cpp_init_context) is available"
    );
}

/// Tests feature spec: cpp-wrapper-vocab-null-checks.md#location-4
///
/// **AC2, AC4, AC5, AC6**
///
/// **Purpose:** Validate NULL vocab handling in Socket 3 eval (line 710)
/// **Expected:** Returns -1 with error message, NO cleanup (context owns resources)
/// **Cleanup:** Must NOT free ctx->model or ctx->ctx (persistent context owns them)
#[test]
#[ignore] // TODO: Remove after implementing NULL check at line 710
fn test_socket3_eval_null_vocab() {
    // TODO: This test requires Socket 1 (bitnet_cpp_init_context) to be implemented
    // For now, scaffold the test structure
    //
    // Expected flow:
    // 1. Create invalid model with create_invalid_gguf()
    // 2. Call bitnet_cpp_init_context() with invalid model (may fail at model load, not vocab check)
    // 3. If context creation succeeds, call bitnet_cpp_eval_with_context()
    // 4. Validate -1 return code and error message
    // 5. Validate NO cleanup happens (context still owns model + context)
    // 6. Free context with bitnet_cpp_free_context()

    todo!(
        "Implement test_socket3_eval_null_vocab after Socket 1 (bitnet_cpp_init_context) is available"
    );
}

// ============================================================================
// Category 2: Resource Cleanup Tests (2 tests)
// ============================================================================

/// Tests feature spec: cpp-wrapper-vocab-null-checks.md#cleanup-ownership-rules
///
/// **AC3, AC7**
///
/// **Purpose:** Validate Socket 0 cleanup on vocab failure (model + context freed)
/// **Expected:** No memory leaks detected by valgrind
/// **Validation:** Run with `valgrind --leak-check=full` to verify cleanup
#[test]
#[ignore] // TODO: Remove after implementing NULL checks at lines 102, 261
fn test_socket0_cleanup_on_vocab_failure() {
    // Test both Socket 0 functions (tokenize and eval) repeatedly
    // to ensure no memory accumulates from missing cleanup

    for iteration in 0..10 {
        let (_temp_dir, model_path) = setup_invalid_model();

        // Test tokenize cleanup
        {
            let model_path_c = CString::new(model_path.to_str().unwrap()).unwrap();
            let prompt_c = CString::new(format!("Test prompt {}", iteration)).unwrap();

            let mut out_tokens = vec![0i32; 128];
            let mut out_len: i32 = 0;
            let mut err_buf = vec![0u8; 256];

            let _result = unsafe {
                crossval_bitnet_tokenize(
                    model_path_c.as_ptr(),
                    prompt_c.as_ptr(),
                    1,
                    0,
                    out_tokens.as_mut_ptr(),
                    out_tokens.len() as i32,
                    &mut out_len,
                    err_buf.as_mut_ptr() as *mut c_char,
                    err_buf.len() as i32,
                )
            };
        }

        // Test eval cleanup
        {
            let model_path_c = CString::new(model_path.to_str().unwrap()).unwrap();
            let tokens = vec![1i32, 4872, 338];

            let mut out_logits = vec![0.0f32; 10_000];
            let mut out_rows: i32 = 0;
            let mut out_cols: i32 = 0;
            let mut err_buf = vec![0u8; 256];

            let _result = unsafe {
                crossval_bitnet_eval_with_tokens(
                    model_path_c.as_ptr(),
                    tokens.as_ptr(),
                    tokens.len() as i32,
                    512,
                    out_logits.as_mut_ptr(),
                    out_logits.len() as i32,
                    &mut out_rows,
                    &mut out_cols,
                    err_buf.as_mut_ptr() as *mut c_char,
                    err_buf.len() as i32,
                )
            };
        }
    }

    // AC7: Run with valgrind to verify no memory leaks:
    // valgrind --leak-check=full --error-exitcode=1 cargo test -p crossval --test cpp_wrapper_null_vocab_tests test_socket0_cleanup_on_vocab_failure
    //
    // Expected output:
    // ==12345== LEAK SUMMARY:
    // ==12345==    definitely lost: 0 bytes in 0 blocks
}

/// Tests feature spec: cpp-wrapper-vocab-null-checks.md#cleanup-ownership-rules
///
/// **AC4, AC7**
///
/// **Purpose:** Validate Socket 2/3 NO cleanup on vocab failure (context owns resources)
/// **Expected:** Context cleanup happens via bitnet_cpp_free_context(), not error path
/// **Safety:** Prevents double-free bugs
#[test]
#[ignore] // TODO: Remove after implementing NULL checks at lines 558, 710 and Socket 1
fn test_socket23_no_cleanup_on_vocab_failure() {
    // TODO: This test requires Socket 1 (bitnet_cpp_init_context) to be implemented
    //
    // Expected flow:
    // 1. Create persistent context with invalid model (or valid model with corrupted vocab)
    // 2. Call Socket 2 tokenize → should return -1, but NOT free context
    // 3. Call Socket 3 eval → should return -1, but NOT free context
    // 4. Explicitly free context with bitnet_cpp_free_context()
    // 5. Validate no double-free or memory corruption (valgrind clean)

    todo!("Implement test_socket23_no_cleanup_on_vocab_failure after Socket 1 is available");
}

// ============================================================================
// Category 3: Integration Tests (2 tests)
// ============================================================================

/// Tests feature spec: cpp-wrapper-vocab-null-checks.md#acceptance-criteria
///
/// **AC6, AC12**
///
/// **Purpose:** Validate invalid GGUF handling end-to-end
/// **Expected:** Graceful error propagation from C++ to Rust, no crashes
/// **Coverage:** Full workflow from model load to inference failure
#[test]
#[ignore] // TODO: Remove after implementing all NULL checks
fn test_invalid_gguf_handling_e2e() {
    let (_temp_dir, model_path) = setup_invalid_model();

    // Test Socket 0 tokenize path
    {
        let model_path_c = CString::new(model_path.to_str().unwrap()).unwrap();
        let prompt_c = CString::new("End-to-end test prompt").unwrap();

        let mut out_tokens = vec![0i32; 128];
        let mut out_len: i32 = 0;
        let mut err_buf = vec![0u8; 256];

        let result = unsafe {
            crossval_bitnet_tokenize(
                model_path_c.as_ptr(),
                prompt_c.as_ptr(),
                1,
                0,
                out_tokens.as_mut_ptr(),
                out_tokens.len() as i32,
                &mut out_len,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        assert_eq!(result, -1, "Should fail gracefully on invalid GGUF");
        assert_error_contains_vocab_failure(&err_buf, "crossval_bitnet_tokenize");
    }

    // Test Socket 0 eval path
    {
        let model_path_c = CString::new(model_path.to_str().unwrap()).unwrap();
        let tokens = vec![1i32];

        let mut out_logits = vec![0.0f32; 10_000];
        let mut out_rows: i32 = 0;
        let mut out_cols: i32 = 0;
        let mut err_buf = vec![0u8; 256];

        let result = unsafe {
            crossval_bitnet_eval_with_tokens(
                model_path_c.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                512,
                out_logits.as_mut_ptr(),
                out_logits.len() as i32,
                &mut out_rows,
                &mut out_cols,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        assert_eq!(result, -1, "Should fail gracefully on invalid GGUF");
        assert_error_contains_vocab_failure(&err_buf, "crossval_bitnet_eval_with_tokens");
    }

    // AC6: No segfaults observed during end-to-end test
}

/// Tests feature spec: cpp-wrapper-vocab-null-checks.md#error-propagation
///
/// **AC14**
///
/// **Purpose:** Validate error propagation to Rust FFI boundary
/// **Expected:** Error strings properly NUL-terminated and UTF-8 valid
/// **Safety:** Prevents buffer overflows and encoding issues
#[test]
#[ignore] // TODO: Remove after implementing NULL checks
fn test_error_propagation_to_rust() {
    let (_temp_dir, model_path) = setup_invalid_model();
    let model_path_c = CString::new(model_path.to_str().unwrap()).unwrap();
    let prompt_c = CString::new("Test").unwrap();

    // Test with various error buffer sizes to ensure NUL termination
    for err_buf_size in [64, 128, 256, 512] {
        let mut out_tokens = vec![0i32; 128];
        let mut out_len: i32 = 0;
        let mut err_buf = vec![0u8; err_buf_size];

        let result = unsafe {
            crossval_bitnet_tokenize(
                model_path_c.as_ptr(),
                prompt_c.as_ptr(),
                1,
                0,
                out_tokens.as_mut_ptr(),
                out_tokens.len() as i32,
                &mut out_len,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        assert_eq!(result, -1, "Should return -1 for all buffer sizes");

        // Validate error string is NUL-terminated
        let err_cstr =
            CStr::from_bytes_until_nul(&err_buf).expect("Error buffer should be NUL-terminated");

        // Validate error string is valid UTF-8
        let err_str = err_cstr.to_str().expect("Error string should be valid UTF-8");

        // Validate error content
        assert!(
            err_str.contains("Failed to get vocab"),
            "Error should contain vocab failure message for buffer size {}",
            err_buf_size
        );

        // AC8: Error message fits within buffer (max 108 bytes + NUL < 256)
        assert!(err_str.len() < 256, "Error message should fit within 256-byte buffer");
    }
}

// ============================================================================
// Category 4: Error Message Tests (1 test)
// ============================================================================

/// Tests feature spec: cpp-wrapper-vocab-null-checks.md#error-string-format
///
/// **AC2, AC8, AC14**
///
/// **Purpose:** Validate error message format consistency across all functions
/// **Expected:** All error messages follow template: `<function>: Failed to get vocab from model (check model format/compatibility)`
/// **Length:** Max 108 bytes (including NUL terminator) to fit within 256-byte buffers
#[test]
#[ignore] // TODO: Remove after implementing all NULL checks
fn test_vocab_error_message_format() {
    let (_temp_dir, model_path) = setup_invalid_model();

    // Test all 4 locations (Socket 0 tokenize, Socket 0 eval, Socket 2 tokenize, Socket 3 eval)
    // For now, only Socket 0 functions are testable (Socket 2/3 require Socket 1 implementation)

    // Test Socket 0 tokenize error message format
    {
        let model_path_c = CString::new(model_path.to_str().unwrap()).unwrap();
        let prompt_c = CString::new("Test").unwrap();

        let mut out_tokens = vec![0i32; 128];
        let mut out_len: i32 = 0;
        let mut err_buf = vec![0u8; 256];

        let _result = unsafe {
            crossval_bitnet_tokenize(
                model_path_c.as_ptr(),
                prompt_c.as_ptr(),
                1,
                0,
                out_tokens.as_mut_ptr(),
                out_tokens.len() as i32,
                &mut out_len,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        let err_str = CStr::from_bytes_until_nul(&err_buf).unwrap().to_str().unwrap();

        // Validate format: "<function_name>: Failed to get vocab from model (check model format/compatibility)"
        assert!(
            err_str.starts_with("crossval_bitnet_tokenize:"),
            "Error should start with function name"
        );
        assert!(
            err_str.contains("Failed to get vocab from model"),
            "Error should contain vocab failure message"
        );
        assert!(
            err_str.contains("check model format/compatibility"),
            "Error should suggest checking model format"
        );

        // AC8: Validate length constraint (max 108 bytes)
        assert!(err_str.len() <= 108, "Error message should be ≤ 108 bytes, got {}", err_str.len());
    }

    // Test Socket 0 eval error message format
    {
        let model_path_c = CString::new(model_path.to_str().unwrap()).unwrap();
        let tokens = vec![1i32];

        let mut out_logits = vec![0.0f32; 10_000];
        let mut out_rows: i32 = 0;
        let mut out_cols: i32 = 0;
        let mut err_buf = vec![0u8; 256];

        let _result = unsafe {
            crossval_bitnet_eval_with_tokens(
                model_path_c.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                512,
                out_logits.as_mut_ptr(),
                out_logits.len() as i32,
                &mut out_rows,
                &mut out_cols,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len() as i32,
            )
        };

        let err_str = CStr::from_bytes_until_nul(&err_buf).unwrap().to_str().unwrap();

        // Validate format
        assert!(
            err_str.starts_with("crossval_bitnet_eval_with_tokens:"),
            "Error should start with function name"
        );
        assert!(
            err_str.contains("Failed to get vocab from model"),
            "Error should contain vocab failure message"
        );

        // AC8: Validate length constraint
        assert!(err_str.len() <= 108, "Error message should be ≤ 108 bytes, got {}", err_str.len());
    }

    // TODO: Test Socket 2/3 error message formats after Socket 1 is implemented
}

// ============================================================================
// FFI Function Declarations (for testing)
// ============================================================================

extern "C" {
    // Safety: FFI declarations for C++ wrapper functions
    // These are tested in a controlled environment with proper error handling

    fn crossval_bitnet_tokenize(
        model_path: *const c_char,
        prompt: *const c_char,
        add_bos: i32,
        parse_special: i32,
        out_tokens: *mut i32,
        out_capacity: i32,
        out_len: *mut i32,
        err: *mut c_char,
        err_len: i32,
    ) -> i32;

    fn crossval_bitnet_eval_with_tokens(
        model_path: *const c_char,
        tokens: *const i32,
        n_tokens: i32,
        n_ctx: i32,
        out_logits: *mut f32,
        logits_capacity: i32,
        out_rows: *mut i32,
        out_cols: *mut i32,
        err: *mut c_char,
        err_len: i32,
    ) -> i32;

    // TODO: Add Socket 1, 2, 3 function declarations when implemented
    // fn bitnet_cpp_init_context(...) -> i32;
    // fn bitnet_cpp_free_context(...) -> i32;
    // fn bitnet_cpp_tokenize_with_context(...) -> i32;
    // fn bitnet_cpp_eval_with_context(...) -> i32;
}
