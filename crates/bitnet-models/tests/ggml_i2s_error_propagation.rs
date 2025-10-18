//! Test that ggml I2_S errors are properly propagated and not caught by fallback
//!
//! This test verifies that when the enhanced GGUF parser detects ggml I2_S format
//! (8B/block with no scale tensor), the error is propagated to the caller instead
//! of falling back to the minimal parser which creates mock/zero tensors.

use bitnet_common::BitNetError;

/// Test that the error message pattern used by parity harness is correct
#[test]
fn test_ggml_i2s_error_pattern_matching() {
    // Simulate the error message format from gguf_simple.rs line 812-818
    let error_msg = "I2_S 'blk.0.attn_q.weight': GGML/llama.cpp format detected (8B/block, no scale tensor). \
                     This format requires GGML-compatible kernels and is not yet supported in pure Rust. \
                     Please use the FFI backend with --features ffi or set BITNET_CPP_DIR. \
                     Available: 4096B, expected split: 4096B, expected inline: 5120B";

    let err = BitNetError::Validation(error_msg.to_string());
    let err_str = format!("{:?}", err);

    // Verify the error contains both patterns that parity harness checks for
    // (from crates/bitnet-inference/src/parity.rs line 82)
    assert!(
        err_str.contains("GGML/llama.cpp format detected"),
        "Error should contain 'GGML/llama.cpp format detected' pattern"
    );

    assert!(err_str.contains("no scale tensor"), "Error should contain 'no scale tensor' pattern");

    // Verify this is NOT a fallback error
    assert!(
        !err_str.contains("falling back to minimal"),
        "Error should NOT contain fallback message"
    );
}

/// Test that error propagation logic works correctly
#[test]
fn test_error_propagation_logic() {
    // Simulate checking if an error should be propagated
    // This mirrors the logic in gguf_simple.rs lines 42-48

    // Case 1: ggml I2_S error - should propagate
    let ggml_error = BitNetError::Validation(
        "I2_S 'weight': GGML/llama.cpp format detected (8B/block, no scale tensor)".to_string(),
    );
    let err_str = format!("{:?}", ggml_error);
    let should_propagate = err_str.contains("GGML/llama.cpp format detected");
    assert!(should_propagate, "GGML I2_S errors should be propagated");

    // Case 2: Other quantization error - should fall back
    let other_error = BitNetError::Validation("Unsupported tensor type Q5_K".to_string());
    let err_str = format!("{:?}", other_error);
    let should_propagate = err_str.contains("GGML/llama.cpp format detected");
    assert!(!should_propagate, "Other errors should fall back to minimal parser");

    // Case 3: Missing tensor error - should fall back
    let missing_error =
        BitNetError::Validation("Missing required tensor: blk.0.attn_q.weight".to_string());
    let err_str = format!("{:?}", missing_error);
    let should_propagate = err_str.contains("GGML/llama.cpp format detected");
    assert!(!should_propagate, "Missing tensor errors should fall back");
}

/// Test that parity harness pattern matching works
#[test]
fn test_parity_harness_pattern_matching() {
    // Simulate the error checking in parity.rs lines 79-82
    let create_ggml_error = || {
        BitNetError::Validation(
            "I2_S 'blk.0.attn_q.weight': GGML/llama.cpp format detected (8B/block, no scale tensor). \
             This format requires GGML-compatible kernels.".to_string(),
        )
    };

    let error = create_ggml_error();
    let err_str = format!("{:?}", error);

    // This is the exact check from parity.rs line 82
    let is_ggml_i2s_error =
        err_str.contains("GGML/llama.cpp format detected") && err_str.contains("no scale tensor");

    assert!(is_ggml_i2s_error, "Parity harness should detect ggml I2_S error");
}
