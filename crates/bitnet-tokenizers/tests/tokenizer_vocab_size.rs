//! AC5: Tokenizer Parity Tests (Issue #469)
//!
//! Tests feature spec: docs/explanation/issue-469-spec.md#ac5-tokenizer-parity
//! API contract: docs/explanation/specs/issue-469-mvp-sprint-polish-spec.md#ac5
//!
//! This test validates tokenizer real_vocab_size() method for cross-validation parity.

#![cfg(all(test, feature = "cpu"))]

/// AC5: Tokenizer trait has real_vocab_size method
///
/// Tests that Tokenizer trait defines real_vocab_size() method.
///
/// # Fixture Requirements
/// - None (unit test for trait definition)
///
/// # Expected Behavior
/// - Tokenizer trait has real_vocab_size() -> usize method
/// - Default implementation returns vocab_size() (no padding)
/// - Method documented with AC5 reference
#[test]
fn test_tokenizer_trait_real_vocab_size_method() {
    // AC5: Verify Tokenizer trait has real_vocab_size method
    use bitnet_tokenizers::{BasicTokenizer, Tokenizer};

    let tokenizer = BasicTokenizer::new();

    // Default implementation should return vocab_size()
    assert_eq!(
        tokenizer.real_vocab_size(),
        tokenizer.vocab_size(),
        "AC5: Default real_vocab_size should equal vocab_size"
    );
}

/// AC5: GGUF tokenizer real vs padded vocab size
///
/// Tests that GGUF tokenizer distinguishes real and padded vocab sizes.
///
/// # Fixture Requirements
/// - tests/fixtures/tokenizer-padded.gguf: GGUF with padded vocab (32000 real, 32064 padded)
///
/// # Expected Behavior
/// - real_vocab_size() returns actual vocab from tokenizer model (e.g., 32000)
/// - vocab_size() returns GGUF-padded size for alignment (e.g., 32064)
/// - Padding amount is vocab_size() - real_vocab_size()
#[test]
#[ignore = "Fixture needed: tests/fixtures/tokenizer-padded.gguf"]
fn test_gguf_tokenizer_real_vocab_size() {
    // AC5: Verify GGUF tokenizer distinguishes real vs padded vocab size
    // FIXTURE NEEDED: tests/fixtures/tokenizer-padded.gguf
    // - Real vocab size: 32000 (from tokenizer.ggml.tokens array length)
    // - Padded vocab size: 32064 (from tokenizer.ggml.vocab_size metadata)
    //
    // Expected:
    //   use bitnet_tokenizers::GgufTokenizer;
    //
    //   let tokenizer = GgufTokenizer::from_file("tests/fixtures/tokenizer-padded.gguf")?;
    //
    //   assert_eq!(tokenizer.real_vocab_size(), 32000, "AC5: Real vocab size from tokenizer model");
    //   assert_eq!(tokenizer.vocab_size(), 32064, "AC5: Padded vocab size from GGUF metadata");
    //   assert_eq!(tokenizer.vocab_size() - tokenizer.real_vocab_size(), 64, "AC5: Padding for alignment");

    panic!(
        "AC5: GGUF tokenizer real_vocab_size not yet implemented. \
         Expected: GgufTokenizer tracks real_vocab_size and padded_vocab_size separately."
    );
}

/// AC5: HuggingFace tokenizer real vs padded vocab size
///
/// Tests that HF tokenizer returns same value for real and padded (no padding).
///
/// # Fixture Requirements
/// - tests/fixtures/tokenizer.json: HuggingFace tokenizer JSON
///
/// # Expected Behavior
/// - real_vocab_size() returns vocab size without special token padding
/// - vocab_size() returns vocab size with special tokens
/// - HF tokenizers don't have alignment padding (real == padded in most cases)
#[test]
fn test_hf_tokenizer_real_vocab_size() {
    // AC5: Verify HF tokenizer real_vocab_size
    use bitnet_tokenizers::{BasicTokenizer, Tokenizer};

    // Use BasicTokenizer as a proxy for HF tokenizer behavior
    // (HF tokenizers also use default implementation: real_vocab_size == vocab_size)
    let tokenizer = BasicTokenizer::new();

    let real_size = tokenizer.real_vocab_size();
    let vocab_size = tokenizer.vocab_size();

    assert!(real_size > 0, "AC5: Real vocab size must be positive");
    // Default implementation: vocab_size == real_vocab_size
    assert_eq!(vocab_size, real_size, "AC5: Default impl has no padding");
}

/// AC5: Tokenizer debug logging shows both sizes
///
/// Tests that tokenizer initialization logs real and padded vocab sizes.
///
/// # Fixture Requirements
/// - Capture log output during tokenizer initialization
///
/// # Expected Behavior
/// - Debug log includes: real_vocab_size, gguf_padded_size, padding amount
/// - Log format: "Tokenizer initialized: real_vocab_size=32000, gguf_padded_size=32064 (padding for alignment)"
#[test]
#[ignore = "Fixture needed: log capture mechanism for tracing output"]
fn test_tokenizer_debug_logging() {
    // AC5: Verify tokenizer debug logging
    // FIXTURE NEEDED: Capture log output during GGUF tokenizer initialization
    //
    // Expected log format:
    //   DEBUG: "Tokenizer initialized: real_vocab_size=32000, gguf_padded_size=32064 (padding: 64 tokens for alignment)"

    // TODO: Implement once GGUF tokenizer logging is available
    // let logs = capture_logs(|| {
    //     let _ = GgufTokenizer::from_file("tests/fixtures/tokenizer-padded.gguf");
    // });
    //
    // assert!(logs.contains("real_vocab_size="), "AC5: Log should show real vocab size");
    // assert!(logs.contains("gguf_padded_size="), "AC5: Log should show padded vocab size");
    // assert!(logs.contains("padding"), "AC5: Log should mention padding");

    panic!(
        "AC5: Tokenizer debug logging not yet implemented. \
         Expected: Debug log during initialization showing real and padded vocab sizes."
    );
}

/// AC5: Parity assertion uses real_vocab_size
///
/// Tests that cross-validation parity assertions use real_vocab_size for comparison.
///
/// # Fixture Requirements
/// - Mock tokenizers (Rust and C++ reference)
///
/// # Expected Behavior
/// - Parity assertion compares real_vocab_size (not padded vocab_size)
/// - Assertion fails if real vocab sizes don't match
/// - Error message mentions real_vocab_size for clarity
#[test]
#[cfg(feature = "crossval")]
fn test_parity_assertion_uses_real_vocab_size() {
    // AC5: Verify parity assertion uses real_vocab_size
    // FIXTURE NEEDED: Mock Rust and C++ tokenizers
    //
    // Expected:
    //   use crossval::parity_harness::validate_tokenizer_parity;
    //
    //   let rust_tokenizer = GgufTokenizer::from_file("tests/fixtures/tokenizer-padded.gguf")?;
    //   let cpp_vocab_size = 32000;  // C++ reference real vocab size
    //
    //   // Should succeed with real_vocab_size
    //   let result = validate_tokenizer_parity(&rust_tokenizer, cpp_vocab_size);
    //   assert!(result.is_ok(), "AC5: Parity should pass with matching real_vocab_size");

    panic!(
        "AC5: Parity assertion using real_vocab_size not yet implemented. \
         Expected: validate_tokenizer_parity compares real_vocab_size values."
    );
}

/// AC5: Parity assertion fails with padded vocab_size mismatch
///
/// Tests that parity assertion would fail if using padded vocab_size instead of real.
///
/// # Fixture Requirements
/// - Mock tokenizers with different padding
///
/// # Expected Behavior
/// - Using vocab_size (padded) would fail parity check
/// - Using real_vocab_size succeeds
/// - Demonstrates importance of real_vocab_size for cross-validation
#[test]
#[cfg(feature = "crossval")]
fn test_parity_fails_with_padded_vocab_size() {
    // AC5: Demonstrate parity failure with padded vocab_size
    // FIXTURE NEEDED: Mock tokenizers
    //
    // Expected:
    //   let rust_tokenizer = GgufTokenizer::from_file("tests/fixtures/tokenizer-padded.gguf")?;
    //   let cpp_vocab_size = 32000;  // C++ reference (real size)
    //
    //   // Rust padded vocab_size = 32064 (alignment)
    //   // C++ real vocab_size = 32000
    //   // Using padded size would fail: 32064 != 32000
    //   assert_ne!(rust_tokenizer.vocab_size(), cpp_vocab_size, "AC5: Padded sizes don't match");
    //
    //   // But using real_vocab_size succeeds: 32000 == 32000
    //   assert_eq!(rust_tokenizer.real_vocab_size(), cpp_vocab_size, "AC5: Real sizes match");

    panic!(
        "AC5: Parity comparison test not yet implemented. \
         Expected: Demonstrates importance of real_vocab_size vs padded vocab_size."
    );
}

/// AC5: GGUF tokenizer metadata parsing
///
/// Tests that GGUF tokenizer correctly parses metadata for vocab sizes.
///
/// # Fixture Requirements
/// - tests/fixtures/tokenizer-padded.gguf with metadata
///
/// # Expected Behavior
/// - Parses tokenizer.ggml.tokens array for real vocab size
/// - Parses tokenizer.ggml.vocab_size metadata for padded size
/// - Falls back to real size if vocab_size metadata missing
#[test]
#[ignore = "Fixture needed: tests/fixtures/tokenizer-padded.gguf"]
fn test_gguf_tokenizer_metadata_parsing() {
    // AC5: Verify GGUF tokenizer metadata parsing
    // FIXTURE NEEDED: tests/fixtures/tokenizer-padded.gguf
    //
    // Expected metadata parsing:
    //   1. Read tokenizer.ggml.tokens array length → real_vocab_size
    //   2. Read tokenizer.ggml.vocab_size metadata → padded_vocab_size
    //   3. If metadata missing, use real_vocab_size for both

    // TODO: Implement once GGUF tokenizer metadata parsing is available
    // let tokenizer = GgufTokenizer::from_file("tests/fixtures/tokenizer-padded.gguf")?;
    //
    // // Verify both sizes parsed
    // assert_eq!(tokenizer.real_vocab_size, 32000, "AC5: Real size from tokens array");
    // assert_eq!(tokenizer.padded_vocab_size, 32064, "AC5: Padded size from metadata");

    panic!(
        "AC5: GGUF tokenizer metadata parsing not yet implemented. \
         Expected: Parse tokenizer.ggml.tokens and tokenizer.ggml.vocab_size from GGUF."
    );
}

/// AC5: Tokenizer vocab_size vs real_vocab_size behavior
///
/// Tests that vocab_size and real_vocab_size return expected values.
///
/// # Fixture Requirements
/// - None (unit test for API contract)
///
/// # Expected Behavior
/// - vocab_size() returns padded size (alignment boundary)
/// - real_vocab_size() returns actual token count
/// - For GGUF: vocab_size() ≥ real_vocab_size()
/// - For HF: vocab_size() may differ by special tokens only
#[test]
fn test_vocab_size_vs_real_vocab_size_contract() {
    // AC5: Verify vocab_size vs real_vocab_size API contract
    use bitnet_tokenizers::{BasicTokenizer, Tokenizer};

    let tokenizer = BasicTokenizer::new();

    // API contract: vocab_size() >= real_vocab_size()
    assert!(
        tokenizer.vocab_size() >= tokenizer.real_vocab_size(),
        "AC5: Padded vocab_size should be ≥ real_vocab_size"
    );

    // Default implementation: they are equal
    assert_eq!(
        tokenizer.vocab_size(),
        tokenizer.real_vocab_size(),
        "AC5: Default impl has no padding"
    );
}

/// AC5: Cross-validation error message clarity
///
/// Tests that parity errors mention real_vocab_size for clarity.
///
/// # Fixture Requirements
/// - Mock tokenizers with mismatched vocab sizes
///
/// # Expected Behavior
/// - Error message mentions "real_vocab_size" (not just "vocab_size")
/// - Error shows both Rust and C++ values
/// - Error provides context: "breaks tokenization parity"
#[test]
#[cfg(feature = "crossval")]
fn test_parity_error_message_clarity() {
    // AC5: Verify parity error message mentions real_vocab_size
    // FIXTURE NEEDED: Mock tokenizers with mismatched real vocab sizes
    //
    // Expected error format:
    //   "Tokenizer vocab size mismatch breaks parity: Rust real_vocab_size=32000, C++ vocab_size=31000"

    // TODO: Implement once parity validation is available
    // let rust_tokenizer = mock_rust_tokenizer(32000);
    // let cpp_vocab_size = 31000;  // Mismatch
    //
    // let result = validate_tokenizer_parity(&rust_tokenizer, cpp_vocab_size);
    // assert!(result.is_err(), "AC5: Should fail with mismatched vocab sizes");
    //
    // let err_msg = result.unwrap_err().to_string();
    // assert!(err_msg.contains("real_vocab_size"), "AC5: Error should mention real_vocab_size");
    // assert!(err_msg.contains("32000"), "AC5: Error should show Rust value");
    // assert!(err_msg.contains("31000"), "AC5: Error should show C++ value");

    panic!(
        "AC5: Parity error message clarity not yet implemented. \
         Expected: Error messages mention real_vocab_size with diagnostic context."
    );
}
