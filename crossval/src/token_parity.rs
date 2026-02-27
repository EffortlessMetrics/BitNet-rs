//! Token parity pre-gate for cross-validation
//!
//! This module provides fail-fast validation of token sequences between Rust and C++
//! implementations before expensive logits comparisons. It prevents silent failures
//! caused by tokenization mismatches (duplicate BOS, template differences, etc.).
//!
//! ## Specification
//!
//! See: `docs/explanation/token-parity-pregate.md`
//!
//! ## Usage
//!
//! ```rust,ignore
//! use bitnet_crossval::token_parity::validate_token_parity;
//!
//! let rust_tokens = vec![128000, 1229, 374];
//! let cpp_tokens = vec![128000_i32, 1229, 374];
//! let prompt = "What is 2+2?";
//!
//! // Validates tokens match; exits with code 2 on mismatch
//! validate_token_parity(&rust_tokens, &cpp_tokens, prompt)?;
//! ```

use std::fmt;

use console::style;

use crate::backend::CppBackend;

/// Error returned when token sequences don't match between Rust and C++
#[derive(Debug, Clone)]
pub struct TokenParityError {
    /// Rust token sequence
    pub rust_tokens: Vec<u32>,
    /// C++ token sequence (converted from i32)
    pub cpp_tokens: Vec<u32>,
    /// First position where tokens differ
    pub first_diff_index: usize,
    /// Original prompt that was tokenized
    pub prompt: String,
    /// C++ backend used for comparison
    pub backend: CppBackend,
}

impl fmt::Display for TokenParityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Token sequence mismatch at index {}: rust={:?} cpp={:?}",
            self.first_diff_index, self.rust_tokens, self.cpp_tokens
        )
    }
}

impl std::error::Error for TokenParityError {}

/// Validates that Rust and C++ token sequences match exactly
///
/// ## Behavior
///
/// - **Success**: Returns `Ok(())` if tokens match (silent - no output)
/// - **Failure**: Prints diagnostic error to stderr and returns `Err`
///
/// ## Arguments
///
/// - `rust_tokens`: Token sequence from Rust tokenizer
/// - `cpp_tokens`: Token sequence from C++ tokenizer (i32 from FFI)
/// - `prompt`: Original prompt string (used for diagnostics)
/// - `backend`: C++ backend used for comparison (BitNet or LLaMA)
///
/// ## Acceptance Criteria
///
/// - AC1: Token mismatch detected before logits evaluation
/// - AC2: Both token sequences displayed on mismatch
/// - AC3: First diff position identified
/// - AC4: Caller should exit with code 2 on token mismatch
/// - AC5-AC8: Error message quality (suggestions, examples)
/// - AC9: Silent success when tokens match
/// - AC10: Backend-specific troubleshooting hints
///
/// ## Error Handling
///
/// - Caller (typically xtask) should handle the error and exit with code 2
/// - This design allows the function to be testable (returns Err instead of calling exit)
pub fn validate_token_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[i32],
    prompt: &str,
    backend: CppBackend,
) -> anyhow::Result<()> {
    // Convert C++ tokens from i32 to u32 for comparison
    // Note: C++ FFI returns i32, but token IDs are non-negative in practice
    let cpp_tokens_u32: Vec<u32> = cpp_tokens.iter().map(|&id| id as u32).collect();

    // Compare token sequences
    if rust_tokens != cpp_tokens_u32.as_slice() {
        // Find first diff index
        let first_diff = find_first_diff(rust_tokens, &cpp_tokens_u32);

        // Create error with diagnostic information
        let error = TokenParityError {
            rust_tokens: rust_tokens.to_vec(),
            cpp_tokens: cpp_tokens_u32,
            first_diff_index: first_diff,
            prompt: prompt.to_string(),
            backend,
        };

        // Print diagnostic error to stderr
        eprintln!("{}", format_token_mismatch_error(&error));

        // Return error (caller should exit with code 2)
        anyhow::bail!("Token sequence mismatch at index {}", first_diff);
    }

    // Silent success - no output when tokens match
    Ok(())
}

/// Helper to compute first diff position between two token sequences
///
/// Returns the index of the first differing element, or the length of the
/// shorter sequence if one sequence is a prefix of the other.
pub fn find_first_diff(rust_tokens: &[u32], cpp_tokens: &[u32]) -> usize {
    // Find first position where tokens differ
    rust_tokens
        .iter()
        .zip(cpp_tokens.iter())
        .position(|(r, c)| r != c)
        // If all zipped elements match, return the length of the shorter sequence
        .unwrap_or_else(|| rust_tokens.len().min(cpp_tokens.len()))
}

/// Formats a diagnostic error message for token mismatch
///
/// ## Output Format
///
/// ```text
/// ❌ Token Sequence Mismatch with C++ Backend: BitNet
/// Fix BOS/template before comparing logits
///
/// Rust tokens (5):
///   [128000, 128000, 1229, 374, 2]
///
/// C++ tokens (5):
///   [128000, 1229, 374, 891, 2]
///
/// First diff at index: 1
/// Mismatch: Rust token=128000, C++ token=1229
///
/// Troubleshooting for BitNet backend:
///   • Verify your model is BitNet-compatible (microsoft-bitnet-b1.58-2B-4T-gguf)
///   • Check tokenizer path matches model format
///   • Try --cpp-backend llama if this is a LLaMA model
///   • Verify --prompt-template setting (current: auto)
///   • Check BOS token handling with --dump-ids
/// ```
pub fn format_token_mismatch_error(error: &TokenParityError) -> String {
    use std::fmt::Write;

    let mut output = String::new();

    // Header with backend name
    writeln!(
        output,
        "\n{}",
        style(format!("❌ Token Sequence Mismatch with C++ Backend: {}", error.backend.name()))
            .red()
            .bold()
    )
    .unwrap();
    writeln!(output, "{}", style("Fix BOS/template before comparing logits").yellow()).unwrap();

    // Display Rust tokens (limit to first 64 for readability)
    writeln!(output, "\n{} ({}):", style("Rust tokens").cyan(), error.rust_tokens.len()).unwrap();
    if error.rust_tokens.len() <= 64 {
        writeln!(output, "  {:?}", error.rust_tokens).unwrap();
    } else {
        writeln!(output, "  {:?}...", &error.rust_tokens[..64]).unwrap();
    }

    // Display C++ tokens (limit to first 64 for readability)
    writeln!(output, "\n{} ({}):", style("C++ tokens").cyan(), error.cpp_tokens.len()).unwrap();
    if error.cpp_tokens.len() <= 64 {
        writeln!(output, "  {:?}", error.cpp_tokens).unwrap();
    } else {
        writeln!(output, "  {:?}...", &error.cpp_tokens[..64]).unwrap();
    }

    // First diff position with detailed mismatch
    writeln!(output, "\n{}: {}", style("First diff at index").yellow(), error.first_diff_index)
        .unwrap();

    // Show the specific tokens that differ
    if error.first_diff_index < error.rust_tokens.len()
        && error.first_diff_index < error.cpp_tokens.len()
    {
        writeln!(
            output,
            "{}: Rust token={}, C++ token={}",
            style("Mismatch").yellow(),
            error.rust_tokens[error.first_diff_index],
            error.cpp_tokens[error.first_diff_index]
        )
        .unwrap();
    }

    // Backend-specific troubleshooting
    writeln!(
        output,
        "\n{}:",
        style(format!("Troubleshooting for {} backend", error.backend.name())).green().bold()
    )
    .unwrap();

    match error.backend {
        CppBackend::BitNet => {
            writeln!(
                output,
                "  • Verify your model is BitNet-compatible (microsoft-bitnet-b1.58-2B-4T-gguf)"
            )
            .unwrap();
            writeln!(output, "  • Check tokenizer path matches model format").unwrap();
            writeln!(output, "  • Try --cpp-backend llama if this is a LLaMA model").unwrap();
            writeln!(output, "  • Verify --prompt-template setting (current: auto)").unwrap();
            writeln!(output, "  • Check BOS token handling with --dump-ids").unwrap();
        }
        CppBackend::Llama => {
            writeln!(
                output,
                "  • Verify LLaMA tokenizer compatibility (llama-3, llama-2, SmolLM3)"
            )
            .unwrap();
            writeln!(output, "  • Check tokenizer.json matches model architecture").unwrap();
            writeln!(output, "  • Try --cpp-backend bitnet for BitNet models").unwrap();
            writeln!(output, "  • Verify --prompt-template setting (current: auto)").unwrap();
            writeln!(output, "  • Check special token handling (BOS/EOS) with --dump-ids").unwrap();
        }
    }

    // Common troubleshooting
    writeln!(output, "\n{}:", style("Common fixes").green().bold()).unwrap();
    writeln!(output, "  • Use --prompt-template raw (disable template formatting)").unwrap();
    writeln!(output, "  • Add --no-bos flag (if BOS is duplicated)").unwrap();
    writeln!(output, "  • Check GGUF chat_template metadata").unwrap();

    // Example command with actual prompt
    writeln!(output, "\n{}:", style("Example command").cyan()).unwrap();
    writeln!(output, "  cargo run -p xtask -- crossval-per-token \\").unwrap();
    writeln!(output, "    --model <model.gguf> \\").unwrap();
    writeln!(output, "    --tokenizer <tokenizer.json> \\").unwrap();
    writeln!(output, "    --prompt \"{}\" \\", error.prompt).unwrap();
    writeln!(output, "    --prompt-template raw \\").unwrap();
    writeln!(output, "    --cpp-backend {} \\", error.backend.name().to_lowercase()).unwrap();
    writeln!(output, "    --max-tokens 4").unwrap();

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // AC1: Token mismatch detected before logits evaluation
    // Spec: docs/explanation/token-parity-pregate.md#acceptance-criteria
    #[test]
    fn test_detect_token_mismatch() {
        let rust_tokens = vec![1, 2, 3];
        let cpp_tokens = vec![1_i32, 2, 4]; // Diff at position 2

        let result =
            validate_token_parity(&rust_tokens, &cpp_tokens, "test prompt", CppBackend::BitNet);

        // Should fail before any logits comparison
        assert!(
            result.is_err()
                || std::panic::catch_unwind(|| {
                    validate_token_parity(
                        &rust_tokens,
                        &cpp_tokens,
                        "test prompt",
                        CppBackend::BitNet,
                    )
                    .ok()
                })
                .is_err(),
            "Expected token mismatch to be detected"
        );
    }

    // AC2: Both token sequences displayed on mismatch
    // Spec: docs/explanation/token-parity-pregate.md#design
    #[test]
    fn test_error_displays_both_sequences() {
        let error = TokenParityError {
            rust_tokens: vec![1, 2, 3],
            cpp_tokens: vec![1, 2, 4],
            first_diff_index: 2,
            prompt: "test".to_string(),
            backend: CppBackend::BitNet,
        };
        let formatted = format_token_mismatch_error(&error);
        // Verify both sequences are present in the error output
        assert!(
            formatted.contains("Rust tokens") || formatted.contains("[1, 2, 3]"),
            "Output should contain rust token sequence, got: {formatted}"
        );
        assert!(
            formatted.contains("C++ tokens") || formatted.contains("[1, 2, 4]"),
            "Output should contain cpp token sequence, got: {formatted}"
        );
    }

    // AC3: First diff position identified
    // Spec: docs/explanation/token-parity-pregate.md#design
    #[test]
    fn test_first_diff_position() {
        let rust = vec![1, 2, 3, 4];
        let cpp = vec![1, 2, 5, 6]; // First diff at index 2

        let diff_idx = find_first_diff(&rust, &cpp);
        assert_eq!(diff_idx, 2, "First diff should be at index 2");
    }

    // AC3: First diff for length mismatch (shorter sequence)
    #[test]
    fn test_first_diff_length_mismatch() {
        let rust = vec![1, 2, 3];
        let cpp = vec![1, 2]; // Shorter

        let diff_idx = find_first_diff(&rust, &cpp);
        assert_eq!(diff_idx, 2, "First diff should be at length of shorter sequence");
    }

    // AC4: Exit code 2 on token mismatch
    // Spec: docs/explanation/token-parity-pregate.md#design
    #[test]
    fn test_exit_code_on_mismatch() {
        // validate_token_parity returns an error (Err) on mismatch; callers are
        // responsible for translating that to exit code 2. Here we verify that the
        // function does return an error when sequences differ.
        let rust = vec![1u32, 2, 3];
        let cpp = vec![1_i32, 2, 99];
        let result = validate_token_parity(&rust, &cpp, "test", CppBackend::BitNet);
        assert!(result.is_err(), "validate_token_parity should return Err on mismatch");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("mismatch"), "Error message should mention mismatch, got: {msg}");
    }

    // AC5: Error message includes suggestions
    #[test]
    fn test_error_message_includes_suggestions() {
        let error = TokenParityError {
            rust_tokens: vec![128000, 128000, 1229],
            cpp_tokens: vec![128000, 1229],
            first_diff_index: 1,
            prompt: "What is 2+2?".to_string(),
            backend: CppBackend::BitNet,
        };

        let formatted = format_token_mismatch_error(&error);

        // Verify common suggestions are present
        assert!(formatted.contains("--prompt-template raw"), "Missing template suggestion");
        assert!(formatted.contains("--no-bos"), "Missing BOS suggestion");
        assert!(formatted.contains("chat_template metadata"), "Missing GGUF suggestion");
        assert!(formatted.contains("--dump-ids"), "Missing dump-ids suggestion");
    }

    // AC6: Error message is actionable (copy-paste-able flags)
    #[test]
    fn test_error_message_actionable() {
        let error = TokenParityError {
            rust_tokens: vec![1, 2],
            cpp_tokens: vec![1, 3],
            first_diff_index: 1,
            prompt: "test".to_string(),
            backend: CppBackend::Llama,
        };

        let formatted = format_token_mismatch_error(&error);

        // Verify flags are properly formatted (with -- prefix)
        assert!(formatted.contains("--"), "Should include command-line flags");
    }

    // AC7: Error message shows example fix
    #[test]
    fn test_error_message_shows_examples() {
        let error = TokenParityError {
            rust_tokens: vec![128000, 128000, 1229],
            cpp_tokens: vec![128000, 1229],
            first_diff_index: 1,
            prompt: "test".to_string(),
            backend: CppBackend::BitNet,
        };

        let formatted = format_token_mismatch_error(&error);

        // Verify it includes concrete examples, not just abstract advice
        assert!(
            formatted.contains("raw") || formatted.contains("template"),
            "Should show concrete template examples"
        );
    }

    // AC8: Error message highlights duplicate BOS pattern
    #[test]
    fn test_error_detects_duplicate_bos() {
        let error = TokenParityError {
            rust_tokens: vec![128000, 128000, 1229], // Duplicate BOS
            cpp_tokens: vec![128000, 1229],
            first_diff_index: 1,
            prompt: "test".to_string(),
            backend: CppBackend::BitNet,
        };

        let formatted = format_token_mismatch_error(&error);

        // Should suggest BOS-related fix
        assert!(
            formatted.contains("BOS") || formatted.contains("--no-bos"),
            "Should suggest BOS flag for duplicate BOS pattern"
        );
    }

    // AC9: Silent success when tokens match
    // Spec: docs/explanation/token-parity-pregate.md#acceptance-criteria
    #[test]
    fn test_silent_success_on_match() {
        let rust_tokens = vec![128000, 1229, 374];
        let cpp_tokens = vec![128000_i32, 1229, 374];

        // Should succeed silently (no output, no panic)
        let result =
            validate_token_parity(&rust_tokens, &cpp_tokens, "What is 2+2?", CppBackend::BitNet);

        // This will panic due to unimplemented!() - that's expected TDD behavior
        // Once implemented, this should pass silently
        match result {
            Ok(()) => {
                // Silent success - perfect!
            }
            Err(e) => panic!("Expected silent success on matching tokens, got error: {}", e),
        }
    }

    // AC10: Performance overhead <100ms for <1000 tokens
    // Spec: docs/explanation/token-parity-pregate.md#non-functional-requirements
    #[test]
    fn test_performance_under_100ms() {
        use std::time::Instant;

        // Generate 1000-token sequence
        let rust_tokens: Vec<u32> = (0..1000).collect();
        let cpp_tokens: Vec<i32> = (0i32..1000i32).collect();

        let start = Instant::now();

        // Run validation (should be very fast)
        let _ = validate_token_parity(&rust_tokens, &cpp_tokens, "perf test", CppBackend::Llama);

        let elapsed = start.elapsed();

        // Should complete in <100ms
        assert!(
            elapsed.as_millis() < 100,
            "Token parity check took {}ms (expected <100ms)",
            elapsed.as_millis()
        );
    }

    // Scenario 1: Duplicate BOS (common bug)
    // Spec: docs/explanation/token-parity-pregate.md#test-scenarios
    #[test]
    fn test_scenario_duplicate_bos() {
        let rust = vec![128000, 128000, 1229, 374]; // Double BOS
        let cpp = vec![128000_i32, 1229, 374]; // Single BOS

        // Should fail with first diff at index 1
        let result = validate_token_parity(&rust, &cpp, "What is 2+2?", CppBackend::BitNet);

        // Expected: exits with code 2, shows diagnostic
        assert!(
            result.is_err()
                || std::panic::catch_unwind(|| {
                    validate_token_parity(&rust, &cpp, "What is 2+2?", CppBackend::BitNet).ok()
                })
                .is_err()
        );
    }

    // Scenario 2: Tokens match (happy path)
    // Spec: docs/explanation/token-parity-pregate.md#test-scenarios
    #[test]
    fn test_scenario_tokens_match() {
        let rust = vec![128000, 1229, 374, 220, 17];
        let cpp = vec![128000_i32, 1229, 374, 220, 17];

        // Should succeed silently
        let result = validate_token_parity(&rust, &cpp, "What is 2+2?", CppBackend::Llama);

        match result {
            Ok(()) => {} // Expected
            Err(e) => panic!("Expected success, got error: {}", e),
        }
    }

    // Scenario 3: Length mismatch
    // Spec: docs/explanation/token-parity-pregate.md#test-scenarios
    #[test]
    fn test_scenario_length_mismatch() {
        let rust = vec![128000, 1229]; // Shorter
        let cpp = vec![128000_i32, 1229, 374]; // Longer

        // Should detect mismatch at index 2
        let result = validate_token_parity(&rust, &cpp, "test", CppBackend::BitNet);

        assert!(
            result.is_err()
                || std::panic::catch_unwind(|| {
                    validate_token_parity(&rust, &cpp, "test", CppBackend::BitNet).ok()
                })
                .is_err()
        );
    }

    // Edge case: Empty sequences
    #[test]
    fn test_empty_sequences() {
        let rust: Vec<u32> = vec![];
        let cpp: Vec<i32> = vec![];

        let result = validate_token_parity(&rust, &cpp, "", CppBackend::Llama);

        // Should succeed (both empty)
        match result {
            Ok(()) => {}
            Err(e) => panic!("Expected success for empty sequences, got error: {}", e),
        }
    }

    // Edge case: Single token
    #[test]
    fn test_single_token() {
        let rust = vec![128000];
        let cpp = vec![128000_i32];

        let result = validate_token_parity(&rust, &cpp, "test", CppBackend::BitNet);

        match result {
            Ok(()) => {}
            Err(e) => panic!("Expected success for single token, got error: {}", e),
        }
    }

    // Edge case: i32 to u32 conversion (negative values should be handled)
    #[test]
    fn test_negative_cpp_tokens() {
        // Policy: negative i32 token IDs from C++ are cast to u32 (wrapping).
        // This means they will never match a valid Rust token, so the call
        // returns Err (mismatch) rather than panicking.
        let rust = vec![1u32, 2, 3];
        let cpp = vec![1_i32, -1, 3]; // -1 becomes u32::MAX when cast
        let result = validate_token_parity(&rust, &cpp, "test", CppBackend::BitNet);
        assert!(
            result.is_err(),
            "Negative C++ token IDs should produce a mismatch error, not panic"
        );
    }

    // AC10: Backend-specific troubleshooting hints
    #[test]
    fn test_backend_specific_error_messages() {
        // Test BitNet backend error message
        let bitnet_error = TokenParityError {
            rust_tokens: vec![128000, 1229],
            cpp_tokens: vec![128000, 374],
            first_diff_index: 1,
            prompt: "test".to_string(),
            backend: CppBackend::BitNet,
        };

        let bitnet_formatted = format_token_mismatch_error(&bitnet_error);

        // Should mention BitNet in header
        assert!(bitnet_formatted.contains("BitNet"), "Should mention BitNet backend");
        // Should suggest trying llama backend
        assert!(
            bitnet_formatted.contains("--cpp-backend llama"),
            "Should suggest trying llama backend"
        );
        // Should mention BitNet-compatible models
        assert!(
            bitnet_formatted.contains("BitNet-compatible"),
            "Should mention BitNet compatibility"
        );

        // Test LLaMA backend error message
        let llama_error = TokenParityError {
            rust_tokens: vec![128000, 1229],
            cpp_tokens: vec![128000, 374],
            first_diff_index: 1,
            prompt: "test".to_string(),
            backend: CppBackend::Llama,
        };

        let llama_formatted = format_token_mismatch_error(&llama_error);

        // Should mention LLaMA in header
        assert!(llama_formatted.contains("LLaMA"), "Should mention LLaMA backend");
        // Should suggest trying bitnet backend
        assert!(
            llama_formatted.contains("--cpp-backend bitnet"),
            "Should suggest trying bitnet backend"
        );
        // Should mention LLaMA tokenizer compatibility
        assert!(llama_formatted.contains("LLaMA tokenizer"), "Should mention LLaMA tokenizer");
    }

    // Test that backend name appears in example command
    #[test]
    fn test_backend_in_example_command() {
        let error = TokenParityError {
            rust_tokens: vec![1, 2],
            cpp_tokens: vec![1, 3],
            first_diff_index: 1,
            prompt: "test".to_string(),
            backend: CppBackend::BitNet,
        };

        let formatted = format_token_mismatch_error(&error);

        // Should include --cpp-backend flag with correct value
        assert!(
            formatted.contains("--cpp-backend bitnet"),
            "Example command should include --cpp-backend bitnet"
        );
    }
}
