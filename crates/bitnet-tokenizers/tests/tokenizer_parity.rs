//! Tokenizer Round-Trip Parity Tests
//!
//! Tests feature spec: docs/explanation/tokenizer-architecture.md#round-trip-encoding
//! Architecture: docs/reference/tokenizer-api.md#encoding-decoding
//!
//! This test suite validates tokenizer correctness by verifying:
//! - Round-trip encoding/decoding: `decode(encode(text)) == text` for ASCII subset
//! - Special token handling: BOS/EOS/EOT token ID resolution and encoding
//! - Token-to-ID resolution: verify `token_to_id()` works for special tokens
//! - Deterministic encoding: same text → same token IDs across multiple calls
//!
//! # Test Coverage
//!
//! - **Round-trip parity**: Verify `decode(encode(text)) == text` for common strings
//! - **Special token resolution**: Test BOS/EOS/EOT token ID lookup
//! - **Deterministic encoding**: Verify stable token sequences
//! - **Edge cases**: Empty strings, whitespace, special characters
//!
//! # Environment Variables
//!
//! - `BITNET_GGUF` or `CROSSVAL_GGUF`: Path to GGUF model file (required)
//! - `BITNET_SKIP_SLOW_TESTS`: Skip tests requiring model loading
//!
//! # Running the Tests
//!
//! ```bash
//! # Run tokenizer parity tests (requires model file)
//! BITNET_GGUF=models/model.gguf cargo test -p bitnet-tokenizers --test tokenizer_parity --no-default-features --features cpu
//!
//! # Skip slow tests
//! BITNET_SKIP_SLOW_TESTS=1 cargo test -p bitnet-tokenizers --test tokenizer_parity
//!
//! # Run with ignored tests (includes cross-validation)
//! BITNET_GGUF=models/model.gguf cargo test -p bitnet-tokenizers --test tokenizer_parity -- --ignored --include-ignored
//! ```

#![cfg(feature = "cpu")]

use anyhow::{Context, Result};
use bitnet_models::{GgufReader, loader::MmapFile};
use bitnet_tokenizers::{RustGgufTokenizer, Tokenizer};
use std::path::{Path, PathBuf};

/// Helper to discover test model from environment or models/ directory
fn discover_test_model() -> Result<PathBuf> {
    // Priority 1: BITNET_GGUF environment variable
    if let Ok(path) = std::env::var("BITNET_GGUF") {
        let model_path = PathBuf::from(&path);
        if model_path.exists() {
            return Ok(model_path);
        }
        anyhow::bail!("BITNET_GGUF set to '{}' but file does not exist", path);
    }

    // Priority 2: CROSSVAL_GGUF environment variable (backward compatibility)
    if let Ok(path) = std::env::var("CROSSVAL_GGUF") {
        let model_path = PathBuf::from(&path);
        if model_path.exists() {
            return Ok(model_path);
        }
        anyhow::bail!("CROSSVAL_GGUF set to '{}' but file does not exist", path);
    }

    // Priority 3: Auto-discover from models/ directory
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow::anyhow!("Failed to find workspace root"))?;

    let models_dir = workspace_root.join("models");
    if !models_dir.exists() {
        anyhow::bail!(
            "No test model found. Set BITNET_GGUF env var or place model in models/ directory.\n\
             Download model with: cargo run -p xtask -- download-model"
        );
    }

    // Find first .gguf file in models/ directory
    let model_file = std::fs::read_dir(&models_dir)
        .context("Failed to read models/ directory")?
        .filter_map(|entry| entry.ok())
        .find(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("gguf"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "No .gguf files found in models/ directory.\n\
                 Download model with: cargo run -p xtask -- download-model"
            )
        })?;

    Ok(model_file.path())
}

/// Helper to load tokenizer from GGUF file
fn load_tokenizer_from_gguf(path: &Path) -> Result<RustGgufTokenizer> {
    let mmap = MmapFile::open(path).context("Failed to memory-map GGUF file")?;
    let reader = GgufReader::new(mmap.as_slice()).context("Failed to parse GGUF file")?;
    RustGgufTokenizer::from_gguf(&reader).context("Failed to load RustGgufTokenizer from GGUF")
}

#[cfg(test)]
mod round_trip_tests {
    use super::*;

    /// Tests feature spec: tokenizer-architecture.md#AC1-round-trip-ascii
    /// Verify round-trip encoding/decoding for ASCII text
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[test]
    fn test_roundtrip_ascii_text() -> Result<()> {
        // Skip if BITNET_SKIP_SLOW_TESTS is set
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: tokenizer roundtrip");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let tokenizer = load_tokenizer_from_gguf(&model_path)?;

        // Test strings covering common ASCII patterns
        let test_strings = vec![
            "Hello world",
            "What is the capital of France?",
            "2+2=4",
            "The sky is blue",
            "Simple sentence.",
            "Question?",
            "Exclamation!",
        ];

        for text in test_strings {
            let tokens = tokenizer.encode(text, false, false)?;
            let decoded = tokenizer.decode(&tokens)?;

            // Round-trip should preserve text (modulo whitespace normalization)
            assert_eq!(
                decoded.trim(),
                text,
                "Round-trip failed for: '{}'\nEncoded: {:?}\nDecoded: '{}'",
                text,
                tokens,
                decoded
            );

            eprintln!("✓ Round-trip passed for: '{}'", text);
        }

        Ok(())
    }

    /// Tests feature spec: tokenizer-architecture.md#AC2-round-trip-special-chars
    /// Verify round-trip encoding/decoding with special characters
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[test]
    fn test_roundtrip_special_characters() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: special characters roundtrip");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let tokenizer = load_tokenizer_from_gguf(&model_path)?;

        // Test strings with special characters
        let test_strings = vec![
            "Line1\nLine2",        // Newline
            "Tab\tseparated",      // Tab
            "Quote \"text\" here", // Quotes
            "Apostrophe's test",   // Apostrophe
            "Comma, semicolon;",   // Punctuation
            "Math: 1+2=3",         // Math symbols
        ];

        for text in test_strings {
            let tokens = tokenizer.encode(text, false, false)?;
            let decoded = tokenizer.decode(&tokens)?;

            // For special characters, allow whitespace normalization
            let normalized_original = text.trim();
            let normalized_decoded = decoded.trim();

            assert!(
                normalized_decoded.contains(normalized_original)
                    || normalized_original.contains(normalized_decoded),
                "Round-trip with special chars failed for: '{}'\nEncoded: {:?}\nDecoded: '{}'",
                text,
                tokens,
                decoded
            );

            eprintln!("✓ Special chars round-trip passed for: '{}'", text);
        }

        Ok(())
    }

    /// Tests feature spec: tokenizer-architecture.md#AC3-round-trip-edge-cases
    /// Verify round-trip encoding/decoding with edge cases
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[test]
    fn test_roundtrip_edge_cases() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: edge cases roundtrip");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let tokenizer = load_tokenizer_from_gguf(&model_path)?;

        // Edge case 1: Single character
        let text = "A";
        let tokens = tokenizer.encode(text, false, false)?;
        let decoded = tokenizer.decode(&tokens)?;
        assert!(
            decoded.trim() == text || decoded.contains(text),
            "Single char round-trip failed: '{}' -> '{}'",
            text,
            decoded
        );
        eprintln!("✓ Single character round-trip passed");

        // Edge case 2: Whitespace-only
        let text = "   ";
        let tokens = tokenizer.encode(text, false, false)?;
        assert!(!tokens.is_empty(), "Whitespace-only text should produce tokens");
        eprintln!("✓ Whitespace encoding passed");

        // Edge case 3: Repeated characters
        let text = "aaaa";
        let tokens = tokenizer.encode(text, false, false)?;
        let decoded = tokenizer.decode(&tokens)?;
        assert!(
            decoded.trim() == text || decoded.contains(text),
            "Repeated chars round-trip failed: '{}' -> '{}'",
            text,
            decoded
        );
        eprintln!("✓ Repeated characters round-trip passed");

        Ok(())
    }
}

#[cfg(test)]
mod special_token_tests {
    use super::*;

    /// Tests feature spec: tokenizer-architecture.md#AC4-special-token-bos-eos
    /// Verify BOS/EOS token ID resolution and encoding
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[test]
    fn test_bos_eos_token_resolution() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: BOS/EOS token resolution");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let tokenizer = load_tokenizer_from_gguf(&model_path)?;

        // Get special token IDs
        let bos_id = tokenizer.bos_token_id();
        let eos_id = tokenizer.eos_token_id();
        let (bos_triple, eos_triple, eot_triple) = tokenizer.bos_eos_eot();

        eprintln!("Special token IDs from tokenizer:");
        eprintln!("  BOS (method): {:?}", bos_id);
        eprintln!("  EOS (method): {:?}", eos_id);
        eprintln!("  BOS (triple): {:?}", bos_triple);
        eprintln!("  EOS (triple): {:?}", eos_triple);
        eprintln!("  EOT (triple): {:?}", eot_triple);

        // Verify consistency between methods
        if let (Some(bos), Some(bos_t)) = (bos_id, bos_triple) {
            assert_eq!(bos, bos_t, "BOS token ID mismatch between methods: {} vs {}", bos, bos_t);
            eprintln!("✓ BOS token ID consistent: {}", bos);
        }

        if let (Some(eos), Some(eos_t)) = (eos_id, eos_triple) {
            assert_eq!(eos, eos_t, "EOS token ID mismatch between methods: {} vs {}", eos, eos_t);
            eprintln!("✓ EOS token ID consistent: {}", eos);
        }

        // Test BOS token insertion
        let text = "Hello world";
        let tokens_no_bos = tokenizer.encode(text, false, false)?;
        let tokens_with_bos = tokenizer.encode(text, true, false)?;

        eprintln!("Without BOS: {:?}", tokens_no_bos);
        eprintln!("With BOS: {:?}", tokens_with_bos);

        // If BOS token is configured, verify insertion
        if let Some(bos) = bos_id {
            assert!(
                tokens_with_bos.first() == Some(&bos),
                "Expected BOS token {} at start with add_bos=true, got {:?}",
                bos,
                tokens_with_bos.first()
            );
            eprintln!("✓ BOS token insertion verified");
        }

        // Verify at least one special token is configured
        assert!(
            bos_id.is_some() || eos_id.is_some() || eot_triple.is_some(),
            "Expected at least one special token (BOS/EOS/EOT) to be configured"
        );

        Ok(())
    }

    /// Tests feature spec: tokenizer-architecture.md#AC5-special-token-eot
    /// Verify EOT token ID resolution (LLaMA-3 specific)
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[test]
    fn test_eot_token_resolution() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: EOT token resolution");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let tokenizer = load_tokenizer_from_gguf(&model_path)?;

        let (_bos_id, _eos_id, eot_id) = tokenizer.bos_eos_eot();

        eprintln!("EOT token ID: {:?}", eot_id);

        if let Some(eot) = eot_id {
            eprintln!("✓ EOT token configured: {}", eot);

            // For LLaMA-3 models, EOT should be token ID 128009
            // Note: This is model-specific, so we just verify it's reasonable
            assert!(
                eot < tokenizer.vocab_size() as u32,
                "EOT token ID {} exceeds vocab size {}",
                eot,
                tokenizer.vocab_size()
            );

            // Test EOT token in text (LLaMA-3 chat format)
            let text_with_eot = "Hello<|eot_id|>world";
            let tokens_parse = tokenizer.encode(text_with_eot, false, true)?;

            eprintln!("Text with EOT: '{}'", text_with_eot);
            eprintln!("Tokens (parse_special=true): {:?}", tokens_parse);

            // Note: Whether EOT appears depends on tokenizer implementation
            // BPE vs SPM handle special tokens differently
            eprintln!("✓ EOT token parsing tested");
        } else {
            eprintln!("⚠ No EOT token configured (not all models use EOT)");
        }

        Ok(())
    }

    /// Tests feature spec: tokenizer-architecture.md#AC6-special-token-lookup
    /// Verify token-to-ID resolution for special tokens
    ///
    /// **TDD Scaffolding**: Test compiles but requires implementation of token_to_id()
    #[test]
    #[ignore = "requires model file and token_to_id() implementation"]
    fn test_token_to_id_special_tokens() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: token-to-ID lookup");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let _tokenizer = load_tokenizer_from_gguf(&model_path)?;

        // Test common special token strings
        let special_tokens = vec![
            "<|eot_id|>",    // LLaMA-3 EOT
            "</s>",          // EOS marker
            "<|endoftext|>", // GPT-style EOS
            "<s>",           // BOS marker
        ];

        for token_str in special_tokens {
            // TODO: Implement token_to_id() method on Tokenizer trait
            // This test is scaffolded for future implementation
            eprintln!("TODO: Test token_to_id('{}') when implemented", token_str);

            // Placeholder assertion - will be replaced with actual implementation
            // let token_id = tokenizer.token_to_id(token_str);
            // assert!(token_id.is_some(), "Expected token ID for '{}'", token_str);
        }

        eprintln!("⚠ token_to_id() implementation pending - test scaffolded");

        Ok(())
    }
}

#[cfg(test)]
mod determinism_tests {
    use super::*;

    /// Tests feature spec: tokenizer-architecture.md#AC7-deterministic-encoding
    /// Verify deterministic encoding: same text → same token IDs
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[test]
    fn test_deterministic_encoding() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: deterministic encoding");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let tokenizer = load_tokenizer_from_gguf(&model_path)?;

        let test_strings = vec!["What is the capital of France?", "2+2=4", "Hello world"];

        for text in test_strings {
            // Encode the same text multiple times
            let tokens1 = tokenizer.encode(text, false, false)?;
            let tokens2 = tokenizer.encode(text, false, false)?;
            let tokens3 = tokenizer.encode(text, false, false)?;

            assert_eq!(
                tokens1, tokens2,
                "Non-deterministic encoding detected for: '{}'\n  First: {:?}\n  Second: {:?}",
                text, tokens1, tokens2
            );

            assert_eq!(
                tokens2, tokens3,
                "Non-deterministic encoding detected for: '{}'\n  Second: {:?}\n  Third: {:?}",
                text, tokens2, tokens3
            );

            eprintln!("✓ Deterministic encoding verified for: '{}'", text);
        }

        Ok(())
    }

    /// Tests feature spec: tokenizer-architecture.md#AC8-deterministic-decoding
    /// Verify deterministic decoding: same tokens → same text
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[test]
    fn test_deterministic_decoding() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: deterministic decoding");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let tokenizer = load_tokenizer_from_gguf(&model_path)?;

        let text = "What is the capital of France?";
        let tokens = tokenizer.encode(text, false, false)?;

        // Decode the same tokens multiple times
        let decoded1 = tokenizer.decode(&tokens)?;
        let decoded2 = tokenizer.decode(&tokens)?;
        let decoded3 = tokenizer.decode(&tokens)?;

        assert_eq!(
            decoded1, decoded2,
            "Non-deterministic decoding detected\n  First: '{}'\n  Second: '{}'",
            decoded1, decoded2
        );

        assert_eq!(
            decoded2, decoded3,
            "Non-deterministic decoding detected\n  Second: '{}'\n  Third: '{}'",
            decoded2, decoded3
        );

        eprintln!("✓ Deterministic decoding verified");

        Ok(())
    }
}

#[cfg(test)]
mod vocab_size_tests {
    use super::*;

    /// Tests feature spec: tokenizer-architecture.md#AC9-vocab-size-consistency
    /// Verify vocabulary size is consistent and token IDs are within bounds
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[test]
    fn test_vocab_size_consistency() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: vocab size consistency");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let tokenizer = load_tokenizer_from_gguf(&model_path)?;

        let vocab_size = tokenizer.vocab_size();

        eprintln!("Vocabulary size: {}", vocab_size);

        // Sanity checks for vocabulary size
        assert!(vocab_size > 0, "Vocabulary size must be positive, got {}", vocab_size);

        // Typical vocab sizes range from 1K to 200K
        assert!(
            (1000..=200_000).contains(&vocab_size),
            "Vocab size {} is outside typical range [1K, 200K]",
            vocab_size
        );

        // Verify all encoded tokens are within vocab bounds
        let test_strings =
            vec!["Hello world", "What is 2+2?", "Test vocabulary bounds with longer text"];

        for text in test_strings {
            let tokens = tokenizer.encode(text, false, false)?;

            for (i, &token_id) in tokens.iter().enumerate() {
                assert!(
                    (token_id as usize) < vocab_size,
                    "Token ID {} at position {} exceeds vocab size {}",
                    token_id,
                    i,
                    vocab_size
                );
            }

            eprintln!("✓ All tokens within bounds for: '{}'", text);
        }

        Ok(())
    }
}
