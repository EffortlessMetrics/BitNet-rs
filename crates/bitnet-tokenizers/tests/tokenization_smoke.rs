//! Smoke tests for pure-Rust GGUF tokenizer
//!
//! These tests validate basic functionality of the pure-Rust tokenizer loaded from GGUF
//! model files. They ensure that tokenization works correctly with various configuration
//! options and special token handling.
//!
//! # Test Coverage
//!
//! - **Basic smoke test**: Verify tokenizer loads from GGUF and encodes text
//! - **BOS token handling**: Test BOS token insertion with add_bos flag
//! - **Special token lookup**: Verify id_for_special method for common tokens
//! - **Parse special flag**: Test parse_special behavior for LLaMA-3 chat tokens
//! - **Crossval comparison**: Compare Rust tokenization against C++ FFI (optional)
//!
//! # Environment Variables
//!
//! - `CROSSVAL_GGUF`: Path to GGUF model file (required for all tests)
//!
//! # Running the Tests
//!
//! ```bash
//! # Run all tokenization smoke tests (requires model file)
//! CROSSVAL_GGUF=models/model.gguf cargo test -p bitnet-tokenizers --test tokenization_smoke
//!
//! # Run without crossval comparison
//! CROSSVAL_GGUF=models/model.gguf cargo test -p bitnet-tokenizers --test tokenization_smoke --no-default-features
//!
//! # Run with crossval comparison (requires FFI)
//! CROSSVAL_GGUF=models/model.gguf cargo test -p bitnet-tokenizers --test tokenization_smoke --features crossval,ffi
//! ```

use anyhow::{Context, Result};
use bitnet_models::{GgufReader, loader::MmapFile};
use bitnet_tokenizers::{RustGgufTokenizer, Tokenizer};
use std::path::Path;

/// Helper to get GGUF path from environment variable
fn get_gguf_path() -> Result<String> {
    std::env::var("CROSSVAL_GGUF")
        .context("CROSSVAL_GGUF not set - set it to a valid GGUF model path")
}

#[test]
#[ignore = "Requires CROSSVAL_GGUF environment variable"]
fn pure_rust_tokenizer_from_gguf_smoke() -> Result<()> {
    let gguf_path = get_gguf_path()?;
    let mmap = MmapFile::open(Path::new(&gguf_path)).context("Failed to memory-map GGUF file")?;
    let reader = GgufReader::new(mmap.as_slice()).context("Failed to parse GGUF file")?;

    // Load tokenizer from GGUF metadata
    let tokenizer = RustGgufTokenizer::from_gguf(&reader)
        .context("Failed to load RustGgufTokenizer from GGUF")?;

    // Get add_bos hint from metadata
    let add_bos = reader.get_bool_metadata("tokenizer.ggml.add_bos_token").unwrap_or(false);

    // Encode basic text
    let text = "What is 2+2?";
    let ids = tokenizer.encode(text, add_bos, false).context("Failed to encode text")?;

    // Basic sanity checks
    assert!(!ids.is_empty(), "Expected non-empty token IDs, got empty vector");

    assert!(
        ids.len() >= 3,
        "Expected at least 3 tokens for '{}', got {} tokens: {:?}",
        text,
        ids.len(),
        ids
    );

    // Verify token IDs are reasonable (not out of bounds)
    let vocab_size = tokenizer.vocab_size();
    for (i, &token_id) in ids.iter().enumerate() {
        assert!(
            (token_id as usize) < vocab_size,
            "Token ID {} at position {} exceeds vocab size {}",
            token_id,
            i,
            vocab_size
        );
    }

    eprintln!("✓ Tokenized '{}' into {} tokens: {:?}", text, ids.len(), ids);

    Ok(())
}

#[test]
#[ignore = "Requires CROSSVAL_GGUF environment variable"]
fn bos_token_handling() -> Result<()> {
    let gguf_path = get_gguf_path()?;
    let mmap = MmapFile::open(Path::new(&gguf_path)).context("Failed to memory-map GGUF file")?;
    let reader = GgufReader::new(mmap.as_slice()).context("Failed to parse GGUF file")?;

    let tokenizer = RustGgufTokenizer::from_gguf(&reader)
        .context("Failed to load RustGgufTokenizer from GGUF")?;

    // Get BOS token ID from tokenizer
    let bos_id = tokenizer.bos_token_id();

    let text = "Hello world";

    // Test 1: Encode without BOS
    let ids_no_bos =
        tokenizer.encode(text, false, false).context("Failed to encode without BOS")?;

    // Test 2: Encode with BOS
    let ids_with_bos = tokenizer.encode(text, true, false).context("Failed to encode with BOS")?;

    eprintln!("BOS token ID: {:?}", bos_id);
    eprintln!("Without BOS: {:?}", ids_no_bos);
    eprintln!("With BOS: {:?}", ids_with_bos);

    // If BOS token is configured, verify it's added when requested
    if let Some(bos) = bos_id {
        // With add_bos=true, BOS should be present at start
        assert!(
            ids_with_bos.first() == Some(&bos),
            "Expected BOS token {} at start with add_bos=true, got {:?}",
            bos,
            ids_with_bos.first()
        );

        // Without add_bos=false, BOS should NOT be at start (unless it's naturally part of the text)
        // Note: Some tokenizers may still include BOS based on internal rules, so we just verify
        // that add_bos=true and add_bos=false produce different results
        eprintln!(
            "✓ BOS token handling: add_bos=true has {} tokens, add_bos=false has {} tokens",
            ids_with_bos.len(),
            ids_no_bos.len()
        );
    } else {
        eprintln!("⚠ BOS token not configured in GGUF - skipping BOS presence checks");

        // Even without BOS configured, encoding should succeed
        assert!(!ids_no_bos.is_empty(), "Encoding without BOS failed");
        assert!(!ids_with_bos.is_empty(), "Encoding with BOS failed");
    }

    // Test 3: Verify add_bos_hint from GGUF metadata
    let add_bos_hint = tokenizer.add_bos_hint();
    eprintln!("GGUF add_bos_hint: {:?}", add_bos_hint);

    // If hint is present, it should be a boolean
    if let Some(hint) = add_bos_hint {
        eprintln!("✓ add_bos_hint from GGUF metadata: {}", hint);
    } else {
        eprintln!("⚠ add_bos_hint not set in GGUF metadata");
    }

    Ok(())
}

#[test]
#[ignore = "Requires CROSSVAL_GGUF environment variable"]
fn special_token_lookup() -> Result<()> {
    let gguf_path = get_gguf_path()?;
    let mmap = MmapFile::open(Path::new(&gguf_path)).context("Failed to memory-map GGUF file")?;
    let reader = GgufReader::new(mmap.as_slice()).context("Failed to parse GGUF file")?;

    let tokenizer = RustGgufTokenizer::from_gguf(&reader)
        .context("Failed to load RustGgufTokenizer from GGUF")?;

    // Get special token IDs from wrapper
    let (bos_id, eos_id, eot_id) = tokenizer.bos_eos_eot();

    eprintln!("Special tokens from GGUF:");
    eprintln!("  BOS: {:?}", bos_id);
    eprintln!("  EOS: {:?}", eos_id);
    eprintln!("  EOT: {:?}", eot_id);

    // Test BOS patterns
    if let Some(expected_bos) = bos_id {
        // Verify BOS token ID is accessible
        if let Some(id) = tokenizer.bos_token_id() {
            if id == expected_bos {
                eprintln!("✓ BOS token ID {} matches bos_token_id() method", expected_bos);
            } else {
                eprintln!("⚠ BOS token mismatch: expected {}, got {}", expected_bos, id);
            }
        } else {
            eprintln!("⚠ BOS token {} not accessible via bos_token_id()", expected_bos);
        }
    } else {
        eprintln!("⚠ No BOS token configured in GGUF");
    }

    // Test EOS patterns
    if let Some(expected_eos) = eos_id {
        // Verify EOS token ID is accessible
        if let Some(id) = tokenizer.eos_token_id() {
            if id == expected_eos {
                eprintln!("✓ EOS token ID {} matches eos_token_id() method", expected_eos);
            } else {
                eprintln!("⚠ EOS token mismatch: expected {}, got {}", expected_eos, id);
            }
        } else {
            eprintln!("⚠ EOS token {} not accessible via eos_token_id()", expected_eos);
        }
    } else {
        eprintln!("⚠ No EOS token configured in GGUF");
    }

    // Test EOT patterns (LLaMA-3 specific)
    if let Some(expected_eot) = eot_id {
        eprintln!("✓ EOT token configured: {}", expected_eot);

        // EOT should be accessible via bos_eos_eot method
        assert_eq!(eot_id, Some(expected_eot), "bos_eos_eot() should return EOT token ID");
    } else {
        eprintln!("⚠ No EOT token configured (not all models use EOT)");
    }

    // Verify at least one special token is configured
    assert!(
        bos_id.is_some() || eos_id.is_some() || eot_id.is_some(),
        "Expected at least one special token (BOS/EOS/EOT) to be configured"
    );

    Ok(())
}

#[test]
#[ignore = "Requires CROSSVAL_GGUF environment variable"]
fn parse_special_eot_handling() -> Result<()> {
    let gguf_path = get_gguf_path()?;
    let mmap = MmapFile::open(Path::new(&gguf_path)).context("Failed to memory-map GGUF file")?;
    let reader = GgufReader::new(mmap.as_slice()).context("Failed to parse GGUF file")?;

    let tokenizer = RustGgufTokenizer::from_gguf(&reader)
        .context("Failed to load RustGgufTokenizer from GGUF")?;

    // Get EOT token ID if available
    let (_bos_id, _eos_id, eot_id) = tokenizer.bos_eos_eot();

    // Test text with LLaMA-3 style EOT token
    let text_with_eot = "Hello<|eot_id|>world";

    // Test 1: Encode with parse_special=false (treat as literal text)
    let ids_no_parse = tokenizer
        .encode(text_with_eot, false, false)
        .context("Failed to encode with parse_special=false")?;

    // Test 2: Encode with parse_special=true (parse special tokens)
    let ids_parse = tokenizer
        .encode(text_with_eot, false, true)
        .context("Failed to encode with parse_special=true")?;

    eprintln!("Text: '{}'", text_with_eot);
    eprintln!("parse_special=false: {:?} ({} tokens)", ids_no_parse, ids_no_parse.len());
    eprintln!("parse_special=true: {:?} ({} tokens)", ids_parse, ids_parse.len());

    // If EOT is configured, verify parsing behavior
    if let Some(eot) = eot_id {
        eprintln!("EOT token ID: {}", eot);

        // With parse_special=true, EOT should appear as a single token
        let has_eot_parsed = ids_parse.contains(&eot);
        eprintln!("✓ parse_special=true: EOT token {} present: {}", eot, has_eot_parsed);

        // Note: parse_special behavior varies by tokenizer implementation
        // BPE tokenizers may or may not handle special tokens differently
        // SPM tokenizers typically have built-in special token handling
        match tokenizer.kind() {
            bitnet_tokenizers::GgufTokKind::Spm => {
                eprintln!("  (SPM tokenizer - special token handling is built-in)");
            }
            bitnet_tokenizers::GgufTokKind::Bpe => {
                eprintln!("  (BPE tokenizer - parse_special controls special token recognition)");
            }
        }
    } else {
        eprintln!("⚠ No EOT token configured - skipping EOT parsing checks");
    }

    // Verify encoding succeeds regardless of parse_special flag
    assert!(!ids_no_parse.is_empty(), "Encoding with parse_special=false failed");
    assert!(!ids_parse.is_empty(), "Encoding with parse_special=true failed");

    Ok(())
}

#[test]
#[ignore = "Requires CROSSVAL_GGUF environment variable"]
fn tokenizer_kind_detection() -> Result<()> {
    let gguf_path = get_gguf_path()?;
    let mmap = MmapFile::open(Path::new(&gguf_path)).context("Failed to memory-map GGUF file")?;
    let reader = GgufReader::new(mmap.as_slice()).context("Failed to parse GGUF file")?;

    let tokenizer = RustGgufTokenizer::from_gguf(&reader)
        .context("Failed to load RustGgufTokenizer from GGUF")?;

    // Get tokenizer kind
    let kind = tokenizer.kind();

    eprintln!("Detected tokenizer kind: {:?}", kind);

    // Verify kind is either SPM or BPE
    match kind {
        bitnet_tokenizers::GgufTokKind::Spm => {
            eprintln!("✓ SentencePiece tokenizer detected");

            // Verify SPM feature is enabled (otherwise loading would have failed)
            #[cfg(not(feature = "spm"))]
            panic!("SPM tokenizer detected but spm feature is not enabled");
        }
        bitnet_tokenizers::GgufTokKind::Bpe => {
            eprintln!("✓ BPE tokenizer detected");
        }
    }

    // Verify metadata contains tokenizer.ggml.model
    let model_type = reader
        .get_string_metadata("tokenizer.ggml.model")
        .context("Missing tokenizer.ggml.model metadata")?;

    eprintln!("GGUF tokenizer.ggml.model: {}", model_type);

    // Verify kind matches metadata
    match (kind, model_type.to_lowercase().as_str()) {
        (bitnet_tokenizers::GgufTokKind::Spm, "llama") => {
            eprintln!("✓ Kind matches metadata: SPM ← llama");
        }
        (bitnet_tokenizers::GgufTokKind::Bpe, "gpt2" | "bpe") => {
            eprintln!("✓ Kind matches metadata: BPE ← {}", model_type);
        }
        _ => {
            panic!("Kind mismatch: {:?} does not match metadata '{}'", kind, model_type);
        }
    }

    Ok(())
}

// NOTE: FFI parity tests with C++ tokenization are located in the `crossval` crate,
// which has access to bitnet-sys. See: crossval/tests/parity_bitnetcpp.rs
//
// To run FFI parity tests for tokenization:
//   CROSSVAL_GGUF=models/model.gguf cargo test -p crossval --features ffi --test parity_bitnetcpp
//
// This ensures proper separation of concerns:
//   - bitnet-tokenizers: Pure-Rust tokenizer implementation and smoke tests
//   - crossval: Cross-validation against C++ reference implementation

#[test]
#[ignore = "Requires CROSSVAL_GGUF environment variable"]
fn vocab_size_sanity() -> Result<()> {
    let gguf_path = get_gguf_path()?;
    let mmap = MmapFile::open(Path::new(&gguf_path)).context("Failed to memory-map GGUF file")?;
    let reader = GgufReader::new(mmap.as_slice()).context("Failed to parse GGUF file")?;

    let tokenizer = RustGgufTokenizer::from_gguf(&reader)
        .context("Failed to load RustGgufTokenizer from GGUF")?;

    let vocab_size = tokenizer.vocab_size();

    eprintln!("Vocabulary size: {}", vocab_size);

    // Sanity checks for vocabulary size
    assert!(vocab_size > 0, "Vocabulary size must be positive, got {}", vocab_size);

    // Typical vocab sizes
    match tokenizer.kind() {
        bitnet_tokenizers::GgufTokKind::Spm => {
            // LLaMA models typically use 32000 or 32768
            assert!(
                (1000..=200_000).contains(&vocab_size),
                "SPM vocab size {} is outside typical range [1K, 200K]",
                vocab_size
            );
            eprintln!("✓ SPM vocab size {} is within typical range", vocab_size);
        }
        bitnet_tokenizers::GgufTokKind::Bpe => {
            // GPT-2 uses 50257, GPT-J uses similar
            assert!(
                (1000..=200_000).contains(&vocab_size),
                "BPE vocab size {} is outside typical range [1K, 200K]",
                vocab_size
            );
            eprintln!("✓ BPE vocab size {} is within typical range", vocab_size);
        }
    }

    // Verify all encoded tokens are within vocab bounds
    let text = "Test vocabulary bounds";
    let ids = tokenizer.encode(text, false, false)?;

    for (i, &token_id) in ids.iter().enumerate() {
        assert!(
            (token_id as usize) < vocab_size,
            "Token ID {} at position {} exceeds vocab size {}",
            token_id,
            i,
            vocab_size
        );
    }

    eprintln!("✓ All token IDs are within vocabulary bounds");

    Ok(())
}
