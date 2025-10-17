//! Golden token tests for tokenizer stability
//!
//! These tests validate that the pure-Rust GGUF tokenizer produces
//! identical outputs to reference tokenizations. This prevents drift
//! and ensures compatibility with llama.cpp and other implementations.
//!
//! To run these tests with a model:
//! ```bash
//! TOKENIZER_TEST_MODEL=/path/to/model.gguf cargo test --test golden_tokens_test
//! ```

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GoldenTokensFile {
    version: String,
    description: String,
    tokenizer_kind: String,
    test_cases: Vec<GoldenTokenCase>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GoldenTokenCase {
    text: String,
    add_bos: bool,
    parse_special: bool,
    expected_tokens: Vec<u32>,
    #[serde(default)]
    note: Option<String>,
}

/// Load golden token test cases for a specific tokenizer kind
fn load_golden_tokens_for_kind(kind: &str) -> Result<GoldenTokensFile> {
    let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");

    // Map tokenizer kind to fixture file
    let fixture_name = match kind.to_lowercase().as_str() {
        "gpt2" | "bpe" => "golden_tokens_gpt2.json",
        "llama" if kind.contains('3') => "golden_tokens_llama3.json",
        "llama" => "golden_tokens_llama.json",
        other => {
            // Try direct lookup first
            let direct_path = tests_dir.join(format!("golden_tokens_{}.json", other));
            if direct_path.exists() {
                return load_golden_tokens_from_path(&direct_path);
            }
            // Fall back to generic llama for unknown SPM variants
            eprintln!("WARNING: Unknown tokenizer kind '{}', falling back to llama fixture", other);
            "golden_tokens_llama.json"
        }
    };

    let json_path = tests_dir.join(fixture_name);
    load_golden_tokens_from_path(&json_path)
}

/// Load golden tokens from a specific path
fn load_golden_tokens_from_path(json_path: &PathBuf) -> Result<GoldenTokensFile> {
    let json_content = std::fs::read_to_string(json_path)
        .with_context(|| format!("Failed to read golden tokens file: {}", json_path.display()))?;

    let golden: GoldenTokensFile =
        serde_json::from_str(&json_content).context("Failed to parse golden tokens JSON")?;

    Ok(golden)
}

/// Load legacy unified golden tokens (deprecated - for backward compatibility)
#[allow(dead_code)]
fn load_legacy_golden_tokens() -> Result<GoldenTokensFile> {
    let json_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("golden_tokens.json");

    if !json_path.exists() {
        anyhow::bail!("Legacy golden_tokens.json not found (migrated to model-specific fixtures)");
    }

    // Convert legacy format to new format
    #[derive(Debug, Deserialize)]
    struct LegacyGoldenTokensFile {
        version: String,
        description: String,
        test_cases: Vec<LegacyGoldenTokenCase>,
    }

    #[derive(Debug, Deserialize)]
    struct LegacyGoldenTokenCase {
        tokenizer_kind: String,
        model_hint: String,
        text: String,
        add_bos: bool,
        parse_special: bool,
        expected_tokens: Vec<u32>,
    }

    let json_content = std::fs::read_to_string(&json_path)?;
    let legacy: LegacyGoldenTokensFile = serde_json::from_str(&json_content)?;

    // Return first matching kind (not ideal, but maintains compatibility)
    Ok(GoldenTokensFile {
        version: legacy.version,
        description: legacy.description,
        tokenizer_kind: "mixed".to_string(),
        test_cases: legacy
            .test_cases
            .into_iter()
            .map(|tc| GoldenTokenCase {
                text: tc.text,
                add_bos: tc.add_bos,
                parse_special: tc.parse_special,
                expected_tokens: tc.expected_tokens,
                note: Some(format!("{} ({})", tc.model_hint, tc.tokenizer_kind)),
            })
            .collect(),
    })
}

#[test]
fn test_golden_tokens_files_load() {
    // Verify all model-specific golden token files load correctly
    let test_kinds = vec!["gpt2", "llama", "llama3"];

    for kind in test_kinds {
        eprintln!("\nLoading golden tokens for kind: {}", kind);
        let golden = load_golden_tokens_for_kind(kind)
            .unwrap_or_else(|e| panic!("Failed to load golden tokens for {}: {}", kind, e));

        assert_eq!(golden.version, "1.0.0");
        assert_eq!(golden.tokenizer_kind, kind);
        assert!(
            !golden.test_cases.is_empty(),
            "Golden tokens file for {} should contain test cases",
            kind
        );

        eprintln!("  Loaded {} test cases for {}", golden.test_cases.len(), kind);
        for (i, case) in golden.test_cases.iter().enumerate() {
            eprintln!(
                "    Case {}: '{}', {} tokens",
                i + 1,
                case.text.chars().take(40).collect::<String>(),
                case.expected_tokens.len()
            );
            if let Some(note) = &case.note {
                eprintln!("      Note: {}", note);
            }
        }
    }
}

#[cfg(feature = "integration-tests")]
#[test]
fn test_golden_tokens_with_model() -> Result<()> {
    use bitnet_models::formats::gguf::GgufReader;
    use bitnet_models::loader::MmapFile;
    use bitnet_tokenizers::gguf_loader::RustTokenizer;

    // Check if test model is provided
    let model_path = match std::env::var("TOKENIZER_TEST_MODEL") {
        Ok(path) => PathBuf::from(path),
        Err(_) => {
            eprintln!("TOKENIZER_TEST_MODEL not set - skipping golden token validation");
            eprintln!("To run: TOKENIZER_TEST_MODEL=/path/to/model.gguf cargo test");
            return Ok(());
        }
    };

    if !model_path.exists() {
        eprintln!("Model not found at {:?} - skipping test", model_path);
        return Ok(());
    }

    // Load tokenizer from model
    let mmap = MmapFile::open(&model_path)?;
    let reader = GgufReader::new(mmap.as_slice())?;
    let tokenizer = RustTokenizer::from_gguf(&reader)?;

    // Detect tokenizer kind from model
    let model_kind =
        reader.get_string_metadata("tokenizer.ggml.model").unwrap_or_else(|| "unknown".to_string());

    eprintln!("Testing with model: {}", model_path.display());
    eprintln!("Model tokenizer kind: {}", model_kind);

    // Load golden tokens for this specific model kind
    let golden = load_golden_tokens_for_kind(&model_kind).with_context(|| {
        format!("Failed to load golden tokens for tokenizer kind '{}'", model_kind)
    })?;

    eprintln!("Loaded {} golden test cases for kind '{}'", golden.test_cases.len(), model_kind);

    // Run all test cases (they're all for this model kind now)
    let mut tests_run = 0;
    let mut tests_passed = 0;

    for (i, case) in golden.test_cases.iter().enumerate() {
        tests_run += 1;

        // Tokenize with Rust implementation
        let tokens = tokenizer.encode(&case.text, case.add_bos, case.parse_special)?;

        // Compare with golden tokens
        if tokens == case.expected_tokens {
            eprintln!("  Case {}: PASS ✓ \"{}\"", i + 1, case.text);
            tests_passed += 1;
        } else {
            eprintln!("  Case {}: FAIL ✗ \"{}\"", i + 1, case.text);
            eprintln!("    Expected: {:?}", case.expected_tokens);
            eprintln!("    Got:      {:?}", tokens);
            eprintln!("    Diff:     {} vs {} tokens", case.expected_tokens.len(), tokens.len());

            // Show first divergence
            for (j, (expected, actual)) in
                case.expected_tokens.iter().zip(tokens.iter()).enumerate()
            {
                if expected != actual {
                    eprintln!("    First divergence at position {}: {} vs {}", j, expected, actual);
                    break;
                }
            }

            panic!("Golden token test failed for case {}", i + 1);
        }
    }

    eprintln!("\nGolden token tests: {}/{} passed", tests_passed, tests_run);

    assert_eq!(tests_passed, tests_run, "All golden token tests must pass");
    assert!(tests_run > 0, "At least one golden token test should run for kind '{}'", model_kind);

    Ok(())
}

#[test]
fn test_bpe_bytelevel_prefix_space() {
    // This test documents the ByteLevel prefix_space behavior fix
    // Without add_prefix_space=true, the first token would be different
    // from subsequent tokens with the same word.

    // Example: "What" at the start vs " What" in the middle
    // With add_prefix_space=true:
    //   - "What is..." -> [3923, 318, ...]  ("What" includes virtual leading space)
    //   - " What is..." -> [3923, 318, ...]  (same, space is already there)
    //
    // Without add_prefix_space (incorrect):
    //   - "What is..." -> [3639, 318, ...]  ("What" without space)
    //   - " What is..." -> [3923, 318, ...]  (" What" with space)

    eprintln!("BPE ByteLevel prefix_space behavior:");
    eprintln!("  With add_prefix_space=true (correct):");
    eprintln!("    'What is...' -> [3923, 318, ...] (first token includes virtual space)");
    eprintln!("  Without add_prefix_space=false (incorrect):");
    eprintln!("    'What is...' -> [3639, 318, ...] (first token differs from llama.cpp)");
    eprintln!();
    eprintln!("This test serves as documentation of the fix applied in PR #468");
}
