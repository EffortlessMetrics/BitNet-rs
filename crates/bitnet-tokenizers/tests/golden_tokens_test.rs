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
struct GoldenTokensFile {
    version: String,
    description: String,
    test_cases: Vec<GoldenTokenCase>,
}

#[derive(Debug, Deserialize)]
struct GoldenTokenCase {
    tokenizer_kind: String,
    model_hint: String,
    text: String,
    add_bos: bool,
    parse_special: bool,
    expected_tokens: Vec<u32>,
}

/// Load golden token test cases from JSON
fn load_golden_tokens() -> Result<GoldenTokensFile> {
    let json_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("golden_tokens.json");

    let json_content = std::fs::read_to_string(&json_path)
        .with_context(|| format!("Failed to read golden tokens file: {}", json_path.display()))?;

    let golden: GoldenTokensFile =
        serde_json::from_str(&json_content).context("Failed to parse golden tokens JSON")?;

    Ok(golden)
}

#[test]
fn test_golden_tokens_file_loads() {
    // Verify the golden tokens file is valid JSON and loads correctly
    let golden = load_golden_tokens().expect("Failed to load golden tokens file");

    assert_eq!(golden.version, "1.0.0");
    assert!(!golden.test_cases.is_empty(), "Golden tokens file should contain test cases");

    eprintln!("Loaded {} golden token test cases", golden.test_cases.len());
    for (i, case) in golden.test_cases.iter().enumerate() {
        eprintln!(
            "  Case {}: {} ({}), {} tokens expected",
            i + 1,
            case.text,
            case.tokenizer_kind,
            case.expected_tokens.len()
        );
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

    // Load golden tokens
    let golden = load_golden_tokens()?;

    // Load tokenizer from model
    let mmap = MmapFile::open(&model_path)?;
    let reader = GgufReader::new(mmap.as_slice())?;
    let tokenizer = RustTokenizer::from_gguf(&reader)?;

    // Detect tokenizer kind from model
    let model_kind =
        reader.get_string_metadata("tokenizer.ggml.model").unwrap_or_else(|| "unknown".to_string());

    eprintln!("Testing with model: {}", model_path.display());
    eprintln!("Model tokenizer kind: {}", model_kind);

    // Run applicable test cases
    let mut tests_run = 0;
    let mut tests_passed = 0;

    for (i, case) in golden.test_cases.iter().enumerate() {
        // Only test cases matching the model's tokenizer kind
        if !model_kind.contains(&case.tokenizer_kind) && !case.model_hint.contains(&model_kind) {
            eprintln!("  Case {}: Skipping (kind mismatch)", i + 1);
            continue;
        }

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

    if tests_run == 0 {
        eprintln!("WARNING: No golden token tests matched the model tokenizer kind");
    }

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
