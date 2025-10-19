//! Greedy Math Q&A Confidence Test
//!
//! Tests feature spec: docs/explanation/cli-ux-improvements-spec.md#greedy-inference
//! Architecture: docs/reference/inference-engine-architecture.md#greedy-sampling
//!
//! This test suite validates greedy inference for simple math Q&A without
//! requiring slow model loading in CI. Tests are designed to catch regressions
//! in deterministic inference behavior.
//!
//! **Performance Target**: < 5 seconds (with small model), SKIP in CI if no model
//! **TDD Approach**: Tests compile successfully and validate when model available
//!
//! # Specification References
//! - Greedy sampling: cli-ux-improvements-spec.md#AC9-greedy-sampling
//! - Deterministic inference: inference-engine-architecture.md#determinism
//! - Template integration: cli-ux-improvements-spec.md#AC10-template-integration

#![cfg(feature = "cpu")]

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

/// Get test model path (auto-discover from models/ or BITNET_GGUF env var)
fn get_test_model_path() -> Result<PathBuf> {
    // Priority 1: BITNET_GGUF environment variable
    if let Ok(path) = std::env::var("BITNET_GGUF") {
        let model_path = PathBuf::from(&path);
        if model_path.exists() {
            return Ok(model_path);
        }
        anyhow::bail!("BITNET_GGUF set to '{}' but file does not exist", path);
    }

    // Priority 2: Auto-discover from models/ directory
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

/// Get bitnet-cli binary path for integration testing
fn get_cli_binary_path() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow::anyhow!("Failed to find workspace root"))?;

    // Try debug build first (most common during development), then release
    let debug_binary = workspace_root.join("target").join("debug").join("bitnet-cli");
    if debug_binary.exists() {
        return Ok(debug_binary);
    }

    let release_binary = workspace_root.join("target").join("release").join("bitnet-cli");
    if release_binary.exists() {
        return Ok(release_binary);
    }

    anyhow::bail!(
        "bitnet-cli binary not found. Build with:\n\
         cargo build -p bitnet-cli --no-default-features --features cpu,full-cli"
    );
}

#[cfg(test)]
mod greedy_math_confidence {
    use super::*;

    /// Tests feature spec: cli-ux-improvements-spec.md#AC9-greedy-math-simple
    /// Verify greedy inference produces deterministic math answer
    ///
    /// **Manual Test Instructions**:
    /// ```bash
    /// # 1. Download test model
    /// cargo run -p xtask -- download-model
    ///
    /// # 2. Build CLI
    /// cargo build -p bitnet-cli --no-default-features --features cpu,full-cli
    ///
    /// # 3. Run greedy math test
    /// BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
    ///   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
    ///   run --model models/model.gguf \
    ///   --prompt-template raw \
    ///   --prompt "2+2=" \
    ///   --max-tokens 4 \
    ///   --greedy
    ///
    /// # Expected: Output should contain "4"
    /// ```
    #[test]
    #[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
    fn test_greedy_math_simple_2plus2() -> Result<()> {
        // Skip if no model available
        let model_path = match get_test_model_path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping greedy math test: {}", e);
                eprintln!("To run this test, set BITNET_GGUF or place model in models/");
                return Ok(()); // Skip gracefully
            }
        };

        let cli_binary = match get_cli_binary_path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping greedy math test: {}", e);
                return Ok(()); // Skip gracefully
            }
        };

        eprintln!("Running greedy math confidence test...");
        eprintln!("Model: {}", model_path.display());
        eprintln!("CLI: {}", cli_binary.display());

        // Run CLI with greedy sampling and deterministic settings
        let output = Command::new(&cli_binary)
            .arg("run")
            .arg("--model")
            .arg(&model_path)
            .arg("--prompt-template")
            .arg("raw") // Raw template for completion-style prompt
            .arg("--prompt")
            .arg("2+2=") // Simple math prompt
            .arg("--max-tokens")
            .arg("4") // Short generation
            .arg("--greedy") // Deterministic greedy sampling
            .env("BITNET_DETERMINISTIC", "1") // Enable determinism
            .env("BITNET_SEED", "42") // Fixed seed
            .env("RAYON_NUM_THREADS", "1") // Single-threaded for reproducibility
            .output()
            .context("Failed to execute bitnet-cli")?;

        // Check that CLI executed successfully
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("CLI execution failed with status {}\nstderr: {}", output.status, stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("CLI output:\n{}", stdout);

        // Verify output contains "4" (correct answer)
        assert!(
            stdout.contains('4'),
            "Greedy inference should produce output containing '4' for prompt '2+2='\n\
             Got output: {}",
            stdout
        );

        eprintln!("✅ Greedy math confidence test passed");

        Ok(())
    }

    /// Tests feature spec: cli-ux-improvements-spec.md#AC9-greedy-qa-format
    /// Verify greedy inference with Instruct template Q&A format
    ///
    /// **Manual Test Instructions**:
    /// ```bash
    /// BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
    ///   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
    ///   run --model models/model.gguf \
    ///   --prompt-template instruct \
    ///   --prompt "What is 2+2?" \
    ///   --max-tokens 16 \
    ///   --greedy
    ///
    /// # Expected: Output should contain "4" or "four"
    /// ```
    #[test]
    #[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
    fn test_greedy_math_qa_format() -> Result<()> {
        // Skip if no model available
        let model_path = match get_test_model_path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping greedy Q&A test: {}", e);
                return Ok(());
            }
        };

        let cli_binary = match get_cli_binary_path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping greedy Q&A test: {}", e);
                return Ok(());
            }
        };

        eprintln!("Running greedy Q&A confidence test...");

        // Run CLI with Instruct template for Q&A format
        let output = Command::new(&cli_binary)
            .arg("run")
            .arg("--model")
            .arg(&model_path)
            .arg("--prompt-template")
            .arg("instruct") // Instruct template for Q&A
            .arg("--prompt")
            .arg("What is 2+2?") // Question format
            .arg("--max-tokens")
            .arg("16") // Slightly longer for full answer
            .arg("--greedy")
            .env("BITNET_DETERMINISTIC", "1")
            .env("BITNET_SEED", "42")
            .env("RAYON_NUM_THREADS", "1")
            .output()
            .context("Failed to execute bitnet-cli")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("CLI execution failed: {}", stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("CLI Q&A output:\n{}", stdout);

        // Verify output contains "4" or "four" (correct answer in various formats)
        let contains_answer = stdout.contains('4') || stdout.to_lowercase().contains("four");

        assert!(
            contains_answer,
            "Greedy Q&A inference should produce answer containing '4' or 'four'\n\
             Got output: {}",
            stdout
        );

        eprintln!("✅ Greedy Q&A confidence test passed");

        Ok(())
    }

    /// Tests feature spec: cli-ux-improvements-spec.md#AC9-greedy-reproducibility
    /// Verify greedy inference is deterministic across runs
    ///
    /// **Manual Test Instructions**:
    /// ```bash
    /// # Run twice with same seed, outputs should be identical
    /// BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
    ///   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
    ///   run --model models/model.gguf \
    ///   --prompt "2+2=" \
    ///   --max-tokens 4 \
    ///   --greedy > output1.txt
    ///
    /// BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
    ///   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
    ///   run --model models/model.gguf \
    ///   --prompt "2+2=" \
    ///   --max-tokens 4 \
    ///   --greedy > output2.txt
    ///
    /// diff output1.txt output2.txt  # Should be identical
    /// ```
    #[test]
    #[ignore = "requires model file and is slow - run manually for regression testing"]
    fn test_greedy_deterministic_reproducibility() -> Result<()> {
        // Skip if no model available
        let model_path = match get_test_model_path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping determinism test: {}", e);
                return Ok(());
            }
        };

        let cli_binary = match get_cli_binary_path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping determinism test: {}", e);
                return Ok(());
            }
        };

        eprintln!("Running greedy determinism test (2 runs)...");

        // Run 1: First execution
        let output1 = Command::new(&cli_binary)
            .arg("run")
            .arg("--model")
            .arg(&model_path)
            .arg("--prompt")
            .arg("2+2=")
            .arg("--max-tokens")
            .arg("4")
            .arg("--greedy")
            .env("BITNET_DETERMINISTIC", "1")
            .env("BITNET_SEED", "42")
            .env("RAYON_NUM_THREADS", "1")
            .output()
            .context("Failed to execute first run")?;

        // Run 2: Second execution (same parameters)
        let output2 = Command::new(&cli_binary)
            .arg("run")
            .arg("--model")
            .arg(&model_path)
            .arg("--prompt")
            .arg("2+2=")
            .arg("--max-tokens")
            .arg("4")
            .arg("--greedy")
            .env("BITNET_DETERMINISTIC", "1")
            .env("BITNET_SEED", "42")
            .env("RAYON_NUM_THREADS", "1")
            .output()
            .context("Failed to execute second run")?;

        assert!(output1.status.success(), "First run failed");
        assert!(output2.status.success(), "Second run failed");

        let stdout1 = String::from_utf8_lossy(&output1.stdout);
        let stdout2 = String::from_utf8_lossy(&output2.stdout);

        eprintln!("Run 1 output:\n{}", stdout1);
        eprintln!("Run 2 output:\n{}", stdout2);

        // Verify outputs are identical (deterministic)
        assert_eq!(
            stdout1, stdout2,
            "Greedy inference should be deterministic with same seed.\n\
             Run 1: {}\n\
             Run 2: {}",
            stdout1, stdout2
        );

        eprintln!("✅ Greedy determinism test passed (outputs identical)");

        Ok(())
    }
}

#[cfg(test)]
mod greedy_stop_sequences {
    use super::*;

    /// Tests feature spec: cli-ux-improvements-spec.md#AC10-stop-sequences-greedy
    /// Verify stop sequences work correctly with greedy sampling
    ///
    /// **Manual Test Instructions**:
    /// ```bash
    /// # Test with Instruct template stop sequences
    /// BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
    ///   cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
    ///   run --model models/model.gguf \
    ///   --prompt-template instruct \
    ///   --prompt "What is 2+2?" \
    ///   --max-tokens 100 \
    ///   --greedy
    ///
    /// # Expected: Should stop at "\n\nQ:" or "\n\nHuman:" (Instruct stop sequences)
    /// # Should NOT generate 100 tokens (stop early due to stop sequence)
    /// ```
    #[test]
    #[ignore = "requires model file - run manually to verify stop sequence behavior"]
    fn test_greedy_respects_stop_sequences() -> Result<()> {
        // Skip if no model available
        let model_path = match get_test_model_path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping stop sequence test: {}", e);
                return Ok(());
            }
        };

        let cli_binary = match get_cli_binary_path() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Skipping stop sequence test: {}", e);
                return Ok(());
            }
        };

        eprintln!("Running greedy stop sequence test...");

        // Run with Instruct template (has default stop sequences)
        let output = Command::new(&cli_binary)
            .arg("run")
            .arg("--model")
            .arg(&model_path)
            .arg("--prompt-template")
            .arg("instruct")
            .arg("--prompt")
            .arg("What is 2+2?")
            .arg("--max-tokens")
            .arg("100") // Request many tokens
            .arg("--greedy")
            .env("BITNET_DETERMINISTIC", "1")
            .env("BITNET_SEED", "42")
            .env("RAYON_NUM_THREADS", "1")
            .output()
            .context("Failed to execute CLI")?;

        assert!(output.status.success(), "CLI execution failed");

        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("CLI output:\n{}", stdout);

        // Verify generation stopped early (not 100 tokens)
        // Count approximate tokens (rough heuristic: whitespace-separated words)
        let word_count = stdout.split_whitespace().count();
        eprintln!("Generated approximately {} words", word_count);

        // Generation should stop well before 100 tokens due to stop sequences
        // (allowing some margin for tokenization differences)
        assert!(
            word_count < 50,
            "Generation should stop early due to stop sequences, not reach max_tokens.\n\
             Generated {} words (expected < 50)",
            word_count
        );

        // Verify output doesn't contain new question markers (stop sequences)
        assert!(
            !stdout.contains("\n\nQ:") && !stdout.contains("\n\nHuman:"),
            "Output should not contain stop sequences (generation should stop before emitting them)"
        );

        eprintln!("✅ Greedy stop sequence test passed");

        Ok(())
    }
}

// ============================================================================
// DOCUMENTATION: Manual Testing Guide
// ============================================================================

/// # Manual Testing Guide for Greedy Math Q&A
///
/// Since these tests require model loading (slow in CI), they are marked with
/// `#[ignore]` and should be run manually or in dedicated CI jobs with models.
///
/// ## Setup
///
/// ```bash
/// # 1. Download test model
/// cargo run -p xtask -- download-model
///
/// # 2. Build CLI with full features
/// cargo build -p bitnet-cli --no-default-features --features cpu,full-cli
/// ```
///
/// ## Test Execution
///
/// ### Run all greedy math tests:
/// ```bash
/// BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
///   cargo test -p bitnet-cli --test qa_greedy_math_confidence -- --ignored
/// ```
///
/// ### Run specific test:
/// ```bash
/// BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
///   cargo test -p bitnet-cli --test qa_greedy_math_confidence test_greedy_math_simple_2plus2 -- --ignored
/// ```
///
/// ## Expected Results
///
/// - **test_greedy_math_simple_2plus2**: Output contains "4"
/// - **test_greedy_math_qa_format**: Output contains "4" or "four"
/// - **test_greedy_deterministic_reproducibility**: Two runs produce identical output
/// - **test_greedy_respects_stop_sequences**: Generation stops early (< 50 words)
///
/// ## Troubleshooting
///
/// ### No model file:
/// ```bash
/// cargo run -p xtask -- download-model
/// # Or set BITNET_GGUF to your model path
/// export BITNET_GGUF=/path/to/model.gguf
/// ```
///
/// ### CLI not built:
/// ```bash
/// cargo build -p bitnet-cli --no-default-features --features cpu,full-cli
/// ```
///
/// ### Non-deterministic results:
/// Ensure environment variables are set:
/// ```bash
/// export BITNET_DETERMINISTIC=1
/// export BITNET_SEED=42
/// export RAYON_NUM_THREADS=1
/// ```
///
/// ## CI Integration
///
/// To run these tests in CI, ensure:
/// 1. Model file is available (download or cache)
/// 2. BITNET_GGUF env var points to model
/// 3. Run with: `cargo test --test qa_greedy_math_confidence -- --ignored`
///
/// ## Performance Notes
///
/// - Each test takes 3-10 seconds depending on model size and hardware
/// - Tests use small max_tokens (4-16) to minimize inference time
/// - Greedy sampling is faster than nucleus sampling (no random sampling overhead)
#[cfg(test)]
mod _manual_testing_guide {}
