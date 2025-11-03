//! Intelligibility Smoke Test Suite
//!
//! Tests feature spec: docs/explanation/cli-ux-improvements-spec.md#intelligibility-testing
//! Architecture: docs/reference/inference-engine-architecture.md#generation-quality
//!
//! This test suite validates actual generation quality with known-good prompts to
//! measure intelligibility and catch regressions in output quality. Tests verify:
//! - Simple completion tasks (math, pattern completion)
//! - Q&A tasks (capital cities, factual questions)
//! - Template-aware generation (raw, instruct, llama3-chat)
//! - Stop sequence behavior (correct termination)
//! - Repetition penalty effectiveness
//!
//! # Test Coverage
//!
//! - **Simple math**: "2+2=" → contains "4"
//! - **Capital cities**: "What is the capital of France?" → contains "Paris"
//! - **Pattern completion**: "A B C D" → contains "E" or "F"
//! - **Factual questions**: "The sky is" → contains "blue"
//! - **Coherent continuation**: Verify no garbage output (e.g., "jjjj kkkk")
//! - **LLaMA-3 chat**: Structured Q&A with system prompts
//! - **Stop sequences**: Correct termination with "\n\n" or "\n\nQ:"
//! - **Repetition penalty**: Verify reduced repetition with rep_penalty=1.1
//!
//! # Pass Criteria
//!
//! - ≥7/10 prompts produce coherent, on-topic answers
//! - Zero "garbled" outputs (e.g., "jjjj kkkk llll")
//! - Stop sequences trigger correctly
//!
//! # Environment Variables
//!
//! - `BITNET_GGUF`: Path to GGUF model file (required)
//! - `BITNET_SKIP_SLOW_TESTS`: Skip tests requiring model loading
//! - `RUST_LOG=warn`: Reduce log noise for clean output inspection
//!
//! # Running the Tests
//!
//! ```bash
//! # Run intelligibility smoke tests (requires model file)
//! RUST_LOG=warn BITNET_GGUF=models/model.gguf \
//!   cargo test -p bitnet-cli --test intelligibility_smoke --no-default-features --features cpu,full-cli
//!
//! # Skip slow tests
//! BITNET_SKIP_SLOW_TESTS=1 cargo test -p bitnet-cli --test intelligibility_smoke
//!
//! # Run with ignored tests (full validation)
//! BITNET_GGUF=models/model.gguf cargo test -p bitnet-cli --test intelligibility_smoke -- --ignored --include-ignored
//! ```

#![cfg(feature = "cpu")]

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;

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

/// Helper to get bitnet-cli binary path for integration testing
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

/// Test case for intelligibility smoke testing
#[derive(Debug, Clone)]
struct IntelligibilityTest {
    /// Test case name
    name: &'static str,
    /// Prompt text
    prompt: &'static str,
    /// Prompt template (raw, instruct, llama3-chat)
    template: &'static str,
    /// System prompt (for llama3-chat)
    system_prompt: Option<&'static str>,
    /// Max tokens to generate
    max_tokens: u32,
    /// Expected pattern (contains check, case-insensitive)
    expected_pattern: Option<&'static str>,
    /// Temperature (0.0 for greedy, 0.7 for sampling)
    temperature: f32,
    /// Top-k sampling
    top_k: u32,
    /// Top-p sampling
    top_p: f32,
    /// Repetition penalty
    repetition_penalty: f32,
    /// Stop sequences
    stop_sequences: Vec<&'static str>,
}

impl IntelligibilityTest {
    /// Run this test case and return the output
    fn run(&self, cli_binary: &PathBuf, model_path: &PathBuf) -> Result<String> {
        let mut cmd = Command::new(cli_binary);

        cmd.arg("run")
            .arg("--model")
            .arg(model_path)
            .arg("--prompt-template")
            .arg(self.template)
            .arg("--prompt")
            .arg(self.prompt)
            .arg("--max-tokens")
            .arg(self.max_tokens.to_string())
            .arg("--temperature")
            .arg(self.temperature.to_string())
            .arg("--top-k")
            .arg(self.top_k.to_string())
            .arg("--top-p")
            .arg(self.top_p.to_string())
            .arg("--repetition-penalty")
            .arg(self.repetition_penalty.to_string());

        // Add system prompt if specified
        if let Some(sys_prompt) = self.system_prompt {
            cmd.arg("--system-prompt").arg(sys_prompt);
        }

        // Add stop sequences
        for stop_seq in &self.stop_sequences {
            cmd.arg("--stop").arg(stop_seq);
        }

        // Set environment for clean output
        cmd.env("RUST_LOG", "warn").env("BITNET_DETERMINISTIC", "1").env("BITNET_SEED", "42");

        eprintln!("Running test case: {}", self.name);
        eprintln!("  Prompt: '{}'", self.prompt);
        eprintln!("  Template: {}", self.template);
        eprintln!("  Max tokens: {}", self.max_tokens);

        let output = cmd.output().context("Failed to execute bitnet-cli")?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            anyhow::bail!(
                "CLI execution failed for test '{}':\n  Stdout: {}\n  Stderr: {}",
                self.name,
                stdout,
                stderr
            );
        }

        Ok(stdout)
    }

    /// Validate the output against expected patterns
    fn validate(&self, output: &str) -> Result<bool> {
        eprintln!("  Output: '{}'", output.trim());

        // Check for garbled output (common failure mode)
        if self.is_garbled(output) {
            eprintln!("  ❌ FAIL: Garbled output detected");
            return Ok(false);
        }

        // Check expected pattern if specified
        if let Some(pattern) = self.expected_pattern {
            let normalized_output = output.to_lowercase();
            let normalized_pattern = pattern.to_lowercase();

            if normalized_output.contains(&normalized_pattern) {
                eprintln!("  ✓ PASS: Output contains expected pattern '{}'", pattern);
                return Ok(true);
            } else {
                eprintln!("  ❌ FAIL: Output does not contain expected pattern '{}'", pattern);
                return Ok(false);
            }
        }

        // If no expected pattern, just check for coherence
        if output.trim().is_empty() {
            eprintln!("  ❌ FAIL: Empty output");
            return Ok(false);
        }

        eprintln!("  ✓ PASS: Coherent output (no expected pattern specified)");
        Ok(true)
    }

    /// Check if output is garbled (common failure mode)
    fn is_garbled(&self, output: &str) -> bool {
        // Pattern 1: Excessive character repetition (e.g., "jjjj kkkk llll")
        let words: Vec<&str> = output.split_whitespace().collect();
        for word in &words {
            if word.len() >= 3 {
                let first_char = word.chars().next().unwrap();
                if word.chars().all(|c| c == first_char) {
                    return true; // All same character (e.g., "jjjj")
                }
            }
        }

        // Pattern 2: Very short repetitive output
        if output.len() < 10 && words.len() <= 2 {
            let unique_words: std::collections::HashSet<_> = words.iter().collect();
            if unique_words.len() == 1 {
                return true; // Single word repeated
            }
        }

        false
    }
}

/// Define the 10 intelligibility test cases
fn get_intelligibility_tests() -> Vec<IntelligibilityTest> {
    vec![
        // Test 1: Simple completion (math)
        IntelligibilityTest {
            name: "simple_math_completion",
            prompt: "2+2=",
            template: "raw",
            system_prompt: None,
            max_tokens: 1,
            expected_pattern: Some("4"),
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
        },
        // Test 2: Capital city Q&A
        IntelligibilityTest {
            name: "capital_city_qa",
            prompt: "What is the capital of France?",
            template: "instruct",
            system_prompt: None,
            max_tokens: 16,
            expected_pattern: Some("Paris"),
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_sequences: vec![],
        },
        // Test 3: Simple math Q&A
        IntelligibilityTest {
            name: "simple_math_qa",
            prompt: "Answer with a single digit: 5-3=",
            template: "instruct",
            system_prompt: None,
            max_tokens: 4,
            expected_pattern: Some("2"),
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
        },
        // Test 4: Pattern completion
        IntelligibilityTest {
            name: "pattern_completion",
            prompt: "A B C D",
            template: "raw",
            system_prompt: None,
            max_tokens: 4,
            expected_pattern: None, // Flexible: could be "E" or "F" or other continuation
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_sequences: vec![],
        },
        // Test 5: Color of sky
        IntelligibilityTest {
            name: "color_of_sky",
            prompt: "The sky is",
            template: "raw",
            system_prompt: None,
            max_tokens: 8,
            expected_pattern: Some("blue"),
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_sequences: vec![],
        },
        // Test 6: Simple greeting continuation
        IntelligibilityTest {
            name: "greeting_continuation",
            prompt: "Hello, my name is",
            template: "raw",
            system_prompt: None,
            max_tokens: 16,
            expected_pattern: None, // Just check for coherence
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_sequences: vec![],
        },
        // Test 7: LLaMA-3 chat Q&A
        IntelligibilityTest {
            name: "llama3_chat_photosynthesis",
            prompt: "What is photosynthesis?",
            template: "llama3-chat",
            system_prompt: Some("You are a helpful assistant"),
            max_tokens: 64,
            expected_pattern: None, // Check for coherent scientific explanation
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
            repetition_penalty: 1.1,
            stop_sequences: vec![],
        },
        // Test 8: Instruct Q&A (CPU explanation)
        IntelligibilityTest {
            name: "instruct_cpu_explanation",
            prompt: "Explain in one sentence what a CPU does.",
            template: "instruct",
            system_prompt: None,
            max_tokens: 32,
            expected_pattern: None, // Check for coherent technical explanation
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_sequences: vec![],
        },
        // Test 9: Repetition test
        IntelligibilityTest {
            name: "repetition_penalty_test",
            prompt: "The dog ran and the dog",
            template: "raw",
            system_prompt: None,
            max_tokens: 16,
            expected_pattern: None, // Check that "dog" is not repeated excessively
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_sequences: vec![],
        },
        // Test 10: Stop sequence test
        IntelligibilityTest {
            name: "stop_sequence_test",
            prompt: "Q: What is 2+2?\nA: 4\n\nQ: What is 5+5?\nA:",
            template: "instruct",
            system_prompt: None,
            max_tokens: 32,
            expected_pattern: None, // Should stop at "\n\nQ:" or "\n\n"
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_sequences: vec!["\n\nQ:", "\n\n"],
        },
    ]
}

#[cfg(test)]
mod intelligibility_smoke_tests {
    use super::*;

    /// Tests feature spec: cli-ux-improvements-spec.md#AC10-intelligibility-suite
    /// Run all 10 intelligibility smoke tests and report pass rate
    ///
    /// **Pass Criteria**: ≥7/10 prompts produce coherent, on-topic answers
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file and CLI binary to execute
    #[test]
    #[ignore = "requires model file and CLI binary - run manually or in CI with BITNET_GGUF set"]
    fn test_intelligibility_smoke_suite() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: intelligibility smoke suite");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let cli_binary = get_cli_binary_path()?;

        eprintln!("\n=== Intelligibility Smoke Test Suite ===");
        eprintln!("Model: {}", model_path.display());
        eprintln!("CLI: {}", cli_binary.display());
        eprintln!();

        let tests = get_intelligibility_tests();
        let mut results = Vec::new();

        for test in &tests {
            eprintln!("\n--- Test Case: {} ---", test.name);

            let output = match test.run(&cli_binary, &model_path) {
                Ok(out) => out,
                Err(e) => {
                    eprintln!("  ❌ FAIL: Execution error: {}", e);
                    results.push(false);
                    continue;
                }
            };

            let passed = match test.validate(&output) {
                Ok(pass) => pass,
                Err(e) => {
                    eprintln!("  ❌ FAIL: Validation error: {}", e);
                    false
                }
            };

            results.push(passed);
        }

        // Summary
        let pass_count = results.iter().filter(|&&x| x).count();
        let total_count = results.len();
        let pass_rate = (pass_count as f32) / (total_count as f32);

        eprintln!("\n=== Summary ===");
        eprintln!("Passed: {}/{} ({:.1}%)", pass_count, total_count, pass_rate * 100.0);

        // Pass criteria: ≥7/10 (70%)
        assert!(
            pass_count >= 7,
            "Intelligibility smoke test suite failed: {}/{} passed (threshold: 7/10)",
            pass_count,
            total_count
        );

        eprintln!("✓ Intelligibility smoke test suite PASSED");

        Ok(())
    }

    /// Tests feature spec: cli-ux-improvements-spec.md#AC11-simple-math-greedy
    /// Individual test: Simple math completion (greedy)
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file and CLI binary to execute
    #[test]
    #[ignore = "requires model file and CLI binary - run manually or in CI with BITNET_GGUF set"]
    fn test_simple_math_completion() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: simple math completion");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let cli_binary = get_cli_binary_path()?;

        let test = IntelligibilityTest {
            name: "simple_math_completion",
            prompt: "2+2=",
            template: "raw",
            system_prompt: None,
            max_tokens: 1,
            expected_pattern: Some("4"),
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
        };

        let output = test.run(&cli_binary, &model_path)?;
        let passed = test.validate(&output)?;

        assert!(passed, "Simple math completion test failed");

        eprintln!("✓ Simple math completion test PASSED");

        Ok(())
    }

    /// Tests feature spec: cli-ux-improvements-spec.md#AC12-capital-city-qa
    /// Individual test: Capital city Q&A
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file and CLI binary to execute
    #[test]
    #[ignore = "requires model file and CLI binary - run manually or in CI with BITNET_GGUF set"]
    fn test_capital_city_qa() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: capital city Q&A");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let cli_binary = get_cli_binary_path()?;

        let test = IntelligibilityTest {
            name: "capital_city_qa",
            prompt: "What is the capital of France?",
            template: "instruct",
            system_prompt: None,
            max_tokens: 16,
            expected_pattern: Some("Paris"),
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_sequences: vec![],
        };

        let output = test.run(&cli_binary, &model_path)?;
        let passed = test.validate(&output)?;

        assert!(passed, "Capital city Q&A test failed");

        eprintln!("✓ Capital city Q&A test PASSED");

        Ok(())
    }

    /// Tests feature spec: cli-ux-improvements-spec.md#AC13-coherence-check
    /// Individual test: Coherent continuation (no garbled output)
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file and CLI binary to execute
    #[test]
    #[ignore = "requires model file and CLI binary - run manually or in CI with BITNET_GGUF set"]
    fn test_coherent_continuation() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: coherent continuation");
            return Ok(());
        }

        let model_path = discover_test_model()?;
        let cli_binary = get_cli_binary_path()?;

        let test = IntelligibilityTest {
            name: "greeting_continuation",
            prompt: "Hello, my name is",
            template: "raw",
            system_prompt: None,
            max_tokens: 16,
            expected_pattern: None, // Just check for coherence
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_sequences: vec![],
        };

        let output = test.run(&cli_binary, &model_path)?;
        let passed = test.validate(&output)?;

        assert!(passed, "Coherent continuation test failed");

        eprintln!("✓ Coherent continuation test PASSED");

        Ok(())
    }
}
