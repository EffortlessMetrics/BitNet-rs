//! Greedy Decode Parity Tests
//!
//! Tests feature spec: docs/explanation/inference-engine-architecture.md#greedy-decoding
//! Architecture: docs/reference/sampling-algorithms.md#greedy-sampling
//!
//! This test suite validates greedy decoding correctness by verifying:
//! - Single-step greedy: same logits → same argmax token (with tie-breaking)
//! - Multi-step greedy: verify token sequence stability and determinism
//! - Temperature=0 equivalence: greedy mode should match temperature=0 sampling
//! - Reproducibility: same prompt + seed → same output across multiple runs
//!
//! # Test Coverage
//!
//! - **Greedy argmax**: Verify argmax selection with deterministic tie-breaking
//! - **Deterministic multi-step**: Same prompt → same output with fixed seed
//! - **Temperature equivalence**: greedy flag should match temperature=0.0
//! - **Reproducibility**: Multiple runs with same config produce identical output
//!
//! # Environment Variables
//!
//! - `BITNET_GGUF` or `CROSSVAL_GGUF`: Path to GGUF model file (required)
//! - `BITNET_SKIP_SLOW_TESTS`: Skip tests requiring model loading
//! - `BITNET_DETERMINISTIC=1` and `BITNET_SEED=42`: Enable deterministic mode
//!
//! # Running the Tests
//!
//! ```bash
//! # Run greedy decode parity tests (requires model file)
//! BITNET_DETERMINISTIC=1 BITNET_SEED=42 \
//!   BITNET_GGUF=models/model.gguf \
//!   cargo test -p bitnet-inference --test greedy_decode_parity --no-default-features --features cpu
//!
//! # Skip slow tests
//! BITNET_SKIP_SLOW_TESTS=1 cargo test -p bitnet-inference --test greedy_decode_parity
//!
//! # Run with ignored tests (includes full inference)
//! BITNET_GGUF=models/model.gguf cargo test -p bitnet-inference --test greedy_decode_parity -- --ignored --include-ignored
//! ```
#![cfg(feature = "cpu")]
use anyhow::{Context, Result};
use bitnet_common::Device as BNDevice;
use bitnet_inference::{GenerationConfig, InferenceEngine};
use bitnet_models::ModelLoader;
use bitnet_tokenizers::auto;
use std::path::{Path, PathBuf};
/// Helper to discover test model from environment or models/ directory
fn discover_test_model() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("BITNET_GGUF") {
        let model_path = PathBuf::from(&path);
        if model_path.exists() {
            return Ok(model_path);
        }
        anyhow::bail!("BITNET_GGUF set to '{}' but file does not exist", path);
    }
    if let Ok(path) = std::env::var("CROSSVAL_GGUF") {
        let model_path = PathBuf::from(&path);
        if model_path.exists() {
            return Ok(model_path);
        }
        anyhow::bail!("CROSSVAL_GGUF set to '{}' but file does not exist", path);
    }
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
/// Helper to perform greedy argmax with deterministic tie-breaking
/// (lower index wins on ties)
fn greedy_argmax(logits: &[f32]) -> usize {
    let mut argmax = 0;
    let mut best = logits[0];
    for (i, &val) in logits.iter().enumerate().skip(1) {
        if val > best {
            best = val;
            argmax = i;
        }
    }
    argmax
}
#[cfg(test)]
mod greedy_argmax_tests {
    use super::*;
    /// Tests feature spec: sampling-algorithms.md#AC1-greedy-argmax-simple
    /// Verify greedy argmax selection with simple logits
    ///
    /// **TDD Scaffolding**: Test compiles and validates argmax logic
    #[test]
    fn test_greedy_argmax_simple() {
        let logits = vec![1.0, 2.0, 5.0, 3.0];
        let argmax = greedy_argmax(&logits);
        assert_eq!(argmax, 2, "Expected argmax=2 for logits {:?}", logits);
        let logits = vec![10.0, 2.0, 3.0, 1.0];
        let argmax = greedy_argmax(&logits);
        assert_eq!(argmax, 0, "Expected argmax=0 for logits {:?}", logits);
        let logits = vec![1.0, 2.0, 3.0, 10.0];
        let argmax = greedy_argmax(&logits);
        assert_eq!(argmax, 3, "Expected argmax=3 for logits {:?}", logits);
        eprintln!("✓ Greedy argmax simple cases passed");
    }
    /// Tests feature spec: sampling-algorithms.md#AC2-greedy-argmax-tie-breaking
    /// Verify greedy argmax with tie-breaking (lower index wins)
    ///
    /// **TDD Scaffolding**: Test compiles and validates tie-breaking behavior
    #[test]
    fn test_greedy_argmax_tie_breaking() {
        let logits = vec![1.0, 5.0, 5.0, 3.0];
        let argmax = greedy_argmax(&logits);
        assert_eq!(
            argmax, 1,
            "Expected lower index to win on tie: argmax=1 for logits {:?}",
            logits
        );
        let logits = vec![7.0, 2.0, 3.0, 7.0];
        let argmax = greedy_argmax(&logits);
        assert_eq!(
            argmax, 0,
            "Expected lower index to win on tie: argmax=0 for logits {:?}",
            logits
        );
        let logits = vec![5.0, 5.0, 5.0, 5.0];
        let argmax = greedy_argmax(&logits);
        assert_eq!(argmax, 0, "Expected index 0 when all equal: argmax=0 for logits {:?}", logits);
        eprintln!("✓ Greedy argmax tie-breaking passed");
    }
    /// Tests feature spec: sampling-algorithms.md#AC3-greedy-argmax-negative
    /// Verify greedy argmax with negative logits
    ///
    /// **TDD Scaffolding**: Test compiles and validates negative value handling
    #[test]
    fn test_greedy_argmax_negative_logits() {
        let logits = vec![-1.0, -2.0, 3.0, -5.0];
        let argmax = greedy_argmax(&logits);
        assert_eq!(argmax, 2, "Expected argmax=2 for mixed logits {:?}", logits);
        let logits = vec![-10.0, -2.0, -5.0, -3.0];
        let argmax = greedy_argmax(&logits);
        assert_eq!(argmax, 1, "Expected argmax=1 for all-negative logits {:?}", logits);
        let logits = vec![-1.0, -2.0, -1.0, -3.0];
        let argmax = greedy_argmax(&logits);
        assert_eq!(
            argmax, 0,
            "Expected lower index for negative tie: argmax=0 for logits {:?}",
            logits
        );
        eprintln!("✓ Greedy argmax with negative logits passed");
    }
}
#[cfg(test)]
mod deterministic_inference_tests {
    use super::*;
    /// Tests feature spec: inference-engine-architecture.md#AC4-deterministic-multi-step
    /// Verify deterministic multi-step greedy decoding
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[tokio::test]
    #[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
    async fn test_deterministic_multi_step_greedy() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: deterministic multi-step greedy");
            return Ok(());
        }
        let model_path = discover_test_model()?;
        let loader = ModelLoader::new(BNDevice::Cpu);
        let model = loader.load::<&Path>(model_path.as_ref())?;
        let tokenizer = auto::load_auto(&model_path, None)?;
        let engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)?;
        let prompt = "What is 2+2?";
        let add_bos = true;
        let config = GenerationConfig::greedy()
            .with_max_tokens(4)
            .with_temperature(0.0)
            .with_top_k(1)
            .with_top_p(1.0)
            .with_repetition_penalty(1.0)
            .with_stop_sequences(vec![])
            .with_stop_token_ids(vec![])
            .with_stop_string_window(64)
            .with_seed(42)
            .with_skip_special_tokens(false)
            .with_eos_token_id(None)
            .with_logits_tap_steps(0)
            .with_logits_topk(0)
            .with_logits_cb(None)
            .with_add_bos(add_bos);
        let ids = engine.tokenizer().encode(prompt, add_bos, false)?;
        let output1 = engine.generate_tokens(&ids, &config).await?;
        let output2 = engine.generate_tokens(&ids, &config).await?;
        eprintln!("Deterministic greedy test:");
        eprintln!("  Prompt: '{}'", prompt);
        eprintln!("  Output 1: {:?}", output1);
        eprintln!("  Output 2: {:?}", output2);
        assert_eq!(
            output1, output2,
            "Greedy decoding is non-deterministic!\n  First run: {:?}\n  Second run: {:?}",
            output1, output2
        );
        eprintln!("✓ Deterministic multi-step greedy passed");
        Ok(())
    }
    /// Tests feature spec: inference-engine-architecture.md#AC5-temperature-zero-equivalence
    /// Verify temperature=0 is equivalent to greedy mode
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[tokio::test]
    #[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
    async fn test_temperature_zero_equivalence() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: temperature=0 equivalence");
            return Ok(());
        }
        let model_path = discover_test_model()?;
        let loader = ModelLoader::new(BNDevice::Cpu);
        let model = loader.load::<&Path>(model_path.as_ref())?;
        let tokenizer = auto::load_auto(&model_path, None)?;
        let engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)?;
        let prompt = "2+2=";
        let add_bos = true;
        let greedy_config = GenerationConfig::greedy()
            .with_max_tokens(4)
            .with_temperature(0.0)
            .with_top_k(1)
            .with_top_p(1.0)
            .with_repetition_penalty(1.0)
            .with_stop_sequences(vec![])
            .with_stop_token_ids(vec![])
            .with_stop_string_window(64)
            .with_seed(42)
            .with_skip_special_tokens(false)
            .with_eos_token_id(None)
            .with_logits_tap_steps(0)
            .with_logits_topk(0)
            .with_logits_cb(None)
            .with_add_bos(add_bos);
        let temp_zero_config = GenerationConfig::greedy().with_temperature(0.0);
        let ids = engine.tokenizer().encode(prompt, add_bos, false)?;
        let greedy_output = engine.generate_tokens(&ids, &greedy_config).await?;
        let temp_output = engine.generate_tokens(&ids, &temp_zero_config).await?;
        eprintln!("Temperature=0 equivalence test:");
        eprintln!("  Prompt: '{}'", prompt);
        eprintln!("  Greedy output: {:?}", greedy_output);
        eprintln!("  Temp=0 output: {:?}", temp_output);
        assert_eq!(
            greedy_output, temp_output,
            "Greedy and temperature=0 produce different outputs!\n  Greedy: {:?}\n  Temp=0: {:?}",
            greedy_output, temp_output
        );
        eprintln!("✓ Temperature=0 equivalence passed");
        Ok(())
    }
    /// Tests feature spec: inference-engine-architecture.md#AC6-reproducibility-with-seed
    /// Verify reproducibility with fixed seed across multiple runs
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[tokio::test]
    #[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
    async fn test_reproducibility_with_seed() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: reproducibility with seed");
            return Ok(());
        }
        let model_path = discover_test_model()?;
        let prompt = "What is the capital of France?";
        let add_bos = true;
        let config = GenerationConfig::greedy()
            .with_max_tokens(8)
            .with_temperature(0.0)
            .with_top_k(1)
            .with_top_p(1.0)
            .with_repetition_penalty(1.0)
            .with_stop_sequences(vec![])
            .with_stop_token_ids(vec![])
            .with_stop_string_window(64)
            .with_seed(42)
            .with_skip_special_tokens(false)
            .with_eos_token_id(None)
            .with_logits_tap_steps(0)
            .with_logits_topk(0)
            .with_logits_cb(None)
            .with_add_bos(add_bos);
        let mut outputs = Vec::new();
        for i in 1..=3 {
            let loader = ModelLoader::new(BNDevice::Cpu);
            let model = loader.load::<&Path>(model_path.as_ref())?;
            let tokenizer = auto::load_auto(&model_path, None)?;
            let engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)?;
            let ids = engine.tokenizer().encode(prompt, add_bos, false)?;
            let output = engine.generate_tokens(&ids, &config).await?;
            eprintln!("Run {}: {:?}", i, output);
            outputs.push(output);
        }
        assert_eq!(
            outputs[0], outputs[1],
            "Non-reproducible outputs between runs 1 and 2!\n  Run 1: {:?}\n  Run 2: {:?}",
            outputs[0], outputs[1]
        );
        assert_eq!(
            outputs[1], outputs[2],
            "Non-reproducible outputs between runs 2 and 3!\n  Run 2: {:?}\n  Run 3: {:?}",
            outputs[1], outputs[2]
        );
        eprintln!("✓ Reproducibility with seed passed");
        Ok(())
    }
}
#[cfg(test)]
mod logits_validation_tests {
    use super::*;
    /// Tests feature spec: inference-engine-architecture.md#AC7-logits-shape-validation
    /// Verify logits output shape matches vocab size
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[tokio::test]
    #[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
    async fn test_logits_shape_validation() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: logits shape validation");
            return Ok(());
        }
        let model_path = discover_test_model()?;
        let loader = ModelLoader::new(BNDevice::Cpu);
        let model = loader.load::<&Path>(model_path.as_ref())?;
        let tokenizer = auto::load_auto(&model_path, None)?;
        let mut engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)?;
        let vocab_size = engine.tokenizer().vocab_size();
        let prompt = "What is 2+2?";
        let add_bos = true;
        let ids = engine.tokenizer().encode(prompt, add_bos, false)?;
        eprintln!("Vocab size: {}", vocab_size);
        eprintln!("Encoded tokens: {:?}", ids);
        let logits = engine.eval_ids(&ids).await?;
        eprintln!("Logits length: {}", logits.len());
        assert_eq!(
            logits.len(),
            vocab_size,
            "Logits length {} does not match vocab size {}",
            logits.len(),
            vocab_size
        );
        let non_zero_count = logits.iter().filter(|&&x| x != 0.0).count();
        eprintln!("Non-zero logits: {}/{}", non_zero_count, vocab_size);
        assert!(non_zero_count > 0, "All logits are zero - inference not producing valid output");
        let argmax = greedy_argmax(&logits);
        assert!(
            argmax < vocab_size,
            "Argmax {} is out of bounds for vocab size {}",
            argmax,
            vocab_size
        );
        eprintln!("✓ Logits shape validation passed");
        Ok(())
    }
    /// Tests feature spec: inference-engine-architecture.md#AC8-logits-argmax-consistency
    /// Verify first generated token matches argmax from eval_ids
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[tokio::test]
    #[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
    async fn test_logits_argmax_consistency() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: logits argmax consistency");
            return Ok(());
        }
        let model_path = discover_test_model()?;
        let loader = ModelLoader::new(BNDevice::Cpu);
        let model = loader.load::<&Path>(model_path.as_ref())?;
        let tokenizer = auto::load_auto(&model_path, None)?;
        let mut engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)?;
        let prompt = "2+2=";
        let add_bos = true;
        let ids = engine.tokenizer().encode(prompt, add_bos, false)?;
        let logits = engine.eval_ids(&ids).await?;
        let expected_argmax = greedy_argmax(&logits);
        eprintln!("Expected argmax from logits: {}", expected_argmax);
        let config = GenerationConfig::greedy()
            .with_max_tokens(1)
            .with_seed(42)
            .with_skip_special_tokens(false)
            .with_add_bos(add_bos);
        let generated = engine.generate_tokens(&ids, &config).await?;
        eprintln!("Generated tokens: {:?}", generated);
        assert!(!generated.is_empty(), "Greedy generation produced no tokens");
        assert_eq!(
            generated[0] as usize, expected_argmax,
            "First generated token {} does not match argmax {}",
            generated[0], expected_argmax
        );
        eprintln!("✓ Logits argmax consistency passed");
        Ok(())
    }
}
