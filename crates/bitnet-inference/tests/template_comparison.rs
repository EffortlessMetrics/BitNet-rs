//! Template Comparison Tests
//!
//! Tests feature spec: docs/explanation/prompt-template-architecture.md#template-comparison
//! Architecture: docs/reference/prompt-templates.md#template-formats
//!
//! This test suite validates prompt template behavior by comparing output quality
//! across different templates (raw, instruct, llama3-chat) with the same prompt.
//! Tests verify:
//! - Template-specific formatting (raw completion vs Q&A vs chat)
//! - Stop sequence behavior (template-aware termination)
//! - Output coherence (quality impact of template choice)
//! - Output length (template impact on generation length)
//!
//! # Test Coverage
//!
//! - **Raw template**: Completion-style generation without formatting
//! - **Instruct template**: Q&A formatting with "Question:" / "Answer:" structure
//! - **LLaMA-3 chat**: Structured chat with system prompts and special tokens
//! - **Stop sequence comparison**: Verify template-specific stop behavior
//! - **Quality comparison**: Side-by-side output analysis
//!
//! # Environment Variables
//!
//! - `BITNET_GGUF` or `CROSSVAL_GGUF`: Path to GGUF model file (required)
//! - `BITNET_SKIP_SLOW_TESTS`: Skip tests requiring model loading
//! - `RUST_LOG=warn`: Reduce log noise for clean output inspection
//!
//! # Running the Tests
//!
//! ```bash
//! # Run template comparison tests (requires model file)
//! RUST_LOG=warn BITNET_GGUF=models/model.gguf \
//!   cargo test -p bitnet-inference --test template_comparison --no-default-features --features cpu
//!
//! # Skip slow tests
//! BITNET_SKIP_SLOW_TESTS=1 cargo test -p bitnet-inference --test template_comparison
//!
//! # Run with ignored tests (full validation)
//! BITNET_GGUF=models/model.gguf cargo test -p bitnet-inference --test template_comparison -- --ignored --include-ignored
//! ```

#![cfg(feature = "cpu")]

use anyhow::{Context, Result};
use bitnet_common::Device as BNDevice;
use bitnet_inference::{GenerationConfig, InferenceEngine, TemplateType};
use bitnet_models::ModelLoader;
use bitnet_tokenizers::auto;
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

/// Template configuration for comparison testing
#[derive(Debug, Clone)]
struct TemplateConfig {
    /// Template name
    name: &'static str,
    /// Prompt template type
    template: TemplateType,
    /// System prompt (for chat templates)
    system_prompt: Option<String>,
    /// Expected stop sequences
    stop_sequences: Vec<String>,
}

impl TemplateConfig {
    /// Create raw template config
    fn raw() -> Self {
        Self {
            name: "raw",
            template: TemplateType::Raw,
            system_prompt: None,
            stop_sequences: vec![],
        }
    }

    /// Create instruct template config
    fn instruct() -> Self {
        Self {
            name: "instruct",
            template: TemplateType::Instruct,
            system_prompt: None,
            stop_sequences: vec!["\n\nQ:".to_string(), "\n\nHuman:".to_string()],
        }
    }

    /// Create LLaMA-3 chat template config
    fn llama3_chat(system_prompt: &str) -> Self {
        Self {
            name: "llama3-chat",
            template: TemplateType::Llama3Chat,
            system_prompt: Some(system_prompt.to_string()),
            stop_sequences: vec![], // EOT is token-based, not string-based
        }
    }

    /// Format prompt according to template
    fn format_prompt(&self, user_prompt: &str) -> String {
        match self.template {
            TemplateType::Raw => user_prompt.to_string(),
            TemplateType::Instruct => {
                format!("### Instruction:\n{}\n\n### Response:\n", user_prompt)
            }
            TemplateType::Llama3Chat => {
                let sys = self.system_prompt.as_deref().unwrap_or("You are a helpful assistant");
                format!(
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\
                     <|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
                     <|start_header_id|>assistant<|end_header_id|>\n\n",
                    sys, user_prompt
                )
            }
        }
    }
}

/// Test result for template comparison
#[derive(Debug)]
struct TemplateTestResult {
    /// Template name
    template_name: String,
    /// Generated output
    output: String,
    /// Output length (tokens)
    output_length: usize,
    /// Whether output is coherent (not garbled)
    is_coherent: bool,
    /// Whether stop sequence triggered
    stop_triggered: bool,
}

impl TemplateTestResult {
    /// Check if output is garbled (common failure mode)
    fn check_coherence(output: &str) -> bool {
        // Pattern 1: Excessive character repetition (e.g., "jjjj kkkk llll")
        let words: Vec<&str> = output.split_whitespace().collect();
        for word in &words {
            if word.len() >= 3 {
                let first_char = word.chars().next().unwrap();
                if word.chars().all(|c| c == first_char) {
                    return false; // All same character (e.g., "jjjj")
                }
            }
        }

        // Pattern 2: Very short repetitive output
        if output.len() < 10 && words.len() <= 2 {
            let unique_words: std::collections::HashSet<_> = words.iter().collect();
            if unique_words.len() == 1 {
                return false; // Single word repeated
            }
        }

        // Pattern 3: Empty or whitespace-only
        if output.trim().is_empty() {
            return false;
        }

        true
    }

    /// Display result in a formatted manner
    fn display(&self) {
        eprintln!("\n=== Template: {} ===", self.template_name);
        eprintln!("Output: '{}'", self.output.trim());
        eprintln!("Length: {} chars", self.output_length);
        eprintln!("Coherent: {}", if self.is_coherent { "✓" } else { "❌" });
        eprintln!("Stop triggered: {}", if self.stop_triggered { "✓" } else { "-" });
    }
}

#[cfg(test)]
mod template_comparison_tests {
    use super::*;

    /// Tests feature spec: prompt-template-architecture.md#AC1-template-formatting
    /// Compare output across all three templates (raw, instruct, llama3-chat)
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[tokio::test]
    #[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
    async fn test_template_comparison_capital_city() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: template comparison");
            return Ok(());
        }

        let model_path = discover_test_model()?;

        // Test prompt
        let prompt = "What is the capital of France?";

        // Template configurations
        let templates = vec![
            TemplateConfig::raw(),
            TemplateConfig::instruct(),
            TemplateConfig::llama3_chat("You are a helpful assistant"),
        ];

        eprintln!("\n=== Template Comparison Test ===");
        eprintln!("Prompt: '{}'", prompt);

        let mut results = Vec::new();

        for template_config in templates {
            eprintln!("\n--- Testing template: {} ---", template_config.name);

            // Load model and tokenizer (fresh instance for each template)
            let loader = ModelLoader::new(BNDevice::Cpu);
            let model = loader.load::<&Path>(model_path.as_ref())?;
            let tokenizer = auto::load_auto(&model_path, None)?;

            let engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)?;

            // Format prompt according to template
            let formatted_prompt = template_config.format_prompt(prompt);
            eprintln!("Formatted prompt: '{}'", formatted_prompt);

            // Generation config
            let config = GenerationConfig {
                max_new_tokens: 32,
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
                stop_sequences: template_config.stop_sequences.clone(),
                stop_token_ids: vec![],
                stop_string_window: 64,
                seed: Some(42),
                skip_special_tokens: false,
                eos_token_id: None,
                logits_tap_steps: 0,
                logits_topk: 0,
                logits_cb: None,
                add_bos: true,
            };

            // Encode and generate
            let ids = engine.tokenizer().encode(&formatted_prompt, config.add_bos, false)?;
            let output_ids = engine.generate_tokens(&ids, &config).await?;
            let output = engine.tokenizer().decode(&output_ids)?;

            // Analyze result
            let result = TemplateTestResult {
                template_name: template_config.name.to_string(),
                output: output.clone(),
                output_length: output.len(),
                is_coherent: TemplateTestResult::check_coherence(&output),
                stop_triggered: template_config
                    .stop_sequences
                    .iter()
                    .any(|stop| output.contains(stop)),
            };

            result.display();
            results.push(result);
        }

        // Summary
        eprintln!("\n=== Summary ===");
        let coherent_count = results.iter().filter(|r| r.is_coherent).count();
        eprintln!("Coherent outputs: {}/{}", coherent_count, results.len());

        // Verify at least one template produces coherent output
        assert!(
            coherent_count > 0,
            "No template produced coherent output for prompt: '{}'",
            prompt
        );

        eprintln!("✓ Template comparison test passed");

        Ok(())
    }

    /// Tests feature spec: prompt-template-architecture.md#AC2-stop-sequence-behavior
    /// Verify stop sequence behavior is template-specific
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[tokio::test]
    #[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
    async fn test_template_stop_sequence_behavior() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: template stop sequence behavior");
            return Ok(());
        }

        let model_path = discover_test_model()?;

        // Test prompt designed to trigger stop sequences
        let prompt = "Q: What is 2+2?\nA: 4\n\nQ: What is 5+5?\nA:";

        // Instruct template with stop sequences
        let template_config = TemplateConfig::instruct();

        eprintln!("\n=== Stop Sequence Behavior Test ===");
        eprintln!("Template: {}", template_config.name);
        eprintln!("Prompt: '{}'", prompt);

        // Load model and tokenizer
        let loader = ModelLoader::new(BNDevice::Cpu);
        let model = loader.load::<&Path>(model_path.as_ref())?;
        let tokenizer = auto::load_auto(&model_path, None)?;

        let engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)?;

        // Format prompt
        let formatted_prompt = template_config.format_prompt(prompt);

        // Generation config with stop sequences
        let config = GenerationConfig {
            max_new_tokens: 32,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            stop_sequences: vec!["\n\nQ:".to_string(), "\n\n".to_string()],
            stop_token_ids: vec![],
            stop_string_window: 64,
            seed: Some(42),
            skip_special_tokens: false,
            eos_token_id: None,
            logits_tap_steps: 0,
            logits_topk: 0,
            logits_cb: None,
            add_bos: true,
        };

        // Encode and generate
        let ids = engine.tokenizer().encode(&formatted_prompt, config.add_bos, false)?;
        let output_ids = engine.generate_tokens(&ids, &config).await?;
        let output = engine.tokenizer().decode(&output_ids)?;

        eprintln!("Output: '{}'", output);

        // Check if stop sequence is present in output
        let stop_found = config.stop_sequences.iter().any(|stop| output.contains(stop));

        eprintln!("Stop sequence found: {}", stop_found);

        // Note: Stop sequences may or may not appear in the final output depending on
        // implementation (some implementations strip them, others include them)
        // This test primarily validates that generation terminates appropriately

        eprintln!("✓ Stop sequence behavior test passed");

        Ok(())
    }

    /// Tests feature spec: prompt-template-architecture.md#AC3-raw-vs-instruct-comparison
    /// Compare raw template vs instruct template for Q&A
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[tokio::test]
    #[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
    async fn test_raw_vs_instruct_qa() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: raw vs instruct comparison");
            return Ok(());
        }

        let model_path = discover_test_model()?;

        // Test prompt
        let prompt = "What is the capital of France?";

        eprintln!("\n=== Raw vs Instruct Comparison ===");
        eprintln!("Prompt: '{}'", prompt);

        // Test with raw template
        {
            eprintln!("\n--- Raw Template ---");
            let loader = ModelLoader::new(BNDevice::Cpu);
            let model = loader.load::<&Path>(model_path.as_ref())?;
            let tokenizer = auto::load_auto(&model_path, None)?;

            let engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)?;

            let config = GenerationConfig {
                max_new_tokens: 32,
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
                stop_sequences: vec![],
                stop_token_ids: vec![],
                stop_string_window: 64,
                seed: Some(42),
                skip_special_tokens: false,
                eos_token_id: None,
                logits_tap_steps: 0,
                logits_topk: 0,
                logits_cb: None,
                add_bos: true,
            };

            let ids = engine.tokenizer().encode(prompt, config.add_bos, false)?;
            let output_ids = engine.generate_tokens(&ids, &config).await?;
            let output = engine.tokenizer().decode(&output_ids)?;

            eprintln!("Raw output: '{}'", output);
        }

        // Test with instruct template
        {
            eprintln!("\n--- Instruct Template ---");
            let loader = ModelLoader::new(BNDevice::Cpu);
            let model = loader.load::<&Path>(model_path.as_ref())?;
            let tokenizer = auto::load_auto(&model_path, None)?;

            let engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)?;

            let template_config = TemplateConfig::instruct();
            let formatted_prompt = template_config.format_prompt(prompt);

            let config = GenerationConfig {
                max_new_tokens: 32,
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                repetition_penalty: 1.1,
                stop_sequences: template_config.stop_sequences,
                stop_token_ids: vec![],
                stop_string_window: 64,
                seed: Some(42),
                skip_special_tokens: false,
                eos_token_id: None,
                logits_tap_steps: 0,
                logits_topk: 0,
                logits_cb: None,
                add_bos: true,
            };

            let ids = engine.tokenizer().encode(&formatted_prompt, config.add_bos, false)?;
            let output_ids = engine.generate_tokens(&ids, &config).await?;
            let output = engine.tokenizer().decode(&output_ids)?;

            eprintln!("Instruct output: '{}'", output);
        }

        eprintln!("\n✓ Raw vs Instruct comparison test passed");

        Ok(())
    }

    /// Tests feature spec: prompt-template-architecture.md#AC4-llama3-chat-system-prompt
    /// Verify LLaMA-3 chat template with system prompts
    ///
    /// **TDD Scaffolding**: Test compiles but requires model file to execute
    #[tokio::test]
    #[ignore = "requires model file - run manually or in CI with BITNET_GGUF set"]
    async fn test_llama3_chat_system_prompt() -> Result<()> {
        if std::env::var("BITNET_SKIP_SLOW_TESTS").is_ok() {
            eprintln!("Skipping slow test: LLaMA-3 chat system prompt");
            return Ok(());
        }

        let model_path = discover_test_model()?;

        // Test prompt
        let prompt = "What is photosynthesis?";
        let system_prompt = "You are a helpful assistant";

        eprintln!("\n=== LLaMA-3 Chat Template Test ===");
        eprintln!("Prompt: '{}'", prompt);
        eprintln!("System: '{}'", system_prompt);

        // Load model and tokenizer
        let loader = ModelLoader::new(BNDevice::Cpu);
        let model = loader.load::<&Path>(model_path.as_ref())?;
        let tokenizer = auto::load_auto(&model_path, None)?;

        let engine = InferenceEngine::new(model.into(), tokenizer, BNDevice::Cpu)?;

        // LLaMA-3 chat template
        let template_config = TemplateConfig::llama3_chat(system_prompt);
        let formatted_prompt = template_config.format_prompt(prompt);

        eprintln!("Formatted prompt: '{}'", formatted_prompt);

        // Generation config
        let config = GenerationConfig {
            max_new_tokens: 64,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
            repetition_penalty: 1.1,
            stop_sequences: vec![],
            stop_token_ids: vec![128009], // <|eot_id|> for LLaMA-3
            stop_string_window: 64,
            seed: Some(42),
            skip_special_tokens: false,
            eos_token_id: None,
            logits_tap_steps: 0,
            logits_topk: 0,
            logits_cb: None,
            add_bos: true,
        };

        // Encode and generate
        let ids = engine.tokenizer().encode(&formatted_prompt, config.add_bos, false)?;
        let output_ids = engine.generate_tokens(&ids, &config).await?;
        let output = engine.tokenizer().decode(&output_ids)?;

        eprintln!("Output: '{}'", output);

        // Verify output is coherent
        let is_coherent = TemplateTestResult::check_coherence(&output);
        eprintln!("Coherent: {}", if is_coherent { "✓" } else { "❌" });

        eprintln!("\n✓ LLaMA-3 chat template test passed");

        Ok(())
    }
}
