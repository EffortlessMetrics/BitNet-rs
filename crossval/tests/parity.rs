//! Deterministic parity tests between Rust and C++ implementations
//!
//! These tests ensure that our Rust implementation produces identical
//! results to the Microsoft BitNet C++ implementation.

#[cfg(feature = "crossval")]
use anyhow::{Context, Result};
#[cfg(feature = "crossval")]
use std::env;

#[cfg(all(test, feature = "crossval"))]
mod tests {
    use super::*;
    use bitnet_inference::{eval_logits_once, get_model_config, get_model_vocab_size};
    use bitnet_sys::wrapper::{self, Session as CppSession};

    /// Tolerance for floating point comparisons
    const LOGIT_TOLERANCE: f32 = 1e-4; // Start with 1e-4, can tighten to 5e-5

    /// Helper to compare logits with tolerance
    fn compare_logits(rust_logits: &[f32], cpp_logits: &[f32], step: usize) -> Result<()> {
        if rust_logits.len() != cpp_logits.len() {
            anyhow::bail!(
                "Step {}: Logit vector length mismatch: Rust {} vs C++ {}",
                step,
                rust_logits.len(),
                cpp_logits.len()
            );
        }

        // Find max difference and location
        let mut max_diff = 0.0f32;
        let mut max_diff_idx = 0;

        for (i, (r, c)) in rust_logits.iter().zip(cpp_logits).enumerate() {
            let diff = (r - c).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
        }

        if max_diff > LOGIT_TOLERANCE {
            // Get top-5 tokens from each side for debugging
            let mut rust_top5: Vec<(usize, f32)> = rust_logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            rust_top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            rust_top5.truncate(5);

            let mut cpp_top5: Vec<(usize, f32)> = cpp_logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            cpp_top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            cpp_top5.truncate(5);

            anyhow::bail!(
                "Step {}: Logits differ by {} at token {} (exceeds tolerance {})\n\
                 Rust top-5: {:?}\n\
                 C++ top-5: {:?}",
                step,
                max_diff,
                max_diff_idx,
                LOGIT_TOLERANCE,
                rust_top5,
                cpp_top5
            );
        }

        Ok(())
    }

    /// Helper to get argmax token from logits
    fn argmax(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[test]
    fn test_model_loading_parity() -> Result<()> {
        // Get model path from environment
        let model_path =
            env::var("CROSSVAL_GGUF").context("Set CROSSVAL_GGUF to path of test model")?;

        // Initialize C++ backend
        wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| wrapper::free_backend());

        // Load model in C++
        let cpp_session = CppSession::load_deterministic(&model_path)?;

        // Check basic model properties
        println!("C++ Model properties:");
        println!("  n_vocab: {}", cpp_session.model.n_vocab());
        println!("  n_ctx_train: {}", cpp_session.model.n_ctx_train());
        println!("  n_embd: {}", cpp_session.model.n_embd());

        // Load model in Rust and compare properties
        let rust_config = get_model_config(&model_path)?;
        println!("\nRust Model properties:");
        println!("  vocab_size: {}", rust_config.model.vocab_size);
        println!("  hidden_size: {}", rust_config.model.hidden_size);
        println!("  num_layers: {}", rust_config.model.num_layers);

        // Compare vocab sizes
        assert_eq!(
            rust_config.model.vocab_size,
            cpp_session.model.n_vocab() as usize,
            "Vocab size mismatch between Rust and C++"
        );

        Ok(())
    }

    #[test]
    fn test_tokenization_parity() -> Result<()> {
        let model_path =
            env::var("CROSSVAL_GGUF").context("Set CROSSVAL_GGUF to path of test model")?;

        // Initialize C++ backend
        wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| wrapper::free_backend());

        // Load model
        let cpp_session = CppSession::load_deterministic(&model_path)?;

        // Test various prompts
        let test_prompts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "1 + 1 = ",
            "def fibonacci(n):",
            "🦀 Rust is awesome! 🚀",
        ];

        for prompt in &test_prompts {
            let cpp_tokens = cpp_session.tokenize(prompt)?;
            println!("Prompt: {:?}", prompt);
            println!("C++ tokens: {:?}", cpp_tokens);

            // TODO: Compare with Rust tokenization
            // let rust_tokens = rust_model.tokenize(prompt)?;
            // assert_eq!(rust_tokens, cpp_tokens, "Tokenization mismatch for: {}", prompt);
        }

        Ok(())
    }

    #[test]
    fn test_single_step_logits_parity() -> Result<()> {
        let model_path =
            env::var("CROSSVAL_GGUF").context("Set CROSSVAL_GGUF to path of test model")?;

        // Initialize C++ backend
        wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| wrapper::free_backend());

        // Load model
        let mut cpp_session = CppSession::load_deterministic(&model_path)?;

        // Simple prompt
        let prompt = "The answer is";
        let tokens = cpp_session.tokenize(prompt)?;

        // Get logits from C++
        let cpp_logits = cpp_session.eval_and_get_logits(&tokens, 0)?;

        // Get argmax token
        let cpp_next_token = argmax(&cpp_logits);

        println!("Prompt: {:?}", prompt);
        println!("Tokens: {:?}", tokens);
        println!("C++ next token ID: {}", cpp_next_token);
        println!("C++ logits shape: {}", cpp_logits.len());

        // Compare with Rust
        println!("\nGetting logits from Rust...");
        let rust_logits = eval_logits_once(&model_path, &tokens)?;
        println!("Rust logits shape: {}", rust_logits.len());

        // Compare logits
        compare_logits(&rust_logits, &cpp_logits, 0)?;

        // Compare argmax tokens
        let rust_next_token = argmax(&rust_logits);
        println!("Rust next token ID: {}", rust_next_token);

        if rust_next_token != cpp_next_token {
            anyhow::bail!(
                "Token selection mismatch! Rust {} vs C++ {}",
                rust_next_token,
                cpp_next_token
            );
        }

        println!("✅ Single-step logits match!");

        Ok(())
    }

    #[test]
    fn test_multi_step_generation_parity() -> Result<()> {
        let model_path =
            env::var("CROSSVAL_GGUF").context("Set CROSSVAL_GGUF to path of test model")?;

        // Initialize C++ backend
        wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| wrapper::free_backend());

        // Load model
        let mut cpp_session = CppSession::load_deterministic(&model_path)?;

        // Generate 10 tokens
        let prompt = "Once upon a time";
        let max_tokens = 10;

        let mut tokens = cpp_session.tokenize(prompt)?;
        let prompt_len = tokens.len();

        // Evaluate prompt
        cpp_session.context.eval(&tokens, 0)?;

        println!("Generating {} tokens from prompt: {:?}", max_tokens, prompt);
        println!("Initial tokens: {:?}", tokens);

        // Generate tokens step by step
        for step in 0..max_tokens {
            // Get logits
            let cpp_logits = cpp_session.context.get_logits()?;

            // Sample greedily (deterministic)
            let next_token = argmax(&cpp_logits) as i32;

            println!("Step {}: Generated token {}", step, next_token);

            // Compare with Rust at each step
            let rust_logits = eval_logits_once(&model_path, &tokens)?;
            compare_logits(&rust_logits, &cpp_logits, step)?;

            let rust_token = argmax(&rust_logits) as i32;
            if rust_token != next_token {
                anyhow::bail!(
                    "Step {}: Token mismatch! Rust {} vs C++ {}",
                    step,
                    rust_token,
                    next_token
                );
            }

            tokens.push(next_token);

            // Evaluate new token
            cpp_session
                .context
                .eval(&[next_token], (prompt_len + step) as i32)?;
        }

        // Decode final text
        let generated_text = cpp_session.decode(&tokens[prompt_len..])?;
        println!("C++ generated text: {:?}", generated_text);
        println!("✅ Multi-step generation matches!");

        Ok(())
    }

    #[test]
    fn test_batch_processing_parity() -> Result<()> {
        let model_path =
            env::var("CROSSVAL_GGUF").context("Set CROSSVAL_GGUF to path of test model")?;

        // Initialize C++ backend
        wrapper::init_backend();
        let _guard = scopeguard::guard((), |_| wrapper::free_backend());

        // Load model
        let mut cpp_session = CppSession::load_deterministic(&model_path)?;

        // Process a longer sequence and get all logits
        let prompt = "The quick brown fox jumps over the lazy dog. This is a test.";
        let tokens = cpp_session.tokenize(prompt)?;

        if tokens.is_empty() {
            anyhow::bail!("Failed to tokenize prompt");
        }

        println!("Testing batch processing with {} tokens", tokens.len());

        // Evaluate all tokens
        cpp_session.context.eval(&tokens, 0)?;

        // Get logits for each position
        let all_cpp_logits = cpp_session.context.get_all_logits(tokens.len())?;

        println!("Got logits for {} positions", all_cpp_logits.len());

        // Verify each position
        for (i, logits) in all_cpp_logits.iter().enumerate() {
            let next_token = argmax(logits);
            println!("Position {}: argmax token = {}", i, next_token);

            // TODO: Compare with Rust batch processing
            // let rust_logits = &all_rust_logits[i];
            // compare_logits(rust_logits, logits, i)?;
        }

        Ok(())
    }
}

#[cfg(all(test, not(feature = "crossval")))]
#[test]
fn test_crossval_disabled() {
    println!("Cross-validation tests are disabled.");
    println!("To enable, run: cargo test --features crossval");
}
