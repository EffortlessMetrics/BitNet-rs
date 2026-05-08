//! Python-to-Rust BitNet inference migration example.
//!
//! The legacy Python version returned untyped dictionaries and performed the
//! decode loop with runtime checks. This Rust version keeps the same observable
//! shape while using typed configuration, typed results, ownership-based model
//! lifetime management, and `Result`-based errors.

use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitNetModel {
    model_path: String,
    eos_token: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationConfig {
    pub max_tokens: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { max_tokens: 100 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationStats {
    pub text: String,
    pub tokens: usize,
    pub elapsed: Duration,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceError {
    EmptyModelPath,
    EmptyPrompt,
    InvalidMaxTokens,
}

impl fmt::Display for InferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyModelPath => formatter.write_str("model_path cannot be empty"),
            Self::EmptyPrompt => formatter.write_str("prompt cannot be empty"),
            Self::InvalidMaxTokens => formatter.write_str("max_tokens must be positive"),
        }
    }
}

impl Error for InferenceError {}

impl BitNetModel {
    pub fn load(model_path: impl Into<String>) -> Result<Self, InferenceError> {
        let model_path = model_path.into();
        if model_path.trim().is_empty() {
            return Err(InferenceError::EmptyModelPath);
        }

        Ok(Self { model_path, eos_token: 0 })
    }

    fn tokenize(&self, prompt: &str) -> Vec<u32> {
        prompt.split_whitespace().map(|part| part.len() as u32).collect()
    }

    fn forward_next_token(&self, tokens: &[u32]) -> u32 {
        tokens.iter().sum::<u32>() % 31 + 1
    }

    fn detokenize(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|token| format!("token_{token}")).collect::<Vec<_>>().join(" ")
    }
}

#[derive(Debug)]
pub struct BitNetInference {
    model: BitNetModel,
}

impl BitNetInference {
    pub fn new(model_path: impl Into<String>) -> Result<Self, InferenceError> {
        Ok(Self { model: BitNetModel::load(model_path)? })
    }

    pub fn generate(
        &self,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<GenerationStats, InferenceError> {
        if prompt.trim().is_empty() {
            return Err(InferenceError::EmptyPrompt);
        }
        if config.max_tokens == 0 {
            return Err(InferenceError::InvalidMaxTokens);
        }

        let started_at = Instant::now();
        let mut context_tokens = self.model.tokenize(prompt);
        let mut output_tokens = Vec::with_capacity(config.max_tokens);

        for _ in 0..config.max_tokens {
            let next_token = self.model.forward_next_token(&context_tokens);
            output_tokens.push(next_token);
            context_tokens.push(next_token);

            if next_token == self.model.eos_token {
                break;
            }
        }

        let elapsed = started_at.elapsed();
        let tokens = output_tokens.len();
        let tokens_per_second =
            if elapsed.is_zero() { 0.0 } else { tokens as f64 / elapsed.as_secs_f64() };

        Ok(GenerationStats {
            text: self.model.detokenize(&output_tokens),
            tokens,
            elapsed,
            tokens_per_second,
        })
    }
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    let model = BitNetInference::new("model.gguf")?;
    let result = model.generate("The future of AI is", GenerationConfig::default())?;

    println!("Generated: {}", result.text);
    println!("Tokens: {}", result.tokens);
    println!("Speed: {:.1} tok/s", result.tokens_per_second);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_model_path() {
        let error = BitNetInference::new("   ").unwrap_err();
        assert_eq!(error, InferenceError::EmptyModelPath);
    }

    #[test]
    fn rejects_empty_prompt() {
        let model = BitNetInference::new("model.gguf").unwrap();
        let error = model.generate(" ", GenerationConfig::default()).unwrap_err();
        assert_eq!(error, InferenceError::EmptyPrompt);
    }

    #[test]
    fn rejects_zero_max_tokens() {
        let model = BitNetInference::new("model.gguf").unwrap();
        let error = model.generate("hello", GenerationConfig { max_tokens: 0 }).unwrap_err();
        assert_eq!(error, InferenceError::InvalidMaxTokens);
    }

    #[test]
    fn generates_typed_stats() {
        let model = BitNetInference::new("model.gguf").unwrap();
        let result =
            model.generate("The future of AI is", GenerationConfig { max_tokens: 4 }).unwrap();

        assert_eq!(result.tokens, 4);
        assert!(!result.text.is_empty());
        assert!(result.text.split_whitespace().all(|part| part.starts_with("token_")));
    }
}
