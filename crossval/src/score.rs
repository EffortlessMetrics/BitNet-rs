// Teacher-forcing perplexity evaluation for cross-validation
// Computes negative log-likelihood (NLL) and perplexity for exact parity testing

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Score output matching llama.cpp format for parity validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreOutput {
    /// Mean negative log-likelihood across all tokens
    pub mean_nll: f64,
    
    /// Perplexity (exp(mean_nll))
    pub perplexity: f64,
    
    /// Total number of tokens evaluated
    pub total_tokens: usize,
    
    /// Number of lines processed
    pub num_lines: usize,
    
    /// Tokens per second throughput
    pub tokens_per_second: f64,
    
    /// Wall-clock time in seconds
    pub elapsed_seconds: f64,
    
    /// Model configuration
    pub model_info: ModelInfo,
    
    /// Tokenizer configuration
    pub tokenizer_info: TokenizerInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_path: String,
    pub vocab_size: usize,
    pub context_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerInfo {
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
}

/// Numerically stable log-softmax computation
/// log_softmax(x) = x - log(sum(exp(x - max(x)))) - max(x)
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    
    // Compute exp(x - max) for numerical stability
    let exp_sum: f32 = logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .sum();
    
    let log_sum = exp_sum.ln() + max_logit;
    
    logits
        .iter()
        .map(|&x| x - log_sum)
        .collect()
}

/// Teacher-forcing evaluation loop
/// 
/// For each line in the input file:
/// 1. Tokenize with proper BOS/EOS handling
/// 2. Run forward pass with teacher forcing (use true tokens as input)
/// 3. Compute NLL for each predicted token
/// 4. Accumulate statistics
pub fn evaluate_perplexity<M, T>(
    model: &mut M,
    tokenizer: &T,
    input_path: &Path,
    max_context: usize,
) -> Result<ScoreOutput>
where
    M: Model,
    T: Tokenizer,
{
    let start_time = std::time::Instant::now();
    let file = File::open(input_path).context("Failed to open input file")?;
    let reader = BufReader::new(file);
    
    let mut total_nll = 0.0;
    let mut total_tokens = 0;
    let mut num_lines = 0;
    
    let vocab_size = model.vocab_size();
    let bos_id = tokenizer.bos_token_id();
    let eos_id = tokenizer.eos_token_id();
    let add_bos = tokenizer.add_bos();
    let add_eos = tokenizer.add_eos();
    
    for line in reader.lines() {
        let line = line.context("Failed to read line")?;
        if line.trim().is_empty() {
            continue;
        }
        
        // Tokenize with proper BOS/EOS handling
        let mut token_ids = tokenizer.encode(&line, add_bos, add_eos)?;
        
        // Skip if too long for context
        if token_ids.len() > max_context {
            eprintln!(
                "Warning: Skipping line with {} tokens (exceeds context {})",
                token_ids.len(),
                max_context
            );
            continue;
        }
        
        // Teacher forcing: for each position, predict next token
        for i in 0..token_ids.len().saturating_sub(1) {
            // Input: tokens[0..=i], target: tokens[i+1]
            let input_slice = &token_ids[0..=i];
            let target_token = token_ids[i + 1];
            
            // Forward pass to get logits
            let logits = model.forward(input_slice, 0)?;
            
            // Get logits for the last position (predicting next token)
            let last_pos = input_slice.len() - 1;
            let logits_at_pos = logits.slice(last_pos, vocab_size)?;
            
            // Compute log probabilities
            let log_probs = log_softmax(&logits_at_pos);
            
            // Get NLL for the target token
            let nll = -log_probs[target_token as usize];
            
            // Validate NLL is finite
            if !nll.is_finite() {
                anyhow::bail!(
                    "Non-finite NLL at position {} for token {}: {}",
                    i,
                    target_token,
                    nll
                );
            }
            
            total_nll += nll as f64;
            total_tokens += 1;
        }
        
        num_lines += 1;
    }
    
    if total_tokens == 0 {
        anyhow::bail!("No tokens evaluated - check input file");
    }
    
    let mean_nll = total_nll / total_tokens as f64;
    let perplexity = mean_nll.exp();
    let elapsed = start_time.elapsed().as_secs_f64();
    let tokens_per_second = total_tokens as f64 / elapsed;
    
    Ok(ScoreOutput {
        mean_nll,
        perplexity,
        total_tokens,
        num_lines,
        tokens_per_second,
        elapsed_seconds: elapsed,
        model_info: ModelInfo {
            model_path: model.model_path().to_string(),
            vocab_size,
            context_length: max_context,
        },
        tokenizer_info: TokenizerInfo {
            bos_token_id: bos_id,
            eos_token_id: eos_id,
            add_bos,
            add_eos,
        },
    })
}

/// Traits that model and tokenizer must implement
pub trait Model {
    fn forward(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Box<dyn Logits>>;
    fn vocab_size(&self) -> usize;
    fn model_path(&self) -> &str;
}

pub trait Tokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Result<Vec<u32>>;
    fn bos_token_id(&self) -> Option<u32>;
    fn eos_token_id(&self) -> Option<u32>;
    fn add_bos(&self) -> bool;
    fn add_eos(&self) -> bool;
}

/// Logits tensor abstraction
pub trait Logits {
    fn slice(&self, position: usize, length: usize) -> Result<Vec<f32>>;
}

/// Parity validation against llama.cpp
pub fn validate_parity(
    rust_output: &ScoreOutput,
    cpp_output: &ScoreOutput,
    tolerance: f64,
) -> Result<()> {
    let nll_diff = (rust_output.mean_nll - cpp_output.mean_nll).abs();
    let ppl_diff = (rust_output.perplexity - cpp_output.perplexity).abs();
    
    if nll_diff > tolerance {
        anyhow::bail!(
            "NLL parity failed: Rust={:.6}, C++={:.6}, diff={:.6} > tolerance={:.6}",
            rust_output.mean_nll,
            cpp_output.mean_nll,
            nll_diff,
            tolerance
        );
    }
    
    // Perplexity tolerance scales with magnitude
    let ppl_tolerance = cpp_output.perplexity * tolerance;
    if ppl_diff > ppl_tolerance {
        anyhow::bail!(
            "Perplexity parity failed: Rust={:.6}, C++={:.6}, diff={:.6} > tolerance={:.6}",
            rust_output.perplexity,
            cpp_output.perplexity,
            ppl_diff,
            ppl_tolerance
        );
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_log_softmax_numerical_stability() {
        // Test with large values that would overflow without stability
        let logits = vec![1000.0, 1001.0, 999.0];
        let log_probs = log_softmax(&logits);
        
        // Should sum to approximately 1 when exp'd
        // Using 1e-4 tolerance for f32 precision
        let sum: f32 = log_probs.iter().map(|&lp| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-4, "Sum: {}, expected ~1.0", sum);
        
        // Highest logit should have highest probability
        let max_idx = log_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 1); // Index of 1001.0
    }
    
    #[test]
    fn test_log_softmax_uniform() {
        // Test with uniform logits
        let logits = vec![0.0; 100];
        let log_probs = log_softmax(&logits);
        
        let expected = -(100.0_f32.ln());
        for &lp in &log_probs {
            assert!((lp - expected).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_parity_validation() {
        let rust = ScoreOutput {
            mean_nll: 2.3456,
            perplexity: 10.4412,
            total_tokens: 1000,
            num_lines: 50,
            tokens_per_second: 150.0,
            elapsed_seconds: 6.67,
            model_info: ModelInfo {
                model_path: "test.gguf".to_string(),
                vocab_size: 32000,
                context_length: 2048,
            },
            tokenizer_info: TokenizerInfo {
                bos_token_id: Some(1),
                eos_token_id: Some(2),
                add_bos: true,
                add_eos: false,
            },
        };
        
        let cpp = rust.clone();
        
        // Should pass with identical outputs
        assert!(validate_parity(&rust, &cpp, 0.01).is_ok());
        
        // Should fail with large difference
        let mut cpp_bad = cpp.clone();
        cpp_bad.mean_nll = 2.5;
        assert!(validate_parity(&rust, &cpp_bad, 0.01).is_err());
    }
}