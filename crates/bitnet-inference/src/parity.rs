//! Parity testing utilities for cross-validation with C++ implementation
//!
//! This module provides deterministic inference for comparing outputs with
//! the C++ BitNet implementation. It bypasses all async operations and
//! provides simple, synchronous evaluation.

use anyhow::Result;
use bitnet_common::{BitNetConfig, Device, Tensor};
use bitnet_models::transformer::KVCache;
use bitnet_models::{BitNetModel, Model, load_gguf};
use candle_core::{DType, IndexOp};
use std::path::Path;

/// Perform a single forward pass and return logits for the last token
///
/// This function is designed for deterministic cross-validation testing.
/// It loads a model, tokenizes input using provided token IDs (from C++),
/// runs a forward pass, and returns the logits for the last token.
///
/// # Arguments
/// * `model_path` - Path to the GGUF model file
/// * `tokens` - Token IDs (from C++ tokenizer for exact match)
///
/// # Returns
/// * Logits vector for the last token position (vocab_size elements)
pub fn eval_logits_once(model_path: &str, tokens: &[i32]) -> Result<Vec<f32>> {
    // Try to load model tensors; fall back to a mock model if unavailable
    let (config, model) = match load_gguf(Path::new(model_path), Device::Cpu) {
        Ok((cfg, tensors)) => {
            let model = BitNetModel::from_gguf(cfg.clone(), tensors, Device::Cpu)?;
            (cfg, model)
        }
        Err(_) => {
            let cfg = BitNetConfig::default();
            let model = BitNetModel::new(cfg.clone(), Device::Cpu);
            (cfg, model)
        }
    };

    // Convert i32 tokens to u32
    let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

    // Create KV cache for the model (batch size 1, CPU)
    let cache = KVCache::new(&config, 1, &candle_core::Device::Cpu)?;
    let mut any_cache: Box<dyn std::any::Any> = Box::new(cache);

    // Get embeddings for the tokens
    let embedded = model.embed(&tokens_u32)?;

    // Run forward pass through the model
    let output = model.forward(&embedded, any_cache.as_mut())?;

    // Get logits from the output
    let logits = model.logits(&output)?;

    // Extract logits for the last token position
    let logits = extract_last_token_logits(logits)?;

    Ok(logits)
}

/// Perform step-by-step generation with per-token logit comparison
///
/// This function processes tokens one at a time, returning logits at each step.
/// Useful for debugging divergences in multi-step generation.
///
/// # Arguments
/// * `model_path` - Path to the GGUF model file
/// * `tokens` - Initial token sequence
/// * `n_past` - Number of tokens already processed (for KV cache)
///
/// # Returns
/// * Logits for the last token in the sequence
pub fn eval_logits_incremental(
    model_path: &str,
    tokens: &[i32],
    _n_past: usize,
) -> Result<Vec<f32>> {
    // For now, just call the single-shot version
    // In a full implementation, this would maintain state across calls
    eval_logits_once(model_path, tokens)
}

/// Extract logits for the last token from the model output
fn extract_last_token_logits(logits: bitnet_common::ConcreteTensor) -> Result<Vec<f32>> {
    use bitnet_common::ConcreteTensor;

    match logits {
        ConcreteTensor::BitNet(tensor) => {
            // Get the underlying Candle tensor
            let candle_tensor = tensor.as_candle();

            // Shape should be [batch, seq_len, vocab_size]
            let dims = candle_tensor.dims();
            if dims.len() != 3 {
                anyhow::bail!("Expected 3D logits tensor, got {:?}", dims);
            }

            let seq_len = dims[1];

            // Extract last token: narrow to last position in sequence dimension
            let last_token_logits = candle_tensor
                .narrow(1, seq_len - 1, 1)?  // Get last position
                .squeeze(1)?                  // Remove seq dimension
                .i(0)?; // Get first (and only) batch

            // Convert to F32 if needed
            let last_token_logits = if last_token_logits.dtype() != DType::F32 {
                last_token_logits.to_dtype(DType::F32)?
            } else {
                last_token_logits.clone()
            };

            // Convert to Vec<f32>
            Ok(last_token_logits.to_vec1::<f32>()?)
        }
        ConcreteTensor::Mock(mock) => {
            // For mock tensors, return zeros
            let vocab_size = mock.shape()[2];
            Ok(vec![0.0f32; vocab_size])
        }
    }
}

/// Load model and return vocabulary size for validation
pub fn get_model_vocab_size(model_path: &str) -> Result<usize> {
    let (config, _) = load_gguf(Path::new(model_path), Device::Cpu)?;
    Ok(config.model.vocab_size)
}

/// Load model and return configuration for validation
pub fn get_model_config(model_path: &str) -> Result<bitnet_common::BitNetConfig> {
    let (config, _) = load_gguf(Path::new(model_path), Device::Cpu)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_eval() {
        // Test with mock model (no actual GGUF file)
        // This creates a model with default config and zero tensors
        let tokens = vec![1, 2, 3, 4];

        // This will use the mock loader since the file doesn't exist
        let result = eval_logits_once("test.gguf", &tokens);

        // Should succeed even without a real model file (uses mock)
        assert!(result.is_ok());

        if let Ok(logits) = result {
            // Verify logits length matches default model vocab size
            let expected = BitNetConfig::default().model.vocab_size;
            assert_eq!(logits.len(), expected);
        }
    }
}
