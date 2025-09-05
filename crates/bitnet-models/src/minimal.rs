//! Minimal model loader for testing and CI
//!
//! This module provides a thin loader that can either load real weights
//! from GGUF files or generate deterministic dummy weights for testing.

use anyhow::{Context, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::Path;

/// Minimal weights containing only embedding and output layers
#[derive(Debug)]
pub struct MinimalWeights {
    /// Token embeddings flattened: \[vocab*dim\]
    pub tok_embeddings: Vec<f32>,
    /// Language model head flattened: \[dim*vocab\]
    pub lm_head: Vec<f32>,
    /// Vocabulary size
    pub vocab: usize,
    /// Hidden dimension size
    pub dim: usize,
}

/// Loading mode for minimal weights
pub enum LoadMode<'a> {
    /// Load from GGUF file (not yet implemented)
    Gguf(&'a Path),
    /// Generate deterministic dummy weights
    Dummy { vocab: usize, dim: usize },
}

/// Load minimal weights based on the specified mode
///
/// # Arguments
/// * `mode` - Either load from GGUF or generate dummy weights
///
/// # Returns
/// MinimalWeights structure with tok_embeddings and lm_head
pub fn load_minimal(mode: LoadMode) -> Result<MinimalWeights> {
    match mode {
        LoadMode::Dummy { vocab, dim } => {
            // Use deterministic RNG for reproducible results
            let mut rng = ChaCha8Rng::from_seed(*b"bitnet-mini-weights!bitnet-mini!");

            // Initialize embeddings with small random values
            let mut tok = vec![0f32; vocab * dim];
            for x in &mut tok {
                *x = rng.r#gen::<f32>() * 0.02 - 0.01;
            }

            // Initialize lm_head with small random values
            let mut head = vec![0f32; dim * vocab];
            for x in &mut head {
                *x = rng.r#gen::<f32>() * 0.02 - 0.01;
            }

            Ok(MinimalWeights { tok_embeddings: tok, lm_head: head, vocab, dim })
        }
        LoadMode::Gguf(path) => {
            // Load tensors from GGUF - now supports I2_S quantized tensors
            let two = crate::gguf_min::load_two(path)
                .with_context(|| format!("load tensors from {}", path.display()))?;
            Ok(MinimalWeights {
                tok_embeddings: two.tok_embeddings,
                lm_head: two.lm_head,
                vocab: two.vocab,
                dim: two.dim,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_weights_deterministic() {
        // Test that dummy weights are deterministic
        let w1 = load_minimal(LoadMode::Dummy { vocab: 100, dim: 64 }).unwrap();
        let w2 = load_minimal(LoadMode::Dummy { vocab: 100, dim: 64 }).unwrap();

        // Check dimensions
        assert_eq!(w1.vocab, 100);
        assert_eq!(w1.dim, 64);
        assert_eq!(w1.tok_embeddings.len(), 100 * 64);
        assert_eq!(w1.lm_head.len(), 64 * 100);

        // Check determinism - first few values should be identical
        for i in 0..10 {
            assert_eq!(w1.tok_embeddings[i], w2.tok_embeddings[i]);
            assert_eq!(w1.lm_head[i], w2.lm_head[i]);
        }

        // Check values are in expected range
        for &val in &w1.tok_embeddings[..10] {
            assert!(val.abs() <= 0.01);
        }
    }

    #[test]
    fn test_gguf_loader_placeholder() {
        // Test that GGUF loading returns expected error
        let path = Path::new("test.gguf");
        let result = load_minimal(LoadMode::Gguf(path));
        assert!(result.is_err());
        let err = result.unwrap_err();
        // Check that we get an error related to loading
        assert!(err.to_string().contains("load tensors") || err.to_string().contains("GGUF"));
    }
}
