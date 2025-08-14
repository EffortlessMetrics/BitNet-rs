//! Simple forward pass implementation for minimal inference
//!
//! This module provides a basic implementation of a forward pass
//! using only embedding and lm_head layers, suitable for testing
//! and CI validation without full attention mechanisms.

use anyhow::{Context, Result};

/// Minimal weights structure containing only embedding and output layers
pub struct Weights<'a> {
    /// Token embeddings: [vocab, dim]
    pub tok_embeddings: &'a [f32],
    /// Language model head: [dim, vocab]
    pub lm_head: &'a [f32],
    /// Vocabulary size
    pub vocab: usize,
    /// Hidden dimension size
    pub dim: usize,
}

/// Compute logits for a single token using minimal forward pass
///
/// This performs:
/// 1. Token embedding lookup: e = tok_embeddings[token_id]
/// 2. Matrix multiplication: logits = e @ lm_head
///
/// # Arguments
/// * `w` - Model weights
/// * `token_id` - Input token ID (must be < vocab)
///
/// # Returns
/// Vector of logits of size `vocab`
pub fn logits_for_token(w: &Weights, token_id: usize) -> Vec<f32> {
    assert!(token_id < w.vocab, "token_id {} >= vocab {}", token_id, w.vocab);
    let mut out = vec![0f32; w.vocab];

    // e = tok_embeddings[token_id]  // [dim]
    let e = &w.tok_embeddings[token_id * w.dim..(token_id + 1) * w.dim];

    // out = e @ lm_head              // [vocab]
    // lm_head is [dim, vocab] row-major
    for v in 0..w.vocab {
        let mut acc = 0f32;
        let mut d = 0;

        // Unrolled loop for better performance
        while d + 4 <= w.dim {
            acc += e[d] * w.lm_head[d * w.vocab + v]
                + e[d + 1] * w.lm_head[(d + 1) * w.vocab + v]
                + e[d + 2] * w.lm_head[(d + 2) * w.vocab + v]
                + e[d + 3] * w.lm_head[(d + 3) * w.vocab + v];
            d += 4;
        }

        // Handle remaining elements
        while d < w.dim {
            acc += e[d] * w.lm_head[d * w.vocab + v];
            d += 1;
        }

        out[v] = acc;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logits_basic() {
        // Create tiny test weights
        let vocab = 4;
        let dim = 2;
        let tok_embeddings = vec![
            1.0, 0.0, // token 0
            0.0, 1.0, // token 1
            0.5, 0.5, // token 2
            -1.0, 1.0, // token 3
        ];
        let lm_head = vec![
            1.0, 0.0, -1.0, 0.5, // dim 0
            0.0, 1.0, 0.5, -0.5, // dim 1
        ];

        let w = Weights { tok_embeddings: &tok_embeddings, lm_head: &lm_head, vocab, dim };

        // Test token 0: [1.0, 0.0] @ [[1.0, 0.0, -1.0, 0.5], [0.0, 1.0, 0.5, -0.5]]
        // Expected: [1.0, 0.0, -1.0, 0.5]
        let logits = logits_for_token(&w, 0);
        assert_eq!(logits.len(), vocab);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[2] - (-1.0)).abs() < 1e-6);
    }
}
