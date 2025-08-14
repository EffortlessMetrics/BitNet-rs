//! Minimal inference example for BitNet models
//!
//! This example demonstrates how to produce real logits using minimal forward pass.
//! Set BITNET_GGUF=/path/to/model.gguf to use real weights (once implemented).
//! Set BITNET_TOKEN_ID to specify which token to score (default: 1).

use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let maybe = env::var_os("BITNET_GGUF").map(PathBuf::from);

    // Pick a token id to score (e.g., 'Hello')
    let token_id: usize = env::var("BITNET_TOKEN_ID")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // Load minimal weights
    let mw = match maybe {
        Some(p) => {
            match bitnet_models::minimal::load_minimal(bitnet_models::minimal::LoadMode::Gguf(&p)) {
                Ok(w) => w,
                Err(e) => {
                    eprintln!(
                        "[info] GGUF not ready yet ({}). Falling back to dummy weights.",
                        e
                    );
                    bitnet_models::minimal::load_minimal(bitnet_models::minimal::LoadMode::Dummy {
                        vocab: 32000,
                        dim: 1024,
                    })?
                }
            }
        }
        None => {
            eprintln!("Set BITNET_GGUF=/path/to/model.gguf to use real weights; using dummy.");
            bitnet_models::minimal::load_minimal(bitnet_models::minimal::LoadMode::Dummy {
                vocab: 32000,
                dim: 1024,
            })?
        }
    };

    // Create weights reference
    let w = bitnet_inference::simple_forward::Weights {
        tok_embeddings: &mw.tok_embeddings,
        lm_head: &mw.lm_head,
        vocab: mw.vocab,
        dim: mw.dim,
    };

    // Compute logits
    let logits = bitnet_inference::simple_forward::logits_for_token(&w, token_id);

    // Print first 8 logits for verification
    println!("logits[0..8]={:?}", &logits[..8.min(logits.len())]);

    // Also print some statistics
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_logit = logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    println!("token_id={}, vocab={}, dim={}", token_id, mw.vocab, mw.dim);
    println!("max_logit={:.4}, min_logit={:.4}", max_logit, min_logit);

    Ok(())
}
