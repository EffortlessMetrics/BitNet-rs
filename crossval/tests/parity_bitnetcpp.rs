//! BitNet.cpp parity harness
//!
//! Validates that the Rust inference engine produces identical outputs to
//! Microsoft's BitNet C++ implementation for deterministic inference.
//!
//! This test is feature-gated and skips gracefully when:
//! - `crossval-bitnetcpp` feature is not enabled
//! - `CROSSVAL_GGUF` environment variable is not set
//! - BitNet C++ is not available

#![cfg(all(feature = "crossval", feature = "integration-tests"))]

use anyhow::{Context, Result};
use serde_json::json;
use std::{env, fs, path::PathBuf, time::SystemTime};

/// Helper function to compute cosine similarity between two vectors
#[allow(dead_code)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[test]
fn parity_bitnetcpp() -> Result<()> {
    // Check if CROSSVAL_GGUF is set; skip if not
    let gguf_path = match env::var("CROSSVAL_GGUF") {
        Ok(s) => PathBuf::from(s),
        Err(_) => {
            eprintln!("CROSSVAL_GGUF not set; skipping parity test");
            return Ok(());
        }
    };

    if !gguf_path.exists() {
        eprintln!("GGUF model not found at {:?}; skipping", gguf_path);
        return Ok(());
    }

    let commit = env::var("GIT_COMMIT").unwrap_or_else(|_| "unknown".into());
    let prompt = env::var("CROSSVAL_PROMPT").unwrap_or_else(|_| "Q: 2+2? A:".into());

    // TODO: Implement actual parity checks when bitnet-sys FFI is ready
    // For now, this is a placeholder that verifies the infrastructure works

    eprintln!("Parity test infrastructure ready");
    eprintln!("Model: {:?}", gguf_path);
    eprintln!("Prompt: {}", prompt);
    eprintln!("Commit: {}", commit);

    // Placeholder receipt - will be replaced with real parity results
    let ts = humantime::format_rfc3339(SystemTime::now()).to_string();
    let date_dir = format!("docs/baselines/{}", chrono::Local::now().format("%Y-%m-%d"));
    let receipt_dir = PathBuf::from(&date_dir);

    // Create baselines directory if it doesn't exist
    if !receipt_dir.exists() {
        fs::create_dir_all(&receipt_dir).context("Failed to create baselines directory")?;
    }

    let receipt = json!({
        "timestamp": ts,
        "commit": commit,
        "model_path": gguf_path.display().to_string(),
        "seed": 0,
        "threads": 1,
        "template": "auto",
        "status": "infrastructure_ready",
        "note": "Parity harness infrastructure in place; awaiting bitnet-sys FFI implementation"
    });

    let receipt_path = receipt_dir.join("parity-bitnetcpp.json");
    fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)
        .context("Failed to write parity receipt")?;

    eprintln!("Parity receipt written to: {:?}", receipt_path);

    Ok(())
}

// TODO: Implement when bitnet-sys FFI is complete:
// fn rust_side_tokenize_and_meta(prompt: &str) -> Result<(Vec<i32>, bool, bool, i32, usize)> {
//     // Call bitnet-tokenizers to tokenize with template-aware BOS/add_special
//     // Return (token_ids, add_bos, add_special, eos_token_id, logits_dimension)
//     unimplemented!("Awaiting bitnet-tokenizers integration")
// }
//
// fn rust_eval_last_logits(ids: &[i32], logits_dim: usize) -> Result<Vec<f32>> {
//     // Call bitnet-inference to evaluate and return last position logits
//     unimplemented!("Awaiting bitnet-inference integration")
// }
//
// fn rust_decode_n_greedy(ids: &[i32], n_steps: usize, eos_id: i32) -> Result<Vec<i32>> {
//     // Call bitnet-inference to perform N-step greedy decoding
//     unimplemented!("Awaiting bitnet-inference greedy decode")
// }
