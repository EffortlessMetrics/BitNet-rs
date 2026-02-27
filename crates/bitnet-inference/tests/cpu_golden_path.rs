//! Phase 5: CPU Golden Path End-to-End Test
//!
//! Always-on, no-model-download E2E test that proves:
//! 1. `InferenceEngine` generates tokens deterministically with temperature=0 (greedy)
//! 2. Receipt invariants hold: compute_path="real", kernel IDs valid, honest-compute passes
//! 3. Backend selection startup contract works
//!
//! Uses synthetic mock model/tokenizer — no model file needed.
#![cfg(feature = "cpu")]
use std::sync::Arc;

use anyhow::Result;
use bitnet_common::{BitNetConfig, ConcreteTensor, Device};
use bitnet_inference::{GenerationConfig, InferenceEngine, KernelRecorder};
use bitnet_models::Model;
use bitnet_receipts::InferenceReceipt;

// --- Minimal mock model ---

struct GreedyMockModel {
    config: BitNetConfig,
}

impl GreedyMockModel {
    fn new() -> Self {
        let mut cfg = BitNetConfig::default();
        cfg.model.vocab_size = 256;
        Self { config: cfg }
    }
}

impl Model for GreedyMockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<ConcreteTensor> {
        // Return mock logits shape [1, vocab_size]; tensor_to_logits falls back to uniform 0.1
        Ok(ConcreteTensor::mock(vec![1, self.config.model.vocab_size]))
    }

    fn embed(&self, _tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 4, 64]))
    }

    fn logits(&self, _hidden: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
        // Shape [B=1, T=1, V=256]; Mock branch in tensor_to_logits returns vec![0.1; vocab_size]
        Ok(ConcreteTensor::mock(vec![1, 1, self.config.model.vocab_size]))
    }
}

// --- Minimal mock tokenizer ---

struct GreedyMockTokenizer {
    vocab_size: usize,
}

impl GreedyMockTokenizer {
    fn new() -> Self {
        Self { vocab_size: 256 }
    }
}

impl bitnet_tokenizers::Tokenizer for GreedyMockTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> bitnet_common::Result<Vec<u32>> {
        // Encode each byte of the input as a token (capped at 8 tokens)
        Ok(text.bytes().take(8).map(|b| b as u32).collect())
    }

    fn decode(&self, tokens: &[u32]) -> bitnet_common::Result<String> {
        Ok(tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn token_to_piece(&self, token: u32) -> Option<String> {
        Some(format!("<{token}>"))
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(255) // EOS = 255; greedy picks 0, so no early termination
    }

    fn pad_token_id(&self) -> Option<u32> {
        None
    }
}

/// Helper: build engine + recorder, run greedy generation, return (tokens, kernel_ids).
async fn run_golden_path(max_new_tokens: usize) -> Result<(Vec<u32>, Vec<String>)> {
    let model = Arc::new(GreedyMockModel::new());
    let tokenizer = Arc::new(GreedyMockTokenizer::new());
    let recorder = KernelRecorder::new();

    let engine =
        InferenceEngine::new(model, tokenizer, Device::Cpu)?.with_recorder(recorder.clone());

    let input_tokens: Vec<u32> = b"hello".iter().map(|&b| b as u32).collect();
    let config =
        GenerationConfig::default().with_max_tokens(max_new_tokens as u32).with_temperature(0.0); // greedy → deterministic

    let generated = engine.generate_tokens(&input_tokens, &config).await?;
    let kernel_ids = recorder.snapshot();
    Ok((generated, kernel_ids))
}

/// Golden path: greedy generation on uniform logits always produces token 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cpu_golden_path_deterministic_output() -> Result<()> {
    let (tokens, _) = run_golden_path(5).await?;
    assert_eq!(tokens.len(), 5, "should generate exactly 5 tokens");
    // With uniform logits (all 0.1) and greedy sampling, argmax ties go to
    // the lowest token ID = 0.
    assert!(
        tokens.iter().all(|&t| t == 0),
        "greedy on uniform logits must always pick token 0; got {tokens:?}"
    );
    Ok(())
}

/// Golden path: two greedy runs produce identical output.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cpu_golden_path_reproducible() -> Result<()> {
    let (tokens1, _) = run_golden_path(4).await?;
    let (tokens2, _) = run_golden_path(4).await?;
    assert_eq!(tokens1, tokens2, "greedy generation must be reproducible");
    Ok(())
}

/// Golden path: kernel recorder captures the expected real kernel IDs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cpu_golden_path_kernel_ids_recorded() -> Result<()> {
    let (_, kernel_ids) = run_golden_path(2).await?;
    // These kernel IDs are recorded by InferenceEngine::forward_pass for every step.
    let required = ["embedding_lookup", "i2s_gemv", "rope_apply", "logits_projection"];
    for req in required {
        assert!(kernel_ids.iter().any(|k| k == req), "expected kernel '{req}' in {kernel_ids:?}");
    }
    Ok(())
}

/// Golden path: receipt passes honest-compute validation (compute_path = "real").
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cpu_golden_path_receipt_honest_compute() -> Result<()> {
    let (_, kernel_ids) = run_golden_path(3).await?;

    // None of the recorded kernel IDs contain "mock", so compute_path must be "real".
    let receipt = InferenceReceipt::generate("cpu-rust", kernel_ids, None)?;

    assert_eq!(receipt.compute_path, "real", "receipt.compute_path must be 'real'");

    // Full validation: schema + compute_path + kernel IDs
    receipt.validate().map_err(|e| anyhow::anyhow!("Receipt validation failed: {e}"))?;
    Ok(())
}

/// Golden path: receipt contains schema version and non-empty kernel list.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cpu_golden_path_receipt_invariants() -> Result<()> {
    let (_, kernel_ids) = run_golden_path(2).await?;
    assert!(!kernel_ids.is_empty(), "kernel_ids must not be empty after generation");

    let receipt = InferenceReceipt::generate("cpu-rust", kernel_ids, None)?;

    assert_eq!(receipt.schema_version, bitnet_receipts::RECEIPT_SCHEMA_VERSION);
    assert_eq!(receipt.backend, "cpu-rust");
    assert!(!receipt.kernels.is_empty(), "receipt must carry kernel IDs");
    Ok(())
}
