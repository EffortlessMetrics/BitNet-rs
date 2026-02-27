//! CPU golden-path E2E integration test — Phase 5.1 of the dual-backend roadmap.
//!
//! This test suite runs on every PR with **no external model downloads**.
//! It exercises the full inference pipeline end-to-end using a tiny synthetic
//! model built entirely in-memory from deterministic weights.
//!
//! # Tests
//!
//! * `test_golden_path_e2e_basic` — always-runs smoke test: synthetic model +
//!   engine + receipt pipeline validates the happy path in < 1 s.
//!
//! * `test_golden_path_e2e_reproducible` — two independent runs with `seed=42`
//!   must produce bit-identical token sequences.
//!
//! * `test_golden_path_e2e_pinned_tokens` — greedy argmax on the fixed synthetic
//!   weights with `seed=42` must match pinned token IDs as a regression guard.
//!
//! * `test_golden_path_e2e_stop_token_halts_early` — a stop token ID must
//!   terminate generation before `max_tokens` is reached.
//!
//! * `test_golden_path_e2e_max_tokens_exact` — `max_tokens` is respected exactly
//!   (no over/under-generation) across small values 1–4.
//!
//! * `test_golden_path_e2e_tokens_in_vocab_range` — every emitted token ID must
//!   be strictly less than `vocab_size`.
//!
//! * `test_golden_path_e2e_receipt_schema_version` — the receipt schema version
//!   is the pinned constant `"1.0.0"`.
//!
//! * `test_golden_path_e2e_receipt_kernel_ids_constraints` — all kernel IDs
//!   satisfy the schema constraints (non-empty, ≤ 128 chars, count ≤ 10 000).
//!
//! * `test_golden_path_mini_gguf_fixture_structural_validity` — the committed
//!   `tests/models/mini.gguf` fixture is accessible and structurally valid.
//!
//! * `test_golden_path_e2e_real_model` — opt-in test, silently skipped unless
//!   `BITNET_MODEL_PATH` is set; exercises the real GGUF loader.

#![cfg(feature = "inference")]

use anyhow::Result;
use bitnet_common::{BitNetConfig, Device};
use bitnet_inference::{GenerationConfig, InferenceEngine, InferenceReceipt, KernelRecorder};
use bitnet_models::BitNetModel;
use bitnet_tokenizers::{MockTokenizer, Tokenizer as _};
use candle_core::Tensor as CandleTensor;
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Synthetic model builder (no file I/O, deterministic weights)
// ---------------------------------------------------------------------------

/// Construct a tiny 2-layer `BitNetModel` from deterministic sin/cos weights.
/// `vocab=512`, `hidden=256`, `layers=2`, `heads=4` — fast enough for < 1 s in CI.
fn synthetic_model() -> Result<Arc<BitNetModel>> {
    let mut cfg = BitNetConfig::default();
    cfg.model.vocab_size = 512;
    cfg.model.hidden_size = 256;
    cfg.model.num_layers = 2;
    cfg.model.num_heads = 4;
    cfg.model.num_key_value_heads = 4;
    cfg.model.intermediate_size = 512;
    cfg.model.max_position_embeddings = 128;

    let dev = candle_core::Device::Cpu;
    let v = cfg.model.vocab_size;
    let h = cfg.model.hidden_size;
    let i = cfg.model.intermediate_size;
    let mut t: HashMap<String, CandleTensor> = HashMap::new();

    let embed: Vec<f32> = (0..v * h).map(|x| (x as f32 * 0.001).sin()).collect();
    t.insert("token_embd.weight".into(), CandleTensor::from_vec(embed, &[v, h], &dev)?);
    let out: Vec<f32> = (0..v * h).map(|x| (x as f32 * 0.001 + 0.1).cos()).collect();
    t.insert("output.weight".into(), CandleTensor::from_vec(out, &[v, h], &dev)?);

    for l in 0..cfg.model.num_layers {
        let p = format!("layers.{l}");
        for name in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"] {
            let d: Vec<f32> = (0..h * h).map(|x| (x as f32 * 1e-4).sin()).collect();
            t.insert(format!("{p}.self_attn.{name}"), CandleTensor::from_vec(d, &[h, h], &dev)?);
        }
        for (name, r, c) in [("gate_proj", i, h), ("up_proj", i, h), ("down_proj", h, i)] {
            let d: Vec<f32> = (0..r * c).map(|x| (x as f32 * 1e-4).cos()).collect();
            t.insert(format!("{p}.mlp.{name}.weight"), CandleTensor::from_vec(d, &[r, c], &dev)?);
        }
        for norm in ["attention_norm", "ffn_norm"] {
            t.insert(
                format!("{p}.{norm}.weight"),
                CandleTensor::from_vec(vec![1.0f32; h], &[h], &dev)?,
            );
            t.insert(
                format!("{p}.{norm}.bias"),
                CandleTensor::from_vec(vec![0.0f32; h], &[h], &dev)?,
            );
        }
    }
    t.insert("final_norm.weight".into(), CandleTensor::from_vec(vec![1.0f32; h], &[h], &dev)?);

    Ok(Arc::new(BitNetModel::from_gguf(cfg, t, HashMap::new(), Device::Cpu)?))
}

// ---------------------------------------------------------------------------
// Basic smoke test + receipt invariants
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_golden_path_e2e_basic() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let recorder = KernelRecorder::new();

    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?
        .with_recorder(recorder.clone());

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    let tokens = engine.generate_tokens(&prompt_ids, &config).await?;

    assert_eq!(tokens.len(), 4, "must generate exactly 4 tokens");

    let kernel_ids = recorder.snapshot();
    assert!(!kernel_ids.is_empty(), "kernel recorder must capture at least one ID");
    assert!(
        !kernel_ids.iter().any(|k| k.contains("mock")),
        "no kernel ID may contain 'mock'; got {kernel_ids:?}"
    );

    let receipt = InferenceReceipt::generate("cpu-rust", kernel_ids, None)?;
    assert_eq!(receipt.compute_path, "real", "compute_path must be 'real'");
    assert!(!receipt.kernels.is_empty(), "receipt must carry kernel IDs");
    receipt.validate().map_err(|e| anyhow::anyhow!("receipt validation failed: {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Reproducibility
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_golden_path_e2e_reproducible() -> Result<()> {
    async fn run_once() -> Result<Vec<u32>> {
        let model = synthetic_model()?;
        let tokenizer = Arc::new(MockTokenizer::new());
        let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
        let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
        let prompt_ids = tokenizer.encode("2+2=", false, false)?;
        engine.generate_tokens(&prompt_ids, &config).await
    }

    let a = run_once().await?;
    let b = run_once().await?;
    assert_eq!(a, b, "greedy inference with seed=42 must be reproducible");
    assert_eq!(a.len(), 4);
    Ok(())
}

// ---------------------------------------------------------------------------
// Pinned regression guard
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_golden_path_e2e_pinned_tokens() -> Result<()> {
    // Captured from the first correct run; update only after an intentional change.
    const GOLDEN: &[u32] = &[140, 459, 459, 459];

    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    let tokens = engine.generate_tokens(&prompt_ids, &config).await?;

    assert_eq!(
        tokens.as_slice(),
        GOLDEN,
        "greedy output diverged from golden; inference pipeline may have regressed"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Stop token halts generation early
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_golden_path_e2e_stop_token_halts_early() -> Result<()> {
    // Pinned golden: [140, 459, 459, 459]; stop on 459 → only [140] emitted.
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
    let config =
        GenerationConfig::greedy().with_seed(42).with_max_tokens(10).with_stop_token_id(459);

    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    let tokens = engine.generate_tokens(&prompt_ids, &config).await?;

    assert_eq!(
        tokens,
        vec![140],
        "stop_token_id=459 must halt before emitting 459; got {tokens:?}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// max_tokens boundary
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_golden_path_e2e_max_tokens_exact() -> Result<()> {
    for &n in &[1u32, 2, 3, 4] {
        let model = synthetic_model()?;
        let tokenizer = Arc::new(MockTokenizer::new());
        let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
        let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(n);
        let prompt_ids = tokenizer.encode("2+2=", false, false)?;
        let tokens = engine.generate_tokens(&prompt_ids, &config).await?;
        assert_eq!(
            tokens.len(),
            n as usize,
            "max_tokens={n}: expected {n} tokens but got {}",
            tokens.len()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Vocab range
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_golden_path_e2e_tokens_in_vocab_range() -> Result<()> {
    const VOCAB: u32 = 512;
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(8);
    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    let tokens = engine.generate_tokens(&prompt_ids, &config).await?;

    assert!(!tokens.is_empty());
    for &id in &tokens {
        assert!(id < VOCAB, "token id {id} out of vocab range [0, {VOCAB})");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Receipt schema version
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_golden_path_e2e_receipt_schema_version() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let recorder = KernelRecorder::new();
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?
        .with_recorder(recorder.clone());
    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(2);
    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    engine.generate_tokens(&prompt_ids, &config).await?;

    let receipt = InferenceReceipt::generate("cpu-rust", recorder.snapshot(), None)?;
    assert_eq!(receipt.schema_version, "1.0.0");
    assert_eq!(receipt.schema_version, bitnet_receipts::RECEIPT_SCHEMA_VERSION);
    Ok(())
}

// ---------------------------------------------------------------------------
// Receipt kernel ID schema constraints
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_golden_path_e2e_receipt_kernel_ids_constraints() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let recorder = KernelRecorder::new();
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?
        .with_recorder(recorder.clone());
    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    engine.generate_tokens(&prompt_ids, &config).await?;

    let ids = recorder.snapshot();
    assert!(!ids.is_empty(), "at least one kernel ID must be recorded");
    assert!(ids.len() <= 10_000, "kernel count {} exceeds schema limit", ids.len());
    for id in &ids {
        assert!(!id.is_empty(), "kernel ID must not be empty");
        assert!(id.len() <= 128, "kernel ID '{id}' length {} exceeds schema limit", id.len());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// mini.gguf fixture structural validity
// ---------------------------------------------------------------------------

#[test]
fn test_golden_path_mini_gguf_fixture_structural_validity() {
    // Walk from the workspace tests/ dir to tests/models/mini.gguf.
    let manifest_dir = env!("CARGO_MANIFEST_DIR"); // …/tests
    let mini = std::path::Path::new(manifest_dir).join("models").join("mini.gguf");

    assert!(
        mini.exists(),
        "mini.gguf fixture must exist at tests/models/mini.gguf (resolved: {})",
        mini.display()
    );

    let bytes = std::fs::read(&mini).expect("mini.gguf must be readable");
    assert_eq!(&bytes[..4], b"GGUF", "mini.gguf must start with GGUF magic");

    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(version, 3, "mini.gguf must be GGUF version 3");

    let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    assert_eq!(tensor_count, 0, "mini.gguf has 0 tensors (metadata-only fixture)");

    let kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    assert_eq!(kv_count, 4, "mini.gguf has exactly 4 metadata entries");
}

// ---------------------------------------------------------------------------
// Real-model opt-in test (skipped in PR CI)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_golden_path_e2e_real_model() -> Result<()> {
    let model_path = match std::env::var("BITNET_MODEL_PATH").or(std::env::var("BITNET_GGUF")) {
        Ok(p) => p,
        Err(_) => return Ok(()), // silently skip when env is not set
    };

    let path = std::path::Path::new(&model_path);
    let loader = bitnet_models::ModelLoader::new(Device::Cpu);
    let model = loader.load(path)?;
    let tokenizer = bitnet_tokenizers::auto::load_auto(path, None)?;

    let recorder = KernelRecorder::new();
    let engine = InferenceEngine::new(model.into(), tokenizer.clone(), Device::Cpu)?
        .with_recorder(recorder.clone());

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    let prompt_ids = tokenizer.encode("2+2=", true, false)?;
    let tokens = engine.generate_tokens(&prompt_ids, &config).await?;

    assert!(!tokens.is_empty(), "real model must generate at least 1 token");

    let receipt = InferenceReceipt::generate("cpu-rust", recorder.snapshot(), None)?;
    assert_eq!(receipt.compute_path, "real");
    assert!(!receipt.kernels.is_empty());
    receipt.validate().map_err(|e| anyhow::anyhow!("receipt validation failed: {e}"))?;

    Ok(())
}
