//! E2E golden-path test: deterministic CPU inference with receipt invariants (Phase 5.1).
//!
//! * `test_e2e_mock_golden_path` — always runs; uses a minimal synthetic-weight model
//!   (no download required) to prove the engine + receipt pipeline works end-to-end.
//!
//! * `test_e2e_real_model_golden_path` — skipped in PR CI; run locally with a real model:
//!   ```sh
//!   BITNET_MODEL_PATH=models/model.gguf \
//!   cargo nextest run -p bitnet-inference --no-default-features --features cpu \
//!     --run-ignored all -E 'test(e2e_)'
//!   ```
#![cfg(feature = "cpu")]

use anyhow::Result;
use bitnet_common::{BitNetConfig, Device};
use bitnet_inference::{GenerationConfig, InferenceEngine, InferenceReceipt, KernelRecorder};
use bitnet_models::BitNetModel;
use bitnet_tokenizers::{MockTokenizer, Tokenizer as _};
use candle_core::Tensor as CandleTensor;
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a tiny 2-layer BitNetModel from synthetic weights — no file I/O.
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
// Always-runs E2E test (no model download needed)
// ---------------------------------------------------------------------------

/// Deterministic CPU inference using a synthetic-weight model.
/// Verifies output shape and receipt invariants without any file download.
#[tokio::test]
async fn test_e2e_mock_golden_path() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let recorder = KernelRecorder::new();

    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?
        .with_recorder(recorder.clone());

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    let tokens = engine.generate_tokens(&prompt_ids, &config).await?;

    assert_eq!(tokens.len(), 4, "should generate exactly 4 tokens");

    let kernel_ids = recorder.snapshot();
    assert!(!kernel_ids.is_empty(), "kernel recorder must capture at least one kernel ID");
    assert!(
        !kernel_ids.iter().any(|k| k.contains("mock")),
        "no kernel ID should contain 'mock'; got {kernel_ids:?}"
    );

    let receipt = InferenceReceipt::generate("cpu-rust", kernel_ids, None)?;
    assert_eq!(receipt.compute_path, "real", "receipt.compute_path must be 'real'");
    assert!(!receipt.kernels.is_empty(), "receipt must carry kernel IDs");
    receipt.validate().map_err(|e| anyhow::anyhow!("receipt validation failed: {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Real-model E2E test (skipped in PR CI)
// ---------------------------------------------------------------------------

/// Deterministic CPU inference with a real GGUF model.
///
/// Skipped unless `BITNET_MODEL_PATH` (or `BITNET_GGUF`) is set.
#[tokio::test]
#[ignore = "requires real model: set BITNET_MODEL_PATH env var"]
async fn test_e2e_real_model_golden_path() -> Result<()> {
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

    let kernel_ids = recorder.snapshot();
    assert!(!kernel_ids.is_empty(), "kernel recorder must capture at least one kernel ID");

    let receipt = InferenceReceipt::generate("cpu-rust", kernel_ids, None)?;
    assert_eq!(receipt.compute_path, "real", "receipt.compute_path must be 'real'");
    assert!(!receipt.kernels.is_empty(), "receipt must carry kernel IDs");
    receipt.validate().map_err(|e| anyhow::anyhow!("receipt validation failed: {e}"))?;

    Ok(())
}
