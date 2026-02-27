//! E2E golden-path test: deterministic CPU inference with receipt invariants (Phase 5.1).
//!
//! * `test_e2e_mock_golden_path` — always runs; uses a minimal synthetic-weight model
//!   (no download required) to prove the engine + receipt pipeline works end-to-end.
//!
//! * `test_e2e_golden_path_reproducible` — runs inference twice with the same seed and
//!   asserts identical token output, proving full determinism of the pipeline.
//!
//! * `test_e2e_golden_path_pinned_output` — pins specific token IDs produced by greedy
//!   decoding on the synthetic model (seed=42) as a regression guard.
//!
//! * `test_e2e_stop_token_id_halts_generation_early` — verifies that a configured stop
//!   token ID terminates generation before `max_tokens` is reached.
//!
//! * `test_e2e_receipt_kernel_ids_schema_constraints` — verifies all recorded kernel IDs
//!   satisfy the receipt schema constraints (non-empty, ≤ 128 chars, count ≤ 10 000).
//!
//! * `test_e2e_receipt_schema_version_is_1_0_0` — verifies the receipt schema version
//!   is the pinned constant "1.0.0".
//!
//! * `test_e2e_max_tokens_boundary` — verifies `max_tokens` is respected exactly when no
//!   stop token is encountered, across several small values (1–4).
//!
//! * `test_e2e_output_token_ids_in_vocab_range` — asserts that every generated token ID
//!   is in `[0, vocab_size)`, proving the engine never emits an out-of-vocabulary token.
//!
//! * `test_e2e_mini_gguf_fixture_accessible` — verifies the `tests/models/mini.gguf`
//!   fixture is accessible and structurally valid (GGUF v3, 0 tensors, 4 metadata keys).
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
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
// Reproducibility: same seed → same tokens
// ---------------------------------------------------------------------------

/// Two independent runs on the same synthetic model with seed=42 must produce
/// identical token sequences, proving the full pipeline is deterministic.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_golden_path_reproducible() -> Result<()> {
    async fn run_once(n: u32) -> Result<Vec<u32>> {
        let model = synthetic_model()?;
        let tokenizer = Arc::new(MockTokenizer::new());
        let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
        let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(n);
        let prompt_ids = tokenizer.encode("2+2=", false, false)?;
        engine.generate_tokens(&prompt_ids, &config).await
    }

    let tokens1 = run_once(4).await?;
    let tokens2 = run_once(4).await?;
    assert_eq!(tokens1, tokens2, "greedy inference must be reproducible across runs");
    assert_eq!(tokens1.len(), 4, "should generate exactly 4 tokens each run");
    Ok(())
}

// ---------------------------------------------------------------------------
// Pinned golden output: regression guard for specific token IDs
// ---------------------------------------------------------------------------

/// Greedy decoding on the fixed-weight synthetic model with seed=42 must produce
/// a stable sequence.  If this test breaks, the inference pipeline regressed.
///
/// Token IDs were captured from the first correct run and are intentionally pinned
/// here as a regression guard.  Update them only after a deliberate model/pipeline
/// change and re-capture.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_golden_path_pinned_output() -> Result<()> {
    // Greedy argmax on the fixed sin/cos synthetic weights, prompt "2+2=".
    // Captured from the first passing run; update only after an intentional change.
    const GOLDEN_TOKENS: &[u32] = &[140, 459, 459, 459];

    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    let tokens = engine.generate_tokens(&prompt_ids, &config).await?;

    assert_eq!(
        tokens.as_slice(),
        GOLDEN_TOKENS,
        "greedy output diverged from golden; inference pipeline may have regressed"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Stop token ID: early termination
// ---------------------------------------------------------------------------

/// A configured stop token ID must terminate generation before `max_tokens` is reached,
/// proving the stop-token logic is correctly wired into the E2E pipeline.
///
/// The pinned golden sequence with seed=42 is `[140, 459, 459, 459]`.
/// Setting stop_token_id=459 must cause the engine to stop after emitting 140,
/// because 459 is checked *before* it is appended to the output.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_stop_token_id_halts_generation_early() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
    let config = GenerationConfig::greedy()
        .with_seed(42)
        .with_max_tokens(10) // generous budget; stop token should fire first
        .with_stop_token_id(459); // second generated token in the pinned golden sequence

    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    let tokens = engine.generate_tokens(&prompt_ids, &config).await?;

    assert_eq!(
        tokens,
        vec![140],
        "stop_token_id=459 must halt generation before emitting 459; got {tokens:?}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Receipt kernel ID schema constraints
// ---------------------------------------------------------------------------

/// All kernel IDs recorded during a real inference pass must satisfy the schema
/// constraints required for honest-compute receipts: non-empty strings, at most
/// 128 characters each, and a total count ≤ 10 000.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_receipt_kernel_ids_schema_constraints() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let recorder = KernelRecorder::new();
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?
        .with_recorder(recorder.clone());

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    engine.generate_tokens(&prompt_ids, &config).await?;

    let kernel_ids = recorder.snapshot();
    assert!(!kernel_ids.is_empty(), "at least one kernel ID must be recorded");
    assert!(
        kernel_ids.len() <= 10_000,
        "kernel count {} exceeds schema limit of 10 000",
        kernel_ids.len()
    );
    for id in &kernel_ids {
        assert!(!id.is_empty(), "kernel ID must not be an empty string");
        assert!(
            id.len() <= 128,
            "kernel ID '{id}' length {} exceeds schema limit of 128",
            id.len()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Receipt schema version
// ---------------------------------------------------------------------------

/// The receipt schema version must always be the pinned literal "1.0.0" and
/// must match the `RECEIPT_SCHEMA_VERSION` constant exported by `bitnet_receipts`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_receipt_schema_version_is_1_0_0() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let recorder = KernelRecorder::new();
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?
        .with_recorder(recorder.clone());

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(2);
    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    engine.generate_tokens(&prompt_ids, &config).await?;

    let receipt = InferenceReceipt::generate("cpu-rust", recorder.snapshot(), None)?;
    assert_eq!(receipt.schema_version, "1.0.0", "receipt schema version must be fixed at '1.0.0'");
    assert_eq!(
        receipt.schema_version,
        bitnet_receipts::RECEIPT_SCHEMA_VERSION,
        "receipt schema version must match the RECEIPT_SCHEMA_VERSION constant"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Max-tokens boundary: exact token count
// ---------------------------------------------------------------------------

/// `max_tokens` must be respected exactly across small values when no stop token
/// is encountered, validating the generation loop termination condition.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_max_tokens_boundary() -> Result<()> {
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
// Vocab range: all generated token IDs must be in [0, vocab_size)
// ---------------------------------------------------------------------------

/// Every token ID produced by the engine must be a valid index into the
/// vocabulary, i.e. strictly less than `vocab_size`.  An out-of-bounds token
/// ID would corrupt any downstream detokenization step.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_e2e_output_token_ids_in_vocab_range() -> Result<()> {
    let model = synthetic_model()?;
    let vocab_size = 512u32; // matches the synthetic_model() configuration
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(8);
    let prompt_ids = tokenizer.encode("2+2=", false, false)?;
    let tokens = engine.generate_tokens(&prompt_ids, &config).await?;

    assert!(!tokens.is_empty(), "must generate at least one token");
    for &id in &tokens {
        assert!(id < vocab_size, "token id {id} is out of vocab range [0, {vocab_size})");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Mini GGUF fixture: structural sanity check
// ---------------------------------------------------------------------------

/// Verifies that the `tests/models/mini.gguf` fixture (committed to the repo)
/// is accessible from the inference-crate test context and contains the
/// expected GGUF v3 header with 0 tensors and 4 metadata keys.
///
/// This is a pure parsing test — the fixture has no model weights — but it
/// confirms the fixture path is correct and the GGUF reader handles it without
/// panicking, providing a sanity check for the shared test infrastructure.
#[test]
fn test_e2e_mini_gguf_fixture_accessible() {
    // CARGO_MANIFEST_DIR → crates/bitnet-inference; walk up two levels to workspace root.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mini_path = std::path::Path::new(manifest_dir)
        .join("..")
        .join("..")
        .join("tests")
        .join("models")
        .join("mini.gguf");

    assert!(
        mini_path.exists(),
        "mini.gguf fixture must exist at tests/models/mini.gguf (resolved: {})",
        mini_path.display()
    );

    let bytes = std::fs::read(&mini_path).expect("mini.gguf must be readable");
    // GGUF magic: 0x47 0x47 0x55 0x46 == "GGUF"
    assert_eq!(&bytes[..4], b"GGUF", "mini.gguf must start with GGUF magic");
    // Version is stored as a little-endian u32 at bytes 4..8; mini.gguf is v3.
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(version, 3, "mini.gguf must be GGUF version 3");
    // Tensor count at bytes 8..16 (u64 LE); the fixture has 0 tensors.
    let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    assert_eq!(tensor_count, 0, "mini.gguf has 0 tensors (metadata-only fixture)");
    // KV count at bytes 16..24 (u64 LE); the fixture has 4 metadata entries.
    let kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    assert_eq!(kv_count, 4, "mini.gguf has exactly 4 metadata entries");
}

// ---------------------------------------------------------------------------
// Real-model E2E test (skipped in PR CI)
// ---------------------------------------------------------------------------

/// Deterministic CPU inference with a real GGUF model.
///
/// Skipped unless `BITNET_MODEL_PATH` (or `BITNET_GGUF`) is set.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
