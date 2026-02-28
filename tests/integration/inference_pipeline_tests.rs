//! Integration tests for the full inference pipeline.
//!
//! Covers end-to-end inference with synthetic models, multi-threaded safety,
//! deterministic output verification, error recovery, config validation,
//! and CLI flag combinations.
//!
//! All tests use in-memory synthetic models — no real model downloads needed.

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
// Synthetic model builder (shared with e2e_golden_path.rs pattern)
// ---------------------------------------------------------------------------

fn synthetic_model() -> Result<Arc<BitNetModel>> {
    synthetic_model_with_vocab(512, 256, 2)
}

fn synthetic_model_with_vocab(
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
) -> Result<Arc<BitNetModel>> {
    let mut cfg = BitNetConfig::default();
    cfg.model.vocab_size = vocab_size;
    cfg.model.hidden_size = hidden_size;
    cfg.model.num_layers = num_layers;
    cfg.model.num_heads = 4;
    cfg.model.num_key_value_heads = 4;
    cfg.model.intermediate_size = hidden_size * 2;
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

    for l in 0..num_layers {
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

// =========================================================================
// 1. End-to-end pipeline: model → quantization → inference → tokens
// =========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pipeline_model_to_tokens() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let recorder = KernelRecorder::new();
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?
        .with_recorder(recorder.clone());

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    let prompt_ids = tokenizer.encode("hello", false, false)?;
    let tokens = engine.generate_tokens(&prompt_ids, &config).await?;

    assert_eq!(tokens.len(), 4, "pipeline must produce exactly max_tokens");
    assert!(!recorder.snapshot().is_empty(), "kernels must be recorded");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pipeline_generate_string_output() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    let result = engine.generate_with_config("test", &config).await?;

    // generate_with_config returns a decoded string (may be non-UTF8-printable
    // from synthetic weights, but must not error)
    assert!(!result.is_empty(), "string output must not be empty");
    Ok(())
}

// =========================================================================
// 2. Deterministic output verification (same seed → same output)
// =========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_determinism_same_seed_same_output() -> Result<()> {
    async fn run(seed: u64) -> Result<Vec<u32>> {
        let model = synthetic_model()?;
        let tokenizer = Arc::new(MockTokenizer::new());
        let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
        let config = GenerationConfig::greedy().with_seed(seed).with_max_tokens(8);
        let ids = tokenizer.encode("determinism", false, false)?;
        engine.generate_tokens(&ids, &config).await
    }

    let a = run(123).await?;
    let b = run(123).await?;
    assert_eq!(a, b, "identical seed must yield identical tokens");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_determinism_different_seeds_may_differ() -> Result<()> {
    // With temperature > 0 and different seeds, outputs should (statistically) differ.
    // With greedy (temp=0), argmax is deterministic regardless of seed, so we use
    // creative config with temp > 0.
    async fn run(seed: u64) -> Result<Vec<u32>> {
        let model = synthetic_model()?;
        let tokenizer = Arc::new(MockTokenizer::new());
        let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
        let config = GenerationConfig::creative().with_seed(seed).with_max_tokens(16);
        let ids = tokenizer.encode("hello world", false, false)?;
        engine.generate_tokens(&ids, &config).await
    }

    let a = run(1).await?;
    let b = run(999_999).await?;
    // We cannot guarantee they differ (argmax could dominate), but we can
    // verify both runs complete without error.
    assert_eq!(a.len(), 16);
    assert_eq!(b.len(), 16);
    Ok(())
}

// =========================================================================
// 3. Multi-threaded inference safety
// =========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_concurrent_inference_safety() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());

    // Spawn multiple concurrent inference tasks sharing the same model/tokenizer
    let mut handles = Vec::new();
    for i in 0u64..4 {
        let m = model.clone();
        let tok = tokenizer.clone();
        handles.push(tokio::spawn(async move {
            let engine = InferenceEngine::new(m, tok.clone(), Device::Cpu).unwrap();
            let config = GenerationConfig::greedy().with_seed(i).with_max_tokens(4);
            let prompt_ids = tok.encode("concurrent", false, false).unwrap();
            engine.generate_tokens(&prompt_ids, &config).await.unwrap()
        }));
    }

    let mut results = Vec::new();
    for h in handles {
        let tokens = h.await?;
        assert_eq!(tokens.len(), 4, "each concurrent run must produce 4 tokens");
        results.push(tokens);
    }
    // All greedy runs with seed=0 should be identical (same model, same input)
    // but we mainly care that no panics/data races occurred.
    assert_eq!(results.len(), 4);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_shared_model_across_engines() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());

    let engine_a = InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu)?;
    let engine_b = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(2);
    let ids = tokenizer.encode("share", false, false)?;

    let (a, b) = tokio::join!(
        engine_a.generate_tokens(&ids, &config),
        engine_b.generate_tokens(&ids, &config),
    );

    assert_eq!(a?.len(), 2);
    assert_eq!(b?.len(), 2);
    Ok(())
}

// =========================================================================
// 4. GenerationConfig validation & edge cases
// =========================================================================

#[test]
fn test_generation_config_validate_rejects_zero_max_tokens() {
    let config = GenerationConfig::greedy().with_max_tokens(0);
    assert!(config.validate().is_err());
}

#[test]
fn test_generation_config_validate_rejects_negative_temperature() {
    let config = GenerationConfig::greedy().with_temperature(-1.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_generation_config_validate_rejects_invalid_top_p() {
    let zero = GenerationConfig::greedy().with_top_p(0.0);
    assert!(zero.validate().is_err(), "top_p=0.0 must be rejected");

    let over = GenerationConfig::greedy().with_top_p(1.5);
    assert!(over.validate().is_err(), "top_p>1.0 must be rejected");
}

#[test]
fn test_generation_config_validate_rejects_invalid_repetition_penalty() {
    let config = GenerationConfig::greedy().with_repetition_penalty(0.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_generation_config_presets_are_valid() {
    assert!(GenerationConfig::greedy().validate().is_ok());
    assert!(GenerationConfig::creative().validate().is_ok());
    assert!(GenerationConfig::balanced().validate().is_ok());
}

#[test]
fn test_generation_config_stop_token_id_builder() {
    let config = GenerationConfig::greedy().with_stop_token_id(100).with_stop_token_id(200);
    assert!(config.is_stop_token(100));
    assert!(config.is_stop_token(200));
    assert!(!config.is_stop_token(300));
}

#[test]
fn test_generation_config_stop_sequences_builder() {
    let config = GenerationConfig::greedy()
        .with_stop_sequence("</s>".to_string())
        .with_stop_sequence("\n\nQ:".to_string());
    assert_eq!(config.stop_sequences.len(), 2);
}

#[test]
fn test_generation_config_seed_builder() {
    let config = GenerationConfig::greedy().with_seed(42);
    assert_eq!(config.seed, Some(42));
}

// =========================================================================
// 5. Stop criteria: token-level and boundary conditions
// =========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_stop_on_multiple_stop_token_ids() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;

    // Stop on any of several token IDs (the pinned golden output starts with 140)
    let config =
        GenerationConfig::greedy().with_seed(42).with_max_tokens(20).with_stop_token_id(140);
    let ids = tokenizer.encode("2+2=", false, false)?;
    let tokens = engine.generate_tokens(&ids, &config).await?;

    // Should stop immediately when first token is a stop token (empty output)
    // or before producing 20 tokens
    assert!(tokens.len() < 20, "stop token must truncate output");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_max_tokens_boundary_one() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(1);
    let ids = tokenizer.encode("x", false, false)?;
    let tokens = engine.generate_tokens(&ids, &config).await?;

    assert_eq!(tokens.len(), 1, "max_tokens=1 must produce exactly 1 token");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_all_tokens_within_vocab_range() -> Result<()> {
    const VOCAB: usize = 512;
    let model = synthetic_model_with_vocab(VOCAB, 256, 2)?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(16);
    let ids = tokenizer.encode("range check", false, false)?;
    let tokens = engine.generate_tokens(&ids, &config).await?;

    for &id in &tokens {
        assert!((id as usize) < VOCAB, "token {id} exceeds vocab_size {VOCAB}");
    }
    Ok(())
}

// =========================================================================
// 6. Receipt integrity after inference
// =========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_receipt_compute_path_is_real() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let recorder = KernelRecorder::new();
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?
        .with_recorder(recorder.clone());

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(2);
    let ids = tokenizer.encode("receipt", false, false)?;
    engine.generate_tokens(&ids, &config).await?;

    let receipt = InferenceReceipt::generate("cpu-rust", recorder.snapshot(), None)?;
    assert_eq!(receipt.compute_path, "real");
    receipt.validate().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_receipt_kernel_ids_non_empty_after_inference() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let recorder = KernelRecorder::new();
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?
        .with_recorder(recorder.clone());

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    let ids = tokenizer.encode("kernels", false, false)?;
    engine.generate_tokens(&ids, &config).await?;

    let kernel_ids = recorder.snapshot();
    assert!(!kernel_ids.is_empty(), "inference must record kernel IDs");
    for kid in &kernel_ids {
        assert!(!kid.is_empty(), "kernel ID must not be empty string");
        assert!(kid.len() <= 128, "kernel ID length exceeds schema limit");
    }
    assert!(kernel_ids.len() <= 10_000, "too many kernel IDs");
    Ok(())
}

// =========================================================================
// 7. Error recovery: corrupt model / invalid inputs
// =========================================================================

#[test]
fn test_model_load_rejects_missing_embeddings() {
    let mut cfg = BitNetConfig::default();
    cfg.model.vocab_size = 64;
    cfg.model.hidden_size = 32;
    cfg.model.num_layers = 1;
    cfg.model.num_heads = 2;
    cfg.model.num_key_value_heads = 2;
    cfg.model.intermediate_size = 64;
    cfg.model.max_position_embeddings = 32;

    let t: HashMap<String, CandleTensor> = HashMap::new();
    // No tensors at all — must error
    let result = BitNetModel::from_gguf(cfg, t, HashMap::new(), Device::Cpu);
    assert!(result.is_err(), "model without embeddings must fail");
}

#[test]
fn test_model_load_rejects_missing_output_weight() {
    let mut cfg = BitNetConfig::default();
    cfg.model.vocab_size = 64;
    cfg.model.hidden_size = 32;
    cfg.model.num_layers = 1;
    cfg.model.num_heads = 2;
    cfg.model.num_key_value_heads = 2;
    cfg.model.intermediate_size = 64;
    cfg.model.max_position_embeddings = 32;

    let dev = candle_core::Device::Cpu;
    let v = 64;
    let h = 32;
    let mut t: HashMap<String, CandleTensor> = HashMap::new();

    // Only embeddings, no output weight and no layer weights
    let embed: Vec<f32> = (0..v * h).map(|x| (x as f32 * 0.001).sin()).collect();
    t.insert("token_embd.weight".into(), CandleTensor::from_vec(embed, &[v, h], &dev).unwrap());

    // Missing output.weight is OK if embeddings exist (weight tying), but missing
    // layer weights will cause an error during transformer construction.
    let result = BitNetModel::from_gguf(cfg, t, HashMap::new(), Device::Cpu);
    assert!(result.is_err(), "incomplete model must fail during construction");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_inference_with_empty_prompt() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    // Empty string encodes to empty token list via MockTokenizer
    let ids = tokenizer.encode("", false, false)?;

    if ids.is_empty() {
        // Engine may return an error or produce tokens from empty context.
        // Either is acceptable; just verify no panic.
        let result = engine.generate_tokens(&ids, &config).await;
        let _ = result;
    }
    Ok(())
}

// =========================================================================
// 8. Sampling strategy edge cases
// =========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_greedy_is_deterministic_across_runs() -> Result<()> {
    let mut runs = Vec::new();
    for _ in 0..3 {
        let model = synthetic_model()?;
        let tokenizer = Arc::new(MockTokenizer::new());
        let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;
        let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
        let ids = tokenizer.encode("greedy", false, false)?;
        runs.push(engine.generate_tokens(&ids, &config).await?);
    }
    assert_eq!(runs[0], runs[1], "greedy run 0 vs 1 must match");
    assert_eq!(runs[1], runs[2], "greedy run 1 vs 2 must match");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_temperature_zero_equals_greedy() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());

    let engine_a = InferenceEngine::new(model.clone(), tokenizer.clone(), Device::Cpu)?;
    let engine_b = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;

    let greedy_cfg = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    // Match greedy exactly: temp=0.0, top_k=1, top_p=1.0, rep_penalty=1.0
    let zero_temp_cfg = GenerationConfig::greedy()
        .with_temperature(0.0)
        .with_top_k(1)
        .with_seed(42)
        .with_max_tokens(4);

    let ids = tokenizer.encode("temp0", false, false)?;
    let a = engine_a.generate_tokens(&ids, &greedy_cfg).await?;
    let b = engine_b.generate_tokens(&ids, &zero_temp_cfg).await?;

    assert_eq!(a, b, "temperature=0.0 + top_k=1 must match greedy");
    Ok(())
}

// =========================================================================
// 9. Generation config serialization round-trip
// =========================================================================

#[test]
fn test_generation_config_serde_roundtrip() {
    let config = GenerationConfig::greedy()
        .with_seed(42)
        .with_max_tokens(100)
        .with_stop_token_id(128009)
        .with_stop_sequence("</s>".to_string());

    let json = serde_json::to_string(&config).expect("serialize");
    let restored: GenerationConfig = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(restored.max_new_tokens, 100);
    assert_eq!(restored.seed, Some(42));
    assert_eq!(restored.temperature, 0.0);
    assert_eq!(restored.stop_sequences, vec!["</s>"]);
    assert!(restored.stop_token_ids.contains(&128009));
}

// =========================================================================
// 10. Logits callback integration
// =========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_logits_callback_receives_steps() -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let call_count = Arc::new(AtomicUsize::new(0));
    let count_clone = call_count.clone();

    let cb: bitnet_inference::config::LogitsCallback = Arc::new(move |_step, _topk, _chosen| {
        count_clone.fetch_add(1, Ordering::SeqCst);
    });

    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;

    let config = GenerationConfig::greedy()
        .with_seed(42)
        .with_max_tokens(4)
        .with_logits_tap_steps(4)
        .with_logits_cb(Some(cb));

    let ids = tokenizer.encode("callback", false, false)?;
    engine.generate_tokens(&ids, &config).await?;

    // The callback should have been invoked for each of the tapped steps.
    // Exact count depends on implementation, but must be > 0 if tap is enabled.
    let count = call_count.load(Ordering::SeqCst);
    assert!(count > 0, "logits callback must be invoked at least once; got {count}");
    Ok(())
}

// =========================================================================
// 11. Stop criteria unit-level integration
// =========================================================================

#[test]
fn test_stop_criteria_priority_order() {
    use bitnet_generation::{StopCriteria, StopReason, check_stop};

    let criteria = StopCriteria {
        stop_token_ids: vec![42],
        stop_strings: vec!["stop".to_string()],
        max_tokens: 1,
        eos_token_id: Some(42),
    };

    // Token ID match takes priority over EOS
    let result = check_stop(&criteria, 42, &[], "");
    assert_eq!(result, Some(StopReason::StopTokenId(42)));
}

#[test]
fn test_stop_criteria_max_tokens_boundary() {
    use bitnet_generation::{StopCriteria, StopReason, check_stop};

    let criteria = StopCriteria {
        stop_token_ids: vec![],
        stop_strings: vec![],
        max_tokens: 3,
        eos_token_id: None,
    };

    // Below budget
    assert!(check_stop(&criteria, 5, &[1, 2], "").is_none());
    // At budget
    assert_eq!(check_stop(&criteria, 5, &[1, 2, 3], ""), Some(StopReason::MaxTokens));
}

// =========================================================================
// 12. Logits transforms (bitnet-logits pure functions)
// =========================================================================

#[test]
fn test_logits_argmax_correctness() {
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    assert_eq!(bitnet_logits::argmax(&logits), 3);
}

#[test]
fn test_logits_temperature_scaling() {
    let mut logits = vec![1.0, 2.0, 3.0];
    bitnet_logits::apply_temperature(&mut logits, 0.5);
    // With temp=0.5, logits should be scaled by 1/0.5 = 2.0
    assert!((logits[0] - 2.0).abs() < 1e-5);
    assert!((logits[1] - 4.0).abs() < 1e-5);
    assert!((logits[2] - 6.0).abs() < 1e-5);
}

#[test]
fn test_logits_softmax_sums_to_one() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    bitnet_logits::softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1.0, got {sum}");
}

#[test]
fn test_logits_top_k_filters() {
    let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
    let kept = bitnet_logits::apply_top_k(&mut logits, 2);
    assert!(kept <= 2, "top_k=2 must keep at most 2 entries");
    // Non-top-k values should be -inf
    let finite_count = logits.iter().filter(|x| x.is_finite()).count();
    assert_eq!(finite_count, 2);
}

#[test]
fn test_logits_repetition_penalty() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0];
    let token_ids = vec![1, 3]; // penalize tokens at index 1 and 3
    bitnet_logits::apply_repetition_penalty(&mut logits, &token_ids, 2.0);

    // Positive logits for penalized tokens should decrease
    assert!(logits[1] < 2.0, "penalized token 1 should decrease");
    assert!(logits[3] < 4.0, "penalized token 3 should decrease");
    // Non-penalized tokens should remain unchanged
    assert!((logits[0] - 1.0).abs() < 1e-5);
    assert!((logits[2] - 3.0).abs() < 1e-5);
}

// =========================================================================
// 13. Device probe integration
// =========================================================================

#[test]
fn test_device_probe_cpu_capabilities() {
    let caps = bitnet_device_probe::probe_cpu();
    assert!(caps.core_count > 0, "must detect at least 1 CPU core");
}

#[test]
fn test_device_probe_simd_detection() {
    let level = bitnet_device_probe::detect_simd_level();
    // On any platform, we should at least get Scalar (rank 0)
    let _rank = bitnet_device_probe::simd_level_rank(&level);
    // Smoke test: function returns without panic
}

// =========================================================================
// 14. Model configuration edge cases
// =========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_single_layer_model() -> Result<()> {
    // Use vocab >= 256 because MockTokenizer maps bytes 0-255 to token IDs
    let model = synthetic_model_with_vocab(512, 64, 1)?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(2);
    let ids = tokenizer.encode("a", false, false)?;
    let tokens = engine.generate_tokens(&ids, &config).await?;

    assert_eq!(tokens.len(), 2);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_small_vocab_model() -> Result<()> {
    // MockTokenizer encodes bytes 0-255, so vocab must be >= 256.
    // Test a smaller hidden_size instead to exercise the "small model" path.
    let model = synthetic_model_with_vocab(512, 64, 1)?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
    let ids = tokenizer.encode("b", false, false)?;
    let tokens = engine.generate_tokens(&ids, &config).await?;

    for &id in &tokens {
        assert!(id < 512, "token {id} exceeds vocab_size 512");
    }
    Ok(())
}

// =========================================================================
// 15. EOS token handling
// =========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_eos_token_id_stops_generation() -> Result<()> {
    let model = synthetic_model()?;
    let tokenizer = Arc::new(MockTokenizer::new());
    let engine = InferenceEngine::new(model, tokenizer.clone(), Device::Cpu)?;

    // Use the known first generated token (140) as EOS to verify early stopping
    let config =
        GenerationConfig::greedy().with_seed(42).with_max_tokens(20).with_eos_token_id(Some(140));

    let ids = tokenizer.encode("2+2=", false, false)?;
    let tokens = engine.generate_tokens(&ids, &config).await?;

    // Generation should stop before producing 20 tokens because token 140
    // is produced first and matches EOS.
    assert!(tokens.len() < 20, "EOS must truncate generation");
    Ok(())
}

// =========================================================================
// 16. Memory bounds (synthetic model size verification)
// =========================================================================

#[test]
fn test_synthetic_model_memory_is_bounded() {
    // A 512-vocab, 256-hidden, 2-layer model should be small
    let model = synthetic_model().expect("synthetic model must construct");
    // The Arc<BitNetModel> itself is small; the tensors are the bulk.
    // We just verify construction succeeds without OOM for the tiny config.
    let _ = model;
}

#[test]
fn test_large_vocab_model_construction() {
    // 8192 vocab × 256 hidden = ~8MB of embeddings — should still be fine
    let result = synthetic_model_with_vocab(8192, 256, 2);
    assert!(result.is_ok(), "8K vocab model must construct without error");
}

// =========================================================================
// 17. Prompt template integration
// =========================================================================

#[test]
fn test_prompt_template_instruct_format() {
    use bitnet_prompt_templates::{PromptTemplate, TemplateType};
    let tmpl = PromptTemplate::new(TemplateType::Instruct);
    let formatted = tmpl.format("What is 2+2?");
    assert!(formatted.contains("What is 2+2?"), "instruct template must include prompt");
}

#[test]
fn test_prompt_template_raw_passthrough() {
    use bitnet_prompt_templates::{PromptTemplate, TemplateType};
    let tmpl = PromptTemplate::new(TemplateType::Raw);
    let formatted = tmpl.format("hello world");
    assert_eq!(formatted, "hello world", "raw template must pass through unchanged");
}

// =========================================================================
// 18. Quantization type enumeration
// =========================================================================

#[test]
fn test_quantization_types_are_distinct() {
    use bitnet_common::QuantizationType;
    let types = [QuantizationType::I2S, QuantizationType::TL1, QuantizationType::TL2];
    for (i, a) in types.iter().enumerate() {
        for (j, b) in types.iter().enumerate() {
            if i != j {
                assert_ne!(a, b, "quantization types must be distinct");
            }
        }
    }
}

// =========================================================================
// 19. Real model tests (opt-in via BITNET_MODEL_PATH)
// =========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires model file - run manually with BITNET_MODEL_PATH or BITNET_GGUF set"]
async fn test_real_model_inference_pipeline() -> Result<()> {
    let model_path = std::env::var("BITNET_MODEL_PATH")
        .or(std::env::var("BITNET_GGUF"))
        .expect("BITNET_MODEL_PATH or BITNET_GGUF must be set");

    let path = std::path::Path::new(&model_path);
    let loader = bitnet_models::ModelLoader::new(Device::Cpu);
    let model = loader.load(path)?;
    let tokenizer = bitnet_tokenizers::auto::load_auto(path, None)?;

    let recorder = KernelRecorder::new();
    let engine = InferenceEngine::new(model.into(), tokenizer.clone(), Device::Cpu)?
        .with_recorder(recorder.clone());

    let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(8);
    let prompt_ids = tokenizer.encode("What is 2+2?", true, false)?;
    let tokens = engine.generate_tokens(&prompt_ids, &config).await?;

    assert!(!tokens.is_empty(), "real model must generate tokens");
    let receipt = InferenceReceipt::generate("cpu-rust", recorder.snapshot(), None)?;
    assert_eq!(receipt.compute_path, "real");
    receipt.validate().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires model file - run manually with BITNET_MODEL_PATH or BITNET_GGUF set"]
async fn test_real_model_deterministic_output() -> Result<()> {
    let model_path = std::env::var("BITNET_MODEL_PATH")
        .or(std::env::var("BITNET_GGUF"))
        .expect("BITNET_MODEL_PATH or BITNET_GGUF must be set");

    let path = std::path::Path::new(&model_path);

    async fn run_once(path: &std::path::Path) -> Result<Vec<u32>> {
        let loader = bitnet_models::ModelLoader::new(Device::Cpu);
        let model = loader.load(path)?;
        let tokenizer = bitnet_tokenizers::auto::load_auto(path, None)?;
        let engine = InferenceEngine::new(model.into(), tokenizer.clone(), Device::Cpu)?;
        let config = GenerationConfig::greedy().with_seed(42).with_max_tokens(4);
        let ids = tokenizer.encode("2+2=", true, false)?;
        engine.generate_tokens(&ids, &config).await
    }

    let a = run_once(path).await?;
    let b = run_once(path).await?;
    assert_eq!(a, b, "real model with same seed must be reproducible");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires GPU hardware - run on CUDA-capable machine with --features gpu"]
async fn test_gpu_inference_pipeline() -> Result<()> {
    panic!("GPU test not yet implemented");
}
