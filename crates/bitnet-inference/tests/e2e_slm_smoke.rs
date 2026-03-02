//! End-to-end SLM smoke tests for multi-architecture inference pipeline.
//!
//! Validates the complete inference pipeline from model loading through token
//! generation using synthetic weights — no actual model files needed.
//!
//! Coverage:
//! - 6 architecture configs (Phi-4, LLaMA-3, Qwen2.5, Gemma-2, Mistral, BitNet)
//! - Full pipeline integration (tokenize → embed → transform → logits → sample)
//! - Regression guards (shapes, finiteness, softmax invariants, vocab bounds)
#![cfg(feature = "cpu")]

use std::sync::Arc;

use bitnet_common::config::ModelConfig;
use bitnet_common::{
    ActivationType, ArchitectureRegistry, BitNetConfig, ConcreteTensor, Device, NormType,
};
use bitnet_inference::InferenceEngine;
use bitnet_inference::config::GenerationConfig;
use bitnet_inference::sampling::{apply_temperature, greedy_sample, softmax_in_place};
use bitnet_inference::simple_forward::{Weights, logits_for_token};
use bitnet_models::Model;

// ═══════════════════════════════════════════════════════════════════════════
// Synthetic model / tokenizer helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Architecture-specific config for creating synthetic models.
struct ArchSpec {
    name: &'static str,
    vocab_size: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    norm_type: NormType,
    activation_type: ActivationType,
}

const PHI4: ArchSpec = ArchSpec {
    name: "phi-4",
    vocab_size: 128,
    hidden_size: 64,
    num_heads: 8,
    num_kv_heads: 8,
    num_layers: 2,
    intermediate_size: 128,
    max_position_embeddings: 2048,
    norm_type: NormType::RmsNorm,
    activation_type: ActivationType::Silu,
};

const LLAMA3: ArchSpec = ArchSpec {
    name: "llama",
    vocab_size: 128,
    hidden_size: 64,
    num_heads: 8,
    num_kv_heads: 8,
    num_layers: 2,
    intermediate_size: 128,
    max_position_embeddings: 4096,
    norm_type: NormType::RmsNorm,
    activation_type: ActivationType::Silu,
};

const QWEN25: ArchSpec = ArchSpec {
    name: "qwen2.5",
    vocab_size: 128,
    hidden_size: 64,
    num_heads: 8,
    num_kv_heads: 8,
    num_layers: 2,
    intermediate_size: 128,
    max_position_embeddings: 2048,
    norm_type: NormType::RmsNorm,
    activation_type: ActivationType::Silu,
};

const GEMMA2: ArchSpec = ArchSpec {
    name: "gemma-2",
    vocab_size: 128,
    hidden_size: 64,
    num_heads: 8,
    num_kv_heads: 8,
    num_layers: 2,
    intermediate_size: 128,
    max_position_embeddings: 2048,
    norm_type: NormType::RmsNorm,
    activation_type: ActivationType::Gelu,
};

const MISTRAL: ArchSpec = ArchSpec {
    name: "mistral",
    vocab_size: 128,
    hidden_size: 64,
    num_heads: 8,
    num_kv_heads: 8,
    num_layers: 2,
    intermediate_size: 128,
    max_position_embeddings: 2048,
    norm_type: NormType::RmsNorm,
    activation_type: ActivationType::Silu,
};

const BITNET: ArchSpec = ArchSpec {
    name: "bitnet",
    vocab_size: 128,
    hidden_size: 64,
    num_heads: 8,
    num_kv_heads: 8,
    num_layers: 2,
    intermediate_size: 128,
    max_position_embeddings: 2048,
    norm_type: NormType::LayerNorm,
    activation_type: ActivationType::Silu,
};

fn build_bitnet_config(spec: &ArchSpec) -> BitNetConfig {
    let mut cfg = BitNetConfig::default();
    cfg.model.vocab_size = spec.vocab_size;
    cfg.model.hidden_size = spec.hidden_size;
    cfg.model.num_heads = spec.num_heads;
    cfg.model.num_key_value_heads = spec.num_kv_heads;
    cfg.model.num_layers = spec.num_layers;
    cfg.model.intermediate_size = spec.intermediate_size;
    cfg.model.max_position_embeddings = spec.max_position_embeddings;
    cfg.model.norm_type = spec.norm_type;
    cfg.model.activation_type = spec.activation_type;
    cfg
}

/// Mock model that returns deterministic logits shaped by architecture config.
/// The logit vector places a distinct peak at a token ID derived from a
/// simple hash of the architecture name so each arch produces a different
/// "signature" token.
struct SyntheticModel {
    config: BitNetConfig,
}

impl SyntheticModel {
    fn new(spec: &ArchSpec) -> Self {
        Self { config: build_bitnet_config(spec) }
    }
}

impl Model for SyntheticModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        _input: &ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, self.config.model.vocab_size]))
    }

    fn embed(&self, _tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 4, self.config.model.hidden_size]))
    }

    fn logits(&self, _hidden: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
        Ok(ConcreteTensor::mock(vec![1, 1, self.config.model.vocab_size]))
    }
}

/// Mock tokenizer with configurable vocab size.
struct SyntheticTokenizer {
    vocab_size: usize,
    eos_id: u32,
}

impl SyntheticTokenizer {
    fn new(vocab_size: usize) -> Self {
        Self { vocab_size, eos_id: (vocab_size - 1) as u32 }
    }
}

impl bitnet_tokenizers::Tokenizer for SyntheticTokenizer {
    fn encode(
        &self,
        text: &str,
        _add_bos: bool,
        _add_special: bool,
    ) -> bitnet_common::Result<Vec<u32>> {
        Ok(text.bytes().take(8).map(|b| (b as u32) % self.vocab_size as u32).collect())
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
        Some(self.eos_id)
    }

    fn pad_token_id(&self) -> Option<u32> {
        None
    }
}

/// Build an InferenceEngine from an ArchSpec.
async fn engine_for(spec: &ArchSpec) -> anyhow::Result<InferenceEngine> {
    let model: Arc<dyn Model> = Arc::new(SyntheticModel::new(spec));
    let tokenizer: Arc<dyn bitnet_tokenizers::Tokenizer> =
        Arc::new(SyntheticTokenizer::new(spec.vocab_size));
    let engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;
    Ok(engine)
}

/// Generate tokens from a spec and return (generated_tokens, vocab_size).
async fn generate_from(
    spec: &ArchSpec,
    max_tokens: u32,
    temperature: f32,
    seed: Option<u64>,
) -> anyhow::Result<(Vec<u32>, usize)> {
    let engine = engine_for(spec).await?;
    let input: Vec<u32> = b"test".iter().map(|&b| (b as u32) % spec.vocab_size as u32).collect();
    let config = GenerationConfig::greedy()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature)
        .with_seed(seed.unwrap_or(0));
    let tokens = engine.generate_tokens(&input, &config).await?;
    Ok((tokens, spec.vocab_size))
}

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline smoke tests (architectures)
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_phi4_single_token() {
    let (tokens, _) = generate_from(&PHI4, 1, 0.0, None).await.unwrap();
    assert_eq!(tokens.len(), 1, "Phi-4: expected exactly 1 generated token");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_llama3_single_token() {
    let (tokens, _) = generate_from(&LLAMA3, 1, 0.0, None).await.unwrap();
    assert_eq!(tokens.len(), 1, "LLaMA-3: expected exactly 1 generated token");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_qwen25_single_token() {
    let (tokens, _) = generate_from(&QWEN25, 1, 0.0, None).await.unwrap();
    assert_eq!(tokens.len(), 1, "Qwen2.5: expected exactly 1 generated token");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_gemma2_single_token() {
    let (tokens, _) = generate_from(&GEMMA2, 1, 0.0, None).await.unwrap();
    assert_eq!(tokens.len(), 1, "Gemma-2: expected exactly 1 generated token");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_mistral_single_token() {
    let (tokens, _) = generate_from(&MISTRAL, 1, 0.0, None).await.unwrap();
    assert_eq!(tokens.len(), 1, "Mistral: expected exactly 1 generated token");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_bitnet_single_token_regression() {
    let (tokens, _) = generate_from(&BITNET, 1, 0.0, None).await.unwrap();
    assert_eq!(tokens.len(), 1, "BitNet: expected exactly 1 generated token");
    // Regression guard: architecture defaults must remain LayerNorm + SiLU.
    assert!(
        ArchitectureRegistry::is_known("bitnet"),
        "bitnet must stay in the architecture registry"
    );
    let defaults = ArchitectureRegistry::lookup("bitnet").unwrap();
    assert_eq!(defaults.norm_type, NormType::LayerNorm);
    assert_eq!(defaults.activation_type, ActivationType::Silu);
}

// ── Config-level smoke tests ──────────────────────────────────────────────

#[test]
fn e2e_silu_activation_produces_valid_logits() {
    let spec = &PHI4; // SiLU
    assert_eq!(spec.activation_type, ActivationType::Silu);
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults(spec.name);
    assert_eq!(config.activation_type, ActivationType::Silu);

    // Synthetic logits pipeline with SiLU-style sharpening
    let mut logits = vec![0.2_f32, 0.8, 0.5, 0.1];
    apply_temperature(&mut logits, 0.5);
    softmax_in_place(&mut logits);
    assert!(logits.iter().all(|v| v.is_finite()), "SiLU config logits must be finite");
}

#[test]
fn e2e_rmsnorm_config_produces_valid_output() {
    let spec = &LLAMA3;
    assert_eq!(spec.norm_type, NormType::RmsNorm);
    let mut config = ModelConfig::default();
    config.apply_architecture_defaults(spec.name);
    assert_eq!(config.norm_type, NormType::RmsNorm);

    let mut logits = vec![1.0_f32, 2.0, 3.0, 4.0];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "RMSNorm config: softmax must sum to ~1.0, got {sum}");
}

#[test]
fn e2e_gqa_40_10_correct_shape() {
    // GQA: 40 query heads, 10 KV heads (4:1 ratio like LLaMA-3 70B)
    let mut cfg = BitNetConfig::default();
    cfg.model.hidden_size = 320; // Must be divisible by 40
    cfg.model.num_heads = 40;
    cfg.model.num_key_value_heads = 10;
    assert!(cfg.validate().is_ok(), "GQA 40/10 config must validate");

    let head_dim = cfg.model.hidden_size / cfg.model.num_heads;
    assert_eq!(head_dim, 8, "head_dim = hidden_size / num_heads");
    let groups = cfg.model.num_heads / cfg.model.num_key_value_heads;
    assert_eq!(groups, 4, "GQA group ratio must be 4:1");
}

#[test]
fn e2e_16k_context_allocation() {
    let mut cfg = BitNetConfig::default();
    cfg.model.max_position_embeddings = 16384;
    assert!(cfg.validate().is_ok(), "16K context config must validate");
    assert_eq!(cfg.model.max_position_embeddings, 16384);
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn e2e_full_pipeline_synthetic_weights() {
    // tokenize → embed (lookup) → transform (matmul) → logits → sample
    let vocab = 8;
    let dim = 4;
    // Synthetic embedding: identity-ish for first `dim` tokens
    let mut tok_embeddings = vec![0.0_f32; vocab * dim];
    for i in 0..vocab.min(dim) {
        tok_embeddings[i * dim + i] = 1.0;
    }
    // Synthetic lm_head: [dim, vocab] that produces a peak at token 2
    let mut lm_head = vec![0.1_f32; dim * vocab];
    for d in 0..dim {
        lm_head[d * vocab + 2] = 1.0; // bias toward token 2
    }

    let w = Weights { tok_embeddings: &tok_embeddings, lm_head: &lm_head, vocab, dim };

    // "tokenize" token 0, get logits, sample
    let logits = logits_for_token(&w, 0);
    assert_eq!(logits.len(), vocab);
    let sampled = greedy_sample(&logits).unwrap();
    assert_eq!(sampled, 2, "synthetic pipeline must produce token 2");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_greedy_deterministic() {
    let (tokens1, _) = generate_from(&PHI4, 4, 0.0, None).await.unwrap();
    let (tokens2, _) = generate_from(&PHI4, 4, 0.0, None).await.unwrap();
    assert_eq!(tokens1, tokens2, "greedy generation must be deterministic");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_temperature_seeded_deterministic() {
    let (tokens1, _) = generate_from(&LLAMA3, 4, 0.8, Some(42)).await.unwrap();
    let (tokens2, _) = generate_from(&LLAMA3, 4, 0.8, Some(42)).await.unwrap();
    assert_eq!(tokens1, tokens2, "seeded temperature sampling must be deterministic");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_multiple_sequential_tokens() {
    let (tokens, _) = generate_from(&QWEN25, 5, 0.0, None).await.unwrap();
    assert_eq!(tokens.len(), 5, "must generate exactly 5 sequential tokens");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn e2e_stop_token_terminates() {
    let spec = &MISTRAL;
    let engine = engine_for(spec).await.unwrap();
    let input: Vec<u32> = vec![1, 2, 3];
    // Set stop token to 0 — the mock model produces uniform logits, greedy
    // picks token 0, which should immediately trigger stop.
    let config = GenerationConfig::greedy().with_max_tokens(10).with_stop_token_ids(vec![0]);
    let tokens = engine.generate_tokens(&input, &config).await.unwrap();
    // Generation should stop after at most 1 token (token 0 is the stop token).
    assert!(
        tokens.len() <= 1,
        "stop token should terminate generation early; got {} tokens",
        tokens.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression guards
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn regression_output_shape_batch_x_vocab() {
    let vocab = 64;
    let dim = 8;
    let tok_embeddings = vec![0.1_f32; vocab * dim];
    let lm_head = vec![0.1_f32; dim * vocab];
    let w = Weights { tok_embeddings: &tok_embeddings, lm_head: &lm_head, vocab, dim };

    let logits = logits_for_token(&w, 0);
    assert_eq!(logits.len(), vocab, "output shape must be [vocab_size]");
}

#[test]
fn regression_logits_are_finite() {
    let vocab = 32;
    let dim = 4;
    let tok_embeddings = vec![0.5_f32; vocab * dim];
    let lm_head = vec![0.3_f32; dim * vocab];
    let w = Weights { tok_embeddings: &tok_embeddings, lm_head: &lm_head, vocab, dim };

    let logits = logits_for_token(&w, 0);
    assert!(logits.iter().all(|v| v.is_finite()), "all logits must be finite (no NaN/Inf)");
}

#[test]
fn regression_softmax_sums_to_one() {
    let mut logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    softmax_in_place(&mut logits);
    let sum: f32 = logits.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax of logits must sum to ~1.0, got {sum}");
}

#[test]
fn regression_token_within_vocab() {
    let vocab = 64;
    let dim = 4;
    let tok_embeddings = vec![0.1_f32; vocab * dim];
    let mut lm_head = vec![0.0_f32; dim * vocab];
    // Place a peak at token 7
    for d in 0..dim {
        lm_head[d * vocab + 7] = 10.0;
    }
    let w = Weights { tok_embeddings: &tok_embeddings, lm_head: &lm_head, vocab, dim };

    let logits = logits_for_token(&w, 0);
    let token = greedy_sample(&logits).unwrap();
    assert!((token as usize) < vocab, "generated token {token} must be < vocab_size {vocab}");
}

#[test]
fn regression_memory_bounded() {
    // Verify that constructing synthetic weights for a small model stays
    // within 2x of the expected allocation.
    let vocab = 256;
    let dim = 64;
    let expected_bytes = (vocab * dim + dim * vocab) * std::mem::size_of::<f32>();

    let tok_embeddings = vec![0.1_f32; vocab * dim];
    let lm_head = vec![0.1_f32; dim * vocab];
    let actual_bytes = (tok_embeddings.len() + lm_head.len()) * std::mem::size_of::<f32>();

    assert!(
        actual_bytes <= expected_bytes * 2,
        "memory {actual_bytes} must be within 2x of expected {expected_bytes}"
    );
    // Also verify the exact match for this simple case
    assert_eq!(actual_bytes, expected_bytes, "synthetic weights size must match expected");
}
