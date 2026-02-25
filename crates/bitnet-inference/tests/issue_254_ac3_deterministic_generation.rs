//! AC3: Autoregressive Deterministic Generation (Issue #254)
//!
//! Tests feature spec: issue-254-real-inference-spec.md#ac3-autoregressive-deterministic-generation
//! API contract: neural-network-operation-requirements.md#generation-requirements
//!
//! This test validates that InferSession::generate() supports seeded greedy/top-k/top-p
//! sampling with BITNET_DETERMINISTIC=1 + RAYON_NUM_THREADS=1 producing identical
//! token sequences across runs.
#![cfg(feature = "cpu")]
mod support;
use anyhow::Result;
use bitnet_common::{Device, Tensor};
use bitnet_inference::{AutoregressiveGenerator, GenConfig};
use bitnet_models::BitNetModel;
use bitnet_tokenizers::{Tokenizer, UniversalTokenizer};
use serial_test::serial;
use support::EnvGuard;
#[tokio::test]
#[serial(bitnet_env)]
#[ignore = "Slow: runs 100+ mock forward passes; run manually with --ignored for integration validation"]
/// AC3.1: Deterministic Generation - SLOW INTEGRATION TEST
///
/// **This test runs 50-token generation (100+ forward passes) and is marked #[ignore].**
///
/// For fast unit testing of determinism, see:
/// - `tests/deterministic_sampling_unit.rs::test_same_seed_identical_samples()` (<5ms)
///
/// Validates ChaCha8Rng seeding and deterministic execution with full autoregressive generation.
///
/// Run manually: `cargo test test_ac3_deterministic_generation_identical_sequences -- --ignored`
async fn test_ac3_deterministic_generation_identical_sequences() -> Result<()> {
    let _g1 = EnvGuard::new("BITNET_DETERMINISTIC");
    _g1.set("1");
    let _g2 = EnvGuard::new("BITNET_SEED");
    _g2.set("42");
    let _g3 = EnvGuard::new("RAYON_NUM_THREADS");
    _g3.set("1");
    let config = GenConfig {
        max_new_tokens: 50,
        temperature: 1.0,
        top_k: Some(50),
        top_p: Some(0.9),
        seed: Some(42),
        ..Default::default()
    };
    let _model = create_test_model()?;
    let tokenizer = create_test_tokenizer()?;
    let prompt = "The future of artificial intelligence is";
    let input_ids: Vec<usize> =
        tokenizer.encode(prompt, false, false)?.iter().map(|&x| x as usize).collect();
    let mut generator1 = AutoregressiveGenerator::new(config.clone(), Device::Cpu)?;
    let tokens1 = generator1.generate(&input_ids, mock_forward_fn).await?;
    let mut generator2 = AutoregressiveGenerator::new(config, Device::Cpu)?;
    let tokens2 = generator2.generate(&input_ids, mock_forward_fn).await?;
    assert_eq!(
        tokens1, tokens2,
        "AC3: Deterministic generation should produce identical sequences.\nRun 1: {:?}\nRun 2: {:?}",
        tokens1, tokens2
    );
    assert!(tokens1.len() > 10, "AC3: Generation too short to validate determinism");
    println!("AC3.1: Deterministic generation test - PASSED");
    Ok(())
}
/// AC:3.2 - Greedy sampling (temperature=0 or top_k=1) is deterministic
/// Validates greedy decoding produces same result without seed
#[tokio::test]
async fn test_ac3_greedy_sampling_deterministic() -> Result<()> {
    let config = GenConfig {
        max_new_tokens: 20,
        temperature: 0.0,
        top_k: Some(1),
        top_p: Some(1.0),
        seed: None,
        ..Default::default()
    };
    let _model = create_test_model()?;
    let tokenizer = create_test_tokenizer()?;
    let prompt = "Once upon a time";
    let input_ids: Vec<usize> =
        tokenizer.encode(prompt, false, false)?.iter().map(|&x| x as usize).collect();
    let mut results = Vec::new();
    for _ in 0..3 {
        let mut generator = AutoregressiveGenerator::new(config.clone(), Device::Cpu)?;
        let tokens = generator.generate(&input_ids, mock_forward_fn).await?;
        results.push(tokens);
    }
    for i in 1..results.len() {
        assert_eq!(results[0], results[i], "AC3: Greedy sampling should produce identical results");
    }
    println!("AC3.2: Greedy sampling determinism test - PENDING IMPLEMENTATION");
    Ok(())
}
#[tokio::test]
#[serial(bitnet_env)]
#[ignore = "Slow: runs 100+ mock forward passes; run manually with --ignored for integration validation"]
/// AC3.3: Top-k Seeded Sampling - SLOW INTEGRATION TEST
///
/// **This test runs 30-token generation (60+ forward passes) and is marked #[ignore].**
///
/// For fast unit testing of top-k determinism, see:
/// - `tests/deterministic_sampling_unit.rs::test_same_seed_identical_samples()` (<5ms)
///
/// Validates top-k sampling respects seed for determinism in full autoregressive generation.
///
/// Run manually: `cargo test test_ac3_top_k_sampling_seeded -- --ignored`
async fn test_ac3_top_k_sampling_seeded() -> Result<()> {
    let _g1 = EnvGuard::new("BITNET_DETERMINISTIC");
    _g1.set("1");
    let _g2 = EnvGuard::new("RAYON_NUM_THREADS");
    _g2.set("1");
    let config = GenConfig {
        max_new_tokens: 30,
        temperature: 1.0,
        top_k: Some(20),
        top_p: Some(1.0),
        seed: Some(42),
        ..Default::default()
    };
    let _model = create_test_model()?;
    let tokenizer = create_test_tokenizer()?;
    let prompt = "In the year 2050";
    let input_ids: Vec<usize> =
        tokenizer.encode(prompt, false, false)?.iter().map(|&x| x as usize).collect();
    let mut generator1 = AutoregressiveGenerator::new(config.clone(), Device::Cpu)?;
    let tokens1 = generator1.generate(&input_ids, mock_forward_fn).await?;
    let mut generator2 = AutoregressiveGenerator::new(config.clone(), Device::Cpu)?;
    let tokens2 = generator2.generate(&input_ids, mock_forward_fn).await?;
    assert_eq!(tokens1, tokens2, "AC3: Top-k sampling with seed should be deterministic");
    println!("AC3.3: Top-k seeded sampling test - PENDING IMPLEMENTATION");
    Ok(())
}
#[tokio::test]
#[serial(bitnet_env)]
#[ignore = "Slow: runs 100+ mock forward passes; run manually with --ignored for integration validation"]
/// AC3.4: Top-p Nucleus Sampling - SLOW INTEGRATION TEST
///
/// **This test runs 25-token generation (50+ forward passes) and is marked #[ignore].**
///
/// For fast unit testing of nucleus sampling determinism, see:
/// - `tests/deterministic_sampling_unit.rs::test_same_seed_identical_samples()` (<5ms)
///
/// Validates nucleus sampling respects seed for determinism in full autoregressive generation.
///
/// Run manually: `cargo test test_ac3_top_p_nucleus_sampling_seeded -- --ignored`
async fn test_ac3_top_p_nucleus_sampling_seeded() -> Result<()> {
    let _g1 = EnvGuard::new("BITNET_DETERMINISTIC");
    _g1.set("1");
    let _g2 = EnvGuard::new("RAYON_NUM_THREADS");
    _g2.set("1");
    let config = GenConfig {
        max_new_tokens: 25,
        temperature: 0.8,
        top_k: Some(0),
        top_p: Some(0.95),
        seed: Some(123),
        ..Default::default()
    };
    let _model = create_test_model()?;
    let tokenizer = create_test_tokenizer()?;
    let prompt = "The secret to happiness is";
    let input_ids: Vec<usize> =
        tokenizer.encode(prompt, false, false)?.iter().map(|&x| x as usize).collect();
    let mut generator1 = AutoregressiveGenerator::new(config.clone(), Device::Cpu)?;
    let tokens1 = generator1.generate(&input_ids, mock_forward_fn).await?;
    let mut generator2 = AutoregressiveGenerator::new(config.clone(), Device::Cpu)?;
    let tokens2 = generator2.generate(&input_ids, mock_forward_fn).await?;
    assert_eq!(tokens1, tokens2, "AC3: Nucleus sampling with seed should be deterministic");
    println!("AC3.4: Nucleus seeded sampling test - PENDING IMPLEMENTATION");
    Ok(())
}
#[tokio::test]
#[serial(bitnet_env)]
#[ignore = "Slow: runs 100+ mock forward passes; run manually with --ignored for integration validation"]
/// AC3.5: Different Seeds Produce Different Outputs - SLOW INTEGRATION TEST
///
/// **This test runs 20-token generation (40+ forward passes) and is marked #[ignore].**
///
/// For fast unit testing of seed variance, see:
/// - `tests/deterministic_sampling_unit.rs::test_different_seeds_different_samples()` (<5ms)
///
/// Validates seed actually affects generation in full autoregressive generation.
///
/// Run manually: `cargo test test_ac3_different_seeds_different_outputs -- --ignored`
async fn test_ac3_different_seeds_different_outputs() -> Result<()> {
    let _g1 = EnvGuard::new("BITNET_DETERMINISTIC");
    _g1.set("1");
    let _g2 = EnvGuard::new("RAYON_NUM_THREADS");
    _g2.set("1");
    let _model = create_test_model()?;
    let tokenizer = create_test_tokenizer()?;
    let prompt = "Once upon a time";
    let input_ids: Vec<usize> =
        tokenizer.encode(prompt, false, false)?.iter().map(|&x| x as usize).collect();
    let config1 = GenConfig {
        max_new_tokens: 20,
        temperature: 1.0,
        top_k: Some(50),
        top_p: Some(0.9),
        seed: Some(42),
        ..Default::default()
    };
    let mut gen1 = AutoregressiveGenerator::new(config1, Device::Cpu)?;
    let tokens1 = gen1.generate(&input_ids, mock_forward_fn).await?;
    let config2 = GenConfig {
        max_new_tokens: 20,
        temperature: 1.0,
        top_k: Some(50),
        top_p: Some(0.9),
        seed: Some(123),
        ..Default::default()
    };
    let mut gen2 = AutoregressiveGenerator::new(config2, Device::Cpu)?;
    let tokens2 = gen2.generate(&input_ids, mock_forward_fn).await?;
    assert_ne!(tokens1, tokens2, "AC3: Different seeds should produce different outputs");
    println!("AC3.5: Different seeds test - PENDING IMPLEMENTATION");
    Ok(())
}
#[tokio::test]
#[serial(bitnet_env)]
#[ignore = "Slow: runs 100+ mock forward passes; run manually with --ignored for integration validation"]
/// AC3.6: Rayon Single-Thread Determinism - SLOW INTEGRATION TEST
///
/// **This test runs 15-token generation 3 times (90+ forward passes) and is marked #[ignore].**
///
/// For fast unit testing of single-threaded determinism, see:
/// - `tests/deterministic_sampling_unit.rs::test_same_seed_identical_samples()` (<5ms)
///
/// Validates single-threaded execution prevents race conditions in full autoregressive generation.
///
/// Run manually: `cargo test test_ac3_rayon_single_thread_determinism -- --ignored`
async fn test_ac3_rayon_single_thread_determinism() -> Result<()> {
    let _g1 = EnvGuard::new("BITNET_DETERMINISTIC");
    _g1.set("1");
    let _g2 = EnvGuard::new("BITNET_SEED");
    _g2.set("42");
    let _g3 = EnvGuard::new("RAYON_NUM_THREADS");
    _g3.set("1");
    let config = GenConfig {
        max_new_tokens: 15,
        temperature: 1.0,
        top_k: Some(30),
        top_p: Some(0.9),
        seed: Some(42),
        ..Default::default()
    };
    let _model = create_test_model()?;
    let tokenizer = create_test_tokenizer()?;
    let prompt = "The meaning of life";
    let input_ids: Vec<usize> =
        tokenizer.encode(prompt, false, false)?.iter().map(|&x| x as usize).collect();
    assert_eq!(
        std::env::var("RAYON_NUM_THREADS").ok(),
        Some("1".to_string()),
        "AC3: RAYON_NUM_THREADS should be set to 1"
    );
    let mut results = Vec::new();
    for _ in 0..3 {
        let mut generator = AutoregressiveGenerator::new(config.clone(), Device::Cpu)?;
        let tokens = generator.generate(&input_ids, mock_forward_fn).await?;
        results.push(tokens);
    }
    for i in 1..results.len() {
        assert_eq!(
            results[0], results[i],
            "AC3: Single-threaded execution should be deterministic"
        );
    }
    println!("AC3.6: Single-threaded determinism test - PENDING IMPLEMENTATION");
    Ok(())
}
fn create_test_model() -> Result<BitNetModel> {
    use bitnet_common::{BitNetConfig, ModelConfig, ModelFormat};
    let model_config = ModelConfig {
        path: None,
        format: ModelFormat::Gguf,
        vocab_size: 50257,
        hidden_size: 768,
        num_layers: 2,
        num_heads: 12,
        num_key_value_heads: 12,
        intermediate_size: 3072,
        max_position_embeddings: 1024,
        rope_theta: Some(10000.0),
        rope_scaling: None,
        rms_norm_eps: None,
        tokenizer: bitnet_common::config::TokenizerConfig::default(),
    };
    let config = BitNetConfig { model: model_config, ..Default::default() };
    Ok(BitNetModel::new(config, Device::Cpu))
}
fn create_test_tokenizer() -> Result<UniversalTokenizer> {
    use bitnet_tokenizers::TokenizerConfig;
    let config = TokenizerConfig {
        model_type: "gpt2".to_string(),
        vocab_size: 50257,
        pre_tokenizer: Some("gpt2".to_string()),
        add_bos: false,
        add_eos: false,
        add_space_prefix: true,
        byte_fallback: true,
        bos_token_id: Some(50256),
        eos_token_id: Some(50256),
        pad_token_id: Some(50257),
        unk_token_id: Some(0),
        vocabulary: None,
        bpe_merges: None,
    };
    UniversalTokenizer::new(config)
        .map_err(|e| anyhow::anyhow!("Failed to create tokenizer: {}", e))
}
async fn mock_forward_fn(
    input: bitnet_common::BitNetTensor,
) -> Result<bitnet_common::BitNetTensor> {
    let vocab_size = 50257;
    let input_candle = input.to_candle()?;
    let input_shape = input_candle.shape();
    let seq_len = if input_shape.dims().len() >= 2 {
        input_shape.dims()[input_shape.dims().len() - 2]
    } else {
        1
    };
    let mut logits_data = vec![0.0f32; vocab_size];
    for (i, item) in logits_data.iter_mut().enumerate() {
        let value = ((i * 17 + seq_len * 31) % 1000) as f32 / 100.0 - 5.0;
        *item = value;
    }
    logits_data[42] = 2.0;
    logits_data[1337] = 1.5;
    logits_data[9999 % vocab_size] = 1.8;
    let logits =
        bitnet_common::BitNetTensor::from_slice(&logits_data, &[1, vocab_size], &Device::Cpu)?;
    Ok(logits)
}
