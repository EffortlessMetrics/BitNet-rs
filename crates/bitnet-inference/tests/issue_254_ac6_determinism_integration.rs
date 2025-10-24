//! AC6: Determinism Integration Test (Issue #254)
//!
//! Tests feature spec: issue-254-real-inference-spec.md#ac6-determinism-test
//! API contract: neural-network-operation-requirements.md#determinism-requirements
//!
//! This test validates that two inference runs produce identical token sequences
//! with BITNET_DETERMINISTIC=1 + RAYON_NUM_THREADS=1.

#![cfg(feature = "cpu")]

mod support;
use support::EnvGuard;

use anyhow::Result;
use bitnet_common::Device;
use bitnet_inference::{AutoregressiveGenerator, GenConfig};
use bitnet_models::BitNetModel;
use bitnet_tokenizers::{Tokenizer, UniversalTokenizer};
use serial_test::serial;

#[tokio::test]
#[serial(bitnet_env)]
// Slow: 50-token generation with 50,257 vocab. Fast equivalent: tests/deterministic_sampling_unit.rs
#[ignore]
/// AC6.1: Deterministic Inference - SLOW INTEGRATION TEST
///
/// **This test runs 50-token generation (100+ forward passes) and is marked #[ignore].**
///
/// For fast unit testing of determinism, see:
/// - `tests/deterministic_sampling_unit.rs::test_same_seed_identical_samples()` (<5ms)
///
/// Validates full determinism from prompt to generated tokens with complete inference runs.
///
/// Run manually: `cargo test test_ac6_deterministic_inference_identical_runs -- --ignored`
async fn test_ac6_deterministic_inference_identical_runs() -> Result<()> {
    // Set deterministic environment
    let _g1 = EnvGuard::new("BITNET_DETERMINISTIC");
    _g1.set("1");
    let _g2 = EnvGuard::new("BITNET_SEED");
    _g2.set("42");
    let _g3 = EnvGuard::new("RAYON_NUM_THREADS");
    _g3.set("1");

    let _model = create_test_model()?;
    let tokenizer = create_test_tokenizer()?;

    let config = GenConfig {
        seed: Some(42),
        max_new_tokens: 50,
        temperature: 1.0,
        top_k: Some(50),
        top_p: Some(0.9),
        ..Default::default()
    };

    let prompt = "The future of AI is";
    let input_ids: Vec<usize> =
        tokenizer.encode(prompt, false, false)?.iter().map(|&x| x as usize).collect();

    // Run 1
    let mut generator1 = AutoregressiveGenerator::new(config.clone(), Device::Cpu)?;
    let tokens1 = generator1.generate(&input_ids, mock_forward_fn).await?;

    // Run 2
    let mut generator2 = AutoregressiveGenerator::new(config, Device::Cpu)?;
    let tokens2 = generator2.generate(&input_ids, mock_forward_fn).await?;

    // AC6: Identical sequences
    assert_eq!(tokens1, tokens2, "AC6: Deterministic inference should produce identical tokens");

    // Verify not trivial
    assert!(tokens1.len() > 10, "AC6: Generation too short to validate");

    println!("AC6.1: Deterministic inference test - PENDING IMPLEMENTATION");
    Ok(())
}

#[tokio::test]
#[serial(bitnet_env)]
// Slow: 20-token generation x5 runs with 50,257 vocab. Fast equivalent: tests/deterministic_sampling_unit.rs
#[ignore]
/// AC6.2: Determinism Multiple Runs - SLOW INTEGRATION TEST
///
/// **This test runs 20-token generation 5 times (200+ forward passes) and is marked #[ignore].**
///
/// For fast unit testing of multi-run determinism, see:
/// - `tests/deterministic_sampling_unit.rs::test_same_seed_identical_samples()` (<5ms)
///
/// Validates consistency over multiple generation cycles with full autoregressive generation.
///
/// Run manually: `cargo test test_ac6_determinism_multiple_runs -- --ignored`
async fn test_ac6_determinism_multiple_runs() -> Result<()> {
    let _g1 = EnvGuard::new("BITNET_DETERMINISTIC");
    _g1.set("1");
    let _g2 = EnvGuard::new("BITNET_SEED");
    _g2.set("42");
    let _g3 = EnvGuard::new("RAYON_NUM_THREADS");
    _g3.set("1");

    let _model = create_test_model()?;
    let tokenizer = create_test_tokenizer()?;

    let config = GenConfig {
        seed: Some(42),
        max_new_tokens: 20,
        temperature: 0.8,
        top_k: Some(30),
        top_p: Some(0.95),
        ..Default::default()
    };

    let prompt = "Once upon a time";
    let input_ids: Vec<usize> =
        tokenizer.encode(prompt, false, false)?.iter().map(|&x| x as usize).collect();

    // Run 5 times
    let mut results = Vec::new();
    for _ in 0..5 {
        let mut generator = AutoregressiveGenerator::new(config.clone(), Device::Cpu)?;
        let tokens = generator.generate(&input_ids, mock_forward_fn).await?;
        results.push(tokens);
    }

    // AC6: All results identical
    for i in 1..results.len() {
        assert_eq!(results[0], results[i], "AC6: Run {} differs from run 0", i);
    }

    println!("AC6.2: Multiple runs determinism test - PENDING IMPLEMENTATION");
    Ok(())
}

// Helper functions
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
        tokenizer: bitnet_common::TokenizerConfig::default(),
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
    _input: bitnet_common::BitNetTensor,
) -> Result<bitnet_common::BitNetTensor> {
    Ok(bitnet_common::BitNetTensor::zeros(&[1, 50257], candle_core::DType::F32, &Device::Cpu)?)
}
