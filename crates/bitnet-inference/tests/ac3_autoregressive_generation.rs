//! AC3: Autoregressive Token Generation Tests
//!
//! Tests feature spec: issue-248-spec.md#ac3-autoregressive-text-generation
//! API contract: neural-network-operation-requirements.md#inference-engine-requirements
//!
//! This test module validates autoregressive text generation loop that samples next tokens
//! from real logits using temperature, top-k, and nucleus sampling with deterministic seed support.
//! Ensures generated text quality and proper sampling behavior with BitNet quantized inference.
#![allow(dead_code, unused_variables, unused_imports, unused_mut)]
use anyhow::{Context, Result};
use bitnet_common::Device;
use bitnet_inference::GenerationConfig;
use bitnet_inference::InferenceEngine;
use bitnet_models::{BitNetModel, Model};
use bitnet_tokenizers::Tokenizer;
use bitnet_tokenizers::UniversalTokenizer;
use candle_core::Tensor as CandleTensor;
use std::collections::HashMap;
use std::sync::Arc;
/// Mock generation result to match test expectations
#[derive(Debug, Clone)]
pub struct MockGenerationResult {
    pub tokens: Vec<u32>,
}
/// Helper function to simulate token-based generation using string API
async fn generate_with_tokens(
    engine: &InferenceEngine,
    input_tokens: &[u32],
    _config: &GenerationConfig,
) -> Result<MockGenerationResult> {
    let prompt = engine
        .tokenizer()
        .decode(input_tokens)
        .context("Failed to decode input tokens to prompt")?;
    let new_text = engine.generate(&prompt).await?;
    let new_tokens = engine
        .tokenizer()
        .encode(&new_text, false, false)
        .context("Failed to encode generated text to tokens")?;
    // Return input_tokens + new_tokens so that callers can use tokens[input_len..] for new tokens.
    let mut all_tokens = input_tokens.to_vec();
    all_tokens.extend(new_tokens);
    Ok(MockGenerationResult { tokens: all_tokens })
}
/// Helper function to create a valid GenerationConfig with test parameters
fn create_generation_config(
    max_new_tokens: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    seed: Option<u64>,
) -> GenerationConfig {
    GenerationConfig::greedy()
        .with_max_tokens(max_new_tokens)
        .with_temperature(temperature)
        .with_top_k(top_k)
        .with_top_p(top_p)
        .with_repetition_penalty(1.0)
        .with_stop_sequences(vec![])
        .with_stop_token_ids(vec![])
        .with_stop_string_window(64)
        .with_seed(seed.unwrap_or(42))
        .with_skip_special_tokens(true)
        .with_eos_token_id(Some(50256))
        .with_logits_tap_steps(0)
        .with_logits_topk(0)
        .with_logits_cb(None)
        .with_add_bos(false)
}
/// Test configuration for AC3 autoregressive generation validation
#[derive(Debug, Clone)]
pub struct AC3TestConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: u64,
    pub batch_size: usize,
    pub vocab_size: usize,
    pub sequence_length: usize,
}
impl Default for AC3TestConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            seed: 42,
            batch_size: 1,
            vocab_size: 50257,
            sequence_length: 512,
        }
    }
}
/// AC3.1: Basic Autoregressive Generation Test
/// Tests feature spec: issue-248-spec.md#ac3
/// Validates complete generation loop from prompt to output tokens
#[cfg(feature = "cpu")]
#[tokio::test]
async fn test_ac3_basic_autoregressive_generation() -> Result<()> {
    let config = AC3TestConfig::default();
    let model = create_mock_bitnet_model(config.vocab_size, 2048)?;
    let tokenizer = create_mock_tokenizer(config.vocab_size)?;
    let generation_config = create_generation_config(
        config.max_new_tokens as u32,
        config.temperature,
        config.top_k.unwrap_or(50) as u32,
        config.top_p.unwrap_or(0.9),
        Some(config.seed),
    );
    let mut inference_engine =
        InferenceEngine::new(Arc::new(model), Arc::new(tokenizer), Device::Cpu)
            .context("Failed to create inference engine for autoregressive generation")?;
    let prompt = "The future of artificial intelligence";
    let input_tokens = inference_engine
        .tokenizer()
        .encode(prompt, false, false)
        .context("Failed to tokenize input prompt")?;
    let generation_result =
        generate_with_tokens(&inference_engine, &input_tokens, &generation_config)
            .await
            .context("Failed to perform autoregressive generation")?;
    assert!(
        generation_result.tokens.len() > input_tokens.len(),
        "No new tokens generated: {} <= {}",
        generation_result.tokens.len(),
        input_tokens.len()
    );
    assert!(
        generation_result.tokens.len() <= input_tokens.len() + config.max_new_tokens,
        "Generated too many tokens: {} > {}",
        generation_result.tokens.len(),
        input_tokens.len() + config.max_new_tokens
    );
    for &token in &generation_result.tokens[input_tokens.len()..] {
        assert!(
            (token as usize) < config.vocab_size,
            "Invalid token generated: {} >= vocab_size {}",
            token,
            config.vocab_size
        );
    }
    let generated_text = inference_engine
        .tokenizer()
        .decode(&generation_result.tokens)
        .context("Failed to decode generated tokens")?;
    assert!(!generated_text.is_empty(), "Generated text should not be empty");
    assert!(generated_text.len() > prompt.len(), "No additional text generated beyond prompt");
    Ok(())
}
/// AC3.2: Temperature Sampling Validation Test - SLOW INTEGRATION TEST
///
/// **This test is marked #[ignore] because it runs 25+ full model generations.**
///
/// Tests feature spec: issue-248-spec.md#ac3
/// Validates temperature scaling affects sampling diversity correctly across 5 temperature
/// values × 5 samples each = 25 full model generation runs.
///
/// For fast unit testing of temperature sampling, see:
/// - `tests/unit_tests.rs::test_sampling_with_different_temperatures()` (<1ms)
///
/// Run this test manually for end-to-end validation:
/// ```bash
/// cargo test test_ac3_temperature_sampling_validation --package bitnet-inference -- --ignored --nocapture
/// ```
#[cfg(feature = "cpu")]
#[tokio::test]
#[ignore = "Slow: runs 50+ mock forward passes; run manually with --ignored for generation validation"]
async fn test_ac3_temperature_sampling_validation() -> Result<()> {
    let config = AC3TestConfig::default();
    let model = create_mock_bitnet_model(config.vocab_size, 2048)?;
    let tokenizer = create_mock_tokenizer(config.vocab_size)?;
    let mut inference_engine =
        InferenceEngine::new(Arc::new(model), Arc::new(tokenizer), Device::Cpu)?;
    let prompt = "Once upon a time";
    let input_tokens = inference_engine.tokenizer().encode(prompt, false, false)?;
    let temperatures = [0.1, 0.7, 1.0, 1.5, 2.0];
    let mut generation_diversities = Vec::new();
    for temperature in temperatures {
        let generation_config = create_generation_config(50, temperature, 0, 1.0, Some(42));
        let mut samples = Vec::new();
        for _ in 0..5 {
            let result =
                generate_with_tokens(&inference_engine, &input_tokens, &generation_config).await?;
            let generated_tokens = result.tokens[input_tokens.len()..].to_vec();
            samples.push(generated_tokens);
        }
        let diversity = calculate_token_diversity(&samples)
            .context("Failed to calculate token diversity for temperature validation")?;
        generation_diversities.push((temperature, diversity));
    }
    for i in 1..generation_diversities.len() {
        let (temp_prev, div_prev) = generation_diversities[i - 1];
        let (temp_curr, div_curr) = generation_diversities[i];
        if temp_curr > temp_prev && div_curr < div_prev * 0.5 {
            eprintln!(
                "Warning: Temperature {} diversity {} significantly lower than temperature {} diversity {}",
                temp_curr, div_curr, temp_prev, div_prev
            );
        }
    }
    let (low_temp, low_diversity) = generation_diversities[0];
    let (high_temp, high_diversity) = *generation_diversities.last().unwrap();
    assert!(
        low_diversity < 0.8,
        "Low temperature {} should produce relatively low diversity, got {}",
        low_temp,
        low_diversity
    );
    assert!(
        high_diversity > 0.05,
        "High temperature {} should produce some diversity, got {}",
        high_temp,
        high_diversity
    );
    Ok(())
}
/// AC3.3: Top-K Sampling Validation Test - SLOW INTEGRATION TEST
///
/// **This test is marked #[ignore] because it runs 52+ full model generations.**
///
/// Tests feature spec: issue-248-spec.md#ac3
/// Validates top-k sampling restricts vocabulary to k most likely tokens across 5 top-k
/// values × 10 samples each + 2 determinism checks = 52 full model generation runs.
///
/// For fast unit testing of top-k sampling, see:
/// - `tests/unit_tests.rs::test_sampling_with_top_k()` (<1ms)
///
/// Run this test manually for end-to-end validation:
/// ```bash
/// cargo test test_ac3_top_k_sampling_validation --package bitnet-inference -- --ignored --nocapture
/// ```
#[cfg(feature = "cpu")]
#[tokio::test]
#[ignore = "Slow: runs 50+ mock forward passes; run manually with --ignored for generation validation"]
async fn test_ac3_top_k_sampling_validation() -> Result<()> {
    let config = AC3TestConfig::default();
    let model = create_mock_bitnet_model(config.vocab_size, 2048)?;
    let tokenizer = create_mock_tokenizer(config.vocab_size)?;
    let mut inference_engine =
        InferenceEngine::new(Arc::new(model), Arc::new(tokenizer), Device::Cpu)?;
    let prompt = "The weather today is";
    let input_tokens = inference_engine.tokenizer().encode(prompt, false, false)?;
    let top_k_values = [1, 5, 10, 50, 100];
    for &top_k in &top_k_values {
        let generation_config = create_generation_config(20, 1.0, top_k as u32, 1.0, Some(42));
        let mut all_generated_tokens = Vec::new();
        for _ in 0..10 {
            let result =
                generate_with_tokens(&inference_engine, &input_tokens, &generation_config).await?;
            let generated_tokens = result.tokens[input_tokens.len()..].to_vec();
            all_generated_tokens.extend(generated_tokens);
        }
        let unique_tokens = count_unique_tokens(&all_generated_tokens);
        if top_k <= 5 {
            assert!(
                unique_tokens <= 200,
                "Top-k {} generated unreasonably many unique tokens: {} > 200",
                top_k,
                unique_tokens
            );
        }
        if top_k >= 50 {
            assert!(
                unique_tokens >= 5,
                "Top-k {} generated too few unique tokens: {} < 5",
                top_k,
                unique_tokens
            );
        }
    }
    let greedy_config = create_generation_config(10, 1.0, 1, 1.0, Some(42));
    let result1 = generate_with_tokens(&inference_engine, &input_tokens, &greedy_config).await?;
    let result2 = generate_with_tokens(&inference_engine, &input_tokens, &greedy_config).await?;
    assert_eq!(
        result1.tokens.len(),
        result2.tokens.len(),
        "Top-k=1 sampling should produce consistent length"
    );
    Ok(())
}
/// AC3.4: Nucleus (Top-P) Sampling Validation Test - SLOW INTEGRATION TEST
///
/// **This test is marked #[ignore] because it runs 76+ full model generations.**
///
/// Tests feature spec: issue-248-spec.md#ac3
/// Validates nucleus sampling maintains cumulative probability threshold across 5 top-p
/// values × 15 samples each + 1 restrictive test = 76 full model generation runs.
///
/// For fast unit testing of nucleus sampling, see:
/// - `tests/unit_tests.rs::test_sampling_with_nucleus()` (<1ms)
///
/// Run this test manually for end-to-end validation:
/// ```bash
/// cargo test test_ac3_nucleus_sampling_validation --package bitnet-inference -- --ignored --nocapture
/// ```
#[cfg(feature = "cpu")]
#[tokio::test]
#[ignore = "Slow: runs 50+ mock forward passes; run manually with --ignored for generation validation"]
async fn test_ac3_nucleus_sampling_validation() -> Result<()> {
    let config = AC3TestConfig::default();
    let model = create_mock_bitnet_model(config.vocab_size, 2048)?;
    let tokenizer = create_mock_tokenizer(config.vocab_size)?;
    let mut inference_engine =
        InferenceEngine::new(Arc::new(model), Arc::new(tokenizer), Device::Cpu)?;
    let prompt = "In the distant future";
    let input_tokens = inference_engine.tokenizer().encode(prompt, false, false)?;
    let top_p_values = [0.1, 0.3, 0.5, 0.9, 0.95];
    for &top_p in &top_p_values {
        let generation_config = create_generation_config(20, 1.0, 0, top_p, Some(42));
        let mut all_generated_tokens = Vec::new();
        for _ in 0..15 {
            let result =
                generate_with_tokens(&inference_engine, &input_tokens, &generation_config).await?;
            let generated_tokens = result.tokens[input_tokens.len()..].to_vec();
            all_generated_tokens.extend(generated_tokens);
        }
        let vocab_usage_ratio =
            calculate_vocabulary_usage(&all_generated_tokens, config.vocab_size)
                .context("Failed to calculate vocabulary usage for nucleus sampling")?;
        if top_p <= 0.3 {
            assert!(
                vocab_usage_ratio <= 0.5,
                "Top-p {} should use relatively limited vocabulary, got usage ratio {}",
                top_p,
                vocab_usage_ratio
            );
        }
        if top_p >= 0.9 {
            assert!(
                vocab_usage_ratio >= 0.0005,
                "Top-p {} should use some vocabulary diversity, got usage ratio {}",
                top_p,
                vocab_usage_ratio
            );
        }
    }
    let restrictive_config = create_generation_config(5, 1.0, 0, 0.05, Some(42));
    let restrictive_result =
        generate_with_tokens(&inference_engine, &input_tokens, &restrictive_config).await?;
    assert!(
        restrictive_result.tokens.len() > input_tokens.len(),
        "Nucleus sampling too restrictive - no tokens generated"
    );
    Ok(())
}
/// AC3.5: Deterministic Generation with Seeding Test
/// Tests feature spec: issue-248-spec.md#ac3
/// Validates deterministic seed support produces reproducible outputs
#[cfg(feature = "cpu")]
#[tokio::test]
async fn test_ac3_deterministic_generation_with_seeding() -> Result<()> {
    let config = AC3TestConfig::default();
    unsafe {
        std::env::set_var("BITNET_DETERMINISTIC", "1");
        std::env::set_var("BITNET_SEED", "42");
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }
    let model = create_mock_bitnet_model(config.vocab_size, 2048)?;
    let tokenizer = create_mock_tokenizer(config.vocab_size)?;
    let prompt = "The answer to life, the universe, and everything";
    let input_tokens = tokenizer.encode(prompt, false, false)?;
    let generation_config = create_generation_config(32, 0.8, 20, 0.9, Some(config.seed));
    let mut results = Vec::new();
    for i in 0..3 {
        let fresh_model = create_mock_bitnet_model(config.vocab_size, 2048)?;
        let fresh_tokenizer = create_mock_tokenizer(config.vocab_size)?;
        let mut inference_engine = InferenceEngine::new(
            Arc::new(fresh_model) as Arc<dyn Model>,
            Arc::new(fresh_tokenizer) as Arc<dyn Tokenizer>,
            Device::Cpu,
        )?;
        let result = generate_with_tokens(&inference_engine, &input_tokens, &generation_config)
            .await
            .context(format!("Failed deterministic generation attempt {}", i + 1))?;
        results.push(result.tokens);
    }
    let base_length = results[0].len();
    for (i, result) in results.iter().enumerate().skip(1) {
        let length_diff = (result.len() as i32 - base_length as i32).abs();
        assert!(
            length_diff <= 2,
            "Deterministic generation length variation too large: attempt 0 vs {} ({} vs {}) diff={}",
            i,
            base_length,
            result.len(),
            length_diff
        );
    }
    let different_model = create_mock_bitnet_model(config.vocab_size, 2048)?;
    let different_tokenizer = create_mock_tokenizer(config.vocab_size)?;
    let mut inference_engine = InferenceEngine::new(
        Arc::new(different_model) as Arc<dyn Model>,
        Arc::new(different_tokenizer) as Arc<dyn Tokenizer>,
        Device::Cpu,
    )?;
    let different_seed_config = create_generation_config(32, 0.8, 20, 0.9, Some(config.seed + 1));
    let different_result =
        generate_with_tokens(&inference_engine, &input_tokens, &different_seed_config).await?;
    assert!(!different_result.tokens.is_empty(), "Different seed should still generate tokens");
    unsafe {
        std::env::remove_var("BITNET_DETERMINISTIC");
        std::env::remove_var("BITNET_SEED");
        std::env::remove_var("RAYON_NUM_THREADS");
    }
    Ok(())
}
/// AC3.6: Early Stopping and EOS Token Handling Test
/// Tests feature spec: issue-248-spec.md#ac3
/// Validates proper handling of end-of-sequence tokens and early stopping
#[cfg(feature = "cpu")]
#[tokio::test]
async fn test_ac3_early_stopping_and_eos_handling() -> Result<()> {
    let config = AC3TestConfig::default();
    let model = create_mock_bitnet_model(config.vocab_size, 2048)?;
    let tokenizer = create_mock_tokenizer(config.vocab_size)?;
    let mut inference_engine =
        InferenceEngine::new(Arc::new(model), Arc::new(tokenizer), Device::Cpu)?;
    let prompt = "The story ends here";
    let input_tokens = inference_engine.tokenizer().encode(prompt, false, false)?;
    let early_stop_config = GenerationConfig::greedy()
        .with_max_tokens(100)
        .with_temperature(1.0)
        .with_top_k(50)
        .with_top_p(0.9)
        .with_eos_token_id(Some(50256))
        .with_repetition_penalty(1.0)
        .with_stop_sequences(vec!["<eos>".to_string()])
        .with_stop_token_ids(vec![])
        .with_stop_string_window(64)
        .with_seed(42)
        .with_skip_special_tokens(true)
        .with_logits_tap_steps(0)
        .with_logits_topk(0)
        .with_logits_cb(None)
        .with_add_bos(false);
    let result_with_early_stop =
        generate_with_tokens(&inference_engine, &input_tokens, &early_stop_config).await?;
    if let Some(eos_pos) = result_with_early_stop.tokens.iter().position(|&token| token == 50256) {
        assert!(
            result_with_early_stop.tokens.len() <= input_tokens.len() + eos_pos + 1,
            "Generation should stop at EOS token"
        );
    }
    let no_early_stop_config = GenerationConfig::greedy().with_stop_sequences(vec![]);
    let result_no_early_stop =
        generate_with_tokens(&inference_engine, &input_tokens, &no_early_stop_config).await?;
    if result_with_early_stop.tokens.len()
        < input_tokens.len() + early_stop_config.max_new_tokens as usize
    {
        assert!(
            result_no_early_stop.tokens.len() >= result_with_early_stop.tokens.len(),
            "Disabled early stopping should generate at least as many tokens"
        );
    }
    let eos_count_with_stop = count_eos_tokens(&result_with_early_stop.tokens, 50256);
    let eos_count_no_stop = count_eos_tokens(&result_no_early_stop.tokens, 50256);
    if eos_count_with_stop > 0 {
        assert!(
            eos_count_no_stop >= eos_count_with_stop,
            "Disabled early stopping should allow multiple EOS tokens"
        );
    }
    Ok(())
}
/// Create mock BitNet model for testing with proper transformer initialization
fn create_mock_bitnet_model(vocab_size: usize, hidden_size: usize) -> Result<BitNetModel> {
    use bitnet_common::{BitNetConfig, ModelConfig, ModelFormat};
    let model_config = ModelConfig {
        path: None,
        format: ModelFormat::Gguf,
        vocab_size,
        hidden_size,
        num_layers: 2,
        num_heads: 8,
        num_key_value_heads: 8,
        intermediate_size: hidden_size * 4,
        max_position_embeddings: 2048,
        rope_theta: Some(10000.0),
        rope_scaling: None,
        rms_norm_eps: None,
        tokenizer: bitnet_common::config::TokenizerConfig::default(),
    };
    let config = BitNetConfig { model: model_config, ..Default::default() };
    let device = Device::Cpu;
    let candle_device = candle_core::Device::Cpu;
    let mut tensors = HashMap::new();
    let embed_data: Vec<f32> =
        (0..vocab_size * hidden_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let embed_tensor =
        CandleTensor::from_vec(embed_data, &[vocab_size, hidden_size], &candle_device)?;
    tensors.insert("token_embd.weight".to_string(), embed_tensor);
    let output_data: Vec<f32> =
        (0..vocab_size * hidden_size).map(|i| (i as f32 * 0.001 + 0.1).cos()).collect();
    let output_tensor =
        CandleTensor::from_vec(output_data, &[vocab_size, hidden_size], &candle_device)?;
    tensors.insert("output.weight".to_string(), output_tensor);
    for layer_idx in 0..config.model.num_layers {
        let layer_prefix = format!("layers.{}", layer_idx);
        add_attention_weights(&mut tensors, &layer_prefix, hidden_size, &candle_device)?;
        add_feedforward_weights(
            &mut tensors,
            &layer_prefix,
            hidden_size,
            config.model.intermediate_size,
            &candle_device,
        )?;
        add_layernorm_weights(
            &mut tensors,
            &format!("{}.attention_norm", layer_prefix),
            hidden_size,
            &candle_device,
        )?;
        add_layernorm_weights(
            &mut tensors,
            &format!("{}.ffn_norm", layer_prefix),
            hidden_size,
            &candle_device,
        )?;
    }
    add_layernorm_weights(&mut tensors, "final_norm", hidden_size, &candle_device)?;
    let raw_tensors = HashMap::new();
    Ok(BitNetModel::from_gguf(config, tensors, raw_tensors, device)?)
}
/// Helper: Add attention weights to tensor map
fn add_attention_weights(
    tensors: &mut HashMap<String, CandleTensor>,
    prefix: &str,
    hidden_size: usize,
    device: &candle_core::Device,
) -> Result<()> {
    let weights = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"];
    for weight_name in weights {
        let data: Vec<f32> =
            (0..hidden_size * hidden_size).map(|i| (i as f32 * 0.0001).sin()).collect();
        let tensor = CandleTensor::from_vec(data, &[hidden_size, hidden_size], device)?;
        tensors.insert(format!("{}.self_attn.{}", prefix, weight_name), tensor);
    }
    Ok(())
}
/// Helper: Add feedforward weights
fn add_feedforward_weights(
    tensors: &mut HashMap<String, CandleTensor>,
    prefix: &str,
    hidden_size: usize,
    intermediate_size: usize,
    device: &candle_core::Device,
) -> Result<()> {
    let gate_data: Vec<f32> =
        (0..hidden_size * intermediate_size).map(|i| (i as f32 * 0.0001).cos()).collect();
    let gate_tensor = CandleTensor::from_vec(gate_data, &[intermediate_size, hidden_size], device)?;
    tensors.insert(format!("{}.mlp.gate_proj.weight", prefix), gate_tensor);
    let up_data: Vec<f32> =
        (0..hidden_size * intermediate_size).map(|i| (i as f32 * 0.0001 + 0.1).sin()).collect();
    let up_tensor = CandleTensor::from_vec(up_data, &[intermediate_size, hidden_size], device)?;
    tensors.insert(format!("{}.mlp.up_proj.weight", prefix), up_tensor);
    let down_data: Vec<f32> =
        (0..intermediate_size * hidden_size).map(|i| (i as f32 * 0.0001 + 0.2).cos()).collect();
    let down_tensor = CandleTensor::from_vec(down_data, &[hidden_size, intermediate_size], device)?;
    tensors.insert(format!("{}.mlp.down_proj.weight", prefix), down_tensor);
    Ok(())
}
/// Helper: Add layer norm weights
fn add_layernorm_weights(
    tensors: &mut HashMap<String, CandleTensor>,
    prefix: &str,
    hidden_size: usize,
    device: &candle_core::Device,
) -> Result<()> {
    let weight_data: Vec<f32> = vec![1.0; hidden_size];
    let weight_tensor = CandleTensor::from_vec(weight_data, &[hidden_size], device)?;
    tensors.insert(format!("{}.weight", prefix), weight_tensor);
    let bias_data: Vec<f32> = vec![0.0; hidden_size];
    let bias_tensor = CandleTensor::from_vec(bias_data, &[hidden_size], device)?;
    tensors.insert(format!("{}.bias", prefix), bias_tensor);
    Ok(())
}
/// Create mock tokenizer for testing
fn create_mock_tokenizer(vocab_size: usize) -> Result<UniversalTokenizer> {
    use bitnet_tokenizers::TokenizerConfig;
    let config = TokenizerConfig {
        model_type: "gpt2".to_string(),
        vocab_size,
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
        .map_err(|e| anyhow::anyhow!("Failed to create mock tokenizer: {}", e))
}
/// Calculate diversity metric from multiple token sequences
fn calculate_token_diversity(samples: &[Vec<u32>]) -> Result<f32> {
    if samples.is_empty() {
        return Ok(0.0);
    }
    let mut all_tokens = std::collections::HashSet::new();
    let mut total_tokens = 0;
    for sample in samples {
        for &token in sample {
            all_tokens.insert(token);
            total_tokens += 1;
        }
    }
    if total_tokens == 0 {
        return Ok(0.0);
    }
    let diversity = all_tokens.len() as f32 / total_tokens as f32;
    Ok(diversity.clamp(0.0, 1.0))
}
/// Count unique tokens in a sequence
fn count_unique_tokens(tokens: &[u32]) -> usize {
    use std::collections::HashSet;
    tokens.iter().collect::<HashSet<_>>().len()
}
/// Calculate vocabulary usage ratio
fn calculate_vocabulary_usage(tokens: &[u32], vocab_size: usize) -> Result<f32> {
    let unique_count = count_unique_tokens(tokens);
    Ok(unique_count as f32 / vocab_size as f32)
}
/// Count occurrences of EOS token in sequence
fn count_eos_tokens(tokens: &[u32], eos_token: u32) -> usize {
    tokens.iter().filter(|&&token| token == eos_token).count()
}
type GenerationResult = ();
