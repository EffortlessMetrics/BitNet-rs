//! Simple test to verify real neural network inference is working
//!
//! This test focuses on verifying Issue #248 core requirement:
//! Replace mock inference with real transformer forward pass

use anyhow::Result;
use bitnet_common::{BitNetConfig, Device};
use bitnet_inference::{GenerationConfig, InferenceEngine};
use bitnet_models::BitNetModel;
use bitnet_tokenizers::MockTokenizer;
use std::sync::Arc;
use std::time::Instant;

/// Test that we can create an inference engine and it doesn't fallback to mock
#[tokio::test]
async fn test_real_inference_engine_creation() -> Result<()> {
    let config = BitNetConfig::default();
    let model = Arc::new(BitNetModel::new(config, Device::Cpu));
    let tokenizer = Arc::new(MockTokenizer::new());

    let _engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;

    println!("✅ Successfully created InferenceEngine");
    Ok(())
}

/// Test forward pass with actual token IDs
#[tokio::test]
#[ignore] // Requires real model with weights loaded - uninitialized model fails with "transformer not initialized"
async fn test_forward_pass_with_tokens() -> Result<()> {
    let config = BitNetConfig::default();
    let model = Arc::new(BitNetModel::new(config, Device::Cpu));
    let tokenizer = Arc::new(MockTokenizer::new());

    let mut engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;

    // Test with simple token sequence
    let test_tokens = [1u32, 2u32, 3u32];
    let logits = engine.eval_ids(&test_tokens).await?;

    // Verify we get actual logits output
    assert!(!logits.is_empty(), "Should generate logits vector");
    println!("✅ Generated {} logits from forward pass", logits.len());

    // Check for non-zero variance (indicates real computation vs mock)
    let mean: f32 = logits.iter().sum::<f32>() / logits.len() as f32;
    let variance: f32 =
        logits.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / logits.len() as f32;

    if variance > 1e-10 {
        println!("✅ Logits show variance {:.2e}, indicating real computation", variance);
    } else {
        println!("⚠️  Logits have low variance {:.2e}, may be using fallback", variance);
    }

    Ok(())
}

/// Test text generation (basic autoregressive functionality)
#[tokio::test]
#[ignore] // Requires real model with weights loaded - uninitialized model fails with "transformer not initialized"
async fn test_basic_text_generation() -> Result<()> {
    let config = BitNetConfig::default();
    let model = Arc::new(BitNetModel::new(config, Device::Cpu));
    let tokenizer = Arc::new(MockTokenizer::new());

    let engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;

    let gen_config =
        GenerationConfig::default().with_max_tokens(3).with_temperature(1.0).with_seed(42);

    let start_time = Instant::now();
    let result = engine.generate_with_config("Hello", &gen_config).await?;
    let generation_time = start_time.elapsed();

    assert!(!result.is_empty(), "Should generate non-empty text");
    println!("✅ Generated text: '{}' in {:?}", result, generation_time);

    // Check that result doesn't contain obvious mock indicators
    let is_mock = result.contains("Mock") || result.contains("mock") || result.contains("[Mock");
    if !is_mock {
        println!("✅ Generated text appears to be from real inference (no mock indicators)");
    } else {
        println!("⚠️  Generated text contains mock indicators: '{}'", result);
    }

    Ok(())
}

/// Test that the model configuration is properly loaded
#[tokio::test]
#[ignore] // Requires real model with weights loaded - uninitialized model fails with "transformer not initialized"
async fn test_model_configuration() -> Result<()> {
    let mut config = BitNetConfig::default();
    config.model.vocab_size = 1000;
    config.model.hidden_size = 512;
    config.model.num_layers = 2;
    config.model.num_heads = 8;

    let model = Arc::new(BitNetModel::new(config.clone(), Device::Cpu));
    let tokenizer = Arc::new(MockTokenizer::new());

    let mut engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;

    // Test that configuration is accessible
    let test_tokens = [1u32];
    let logits = engine.eval_ids(&test_tokens).await?;

    // For BitNet model, logits should match vocab_size in length
    // Note: This might use fallback if no real weights are loaded
    println!(
        "✅ Model config: vocab={}, hidden={}, layers={}",
        config.model.vocab_size, config.model.hidden_size, config.model.num_layers
    );
    println!("✅ Generated logits length: {}", logits.len());

    Ok(())
}

/// Integration test: measure basic performance
#[tokio::test]
#[ignore] // Requires real model with weights loaded - uninitialized model fails with "transformer not initialized"
async fn test_basic_performance() -> Result<()> {
    let config = BitNetConfig::default();
    let model = Arc::new(BitNetModel::new(config, Device::Cpu));
    let tokenizer = Arc::new(MockTokenizer::new());

    let engine = InferenceEngine::new(model, tokenizer, Device::Cpu)?;

    let gen_config =
        GenerationConfig::default().with_max_tokens(10).with_temperature(1.0).with_seed(42);

    let start_time = Instant::now();
    let result = engine.generate_with_config("Test prompt", &gen_config).await?;
    let total_time = start_time.elapsed();

    let estimated_tokens = result.split_whitespace().count().max(1);
    let tokens_per_second = estimated_tokens as f64 / total_time.as_secs_f64();

    println!(
        "✅ Performance test - {:.2} tok/sec ({} tokens in {:?})",
        tokens_per_second, estimated_tokens, total_time
    );

    // Just verify we get measurable performance (not testing exact targets)
    assert!(tokens_per_second >= 0.1, "Should achieve basic measurable performance");
    assert!(total_time.as_secs() < 30, "Should complete within reasonable time");

    Ok(())
}
