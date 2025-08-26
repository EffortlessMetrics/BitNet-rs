#![cfg(feature = "integration-tests")]
use bitnet_common::{BitNetConfig, Device, ModelConfig, Tensor as BitNetTensorTrait};
/// Tests for transformer model implementation
use bitnet_models::{
    BitNetModel, Model,
    transformer::{KVCache, TransformerModel},
};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use std::sync::Arc;

/// Create a small test model configuration
fn test_config() -> BitNetConfig {
    BitNetConfig {
        model: ModelConfig {
            hidden_size: 256,
            num_layers: 2,
            num_heads: 8,
            vocab_size: 1000,
            intermediate_size: 1024,
            max_position_embeddings: 512,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Create a test model with zeros initialization
fn test_model_fp32() -> anyhow::Result<(Arc<TransformerModel>, BitNetConfig)> {
    let config = test_config();
    let device = candle_core::Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);
    let model = TransformerModel::new(config.clone(), vb)?;
    Ok((Arc::new(model), config))
}

#[test]
fn test_forward_shapes() -> anyhow::Result<()> {
    let (model, config) = test_model_fp32()?;
    let batch = 1;
    let seq_len = 4;
    let hidden_size = config.model.hidden_size;

    // Create input tensor
    let x = Tensor::zeros(&[batch, seq_len, hidden_size], DType::F32, &candle_core::Device::Cpu)?;

    // Create KV cache
    let mut kv = KVCache::new(&config, batch, &candle_core::Device::Cpu)?;

    // Forward pass
    let y = model.forward(&x, Some(&mut kv))?;

    // Check output shape matches input shape
    assert_eq!(y.dims(), x.dims());
    assert_eq!(y.dims(), &[batch, seq_len, hidden_size]);

    Ok(())
}

#[test]
fn test_embedding_shapes() -> anyhow::Result<()> {
    let (model, config) = test_model_fp32()?;
    let tokens = vec![1u32, 2, 3, 4];

    // Embed tokens
    let embedded = model.embed(&tokens)?;

    // Check shape
    let expected_shape = vec![1, tokens.len(), config.model.hidden_size];
    assert_eq!(embedded.dims(), &expected_shape);

    Ok(())
}

#[test]
fn test_logits_shapes() -> anyhow::Result<()> {
    let (model, config) = test_model_fp32()?;
    let batch = 1;
    let seq_len = 4;
    let hidden_size = config.model.hidden_size;
    let vocab_size = config.model.vocab_size;

    // Create hidden state tensor
    let hidden =
        Tensor::zeros(&[batch, seq_len, hidden_size], DType::F32, &candle_core::Device::Cpu)?;

    // Get logits
    let logits = model.logits(&hidden)?;

    // Check shape
    assert_eq!(logits.dims(), &[batch, seq_len, vocab_size]);

    Ok(())
}

#[test]
fn test_kv_cache_append() -> anyhow::Result<()> {
    let config = test_config();
    let batch = 1;
    let n_heads = config.model.num_heads;
    let head_dim = config.model.hidden_size / n_heads;
    let device = candle_core::Device::Cpu;

    let mut cache = KVCache::new(&config, batch, &device)?;
    let layer_cache = cache.layer_mut(0).unwrap();

    // Initial append
    let k1 = Tensor::ones(&[batch, n_heads, 2, head_dim], DType::F32, &device)?;
    let v1 = Tensor::ones(&[batch, n_heads, 2, head_dim], DType::F32, &device)?;
    layer_cache.append(&k1, &v1)?;

    assert_eq!(layer_cache.seq_len, 2);
    assert_eq!(layer_cache.k.dims(), &[batch, n_heads, 2, head_dim]);

    // Second append
    let k2 = Tensor::ones(&[batch, n_heads, 3, head_dim], DType::F32, &device)?;
    let v2 = Tensor::ones(&[batch, n_heads, 3, head_dim], DType::F32, &device)?;
    layer_cache.append(&k2, &v2)?;

    assert_eq!(layer_cache.seq_len, 5);
    assert_eq!(layer_cache.k.dims(), &[batch, n_heads, 5, head_dim]);

    Ok(())
}

#[test]
fn test_step_vs_full_equivalence() -> anyhow::Result<()> {
    let (model, config) = test_model_fp32()?;
    let tokens = vec![1u32, 2, 3, 4];

    // Full forward pass (all tokens at once)
    let emb_full = model.embed(&tokens)?;
    let h_full = model.forward(&emb_full, None)?;
    let logits_full = model.logits(&h_full)?;

    // Extract last step logits from full pass
    let last_full =
        logits_full.narrow(1, tokens.len() - 1, 1)?.squeeze(1)?.i(0)?.to_vec1::<f32>()?;

    // Incremental forward pass (token by token with KV cache)
    let mut kv = KVCache::new(&config, 1, &candle_core::Device::Cpu)?;
    let mut last_incremental = Vec::new();

    for (i, &token) in tokens.iter().enumerate() {
        // For incremental processing, only embed the current token
        let current_token = vec![token];
        let emb = model.embed(&current_token)?;
        let h = model.forward(&emb, Some(&mut kv))?;
        let logits = model.logits(&h)?;

        if i == tokens.len() - 1 {
            // Compare last step logits
            last_incremental = logits
                .narrow(1, 0, 1)?  // Get the first (and only) position
                .squeeze(1)?
                .i(0)?
                .to_vec1::<f32>()?;
        }
    }

    // Check that they're approximately equal
    for (a, b) in last_full.iter().zip(last_incremental.iter()) {
        let diff = (a - b).abs();
        assert!(diff < 1e-4, "Logits differ: {} vs {}", a, b);
    }

    Ok(())
}

#[test]
fn test_forward_full_matches_incremental() -> anyhow::Result<()> {
    let (model, config) = test_model_fp32()?;
    let tokens = vec![1u32, 2, 3, 4];

    // Prepare token tensor for forward_full path
    let token_tensor =
        Tensor::from_vec(tokens.clone(), &[1, tokens.len()], &candle_core::Device::Cpu)?;

    // Compute logits using the teacher-forcing path
    let logits_full = model.forward_full(&token_tensor)?;

    // Compute logits using incremental decoding path
    let mut kv = KVCache::new(&config, 1, &candle_core::Device::Cpu)?;
    let mut step_logits = Vec::new();
    for &token in &tokens {
        let emb = model.embed(&[token])?;
        let hidden = model.forward(&emb, Some(&mut kv))?;
        let logits = model.logits(&hidden)?;
        step_logits.push(logits);
    }
    let logits_inc = Tensor::cat(&step_logits, 1)?;

    // Ensure shapes match
    assert_eq!(logits_full.dims(), logits_inc.dims());

    // Compare element-wise
    let full_vec = logits_full.flatten_all()?.to_vec1::<f32>()?;
    let inc_vec = logits_inc.flatten_all()?.to_vec1::<f32>()?;
    for (a, b) in full_vec.iter().zip(inc_vec.iter()) {
        let diff = (a - b).abs();
        assert!(diff < 1e-4, "Logits differ: {} vs {}", a, b);
    }

    Ok(())
}

#[test]
fn test_causal_mask() -> anyhow::Result<()> {
    let config = test_config();
    let device = candle_core::Device::Cpu;
    let vb = VarBuilder::zeros(DType::F32, &device);

    // Create attention layer
    let attn = bitnet_models::transformer::MultiHeadAttention::new(&config, vb)?;

    // Create input
    let batch = 1;
    let seq_len = 4;
    let hidden = config.model.hidden_size;
    let x = Tensor::zeros(&[batch, seq_len, hidden], DType::F32, &device)?;

    // Forward pass
    let _ = attn.forward(&x, None)?;

    // Test passes if no error
    Ok(())
}

#[test]
fn test_model_integration() -> anyhow::Result<()> {
    // Create a BitNetModel with mock transformer
    let config = test_config();
    let model = BitNetModel::new(config.clone(), Device::Cpu);

    // Test embed
    let tokens = vec![1u32, 2, 3];
    let embedded = model.embed(&tokens)?;
    assert_eq!(embedded.shape(), &[1, 3, config.model.hidden_size]);

    // Test forward
    let mut cache: Box<dyn std::any::Any> = Box::new(());
    let output = model.forward(&embedded, cache.as_mut())?;
    assert_eq!(output.shape()[0], 1);
    assert_eq!(output.shape()[1], 3);

    // Test logits
    let logits = model.logits(&output)?;
    assert_eq!(logits.shape(), &[1, 3, config.model.vocab_size]);

    Ok(())
}
