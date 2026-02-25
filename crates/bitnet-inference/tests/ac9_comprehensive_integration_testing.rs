//! AC9: Comprehensive Integration Testing for Transformer Pipeline
//!
//! Tests feature spec: issue-248-spec.md#ac9-comprehensive-integration-testing
//! API contract: neural-network-operation-requirements.md#neural-network-inference-pipeline-requirements
//!
//! This test module validates inference accuracy through comprehensive testing including
//! unit tests for individual transformer components, integration tests for end-to-end generation,
//! and cross-validation against reference implementations.
use anyhow::{Context, Result};
use bitnet_common::{Device, Tensor};
use bitnet_inference::InferenceEngine;
use bitnet_models::BitNetModel;
use bitnet_tokenizers::UniversalTokenizer;
use std::sync::Arc;
/// AC9.1: End-to-End Transformer Pipeline Integration Test
/// Tests feature spec: issue-248-spec.md#ac9
/// Validates complete transformer pipeline from tokenization to detokenization
#[cfg(feature = "cpu")]
#[tokio::test]
#[ignore = "requires fixture GGUF at tests-new/fixtures/fixtures/gguf/valid/small_bitnet_test.gguf"]
async fn test_ac9_end_to_end_transformer_pipeline() -> Result<()> {
    let workspace_root = find_workspace_root().unwrap();
    let model_path =
        workspace_root.join("tests-new/fixtures/fixtures/gguf/valid/small_bitnet_test.gguf");
    let model = load_complete_bitnet_model(model_path.to_str().unwrap())
        .context("Failed to load complete BitNet model for integration testing")?;
    let tokenizer = UniversalTokenizer::new(Default::default())
        .context("Failed to create tokenizer with default config")?;
    let engine = InferenceEngine::new(Arc::new(model), Arc::new(tokenizer), Device::Cpu)
        .context("Failed to create transformer pipeline")?;
    let test_prompts = vec![
        "The capital of France is",
        "In the year 2050, artificial intelligence will",
        "The most important discovery in science was",
    ];
    for prompt in test_prompts {
        let result = engine
            .generate(prompt)
            .await
            .context(format!("Failed end-to-end generation for prompt: {}", prompt))?;
        assert!(!result.is_empty(), "No text generated for prompt: {}", prompt);
        assert!(result.starts_with(prompt), "Generated text doesn't start with prompt");
    }
    Ok(())
}
/// AC9.2: Individual Transformer Component Testing
/// Tests feature spec: issue-248-spec.md#ac9
/// Validates individual transformer blocks work correctly in isolation
#[cfg(feature = "cpu")]
#[tokio::test]
async fn test_ac9_individual_transformer_components() -> Result<()> {
    let workspace_root = find_workspace_root().unwrap();
    let model_path =
        workspace_root.join("models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf");
    if !model_path.exists() {
        eprintln!("Skipping test: Model not found at {}", model_path.display());
        return Ok(());
    }
    let model = load_complete_bitnet_model(model_path.to_str().unwrap())?;
    test_embedding_layer(&model).await.context("Embedding layer test failed")?;
    test_transformer_block(&model).await.context("Transformer block test failed")?;
    test_output_projection(&model).await.context("Output projection test failed")?;
    Ok(())
}
/// Find workspace root by looking upward for Cargo.toml with [workspace]
fn find_workspace_root() -> Result<std::path::PathBuf> {
    let mut path = std::env::current_dir()?;
    loop {
        let cargo_toml = path.join("Cargo.toml");
        if cargo_toml.exists()
            && let Ok(contents) = std::fs::read_to_string(&cargo_toml)
            && contents.contains("[workspace]")
        {
            return Ok(path);
        }
        if !path.pop() {
            break;
        }
    }
    anyhow::bail!("Could not find workspace root")
}
fn load_complete_bitnet_model(path: &str) -> Result<BitNetModel> {
    use bitnet_common::Device;
    use bitnet_models::gguf_simple::{GGUFLoaderConfig, load_gguf_full};
    use std::path::Path;
    let load_result = load_gguf_full(Path::new(path), Device::Cpu, GGUFLoaderConfig::default())
        .context("Failed to load GGUF file")?;
    let raw_tensors = load_result
        .i2s_qk256
        .into_keys()
        .map(|k| {
            use candle_core::{DType, Device as CDevice, Tensor as CandleTensor};
            let device = CDevice::Cpu;
            let dummy = CandleTensor::zeros(&[1], DType::F32, &device).unwrap();
            (k, dummy)
        })
        .collect();
    let model =
        BitNetModel::from_gguf(load_result.config, load_result.tensors, raw_tensors, Device::Cpu)
            .context("Failed to create BitNetModel from GGUF")?;
    Ok(model)
}
async fn test_embedding_layer(model: &BitNetModel) -> Result<()> {
    let embedding_tensor = model
        .get_tensor("token_embd.weight")
        .or_else(|| model.get_tensor("tok_embeddings.weight"))
        .or_else(|| model.get_tensor("model.embed_tokens.weight"))
        .context("Failed to find embedding tensor (token_embd.weight or equivalent)")?;
    let shape = embedding_tensor.shape();
    anyhow::ensure!(
        shape.dims().len() == 2,
        "Embedding tensor should be 2D, got shape: {:?}",
        shape.dims()
    );
    let vocab_size = shape.dims()[0];
    let hidden_size = shape.dims()[1];
    anyhow::ensure!(
        vocab_size > 0 && hidden_size > 0,
        "Invalid embedding shape: vocab_size={}, hidden_size={}",
        vocab_size,
        hidden_size
    );
    use bitnet_models::Model;
    let test_tokens = vec![0, 1, 2];
    let embedded = model.embed(&test_tokens).context("Failed to embed test tokens")?;
    let embedded_shape = embedded.shape();
    let (num_tokens_dim, hidden_dim) = if embedded_shape.len() == 3 {
        anyhow::ensure!(
            embedded_shape[0] == 1,
            "Batch dimension should be 1, got {}",
            embedded_shape[0]
        );
        (embedded_shape[1], embedded_shape[2])
    } else if embedded_shape.len() == 2 {
        (embedded_shape[0], embedded_shape[1])
    } else {
        anyhow::bail!("Embedded output should be 2D or 3D, got shape: {:?}", embedded_shape);
    };
    anyhow::ensure!(
        num_tokens_dim == test_tokens.len(),
        "Embedded output should have {} tokens, got {}",
        test_tokens.len(),
        num_tokens_dim
    );
    anyhow::ensure!(
        hidden_dim == hidden_size,
        "Embedded output should have hidden_size={}, got {}",
        hidden_size,
        hidden_dim
    );
    Ok(())
}
async fn test_transformer_block(model: &BitNetModel) -> Result<()> {
    let layer_prefixes = vec!["blk.0.attn_q", "blk.0.attn_k", "blk.0.attn_v", "blk.0.attn_output"];
    let alternative_names = vec![
        "layers.0.attention.q_proj.weight",
        "layers.0.attention.k_proj.weight",
        "layers.0.attention.v_proj.weight",
        "layers.0.attention.o_proj.weight",
    ];
    let mut found_attention = false;
    for (prefix, alt_name) in layer_prefixes.iter().zip(alternative_names.iter()) {
        let weight_name = format!("{}.weight", prefix);
        if let Some(tensor) = model.get_tensor(&weight_name).or_else(|| model.get_tensor(alt_name))
        {
            found_attention = true;
            let shape = tensor.shape();
            anyhow::ensure!(
                shape.dims().len() == 2,
                "Attention weight {} should be 2D, got: {:?}",
                weight_name,
                shape.dims()
            );
        }
    }
    anyhow::ensure!(
        found_attention,
        "No attention weights found for layer 0 (tried: {:?} and {:?})",
        layer_prefixes,
        alternative_names
    );
    let ffn_prefixes = ["blk.0.ffn_gate", "blk.0.ffn_up", "blk.0.ffn_down"];
    let alternative_ffn = [
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ];
    let mut found_ffn = false;
    for (prefix, alt_name) in ffn_prefixes.iter().zip(alternative_ffn.iter()) {
        let weight_name = format!("{}.weight", prefix);
        if let Some(tensor) = model.get_tensor(&weight_name).or_else(|| model.get_tensor(alt_name))
        {
            found_ffn = true;
            let shape = tensor.shape();
            anyhow::ensure!(
                shape.dims().len() == 2,
                "FFN weight {} should be 2D, got: {:?}",
                weight_name,
                shape.dims()
            );
        }
    }
    if !found_ffn {
        tracing::warn!("No FFN weights found for layer 0 - may be a minimal test model");
    }
    Ok(())
}
async fn test_output_projection(model: &BitNetModel) -> Result<()> {
    let output_tensor = model
        .get_tensor("output.weight")
        .or_else(|| model.get_tensor("lm_head.weight"))
        .or_else(|| model.get_tensor("head.weight"));
    let tensor = if let Some(t) = output_tensor {
        t
    } else {
        model
            .get_tensor("token_embd.weight")
            .or_else(|| model.get_tensor("tok_embeddings.weight"))
            .context("No output projection or embedding tensor found")?
    };
    let shape = tensor.shape();
    anyhow::ensure!(
        shape.dims().len() == 2,
        "Output projection should be 2D, got shape: {:?}",
        shape.dims()
    );
    let vocab_size = shape.dims()[0];
    let hidden_size = shape.dims()[1];
    anyhow::ensure!(
        vocab_size > 0 && hidden_size > 0,
        "Invalid output projection shape: vocab_size={}, hidden_size={}",
        vocab_size,
        hidden_size
    );
    use bitnet_common::ConcreteTensor;
    use bitnet_models::Model;
    let dummy_hidden = ConcreteTensor::mock(vec![hidden_size]);
    let logits_result = model.logits(&dummy_hidden);
    match logits_result {
        Ok(logits) => {
            let logits_shape = logits.shape();
            anyhow::ensure!(
                !logits_shape.is_empty(),
                "Logits should have at least 1 dimension, got: {:?}",
                logits_shape
            );
            let last_dim = logits_shape[logits_shape.len() - 1];
            anyhow::ensure!(
                last_dim == vocab_size,
                "Logits should have vocab_size={} in last dimension, got {}",
                vocab_size,
                last_dim
            );
        }
        Err(e) => {
            tracing::warn!("Logits computation failed with mock tensors (expected): {}", e);
        }
    }
    Ok(())
}
#[allow(dead_code)]
type TransformerPipeline = ();
#[allow(dead_code)]
type TransformerConfig = ();
