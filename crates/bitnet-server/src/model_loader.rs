//! Model loading utilities for the server

use anyhow::{Context, Result};
use bitnet_common::Device;
use bitnet_models::bitnet::BitNetModel;
use bitnet_models::Model;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Load a BitNet model from a GGUF file
pub fn load_model_from_gguf(path: &Path, device: Device) -> Result<Arc<dyn Model>> {
    // Use the minimal GGUF loader from gguf_min
    let (tensors, metadata) = bitnet_models::gguf_min::load_gguf_minimal(path)
        .with_context(|| format!("Failed to load GGUF file: {}", path.display()))?;

    // Extract model configuration from metadata
    let config = extract_config_from_metadata(&metadata)?;

    // Create the model
    let model = BitNetModel::from_gguf(config, tensors, device)?;

    Ok(Arc::new(model))
}

/// Extract BitNetConfig from GGUF metadata
fn extract_config_from_metadata(metadata: &HashMap<String, String>) -> Result<bitnet_common::BitNetConfig> {
    let mut config = bitnet_common::BitNetConfig::default();

    // Parse key metadata fields
    if let Some(vocab_size) = metadata.get("llama.vocab_size") {
        config.model.vocab_size = vocab_size.parse()
            .context("Failed to parse vocab_size")?;
    }

    if let Some(hidden_size) = metadata.get("llama.embedding_length") {
        config.model.hidden_size = hidden_size.parse()
            .context("Failed to parse hidden_size")?;
    }

    if let Some(num_layers) = metadata.get("llama.block_count") {
        config.model.num_hidden_layers = num_layers.parse()
            .context("Failed to parse num_hidden_layers")?;
    }

    if let Some(num_heads) = metadata.get("llama.attention.head_count") {
        config.model.num_attention_heads = num_heads.parse()
            .context("Failed to parse num_attention_heads")?;
    }

    if let Some(intermediate_size) = metadata.get("llama.feed_forward_length") {
        config.model.intermediate_size = intermediate_size.parse()
            .context("Failed to parse intermediate_size")?;
    }

    // Set quantization type based on tensor types found
    config.quantization.bits = 2; // BitNet uses 2-bit quantization
    config.quantization.group_size = 128; // Standard group size

    Ok(config)
}

/// Load a dummy model for testing
pub fn load_dummy_model(vocab_size: usize, hidden_size: usize, device: Device) -> Arc<dyn Model> {
    let mut config = bitnet_common::BitNetConfig::default();
    config.model.vocab_size = vocab_size;
    config.model.hidden_size = hidden_size;
    config.model.num_hidden_layers = 12;
    config.model.num_attention_heads = 12;
    config.model.intermediate_size = hidden_size * 4;

    let model = BitNetModel::new(config, device);
    Arc::new(model)
}
