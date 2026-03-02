//! Model loading utilities for the server

use anyhow::{Context, Result};
use bitnet_gguf_metadata_core::bitnet_config_from_metadata;
use bitnet_common::Device;
use bitnet_models::bitnet::BitNetModel;
use bitnet_models::Model;
use std::path::Path;
use std::sync::Arc;

/// Load a BitNet model from a GGUF file
pub fn load_model_from_gguf(path: &Path, device: Device) -> Result<Arc<dyn Model>> {
    // Use the minimal GGUF loader from gguf_min
    let (tensors, metadata) = bitnet_models::gguf_min::load_gguf_minimal(path)
        .with_context(|| format!("Failed to load GGUF file: {}", path.display()))?;

    // Extract model configuration from metadata
    let config = bitnet_config_from_metadata(&metadata)?;

    // Create the model
    let model = BitNetModel::from_gguf(config, tensors, device)?;

    Ok(Arc::new(model))
}

/// Load a dummy model for testing
pub fn load_dummy_model(vocab_size: usize, hidden_size: usize, device: Device) -> Arc<dyn Model> {
    let mut config = bitnet_common::BitNetConfig::default();
    config.model.vocab_size = vocab_size;
    config.model.hidden_size = hidden_size;
    config.model.num_layers = 12;
    config.model.num_heads = 12;
    config.model.intermediate_size = hidden_size * 4;

    let model = BitNetModel::new(config, device);
    Arc::new(model)
}
