//! BitNet model implementation

use crate::transformer::{KVCache, TransformerModel};
use bitnet_common::{
    BitNetConfig, BitNetError, BitNetTensor, ConcreteTensor, Device, Result, Tensor,
};
use candle_core::{DType, Tensor as CandleTensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::sync::Arc;

/// Trait for BitNet models
pub trait Model: Send + Sync {
    fn config(&self) -> &BitNetConfig;
    fn forward(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor>;
    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor>;
    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor>;
}

/// BitNet model implementation
pub struct BitNetModel {
    config: BitNetConfig,
    device: Device,
    tensors: HashMap<String, CandleTensor>,
    transformer: Option<Arc<TransformerModel>>,
}

impl BitNetModel {
    pub fn new(config: BitNetConfig, device: Device) -> Self {
        Self { config, device, tensors: HashMap::new(), transformer: None }
    }

    /// Create a BitNet model from GGUF tensors
    pub fn from_gguf(
        config: BitNetConfig,
        tensors: HashMap<String, CandleTensor>,
        device: Device,
    ) -> Result<Self> {
        // Validate that required tensors are present
        // LM head can be tied to embeddings, so check for either output.weight or embeddings
        let has_output = tensors.contains_key("output.weight")
            || tensors.contains_key("lm_head.weight")
            || tensors.contains_key("head.weight");

        let has_embeddings = tensors.contains_key("token_embd.weight")
            || tensors.contains_key("tok_embeddings.weight")
            || tensors.contains_key("model.embed_tokens.weight");

        if !has_embeddings {
            return Err(BitNetError::Validation(
                "Missing required tensor: token embeddings (token_embd.weight or equivalent)"
                    .to_string(),
            ));
        }

        if !has_output && !has_embeddings {
            return Err(BitNetError::Validation(
                "Missing both output.weight and token_embd.weight - cannot compute logits"
                    .to_string(),
            ));
        }

        // Try to build transformer model if we have weights
        let transformer = match Self::build_transformer(&config, &tensors, &device) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::warn!("Failed to build transformer model: {}", e);
                None
            }
        };

        Ok(Self { config, device, tensors, transformer })
    }

    /// Build transformer model from loaded tensors
    fn build_transformer(
        config: &BitNetConfig,
        tensors: &HashMap<String, CandleTensor>,
        device: &Device,
    ) -> Result<Arc<TransformerModel>> {
        use crate::weight_mapper::{
            create_var_builder, normalize_model_tensors, remap_gguf_weights,
        };

        // Create a VarBuilder that uses our loaded tensors
        let device = match device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda(id) => candle_core::Device::new_cuda(*id)?,
            Device::Metal => {
                return Err(BitNetError::Validation("Metal not yet supported".to_string()));
            }
        };

        // If we have tensors, try to use them
        let vb = if !tensors.is_empty() {
            // Remap tensor names to match our transformer module structure
            let mut mapped = remap_gguf_weights(tensors)?;

            // Normalize embeddings and lm_head tensors, detect vocab size and hidden size
            let (detected_vocab, detected_hidden) =
                normalize_model_tensors(&mut mapped, config.model.hidden_size)?;

            // Update config with detected values
            let mut updated_config = config.clone();
            if updated_config.model.vocab_size != detected_vocab {
                tracing::info!(
                    "Updating vocab_size from {} to {} based on tensor shapes",
                    updated_config.model.vocab_size,
                    detected_vocab
                );
                updated_config.model.vocab_size = detected_vocab;
            }
            if updated_config.model.hidden_size != detected_hidden {
                tracing::info!(
                    "Updating hidden_size from {} to {} based on tensor shapes",
                    updated_config.model.hidden_size,
                    detected_hidden
                );
                updated_config.model.hidden_size = detected_hidden;
            }

            let vb = create_var_builder(mapped, DType::F32, &device)?;
            let model = TransformerModel::new(updated_config, vb)?;
            return Ok(Arc::new(model));
        } else {
            // Fallback to zeros for testing
            VarBuilder::zeros(DType::F32, &device)
        };

        let model = TransformerModel::new(config.clone(), vb)?;
        Ok(Arc::new(model))
    }

    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&CandleTensor> {
        self.tensors.get(name)
    }

    /// List all tensor names
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Convert ConcreteTensor to Candle tensor
    fn to_candle_tensor(&self, tensor: &ConcreteTensor) -> Result<CandleTensor> {
        match tensor {
            ConcreteTensor::BitNet(t) => t.to_candle(),
            ConcreteTensor::Mock(mock) => {
                // Create a dummy tensor for mock
                let shape = mock.shape();
                let device = match self.device {
                    Device::Cpu => candle_core::Device::Cpu,
                    Device::Cuda(id) => candle_core::Device::new_cuda(id)?,
                    Device::Metal => {
                        return Err(BitNetError::Validation("Metal not yet supported".to_string()));
                    }
                };
                Ok(CandleTensor::zeros(shape, DType::F32, &device)?)
            }
        }
    }

    /// Convert Candle tensor to ConcreteTensor
    fn candle_to_concrete(&self, tensor: CandleTensor) -> ConcreteTensor {
        ConcreteTensor::BitNet(BitNetTensor::new(tensor))
    }
}

impl Model for BitNetModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        input: &ConcreteTensor,
        cache: &mut dyn std::any::Any,
    ) -> Result<ConcreteTensor> {
        if let Some(transformer) = &self.transformer {
            // Get or create KV cache
            let kv_cache = cache.downcast_mut::<KVCache>();

            // Convert input to Candle tensor
            let input_tensor = self.to_candle_tensor(input)?;

            // Run transformer forward pass
            let output = transformer.forward(&input_tensor, kv_cache)?;

            // Convert back to ConcreteTensor
            Ok(self.candle_to_concrete(output))
        } else {
            // Fallback to mock implementation
            let batch_size = input.shape()[0];
            let seq_len = input.shape()[1];
            let hidden_size = self.config.model.hidden_size;
            Ok(ConcreteTensor::mock(vec![batch_size, seq_len, hidden_size]))
        }
    }

    fn embed(&self, tokens: &[u32]) -> Result<ConcreteTensor> {
        if let Some(transformer) = &self.transformer {
            let embedded = transformer.embed(tokens)?;
            Ok(self.candle_to_concrete(embedded))
        } else {
            // Mock embedding
            let seq_len = tokens.len();
            let hidden_size = self.config.model.hidden_size;
            Ok(ConcreteTensor::mock(vec![1, seq_len, hidden_size]))
        }
    }

    fn logits(&self, hidden: &ConcreteTensor) -> Result<ConcreteTensor> {
        if let Some(transformer) = &self.transformer {
            let hidden_tensor = self.to_candle_tensor(hidden)?;
            let logits = transformer.logits(&hidden_tensor)?;
            Ok(self.candle_to_concrete(logits))
        } else {
            // Mock logits
            let batch_size = hidden.shape()[0];
            let seq_len = hidden.shape()[1];
            let vocab_size = self.config.model.vocab_size;
            Ok(ConcreteTensor::mock(vec![batch_size, seq_len, vocab_size]))
        }
    }
}
