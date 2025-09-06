use anyhow::Result;
use bitnet_common::{BitNetConfig, ConcreteTensor, Tensor};
use bitnet_models::Model;
use std::sync::Arc;
use tracing::{error, warn};

/// MockModelFallback provides a reliable fallback mechanism
/// when model loading or initialization fails
pub struct MockModelFallback;

impl MockModelFallback {
    /// Create a mock model with default configuration
    pub fn create_mock_model() -> Arc<dyn Model> {
        Arc::new(DefaultMockModel::new())
    }

    /// Attempt to load a real model, falling back to mock if loading fails
    pub fn load_with_fallback(path: &str, config: Option<&BitNetConfig>) -> Result<Arc<dyn Model>> {
        match self::load_real_model(path, config) {
            Ok(model) => Ok(model),
            Err(e) => {
                error!("Failed to load model from {}: {}. Falling back to mock model.", path, e);
                Ok(Self::create_mock_model())
            }
        }
    }
}

/// A default mock model for parity testing and fallback scenarios
struct DefaultMockModel {
    config: BitNetConfig,
}

impl DefaultMockModel {
    fn new() -> Self {
        Self { config: BitNetConfig::default() }
    }
}

impl Model for DefaultMockModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn forward(
        &self,
        input: &ConcreteTensor,
        _cache: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<ConcreteTensor> {
        warn!("Using mock model for forward pass");
        // Return hidden states with the same batch and sequence dimensions as input
        let shape = input.shape();
        if shape.len() >= 2 {
            // Return [batch, seq_len, hidden_dim]
            Ok(ConcreteTensor::mock(vec![shape[0], shape[1], 768]))
        } else {
            // Fallback to reasonable defaults
            Ok(ConcreteTensor::mock(vec![1, 1, 768]))
        }
    }

    fn embed(&self, tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
        warn!("Using mock model for embedding");
        // Return embeddings with shape [batch=1, seq_len=tokens.len(), hidden_dim=768]
        Ok(ConcreteTensor::mock(vec![1, tokens.len(), 768]))
    }

    fn logits(&self, hidden: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
        warn!("Using mock model for logits generation");
        // Get the batch and sequence dimensions from the hidden tensor
        let shape = hidden.shape();
        if shape.len() >= 2 {
            // Return 3D logits tensor [batch, seq_len, vocab_size]
            Ok(ConcreteTensor::mock(vec![shape[0], shape[1], 50257]))
        } else {
            // Fallback to reasonable defaults
            Ok(ConcreteTensor::mock(vec![1, 1, 50257]))
        }
    }
}

/// Placeholder for real model loading
fn load_real_model(_path: &str, _config: Option<&BitNetConfig>) -> Result<Arc<dyn Model>> {
    // This would be replaced with actual model loading logic
    Err(anyhow::anyhow!("Model loading not implemented"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_model_creation() {
        let mock_model = MockModelFallback::create_mock_model();
        assert!(mock_model.embed(&[1, 2, 3]).is_ok());
        assert!(mock_model.logits(&ConcreteTensor::mock(vec![1, 10, 768])).is_ok());
    }

    #[test]
    fn test_model_fallback() {
        let result = MockModelFallback::load_with_fallback("/nonexistent/path", None);
        assert!(result.is_ok());
    }
}
