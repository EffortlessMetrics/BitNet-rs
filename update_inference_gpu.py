import re

file_path = 'crates/bitnet-inference/src/gpu.rs'

with open(file_path, 'r') as f:
    content = f.read()

# 1. Update imports
search_imports = r"""use bitnet_models::Model;
use candle_core::Device;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use std::time::Instant;"""

replace_imports = r"""use bitnet_models::Model;
use candle_core::Device;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;"""

content = content.replace(search_imports, replace_imports)

# 2. Update forward_gpu
search_forward = r"""    /// Forward pass with GPU optimizations
    fn forward_gpu(&self, input: &BitNetTensor, _step: usize) -> Result<BitNetTensor> {
        let compute_start = Instant::now();

        // This is a simplified synchronous version
        // In a full async implementation, we would use model.read().await

        // For now, create a placeholder result
        let result = BitNetTensor::zeros(&[1, 32000], candle_core::DType::F32, &self.backend.device)?;

        // Update compute time metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.compute_time_ms = compute_start.elapsed().as_millis() as f64;
        }

        Ok(result)
    }"""

replace_forward = r"""    /// Forward pass with GPU optimizations
    fn forward_gpu(&self, input: &BitNetTensor, _step: usize) -> Result<BitNetTensor> {
        let compute_start = Instant::now();

        // Lock the model for reading
        // Using unwrap is safe here as we don't expect poisoning in normal operation
        let model = self.model.read().map_err(|_| BitNetError::Validation("Model lock poisoned".to_string()))?;

        // Perform forward pass
        // Input tensor is already on the correct device thanks to generate_tokens_gpu
        let result = model.forward(input)?;

        // Update compute time metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.compute_time_ms = compute_start.elapsed().as_millis() as f64;
        }

        Ok(result)
    }"""

content = content.replace(search_forward, replace_forward)

with open(file_path, 'w') as f:
    f.write(content)

print("Updated gpu.rs")
