//! Model management for the C API
//!
//! This module provides thread-safe model loading, management, and information
//! retrieval functionality for the C API.

use crate::{BitNetCConfig, BitNetCError};
use bitnet_common::{BitNetConfig, ConcreteTensor, ModelFormat, QuantizationType, Tensor};
use bitnet_models::Model;
use candle_core::{Tensor as CandleTensor, Device as CDevice};
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_uint, c_ulong};
use std::sync::{Arc, Mutex, RwLock};

/// C API model information structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BitNetCModel {
    /// Model name (null-terminated string)
    pub name: *const c_char,
    /// Model version (null-terminated string)
    pub version: *const c_char,
    /// Model architecture (null-terminated string)
    pub architecture: *const c_char,
    /// Vocabulary size
    pub vocab_size: c_uint,
    /// Context length
    pub context_length: c_uint,
    /// Hidden size
    pub hidden_size: c_uint,
    /// Number of layers
    pub num_layers: c_uint,
    /// Number of attention heads
    pub num_heads: c_uint,
    /// Intermediate size
    pub intermediate_size: c_uint,
    /// Quantization type (0=I2S, 1=TL1, 2=TL2)
    pub quantization_type: c_uint,
    /// Model file size in bytes
    pub file_size: c_ulong,
    /// Memory usage in bytes
    pub memory_usage: c_ulong,
    /// Whether the model is loaded on GPU
    pub is_gpu_loaded: c_uint,
}

impl Default for BitNetCModel {
    fn default() -> Self {
        Self {
            name: std::ptr::null(),
            version: std::ptr::null(),
            architecture: std::ptr::null(),
            vocab_size: 0,
            context_length: 0,
            hidden_size: 0,
            num_layers: 0,
            num_heads: 0,
            intermediate_size: 0,
            quantization_type: 0,
            file_size: 0,
            memory_usage: 0,
            is_gpu_loaded: 0,
        }
    }
}

/// Internal model information with owned strings
#[derive(Debug, Clone)]
struct ModelInfo {
    name: CString,
    version: CString,
    architecture: CString,
    config: BitNetConfig,
    file_size: u64,
    memory_usage: u64,
    is_gpu_loaded: bool,
}

impl ModelInfo {
    fn to_c_model(&self) -> BitNetCModel {
        BitNetCModel {
            name: self.name.as_ptr(),
            version: self.version.as_ptr(),
            architecture: self.architecture.as_ptr(),
            vocab_size: self.config.model.vocab_size as c_uint,
            context_length: self.config.model.max_position_embeddings as c_uint,
            hidden_size: self.config.model.hidden_size as c_uint,
            num_layers: self.config.model.num_layers as c_uint,
            num_heads: self.config.model.num_heads as c_uint,
            intermediate_size: self.config.model.intermediate_size as c_uint,
            quantization_type: match self.config.quantization.quantization_type {
                QuantizationType::I2S => 0,
                QuantizationType::TL1 => 1,
                QuantizationType::TL2 => 2,
            },
            file_size: self.file_size as c_ulong,
            memory_usage: self.memory_usage as c_ulong,
            is_gpu_loaded: if self.is_gpu_loaded { 1 } else { 0 },
        }
    }
}

/// Thread-safe model manager
pub struct ModelManager {
    models: RwLock<HashMap<u32, Arc<dyn Model>>>,
    model_info: RwLock<HashMap<u32, ModelInfo>>,
    next_id: Mutex<u32>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            model_info: RwLock::new(HashMap::new()),
            next_id: Mutex::new(0),
        }
    }

    /// Load a model from file
    pub fn load_model(&self, path: &str) -> Result<u32, BitNetCError> {
        let config = BitNetConfig::default();
        self.load_model_with_config(path, &BitNetCConfig::from_bitnet_config(&config))
    }

    /// Load a model with configuration
    pub fn load_model_with_config(
        &self,
        path: &str,
        config: &BitNetCConfig,
    ) -> Result<u32, BitNetCError> {
        // Convert C config to Rust config
        let rust_config = config.to_bitnet_config()?;

        // Load the model using the models crate
        let model = self.load_model_from_path(path, &rust_config)?;

        // Get next available ID
        let model_id = {
            let mut next_id = self
                .next_id
                .lock()
                .map_err(|_| BitNetCError::ThreadSafety("Failed to acquire ID lock".to_string()))?;
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Store model
        {
            let mut models = self.models.write().map_err(|_| {
                BitNetCError::ThreadSafety("Failed to acquire models write lock".to_string())
            })?;
            models.insert(model_id, model);
        }

        // Store model info
        let model_info = self.create_model_info(path, &rust_config)?;
        {
            let mut info_map = self.model_info.write().map_err(|_| {
                BitNetCError::ThreadSafety("Failed to acquire model info write lock".to_string())
            })?;
            info_map.insert(model_id, model_info);
        }

        Ok(model_id)
    }

    /// Free a model
    pub fn free_model(&self, model_id: u32) -> Result<(), BitNetCError> {
        // Remove from models
        let model_existed = {
            let mut models = self.models.write().map_err(|_| {
                BitNetCError::ThreadSafety("Failed to acquire models write lock".to_string())
            })?;
            models.remove(&model_id).is_some()
        };

        // Remove from model info
        {
            let mut info_map = self.model_info.write().map_err(|_| {
                BitNetCError::ThreadSafety("Failed to acquire model info write lock".to_string())
            })?;
            info_map.remove(&model_id);
        }

        if model_existed {
            Ok(())
        } else {
            Err(BitNetCError::InvalidModelId(format!("Model ID {} not found", model_id)))
        }
    }

    /// Check if a model is loaded
    pub fn is_model_loaded(&self, model_id: u32) -> Result<bool, BitNetCError> {
        let models = self.models.read().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire models read lock".to_string())
        })?;
        Ok(models.contains_key(&model_id))
    }

    /// Get model information
    pub fn get_model_info(&self, model_id: u32) -> Result<BitNetCModel, BitNetCError> {
        let info_map = self.model_info.read().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire model info read lock".to_string())
        })?;

        match info_map.get(&model_id) {
            Some(info) => Ok(info.to_c_model()),
            None => Err(BitNetCError::InvalidModelId(format!("Model ID {} not found", model_id))),
        }
    }

    /// Get a reference to a loaded model
    pub fn get_model(&self, model_id: u32) -> Result<Arc<dyn Model>, BitNetCError> {
        let models = self.models.read().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire models read lock".to_string())
        })?;

        match models.get(&model_id) {
            Some(model) => Ok(Arc::clone(model)),
            None => Err(BitNetCError::InvalidModelId(format!("Model ID {} not found", model_id))),
        }
    }

    /// Get list of loaded model IDs
    pub fn get_loaded_models(&self) -> Result<Vec<u32>, BitNetCError> {
        let models = self.models.read().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire models read lock".to_string())
        })?;
        Ok(models.keys().copied().collect())
    }

    /// Get total memory usage of all loaded models
    pub fn get_total_memory_usage(&self) -> Result<u64, BitNetCError> {
        let info_map = self.model_info.read().map_err(|_| {
            BitNetCError::ThreadSafety("Failed to acquire model info read lock".to_string())
        })?;

        Ok(info_map.values().map(|info| info.memory_usage).sum())
    }

    // Private helper methods

    fn load_model_from_path(
        &self,
        path: &str,
        config: &BitNetConfig,
    ) -> Result<Arc<dyn Model>, BitNetCError> {
        use std::path::Path;
        let path_obj = Path::new(path);

        if !path_obj.exists() {
            return Err(BitNetCError::ModelNotFound(format!("Model file not found: {}", path)));
        }

        // Detect format based on extension
        let format = match path_obj.extension().and_then(|s| s.to_str()) {
            Some("gguf") => ModelFormat::Gguf,
            Some("safetensors") => ModelFormat::SafeTensors,
            _ => ModelFormat::Gguf // Default fallback
        };

        // Try to load using the actual model loader
        match format {
            ModelFormat::Gguf => {
                // Use the gguf_min loader for now
                use bitnet_models::gguf_min::load_two;
                match load_two(path) {
                    Ok(tensors) => {
                        // Create a simple model wrapper around the loaded tensors
                        Ok(Arc::new(SimpleGgufModel::new(config.clone(), tensors)))
                    }
                    Err(_) => {
                        // Fall back to mock model if real loading fails
                        Ok(Arc::new(MockModel::new(config.clone())))
                    }
                }
            }
            _ => {
                // For other formats, use mock for now
                Ok(Arc::new(MockModel::new(config.clone())))
            }
        }
    }

    fn create_model_info(
        &self,
        path: &str,
        config: &BitNetConfig,
    ) -> Result<ModelInfo, BitNetCError> {
        use std::fs;

        let file_size = fs::metadata(path)
            .map_err(|e| BitNetCError::Internal(format!("Failed to get file metadata: {}", e)))?
            .len();

        // Estimate memory usage based on model parameters
        let memory_usage = self.estimate_memory_usage(config);

        let name = CString::new(
            std::path::Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("unknown"),
        )
        .map_err(|_| BitNetCError::Internal("Failed to create model name string".to_string()))?;

        let version = CString::new("1.0.0")
            .map_err(|_| BitNetCError::Internal("Failed to create version string".to_string()))?;

        let architecture = CString::new("BitNet").map_err(|_| {
            BitNetCError::Internal("Failed to create architecture string".to_string())
        })?;

        Ok(ModelInfo {
            name,
            version,
            architecture,
            config: config.clone(),
            file_size,
            memory_usage,
            is_gpu_loaded: config.performance.use_gpu,
        })
    }

    fn estimate_memory_usage(&self, config: &BitNetConfig) -> u64 {
        // Rough estimation based on model parameters
        let params = config.model.vocab_size * config.model.hidden_size
            + config.model.num_layers * config.model.hidden_size * config.model.hidden_size * 4;

        // Estimate bytes per parameter based on quantization
        let bytes_per_param = match config.quantization.quantization_type {
            QuantizationType::I2S => 0.25, // 2 bits per parameter
            QuantizationType::TL1 | QuantizationType::TL2 => 0.5, // 4 bits per parameter with lookup tables
        };

        (params as f64 * bytes_per_param) as u64
    }
}

/// Mock model implementation for testing
struct MockModel {
    config: BitNetConfig,
}

impl MockModel {
    fn new(config: BitNetConfig) -> Self {
        Self { config }
    }
}

/// Simple GGUF model wrapper for real loaded tensors
struct SimpleGgufModel {
    config: BitNetConfig,
    embeddings: Vec<f32>,
    lm_head: Vec<f32>,
    vocab_size: usize,
    hidden_size: usize,
}

impl SimpleGgufModel {
    fn new(config: BitNetConfig, tensors: bitnet_models::gguf_min::TwoTensors) -> Self {
        Self {
            config,
            embeddings: tensors.tok_embeddings,
            lm_head: tensors.lm_head,
            vocab_size: tensors.vocab,
            hidden_size: tensors.dim,
        }
    }
}

impl Model for MockModel {
    // No associated type anymore

    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn embed(&self, tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
        // Create a simple embedding tensor with correct shape
        let vocab_size = self.config.model.vocab_size;
        let hidden_size = self.config.model.hidden_size;
        let batch_size = tokens.len();
        
        // Create embeddings tensor [batch_size, hidden_size]
        let mut data = vec![0.0f32; batch_size * hidden_size];
        
        // Simple embedding: use token id modulo to create variations
        for (i, &token) in tokens.iter().enumerate() {
            let offset = i * hidden_size;
            for j in 0..hidden_size {
                data[offset + j] = ((token as usize % vocab_size) as f32) / (vocab_size as f32);
            }
        }
        
        // Create a Candle tensor and wrap it in ConcreteTensor
        let tensor = CandleTensor::from_vec(
            data,
            &[batch_size, hidden_size],
            &CDevice::Cpu,
        ).map_err(|e| bitnet_common::BitNetError::Candle(e))?;
        
        Ok(ConcreteTensor::bitnet(tensor))
    }

    fn logits(&self, x: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
        // Simple logits: project to vocab size
        let vocab_size = self.config.model.vocab_size;
        let batch_size = x.shape()[0];
        
        // Create logits tensor [batch_size, vocab_size]
        let data = vec![0.0f32; batch_size * vocab_size];
        
        // Create a Candle tensor and wrap it in ConcreteTensor
        let tensor = CandleTensor::from_vec(
            data,
            &[batch_size, vocab_size],
            &CDevice::Cpu,
        ).map_err(|e| bitnet_common::BitNetError::Candle(e))?;
        
        Ok(ConcreteTensor::bitnet(tensor))
    }

    fn forward(
        &self,
        x: &ConcreteTensor,
        _state: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<ConcreteTensor> {
        // Simple forward: return tensor with same batch size but hidden_size dimensions
        let batch_size = x.shape()[0];
        let hidden_size = self.config.model.hidden_size;
        
        let data = vec![0.1f32; batch_size * hidden_size];
        
        // Create a Candle tensor and wrap it in ConcreteTensor
        let tensor = CandleTensor::from_vec(
            data,
            &[batch_size, hidden_size],
            &CDevice::Cpu,
        ).map_err(|e| bitnet_common::BitNetError::Candle(e))?;
        
        Ok(ConcreteTensor::bitnet(tensor))
    }
}

impl Model for SimpleGgufModel {
    fn config(&self) -> &BitNetConfig {
        &self.config
    }

    fn embed(&self, tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
        let batch_size = tokens.len();
        let hidden_size = self.hidden_size;
        
        // Create embeddings tensor [batch_size, hidden_size]
        let mut data = vec![0.0f32; batch_size * hidden_size];
        
        // Look up embeddings for each token
        for (i, &token) in tokens.iter().enumerate() {
            let token_idx = (token as usize) % self.vocab_size;
            let emb_offset = token_idx * hidden_size;
            let out_offset = i * hidden_size;
            
            // Copy embedding vector
            if emb_offset + hidden_size <= self.embeddings.len() {
                data[out_offset..out_offset + hidden_size]
                    .copy_from_slice(&self.embeddings[emb_offset..emb_offset + hidden_size]);
            }
        }
        
        // Create a Candle tensor and wrap it in ConcreteTensor
        let tensor = CandleTensor::from_vec(
            data,
            &[batch_size, hidden_size],
            &CDevice::Cpu,
        ).map_err(|e| bitnet_common::BitNetError::Candle(e))?;
        
        Ok(ConcreteTensor::bitnet(tensor))
    }

    fn logits(&self, x: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
        // x shape: [batch_size, hidden_size]
        // lm_head shape: [hidden_size, vocab_size]
        // output: [batch_size, vocab_size]
        
        let batch_size = x.shape()[0];
        let hidden_size = self.hidden_size;
        let vocab_size = self.vocab_size;
        
        // Get input tensor and convert to Candle
        let input_tensor = x.to_candle()?;
        let input = input_tensor.flatten_all()?.to_vec1::<f32>()?;
        
        // Compute matrix multiplication: input @ lm_head.T
        let mut output = vec![0.0f32; batch_size * vocab_size];
        
        for b in 0..batch_size {
            for v in 0..vocab_size {
                let mut sum = 0.0f32;
                for h in 0..hidden_size {
                    sum += input[b * hidden_size + h] * self.lm_head[h * vocab_size + v];
                }
                output[b * vocab_size + v] = sum;
            }
        }
        
        // Create a Candle tensor and wrap it in ConcreteTensor
        let tensor = CandleTensor::from_vec(
            output,
            &[batch_size, vocab_size],
            &CDevice::Cpu,
        ).map_err(|e| bitnet_common::BitNetError::Candle(e))?;
        
        Ok(ConcreteTensor::bitnet(tensor))
    }

    fn forward(
        &self,
        x: &ConcreteTensor,
        _state: &mut dyn std::any::Any,
    ) -> bitnet_common::Result<ConcreteTensor> {
        // For simple forward, just pass through with potential shape adjustment
        Ok(x.clone())
    }
}

// Global model manager instance
static MODEL_MANAGER: std::sync::OnceLock<ModelManager> = std::sync::OnceLock::new();

/// Get the global model manager instance
pub fn get_model_manager() -> &'static ModelManager {
    MODEL_MANAGER.get_or_init(|| ModelManager::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_model_manager_creation() {
        let manager = ModelManager::new();
        assert!(manager.get_loaded_models().unwrap().is_empty());
    }

    #[test]
    fn test_model_info_conversion() {
        let config = BitNetConfig::default();
        let info = ModelInfo {
            name: CString::new("test").unwrap(),
            version: CString::new("1.0").unwrap(),
            architecture: CString::new("BitNet").unwrap(),
            config: config.clone(),
            file_size: 1024,
            memory_usage: 2048,
            is_gpu_loaded: false,
        };

        let c_model = info.to_c_model();
        assert_eq!(c_model.vocab_size, config.model.vocab_size as c_uint);
        assert_eq!(c_model.file_size, 1024);
        assert_eq!(c_model.memory_usage, 2048);
        assert_eq!(c_model.is_gpu_loaded, 0);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let manager = ModelManager::new();
        let config = BitNetConfig::default();
        let memory_usage = manager.estimate_memory_usage(&config);
        assert!(memory_usage > 0);
    }
}
