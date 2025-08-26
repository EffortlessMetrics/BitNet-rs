//! HuggingFace format loader

use crate::loader::{FormatLoader, LoadConfig};
use crate::{BitNetModel, Model};
use bitnet_common::{BitNetConfig, BitNetError, Device, ModelError, ModelMetadata, Result};
use candle_core::Tensor;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// HuggingFace format loader
pub struct HuggingFaceLoader;

impl FormatLoader for HuggingFaceLoader {
    fn name(&self) -> &'static str {
        "HuggingFace"
    }

    fn can_load(&self, path: &Path) -> bool {
        path.is_dir() && path.join("config.json").exists()
    }

    fn detect_format(&self, path: &Path) -> Result<bool> {
        if !path.exists() {
            return Ok(false);
        }

        Ok(self.can_load(path))
    }

    fn extract_metadata(&self, path: &Path) -> Result<ModelMetadata> {
        debug!("Extracting HuggingFace metadata from: {}", path.display());

        let config = self.read_config(path)?;

        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
        let version = config
            .get("transformers_version")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let architecture = config
            .get("model_type")
            .or_else(|| config.get("architectures").and_then(|v| v.get(0)))
            .and_then(|v| v.as_str())
            .unwrap_or("bitnet")
            .to_string();
        let vocab_size =
            config.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(32000) as usize;
        let context_length = config
            .get("max_position_embeddings")
            .or_else(|| config.get("n_positions"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;

        let metadata = ModelMetadata {
            name,
            version,
            architecture,
            vocab_size,
            context_length,
            quantization: None,
        };

        debug!("Extracted HuggingFace metadata: {:?}", metadata);
        Ok(metadata)
    }

    fn load(&self, path: &Path, device: &Device, config: &LoadConfig) -> Result<Box<dyn Model>> {
        info!("Loading HuggingFace model from: {}", path.display());

        let json = self.read_config(path)?;
        let model_config = self.config_from_json(&json);

        let weight_files = self.find_weight_files(path)?;
        if weight_files.is_empty() {
            return Err(BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("No weight files found in {}", path.display()),
            }));
        }

        let tensors = self.load_tensors(&weight_files, device, config)?;

        let model = BitNetModel::from_gguf(model_config, tensors, *device)?;

        Ok(Box::new(model))
    }
}

impl HuggingFaceLoader {
    fn read_config(&self, path: &Path) -> Result<Value> {
        let config_path = path.join("config.json");
        let data = fs::read_to_string(config_path).map_err(BitNetError::Io)?;
        let value =
            serde_json::from_str(&data).map_err(|e| BitNetError::Validation(e.to_string()))?;
        Ok(value)
    }

    fn config_from_json(&self, json: &Value) -> BitNetConfig {
        let mut config = BitNetConfig::default();

        if let Some(v) = json.get("vocab_size").and_then(|v| v.as_u64()) {
            config.model.vocab_size = v as usize;
        }
        if let Some(v) =
            json.get("hidden_size").or_else(|| json.get("n_embd")).and_then(|v| v.as_u64())
        {
            config.model.hidden_size = v as usize;
        }
        if let Some(v) =
            json.get("num_hidden_layers").or_else(|| json.get("n_layer")).and_then(|v| v.as_u64())
        {
            config.model.num_layers = v as usize;
        }
        if let Some(v) =
            json.get("num_attention_heads").or_else(|| json.get("n_head")).and_then(|v| v.as_u64())
        {
            config.model.num_heads = v as usize;
        }
        if let Some(v) =
            json.get("intermediate_size").or_else(|| json.get("n_inner")).and_then(|v| v.as_u64())
        {
            config.model.intermediate_size = v as usize;
        }
        if let Some(v) = json
            .get("max_position_embeddings")
            .or_else(|| json.get("n_positions"))
            .and_then(|v| v.as_u64())
        {
            config.model.max_position_embeddings = v as usize;
        }

        config
    }

    fn find_weight_files(&self, path: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        if path.join("pytorch_model.bin").exists() {
            files.push(path.join("pytorch_model.bin"));
        }
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("pytorch_model-") && name.ends_with(".bin") {
                    files.push(entry.path());
                }
            }
        }
        files.sort();
        Ok(files)
    }

    fn load_tensors(
        &self,
        files: &[PathBuf],
        device: &Device,
        config: &LoadConfig,
    ) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        for (file_idx, file) in files.iter().enumerate() {
            if let Some(callback) = &config.progress_callback {
                let progress = 0.3 + (file_idx as f32 / files.len() as f32) * 0.6;
                callback(progress, &format!("Loading shard {}/{}", file_idx + 1, files.len()));
            }
            let data = fs::read(file).map_err(BitNetError::Io)?;
            let safetensors = SafeTensors::deserialize(&data).map_err(|e| {
                BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Failed to parse {}: {}", file.display(), e),
                })
            })?;
            for name in safetensors.names() {
                let view = safetensors.tensor(name).map_err(|e| {
                    BitNetError::Model(ModelError::LoadingFailed {
                        reason: format!("Failed to get tensor '{}': {}", name, e),
                    })
                })?;
                let tensor = self.convert_tensor(&view, device)?;
                tensors.insert(name.to_string(), tensor);
            }
        }
        Ok(tensors)
    }

    fn convert_tensor(
        &self,
        tensor_view: &safetensors::tensor::TensorView,
        device: &Device,
    ) -> Result<Tensor> {
        let shape = tensor_view.shape();
        let data = tensor_view.data();
        let device = Self::device_to_candle(device)?;

        let tensor = match tensor_view.dtype() {
            SafeDtype::F32 => {
                let float_data = bytemuck::cast_slice::<u8, f32>(data);
                Tensor::from_slice(float_data, shape, &device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::F16 => {
                let half_data = bytemuck::cast_slice::<u8, u16>(data);
                let float_data: Vec<f32> =
                    half_data.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
                Tensor::from_slice(&float_data, shape, &device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::I32 => {
                let int_data = bytemuck::cast_slice::<u8, i32>(data);
                let uint_data: Vec<u32> = int_data.iter().map(|&x| x as u32).collect();
                Tensor::from_slice(&uint_data, shape, &device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            dtype => {
                return Err(BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Unsupported tensor dtype: {:?}", dtype),
                }));
            }
        };
        Ok(tensor)
    }

    fn device_to_candle(device: &Device) -> Result<candle_core::Device> {
        match device {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda(id) => {
                #[cfg(feature = "gpu")]
                {
                    use candle_core::backend::BackendDevice;
                    let cuda = candle_core::CudaDevice::new(*id)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    Ok(candle_core::Device::Cuda(cuda))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    let _ = id;
                    Err(BitNetError::Validation(
                        "CUDA support not enabled; rebuild with --features gpu".to_string(),
                    ))
                }
            }
            #[cfg(all(target_os = "macos", feature = "gpu"))]
            Device::Metal => {
                use candle_core::backend::BackendDevice;
                let metal = candle_core::MetalDevice::new(0)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?;
                Ok(candle_core::Device::Metal(metal))
            }
            #[cfg(not(all(target_os = "macos", feature = "gpu")))]
            Device::Metal => Err(BitNetError::Validation(
                "Metal support not enabled; rebuild with --features gpu on macOS".to_string(),
            )),
        }
    }
}
