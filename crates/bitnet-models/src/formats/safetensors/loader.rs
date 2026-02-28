//! SafeTensors format loader

use crate::loader::{FormatLoader, LoadConfig, MmapFile};
use crate::{BitNetModel, Model};
use bitnet_common::{BitNetConfig, BitNetError, Device, ModelError, ModelMetadata, Result};
use candle_core::Tensor;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info};

/// SafeTensors format loader
pub struct SafeTensorsLoader;

impl SafeTensorsLoader {
    pub(crate) fn parse_header_metadata_from_bytes(data: &[u8]) -> HashMap<String, String> {
        if data.len() < 8 {
            return HashMap::new();
        }

        let mut header_len_bytes = [0_u8; 8];
        header_len_bytes.copy_from_slice(&data[0..8]);
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;
        if header_len == 0 || data.len() < 8 + header_len {
            return HashMap::new();
        }

        let Ok(header) = serde_json::from_slice::<serde_json::Value>(&data[8..8 + header_len])
        else {
            return HashMap::new();
        };

        header
            .get("__metadata__")
            .and_then(serde_json::Value::as_object)
            .map(|meta| {
                meta.iter()
                    .map(|(k, v)| {
                        let normalized = v
                            .as_str()
                            .map_or_else(|| v.to_string(), std::string::ToString::to_string);
                        (k.clone(), normalized)
                    })
                    .collect::<HashMap<String, String>>()
            })
            .unwrap_or_default()
    }

    fn parse_header_metadata(path: &Path) -> Result<HashMap<String, String>> {
        let mmap = MmapFile::open(path)?;
        Ok(Self::parse_header_metadata_from_bytes(mmap.as_slice()))
    }

    /// Convert our Device to candle Device
    fn device_to_candle(device: &Device) -> Result<candle_core::Device> {
        match device {
            Device::Cpu => Ok(candle_core::Device::Cpu),
            Device::Cuda(id) => {
                #[cfg(any(feature = "gpu", feature = "cuda"))]
                {
                    use candle_core::backend::BackendDevice;
                    let cuda = candle_core::CudaDevice::new(*id)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    Ok(candle_core::Device::Cuda(cuda))
                }
                #[cfg(not(any(feature = "gpu", feature = "cuda")))]
                {
                    let _ = id; // Suppress unused variable warning
                    Err(BitNetError::Validation(
                        "CUDA support not enabled; rebuild with --features gpu".to_string(),
                    ))
                }
            }
            // Compile this arm only on macOS with the 'gpu' feature.
            #[cfg(all(target_os = "macos", any(feature = "gpu", feature = "metal")))]
            Device::Metal => {
                use candle_core::backend::BackendDevice; // provides `new`
                let metal = candle_core::MetalDevice::new(0)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?;
                Ok(candle_core::Device::Metal(metal))
            }
            // Everywhere else, emit a clear error without referencing Metal symbols.
            #[cfg(not(all(target_os = "macos", any(feature = "gpu", feature = "metal"))))]
            Device::Metal => Err(BitNetError::Validation(
                "Metal support not enabled; rebuild with --features metal (or gpu) on macOS"
                    .to_string(),
            )),
            Device::Hip(_) | Device::Npu => Err(BitNetError::Validation(
                "HIP/NPU devices are not yet supported for model loading".to_string(),
            )),
        }
    }
}

impl FormatLoader for SafeTensorsLoader {
    fn name(&self) -> &'static str {
        "SafeTensors"
    }

    fn can_load(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase() == "safetensors")
            .unwrap_or(false)
    }

    fn detect_format(&self, path: &Path) -> Result<bool> {
        if !path.exists() {
            return Ok(false);
        }

        // Check file extension first
        if self.can_load(path) {
            return Ok(true);
        }

        // Check SafeTensors header format
        let mmap = MmapFile::open(path)?;
        if mmap.len() < 8 {
            return Ok(false);
        }

        // SafeTensors files start with an 8-byte header length
        let header_len_bytes = &mmap.as_slice()[0..8];
        let header_len = u64::from_le_bytes([
            header_len_bytes[0],
            header_len_bytes[1],
            header_len_bytes[2],
            header_len_bytes[3],
            header_len_bytes[4],
            header_len_bytes[5],
            header_len_bytes[6],
            header_len_bytes[7],
        ]);
        {
            // Header length should be reasonable (less than 1MB)
            if header_len > 0 && header_len < 1024 * 1024 && (header_len as usize + 8) <= mmap.len()
            {
                // Try to parse the header as JSON
                let header_data = &mmap.as_slice()[8..8 + header_len as usize];
                if serde_json::from_slice::<serde_json::Value>(header_data).is_ok() {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn extract_metadata(&self, path: &Path) -> Result<ModelMetadata> {
        debug!("Extracting SafeTensors metadata from: {}", path.display());

        let mmap = MmapFile::open(path)?;
        let safetensors = SafeTensors::deserialize(mmap.as_slice()).map_err(|e| {
            BitNetError::Model(ModelError::InvalidFormat {
                format: format!("Failed to parse SafeTensors file: {}", e),
            })
        })?;

        // SafeTensors crate does not expose __metadata__, so parse the header directly.
        let header_metadata = Self::parse_header_metadata(path)?;

        let name = header_metadata
            .get("name")
            .or_else(|| header_metadata.get("model_name"))
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string()
            });

        let version = header_metadata
            .get("version")
            .or_else(|| header_metadata.get("model_version"))
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let architecture = header_metadata
            .get("architecture")
            .or_else(|| header_metadata.get("model_type"))
            .map(|s| s.to_string())
            .unwrap_or_else(|| "bitnet".to_string());

        // Try to infer vocab_size and context_length from tensor shapes
        let (vocab_size, context_length) = self.infer_model_dimensions(&safetensors);

        let metadata = ModelMetadata {
            name,
            version,
            architecture,
            vocab_size,
            context_length,
            quantization: None, // SafeTensors typically stores unquantized weights
            fingerprint: None,
            corrections_applied: None,
        };

        debug!("Extracted SafeTensors metadata: {:?}", metadata);
        Ok(metadata)
    }

    fn load(&self, path: &Path, device: &Device, config: &LoadConfig) -> Result<Box<dyn Model>> {
        info!("Loading SafeTensors model from: {}", path.display());

        let mmap = if config.use_mmap { Some(MmapFile::open(path)?) } else { None };
        let owned_data = if config.use_mmap {
            None
        } else {
            Some(std::fs::read(path).map_err(BitNetError::Io)?)
        };

        let data = if let Some(ref mmap) = mmap {
            mmap.as_slice()
        } else {
            owned_data.as_deref().unwrap_or_default()
        };

        let safetensors = SafeTensors::deserialize(data).map_err(|e| {
            BitNetError::Model(ModelError::InvalidFormat {
                format: format!("Failed to parse SafeTensors file: {}", e),
            })
        })?;

        // Report progress
        if let Some(callback) = &config.progress_callback {
            callback(0.3, "Parsing SafeTensors header...");
        }

        // Extract model configuration
        let header_metadata = Self::parse_header_metadata_from_bytes(data);
        let model_config = self.extract_config(&safetensors, &header_metadata)?;

        if let Some(callback) = &config.progress_callback {
            callback(0.5, "Loading tensors...");
        }

        // Load tensors
        let tensors = self.load_tensors(&safetensors, device, config)?;

        if let Some(callback) = &config.progress_callback {
            callback(0.9, "Initializing model...");
        }

        // Create model instance
        // SafeTensors loader doesn't support QK256 raw tensors yet (passes empty HashMap)
        let raw_tensors = std::collections::HashMap::new();
        let model = BitNetModel::from_gguf(model_config, tensors, raw_tensors, *device)?;

        Ok(Box::new(model))
    }
}

impl SafeTensorsLoader {
    fn extract_config(
        &self,
        safetensors: &SafeTensors,
        metadata: &HashMap<String, String>,
    ) -> Result<BitNetConfig> {
        let mut config = BitNetConfig::default();

        if let Some(vocab_size_str) = metadata.get("vocab_size")
            && let Ok(vocab_size) = vocab_size_str.parse::<usize>()
        {
            config.model.vocab_size = vocab_size;
        }

        if let Some(hidden_size_str) = metadata.get("hidden_size")
            && let Ok(hidden_size) = hidden_size_str.parse::<usize>()
        {
            config.model.hidden_size = hidden_size;
        }

        if let Some(num_layers_str) = metadata.get("num_layers")
            && let Ok(num_layers) = num_layers_str.parse::<usize>()
        {
            config.model.num_layers = num_layers;
        }

        if let Some(num_heads_str) = metadata.get("num_attention_heads")
            && let Ok(num_heads) = num_heads_str.parse::<usize>()
        {
            config.model.num_heads = num_heads;
        }

        if let Some(intermediate_size_str) = metadata.get("intermediate_size")
            && let Ok(intermediate_size) = intermediate_size_str.parse::<usize>()
        {
            config.model.intermediate_size = intermediate_size;
        }

        if let Some(max_position_embeddings_str) = metadata.get("max_position_embeddings")
            && let Ok(max_position_embeddings) = max_position_embeddings_str.parse::<usize>()
        {
            config.model.max_position_embeddings = max_position_embeddings;
        }

        // If metadata is not available, try to infer from tensor shapes
        if config.model.vocab_size == BitNetConfig::default().model.vocab_size {
            let (vocab_size, context_length) = self.infer_model_dimensions(safetensors);
            config.model.vocab_size = vocab_size;
            if config.model.max_position_embeddings
                == BitNetConfig::default().model.max_position_embeddings
            {
                config.model.max_position_embeddings = context_length;
            }
        }

        Ok(config)
    }

    fn load_tensors(
        &self,
        safetensors: &SafeTensors,
        device: &Device,
        config: &LoadConfig,
    ) -> Result<HashMap<String, Tensor>> {
        let tensor_names = safetensors.names();
        let tensor_count = tensor_names.len();
        let mut tensors = HashMap::new();

        info!("Loading {} tensors from SafeTensors", tensor_count);

        for (i, tensor_name) in tensor_names.iter().enumerate() {
            if let Some(callback) = &config.progress_callback {
                let progress = 0.5 + (i as f32 / tensor_count as f32) * 0.4;
                callback(progress, &format!("Loading tensor {}/{}", i + 1, tensor_count));
            }

            let tensor_view = safetensors.tensor(tensor_name).map_err(|e| {
                BitNetError::Model(ModelError::LoadingFailed {
                    reason: format!("Failed to get tensor '{}': {}", tensor_name, e),
                })
            })?;

            debug!(
                "Loading tensor '{}' with shape {:?} and dtype {:?}",
                tensor_name,
                tensor_view.shape(),
                tensor_view.dtype()
            );

            // Convert SafeTensors tensor to Candle tensor
            let candle_tensor = self.convert_tensor(&tensor_view, device)?;
            tensors.insert(tensor_name.to_string(), candle_tensor);
        }

        info!("Successfully loaded {} tensors from SafeTensors", tensors.len());
        Ok(tensors)
    }

    fn convert_tensor(
        &self,
        tensor_view: &safetensors::tensor::TensorView,
        device: &Device,
    ) -> Result<Tensor> {
        let shape = tensor_view.shape();
        let data = tensor_view.data();
        let candle_device = Self::device_to_candle(device)?;

        let candle_tensor = match tensor_view.dtype() {
            SafeDtype::F32 => {
                let float_data = bytemuck::try_cast_slice::<u8, f32>(data).map_err(|_| {
                    BitNetError::Model(ModelError::InvalidFormat {
                        format: format!(
                            "Tensor data length {} is not valid for F32 tensor {:?}",
                            data.len(),
                            shape
                        ),
                    })
                })?;
                Tensor::from_slice(float_data, shape, &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::F16 => {
                let half_data = bytemuck::try_cast_slice::<u8, u16>(data).map_err(|_| {
                    BitNetError::Model(ModelError::InvalidFormat {
                        format: format!(
                            "Tensor data length {} is not valid for F16 tensor {:?}",
                            data.len(),
                            shape
                        ),
                    })
                })?;
                let float_data: Vec<f32> =
                    half_data.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
                Tensor::from_slice(&float_data, shape, &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::BF16 => {
                let bf16_data = bytemuck::try_cast_slice::<u8, u16>(data).map_err(|_| {
                    BitNetError::Model(ModelError::InvalidFormat {
                        format: format!(
                            "Tensor data length {} is not valid for BF16 tensor {:?}",
                            data.len(),
                            shape
                        ),
                    })
                })?;
                let float_data: Vec<f32> =
                    bf16_data.iter().map(|&h| half::bf16::from_bits(h).to_f32()).collect();
                Tensor::from_slice(&float_data, shape, &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::I32 => {
                let int_data = bytemuck::try_cast_slice::<u8, i32>(data).map_err(|_| {
                    BitNetError::Model(ModelError::InvalidFormat {
                        format: format!(
                            "Tensor data length {} is not valid for I32 tensor {:?}",
                            data.len(),
                            shape
                        ),
                    })
                })?;
                // Convert i32 to u32 for Candle compatibility
                let uint_data: Vec<u32> = int_data.iter().map(|&x| x as u32).collect();
                Tensor::from_slice(&uint_data, shape, &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::I64 => {
                let int_data = bytemuck::try_cast_slice::<u8, i64>(data).map_err(|_| {
                    BitNetError::Model(ModelError::InvalidFormat {
                        format: format!(
                            "Tensor data length {} is not valid for I64 tensor {:?}",
                            data.len(),
                            shape
                        ),
                    })
                })?;
                Tensor::from_slice(int_data, shape, &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::U8 => Tensor::from_slice(data, shape, &candle_device)
                .map_err(|e| BitNetError::Validation(e.to_string()))?,
            _ => {
                return Err(BitNetError::Model(ModelError::InvalidFormat {
                    format: format!("Unsupported SafeTensors dtype: {:?}", tensor_view.dtype()),
                }));
            }
        };

        Ok(candle_tensor)
    }

    fn infer_model_dimensions(&self, safetensors: &SafeTensors) -> (usize, usize) {
        let mut vocab_size = 32000; // Default
        let mut context_length = 2048; // Default

        // Look for common tensor names to infer dimensions
        for tensor_name in safetensors.names() {
            if let Ok(tensor_view) = safetensors.tensor(tensor_name) {
                let shape = tensor_view.shape();

                // Look for embedding or output layer to infer vocab size
                if (tensor_name.contains("embed")
                    || tensor_name.contains("lm_head")
                    || tensor_name.contains("output"))
                    && shape.len() >= 2
                {
                    vocab_size = shape[0].max(shape[1]);
                }

                // Look for positional embeddings to infer context length
                if (tensor_name.contains("position") || tensor_name.contains("pos_emb"))
                    && shape.len() >= 2
                {
                    context_length = shape[0].max(shape[1]);
                }
            }
        }

        (vocab_size, context_length)
    }
}
