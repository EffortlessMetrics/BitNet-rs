//! HuggingFace format loader

use crate::loader::{FormatLoader, LoadConfig, MmapFile};
use crate::{BitNetModel, Model};
use bitnet_common::{BitNetConfig, BitNetError, Device, ModelError, ModelMetadata, Result};
use candle_core::Tensor;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
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

        let config_path = path.join("config.json");
        let cfg: Value = serde_json::from_slice(&fs::read(&config_path).map_err(BitNetError::Io)?)
            .map_err(|e| BitNetError::Model(ModelError::InvalidFormat { format: e.to_string() }))?;

        let name = cfg
            .get("_name_or_path")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown"))
            .to_string();

        let version = cfg
            .get("transformers_version")
            .or_else(|| cfg.get("model_revision"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let architecture = cfg
            .get("model_type")
            .or_else(|| cfg.get("architectures").and_then(|v| v.get(0)))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let vocab_size = cfg.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(32000) as usize;

        let context_length = cfg
            .get("max_position_embeddings")
            .or_else(|| cfg.get("n_positions"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize;

        let metadata = ModelMetadata {
            name,
            version,
            architecture,
            vocab_size,
            context_length,
            quantization: None,
            fingerprint: None,
            corrections_applied: None,
        };

        debug!("Extracted HuggingFace metadata: {:?}", metadata);
        Ok(metadata)
    }

    fn load(&self, path: &Path, device: &Device, config: &LoadConfig) -> Result<Box<dyn Model>> {
        info!("Loading HuggingFace model from: {}", path.display());

        let model_config = self.parse_config(path)?;
        let shard_files = self.find_shards(path)?;

        let mut tensors = HashMap::new();
        for shard in shard_files {
            let data = if config.use_mmap {
                let mmap = MmapFile::open(&shard)?;
                mmap.as_slice().to_vec()
            } else {
                fs::read(&shard).map_err(BitNetError::Io)?
            };
            let st = SafeTensors::deserialize(&data).map_err(|e| {
                BitNetError::Model(ModelError::InvalidFormat {
                    format: format!("Failed to parse {}: {}", shard.display(), e),
                })
            })?;

            for name in st.names() {
                let view = st.tensor(name).map_err(|e| {
                    BitNetError::Model(ModelError::LoadingFailed {
                        reason: format!("Failed to get tensor '{}': {}", name, e),
                    })
                })?;
                let tensor = self.convert_tensor(&view, device)?;
                tensors.insert(name.to_string(), tensor);
            }
        }

        // HuggingFace loader doesn't support QK256 raw tensors yet (passes empty HashMap)
        let raw_tensors = std::collections::HashMap::new();
        let model = BitNetModel::from_gguf(model_config, tensors, raw_tensors, *device)?;
        Ok(Box::new(model))
    }
}

impl HuggingFaceLoader {
    fn parse_config(&self, path: &Path) -> Result<BitNetConfig> {
        let config_path = path.join("config.json");
        let cfg: Value = serde_json::from_slice(&fs::read(&config_path).map_err(BitNetError::Io)?)
            .map_err(|e| BitNetError::Model(ModelError::InvalidFormat { format: e.to_string() }))?;

        let mut config = BitNetConfig::default();
        config.model.format = bitnet_common::ModelFormat::HuggingFace;
        if let Some(v) = cfg.get("vocab_size").and_then(|v| v.as_u64()) {
            config.model.vocab_size = v as usize;
        }
        if let Some(v) =
            cfg.get("hidden_size").or_else(|| cfg.get("n_embd")).and_then(|v| v.as_u64())
        {
            config.model.hidden_size = v as usize;
        }
        if let Some(v) =
            cfg.get("num_hidden_layers").or_else(|| cfg.get("n_layer")).and_then(|v| v.as_u64())
        {
            config.model.num_layers = v as usize;
        }
        if let Some(v) =
            cfg.get("num_attention_heads").or_else(|| cfg.get("n_head")).and_then(|v| v.as_u64())
        {
            config.model.num_heads = v as usize;
        }
        if let Some(v) =
            cfg.get("intermediate_size").or_else(|| cfg.get("n_inner")).and_then(|v| v.as_u64())
        {
            config.model.intermediate_size = v as usize;
        }
        if let Some(v) = cfg
            .get("max_position_embeddings")
            .or_else(|| cfg.get("n_positions"))
            .and_then(|v| v.as_u64())
        {
            config.model.max_position_embeddings = v as usize;
        }

        Ok(config)
    }

    fn find_shards(&self, path: &Path) -> Result<Vec<PathBuf>> {
        let index_path = path.join("model.safetensors.index.json");
        if index_path.exists() {
            let idx: Value = serde_json::from_slice(
                &fs::read(&index_path).map_err(BitNetError::Io)?,
            )
            .map_err(|e| {
                BitNetError::Model(ModelError::InvalidFormat {
                    format: format!("Failed to parse index: {}", e),
                })
            })?;
            let mut files = HashSet::new();
            if let Some(map) = idx.get("weight_map").and_then(|v| v.as_object()) {
                for v in map.values() {
                    if let Some(f) = v.as_str() {
                        files.insert(path.join(f));
                    }
                }
            }
            Ok(files.into_iter().collect())
        } else {
            let mut shards = Vec::new();
            for entry in fs::read_dir(path).map_err(BitNetError::Io)? {
                let entry = entry.map_err(BitNetError::Io)?;
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    shards.push(p);
                }
            }
            Ok(shards)
        }
    }

    fn convert_tensor(
        &self,
        tensor_view: &safetensors::tensor::TensorView,
        device: &Device,
    ) -> Result<Tensor> {
        let shape = tensor_view.shape();
        let data = tensor_view.data();
        let candle_device = Self::device_to_candle(device)?;

        let tensor = match tensor_view.dtype() {
            SafeDtype::F32 => {
                let d = bytemuck::cast_slice::<u8, f32>(data);
                Tensor::from_slice(d, shape, &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::F16 => {
                let half = bytemuck::cast_slice::<u8, u16>(data);
                let float: Vec<f32> =
                    half.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
                Tensor::from_slice(&float, shape, &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::BF16 => {
                let half = bytemuck::cast_slice::<u8, u16>(data);
                let float: Vec<f32> =
                    half.iter().map(|&h| half::bf16::from_bits(h).to_f32()).collect();
                Tensor::from_slice(&float, shape, &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::I32 => {
                let int = bytemuck::cast_slice::<u8, i32>(data);
                let uint: Vec<u32> = int.iter().map(|&x| x as u32).collect();
                Tensor::from_slice(&uint, shape, &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::I64 => {
                let int = bytemuck::cast_slice::<u8, i64>(data);
                Tensor::from_slice(int, shape, &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?
            }
            SafeDtype::U8 => Tensor::from_slice(data, shape, &candle_device)
                .map_err(|e| BitNetError::Validation(e.to_string()))?,
            _ => {
                return Err(BitNetError::Model(ModelError::InvalidFormat {
                    format: format!("Unsupported dtype: {:?}", tensor_view.dtype()),
                }));
            }
        };

        Ok(tensor)
    }

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
                    let _ = id;
                    Err(BitNetError::Validation(
                        "CUDA support not enabled; rebuild with --features gpu".to_string(),
                    ))
                }
            }
            #[cfg(all(target_os = "macos", any(feature = "gpu", feature = "metal")))]
            Device::Metal => {
                use candle_core::backend::BackendDevice;
                let metal = candle_core::MetalDevice::new(0)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?;
                Ok(candle_core::Device::Metal(metal))
            }
            #[cfg(not(all(target_os = "macos", any(feature = "gpu", feature = "metal"))))]
            Device::Metal => Err(BitNetError::Validation(
                "Metal support not enabled; rebuild with --features metal (or gpu) on macOS"
                    .to_string(),
            )),
            Device::OpenCL(_) => Ok(candle_core::Device::Cpu), // OpenCL uses its own buffer management
        }
    }
}
