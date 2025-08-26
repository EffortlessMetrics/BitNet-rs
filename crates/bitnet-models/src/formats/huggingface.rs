//! HuggingFace format loader

use crate::loader::{FormatLoader, LoadConfig};
use crate::{BitNetModel, Model};
use bitnet_common::{
    BitNetConfig, BitNetError, Device, ModelError, ModelMetadata, QuantizationType, Result,
};
use candle_core::Tensor;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde::Deserialize;
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

        let cfg = parse_config(&path.join("config.json"))?;

        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
        let architecture = cfg
            .model_type
            .clone()
            .or(cfg.architecture.clone())
            .unwrap_or_else(|| "unknown".to_string());
        let vocab_size = cfg.vocab_size.unwrap_or(0);
        let context_length = cfg
            .max_position_embeddings
            .or(cfg.n_positions)
            .unwrap_or(0);
        let quantization = cfg
            .quantization_config
            .as_ref()
            .and_then(|q| q.quantization_type.as_ref())
            .or(cfg.quantization.as_ref())
            .and_then(|s| parse_quantization_type(s));

        let metadata = ModelMetadata {
            name,
            version: "unknown".to_string(),
            architecture,
            vocab_size,
            context_length,
            quantization,
        };

        debug!("Extracted HuggingFace metadata: {:?}", metadata);
        Ok(metadata)
    }

    fn load(&self, path: &Path, device: &Device, _config: &LoadConfig) -> Result<Box<dyn Model>> {
        info!("Loading HuggingFace model from: {}", path.display());

        let cfg_path = path.join("config.json");
        let hf_cfg = parse_config(&cfg_path)?;
        let mut model_config = BitNetConfig::default();
        model_config.model.format = bitnet_common::config::ModelFormat::HuggingFace;
        if let Some(v) = hf_cfg.vocab_size {
            model_config.model.vocab_size = v;
        }
        if let Some(h) = hf_cfg.hidden_size {
            model_config.model.hidden_size = h;
        }
        if let Some(l) = hf_cfg.num_hidden_layers {
            model_config.model.num_layers = l;
        }
        if let Some(h) = hf_cfg.num_attention_heads {
            model_config.model.num_heads = h;
        }
        if let Some(i) = hf_cfg.intermediate_size {
            model_config.model.intermediate_size = i;
        }
        if let Some(m) = hf_cfg.max_position_embeddings.or(hf_cfg.n_positions) {
            model_config.model.max_position_embeddings = m;
        }
        if let Some(theta) = hf_cfg.rope_theta {
            model_config.model.rope_theta = Some(theta);
        }

        let weight_path = find_weight_file(path)?;
        let tensors = load_safetensors(&weight_path, device)?;

        let model = BitNetModel::from_gguf(model_config, tensors, *device)?;

        Ok(Box::new(model))
    }
}

#[derive(Debug, Deserialize, Clone)]
struct HfQuantConfig {
    quantization_type: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct HfConfig {
    model_type: Option<String>,
    architecture: Option<String>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    n_positions: Option<usize>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    intermediate_size: Option<usize>,
    rope_theta: Option<f32>,
    quantization_config: Option<HfQuantConfig>,
    quantization: Option<String>,
}

fn parse_config(path: &Path) -> Result<HfConfig> {
    let text = fs::read_to_string(path)?;
    let cfg: HfConfig = serde_json::from_str(&text).map_err(|e| {
        BitNetError::Model(ModelError::InvalidFormat {
            format: format!("Invalid config.json: {}", e),
        })
    })?;
    Ok(cfg)
}

fn find_weight_file(dir: &Path) -> Result<PathBuf> {
    let st = dir.join("model.safetensors");
    if st.exists() {
        return Ok(st);
    }
    let pt = dir.join("pytorch_model.bin");
    if pt.exists() {
        return Ok(pt);
    }
    Err(BitNetError::Model(ModelError::NotFound { path: dir.display().to_string() }))
}

fn parse_quantization_type(s: &str) -> Option<QuantizationType> {
    match s.to_uppercase().as_str() {
        "I2S" | "I2_S" => Some(QuantizationType::I2S),
        "TL1" => Some(QuantizationType::TL1),
        "TL2" => Some(QuantizationType::TL2),
        _ => None,
    }
}

fn load_safetensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let data = fs::read(path)?;
    let st = SafeTensors::deserialize(&data).map_err(|e| {
        BitNetError::Model(ModelError::InvalidFormat {
            format: format!("Failed to parse SafeTensors: {}", e),
        })
    })?;

    let mut tensors = HashMap::new();
    for name in st.names() {
        let view = st.tensor(name).map_err(|e| {
            BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Failed to load tensor '{}': {}", name, e),
            })
        })?;
        let tensor = convert_tensor(&view, device)?;
        tensors.insert(name.to_string(), tensor);
    }
    Ok(tensors)
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

fn convert_tensor(view: &safetensors::tensor::TensorView, device: &Device) -> Result<Tensor> {
    let shape = view.shape();
    let data = view.data();
    let dev = device_to_candle(device)?;
    let tensor = match view.dtype() {
        SafeDtype::F32 => {
            let floats = bytemuck::cast_slice::<u8, f32>(data);
            Tensor::from_slice(floats, shape, &dev).map_err(|e| BitNetError::Validation(e.to_string()))?
        }
        SafeDtype::F16 => {
            let halfs = bytemuck::cast_slice::<u8, u16>(data);
            let floats: Vec<f32> = halfs.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
            Tensor::from_slice(&floats, shape, &dev)
                .map_err(|e| BitNetError::Validation(e.to_string()))?
        }
        SafeDtype::BF16 => {
            let halfs = bytemuck::cast_slice::<u8, u16>(data);
            let floats: Vec<f32> = halfs.iter().map(|&h| half::bf16::from_bits(h).to_f32()).collect();
            Tensor::from_slice(&floats, shape, &dev)
                .map_err(|e| BitNetError::Validation(e.to_string()))?
        }
        SafeDtype::I32 => {
            let ints = bytemuck::cast_slice::<u8, i32>(data);
            let uints: Vec<u32> = ints.iter().map(|&x| x as u32).collect();
            Tensor::from_slice(&uints, shape, &dev)
                .map_err(|e| BitNetError::Validation(e.to_string()))?
        }
        SafeDtype::I64 => {
            let ints = bytemuck::cast_slice::<u8, i64>(data);
            Tensor::from_slice(ints, shape, &dev)
                .map_err(|e| BitNetError::Validation(e.to_string()))?
        }
        SafeDtype::U8 => Tensor::from_slice(data, shape, &dev)
            .map_err(|e| BitNetError::Validation(e.to_string()))?,
        _ => {
            return Err(BitNetError::Model(ModelError::InvalidFormat {
                format: format!("Unsupported dtype: {:?}", view.dtype()),
            }))
        }
    };
    Ok(tensor)
}

