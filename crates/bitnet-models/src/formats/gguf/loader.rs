//! GGUF format loader implementation

use super::{GgufReader, GgufTensorType, GgufTensors};
use crate::loader::{FormatLoader, LoadConfig, MmapFile};
use crate::{BitNetModel, Model};
use bitnet_common::{BitNetConfig, BitNetError, Device, ModelError, ModelMetadata, Result};
use candle_core::{DType, Tensor};
use std::path::Path;
use tracing::{debug, info};

/// GGUF format loader
pub struct GgufLoader;

impl GgufLoader {
    /// Helper to fetch an unsigned integer by trying a list of keys
    fn get_u32_any(reader: &GgufReader, keys: &[&str]) -> Option<u32> {
        for k in keys {
            if let Some(v) = reader.get_u32_metadata(k) { return Some(v); }
            if let Some(v) = reader.get_i32_metadata(k) { if v >= 0 { return Some(v as u32); } }
        }
        None
    }

    /// Convert our Device to candle Device
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
                    let _ = id; // Suppress unused variable warning
                    Err(BitNetError::Validation(
                        "CUDA support not enabled; rebuild with --features gpu".to_string(),
                    ))
                }
            }
            // Compile this arm only on macOS with the 'gpu' feature.
            #[cfg(all(target_os = "macos", feature = "gpu"))]
            Device::Metal => {
                use candle_core::backend::BackendDevice; // provides `new`
                let metal = candle_core::MetalDevice::new(0)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?;
                Ok(candle_core::Device::Metal(metal))
            }
            // Everywhere else, emit a clear error without referencing Metal symbols.
            #[cfg(not(all(target_os = "macos", feature = "gpu")))]
            Device::Metal => Err(BitNetError::Validation(
                "Metal support not enabled; rebuild with --features gpu on macOS".to_string(),
            )),
        }
    }
}

impl FormatLoader for GgufLoader {
    fn name(&self) -> &'static str {
        "GGUF"
    }

    fn can_load(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase() == "gguf")
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

        // Check magic bytes
        let mmap = MmapFile::open(path)?;
        if mmap.len() < 4 {
            return Ok(false);
        }

        let magic = &mmap.as_slice()[0..4];
        Ok(magic == b"GGUF")
    }

    fn extract_metadata(&self, path: &Path) -> Result<ModelMetadata> {
        debug!("Extracting GGUF metadata from: {}", path.display());

        let mmap = MmapFile::open(path)?;
        let reader = GgufReader::new(mmap.as_slice())?;

        // Validate the file structure
        reader.validate()?;

        let metadata = ModelMetadata {
            name: reader.get_string_metadata("general.name").unwrap_or_else(|| {
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string()
            }),
            version: reader
                .get_string_metadata("general.version")
                .unwrap_or_else(|| format!("gguf-v{}", reader.version())),
            architecture: reader
                .get_string_metadata("general.architecture")
                .unwrap_or_else(|| "bitnet".to_string()),
            vocab_size: reader
                .get_u32_metadata("llama.vocab_size")
                .or_else(|| reader.get_u32_metadata("tokenizer.ggml.tokens"))
                .unwrap_or(32000) as usize,
            context_length: reader
                .get_u32_metadata("llama.context_length")
                .or_else(|| reader.get_u32_metadata("llama.rope.dimension_count"))
                .unwrap_or(2048) as usize,
            quantization: reader.get_quantization_type(),
        };

        debug!("Extracted GGUF metadata: {:?}", metadata);
        Ok(metadata)
    }

    fn load(&self, path: &Path, device: &Device, config: &LoadConfig) -> Result<Box<dyn Model>> {
        info!("Loading GGUF model from: {}", path.display());

        let mmap = if config.use_mmap { Some(MmapFile::open(path)?) } else { None };

        let data = if let Some(ref mmap) = mmap {
            mmap.as_slice()
        } else {
            // Read entire file into memory
            &std::fs::read(path).map_err(BitNetError::Io)?
        };

        let reader = GgufReader::new(data)?;

        // Report progress
        if let Some(callback) = &config.progress_callback {
            callback(0.3, "Parsing GGUF header...");
        }

        // Validate file structure
        reader.validate()?;

        // Extract model configuration
        let model_config = self.extract_config(&reader)?;

        if let Some(callback) = &config.progress_callback {
            callback(0.5, "Loading tensors...");
        }

        // Load tensors
        let tensors = self.load_tensors(&reader, device, config)?;

        if let Some(callback) = &config.progress_callback {
            callback(0.9, "Initializing model...");
        }

        // Create model instance
        let model = BitNetModel::from_gguf(model_config, tensors, *device)?;

        Ok(Box::new(model))
    }
}

impl GgufLoader {
    /// Check if a tensor name indicates it's an embedding tensor
    fn is_embedding_tensor(name: &str) -> bool {
        matches!(
            name,
            "embed_tokens.weight" |
            "tok_embeddings.weight" |
            "token_embd.weight" |
            "model.embed_tokens.weight" |
            "transformer.wte.weight"
        )
    }

    /// Heuristic: Microsoft 2B ships [hidden, vocab]; we want [vocab, hidden].
    fn embedding_is_transposed(dims: &[usize]) -> bool {
        dims.len() == 2 && dims[0] < dims[1] && dims[1] >= 32768
    }

    /// Helper to transpose F16 data to F32 transposed layout
    fn transpose_f16_to_f32(bytes: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
        use std::io::Read;
        let (rows, cols) = (dims[0], dims[1]);
        let mut out = vec![0f32; rows * cols]; // transposed [cols, rows]
        let mut rdr = std::io::Cursor::new(bytes);
        for r in 0..rows {
            for c in 0..cols {
                let mut buf = [0u8; 2];
                rdr.read_exact(&mut buf).map_err(BitNetError::Io)?;
                let v = half::f16::from_bits(u16::from_le_bytes(buf)).to_f32();
                out[c * rows + r] = v;
            }
        }
        Ok(out)
    }

    /// Helper to transpose F32 data to F32 transposed layout
    fn transpose_f32_to_f32(bytes: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
        use std::io::Read;
        let (rows, cols) = (dims[0], dims[1]);
        let mut out = vec![0f32; rows * cols]; // transposed [cols, rows]
        let mut rdr = std::io::Cursor::new(bytes);
        for r in 0..rows {
            for c in 0..cols {
                let mut buf = [0u8; 4];
                rdr.read_exact(&mut buf).map_err(BitNetError::Io)?;
                out[c * rows + r] = f32::from_le_bytes(buf);
            }
        }
        Ok(out)
    }

    fn extract_config(&self, reader: &GgufReader) -> Result<BitNetConfig> {
        let mut config = BitNetConfig::default();

        // Extract model configuration from GGUF metadata
        if let Some(vocab_size) = reader.get_u32_metadata("llama.vocab_size") {
            config.model.vocab_size = vocab_size as usize;
        }

        // Try multiple keys for hidden size
        if let Some(hidden_size) = reader
            .get_u32_metadata("llama.embedding_length")
            .or_else(|| reader.get_u32_metadata("llama.hidden_size"))
            .or_else(|| reader.get_u32_metadata("bitnet.hidden_size"))
        {
            config.model.hidden_size = hidden_size as usize;
        }

        if let Some(num_layers) = reader.get_u32_metadata("llama.block_count") {
            config.model.num_layers = num_layers as usize;
        }

        if let Some(num_heads) = reader.get_u32_metadata("llama.attention.head_count") {
            config.model.num_heads = num_heads as usize;
        }

        // GQA/MQA: parse K/V head count (vendors use various keys)
        let kv_keys = [
            "n_head_kv",             // llama.cpp style
            "n_kv_heads",
            "attn.n_kv_heads",       // some HF exports
            "attn_n_kv_heads",
            "num_key_value_heads",   // transformers config key
            "llama.attention.head_count_kv", // GGUF standard
        ];
        config.model.num_key_value_heads = Self::get_u32_any(reader, &kv_keys)
            .map(|v| v as usize)
            .unwrap_or(0);
        if config.model.num_key_value_heads == 0 {
            // default to full MHA if not present
            config.model.num_key_value_heads = config.model.num_heads;
        }

        if let Some(intermediate_size) = reader.get_u32_metadata("llama.feed_forward_length") {
            config.model.intermediate_size = intermediate_size as usize;
        }

        if let Some(context_length) = reader.get_u32_metadata("llama.context_length") {
            config.model.max_position_embeddings = context_length as usize;
        }

        // Set quantization type based on tensor types
        if let Some(qtype) = reader.get_quantization_type() {
            config.quantization.quantization_type = qtype;
        }

        // Extract additional BitNet-specific configuration
        if let Some(block_size) = reader.get_u32_metadata("bitnet.block_size") {
            config.quantization.block_size = block_size as usize;
        }

        if let Some(precision) = reader.get_f32_metadata("bitnet.precision") {
            config.quantization.precision = precision;
        }

        Ok(config)
    }

    fn load_tensors(
        &self,
        reader: &GgufReader,
        device: &Device,
        config: &LoadConfig,
    ) -> Result<GgufTensors> {
        let tensor_count = reader.tensor_count() as usize;
        let mut tensors = GgufTensors::new();

        info!("Loading {} tensors", tensor_count);

        for i in 0..tensor_count {
            if let Some(callback) = &config.progress_callback {
                let progress = 0.5 + (i as f32 / tensor_count as f32) * 0.4;
                callback(progress, &format!("Loading tensor {}/{}", i + 1, tensor_count));
            }

            let tensor_info = reader.get_tensor_info(i)?;
            let tensor_data = reader.get_tensor_data(i)?;

            debug!(
                "Loading tensor '{}' with shape {:?} and type {:?}",
                tensor_info.name, tensor_info.shape, tensor_info.tensor_type
            );

            // Convert to Candle tensor
            let candle_tensor = self.create_candle_tensor(tensor_info, tensor_data, device)?;
            tensors.insert(tensor_info.name.clone(), candle_tensor);
        }

        info!("Successfully loaded {} tensors", tensors.len());
        Ok(tensors)
    }

    fn create_candle_tensor(
        &self,
        info: &crate::formats::gguf::TensorInfo,
        data: &[u8],
        device: &Device,
    ) -> Result<Tensor> {
        let dtype = match info.tensor_type {
            GgufTensorType::F32 => DType::F32,
            GgufTensorType::F16 => DType::F16,
            GgufTensorType::Q4_0
            | GgufTensorType::Q4_1
            | GgufTensorType::Q5_0
            | GgufTensorType::Q5_1
            | GgufTensorType::Q8_0
            | GgufTensorType::Q8_1
            | GgufTensorType::Q2_K
            | GgufTensorType::Q3_K
            | GgufTensorType::Q4_K
            | GgufTensorType::Q5_K
            | GgufTensorType::Q6_K
            | GgufTensorType::Q8_K
            | GgufTensorType::IQ2_S
            | GgufTensorType::I2_S => DType::U8, // Quantized types stored as bytes
        };

        let candle_device = Self::device_to_candle(device)?;

        // For quantized tensors, we need special handling
        if info.tensor_type.is_quantized() {
            // Handle IQ2_S quantization with FFI dequantization
            #[cfg(feature = "iq2s-ffi")]
            if matches!(info.tensor_type, GgufTensorType::IQ2_S) {
                use crate::quant::iq2s;
                let f32_data = iq2s::dequantize_to_f32(data, &info.shape)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?;
                let tensor = Tensor::from_slice(&f32_data, info.shape.as_slice(), &candle_device)
                    .map_err(|e| BitNetError::Validation(e.to_string()))?;
                return Ok(tensor);
            }

            // For IQ2_S without FFI support, fail with clear message
            #[cfg(not(feature = "iq2s-ffi"))]
            if matches!(info.tensor_type, GgufTensorType::IQ2_S) {
                return Err(BitNetError::Model(ModelError::InvalidFormat {
                    format: format!(
                        "IQ2_S tensor '{}' found but support not compiled in. \
                        Rebuild with `--features iq2s-ffi` to enable IQ2_S support.",
                        info.name
                    ),
                }));
            }

            // Handle I2_S quantization with native Rust dequantization
            if matches!(info.tensor_type, GgufTensorType::I2_S) {
                use crate::quant::i2s;

                // Check for embedding transposition
                if Self::is_embedding_tensor(&info.name) && Self::embedding_is_transposed(&info.shape) {
                    info!("Embedding appears transposed ({:?}) -> decoding transposed", info.shape);
                    let f32_data = i2s::dequantize_to_f32_transposed(data, &info.shape)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    // Now dims become [vocab, hidden]
                    let (rows, cols) = (info.shape[1], info.shape[0]);
                    let tensor = Tensor::from_slice(&f32_data, &[rows, cols], &candle_device)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    return Ok(tensor);
                } else {
                    // Normal I2_S dequantization
                    let f32_data = i2s::dequantize_to_f32(data, &info.shape)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    let tensor = Tensor::from_slice(&f32_data, info.shape.as_slice(), &candle_device)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    return Ok(tensor);
                }
            }

            // For other quantized types, keep as raw bytes for now
            // (would need specific dequantizers for Q4_0, Q8_0, etc.)
            let tensor = Tensor::from_raw_buffer(data, dtype, &info.shape, &candle_device)
                .map_err(|e| BitNetError::Validation(e.to_string()))?;
            Ok(tensor)
        } else {
            // For regular tensors, interpret the bytes according to the data type
            match dtype {
                DType::F32 => {
                    // Check for embedding transposition
                    if Self::is_embedding_tensor(&info.name) && Self::embedding_is_transposed(&info.shape) {
                        info!("Embedding appears transposed ({:?}) -> decoding transposed", info.shape);
                        let f32_data = Self::transpose_f32_to_f32(data, &info.shape)?;
                        // Now dims become [vocab, hidden]
                        let (rows, cols) = (info.shape[1], info.shape[0]);
                        Tensor::from_slice(&f32_data, &[rows, cols], &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))
                    } else {
                        let float_data = bytemuck::cast_slice::<u8, f32>(data);
                        Tensor::from_slice(float_data, info.shape.as_slice(), &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))
                    }
                }
                DType::F16 => {
                    // Check for embedding transposition
                    if Self::is_embedding_tensor(&info.name) && Self::embedding_is_transposed(&info.shape) {
                        info!("Embedding appears transposed ({:?}) -> decoding transposed", info.shape);
                        let f32_data = Self::transpose_f16_to_f32(data, &info.shape)?;
                        // Now dims become [vocab, hidden]
                        let (rows, cols) = (info.shape[1], info.shape[0]);
                        Tensor::from_slice(&f32_data, &[rows, cols], &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))
                    } else {
                        // For now, convert F16 data to F32 for compatibility
                        let half_data = bytemuck::cast_slice::<u8, u16>(data);
                        let float_data: Vec<f32> =
                            half_data.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
                        Tensor::from_slice(&float_data, info.shape.as_slice(), &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))
                    }
                }
                _ => Err(BitNetError::Model(ModelError::InvalidFormat {
                    format: format!("Unsupported data type: {:?}", dtype),
                })),
            }
        }
    }

    /// Validate tensor data integrity
    #[cfg(any(test, feature = "validation"))]
    #[allow(dead_code)]
    fn validate_tensor_data(
        &self,
        info: &crate::formats::gguf::TensorInfo,
        data: &[u8],
    ) -> Result<()> {
        // Check data size matches expected size
        let expected_size = info.size as usize;
        if data.len() != expected_size {
            return Err(BitNetError::Validation(format!(
                "Tensor '{}' data size mismatch: expected {}, got {}",
                info.name,
                expected_size,
                data.len()
            )));
        }

        // For quantized tensors, validate block alignment
        if info.tensor_type.is_quantized() {
            let block_size = info.tensor_type.block_size();
            let total_elements: usize = info.shape.iter().product();

            if total_elements % block_size != 0 {
                return Err(BitNetError::Validation(format!(
                    "Tensor '{}' elements ({}) not aligned to block size ({})",
                    info.name, total_elements, block_size
                )));
            }
        }

        Ok(())
    }
}
