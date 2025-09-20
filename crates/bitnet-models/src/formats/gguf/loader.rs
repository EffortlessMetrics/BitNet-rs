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
    #[inline]
    fn is_projection_weight(name: &str) -> bool {
        // Linear projections (attn + ffn) that should be [out,in] in memory
        name.ends_with(".q_proj.weight")
            || name.ends_with(".k_proj.weight")
            || name.ends_with(".v_proj.weight")
            || name.ends_with(".o_proj.weight")
            || name.ends_with(".gate_proj.weight")
            || name.ends_with(".up_proj.weight")
            || name.ends_with(".down_proj.weight")
    }

    #[inline]
    fn maybe_transpose_to_out_in(shape: &[usize], name: &str) -> bool {
        // All projection weights are stored/consumed as [out,in] in our kernels.
        // GGUF frequently provides them as [in,out]. Normalize here once.
        // Use name-only gating since model dims vary across architectures.
        Self::is_projection_weight(name) && shape.len() == 2
    }

    /// Helper to fetch an unsigned integer by trying a list of keys
    fn get_u32_any(reader: &GgufReader, keys: &[&str]) -> Option<u32> {
        for k in keys {
            if let Some(v) = reader.get_u32_metadata(k) {
                return Some(v);
            }
            if let Some(v) = reader.get_i32_metadata(k)
                && v >= 0
            {
                return Some(v as u32);
            }
        }
        None
    }

    /// Infer hidden_size from embedding tensor shapes when metadata is missing.
    fn infer_hidden_size_from_tensors(reader: &GgufReader) -> Option<usize> {
        let emb_names = [
            // common names across llama.cpp/HF exports
            "token_embd.weight",
            "tok_embeddings.weight",
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "transformer.wte.weight",
        ];
        for n in &emb_names {
            if let Some(info) = reader.get_tensor_info_by_name(n)
                && info.shape.len() == 2
            {
                let a = info.shape[0];
                let b = info.shape[1];
                // Heuristic: vocab is big (>= 32768). Hidden is the other dim.
                let hidden = if a >= 32768 && b < a {
                    b
                } else if b >= 32768 && a < b {
                    a
                } else {
                    a.min(b)
                }; // fallback: pick the smaller
                tracing::info!("inferred hidden_size={} from {}", hidden, n);
                return Some(hidden);
            }
        }
        None
    }

    /// Infer intermediate_size from feed-forward tensor shapes when metadata is missing.
    fn infer_intermediate_size_from_tensors(
        reader: &GgufReader,
        hidden_size: usize,
    ) -> Option<usize> {
        let ffn_names = [
            // Common feed-forward projection tensor names
            "blk.0.ffn_gate.weight", // Microsoft BitNet style
            "layers.0.feed_forward.gate_proj.weight", // LLaMA style
            "model.layers.0.mlp.gate_proj.weight",
            "transformer.h.0.mlp.c_fc.weight",
        ];
        for n in &ffn_names {
            if let Some(info) = reader.get_tensor_info_by_name(n)
                && info.shape.len() == 2
            {
                let w_in = info.shape[0];
                let w_out = info.shape[1];
                // gate_proj should be [hidden_size, intermediate_size]
                if w_in == hidden_size {
                    tracing::info!("inferred intermediate_size={} from {}", w_out, n);
                    return Some(w_out);
                }
                // Handle transposed case [intermediate_size, hidden_size]
                if w_out == hidden_size {
                    tracing::info!("inferred intermediate_size={} from {} (transposed)", w_in, n);
                    return Some(w_in);
                }
            }
        }
        None
    }

    /// Infer number of layers from tensor names when metadata is missing or incorrect.
    fn infer_num_layers_from_tensors(reader: &GgufReader) -> Option<usize> {
        let mut max_layer = 0;
        let tensor_names = reader.tensor_names();

        for name in tensor_names {
            // Look for patterns like "blk.N." or "layers.N."
            if let Some(layer_num) = Self::extract_layer_number(name) {
                max_layer = max_layer.max(layer_num);
            }
        }

        if max_layer > 0 {
            // Layer numbers are 0-indexed, so add 1 to get total count
            Some(max_layer + 1)
        } else {
            None
        }
    }

    /// Extract layer number from tensor name patterns like "blk.N." or "layers.N."
    fn extract_layer_number(name: &str) -> Option<usize> {
        // Check for "blk.N." pattern
        if let Some(start) = name.find("blk.") {
            let after_blk = &name[start + 4..];
            if let Some(dot_pos) = after_blk.find('.') {
                let number_str = &after_blk[..dot_pos];
                if let Ok(layer_num) = number_str.parse::<usize>() {
                    return Some(layer_num);
                }
            }
        }

        // Check for "layers.N." pattern
        if let Some(start) = name.find("layers.") {
            let after_layers = &name[start + 7..];
            if let Some(dot_pos) = after_layers.find('.') {
                let number_str = &after_layers[..dot_pos];
                if let Ok(layer_num) = number_str.parse::<usize>() {
                    return Some(layer_num);
                }
            }
        }

        None
    }

    /// Infer number of KV heads from tensor shapes (for models without explicit metadata)
    fn infer_kv_heads_from_tensors(reader: &GgufReader, config: &BitNetConfig) -> Result<usize> {
        let hidden_size = config.model.hidden_size;
        let num_heads = config.model.num_heads;

        debug!("Shape inference: hidden_size={}, num_heads={}", hidden_size, num_heads);

        if num_heads == 0 || hidden_size == 0 {
            debug!("Cannot infer GQA: missing basic dimensions");
            return Ok(num_heads); // fallback to MHA
        }

        let head_dim = hidden_size / num_heads;
        debug!("Calculated head_dim: {}", head_dim);

        // Look for k_proj tensor in first layer to infer KV head count
        let k_proj_names = [
            "blk.0.attn_k.weight",              // Microsoft BitNet style
            "layers.0.attention.k_proj.weight", // LLaMA style
            "model.layers.0.self_attn.k_proj.weight",
            "transformer.h.0.attn.k_proj.weight",
        ];

        for tensor_name in &k_proj_names {
            debug!("Checking tensor: {}", tensor_name);
            if let Some(info) = reader.get_tensor_info_by_name(tensor_name) {
                debug!("Found tensor {} with shape {:?}", tensor_name, info.shape);
                if info.shape.len() == 2 {
                    let w_in = info.shape[0];
                    let w_out = info.shape[1];
                    // Microsoft 2B: [hidden=2560, kv_out=640]
                    if w_in == hidden_size && w_out % head_dim == 0 {
                        let inferred_kv_heads = w_out / head_dim;
                        debug!("inferred_kv_heads={}, num_heads={}", inferred_kv_heads, num_heads);
                        if inferred_kv_heads != 0
                            && inferred_kv_heads <= num_heads
                            && num_heads.is_multiple_of(inferred_kv_heads)
                        {
                            info!(
                                "Inferred GQA: {} KV heads from tensor {} shape {:?}",
                                inferred_kv_heads, tensor_name, info.shape
                            );
                            return Ok(inferred_kv_heads);
                        }
                    }
                }
            } else {
                debug!("Tensor {} not found", tensor_name);
            }
        }

        // No inference possible, default to MHA
        Ok(num_heads)
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

impl GgufLoader {
    /// PATCH 7: Debug probe for I2_S dequantization to catch zero outputs
    fn debug_probe_i2s_tensor(name: &str, data: &[f32], shape: &[usize], max_probe: usize) {
        if data.is_empty() {
            eprintln!("[i2s probe] {name}: shape={:?} -> EMPTY DATA", shape);
            return;
        }

        let probe_len = max_probe.min(data.len());
        let slice = &data[..probe_len];

        let (min, max, sum) =
            slice.iter().fold((f32::INFINITY, f32::NEG_INFINITY, 0.0), |(mi, ma, s), &x| {
                (mi.min(x), ma.max(x), s + x)
            });
        let mean = sum / (slice.len() as f32);
        let non_zero_count = slice.iter().filter(|&&x| x != 0.0).count();

        eprintln!(
            "[i2s probe] {name}: shape={:?} probe_len={probe_len} -> min={min:.6} max={max:.6} mean={mean:.6} non_zeros={non_zero_count}",
            shape
        );

        if non_zero_count == 0 {
            eprintln!("⚠️  [i2s probe] {name}: ALL ZEROS detected - quantization failure!");
        } else if non_zero_count < probe_len / 4 {
            eprintln!(
                "⚠️  [i2s probe] {name}: Mostly zeros ({non_zero_count}/{probe_len}) - possible quantization issue"
            );
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
            "embed_tokens.weight"
                | "tok_embeddings.weight"
                | "token_embd.weight"
                | "model.embed_tokens.weight"
                | "transformer.wte.weight"
        )
    }

    /// Check if a tensor name indicates it's a projection tensor that needs transposition
    /// This includes both attention and feed-forward projection tensors
    fn is_projection_tensor(name: &str) -> bool {
        // Attention projection tensors
        name.contains("attn_q.weight") ||
        name.contains("attn_k.weight") ||
        name.contains("attn_v.weight") ||
        name.contains("attn_output.weight") ||
        name.contains("q_proj.weight") ||
        name.contains("k_proj.weight") ||
        name.contains("v_proj.weight") ||
        name.contains("o_proj.weight") ||
        // Feed-forward projection tensors
        name.contains("ffn_gate.weight") ||
        name.contains("ffn_up.weight") ||
        name.contains("ffn_down.weight") ||
        name.contains("gate_proj.weight") ||
        name.contains("up_proj.weight") ||
        name.contains("down_proj.weight")
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

    /// Helper to create a transposed I2_S tensor (for attention projections)
    fn create_transposed_i2s_tensor(
        data: &[u8],
        dims: &[usize],
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        use crate::quant::i2s;

        // First dequantize to F32 with original layout
        let f32_data = i2s::dequantize_to_f32(data, dims)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;

        // PATCH 7: Probe the dequantized data for debugging
        Self::debug_probe_i2s_tensor(&"transposed_projection".to_string(), &f32_data, dims, 1000);

        // Then transpose from [rows, cols] to [cols, rows]
        let (rows, cols) = (dims[0], dims[1]);
        let mut transposed = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = f32_data[r * cols + c];
            }
        }

        // Create tensor with transposed dimensions
        let tensor = Tensor::from_slice(&transposed, &[cols, rows], device)
            .map_err(|e| BitNetError::Validation(e.to_string()))?;
        Ok(tensor)
    }

    fn extract_config(&self, reader: &GgufReader) -> Result<BitNetConfig> {
        let mut config = BitNetConfig::default();

        // Extract model configuration from GGUF metadata
        if let Some(vocab_size) = reader.get_u32_metadata("llama.vocab_size") {
            config.model.vocab_size = vocab_size as usize;
        }

        if let Some(num_layers) = reader.get_u32_metadata("llama.block_count") {
            config.model.num_layers = num_layers as usize;
        }

        // If layer count wasn't in metadata or seems wrong, infer from tensors
        if (config.model.num_layers == 0
            || config.model.num_layers == BitNetConfig::default().model.num_layers)
            && let Some(layers) = Self::infer_num_layers_from_tensors(reader)
        {
            tracing::info!("Inferred num_layers={} from tensor analysis", layers);
            config.model.num_layers = layers;
        }

        // 1) hidden_size: try metadata, else infer from embeddings
        if let Some(h) =
            Self::get_u32_any(reader, &["llama.embedding_length", "n_embd", "hidden_size"])
        {
            config.model.hidden_size = h as usize;
        }
        if (config.model.hidden_size == 0
            || config.model.hidden_size == BitNetConfig::default().model.hidden_size)
            && let Some(h) = Self::infer_hidden_size_from_tensors(reader)
        {
            config.model.hidden_size = h;
        }

        // 2) num_heads: broaden key set (MS 2B commonly has "n_head")
        if let Some(h) = Self::get_u32_any(
            reader,
            &["llama.attention.head_count", "n_head", "attn.n_heads", "num_attention_heads"],
        ) {
            config.model.num_heads = h as usize;
        }

        // 3) num_key_value_heads:
        //    a) metadata if present
        let kv_keys = [
            "n_head_kv",
            "n_kv_heads",
            "attn.n_kv_heads",
            "attn_n_kv_heads",
            "num_key_value_heads",
            "llama.attention.head_count_kv",
        ];
        config.model.num_key_value_heads =
            Self::get_u32_any(reader, &kv_keys).map(|v| v as usize).unwrap_or(0);

        //    b) if not present, infer from tensor shapes (now that hidden_size & num_heads are set)
        if config.model.num_key_value_heads == 0
            && config.model.num_heads > 0
            && config.model.hidden_size > 0
        {
            debug!("No explicit GQA metadata found, attempting shape inference...");
            config.model.num_key_value_heads = Self::infer_kv_heads_from_tensors(reader, &config)?;
            debug!("Final num_key_value_heads: {}", config.model.num_key_value_heads);
        }

        //    c) final fallback: MHA
        if config.model.num_key_value_heads == 0 {
            config.model.num_key_value_heads = config.model.num_heads;
        }

        // Log one-liner so you can grep it during runs
        let hidden = config.model.hidden_size;
        let q = config.model.num_heads;
        let kv = config.model.num_key_value_heads;
        if q > 0 && hidden % q == 0 && kv > 0 && q % kv == 0 {
            let head_dim = hidden / q;
            let group = q / kv;
            info!("heads: q={} kv={} (group={}) head_dim={}", q, kv, group, head_dim);
        }

        // 4) intermediate_size: try metadata, else infer from feed-forward tensors
        if let Some(intermediate_size) = reader.get_u32_metadata("llama.feed_forward_length") {
            config.model.intermediate_size = intermediate_size as usize;
        }
        // If no metadata or if it seems wrong (based on tensor shapes), infer from tensors
        if (config.model.intermediate_size == 0
            || config.model.intermediate_size == BitNetConfig::default().model.intermediate_size)
            && let Some(inferred_size) =
                Self::infer_intermediate_size_from_tensors(reader, config.model.hidden_size)
        {
            config.model.intermediate_size = inferred_size;
        }

        if let Some(context_length) = reader.get_u32_metadata("llama.context_length") {
            config.model.max_position_embeddings = context_length as usize;
        }

        // Log final model configuration
        info!(
            "model dimensions: hidden={}, intermediate={}, layers={}, vocab={}",
            config.model.hidden_size,
            config.model.intermediate_size,
            config.model.num_layers,
            config.model.vocab_size
        );

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
                if Self::is_embedding_tensor(&info.name)
                    && Self::embedding_is_transposed(&info.shape)
                {
                    info!("Embedding appears transposed ({:?}) -> decoding transposed", info.shape);
                    let f32_data = i2s::dequantize_to_f32_transposed(data, &info.shape)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;

                    // PATCH 7: Probe the dequantized data for debugging
                    Self::debug_probe_i2s_tensor(&info.name, &f32_data, &info.shape, 1000);
                    // Now dims become [vocab, hidden]
                    let (rows, cols) = (info.shape[1], info.shape[0]);
                    let tensor = Tensor::from_slice(&f32_data, &[rows, cols], &candle_device)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    return Ok(tensor);
                } else if Self::is_projection_tensor(&info.name) && info.shape.len() == 2 {
                    // Projection tensors need transposition for linear layer compatibility
                    debug!(
                        "Transposing projection tensor '{}' from {:?} to {:?}",
                        info.name,
                        info.shape,
                        [info.shape[1], info.shape[0]]
                    );
                    return Self::create_transposed_i2s_tensor(data, &info.shape, &candle_device);
                } else {
                    // Normal I2_S dequantization
                    let mut f32_data = i2s::dequantize_to_f32(data, &info.shape)
                        .map_err(|e| BitNetError::Validation(e.to_string()))?;
                    // Transpose once to [out,in] if this is a projection weight
                    let (mut rows, mut cols) = (info.shape[0], info.shape[1]);
                    let mut want_shape = info.shape.clone();
                    if Self::maybe_transpose_to_out_in(&info.shape, &info.name) {
                        // f32_data currently [rows, cols]=[in,out]; flip to [out,in]
                        let mut transposed = vec![0f32; rows * cols];
                        for r in 0..rows {
                            for c in 0..cols {
                                transposed[c * rows + r] = f32_data[r * cols + c];
                            }
                        }
                        f32_data = transposed;
                        (rows, cols) = (cols, rows);
                        want_shape = vec![rows, cols];
                        tracing::debug!(
                            "pre-transposed {} to [out,in]={:?}",
                            info.name,
                            want_shape
                        );
                    }

                    // PATCH 7: Probe the dequantized data for debugging
                    Self::debug_probe_i2s_tensor(&info.name, &f32_data, &want_shape, 1000);
                    let tensor =
                        Tensor::from_slice(&f32_data, want_shape.as_slice(), &candle_device)
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
                    if Self::is_embedding_tensor(&info.name)
                        && Self::embedding_is_transposed(&info.shape)
                    {
                        info!(
                            "Embedding appears transposed ({:?}) -> decoding transposed",
                            info.shape
                        );
                        let f32_data = Self::transpose_f32_to_f32(data, &info.shape)?;
                        // Now dims become [vocab, hidden]
                        let (rows, cols) = (info.shape[1], info.shape[0]);
                        Tensor::from_slice(&f32_data, &[rows, cols], &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))
                    } else if Self::maybe_transpose_to_out_in(&info.shape, &info.name) {
                        // Apply unified transpose logic for F32 projection weights
                        debug!(
                            "pre-transposing F32 projection '{}' from {:?} to [out,in]",
                            info.name, info.shape
                        );
                        let f32_data = Self::transpose_f32_to_f32(data, &info.shape)?;
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
                    if Self::is_embedding_tensor(&info.name)
                        && Self::embedding_is_transposed(&info.shape)
                    {
                        info!(
                            "Embedding appears transposed ({:?}) -> decoding transposed",
                            info.shape
                        );
                        let f32_data = Self::transpose_f16_to_f32(data, &info.shape)?;
                        // Now dims become [vocab, hidden]
                        let (rows, cols) = (info.shape[1], info.shape[0]);
                        Tensor::from_slice(&f32_data, &[rows, cols], &candle_device)
                            .map_err(|e| BitNetError::Validation(e.to_string()))
                    } else if Self::maybe_transpose_to_out_in(&info.shape, &info.name) {
                        // Apply unified transpose logic for F16 projection weights
                        debug!(
                            "pre-transposing F16 projection '{}' from {:?} to [out,in]",
                            info.name, info.shape
                        );
                        let f32_data = Self::transpose_f16_to_f32(data, &info.shape)?;
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
