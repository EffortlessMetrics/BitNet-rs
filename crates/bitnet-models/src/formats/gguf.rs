//! GGUF format loader

use crate::loader::{FormatLoader, LoadConfig, MmapFile};
use crate::{Model, BitNetModel};
use bitnet_common::{BitNetConfig, ModelMetadata, QuantizationType, Result, BitNetError, ModelError};
use candle_core::Device;
use std::path::Path;
use tracing::{debug, info, warn};

/// GGUF format loader
pub struct GgufLoader;

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
        
        let metadata = ModelMetadata {
            name: reader.get_string_metadata("general.name")
                .unwrap_or_else(|| path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string()),
            version: reader.get_string_metadata("general.version")
                .unwrap_or_else(|| "unknown".to_string()),
            architecture: reader.get_string_metadata("general.architecture")
                .unwrap_or_else(|| "bitnet".to_string()),
            vocab_size: reader.get_u32_metadata("llama.vocab_size")
                .unwrap_or(32000) as usize,
            context_length: reader.get_u32_metadata("llama.context_length")
                .unwrap_or(2048) as usize,
            quantization: reader.get_quantization_type(),
        };
        
        debug!("Extracted GGUF metadata: {:?}", metadata);
        Ok(metadata)
    }
    
    fn load(
        &self,
        path: &Path,
        device: &Device,
        config: &LoadConfig,
    ) -> Result<Box<dyn Model<Config = BitNetConfig>>> {
        info!("Loading GGUF model from: {}", path.display());
        
        let mmap = if config.use_mmap {
            Some(MmapFile::open(path)?)
        } else {
            None
        };
        
        let data = if let Some(ref mmap) = mmap {
            mmap.as_slice()
        } else {
            // Read entire file into memory
            &std::fs::read(path).map_err(|e| BitNetError::Io(e))?
        };
        
        let reader = GgufReader::new(data)?;
        
        // Report progress
        if let Some(callback) = &config.progress_callback {
            callback(0.3, "Parsing GGUF header...");
        }
        
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
        let model = BitNetModel::from_gguf(model_config, tensors, device.clone())?;
        
        Ok(Box::new(model))
    }
}

impl GgufLoader {
    fn extract_config(&self, reader: &GgufReader) -> Result<BitNetConfig> {
        let mut config = BitNetConfig::default();
        
        // Extract model configuration from GGUF metadata
        if let Some(vocab_size) = reader.get_u32_metadata("llama.vocab_size") {
            config.model.vocab_size = vocab_size as usize;
        }
        
        if let Some(hidden_size) = reader.get_u32_metadata("llama.embedding_length") {
            config.model.hidden_size = hidden_size as usize;
        }
        
        if let Some(num_layers) = reader.get_u32_metadata("llama.block_count") {
            config.model.num_layers = num_layers as usize;
        }
        
        if let Some(num_heads) = reader.get_u32_metadata("llama.attention.head_count") {
            config.model.num_heads = num_heads as usize;
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
        
        Ok(config)
    }
    
    fn load_tensors(
        &self,
        reader: &GgufReader,
        device: &Device,
        config: &LoadConfig,
    ) -> Result<GgufTensors> {
        let tensor_count = reader.tensor_count();
        let mut tensors = GgufTensors::new();
        
        for i in 0..tensor_count {
            if let Some(callback) = &config.progress_callback {
                let progress = 0.5 + (i as f32 / tensor_count as f32) * 0.4;
                callback(progress, &format!("Loading tensor {}/{}", i + 1, tensor_count));
            }
            
            let tensor_info = reader.get_tensor_info(i)?;
            let tensor_data = reader.get_tensor_data(i)?;
            
            // Convert to Candle tensor
            let candle_tensor = self.create_candle_tensor(&tensor_info, tensor_data, device)?;
            tensors.insert(tensor_info.name.clone(), candle_tensor);
        }
        
        Ok(tensors)
    }
    
    fn create_candle_tensor(
        &self,
        info: &TensorInfo,
        data: &[u8],
        device: &Device,
    ) -> Result<candle_core::Tensor> {
        use candle_core::{DType, Tensor};
        
        let dtype = match info.tensor_type {
            GgufTensorType::F32 => DType::F32,
            GgufTensorType::F16 => DType::F16,
            GgufTensorType::Q4_0 => DType::U8, // Quantized types stored as bytes
            GgufTensorType::Q4_1 => DType::U8,
            GgufTensorType::Q5_0 => DType::U8,
            GgufTensorType::Q5_1 => DType::U8,
            GgufTensorType::Q8_0 => DType::U8,
            _ => return Err(BitNetError::Model(ModelError::InvalidFormat {
                format: format!("Unsupported tensor type: {:?}", info.tensor_type),
            })),
        };
        
        // For quantized tensors, we need special handling
        if matches!(info.tensor_type, 
            GgufTensorType::Q4_0 | GgufTensorType::Q4_1 | 
            GgufTensorType::Q5_0 | GgufTensorType::Q5_1 | 
            GgufTensorType::Q8_0
        ) {
            // Create tensor from raw bytes for quantized data
            let tensor = Tensor::from_raw_buffer(data, dtype, &info.shape, device)
                .map_err(|e| BitNetError::Validation(e.to_string()))?;
            Ok(tensor)
        } else {
            // For regular tensors, interpret the bytes according to the data type
            match dtype {
                DType::F32 => {
                    let float_data = bytemuck::cast_slice::<u8, f32>(data);
                    Tensor::from_slice(float_data, &info.shape, device)
                        .map_err(|e| BitNetError::Validation(e.to_string()))
                }
                DType::F16 => {
                    let half_data = bytemuck::cast_slice::<u8, candle_core::half::f16>(data);
                    Tensor::from_slice(half_data, &info.shape, device)
                        .map_err(|e| BitNetError::Validation(e.to_string()))
                }
                _ => Err(BitNetError::Model(ModelError::InvalidFormat {
                    format: format!("Unsupported data type: {:?}", dtype),
                })),
            }
        }
    }
}

/// GGUF file reader
pub struct GgufReader<'a> {
    data: &'a [u8],
    header: GgufHeader,
    metadata: Vec<GgufMetadata>,
    tensor_infos: Vec<TensorInfo>,
}

impl<'a> GgufReader<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 16 {
            return Err(BitNetError::Model(ModelError::InvalidFormat {
                format: "File too small to be a valid GGUF file".to_string(),
            }));
        }
        
        let mut offset = 0;
        
        // Read header
        let header = GgufHeader::read(data, &mut offset)?;
        
        // Read metadata
        let mut metadata = Vec::new();
        for _ in 0..header.metadata_kv_count {
            metadata.push(GgufMetadata::read(data, &mut offset)?);
        }
        
        // Read tensor infos
        let mut tensor_infos = Vec::new();
        for _ in 0..header.tensor_count {
            tensor_infos.push(TensorInfo::read(data, &mut offset)?);
        }
        
        Ok(Self {
            data,
            header,
            metadata,
            tensor_infos,
        })
    }
    
    pub fn tensor_count(&self) -> usize {
        self.header.tensor_count as usize
    }
    
    pub fn get_tensor_info(&self, index: usize) -> Result<&TensorInfo> {
        self.tensor_infos.get(index)
            .ok_or_else(|| BitNetError::Validation(
                format!("Tensor index {} out of bounds", index)
            ))
    }
    
    pub fn get_tensor_data(&self, index: usize) -> Result<&[u8]> {
        let info = self.get_tensor_info(index)?;
        let start = info.offset as usize;
        let end = start + info.size as usize;
        
        if end > self.data.len() {
            return Err(BitNetError::Model(ModelError::LoadingFailed {
                reason: format!("Tensor data extends beyond file bounds"),
            }));
        }
        
        Ok(&self.data[start..end])
    }
    
    pub fn get_string_metadata(&self, key: &str) -> Option<String> {
        self.metadata.iter()
            .find(|m| m.key == key)
            .and_then(|m| match &m.value {
                GgufValue::String(s) => Some(s.clone()),
                _ => None,
            })
    }
    
    pub fn get_u32_metadata(&self, key: &str) -> Option<u32> {
        self.metadata.iter()
            .find(|m| m.key == key)
            .and_then(|m| match &m.value {
                GgufValue::U32(v) => Some(*v),
                _ => None,
            })
    }
    
    pub fn get_quantization_type(&self) -> Option<QuantizationType> {
        // Infer quantization type from tensor types
        for tensor_info in &self.tensor_infos {
            match tensor_info.tensor_type {
                GgufTensorType::Q4_0 | GgufTensorType::Q4_1 => return Some(QuantizationType::I2S),
                GgufTensorType::Q5_0 | GgufTensorType::Q5_1 => return Some(QuantizationType::TL1),
                GgufTensorType::Q8_0 => return Some(QuantizationType::TL2),
                _ => continue,
            }
        }
        None
    }
}

/// GGUF file header
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    fn read(data: &[u8], offset: &mut usize) -> Result<Self> {
        if data.len() < *offset + 24 {
            return Err(BitNetError::Model(ModelError::InvalidFormat {
                format: "Insufficient data for GGUF header".to_string(),
            }));
        }
        
        let magic = [data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3]];
        *offset += 4;
        
        if &magic != b"GGUF" {
            return Err(BitNetError::Model(ModelError::InvalidFormat {
                format: "Invalid GGUF magic number".to_string(),
            }));
        }
        
        let version = u32::from_le_bytes([
            data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3]
        ]);
        *offset += 4;
        
        let tensor_count = u64::from_le_bytes([
            data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3],
            data[*offset + 4], data[*offset + 5], data[*offset + 6], data[*offset + 7],
        ]);
        *offset += 8;
        
        let metadata_kv_count = u64::from_le_bytes([
            data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3],
            data[*offset + 4], data[*offset + 5], data[*offset + 6], data[*offset + 7],
        ]);
        *offset += 8;
        
        Ok(Self {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

/// GGUF metadata entry
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub key: String,
    pub value: GgufValue,
}

impl GgufMetadata {
    fn read(data: &[u8], offset: &mut usize) -> Result<Self> {
        let key = read_string(data, offset)?;
        let value = GgufValue::read(data, offset)?;
        Ok(Self { key, value })
    }
}

/// GGUF value types
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    fn read(data: &[u8], offset: &mut usize) -> Result<Self> {
        if *offset >= data.len() {
            return Err(BitNetError::Model(ModelError::InvalidFormat {
                format: "Unexpected end of data while reading GGUF value".to_string(),
            }));
        }
        
        let value_type = data[*offset];
        *offset += 1;
        
        match value_type {
            0 => Ok(GgufValue::U8(read_u8(data, offset)?)),
            1 => Ok(GgufValue::I8(read_i8(data, offset)?)),
            2 => Ok(GgufValue::U16(read_u16(data, offset)?)),
            3 => Ok(GgufValue::I16(read_i16(data, offset)?)),
            4 => Ok(GgufValue::U32(read_u32(data, offset)?)),
            5 => Ok(GgufValue::I32(read_i32(data, offset)?)),
            6 => Ok(GgufValue::F32(read_f32(data, offset)?)),
            7 => Ok(GgufValue::Bool(read_bool(data, offset)?)),
            8 => Ok(GgufValue::String(read_string(data, offset)?)),
            9 => {
                // Array type
                let array_type = data[*offset];
                *offset += 1;
                let array_len = read_u64(data, offset)? as usize;
                
                let mut array = Vec::with_capacity(array_len);
                for _ in 0..array_len {
                    // Temporarily set the type byte for reading array elements
                    let saved_offset = *offset;
                    *offset = saved_offset.saturating_sub(1);
                    data[*offset] = array_type;
                    *offset += 1;
                    
                    array.push(GgufValue::read(data, offset)?);
                }
                Ok(GgufValue::Array(array))
            }
            _ => Err(BitNetError::Model(ModelError::InvalidFormat {
                format: format!("Unknown GGUF value type: {}", value_type),
            })),
        }
    }
}

/// Tensor information
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub tensor_type: GgufTensorType,
    pub offset: u64,
    pub size: u64,
}

impl TensorInfo {
    fn read(data: &[u8], offset: &mut usize) -> Result<Self> {
        let name = read_string(data, offset)?;
        
        let n_dims = read_u32(data, offset)? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(read_u64(data, offset)? as usize);
        }
        
        let tensor_type = GgufTensorType::from_u32(read_u32(data, offset)?)?;
        let tensor_offset = read_u64(data, offset)?;
        
        // Calculate tensor size
        let element_size = tensor_type.element_size();
        let total_elements: usize = shape.iter().product();
        let size = (total_elements * element_size) as u64;
        
        Ok(Self {
            name,
            shape,
            tensor_type,
            offset: tensor_offset,
            size,
        })
    }
}

/// GGUF tensor types
#[derive(Debug, Clone, Copy)]
pub enum GgufTensorType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
}

impl GgufTensorType {
    fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            _ => Err(BitNetError::Model(ModelError::InvalidFormat {
                format: format!("Unknown tensor type: {}", value),
            })),
        }
    }
    
    fn element_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18, // 16 4-bit values + 2 bytes for scale
            Self::Q4_1 => 20, // 16 4-bit values + 4 bytes for scale and min
            Self::Q5_0 => 22, // 16 5-bit values + extras
            Self::Q5_1 => 24,
            Self::Q8_0 => 34, // 32 8-bit values + 2 bytes for scale
            Self::Q8_1 => 36,
            Self::Q2_K => 82,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 256,
        }
    }
}

/// Container for loaded tensors
pub type GgufTensors = std::collections::HashMap<String, candle_core::Tensor>;

// Helper functions for reading binary data
fn read_u8(data: &[u8], offset: &mut usize) -> Result<u8> {
    if *offset >= data.len() {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: "Unexpected end of data".to_string(),
        }));
    }
    let value = data[*offset];
    *offset += 1;
    Ok(value)
}

fn read_i8(data: &[u8], offset: &mut usize) -> Result<i8> {
    Ok(read_u8(data, offset)? as i8)
}

fn read_u16(data: &[u8], offset: &mut usize) -> Result<u16> {
    if *offset + 2 > data.len() {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: "Unexpected end of data".to_string(),
        }));
    }
    let value = u16::from_le_bytes([data[*offset], data[*offset + 1]]);
    *offset += 2;
    Ok(value)
}

fn read_i16(data: &[u8], offset: &mut usize) -> Result<i16> {
    Ok(read_u16(data, offset)? as i16)
}

fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32> {
    if *offset + 4 > data.len() {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: "Unexpected end of data".to_string(),
        }));
    }
    let value = u32::from_le_bytes([
        data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3]
    ]);
    *offset += 4;
    Ok(value)
}

fn read_i32(data: &[u8], offset: &mut usize) -> Result<i32> {
    Ok(read_u32(data, offset)? as i32)
}

fn read_u64(data: &[u8], offset: &mut usize) -> Result<u64> {
    if *offset + 8 > data.len() {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: "Unexpected end of data".to_string(),
        }));
    }
    let value = u64::from_le_bytes([
        data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3],
        data[*offset + 4], data[*offset + 5], data[*offset + 6], data[*offset + 7],
    ]);
    *offset += 8;
    Ok(value)
}

fn read_f32(data: &[u8], offset: &mut usize) -> Result<f32> {
    Ok(f32::from_le_bytes([
        data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3]
    ]))
}

fn read_bool(data: &[u8], offset: &mut usize) -> Result<bool> {
    Ok(read_u8(data, offset)? != 0)
}

fn read_string(data: &[u8], offset: &mut usize) -> Result<String> {
    let len = read_u64(data, offset)? as usize;
    if *offset + len > data.len() {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: "String extends beyond data bounds".to_string(),
        }));
    }
    
    let string_data = &data[*offset..*offset + len];
    *offset += len;
    
    String::from_utf8(string_data.to_vec())
        .map_err(|_| BitNetError::Model(ModelError::InvalidFormat {
            format: "Invalid UTF-8 in string".to_string(),
        }))
}