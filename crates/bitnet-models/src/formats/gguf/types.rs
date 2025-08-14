//! GGUF type definitions

use bitnet_common::{BitNetError, ModelError, Result};

/// GGUF file header
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    pub fn read(data: &[u8], offset: &mut usize) -> Result<Self> {
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
    pub fn read(data: &[u8], offset: &mut usize) -> Result<Self> {
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
    pub fn read(data: &[u8], offset: &mut usize) -> Result<Self> {
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
                    // Create a temporary buffer with the array type byte
                    let mut temp_data = vec![array_type];
                    temp_data.extend_from_slice(&data[*offset..]);
                    let mut temp_offset = 0;
                    
                    array.push(GgufValue::read(&temp_data, &mut temp_offset)?);
                    *offset += temp_offset - 1; // Adjust for the type byte we added
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
    pub fn read(data: &[u8], offset: &mut usize) -> Result<Self> {
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
#[allow(non_camel_case_types)]
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
    I2_S,  // BitNet 2-bit signed quantization (type 36)
}

impl GgufTensorType {
    pub fn from_u32(value: u32) -> Result<Self> {
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
            36 => Ok(Self::I2_S),  // BitNet I2_S format
            _ => Err(BitNetError::Model(ModelError::InvalidFormat {
                format: format!("Unknown tensor type: {}", value),
            })),
        }
    }
    
    pub fn element_size(&self) -> usize {
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
            Self::I2_S => {
                // Delegate to centralized I2SLayout
                use bitnet_quantization::I2SLayout;
                I2SLayout::default().bytes_per_block
            }
        }
    }
    
    /// Check if this tensor type represents quantized data
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::F32 | Self::F16)
    }
    
    /// Get the block size for quantized types
    pub fn block_size(&self) -> usize {
        match self {
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K => 256,
            Self::Q3_K => 256,
            Self::Q4_K => 256,
            Self::Q5_K => 256,
            Self::Q6_K => 256,
            Self::Q8_K => 256,
            Self::I2_S => {
                // Delegate to centralized I2SLayout
                use bitnet_quantization::I2SLayout;
                I2SLayout::default().block_size
            }
            _ => 1, // Non-quantized types
        }
    }
}

/// Container for loaded tensors
pub type GgufTensors = std::collections::HashMap<String, candle_core::Tensor>;

// Helper functions for reading binary data
pub fn read_u8(data: &[u8], offset: &mut usize) -> Result<u8> {
    if *offset >= data.len() {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: "Unexpected end of data".to_string(),
        }));
    }
    let value = data[*offset];
    *offset += 1;
    Ok(value)
}

pub fn read_i8(data: &[u8], offset: &mut usize) -> Result<i8> {
    Ok(read_u8(data, offset)? as i8)
}

pub fn read_u16(data: &[u8], offset: &mut usize) -> Result<u16> {
    if *offset + 2 > data.len() {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: "Unexpected end of data".to_string(),
        }));
    }
    let value = u16::from_le_bytes([data[*offset], data[*offset + 1]]);
    *offset += 2;
    Ok(value)
}

pub fn read_i16(data: &[u8], offset: &mut usize) -> Result<i16> {
    Ok(read_u16(data, offset)? as i16)
}

pub fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32> {
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

pub fn read_i32(data: &[u8], offset: &mut usize) -> Result<i32> {
    Ok(read_u32(data, offset)? as i32)
}

pub fn read_u64(data: &[u8], offset: &mut usize) -> Result<u64> {
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

pub fn read_f32(data: &[u8], offset: &mut usize) -> Result<f32> {
    let bytes = [
        data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3]
    ];
    *offset += 4;
    Ok(f32::from_le_bytes(bytes))
}

pub fn read_bool(data: &[u8], offset: &mut usize) -> Result<bool> {
    Ok(read_u8(data, offset)? != 0)
}

pub fn read_string(data: &[u8], offset: &mut usize) -> Result<String> {
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