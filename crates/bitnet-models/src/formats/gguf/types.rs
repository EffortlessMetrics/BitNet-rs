//! GGUF type definitions
//!
//! Supports GGUF v2 and v3 headers. For v3, `alignment` (u32) and `data_offset` (u64)
//! are read from the header and sanitized:
//! - `alignment` is clamped to 32 if it is 0 or not a power of two.
//! - `data_offset` is only *used* by the reader if it is ≥ end-of-KV, ≤ file size,
//!   and aligned; otherwise the reader falls back to `align_up(kv_end, alignment)`.

use bitnet_common::{BitNetError, ModelError, Result};

/// Returns the smallest `x >= off` such that `x % align == 0`.
/// Safe for any `align >= 1`.
#[inline]
pub fn align_up(off: usize, align: usize) -> usize {
    if align == 0 {
        return off;
    }
    debug_assert!(align.is_power_of_two(), "alignment should be power-of-two");
    (off + align - 1) & !(align - 1)
}

/// Parsed GGUF header (v2/v3).
/// - For v2, `alignment = 32`, `data_offset = 0`.
/// - For v3, both are read from the header; invalid `alignment` is clamped to 32.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    /// Metadata/tensor alignment in bytes (always a power of two; defaults to 32).
    pub alignment: u32,
    /// Byte offset to tensor data for v3. Zero for v2 or if absent.
    pub data_offset: u64
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
            data[*offset],
            data[*offset + 1],
            data[*offset + 2],
            data[*offset + 3],
        ]);
        *offset += 4;

        let tensor_count = u64::from_le_bytes([
            data[*offset],
            data[*offset + 1],
            data[*offset + 2],
            data[*offset + 3],
            data[*offset + 4],
            data[*offset + 5],
            data[*offset + 6],
            data[*offset + 7],
        ]);
        *offset += 8;

        let metadata_kv_count = u64::from_le_bytes([
            data[*offset],
            data[*offset + 1],
            data[*offset + 2],
            data[*offset + 3],
            data[*offset + 4],
            data[*offset + 5],
            data[*offset + 6],
            data[*offset + 7],
        ]);
        *offset += 8;

        // v3 typically adds alignment (u32) and data_offset (u64)
        // However, some files claim v3 but omit these fields (early v3 format)
        // We detect this by checking if the next bytes look like a KV pair
        let (alignment, data_offset) = if version >= 3 {
            // Check we have enough data to peek
            if data.len() < *offset + 12 {
                // Not enough data for v3 fields, assume v2-style layout
                tracing::warn!("GGUF v3 with insufficient header data, using v2 layout");
                (32u32, 0u64)
            } else {
                // Peek at the next 8 bytes to see if it looks like a u64 string length
                let potential_strlen = u64::from_le_bytes([
                    data[*offset],
                    data[*offset + 1],
                    data[*offset + 2],
                    data[*offset + 3],
                    data[*offset + 4],
                    data[*offset + 5],
                    data[*offset + 6],
                    data[*offset + 7],
                ]);
                
                // Metadata key strings are typically 10-50 chars (e.g., "general.architecture")
                // If we see a reasonable string length, this is likely a KV pair, not alignment/offset
                const STRING_SAMPLE_LEN: usize = 20;
                if potential_strlen > 0 && potential_strlen < 256 {
                    // Check if following bytes could be ASCII text
                    if *offset + 8 + potential_strlen as usize <= data.len() {
                        let potential_string = &data[*offset + 8..*offset + 8 + (potential_strlen as usize).min(20)];
                        let looks_like_key = potential_string.iter()
                            .all(|&b| (b >= b'a' && b <= b'z') || 
                                     (b >= b'A' && b <= b'Z') || 
                                     b == b'.' || b == b'_' || b == b'-');
                        
                        if looks_like_key {
                            // This is an early v3 format without alignment/data_offset
                            tracing::warn!("GGUF v3 without alignment/data_offset fields (early format)");
                            (32u32, 0u64)
                        } else {
                            // Standard v3 with alignment and data_offset
                            let mut align = u32::from_le_bytes([
                                data[*offset],
                                data[*offset + 1],
                                data[*offset + 2],
                                data[*offset + 3],
                            ]);
                            *offset += 4;
                            
                            // Validate alignment
                            if align == 0 || !align.is_power_of_two() {
                                tracing::warn!("GGUF v{}: alignment {} is invalid; using 32", version, align);
                                align = 32;
                            }
                            
                            let doff = u64::from_le_bytes([
                                data[*offset],
                                data[*offset + 1],
                                data[*offset + 2],
                                data[*offset + 3],
                                data[*offset + 4],
                                data[*offset + 5],
                                data[*offset + 6],
                                data[*offset + 7],
                            ]);
                            *offset += 8;
                            
                            (align, doff)
                        }
                    } else {
                        // Can't peek far enough, assume standard v3
                        let mut align = u32::from_le_bytes([
                            data[*offset],
                            data[*offset + 1],
                            data[*offset + 2],
                            data[*offset + 3],
                        ]);
                        *offset += 4;
                        
                        if align == 0 || !align.is_power_of_two() {
                            tracing::warn!("GGUF v{}: alignment {} invalid; using 32", version, align);
                            align = 32;
                        }
                        
                        let doff = u64::from_le_bytes([
                            data[*offset],
                            data[*offset + 1],
                            data[*offset + 2],
                            data[*offset + 3],
                            data[*offset + 4],
                            data[*offset + 5],
                            data[*offset + 6],
                            data[*offset + 7],
                        ]);
                        *offset += 8;
                        
                        (align, doff)
                    }
                } else {
                    // Unreasonably large string length, assume standard v3
                    let mut align = u32::from_le_bytes([
                        data[*offset],
                        data[*offset + 1],
                        data[*offset + 2],
                        data[*offset + 3],
                    ]);
                    *offset += 4;
                    
                    if align == 0 || !align.is_power_of_two() {
                        tracing::warn!("GGUF v{}: alignment {} invalid; using 32", version, align);
                        align = 32;
                    }
                    
                    let doff = u64::from_le_bytes([
                        data[*offset],
                        data[*offset + 1],
                        data[*offset + 2],
                        data[*offset + 3],
                        data[*offset + 4],
                        data[*offset + 5],
                        data[*offset + 6],
                        data[*offset + 7],
                    ]);
                    *offset += 8;
                    
                    (align, doff)
                }
            }
        } else {
            // v2: 32-byte alignment, no data_offset
            (32u32, 0u64)
        };

        Ok(Self { magic, version, tensor_count, metadata_kv_count, alignment, data_offset })
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
        if *offset + 4 > data.len() {
            return Err(BitNetError::Model(ModelError::InvalidFormat {
                format: "Unexpected end of data while reading GGUF value type".to_string(),
            }));
        }

        // GGUF v3 uses 4-byte type field
        let value_type = read_u32(data, offset)?;

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
                let array_type = read_u32(data, offset)?;
                let array_len = read_u64(data, offset)? as usize;

                let mut array = Vec::with_capacity(array_len);

                // Read elements based on the array element type
                match array_type {
                    0 => {
                        // Array of U8
                        for _ in 0..array_len {
                            array.push(GgufValue::U8(read_u8(data, offset)?));
                        }
                    }
                    1 => {
                        // Array of I8
                        for _ in 0..array_len {
                            array.push(GgufValue::I8(read_i8(data, offset)?));
                        }
                    }
                    2 => {
                        // Array of U16
                        for _ in 0..array_len {
                            array.push(GgufValue::U16(read_u16(data, offset)?));
                        }
                    }
                    3 => {
                        // Array of I16
                        for _ in 0..array_len {
                            array.push(GgufValue::I16(read_i16(data, offset)?));
                        }
                    }
                    4 => {
                        // Array of U32
                        for _ in 0..array_len {
                            array.push(GgufValue::U32(read_u32(data, offset)?));
                        }
                    }
                    5 => {
                        // Array of I32
                        for _ in 0..array_len {
                            array.push(GgufValue::I32(read_i32(data, offset)?));
                        }
                    }
                    6 => {
                        // Array of F32
                        for _ in 0..array_len {
                            array.push(GgufValue::F32(read_f32(data, offset)?));
                        }
                    }
                    7 => {
                        // Array of Bool
                        for _ in 0..array_len {
                            array.push(GgufValue::Bool(read_bool(data, offset)?));
                        }
                    }
                    8 => {
                        // Array of String - most common for token pieces
                        for _ in 0..array_len {
                            array.push(GgufValue::String(read_string(data, offset)?));
                        }
                    }
                    _ => {
                        return Err(BitNetError::Model(ModelError::InvalidFormat {
                            format: format!("Unsupported GGUF array element type: {}", array_type),
                        }));
                    }
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
        let total_elements: usize = shape.iter().product();
        let size = if tensor_type.is_quantized() {
            // For quantized types, element_size is actually bytes per block
            let block_size = tensor_type.block_size();
            let bytes_per_block = tensor_type.element_size();
            let num_blocks = (total_elements + block_size - 1) / block_size;
            (num_blocks * bytes_per_block) as u64
        } else {
            // For non-quantized types, element_size is bytes per element
            (total_elements * tensor_type.element_size()) as u64
        };

        Ok(Self { name, shape, tensor_type, offset: tensor_offset, size })
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
    I2_S, // BitNet 2-bit signed quantization (type 36)
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
            36 => Ok(Self::I2_S), // BitNet I2_S format
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
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
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
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
        data[*offset + 4],
        data[*offset + 5],
        data[*offset + 6],
        data[*offset + 7],
    ]);
    *offset += 8;
    Ok(value)
}

pub fn read_f32(data: &[u8], offset: &mut usize) -> Result<f32> {
    let bytes = [data[*offset], data[*offset + 1], data[*offset + 2], data[*offset + 3]];
    *offset += 4;
    Ok(f32::from_le_bytes(bytes))
}

pub fn read_bool(data: &[u8], offset: &mut usize) -> Result<bool> {
    Ok(read_u8(data, offset)? != 0)
}

pub fn read_string(data: &[u8], offset: &mut usize) -> Result<String> {
    let len = read_u64(data, offset)? as usize;

    // Sanity check for reasonable string length (e.g., < 1MB)
    const MAX_STRING_LEN: usize = 1024 * 1024;
    if len > MAX_STRING_LEN {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: format!(
                "String length {} exceeds maximum {} at offset {}",
                len,
                MAX_STRING_LEN,
                *offset - 8
            ),
        }));
    }

    if *offset + len > data.len() {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: format!(
                "String extends beyond data bounds (offset: {}, len: {}, data size: {})",
                *offset,
                len,
                data.len()
            ),
        }));
    }

    let string_data = &data[*offset..*offset + len];
    *offset += len;

    // GGUF may store byte strings (e.g., token pieces) that are not valid UTF-8.
    // Use lossy decoding to handle this gracefully instead of failing.
    match String::from_utf8(string_data.to_vec()) {
        Ok(s) => Ok(s),
        Err(e) => {
            let bytes = e.into_bytes();
            // Log once per problematic string
            tracing::warn!("GGUF string contained invalid UTF-8; decoding lossily");
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
    }
}

/// For fields that genuinely need raw bytes (e.g., token pieces arrays),
/// keep the bytes verbatim.
pub fn read_bytes(data: &[u8], offset: &mut usize) -> Result<Vec<u8>> {
    let len = read_u64(data, offset)? as usize;
    if *offset + len > data.len() {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: "Bytes extend beyond data bounds".to_string(),
        }));
    }

    let bytes = data[*offset..*offset + len].to_vec();
    *offset += len;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lossy_utf8_string_does_not_panic() {
        // Create test data with invalid UTF-8
        // Length = 2, then bytes [0xC3, 0x28] which is invalid UTF-8
        let mut data = Vec::new();
        data.extend_from_slice(&2u64.to_le_bytes()); // length
        data.extend_from_slice(&[0xC3, 0x28]); // invalid UTF-8 sequence

        let mut offset = 0;
        // Should not error; should return a String with replacement char
        let result = read_string(&data, &mut offset);
        assert!(result.is_ok(), "lossy decode should succeed");

        let s = result.unwrap();
        assert!(s.contains('\u{FFFD}'), "expected replacement char in lossy decode");
        assert_eq!(offset, 10, "offset should be updated correctly");
    }

    #[test]
    fn test_valid_utf8_string() {
        // Create test data with valid UTF-8
        let test_str = "Hello, GGUF!";
        let mut data = Vec::new();
        data.extend_from_slice(&(test_str.len() as u64).to_le_bytes());
        data.extend_from_slice(test_str.as_bytes());

        let mut offset = 0;
        let result = read_string(&data, &mut offset);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_str);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }
}
