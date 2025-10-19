//! GGUF type definitions
//!
//! Supports GGUF v2 and v3 headers. For v3, `alignment` (u32) and `data_offset` (u64)
//! are read from the header and sanitized:
//! - `alignment` is clamped to 32 if it is 0 or not a power of two.
//! - `data_offset` is only *used* by the reader if it is ≥ end-of-KV, ≤ file size,
//!   and aligned; otherwise the reader falls back to `align_up(kv_end, alignment)`.
//!
//! ## Security Hardening
//! This module includes comprehensive input validation to prevent memory allocation attacks:
//! - Bounded string lengths and array sizes to prevent DoS
//! - Tensor dimension overflow protection
//! - Progressive memory allocation with safety checks
//! - Resource limits to prevent memory bombs

use bitnet_common::{BitNetError, ModelError, Result, SecurityError, SecurityLimits};

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
    pub data_offset: u64,
}

impl GgufHeader {
    pub fn read(data: &[u8], offset: &mut usize) -> Result<Self> {
        Self::read_with_limits(data, offset, &SecurityLimits::default())
    }

    pub fn read_with_limits(
        data: &[u8],
        offset: &mut usize,
        limits: &SecurityLimits,
    ) -> Result<Self> {
        if data.len() < *offset + 24 {
            return Err(BitNetError::Model(ModelError::InvalidFormat {
                format: "Insufficient data for GGUF header".to_string(),
            }));
        }

        // Check for reasonable file size to prevent processing of maliciously large files
        if data.len() > limits.max_metadata_size * 100 {
            // Allow 10GB max for complete GGUF files
            return Err(BitNetError::Security(SecurityError::MemoryBomb {
                reason: format!("File size {} exceeds safety limit", data.len()),
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

        // Security: Validate tensor count to prevent memory allocation bombs
        if tensor_count > 100_000 {
            // Maximum 100K tensors
            return Err(BitNetError::Security(SecurityError::ResourceLimit {
                resource: "tensor_count".to_string(),
                value: tensor_count,
                limit: 100_000,
            }));
        }

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

        // Security: Validate metadata count to prevent memory allocation bombs
        if metadata_kv_count > 10_000 {
            // Maximum 10K metadata entries
            return Err(BitNetError::Security(SecurityError::ResourceLimit {
                resource: "metadata_kv_count".to_string(),
                value: metadata_kv_count,
                limit: 10_000,
            }));
        }

        // GGUF v3 Format Variants:
        // 1. Standard v3: Has alignment (u32) and data_offset (u64) fields after metadata_kv_count
        // 2. Early v3 variant: Omits alignment and data_offset, goes directly to KV pairs
        //
        // The early v3 variant is used by some models like Microsoft's BitNet models.
        // These files claim to be v3 but use a simpler header structure similar to v2.
        //
        // Detection strategy:
        // - Read the next 8 bytes as a potential u64 string length
        // - If it's a reasonable length (0-256) and followed by ASCII text, it's likely a KV pair
        // - This means we're looking at the early v3 variant without alignment/data_offset
        // - Otherwise, parse as standard v3 with alignment and data_offset fields
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

                // Heuristic: Metadata keys are typically short strings (10-50 chars)
                // Examples: "general.architecture", "llama.attention.head_count"
                // If we see a reasonable string length followed by ASCII text,
                // we're likely looking at the first KV pair, not alignment/data_offset
                const SAMPLE: usize = 20;

                if potential_strlen > 0 && potential_strlen < 256 {
                    // Check if following bytes could be ASCII text
                    if *offset + 8 + potential_strlen as usize <= data.len() {
                        let sample =
                            &data[*offset + 8..*offset + 8 + SAMPLE.min(potential_strlen as usize)];

                        // Keys typically contain: [A-Za-z0-9._-]
                        let looks_like_key = sample.iter().all(|&b| {
                            matches!(b,
                                b'0'..=b'9' |
                                b'a'..=b'z' |
                                b'A'..=b'Z' |
                                b'.' | b'_' | b'-'
                            )
                        });

                        if looks_like_key {
                            // Early v3 variant detected: Missing alignment/data_offset fields
                            // This format is used by Microsoft BitNet models and others
                            // We handle it gracefully by using default alignment (32) and computing offset later
                            tracing::warn!(
                                "GGUF v3 early variant detected (missing alignment/data_offset) - handling gracefully"
                            );
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
                                tracing::warn!(
                                    "GGUF v{}: alignment {} is invalid; using 32",
                                    version,
                                    align
                                );
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
                            tracing::warn!(
                                "GGUF v{}: alignment {} invalid; using 32",
                                version,
                                align
                            );
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
        Self::read_with_limits(data, offset, &SecurityLimits::default())
    }

    pub fn read_with_limits(
        data: &[u8],
        offset: &mut usize,
        limits: &SecurityLimits,
    ) -> Result<Self> {
        let key = read_string_with_limits(data, offset, limits)?;
        let value = GgufValue::read_with_limits(data, offset, limits)?;
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
        Self::read_with_limits(data, offset, &SecurityLimits::default())
    }

    pub fn read_with_limits(
        data: &[u8],
        offset: &mut usize,
        limits: &SecurityLimits,
    ) -> Result<Self> {
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

                // Security: Validate array length to prevent memory allocation bombs
                if array_len > limits.max_array_length {
                    return Err(BitNetError::Security(SecurityError::ResourceLimit {
                        resource: "array_length".to_string(),
                        value: array_len as u64,
                        limit: limits.max_array_length as u64,
                    }));
                }

                // Calculate memory requirements to prevent integer overflow
                // For string arrays, we use a reasonable average (256 bytes) instead of max_string_length
                // to avoid false positives on large vocabularies (e.g., 128K tokens)
                // Individual strings are still validated against max_string_length during parsing
                let element_size = match array_type {
                    0 | 1 => 1, // U8, I8
                    2 | 3 => 2, // U16, I16
                    4..=6 => 4, // U32, I32, F32
                    7 => 1,     // Bool
                    8 => 256,   // String (reasonable average for tokenizer vocab)
                    _ => {
                        return Err(BitNetError::Security(SecurityError::MalformedData {
                            reason: format!("Invalid array element type: {}", array_type),
                        }));
                    }
                };

                let memory_required = array_len.saturating_mul(element_size);
                if memory_required > limits.max_memory_allocation {
                    return Err(BitNetError::Security(SecurityError::MemoryBomb {
                        reason: format!(
                            "Array memory requirement {} exceeds limit {}",
                            memory_required, limits.max_memory_allocation
                        ),
                    }));
                }

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
                            array.push(GgufValue::String(read_string_with_limits(
                                data, offset, limits,
                            )?));
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
        Self::read_with_limits(data, offset, &SecurityLimits::default())
    }

    pub fn read_with_limits(
        data: &[u8],
        offset: &mut usize,
        limits: &SecurityLimits,
    ) -> Result<Self> {
        let name = read_string_with_limits(data, offset, limits)?;

        let n_dims = read_u32(data, offset)? as usize;

        // Security: Limit tensor dimensions to prevent memory bombs
        if n_dims > 8 {
            // Maximum 8 dimensions for any reasonable tensor
            return Err(BitNetError::Security(SecurityError::ResourceLimit {
                resource: "tensor_dimensions".to_string(),
                value: n_dims as u64,
                limit: 8,
            }));
        }

        let mut shape = Vec::with_capacity(n_dims);
        let mut total_elements = 1u64;

        for i in 0..n_dims {
            let dim_u64 = read_u64(data, offset)?;

            // Security: Check for dimension overflow (u64 -> usize conversion)
            let dim = if dim_u64 > usize::MAX as u64 {
                return Err(BitNetError::Security(SecurityError::ResourceLimit {
                    resource: "tensor_dimension_size".to_string(),
                    value: dim_u64,
                    limit: usize::MAX as u64,
                }));
            } else {
                dim_u64 as usize
            };

            // Security: Check for dimension overflow and unreasonable sizes
            if dim == 0 {
                return Err(BitNetError::Security(SecurityError::MalformedData {
                    reason: format!("Tensor dimension {} cannot be zero", i),
                }));
            }

            if dim > 1_000_000_000 {
                // 1B elements per dimension max
                return Err(BitNetError::Security(SecurityError::ResourceLimit {
                    resource: "tensor_dimension".to_string(),
                    value: dim as u64,
                    limit: 1_000_000_000,
                }));
            }

            // Security: Check for multiplication overflow using checked arithmetic
            total_elements = total_elements.checked_mul(dim as u64).ok_or_else(|| {
                BitNetError::Security(SecurityError::MemoryBomb {
                    reason: format!("Tensor dimension multiplication overflow at dimension {}", i),
                })
            })?;

            if total_elements > limits.max_tensor_elements {
                return Err(BitNetError::Security(SecurityError::ResourceLimit {
                    resource: "tensor_elements".to_string(),
                    value: total_elements,
                    limit: limits.max_tensor_elements,
                }));
            }

            shape.push(dim);
        }

        let tensor_type = GgufTensorType::from_u32(read_u32(data, offset)?)?;
        let tensor_offset = read_u64(data, offset)?;

        // Calculate tensor size with overflow protection
        let total_elements: usize = total_elements as usize; // Already validated above

        let size = if tensor_type.is_quantized() {
            // For quantized types, element_size is actually bytes per block
            let block_size = tensor_type.block_size();
            let bytes_per_block = tensor_type.element_size();
            let num_blocks = total_elements.div_ceil(block_size);

            // Security: Check for size calculation overflow
            let size_bytes = num_blocks.saturating_mul(bytes_per_block);
            if size_bytes > limits.max_memory_allocation {
                return Err(BitNetError::Security(SecurityError::MemoryBomb {
                    reason: format!(
                        "Tensor memory requirement {} exceeds limit {}",
                        size_bytes, limits.max_memory_allocation
                    ),
                }));
            }
            size_bytes as u64
        } else {
            // For non-quantized types, element_size is bytes per element
            let size_bytes = total_elements.saturating_mul(tensor_type.element_size());
            if size_bytes > limits.max_memory_allocation {
                return Err(BitNetError::Security(SecurityError::MemoryBomb {
                    reason: format!(
                        "Tensor memory requirement {} exceeds limit {}",
                        size_bytes, limits.max_memory_allocation
                    ),
                }));
            }
            size_bytes as u64
        };

        Ok(Self { name, shape, tensor_type, offset: tensor_offset, size })
    }
}

/// GGUF tensor types
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub enum GgufTensorType {
    F32,
    F16,
    F64,
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
    IQ2_S, // GGML IQ2_S quantization (type 24)
    I2_S,  // BitNet 2-bit signed quantization (type 36)
}

impl GgufTensorType {
    /// Parse quantization type from string, tolerating common aliases
    pub fn from_quant_string(s: &str) -> Option<Self> {
        let s = s.to_ascii_lowercase();
        match s.as_str() {
            "i2_s" | "is_2" | "is2" => Some(Self::I2_S),
            "iq2_s" => Some(Self::IQ2_S),
            "q4_0" => Some(Self::Q4_0),
            "q4_1" => Some(Self::Q4_1),
            "q5_0" => Some(Self::Q5_0),
            "q5_1" => Some(Self::Q5_1),
            "q8_0" => Some(Self::Q8_0),
            "q8_1" => Some(Self::Q8_1),
            "q2_k" => Some(Self::Q2_K),
            "q3_k" => Some(Self::Q3_K),
            "q4_k" => Some(Self::Q4_K),
            "q5_k" => Some(Self::Q5_K),
            "q6_k" => Some(Self::Q6_K),
            "q8_k" => Some(Self::Q8_K),
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }

    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            4 => Ok(Self::F64), // F64 type (rarely used)
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
            24 => Ok(Self::IQ2_S), // GGML IQ2_S format
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
            Self::F64 => 8,
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
            Self::IQ2_S => 82, // GGML IQ2_S block size: 64 bytes + 2 scale + 8 qh + 8 scales
            Self::I2_S => {
                // GGML I2_S: 8 bytes packed data per block (scales stored separately or externally)
                // 32 elements * 2 bits / 8 bits/byte = 8 bytes
                // NOTE: Scales may be stored in separate scale tensors, not inline
                8
            }
        }
    }

    /// Check if this tensor type represents quantized data
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::F64)
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
            Self::IQ2_S => 256, // GGML IQ2_S uses 256-element blocks
            Self::I2_S => {
                // GGML I2_S uses 32-element blocks (32 * 2 bits = 64 bits = 8 bytes packed data)
                // This matches BitNet's internal I2SLayout.block_size
                32
            }
            _ => 1, // Non-quantized types
        }
    }
}

/// I2_S quantization layout detection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I2SLayoutKind {
    /// GGML split layout: 32 elems per block, 8B packed data, scales in separate tensor (f32/f16/f64)
    GgmlSplit,
    /// Inline legacy layout: 32 elems per block, 8B packed qbits + 2B f16 scale = 10B per block
    InlineF16,
}

impl I2SLayoutKind {
    pub fn block_size(&self) -> usize {
        32
    }
    pub fn data_bytes_per_block(&self) -> usize {
        8
    }
    pub fn total_bytes_per_block(&self) -> usize {
        match self {
            I2SLayoutKind::GgmlSplit => 8,  // data only; scales elsewhere
            I2SLayoutKind::InlineF16 => 10, // 8 data + 2 scale
        }
    }
}

/// I2_S flavor enum for comprehensive layout detection
///
/// Supports multiple I2_S quantization formats found in the wild:
/// - BitNet32F16: Original BitNet format (32 elem blocks, 10 B/block with inline f16 scales)
/// - Split32WithSibling: Research-style split format (32 elem blocks, 8 B/block + separate scale tensor)
/// - GgmlQk256NoScale: GGML/llama.cpp format (256 elem blocks, 64 B/block, no per-block scales)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I2SFlavor {
    /// BitNet original format: 32 elem per block, 10 B/block (8B packed + 2B f16 scale inline)
    BitNet32F16,
    /// Split format with sibling scale tensor: 32 elem per block, 8 B/block (data only)
    /// Requires separate scale tensor (f32/f16/f64) with matching block count
    Split32WithSibling,
    /// GGML/llama.cpp format: 256 elem per block, 64 B/block, no per-block scales
    /// Requires native GGML kernels (not yet supported in pure Rust)
    GgmlQk256NoScale,
}

impl I2SFlavor {
    /// Get the number of elements per block for this flavor
    pub fn block_size(&self) -> usize {
        match self {
            I2SFlavor::BitNet32F16 => 32,
            I2SFlavor::Split32WithSibling => 32,
            I2SFlavor::GgmlQk256NoScale => 256,
        }
    }

    /// Get the number of bytes per block for this flavor (data portion only)
    pub fn data_bytes_per_block(&self) -> usize {
        match self {
            I2SFlavor::BitNet32F16 => 8,
            I2SFlavor::Split32WithSibling => 8,
            I2SFlavor::GgmlQk256NoScale => 64,
        }
    }

    /// Get the total bytes per block including inline metadata
    pub fn total_bytes_per_block(&self) -> usize {
        match self {
            I2SFlavor::BitNet32F16 => 10,       // 8 data + 2 f16 scale
            I2SFlavor::Split32WithSibling => 8, // data only (scales in sibling tensor)
            I2SFlavor::GgmlQk256NoScale => 64,  // data only (no per-block scales)
        }
    }

    /// Convert to legacy I2SLayoutKind for backward compatibility
    pub fn to_layout_kind(&self) -> I2SLayoutKind {
        match self {
            I2SFlavor::BitNet32F16 => I2SLayoutKind::InlineF16,
            I2SFlavor::Split32WithSibling => I2SLayoutKind::GgmlSplit,
            I2SFlavor::GgmlQk256NoScale => I2SLayoutKind::GgmlSplit,
        }
    }
}

/// Detect I2_S flavor from tensor metadata and available bytes
///
/// # Arguments
/// * `info` - Tensor metadata (name, shape, size)
/// * `has_scale_sibling` - Whether a separate scale tensor was found
/// * `nelems` - Total number of elements in the tensor
///
/// # Returns
/// * `Ok(I2SFlavor)` - Detected flavor with detailed logging
/// * `Err(BitNetError)` - If no valid flavor matches (fail-closed with diagnostic info)
///
/// # Detection logic
/// 1. Calculate expected bytes for each flavor:
///    - blocks32 = (nelems + 31) / 32
///    - blocks256 = (nelems + 255) / 256
///    - split_need = blocks32 * 8
///    - inline_need = blocks32 * 10
///    - qk256_need = blocks256 * 64
/// 2. Match available bytes against expected (with ±64 byte tolerance for alignment)
/// 3. Priority: Split32WithSibling (if sibling) > BitNet32F16 > GgmlQk256NoScale
/// 4. Fail-closed with detailed error if no match
pub fn detect_i2s_flavor(
    info: &TensorInfo,
    has_scale_sibling: bool,
    nelems: usize,
) -> Result<I2SFlavor> {
    let blocks32 = nelems.div_ceil(32);
    let blocks256 = nelems.div_ceil(256);
    let split_need = blocks32 * 8;
    let inline_need = blocks32 * 10;
    let qk256_need = blocks256 * 64;
    let available = info.size as usize;

    // AC2: Centralized tolerance
    // - Strict mode: tight 8 bytes (fail-fast)
    // - Default: size-proportional (~0.1%) using quantization helper
    let strict = std::env::var("BITNET_STRICT_MODE").as_deref() == Ok("1");
    let tolerance = if strict {
        8usize
    } else {
        // pick a representative expected size (qk256/split) to compute tolerance bytes
        let expected_any = core::cmp::min(split_need, qk256_need);
        bitnet_quantization::qk256_tolerance_bytes(expected_any)
    };

    tracing::debug!(
        "I2_S flavor detection for '{}': nelems={}, blocks32={}, blocks256={}, available={}, split_need={}, inline_need={}, qk256_need={}, has_sibling={}, tolerance={} (strict={})",
        info.name,
        nelems,
        blocks32,
        blocks256,
        available,
        split_need,
        inline_need,
        qk256_need,
        has_scale_sibling,
        tolerance,
        strict
    );

    // Calculate diff for each flavor
    let diff_split32 = available.abs_diff(split_need);
    let diff_inline = available.abs_diff(inline_need);
    let diff_qk256 = available.abs_diff(qk256_need);

    //  Priority logic with adaptive tolerance:
    // 1. Exact matches (diff == 0) - prefer larger block sizes (qk256 > inline > split32)
    // 2. Close matches (within tolerance) - prefer split32 with sibling, then inline, then qk256
    // 3. Split32 without sibling (warn) - data-only format, possibly incomplete

    // Priority 1: Exact matches (diff == 0) - prefer larger block sizes
    if diff_qk256 == 0 {
        tracing::debug!(
            "I2_S '{}': detected GgmlQk256NoScale (exact match: available={}, qk256_need={}) - GGML format",
            info.name,
            available,
            qk256_need
        );
        return Ok(I2SFlavor::GgmlQk256NoScale);
    }
    if diff_inline == 0 {
        tracing::debug!(
            "I2_S '{}': detected BitNet32F16 (exact match: available={}, inline_need={})",
            info.name,
            available,
            inline_need
        );
        return Ok(I2SFlavor::BitNet32F16);
    }
    if diff_split32 == 0 && has_scale_sibling {
        tracing::debug!(
            "I2_S '{}': detected Split32WithSibling (exact match: available={}, split_need={}, has_sibling=true)",
            info.name,
            available,
            split_need
        );
        return Ok(I2SFlavor::Split32WithSibling);
    }

    // Priority 2: Close matches (within tolerance) - prefer QK256 for specificity
    if diff_qk256 <= tolerance {
        tracing::debug!(
            "I2_S '{}': detected GgmlQk256NoScale (close match: available={}, qk256_need={}, diff={}) - GGML format",
            info.name,
            available,
            qk256_need,
            diff_qk256
        );
        return Ok(I2SFlavor::GgmlQk256NoScale);
    }
    if has_scale_sibling && diff_split32 <= tolerance {
        tracing::debug!(
            "I2_S '{}': detected Split32WithSibling (close match: available={}, split_need={}, diff={}, has_sibling=true)",
            info.name,
            available,
            split_need,
            diff_split32
        );
        return Ok(I2SFlavor::Split32WithSibling);
    }
    if diff_inline <= tolerance {
        tracing::debug!(
            "I2_S '{}': detected BitNet32F16 (close match: available={}, inline_need={}, diff={})",
            info.name,
            available,
            inline_need,
            diff_inline
        );
        return Ok(I2SFlavor::BitNet32F16);
    }

    // Priority 3: Split32 without sibling (data-only, warn about missing scales)
    if diff_split32 <= tolerance {
        tracing::warn!(
            "I2_S '{}': bytes match split layout (close match: available={}, split_need={}, diff={}) but no scale sibling found - may be incomplete",
            info.name,
            available,
            split_need,
            diff_split32
        );
        return Ok(I2SFlavor::Split32WithSibling);
    }

    // Fail-closed: no flavor matches
    Err(BitNetError::Validation(format!(
        "I2_S '{}': no valid flavor detected. Byte accounting:\n\
         - available: {}\n\
         - split_need (32-elem blocks, 8B/block): {} (diff: {})\n\
         - inline_need (32-elem blocks, 10B/block): {} (diff: {})\n\
         - qk256_need (256-elem blocks, 64B/block): {} (diff: {})\n\
         - has_scale_sibling: {}\n\
         - tolerance: ±{} bytes ({})\n\
         All diffs exceed tolerance. This indicates an unsupported I2_S variant or corrupted data.",
        info.name,
        available,
        split_need,
        available.abs_diff(split_need),
        inline_need,
        available.abs_diff(inline_need),
        qk256_need,
        available.abs_diff(qk256_need),
        has_scale_sibling,
        tolerance,
        if strict { "strict mode" } else { "~0.1% size-proportional" }
    )))
}

/// Container for loaded tensors
pub type GgufTensors = std::collections::HashMap<String, candle_core::Tensor>;

/// Raw quantized tensor container (for preserving 2-bit QK256 data without eager dequantization)
#[derive(Clone)]
pub struct RawQuantTensor {
    pub bytes: std::sync::Arc<[u8]>,
    pub rows: usize,
    pub cols: usize,
    pub block_cols: usize,       // 256 for QK256
    pub row_stride_bytes: usize, // bytes per row in packed form
    pub flavor: I2SFlavor,       // include GgmlQk256NoScale, Split32WithSibling, BitNet32F16
}

/// Container for raw quantized tensors
pub type RawQuantTensors = std::collections::HashMap<String, RawQuantTensor>;

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
    // Security: Check offset bounds before any array access
    if *offset >= data.len() || data.len().saturating_sub(*offset) < 4 {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: format!(
                "Unexpected end of data at offset {} for u32 read (data len: {})",
                *offset,
                data.len()
            ),
        }));
    }

    // Security: Use safe array indexing with bounds checking
    let bytes = data.get(*offset..*offset + 4).ok_or_else(|| {
        BitNetError::Model(ModelError::InvalidFormat {
            format: format!("Buffer bounds violation reading u32 at offset {}", *offset),
        })
    })?;

    let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    *offset += 4;
    Ok(value)
}

pub fn read_i32(data: &[u8], offset: &mut usize) -> Result<i32> {
    Ok(read_u32(data, offset)? as i32)
}

pub fn read_u64(data: &[u8], offset: &mut usize) -> Result<u64> {
    // Security: Check offset bounds before any array access
    if *offset >= data.len() || data.len().saturating_sub(*offset) < 8 {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: format!(
                "Unexpected end of data at offset {} for u64 read (data len: {})",
                *offset,
                data.len()
            ),
        }));
    }

    // Security: Use safe array indexing with bounds checking
    let bytes = data.get(*offset..*offset + 8).ok_or_else(|| {
        BitNetError::Model(ModelError::InvalidFormat {
            format: format!("Buffer bounds violation reading u64 at offset {}", *offset),
        })
    })?;

    let value = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    *offset += 8;
    Ok(value)
}

pub fn read_f32(data: &[u8], offset: &mut usize) -> Result<f32> {
    // Security: Check offset bounds before any array access
    if *offset >= data.len() || data.len().saturating_sub(*offset) < 4 {
        return Err(BitNetError::Model(ModelError::InvalidFormat {
            format: format!(
                "Unexpected end of data at offset {} for f32 read (data len: {})",
                *offset,
                data.len()
            ),
        }));
    }

    // Security: Use safe array indexing with bounds checking
    let bytes = data.get(*offset..*offset + 4).ok_or_else(|| {
        BitNetError::Model(ModelError::InvalidFormat {
            format: format!("Buffer bounds violation reading f32 at offset {}", *offset),
        })
    })?;

    let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    *offset += 4;
    Ok(value)
}

pub fn read_bool(data: &[u8], offset: &mut usize) -> Result<bool> {
    Ok(read_u8(data, offset)? != 0)
}

pub fn read_string(data: &[u8], offset: &mut usize) -> Result<String> {
    read_string_with_limits(data, offset, &SecurityLimits::default())
}

pub fn read_string_with_limits(
    data: &[u8],
    offset: &mut usize,
    limits: &SecurityLimits,
) -> Result<String> {
    let len = read_u64(data, offset)? as usize;

    // Security: Enforce strict string length limits to prevent DoS
    if len > limits.max_string_length {
        return Err(BitNetError::Security(SecurityError::ResourceLimit {
            resource: "string_length".to_string(),
            value: len as u64,
            limit: limits.max_string_length as u64,
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
    read_bytes_with_limits(data, offset, &SecurityLimits::default())
}

pub fn read_bytes_with_limits(
    data: &[u8],
    offset: &mut usize,
    limits: &SecurityLimits,
) -> Result<Vec<u8>> {
    let len = read_u64(data, offset)? as usize;

    // Security: Enforce byte array length limits
    if len > limits.max_string_length {
        // Reuse string limit for bytes
        return Err(BitNetError::Security(SecurityError::ResourceLimit {
            resource: "byte_array_length".to_string(),
            value: len as u64,
            limit: limits.max_string_length as u64,
        }));
    }

    if *offset + len > data.len() {
        return Err(BitNetError::Security(SecurityError::MalformedData {
            reason: "Byte array extends beyond data bounds".to_string(),
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
